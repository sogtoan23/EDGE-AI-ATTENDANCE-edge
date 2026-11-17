import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import json
import time
import base64
import threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import requests
import websocket
from PIL import Image, ImageDraw, ImageFont

from config import (
    SERVER_HTTP_URL,
    SERVER_WS_URL,
    LIVENESS_THRESHOLD,
    MATCH_THRESHOLD,
    MAX_FACES_PER_FRAME,
    TRACK_IOU_THRESHOLD,
    TRACK_MAX_AGE,
    CONSISTENT_FRAMES,
    IN_FLIGHT_LOCK_SECONDS,
    SESSION_COOLDOWN_SECONDS,
    IGNORED_COOLDOWN_SECONDS,
    WEBCAM_REGISTRY_FILE,
    CACHE_VECTORS_FILE,
    OFFLINE_QUEUE_FILE,
    PUNCH_CACHE_FILE,
)
from ai_pipeline import EdgeAIPipeline


# ======================================================================
# 1. BỘ QUẢN LÝ DANH SÁCH WEBCAM (TỰ QUÉT MỚI, THÊM, SỬA TOKEN, XÓA)
# ======================================================================
class WebcamManager:
    def __init__(self, registry_path=WEBCAM_REGISTRY_FILE):
        self.registry_path = registry_path
        self.webcams = self._load_registry()

    def _load_registry(self) -> list:
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception as e:
                print(f"[WARN] Lỗi đọc file danh sách webcam: {e}")
        return []

    def _save_registry(self):
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.webcams, f, ensure_ascii=False, indent=2)

    @staticmethod
    def scan_physical_cameras(max_ports=6) -> list:
        """Quét tất cả cổng webcam vật lý (dùng DirectShow trên Windows cho cam USB)."""
        found = []
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        for idx in range(max_ports):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, frame = False, None
                for _ in range(3):
                    ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    found.append({"port": idx, "resolution": f"{w}x{h}"})
                cap.release()
        return found

    @staticmethod
    def verify_token_with_server(token: str) -> bool:
        """Kiểm tra mã X-Device-Token với Server Render/Local."""
        print("   ⏳ Đang xác thực X-Device-Token với Server...")
        try:
            res = requests.get(
                f"{SERVER_HTTP_URL}/api/v1/students/sync-vectors",
                headers={"X-Device-Token": token},
                timeout=10,
            )
            if res.status_code == 200:
                print("   ✅ Token hợp lệ! Kết nối Server thành công.")
                return True
            elif res.status_code in (401, 403):
                print("   ❌ Token không tồn tại hoặc đã bị vô hiệu hóa trên Server!")
                return False
            else:
                print(f"   ⚠️ Server trả về mã {res.status_code}. Cho phép lưu tạm.")
                return True
        except Exception:
            print("   ⚠️ Không có kết nối mạng tới Server. Lưu Token để chạy Offline.")
            return True

    def add_webcam_interactive(self, prefill_source=None):
        print("\n--- ➕ THÊM WEBCAM MỚI VÀO HỆ THỐNG ---")
        if prefill_source is not None:
            camera_source = prefill_source
            print(f"   • Nguồn Camera tự động chọn: Cổng [{camera_source}]")
        else:
            while True:
                src = input("👉 Nhập số cổng Camera (VD: 0, 1) hoặc link RTSP (rtsp://...): ").strip()
                if src.isdigit():
                    camera_source = int(src)
                    break
                elif src.startswith(("rtsp://", "http://", "https://")):
                    camera_source = src
                    break
                print("   ❌ Định dạng không hợp lệ. Vui lòng nhập số cổng hoặc link rtsp://")

        room_name = input("👉 Nhập tên Phòng học gắn kèm (VD: F404, P302): ").strip().upper() or "CHƯA_GÁN_PHÒNG"
        device_name = input("👉 Nhập tên gợi nhớ cho Webcam (VD: Cam Cửa Chính): ").strip() or f"Camera {room_name}"

        while True:
            token = input("👉 Dán mã X-Device-Token (DEV_...): ").strip()
            if not token:
                print("   ❌ Token không được để trống!")
                continue
            if self.verify_token_with_server(token):
                break
            retry = input("   Bạn có muốn nhập lại Token khác không? (y/n): ").strip().lower()
            if retry != "y":
                return

        new_entry = {
            "camera_source": camera_source,
            "room_name": room_name,
            "device_name": device_name,
            "device_token": token,
            "updated_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
        }
        self.webcams.append(new_entry)
        self._save_registry()
        print(f"✅ Đã thêm [{device_name} - Phòng {room_name}] vào danh sách quản lý!")

    def edit_webcam_interactive(self):
        if not self.webcams:
            print("⚠️ Danh sách webcam đang trống.")
            return
        self.print_webcam_table()
        idx_str = input("👉 Nhập số thứ tự (STT) Webcam muốn chỉnh sửa: ").strip()
        if not idx_str.isdigit() or not (1 <= int(idx_str) <= len(self.webcams)):
            print("❌ STT không hợp lệ.")
            return

        cam = self.webcams[int(idx_str) - 1]
        print(f"\n--- ✏️ CHỈNH SỬA WEBCAM: {cam['device_name']} (Phòng {cam['room_name']}) ---")
        print("   (Nhấn Enter nếu muốn giữ nguyên giá trị hiện tại)")

        new_src = input(f"   • Nguồn Camera [{cam['camera_source']}]: ").strip()
        if new_src:
            cam["camera_source"] = int(new_src) if new_src.isdigit() else new_src

        new_room = input(f"   • Phòng học [{cam['room_name']}]: ").strip().upper()
        if new_room:
            cam["room_name"] = new_room

        new_name = input(f"   • Tên Webcam [{cam['device_name']}]: ").strip()
        if new_name:
            cam["device_name"] = new_name

        masked = cam["device_token"][:8] + "••••" + cam["device_token"][-4:]
        new_token = input(f"   • Mã X-Device-Token mới [{masked}]: ").strip()
        if new_token:
            if self.verify_token_with_server(new_token):
                cam["device_token"] = new_token
            else:
                print("   ⚠️ Giữ nguyên Token cũ do Token mới không hợp lệ.")

        cam["updated_at"] = datetime.now().strftime("%d/%m/%Y %H:%M")
        self._save_registry()
        print("✅ Đã cập nhật thông tin Webcam thành công!")

    def delete_webcam_interactive(self):
        if not self.webcams:
            print("⚠️ Danh sách webcam đang trống.")
            return
        self.print_webcam_table()
        idx_str = input("👉 Nhập số thứ tự (STT) Webcam muốn xóa: ").strip()
        if not idx_str.isdigit() or not (1 <= int(idx_str) <= len(self.webcams)):
            print("❌ STT không hợp lệ.")
            return

        target = self.webcams[int(idx_str) - 1]
        confirm = input(f"⚠️ Xác nhận xóa '{target['device_name']}' (Phòng {target['room_name']})? (y/n): ").strip().lower()
        if confirm == "y":
            self.webcams.pop(int(idx_str) - 1)
            self._save_registry()
            print("🗑️ Đã xóa Webcam khỏi danh sách!")

    def print_webcam_table(self):
        print("\n" + "=" * 82)
        print(f"{'STT':<4} | {'Nguồn':<12} | {'Phòng':<8} | {'Tên Webcam':<24} | {'X-Device-Token'}")
        print("-" * 82)
        if not self.webcams:
            print("   (Chưa có webcam nào được lưu trong hệ thống)")
        else:
            for i, c in enumerate(self.webcams, 1):
                tok = c["device_token"]
                masked_tok = (tok[:8] + "••••••••" + tok[-4:]) if len(tok) > 12 else "••••••••"
                src_str = f"Cổng {c['camera_source']}" if isinstance(c["camera_source"], int) else "RTSP Stream"
                print(f"{i:<4} | {src_str:<12} | {c['room_name']:<8} | {c['device_name']:<24} | {masked_tok}")
        print("=" * 82)

    def check_and_prompt_new_webcams(self):
        """Quét phần cứng và hỏi thêm ngay nếu phát hiện webcam mới cắm vào."""
        print("\n🔍 Đang quét các webcam kết nối trên thiết bị...")
        physical_cams = self.scan_physical_cameras()
        saved_ports = {c["camera_source"] for c in self.webcams if isinstance(c["camera_source"], int)}

        new_cams = [c for c in physical_cams if c["port"] not in saved_ports]
        if new_cams:
            for nc in new_cams:
                print(f"\n🆕 PHÁT HIỆN WEBCAM MỚI: Cổng [{nc['port']}] (Độ phân giải {nc['resolution']})")
                ans = input(f"👉 Bạn có muốn thêm webcam cổng [{nc['port']}] vào danh sách quản lý không? (y/n): ").strip().lower()
                if ans == "y":
                    self.add_webcam_interactive(prefill_source=nc["port"])
        elif physical_cams:
            print(f"✅ Đã nhận diện {len(physical_cams)} webcam vật lý (tất cả đều đã có trong danh sách).")
        else:
            print("⚠️ Không tìm thấy webcam USB vật lý nào (bạn vẫn có thể thêm link RTSP).")

    def open_manager_menu(self) -> dict | None:
        self.check_and_prompt_new_webcams()

        while True:
            self.print_webcam_table()
            print("📌 BẢNG ĐIỀU KHIỂN QUẢN LÝ WEBCAM:")
            print("   [1] ▶️  Chọn Webcam và BẮT ĐẦU ĐIỂM DANH")
            print("   [2] ➕  Thêm Webcam mới (USB / RTSP)")
            print("   [3] ✏️  Chỉnh sửa thông tin / Đổi mã X-Device-Token")
            print("   [4] 🗑️  Xóa Webcam khỏi danh sách")
            print("   [5] 🔄  Quét lại cổng Webcam vật lý")
            print("   [0] ❌  Thoát chương trình")

            choice = input("\n👉 Nhập lựa chọn của bạn (0-5): ").strip()

            if choice == "1":
                if not self.webcams:
                    print("⚠️ Danh sách trống! Vui lòng thêm ít nhất 1 webcam trước.")
                    self.add_webcam_interactive()
                    continue
                if len(self.webcams) == 1:
                    selected = self.webcams[0]
                    print(f"🚀 Tự động chọn webcam duy nhất: [{selected['device_name']} - Phòng {selected['room_name']}]")
                    return selected

                stt = input(f"👉 Nhập STT Webcam muốn chạy (1 - {len(self.webcams)}): ").strip()
                if stt.isdigit() and 1 <= int(stt) <= len(self.webcams):
                    return self.webcams[int(stt) - 1]
                print("❌ STT không hợp lệ!")

            elif choice == "2":
                self.add_webcam_interactive()
            elif choice == "3":
                self.edit_webcam_interactive()
            elif choice == "4":
                self.delete_webcam_interactive()
            elif choice == "5":
                self.check_and_prompt_new_webcams()
            elif choice == "0":
                return None
            else:
                print("❌ Vui lòng chọn số từ 0 đến 5.")


# ======================================================================
# 2. THUẬT TOÁN THEO DÕI ĐỐI TƯỢNG ĐA KHUNG HÌNH (MULTI-FACE IOU TRACKER)
# ======================================================================
class FaceTrack:
    def __init__(self, track_id: int, bbox: np.ndarray):
        self.track_id = track_id
        self.bbox = bbox
        self.missed_frames = 0

        self.verified = False
        self.student_id = None
        self.display_name = None
        self.similarity = 0.0
        self.hit_count = 0
        self.candidate_id = None
        self.last_status_text = ""
        self.is_spoof = False
        self.spoof_score = 0.0


class MultiFaceTracker:
    def __init__(self, iou_threshold=0.35, max_age=15):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.tracks: dict[int, FaceTrack] = {}

    @staticmethod
    def _compute_iou(boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def update(self, detections: list) -> list:
        matched_pairs = []
        unmatched_track_ids = set(self.tracks.keys())
        unmatched_det_indices = set(range(len(detections)))

        if self.tracks and detections:
            iou_candidates = []
            for tid, track in self.tracks.items():
                for d_idx, det in enumerate(detections):
                    iou = self._compute_iou(track.bbox, det["bbox"])
                    if iou >= self.iou_threshold:
                        iou_candidates.append((iou, tid, d_idx))

            iou_candidates.sort(key=lambda x: x[0], reverse=True)
            for iou, tid, d_idx in iou_candidates:
                if tid in unmatched_track_ids and d_idx in unmatched_det_indices:
                    unmatched_track_ids.remove(tid)
                    unmatched_det_indices.remove(d_idx)
                    track = self.tracks[tid]
                    track.bbox = detections[d_idx]["bbox"]
                    track.missed_frames = 0
                    matched_pairs.append((track, detections[d_idx]))

        for tid in unmatched_track_ids:
            self.tracks[tid].missed_frames += 1
            if self.tracks[tid].missed_frames > self.max_age:
                del self.tracks[tid]

        for d_idx in unmatched_det_indices:
            new_track = FaceTrack(self.next_id, detections[d_idx]["bbox"])
            self.tracks[self.next_id] = new_track
            self.next_id += 1
            matched_pairs.append((new_track, detections[d_idx]))

        return matched_pairs


# ======================================================================
# 3. LÕI XỬ LÝ ĐIỂM DANH TẠI BIÊN (CHỐNG SPAM + TỰ MỞ KHÓA CA TIẾP THEO)
# ======================================================================
class EdgeAttendanceAgent:
    def __init__(self, device_cfg: dict):
        self.camera_source = device_cfg["camera_source"]
        self.room_name = device_cfg["room_name"]
        self.device_name = device_cfg["device_name"]
        self.device_token = device_cfg["device_token"]

        self.ai = EdgeAIPipeline()
        self.tracker = MultiFaceTracker(iou_threshold=TRACK_IOU_THRESHOLD, max_age=TRACK_MAX_AGE)

        self.lock = threading.Lock()
        self.student_ids = []
        self.resolved_names = {}
        self.face_matrix = None

        self.headers = {
            "X-Device-Token": self.device_token,
            "Content-Type": "application/json",
        }

        # Bộ nhớ chống spam theo từng sinh viên & ca học:
        # Key: f"{room_name}:{student_id}"
        # Value: {"cooldown_until": float, "status": str, "full_name": str, "result_status": str, "schedule_id": int|None}
        self.punched_records = {}
        self.recent_checkins = deque(maxlen=4)

        self._load_punch_cache()
        self._load_local_cache()
        self.sync_vectors_from_server()

    # ------------------------------------------------------------------
    # QUẢN LÝ BỘ NHỚ CHỐNG SPAM (SESSION COOLDOWN CACHE)
    # ------------------------------------------------------------------
    def _load_punch_cache(self):
        if not os.path.exists(PUNCH_CACHE_FILE):
            return
        try:
            with open(PUNCH_CACHE_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            now = time.time()
            valid = {}
            for k, v in raw.items():
                if v.get("cooldown_until", 0) > now:
                    valid[k] = v
                    if v.get("full_name") and ":" in k:
                        sid = k.split(":", 1)[1]
                        self.resolved_names[sid] = v["full_name"]
            self.punched_records = valid
        except Exception:
            self.punched_records = {}

    def _save_punch_cache(self):
        # Lưu ý: Hàm này được gọi bên trong self.lock hoặc sau khi lấy bản sao an toàn
        now = time.time()
        clean_data = {
            k: v for k, v in self.punched_records.items() if v.get("cooldown_until", 0) > now
        }
        try:
            with open(PUNCH_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(clean_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _try_acquire_punch_slot(self, student_id: str) -> tuple[bool, dict | None]:
        """
        Kiểm tra xem sinh viên này có được phép gửi punch lên Server lúc này không.
        - Trả về (True, None) và KHÓA NGAY lập tức nếu hợp lệ (chống gửi trùng đa luồng).
        - Trả về (False, record_dict) nếu đang trong thời gian khóa chống spam của ca hiện tại.
        """
        key = f"{self.room_name}:{student_id}"
        now = time.time()
        with self.lock:
            rec = self.punched_records.get(key)
            if rec and now < rec.get("cooldown_until", 0):
                return False, dict(rec)

            # Cấp quyền gửi và khóa tạm thời ngay lập tức trong lúc chờ HTTP response
            self.punched_records[key] = {
                "cooldown_until": now + IN_FLIGHT_LOCK_SECONDS,
                "status": "in_flight",
                "full_name": self.resolved_names.get(student_id, student_id),
                "result_status": "Đang gửi...",
                "schedule_id": None,
            }
            return True, None

    # ------------------------------------------------------------------
    # ĐỒNG BỘ VECTOR & QUẢN LÝ LOCAL CACHE
    # ------------------------------------------------------------------
    def _rebuild_matrix(self, data_list):
        with self.lock:
            ids = []
            vecs = []
            for item in data_list:
                sid = item.get("student_id")
                emb = item.get("embedding")
                if sid and isinstance(emb, list) and len(emb) == 512:
                    ids.append(sid)
                    vecs.append(emb)

            self.student_ids = ids
            if len(vecs) > 0:
                mat = np.array(vecs, dtype=np.float32)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                self.face_matrix = mat / np.maximum(norms, 1e-6)
            else:
                self.face_matrix = None

    def _save_local_cache(self):
        with self.lock:
            if self.face_matrix is None:
                data = []
            else:
                data = [
                    {"student_id": sid, "embedding": self.face_matrix[i].tolist()}
                    for i, sid in enumerate(self.student_ids)
                ]
        with open(CACHE_VECTORS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def _load_local_cache(self):
        if os.path.exists(CACHE_VECTORS_FILE):
            try:
                with open(CACHE_VECTORS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._rebuild_matrix(data)
                print(f"[CACHE] Đã nạp {len(self.student_ids)} vector khuôn mặt từ bộ nhớ cục bộ.")
            except Exception as e:
                print(f"[CACHE] Lỗi đọc cache: {e}")

    def sync_vectors_from_server(self):
        print("[SYNC] Đang gọi GET /api/v1/students/sync-vectors...")
        try:
            res = requests.get(
                f"{SERVER_HTTP_URL}/api/v1/students/sync-vectors",
                headers={"X-Device-Token": self.device_token},
                timeout=15,
            )
            if res.status_code == 200:
                data = res.json()
                self._rebuild_matrix(data)
                self._save_local_cache()
                print(f"[SYNC] Thành công! Tổng số vector trong RAM: {len(self.student_ids)}")
            else:
                print(f"[SYNC] Server phản hồi lỗi ({res.status_code}): {res.text}")
        except Exception as e:
            print(f"[SYNC] Mất kết nối Server, chạy chế độ Offline với cache hiện tại: {e}")

    def upsert_single_face(self, student_id: str, embedding: list):
        vec = np.array(embedding, dtype=np.float32)
        vec = vec / max(np.linalg.norm(vec), 1e-6)

        with self.lock:
            if student_id in self.student_ids:
                idx = self.student_ids.index(student_id)
                self.face_matrix[idx] = vec
                print(f"[WS-SYNC] Cập nhật vector SV: {student_id}")
            else:
                self.student_ids.append(student_id)
                if self.face_matrix is None or len(self.face_matrix) == 0:
                    self.face_matrix = np.array([vec], dtype=np.float32)
                else:
                    self.face_matrix = np.vstack([self.face_matrix, vec])
                print(f"[WS-SYNC] Thêm mới vector SV: {student_id}")
        self._save_local_cache()

    def delete_single_face(self, student_id: str):
        with self.lock:
            if student_id in self.student_ids:
                idx = self.student_ids.index(student_id)
                self.student_ids.pop(idx)
                self.resolved_names.pop(student_id, None)
                self.face_matrix = np.delete(self.face_matrix, idx, axis=0)
                if len(self.student_ids) == 0:
                    self.face_matrix = None
                print(f"[WS-SYNC] Đã xóa vector SV: {student_id}")
        self._save_local_cache()

    def match_embedding(self, query_vec: np.ndarray):
        with self.lock:
            if self.face_matrix is None or len(self.student_ids) == 0:
                return None, 0.0
            scores = np.dot(self.face_matrix, query_vec)
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            if best_score >= MATCH_THRESHOLD:
                return self.student_ids[best_idx], best_score
        return None, best_score

    # ------------------------------------------------------------------
    # HÀNG ĐỢI OFFLINE & GỬI ĐIỂM DANH LÊN SERVER
    # ------------------------------------------------------------------
    def _enqueue_offline(self, payload: dict):
        with self.lock:
            queue = []
            if os.path.exists(OFFLINE_QUEUE_FILE):
                try:
                    with open(OFFLINE_QUEUE_FILE, "r", encoding="utf-8") as f:
                        queue = json.load(f)
                except Exception:
                    queue = []
            queue.append(payload)
            with open(OFFLINE_QUEUE_FILE, "w", encoding="utf-8") as f:
                json.dump(queue, f, ensure_ascii=False)
            print(f"[OFFLINE] Đã lưu tạm vào hàng đợi ({len(queue)} bản ghi chờ đồng bộ).")

    def flush_offline_queue(self):
        with self.lock:
            if not os.path.exists(OFFLINE_QUEUE_FILE):
                return
            try:
                with open(OFFLINE_QUEUE_FILE, "r", encoding="utf-8") as f:
                    queue = json.load(f)
            except Exception:
                return

        if not queue:
            return

        remaining = []
        for item in queue:
            try:
                res = requests.post(
                    f"{SERVER_HTTP_URL}/api/v1/attendance/punch",
                    json=item,
                    headers=self.headers,
                    timeout=5,
                )
                if res.status_code != 200:
                    remaining.append(item)
            except Exception:
                remaining.append(item)
                break

        with self.lock:
            with open(OFFLINE_QUEUE_FILE, "w", encoding="utf-8") as f:
                json.dump(remaining, f, ensure_ascii=False)
        if len(remaining) < len(queue):
            print(f"[OFFLINE-SYNC] Đã đồng bộ bù {len(queue) - len(remaining)} bản ghi lên Server.")

    def send_punch_async(self, track: FaceTrack, frame_snapshot):
        student_id = track.student_id
        similarity = track.similarity
        key = f"{self.room_name}:{student_id}"

        def _worker():
            _, buf = cv2.imencode(".jpg", frame_snapshot, [cv2.IMWRITE_JPEG_QUALITY, 65])
            b64_img = base64.b64encode(buf).decode("utf-8")
            now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            now_hms = datetime.now().strftime("%H:%M:%S")

            payload = {
                "student_id": student_id,
                "similarity": round(float(similarity), 4),
                "timestamp": now_iso,
                "image_base64": b64_img,
            }
            try:
                res = requests.post(
                    f"{SERVER_HTTP_URL}/api/v1/attendance/punch",
                    json=payload,
                    headers=self.headers,
                    timeout=6,
                )
                if res.status_code == 200:
                    data = res.json()
                    resp_status = data.get("status")

                    if resp_status == "success":
                        full_name = data.get("full_name", student_id)
                        class_name = data.get("class_name", "")
                        subject_name = data.get("subject_name", "")
                        result_status = data.get("result_status", "Đã điểm danh")
                        schedule_id = data.get("schedule_id")

                        track.display_name = full_name
                        track.last_status_text = result_status

                        with self.lock:
                            self.resolved_names[student_id] = full_name
                            # Khóa chống spam trong 15 phút cho ca học hiện tại; hết 15p tự mở cho ca sau
                            self.punched_records[key] = {
                                "cooldown_until": time.time() + SESSION_COOLDOWN_SECONDS,
                                "status": "success",
                                "full_name": full_name,
                                "result_status": result_status,
                                "schedule_id": schedule_id,
                            }
                            self._save_punch_cache()
                            self.recent_checkins.appendleft(
                                f"✔ {student_id} - {full_name} ({class_name}) | {result_status} [{now_hms}]"
                            )

                        print(
                            f"[{self.room_name} | SUCCESS] {student_id} - {full_name} "
                            f"| Ca ID: {schedule_id} ({subject_name}) | {result_status} "
                            f"(Khóa chống spam {SESSION_COOLDOWN_SECONDS // 60} phút)"
                        )

                    elif resp_status == "ignored":
                        msg = data.get("message", "Bỏ qua")
                        already_checked = "đã điểm danh" in msg.lower() or "đã được ghi nhận" in msg.lower()
                        cooldown_dur = SESSION_COOLDOWN_SECONDS if already_checked else IGNORED_COOLDOWN_SECONDS
                        status_label = "Đã điểm danh ca này" if already_checked else "Ngoài giờ / Chưa tới ca"

                        track.last_status_text = status_label
                        with self.lock:
                            self.punched_records[key] = {
                                "cooldown_until": time.time() + cooldown_dur,
                                "status": "ignored",
                                "full_name": self.resolved_names.get(student_id, student_id),
                                "result_status": status_label,
                                "schedule_id": None,
                            }
                            self._save_punch_cache()
                            self.recent_checkins.appendleft(f"ℹ {student_id}: {msg[:44]}...")

                        print(f"[{self.room_name} | IGNORED - Khóa {cooldown_dur}s] {msg}")

                    self.flush_offline_queue()
                else:
                    print(f"[{self.room_name} | HTTP {res.status_code}] {res.text}")
                    with self.lock:
                        self.punched_records[key] = {
                            "cooldown_until": time.time() + IGNORED_COOLDOWN_SECONDS,
                            "status": "error",
                            "full_name": self.resolved_names.get(student_id, student_id),
                            "result_status": f"Lỗi HTTP {res.status_code}",
                            "schedule_id": None,
                        }
            except Exception as e:
                print(f"[{self.room_name} | OFFLINE] Không kết nối được Server: {e}")
                with self.lock:
                    self.punched_records[key] = {
                        "cooldown_until": time.time() + SESSION_COOLDOWN_SECONDS,
                        "status": "offline",
                        "full_name": self.resolved_names.get(student_id, student_id),
                        "result_status": "Đã lưu Offline",
                        "schedule_id": None,
                    }
                    self._save_punch_cache()
                    self.recent_checkins.appendleft(f"✔ {student_id} (Lưu Offline - {now_hms})")
                self._enqueue_offline(payload)

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # WEBSOCKET LẮNG NGHE ĐỒNG BỘ REALTIME TỪ SERVER
    # ------------------------------------------------------------------
    def start_ws_listener(self):
        self._ws_running = True

        def on_message(ws, message):
            try:
                msg = json.loads(message)
                event = msg.get("event")
                data = msg.get("data", {})
                if event == "SYNC_UPSERT_FACE":
                    self.upsert_single_face(data["student_id"], data["embedding"])
                elif event == "SYNC_DELETE_FACE":
                    self.delete_single_face(data["student_id"])
                elif event == "SYNC_FULL_GALLERY":
                    if isinstance(data, list):
                        self._rebuild_matrix(data)
                        self._save_local_cache()
                    else:
                        self.sync_vectors_from_server()
            except Exception as err:
                print(f"[WS] Lỗi xử lý gói tin: {err}")

        def _run_forever():
            while self._ws_running:
                try:
                    self._ws_app = websocket.WebSocketApp(
                        SERVER_WS_URL,
                        header=[f"X-Device-Token: {self.device_token}"],
                        on_message=on_message,
                    )
                    self._ws_app.run_forever(ping_interval=25, ping_timeout=10)
                except Exception:
                    pass
                if self._ws_running:
                    time.sleep(5)

        threading.Thread(target=_run_forever, daemon=True).start()

    def stop_ws_listener(self):
        self._ws_running = False
        if hasattr(self, "_ws_app") and self._ws_app:
            try:
                self._ws_app.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # VẼ KHUNG ROI ZONE, TRACK_ID & BẢNG PHẢN HỒI TẠI CỬA LỚP
    # ------------------------------------------------------------------
    def _draw_overlay(self, frame, tracks_to_draw, roi_bounds):
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 19)
            font_small = ImageFont.truetype("arial.ttf", 15)
        except Exception:
            font = ImageFont.load_default()
            font_small = font

        h, w, _ = frame.shape
        rx1, ry1, rx2, ry2 = roi_bounds

        draw.rectangle([rx1, ry1, rx2, ry2], outline=(56, 189, 248), width=2)
        draw.text((rx1 + 8, ry1 + 6), "VÙNG QUÉT ROI (0.5m - 1.5m)", font=font_small, fill=(56, 189, 248))

        draw.rectangle([0, 0, w, 36], fill=(15, 23, 42))
        header_txt = (
            f"PHÒNG: {self.room_name} | {self.device_name} | "
            f"Dữ liệu: {len(self.student_ids)} SV | Phím: [M] Quản lý Cam  [S] Đồng bộ  [Q] Thoát"
        )
        draw.text((15, 8), header_txt, font=font_small, fill=(248, 250, 252))

        for track in tracks_to_draw:
            x1, y1, x2, y2 = track.bbox.astype(int)
            if track.is_spoof:
                color = (220, 38, 38)
                label = f"[ID#{track.track_id}] CẢNH BÁO GIẢ MẠO ({track.spoof_score:.2f})"
            elif track.verified:
                color = (22, 163, 74)
                with self.lock:
                    name_show = track.display_name or self.resolved_names.get(track.student_id, track.student_id)
                st_suffix = f" - {track.last_status_text}" if track.last_status_text else ""
                label = f"✅ [ID#{track.track_id}] {name_show}{st_suffix}"
            elif track.candidate_id:
                color = (234, 179, 8)
                label = f"⏳ [ID#{track.track_id}] Đang xác thực {track.candidate_id}..."
            else:
                color = (37, 99, 235)
                label = f"[ID#{track.track_id}] Người lạ ({track.similarity:.2f})"

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            text_w = max(180, len(label) * 10 + 16)
            draw.rectangle([x1, max(36, y1 - 28), x1 + text_w, max(36, y1)], fill=color)
            draw.text((x1 + 6, max(38, y1 - 25)), label, font=font, fill=(255, 255, 255))

        with self.lock:
            recent_snapshot = list(self.recent_checkins)

        if recent_snapshot:
            overlay_h = 72 + len(recent_snapshot) * 23
            draw.rectangle([10, 42, 540, overlay_h], fill=(15, 23, 42))
            draw.text((18, 46), "NHẬT KÝ CỬA LỚP GẦN NHẤT:", font=font_small, fill=(56, 189, 248))
            for idx, line in enumerate(recent_snapshot):
                draw.text((18, 68 + idx * 22), line, font=font_small, fill=(74, 222, 128))

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # VÒNG LẶP CAMERA CHÍNH: ROI -> TOP-K -> TRACKING -> ANTI-SPAM PUNCH
    # ------------------------------------------------------------------
    def run(self) -> str:
        self.start_ws_listener()
        backend = cv2.CAP_DSHOW if os.name == "nt" and isinstance(self.camera_source, int) else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.camera_source, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print(f"❌ Không thể mở nguồn camera [{self.camera_source}]. Quay lại trình quản lý Webcam...")
            self.stop_ws_listener()
            return "MENU"

        print(f"\n🚀 [{self.room_name} - {self.device_name}] Đang chạy điểm danh!")
        print("   👉 Phím tắt trên cửa sổ camera: [M] Quản lý Webcam | [S] Đồng bộ vector | [Q] Thoát")

        next_action = "EXIT"

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            roi_bounds = self.ai.get_roi_bounds(frame.shape)
            top_k_detections = self.ai.detect_faces_in_roi(frame, max_faces=MAX_FACES_PER_FRAME)
            tracked_pairs = self.tracker.update(top_k_detections)
            active_tracks = []

            for track, det in tracked_pairs:
                active_tracks.append(track)

                # TẦNG 1 CHỐNG SPAM: Track_ID này đã xác thực rồi -> Bỏ qua hoàn toàn AI & HTTP
                if track.verified:
                    continue

                liveness = self.ai.check_liveness(frame, det["bbox"])
                if liveness < LIVENESS_THRESHOLD:
                    track.is_spoof = True
                    track.spoof_score = liveness
                    track.hit_count = 0
                    continue

                track.is_spoof = False

                aligned = self.ai.align_face(frame, det["kps"])
                emb = self.ai.extract_embedding(aligned)

                sid, score = self.match_embedding(emb)
                track.similarity = score

                if sid:
                    if track.candidate_id == sid:
                        track.hit_count += 1
                    else:
                        track.candidate_id = sid
                        track.hit_count = 1

                    if track.hit_count >= CONSISTENT_FRAMES:
                        track.verified = True
                        track.student_id = sid

                        # TẦNG 2 & 3 CHỐNG SPAM: Kiểm tra & khóa trước khi gửi lên Server
                        can_send, existing_rec = self._try_acquire_punch_slot(sid)
                        if can_send:
                            with self.lock:
                                track.display_name = self.resolved_names.get(sid, sid)
                            self.send_punch_async(track, frame.copy())
                        else:
                            # Sinh viên này đã được điểm danh trong ca hiện tại -> Hiển thị luôn trạng thái đã lưu, KHÔNG spam Server
                            track.display_name = existing_rec.get("full_name", sid)
                            track.last_status_text = existing_rec.get("result_status", "Đã điểm danh")
                else:
                    track.candidate_id = None
                    track.hit_count = 0

            display_frame = self._draw_overlay(frame, active_tracks, roi_bounds)
            cv2.imshow(f"EDGE AI - {self.room_name} ({self.device_name})", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                next_action = "EXIT"
                break
            elif key == ord("s"):
                self.sync_vectors_from_server()
            elif key == ord("m"):
                next_action = "MENU"
                break

        cap.release()
        cv2.destroyAllWindows()
        self.stop_ws_listener()
        return next_action


# ======================================================================
# ĐIỂM KHỞI CHẠY CHÍNH
# ======================================================================
if __name__ == "__main__":
    manager = WebcamManager()
    while True:
        selected_cam = manager.open_manager_menu()
        if selected_cam is None:
            print("👋 Đã thoát hệ thống Edge AI.")
            break

        agent = EdgeAttendanceAgent(selected_cam)
        action = agent.run()
        if action != "MENU":
            print("👋 Đã dừng hệ thống Edge AI.")
            break