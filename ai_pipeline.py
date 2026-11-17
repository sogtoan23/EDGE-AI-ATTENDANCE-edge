import os
import cv2
import numpy as np
import onnxruntime as ort
from config import (
    MODEL_DIR,
    DET_CONF_THRESHOLD,
    ROI_X_MARGIN_RATIO,
    ROI_Y_MARGIN_RATIO,
    MIN_FACE_AREA_RATIO,
)

# Tọa độ 5 điểm mốc chuẩn ArcFace (112x112)
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


class EdgeAIPipeline:
    def __init__(self):
        providers = ["CPUExecutionProvider"]
        self.det_sess = ort.InferenceSession(
            os.path.join(MODEL_DIR, "det_500m.onnx"), providers=providers
        )
        self.fas_sess = ort.InferenceSession(
            os.path.join(MODEL_DIR, "minifasnet.onnx"), providers=providers
        )
        self.rec_sess = ort.InferenceSession(
            os.path.join(MODEL_DIR, "w600k_mbf.onnx"), providers=providers
        )

        self.det_input_name = self.det_sess.get_inputs()[0].name
        self.fas_input_name = self.fas_sess.get_inputs()[0].name
        self.rec_input_name = self.rec_sess.get_inputs()[0].name

        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self._center_cache = {}

    def get_roi_bounds(self, frame_shape):
        """Tính tọa độ vùng nhận diện trung tâm (ROI Zone) cho cự ly 0.5m - 1.5m."""
        h, w = frame_shape[:2]
        rx1 = int(w * ROI_X_MARGIN_RATIO)
        ry1 = int(h * ROI_Y_MARGIN_RATIO)
        rx2 = int(w * (1.0 - ROI_X_MARGIN_RATIO))
        ry2 = int(h * (1.0 - ROI_Y_MARGIN_RATIO))
        return (rx1, ry1, rx2, ry2)

    # ------------------------------------------------------------------
    # MODEL 1: SCRFD FACE DETECTION + LỌC VÙNG ROI + ƯU TIÊN TOP-K
    # ------------------------------------------------------------------
    def _distance2bbox(self, points, distance):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _distance2kps(self, points, distance):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def _nms(self, dets, thresh=0.4):
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def detect_faces_in_roi(self, img, input_size=(640, 640), max_faces=3):
        """
        1. Chạy SCRFD phát hiện toàn bộ khuôn mặt.
        2. Lọc bỏ khuôn mặt ngoài vùng ROI hoặc quá nhỏ (xa > 1.5m).
        3. Sắp xếp giảm dần theo diện tích và chỉ lấy Top-K khuôn mặt lớn nhất.
        """
        frame_h, frame_w = img.shape[:2]
        frame_area = float(frame_h * frame_w)
        rx1, ry1, rx2, ry2 = self.get_roi_bounds(img.shape)

        im_ratio = float(frame_h) / frame_w
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / frame_h
        resized_img = cv2.resize(img, (new_width, new_height))

        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        blob = cv2.dnn.blobFromImage(
            det_img, 1.0 / 128.0, input_size, (127.5, 127.5, 127.5), swapRB=True
        )
        net_outs = self.det_sess.run(None, {self.det_input_name: blob})

        scores_list, bboxes_list, kpss_list = [], [], []
        fmc = 3
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc] * stride
            kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_size[1] // stride
            width = input_size[0] // stride
            key = (height, width, stride)
            if key in self._center_cache:
                anchor_centers = self._center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                self._center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= DET_CONF_THRESHOLD)[0]
            bboxes = self._distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])

            kpss = self._distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            kpss_list.append(kpss[pos_inds])

        scores = np.vstack(scores_list)
        if len(scores) == 0:
            return []

        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self._nms(pre_det)
        det = pre_det[keep, :]
        kpss = kpss[order, :, :][keep, :, :]

        # Lọc theo ROI Zone & Kích thước tối thiểu (0.5m - 1.5m)
        valid_candidates = []
        for i in range(len(det)):
            x1, y1, x2, y2, score = det[i]
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            # Bỏ qua khuôn mặt ngoài vùng ROI (ngoài hành lang)
            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                continue
            # Bỏ qua khuôn mặt quá nhỏ (đứng quá xa > 1.5m)
            if (area / frame_area) < MIN_FACE_AREA_RATIO:
                continue

            valid_candidates.append({
                "bbox": det[i, :4],
                "score": float(score),
                "kps": kpss[i],
                "area": area
            })

        # Ưu tiên Top-K khuôn mặt có diện tích lớn nhất (đứng sát cửa nhất)
        valid_candidates.sort(key=lambda item: item["area"], reverse=True)
        return valid_candidates[:max_faces]

    # ------------------------------------------------------------------
    # MODEL 2: MINIFASNET ANTI-SPOOFING (Kiểm tra thực thể sống)
    # ------------------------------------------------------------------
    def _crop_with_scale(self, img, bbox, scale=2.7, out_w=80, out_h=80):
        src_h, src_w, _ = img.shape
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        scale = min((src_h - 1) / max(h, 1), min((src_w - 1) / max(w, 1), scale))

        new_w, new_h = w * scale, h * scale
        center_x, center_y = w / 2 + x1, h / 2 + y1

        left_top_x = max(0, int(center_x - new_w / 2))
        left_top_y = max(0, int(center_y - new_h / 2))
        right_bottom_x = min(src_w - 1, int(center_x + new_w / 2))
        right_bottom_y = min(src_h - 1, int(center_y + new_h / 2))

        cropped = img[left_top_y:right_bottom_y + 1, left_top_x:right_bottom_x + 1]
        if cropped.size == 0:
            return None
        return cv2.resize(cropped, (out_w, out_h))

    def check_liveness(self, img, bbox) -> float:
        patch = self._crop_with_scale(img, bbox, scale=2.7, out_w=80, out_h=80)
        if patch is None:
            return 0.0

        blob = patch.astype(np.float32)
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        logits = self.fas_sess.run(None, {self.fas_input_name: blob})[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return float(probs[1])

    # ------------------------------------------------------------------
    # CĂN CHỈNH MẶT (AFFINE) & MODEL 3: MOBILEFACENET (VECTOR 512-D)
    # ------------------------------------------------------------------
    def align_face(self, img, kps):
        M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARCFACE_DST, method=cv2.LMEDS)
        return cv2.warpAffine(img, M, (112, 112), borderValue=0.0)

    def extract_embedding(self, aligned_face) -> np.ndarray:
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        emb = self.rec_sess.run(None, {self.rec_input_name: blob})[0][0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb