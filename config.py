import os

# ================= CẤU HÌNH SERVER RENDER =================
SERVER_HTTP_URL = "https://face-attendance-server-j1bf.onrender.com/"
SERVER_WS_URL = "wss://face-attendance-server-j1bf.onrender.com/ws"

# ================= NGƯỠNG XỬ LÝ AI =================
DET_CONF_THRESHOLD = 0.55      # Ngưỡng phát hiện khuôn mặt (SCRFD)
LIVENESS_THRESHOLD = 0.70      # Ngưỡng người thật (MiniFASNet Index 1)
MATCH_THRESHOLD = 0.50         # Ngưỡng nhận diện danh tính (Cosine Similarity)

# ================= TỐI ƯU TÀI NGUYÊN BIÊN (ROI, TOP-K, TRACKING) =================
MAX_FACES_PER_FRAME = 3        # Top-K: Tối đa 3 khuôn mặt gần cửa nhất mỗi frame
ROI_X_MARGIN_RATIO = 0.12      # Cắt bỏ 12% rìa trái & phải (loại người đi ngoài hành lang)
ROI_Y_MARGIN_RATIO = 0.05      # Cắt bỏ 5% mép trên/dưới
MIN_FACE_AREA_RATIO = 0.008    # Diện tích mặt tối thiểu >= 0.8% khung hình (tương thích cả cam VGA/HD)

TRACK_IOU_THRESHOLD = 0.35     # Ngưỡng IoU ghép cặp khuôn mặt giữa 2 frame liên tiếp
TRACK_MAX_AGE = 15             # Số frame tối đa giữ Track_ID khi mất dấu tạm thời
CONSISTENT_FRAMES = 2          # Số frame xác nhận liên tiếp trước khi chốt danh tính

# ================= CHỐNG SPAM SERVER & MỞ KHÓA CA TIẾP THEO =================
IN_FLIGHT_LOCK_SECONDS = 15    # Khóa tạm 15s ngay lúc đang gửi request lên Server
SESSION_COOLDOWN_SECONDS = 900 # Khóa 15 phút (900s) sau khi điểm danh thành công ca hiện tại -> tự mở cho ca sau
IGNORED_COOLDOWN_SECONDS = 60  # Khóa 60s nếu quét ngoài giờ/chưa tới ca để tránh spam khi đứng chờ

# ================= ĐƯỜNG DẪN LƯU TRỮ CỤC BỘ =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
WEBCAM_REGISTRY_FILE = os.path.join(BASE_DIR, "webcams_registry.json")
CACHE_VECTORS_FILE = os.path.join(BASE_DIR, "local_vectors_cache.json")
OFFLINE_QUEUE_FILE = os.path.join(BASE_DIR, "offline_attendance_queue.json")
PUNCH_CACHE_FILE = os.path.join(BASE_DIR, "session_punch_cache.json")