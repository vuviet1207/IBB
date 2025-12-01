# process.py
import os
import cv2
import math
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import uuid
from ultralytics import YOLO
# =======================
# Model & I/O Config
# =======================
OUTPUT_FOLDER = "check"

# Load smile model (expects grayscale 32x32 input)
try:
    SMILE_MODEL_PATH = os.getenv("SMILE_MODEL_PATH", r"models\smilemain.h5")
    smile_model = load_model(SMILE_MODEL_PATH)  # Đảm bảo đường dẫn đúng
except Exception as e:
    print(f"[warn] Error loading smile_model: {e}")
    smile_model = None

# =======================
# Thresholds
# =======================

# ==== LEG (skin-exposure) ====
# Bề rộng dải kiểm tra quanh trục chân (tỉ lệ theo chiều dài đoạn hip→(ankle/knee))
LEG_BAND_WIDTH_RATIO = float(os.getenv("LEG_BAND_WIDTH_RATIO", "0.12"))
# Dùng lại ngưỡng nhận diện da có sẵn:
# - HSV_SKIN_LOOSE, YCRCB_SKIN (tạo mask da "loose")
# - HSV_SKIN_TIGHT (kiểm tra median nằm trong vùng "da thật")

def _poly_mask(h, w, pts):
    """Tạo mask nhị phân từ polygon pts (Nx2 int)."""
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, pts.astype(np.int32), 255)
    return m

def _leg_band_polygon(img_shape, p_start, p_end, width_ratio=LEG_BAND_WIDTH_RATIO):
    """
    Tạo hình chữ nhật xoay (polygon 4 điểm) bao quanh đoạn thẳng p_start→p_end
    với bề rộng = width_ratio * chiều dài đoạn.
    p_start, p_end: (x, y) theo pixel.
    """
    h, w = img_shape[:2]
    v = np.array(p_end, dtype=np.float32) - np.array(p_start, dtype=np.float32)
    L = np.linalg.norm(v)
    if L < 1e-3:
        return None
    d = v / L
    n = np.array([-d[1], d[0]], dtype=np.float32)  # pháp tuyến
    half_w = max(6.0, width_ratio * L * 0.5)

    p1 = np.array(p_start) + n * half_w
    p2 = np.array(p_end)   + n * half_w
    p3 = np.array(p_end)   - n * half_w
    p4 = np.array(p_start) - n * half_w
    pts = np.vstack([p1, p2, p3, p4])
    return np.clip(pts, [0, 0], [w - 1, h - 1]).astype(np.int32)

SHOULDER_LEVEL_THRESHOLD = float(os.getenv("SHOULDER_LEVEL_THRESHOLD", "0.015"))  # chênh lệch y-normalized ≤ 0.02
ARM_STRAIGHT_ANGLE_THRESHOLD = float(os.getenv("ARM_STRAIGHT_ANGLE_THRESHOLD", "150"))  # góc khuỷu ≥ 150°
THRESH_SMILE_CONF = float(os.getenv("THRESH_SMILE_CONF", "0.35"))  # ngưỡng xác suất cười

# Lưu ảnh crop khuôn mặt (dùng cho kiểm tra cười)
SAVE_SMILE_FACE_CROP = bool(int(os.getenv("SAVE_SMILE_FACE_CROP", "1")))
SMILE_FACE_CROP_NAME = os.getenv("SMILE_FACE_CROP_NAME", "face_smile_crop.jpg")

# Điểm bắt đầu dải chân: lùi từ HIP xuống phía KNEE
LEG_START_FROM_HIP_RATIO = float(os.getenv("LEG_START_FROM_HIP_RATIO", "0.4"))  # % độ dài HIP→KNEE
LEG_START_FROM_HIP_MINPX = int(os.getenv("LEG_START_FROM_HIP_MINPX", "8"))       # tối thiểu mấy pixel

# thêm ở phần config
MIN_FOOT_VEC_LEN_PX = int(os.getenv("MIN_FOOT_VEC_LEN_PX", "30"))  # độ dài A→T tối thiểu (px)
BORDER_FAIL_PX      = int(os.getenv("BORDER_FAIL_PX", "8"))        # cách mép ảnh < ngưỡng ⇒ fail

# ==== Tham số phát hiện lộ da ở VÙNG GIÀY ====
# Dải da "loose" cho mask nền (HSV ∩ YCrCb)
HSV_SKIN_LOOSE = (
    np.array([4, 30, 60], dtype=np.uint8),
    np.array([18, 255, 255], dtype=np.uint8),
)
YCRCB_SKIN = (
    np.array([0, 133, 77], dtype=np.uint8),
    np.array([255, 173, 127], dtype=np.uint8),
)
HSV_SKIN_TIGHT = (np.array([4, 57, 60], dtype=np.uint8),  np.array([18, 200, 255], dtype=np.uint8))
# Cửa sổ "tight" để xác nhận median HSV thật là da (giảm false positive)

# Tỉ lệ da tối thiểu trong crop giày để xét tiếp median & giao hình học
MIN_SKIN_AREA_SHOE_FULL = float(os.getenv("MIN_SKIN_AREA_SHOE_FULL", "0.15"))  # 15%
THRESH_SKIN_PANTS = float(os.getenv("THRESH_SKIN_PANTS", str(MIN_SKIN_AREA_SHOE_FULL)))
# Hình học bàn chân (ankle→toe)
MIN_VIS = float(os.getenv("MIN_VIS", "0.59"))
SCALE_LEN = float(os.getenv("SCALE_LEN", "1.35"))
WIDTH_RATIO = float(os.getenv("WIDTH_RATIO", "0.55"))

HAND_MIN_VIS = float(os.getenv("HAND_MIN_VIS", str(MIN_VIS)))  # có thể chỉnh riêng, mặc định = MIN_VIS
# =======================
# Fail-box & crop configs (có thể override qua ENV)
# =======================
# Fail-box margins (pixels)
MARGIN_FACE = int(os.getenv("MARGIN_FACE", "20"))
MARGIN_SHOULDER = int(os.getenv("MARGIN_SHOULDER", "50"))
MARGIN_ARM = int(os.getenv("MARGIN_ARM", "40"))
MARGIN_SHOE = int(os.getenv("MARGIN_SHOE", "12"))

# Độ dày nét & màu khung fail
FAIL_BOX_THICKNESS = int(os.getenv("FAIL_BOX_THICKNESS", "2"))
FAIL_BOX_COLOR = (0, 0, 255)  # đỏ (B, G, R)

# Cỡ chữ & độ dày chữ label BB
FAIL_TEXT_FONT_SCALE = float(os.getenv("FAIL_TEXT_FONT_SCALE", "0.9"))  # tăng/giảm cỡ chữ
FAIL_TEXT_THICKNESS  = int(os.getenv("FAIL_TEXT_THICKNESS", "2"))       # dày chữ

# Kích thước CROP GIÀY (bao vùng xét & khung hiển thị)
# Hộp crop quanh mắt cá: [px - HALF_W, py - TOP] → [px + HALF_W, py + BOTTOM]
SHOE_CROP_HALF_W = int(os.getenv("SHOE_CROP_HALF_W", "60"))
SHOE_CROP_TOP = int(os.getenv("SHOE_CROP_TOP", "30"))
SHOE_CROP_BOTTOM = int(os.getenv("SHOE_CROP_BOTTOM", "200"))

FEMALE_KNEE_EXT_RATIO = float(os.getenv("FEMALE_KNEE_EXT_RATIO", "0.15"))  # % độ dài knee→ankle
FEMALE_KNEE_EXT_MINPX = int(os.getenv("FEMALE_KNEE_EXT_MINPX", "8"))       # tối thiểu mấy pixel

# --- Male leg band tweak (đẩy điểm cuối lên khỏi mắt cá) ---
MALE_ANKLE_OFFSET_RATIO = float(os.getenv("MALE_ANKLE_OFFSET_RATIO", "0.15"))  # tỉ lệ độ dài knee→ankle
MALE_ANKLE_OFFSET_MINPX = int(os.getenv("MALE_ANKLE_OFFSET_MINPX", "8"))       # tối thiểu mấy pixel

# --- Shoe rectangle tweak (rút ngắn ở đầu gần mắt cá) ---
FOOT_RECT_CENTER_T = float(os.getenv("FOOT_RECT_CENTER_T", "0.5"))            # tâm rect dọc theo A→T (trước ~0.70)
SHOE_TRIM_AT_ANKLE_RATIO = float(os.getenv("SHOE_TRIM_AT_ANKLE_RATIO", "1")) # % chiều dài A→T cắt ở đầu mắt cá

# --- Extra horizontal shift (dịch theo trục X màn hình) ---
FOOT_RECT_HSHIFT_PX     = float(os.getenv("FOOT_RECT_HSHIFT_PX", "15"))   # px
FOOT_RECT_HSHIFT_RATIO  = float(os.getenv("FOOT_RECT_HSHIFT_RATIO", "0")) # tỉ lệ theo bề rộng ảnh (0..1)

# =======================
# YOLO scollar (cổ áo)
# =======================

SCOLLAR_MODEL_PATH = os.getenv("SCOLLAR_MODEL_PATH", r"models\collar.pt")
SCOLLAR_CONF_THRES = float(os.getenv("SCOLLAR_CONF_THRES", "0.70"))  # ngưỡng pass

# Nhóm tên lớp POSITIVE (có cổ) và NEGATIVE (không cổ)
SCOLLAR_POS_CLASS_NAMES = os.getenv("SCOLLAR_CLASS_NAMES", "scollar,collar,shirt_collar")
SCOLLAR_NEG_CLASS_NAMES = os.getenv("SCOLLAR_NEG_CLASS_NAMES", "no_collar,nocollar,no_scollar")

# Lưu crop NGƯỜI để check cổ áo (1 file cố định, lần sau ghi đè)
SAVE_SCOLLAR_CROP  = bool(int(os.getenv("SAVE_SCOLLAR_CROP", "1")))
SCOLLAR_CROP_NAME  = os.getenv("SCOLLAR_CROP_NAME", "scollar_person_crop.jpg")

# Ngưỡng IoU để coi là "đè nhau / cắt nhau" giữa collar và no_collar
SCOLLAR_CONFLICT_IOU = float(os.getenv("SCOLLAR_CONFLICT_IOU", "0.3"))

try:
    scollar_model = YOLO(SCOLLAR_MODEL_PATH) if YOLO is not None else None
except Exception as e:
    print(f"[warn] Error loading scollar YOLO: {e}")
    scollar_model = None

# Điều chỉnh 4 hướng crop người cho cổ áo (tính theo tỉ lệ kích thước bbox người ban đầu)
# dương = nới rộng thêm, âm = cắt bớt vào trong
SCOLLAR_CROP_TOP_RATIO=0.05
SCOLLAR_CROP_BOTTOM_RATIO=0.4
SCOLLAR_CROP_LEFT_RATIO=0.05
SCOLLAR_CROP_RIGHT_RATIO=0.05
# ==== Collar fail-box tweak ====
# Dịch ô fail lên trên (tính theo % chiều cao bbox vai) và thu hẹp bề ngang (% mỗi bên)
SCOLLAR_FAIL_SHIFT_UP_RATIO      = float(os.getenv("SCOLLAR_FAIL_SHIFT_UP_RATIO", "0.45"))  # 10% cao lên
SCOLLAR_FAIL_SHRINK_X_RATIO      = float(os.getenv("SCOLLAR_FAIL_SHRINK_X_RATIO", "0.30"))  # thu hẹp 20% mỗi bên
SCOLLAR_FAIL_MIN_WIDTH_PX        = int(os.getenv("SCOLLAR_FAIL_MIN_WIDTH_PX", "60"))        # tối thiểu để không quá bé
SCOLLAR_FAIL_EXTRA_TOP_MARGIN_PX = int(os.getenv("SCOLLAR_FAIL_EXTRA_TOP_MARGIN_PX", "10")) # nới thêm phần trên

# ==== EYEBROW–HAIR (mới) ====
HAIR_UP_PX           = int(os.getenv("HAIR_UP_PX", "18"))     # cao của forehead band
HAIR_EXPAND_PX       = int(os.getenv("HAIR_EXPAND_PX", "4"))  # nới ngang band
HAIR_MIN_THICK_PX    = float(os.getenv("HAIR_MIN_THICK_PX", "0.5"))
HAIR_MIN_AREA_RATIO  = float(os.getenv("HAIR_MIN_AREA_RATIO", "0.00001"))
HAIR_MORPH_ITERS     = int(os.getenv("HAIR_MORPH_ITERS", "1"))
ROI_TOUCH_DILATE     = int(os.getenv("ROI_TOUCH_DILATE", "5"))

# Ngưỡng fallback khi không đủ mẫu adaptive
DARK_V_THRESH        = int(os.getenv("DARK_V_THRESH", "110"))
DARK_GRAY_THRESH     = int(os.getenv("DARK_GRAY_THRESH", "55"))
# ==== FaceMesh crop (vuông – ổn định) ====
FACE_MESH_CROP_SCALE = float(os.getenv("FACE_MESH_CROP_SCALE", "1.6"))
FACE_MESH_MIN_SIDE   = int(os.getenv("FACE_MESH_MIN_SIDE", "220"))
FACE_MESH_MAX_SIDE   = int(os.getenv("FACE_MESH_MAX_SIDE", "640"))

# ==== Eyebrow ROI tweak ====
EYEBROW_SHRINK_PX    = int(os.getenv("EYEBROW_SHRINK_PX", "2"))
EYEBROW_UPSHIFT_PX   = int(os.getenv("EYEBROW_UPSHIFT_PX", "3"))
EYEBROW_TOP_GROW_PX  = int(os.getenv("EYEBROW_TOP_GROW_PX", "3"))
EYEBROW_SIDE_TOP_RATIO = float(os.getenv("EYEBROW_SIDE_TOP_RATIO", "0.67"))
INNER_RIM_GUARD_PX     = int(os.getenv("INNER_RIM_GUARD_PX", "6"))
INNER_RIM_GUARD_RATIO  = float(os.getenv("INNER_RIM_GUARD_RATIO", "0.18"))

# Lưu crop mặt phục vụ debug lông mày
DEBUG_FACE_SAVE        = bool(int(os.getenv("DEBUG_FACE_SAVE", "0")))
FACE_CROP_RAW_NAME     = os.getenv("FACE_CROP_RAW_NAME", "face_facemesh_raw.jpg")
FACE_CROP_OVERLAY_NAME = os.getenv("FACE_CROP_OVERLAY_NAME", "face_facemesh_overlay.jpg")

# Kernel lọc morphology cho mask tóc (thiếu)
HAIR_MORPH_K = int(os.getenv("HAIR_MORPH_K", "3"))

def _clean_dark_mask(mask, k=HAIR_MORPH_K, iters=HAIR_MORPH_ITERS):
    # mở rồi đóng để khử nhiễu các đốm nhỏ
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
    return mask

# ==== EAR YOLO (phát hiện 2 tai) ====
EAR_MODEL_PATH = os.getenv("EAR_MODEL_PATH", r"models\ear.pt")  # đặt ear.pt ở đây
EAR_CONF_THRES = float(os.getenv("EAR_CONF_THRES", "0.50"))     # ngưỡng 50%
EAR_SAVE_CROP  = bool(int(os.getenv("EAR_SAVE_CROP", "1")))     # 1 = luôn lưu crop
EAR_CROP_NAME  = os.getenv("EAR_CROP_NAME", "face_ear_crop.jpg")
EAR_YOLO_IMGSZ = int(os.getenv("EAR_YOLO_IMGSZ", "640"))
# CROP TAI TO HƠN CROP CƯỜI
EAR_FACE_EXTRA_MARGIN_X = int(os.getenv("EAR_FACE_EXTRA_MARGIN_X", "30"))  # nới ngang thêm (px)
EAR_FACE_EXTRA_MARGIN_Y = int(os.getenv("EAR_FACE_EXTRA_MARGIN_Y", "40"))  # nới dọc thêm (px)
try:
    ear_model = YOLO(EAR_MODEL_PATH) if YOLO is not None else None
except Exception as e:
    print(f"[warn] Error loading ear YOLO: {e}")
    ear_model = None

# =======================
# MediaPipe
# =======================
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh  # hiện chưa dùng, để sau dễ mở rộng


# =======================
# Helpers
# =======================
def _check_required_landmarks(pose_landmarks, vis_thresh=0.5):
    """
    Kiểm tra 3 nhóm landmark chính:
      - upper: thân trên (vai, khuỷu tay, cổ tay)
      - lower: thân dưới (hông, gối, mắt cá, gót, mũi chân)
      - face:  khuôn mặt (mũi, mắt, tai)

    Nếu BẤT KỲ nhóm nào thiếu / visibility < vis_thresh / tọa độ out-of-range
    thì coi như KHÔNG ĐỦ POSE -> trả về overall=False.
    """
    lm = pose_landmarks.landmark

    upper_ids = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value,
        mp_pose.PoseLandmark.RIGHT_WRIST.value,
    ]
    lower_ids = [
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        mp_pose.PoseLandmark.LEFT_HEEL.value,
        mp_pose.PoseLandmark.RIGHT_HEEL.value,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
    ]
    face_ids = [
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.LEFT_EYE.value,
        mp_pose.PoseLandmark.RIGHT_EYE.value,
        mp_pose.PoseLandmark.LEFT_EAR.value,
        mp_pose.PoseLandmark.RIGHT_EAR.value,
    ]

    groups = {
        "upper": upper_ids,
        "lower": lower_ids,
        "face":  face_ids,
    }

    groups_ok = {}
    missing = {}

    def _lm_bad(lm_one):
        # visibility thấp hoặc tọa độ ra ngoài [0,1] thì coi là "không thấy"
        if lm_one.visibility is None or np.isnan(lm_one.visibility):
            return True
        if lm_one.visibility < vis_thresh:
            return True
        if not (0.0 <= lm_one.x <= 1.0) or not (0.0 <= lm_one.y <= 1.0):
            return True
        return False

    # Ngưỡng tỉ lệ landmark "tốt" tối thiểu cho từng vùng
    # Có thể override bằng ENV nếu cần tinh chỉnh
    upper_min_ratio = float(os.getenv("UPPER_MIN_OK_RATIO", "0.3"))
    lower_min_ratio = float(os.getenv("LOWER_MIN_OK_RATIO", "0.3"))
    face_min_ratio  = float(os.getenv("FACE_MIN_OK_RATIO",  "0.3"))

    ratios = {}

    for gname, id_list in groups.items():
        bad_idx = []
        for idx in id_list:
            try:
                if _lm_bad(lm[idx]):
                    bad_idx.append(idx)
            except Exception:
                # Nếu index không tồn tại trong list landmark -> cũng coi là thiếu
                bad_idx.append(idx)

        missing[gname] = bad_idx

        total = len(id_list)
        good  = total - len(bad_idx)
        ratio = float(good) / float(total) if total > 0 else 0.0
        ratios[gname] = ratio

        if gname == "upper":
            min_ratio = upper_min_ratio
        elif gname == "lower":
            min_ratio = lower_min_ratio
        else:  # "face"
            min_ratio = face_min_ratio

        # Nhóm OK nếu:
        #  - có ít nhất 1 landmark tốt
        #  - và tỉ lệ landmark tốt >= min_ratio
        groups_ok[gname] = (good > 0 and ratio >= min_ratio)

    overall_ok = all(groups_ok.values())

    # Log ra console cho dễ debug
    if not overall_ok:
        print("[pose_visibility] NOT ENOUGH landmarks:")
        for gname in ["upper", "lower", "face"]:
            print(
                f"  - {gname}: ok={groups_ok[gname]}, "
                f"ratio={ratios.get(gname, 0.0):.2f}, "
                f"missing_ids={missing[gname]}"
            )

    return overall_ok, groups_ok, missing

    # Log ra console cho dễ debug
    if not overall_ok:
        print("[pose_visibility] NOT ENOUGH landmarks:")
        for gname in ["upper", "lower", "face"]:
            print(f"  - {gname}: ok={groups_ok[gname]}, missing_ids={missing[gname]}")

    return overall_ok, groups_ok, missing

def _expand_box_xy(box, img_shape, margin_x, margin_y):
    """
    Nới rộng box theo 2 hướng X/Y độc lập, rồi clip vào ảnh.
    """
    if box is None:
        return None
    h, w = img_shape[:2]
    x1 = box["x1"] - int(margin_x)
    y1 = box["y1"] - int(margin_y)
    x2 = box["x2"] + int(margin_x)
    y2 = box["y2"] + int(margin_y)
    return _clip_box_raw(x1, y1, x2, y2, w, h)

def detect_ears_yolo(src_image, face_box, conf_thres=EAR_CONF_THRES, min_y=None):
    """
    Dùng YOLO ear.pt trên crop khuôn mặt:
    - Crop theo face_box (dùng riêng cho EAR, có thể to hơn SMILE).
    - Lưu crop vào OUTPUT_FOLDER/EAR_CROP_NAME (ghi đè).
    - Chỉ tính các bbox:
        + Nằm BÊN DƯỚI chân mày (center_y_global >= min_y, nếu min_y != None)
    - Nếu có >= 2 bbox với conf >= conf_thres ⇒ 'pass'
      Ngược lại ⇒ 'fail'.
    Trả: (status, debug_dict)
    """
    debug = {
        "model_loaded": ear_model is not None,
        "face_box": face_box,
        "saved_crop": None,
        "num_dets": 0,
        "num_valid": 0,
        "all_confs": [],
        "valid_confs": [],
        "conf_thres": conf_thres,
        "reason": None,
        "boxes_crop": [],   # bbox trong hệ toạ độ CROP (sau khi lọc)
        "boxes_global": [], # bbox đã map ra toàn ảnh (sau khi lọc)
        "min_y": min_y,     # ngưỡng dưới chân mày (global y)
    }

    if ear_model is None:
        debug["reason"] = "model_not_loaded"
        print("[ear] MISSING: YOLO ear model not loaded.")
        return "missing", debug

    if face_box is None:
        debug["reason"] = "no_face_box"
        print("[ear] FAIL: no face box for ear detection.")
        return "missing", debug

    # Crop từ ảnh gốc (không bị vẽ overlay)
    crop = _crop_box(src_image, face_box)
    if crop is None or crop.size == 0:
        debug["reason"] = "empty_crop"
        print("[ear] FAIL: empty face crop.")
        return "missing", debug

    # Lưu crop (ghi đè mỗi lần)
    if EAR_SAVE_CROP:
        out_path = os.path.join(OUTPUT_FOLDER, EAR_CROP_NAME)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if cv2.imwrite(out_path, crop):
            debug["saved_crop"] = out_path

    # Chạy YOLO ear trên crop
    try:
        r = ear_model.predict(
            source=crop,
            imgsz=EAR_YOLO_IMGSZ,
            conf=0.001,      # lấy hết, rồi tự lọc theo EAR_CONF_THRES
            iou=0.5,
            verbose=False
        )[0]
    except Exception as e:
        debug["reason"] = f"infer_error: {e}"
        print(f"[ear] FAIL: inference error: {e}")
        return "fail", debug

    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        debug["reason"] = "no_boxes"
        print("[ear] --- No ear boxes ---")
        return "fail", debug

    # Lấy cls/conf/xyxy trong CROP
    try:
        confs_all = boxes.conf.detach().cpu().numpy().tolist()
    except Exception:
        confs_all = []

    try:
        xyxy_all = boxes.xyxy.detach().cpu().numpy().tolist()
    except Exception:
        xyxy_all = []

    # ===== LỌC THEO VỊ TRÍ DƯỚI CHÂN MÀY (center_y_global >= min_y) =====
    h_img, w_img = src_image.shape[:2]
    fx1, fy1 = face_box["x1"], face_box["y1"]

    boxes_crop = []
    confs = []
    boxes_global = []

    for b, c in zip(xyxy_all, confs_all):
        # toạ độ global của bbox
        gx1 = b[0] + fx1
        gy1 = b[1] + fy1
        gx2 = b[2] + fx1
        gy2 = b[3] + fy1

        # nếu có min_y (đáy lông mày) thì chỉ giữ bbox dưới đó
        if min_y is not None:
            cy = 0.5 * (gy1 + gy2)
            if cy < float(min_y):
                continue

        g = _clip_box_raw(gx1, gy1, gx2, gy2, w_img, h_img)
        if g is None:
            continue

        boxes_crop.append(b)
        confs.append(float(c))
        boxes_global.append(g)

    # nếu sau khi lọc không còn bbox nào -> fail
    debug["boxes_crop"] = boxes_crop
    debug["boxes_global"] = boxes_global
    debug["num_dets"] = len(confs)
    debug["all_confs"] = confs

    if len(confs) == 0:
        debug["reason"] = "no_boxes_after_brow_filter"
        print("[ear] --- No ear boxes after filtering under eyebrows ---")
        return "fail", debug

    # ===== CHỈ XÉT 2 BBOX CÓ CONF CAO NHẤT =====
    if len(confs) < 2:
        # Không đủ 2 bbox để xét tai
        debug["num_valid"] = 0
        debug["valid_confs"] = []
        debug["reason"] = "not_enough_boxes_for_top2"
        print(
            f"[ear] fail | not enough boxes (<2) after brow filter"
            f" | all_confs={[f'{c:.3f}' for c in confs]}"
            f" | min_y={min_y}"
        )
        return "fail", debug

    # Sắp xếp index theo conf giảm dần
    idx_sorted = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
    top2_idx = idx_sorted[:2]
    top2_confs = [float(confs[i]) for i in top2_idx]

    # Chỉ 2 bbox này mới được tính để quyết định pass/fail
    valid_top2 = [c for c in top2_confs if c >= conf_thres]
    debug["num_valid"] = len(valid_top2)
    debug["valid_confs"] = valid_top2  # chỉ log 2 bbox được xét

    status = "pass" if len(valid_top2) == 2 else "fail"
    debug["reason"] = "top2_above_threshold" if status == "pass" else "top2_not_enough_or_below_threshold"

    print(
        f"[ear] {status}"
        f" | top2_confs={[f'{c:.3f}' for c in top2_confs]} (>= {conf_thres})"
        f" | all_confs={[f'{c:.3f}' for c in confs]}"
        f" | min_y={min_y}"
    )

    return status, debug




def _tweak_collar_fail_box(box, img_shape):
    """
    - Dịch box lên trên một chút (SHIFT_UP_RATIO * height)
    - Thu hẹp bề ngang (SHRINK_X_RATIO từ mỗi bên)
    - Nới phần trên thêm vài px để ô “cao” hơn ở vùng cổ
    """
    if box is None:
        return None
    h, w = img_shape[:2]
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)

    # Thu hẹp ngang
    shrink_each = int(bw * SCOLLAR_FAIL_SHRINK_X_RATIO)
    nx1 = x1 + shrink_each
    nx2 = x2 - shrink_each

    # Dịch lên
    shift_up = int(bh * SCOLLAR_FAIL_SHIFT_UP_RATIO)
    ny1 = max(0, y1 - shift_up - SCOLLAR_FAIL_EXTRA_TOP_MARGIN_PX)
    ny2 = y2 - shift_up

    # Đảm bảo không quá nhỏ
    if (nx2 - nx1) < SCOLLAR_FAIL_MIN_WIDTH_PX:
        cx = (nx1 + nx2) // 2
        nx1 = max(0, cx - SCOLLAR_FAIL_MIN_WIDTH_PX // 2)
        nx2 = min(w - 1, nx1 + SCOLLAR_FAIL_MIN_WIDTH_PX)

    return _clip_box_raw(nx1, ny1, nx2, ny2, w, h)

def get_facemesh_box(image, pose_landmarks, scale=FACE_MESH_CROP_SCALE, min_side=FACE_MESH_MIN_SIDE, max_side=FACE_MESH_MAX_SIDE):
    if pose_landmarks is None:
        return None
    h, w = image.shape[:2]
    lm = pose_landmarks.landmark
    try:
        nose      = _lm_xy(lm[mp_pose.PoseLandmark.NOSE], w, h)
        left_ear  = _lm_xy(lm[mp_pose.PoseLandmark.LEFT_EAR], w, h)
        right_ear = _lm_xy(lm[mp_pose.PoseLandmark.RIGHT_EAR], w, h)
    except Exception:
        return None

    ear_dist = float(np.linalg.norm(left_ear - right_ear))
    if ear_dist < 1.0:
        return None

    side = int(round(scale * ear_dist))
    side = max(min_side, min(max_side, side))

    cx, cy = int(round(nose[0])), int(round(nose[1]))
    half = side // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + side, y1 + side
    return _clip_box_raw(x1, y1, x2, y2, w, h)

def detect_face_mesh_landmarks_on_crop(image, crop_box):
    if crop_box is None:
        return None
    x1, y1, x2, y2 = crop_box["x1"], crop_box["y1"], crop_box["x2"], crop_box["y2"]
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    try:
        with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as fm:
            res = fm.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"[warn] MediaPipe FaceMesh (crop) error: {e}")
        return None
    if not res.multi_face_landmarks:
        return None

    # Map về toạ độ normalized GLOBAL giống process.py
    h, w = image.shape[:2]
    ch, cw = crop.shape[:2]
    lms = res.multi_face_landmarks[0].landmark
    mapped = []
    for lm in lms:
        px = x1 + lm.x * cw
        py = y1 + lm.y * ch
        class _LM: pass
        lm2 = _LM()
        lm2.x = px / w
        lm2.y = py / h
        mapped.append(lm2)
    return mapped



# ===== Face Mesh eyebrow indices (MediaPipe 468) =====
LEFT_BROW_UP   = [70, 63, 105, 66, 107]
LEFT_BROW_LOW  = [55, 65, 52, 53, 46]
RIGHT_BROW_UP  = [336, 296, 334, 293, 300]
RIGHT_BROW_LOW = [285, 295, 282, 283, 276]

def _to_px(pt, w, h):
    return np.array([pt.x * w, pt.y * h], dtype=np.float32)

def _extend_poly_upwards(poly, img_shape, grow_px=2):
    """
    Mở rộng polygon theo HƯỚNG LÊN TRÊN thêm grow_px (px).
    Cách làm: tạo mask từ polygon -> dịch mask lên grow_px -> union với mask gốc -> lấy contour/hull.
    """
    if poly is None or grow_px <= 0:
        return poly

    h, w = img_shape[:2]
    # mask gốc
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, poly.astype(np.int32), 255)

    # dịch mask lên trên grow_px (y giảm)
    shifted = np.zeros_like(mask)
    if grow_px < h:
        shifted[0:h-grow_px, :] = mask[grow_px:h, :]
    # union để “nở” phần phía trên
    grown = cv2.bitwise_or(mask, shifted)

    # lấy contour lớn nhất -> convex hull
    cnts, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return poly
    biggest = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(biggest).reshape(-1, 2)
    return hull.astype(np.int32)

def _convex_poly_from_indices(lms, idx_list, w, h, expand_px=2):
    """Lấy điểm theo index → convex hull → nới nhẹ ra ngoài (expand_px)."""
    if lms is None or len(idx_list) == 0:
        return None
    pts = np.array([_to_px(lms[i], w, h) for i in idx_list], dtype=np.float32)
    if len(pts) < 3:
        return None
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)

    c = hull.mean(axis=0, keepdims=True)
    dirs = hull - c
    norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-6
    unit = dirs / norms
    expanded = (hull + unit * float(expand_px)).astype(np.int32)
    return np.clip(expanded, [0,0], [w-1, h-1])

def eyebrow_polygons_from_facemesh(face_lms, img_shape, expand_px=2):
    """
    Tạo 2 polygon (left, right) cho lông mày.
    Ở đây ta:
      - Co hẹp ROI bằng expand âm = -EYEBROW_SHRINK_PX
      - Dịch ROI lên trên EYEBROW_UPSHIFT_PX
    """
    if face_lms is None:
        return None, None
    h, w = img_shape[:2]

    left_idx  = LEFT_BROW_UP + LEFT_BROW_LOW
    right_idx = RIGHT_BROW_UP + RIGHT_BROW_LOW

    # Co hẹp: dùng expand âm
    shrink = int(EYEBROW_SHRINK_PX)
    L_poly = _convex_poly_from_indices(face_lms, left_idx,  w, h, expand_px=-shrink)
    R_poly = _convex_poly_from_indices(face_lms, right_idx, w, h, expand_px=-shrink)

    # Dịch lên trên: y -= EYEBROW_UPSHIFT_PX (clip về [0, h-1])
    up = int(EYEBROW_UPSHIFT_PX)
    if L_poly is not None:
        L_poly = L_poly.copy()
        L_poly[:, 1] = np.clip(L_poly[:, 1] - up, 0, h - 1)
    if R_poly is not None:
        R_poly = R_poly.copy()
        R_poly[:, 1] = np.clip(R_poly[:, 1] - up, 0, h - 1)

    # MỚI: nới rộng ROI theo hướng LÊN TRÊN
    grow = int(EYEBROW_TOP_GROW_PX)
    if grow > 0:
        if L_poly is not None:
            L_poly = _extend_poly_upwards(L_poly, img_shape, grow_px=grow)
        if R_poly is not None:
            R_poly = _extend_poly_upwards(R_poly, img_shape, grow_px=grow)

    return L_poly, R_poly



def _dilate_mask(mask, k=3, iters=1):
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iters)


def check_eyebrow_hair_overlap(image, face_lms, face_box=None):
    """
    FAIL nếu có vùng tối (đen/nâu đậm) CHẠM VÀO ROI LÔNG MÀY
    Chỉ chấp nhận các tiếp xúc:
      - TỪ TRÊN (top), hoặc
      - TỪ HAI BÊN (sides) nhưng chỉ phần 2/3 trên.
    KHÔNG TÍNH:
      - Tiếp xúc từ DƯỚI (bottom),
      - Mép TRONG gần sống mũi (inner rim),
      - Điểm tối nằm hoàn toàn bên trong ROI.
    Trả: (overall, detail_dict, (L_poly, R_poly))
    """
    if face_lms is None:
        return "missing", {}, (None, None)

    h, w = image.shape[:2]
    L_poly, R_poly = eyebrow_polygons_from_facemesh(face_lms, image.shape, expand_px=2)
    if L_poly is None and R_poly is None:
        return "missing", {}, (L_poly, R_poly)

    # 1) mask TỐI (nhận tóc/miền tối) = (gray < thr) OR (HSV.V < thr)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dark_gray = (gray < DARK_GRAY_THRESH).astype(np.uint8) * 255
    dark_v    = (hsv[:, :, 2] < DARK_V_THRESH).astype(np.uint8) * 255
    dark_any  = cv2.bitwise_or(dark_gray, dark_v)
    dark_any  = _clean_dark_mask(dark_any, k=HAIR_MORPH_K, iters=HAIR_MORPH_ITERS)

    # Trục giữa khuôn mặt (để bỏ "mép trong")
    if face_box is not None:
        cx_mid = (face_box["x1"] + face_box["x2"]) // 2
    else:
        cx_mid = w // 2

    def _touch_from_allowed_dirs(poly, side):
        if poly is None:
            return {"touch_px": 0, "has_touch": False, "num_comp": 0, "max_comp_area": 0}

        roi = np.zeros_like(dark_any, dtype=np.uint8)
        cv2.fillConvexPoly(roi, poly, 255)

        # Vành ngoài của ROI
        rim = _dilate_mask(roi, k=ROI_TOUCH_DILATE, iters=1)
        rim_outside = cv2.bitwise_and(rim, cv2.bitwise_not(roi))

        xmin, ymin = int(poly[:,0].min()), int(poly[:,1].min())
        xmax, ymax = int(poly[:,0].max()), int(poly[:,1].max())
        yy, xx = np.indices(dark_any.shape, dtype=np.int32)

        # CHỈ cho TOP + HAI BÊN; LOẠI BOTTOM
        top_allow   = ((yy < ymin).astype(np.uint8) * 255)
        left_allow  = ((xx < xmin).astype(np.uint8) * 255)
        right_allow = ((xx > xmax).astype(np.uint8) * 255)

        # Sides chỉ 2/3 trên
        roi_h = max(1, ymax - ymin)
        y_cut = int(ymin + EYEBROW_SIDE_TOP_RATIO * roi_h)
        side_top_mask = ((yy <= y_cut).astype(np.uint8) * 255)
        left_allow_top  = cv2.bitwise_and(left_allow,  side_top_mask)
        right_allow_top = cv2.bitwise_and(right_allow, side_top_mask)

        allowed_dirs = cv2.bitwise_or(top_allow, cv2.bitwise_or(left_allow_top, right_allow_top))
        allowed_rim  = cv2.bitwise_and(rim_outside, allowed_dirs)

        # BỎ “mép trong” (gần sống mũi)
        roi_w   = max(1, xmax - xmin)
        guard_w = int(max(int(INNER_RIM_GUARD_PX), INNER_RIM_GUARD_RATIO * roi_w))
        if side == "left":
            inner_strip = ((xx >= (xmax - guard_w)).astype(np.uint8) * 255)
        else:
            inner_strip = ((xx <= (xmin + guard_w)).astype(np.uint8) * 255)
        allowed_rim = cv2.bitwise_and(allowed_rim, cv2.bitwise_not(inner_strip))

        # Giao với vùng tối
        touch = cv2.bitwise_and(dark_any, allowed_rim)

        touch_px = int(np.count_nonzero(touch))
        lab, labels, stats, _ = cv2.connectedComponentsWithStats((touch > 0).astype(np.uint8), connectivity=8)
        num_comp = max(0, lab - 1)
        max_area = 0 if num_comp == 0 else int(np.max(stats[1:, cv2.CC_STAT_AREA]))
        return {"touch_px": touch_px, "has_touch": (touch_px > 0), "num_comp": num_comp, "max_comp_area": max_area}

    Ls = _touch_from_allowed_dirs(L_poly, side="left")
    Rs = _touch_from_allowed_dirs(R_poly, side="right")

    def _side_status(poly, touch_dict):
        if poly is None:
            return "missing"
        return "fail" if touch_dict["has_touch"] else "pass"

    left_status  = _side_status(L_poly, Ls)
    right_status = _side_status(R_poly, Rs)

    if "fail" in (left_status, right_status):
        overall = "fail"
    elif "missing" in (left_status, right_status):
        overall = "missing"
    else:
        overall = "pass"

    detail = {
        "eyebrow_left":  left_status,
        "eyebrow_right": right_status,
        "left_brow": left_status,
        "right_brow": right_status,
        "left_touch_px": Ls["touch_px"],
        "right_touch_px": Rs["touch_px"],
        "left_num_comp": Ls["num_comp"],
        "right_num_comp": Rs["num_comp"],
        "left_max_comp_area": Ls["max_comp_area"],
        "right_max_comp_area": Rs["max_comp_area"],
        "dark_gray_thresh": int(DARK_GRAY_THRESH),
        "dark_v_thresh": int(DARK_V_THRESH),
        "roi_dilate": int(ROI_TOUCH_DILATE),
        "rule": "outside-only & (top OR sides), exclude bottom, exclude inner rim near midline",
    }

    # DEBUG overlay ROI + mask tối
    if DEBUG_FACE_SAVE:
        if L_poly is not None: cv2.polylines(image, [L_poly], True, (255, 255, 0), 2)
        if R_poly is not None: cv2.polylines(image, [R_poly], True, (255, 255, 0), 2)
        image[:] = overlay_mask_on_image(image, dark_any, color=(255, 0, 255), alpha=0.35)
        cv2.line(image, (cx_mid, 0), (cx_mid, h-1), (0, 255, 255), 1)

    return overall, detail, (L_poly, R_poly)
def annotate_ears(image, results):
    """
    Vẽ khung tổng cho tai:
      - KHÔNG vẽ bbox từng tai nữa.
      - Chỉ vẽ 1 khung lớn quanh ear_face_box khi ear != 'pass'
        (fail/missing).
    """
    out = image.copy()
    ear_status = results.get("ear", "missing")
    ear_face_box = results.get("ear_face_box")

    # Chỉ vẽ khi không pass
    if ear_face_box is not None and ear_status != "pass":
        out = _draw_labeled_box(out, ear_face_box, f"ear: {ear_status}", color=FAIL_BOX_COLOR)

    return out



def annotate_fail_eyebrows(image, brow_polys, results):
    """
    Vẽ ô fail cho lông mày:
      - Nếu overall == "pass" -> không vẽ gì.
      - Nếu overall == "fail" -> bên nào có polygon thì vẽ ĐỎ CẢ 2 BÊN.
      - Nếu overall "missing" -> giữ logic cũ: chỉ vẽ bên có status != "pass".
    """
    out = image.copy()
    overall = results.get("eyebrow_hair", "missing")
    if overall == "pass":
        return out

    left_poly, right_poly = brow_polys if brow_polys is not None else (None, None)

    def _bb_from_poly(poly):
        x1 = int(poly[:, 0].min()); y1 = int(poly[:, 1].min())
        x2 = int(poly[:, 0].max()); y2 = int(poly[:, 1].max())
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    for side, poly, key in [
        ("left",  left_poly,  "eyebrow_left"),
        ("right", right_poly, "eyebrow_right"),
    ]:
        if poly is None:
            continue

        # Nếu overall FAIL: ép cả 2 bên có polygon thành fail để vẽ đỏ symmetry
        if overall == "fail":
            status = "fail"
        else:
            # Trường hợp overall "missing" -> giữ behavior cũ
            status = results.get(key, overall)
            if status == "pass":
                continue

        # 1) Viền polygon mảnh (để đối chiếu vùng)
        cv2.polylines(out, [poly.astype(np.int32)], isClosed=True,
                      color=FAIL_BOX_COLOR, thickness=1)

        # 2) Ô fail = bounding box của polygon
        bb = _bb_from_poly(poly.astype(np.int32))
        label = f"{key}: {status}"
        out = _draw_labeled_box(out, bb, label)

    return out


def _iou_xyxy(box1, box2):
    """
    box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter + 1e-9
    return float(inter / union)

def _save_face_crop(image, box, out_path):
    """Lưu crop khuôn mặt theo box {x1,y1,x2,y2}. Trả True/False."""
    if box is None:
        return False
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return cv2.imwrite(out_path, crop)

# ==== DEBUG (shoe) ====
DEBUG_SHOE = bool(int(os.getenv("DEBUG_SHOE", "0")))  # 1: bật overlay debug vùng da bàn chân
DEBUG_SAVE_EACH_SHOE = bool(int(os.getenv("DEBUG_SAVE_EACH_SHOE", "0")))  # lưu ảnh debug từng bên

# Cửa sổ "tight" để xác nhận median HSV thật là da (dùng riêng cho giày)
HSV_SKIN_TIGHT_SHOE = (
    np.array([4, 38, 60], dtype=np.uint8),
    np.array([16, 160, 255], dtype=np.uint8),
)

def overlay_mask_on_image(base_img, mask, color=(0, 255, 0), alpha=0.5):
    """Vẽ mask (0/255) đổ màu bán trong suốt lên base_img."""
    if base_img is None or mask is None:
        return base_img
    overlay = base_img.copy()
    color_img = np.zeros_like(base_img, dtype=np.uint8)
    color_img[:, :] = color
    mask_bool = (mask > 0).astype(np.uint8)
    color_mask = cv2.bitwise_and(color_img, color_img, mask=mask_bool)
    overlay = cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0)
    return overlay
def _skin_stats_in_polygon_full(bgr_img, poly_pts):
    """
    Tính tỉ lệ da và median HSV ngay trên ảnh gốc, GIỚI HẠN bởi polygon (foot-rectangle).
    Trả (skin_ratio, median_hsv, median_is_skin_tight, area_poly, skin_cnt).
    """
    if poly_pts is None:
        return 0.0, None, False, 0, 0

    h, w = bgr_img.shape[:2]
    poly_mask = _poly_mask(h, w, poly_pts)

    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)

    m_h = cv2.inRange(hsv,  HSV_SKIN_LOOSE[0],  HSV_SKIN_LOOSE[1])
    m_y = cv2.inRange(ycc,  YCRCB_SKIN[0],     YCRCB_SKIN[1])
    m   = _clean_mask(cv2.bitwise_and(m_h, m_y), k=3, iters=1)

    m_roi = cv2.bitwise_and(m, poly_mask)

    area_poly = int(np.count_nonzero(poly_mask))
    if area_poly <= 0:
        return 0.0, None, False, 0, 0

    skin_cnt = int(np.count_nonzero(m_roi))
    skin_ratio = float(skin_cnt) / float(area_poly)

    ys, xs = np.where(m_roi > 0)
    if len(xs) == 0:
        return skin_ratio, None, False, area_poly, skin_cnt

    med = np.median(hsv[ys, xs, :], axis=0).astype(np.uint8)
    lo, hi = HSV_SKIN_TIGHT_SHOE
    median_is_skin = bool(np.all(med >= lo) and np.all(med <= hi))
    return skin_ratio, med, median_is_skin, area_poly, skin_cnt

def _skin_ratio_in_polygon(bgr_img, poly_pts):
    """
    Tính tỉ lệ pixel da trong polygon trên ảnh BGR.
    Skin mask = HSV_SKIN_LOOSE ∩ YCRCB_SKIN, lọc nhiễu nhẹ.
    Trả (skin_ratio, median_hsv, median_is_skin_tight).
    """
    if poly_pts is None:
        return 0.0, None, False

    h, w = bgr_img.shape[:2]
    poly_mask = _poly_mask(h, w, poly_pts)

    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)

    m_h = cv2.inRange(hsv,  HSV_SKIN_LOOSE[0],  HSV_SKIN_LOOSE[1])
    m_y = cv2.inRange(ycc,  YCRCB_SKIN[0],     YCRCB_SKIN[1])
    m   = cv2.bitwise_and(m_h, m_y)
    m   = _clean_mask(m, k=3, iters=1)

    m_poly = cv2.bitwise_and(m, poly_mask)
    area = np.count_nonzero(poly_mask)
    if area <= 0:
        return 0.0, None, False

    skin_cnt = np.count_nonzero(m_poly)
    skin_ratio = float(skin_cnt) / float(area)

    # median HSV trong vùng da bên trong polygon
    ys, xs = np.where(m_poly > 0)
    if len(xs) == 0:
        return skin_ratio, None, False
    med = np.median(hsv[ys, xs, :], axis=0).astype(np.uint8)
    lo, hi = HSV_SKIN_TIGHT
    median_is_skin = bool(np.all(med >= lo) and np.all(med <= hi))
    return skin_ratio, med, median_is_skin
def check_leg_skin_exposure(test_image, pose_landmarks, gender="male"):
    h, w = test_image.shape[:2]
    lm = pose_landmarks.landmark

    def P(i):
        return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

    def make_poly(side="left"):
        if side == "left":
            hip_i, knee_i, ankle_i = (mp_pose.PoseLandmark.LEFT_HIP,
                                      mp_pose.PoseLandmark.LEFT_KNEE,
                                      mp_pose.PoseLandmark.LEFT_ANKLE)
        else:
            hip_i, knee_i, ankle_i = (mp_pose.PoseLandmark.RIGHT_HIP,
                                      mp_pose.PoseLandmark.RIGHT_KNEE,
                                      mp_pose.PoseLandmark.RIGHT_ANKLE)

        hip   = P(hip_i)
        knee  = P(knee_i)
        ankle = P(ankle_i)

        # BẮT ĐẦU: từ giữa đùi (dịch xuống từ hip về phía gối)
        v_hk = knee - hip
        L_hk = np.linalg.norm(v_hk)
        if L_hk > 1e-3:
            v_hk_dir = v_hk / L_hk
        else:
            v_hk_dir = np.array([0.0, 1.0], dtype=np.float32)
        start_shift = max(LEG_START_FROM_HIP_MINPX, LEG_START_FROM_HIP_RATIO * L_hk)
        start_pt = hip + v_hk_dir * start_shift   # nằm trên đùi

        # KẾT THÚC: ngay tại gối (không kéo xuống mắt cá nữa)
        end_pt = knee

        # Tạo dải hình chữ nhật xoay giữa start_pt ↔ end_pt
        return _leg_band_polygon(
            test_image.shape,
            tuple(start_pt.astype(np.int32)),
            tuple(end_pt.astype(np.int32))
        )



    L_poly = make_poly("left")
    R_poly = make_poly("right")

    # phần dưới giữ nguyên…
    L_ratio, L_med, L_okmed = _skin_ratio_in_polygon(test_image, L_poly)
    R_ratio, R_med, R_okmed = _skin_ratio_in_polygon(test_image, R_poly)

    L_exposed = (L_ratio >= THRESH_SKIN_PANTS) and L_okmed
    R_exposed = (R_ratio >= THRESH_SKIN_PANTS) and R_okmed

    detail = {
        "left_leg":  "fail" if L_exposed else "pass",
        "right_leg": "fail" if R_exposed else "pass",
        "left_ratio":  L_ratio, "right_ratio": R_ratio,
        "left_median": L_med,   "right_median": R_med
    }
    overall = "pass" if (not L_exposed and not R_exposed) else "fail"
    return overall, detail, (L_poly, R_poly)


def _lm_xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def _clip_box_raw(x1, y1, x2, y2, w, h):
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _expand_and_clip(box, w, h, margin):
    if box is None:
        return None
    return _clip_box_raw(
        box["x1"] - margin, box["y1"] - margin,
        box["x2"] + margin, box["y2"] + margin, w, h
    )


def _draw_labeled_box(img, box, text, color=None, thick=None):
    if box is None:
        return img
    if color is None:
        color = FAIL_BOX_COLOR
    if thick is None:
        thick = FAIL_BOX_THICKNESS
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

    # TÍNH KÍCH THƯỚC CHỮ THEO CẤU HÌNH
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, FAIL_TEXT_FONT_SCALE, FAIL_TEXT_THICKNESS
    )
    y_top = max(0, y1 - th - 8)
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 10, y_top + th + 8), color, -1)
    cv2.putText(
        img, text, (x1 + 5, y_top + th + 1),
        cv2.FONT_HERSHEY_SIMPLEX, FAIL_TEXT_FONT_SCALE, (255, 255, 255),
        FAIL_TEXT_THICKNESS, cv2.LINE_AA
    )
    return img

def _person_crop_box_from_pose(image_shape, pose_landmarks, margin_ratio=0.12, min_margin_px=24):
    """
    Sinh bbox người dựa vào toàn bộ pose landmarks rồi nới biên.
    Trả: dict {x1,y1,x2,y2} hoặc None nếu thiếu landmarks.
    """
    if pose_landmarks is None:
        return None
    h, w = image_shape[:2]
    xs, ys = [], []
    for lm in pose_landmarks.landmark:
        # lọc landmark có toạ độ hợp lệ
        if 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))
    if not xs or not ys:
        return None

    x1, y1 = max(0, min(xs)), max(0, min(ys))
    x2, y2 = min(w - 1, max(xs)), min(h - 1, max(ys))
    bw, bh  = x2 - x1 + 1, y2 - y1 + 1
    if bw <= 0 or bh <= 0:
        return None

    # nới biên đều 4 phía (bbox người gốc)
    m = int(max(min_margin_px, margin_ratio * max(bw, bh)))
    return _clip_box_raw(x1 - m, y1 - m, x2 + m, y2 + m, w, h)


def _adjust_box_for_collar_crop(box, img_shape):
    """
    Điều chỉnh box người riêng cho cổ áo theo 4 hướng.

    SCOLLAR_CROP_*_RATIO được hiểu là TỈ LỆ CẮT BỚT:
      - >= 0: cắt bớt từ phía đó (0.3 = cắt 30% từ cạnh đó)
      - <  0: nới rộng thêm (|ratio| * kích thước)
    Nếu cắt quá tay khiến bbox mới quá nhỏ hoặc invalid
    thì fallback dùng lại box gốc.
    """
    if box is None:
        return None

    h, w = img_shape[:2]
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # --- TOP ---
    if SCOLLAR_CROP_TOP_RATIO >= 0:
        cut_top = int(SCOLLAR_CROP_TOP_RATIO * bh)
        new_y1 = y1 + cut_top
    else:
        ext_top = int(abs(SCOLLAR_CROP_TOP_RATIO) * bh)
        new_y1 = y1 - ext_top

    # --- BOTTOM ---
    if SCOLLAR_CROP_BOTTOM_RATIO >= 0:
        cut_bot = int(SCOLLAR_CROP_BOTTOM_RATIO * bh)
        new_y2 = y2 - cut_bot
    else:
        ext_bot = int(abs(SCOLLAR_CROP_BOTTOM_RATIO) * bh)
        new_y2 = y2 + ext_bot

    # --- LEFT ---
    if SCOLLAR_CROP_LEFT_RATIO >= 0:
        cut_l = int(SCOLLAR_CROP_LEFT_RATIO * bw)
        new_x1 = x1 + cut_l
    else:
        ext_l = int(abs(SCOLLAR_CROP_LEFT_RATIO) * bw)
        new_x1 = x1 - ext_l

    # --- RIGHT ---
    if SCOLLAR_CROP_RIGHT_RATIO >= 0:
        cut_r = int(SCOLLAR_CROP_RIGHT_RATIO * bw)
        new_x2 = x2 - cut_r
    else:
        ext_r = int(abs(SCOLLAR_CROP_RIGHT_RATIO) * bw)
        new_x2 = x2 + ext_r

    # Nếu sau khi chỉnh mà bbox quá nhỏ hoặc invalid thì dùng lại box gốc
    min_w = 10
    min_h = 10
    if (new_x2 - new_x1) < min_w or (new_y2 - new_y1) < min_h:
        # fallback
        return box

    return _clip_box_raw(new_x1, new_y1, new_x2, new_y2, w, h)



def _norm_name(s: str) -> str:
    """chuẩn hoá: lowercase, bỏ khoảng trắng/gạch, chỉ a-z0-9"""
    import re
    if s is None:
        return ""
    s = s.strip().lower()
    s = s.replace("_", "").replace("-", "").replace(" ", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def detect_collar_scollar_yolo(image, pose_landmarks, conf_thres=SCOLLAR_CONF_THRES):
    """
    Chạy YOLO scollar trên person-crop.

    Logic mới:
    - Lưu crop người vào file cố định SCOLLAR_CROP_NAME (ghi đè mỗi lần).
    - Nếu tất cả detection hợp lệ đều là "có cổ áo"  (POS)  → lấy box conf cao nhất, conf >= thres thì pass.
    - Nếu tất cả detection hợp lệ đều là "no_collar" (NEG) → lấy box conf cao nhất nhưng ALWAYS fail.
    - Nếu trong cùng frame có cả POS và NEG:
        + Nếu tồn tại cặp box POS–NEG có IoU >= SCOLLAR_CONFLICT_IOU
          => ưu tiên NEG (no_collar), fail.
        + Nếu không overlap => xử lý như trường hợp POS-only: dùng best POS so với threshold.
    """

    debug = {
        "model_loaded": scollar_model is not None,
        "crop_box": None,
        "num_dets": 0,
        "best_name": None,
        "best_conf": None,
        "best_scollar_conf": None,   # để run_inference lấy ra
        "scollar_confs": [],         # list các conf POS/NEG
        "conf_thres": conf_thres,
        "reason": None,
        "saved_crop": None,
        "raw_pos": SCOLLAR_POS_CLASS_NAMES,
        "raw_neg": SCOLLAR_NEG_CLASS_NAMES,
        "pos_names": None,
        "neg_names": None,
        "pos_dets": [],
        "neg_dets": [],
        "other_dets": [],
        "conflict_iou": SCOLLAR_CONFLICT_IOU,
    }

    if scollar_model is None:
        debug["reason"] = "model_not_loaded"
        print("[scollar] MISSING: YOLO model not loaded.")
        return "missing", debug

    # ===== 1) Cắt person-crop từ pose =====
    base_box = _person_crop_box_from_pose(image.shape, pose_landmarks)
    if base_box is None:
        debug["reason"] = "no_person_box_from_pose"
        print("[scollar] FAIL: cannot derive person crop from pose")
        return "fail", debug

    # Điều chỉnh 4 hướng riêng cho collar
    box = _adjust_box_for_collar_crop(base_box, image.shape)

    crop = _crop_box(image, box)
    debug["crop_box"] = box
    debug["raw_person_box"] = base_box  # để debug nếu cần


    # Lưu crop người vào file cố định (ghi đè)
    if SAVE_SCOLLAR_CROP and crop is not None and crop.size > 0:
        out_path = os.path.join(OUTPUT_FOLDER, SCOLLAR_CROP_NAME)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if cv2.imwrite(out_path, crop):
            debug["saved_crop"] = out_path

    # ===== 2) Chuẩn hoá tên class POS/NEG =====
    pos_alias = [a for a in (SCOLLAR_POS_CLASS_NAMES or "").split(",") if a.strip() != ""]
    neg_alias = [a for a in (SCOLLAR_NEG_CLASS_NAMES or "").split(",") if a.strip() != ""]
    pos_norm = {_norm_name(a) for a in pos_alias}
    neg_norm = {_norm_name(a) for a in neg_alias}
    debug["pos_names"] = sorted(list(pos_norm))
    debug["neg_names"] = sorted(list(neg_norm))

    # ===== 3) Chạy YOLO trên crop =====
    try:
        r = scollar_model.predict(source=crop, imgsz=640, conf=0.001, iou=0.5, verbose=False)[0]
    except Exception as e:
        debug["reason"] = f"infer_error: {e}"
        print(f"[scollar] FAIL: inference error: {e}")
        return "fail", debug

    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        debug["reason"] = "no_boxes"
        print("[scollar] --- No boxes ---")
        return "fail", debug

    # map id -> name (nếu có)
    names = None
    try:
        names = r.names if hasattr(r, "names") and r.names else getattr(scollar_model, "names", None)
    except Exception:
        names = getattr(scollar_model, "names", None)

    # lấy cls, conf, box xyxy
    try:
        cls_ids = boxes.cls.detach().cpu().numpy().tolist() if hasattr(boxes, "cls") else []
        confs   = boxes.conf.detach().cpu().numpy().tolist() if hasattr(boxes, "conf") else []
        xyxy    = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes, "xyxy") else None
    except Exception:
        cls_ids, confs, xyxy = [], [], None

    debug["num_dets"] = int(len(confs))
    if xyxy is None:
        xyxy = np.zeros((len(confs), 4), dtype=np.float32)

    pos_dets = []
    neg_dets = []
    other_dets = []

    for i, (cid, cf) in enumerate(zip(cls_ids, confs)):
        # tên lớp
        if names is not None:
            try:
                cname = names[int(cid)]
            except Exception:
                cname = str(int(cid))
        else:
            cname = str(int(cid))

        cname_norm = _norm_name(str(cname))
        b = xyxy[i].tolist()  # [x1,y1,x2,y2]
        det_rec = {
            "name": str(cname),
            "norm": cname_norm,
            "conf": float(cf),
            "box": [float(v) for v in b],
        }

        if cname_norm in pos_norm:
            pos_dets.append(det_rec)
        elif cname_norm in neg_norm:
            neg_dets.append(det_rec)
        else:
            other_dets.append(det_rec)

    debug["pos_dets"] = pos_dets
    debug["neg_dets"] = neg_dets
    debug["other_dets"] = other_dets
    debug["scollar_confs"] = [d["conf"] for d in pos_dets + neg_dets]

    def _best(det_list):
        return max(det_list, key=lambda d: d["conf"]) if det_list else None

    best_pos = _best(pos_dets)
    best_neg = _best(neg_dets)

    # ===== 4) Không có POS/NEG nào => fail =====
    if best_pos is None and best_neg is None:
        debug["reason"] = "no_valid_pos_or_neg_class"
        print("[scollar] No valid collar/no_collar class found.")
        return "fail", debug

    # ===== 5) Chỉ có NEG (no_collar) => luôn FAIL =====
    if best_neg is not None and best_pos is None:
        debug["best_name"] = best_neg["name"]
        debug["best_conf"] = round(best_neg["conf"], 4)
        debug["best_scollar_conf"] = debug["best_conf"]
        debug["reason"] = "only_negative_classes"
        print(f"[scollar] ONLY NEG: best={debug['best_name']}:{debug['best_conf']}")
        return "fail", debug

    # ===== 6) Chỉ có POS => dùng threshold như cũ =====
    if best_pos is not None and best_neg is None:
        debug["best_name"] = best_pos["name"]
        debug["best_conf"] = round(best_pos["conf"], 4)
        debug["best_scollar_conf"] = debug["best_conf"]
        print(f"[scollar] POS-ONLY dets={debug['num_dets']} | best={debug['best_name']}:{debug['best_conf']} "
              f"| thres={conf_thres} | crop={debug['crop_box']} | allow_pos={debug['pos_names']}")
        if best_pos["conf"] >= conf_thres:
            return "pass", debug
        else:
            debug["reason"] = "best_pos_below_threshold"
            return "fail", debug

    # ===== 7) Có cả POS và NEG =====
    # Bỏ conflict IoU, chỉ lấy detection có confidence lớn nhất
    if best_pos is not None and best_neg is not None:
        # Chọn detection có conf cao nhất
        if best_pos["conf"] >= best_neg["conf"]:
            # Chọn POS
            debug["best_name"] = best_pos["name"]
            debug["best_conf"] = round(best_pos["conf"], 4)
            debug["best_scollar_conf"] = debug["best_conf"]
            debug["reason"] = "pos_neg_pick_pos_by_best_conf"

            print(
                f"[scollar] POS+NEG pick POS: {debug['best_name']}:{debug['best_conf']} "
                f"| thres={conf_thres} | crop={debug['crop_box']} | allow_pos={debug['pos_names']}"
            )

            # Giống POS-only: vẫn dùng threshold
            if best_pos["conf"] >= conf_thres:
                return "pass", debug
            else:
                debug["reason"] = "best_pos_below_threshold_with_neg_present"
                return "fail", debug

        else:
            # Chọn NEG (no_collar) => luôn FAIL
            debug["best_name"] = best_neg["name"]
            debug["best_conf"] = round(best_neg["conf"], 4)
            debug["best_scollar_conf"] = debug["best_conf"]
            debug["reason"] = "pos_neg_pick_neg_by_best_conf"

            print(
                f"[scollar] POS+NEG pick NEG => FAIL: {debug['best_name']}:{debug['best_conf']} "
                f"| crop={debug['crop_box']} | allow_neg={debug['neg_names']}"
            )
            return "fail", debug

    # Trường hợp không rơi vào nhánh nào ở trên (phòng hờ)
    debug["reason"] = "unexpected_branch_after_pos_neg"
    return "fail", debug




def detect_pose_landmarks(image):
    """Detect pose landmarks using MediaPipe. Trả None nếu không có landmarks hoặc có lỗi."""
    try:
        with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False) as pose:
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"[warn] MediaPipe Pose error: {e}")
        return None

    if not results or not results.pose_landmarks:
        return None
    return results.pose_landmarks


def get_face_box(image, landmarks):
    """Tính box khuôn mặt dùng cho smile (CRUDE: nose & ears) và trả (crop, box)."""
    h, w = image.shape[:2]
    nose = _lm_xy(landmarks.landmark[mp_pose.PoseLandmark.NOSE], w, h)
    left_ear = _lm_xy(landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR], w, h)
    right_ear = _lm_xy(landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR], w, h)
    x1 = int(min(left_ear[0], right_ear[0]) - 50)
    y1 = int(nose[1] - 100)
    x2 = int(max(left_ear[0], right_ear[0]) + 50)
    y2 = int(nose[1] + 100)
    box = _clip_box_raw(x1, y1, x2, y2, w, h)
    crop = image[box["y1"]:box["y2"], box["x1"]:box["x2"]] if box else None
    return crop, box


def detect_smile(face_img, threshold=THRESH_SMILE_CONF):
    """Detect smile từ crop face (grayscale 32x32)."""
    if face_img is None or face_img.size == 0 or smile_model is None:
        return "missing"
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        array_img = img_to_array(resized)  # (32,32,1)
        array_img = array_img.astype("float") / 255.0
        array_img = np.expand_dims(array_img, axis=0)  # (1,32,32,1)
        pred = float(smile_model.predict(array_img, verbose=0)[0][0])  # sigmoid
        return "smile" if pred > threshold else "no_smile"
    except Exception as e:
        print(f"[warn] Error in smile detection: {e}")
        return "missing"




def shoulders_box(landmarks, image_shape):
    """Box bao 2 vai (để annotate)."""
    h, w = image_shape[:2]
    L = _lm_xy(landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER], w, h)
    R = _lm_xy(landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER], w, h)
    x1, y1 = min(L[0], R[0]), min(L[1], R[1])
    x2, y2 = max(L[0], R[0]), max(L[1], R[1])
    base = _clip_box_raw(x1, y1, x2, y2, w, h)
    return base  # margin sẽ nới khi vẽ


def calculate_angle(p1, p2, p3):
    """Góc giữa p1-p2-p3 (dùng normalized coords)."""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    angle = np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return np.degrees(angle)


def check_arms_straight_down(landmarks):
    """
    Trả:
      both_ok (bool),
      (left_angle, right_angle),
      (left_ok, right_ok)
    """
    lS = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    lE = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    lW = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    rS = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    rE = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    rW = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_angle = calculate_angle(lS, lE, lW)
    right_angle = calculate_angle(rS, rE, rW)
    left_ok = (left_angle >= ARM_STRAIGHT_ANGLE_THRESHOLD) and (lW.y > lE.y > lS.y)
    right_ok = (right_angle >= ARM_STRAIGHT_ANGLE_THRESHOLD) and (rW.y > rE.y > rS.y)
    return (left_ok and right_ok), (left_angle, right_angle), (left_ok, right_ok)


def arm_box(landmarks, image_shape, left=True):
    """Box bao phủ shoulder-elbow-wrist của 1 tay (để annotate)."""
    h, w = image_shape[:2]
    if left:
        ids = [mp_pose.PoseLandmark.LEFT_SHOULDER,
               mp_pose.PoseLandmark.LEFT_ELBOW,
               mp_pose.PoseLandmark.LEFT_WRIST]
    else:
        ids = [mp_pose.PoseLandmark.RIGHT_SHOULDER,
               mp_pose.PoseLandmark.RIGHT_ELBOW,
               mp_pose.PoseLandmark.RIGHT_WRIST]
    pts = np.array([_lm_xy(landmarks.landmark[i], w, h) for i in ids], dtype=np.float32)
    x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
    x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])
    base = _clip_box_raw(x1, y1, x2, y2, w, h)
    return base  # margin sẽ nới khi vẽ


# =======================
# Skin & Foot geometry (SHOES)
# =======================
def _clean_mask(mask, k=3, iters=1):
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
    return mask


def median_hsv_on_mask(hsv_img, mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    sel = hsv_img[ys, xs, :]
    med = np.median(sel, axis=0)
    return np.array(med, dtype=np.uint8)


def hsv_in_range(hsv_color, rng):
    lo, hi = rng
    return bool(np.all(hsv_color >= lo) and np.all(hsv_color <= hi))


def _rotated_rect_points(center, size, angle_deg):
    rect = (tuple(center), tuple(size), angle_deg)
    box = cv2.boxPoints(rect)
    return box.astype(np.int32)


def foot_rectangle_ankle_to_toe(landmarks, img_shape, left=True):
    """Hình chữ nhật xoay theo hướng (ankle→toe) để ràng buộc vùng bàn chân."""
    h, w = img_shape[:2]
    if left:
        idx_ankle = mp_pose.PoseLandmark.LEFT_ANKLE.value
        idx_toe = mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
    else:
        idx_ankle = mp_pose.PoseLandmark.RIGHT_ANKLE.value
        idx_toe = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value

    lms = landmarks.landmark
    if any(lms[i].visibility < MIN_VIS for i in [idx_ankle, idx_toe]):
        return None, False

    A = _lm_xy(lms[idx_ankle], w, h)
    T = _lm_xy(lms[idx_toe], w, h)

    v = T - A
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-3:
        return None, False
    v_dir = v / v_norm
    v_perp = np.array([-v_dir[1], v_dir[0]])

    # Tâm dịch về phía ngón chân hơn (FOOT_RECT_CENTER_T > 0.70 sẽ xa mắt cá hơn)
    center = A + FOOT_RECT_CENTER_T * v

    # Dịch ngang nhẹ để hai chân tách nhau (như cũ)
    shift_px = v_norm * 0.02
    center = center - shift_px * v_perp if left else center + shift_px * v_perp

    # Chiều dài cơ bản
    base_length = v_norm * SCALE_LEN

    # RÚT NGẮN ở đầu gần mắt cá: giảm length và đẩy center tiến về ngón chân nửa phần trim
    trim = max(0.0, SHOE_TRIM_AT_ANKLE_RATIO * v_norm)
    length = max(20.0, base_length - trim)
    center = center + v_dir * (trim * 0.5)

    width = max(20.0, v_norm * WIDTH_RATIO)
    angle = math.degrees(math.atan2(v_dir[1], v_dir[0]))
    h_img, w_img = img_shape[:2]
    extra_x = float(FOOT_RECT_HSHIFT_PX) + float(FOOT_RECT_HSHIFT_RATIO) * float(w_img)
    if left:
        center[0] += extra_x
    else:
        center[0] -= extra_x

    pts = _rotated_rect_points(center, (length, width), angle)
    return pts, True




def _crop_box(img, box):
    """Crop ảnh theo box {x1,y1,x2,y2}. Trả None nếu box None hoặc invalid."""
    if box is None:
        return None
    return img[box["y1"]:box["y2"], box["x1"]:box["x2"]]





def verify_shoe_skin(label, test_image, landmarks):
    h, w = test_image.shape[:2]
    debug = {
        "skin_ratio": None, "median_hsv": None, "median_is_skin": False,
        "foot_rect_ok": False, "foot_rect_pts": None,
        "area_poly": 0, "skin_pixels": 0,
        "reason": None,
        # NEW: log visibility từng điểm chân
        "foot_vis": {}
    }

    left = (label == "left_shoe")
    if left:
        foot_ids = {
            "ankle": mp_pose.PoseLandmark.LEFT_ANKLE.value,
            "heel":  mp_pose.PoseLandmark.LEFT_HEEL.value,
            "toe":   mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
        }
    else:
        foot_ids = {
            "ankle": mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            "heel":  mp_pose.PoseLandmark.RIGHT_HEEL.value,
            "toe":   mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
        }

    lms = landmarks.landmark
    # --- 1) BẮT BUỘC THẤY HẾT: chỉ cần MỘT điểm < MIN_VIS là FAIL ---
    vis = {name: float(lms[idx].visibility) for name, idx in foot_ids.items()}
    debug["foot_vis"] = vis
    low = {k: v for k, v in vis.items() if (v < MIN_VIS) or np.isnan(v)}

    if len(low) > 0:
        debug["reason"] = "foot_landmark_missing_or_low_visibility"
        print(f"[{label}] FAIL (foot landmarks): " +
              ", ".join([f"{k}={v:.3f}" for k, v in vis.items()]) +
              f"  (MIN_VIS={MIN_VIS})")
        return "fail", debug

    # --- 2) Tạo foot-rectangle như cũ ---
    pts, ok = foot_rectangle_ankle_to_toe(landmarks, test_image.shape, left=left)
    if (not ok) or (pts is None):
        debug["reason"] = "foot_rect_generation_failed"
        print(f"[{label}] FAIL: cannot build foot rectangle (ok={ok})")
        return "fail", debug

    debug["foot_rect_ok"]  = True
    debug["foot_rect_pts"] = pts.reshape(-1, 2).tolist()

    # --- 3) Tính skin trong polygon (giữ nguyên logic lộ da) ---
    skin_ratio, med, med_ok, area_poly, skin_cnt = _skin_stats_in_polygon_full(test_image, pts)
    debug["skin_ratio"] = skin_ratio
    debug["median_hsv"] = None if med is None else tuple(int(x) for x in med.tolist())
    debug["median_is_skin"] = bool(med_ok)
    debug["area_poly"] = int(area_poly)
    debug["skin_pixels"] = int(skin_cnt)

    result = "fail" if (skin_ratio >= MIN_SKIN_AREA_SHOE_FULL and med_ok) else "pass"

    print(
        f"[{label}] res={result}"
        f" | skin_ratio={skin_ratio:.4f}"
        f" | median={debug['median_hsv']}"
        f" | median_is_skin={med_ok}"
        f" | area={area_poly}"
        f" | skin_px={skin_cnt}"
        f" | vis=" + ", ".join([f"{k}={v:.3f}" for k, v in vis.items()])
    )

    if DEBUG_SHOE:
        cv2.polylines(test_image, [pts], isClosed=True, color=(255, 0, 0), thickness=3)
        poly_mask = _poly_mask(h, w, pts)
        hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
        ycc = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)
        m_h = cv2.inRange(hsv,  HSV_SKIN_LOOSE[0],  HSV_SKIN_LOOSE[1])
        m_y = cv2.inRange(ycc,  YCRCB_SKIN[0],     YCRCB_SKIN[1])
        m   = _clean_mask(cv2.bitwise_and(m_h, m_y), k=3, iters=1)
        m_roi = cv2.bitwise_and(m, poly_mask)
        test_image[:] = overlay_mask_on_image(test_image, m_roi, color=(0, 255, 0), alpha=0.45)
        if result == "fail":
            cv2.polylines(test_image, [pts], isClosed=True, color=(0, 0, 255), thickness=3)

    return result, debug




# =======================
# Annotate (vẽ BB đỏ cho mọi phần FAIL/MISSING)
# =======================
def annotate_fail_shoes(image, foot_rects, results):
    """
    Vẽ khung đỏ TRÙNG khít foot-rectangle cho các bên 'fail' hoặc 'missing'.
    (foot_rects[label] là ndarray (N,2) các điểm polygon)
    """
    out = image.copy()
    for label in ["left_shoe", "right_shoe"]:
        status = results.get(label, "missing")
        if status != "pass":
            pts = foot_rects.get(label)
            if pts is not None:
                # vẽ polygon đỏ
                cv2.polylines(out, [pts.astype(np.int32)], True, FAIL_BOX_COLOR, FAIL_BOX_THICKNESS)
                # dán nhãn gần tâm polygon
                cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                out = _draw_labeled_box(
                    out,
                    {"x1": cx-40, "y1": cy-24, "x2": cx+40, "y2": cy-4},
                    f"{label}: {status}"
                )
    return out



def annotate_fail_general(image, results, pose_landmarks, face_box):
    """
    - Vai: nếu fail → vẽ box quanh 2 vai (nới MARGIN_SHOULDER)
    - Tay: nếu left/right fail → vẽ box từng tay (nới MARGIN_ARM)
    - Cười: nếu 'no_smile' hoặc 'missing' → vẽ box khuôn mặt (nới MARGIN_FACE)
    """
    out = image.copy()
    h, w = image.shape[:2]


    # Tay
    if results.get("left_arm", "pass") != "pass":
        base = arm_box(pose_landmarks, image.shape, left=True)
        box = _expand_and_clip(base, w, h, MARGIN_ARM)
        out = _draw_labeled_box(out, box, "left_arm: fail")
    if results.get("right_arm", "pass") != "pass":
        base = arm_box(pose_landmarks, image.shape, left=False)
        box = _expand_and_clip(base, w, h, MARGIN_ARM)
        out = _draw_labeled_box(out, box, "right_arm: fail")

    # Nụ cười
    smile_status = results.get("smile", "missing")
    if smile_status != "smile":
        box = _expand_and_clip(face_box, w, h, MARGIN_FACE)
        out = _draw_labeled_box(out, box, f"smile: {smile_status}")

    # Cổ áo (scollar)
    if results.get("scollar", "pass") != "pass":
        base = shoulders_box(pose_landmarks, image.shape)
        # nới rộng cơ bản để phủ vùng cổ/ức (như cũ)
        base_expanded = _expand_and_clip(base, w, h, max(MARGIN_SHOULDER, 60))
        # rồi tinh chỉnh: cao lên + thu hẹp ngang
        box = _tweak_collar_fail_box(base_expanded, image.shape)
        out = _draw_labeled_box(out, box, f"scollar: {results.get('scollar')}")


    return out

def get_face_smile_box_like_process(image, pose_landmarks):
    """
    Tạo face-smile box giống file process đã upload:
    - Lấy các mốc đầu (0..8): mũi, mắt, tai...
    - Nới theo tỉ lệ margin_x=0.15*face_w, margin_y=3*face_h
    - Trả (crop, box)
    """
    if pose_landmarks is None:
        return None, None

    h, w = image.shape[:2]
    lm = pose_landmarks.landmark

    # Các mốc đầu như file process: 0..8 (NOSE, mắt, tai…)
    head_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    xs, ys = [], []
    for i in head_ids:
        xi = int(lm[i].x * w)
        yi = int(lm[i].y * h)
        xs.append(xi); ys.append(yi)
    if not xs or not ys:
        return None, None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    face_w = max(1, x_max - x_min)
    face_h = max(1, y_max - y_min)

    # Giống tỉ lệ trong file upload
    margin_x = int(face_w * 0.15)
    margin_y = int(face_h * 3.0)

    x1 = max(x_min - margin_x, 0)
    y1 = max(y_min - margin_y, 0)
    x2 = min(x_max + margin_x, w)
    y2 = min(y_max + margin_y, h)

    box = _clip_box_raw(x1, y1, x2, y2, w, h)
    crop = image[box["y1"]:box["y2"], box["x1"]:box["x2"]] if box else None
    return crop, box
def _hand_visibility_ok(landmarks, side="left", thresh=0.5):
    """
    Kiểm tra các điểm của bàn tay có visibility >= thresh không.
    side: "left" | "right"
    Trả về: (ok: bool, vis_dict: {wrist,index,thumb,pinky}, low_keys: [keys dưới ngưỡng])
    """
    if side == "left":
        keys = {
            "wrist": mp_pose.PoseLandmark.LEFT_WRIST.value,
            "index": mp_pose.PoseLandmark.LEFT_INDEX.value,
            "thumb": mp_pose.PoseLandmark.LEFT_THUMB.value,
            "pinky": mp_pose.PoseLandmark.LEFT_PINKY.value,
        }
    else:
        keys = {
            "wrist": mp_pose.PoseLandmark.RIGHT_WRIST.value,
            "index": mp_pose.PoseLandmark.RIGHT_INDEX.value,
            "thumb": mp_pose.PoseLandmark.RIGHT_THUMB.value,
            "pinky": mp_pose.PoseLandmark.RIGHT_PINKY.value,
        }

    lms = landmarks.landmark
    vis = {k: float(lms[idx].visibility) for k, idx in keys.items()}
    low = [k for k, v in vis.items() if (np.isnan(v) or v < thresh)]
    return (len(low) == 0), vis, low

# =======================
# Main inference API
# =======================
def run_inference(test_image, test_image_path=None, gender="male"):
    """
    Trả:
      - output_path: đường dẫn ảnh đã annotate
      - results: dict gồm {shoulders, arms, left_arm, right_arm, smile, left_shoe, right_shoe, shoes}
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    results = {}

    raw_image = test_image.copy()
    # 1) Pose
    pose_landmarks = detect_pose_landmarks(test_image)
    if not pose_landmarks:
        results["_error"] = "Vui lòng chụp toàn thân, đủ sáng và không che khuất."
        # Không cần lưu ảnh annotate khi lỗi; vẫn trả đường dẫn dự kiến (hoặc None)
        return None, results

    # 1.1) MỚI: kiểm tra bắt buộc phải thấy rõ thân trên, thân dưới và khuôn mặt
    body_ok, groups_ok, missing = _check_required_landmarks(
        pose_landmarks,
        vis_thresh=MIN_VIS   # dùng cùng ngưỡng với chân/giày
    )
    if not body_ok:
        # Gộp thông báo chung cho user
        results["_error"] = (
            "Vui lòng chụp toàn thân (thấy rõ tay, chân và khuôn mặt), "
            "đứng thẳng, đủ sáng và không che khuất."
        )
        # Option: trả thêm thông tin debug cho JSON (nếu cần xem ở /result)
        results["_body_visibility"] = {
            "upper_ok": groups_ok.get("upper", False),
            "lower_ok": groups_ok.get("lower", False),
            "face_ok":  groups_ok.get("face", False),
            "missing_ids": missing,  # id các landmark thiếu / visibility thấp
        }
        return None, results

    # 3) Tay (điều kiện tổng: cả 2 chuẩn mới pass)
    both_ok, (l_angle, r_angle), (l_ok, r_ok) = check_arms_straight_down(pose_landmarks)

    # >>> NEW: bắt buộc thấy rõ bàn tay cho từng bên
    left_hand_ok, left_vis, left_low = _hand_visibility_ok(pose_landmarks, side="left", thresh=HAND_MIN_VIS)
    right_hand_ok, right_vis, right_low = _hand_visibility_ok(pose_landmarks, side="right", thresh=HAND_MIN_VIS)

    # Nếu bàn tay bên nào không đạt visibility, ép cánh tay bên đó fail
    if not left_hand_ok:
        l_ok = False
        # (tuỳ chọn) log để debug
        print(f"[left_arm] FAIL (hand visibility): {left_vis}  low={left_low}, THRESH={HAND_MIN_VIS}")
    if not right_hand_ok:
        r_ok = False
        print(f"[right_arm] FAIL (hand visibility): {right_vis}  low={right_low}, THRESH={HAND_MIN_VIS}")

    # Tính lại tổng thể arms sau khi ép điều kiện bàn tay
    both_ok = (l_ok and r_ok)

    results["arms"] = "pass" if both_ok else "fail"
    results["left_arm"] = "pass" if l_ok else "fail"
    results["right_arm"] = "pass" if r_ok else "fail"
    # 3.5) CHÂN (theo giới tính)
    leg_overall, leg_detail, leg_polys = check_leg_skin_exposure(test_image, pose_landmarks, gender=gender)
    results["legs"] = leg_overall
    results.update(leg_detail)

    scollar_status, sc_dbg = detect_collar_scollar_yolo(test_image, pose_landmarks)
    results["scollar"] = scollar_status
    results["scollar_debug"] = sc_dbg  # để trả JSON nếu bạn hiển thị
    results["scollar_best_conf"] = float(sc_dbg.get("best_scollar_conf") or -1.0)
    results["scollar_confs"] = sc_dbg.get("scollar_confs") or []
    results["scollar_crop_path"] = sc_dbg.get("saved_crop")

    # 4.1) FaceMesh chạy trên CROP MẶT VUÔNG ỔN ĐỊNH (để bắt lông mày ổn định)
    facemesh_box = get_facemesh_box(test_image, pose_landmarks, scale=FACE_MESH_CROP_SCALE)
    if DEBUG_FACE_SAVE and facemesh_box is not None:
        _save_face_crop(test_image, facemesh_box, os.path.join(OUTPUT_FOLDER, FACE_CROP_RAW_NAME))

    face_lms = detect_face_mesh_landmarks_on_crop(test_image, facemesh_box)
    brow_overall, brow_detail, (L_poly, R_poly) = check_eyebrow_hair_overlap(
        test_image, face_lms, face_box=facemesh_box
    )
    results["eyebrow_hair"] = brow_overall
    results.update(brow_detail)
    brow_polys = (L_poly, R_poly)
    # (MỚI) Lưu bản OVERLAY của crop mặt (sau khi check_eyebrow_hair_overlap đã vẽ ROI + tóc)
    if DEBUG_FACE_SAVE and facemesh_box is not None:
        _save_face_crop(test_image, facemesh_box, os.path.join(OUTPUT_FOLDER, FACE_CROP_OVERLAY_NAME))
    # TÍNH NGƯỠNG Y DƯỚI CHÂN MÀY CHO EAR (dùng đáy ROI lông mày)
    ear_min_y = None
    ys = []
    if L_poly is not None:
        ys.append(float(L_poly[:, 1].max()))  # đáy lông mày trái
    if R_poly is not None:
        ys.append(float(R_poly[:, 1].max()))  # đáy lông mày phải
    if ys:
        ear_min_y = max(ys)  # bbox tai phải có center_y >= ear_min_y
    results["ear_min_y"] = ear_min_y



    # Vẽ cảnh báo nếu FAIL (viền đỏ quanh dải kiểm tra)
    if leg_overall != "pass":
        for poly in leg_polys:
            if poly is not None:
                cv2.polylines(test_image, [poly], isClosed=True, color=(0, 0, 255), thickness=FAIL_BOX_THICKNESS)
                # dán nhãn nhỏ
                cx, cy = int(poly[:,0].mean()), int(poly[:,1].mean())
                cv2.putText(test_image, "leg_skin: fail", (cx-40, cy-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    # 4) Nụ cười + box khuôn mặt (dùng ảnh gốc để không bị overlay)
    face_crop_smile, face_box_smile = get_face_smile_box_like_process(raw_image, pose_landmarks)
    results["smile"] = detect_smile(face_crop_smile)

    # Lưu crop mặt (smile)
    if SAVE_SMILE_FACE_CROP and face_box_smile is not None:
        out_face_path = os.path.join(OUTPUT_FOLDER, SMILE_FACE_CROP_NAME)
        saved_ok = _save_face_crop(raw_image, face_box_smile, out_face_path)
        results["face_crop_saved"] = out_face_path if saved_ok else None

    # 4.1) Tai: dùng CROP TO HƠN so với SMILE, chỉ tính bbox dưới chân mày
    ear_face_box = _expand_box_xy(
        face_box_smile,
        raw_image.shape,
        margin_x=EAR_FACE_EXTRA_MARGIN_X,
        margin_y=EAR_FACE_EXTRA_MARGIN_Y,
    )
    ear_min_y = results.get("ear_min_y")  # có thể None nếu không detect được lông mày
    ear_status, ear_dbg = detect_ears_yolo(raw_image, ear_face_box, min_y=ear_min_y)

    results["ear"] = ear_status
    results["ear_debug"] = ear_dbg
    results["ear_num_valid"] = int(ear_dbg.get("num_valid", 0))
    results["ear_confs"] = ear_dbg.get("all_confs", [])
    results["ear_crop_path"] = ear_dbg.get("saved_crop")
    results["ear_face_box"] = ear_face_box  # để annotate sau này

    # GIỮ lại face_box_smile để annotate phần smile
    face_box = face_box_smile


    # 5) Giày (left/right) + tổng hợp "shoes" (KHÔNG dùng shoe_box; chỉ foot-rectangle)
    foot_rects = {}
    for label in ["left_shoe", "right_shoe"]:
        res, dbg = verify_shoe_skin(
            label=label, test_image=test_image,
            landmarks=pose_landmarks
        )
        results[label] = res
        if dbg.get("foot_rect_ok") and dbg.get("foot_rect_pts") is not None:
            foot_rects[label] = np.array(dbg["foot_rect_pts"], dtype=np.int32).reshape(-1, 2)

        # (tuỳ chọn) log nhanh
        print(f"[{label}] skin_ratio={dbg['skin_ratio'] if dbg['skin_ratio'] is not None else -1:.4f} | "
              f"median={dbg['median_hsv']} | median_is_skin={dbg['median_is_skin']} | "
              f"foot_rect_ok={dbg['foot_rect_ok']} | area={dbg['area_poly']} | skin_px={dbg['skin_pixels']}")

    # Tổng hợp: chỉ pass nếu cả trái & phải đều pass
    results["shoes"] = "pass" if (results.get("left_shoe") == "pass" and results.get("right_shoe") == "pass") else "fail"

    # 6) Annotate các phần fail/missing
    annotated = annotate_fail_general(test_image, results, pose_landmarks, face_box)
    annotated = annotate_fail_shoes(annotated, foot_rects, results)
    annotated = annotate_fail_eyebrows(annotated, brow_polys, results)
    annotated = annotate_ears(annotated, results)  # <<< vẽ debug tai lên ảnh

    # 7) Lưu ảnh
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, "test_result.jpg")
    cv2.imwrite(output_path, annotated)

    return output_path, results
