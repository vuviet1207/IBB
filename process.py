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


# ==== EYEBROW–HAIR (mới) ====
HAIR_UP_PX           = int(os.getenv("HAIR_UP_PX", "18"))     # cao của forehead band
HAIR_EXPAND_PX       = int(os.getenv("HAIR_EXPAND_PX", "4"))  # nới ngang band
HAIR_MIN_THICK_PX    = float(os.getenv("HAIR_MIN_THICK_PX", "1"))
HAIR_MIN_AREA_RATIO  = float(os.getenv("HAIR_MIN_AREA_RATIO", "0.00001"))  # so với diện tích face-crop
HAIR_MORPH_ITERS     = int(os.getenv("HAIR_MORPH_ITERS", "1"))
ROI_TOUCH_DILATE     = int(os.getenv("ROI_TOUCH_DILATE", "5"))  # nếu bạn đang để 3 thì nâng lên 5

# Ngưỡng fallback khi không đủ mẫu để làm adaptive
DARK_V_THRESH        = int(os.getenv("DARK_V_THRESH", "110"))
DARK_GRAY_THRESH     = int(os.getenv("DARK_GRAY_THRESH", "55"))

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
MIN_SKIN_AREA_SHOE_FULL = float(os.getenv("MIN_SKIN_AREA_SHOE_FULL", "0.19"))  # 15%
THRESH_SKIN_PANTS = float(os.getenv("THRESH_SKIN_PANTS", str(MIN_SKIN_AREA_SHOE_FULL)))
# Hình học bàn chân (ankle→toe)
MIN_VIS = float(os.getenv("MIN_VIS", "0.59"))
SCALE_LEN = float(os.getenv("SCALE_LEN", "1.35"))
WIDTH_RATIO = float(os.getenv("WIDTH_RATIO", "0.6"))

HAND_MIN_VIS = float(os.getenv("HAND_MIN_VIS", str(MIN_VIS)))  # có thể chỉnh riêng, mặc định = MIN_VIS
# =======================
# Fail-box & crop configs (có thể override qua ENV)
# =======================
# Fail-box margins (pixels)
MARGIN_FACE = int(os.getenv("MARGIN_FACE", "20"))
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

FEMALE_KNEE_EXT_RATIO = float(os.getenv("FEMALE_KNEE_EXT_RATIO", "0.25"))  # % độ dài knee→ankle
FEMALE_KNEE_EXT_MINPX = int(os.getenv("FEMALE_KNEE_EXT_MINPX", "8"))       # tối thiểu mấy pixel

# --- Male leg band tweak (đẩy điểm cuối lên khỏi mắt cá) ---
MALE_ANKLE_OFFSET_RATIO = float(os.getenv("MALE_ANKLE_OFFSET_RATIO", "0.12"))  # tỉ lệ độ dài knee→ankle
MALE_ANKLE_OFFSET_MINPX = int(os.getenv("MALE_ANKLE_OFFSET_MINPX", "8"))       # tối thiểu mấy pixel

# --- Shoe rectangle tweak (rút ngắn ở đầu gần mắt cá) ---
FOOT_RECT_CENTER_T = float(os.getenv("FOOT_RECT_CENTER_T", "0.5"))            # tâm rect dọc theo A→T (trước ~0.70)
SHOE_TRIM_AT_ANKLE_RATIO = float(os.getenv("SHOE_TRIM_AT_ANKLE_RATIO", "0.95")) # % chiều dài A→T cắt ở đầu mắt cá

# --- Extra horizontal shift (dịch theo trục X màn hình) ---
FOOT_RECT_HSHIFT_PX     = float(os.getenv("FOOT_RECT_HSHIFT_PX", "15"))   # px
FOOT_RECT_HSHIFT_RATIO  = float(os.getenv("FOOT_RECT_HSHIFT_RATIO", "0")) # tỉ lệ theo bề rộng ảnh (0..1)



# ==== DEBUG FACE SAVE ====
DEBUG_FACE_SAVE = bool(int(os.getenv("DEBUG_FACE_SAVE", "1")))
FACE_CROP_RAW_NAME     = os.getenv("FACE_CROP_RAW_NAME", "face_crop_facemesh_raw.jpg")
FACE_CROP_OVERLAY_NAME = os.getenv("FACE_CROP_OVERLAY_NAME", "face_crop_facemesh_overlay.jpg")

# Tỉ lệ diện tích tối thiểu (so với diện tích face-box) để coi là "mảng lớn" (không bắt buộc dùng, để dự phòng)
MIN_HAIR_AREA_RATIO  = float(os.getenv("MIN_HAIR_AREA_RATIO", "0.002"))
# Dãn ROI (px) để xét “chạm” (tạo vành ngoài ROI lông mày)

# Làm trơn/khử nhiễu mặt nạ tối
HAIR_MORPH_K         = int(os.getenv("HAIR_MORPH_K", "3"))


# ==== FaceMesh crop (vuông – ổn định) ====
FACE_MESH_CROP_SCALE = float(os.getenv("FACE_MESH_CROP_SCALE", "1.6"))  # * khoảng cách 2 tai
FACE_MESH_MIN_SIDE   = int(os.getenv("FACE_MESH_MIN_SIDE", "220"))
FACE_MESH_MAX_SIDE   = int(os.getenv("FACE_MESH_MAX_SIDE", "640"))

# ==== Eyebrow ROI tweak ====
# Co hẹp ROI lông mày (px) và đẩy ROI lên trên (px). Có thể override qua ENV.
EYEBROW_SHRINK_PX = int(os.getenv("EYEBROW_SHRINK_PX", "0"))   # 2..3 là hợp lý
EYEBROW_UPSHIFT_PX = int(os.getenv("EYEBROW_UPSHIFT_PX", "3")) # 2..3 là hợp lý
EYEBROW_TOP_GROW_PX = int(os.getenv("EYEBROW_TOP_GROW_PX", "3"))  # 2..4 hợp lý

EYEBROW_SIDE_TOP_RATIO = float(os.getenv("EYEBROW_SIDE_TOP_RATIO", "0.67"))
# Bỏ chạm ở mép trong ROI lông mày (sát sống mũi)
INNER_RIM_GUARD_PX     = int(os.getenv("INNER_RIM_GUARD_PX", "6"))     # tối thiểu px
INNER_RIM_GUARD_RATIO  = float(os.getenv("INNER_RIM_GUARD_RATIO", "0.18"))  # % bề rộng ROI

# =======================
# MediaPipe
# =======================
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh  # hiện chưa dùng, để sau dễ mở rộng


# =======================
# Helpers
# =======================
def _clean_mask(mask, k=3, iters=1, mode="open"):
    kernel = np.ones((k, k), np.uint8)
    if mode == "open":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
    if mode == "close":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
    return mask

def _make_skin_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m1 = cv2.inRange(hsv, HSV_SKIN_LOOSE[0], HSV_SKIN_LOOSE[1])
    m2 = cv2.inRange(ycc, YCRCB_SKIN[0], YCRCB_SKIN[1])
    skin = cv2.bitwise_and(m1, m2)
    return _clean_mask(skin, k=3, iters=1, mode="open")

def _forehead_band_from_brows(img_shape, poly_L, poly_R, up_px=HAIR_UP_PX, expand_px=HAIR_EXPAND_PX):
    H, W = img_shape[:2]
    band = np.zeros((H, W), np.uint8)
    for poly in (poly_L, poly_R):
        x, y, w, h = cv2.boundingRect(np.array(poly, dtype=np.int32))
        y = max(0, y - up_px)
        h = h + up_px
        x = max(0, x - expand_px)
        w = min(W - x, w + 2*expand_px)
        band[y:y+h, x:x+w] = 255
    band = cv2.dilate(band, np.ones((3,3), np.uint8), iterations=1)
    return band

def _adaptive_dark_masks(bgr, forehead_band):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V    = hsv[:,:,2]

    v_vals = V[forehead_band > 0]
    if v_vals.size > 100:
        p30 = np.percentile(v_vals, 30)
        dv  = int(min(110, p30))
        dg  = int(max(35, 0.5 * dv))
    else:
        dv, dg = DARK_V_THRESH, DARK_GRAY_THRESH

    dark = ((gray < dg) | (V < dv)).astype(np.uint8) * 255
    dark = _clean_mask(dark, k=3, iters=1, mode="close")
    return dark

def _make_allowed_rim(mask_polygons, dilate_px=ROI_TOUCH_DILATE):
    rim = np.zeros(mask_polygons.shape[:2], np.uint8)  # nếu bạn có ảnh, truyền shape ảnh
    # mask_polygons là mask đã tô kín 2 polygon lông mày
    rim_d = cv2.dilate(mask_polygons, np.ones((dilate_px,dilate_px),np.uint8), iterations=1)
    rim_e = cv2.erode(mask_polygons,  np.ones((max(1,dilate_px-2),max(1,dilate_px-2)),np.uint8), iterations=1)
    rim   = cv2.subtract(rim_d, rim_e)
    return rim

def _hair_mask_from_brows(face_bgr, allowed_rim_mask, poly_L, poly_R):
    """
    Tạo mask tóc toàn phần (dark & ~skin), lọc theo:
      - rooted với forehead band
      - diện tích tối thiểu
      - độ dày tối thiểu
    Nếu provided allowed_rim_mask → chỉ dùng ở CUỐI cùng để giới hạn hiển thị,
    KHÔNG ảnh hưởng bước lọc thickness/area (tránh mất tóc ở mép).
    """
    skin     = _make_skin_mask(face_bgr)
    forehead = _forehead_band_from_brows(face_bgr.shape, poly_L, poly_R)
    dark     = _adaptive_dark_masks(face_bgr, forehead)

    # 1) Ứng viên tóc toàn phần (chưa áp rim)
    cand_full = cv2.bitwise_and(dark, cv2.bitwise_not(skin))

    # 2) Connected components + rooted + thickness/area (trên toàn phần)
    num, lab = cv2.connectedComponents(cand_full)
    keep = np.zeros_like(cand_full)
    H, W = cand_full.shape
    min_area = max(12, int(HAIR_MIN_AREA_RATIO * H * W))

    for i in range(1, num):
        comp = (lab == i).astype(np.uint8) * 255

        # rooted với trán
        if cv2.countNonZero(cv2.bitwise_and(comp, forehead)) == 0:
            continue

        # diện tích
        if cv2.countNonZero(comp) < min_area:
            continue

        # độ dày (đo trên blob đầy đủ, không phải vành mỏng)
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        if dist.max() < HAIR_MIN_THICK_PX:
            continue

        keep = cv2.bitwise_or(keep, comp)

    # 3) Morph nhẹ
    if HAIR_MORPH_ITERS > 0:
        keep = _clean_mask(keep, k=3, iters=HAIR_MORPH_ITERS, mode="close")

    # 4) (tuỳ chọn) Nếu muốn xem chỉ vùng sát lông mày, mới áp rim ở CUỐI
    if allowed_rim_mask is not None:
        keep = cv2.bitwise_and(keep, allowed_rim_mask)

    return keep


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

# ===== Face Mesh eyebrow indices (MediaPipe 468) =====
LEFT_BROW_UP   = [70, 63, 105, 66, 107]
LEFT_BROW_LOW  = [55, 65, 52, 53, 46]
RIGHT_BROW_UP  = [336, 296, 334, 293, 300]
RIGHT_BROW_LOW = [285, 295, 282, 283, 276]

def _to_px(pt, w, h):
    return np.array([pt.x * w, pt.y * h], dtype=np.float32)

def _dilate_mask(mask, k=3, iters=1):
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iters)

def _clean_dark_mask(mask, k=3, iters=1):
    # mở rồi đóng để khử nhiễu các đốm nhỏ
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
    return mask
def get_facemesh_box(image,
                     pose_landmarks,
                     scale=FACE_MESH_CROP_SCALE,
                     min_side=FACE_MESH_MIN_SIDE,
                     max_side=FACE_MESH_MAX_SIDE):
    """
    Box vuông FaceMesh: cạnh ~ scale * (khoảng cách 2 tai), clamp [min_side, max_side],
    tâm đặt tại mũi. Trả dict {x1,y1,x2,y2} đã CLIP.
    """
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
    """
    Chạy MediaPipe FaceMesh trên vùng crop_box rồi map landmarks về toạ độ ảnh gốc.
    Trả: list 468 landmark (mỗi landmark có .x/.y normalized theo ảnh GỐC) hoặc None.
    """
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

    # Map tọa độ normalized của crop -> pixel crop -> pixel global -> normalized global
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

def check_eyebrow_hair_overlap(image, face_lms, face_box=None):
    """
    FAIL nếu có vùng tối (đen/nâu đậm) CHẠM VÀO ROI LÔNG MÀY
    nhưng chỉ chấp nhận các tiếp xúc:
      - TỪ TRÊN xuống (y < ymin), HOẶC
      - TỪ HAI BÊN (x < xmin hoặc x > xmax).
    KHÔNG TÍNH:
      - Điểm tối nằm BÊN TRONG ROI,
      - Tiếp xúc TỪ DƯỚI lên (y >= ymax),
      - Mép TRONG của mỗi lông mày (sát trục giữa khuôn mặt).
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

        # Vành ngoài của ROI (rim outside)
        rim = _dilate_mask(roi, k=ROI_TOUCH_DILATE, iters=1)
        rim_outside = cv2.bitwise_and(rim, cv2.bitwise_not(roi))

        xmin, ymin = int(poly[:,0].min()), int(poly[:,1].min())
        xmax, ymax = int(poly[:,0].max()), int(poly[:,1].max())
        yy, xx = np.indices(dark_any.shape, dtype=np.int32)

        # CHỈ cho TOP + HAI BÊN; LOẠI BOTTOM
        top_allow   = ((yy < ymin).astype(np.uint8) * 255)
        left_allow  = ((xx < xmin).astype(np.uint8) * 255)
        right_allow = ((xx > xmax).astype(np.uint8) * 255)

        # -- MỚI: với HAI BÊN, chỉ tính phần "2/3 trên" (hoặc tỷ lệ cấu hình)
        roi_h = max(1, ymax - ymin)
        y_cut = int(ymin + EYEBROW_SIDE_TOP_RATIO * roi_h)   # ví dụ ~ ymin + 0.67*h
        side_top_mask = ((yy <= y_cut).astype(np.uint8) * 255)

        left_allow_top  = cv2.bitwise_and(left_allow,  side_top_mask)
        right_allow_top = cv2.bitwise_and(right_allow, side_top_mask)

        # Cho phép: TOP (toàn phần) + SIDES (chỉ 2/3 trên)
        allowed_dirs = cv2.bitwise_or(top_allow, cv2.bitwise_or(left_allow_top, right_allow_top))

        # Áp lên vành ngoài ROI
        allowed_rim = cv2.bitwise_and(rim_outside, allowed_dirs)

        # BỎ “mép trong” gần sống mũi bằng guard theo bề rộng ROI
        roi_w = max(1, xmax - xmin)
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
DEBUG_SHOE = bool(int(os.getenv("DEBUG_SHOE", "1")))  # 1: bật overlay debug vùng da bàn chân
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
def check_leg_skin_exposure(test_image, pose_landmarks, gender="female"):
    """
    Luôn dùng tiêu chí 'nữ':
      - Start: lùi từ HIP xuống phía KNEE (LEG_START_FROM_HIP_RATIO/MINPX)
      - End  : từ KNEE kéo ra phía ANKLE một đoạn (FEMALE_KNEE_EXT_RATIO/MINPX)
    Trả (overall, detail, (L_poly, R_poly)) như cũ.
    """
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

        # START: từ HIP xuống phía KNEE
        v_hk = knee - hip
        L_hk = np.linalg.norm(v_hk)
        v_hk_dir = v_hk / L_hk if L_hk > 1e-3 else np.array([0.0, 1.0], dtype=np.float32)
        start_shift = max(LEG_START_FROM_HIP_MINPX, LEG_START_FROM_HIP_RATIO * L_hk)
        start_pt = hip + v_hk_dir * start_shift

        # END: như NỮ — từ KNEE kéo về phía ANKLE một đoạn
        dir_ka = ankle - knee
        L_ka = np.linalg.norm(dir_ka)
        if L_ka > 1e-3:
            dir_ka /= L_ka
        ext_len = max(FEMALE_KNEE_EXT_MINPX, FEMALE_KNEE_EXT_RATIO * max(L_ka, 1e-3))
        end_pt = knee + dir_ka * ext_len

        return _leg_band_polygon(test_image.shape,
                                 tuple(start_pt.astype(np.int32)),
                                 tuple(end_pt.astype(np.int32)))

    L_poly = make_poly("left")
    R_poly = make_poly("right")

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


def _clip_box(x1, y1, x2, y2, w, h):
    return _clip_box_raw(x1, y1, x2, y2, w, h)


def _crop_box(img, box):
    return img[box["y1"]:box["y2"], box["x1"]:box["x2"]]


def _shoe_boxes(image, landmarks):
    """Sinh crop box quanh mắt cá chân (ankle) theo cấu hình half-width/top/bottom."""
    h, w = image.shape[:2]
    lm = landmarks.landmark
    res = {}
    for label, a_idx in [("left_shoe", mp_pose.PoseLandmark.LEFT_ANKLE),
                         ("right_shoe", mp_pose.PoseLandmark.RIGHT_ANKLE)]:
        px = int(lm[a_idx].x * w)
        py = int(lm[a_idx].y * h)
        box = _clip_box(
            px - SHOE_CROP_HALF_W, py - SHOE_CROP_TOP,
            px + SHOE_CROP_HALF_W, py + SHOE_CROP_BOTTOM, w, h
        )
        if box:
            res[label] = box
    return res


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

    # Vai

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
def _hand_visibility_ok(landmarks, side="left", thresh=0.59):
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
def run_inference(test_image, test_image_path=None, gender="female"):
    """
    Trả:
      - output_path: đường dẫn ảnh đã annotate
      - results: dict gồm {shoulders, arms, left_arm, right_arm, smile, left_shoe, right_shoe, shoes}
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    results = {}

    # 1) Pose
    pose_landmarks = detect_pose_landmarks(test_image)
    if not pose_landmarks:
        results["_error"] = "Vui lòng chụp toàn thân, đủ sáng và không che khuất."
        # Không cần lưu ảnh annotate khi lỗi; vẫn trả đường dẫn dự kiến (hoặc None)
        return None, results

    # 2) Vai

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

    # Vẽ cảnh báo nếu FAIL (viền đỏ quanh dải kiểm tra)
    if leg_overall != "pass":
        for poly in leg_polys:
            if poly is not None:
                cv2.polylines(test_image, [poly], isClosed=True, color=(0, 0, 255), thickness=FAIL_BOX_THICKNESS)
                # dán nhãn nhỏ
                cx, cy = int(poly[:,0].mean()), int(poly[:,1].mean())
                cv2.putText(test_image, "leg_skin: fail", (cx-40, cy-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    # 4) Nụ cười + box khuôn mặt (crop kiểu file process + lưu ảnh)
    face_crop, face_box = get_face_smile_box_like_process(test_image, pose_landmarks)
    results["smile"] = detect_smile(face_crop)

    # Lưu crop mặt như file upload (check/face_crop/<basename>_face.jpg)
    if SAVE_SMILE_FACE_CROP and face_box is not None:
        out_face_path = os.path.join(OUTPUT_FOLDER, SMILE_FACE_CROP_NAME)
        saved_ok = _save_face_crop(test_image, face_box, out_face_path)
        results["face_crop_saved"] = out_face_path if saved_ok else None
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

    # (MỚI) Lưu bản OVERLAY của crop mặt (sau khi check_eyebrow_hair_overlap đã vẽ ROI + tóc)
    if DEBUG_FACE_SAVE and facemesh_box is not None:
        _save_face_crop(test_image, facemesh_box, os.path.join(OUTPUT_FOLDER, FACE_CROP_OVERLAY_NAME))

    # Vẽ cảnh báo nếu FAIL: viền poly lông mày
    if brow_overall != "pass":
        if L_poly is not None:
            cv2.polylines(test_image, [L_poly], True, FAIL_BOX_COLOR, FAIL_BOX_THICKNESS)
            cx, cy = int(L_poly[:,0].mean()), int(L_poly[:,1].mean())
            cv2.putText(test_image, "brow: fail", (cx-30, cy-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, FAIL_BOX_COLOR, 2)
        if R_poly is not None:
            cv2.polylines(test_image, [R_poly], True, FAIL_BOX_COLOR, FAIL_BOX_THICKNESS)
            cx, cy = int(R_poly[:,0].mean()), int(R_poly[:,1].mean())
            cv2.putText(test_image, "brow: fail", (cx-30, cy-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, FAIL_BOX_COLOR, 2)


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

    # 7) Lưu ảnh
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, "test_result.jpg")
    cv2.imwrite(output_path, annotated)

    return output_path, results
