
"""영상 읽기·크기 맞추기·노면 띠(ROI) 자르기·노면 마스크, roi.csv와 사진 이름 해석."""
import os
import csv
import numpy as np
import cv2 as cv

from src import config as C


def load_image(path):
    """사진을 BGR 컬러로 읽는다. 못 읽으면 FileNotFoundError를 낸다(한글 경로도 지원)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    img = cv.imread(path, cv.IMREAD_COLOR)
    
    # imread가 실패하면(윈도 한글 경로 등) 바이트로 읽어 다시 시도
    if img is None:
        try:
            img_array = np.fromfile(path, np.uint8)
            img = cv.imdecode(img_array, cv.IMREAD_COLOR)
        except Exception:
            pass

    if img is None:
        raise FileNotFoundError(f"이미지 파일을 읽을 수 없습니다: {path}")
        
    return img


def resize_width(img, width=C.TARGET_WIDTH):
    """가로를 width로 맞춘다(세로는 비율대로). 크기가 같아야 파라미터가 모든 사진에서 같은 뜻을 가진다."""
    h, w = img.shape[:2]
    if w == width:
        return img.copy()

    interp = cv.INTER_AREA if width < w else cv.INTER_LINEAR    # 줄일 때 AREA, 키울 때 LINEAR
    return cv.resize(img, (width, round(h * width / w)), interpolation=interp)


def crop_roi(img, top=C.ROI_TOP_DEFAULT, bottom=C.ROI_BOTTOM_DEFAULT):
    """위·아래 비율(0~1)로 잘라 노면 띠만 남긴다. (노면 띠, 위에서 잘라 낸 픽셀 수 y0)를 돌려준다."""
    h = img.shape[0]
    y0, y1 = int(h * top), int(h * bottom)
    
    # 최소 1줄은 남게 범위 제한
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))
    
    return img[y0:y1].copy(), y0

def _fill_holes(mask):
    """마스크 안에서 완전히 둘러싸인 구멍(포트홀·차선 자리)을 채운다."""

    h, w = mask.shape

    # 1px 테두리를 붙여 바깥 배경을 하나로 이은 뒤 바깥에서부터 채움
    padded = cv.copyMakeBorder(
        mask,
        1,
        1,
        1,
        1,
        cv.BORDER_CONSTANT,
        value=0,
    )
    flood = padded.copy()
    flood_mask = np.zeros(
        (h + 4, w + 4),
        dtype=np.uint8,
    )
    cv.floodFill(
        flood,
        flood_mask,
        (0, 0),
        255,
    )
    # 채워지지 않은 곳 = 둘러싸인 구멍
    holes = cv.bitwise_not(flood)

    filled = cv.bitwise_or(
        padded,
        holes,
    )
    return filled[1:-1, 1:-1]

def _lane_pixels(L, a, b, L_med):
    """차선처럼 보이는 픽셀(노면보다 밝고 가는 흰색·노란색 줄)을 bool 영상으로 돌려준다.
    가로·세로 탑햇으로 가는 줄만 골라, 하늘·흰 벽처럼 넓은 밝은 면은 빠진다."""
    k = C.ROAD_LANE_KSIZE
    top_h = cv.morphologyEx(L, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (k, 1)))
    top_v = cv.morphologyEx(L, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (1, k)))
    thin_bright = (np.maximum(top_h, top_v) >= C.ROAD_LANE_DELTA) & (L > L_med)
    da, db = a - 128, b - 128
    white = np.sqrt(da ** 2 + db ** 2) < C.ROAD_LANE_WHITE_CHROMA
    yellow = (db >= C.ROAD_LANE_YELLOW_B) & (db >= 2 * np.abs(da))
    return thin_bright & (white | yellow)


def road_mask(bgr_roi):
    """노면 마스크(노면 255, 아님 0)를 만든다.
    띠 아래 가운데(씨앗)의 색·밝기와 비슷하고 씨앗과 이어진 영역을 노면으로 본다."""

    img = bgr_roi.copy()

    h, w = img.shape[:2]

    # 빈 영상이면 빈 마스크
    if h == 0 or w == 0:
        return np.zeros((h, w), np.uint8)
    # 골재 결·잡음이 색 비교를 흔들지 않게 흐린 뒤 Lab으로
    blurred = cv.GaussianBlur(
        img,
        (0, 0),
        C.ROAD_BLUR_SIGMA
    )
    lab = cv.cvtColor(blurred, cv.COLOR_BGR2LAB)

    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    # 씨앗 영역: 노면일 가능성이 가장 높은 아래 가운데
    x0_ratio, y0_ratio, x1_ratio, y1_ratio = C.ROAD_SEED_BOX

    x0 = int(w * x0_ratio)
    y0 = int(h * y0_ratio)
    x1 = int(w * x1_ratio)
    y1 = int(h * y1_ratio)

    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))

    seed = lab[y0:y1, x0:x1]

    seed_L = seed[:, :, 0].astype(np.float32)
    seed_a = seed[:, :, 1].astype(np.float32)
    seed_b = seed[:, :, 2].astype(np.float32)
    
    # 씨앗의 대표 색·밝기 (중앙값)
    L_med = float(np.median(seed_L))
    a_med = float(np.median(seed_a))
    b_med = float(np.median(seed_b))

    # 씨앗 색과의 거리 (a·b 평면)
    chroma_dist = np.sqrt(
        (a - a_med) ** 2 +
        (b - b_med) ** 2
    )

    # 밝기 허용 범위: MAD(중앙값 절대 편차)로 잰 강건한 퍼짐
    mad = float(
        np.median(
            np.abs(seed_L - L_med)
        )
    )

    l_threshold = (
        C.ROAD_L_K *
        (1.4826 * mad + 2.0)
    )

    # 색과 밝기가 모두 씨앗과 비슷한 픽셀
    candidate = (
        (chroma_dist < C.ROAD_CHROMA_MAX) &
        (np.abs(L - L_med) < l_threshold)
    )

    # 차선 픽셀도 노면으로 다뤄 차선 건너편 노면과 이어지게 함
    if C.ROAD_LANE_BRIDGE:
        candidate = candidate | _lane_pixels(L, a, b, L_med)

    mask = candidate.astype(np.uint8) * 255

    # 작은 점 제거
    open_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (C.ROAD_OPEN_KSIZE, C.ROAD_OPEN_KSIZE)
    )

    mask = cv.morphologyEx(
        mask,
        cv.MORPH_OPEN,
        open_kernel
    )

    # 씨앗 연결을 보기 전에 닫힘: 가는 차선으로 끊긴 건너편 노면을 이음
    if C.ROAD_CLOSE_BEFORE_SEED:
        mask = cv.morphologyEx(
            mask,
            cv.MORPH_CLOSE,
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (C.ROAD_CLOSE_KSIZE, C.ROAD_CLOSE_KSIZE))
        )

    # 씨앗과 이어진 덩어리만 남김
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    keep_labels = set()

    seed_labels = labels[y0:y1, x0:x1]

    for label in np.unique(seed_labels):
        if label != 0:
            keep_labels.add(int(label))

    connected = np.zeros_like(mask)

    for label in keep_labels:
        connected[labels == label] = 255

    # 씨앗과 이어진 덩어리가 없으면 후보 전체를 씀
    if not keep_labels:
        connected = mask.copy()

    # 균열·차선으로 생긴 틈 메우기
    close_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (C.ROAD_CLOSE_KSIZE, C.ROAD_CLOSE_KSIZE)
    )

    connected = cv.morphologyEx(
        connected,
        cv.MORPH_CLOSE,
        close_kernel
    )

    # 둘러싸인 구멍 채우기
    connected = _fill_holes(connected)

    return connected.astype(np.uint8)

def load_roi_table(path):
    """roi.csv에서 사진별 노면 띠 범위와 촬영 조건을 읽어 {파일명: {top, bottom, condition}}으로 돌려준다.
    파일이 없으면 빈 dict, 형식이 잘못되면 ValueError."""
    if not os.path.exists(path):
        return {}

    roi_dict = {}
    try:
        # utf-8-sig: 엑셀이 붙이는 BOM 대비
        with open(path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["file"].strip()
                top_val = float(row["roi_top"])
                bottom_val = float(row["roi_bottom"])
                condition = row.get("condition", "").strip()
                
                roi_dict[filename] = {
                    "top": top_val,
                    "bottom": bottom_val,
                    "condition": condition
                }
    except KeyError as e:
        raise ValueError(f"roi.csv에 필수 컬럼이 누락되었습니다: {e}")
    except ValueError as e:
        raise ValueError(f"roi.csv의 숫자 형식이 잘못되었습니다: {e}")

    return roi_dict


def _is_number(token):
    """번호 칸이 0~9 숫자로만 되어 있는지 본다."""
    return token.isascii() and token.isdigit()


def parse_name(path):
    """사진 파일 이름에서 출처·손상·촬영 조건을 읽는다.
    own_번호_손상_조건.jpg → own, pair번호_기호_조건.jpg → pair, 그 밖 → provided(제공 영상).
    own·pair 이름이 규칙을 어기면 ValueError."""
    filename = os.path.basename(path)
    tokens = os.path.splitext(filename)[0].split("_")

    # 1. 직접 촬영: own_번호_손상_조건 (예: own_01_crack_dark)
    if tokens[0] == "own":
        if len(tokens) != 4:
            raise ValueError(f"own 이름은 own_번호_손상_조건 4칸이어야 함 ({len(tokens)}칸): {filename}")
        _, number, damage, condition = tokens
        if not _is_number(number):
            raise ValueError(f"own 번호가 숫자가 아님 ('{number}'): {filename}")
        if damage not in C.DAMAGE_TAGS:
            raise ValueError(f"손상 '{damage}'가 {C.DAMAGE_TAGS}에 없음: {filename}")
        if condition not in C.CONDITION_TAGS:
            raise ValueError(f"조건 '{condition}'이 {C.CONDITION_TAGS}에 없음: {filename}")
        return {"source": "own", "damage": damage, "condition": condition}

    # 2. 매칭용 쌍: pair번호_기호_조건 (예: pair01_A_ref)
    if tokens[0].startswith("pair"):
        if len(tokens) != 3:
            raise ValueError(f"pair 이름은 pair번호_기호_조건 3칸이어야 함 ({len(tokens)}칸): {filename}")
        pair_token, role, condition = tokens
        number = pair_token[len("pair"):]
        if not _is_number(number):
            raise ValueError(f"pair 번호가 숫자가 아님 ('{number}'): {filename}")
        if role not in C.PAIR_ROLES:
            raise ValueError(f"기호 '{role}'가 {C.PAIR_ROLES}에 없음: {filename}")
        if condition not in C.PAIR_CONDITIONS:
            raise ValueError(f"쌍 조건 '{condition}'이 {C.PAIR_CONDITIONS}에 없음: {filename}")
        return {"source": "pair", "pair": number, "role": role, "condition": condition}

    # 3. 제공 영상
    return {"source": "provided", "damage": "", "condition": ""}
