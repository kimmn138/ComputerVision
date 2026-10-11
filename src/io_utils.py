
"""(파트 A) 영상 읽기·가로 크기 맞추기·노면(ROI) 자르기, roi.csv와 사진 이름 해석."""
import os
import csv
import numpy as np
import cv2 as cv

from src import config as C


def load_image(path):
    """사진을 BGR uint8 컬러로 읽는다. 못 읽으면 None 대신 FileNotFoundError를 내서 바로 알아채게 한다.
    윈도우 한글 경로 대비 cv.imdecode(np.fromfile(path, np.uint8), cv.IMREAD_COLOR)로 한 번 더 시도한다."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    # 1차 시도: 기본 imread
    img = cv.imread(path, cv.IMREAD_COLOR)
    
    # 2차 시도: 윈도우 환경 한글 경로 인식 오류 대비
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
    """영상마다 크기가 달라도 파라미터가 같은 뜻을 갖도록 가로를 width로 맞춘다.
    세로는 비율대로 바꾸고 채널 수는 그대로 둔다."""
    h, w = img.shape[:2]
    if w == width:
        return img.copy()

    interp = cv.INTER_AREA if width < w else cv.INTER_LINEAR    # 줄일 때 AREA, 키울 때 LINEAR
    return cv.resize(img, (width, round(h * width / w)), interpolation=interp)


def crop_roi(img, top=C.ROI_TOP_DEFAULT, bottom=C.ROI_BOTTOM_DEFAULT):
    """노면 밖(하늘·차·건물)을 빼려고 위·아래 비율(0~1)로 잘라 노면만 남긴다.
    (노면 영상, 위에서 잘라 낸 픽셀 수 y0)를 돌려준다."""
    h = img.shape[0]
    y0, y1 = int(h * top), int(h * bottom)
    
    # 안전하게 인덱스 범위 제한
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))
    
    return img[y0:y1].copy(), y0

def _fill_holes(mask):
    """노면 내부의 포트홀·차선처럼 둘러싸인 빈 영역을 채운다."""
    h, w = mask.shape

    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    # 가장자리의 배경 영역을 채운다.
    cv.floodFill(flood, flood_mask, (0, 0), 255)

    # floodFill 후에도 0으로 남아 있는 곳이 내부 구멍이다.
    holes = cv.bitwise_not(flood)

    return cv.bitwise_or(mask, holes)

def road_mask(bgr_roi):
    """노면과 비슷한 색·밝기의 연결 영역을 찾아 자동 노면 마스크를 만든다."""

    # 입력 영상을 직접 수정하지 않도록 복사한다.
    img = bgr_roi.copy()

    h, w = img.shape[:2]

    # 혹시 크기가 0인 잘못된 입력이 들어오면 빈 마스크 반환
    if h == 0 or w == 0:
        return np.zeros((h, w), np.uint8)
    # 작은 잡음의 영향을 줄이기 위해 흐림
    blurred = cv.GaussianBlur(
        img,
        (0, 0),
        C.ROAD_BLUR_SIGMA
    )
    lab = cv.cvtColor(blurred, cv.COLOR_BGR2LAB)

    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    # 아래 가운데 seed 영역 지정
    x0_ratio, y0_ratio, x1_ratio, y1_ratio = C.ROAD_SEED_BOX

    x0 = int(w * x0_ratio)
    y0 = int(h * y0_ratio)
    x1 = int(w * x1_ratio)
    y1 = int(h * y1_ratio)

    # 인덱스 안전 처리
    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))

    seed = lab[y0:y1, x0:x1]

    seed_L = seed[:, :, 0].astype(np.float32)
    seed_a = seed[:, :, 1].astype(np.float32)
    seed_b = seed[:, :, 2].astype(np.float32)
    
    # seed 영역의 대표값
    L_med = float(np.median(seed_L))
    a_med = float(np.median(seed_a))
    b_med = float(np.median(seed_b))

    # 색 거리: a-b 평면에서 seed 색과 얼마나 다른가
    chroma_dist = np.sqrt(
        (a - a_med) ** 2 +
        (b - b_med) ** 2
    )

    # 밝기 범위 계산
    # MAD는 극단적인 픽셀의 영향을 덜 받는 편
    mad = float(
        np.median(
            np.abs(seed_L - L_med)
        )
    )

    l_threshold = (
        C.ROAD_L_K *
        (1.4826 * mad + 2.0)
    )

    # 색과 밝기가 모두 노면과 비슷한 픽셀
    candidate = (
        (chroma_dist < C.ROAD_CHROMA_MAX) &
        (np.abs(L - L_med) < l_threshold)
    )

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

    # seed 영역과 연결된 노면 덩어리만 유지
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

    # seed와 연결된 것이 하나도 없다면
    # 너무 공격적으로 전부 없애지 않고 기존 후보를 사용
    if not keep_labels:
        connected = mask.copy()

    # 균열·차선 등에 의해 끊긴 작은 틈 연결
    close_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (C.ROAD_CLOSE_KSIZE, C.ROAD_CLOSE_KSIZE)
    )

    connected = cv.morphologyEx(
        connected,
        cv.MORPH_CLOSE,
        close_kernel
    )

    # 노면 내부의 포트홀·차선 등의 구멍 채우기
    connected = _fill_holes(connected)

    return connected.astype(np.uint8)

def load_roi_table(path):
    """사진마다 다른 노면 범위(top·bottom 비율)와 촬영 조건을 roi.csv에서 읽는다.
    utf-8-sig로 읽어 {파일명: {"top": float, "bottom": float, "condition": str}} 형태로 반환하며, 
    파일이 없으면 빈 dict를 반환하고 데이터 형식이 잘못된 행은 ValueError를 낸다."""
    if not os.path.exists(path):
        return {}

    roi_dict = {}
    try:
        # utf-8-sig를 사용하여 BOM 문제 해결
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
    """이름의 번호 칸이 0~9 숫자로만 되어 있는지 본다(빈 칸·'²' 같은 유니코드 숫자는 거짓)."""
    return token.isascii() and token.isdigit()


def parse_name(path):
    """사진 파일 이름에서 출처·손상·촬영 조건을 읽어 결과 표에 붙인다.
    own_번호_손상_조건.jpg → {source: "own", damage, condition}
    pair번호_기호_조건.jpg → {source: "pair", pair, role, condition}
    그 밖은 제공 영상(예: Japan_000107.jpg, United_States_004830.jpg) → {source: "provided", damage: "", condition: ""}.
    제공 영상 이름에는 손상·조건 정보가 없으므로 밑줄 개수와 상관없이 해석하지 않고, 조건은 roi.csv에서 읽는다.
    own·pair 이름이 칸 수·번호·config A 구역의 태그 규칙을 어기면 ValueError."""
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
