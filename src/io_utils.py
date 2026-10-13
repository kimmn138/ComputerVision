
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
    """노면 내부의 포트홀·차선처럼 완전히 둘러싸인 빈 영역만 채운다."""

    h, w = mask.shape

    # 영상 가장자리의 모든 배경을 서로 연결하기 위해 바깥쪽에 1px짜리 0 테두리를 추가한다.
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
    # floodFill용 마스크는 대상 영상보다 가로·세로가 각각 2픽셀 더 커야 한다.
    flood_mask = np.zeros(
        (h + 4, w + 4),
        dtype=np.uint8,
    )
    # 바깥 배경을 모두 채운다.
    cv.floodFill(
        flood,
        flood_mask,
        (0, 0),
        255,
    )
    # flood 후에도 0인 영역만 노면에 완전히 둘러싸인 내부 구멍이다.
    holes = cv.bitwise_not(flood)

    filled = cv.bitwise_or(
        padded,
        holes,
    )
    # 처음 추가했던 1px 테두리를 제거한다.
    return filled[1:-1, 1:-1]

def _lane_pixels(L, a, b, L_med):
    """흰 차선·노란 중앙선이 노면 연결을 끊지 않게, 차선처럼 보이는 픽셀(밝고 가는 무채색·노란 줄)을 찾는다(v2 차선 (가)).
    가는 줄 = 가로 또는 세로 ROAD_LANE_KSIZE 선 커널 탑햇이 ROAD_LANE_DELTA 이상(주변보다 밝음). 한 방향으로만 가늘어도 되므로
    세로 차선·가로 정지선을 모두 잡고, 하늘·흰 벽·밝은 인도처럼 두 방향으로 넓은 밝은 면은 빠진다.
    색 = 무채색(a·b가 128에서 ROAD_LANE_WHITE_CHROMA 안) 또는 노랑(b − 128 ≥ ROAD_LANE_YELLOW_B이고 |a − 128|의 2배 이상).
    씨앗 노면 밝기(L_med)보다 밝아야 한다. 받은 배열은 바꾸지 않고 bool (H, W)를 돌려준다."""
    k = C.ROAD_LANE_KSIZE
    top_h = cv.morphologyEx(L, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (k, 1)))
    top_v = cv.morphologyEx(L, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (1, k)))
    thin_bright = (np.maximum(top_h, top_v) >= C.ROAD_LANE_DELTA) & (L > L_med)
    da, db = a - 128, b - 128
    white = np.sqrt(da ** 2 + db ** 2) < C.ROAD_LANE_WHITE_CHROMA
    yellow = (db >= C.ROAD_LANE_YELLOW_B) & (db >= 2 * np.abs(da))
    return thin_bright & (white | yellow)


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

    # 시드 픽셀(bool). 기본은 시드 상자 전체, (A2r) 켜져 있으면 노면답지 않은 픽셀을 뺀다
    seed_sel = np.zeros((h, w), bool)
    seed_sel[y0:y1, x0:x1] = True
    if C.ROAD_SEED_FILTER:
        good = seed_sel & ~_not_road_seed(img, L, a, b)
        if np.count_nonzero(good) >= C.ROAD_SEED_KEEP_MIN * np.count_nonzero(seed_sel):
            seed_sel = good

    seed_L = L[seed_sel]
    seed_a = a[seed_sel]
    seed_b = b[seed_sel]

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
    if C.ROAD_L_RANGE_MAX > 0:                          # (A3) 시드가 그늘·햇빛·지평선을 섞어 허용 폭이 0~255를 덮지 않게
        l_threshold = min(l_threshold, C.ROAD_L_RANGE_MAX)

    # 색과 밝기가 모두 노면과 비슷한 픽셀
    candidate = (
        (chroma_dist < C.ROAD_CHROMA_MAX) &
        (np.abs(L - L_med) < l_threshold)
    )

    # (v2 8a 그림자·햇빛) 색이 씨앗과 더 가까우면(ROAD_SHADOW_CHROMA) 밝기 범위를 위아래 모두 ROAD_SHADOW_L_K배로 넓힘.
    # 그림자와 햇빛은 밝기만 바꾸고 색은 거의 그대로라, 씨앗이 어느 쪽에 있든 반대쪽 노면을 잇는다 (Plan.md 3.3 (5))
    if C.ROAD_SHADOW_L_K > 0:
        candidate = candidate | (
            (chroma_dist < C.ROAD_SHADOW_CHROMA) &
            (np.abs(L - L_med) < C.ROAD_SHADOW_L_K * l_threshold)
        )

    # (v2 8c 건물·인도·비탈) 창 안 에지 밀도가 씨앗 창들의 상위 분위보다 크면 노면이 아닌 것으로 봄. 균열이 많은 노면도 잘릴 수 있다
    if C.ROAD_EDGE_DENSITY:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edges = (cv.Canny(gray, *C.ROAD_EDGE_CANNY) > 0).astype(np.float32)
        density = cv.boxFilter(edges, -1, (C.ROAD_EDGE_KSIZE, C.ROAD_EDGE_KSIZE), borderType=cv.BORDER_REFLECT)
        limit = float(np.quantile(density[y0:y1, x0:x1], C.ROAD_EDGE_QUANTILE))
        candidate = candidate & (density <= limit)

    # (v2 차선 (가)) 차선 픽셀도 노면으로 다뤄 차선 건너편 노면과 이어지게 함
    if C.ROAD_LANE_BRIDGE:
        candidate = candidate | _lane_pixels(L, a, b, L_med)

    # (B2) 노면 위의 가늘고 긴 흰·노란 줄을 노면으로: 건너편 노면과 이어지게
    if C.ROAD_LANE_AS_ROAD:
        candidate = candidate | _lane_on_road(L, a, b, L_med, candidate)

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

    # (v2 차선 (나)) 연결 요소를 고르기 전에 닫힘: 닫힘 폭보다 가는 차선으로 끊긴 건너편 노면을 이음
    if C.ROAD_CLOSE_BEFORE_SEED:
        mask = cv.morphologyEx(
            mask,
            cv.MORPH_CLOSE,
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (C.ROAD_CLOSE_KSIZE, C.ROAD_CLOSE_KSIZE))
        )

    # seed 영역과 연결된 노면 덩어리만 유지
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    keep_labels = set()

    seed_labels = labels[seed_sel]

    for label in np.unique(seed_labels):
        if label != 0:
            keep_labels.add(int(label))

    # (B1) 시드와 이어지지 않았어도 큰 덩어리, 아래쪽에 닿은 덩어리는 노면으로(흰 선에 막힌 건너편)
    if C.ROAD_KEEP_BLOBS:
        bottom = int(h * (1 - C.ROAD_BLOB_BOTTOM))
        for label in range(1, num_labels):
            area = stats[label, cv.CC_STAT_AREA]
            touches = stats[label, cv.CC_STAT_TOP] + stats[label, cv.CC_STAT_HEIGHT] > bottom
            if area >= C.ROAD_BLOB_BIG * h * w or (area >= C.ROAD_BLOB_SMALL * h * w and touches):
                keep_labels.add(label)

    connected = np.isin(labels, list(keep_labels)).astype(np.uint8) * 255

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

    # (v2 8b 보닛·와이퍼) 아래 테두리부터 위로, 결이 씨앗보다 훨씬 매끈한 행이 이어지는 띠를 뺀다
    if C.ROAD_HOOD_CUT:
        connected[_hood_rows(img, (x0, y0, x1, y1))] = 0

    return connected.astype(np.uint8)


def _not_road_seed(bgr, L, a, b):
    """(A2r) 시드로 쓰면 안 되는 픽셀(bool): 그 사진 ROI 아래쪽(ROAD_SEED_REF_BOX)의 채도 중앙값보다 ROAD_SEED_CHROMA_REL 넘게 채도가 높은
    픽셀(풀·차), 밝은 표시(검출기 1단계와 같은 기준: 1/DARK_BG_DOWNSCALE 중앙값 거친 배경 + DARK_BRIGHT_DELTA보다 밝음, 흰 선),
    노란 선 색(차선 (가)와 같은 노랑 규칙). 채도를 사진마다 상대적으로 보므로 빨간·푸른 아스팔트 자체는 빠지지 않는다."""
    h, w = L.shape
    da, db = a - 128, b - 128
    chroma = np.sqrt(da ** 2 + db ** 2)
    fx0, fy0, fx1, fy1 = C.ROAD_SEED_REF_BOX
    ref = float(np.median(chroma[int(h * fy0):int(h * fy1), int(w * fx0):int(w * fx1)]))
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    s = C.DARK_BG_DOWNSCALE
    size = (max(1, int(round(w / s))), max(1, int(round(h / s))))
    coarse = cv.resize(cv.medianBlur(cv.resize(gray, size, interpolation=cv.INTER_AREA), C.DARK_BG_KSIZE_SMALL | 1), (w, h),
                       interpolation=cv.INTER_LINEAR).astype(np.float32)
    bright = gray.astype(np.float32) > coarse + C.DARK_BRIGHT_DELTA
    yellow = (db >= C.ROAD_LANE_YELLOW_B) & (db >= 2 * np.abs(da))
    return (chroma - ref > C.ROAD_SEED_CHROMA_REL) | bright | yellow


def _lane_on_road(L, a, b, L_med, candidate):
    """(B2) _lane_pixels 덩어리 가운데 가늘고 길며(긴 변 ÷ 짧은 변 ≥ ROAD_LANE_ELONG) 둘레 고리(ROAD_LANE_RING px)의 ROAD_LANE_RING_ROAD
    이상이 노면 색(candidate)인 줄만 골라, 흐림 번짐 폭(ROAD_LANE_EDGE)만큼 넓혀 bool로 돌려준다. 하늘·벽 같은 밝은 면은 둘레가 노면이 아니라 빠진다."""
    lane = _lane_pixels(L, a, b, L_med).astype(np.uint8)
    n, labels, stats, _ = cv.connectedComponentsWithStats(lane, connectivity=8)
    out = np.zeros(lane.shape, bool)
    r = C.ROAD_LANE_RING
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    for i in range(1, n):
        if stats[i, cv.CC_STAT_AREA] < C.ROAD_LANE_MIN_AREA:
            continue
        x, y, bw, bh = stats[i, :4]
        comp = labels[y:y + bh, x:x + bw] == i
        (_, _), (rw, rh), _ = cv.minAreaRect(np.column_stack(np.nonzero(comp)[::-1]).astype(np.float32))
        if max(rw, rh) < C.ROAD_LANE_ELONG * max(1.0, min(rw, rh)):
            continue
        X0, Y0 = max(0, x - r), max(0, y - r)
        X1, Y1 = min(lane.shape[1], x + bw + r), min(lane.shape[0], y + bh + r)
        local = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
        local[y - Y0:y - Y0 + bh, x - X0:x - X0 + bw] = comp
        ring = (cv.dilate(local, k) > 0) & (local == 0) & (lane[Y0:Y1, X0:X1] == 0)
        if ring.any() and candidate[Y0:Y1, X0:X1][ring].mean() >= C.ROAD_LANE_RING_ROAD:
            out[Y0:Y1, X0:X1] |= local > 0
    d = 2 * C.ROAD_LANE_EDGE + 1
    return cv.dilate(out.astype(np.uint8), cv.getStructuringElement(cv.MORPH_ELLIPSE, (d, d))) > 0


def _hood_rows(bgr, seed_box):
    """노면 띠 아래 테두리에 붙은 보닛·와이퍼 행을 bool (H,) 배열로 찾는다. 행마다 결(ROAD_HOOD_TEX_KSIZE 창의 흑백 국소 표준편차)의
    중앙값을 내어, 맨 아래 행부터 위로 '씨앗 결 중앙값 × ROAD_HOOD_TEX_RATIO'보다 작은 행이 이어지는 동안만 보닛으로 본다.
    보닛은 도장면이라 매끈하고 아스팔트는 골재 결이 있다. 매끈한 행이 아래 테두리에 붙어 있지 않으면 아무 행도 빼지 않는다."""
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY).astype(np.float32)
    k = (C.ROAD_HOOD_TEX_KSIZE, C.ROAD_HOOD_TEX_KSIZE)
    mean = cv.boxFilter(gray, -1, k, borderType=cv.BORDER_REFLECT)
    sq = cv.boxFilter(gray * gray, -1, k, borderType=cv.BORDER_REFLECT)
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    x0, y0, x1, y1 = seed_box
    limit = float(np.median(std[y0:y1, x0:x1])) * C.ROAD_HOOD_TEX_RATIO
    smooth = np.median(std, axis=1) < limit
    rough_from_bottom = np.flatnonzero(~smooth[::-1])                # 아래에서 센 '매끈하지 않은' 행의 순번
    n = int(rough_from_bottom[0]) if rough_from_bottom.size else len(smooth)
    rows = np.zeros(gray.shape[0], bool)
    rows[len(smooth) - n:] = True                                      # 아래 테두리부터 이어진 매끈한 행 n개
    return rows

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
