
"""사진 읽기, 가로 640px 맞춤, ROI 자르기, 노면 마스크, roi.csv·사진 이름 해석."""
import os
import csv
import numpy as np
import cv2 as cv

from src import config as C


def load_image(path):
    """사진을 BGR 컬러로 읽는다. 못 읽으면 None 대신 FileNotFoundError를 내서 바로 알아채게 한다."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    # 첫 시도: 기본 imread
    img = cv.imread(path, cv.IMREAD_COLOR)

    # 다시 시도: 윈도의 한글 경로는 imread가 못 읽어서 바이트로 읽은 뒤 디코딩
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
    """사진 크기가 달라도 파라미터가 같은 뜻을 갖도록 가로를 width로 맞춘다(세로는 비율대로)."""
    h, w = img.shape[:2]
    if w == width:
        return img.copy()

    interp = cv.INTER_AREA if width < w else cv.INTER_LINEAR    # 줄일 때 AREA, 키울 때 LINEAR
    return cv.resize(img, (width, round(h * width / w)), interpolation=interp)


def crop_roi(img, top=C.ROI_TOP_DEFAULT, bottom=C.ROI_BOTTOM_DEFAULT):
    """노면 밖(하늘·차·건물)을 빼려고 위·아래 비율(0~1)로 잘라 ROI만 남긴다.
    (ROI 영상, 위에서 잘라 낸 픽셀 수 y0)를 돌려준다."""
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
    """흰 차선·노란 중앙선처럼 밝고 가는 줄의 픽셀(bool)을 찾는다. 차선이 노면을 둘로 끊지 않게 노면으로 다룰 때 쓴다.
    한 방향으로만 가늘면 되므로 세로 차선·가로 정지선은 잡히고, 하늘·흰 벽처럼 넓은 밝은 면은 빠진다."""
    k = C.ROAD_LANE_KSIZE
    top_h = cv.morphologyEx(L, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (k, 1)))
    top_v = cv.morphologyEx(L, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (1, k)))
    thin_bright = (np.maximum(top_h, top_v) >= C.ROAD_LANE_DELTA) & (L > L_med)
    da, db = a - 128, b - 128
    white = np.sqrt(da ** 2 + db ** 2) < C.ROAD_LANE_WHITE_CHROMA
    yellow = (db >= C.ROAD_LANE_YELLOW_B) & (db >= 2 * np.abs(da))
    return thin_bright & (white | yellow)


def road_mask(bgr_roi):
    """ROI 아래 가운데(시드 영역)와 색·밝기가 비슷하고 이어진 영역을 노면 마스크(노면 255, 나머지 0)로 만든다."""

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

    # 시드 영역: ROI 아래 가운데 상자(노면일 가능성이 가장 큰 곳)
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

    # 시드 픽셀: 상자에서 풀·차·흰 선처럼 노면답지 않은 픽셀을 뺌(남는 픽셀이 너무 적으면 상자 전체)
    seed_sel = np.zeros((h, w), bool)
    seed_sel[y0:y1, x0:x1] = True
    if C.ROAD_SEED_FILTER:
        good = seed_sel & ~_not_road_seed(img, L, a, b)
        if np.count_nonzero(good) >= C.ROAD_SEED_KEEP_MIN * np.count_nonzero(seed_sel):
            seed_sel = good

    seed_L = L[seed_sel]
    seed_a = a[seed_sel]
    seed_b = b[seed_sel]

    # 시드 영역의 대표 색·밝기(중앙값)
    L_med = float(np.median(seed_L))
    a_med = float(np.median(seed_a))
    b_med = float(np.median(seed_b))

    # 색 거리: Lab의 a·b 평면에서 시드 색과 얼마나 다른가
    chroma_dist = np.sqrt(
        (a - a_med) ** 2 +
        (b - b_med) ** 2
    )

    # 밝기 허용 폭: MAD(중앙값 절대 편차)를 써서 튀는 픽셀의 영향을 덜 받음
    mad = float(
        np.median(
            np.abs(seed_L - L_med)
        )
    )

    l_threshold = (
        C.ROAD_L_K *
        (1.4826 * mad + 2.0)
    )
    if C.ROAD_L_RANGE_MAX > 0:                          # 허용 폭 상한: 시드에 그늘·햇빛이 섞여도 모든 밝기를 노면으로 보지 않게
        l_threshold = min(l_threshold, C.ROAD_L_RANGE_MAX)

    # 색과 밝기가 모두 노면과 비슷한 픽셀
    candidate = (
        (chroma_dist < C.ROAD_CHROMA_MAX) &
        (np.abs(L - L_med) < l_threshold)
    )

    # 그림자·햇빛: 색이 시드와 아주 가까우면 밝기 허용 폭을 넓힘(그림자는 밝기만 바꾸고 색은 거의 그대로라)
    if C.ROAD_SHADOW_L_K > 0:
        candidate = candidate | (
            (chroma_dist < C.ROAD_SHADOW_CHROMA) &
            (np.abs(L - L_med) < C.ROAD_SHADOW_L_K * l_threshold)
        )

    # 에지가 시드 영역보다 훨씬 빽빽한 곳(건물·인도)은 노면에서 뺌
    if C.ROAD_EDGE_DENSITY:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edges = (cv.Canny(gray, *C.ROAD_EDGE_CANNY) > 0).astype(np.float32)
        density = cv.boxFilter(edges, -1, (C.ROAD_EDGE_KSIZE, C.ROAD_EDGE_KSIZE), borderType=cv.BORDER_REFLECT)
        limit = float(np.quantile(density[y0:y1, x0:x1], C.ROAD_EDGE_QUANTILE))
        candidate = candidate & (density <= limit)

    # 차선 픽셀을 모두 노면으로 다뤄 차선 건너편 노면과 이어지게 함
    if C.ROAD_LANE_BRIDGE:
        candidate = candidate | _lane_pixels(L, a, b, L_med)

    # 노면 위의 가늘고 긴 흰·노란 줄만 골라 노면으로 더함(건너편 노면과 이어지게)
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

    # 연결 영역을 고르기 전에 닫힘 연산으로, 가는 차선에 끊긴 건너편 노면을 이음
    if C.ROAD_CLOSE_BEFORE_SEED:
        mask = cv.morphologyEx(
            mask,
            cv.MORPH_CLOSE,
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (C.ROAD_CLOSE_KSIZE, C.ROAD_CLOSE_KSIZE))
        )

    # 시드 영역과 이어진 노면 덩어리만 남김
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    keep_labels = set()

    seed_labels = labels[seed_sel]

    for label in np.unique(seed_labels):
        if label != 0:
            keep_labels.add(int(label))

    # 시드와 떨어져 있어도 큰 덩어리나 아래쪽에 닿은 덩어리는 노면으로 남김(흰 선에 막힌 건너편 노면)
    if C.ROAD_KEEP_BLOBS:
        bottom = int(h * (1 - C.ROAD_BLOB_BOTTOM))
        for label in range(1, num_labels):
            area = stats[label, cv.CC_STAT_AREA]
            touches = stats[label, cv.CC_STAT_TOP] + stats[label, cv.CC_STAT_HEIGHT] > bottom
            if area >= C.ROAD_BLOB_BIG * h * w or (area >= C.ROAD_BLOB_SMALL * h * w and touches):
                keep_labels.add(label)

    connected = np.isin(labels, list(keep_labels)).astype(np.uint8) * 255

    # 남길 덩어리가 하나도 없으면 전부 지우지 않고 후보 전체를 씀
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

    # 아래 테두리에 붙은 보닛·와이퍼(질감이 매끈한 행)를 뺌
    if C.ROAD_HOOD_CUT:
        connected[_hood_rows(img, (x0, y0, x1, y1))] = 0

    return connected.astype(np.uint8)


def _not_road_seed(bgr, L, a, b):
    """시드로 쓰면 안 되는 픽셀(bool)을 찾는다: 이 사진의 노면보다 채도가 유독 높은 픽셀(풀·차), 주변보다 밝은 표시(흰 선), 노란 선.
    채도는 사진마다 상대적으로 비교하므로 붉거나 푸른 아스팔트는 빠지지 않는다."""
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
    """차선 픽셀 덩어리 가운데 가늘고 길며 둘레가 노면 색인 줄만 골라, 노면으로 더할 픽셀(bool)을 돌려준다.
    하늘·벽 같은 밝은 면은 둘레가 노면이 아니라서 빠진다."""
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
    """ROI 아래 테두리에 붙은 보닛·와이퍼 행(bool (H,))을 찾는다.
    도장면은 아스팔트보다 질감이 매끈하므로, 맨 아래부터 질감이 시드 영역보다 훨씬 작은 행이 이어지는 동안만 보닛으로 본다."""
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
    """roi.csv에서 사진마다 ROI 범위(top·bottom 비율)와 촬영 조건을 읽는다.
    {파일명: {"top", "bottom", "condition"}}을 돌려주고, 파일이 없으면 빈 dict, 형식이 틀린 행이 있으면 ValueError."""
    if not os.path.exists(path):
        return {}

    roi_dict = {}
    try:
        # utf-8-sig: 엑셀이 붙이는 BOM을 무시
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
    그 밖은 제공 영상 → {source: "provided", damage: "", condition: ""} (조건은 roi.csv에서 읽음)
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
