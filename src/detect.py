"""전처리한 노면 영상에서 균열·포트홀 후보 영역을 찾는다."""
import cv2 as cv
import numpy as np

from src import config as C


def region_darkness(gray, contour):
    """후보가 주변보다 얼마나 어두운지 잰다: 주변 띠의 어두운 쪽 밝기 − 후보 안 평균 밝기.
    균열처럼 어두운 선은 양수, 흰 차선처럼 밝은 선은 음수가 된다."""
    x, y, w, h = cv.boundingRect(contour)
    margin = C.DARK_RING_PX + 1

    x0 = max(x - margin, 0)
    y0 = max(y - margin, 0)
    x1 = min(x + w + margin, gray.shape[1])
    y1 = min(y + h + margin, gray.shape[0])

    inner = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv.drawContours(inner, [contour - [x0, y0]], -1, 255, -1)

    kernel = np.ones((2 * C.DARK_RING_PX + 1, 2 * C.DARK_RING_PX + 1), np.uint8)
    ring = cv.dilate(inner, kernel) & ~inner

    if not inner.any() or not ring.any():
        return 0.0

    patch = gray[y0:y1, x0:x1].astype(np.float32)
    surround = np.quantile(patch[ring > 0], C.DARK_RING_QUANTILE)

    return float(surround - patch[inner > 0].mean())


def shape_metrics(region_mask, contour, area):
    """균열(가늘고 긺)과 포트홀(둥긂)을 가르는 형태 지표를 잰다.
    thickness = 2·넓이/둘레, circ_inv = 둘레² / (4π·넓이) (원 ≈ 1, 길수록 큼), dt_width = 거리 변환으로 잰 폭."""
    perimeter = cv.arcLength(contour, True)
    # 1px 배경 테두리를 붙여 가장자리에서도 거리를 바르게 잼
    padded = cv.copyMakeBorder(region_mask, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
    dist = cv.distanceTransform(padded, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    inside = dist[padded > 0]
    dt_width = 2.0 * float(np.median(inside)) if inside.size else float("nan")
    if perimeter <= 0 or area <= 0:
        return {"thickness": float("nan"), "dt_width": dt_width, "circ_inv": float("nan")}
    return {"thickness": float(2.0 * area / perimeter),
            "dt_width": dt_width,
            "circ_inv": float(perimeter ** 2 / (4.0 * np.pi * area))}


def describe_regions(mask, min_pixels=30, gray=None):
    """이진 마스크의 연결 영역마다 모양 지표를 재서 후보 dict list로 돌려준다.
    elong = 긴 변 / 짧은 변, fill = 윤곽 넓이 / 회전 사각형 넓이, solidity = 윤곽 넓이 / 볼록 껍질 넓이.
    gray를 주면 darkness(주변보다 어두운 정도)도 더한다."""
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    regions = []

    for label in range(1, num_labels):
        pixel_count = stats[label, cv.CC_STAT_AREA]

        if pixel_count < min_pixels:
            continue

        component_mask = np.zeros_like(mask)
        component_mask[labels == label] = 255

        contours, _ = cv.findContours(
            component_mask,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        contour = max(contours, key=cv.contourArea)

        x, y, w, h = cv.boundingRect(contour)

        rect = cv.minAreaRect(contour)
        rw, rh = rect[1]

        short_side = min(rw, rh)
        long_side = max(rw, rh)

        if short_side <= 0:
            continue

        elong = long_side / short_side
        contour_area = cv.contourArea(contour)

        rect_area = rw * rh
        fill = contour_area / rect_area if rect_area > 0 else 0.0

        hull = cv.convexHull(contour)
        hull_area = cv.contourArea(hull)

        solidity = (
            contour_area / hull_area
            if hull_area > 0
            else 0.0
        )

        region = {
            "area": int(pixel_count),
            "elong": float(elong),
            "fill": float(fill),
            "solidity": float(solidity),
            "bbox": (
                int(x),
                int(y),
                int(w),
                int(h),
            ),
            "contour": contour.astype(np.int32),
        }

        sx, sy, sw, sh = stats[label, :4]              # 영역의 bbox만 잘라 계산을 빠르게
        region.update(shape_metrics(component_mask[sy:sy + sh, sx:sx + sw], contour, int(pixel_count)))
        region["contrast"] = float("nan")
        region["score"] = float("nan")

        if gray is not None:
            region["darkness"] = region_darkness(gray, contour)

        regions.append(region)

    return regions


def _detect_mode():
    """config의 DETECT_MODE("legacy" 또는 "dark")를 읽는다. 다른 값이면 ValueError."""
    if C.DETECT_MODE not in ("legacy", "dark"):
        raise ValueError(f"DETECT_MODE는 'legacy' 또는 'dark'여야 함 ({C.DETECT_MODE!r})")
    return C.DETECT_MODE


def extract_dark_regions(gray, road=None):
    """배경보다 어두운 영역을 뽑아 균열·포트홀 공통 후보로 만든다. (영역 list, 어두운 영역 마스크)를 돌려준다.
    1. 흰 차선 같은 밝은 표시를 배경값으로 메운다.
    2. 중앙값 필터로 배경을 추정하고, 배경 − 영상 = 어두운 정도를 구한다.
    3. 잡음 σ(MAD)에 비례하는 이중 문턱으로 어두운 영역을 고른다.
    4. 닫힘으로 조각을 잇고 작은 영역을 버린 뒤, 영역마다 형태 지표와 대비를 잰다."""
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("gray는 2차원 uint8 배열이어야 합니다.")
    if gray.size == 0:
        raise ValueError("gray는 빈 영상일 수 없습니다.")
    if road is not None and np.shape(road) != gray.shape:
        raise ValueError("road와 gray의 크기가 같아야 합니다.")
    if C.DARK_BG_DOWNSCALE < 1 or C.DARK_SIGMA_MIN <= 0:
        raise ValueError("축소 배율은 1 이상, σ 하한은 양수여야 합니다.")
    if not (0 < C.DARK_K_LO <= C.DARK_K_HI):
        raise ValueError("0 < DARK_K_LO <= DARK_K_HI여야 합니다.")
    if C.DARK_ABS_MIN <= 0 or C.DARK_MIN_AREA < 1:
        raise ValueError("절대 문턱은 양수, 최소 넓이는 1 이상이어야 합니다.")
    
    h, w = gray.shape
    scale = C.DARK_BG_DOWNSCALE
    small_size = (max(1, int(round(w / scale))), max(1, int(round(h / scale))))

    def median_background(image, small_window):
        """축소 영상에서 중앙값 배경을 구해 큰 창의 계산량을 줄인다."""
        ksize = max(1, int(round(small_window)))
        if ksize % 2 == 0:
            ksize += 1
        small = cv.resize(image, small_size, interpolation=cv.INTER_AREA)
        filtered = cv.medianBlur(small, ksize) if ksize > 1 else small
        return cv.resize(filtered, (w, h), interpolation=cv.INTER_LINEAR).astype(np.float32)


    # 1. 밝은 표시 억제
    coarse = median_background(gray, C.DARK_BG_KSIZE_SMALL)
    gray_float = gray.astype(np.float32)
    bright = gray_float > coarse + C.DARK_BRIGHT_DELTA
    g2 = gray.copy()
    g2[bright] = coarse[bright].astype(np.uint8)
    # 2. 배경 = 작은 창·큰 창 중앙값 중 큰 값 (큰 포트홀 가운데가 비지 않게)
    bg_small = median_background(g2, h * C.DARK_BG_FRAC / scale)
    bg_large = median_background(g2, h * C.DARK_BG_FRAC_LARGE / scale)
    residual = np.maximum(bg_small, bg_large) - g2.astype(np.float32)

    # 3. 잡음 σ 추정과 이중 문턱 (약 문턱 영역 중 강 픽셀을 품은 것만 남김)
    samples = residual.ravel()
    if road is not None and C.DARK_SIGMA_IN_ROAD:
        road_pixels = np.asarray(road) > 0
        if np.any(road_pixels):
            samples = residual[road_pixels]
    median = np.median(samples)
    sigma = max(1.4826 * float(np.median(np.abs(samples - median))), C.DARK_SIGMA_MIN)
    darkness = np.maximum(residual, 0)
    strong = darkness >= max(C.DARK_K_HI * sigma, C.DARK_ABS_MIN)
    weak = (darkness >= C.DARK_K_LO * sigma).astype(np.uint8)
    count, weak_labels = cv.connectedComponents(weak, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[np.unique(weak_labels[strong])] = True
    keep[0] = False  # 배경 라벨 제외
    mask = (keep[weak_labels].astype(np.uint8) * 255)

    # 4. 닫힘 → 작은 영역 제거 → 영역별 지표 (둘레는 구멍 윤곽까지 포함)
    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (C.DARK_CLOSE_KSIZE, C.DARK_CLOSE_KSIZE)
    )
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv.CC_STAT_AREA] >= C.DARK_MIN_AREA
    keep[0] = False
    mask = keep[labels].astype(np.uint8) * 255
    regions = describe_regions(mask, C.DARK_MIN_AREA, gray)

    for region in regions:
        x, y, rw, rh = region["bbox"]
        px, py = region["contour"][0, 0]
        label = labels[py, px]
        component = labels[y:y + rh, x:x + rw] == label
        local_mask = component.astype(np.uint8) * 255
        contours, _ = cv.findContours(local_mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv.arcLength(contour, True) for contour in contours)
        area = region["area"]
        if perimeter > 0:
            region["thickness"] = float(2.0 * area / perimeter)
            region["circ_inv"] = float(perimeter ** 2 / (4.0 * np.pi * area))
        region["contrast"] = float(darkness[y:y + rh, x:x + rw][component].mean() / sigma)
        # score는 쓰지 않아 NaN으로 둠
    return regions, mask

def detect_cracks(gray, road=None):
    """가늘고 긴 어두운 선을 균열 후보로 찾는다. (후보 list, 캐니 에지 0/255)를 돌려준다.
    dark: 어두운 영역 중 비원형도가 큰 것. legacy: 캐니 → 닫힘 → 길쭉하거나 채움이 낮고 주변보다 어두운 것."""
    if _detect_mode() == "dark":
        regions, _ = extract_dark_regions(gray, road)
        
        candidates = [r for r in regions if r["circ_inv"] >= C.CRACK_MIN_CIRC_INV]
        edges = cv.Canny(gray, C.CANNY_LOW, C.CANNY_HIGH)
        return candidates, edges
    
    edges=cv.Canny(
        gray, 
        C.CANNY_LOW,
        C.CANNY_HIGH,
    )
    
    kernel = cv.getStructuringElement(
        cv.MORPH_RECT,
        (C.CRACK_CLOSE_KSIZE, C.CRACK_CLOSE_KSIZE),
    )
    closed = cv.morphologyEx(
        edges,
        cv.MORPH_CLOSE,
        kernel,
    )

    regions = describe_regions(
        closed,
        min_pixels=C.CRACK_MIN_AREA,
        gray=gray,
    )

    candidates = [
        region
        for region in regions
        if region["area"] >= C.CRACK_MIN_AREA
        and (
            region["elong"] >= C.CRACK_MIN_ELONG
            or region["fill"] <= C.CRACK_MAX_FILL
        )
        and region["darkness"] >= C.CRACK_MIN_DARKNESS
    ]

    return candidates, edges


def detect_potholes(gray, road=None):
    """주변보다 어두운 둥근 덩어리를 포트홀 후보로 찾는다. (후보 list, 이진 마스크 0/255)를 돌려준다.
    dark: 어두운 영역 중 둥글고 꽉 찬 것. legacy: DoG 문턱 → 열림 → 크기·모양 조건."""
    if _detect_mode() == "dark":

        regions, mask = extract_dark_regions(gray, road)
        
        candidates = [
            r for r in regions
            if r["circ_inv"] <= C.POT_MAX_CIRC_INV
            and r["solidity"] >= C.POT_MIN_SOLIDITY
        ]
        return candidates, mask
    
    sigma_small, sigma_large = C.DOG_SIGMAS

    blur_small = cv.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=sigma_small,
    )

    blur_large = cv.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=sigma_large,
    )

    dog = cv.subtract(blur_large, blur_small)

    _, mask = cv.threshold(
        dog,
        C.DOG_THRESH,
        255,
        cv.THRESH_BINARY,
    )

    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (C.POT_OPEN_KSIZE, C.POT_OPEN_KSIZE),
    )

    opened = cv.morphologyEx(
        mask,
        cv.MORPH_OPEN,
        kernel,
    )

    regions = describe_regions(
        opened,
        min_pixels=C.POT_MIN_AREA,
        gray=gray,                  # 균열과 같은 형식을 위해 darkness도 붙임
    )

    roi_area = gray.shape[0] * gray.shape[1]
    max_area = roi_area * C.POT_MAX_AREA_RATIO

    candidates = [
        region
        for region in regions
        if region["area"] >= C.POT_MIN_AREA
        and region["area"] <= max_area
        and region["elong"] <= C.POT_MAX_ELONG
        and region["solidity"] >= C.POT_MIN_SOLIDITY
    ]

    return candidates, opened
    
