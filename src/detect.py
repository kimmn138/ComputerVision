"""전처리한 ROI 영상에서 균열·포트홀 후보 영역을 찾는다."""
import warnings

import cv2 as cv
import numpy as np

from src import config as C


def region_darkness(gray, contour):
    """후보가 주변보다 얼마나 어두운지 잰다: 후보 둘레 고리의 어두운 쪽 밝기 − 후보 안 평균 밝기.
    균열은 양수, 흰 선·횡단보도처럼 주변보다 밝은 선은 음수가 된다."""
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
    """후보 하나의 두께·거리 변환 폭·비원형도를 재서, 균열(가늘고 긺)과 포트홀(둥긂)을 모양으로 가를 수 있게 한다.
    thickness = 2·넓이/둘레, dt_width = 거리 변환 값 중앙값의 2배, circ_inv(비원형도) = 둘레² / (4π·넓이)(원 ≈ 1, 길수록 큼)."""
    perimeter = cv.arcLength(contour, True)
    # 둘레 1px을 배경으로 둘러, 잘라 낸 조각에서도 영역 밖까지의 거리를 바르게 잰다
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
    """이진 마스크의 연결 영역마다 모양 지표를 계산해 후보 dict로 만든다.
    area(픽셀 수)·elong(긴 변 ÷ 짧은 변)·fill(윤곽 넓이 ÷ 회전 사각형 넓이)·solidity(윤곽 넓이 ÷ 볼록 껍질 넓이)·bbox·contour와
    shape_metrics 값을 담는다. gray를 주면 darkness도 더한다. contrast·score는 NaN으로 두고 dark 방식에서 채운다."""
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

        sx, sy, sw, sh = stats[label, :4]              # 이 영역을 감싸는 bbox만 잘라 거리 변환을 빠르게
        region.update(shape_metrics(component_mask[sy:sy + sh, sx:sx + sw], contour, int(pixel_count)))
        region["contrast"] = float("nan")
        region["score"] = float("nan")

        if gray is not None:
            region["darkness"] = region_darkness(gray, contour)

        regions.append(region)

    return regions


def region_scores(region, perimeter):
    """후보 하나의 균열용·포트홀용 점수를 SCORE_FORMULA 식으로 계산해 (균열 점수, 포트홀 점수)를 돌려준다.
    점수가 낮은 후보를 임계값으로 걸러 오검출을 줄이는 데 쓴다. 종류가 정해지기 전이라 두 값을 다 낸다.
    "margin" = 대비 × 분류 경계에서의 여유, "snr" = 대비 × √넓이, "shape" = 대비 × 길이·넓이 항, "none" = NaN."""
    formula = C.SCORE_FORMULA
    if formula == "none":
        return float("nan"), float("nan")
    c, area, ci, solidity = region["contrast"], region["area"], region["circ_inv"], region["solidity"]
    if formula == "margin":
        solid = min(1.0, (solidity - C.POT_MIN_SOLIDITY) / (1.0 - C.POT_MIN_SOLIDITY))
        crack = c * (1.0 - np.exp(-(ci - C.CRACK_MIN_CIRC_INV) / C.SCORE_MARGIN_SCALE))
        pothole = c * (1.0 - np.exp(-(C.POT_MAX_CIRC_INV - ci) / C.SCORE_MARGIN_SCALE_POT)) * solid
    elif formula == "snr":
        crack = pothole = c * np.sqrt(area)
    elif formula == "shape":
        capped = min(c, C.SCORE_C_CAP)
        crack = capped * min(1.0, (perimeter / 2.0) / C.SCORE_LEN_REF)
        pothole = capped * solidity * min(1.0, area / C.SCORE_AREA_REF)
    else:
        raise ValueError(f"SCORE_FORMULA는 'none'·'margin'·'snr'·'shape' 중 하나여야 함 ({formula!r})")
    return float(max(crack, 0.0)), float(max(pothole, 0.0))


def _keep_by_score(candidates, kind):
    """종류(crack·pothole)의 점수를 score에 넣고, USE_SCORE_TAU가 켜져 있으면 점수 임계값 이상인 후보만 남긴다."""
    for r in candidates:
        r["score"] = r[f"score_{kind}"]
    if not C.USE_SCORE_TAU:
        return candidates
    if C.SCORE_FORMULA == "none":
        raise ValueError("USE_SCORE_TAU를 켜려면 SCORE_FORMULA가 'none'이 아니어야 함")
    tau = C.SCORE_TAU_CRACK if kind == "crack" else C.SCORE_TAU_POTHOLE
    return [r for r in candidates if r["score"] >= tau]


def _detect_mode():
    """검출 방식(DETECT_MODE)을 읽는다: "dark" = 어두운 영역을 뽑아 모양으로 분류, "legacy" = Canny 균열 + DoG 포트홀."""
    if C.DETECT_MODE not in ("legacy", "dark"):
        raise ValueError(f"DETECT_MODE는 'legacy' 또는 'dark'여야 함 ({C.DETECT_MODE!r})")
    return C.DETECT_MODE


def local_sigma_map(residual, median, sigma_global, step):
    """지역 잡음 추정: 위치마다 주변 창에서 잰 잡음 σ 지도. 질감이 거친 곳은 σ가 커져 임계값이 저절로 오른다.
    큰 포트홀 같은 어두운 손상이 σ를 부풀리지 않게 배경보다 밝은 쪽 잔차만 쓰고, 표본이 적은 창은 전역 σ를 쓴다."""
    h, w = residual.shape
    sub = residual[::step, ::step]
    bright = np.where(sub < median, median - sub, np.nan).astype(np.float32)
    k = max(3, int(round(h * C.DARK_SIGMA_LOCAL_FRAC / step))) | 1
    stride = max(1, k // 2)
    padded = np.pad(bright, k // 2, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k))[::stride, ::stride]
    flat = windows.reshape(windows.shape[0], windows.shape[1], -1)
    count = np.count_nonzero(~np.isnan(flat), axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # 표본이 하나도 없는 창(모두 NaN)은 아래에서 전역 σ로 바꾼다
        local = 1.4826 * np.nanmedian(flat, axis=2)
    local = np.where(count >= C.DARK_SIGMA_LOCAL_MIN_N, local, sigma_global).astype(np.float32)
    return cv.resize(local, (w, h), interpolation=cv.INTER_LINEAR)


def extract_dark_regions(gray, road=None):
    """배경보다 어두운 영역을 한 번에 뽑아, 균열·포트홀 후보를 같은 기준(배경 대비 ÷ 잡음 σ)으로 만든다.
    출력: (영역 list, 어두운 영역 마스크 uint8 0/255). road(노면 마스크)를 주면 잡음 σ를 노면 안에서 잰다.
    순서: 밝은 표시(흰 차선 등) 메우기 → 큰 창 중앙값으로 배경 밝기 → 배경보다 어두운 정도(잔차)와 잡음 σ
    → 이중 임계값(약한 영역 가운데 강한 픽셀을 품은 것만) → 닫힘·작은 영역 제거 → 영역마다 모양·대비·점수."""
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


    # 1. 밝은 표시 메우기: 기준 밝기(거친 배경)보다 크게 밝은 픽셀(흰 차선·횡단보도)을 기준 밝기로 바꿈
    coarse = median_background(gray, C.DARK_BG_KSIZE_SMALL)
    gray_float = gray.astype(np.float32)
    if C.DARK_BRIGHT_REF == "coarse":
        reference = coarse
    elif C.DARK_BRIGHT_REF == "road_median":
        # 기준 밝기를 노면 밝기 중앙값 하나로(넓은 흰 줄무늬는 거친 배경 자체가 밝아져 못 거르므로)
        road_pixels = np.asarray(road) > 0 if road is not None else None
        pixels = gray_float[road_pixels] if road_pixels is not None and np.any(road_pixels) else gray_float
        reference = np.full_like(gray_float, float(np.median(pixels)))
    else:
        raise ValueError(f"DARK_BRIGHT_REF는 'coarse' 또는 'road_median'이어야 함 ({C.DARK_BRIGHT_REF!r})")
    bright = gray_float > reference + C.DARK_BRIGHT_DELTA
    if C.DARK_BRIGHT_DILATE > 0:
        # 밝은 표시 주변 보정: 아주 밝은 픽셀만 넓혀, 샤프닝이 밝은 선 옆에 만든 어두운 테두리까지 메움
        smooth = cv.medianBlur(gray, C.DARK_BRIGHT_SEED_KSIZE).astype(np.float32)   # 샤프닝이 키운 작은 점은 빼고 고름
        seed = (smooth > reference + C.DARK_BRIGHT_DILATE_DELTA).astype(np.uint8)
        size = 2 * C.DARK_BRIGHT_DILATE + 1
        bright = bright | (cv.dilate(seed, cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))) > 0)
    g2 = gray.copy()
    g2[bright] = reference[bright].astype(np.uint8)
    # 2. 배경 밝기 = 두 크기 창 중앙값의 큰 값(창 하나면 큰 포트홀 가운데가 빔), 잔차 = 배경보다 어두운 정도
    bg_small = median_background(g2, h * C.DARK_BG_FRAC / scale)
    bg_large = median_background(g2, h * C.DARK_BG_FRAC_LARGE / scale)
    residual = np.maximum(bg_small, bg_large) - g2.astype(np.float32)

    # 3. 잡음 σ = 잔차의 MAD × 1.4826(road가 있으면 노면 안에서). DARK_SIGMA_LOCAL이면 위치마다 잼
    samples = residual.ravel()
    if road is not None and C.DARK_SIGMA_IN_ROAD:
        road_pixels = np.asarray(road) > 0
        if np.any(road_pixels):
            samples = residual[road_pixels]
    median = np.median(samples)
    sigma = max(1.4826 * float(np.median(np.abs(samples - median))), C.DARK_SIGMA_MIN)
    if C.DARK_SIGMA_LOCAL:
        sigma = np.maximum(local_sigma_map(residual, median, sigma, scale), sigma)
    # 4. 이중 임계값: 약한 임계값을 넘는 영역 가운데 강한 임계값을 넘는 픽셀을 품은 것만 남김
    darkness = np.maximum(residual, 0)
    strong = darkness >= (np.maximum(C.DARK_K_HI * sigma, C.DARK_ABS_MIN) if np.ndim(sigma) else max(C.DARK_K_HI * sigma, C.DARK_ABS_MIN))
    weak = (darkness >= C.DARK_K_LO * sigma).astype(np.uint8)
    count, weak_labels = cv.connectedComponents(weak, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[np.unique(weak_labels[strong])] = True
    keep[0] = False  # 배경 라벨은 후보에서 제외한다
    mask = (keep[weak_labels].astype(np.uint8) * 255)

    # 5. 닫힘으로 끊긴 조각을 잇고, 아주 작은 영역(골재 무늬)은 지움
    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (C.DARK_CLOSE_KSIZE, C.DARK_CLOSE_KSIZE)
    )
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    if C.DARK_SPLIT_BLOB:
        # 덩어리와 선 나누기: 원판 열림으로 굵은 덩어리(포트홀)만 떼어, 균열이 붙은 포트홀을 두 영역으로 가름
        disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (C.DARK_SPLIT_KSIZE, C.DARK_SPLIT_KSIZE))
        blob = cv.morphologyEx(mask, cv.MORPH_OPEN, disc)
        if blob.any():
            rest = cv.bitwise_and(mask, cv.bitwise_not(cv.dilate(blob, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))))
            mask = cv.bitwise_or(blob, rest)
    count, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    min_area = np.full(count, float(C.DARK_MIN_AREA))
    if C.DARK_AREA_PERSPECTIVE:
        # 원근 보정: ROI 위쪽(먼 곳)일수록 최소 넓이를 줄여 먼 곳의 작은 균열 조각을 남김
        cy = (stats[:, cv.CC_STAT_TOP] + stats[:, cv.CC_STAT_HEIGHT] / 2.0) / max(h - 1, 1)
        min_area = C.DARK_MIN_AREA_TOP + (C.DARK_MIN_AREA - C.DARK_MIN_AREA_TOP) * np.clip(cy, 0.0, 1.0)
    keep = stats[:, cv.CC_STAT_AREA] >= min_area
    keep[0] = False
    mask = keep[labels].astype(np.uint8) * 255
    regions = describe_regions(mask, min(C.DARK_MIN_AREA, C.DARK_MIN_AREA_TOP) if C.DARK_AREA_PERSPECTIVE else C.DARK_MIN_AREA, gray)
    if C.SCORE_USE_HESSIAN:                             # 헤시안 선 강조 지도: ROI 전체에서 한 번 계산
        ridge_num, ridge_den = _ridge_map(gray)
    if C.SCORE_USE_EDGE:                                # 경계 날카로움을 잴 기울기 크기(밝은 표시를 메운 영상 기준)
        g2f = g2.astype(np.float32)
        gradient = np.hypot(cv.Sobel(g2f, cv.CV_32F, 1, 0, ksize=3), cv.Sobel(g2f, cv.CV_32F, 0, 1, ksize=3)) / 8.0

    # 6. 영역마다 둘레(구멍 둘레 포함)·대비·점수를 잼
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
        if np.ndim(sigma):                              # 지역 잡음 추정: 픽셀마다 그 자리 σ로 나눈 평균
            region["contrast"] = float((darkness[y:y + rh, x:x + rw][component] / sigma[y:y + rh, x:x + rw][component]).mean())
        else:
            region["contrast"] = float(darkness[y:y + rh, x:x + rw][component].mean() / sigma)
        # 점수는 종류가 정해진 뒤 detect_cracks·detect_potholes가 score에 넣음
        region["score_crack"], region["score_pothole"] = region_scores(region, perimeter)
        if C.SCORE_USE_EDGE:
            # 경계 날카로움(바깥 고리 기울기 ÷ 안쪽 어둡기)으로 점수를 깎음: 그림자는 가장자리가 흐려 값이 작음
            x0, y0, x1, y1 = max(x - 1, 0), max(y - 1, 0), min(x + rw + 1, w), min(y + rh + 1, h)
            inside = labels[y0:y1, x0:x1] == label
            ring = cv.dilate(inside.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~inside
            depth = float(darkness[y:y + rh, x:x + rw][component].mean())
            region["edge"] = float(gradient[y0:y1, x0:x1][ring].mean() / depth) if ring.any() and depth > 0 else 0.0
            term = min(1.0, region["edge"] / C.SCORE_EDGE_REF)
            region["score_crack"], region["score_pothole"] = region["score_crack"] * term, region["score_pothole"] * term
        if C.SCORE_USE_HESSIAN:
            # 영역 안 선 강조(가는 어두운 선이면 1에 가까움)를 균열 점수에만 곱함
            den = float(ridge_den[y:y + rh, x:x + rw][component].sum())
            region["ridge"] = float(ridge_num[y:y + rh, x:x + rw][component].sum()) / den if den > 0 else 0.0
            region["score_crack"] = region["score_crack"] * region["ridge"]
    return regions, mask


def _ridge_map(gray):
    """헤시안(밝기의 2차 미분) 고윳값으로 픽셀마다 '어두운 선' 모양을 잰다(Frangi 필터의 단순형).
    선 강조 = 분자 합 ÷ 분모 합으로 쓰도록 (분자, 분모) float32 두 장을 돌려준다."""
    g = cv.GaussianBlur(gray.astype(np.float32), (0, 0), C.HESSIAN_SIGMA)
    ixx = cv.Sobel(g, cv.CV_32F, 2, 0, ksize=3)
    iyy = cv.Sobel(g, cv.CV_32F, 0, 2, ksize=3)
    ixy = cv.Sobel(g, cv.CV_32F, 1, 1, ksize=3)
    root = np.sqrt((ixx - iyy) ** 2 + 4.0 * ixy ** 2)
    l1, l2 = (ixx + iyy + root) / 2.0, (ixx + iyy - root) / 2.0
    num = np.where(l1 > 0, np.maximum(l1 - np.abs(l2), 0.0), 0.0).astype(np.float32)
    return num, (np.abs(l1) + np.abs(l2)).astype(np.float32)


def _bright_gravel_regions(gray, road, existing):
    """자갈이 차 있어 어둡지 않은 포트홀 후보를 질감으로 찾는다: 질감이 노면보다 훨씬 거칠고 포트홀 모양인 덩어리.
    이미 있는 후보와 겹치지 않는 것만 돌려주고, contrast 자리에는 질감 비율(영역 ÷ 노면)을 넣는다."""
    g = gray.astype(np.float32)
    k = (C.BRIGHT_POT_TEX_KSIZE, C.BRIGHT_POT_TEX_KSIZE)
    mean = cv.boxFilter(g, -1, k, borderType=cv.BORDER_REFLECT)
    std = np.sqrt(np.maximum(cv.boxFilter(g * g, -1, k, borderType=cv.BORDER_REFLECT) - mean * mean, 0.0))
    on_road = np.asarray(road) > 0 if road is not None else np.ones(gray.shape, bool)
    if not on_road.any():
        return []
    ref = float(np.median(std[on_road]))
    if ref <= 0:
        return []
    mask = ((std > C.BRIGHT_POT_TEX_RATIO * ref) & on_road).astype(np.uint8) * 255
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (C.POT_OPEN_KSIZE, C.POT_OPEN_KSIZE)))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (C.DARK_CLOSE_KSIZE, C.DARK_CLOSE_KSIZE)))

    def inside(box, region):
        x, y, w, h = box
        rx, ry, rw, rh = region["bbox"]
        cx, cy = rx + rw / 2, ry + rh / 2
        return x <= cx <= x + w and y <= cy <= y + h

    out = []
    for r in describe_regions(mask, C.POT_MIN_AREA, gray):
        if not (r["circ_inv"] <= C.POT_MAX_CIRC_INV and r["solidity"] >= C.POT_MIN_SOLIDITY):
            continue
        if any(inside(e["bbox"], r) for e in existing):
            continue
        x, y, w, h = r["bbox"]
        fill = np.zeros((h, w), np.uint8)
        cv.drawContours(fill, [r["contour"]], -1, 255, -1, offset=(-x, -y))
        r["contrast"] = float(std[y:y + h, x:x + w][fill > 0].mean() / ref)
        r["score_crack"], r["score_pothole"] = region_scores(r, cv.arcLength(r["contour"], True))
        r["bright_gravel"] = True
        out.append(r)
    return out

def detect_cracks(gray, road=None):
    """가늘고 긴 어두운 선을 균열 후보로 찾는다. (후보 list, Canny 에지 uint8 0/255)를 돌려준다.
    "dark" 방식: 어두운 영역 가운데 비원형도가 CRACK_MIN_CIRC_INV 이상인 것(점수 임계값 적용).
    "legacy" 방식: Canny 에지를 닫힘으로 이어, 길쭉하거나 채움이 낮고 주변보다 어두운 영역. 노면 밖 후보는 pipeline이 지운다."""
    if _detect_mode() == "dark":
        regions, _ = extract_dark_regions(gray, road)
        
        candidates = _keep_by_score([r for r in regions if r["circ_inv"] >= C.CRACK_MIN_CIRC_INV], "crack")
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
    """주변보다 어두운 둥근 덩어리를 포트홀 후보로 찾는다. (후보 list, 이진 마스크 uint8 0/255)를 돌려준다.
    "dark" 방식: 비원형도가 낮고 속이 찬 어두운 영역(점수 임계값 적용). "legacy" 방식: DoG로 찾은 어두운 덩어리."""
    if _detect_mode() == "dark":

        regions, mask = extract_dark_regions(gray, road)
        
        shaped = [
            r for r in regions
            if r["circ_inv"] <= C.POT_MAX_CIRC_INV
            and r["solidity"] >= C.POT_MIN_SOLIDITY
        ]
        if C.BRIGHT_POTHOLE:                            # 어둡지 않은 자갈 포트홀을 질감으로 더함
            shaped = shaped + _bright_gravel_regions(gray, road, shaped)
        candidates = _keep_by_score(shaped, "pothole")
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
        gray=gray,                  # 균열 후보와 같은 형식이 되게 darkness도 붙임 (거르는 데는 쓰지 않음)
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
    
