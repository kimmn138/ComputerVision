"""(파트 B) 전처리한 노면 영상에서 균열·포트홀 후보 영역을 찾는다."""
import warnings

import cv2 as cv
import numpy as np

from src import config as C


def region_darkness(gray, contour):
    """후보가 주변보다 얼마나 어두운지 잰다: 후보를 DARK_RING_PX만큼 넓힌 띠에서 어두운 쪽 밝기
    (하위 DARK_RING_QUANTILE) − 후보 윤곽 안의 평균. 균열은 양쪽 모두보다 어두운 선이라 양수,
    흰 선·횡단보도 점선은 음수가 된다. 띠의 평균을 쓰면 넓은 흰 차선의 가장자리(한쪽만 밝음)가
    양수로 나와 걸러지지 않으므로 어두운 쪽과 비교한다. 영상 전체가 아니라 후보 주변만 잘라 계산한다."""
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
    """후보 하나의 두께·거리 변환 폭·비원형도를 재서, 균열(가늘고 긺)과 포트홀(둥긂)을 형태로 가를 수 있게 한다(Plan.md 3.5 (1)).
    region_mask: 그 영역만 255인 uint8 마스크(영역을 감싸는 bbox로 잘라도 됨), contour: 바깥 윤곽, area: 픽셀 수.
    P = 바깥 윤곽 길이(cv.arcLength). thickness = 2·area/P(가는 선이면 선 폭, 원이면 반지름에 가까움),
    circ_inv = P² / (4π·area)(원 ≈ 1, 길거나 갈래가 많을수록 큼), dt_width = 2 × 영역 안 거리 변환 값의 중앙값.
    넓이는 윤곽 넓이가 아니라 픽셀 수로 센다(1px 선의 윤곽 넓이는 0). P가 0이면 thickness·circ_inv는 NaN.
    구멍 둘레를 P에 더하는 계산(cv.RETR_CCOMP, 거북등 균열)은 B가 WP2에서 한다. 지금은 바깥 윤곽만 쓴다."""
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
    area = 픽셀 수, elong = 회전 사각형의 긴 변 / 짧은 변, fill = 윤곽 넓이 / 회전 사각형 넓이,
    solidity = 윤곽 넓이 / 볼록 껍질 넓이. gray를 주면 darkness(주변보다 어두운 정도, region_darkness)도 더한다.
    thickness·dt_width·circ_inv는 shape_metrics로 잰다. contrast(배경 대비 / 잡음 σ)와 score(대비 × 형태 점수)는
    dark 방식의 값이라 여기서는 NaN으로 둔다(B가 WP2·WP3에서 채움)."""
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
    """후보 하나의 균열용·포트홀용 점수를 config B 구역의 SCORE_FORMULA로 계산해, 점수 문턱 τ로 오검출을 줄일 동작점을 고를 수 있게 한다
    (보완 1번, docs/tasks/supplement.md 2.3). 종류는 아직 정해지지 않았으므로 두 값을 다 돌려준다 → (균열 점수, 포트홀 점수).
    c = contrast(영역 안 어둡기 평균 ÷ 잡음 σ), A = 픽셀 수, ci = 비원형도, s = solidity, L = 둘레 ÷ 2(가는 선이면 길이).
    "none": 둘 다 NaN(점수 없음, 지금과 같음).
    "margin": 대비 × 분류 문턱에서의 여유. 균열 c·(1 − e^(−(ci − CRACK_MIN_CIRC_INV)/k)),
              포트홀 c·(1 − e^(−(POT_MAX_CIRC_INV − ci)/k'))·min(1, (s − POT_MIN_SOLIDITY)/(1 − POT_MIN_SOLIDITY)),
              k = SCORE_MARGIN_SCALE(균열), k' = SCORE_MARGIN_SCALE_POT(포트홀). 두 종류의 거리 범위가 달라 따로 둔다.
    "snr": c·√A(영역 평균 어둡기가 잡음 평균의 표준오차 σ/√A의 몇 배인지). 두 종류 같음.
    "shape": 균열 min(c, SCORE_C_CAP)·min(1, L / SCORE_LEN_REF), 포트홀 min(c, SCORE_C_CAP)·s·min(1, A / SCORE_AREA_REF).
    여유가 음수(그 종류의 문턱 밖)면 0으로 둔다. 그 밖의 식 이름은 ValueError."""
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
    """종류(crack·pothole)의 점수를 후보의 score에 넣고, USE_SCORE_TAU가 켜져 있으면 score ≥ SCORE_TAU_<종류>인 후보만 남긴다.
    dark 방식에서만 부른다. 점수식이 none인데 τ를 켜면 점수가 NaN이라 후보가 모두 지워지므로 ValueError로 알린다."""
    for r in candidates:
        r["score"] = r[f"score_{kind}"]
    if not C.USE_SCORE_TAU:
        return candidates
    if C.SCORE_FORMULA == "none":
        raise ValueError("USE_SCORE_TAU를 켜려면 SCORE_FORMULA가 'none'이 아니어야 함")
    tau = C.SCORE_TAU_CRACK if kind == "crack" else C.SCORE_TAU_POTHOLE
    return [r for r in candidates if r["score"] >= tau]


def _detect_mode():
    """config B 구역의 DETECT_MODE를 읽는다. "legacy" = 1차 방식(Canny 균열 + DoG 포트홀, road를 쓰지 않음),
    "dark" = 어두운 영역 한 번 추출 + 형태 분류(WP2·WP3). 다른 값이면 ValueError로 바로 알린다."""
    if C.DETECT_MODE not in ("legacy", "dark"):
        raise ValueError(f"DETECT_MODE는 'legacy' 또는 'dark'여야 함 ({C.DETECT_MODE!r})")
    return C.DETECT_MODE


def local_sigma_map(residual, median, sigma_global, step):
    """(보완 2-1, 12번) 국소 σ 지도: 위치마다 주변 창에서 잰 잡음 크기. 골재 결이 거친 곳은 σ가 커져 문턱이 저절로 오른다.
    배경보다 밝은 쪽 잔차(잔차 < 전역 중앙값)만 쓴다. 어두운 손상(큰 포트홀)이 창을 채워도 σ가 부풀지 않게 하려고다(안 A, 10/13).
    노면 잔차는 중앙값 둘레에서 대칭이라(개발셋 51장: 밝은 쪽 σ ÷ 양쪽 σ 중앙값 1.000) 양쪽과 같은 배율 1.4826을 쓴다.
    잔차를 step 간격으로 뽑아(평균하지 않음) 창(띠 높이 × DARK_SIGMA_LOCAL_FRAC)을 창 절반 간격으로 옮기며 밝은 쪽 편차 (중앙값 − 잔차)의
    중앙값 × 1.4826을 잰다. 창 안 밝은 쪽 표본이 DARK_SIGMA_LOCAL_MIN_N개보다 적으면 전역 σ를 쓴다. 원래 크기로 늘려 돌려준다(하한은 부르는 쪽)."""
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
    """배경보다 어두운 영역을 한 번에 뽑아, 균열·포트홀 후보를 같은 기준(배경 대비 / 잡음 σ)으로 만든다(Plan.md 3.4, B 전용 · 약속 표 밖).
    입력: 흑백 노면 띠 uint8 (H, W), road = 노면 마스크(같은 크기 0/255) 또는 None. 입력은 바꾸지 않는다.
    출력: (영역 list, 어두운 영역 마스크 uint8 0/255). 영역 dict = 후보 형식(area·elong·fill·solidity·bbox·contour)
    + thickness·dt_width·circ_inv(구멍 둘레 포함)·contrast·darkness. 숫자는 config B 구역의 DARK_* 값을 쓴다.
    1. 밝은 표시 억제: 1/DARK_BG_DOWNSCALE로 줄여 중앙값(DARK_BG_KSIZE_SMALL)으로 거친 배경 Bc를 만들고,
       gray > Bc + DARK_BRIGHT_DELTA인 픽셀(흰 차선·횡단보도)을 Bc 값으로 바꾼 복사본 g2를 만든다.
       (보완 2-2) DARK_BRIGHT_REF "road_median"이면 Bc 대신 노면 밝기 중앙값 하나를 기준·메움 값으로 쓰고(9a 넓은 줄무늬),
       DARK_BRIGHT_DILATE > 0이면 밝은 픽셀 마스크를 그만큼 팽창해 밝은 선 옆 어두운 테두리도 메운다(13번).
       한계: 억제 창보다 넓은 밝은 영역 옆은 '어두운 선'이 된다(Japan_000107 자갈 경계).
    2. 배경 B = 두 크기 창 중앙값의 큰 값: max(bg(g2, 띠 높이 × DARK_BG_FRAC), bg(g2, 띠 높이 × DARK_BG_FRAC_LARGE)).
       bg(영상, 창) = 1/DARK_BG_DOWNSCALE로 줄여 중앙값(창 / DARK_BG_DOWNSCALE을 홀수로) → 원래 크기. 한 창이면 큰 포트홀 가운데가 빈다.
    3. 잔차와 잡음(고친 식, R1): r = B − g2(부호 있는 실수). rr = r, 단 road가 있고 DARK_SIGMA_IN_ROAD이면 노면 안의 r만
       (노면 픽셀이 하나도 없으면 띠 전체). σ = max(1.4826 × median(|rr − median(rr)|), DARK_SIGMA_MIN), D = max(r, 0).
       max(r, 0)의 MAD를 쓰면 D = 0이 절반을 넘어 σ = 0이 된다.
    4. 이중 문턱: 강 D ≥ max(DARK_K_HI × σ, DARK_ABS_MIN), 약 D ≥ DARK_K_LO × σ. 약 마스크의 연결 요소 중 강 픽셀을 품은 것만 남긴다.
       (보완 2-1) DARK_SIGMA_LOCAL이면 σ는 위치마다의 국소 σ 지도(local_sigma_map, 밝은 쪽 잔차, 하한 = 전역 σ)이고
       contrast도 픽셀마다 국소 σ로 나눈 평균이다.
    5. 닫힘(타원 DARK_CLOSE_KSIZE) 뒤 넓이 DARK_MIN_AREA 미만을 지운다(골재 무늬).
       (보완 2-3) DARK_SPLIT_BLOB이면 그 전에 원판 열림으로 덩어리와 선을 나눈다(균열이 붙은 포트홀).
    6. 영역마다 area(픽셀 수), P(바깥 + 구멍 윤곽 길이, cv.RETR_CCOMP), thickness = 2A/P, dt_width, circ_inv = P²/(4πA),
       elong·fill·solidity(describe_regions), contrast = 영역 안 D의 평균 / σ, darkness(region_darkness)를 잰다.
       (보완 1-1) 균열용·포트홀용 점수 score_crack·score_pothole(region_scores, SCORE_FORMULA가 none이면 NaN)도 붙인다.
       (보완 2-4) SCORE_USE_EDGE면 경계 날카로움 edge를 재 두 점수에 min(1, edge / SCORE_EDGE_REF)를 곱한다(흐린 그림자를 깎음).
    TODO(B): WP2에서 구현한다. 지금은 자리만 있다."""
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


    #비교와 잔차 계산은 float32로 한다.
    coarse = median_background(gray, C.DARK_BG_KSIZE_SMALL)
    gray_float = gray.astype(np.float32)
    if C.DARK_BRIGHT_REF == "coarse":
        reference = coarse
    elif C.DARK_BRIGHT_REF == "road_median":
        # (보완 2-2, 9a) 기준 밝기를 노면(없으면 띠 전체) 밝기 중앙값 하나로: 억제 창보다 넓은 흰 줄무늬는 거친 배경 자체가 밝아져 못 거름
        road_pixels = np.asarray(road) > 0 if road is not None else None
        pixels = gray_float[road_pixels] if road_pixels is not None and np.any(road_pixels) else gray_float
        reference = np.full_like(gray_float, float(np.median(pixels)))
    else:
        raise ValueError(f"DARK_BRIGHT_REF는 'coarse' 또는 'road_median'이어야 함 ({C.DARK_BRIGHT_REF!r})")
    bright = gray_float > reference + C.DARK_BRIGHT_DELTA
    if C.DARK_BRIGHT_DILATE > 0:
        # (보완 2-2, 13번) 아주 밝은 픽셀(기준 + DARK_BRIGHT_DILATE_DELTA, 차선·횡단보도)만 팽창해 샤프닝이 그 옆에 만든 어두운 테두리까지
        # 기준 밝기로 메움. 샤프닝은 균열 양옆에도 밝은 오버슈트를 만들므로, 덜 밝은 픽셀까지 팽창하면 균열 자체를 덮어 지운다
        smooth = cv.medianBlur(gray, C.DARK_BRIGHT_SEED_KSIZE).astype(np.float32)   # 샤프닝이 키운 잡무늬 점은 씨앗이 되지 않게
        seed = (smooth > reference + C.DARK_BRIGHT_DILATE_DELTA).astype(np.uint8)
        size = 2 * C.DARK_BRIGHT_DILATE + 1
        bright = bright | (cv.dilate(seed, cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))) > 0)
    g2 = gray.copy()
    g2[bright] = reference[bright].astype(np.uint8)
    bg_small = median_background(g2, h * C.DARK_BG_FRAC / scale)
    bg_large = median_background(g2, h * C.DARK_BG_FRAC_LARGE / scale)
    residual = np.maximum(bg_small, bg_large) - g2.astype(np.float32)

    samples = residual.ravel()
    if road is not None and C.DARK_SIGMA_IN_ROAD:
        road_pixels = np.asarray(road) > 0
        if np.any(road_pixels):
            samples = residual[road_pixels]
    median = np.median(samples)
    sigma = max(1.4826 * float(np.median(np.abs(samples - median))), C.DARK_SIGMA_MIN)
    if C.DARK_SIGMA_LOCAL:
        sigma = np.maximum(local_sigma_map(residual, median, sigma, scale), sigma)
    darkness = np.maximum(residual, 0)
    strong = darkness >= (np.maximum(C.DARK_K_HI * sigma, C.DARK_ABS_MIN) if np.ndim(sigma) else max(C.DARK_K_HI * sigma, C.DARK_ABS_MIN))
    weak = (darkness >= C.DARK_K_LO * sigma).astype(np.uint8)
    count, weak_labels = cv.connectedComponents(weak, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[np.unique(weak_labels[strong])] = True
    keep[0] = False  # 배경 라벨은 후보에서 제외한다
    mask = (keep[weak_labels].astype(np.uint8) * 255)

    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (C.DARK_CLOSE_KSIZE, C.DARK_CLOSE_KSIZE)
    )
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    if C.DARK_SPLIT_BLOB:
        # (보완 2-3, 9c) 덩어리와 선 나누기: 원판(DARK_SPLIT_KSIZE) 열림으로 원판보다 굵은 덩어리(포트홀)만 남기고,
        # 나머지(원래 영역 − 1px 넓힌 덩어리)를 따로 둬 균열이 붙은 포트홀을 두 영역으로 가른다. 가는 선만 있는 영역은 그대로다
        disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (C.DARK_SPLIT_KSIZE, C.DARK_SPLIT_KSIZE))
        blob = cv.morphologyEx(mask, cv.MORPH_OPEN, disc)
        if blob.any():
            rest = cv.bitwise_and(mask, cv.bitwise_not(cv.dilate(blob, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))))
            mask = cv.bitwise_or(blob, rest)
    count, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv.CC_STAT_AREA] >= C.DARK_MIN_AREA
    keep[0] = False
    mask = keep[labels].astype(np.uint8) * 255
    regions = describe_regions(mask, C.DARK_MIN_AREA, gray)
    if C.SCORE_USE_EDGE:                                # (보완 2-4, 9b) 경계 날카로움을 잴 기울기 크기(밝은 표시를 메운 g2 기준)
        g2f = g2.astype(np.float32)
        gradient = np.hypot(cv.Sobel(g2f, cv.CV_32F, 1, 0, ksize=3), cv.Sobel(g2f, cv.CV_32F, 0, 1, ksize=3)) / 8.0

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
        if np.ndim(sigma):                              # 국소 σ: 픽셀마다 나눈 평균
            region["contrast"] = float((darkness[y:y + rh, x:x + rw][component] / sigma[y:y + rh, x:x + rw][component]).mean())
        else:
            region["contrast"] = float(darkness[y:y + rh, x:x + rw][component].mean() / sigma)
        # 점수는 종류가 정해진 뒤 detect_cracks·detect_potholes가 score에 넣는다(보완 1-1)
        region["score_crack"], region["score_pothole"] = region_scores(region, perimeter)
        if C.SCORE_USE_EDGE:
            # (보완 2-4, 9b) 경계 날카로움 = 영역 바로 바깥 1px 고리의 평균 기울기 ÷ 영역 안 평균 어둡기. 그림자는 가장자리가 흐려 작다.
            # 점수에 min(1, 날카로움 / SCORE_EDGE_REF)를 곱한다(가장자리 흐림 σ 2px까지는 깎지 않음)
            x0, y0, x1, y1 = max(x - 1, 0), max(y - 1, 0), min(x + rw + 1, w), min(y + rh + 1, h)
            inside = labels[y0:y1, x0:x1] == label
            ring = cv.dilate(inside.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~inside
            depth = float(darkness[y:y + rh, x:x + rw][component].mean())
            region["edge"] = float(gradient[y0:y1, x0:x1][ring].mean() / depth) if ring.any() and depth > 0 else 0.0
            term = min(1.0, region["edge"] / C.SCORE_EDGE_REF)
            region["score_crack"], region["score_pothole"] = region["score_crack"] * term, region["score_pothole"] * term
    return regions, mask

def detect_cracks(gray, road=None):
    """가늘고 긴 어두운 선을 균열 후보로 찾는다. (후보 list, 캐니 에지 uint8 0/255)를 돌려준다.
    road: 노면 마스크(노면 띠와 같은 크기 uint8 0/255) 또는 None(기본값, 지금과 같음). 노면 밖 후보를 지우는 일은 pipeline(C)이 한다.
    DETECT_MODE "legacy"(지금 방식)는 road를 쓰지 않는다:
    캐니 → 닫힘 → 넓이 CRACK_MIN_AREA 이상 중에서 길쭉함 ≥ CRACK_MIN_ELONG(곧은 균열) 이거나
    채움 ≤ CRACK_MAX_FILL(구불구불·갈래 균열)이면 균열 후보. 곧은 균열은 닫힘 뒤 띠가 되어 채움이 높으므로
    두 조건을 함께 요구하면(그리고) 가장 흔한 곧은 균열을 놓친다.
    마지막으로 주변보다 CRACK_MIN_DARKNESS 이상 어둡지 않은 후보(흰 차선·횡단보도 점선 같은 밝은 선)는 뺀다."""
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
    """주변보다 어두운 덩어리를 포트홀 후보로 찾는다. (후보 list, 이진 마스크 uint8 0/255)를 돌려준다.
    road: 노면 마스크 또는 None(기본값, 지금과 같음). DETECT_MODE "legacy"(DoG 방식)는 road를 쓰지 않는다."""
    if _detect_mode() == "dark":

        regions, mask = extract_dark_regions(gray, road)
        
        candidates = _keep_by_score([
            r for r in regions
            if r["circ_inv"] <= C.POT_MAX_CIRC_INV
            and r["solidity"] >= C.POT_MIN_SOLIDITY
        ], "pothole")
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
    
