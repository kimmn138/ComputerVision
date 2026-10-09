"""(파트 B) 전처리한 노면 영상에서 균열·포트홀 후보 영역을 찾는다."""
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


def _detect_mode():
    """config B 구역의 DETECT_MODE를 읽는다. "legacy" = 1차 방식(Canny 균열 + DoG 포트홀, road를 쓰지 않음),
    "dark" = 어두운 영역 한 번 추출 + 형태 분류(WP2·WP3). 다른 값이면 ValueError로 바로 알린다."""
    if C.DETECT_MODE not in ("legacy", "dark"):
        raise ValueError(f"DETECT_MODE는 'legacy' 또는 'dark'여야 함 ({C.DETECT_MODE!r})")
    return C.DETECT_MODE


def extract_dark_regions(gray, road=None):
    """배경보다 어두운 영역을 한 번에 뽑아, 균열·포트홀 후보를 같은 기준(배경 대비 / 잡음 σ)으로 만든다(Plan.md 3.4, B 전용 · 약속 표 밖).
    입력: 흑백 노면 띠 uint8 (H, W), road = 노면 마스크(같은 크기 0/255) 또는 None. 입력은 바꾸지 않는다.
    출력: (영역 list, 어두운 영역 마스크 uint8 0/255). 영역 dict = 후보 형식(area·elong·fill·solidity·bbox·contour)
    + thickness·dt_width·circ_inv(구멍 둘레 포함)·contrast·darkness. 숫자는 config B 구역의 DARK_* 값을 쓴다.
    1. 밝은 표시 억제: 1/DARK_BG_DOWNSCALE로 줄여 중앙값(DARK_BG_KSIZE_SMALL)으로 거친 배경 Bc를 만들고,
       gray > Bc + DARK_BRIGHT_DELTA인 픽셀(흰 차선·횡단보도)을 Bc 값으로 바꾼 복사본 g2를 만든다.
       한계: 억제 창보다 넓은 밝은 영역 옆은 '어두운 선'이 된다(Japan_000107 자갈 경계).
    2. 배경 B = 두 크기 창 중앙값의 큰 값: max(bg(g2, 띠 높이 × DARK_BG_FRAC), bg(g2, 띠 높이 × DARK_BG_FRAC_LARGE)).
       bg(영상, 창) = 1/DARK_BG_DOWNSCALE로 줄여 중앙값(창 / DARK_BG_DOWNSCALE을 홀수로) → 원래 크기. 한 창이면 큰 포트홀 가운데가 빈다.
    3. 잔차와 잡음(고친 식, R1): r = B − g2(부호 있는 실수). rr = r, 단 road가 있고 DARK_SIGMA_IN_ROAD이면 노면 안의 r만
       (노면 픽셀이 하나도 없으면 띠 전체). σ = max(1.4826 × median(|rr − median(rr)|), DARK_SIGMA_MIN), D = max(r, 0).
       max(r, 0)의 MAD를 쓰면 D = 0이 절반을 넘어 σ = 0이 된다.
    4. 이중 문턱: 강 D ≥ max(DARK_K_HI × σ, DARK_ABS_MIN), 약 D ≥ DARK_K_LO × σ. 약 마스크의 연결 요소 중 강 픽셀을 품은 것만 남긴다.
    5. 닫힘(타원 DARK_CLOSE_KSIZE) 뒤 넓이 DARK_MIN_AREA 미만을 지운다(골재 무늬).
    6. 영역마다 area(픽셀 수), P(바깥 + 구멍 윤곽 길이, cv.RETR_CCOMP), thickness = 2A/P, dt_width, circ_inv = P²/(4πA),
       elong·fill·solidity(describe_regions), contrast = 영역 안 D의 평균 / σ, darkness(region_darkness)를 잰다.
    TODO(B): WP2에서 구현한다. 지금은 자리만 있다."""
    raise NotImplementedError("TODO(B): extract_dark_regions는 WP2에서 구현 (Plan.md 3.4 (2), docs/tasks/B.md)")


def detect_cracks(gray, road=None):
    """가늘고 긴 어두운 선을 균열 후보로 찾는다. (후보 list, 캐니 에지 uint8 0/255)를 돌려준다.
    road: 노면 마스크(노면 띠와 같은 크기 uint8 0/255) 또는 None(기본값, 지금과 같음). 노면 밖 후보를 지우는 일은 pipeline(C)이 한다.
    DETECT_MODE "legacy"(지금 방식)는 road를 쓰지 않는다:
    캐니 → 닫힘 → 넓이 CRACK_MIN_AREA 이상 중에서 길쭉함 ≥ CRACK_MIN_ELONG(곧은 균열) 이거나
    채움 ≤ CRACK_MAX_FILL(구불구불·갈래 균열)이면 균열 후보. 곧은 균열은 닫힘 뒤 띠가 되어 채움이 높으므로
    두 조건을 함께 요구하면(그리고) 가장 흔한 곧은 균열을 놓친다.
    마지막으로 주변보다 CRACK_MIN_DARKNESS 이상 어둡지 않은 후보(흰 차선·횡단보도 점선 같은 밝은 선)는 뺀다."""
    if _detect_mode() == "dark":
        # TODO(B) WP2·WP3: extract_dark_regions(gray, road)의 영역 중 circ_inv ≥ CRACK_MIN_CIRC_INV인 것을 균열로.
        # 에지는 row의 edges를 1차 결과와 비교할 수 있게 지금과 같은 Canny로 계산해 함께 돌려준다.
        raise NotImplementedError("TODO(B): DETECT_MODE='dark' 균열 후보 (Plan.md 3.4·3.5, docs/tasks/B.md)")

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
        # TODO(B) WP2·WP3: extract_dark_regions(gray, road)의 영역 중 circ_inv ≤ POT_MAX_CIRC_INV이고
        # solidity ≥ POT_MIN_SOLIDITY인 것을 포트홀로. 마스크는 어두운 영역 마스크(0/255)를 돌려준다.
        raise NotImplementedError("TODO(B): DETECT_MODE='dark' 포트홀 후보 (Plan.md 3.4·3.5, docs/tasks/B.md)")

    # TODO(B): DoG → 이진화 → 열림 → 모양으로 거르기, (후보, 마스크) 반환
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
    
