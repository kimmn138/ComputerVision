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


def describe_regions(mask, min_pixels=30, gray=None):
    """이진 마스크의 연결 영역마다 모양 지표를 계산해 후보 dict로 만든다.
    area = 픽셀 수, elong = 회전 사각형의 긴 변 / 짧은 변, fill = 윤곽 넓이 / 회전 사각형 넓이,
    solidity = 윤곽 넓이 / 볼록 껍질 넓이. gray를 주면 darkness(주변보다 어두운 정도, region_darkness)도 더한다."""
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

        if gray is not None:
            region["darkness"] = region_darkness(gray, contour)

        regions.append(region)

    return regions


def detect_cracks(gray):
    """가늘고 긴 어두운 선을 균열 후보로 찾는다. (후보 list, 캐니 에지 uint8 0/255)를 돌려준다.
    캐니 → 닫힘 → 넓이 CRACK_MIN_AREA 이상 중에서 길쭉함 ≥ CRACK_MIN_ELONG(곧은 균열) 이거나
    채움 ≤ CRACK_MAX_FILL(구불구불·갈래 균열)이면 균열 후보. 곧은 균열은 닫힘 뒤 띠가 되어 채움이 높으므로
    두 조건을 함께 요구하면(그리고) 가장 흔한 곧은 균열을 놓친다.
    마지막으로 주변보다 CRACK_MIN_DARKNESS 이상 어둡지 않은 후보(흰 차선·횡단보도 점선 같은 밝은 선)는 뺀다."""
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


def detect_potholes(gray):
    """주변보다 어두운 덩어리를 포트홀 후보로 찾는다. (후보 list, 이진 마스크 uint8 0/255)를 돌려준다."""
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
    
