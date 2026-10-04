"""(파트 B) 전처리한 노면 영상에서 균열·포트홀 후보 영역을 찾는다."""
import cv2 as cv
import numpy as np

from src import config as C
def describe_regions(mask, min_pixels=30):
    """이진 마스크의 연결 영역마다 모양 지표를 계산해 후보 dict로 만든다."""
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

        regions.append({
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
        })

    return regions
def detect_cracks(gray):
    """가늘고 긴 어두운 선을 균열 후보로 찾는다. (후보 list, 캐니 에지 uint8 0/255)를 돌려준다."""
    # TODO(B): 캐니 → 닫힘 → 모양으로 거르기, (후보, 캐니 에지) 반환
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
    )

    candidates = [
        region
        for region in regions
        if region["area"] >= C.CRACK_MIN_AREA
        and region["elong"] >= C.CRACK_MIN_ELONG
        and region["fill"] <= C.CRACK_MAX_FILL
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
    
