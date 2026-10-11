"""영상 1장의 전체 처리 흐름. 실험(run_experiment)과 시연(demo)이 같은 함수를 쓴다."""
import time

import cv2 as cv
import numpy as np

from src import analyze, detect, features, io_utils, preprocess
from src import config as C


def local_std(gray, ksize):
    """픽셀마다 ksize × ksize 창 안의 밝기 표준편차(결)를 float32 영상으로 돌려준다.
    아스팔트는 골재 때문에 값이 크고, 하늘처럼 매끈한 곳은 작다."""
    g = gray.astype(np.float32)
    mean = cv.blur(g, (ksize, ksize))
    return np.sqrt(np.maximum(cv.blur(g * g, (ksize, ksize)) - mean * mean, 0))


def auto_band_top(small, top):
    """노면 마스크를 보고 노면 띠의 위쪽 경계를 위로 넓힌다(노면만 찍힌 드론·오토바이 시점 사진용).
    바로 위 구간이 노면 비율 조건(과 결 조건)을 만족하는 동안 넓히고, 정한 경계(비율)를 돌려준다."""
    road = io_utils.road_mask(small) > 0
    h = road.shape[0]
    if C.AUTO_BAND_USE_TEXTURE:
        tex = local_std(cv.cvtColor(small, cv.COLOR_BGR2GRAY), C.AUTO_BAND_TEX_KSIZE)
        ref = tex[int(h * top):][road[int(h * top):]]
        ref_tex = float(np.median(ref)) if ref.size else 0.0
        if ref_tex <= 0:
            return top
    cur = top
    while cur > 0:
        nxt = max(0.0, round(cur - C.AUTO_BAND_STEP, 6))    # 0.1씩 빼며 생기는 소수 오차 제거
        seg = road[int(h * nxt):int(h * cur)]
        if seg.size == 0 or np.count_nonzero(seg) < C.AUTO_BAND_ROAD_MIN * seg.size:
            break
        if C.AUTO_BAND_USE_TEXTURE:
            ratio = float(np.median(tex[int(h * nxt):int(h * cur)][seg])) / ref_tex
            if not 1 / C.AUTO_BAND_TEX_RATIO <= ratio <= C.AUTO_BAND_TEX_RATIO:
                break
        cur = nxt
    return cur


def keep_on_road(regions, road, rule=None):
    """노면 밖(건물·차·풀숲 등)에 있는 후보를 지운 새 list를 돌려준다.
    rule "center"는 상자 중심이 노면 안이면, "ratio"는 후보 영역의 ROAD_KEEP_RATIO_MIN 이상이 노면 안이면 남긴다."""
    rule = C.ROAD_KEEP_RULE if rule is None else rule
    if rule not in ("center", "ratio"):
        raise ValueError(f"ROAD_KEEP_RULE은 'center'·'ratio'만 됨 ({rule!r})")
    h, w = road.shape
    kept = []
    for r in regions:
        x, y, bw, bh = r["bbox"]
        if rule == "center":
            on = road[min(int(y + bh / 2), h - 1), min(int(x + bw / 2), w - 1)] > 0
        else:
            on = road_ratio(r, road) >= C.ROAD_KEEP_RATIO_MIN
        if on:
            kept.append(r)
    return kept


def road_ratio(region, road):
    """후보 영역(윤곽 안쪽까지 채운 픽셀) 중 노면 안에 있는 비율을 돌려준다. 채운 픽셀이 없으면 0.0."""
    x, y, bw, bh = region["bbox"]
    fill = np.zeros((bh, bw), np.uint8)
    cv.drawContours(fill, [region["contour"]], -1, 255, -1, offset=(-x, -y))
    inside = fill > 0
    n = np.count_nonzero(inside)
    return float(np.count_nonzero(road[y:y + bh, x:x + bw][inside])) / n if n else 0.0


def run_pipeline(bgr, top, bottom, use_preprocess):
    """사진 1장을 크기 조정 → 노면 띠 자르기 → (전처리) → 검출 → 측정까지 처리해 결과 dict로 돌려준다.
    전처리 끔·켬의 차이는 steps 하나뿐이고, 나머지 과정과 파라미터는 항상 같다."""
    small = io_utils.resize_width(bgr)
    t_band_ms = 0.0
    if C.AUTO_BAND_TOP:
        t_band = time.perf_counter()
        top = auto_band_top(small, top)
        t_band_ms = round((time.perf_counter() - t_band) * 1000, 2)
    roi, y0 = io_utils.crop_roi(small, top, bottom)
    t_road = time.perf_counter()
    road = io_utils.road_mask(roi)
    t_road_ms = round((time.perf_counter() - t_road) * 1000, 2)
    road_arg = road if C.USE_ROAD_MASK else None
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    t0 = time.perf_counter()
    if use_preprocess:                                  # 상태 분석도 전처리 시간에 포함
        quality = analyze.measure_quality(gray)
        steps = analyze.choose_steps(quality)
    else:
        quality, steps = None, dict(analyze.NO_STEPS)
    proc = preprocess.preprocess(gray, steps)
    t1 = time.perf_counter()
    cracks, edges = detect.detect_cracks(proc, road_arg)
    potholes, pmask = detect.detect_potholes(proc, road_arg)
    t2 = time.perf_counter()
    if C.USE_ROAD_MASK:                                 # 노면 밖 후보 제거 (검출 시간에는 넣지 않음)
        cracks, potholes = keep_on_road(cracks, road), keep_on_road(potholes, road)

    row = {"edges": features.count_edges(edges),
           "harris": features.count_harris(proc),
           "n_crack": len(cracks), "area_crack": sum(r["area"] for r in cracks),
           "n_pothole": len(potholes), "area_pothole": sum(r["area"] for r in potholes),
           "roi_pixels": gray.size,
           "t_pre_ms": round((t1 - t0) * 1000, 2), "t_detect_ms": round((t2 - t1) * 1000, 2)}
    return {"small": small, "y0": y0, "gray": gray, "proc": proc, "edges": edges,
            "pmask": pmask, "cracks": cracks, "potholes": potholes,
            "steps": steps, "quality": quality, "row": row,
            "road": road, "t_road_ms": t_road_ms, "top": top, "t_band_ms": t_band_ms}
