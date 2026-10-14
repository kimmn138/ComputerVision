"""사진 1장의 전체 처리 흐름. 실험(run_experiment)과 시연(demo)이 같은 함수를 쓴다."""
import time

import cv2 as cv
import numpy as np

from src import analyze, detect, features, io_utils, preprocess
from src import config as C


def local_std(gray, ksize):
    """픽셀마다 ksize × ksize 창 안의 밝기 표준편차(질감)를 float32 영상으로 돌려준다. 아스팔트는 골재 때문에 크고 하늘은 작다."""
    g = gray.astype(np.float32)
    mean = cv.blur(g, (ksize, ksize))
    return np.sqrt(np.maximum(cv.blur(g * g, (ksize, ksize)) - mean * mean, 0))


def auto_band_top(small, top):
    """노면이 위쪽까지 찍힌 사진(드론·오토바이 시점)에서 노면을 버리지 않도록 ROI 상단 경계를 자동으로 올린다.
    top에서 시작해 바로 위 구간이 거의 다 노면이고 질감이 아래 ROI와 비슷하면 그 구간까지 올리고, 아니면 멈춘다.
    질감 조건은 노면 마스크가 색만 보고 잡은 흐린 하늘·나무를 거르려고 둔다. 정한 경계(비율)를 돌려준다."""
    road = io_utils.road_mask(small) > 0
    h = road.shape[0]
    if C.AUTO_BAND_USE_TEXTURE:
        tex = local_std(cv.cvtColor(small, cv.COLOR_BGR2GRAY), C.AUTO_BAND_TEX_KSIZE)
        ref = tex[int(h * top):][road[int(h * top):]]
        ref_tex = float(np.median(ref)) if ref.size else 0.0
        if ref_tex <= 0:
            return top
    cur, step = top, 0
    while cur > 0:
        nxt = max(0.0, round(cur - C.AUTO_BAND_STEP, 6))    # 0.5 − 0.1 − 0.1 …의 소수 오차(0.30000000000000004)를 없앰
        seg = road[int(h * nxt):int(h * cur)]
        if seg.size == 0 or np.count_nonzero(seg) < C.AUTO_BAND_ROAD_MIN * seg.size:
            break
        if C.AUTO_BAND_USE_TEXTURE:
            ratio = float(np.median(tex[int(h * nxt):int(h * cur)][seg])) / ref_tex
            low = 1 / C.AUTO_BAND_TEX_RATIO
            if C.AUTO_BAND_TEX_PERSPECTIVE:
                low *= C.AUTO_BAND_TEX_DECAY ** (step + 1)
            if not low <= ratio <= C.AUTO_BAND_TEX_RATIO:
                break
        cur, step = nxt, step + 1
    return cur


def keep_on_road(regions, road, rule=None):
    """노면 마스크 밖의 후보를 지워, 건물·차·풀숲·보닛에서 나온 오검출을 뺀다(새 list를 돌려줌).
    rule "ratio": 후보 영역 중 노면 안 비율이 ROAD_KEEP_RATIO_MIN 이상이면 남김, "center": 상자 중심이 노면 안이면 남김."""
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
    """후보 하나의 영역(윤곽 안쪽까지 채운 픽셀) 중 노면 안 비율을 잰다. 채운 픽셀이 없으면 0.0."""
    x, y, bw, bh = region["bbox"]
    fill = np.zeros((bh, bw), np.uint8)
    cv.drawContours(fill, [region["contour"]], -1, 255, -1, offset=(-x, -y))
    inside = fill > 0
    n = np.count_nonzero(inside)
    return float(np.count_nonzero(road[y:y + bh, x:x + bw][inside])) / n if n else 0.0


def run_pipeline(bgr, top, bottom, use_preprocess):
    """사진 1장을 처리한다: 640px 맞춤 → ROI 상단 경계 자동 조정 → ROI 자르기 → 노면 마스크 → 화질 측정·전처리 → 검출 → 노면 밖 후보 제거.
    use_preprocess(전처리 적용·미적용)만 바꾸고 나머지는 늘 같다. 결과 dict에 중간 영상·후보·row(지표 9개)·처리 시간을 담는다."""
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
    if use_preprocess:                                  # 화질 측정도 전처리 시간에 넣음
        quality = analyze.measure_quality(gray)
        steps = analyze.choose_steps(quality)
    else:
        quality, steps = None, dict(analyze.NO_STEPS)
    proc = preprocess.preprocess(gray, steps)
    t1 = time.perf_counter()
    cracks, edges = detect.detect_cracks(proc, road_arg)
    potholes, pmask = detect.detect_potholes(proc, road_arg)
    t2 = time.perf_counter()
    if C.USE_ROAD_MASK:                                 # 노면 밖 후보 제거는 검출 시간에 넣지 않음
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
