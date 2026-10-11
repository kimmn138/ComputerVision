"""(파트 C) 영상 1장의 전체 흐름. 실험(run_experiment)과 시연(demo)이 같은 함수를 쓴다."""
import time

import cv2 as cv
import numpy as np

from src import analyze, detect, features, io_utils, preprocess
from src import config as C


def auto_band_top(small, top):
    """노면만 찍힌 사진(드론·오토바이 시점)에서 띠 위쪽 노면을 버리지 않으려고, 사진 전체 노면 마스크를 보고 띠 위쪽 경계를 정한다.
    small은 640px 컬러 사진 전체, top은 시작 경계(비율). 바로 위 구간(config C 구역 AUTO_BAND_STEP 높이)의 노면 픽셀 비율이
    AUTO_BAND_ROAD_MIN 이상이면 그 구간까지 넓히고, 모자란 구간이 나오면 멈춘다. 0.0까지 넓어질 수 있고 좁히지는 않는다.
    노면 마스크는 A의 io_utils.road_mask를 사진 전체에 한 번 부른다. 구간 경계는 crop_roi와 같이 int(높이 × 비율)로 자른다.
    받은 영상은 바꾸지 않고, 정한 경계(비율, float)를 돌려준다."""
    road = io_utils.road_mask(small)
    h = road.shape[0]
    cur = top
    while cur > 0:
        nxt = max(0.0, round(cur - C.AUTO_BAND_STEP, 6))    # 0.5 − 0.1 − 0.1 …의 소수 오차(0.30000000000000004)를 없앰
        seg = road[int(h * nxt):int(h * cur)]
        if seg.size == 0 or np.count_nonzero(seg) < C.AUTO_BAND_ROAD_MIN * seg.size:
            break
        cur = nxt
    return cur


def keep_on_road(regions, road, rule=None):
    """노면 마스크 밖의 후보를 지워, 노면 밖 배경(건물·차·풀숲·보닛)에서 나온 오검출을 뺀다. 받은 list는 바꾸지 않고 새 list를 돌려준다.
    rule(기본 config C 구역의 ROAD_KEEP_RULE) "center": 후보 bbox의 중심 픽셀이 노면(255)이면 남긴다. 후보와 road는 둘 다 노면 띠 좌표다.
    다른 규칙(예: 윤곽 픽셀의 50% 이상이 노면 안)은 C가 WP5에서 비교한다(Plan.md 3.3 (7), Should)."""
    rule = C.ROAD_KEEP_RULE if rule is None else rule
    if rule != "center":
        raise ValueError(f"ROAD_KEEP_RULE은 지금 'center'만 됨 ({rule!r})")
    h, w = road.shape
    kept = []
    for r in regions:
        x, y, bw, bh = r["bbox"]
        if road[min(int(y + bh / 2), h - 1), min(int(x + bw / 2), w - 1)] > 0:
            kept.append(r)
    return kept


def run_pipeline(bgr, top, bottom, use_preprocess):
    """전처리 여부만 바꾸고, 나머지(크기·ROI·노면 마스크·검출·파라미터)는 항상 같다.
    노면 마스크(road)는 전처리와 상관없이 컬러 노면 띠에서 계산하므로 끔·켬의 차이는 steps 하나뿐이다. 걸린 시간은 t_road_ms.
    config C 구역의 USE_ROAD_MASK가 켜져 있으면 road를 검출 함수에 넘기고(잡음 σ를 노면 안에서 재게) bbox 중심이 노면 밖인
    후보를 지운 뒤 row를 센다. 꺼져 있으면 road를 넘기지 않고 후보도 지우지 않는다(1차 결과와 같음). road는 어느 쪽이든 결과에 담는다.
    config C 구역의 AUTO_BAND_TOP이 켜져 있으면 받은 top에서 시작해 auto_band_top으로 띠 위쪽 경계를 넓힌 뒤 자른다(컬러 사진만 보므로
    끔·켬에 같음). 실제로 쓴 경계는 top, 그 계산 시간은 t_band_ms(꺼져 있으면 0.0)로 결과에 담는다. 꺼져 있으면 받은 top 그대로다."""
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
    if use_preprocess:                                  # 상태 분석도 전처리 시간에 넣는다
        quality = analyze.measure_quality(gray)
        steps = analyze.choose_steps(quality)
    else:
        quality, steps = None, dict(analyze.NO_STEPS)
    proc = preprocess.preprocess(gray, steps)
    t1 = time.perf_counter()
    cracks, edges = detect.detect_cracks(proc, road_arg)
    potholes, pmask = detect.detect_potholes(proc, road_arg)
    t2 = time.perf_counter()
    if C.USE_ROAD_MASK:                                 # 노면 밖 후보 제거는 검출 시간에 넣지 않는다(row 시간 열은 1차와 같은 뜻)
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
