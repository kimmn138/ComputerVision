"""(파트 C) 영상 1장의 전체 흐름. 실험(run_experiment)과 시연(demo)이 같은 함수를 쓴다."""
import time

import cv2 as cv

from src import analyze, detect, features, io_utils, preprocess
from src import config as C


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
    후보를 지운 뒤 row를 센다. 꺼져 있으면 road를 넘기지 않고 후보도 지우지 않는다(1차 결과와 같음). road는 어느 쪽이든 결과에 담는다."""
    small = io_utils.resize_width(bgr)
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
            "road": road, "t_road_ms": t_road_ms}
