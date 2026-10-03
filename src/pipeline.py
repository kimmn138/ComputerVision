"""(파트 C) 영상 1장의 전체 흐름. 실험(run_experiment)과 시연(demo)이 같은 함수를 쓴다."""
import time

import cv2 as cv

from src import analyze, detect, features, io_utils, preprocess


def run_pipeline(bgr, top, bottom, use_preprocess):
    """전처리 여부만 바꾸고, 나머지(크기·ROI·검출·파라미터)는 항상 같다."""
    small = io_utils.resize_width(bgr)
    roi, y0 = io_utils.crop_roi(small, top, bottom)
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    t0 = time.perf_counter()
    if use_preprocess:                                  # 상태 분석도 전처리 시간에 넣는다
        quality = analyze.measure_quality(gray)
        steps = analyze.choose_steps(quality)
    else:
        quality, steps = None, dict(analyze.NO_STEPS)
    proc = preprocess.preprocess(gray, steps)
    t1 = time.perf_counter()
    cracks, edges = detect.detect_cracks(proc)
    potholes, pmask = detect.detect_potholes(proc)
    t2 = time.perf_counter()

    row = {"edges": features.count_edges(edges),
           "harris": features.count_harris(proc),
           "n_crack": len(cracks), "area_crack": sum(r["area"] for r in cracks),
           "n_pothole": len(potholes), "area_pothole": sum(r["area"] for r in potholes),
           "roi_pixels": gray.size,
           "t_pre_ms": round((t1 - t0) * 1000, 2), "t_detect_ms": round((t2 - t1) * 1000, 2)}
    return {"small": small, "y0": y0, "gray": gray, "proc": proc, "edges": edges,
            "pmask": pmask, "cracks": cracks, "potholes": potholes,
            "steps": steps, "quality": quality, "row": row}
