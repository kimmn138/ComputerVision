"""(파트 C) 영상 1장의 전체 흐름. 실험(run_experiment)과 시연(demo)이 같은 함수를 쓴다."""
import time

import cv2 as cv
import numpy as np

from src import analyze, detect, features, io_utils, preprocess
from src import config as C


def local_std(gray, ksize):
    """흑백 영상의 픽셀마다 ksize × ksize 창 안 밝기 표준편차(결)를 float32 영상으로 돌려준다. 아스팔트는 골재 때문에 크고 하늘은 작다.
    창 평균과 제곱 평균(cv.blur)으로 계산해 픽셀마다 도는 for 문을 쓰지 않는다. 받은 영상은 바꾸지 않는다."""
    g = gray.astype(np.float32)
    mean = cv.blur(g, (ksize, ksize))
    return np.sqrt(np.maximum(cv.blur(g * g, (ksize, ksize)) - mean * mean, 0))


def fixed_seed_box(width, height, top, bottom, y0):
    """(v5 V1c) 시드 상자를 '원래 ROI(top~bottom 비율)'의 ROAD_SEED_BOX 자리에 두고, 위에서 y0픽셀을 잘라 낸 영상의 좌표로 돌려준다
    → (x0, y0, x1, y1) 픽셀. ROI가 넓어져도 시드가 위쪽(나무·하늘)으로 올라가지 않게 하려고다. 넓어지지 않았으면(y0 = 원래 ROI 시작)
    io_utils.road_mask의 상자와 같다."""
    fx0, fy0, fx1, fy1 = C.ROAD_SEED_BOX
    r0, r1 = int(height * top), int(height * bottom)
    span = max(1, r1 - r0)
    return (int(width * fx0), r0 + int(span * fy0) - y0, int(width * fx1), r0 + int(span * fy1) - y0)


def auto_band_top(small, top):
    """노면만 찍힌 사진(드론·오토바이 시점)에서 띠 위쪽 노면을 버리지 않으려고, 사진 전체 노면 마스크를 보고 띠 위쪽 경계를 정한다.
    small은 640px 컬러 사진 전체, top은 시작 경계(비율). 바로 위 구간(config C 구역 AUTO_BAND_STEP 높이)의 노면 픽셀 비율이
    AUTO_BAND_ROAD_MIN 이상이면 그 구간까지 넓히고, 모자란 구간이 나오면 멈춘다. 0.0까지 넓어질 수 있고 좁히지는 않는다.
    AUTO_BAND_USE_TEXTURE가 켜져 있으면 결도 본다: 구간 노면 픽셀의 결(local_std) 중앙값 ÷ 시작 띠(top 아래) 노면 픽셀의 결 중앙값이
    1/AUTO_BAND_TEX_RATIO ~ AUTO_BAND_TEX_RATIO 밖이면 멈춘다. 노면 마스크가 색만 보고 칠한 흐린 하늘(매끈함)·나무(거침)를 거르려고다.
    AUTO_BAND_TEX_PERSPECTIVE가 켜져 있으면 위로 k번째 구간의 아래 한계를 (1/AUTO_BAND_TEX_RATIO) × AUTO_BAND_TEX_DECAY^k로 낮춘다
    (원근 때문에 먼 노면의 결이 매끈해 보이는 만큼 봐줌, 보완 4-4). 위 한계는 그대로다.
    시작 띠에 노면 픽셀이 없거나 결이 0이면 견줄 기준이 없어 넓히지 않는다.
    노면 마스크는 A의 io_utils.road_mask를 사진 전체에 한 번 부른다. 구간 경계는 crop_roi와 같이 int(높이 × 비율)로 자른다.
    받은 영상은 바꾸지 않고, 정한 경계(비율, float)를 돌려준다."""
    if C.AUTO_BAND_SEED_FIXED:                          # (v5 V1c) 사진 전체 마스크의 시드도 원래 아래쪽 ROI(top~ROI_BOTTOM_DEFAULT) 자리
        h_img, w_img = small.shape[:2]
        road = io_utils.road_mask_seeded(small, fixed_seed_box(w_img, h_img, top, C.ROI_BOTTOM_DEFAULT, 0)) > 0
    else:
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
    """노면 마스크 밖의 후보를 지워, 노면 밖 배경(건물·차·풀숲·보닛)에서 나온 오검출을 뺀다. 받은 list는 바꾸지 않고 새 list를 돌려준다.
    rule(기본 config C 구역의 ROAD_KEEP_RULE) "center": 후보 bbox의 중심 픽셀이 노면(255)이면 남긴다. 후보와 road는 둘 다 노면 띠 좌표다.
    "ratio": 후보 영역(윤곽 안쪽까지 채운 픽셀) 중 노면 안 비율이 ROAD_KEEP_RATIO_MIN 이상이면 남긴다. 노면 경계가 비스듬하면
    바깥 후보의 상자 중심이 노면에 떨어져 "center"로는 남기 때문이다(Plan.md 3.3 (7)). 그 밖의 규칙은 ValueError."""
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
    """후보 하나의 영역(윤곽 안쪽까지 채운 픽셀) 중 노면(255) 안 비율을 잰다. 'ratio' 남기기 규칙과 확인용 표가 같은 계산을 쓰게 따로 둔다.
    bbox 크기의 작은 영상에 윤곽을 채워 그려 사진 전체 크기 배열을 만들지 않는다. 채운 픽셀이 없으면 0.0."""
    x, y, bw, bh = region["bbox"]
    fill = np.zeros((bh, bw), np.uint8)
    cv.drawContours(fill, [region["contour"]], -1, 255, -1, offset=(-x, -y))
    inside = fill > 0
    n = np.count_nonzero(inside)
    return float(np.count_nonzero(road[y:y + bh, x:x + bw][inside])) / n if n else 0.0


def run_pipeline(bgr, top, bottom, use_preprocess):
    """전처리 여부만 바꾸고, 나머지(크기·ROI·노면 마스크·검출·파라미터)는 항상 같다.
    노면 마스크(road)는 전처리와 상관없이 컬러 노면 띠에서 계산하므로 끔·켬의 차이는 steps 하나뿐이다. 걸린 시간은 t_road_ms.
    config C 구역의 USE_ROAD_MASK가 켜져 있으면 road를 검출 함수에 넘기고(잡음 σ를 노면 안에서 재게) bbox 중심이 노면 밖인
    후보를 지운 뒤 row를 센다. 꺼져 있으면 road를 넘기지 않고 후보도 지우지 않는다(1차 결과와 같음). road는 어느 쪽이든 결과에 담는다.
    config C 구역의 AUTO_BAND_TOP이 켜져 있으면 받은 top에서 시작해 auto_band_top으로 띠 위쪽 경계를 넓힌 뒤 자른다(컬러 사진만 보므로
    끔·켬에 같음). 실제로 쓴 경계는 top, 그 계산 시간은 t_band_ms(꺼져 있으면 0.0)로 결과에 담는다. 꺼져 있으면 받은 top 그대로다."""
    small = io_utils.resize_width(bgr)
    t_band_ms = 0.0
    top0 = top
    if C.AUTO_BAND_TOP:
        t_band = time.perf_counter()
        top = auto_band_top(small, top)
        t_band_ms = round((time.perf_counter() - t_band) * 1000, 2)
    roi, y0 = io_utils.crop_roi(small, top, bottom)
    t_road = time.perf_counter()
    if C.AUTO_BAND_SEED_FIXED and top != top0:          # (v5 V1c) ROI가 넓어졌으면 시드는 받은 top의 ROI 자리(넓어지지 않았으면 같은 상자라 road_mask)
        road = io_utils.road_mask_seeded(roi, fixed_seed_box(small.shape[1], small.shape[0], top0, bottom, y0))
    else:
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
