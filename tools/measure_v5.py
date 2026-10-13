"""(파트 C) 보고서 v12 뒤 남은 문제 세 가지((1) ROI 상단 경계, (2) 밝은 포트홀, (3) 노면 마스크)를 고치기 전에 지금 config 그대로 잰다.
사용법: 맨 위 폴더에서  python tools/measure_v5.py [--limit N]
개발셋 652장(evaluate_train.assign_split의 dev)과 39장(data/provided·data/own, 자동 ROI = 시작 경계 0.5)만 읽는다. 시험셋 사진은 열지 않는다.
결과: results/measure_v5/
- A_roi_<셋>.csv: 사진마다 시작·정한 ROI 상단 경계, 사진 전체 노면 비율, ROI 위로 빠진 정답 상자 수, 첫 확장 구간의 노면 비율·질감 비
  (시작 ROI와 비교 = 지금, 바로 아래 구간과 비교 = 설계안 ②), 설계안 ①·②로 정했을 경계
- B_boxes_<셋>.csv: 정답 상자(포트홀·균열)와 가짜 대조군 상자마다 안·바깥 고리의 평균 L·a·b(CIE, L 0~100), ΔL, ΔE76, 질감 비, 밝은 표시 보정이 메운 비율
- C_mask_<셋>.csv: 사진마다 시드 영역의 밝은 표시·높은 채도·마스크 밖 비율, 노면 색인데 시드와 이어지지 않아 버려진 픽셀 비율(ROI 대비)
- summary.txt: 보고용 핵심 숫자, fig_*.png: 노면 마스크 확인 그림(사진 · 지금 마스크 · 버려진 노면 색 픽셀)
src의 함수(pipeline.auto_band_top, io_utils.road_mask)를 그대로 부르고, 중간값(첫 구간 조건·연결 전 후보)은 같은 식으로 따로 계산한 뒤
결과가 src와 같은지 사진마다 대조한다(다르면 summary에 개수를 적음)."""
import argparse
import ast
import sys
import zlib
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiment as rx  # noqa: E402
from src import config as C  # noqa: E402
from src import evaluate, io_utils, pipeline  # noqa: E402
from tools import evaluate_train as et  # noqa: E402

OUT_DIR = Path("results/measure_v5")
ROI_PATH = "data/roi.csv"
FULL_ROAD_LEVELS = (0.8, 0.9)   # 사진 전체 노면 비율 기준(사용자 지정)
RING_MIN_PX = 8                 # px, 바깥 고리 최소 폭. 노면 마스크의 흐림 σ 3px의 두 배보다 넓게(색 경계 번짐을 넘어서 재게)
RING_MIN_N = 50                 # 개, 고리 안 노면 픽셀이 이보다 적으면 '주변 노면 없음'으로 뺌
FAKE_TRIES = 200                # 번, 가짜 상자 자리를 찾는 시도 수
FAKE_ROAD_MIN = 0.8             # 비율, 가짜 상자 안 노면 마스크 비율 하한(정답 없는 '노면' 상자)
SMALL_DL = 5.0                  # L*(0~100), |ΔL| < 이 값이면 '차이 작음'. 색차 인지 한계(ΔE ≈ 2.3)의 약 두 배. 가짜 대조군 분위와 함께 보고
HIGH_CHROMA = 20                # Lab a·b 거리(128 기준), 높은 채도. 노면 색 허용 거리 ROAD_CHROMA_MAX 10의 두 배
SEED_BAD_RATIO = 0.2            # 비율, 시드에서 밝은 표시·높은 채도·마스크 밖 합이 이 값 이상이면 '오염된 시드' 목록
DISCARD_BIG = 0.05              # 비율(ROI 대비), 버려진 노면 색 픽셀이 이 값 이상이면 '큰 사진' 목록
N_FIGURES = 10


def dev_items():
    """개발셋 652장의 (이름, 경로, 정답 경로, ROI 시작 경계). 시험셋은 이름만 보고 걸러 사진을 열지 않는다."""
    paths = sorted(Path(C.TRAIN_IMAGE_DIR).glob("*.jpg"))
    split = et.assign_split([p.name for p in paths])
    table = io_utils.load_roi_table(ROI_PATH)
    items = []
    for p in paths:
        if split[p.name] != "dev":
            continue
        roi, _ = rx.get_roi(p.name, table)
        items.append((p.name, p, Path(C.TRAIN_GT_DIR) / f"{p.stem}.csv", roi["top"], roi["bottom"]))
    return items


def own_items():
    """39장의 (이름, 경로, 정답 경로, 자동 ROI 시작 경계). 자동 ROI(roi.csv 범위를 쓰지 않음)라 시작은 기본 0.5다."""
    table = io_utils.load_roi_table(ROI_PATH)
    items = []
    for p in map(Path, rx.list_images()):
        roi, _ = rx.get_roi(p.name, table, mode="auto")
        items.append((p.name, p, Path("gt") / f"{p.stem}.csv", roi["top"], roi["bottom"]))
    return items


def band_trace(small, top, road, tex):
    """pipeline.auto_band_top과 같은 계산으로 첫 확장 구간의 조건 값과 설계안 ②(질감 비를 바로 아래 구간과 비교)의 경계를 낸다."""
    h = road.shape[0]
    ref = tex[int(h * top):][road[int(h * top):]]
    ref_tex = float(np.median(ref)) if ref.size else 0.0
    nxt = max(0.0, round(top - C.AUTO_BAND_STEP, 6))
    seg = road[int(h * nxt):int(h * top)]
    below_end = min(1.0, round(top + C.AUTO_BAND_STEP, 6))
    below = road[int(h * top):int(h * below_end)]
    road1 = float(np.count_nonzero(seg)) / seg.size if seg.size else np.nan
    seg_tex = tex[int(h * nxt):int(h * top)][seg]
    below_tex = tex[int(h * top):int(h * below_end)][below]
    tex1 = float(np.median(seg_tex)) / ref_tex if seg_tex.size and ref_tex > 0 else np.nan
    tex1_below = float(np.median(seg_tex)) / float(np.median(below_tex)) if seg_tex.size and below_tex.size else np.nan
    in_range = lambda r: bool(np.isfinite(r) and 1 / C.AUTO_BAND_TEX_RATIO <= r <= C.AUTO_BAND_TEX_RATIO)
    # 설계안 ②: 위로 한 구간씩 가며 질감 비를 바로 아래 구간(이미 넓힌 마지막 구간)의 노면 질감과 비교
    cur = top
    while cur > 0:
        nx = max(0.0, round(cur - C.AUTO_BAND_STEP, 6))
        lo = min(1.0, round(cur + C.AUTO_BAND_STEP, 6))
        s = road[int(h * nx):int(h * cur)]
        if s.size == 0 or np.count_nonzero(s) < C.AUTO_BAND_ROAD_MIN * s.size:
            break
        b = road[int(h * cur):int(h * lo)]
        st, bt = tex[int(h * nx):int(h * cur)][s], tex[int(h * cur):int(h * lo)][b]
        if not (st.size and bt.size and in_range(float(np.median(st)) / float(np.median(bt)))):
            break
        cur = nx
    return {"road1": road1, "tex1": tex1, "tex1_below": tex1_below,
            "road1_ok": bool(road1 >= C.AUTO_BAND_ROAD_MIN), "tex1_ok": in_range(tex1), "tex1_below_ok": in_range(tex1_below),
            "top_alt2": cur}


def road_parts(bgr_roi):
    """io_utils.road_mask와 같은 식으로 (연결 전 노면 색 후보(열림 뒤), 시드 상자, 밝기 허용 폭 ±L)을 낸다. 지금 config에서 꺼진 v2 스위치(그림자·
    에지 밀도·차선·연결 전 닫힘·보닛)가 켜져 있으면 식이 달라지므로 멈춘다."""
    if C.ROAD_SHADOW_L_K or C.ROAD_EDGE_DENSITY or C.ROAD_LANE_BRIDGE or C.ROAD_CLOSE_BEFORE_SEED or C.ROAD_HOOD_CUT:
        raise RuntimeError("road_parts는 노면 마스크 v2 스위치가 모두 꺼진 config만 다시 계산한다")
    h, w = bgr_roi.shape[:2]
    lab = cv.cvtColor(cv.GaussianBlur(bgr_roi, (0, 0), C.ROAD_BLUR_SIGMA), cv.COLOR_BGR2LAB)
    L, a, b = (lab[:, :, i].astype(np.float32) for i in range(3))
    fx0, fy0, fx1, fy1 = C.ROAD_SEED_BOX
    x0, y0 = max(0, min(int(w * fx0), w - 1)), max(0, min(int(h * fy0), h - 1))
    x1, y1 = max(x0 + 1, min(int(w * fx1), w)), max(y0 + 1, min(int(h * fy1), h))
    sL, sa, sb = L[y0:y1, x0:x1], a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    L_med, a_med, b_med = float(np.median(sL)), float(np.median(sa)), float(np.median(sb))
    mad = float(np.median(np.abs(sL - L_med)))
    cand = (np.sqrt((a - a_med) ** 2 + (b - b_med) ** 2) < C.ROAD_CHROMA_MAX) & (np.abs(L - L_med) < C.ROAD_L_K * (1.4826 * mad + 2.0))
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (C.ROAD_OPEN_KSIZE, C.ROAD_OPEN_KSIZE))
    opened = cv.morphologyEx(cand.astype(np.uint8) * 255, cv.MORPH_OPEN, k) > 0
    return opened, (x0, y0, x1, y1), C.ROAD_L_K * (1.4826 * mad + 2.0)


def rebuild_mask(opened, seed_box):
    """road_parts의 연결 전 후보에서 io_utils.road_mask의 나머지(시드 연결 요소 → 닫힘 → 구멍 채우기)를 다시 해 bool 마스크를 낸다.
    src 결과와 같으면 '버려진 픽셀 = 연결 전 후보 − 마스크'가 road_mask의 실제 동작을 잰 것이다."""
    x0, y0, x1, y1 = seed_box
    _, labels = cv.connectedComponents(opened.astype(np.uint8) * 255, connectivity=8)
    keep = np.unique(labels[y0:y1, x0:x1])
    keep = keep[keep != 0]
    conn = np.isin(labels, keep).astype(np.uint8) * 255 if keep.size else opened.astype(np.uint8) * 255
    conn = cv.morphologyEx(conn, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (C.ROAD_CLOSE_KSIZE, C.ROAD_CLOSE_KSIZE)))
    return io_utils._fill_holes(conn) > 0


def figure_data(item):
    """확인 그림용으로 사진 1장의 640px 컬러, ROI y0, 지금 노면 마스크, 버려진 노면 색 픽셀, 시드 상자를 다시 계산한다."""
    name, path, _, top0, bottom = item
    small = io_utils.resize_width(io_utils.load_image(path))
    top = pipeline.auto_band_top(small, top0) if C.AUTO_BAND_TOP else top0
    roi, y0 = io_utils.crop_roi(small, top, bottom)
    road = io_utils.road_mask(roi) > 0
    opened, seed, _ = road_parts(roi)
    return {"small": small, "y0": y0, "road": road, "discarded": opened & ~road, "seed": seed}


def bright_fill(gray, dilate=True):
    """detect.extract_dark_regions 1단계(밝은 표시 보정)가 메우는 픽셀(bool)을 같은 식으로 낸다(DARK_BRIGHT_REF "coarse"만).
    dilate=False면 팽창 전 '거친 배경 + DARK_BRIGHT_DELTA보다 밝은 픽셀'만(시드의 밝은 표시 비율에 씀: 검출기가 밝은 표시로 보는 기준과 같고,
    햇빛 든 넓은 노면은 거친 배경도 함께 밝아 세지 않음)."""
    if C.DARK_BRIGHT_REF != "coarse":
        raise RuntimeError("bright_fill은 DARK_BRIGHT_REF 'coarse'만 다시 계산한다")
    h, w = gray.shape
    s = C.DARK_BG_DOWNSCALE
    size = (max(1, int(round(w / s))), max(1, int(round(h / s))))
    k = C.DARK_BG_KSIZE_SMALL | 1
    coarse = cv.resize(cv.medianBlur(cv.resize(gray, size, interpolation=cv.INTER_AREA), k), (w, h),
                       interpolation=cv.INTER_LINEAR).astype(np.float32)
    bright = gray.astype(np.float32) > coarse + C.DARK_BRIGHT_DELTA
    if dilate and C.DARK_BRIGHT_DILATE > 0:
        seed = (cv.medianBlur(gray, C.DARK_BRIGHT_SEED_KSIZE).astype(np.float32) > coarse + C.DARK_BRIGHT_DILATE_DELTA).astype(np.uint8)
        d = 2 * C.DARK_BRIGHT_DILATE + 1
        bright |= cv.dilate(seed, cv.getStructuringElement(cv.MORPH_ELLIPSE, (d, d))) > 0
    return bright


def blurred_lab(small):
    """노면 마스크와 같은 σ(ROAD_BLUR_SIGMA)로 흐린 CIE Lab(float, L 0~100). 픽셀 ΔE가 골재 질감·잡음에 흔들리지 않게."""
    return cv.cvtColor(cv.GaussianBlur(small, (0, 0), C.ROAD_BLUR_SIGMA).astype(np.float32) / 255.0, cv.COLOR_BGR2LAB)


def ring_width(w, h):
    """고리 폭 d: (w + 2d)(h + 2d) = 2wh, 즉 고리 넓이 = 상자 넓이가 되게(안·바깥 표본 수를 맞춤). 최소 RING_MIN_PX."""
    d = (-(w + h) + np.sqrt((w + h) ** 2 + 4 * w * h)) / 4
    return max(RING_MIN_PX, int(round(d)))


def box_stats(box, lab, tex, road_full, gt_any, fill=None, labb=None):
    """상자 안(모든 픽셀)과 바깥 고리(노면 마스크 안, 다른 정답 상자 밖)의 평균 L·a·b와 질감 중앙값을 비교한다. 고리가 모자라면 None.
    labb(노면 마스크와 같은 σ로 흐린 Lab)를 주면 픽셀 단위도 잰다: 고리 평균색과의 픽셀 ΔE가 고리 픽셀 자신의 95백분위를 넘는
    상자 안 픽셀 비율(px_out, 노면과 같으면 약 5%)과 그중 고리보다 밝은·어두운 픽셀 비율(px_out_bright·px_out_dark).
    상자 평균은 정답 상자의 여백 노면과 섞여 희석되므로, 색 거리로 픽셀을 고르는 경로가 실제로 볼 값을 따로 본다."""
    H, W = road_full.shape
    x, y, w, h = box
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)
    if w <= 0 or h <= 0:
        return None
    d = ring_width(w, h)
    X0, Y0, X1, Y1 = max(0, x - d), max(0, y - d), min(W, x + w + d), min(H, y + h + d)
    ring = road_full[Y0:Y1, X0:X1] & ~gt_any[Y0:Y1, X0:X1]
    ring[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = False
    if np.count_nonzero(ring) < RING_MIN_N:
        return None
    inn = lab[y:y + h, x:x + w].reshape(-1, 3).mean(0)
    out = lab[Y0:Y1, X0:X1][ring].mean(0)
    t_in, t_out = float(np.median(tex[y:y + h, x:x + w])), float(np.median(tex[Y0:Y1, X0:X1][ring]))
    row = {"L_in": inn[0], "a_in": inn[1], "b_in": inn[2], "L_ring": out[0], "a_ring": out[1], "b_ring": out[2],
           "dL": inn[0] - out[0], "dE": float(np.linalg.norm(inn - out)), "tex_ratio": t_in / t_out if t_out > 0 else np.nan,
           "ring_px": int(d), "ring_n": int(np.count_nonzero(ring))}
    if fill is not None:
        row["fill_ratio"] = float(fill[y:y + h, x:x + w].mean())
    if labb is not None:
        ref = labb[Y0:Y1, X0:X1][ring].mean(0)
        t95 = float(np.quantile(np.linalg.norm(labb[Y0:Y1, X0:X1][ring] - ref, axis=1), 0.95))
        px = labb[y:y + h, x:x + w].reshape(-1, 3)
        out_px = np.linalg.norm(px - ref, axis=1) > t95
        row.update({"px_t95": t95, "px_out": float(out_px.mean()), "px_out_bright": float((out_px & (px[:, 0] > ref[0])).mean()),
                    "px_out_dark": float((out_px & (px[:, 0] <= ref[0])).mean())})
    return row


def place_fake(rng, size, y_min, road_full, gt_any):
    """같은 크기 상자를 ROI 안 노면(노면 비율 ≥ FAKE_ROAD_MIN), 정답 상자(고리 폭만큼 넓힘)와 겹치지 않는 자리에 무작위로 놓는다."""
    H, W = road_full.shape
    w, h = size
    if w > W or h > H - y_min:
        return None
    d = ring_width(w, h)
    for _ in range(FAKE_TRIES):
        x, y = int(rng.integers(0, W - w + 1)), int(rng.integers(y_min, H - h + 1))
        if gt_any[max(0, y - d):y + h + d, max(0, x - d):x + w + d].any():
            continue
        if road_full[y:y + h, x:x + w].mean() >= FAKE_ROAD_MIN:
            return (x, y, w, h)
    return None


def analyze_image(item, set_name, pending):
    """사진 1장을 재서 (A 행, B 행 list, C 행)을 돌려준다. 같은 사진 안에서 가짜 자리를 못 찾은 정답 크기는 pending에 넣는다."""
    name, path, gt_path, top0, bottom = item
    small = io_utils.resize_width(io_utils.load_image(path))
    H, W = small.shape[:2]
    gt = evaluate.load_gt(str(gt_path)) or []
    gray_full = cv.cvtColor(small, cv.COLOR_BGR2GRAY)

    # A. ROI 상단 경계
    road_whole = io_utils.road_mask(small) > 0
    _, _, l_range_full = road_parts(small)
    tex_whole = pipeline.local_std(gray_full, C.AUTO_BAND_TEX_KSIZE)
    top = pipeline.auto_band_top(small, top0) if C.AUTO_BAND_TOP else top0
    tr = band_trace(small, top0, road_whole, tex_whole)
    full_ratio = float(road_whole.mean())
    y0 = int(H * top)
    a_row = {"set": set_name, "image": name, "source": et.source_of(name), "height": H, "top_start": top0, "top": top,
             "full_road": full_ratio, "l_range_full": l_range_full, "n_gt": len(gt), "gt_above": sum(1 for x, y, w, h, k in gt if y + h / 2 < y0),
             "pothole_above": sum(1 for x, y, w, h, k in gt if k == "pothole" and y + h / 2 < y0), **tr,
             **{f"top_alt1_{int(v * 100)}": (0.0 if full_ratio >= v else top) for v in FULL_ROAD_LEVELS}}

    # C. 노면 마스크(파이프라인과 같은 ROI)
    roi, y0 = io_utils.crop_roi(small, top, bottom)
    road = io_utils.road_mask(roi) > 0
    opened, (sx0, sy0, sx1, sy1), l_range = road_parts(roi)
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    lab8 = cv.cvtColor(roi, cv.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.sqrt((lab8[:, :, 1] - 128) ** 2 + (lab8[:, :, 2] - 128) ** 2)
    seed = (slice(sy0, sy1), slice(sx0, sx1))
    bright_seed = bright_fill(gray, dilate=False)[seed]
    hi_chroma = chroma[seed] > HIGH_CHROMA
    off_mask = ~road[seed]
    discarded = opened & ~road
    again = rebuild_mask(opened, (sx0, sy0, sx1, sy1))
    gt_center_in_band = [(x + w / 2, y + h / 2) for x, y, w, h, k in gt if y + h / 2 >= y0]
    in_mask = sum(1 for cx, cy in gt_center_in_band if road[min(int(cy) - y0, road.shape[0] - 1), min(int(cx), W - 1)])
    c_row = {"set": set_name, "image": name, "source": et.source_of(name), "top": top,
             "seed_bright": float(bright_seed.mean()), "seed_chroma": float(hi_chroma.mean()), "seed_off_mask": float(off_mask.mean()),
             "seed_bad": float((bright_seed | hi_chroma | off_mask).mean()),
             "l_range": l_range, "road_frac": float(road.mean()), "roadcolor_frac": float(opened.mean()), "discarded_frac": float(discarded.mean()),
             "gt_in_band": len(gt_center_in_band), "gt_in_mask": in_mask,
             "check_mask": bool(np.array_equal(again, road)),
             "check_band": bool((top < top0) == (tr["road1_ok"] and tr["tex1_ok"])) if C.AUTO_BAND_TOP else True}

    # B. 정답 상자 색(640px 사진 전체 좌표, 바깥 고리는 파이프라인 노면 마스크 안만)
    road_full = np.zeros((H, W), bool)
    road_full[y0:y0 + road.shape[0]] = road
    fill_full = np.zeros((H, W), bool)
    fill_full[y0:y0 + road.shape[0]] = bright_fill(gray)
    gt_any = np.zeros((H, W), bool)
    for x, y, w, h, k in gt:
        gt_any[max(0, y):y + h, max(0, x):x + w] = True
    labf = cv.cvtColor(small.astype(np.float32) / 255.0, cv.COLOR_BGR2LAB)
    labb = blurred_lab(small)
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    b_rows = []
    for i, (x, y, w, h, k) in enumerate(gt):
        base = {"set": set_name, "image": name, "source": et.source_of(name), "kind": k, "idx": i, "x": x, "y": y, "w": w, "h": h,
                "in_band": y + h / 2 >= y0}
        s = box_stats((x, y, w, h), labf, tex_whole, road_full, gt_any, fill_full, labb)
        b_rows.append({**base, **(s or {}), "ok": s is not None})
        fake = place_fake(rng, (w, h), y0, road_full, gt_any)
        fs = box_stats(fake, labf, tex_whole, road_full, gt_any, fill_full, labb) if fake else None
        if fs is None:
            pending.append((set_name, k, name, i, w, h))
        else:
            b_rows.append({**base, "kind": f"fake_{k}", "x": fake[0], "y": fake[1], "fake_from": "same", **fs, "ok": True})
    return a_row, b_rows, c_row


def fill_pending(pending, none_items):
    """같은 사진에서 가짜 자리를 못 찾은 정답 크기를 정답 없는 개발셋 사진에 차례로 놓는다(사진 순서 고정이라 같은 결과)."""
    rows = []
    if not pending or not none_items:
        return rows
    queue = list(pending)
    for j, item in enumerate(none_items):
        if not queue:
            break
        name, path, _, top0, bottom = item
        small = io_utils.resize_width(io_utils.load_image(path))
        H, W = small.shape[:2]
        top = pipeline.auto_band_top(small, top0) if C.AUTO_BAND_TOP else top0
        roi, y0 = io_utils.crop_roi(small, top, bottom)
        road_full = np.zeros((H, W), bool)
        road_full[y0:y0 + roi.shape[0]] = io_utils.road_mask(roi) > 0
        fill_full = np.zeros((H, W), bool)
        fill_full[y0:y0 + roi.shape[0]] = bright_fill(cv.cvtColor(roi, cv.COLOR_BGR2GRAY))
        tex = pipeline.local_std(cv.cvtColor(small, cv.COLOR_BGR2GRAY), C.AUTO_BAND_TEX_KSIZE)
        labf = cv.cvtColor(small.astype(np.float32) / 255.0, cv.COLOR_BGR2LAB)
        labb = blurred_lab(small)
        no_gt = np.zeros((H, W), bool)
        rng = np.random.default_rng(zlib.crc32(name.encode()) + 1)
        left = []
        for set_name, k, src, i, w, h in queue:
            fake = place_fake(rng, (w, h), y0, road_full, no_gt)
            fs = box_stats(fake, labf, tex, road_full, no_gt, fill_full, labb) if fake else None
            if fs is None:
                left.append((set_name, k, src, i, w, h))
            else:
                rows.append({"set": set_name, "image": name, "source": et.source_of(name), "kind": f"fake_{k}", "idx": i,
                             "x": fake[0], "y": fake[1], "w": w, "h": h, "in_band": True, "fake_from": f"none:{src}", **fs, "ok": True})
        queue = left
    return rows


def auc(pos, neg):
    """pos가 neg보다 클 확률(Mann–Whitney AUC). 0.5면 가를 수 없음."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def overlap(pos, neg, bins=40):
    """두 분포의 겹침 계수(정규화 히스토그램의 min 합, 0 = 완전히 갈림, 1 = 같음)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    edges = np.histogram_bin_edges(np.concatenate([pos, neg]), bins=bins)
    hp, _ = np.histogram(pos, edges)
    hn, _ = np.histogram(neg, edges)
    return float(np.minimum(hp / len(pos), hn / len(neg)).sum())


def sep_lines(df, set_name, pos_kind, neg_kind):
    """ΔE·|ΔL|·질감 비로 pos(정답)와 neg(가짜)를 얼마나 가르는지: AUC, 겹침, pos 50%·80%를 잡는 임계값에서 neg가 넘는 비율."""
    p = df[(df.set == set_name) & (df.kind == pos_kind) & df.ok]
    n = df[(df.set == set_name) & (df.kind == neg_kind) & df.ok]
    lines = [f"  {pos_kind} {len(p)}개 vs {neg_kind} {len(n)}개"]
    for col, f in (("dE", lambda v: v), ("dL", np.abs), ("tex_ratio", lambda v: v), ("px_out", lambda v: v)):
        pv, nv = f(p[col].to_numpy(float)), f(n[col].to_numpy(float))
        pv, nv = pv[np.isfinite(pv)], nv[np.isfinite(nv)]
        if not len(pv) or not len(nv):
            continue
        q = [float(np.quantile(pv, 1 - r)) for r in (0.5, 0.8)]
        over = [float((nv >= t).mean()) for t in q]
        label = "|dL|" if col == "dL" else col
        lines.append(f"    {label:9s} AUC {auc(pv, nv):.3f}  겹침 {overlap(pv, nv):.2f}  중앙값 정답 {np.median(pv):.2f} / 가짜 {np.median(nv):.2f}"
                     f"  | 정답 50% 잡는 임계값 {q[0]:.2f}에서 가짜 {over[0]:.1%}, 80% 잡는 {q[1]:.2f}에서 가짜 {over[1]:.1%}")
    return lines


def save_figures(figs, names, path_prefix):
    """대표 사진(figs: 이름 → 측정 항목)마다 사진(ROI 상단선) · 지금 마스크(초록) · 버려진 노면 색 픽셀(자홍)과 시드 상자(노랑)를 나란히 저장한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = C.FONT_BY_PLATFORM.get(sys.platform, C.FONT_DEFAULT)
    saved = []
    for rank, (name, why) in enumerate(names, start=1):
        f = figure_data(figs[name])
        small, y0, road, disc = f["small"], f["y0"], f["road"], f["discarded"]
        top_img = small.copy()
        cv.line(top_img, (0, y0), (small.shape[1] - 1, y0), (0, 255, 255), 2)
        roi = small[y0:y0 + road.shape[0]]
        m = roi.copy()
        m[road] = (0.5 * m[road] + 0.5 * np.array((0, 255, 0))).astype(np.uint8)
        d = roi.copy()
        d[road] = (0.6 * d[road] + 0.4 * np.array((0, 255, 0))).astype(np.uint8)
        d[disc] = (255, 0, 255)
        x0, s0, x1, s1 = f["seed"]
        for img in (m, d):
            cv.rectangle(img, (x0, s0), (x1 - 1, s1 - 1), (0, 255, 255), 2)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, img, title in zip(axes, (top_img, m, d), ("사진(노랑 = ROI 상단 경계)", "지금 노면 마스크(초록), 시드 영역(노랑)",
                                                          "버려진 노면 색 픽셀(자홍)")):
            ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"{name} — {why}", fontsize=11)
        fig.tight_layout()
        out = Path(f"{path_prefix}{rank:02d}_{Path(name).stem}.png")
        fig.savefig(out, dpi=80)
        plt.close(fig)
        saved.append(out.as_posix())
    return saved


def pct(v):
    return f"{v:.1%}" if np.isfinite(v) else "-"


def summary_a(A, lines):
    """측정 A 요약: 사진 전체 노면 비율이 높은데 시작 경계에 머문 사진 수와, 첫 구간에서 걸린 조건."""
    lines.append("== A. ROI 상단 경계 ==")
    for s in ("dev", "39"):
        a = A[A.set == s]
        lines.append(f"[{s}] 사진 {len(a)}장, 시작 경계에 머문 사진 {int((a.top == a.top_start).sum())}장, "
                     f"정답 상자 {int(a.n_gt.sum())}개 중 중심이 ROI 위 {int(a.gt_above.sum())}개(포트홀 {int(a.pothole_above.sum())}개)")
        for v in FULL_ROAD_LEVELS:
            st = a[(a.full_road >= v) & (a.top == a.top_start) & (a.top_start == 0.5)]
            lines.append(f"  사진 전체 노면 ≥ {v}: {int((a.full_road >= v).sum())}장, 그중 경계 0.5에 머문 사진 {len(st)}장"
                         f" (정답 상자 중심 ROI 위 {int(st.gt_above.sum())}개)")
            if len(st):
                by = st.groupby("source").size().to_dict()
                lines.append("    출처별: " + ", ".join(f"{k} {n}" for k, n in sorted(by.items())))
                road_only = int((~st.road1_ok & st.tex1_ok).sum())
                tex_only = int((st.road1_ok & ~st.tex1_ok).sum())
                both = int((~st.road1_ok & ~st.tex1_ok).sum())
                lines.append(f"    첫 구간에서 걸린 조건: 노면 비율만 {road_only}, 질감 비만 {tex_only}, 둘 다 {both}"
                             f" | 노면 비율 중앙값 {st.road1.median():.3f}, 질감 비 중앙값 {st.tex1.median():.3f}")
                lines.append(f"    질감 비를 바로 아래 구간과 비교하면 통과: {int(st.tex1_below_ok.sum())}장"
                             f"(질감에서 걸린 {int((~st.tex1_ok).sum())}장 중 {int((~st.tex1_ok & st.tex1_below_ok).sum())}장),"
                             f" 아래 구간 비 중앙값 {st.tex1_below.median():.3f}")
                if s == "39":
                    lines.append("    이름: " + ", ".join(f"{r.image}(노면 {r.full_road:.2f}, 첫 구간 노면 {r.road1:.2f}·질감 {r.tex1:.2f}·아래 비교 {r.tex1_below:.2f})"
                                                       for r in st.itertuples()))
        lines.append(f"  사진 전체 노면 마스크의 밝기 허용 폭(±L, 0~255): 중앙값 {a.l_range_full.median():.0f}, ≥ 128(밝기 조건이 사실상 꺼짐) "
                     f"{int((a.l_range_full >= 128).sum())}장, 그중 전체 노면 ≥ 0.9 {int(((a.l_range_full >= 128) & (a.full_road >= 0.9)).sum())}장"
                     f" | 전체 노면 ≥ 0.9인 사진의 허용 폭 중앙값 {a[a.full_road >= 0.9].l_range_full.median():.0f}")
        lines.append(f"  설계안 ① 전체 노면 ≥ 0.8 → 경계 0: {int((a.top_alt1_80 < a.top).sum())}장 바뀜, ≥ 0.9: {int((a.top_alt1_90 < a.top).sum())}장 바뀜")
        lines.append(f"  설계안 ② 질감을 바로 아래 구간과 비교: 경계가 지금보다 위로 {int((a.top_alt2 < a.top).sum())}장, "
                     f"아래로 {int((a.top_alt2 > a.top).sum())}장")
    if (A.set == "39").any():
        a = A[A.set == "39"]
        lines.append("  39장 사진별(시작 → 지금, 전체 노면): " + ", ".join(f"{r.image} {r.top_start}→{r.top}({r.full_road:.2f})" for r in a.itertuples()))


def summary_b(B, lines):
    """측정 B 요약: 포트홀·균열 정답과 가짜 상자의 밝기 차 방향, ΔE 분리, 밝은 표시 보정이 메운 비율."""
    lines.append("== B. 포트홀 밝기와 색 (L* 0~100, ΔE76) ==")
    for s in ("dev", "39"):
        b = B[B.set == s]
        for k in ("pothole", "crack", "fake_pothole", "fake_crack"):
            r = b[(b.kind == k) & b.ok]
            tot = int((b.kind == k).sum())
            if not tot:
                continue
            dl = r.dL.to_numpy(float)
            lines.append(f"[{s}] {k}: {tot}개 중 고리 측정 {len(r)}개 | 밝음(ΔL ≥ {SMALL_DL:g}) {pct((dl >= SMALL_DL).mean())}, "
                         f"어두움(ΔL ≤ -{SMALL_DL:g}) {pct((dl <= -SMALL_DL).mean())}, 차이 작음 {pct((np.abs(dl) < SMALL_DL).mean())}"
                         f" | ΔL > 0 {pct((dl > 0).mean())} | ΔL 중앙값 {np.median(dl):.2f}, ΔE 중앙값 {r.dE.median():.2f},"
                         f" 질감 비 중앙값 {r.tex_ratio.median():.2f}")
            if "px_out" in r:
                lines.append(f"    픽셀 단위: 고리와 다른 픽셀 중앙값 {r.px_out.median():.1%}(밝은 쪽 {r.px_out_bright.median():.1%}·어두운 쪽 {r.px_out_dark.median():.1%})"
                             f" | 다른 픽셀이 밝은 쪽이 더 많은 상자 {pct((r.px_out_bright > r.px_out_dark).mean())},"
                             f" 다른 픽셀 ≥ 20%이면서 밝은 쪽이 더 많은 상자 {pct(((r.px_out >= 0.2) & (r.px_out_bright > r.px_out_dark)).mean())},"
                             f" 어두운 쪽이 더 많은 상자 {pct(((r.px_out >= 0.2) & (r.px_out_dark >= r.px_out_bright)).mean())}")
            if k == "pothole" and "fill_ratio" in r:
                fr = r[r.in_band].fill_ratio.to_numpy(float)
                if len(fr):
                    lines.append(f"    밝은 표시 보정이 메운 비율(ROI 안 상자 {len(fr)}개): 평균 {fr.mean():.1%}, 중앙값 {np.median(fr):.1%}, "
                                 f"25% 이상 {pct((fr >= 0.25).mean())}, 50% 이상 {pct((fr >= 0.5).mean())}"
                                 f" | 밝은 포트홀만: 평균 {pct(r[r.in_band & (r.dL >= SMALL_DL)].fill_ratio.mean())}")
        fk = b[(b.kind.str.startswith("fake")) & b.ok]
        if len(fk):
            lines.append(f"  가짜 대조군 |ΔL| 50·75·90백분위: {np.quantile(np.abs(fk.dL), [0.5, 0.75, 0.9]).round(2).tolist()}"
                         f" (같은 사진 {int((fk.fake_from == 'same').sum())}개, 무손상 사진 {int((fk.fake_from != 'same').sum())}개)")
        lines += sep_lines(b, s, "pothole", "fake_pothole") + sep_lines(b, s, "crack", "fake_crack") + sep_lines(b, s, "pothole", "crack")
        if s == "39":
            r = b[(b.kind == "pothole")]
            lines.append("  39장 포트홀 상자별: " + "; ".join(
                f"{x.image}#{x.idx} " + (f"ΔL {x.dL:+.1f} ΔE {x.dE:.1f} 질감 {x.tex_ratio:.2f} 메움 {x.fill_ratio:.0%}"
                                         f" 다른 픽셀 {x.px_out:.0%}(밝 {x.px_out_bright:.0%}·어 {x.px_out_dark:.0%})" if x.ok else "고리 없음")
                for x in r.itertuples()))


def summary_c(Cm, lines):
    """측정 C 요약: 오염된 시드와 버려진 노면 색 픽셀이 큰 사진."""
    lines.append("== C. 노면 마스크 ==")
    for s in ("dev", "39"):
        c = Cm[Cm.set == s]
        lines.append(f"[{s}] 사진 {len(c)}장 | 시드 평균: 밝은 표시 {c.seed_bright.mean():.1%}, 높은 채도 {c.seed_chroma.mean():.1%}, "
                     f"마스크 밖 {c.seed_off_mask.mean():.1%} | 셋 중 하나라도 {SEED_BAD_RATIO:.0%} 이상인 사진 {int((c.seed_bad >= SEED_BAD_RATIO).sum())}장"
                     f" (밝은 표시 ≥ 10% {int((c.seed_bright >= 0.1).sum())}, 채도 ≥ 10% {int((c.seed_chroma >= 0.1).sum())},"
                     f" 마스크 밖 ≥ 20% {int((c.seed_off_mask >= 0.2).sum())})")
        lines.append(f"  노면 마스크 ROI의 {c.road_frac.mean():.1%}, 노면 색(연결 전) {c.roadcolor_frac.mean():.1%}, 버려짐 평균 {c.discarded_frac.mean():.1%}"
                     f" / 중앙값 {c.discarded_frac.median():.1%} | ≥ {DISCARD_BIG:.0%}인 사진 {int((c.discarded_frac >= DISCARD_BIG).sum())}장,"
                     f" ≥ 20% {int((c.discarded_frac >= 0.2).sum())}장")
        lines.append(f"  ROI 노면 마스크의 밝기 허용 폭(±L): 중앙값 {c.l_range.median():.0f}, 90백분위 {c.l_range.quantile(0.9):.0f}, ≥ 128 {int((c.l_range >= 128).sum())}장")
        lines.append(f"  띠 안 정답 중심이 마스크 안: {int(c.gt_in_mask.sum())}/{int(c.gt_in_band.sum())} = {c.gt_in_mask.sum() / max(1, c.gt_in_band.sum()):.2%}")
        top = c.sort_values("seed_bad", ascending=False).head(8)
        lines.append("  시드 오염 상위: " + ", ".join(f"{r.image}({r.seed_bad:.0%}: 밝 {r.seed_bright:.0%}·채 {r.seed_chroma:.0%}·밖 {r.seed_off_mask:.0%})"
                                                for r in top.itertuples()))
        top = c.sort_values("discarded_frac", ascending=False).head(8)
        lines.append("  버려진 노면 색 상위: " + ", ".join(f"{r.image}({r.discarded_frac:.0%})" for r in top.itertuples()))


def pick_figures(Cm):
    """확인 그림 대표 사진: 버려진 노면 색 상위 4장(개발셋), 시드 오염 상위 3장(개발셋), 39장의 버려짐·오염 상위 각 1~2장 (겹치면 다음)."""
    chosen = []

    def take(frame, col, n, why):
        for r in frame.sort_values(col, ascending=False).itertuples():
            if n == 0:
                break
            if r.image not in [c[0] for c in chosen]:
                chosen.append((r.image, f"{why} {getattr(r, col):.0%}"))
                n -= 1
    dev, own = Cm[Cm.set == "dev"], Cm[Cm.set == "39"]
    take(dev, "discarded_frac", 4, "버려진 노면 색(ROI 대비)")
    take(dev, "seed_bad", 3, "시드 오염")
    take(own, "discarded_frac", 2, "버려진 노면 색(ROI 대비)")
    take(own, "seed_bad", 1, "시드 오염")
    return chosen[:N_FIGURES]


def candidate_texture(region, tex, road):
    """후보 하나의 질감 비 = 후보 영역(윤곽 안쪽까지 채움) 7×7 지역 표준편차 중앙값 ÷ 바깥 고리(폭 ring_width, 노면 마스크 안) 중앙값.
    tex·road는 ROI 좌표. 고리 픽셀이 RING_MIN_N보다 적거나 바깥 중앙값이 0이면 NaN."""
    x, y, w, h = region["bbox"]
    d = ring_width(w, h)
    H, W = tex.shape
    X0, Y0, X1, Y1 = max(0, x - d), max(0, y - d), min(W, x + w + d), min(H, y + h + d)
    inside = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
    cv.drawContours(inside, [region["contour"]], -1, 255, -1, offset=(-X0, -Y0))
    grown = cv.dilate(inside, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))) > 0
    ring = grown & (inside == 0) & road[Y0:Y1, X0:X1]
    if np.count_nonzero(ring) < RING_MIN_N or not inside.any():
        return np.nan
    out = float(np.median(tex[Y0:Y1, X0:X1][ring]))
    return float(np.median(tex[Y0:Y1, X0:X1][inside > 0])) / out if out > 0 else np.nan


def measure_candidates(items):
    """(2') 실제 포트홀 후보(점수 임계값을 끈 dark 검출, 전처리 미적용, 노면 밖 후보 제거 뒤)마다 질감 비와 포트홀 점수를 재고,
    중심이 포트홀 정답 상자 안인지(검출 성공) 적는다. 비교용으로 포트홀 정답 안에 중심이 든 균열 후보(분류만 균열로 된 영역)도 남긴다.
    → DataFrame(image, kind, hit_pothole, score, tex_ratio, area, circ_inv)"""
    rows = []
    backup = C.USE_SCORE_TAU
    C.USE_SCORE_TAU = False
    try:
        for n, (name, path, gt_path, top0, bottom) in enumerate(items, start=1):
            gt = evaluate.load_gt(str(gt_path)) or []
            pot_gt = [(x, y, w, h) for x, y, w, h, k in gt if k == "pothole"]
            out = pipeline.run_pipeline(io_utils.load_image(path), top0, bottom, False)
            tex = pipeline.local_std(out["gray"], C.AUTO_BAND_TEX_KSIZE)
            road = out["road"] > 0
            for kind in ("pothole", "crack"):
                for r in out[f"{kind}s"]:
                    x, y, w, h = r["bbox"]
                    hit = bool(evaluate.count_center_hits([(x, y + out["y0"], w, h)], pot_gt)["tp"]) if pot_gt else False
                    if kind == "crack" and not hit:
                        continue
                    rows.append({"image": name, "kind": kind, "hit_pothole": hit, "score": r.get(f"score_{kind}", r["score"]),
                                 "score_pothole": r.get("score_pothole", np.nan), "tex_ratio": candidate_texture(r, tex, road),
                                 "area": r["area"], "circ_inv": r["circ_inv"]})
            if n % 100 == 0:
                print(f"후보 {n}/{len(items)}")
    finally:
        C.USE_SCORE_TAU = backup
    return pd.DataFrame(rows)


def candidate_lines(df):
    """(2') 요약: 포트홀 후보 중 검출 성공 vs 오검출의 질감 비·점수·점수 × 질감 비 AUC(표본 수와 함께)."""
    pot = df[df.kind == "pothole"]
    pos, neg = pot[pot.hit_pothole], pot[~pot.hit_pothole]
    lines = [f"(2') 포트홀 후보 {len(pot)}개(점수 임계값 끔, 전처리 미적용): 검출 성공 {len(pos)}개, 오검출 {len(neg)}개"]
    for col, label in (("tex_ratio", "질감 비"), ("score", "포트홀 점수")):
        pv, nv = pos[col].to_numpy(float), neg[col].to_numpy(float)
        pv, nv = pv[np.isfinite(pv)], nv[np.isfinite(nv)]
        lines.append(f"  {label}: AUC {auc(pv, nv):.3f} (성공 {len(pv)}·오검출 {len(nv)}개), 중앙값 성공 {np.median(pv) if len(pv) else np.nan:.2f} / "
                     f"오검출 {np.median(nv) if len(nv) else np.nan:.2f}")
    prod = lambda f: (f.score * f.tex_ratio).to_numpy(float)
    pv, nv = prod(pos), prod(neg)
    lines.append(f"  점수 × 질감 비: AUC {auc(pv[np.isfinite(pv)], nv[np.isfinite(nv)]):.3f}")
    cr = df[(df.kind == "crack") & df.hit_pothole]
    allpos = pd.concat([pos, cr])
    pv, nv = allpos.tex_ratio.to_numpy(float), neg.tex_ratio.to_numpy(float)
    lines.append(f"  포트홀 정답 안의 모든 어두운 후보(균열로 분류된 {len(cr)}개 포함 {len(allpos)}개) vs 포트홀 오검출: 질감 비 AUC "
                 f"{auc(pv[np.isfinite(pv)], nv[np.isfinite(nv)]):.3f}, 중앙값 {np.nanmedian(pv):.2f} / {np.nanmedian(nv):.2f}")
    if len(pos):
        lines.append("  검출 성공 포트홀 후보: " + ", ".join(f"{r.image}(점수 {r.score:.2f}, 질감 {r.tex_ratio:.2f})" for r in pos.itertuples()))
    return lines


STUCK_SOURCES = ("China_Drone", "China_MotorBike")   # (1) 재측정 대상: 사진 전체 노면 ≥ 0.8인데 시작 경계에 머문 드론·오토바이 사진


def stuck_check(sets):
    """(1) 재측정: A_roi_dev.csv(지금 config로 잰 기준)에서 사진 전체 노면 ≥ 0.8이면서 0.5에 머문 드론·오토바이 사진마다, 지금 실행의
    config(--set으로 바꾼 값 포함)로 ROI 상단 경계와 ROI 위로 빠진 정답 수를 다시 잰다. → (DataFrame, 요약 줄 list)"""
    base = pd.read_csv(OUT_DIR / "A_roi_dev.csv", encoding="utf-8-sig")
    base = base[(base.full_road >= FULL_ROAD_LEVELS[0]) & (base.top == base.top_start) & (base.top_start == 0.5)
                & base.source.isin(STUCK_SOURCES)]
    items = {it[0]: it for it in sets["dev"]}
    rows = []
    for r in base.itertuples():
        name, path, gt_path, top0, bottom = items[r.image]
        small = io_utils.resize_width(io_utils.load_image(path))
        top = pipeline.auto_band_top(small, top0) if C.AUTO_BAND_TOP else top0
        y0 = int(small.shape[0] * top)
        gt = evaluate.load_gt(str(gt_path)) or []
        rows.append({"image": name, "top_before": r.top, "top": top, "gt_above_before": r.gt_above,
                     "gt_above": sum(1 for x, y, w, h, k in gt if y + h / 2 < y0)})
    df = pd.DataFrame(rows)
    lines = [f"(1) 재측정: 드론·오토바이 {len(df)}장(ROI 위 정답 {int(df.gt_above_before.sum())}개) → 넓어진 사진 {int((df.top < df.top_before).sum())}장, "
             f"ROI 위 정답 {int(df.gt_above.sum())}개(회복 {int(df.gt_above_before.sum() - df.gt_above.sum())}개)",
             "  " + ", ".join(f"{r.image} {r.top_before}→{r.top}(위 정답 {r.gt_above_before}→{r.gt_above})" for r in df.itertuples())]
    return df, lines


def main(argv=None):
    """개발셋 652장과 39장을 재서 results/measure_v5/에 CSV·요약·확인 그림을 저장하고 요약을 출력한다."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=0, help="셋마다 앞의 N장만(확인용, 0 = 전부)")
    parser.add_argument("--from-csv", action="store_true", help="측정은 건너뛰고 저장된 CSV로 요약·그림만 다시 만든다")
    parser.add_argument("--candidates", action="store_true",
                        help="(2') 개발셋 실제 포트홀 후보의 질감 비만 잰다 → B2_pothole_candidates.csv, summary_candidates.txt")
    parser.add_argument("--stuck", action="store_true", help="(1) 재측정: 0.5에 머문 드론·오토바이 사진만 지금 config로 다시 잰다")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="이번 실행에만 바꿀 config 값(여러 번). 값은 파이썬 리터럴(예: ROAD_L_RANGE_MAX=64, ROAD_COLOR_DIST='mahal')")
    args = parser.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for item in args.set:
        key, value = item.split("=", 1)
        if not hasattr(C, key):
            parser.error(f"config에 없는 이름: {key}")
        setattr(C, key, ast.literal_eval(value))
    if args.stuck:
        df, lines = stuck_check({"dev": dev_items()})
        tag = "_".join(s.replace("=", "-").replace("'", "") for s in args.set) or "config"
        df.to_csv(OUT_DIR / f"A_stuck_{tag}.csv", index=False, encoding="utf-8-sig")
        print(f"설정: {', '.join(args.set) or 'config 그대로'}")
        for line in lines:
            print(line)
        return
    if args.candidates:
        items = dev_items()[:args.limit] if args.limit else dev_items()
        df = measure_candidates(items)
        df.to_csv(OUT_DIR / "B2_pothole_candidates.csv", index=False, encoding="utf-8-sig")
        text = "\n".join(candidate_lines(df))
        (OUT_DIR / "summary_candidates.txt").write_text(text + "\n", encoding="utf-8")
        print(text)
        return
    sets = {"dev": dev_items(), "39": own_items()}
    if args.limit:
        sets = {k: v[:args.limit] for k, v in sets.items()}
    figs = {item[0]: item for items in sets.values() for item in items}
    if args.from_csv:
        read = lambda kind: pd.concat([pd.read_csv(OUT_DIR / f"{kind}_{s}.csv", encoding="utf-8-sig", dtype={"set": str}) for s in sets])
        A, B, Cm = read("A_roi"), read("B_boxes"), read("C_mask")
        fake = B[B.kind.str.startswith("fake")]
        n_pending, n_none = int((fake.fake_from != "same").sum()), "-"
    else:
        A, B, Cm, pending = [], [], [], []
        for set_name, items in sets.items():
            for n, item in enumerate(items, start=1):
                a, b, c = analyze_image(item, set_name, pending)
                A.append(a), B.extend(b), Cm.append(c)
                if n % 100 == 0:
                    print(f"{set_name} {n}/{len(items)}")
        none_dev = [it for it in sets["dev"] if not (evaluate.load_gt(str(it[2])) or [])]
        n_pending, n_none = len(pending), len(none_dev)
        B.extend(fill_pending(pending, none_dev))
        A, B, Cm = pd.DataFrame(A), pd.DataFrame(B), pd.DataFrame(Cm)
        for s in sets:
            A[A.set == s].to_csv(OUT_DIR / f"A_roi_{s}.csv", index=False, encoding="utf-8-sig")
            B[B.set == s].to_csv(OUT_DIR / f"B_boxes_{s}.csv", index=False, encoding="utf-8-sig")
            Cm[Cm.set == s].to_csv(OUT_DIR / f"C_mask_{s}.csv", index=False, encoding="utf-8-sig")
    lines = [f"측정 v5 (config 그대로, 개발셋 {int((A.set == 'dev').sum())}장 + 39장 {int((A.set == '39').sum())}장, 시험셋 사용 안 함)",
             f"가짜 대조군: 같은 사진에서 자리를 못 찾은 {n_pending}개를 무손상 개발셋 사진({n_none}장)에 놓음"]
    lines.append(f"src 대조: 노면 마스크 다시 계산이 road_mask와 다른 사진 {int((~Cm.check_mask).sum())}장, "
                 f"첫 구간 조건과 auto_band_top 결과가 어긋난 사진 {int((~Cm.check_band).sum())}장")
    summary_a(A, lines)
    summary_b(B, lines)
    summary_c(Cm, lines)
    chosen = pick_figures(Cm)
    saved = save_figures(figs, chosen, (OUT_DIR / "fig_").as_posix())
    lines.append("확인 그림: " + ", ".join(saved))
    text = "\n".join(lines)
    (OUT_DIR / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
