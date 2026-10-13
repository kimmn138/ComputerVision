"""(파트 C, 10/14) 노면 마스크의 두 결함을 고치는 변형(A1·A2·A3·B1·B2와 조합)을 개발셋 652장 + 39장에서 노면 기준으로 비교한다.
사용법(맨 위 폴더에서):
  python tools/road_v6.py --calib            A2·A3 기준값을 정한다(개발셋, 지금 config 시드 상자) → results/road_v6/calib.json·calib.txt
  python tools/road_v6.py [--limit N]        변형마다 사진별 숫자 → results/road_v6/per_image.csv, 조합별 표 summary.csv·summary.txt
  python tools/road_v6.py --figs             확인 그림(39장 전부 + 개발셋 시점별 20장) → results/road_v6/fig_<셋>_<사진>.png
결함 A: 시드 상자가 노면이 아닌 물체(차·흰 선·그림자·풀)에 찍히면 노면 색을 잘못 정한다.
결함 B: 시드와 이어진 곳으로만 퍼져 흰 선에 막히면 건너편 노면을 버린다.
변형(모두 이 도구 안에서만, src는 고치지 않음. 변형이 없으면 io_utils.road_mask와 픽셀 하나까지 같음을 사진마다 대조):
- A1 여러 점 다수결: 노면 띠 아래쪽 격자 점마다 작은 창의 Lab 중앙값을 내고, 다른 점의 노면 색 기준(a·b 거리 < ROAD_CHROMA_MAX,
  밝기 차 < ROAD_L_K × (1.4826 × MAD + 2)) 안에 드는 점이 가장 많은 점의 묶음을 시드로 쓴다.
- A2 노면다운지 검사: 높은 채도(calib의 채도 기준), 밝은 표시(검출기와 같은 기준: 거친 배경 + DARK_BRIGHT_DELTA보다 밝음),
  노란 선 색(ROAD_LANE_YELLOW_B 규칙)인 픽셀·점을 시드에서 뺀다. A2b는 채도 기준을 뺀 판(빨간·푸른 아스팔트 대비).
- A3 밝기 허용 폭 상한: ±ROAD_L_K × (1.4826 × MAD + 2)를 calib의 상한으로 자른다.
- B1 연결 조건 없애기: 노면 색 픽셀 덩어리 가운데 시드와 이어진 것 + ROI 넓이의 B1_BIG 이상 + (B1_SMALL 이상이고 아래 B1_BOTTOM에 닿은 것)을 남긴다.
- B2 차선을 노면으로: 밝고 가는 흰색·노란색 줄(io_utils._lane_pixels) 덩어리 중 가늘고 길며 둘레의 B2_RING_ROAD 이상이 노면 색인 것만 노면에 더한다.
자동 ROI(사진 전체 노면 마스크)와 ROI 노면 마스크 모두 같은 변형을 쓴다: pipeline.run_pipeline을 그대로 부르되 io_utils.road_mask만
변형으로 바꿔 끼운다(마스크를 따로 다시 만들지 않아 파이프라인과 어긋나지 않음). 시험셋 사진은 열지 않는다.
노면 정답은 없다. 숫자(시드 오염, 차선 양쪽 칠함, 마스크 안 정답, ROI 위 정답, 위쪽 번짐, ROI가 넓어진 사진)와 그림으로 본다.
오검출(중심 기준, 점수 임계값 config 그대로, 전처리 미적용)은 함께 적기만 한다."""
import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from contextlib import contextmanager
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src import evaluate, io_utils, pipeline  # noqa: E402
from tools import evaluate_train as et  # noqa: E402
from tools import measure_v5 as mv  # noqa: E402

OUT_DIR = Path("results/road_v6")
VARIANTS = {"now": (), "A1": ("A1",), "A2": ("A2",), "A3": ("A3",), "B1": ("B1",), "B2": ("B2",),
            "A1A2": ("A1", "A2"), "A1A2A3": ("A1", "A2", "A3"), "B1B2": ("B1", "B2"), "all": ("A1", "A2", "A3", "B1", "B2"),
            # 효과가 큰 조합(첫 실행 뒤): A1은 시드 오염을 줄이지만 밝기 허용 폭 ≥ 128을 늘려(71 → 111장) A1 없는 조합과 B2만 더한 조합을 견줌
            "A2A3": ("A2", "A3"), "A2A3B1B2": ("A2", "A3", "B1", "B2"), "A1A2A3B2": ("A1", "A2", "A3", "B2"),
            # A2b = A2에서 채도 기준을 뺌(밝은 표시·노란 선만). 빨간·푸른 아스팔트(Czech_001106, United_States_003320)에서 A2가 노면을 빼 실패해서 더함
            "A2b": ("A2b",), "A2bA3B1B2": ("A2b", "A3", "B1", "B2"), "A1A2bA3B1B2": ("A1", "A2b", "A3", "B1", "B2")}
GRID = (8, 4)                    # 점 수 (가로, 세로), A1. 띠 아래쪽을 고르게 덮음
GRID_BOX = (0.05, 0.35, 0.95, 0.90)   # 비율 (가로 시작, 세로 시작, 가로 끝, 세로 끝), A1 점을 놓을 범위. 맨 아래 0.1(보닛)은 뺌
PATCH = 15                       # px, A1 점 하나의 창 한 변. 골재 결(σ 3 흐림 뒤)보다 넓고 차선 폭(8~40px)과 비슷해 차선 위 점은 차선 색이 됨
A2_MIN_KEEP = 0.25               # 비율, A2로 시드 상자에서 빼고 남은 픽셀이 상자의 이 비율보다 적으면 빼지 않음(시드가 사라지지 않게)
A2_POINT_BAD = 0.5               # 비율, A1 점 창에서 밝은 표시·높은 채도·노란 선 픽셀이 이 비율 이상이면 그 점을 뺌
B1_SMALL = 0.005                 # ROI 넓이 비율, B1에서 이보다 작은 덩어리는 버림
B1_BIG = 0.05                    # ROI 넓이 비율, B1에서 이 이상이면 위치와 상관없이 남김(10/13 설계안 (3)②의 5%)
B1_BOTTOM = 0.25                 # 높이 비율, B1에서 ROI 아래 이 범위에 닿은 덩어리는 B1_SMALL 이상이면 남김
B2_MIN_AREA = 30                 # px, B2 차선 덩어리 최소 넓이
B2_ELONG = 3.0                   # 배, B2 차선 덩어리 긴 변 ÷ 짧은 변 하한(가늘고 김)
B2_RING = 9                      # px, B2 둘레 고리 폭(차선 양옆 노면을 봄)
B2_RING_ROAD = 0.5               # 비율, B2 둘레 고리에서 노면 색 픽셀이 이 이상이어야 노면 위의 차선으로 봄
B2_EDGE = 2 * C.ROAD_BLUR_SIGMA  # px, B2로 더한 차선을 이만큼 넓힘. 흐림 σ의 두 배(가장자리 번짐 범위)
LANE_MIN_LEN = 40                # px, 차선 지표: 긴 변이 이 이상인 가늘고 긴(B2_ELONG 이상) 밝은 흰·노란 줄
LANE_SIDE_IN, LANE_SIDE_OUT = 4, 18   # px, 차선 지표: 줄에서 이만큼 떨어진 양옆 띠(4~18px)
LANE_SIDE_MIN_N = 30             # 개, 한쪽 띠 픽셀이 이보다 적으면 그 줄은 셈하지 않음
LANE_SIDE_DL = 40                # 밝기 단계(Lab L 0~255), 양옆 노면 밝기 차가 이보다 크면 같은 노면으로 보지 않음(그늘·햇빛은 허용)
LANE_PAINT = 0.5                 # 비율, 한쪽 띠의 마스크 비율이 이 이상이면 칠해짐
SPREAD_ROWS, SPREAD_MIN = 0.3, 0.3    # 사진 위 30% 행에서 마스크가 30% 이상이면 '위쪽까지 번짐'
DRONE_SOURCES = ("China_Drone", "China_MotorBike")
N_FIG_DRONE, N_FIG_CAR = 6, 14


def view_of(name):
    """사진 시점 묶음: 드론·오토바이 / 직접 촬영(own) / 차량 시점(나머지 제공·개발셋)."""
    src = et.source_of(name)
    if src in DRONE_SOURCES:
        return "drone_moto"
    return "own" if name.startswith("own_") else "car"


def load_calib():
    path = OUT_DIR / "calib.json"
    if not path.exists():
        raise SystemExit("먼저 python tools/road_v6.py --calib 로 A2·A3 기준값을 정한다")
    return json.loads(path.read_text(encoding="utf-8"))


def not_road_pixels(bgr, lab, calib, use_chroma=True):
    """A2: 시드로 쓰면 안 되는 픽셀(bool). 높은 채도 = 흐린 Lab a·b의 무채색(128, 128) 거리 > calib 채도 기준,
    밝은 표시 = 검출기 1단계와 같은 기준(팽창 전), 노란 선 색 = io_utils._lane_pixels와 같은 노랑 규칙."""
    da, db = lab[:, :, 1] - 128, lab[:, :, 2] - 128
    chroma = (np.sqrt(da ** 2 + db ** 2) > calib["chroma_max"]) if use_chroma else np.zeros(lab.shape[:2], bool)
    bright = mv.bright_fill(cv.cvtColor(bgr, cv.COLOR_BGR2GRAY), dilate=False)
    yellow = (db >= C.ROAD_LANE_YELLOW_B) & (db >= 2 * np.abs(da))
    return chroma | bright | yellow


def grid_seed(lab, bad, opts, h, w):
    """A1: 격자 점마다 PATCH 창의 Lab 중앙값·L MAD를 내고, 다른 점의 노면 색 기준 안에 드는 점이 가장 많은 점의 묶음 창들을 시드로.
    A2가 켜져 있으면 나쁜 픽셀이 A2_POINT_BAD 이상인 점은 처음부터 뺀다. → (시드 bool, 점 list[(x, y, 묶음에 듦)]) 또는 (None, [])."""
    gx, gy = GRID
    fx0, fy0, fx1, fy1 = GRID_BOX
    r = PATCH // 2
    pts = []
    for j in range(gy):
        for i in range(gx):
            x = int(w * (fx0 + (fx1 - fx0) * (i + 0.5) / gx))
            y = int(h * (fy0 + (fy1 - fy0) * (j + 0.5) / gy))
            sl = (slice(max(0, y - r), y + r + 1), slice(max(0, x - r), x + r + 1))
            if bad is not None and bad[sl].mean() >= A2_POINT_BAD:
                continue
            p = lab[sl].reshape(-1, 3)
            med = np.median(p, axis=0)
            mad = float(np.median(np.abs(p[:, 0] - med[0])))
            pts.append((x, y, sl, med, C.ROAD_L_K * (1.4826 * mad + 2.0)))
    if not pts:
        return None, []
    meds = np.array([p[3] for p in pts])
    tol = np.array([p[4] for p in pts])
    ab = np.sqrt((meds[:, None, 1] - meds[None, :, 1]) ** 2 + (meds[:, None, 2] - meds[None, :, 2]) ** 2)
    member = (ab < C.ROAD_CHROMA_MAX) & (np.abs(meds[:, None, 0] - meds[None, :, 0]) < tol[:, None])   # [i, j]: j가 i의 노면 색 기준 안
    best = int(np.argmax(member.sum(1)))        # 동점이면 앞(위·왼쪽) 점. 점 순서가 고정이라 같은 입력이면 같은 결과
    seed = np.zeros((h, w), bool)
    for k, p in enumerate(pts):
        if member[best, k]:
            seed[p[2]] = True
    return seed, [(p[0], p[1], bool(member[best, k])) for k, p in enumerate(pts)]


def lane_bridge(L, a, b, L_med, cand):
    """B2: 밝고 가는 흰·노란 줄 덩어리 가운데 가늘고 길며(긴 변 ÷ 짧은 변 ≥ B2_ELONG) 둘레 고리의 B2_RING_ROAD 이상이 노면 색인 것(bool)."""
    lane = io_utils._lane_pixels(L, a, b, L_med).astype(np.uint8)
    n, labels, stats, _ = cv.connectedComponentsWithStats(lane, connectivity=8)
    out = np.zeros(lane.shape, bool)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * B2_RING + 1, 2 * B2_RING + 1))
    for i in range(1, n):
        if stats[i, cv.CC_STAT_AREA] < B2_MIN_AREA:
            continue
        x, y, bw, bh = stats[i, :4]
        comp = labels[y:y + bh, x:x + bw] == i
        (_, _), (rw, rh), _ = cv.minAreaRect(np.column_stack(np.nonzero(comp)[::-1]).astype(np.float32))
        if max(rw, rh) < B2_ELONG * max(1.0, min(rw, rh)):
            continue
        X0, Y0 = max(0, x - B2_RING), max(0, y - B2_RING)
        X1, Y1 = min(lane.shape[1], x + bw + B2_RING), min(lane.shape[0], y + bh + B2_RING)
        local = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
        local[y - Y0:y - Y0 + bh, x - X0:x - X0 + bw] = comp
        ring = (cv.dilate(local, k) > 0) & (local == 0) & (lane[Y0:Y1, X0:X1] == 0)
        if ring.any() and cand[Y0:Y1, X0:X1][ring].mean() >= B2_RING_ROAD:
            out[Y0:Y1, X0:X1] |= local > 0
    d = 2 * B2_EDGE + 1                    # 흐림(σ ROAD_BLUR_SIGMA) 때문에 차선 가장자리에 '노면 색도 차선도 아닌' 틈이 남아 연결이 끊기지 않게 넓힘
    return cv.dilate(out.astype(np.uint8), cv.getStructuringElement(cv.MORPH_ELLIPSE, (d, d))) > 0


def keep_components(opened, seed, h):
    """B1: 시드와 이어진 덩어리 + 넓이 ≥ B1_BIG + (넓이 ≥ B1_SMALL이고 아래 B1_BOTTOM에 닿은 덩어리)."""
    n, labels, stats, _ = cv.connectedComponentsWithStats(opened.astype(np.uint8), connectivity=8)
    total = opened.size
    keep = set(np.unique(labels[seed]).tolist()) - {0}
    bottom = int(h * (1 - B1_BOTTOM))
    for i in range(1, n):
        area = stats[i, cv.CC_STAT_AREA]
        if area >= B1_BIG * total or (area >= B1_SMALL * total and stats[i, cv.CC_STAT_TOP] + stats[i, cv.CC_STAT_HEIGHT] > bottom):
            keep.add(i)
    return keep, labels


def road_mask_v6(bgr_roi, opts=(), calib=None, info=None):
    """io_utils.road_mask(보완 최종)의 식에 변형(opts ⊂ {A1, A2, A3, B1, B2})을 더한 노면 마스크(uint8 0/255). opts가 비면 road_mask와 같다.
    info(dict)를 주면 시드 픽셀(bool)·격자 점·밝기 허용 폭을 담는다. 받은 영상은 바꾸지 않는다."""
    if C.ROAD_SHADOW_L_K or C.ROAD_EDGE_DENSITY or C.ROAD_LANE_BRIDGE or C.ROAD_CLOSE_BEFORE_SEED or C.ROAD_HOOD_CUT:
        raise RuntimeError("road_mask_v6는 노면 마스크 v2 스위치가 모두 꺼진 config(보완 최종)만 다룬다")
    h, w = bgr_roi.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((h, w), np.uint8)
    lab = cv.cvtColor(cv.GaussianBlur(bgr_roi, (0, 0), C.ROAD_BLUR_SIGMA), cv.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    fx0, fy0, fx1, fy1 = C.ROAD_SEED_BOX
    x0, y0 = max(0, min(int(w * fx0), w - 1)), max(0, min(int(h * fy0), h - 1))
    x1, y1 = max(x0 + 1, min(int(w * fx1), w)), max(y0 + 1, min(int(h * fy1), h))
    a2 = "A2" in opts or "A2b" in opts
    bad = not_road_pixels(bgr_roi, lab, calib, use_chroma="A2" in opts) if a2 else None
    seed, points = (grid_seed(lab, bad, opts, h, w) if "A1" in opts else (None, []))
    if seed is None:
        seed = np.zeros((h, w), bool)
        seed[y0:y1, x0:x1] = True
        if a2:
            good = seed & ~bad
            if np.count_nonzero(good) >= A2_MIN_KEEP * np.count_nonzero(seed):
                seed = good
    L_med, a_med, b_med = (float(np.median(ch[seed])) for ch in (L, a, b))
    mad = float(np.median(np.abs(L[seed] - L_med)))
    l_thr = C.ROAD_L_K * (1.4826 * mad + 2.0)
    if "A3" in opts:
        l_thr = min(l_thr, calib["l_range_max"])
    cand = (np.sqrt((a - a_med) ** 2 + (b - b_med) ** 2) < C.ROAD_CHROMA_MAX) & (np.abs(L - L_med) < l_thr)
    if "B2" in opts:
        cand = cand | lane_bridge(L, a, b, L_med, cand)
    opened = cv.morphologyEx(cand.astype(np.uint8) * 255, cv.MORPH_OPEN,
                             cv.getStructuringElement(cv.MORPH_ELLIPSE, (C.ROAD_OPEN_KSIZE, C.ROAD_OPEN_KSIZE))) > 0
    if "B1" in opts:
        keep, labels = keep_components(opened, seed, h)
    else:
        _, labels = cv.connectedComponents(opened.astype(np.uint8), connectivity=8)
        keep = set(np.unique(labels[seed]).tolist()) - {0}
    conn = (np.isin(labels, list(keep)) if keep else opened).astype(np.uint8) * 255
    conn = cv.morphologyEx(conn, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (C.ROAD_CLOSE_KSIZE, C.ROAD_CLOSE_KSIZE)))
    out = io_utils._fill_holes(conn).astype(np.uint8)
    if info is not None:
        info.update({"seed": seed, "points": points, "l_thr": l_thr})
    return out


@contextmanager
def mask_variant(opts, calib):
    """io_utils.road_mask를 변형으로 바꿔 끼운다(pipeline.run_pipeline·auto_band_top이 모두 이것을 부름). 끝나면 되돌린다."""
    original = io_utils.road_mask
    io_utils.road_mask = lambda bgr: road_mask_v6(bgr, opts, calib)
    try:
        yield
    finally:
        io_utils.road_mask = original


def lanes_with_road(small, r0, r1):
    """차선 지표(변형과 무관, 사진마다 한 번): 시작 ROI 띠(r0~r1행) 안의 가늘고 긴 밝은 흰·노란 줄 가운데 양옆 띠(LANE_SIDE_IN~OUT px)가
    모두 낮은 채도(측정 C의 높은 채도 기준 미만)이고 서로 같은 노면 색(a·b 거리 < ROAD_CHROMA_MAX, 밝기 차 < LANE_SIDE_DL)인 줄.
    → [(한쪽 띠 bool, 다른 쪽 띠 bool)] (사진 전체 좌표, 긴 줄 순)."""
    band = small[r0:r1]
    lab = cv.cvtColor(cv.GaussianBlur(band, (0, 0), C.ROAD_BLUR_SIGMA), cv.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    lane = io_utils._lane_pixels(L, a, b, float(np.median(L))).astype(np.uint8)
    n, labels, stats, _ = cv.connectedComponentsWithStats(lane, connectivity=8)
    H, W = small.shape[:2]
    k_in = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * LANE_SIDE_IN + 1,) * 2)
    k_out = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * LANE_SIDE_OUT + 1,) * 2)
    found = []
    for i in range(1, n):
        if stats[i, cv.CC_STAT_AREA] < B2_MIN_AREA:
            continue
        comp = (labels == i).astype(np.uint8)
        ys, xs = np.nonzero(comp)
        (cx, cy), (rw, rh), _ = cv.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        long_, short = max(rw, rh), max(1.0, min(rw, rh))
        if long_ < LANE_MIN_LEN or long_ < B2_ELONG * short:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float64)
        _, _, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
        normal = np.array([-vt[0, 1], vt[0, 0]])
        ring = (cv.dilate(comp, k_out) > 0) & ~(cv.dilate(comp, k_in) > 0) & (lane == 0)
        ry, rx = np.nonzero(ring)
        side = ((np.column_stack([rx, ry]) - pts.mean(0)) @ normal) > 0
        sides = []
        for s in (side, ~side):
            if s.sum() < LANE_SIDE_MIN_N:
                break
            m = np.median(lab[ry[s], rx[s]], axis=0)
            mask = np.zeros((H, W), bool)
            mask[ry[s] + r0, rx[s]] = True
            sides.append((m, mask))
        if len(sides) < 2:
            continue
        (m1, s1), (m2, s2) = sides
        low = all(np.hypot(m[1] - 128, m[2] - 128) <= mv.HIGH_CHROMA for m in (m1, m2))
        same = np.hypot(m1[1] - m2[1], m1[2] - m2[2]) < C.ROAD_CHROMA_MAX and abs(m1[0] - m2[0]) < LANE_SIDE_DL
        if low and same:
            found.append((long_, s1, s2))
    return [(s1, s2) for _, s1, s2 in sorted(found, key=lambda t: -t[0])]


def items_all(limit=None):
    """개발셋 652장 + 39장의 (셋, 항목). 항목은 measure_v5와 같은 (이름, 경로, 정답 경로, 시작 경계, 아래 경계)."""
    out = [("dev", it) for it in mv.dev_items()] + [("39", it) for it in mv.own_items()]
    return out[:limit] if limit else out


def calibrate():
    """A2 채도 기준·A3 밝기 허용 폭 상한을 개발셋 '깨끗한 시드' 사진(측정 C 오염 합 < 5%, results/measure_v5/C_mask_dev.csv,
    보완 최종 config)에서 정한다: 채도 기준 = 시드 상자 픽셀(흐린 Lab) 채도의 99.5백분위, 밝기 상한 = ROI 밝기 허용 폭의 95백분위.
    깨끗한 시드에서 거의 아무것도 바꾸지 않는 값이라, 오염된 시드·허용 폭이 터진 사진만 걸린다. 정답은 보지 않는다."""
    cm = pd.read_csv("results/measure_v5/C_mask_dev.csv", encoding="utf-8-sig")
    clean = cm[cm["seed_bad"] < 0.05]
    items = {it[0]: it for it in mv.dev_items()}
    chroma = []
    for name in clean["image"]:
        _, path, _, top0, bottom = items[name]
        small = io_utils.resize_width(io_utils.load_image(path))
        top = pipeline.auto_band_top(small, top0) if C.AUTO_BAND_TOP else top0
        roi, _ = io_utils.crop_roi(small, top, bottom)
        h, w = roi.shape[:2]
        lab = cv.cvtColor(cv.GaussianBlur(roi, (0, 0), C.ROAD_BLUR_SIGMA), cv.COLOR_BGR2LAB).astype(np.float32)
        fx0, fy0, fx1, fy1 = C.ROAD_SEED_BOX
        s = lab[int(h * fy0):int(h * fy1), int(w * fx0):int(w * fx1)].reshape(-1, 3)
        chroma.append(np.hypot(s[:, 1] - 128, s[:, 2] - 128)[::7])
    chroma = np.concatenate(chroma)
    calib = {"chroma_max": round(float(np.quantile(chroma, 0.995)), 2), "l_range_max": round(float(clean["l_range"].quantile(0.95)), 1),
             "n_clean": int(len(clean)), "n_dev": int(len(cm))}
    lines = [f"깨끗한 시드(측정 C 오염 합 < 5%): 개발셋 {calib['n_clean']}/{calib['n_dev']}장",
             f"시드 상자 픽셀 채도(a·b의 무채색 거리, 0~255 Lab): 중앙값 {np.median(chroma):.2f}, 95백분위 {np.quantile(chroma, 0.95):.2f}, "
             f"99백분위 {np.quantile(chroma, 0.99):.2f}, 99.5백분위 {calib['chroma_max']} → A2 채도 기준 {calib['chroma_max']}",
             f"ROI 밝기 허용 폭 ±L: 깨끗한 시드 중앙값 {clean['l_range'].median():.1f}, 90백분위 {clean['l_range'].quantile(0.9):.1f}, "
             f"95백분위 {calib['l_range_max']} (개발셋 전체 중앙값 {cm['l_range'].median():.1f}, ≥ 128 {int((cm['l_range'] >= 128).sum())}장) "
             f"→ A3 상한 {calib['l_range_max']}",
             f"상한에 걸리는 개발셋 사진: {int((cm['l_range'] > calib['l_range_max']).sum())}장"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "calib.json").write_text(json.dumps(calib, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_DIR / "calib.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def fp_counts(out, gt):
    """후보 중심이 같은 종류 정답 상자 밖이면 오검출(evaluate.count_center_hits, 640px 전체 좌표)."""
    y0 = out["y0"]
    res = {}
    for kind, cands in (("crack", out["cracks"]), ("pothole", out["potholes"])):
        boxes = [(x, y + y0, w, h) for x, y, w, h in (r["bbox"] for r in cands)]
        truth = [g[:4] for g in gt if g[4] == kind]
        hits = evaluate.count_center_hits(boxes, truth)
        res[f"n_{kind}"] = len(boxes)
        res[f"fp_{kind}"] = len(boxes) - hits["tp"]
    return res


def measure_image(set_name, item, calib, variants):
    """사진 1장을 변형마다 파이프라인에 돌려 노면 지표 행 list를 낸다."""
    name, path, gt_path, top0, bottom = item
    bgr = io_utils.load_image(path)
    small = io_utils.resize_width(bgr)
    H, W = small.shape[:2]
    gt = evaluate.load_gt(str(gt_path)) or []
    lanes = lanes_with_road(small, int(H * top0), int(H * bottom))
    rows = []
    for vid in variants:
        opts = VARIANTS[vid]
        with mask_variant(opts, calib):
            out = pipeline.run_pipeline(bgr, top0, bottom, False)
        y0, road = out["y0"], out["road"] > 0
        info = {}
        again = road_mask_v6(io_utils.crop_roi(small, out["top"], bottom)[0], opts, calib, info) > 0
        roi = io_utils.crop_roi(small, out["top"], bottom)[0]
        full = np.zeros((H, W), bool)
        full[y0:y0 + road.shape[0]] = road
        seed = info["seed"]
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        lab8 = cv.cvtColor(roi, cv.COLOR_BGR2LAB).astype(np.float32)
        chroma = np.hypot(lab8[:, :, 1] - 128, lab8[:, :, 2] - 128)
        s_bright, s_chroma, s_off = mv.bright_fill(gray, dilate=False)[seed], chroma[seed] > mv.HIGH_CHROMA, ~road[seed]
        row = {"variant": vid, "set": set_name, "image": name, "source": et.source_of(name), "view": view_of(name),
               "top_start": top0, "top": out["top"], "widened": bool(out["top"] < top0), "road_frac": float(road.mean()),
               "l_thr": info["l_thr"], "seed_px": int(seed.sum()), "seed_bright": float(s_bright.mean()), "seed_chroma": float(s_chroma.mean()),
               "seed_off_mask": float(s_off.mean()), "seed_bad": float((s_bright | s_chroma | s_off).mean()),
               "spread_top": float(full[:int(H * SPREAD_ROWS)].mean()), "same_as_pipeline": bool(np.array_equal(again, road)),
               "has_lane": bool(lanes), "lane_both": bool(lanes) and all(full[s1].mean() >= LANE_PAINT and full[s2].mean() >= LANE_PAINT
                                                                         for s1, s2 in lanes[:1]),
               "n_gt": len(gt), "gt_above": sum(1 for x, y, w, h, k in gt if y + h / 2 < y0)}
        for kind in ("crack", "pothole"):
            band = [(int(x + w / 2), int(y + h / 2)) for x, y, w, h, k in gt if k == kind and y0 <= int(y + h / 2) < y0 + road.shape[0]]
            row[f"{kind}_in_band"] = len(band)
            row[f"{kind}_in_mask"] = sum(1 for cx, cy in band if full[cy, min(cx, W - 1)])
        if vid == "now":
            row["same_as_src"] = bool(np.array_equal(road, io_utils.road_mask(roi) > 0)
                                      and np.array_equal(road_mask_v6(small, (), calib), io_utils.road_mask(small)))
        row.update(fp_counts(out, gt))
        rows.append(row)
    return rows


def summarize(df):
    """조합별 노면 지표 표(개발셋·39장 따로). 시점별(드론·오토바이 / 차량 / 직접 촬영) 열을 함께 낸다."""
    out = []
    for (vid, set_name), g in df.groupby(["variant", "set"], sort=False):
        lanes = g[g["has_lane"]]
        dm = g[g["view"] == "drone_moto"]
        car = g[g["view"] == "car"]
        own = g[g["view"] == "own"]
        out.append({"variant": vid, "set": set_name, "images": len(g),
                    "seed_bad_20": int((g["seed_bad"] >= mv.SEED_BAD_RATIO).sum()),
                    "seed_bad_mean": g["seed_bad"].mean(),
                    "l_thr_ge128": int((g["l_thr"] >= 128).sum()), "l_thr_median": g["l_thr"].median(),
                    "lane_images": len(lanes), "lane_both": int(lanes["lane_both"].sum()),
                    "crack_mask_ratio": g["crack_in_mask"].sum() / max(1, g["crack_in_band"].sum()),
                    "crack_in_mask": int(g["crack_in_mask"].sum()), "crack_in_band": int(g["crack_in_band"].sum()),
                    "pothole_mask_ratio": g["pothole_in_mask"].sum() / max(1, g["pothole_in_band"].sum()),
                    "pothole_in_mask": int(g["pothole_in_mask"].sum()), "pothole_in_band": int(g["pothole_in_band"].sum()),
                    "gt_above": int(g["gt_above"].sum()), "gt_above_dm": int(dm["gt_above"].sum()), "gt_above_car": int(car["gt_above"].sum()),
                    "spread_top": int((g["spread_top"] >= SPREAD_MIN).sum()),
                    "spread_top_dm": int((dm["spread_top"] >= SPREAD_MIN).sum()), "spread_top_car": int((car["spread_top"] >= SPREAD_MIN).sum()),
                    "spread_top_own": int((own["spread_top"] >= SPREAD_MIN).sum()),
                    "widened_dm": int(dm["widened"].sum()), "n_dm": len(dm), "widened_car": int(car["widened"].sum()), "n_car": len(car),
                    "widened_own": int(own["widened"].sum()), "n_own": len(own),
                    "road_frac_mean": g["road_frac"].mean(),
                    "fp_per_image": (g["fp_crack"].sum() + g["fp_pothole"].sum()) / len(g),
                    "fp_crack_per_image": g["fp_crack"].sum() / len(g), "fp_pothole_per_image": g["fp_pothole"].sum() / len(g),
                    "same_as_pipeline": int(g["same_as_pipeline"].sum())})
    return pd.DataFrame(out)


def summary_lines(s, df):
    pct = lambda v: f"{100 * v:.2f}%"
    lines = ["노면 마스크 변형 비교(10/14, 보완 최종 config 위, 전처리 미적용, 점수 임계값 config 그대로). 시험셋 사진은 열지 않음.", ""]
    now = df[df["variant"] == "now"]
    lines.append(f"대조: 변형 없음(now)이 io_utils.road_mask와 같은 사진 {int(now['same_as_src'].sum())}/{len(now)}장, "
                 f"모든 행에서 파이프라인 마스크와 도구 마스크가 같은 행 {int(df['same_as_pipeline'].sum())}/{len(df)}")
    for set_name in ("dev", "39"):
        lines += ["", f"[{'개발셋 652장' if set_name == 'dev' else '39장(자동 ROI)'}]",
                  "| 조합 | 시드 오염 ≥ 20% | 밝기 허용 폭 ≥ 128 | 차선 양쪽 칠함 / 차선 사진 | 균열 마스크 안 | 포트홀 마스크 안 | ROI 위 정답(드론·오토바이·차량) "
                  "| 위쪽 번짐(드론·오토바이·차량·직접) | ROI 넓어짐 드론·오토바이 / 차량 / 직접 | 오검출/장(균열·포트홀) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in s[s["set"] == set_name].iterrows():
            lines.append(f"| {r['variant']} | {r['seed_bad_20']} | {r['l_thr_ge128']} | {r['lane_both']} / {r['lane_images']} | "
                         f"{pct(r['crack_mask_ratio'])} ({r['crack_in_mask']}/{r['crack_in_band']}) | "
                         f"{pct(r['pothole_mask_ratio'])} ({r['pothole_in_mask']}/{r['pothole_in_band']}) | "
                         f"{r['gt_above']} ({r['gt_above_dm']}·{r['gt_above_car']}) | "
                         f"{r['spread_top']} ({r['spread_top_dm']}·{r['spread_top_car']}·{r['spread_top_own']}) | "
                         f"{r['widened_dm']}/{r['n_dm']} · {r['widened_car']}/{r['n_car']} · {r['widened_own']}/{r['n_own']} | "
                         f"{r['fp_per_image']:.2f} ({r['fp_crack_per_image']:.2f}·{r['fp_pothole_per_image']:.2f}) |")
    return lines


def pick_dev_figures(df):
    """개발셋 확인 그림 20장(변형 없음 기준, 정답은 보지 않음): 드론·오토바이 6장 = 차선 양쪽이 안 칠해진 사진 → 시드 오염 큰 사진 순,
    차량 시점 14장 = 시드 오염 큰 5장 + 차선 양쪽이 안 칠해진 5장 + ROI가 넓어졌거나 위쪽까지 번진 4장. 같은 순위는 이름순."""
    now = df[(df["variant"] == "now") & (df["set"] == "dev")].copy()
    now["lane_cut"] = now["has_lane"] & ~now["lane_both"]
    picked = []

    def take(frame, n):
        for name in frame["image"]:
            if n <= 0:
                break
            if name not in picked:
                picked.append(name)
                n -= 1

    dm = now[now["view"] == "drone_moto"].sort_values(["lane_cut", "seed_bad", "image"], ascending=[False, False, True])
    take(dm, N_FIG_DRONE)
    car = now[now["view"] == "car"]
    take(car.sort_values(["seed_bad", "image"], ascending=[False, True]), 5)
    take(car[car["lane_cut"]].sort_values(["road_frac", "image"]), 5)
    take(car[car["widened"] | (car["spread_top"] >= SPREAD_MIN)].sort_values(["spread_top", "image"], ascending=[False, True]), 4)
    return picked


FIG_VARIANTS = ("now", "A1", "A2", "A2b", "A3", "B1", "B2", "B1B2", "A2A3B1B2", "A2bA3B1B2", "A1A2bA3B1B2", "all")


def panel(small, road_full, top_y, seed_full, label):
    """확인 그림 한 칸: 노면 = 반투명 초록, ROI 상단 경계 = 빨간 선, 시드 = 자홍 점. 640px → 절반."""
    img = small.copy()
    if road_full is not None:
        green = img.copy()
        green[road_full] = (0, 200, 0)
        img = cv.addWeighted(green, 0.45, img, 0.55, 0)
        cv.line(img, (0, top_y), (img.shape[1] - 1, top_y), (0, 0, 255), 2)
    if seed_full is not None:
        edge = cv.morphologyEx(seed_full.astype(np.uint8), cv.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        img[edge] = (255, 0, 255)
    img = cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv.INTER_AREA)
    cv.rectangle(img, (0, 0), (8 + 11 * len(label), 22), (0, 0, 0), -1)
    cv.putText(img, label, (4, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    return img


def make_figures(df, calib):
    """39장 전부와 개발셋 20장의 확인 그림(원본 + 변형 12개, 13칸을 2줄 × 7칸, 마지막 칸은 빈칸)을 저장한다."""
    picks = [("39", it) for it in mv.own_items()]
    dev = {it[0]: it for it in mv.dev_items()}
    picks += [("dev", dev[n]) for n in pick_dev_figures(df)]
    for set_name, (name, path, _, top0, bottom) in picks:
        bgr = io_utils.load_image(path)
        small = io_utils.resize_width(bgr)
        H, W = small.shape[:2]
        tiles = [panel(small, None, 0, None, f"{name[:-4]} ({view_of(name)})")]
        for vid in FIG_VARIANTS:
            with mask_variant(VARIANTS[vid], calib):
                out = pipeline.run_pipeline(bgr, top0, bottom, False)
            info = {}
            road_mask_v6(io_utils.crop_roi(small, out["top"], bottom)[0], VARIANTS[vid], calib, info)
            full, seed = np.zeros((H, W), bool), np.zeros((H, W), bool)
            full[out["y0"]:out["y0"] + out["road"].shape[0]] = out["road"] > 0
            seed[out["y0"]:out["y0"] + out["road"].shape[0]] = info["seed"]
            tiles.append(panel(small, full, out["y0"], seed, vid))
        tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 7]) for i in (0, 7)]
        cv.imwrite(str(OUT_DIR / f"fig_{set_name}_{name[:-4]}.png"), np.vstack(rows))
    (OUT_DIR / "figures.txt").write_text("\n".join(f"{s} {it[0]} {view_of(it[0])}" for s, it in picks) + "\n", encoding="utf-8")
    print(f"그림 {len(picks)}장 → {OUT_DIR}")


def _measure_one(pair, calib, variants):
    return measure_image(pair[0], pair[1], calib, variants)


class _Serial:
    """--jobs 1: Pool과 같은 모양(imap)으로 한 프로세스에서 차례로 돈다."""
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @staticmethod
    def imap(fn, items):
        return map(fn, items)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--figs", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--jobs", type=int, default=1, help="사진을 나눠 돌릴 프로세스 수(결과 순서·값은 같음)")
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.calib:
        calibrate()
        return 0
    calib = load_calib()
    if args.figs:
        make_figures(pd.read_csv(OUT_DIR / "per_image.csv", encoding="utf-8-sig"), calib)
        return 0
    variants = args.variants.split(",")
    rows = []
    items = items_all(args.limit)
    work = partial(_measure_one, calib=calib, variants=variants)
    with Pool(args.jobs, initializer=cv.setNumThreads, initargs=(1,)) if args.jobs > 1 else _Serial() as pool:
        for i, part in enumerate(pool.imap(work, items)):
            rows += part
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(items)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_image.csv", encoding="utf-8-sig", index=False)
    s = summarize(df)
    s.to_csv(OUT_DIR / "summary.csv", encoding="utf-8-sig", index=False)
    lines = summary_lines(s, df)
    (OUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
