"""(파트 A) 전처리 파이프라인(io_utils, analyze, preprocess) 통합 및 단위 테스트
사용법: 맨 위 폴더에서  python tests/test_a.py   (pytest 없이 실행되고, pytest로 돌려도 됨)"""
import contextlib
import functools
import glob
import os
import sys
import traceback
import unittest

import numpy as np
import cv2 as cv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import io_utils, analyze, preprocess  # noqa: E402
from src import config as C  # noqa: E402


def todo(reason):
    """아직 구현하지 않은 기능의 시험 표시: 기대값은 적어 두되 지금은 건너뛴다(unittest.SkipTest라 pytest도 건너뜀).
    구현한 사람이 이 줄(@todo)을 지우면 시험이 돈다. 이 시험이 그 작업의 통과 기준이다(docs/tasks/A.md 4장)."""
    def mark(fn):
        @functools.wraps(fn)
        def skipped(*args, **kwargs):
            raise unittest.SkipTest(reason)
        return skipped
    return mark

# ==========================================
# 1. io_utils.py 테스트
# ==========================================
def test_resize_width():
    """가로 크기 정규화 테스트 (비율 유지 확인)"""
    # 가로 100, 세로 200인 가상 컬러 이미지 생성
    dummy_img = np.zeros((200, 100, 3), dtype=np.uint8)
    
    # 설정된 TARGET_WIDTH(예: 640)로 변환
    target_w = getattr(C, 'TARGET_WIDTH', 640)
    resized = io_utils.resize_width(dummy_img, width=target_w)
    
    h, w = resized.shape[:2]
    assert w == target_w
    # 가로가 6.4배 커졌으므로 세로도 6.4배(1280) 커져야 함
    assert h == int(round(200 * (target_w / 100)))

def test_crop_roi():
    """관심 영역(ROI) 크롭 테스트"""
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    # 상위 35%에서 하단 100%까지 자르기
    cropped, y0 = io_utils.crop_roi(dummy_img, top=0.35, bottom=1.0)
    
    assert y0 == 35
    assert cropped.shape[0] == 65  # 100 - 35 = 65

def test_parse_name():
    """파일명 분석 규칙 테스트: 약속 표의 반환 형식과 규칙 위반 ValueError"""
    provided = {"source": "provided", "damage": "", "condition": ""}
    # 1. 제공 데이터: 밑줄 개수와 상관없이 provided (조건은 roi.csv에서 읽음)
    for name in ("data/provided/Japan_000107.jpg", "data/provided/United_States_004830.jpg",
                 "data/provided/China_MotorBike_000123.jpg"):
        res = io_utils.parse_name(name)
        assert res == provided, f"{name} -> {res}"

    # 2. own 규칙: {source, damage, condition}
    res2 = io_utils.parse_name("data/own/own_01_crack_dark.jpg")
    assert res2 == {"source": "own", "damage": "crack", "condition": "dark"}
    res2b = io_utils.parse_name("own_03_none_normal.jpg")
    assert res2b == {"source": "own", "damage": "none", "condition": "normal"}

    # 3. pair 규칙: {source, pair, role, condition}
    res3 = io_utils.parse_name("data/pairs/pair01_A_ref.jpg")
    assert res3 == {"source": "pair", "pair": "01", "role": "A", "condition": "ref"}

    # 4. 예외 처리: 규칙을 어긴 own·pair 이름은 ValueError
    bad_names = [
        "own_01_crack.jpg",            # 칸 부족
        "own_01_crack_dark_x.jpg",     # 칸 초과
        "own_xx_crack_dark.jpg",       # 번호가 숫자가 아님
        "own_01_normal_dark.jpg",      # 손상 태그 아님 (손상 없음은 none)
        "own_01_crack_day.jpg",        # 조건 태그 아님 (낮은 normal)
        "pair01_A.jpg",                # 칸 부족
        "pair_A_ref.jpg",              # pair 번호 없음
        "pair01_Z_ref.jpg",            # 기호 태그 아님
        "pair01_A_day.jpg",            # 쌍 조건 태그 아님 (기준은 ref)
    ]
    accepted = []
    for name in bad_names:
        try:
            io_utils.parse_name(name)
            accepted.append(name)
        except ValueError:
            pass
    assert not accepted, f"규칙 위반인데 ValueError가 나지 않음: {accepted}"


def road_band(seed=0, split_lane=False):
    """노면 마스크 시험용 컬러 띠(240×640)와 영역별 위치. 회색 아스팔트 + 왼쪽 풀(초록) + 오른쪽 위 건물(주황빛)
    + 노면 안 포트홀(어두운 원) + 흰 차선. split_lane이면 차선이 띠 아래에서 건물까지 이어져 노면을 둘로 가르고,
    아니면 테두리에 닿지 않는 짧은 차선 조각이다. (컬러 띠, {영역 이름: bool 마스크})를 돌려준다."""
    rng = np.random.default_rng(seed)
    gray = cv.GaussianBlur(rng.normal(115, 10, (240, 640)).clip(0, 255).astype(np.uint8), (0, 0), 1.2)
    bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    grass = np.zeros((240, 640), bool)
    grass[:, :120] = True
    bgr[grass] = (np.array([45, 140, 70]) + rng.normal(0, 8, (grass.sum(), 3))).clip(0, 255).astype(np.uint8)
    building = np.zeros((240, 640), bool)
    building[:60, 420:] = True
    bgr[building] = (np.array([90, 130, 190]) + rng.normal(0, 6, (building.sum(), 3))).clip(0, 255).astype(np.uint8)
    pothole = np.zeros((240, 640), np.uint8)
    cv.circle(pothole, (250, 170), 25, 255, -1)
    pothole = pothole > 0
    bgr[pothole] = 45
    lane = np.zeros((240, 640), np.uint8)
    if split_lane:
        cv.line(lane, (520, 239), (480, 60), 255, 8)
    else:
        cv.line(lane, (500, 200), (480, 110), 255, 8)
    lane = (lane > 0) & ~building
    bgr[lane] = 230
    road = ~grass & ~building & ~pothole & ~lane
    right = road & (np.arange(640)[None, :] > 540)
    return bgr, {"road": road, "grass": grass, "building": building, "pothole": pothole, "lane": lane, "right": right}


def test_road_mask_format():
    """road_mask는 같은 크기 uint8 0/255를 돌려주고, 입력을 바꾸지 않으며, 빈 영상에서도 오류가 없고, 같은 입력이면 같은 출력."""
    bgr, _ = road_band()
    before = bgr.copy()
    for img in (bgr, np.zeros_like(bgr), np.full_like(bgr, 255)):
        mask = io_utils.road_mask(img)
        assert isinstance(mask, np.ndarray) and mask.dtype == np.uint8 and mask.shape == img.shape[:2], \
            f"road_mask 형식이 (H, W) uint8이 아님: {getattr(mask, 'dtype', None)} {getattr(mask, 'shape', None)}"
        assert set(np.unique(mask).tolist()) <= {0, 255}, f"road_mask 값이 0/255가 아님: {np.unique(mask)}"
    assert np.array_equal(bgr, before), "road_mask가 입력 영상을 바꿈"
    assert np.array_equal(io_utils.road_mask(bgr), io_utils.road_mask(bgr)), "같은 입력인데 road_mask 결과가 다름"

def test_road_mask_v1_synthetic():
    """v1: 씨앗(아래 가운데)과 색·밝기가 비슷하고 이어진 노면은 남기고 풀·건물은 뺀다.
    노면 안의 포트홀과 테두리에 닿지 않는 짧은 차선 조각은 구멍 메우기·닫힘으로 마스크에 넣는다.
    (C가 부록 C 코드로 메모리에서 확인: 노면 0.99, 포트홀 1.0, 차선 조각 1.0, 풀 0.0, 건물 0.001)"""
    bgr, part = road_band()
    road = io_utils.road_mask(bgr) == 255
    ratio = {name: round(float(road[where].mean()), 3) for name, where in part.items()}
    assert ratio["road"] >= 0.95 and ratio["pothole"] >= 0.95 and ratio["lane"] >= 0.95, ratio
    assert ratio["grass"] <= 0.05 and ratio["building"] <= 0.05, ratio


def test_road_mask_v2_shadow():
    """v2 8a(보완 4-2): 오른쪽 노면에 같은 색의 그림자(밝기 −50)를 드리우면 v1은 그늘진 노면을 빼고, ROAD_SHADOW_L_K를 켜면 잇는다.
    풀·건물(색이 다름)은 켜도 들어오지 않는다. 합성 씨앗은 결이 고와 밝기 문턱이 작아(약 14) 시험에서는 K 5로 장치만 확인한다
    (개발셋 띠의 밝기 문턱은 중앙값 55라 후보 K는 1.5·2, part_C 4-2). (C가 메모리에서 확인: K 0 → 그림자 0.0, K 5 → 0.98)"""
    bgr, part = road_band()
    shadow = part["road"] & (np.arange(640)[None, :] >= 500)
    img = bgr.copy()
    img[shadow] = (img[shadow].astype(int) - 50).clip(0, 255).astype(np.uint8)
    with config_values(ROAD_SHADOW_L_K=0):
        off = io_utils.road_mask(img) == 255
    with config_values(ROAD_SHADOW_L_K=5):
        on = io_utils.road_mask(img) == 255
    assert off[shadow].mean() <= 0.05, f"v1이 그림자 노면을 이미 넣음 ({off[shadow].mean():.3f}): 시험 장면이 그림자를 만들지 못함"
    assert on[shadow].mean() >= 0.90, f"8a를 켰는데 그림자 노면이 빠짐 ({on[shadow].mean():.3f})"
    assert on[part["grass"]].mean() <= 0.05 and on[part["building"]].mean() <= 0.05, "8a를 켜자 색이 다른 풀·건물이 들어옴"


def test_road_mask_v2_hood():
    """v2 8b(보완 4-2): 띠 아래 30행을 매끈한 보닛(같은 회색, 결 없음)으로 덮으면 v1은 노면에 넣고, ROAD_HOOD_CUT을 켜면 뺀다.
    보닛 위 노면은 그대로다. 매끈한 행이 아래 테두리에 붙어 있지 않으면(보닛 없음) 아무 행도 빼지 않는다."""
    bgr, part = road_band()
    img = bgr.copy()
    img[210:] = 110
    with config_values(ROAD_HOOD_CUT=False):
        off = io_utils.road_mask(img) == 255
    with config_values(ROAD_HOOD_CUT=True):
        on = io_utils.road_mask(img) == 255
        plain = io_utils.road_mask(bgr) == 255
    above = part["road"].copy()
    above[210:] = False
    assert off[210:].mean() >= 0.90, f"v1이 보닛을 노면에 넣지 않음 ({off[210:].mean():.3f}): 시험 장면이 보닛을 만들지 못함"
    assert on[210:].mean() <= 0.15 and on[above].mean() >= 0.95, (on[210:].mean(), on[above].mean())
    with config_values(ROAD_HOOD_CUT=False):
        assert np.array_equal(plain, io_utils.road_mask(bgr) == 255), "보닛이 없는 띠에서 8b가 행을 뺌"


def test_road_mask_v2_edge_density():
    """v2 8c(보완 4-2): 오른쪽 아래를 노면과 같은 평균 회색의 촘촘한 줄무늬(인도 블록·건물 벽 흉내)로 바꾸면 v1은 노면에 넣고,
    ROAD_EDGE_DENSITY를 켜면 뺀다. 나머지 노면과 노면 안 포트홀은 남는다."""
    bgr, part = road_band()
    img = bgr.copy()
    dense = np.zeros((240, 640), bool)
    dense[120:, 560:] = True
    stripe = ((np.arange(640) // 3) % 2 == 0)[None, :].repeat(240, 0)
    img[dense & stripe] = 150
    img[dense & ~stripe] = 85
    with config_values(ROAD_EDGE_DENSITY=False):
        off = io_utils.road_mask(img) == 255
    with config_values(ROAD_EDGE_DENSITY=True):
        on = io_utils.road_mask(img) == 255
    rest = part["road"] & ~dense
    assert off[dense].mean() >= 0.90, f"v1이 줄무늬를 노면에 넣지 않음 ({off[dense].mean():.3f}): 시험 장면이 맞지 않음"
    assert on[dense].mean() <= 0.10, f"8c를 켰는데 촘촘한 줄무늬가 노면에 남음 ({on[dense].mean():.3f})"
    assert on[rest].mean() >= 0.95 and on[part["pothole"]].mean() >= 0.95, (on[rest].mean(), on[part["pothole"]].mean())


def test_road_mask_seed_cells():
    """v5 (3)①: 시드 상자의 오른쪽 절반을 띠 위 테두리까지 이어진 풀(구멍이 아님)로 덮으면, 상자 시드("box")는 대표색이 풀과 노면
    사이에 떨어져 노면을 못 칠하고, 칸 시드("cells")는 노면 칸 묶음을 골라 노면을 칠하며 풀은 뺀다. 풀이 없는 띠에서는 두 방식 모두 v1과 같은 수준.
    (C가 메모리에서 확인: box 노면 0.0, cells 노면 0.971·풀 조각 0.0·왼쪽 풀 0.0·포트홀 1.0)"""
    bgr, part = road_band()
    img = bgr.copy()
    patch = np.zeros((240, 640), bool)
    patch[:200, 320:460] = True
    rng = np.random.default_rng(1)
    img[patch] = (np.array([45, 140, 70]) + rng.normal(0, 8, (patch.sum(), 3))).clip(0, 255).astype(np.uint8)
    road = part["road"] & ~patch
    with config_values(ROAD_SEED_MODE="box", ROAD_SEED_CHECK=False):     # 시드 검사(V6a, 10/13 밤 채택)를 끈 상자 시드 그대로
        box = io_utils.road_mask(img) == 255
    with config_values(ROAD_SEED_MODE="cells"):
        cells = io_utils.road_mask(img) == 255
        plain = io_utils.road_mask(bgr) == 255
    assert box[road].mean() <= 0.5, f"상자 시드가 이미 노면을 칠함 ({box[road].mean():.3f}): 시험 장면이 시드를 오염시키지 못함"
    assert cells[road].mean() >= 0.90 and cells[part["pothole"]].mean() >= 0.95, (cells[road].mean(), cells[part["pothole"]].mean())
    assert cells[patch].mean() <= 0.05 and cells[part["grass"]].mean() <= 0.05, (cells[patch].mean(), cells[part["grass"]].mean())
    assert plain[part["road"]].mean() >= 0.95 and plain[part["building"]].mean() <= 0.05, "풀 없는 띠에서 칸 시드가 v1보다 나쁨"
    with config_values(ROAD_SEED_MODE="nope"):
        try:
            io_utils.road_mask(bgr)
        except ValueError:
            pass
        else:
            raise AssertionError("ROAD_SEED_MODE가 잘못된 값인데 ValueError가 나지 않음")


def test_road_mask_seed_check():
    """v5 V6a: ROAD_SEED_CHECK를 켜도 시드 상자가 노면 한 가지인 띠(v1 합성)는 끔과 픽셀 하나까지 같고, 시드 상자 오른쪽 절반이 풀(색이 다른 면)인
    띠는 검사에 걸려 칸 시드로 바뀌어 노면을 칠한다(칸 시드 결과와 같음). 시드를 밖에서 주는 road_mask_seeded(None)는 road_mask와 같다."""
    bgr, part = road_band()
    img = bgr.copy()
    patch = np.zeros((240, 640), bool)
    patch[:200, 320:460] = True
    rng = np.random.default_rng(1)
    img[patch] = (np.array([45, 140, 70]) + rng.normal(0, 8, (patch.sum(), 3))).clip(0, 255).astype(np.uint8)
    road = part["road"] & ~patch
    with config_values(ROAD_SEED_CHECK=False):
        off_plain, off_img = io_utils.road_mask(bgr), io_utils.road_mask(img)
    with config_values(ROAD_SEED_CHECK=True):
        on_plain, on_img = io_utils.road_mask(bgr), io_utils.road_mask(img)
        seeded = io_utils.road_mask_seeded(bgr, None)
    with config_values(ROAD_SEED_MODE="cells"):
        cells_img = io_utils.road_mask(img)
    assert np.array_equal(off_plain, on_plain) and np.array_equal(seeded, on_plain), "시드 검사를 통과한 띠가 끔과 다름"
    assert (off_img == 255)[road].mean() <= 0.5, f"검사 끔인데 오염된 시드로 노면을 칠함 ({(off_img == 255)[road].mean():.3f}): 장면이 맞지 않음"
    assert np.array_equal(on_img, cells_img) and (on_img == 255)[road].mean() >= 0.90, "오염된 시드가 검사에 걸려 칸 시드로 바뀌지 않음"


def test_road_mask_l_range_max():
    """v5 (3)①: ROAD_L_RANGE_MAX를 주면 노면 밝기 범위가 그 값을 넘지 않는다. 어두운 노면(밝기 −60)을 오른쪽에 두면
    상한이 없을 때(범위 넓게, ROAD_L_K 20)는 잇고, 상한 30이면 뺀다. 상한은 ab_L 색 기준의 L 범위에만 걸리므로(mahal은 L 범위 식을 쓰지 않음)
    10/13 V5a 채택으로 기본이 "mahal"이 된 뒤에도 ab_L을 명시해 이 장치를 시험한다."""
    bgr, part = road_band()
    dark = part["road"] & (np.arange(640)[None, :] >= 500)
    img = bgr.copy()
    img[dark] = (img[dark].astype(int) - 60).clip(0, 255).astype(np.uint8)
    with config_values(ROAD_COLOR_DIST="ab_L", ROAD_L_K=20, ROAD_L_RANGE_MAX=0):
        wide = io_utils.road_mask(img) == 255
    with config_values(ROAD_COLOR_DIST="ab_L", ROAD_L_K=20, ROAD_L_RANGE_MAX=30):
        capped = io_utils.road_mask(img) == 255
    assert wide[dark].mean() >= 0.90 and capped[dark].mean() <= 0.10, (wide[dark].mean(), capped[dark].mean())


def test_road_mask_mahal():
    """v5 (3)③: ROAD_COLOR_DIST "mahal"(시드 Lab 분포의 마할라노비스 거리)도 v1 합성 띠에서 노면·포트홀·차선 조각은 남기고 풀·건물은 뺀다.
    오른쪽 노면이 같은 색으로 조금 어두워지면(−8, 하한으로 둔 최소 범위 안) 잇는다. 잘못된 값은 ValueError.
    (C가 메모리에서 확인: ab_L·mahal 99% 모두 노면 0.989·풀 0.0·건물 0.001·포트홀 1.0. 오른쪽 −4·−6·−8은 둘 다 0.989,
    −12는 ab_L 0.989·mahal 0.011 — 결이 고운 시드에서는 mahal의 밝기 범위가 ab_L보다 좁다)"""
    bgr, part = road_band()
    with config_values(ROAD_COLOR_DIST="mahal"):
        road = io_utils.road_mask(bgr) == 255
        ratio = {name: round(float(road[where].mean()), 3) for name, where in part.items()}
        img = bgr.copy()
        right = part["road"] & (np.arange(640)[None, :] >= 500)
        img[right] = (img[right].astype(int) - 8).clip(0, 255).astype(np.uint8)
        dim = io_utils.road_mask(img) == 255
    assert ratio["road"] >= 0.95 and ratio["pothole"] >= 0.95 and ratio["lane"] >= 0.95, ratio
    assert ratio["grass"] <= 0.05 and ratio["building"] <= 0.05, ratio
    assert dim[right].mean() >= 0.90, f"mahal이 조금 어두운 같은 색 노면을 뺌 ({dim[right].mean():.3f})"
    with config_values(ROAD_COLOR_DIST="nope"):
        try:
            io_utils.road_mask(bgr)
        except ValueError:
            pass
        else:
            raise AssertionError("ROAD_COLOR_DIST가 잘못된 값인데 ValueError가 나지 않음")


@todo("A WP1 v2 (Should): 차선이 노면을 가르는 영상 (Plan.md 3.3 (5) 차선)")
def test_road_mask_v2_lane_split():
    """v2: 흰 차선이 띠 아래에서 위까지 노면을 갈라도 차선 건너편 노면과 차선을 마스크에 넣는다.
    (v1은 건너편을 통째로 뺀다. C가 부록 C 코드로 메모리에서 확인: 건너편 0.0, 차선 0.0)"""
    bgr, part = road_band(split_lane=True)
    road = io_utils.road_mask(bgr) == 255
    ratio = {name: round(float(road[where].mean()), 3) for name, where in part.items()}
    assert ratio["right"] >= 0.90 and ratio["lane"] >= 0.90, ratio
    assert ratio["grass"] <= 0.05 and ratio["building"] <= 0.05, ratio

# ==========================================
# 2. analyze.py 테스트
# ==========================================
def test_measure_quality():
    """품질 측정 지표 반환 테스트"""
    # 밝기 100으로 채워진 가상 흑백 이미지
    dummy_gray = np.full((50, 50), 100, dtype=np.uint8)
    q = analyze.measure_quality(dummy_gray)
    
    assert "mean" in q and "std" in q and "lap_var" in q and "noise" in q
    assert q["mean"] == 100.0
    assert q["std"] == 0.0  # 모두 같은 색이므로 표준편차는 0

@contextlib.contextmanager
def config_values(**values):
    """config 값을 잠시 바꿨다가 시험이 끝나면(실패해도) 되돌린다."""
    old = {k: getattr(C, k) for k in values}
    for k, v in values.items():
        setattr(C, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(C, k, v)


def good_quality(**changes):
    """config 기준값을 모두 넉넉히 통과하는 상태 지표(전처리가 필요 없는 영상)에서 일부만 바꾼 dict."""
    q = {"mean": (C.DARK_MEAN_MAX + C.BRIGHT_MEAN_MIN) / 2, "std": C.CONTRAST_STD_MIN * 2,
         "lap_var": C.BLUR_VAR_MAX * 2, "noise": C.NOISE_SIGMA_MAX / 2}
    q.update(changes)
    return q


def test_noise_ignores_texture():
    """잡티 지표는 결·경계를 잡음으로 세지 않는다: 깨끗한 무늬 영상은 기준 아래, 같은 영상에 잡음을 더하면 기준 위."""
    rng = np.random.default_rng(2)
    gray = np.full((240, 640), 120, np.uint8)
    # 왼쪽 절반: 대비가 큰 결(나뭇잎·알갱이 흉내), 오른쪽 절반: 평평한 노면 + 경계선
    gray[:, :320] = rng.integers(40, 220, (240, 320), dtype=np.uint8)
    cv.line(gray, (330, 20), (630, 220), 30, 3)
    clean = analyze.measure_quality(gray)["noise"]
    assert clean < C.NOISE_SIGMA_MAX, f"깨끗한 무늬 영상의 잡티 {clean:.2f}가 기준 {C.NOISE_SIGMA_MAX} 이상"
    noisy_img = (gray.astype(np.float32) + rng.normal(0, 2 * C.NOISE_SIGMA_MAX, gray.shape)).clip(0, 255).astype(np.uint8)
    noisy = analyze.measure_quality(noisy_img)["noise"]
    assert noisy > C.NOISE_SIGMA_MAX, f"σ {2 * C.NOISE_SIGMA_MAX} 잡음을 더했는데 잡티 {noisy:.2f}가 기준 이하"
    # 블록보다 작은 영상도 오류 없이 잰다
    assert np.isfinite(analyze.measure_quality(gray[:5, :5])["noise"])


def test_choose_steps():
    """진단 결과에 따른 전처리 단계 선택 테스트 (기준값은 config A 구역에서 읽음)"""
    # 1. 정상 (아무 처리 필요 없음)
    steps1 = analyze.choose_steps(good_quality())
    assert steps1 == analyze.NO_STEPS

    # 2. 저조도 → 밝게 하는 감마(< 1), 고조도 → 어둡게 하는 감마(> 1)
    steps2 = analyze.choose_steps(good_quality(mean=C.DARK_MEAN_MAX - 30))
    assert steps2["tone"] == "gamma"
    assert steps2["gamma"] < 1.0, steps2
    steps2b = analyze.choose_steps(good_quality(mean=C.BRIGHT_MEAN_MIN + 30))
    assert steps2b["tone"] == "gamma" and steps2b["gamma"] > 1.0, steps2b

    # 3. 감마는 GAMMA_RANGE 안으로 자름 (평균 0·255여도 오류 없음)
    lo, hi = C.GAMMA_RANGE
    assert analyze.choose_steps(good_quality(mean=0.0))["gamma"] == lo
    assert analyze.choose_steps(good_quality(mean=255.0))["gamma"] == hi

    # 4. 흐림 + 노이즈
    steps3 = analyze.choose_steps(good_quality(lap_var=C.BLUR_VAR_MAX / 2, noise=C.NOISE_SIGMA_MAX * 2))
    assert steps3["sharpen"] is True
    assert steps3["denoise"] is True

    # 5. 대비 부족 → 평활화, USE_CLAHE면 CLAHE
    low = good_quality(std=C.CONTRAST_STD_MIN / 2)
    with config_values(USE_CLAHE=False):
        assert analyze.choose_steps(low)["tone"] == "equalize"
    with config_values(USE_CLAHE=True):
        assert analyze.choose_steps(low)["tone"] == "clahe"


def test_choose_steps_follows_config():
    """config 기준값을 바꾸면 고르는 단계도 바뀐다 (함수 안에 숫자를 두지 않음)."""
    q = good_quality()
    with config_values(NOISE_SIGMA_MAX=q["noise"] / 2, BLUR_VAR_MAX=q["lap_var"] * 2,
                       DARK_MEAN_MAX=q["mean"] + 1, CONTRAST_STD_MIN=q["std"] * 2):
        s = analyze.choose_steps(q)
    assert s["denoise"] and s["sharpen"] and s["tone"] == "gamma", s
    with config_values(GAMMA_RANGE=(0.9, 1.1)):
        assert analyze.choose_steps(good_quality(mean=10.0))["gamma"] == 0.9


def test_blur_ratio_key():
    """quality에 blur_ratio 키가 있고 float다(약속 표). A가 WP4에서 채우기 전까지는 NaN이라 choose_steps 결과는 그대로다."""
    q = analyze.measure_quality(np.full((50, 50), 100, np.uint8))
    assert "blur_ratio" in q and isinstance(q["blur_ratio"], float), q


def test_blur_ratio_synthetic():
    """재흐림 비율 = Sobel 기울기 크기 평균 ÷ 가우시안(σ 1.5)을 건 뒤의 같은 값. 흐린 영상은 1에 가깝고 선명한 영상은 크다.
    비율이라 결의 세기(대비)와 상관없다. 인위 저하 E6의 blur(가우시안 σ 2)는 흐림으로 잡혀야 한다.
    (C가 메모리에서 이 식으로 확인: 선명 2.55, 대비 절반 2.54, σ 2 흐림 1.41)"""
    rng = np.random.default_rng(0)
    sharp = cv.GaussianBlur(rng.normal(120, 25, (240, 640)).clip(0, 255).astype(np.uint8), (0, 0), 1.0)
    blurred = cv.GaussianBlur(sharp, (0, 0), 2.0)
    weak = (sharp.astype(np.float32) * 0.5 + 60).astype(np.uint8)        # 같은 무늬, 대비만 절반
    r_sharp, r_blur, r_weak = (analyze.measure_quality(g)["blur_ratio"] for g in (sharp, blurred, weak))
    assert 1.0 <= r_blur < C.BLUR_RATIO_MAX <= r_sharp, (r_blur, C.BLUR_RATIO_MAX, r_sharp)
    assert abs(r_weak - r_sharp) / r_sharp < 0.05, f"대비만 바꿨는데 blur_ratio가 5% 넘게 달라짐 ({r_sharp:.3f} → {r_weak:.3f})"
    flat = analyze.measure_quality(np.full((50, 50), 100, np.uint8))["blur_ratio"]
    assert isinstance(flat, float), "평평한 영상에서 blur_ratio가 float가 아님 (NaN은 허용)"


def test_choose_steps_blur_ratio():
    """lap_var < BLUR_VAR_MAX 또는 blur_ratio < BLUR_RATIO_MAX이면 샤프닝. blur_ratio가 NaN이거나 키가 없으면 지금처럼 lap_var만 본다."""
    assert analyze.choose_steps(good_quality(blur_ratio=C.BLUR_RATIO_MAX / 2))["sharpen"] is True
    assert analyze.choose_steps(good_quality(blur_ratio=C.BLUR_RATIO_MAX * 2))["sharpen"] is False
    assert analyze.choose_steps(good_quality(blur_ratio=float("nan")))["sharpen"] is False
    assert analyze.choose_steps(good_quality())["sharpen"] is False                     # 키가 없어도 오류 없음
    assert analyze.choose_steps(good_quality(lap_var=C.BLUR_VAR_MAX / 2, blur_ratio=C.BLUR_RATIO_MAX * 2))["sharpen"] is True


def motion_blur(gray, length, angle):
    """길이 length px, 방향 angle도(0 = 가로)의 선 커널로 움직임 흐림을 넣은 새 영상(흔들림 흉내)."""
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    rot = cv.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), angle, 1.0)
    k = cv.warpAffine(k, rot, (length, length))
    return cv.filter2D(gray, -1, k / k.sum())


def test_blur_dir_motion():
    """방향별 흐림 비율(blur_dir = 방향별 기울기 제곱 평균의 최소 ÷ 최대, 보완 3-1): 방향이 고른 결은 1에 가깝고, 같은 결에 가로·세로
    움직임 흐림(15px)을 넣으면 작아지며, 가우시안 흐림(E6 blur σ 2)은 모든 방향을 같이 뭉개 크게 바뀌지 않는다. 평평한 영상은 NaN.
    (C가 메모리에서 이 합성으로 확인: 원본 0.99, 가우시안 0.97, 가로 15px 0.10, 세로 15px 0.10, 대각 15px 0.12)"""
    rng = np.random.default_rng(0)
    sharp = cv.GaussianBlur(rng.normal(120, 25, (240, 640)).clip(0, 255).astype(np.uint8), (0, 0), 1.0)
    d = {name: analyze.measure_quality(g)["blur_dir"] for name, g in
         (("sharp", sharp), ("gauss", cv.GaussianBlur(sharp, (0, 0), 2.0)),
          ("motion0", motion_blur(sharp, 15, 0)), ("motion90", motion_blur(sharp, 15, 90)))}
    assert all(isinstance(v, float) for v in d.values()), d
    assert d["sharp"] > 0.8 and d["gauss"] > 0.8, f"방향이 고른 결·가우시안 흐림의 blur_dir이 작음 {d}"
    assert d["motion0"] < 0.2 and d["motion90"] < 0.2, f"움직임 흐림의 blur_dir이 작아지지 않음 {d}"
    assert np.isnan(analyze.measure_quality(np.full((50, 50), 100, np.uint8))["blur_dir"])


def test_blur_dir_off_same():
    """USE_BLUR_DIR가 꺼져 있으면 blur_dir이 아무리 작아도 고르는 단계가 그대로이고(v9와 같음), 켜면 blur_dir < BLUR_DIR_MIN일 때만
    샤프닝을 더한다. blur_dir이 NaN이거나 키가 없으면 켜도 샤프닝하지 않는다."""
    low, high = good_quality(blur_dir=0.01), good_quality(blur_dir=0.99)
    with config_values(USE_BLUR_DIR=False):
        assert analyze.choose_steps(low) == analyze.NO_STEPS
    with config_values(USE_BLUR_DIR=True, BLUR_DIR_MIN=0.3):
        assert analyze.choose_steps(low) == {**analyze.NO_STEPS, "sharpen": True}
        assert analyze.choose_steps(high) == analyze.NO_STEPS
        assert analyze.choose_steps(good_quality(blur_dir=float("nan"))) == analyze.NO_STEPS
        assert analyze.choose_steps(good_quality()) == analyze.NO_STEPS


def test_night_denoise_switch():
    """야간 잡티 제거(보완 3-5): NIGHT_DENOISE가 꺼져 있으면 지금과 똑같고(가우시안), 켜면 '잡티 제거 + 밝게 하는 감마(< 1)'인 영상만
    가우시안 대신 NLM을 쓴다. 잡티 제거만 고른 영상(감마 없음·어둡게 하는 감마)은 켜도 그대로다. 받은 영상은 바꾸지 않는다."""
    rng = np.random.default_rng(5)
    dark = (rng.normal(40, 10, (120, 200)).clip(0, 255)).astype(np.uint8)
    cv.line(dark, (10, 60), (190, 60), 10, 3)
    before = dark.copy()
    night = {**analyze.NO_STEPS, "denoise": True, "tone": "gamma", "gamma": 0.5}
    plain = {**analyze.NO_STEPS, "denoise": True}
    bright = {**analyze.NO_STEPS, "denoise": True, "tone": "gamma", "gamma": 2.0}
    with config_values(NIGHT_DENOISE=False):
        off = {k: preprocess.preprocess(dark, s) for k, s in (("night", night), ("plain", plain), ("bright", bright))}
    with config_values(NIGHT_DENOISE=True):
        on = {k: preprocess.preprocess(dark, s) for k, s in (("night", night), ("plain", plain), ("bright", bright))}
    table = np.clip(np.round((np.arange(256) / 255.0) ** 0.5 * 255), 0, 255).astype(np.uint8)
    assert np.array_equal(off["night"], cv.LUT(cv.GaussianBlur(dark, C.GAUSS_KSIZE, C.GAUSS_SIGMA), table)), "끔인데 가우시안 + 감마가 아님"
    assert not np.array_equal(on["night"], off["night"]), "켰는데 야간 영상의 결과가 같음(NLM을 쓰지 않음)"
    assert on["night"].shape == dark.shape and on["night"].dtype == np.uint8
    assert np.array_equal(on["plain"], off["plain"]) and np.array_equal(on["bright"], off["bright"]), "야간이 아닌 영상까지 바뀜"
    assert np.array_equal(dark, before), "받은 영상을 바꿈"


def test_noise_aggregate_texture():
    """잡티 지표는 거친 골재 결(크기가 여러 가지인 알갱이, 잡음 없음)을 잡음으로 세지 않는다(보완 3-2). 같은 영상에 σ 2 × NOISE_SIGMA_MAX
    잡음을 더하면 기준 위다. 결만 있는 깨끗한 사진에 가우시안을 고르면 균열까지 뿌예진다(own_19_none_normal의 사례)."""
    rng = np.random.default_rng(7)
    gray = np.full((240, 640), 120, np.float32)
    for r, n in ((1, 6000), (2, 2500), (4, 600)):         # 골재 알갱이: 반지름 1·2·4px, 밝기 ±40
        ys, xs, vs = rng.integers(0, 240, n), rng.integers(0, 640, n), rng.integers(-40, 41, n)
        for y, x, v in zip(ys, xs, vs):
            cv.circle(gray, (int(x), int(y)), r, float(120 + v), -1)
    clean = cv.GaussianBlur(gray, (0, 0), 0.8).clip(0, 255).astype(np.uint8)
    q = analyze.measure_quality(clean)["noise"]
    assert q < C.NOISE_SIGMA_MAX, f"잡음 없는 골재 결의 잡티 {q:.2f}가 기준 {C.NOISE_SIGMA_MAX} 이상"
    noisy = (clean.astype(np.float32) + rng.normal(0, 2 * C.NOISE_SIGMA_MAX, clean.shape)).clip(0, 255).astype(np.uint8)
    assert analyze.measure_quality(noisy)["noise"] > C.NOISE_SIGMA_MAX, "잡음을 더했는데 잡티가 기준 이하"


def test_blur_ratio_provided():
    """사람이 흐리다고 판정한 제공 3장만 blur_ratio < BLUR_RATIO_MAX이고, 나머지 10장은 흐림으로 잡히지 않는다(roi.csv 띠).
    사진이 없으면 건너뛴다. 같은 장소 5쌍(blur vs normal)은 시험이 아니라 E5의 상태 판정 검증으로 C가 보고한다."""
    blurry = {"China_MotorBike_002209.jpg", "Japan_000156.jpg", "United_States_004952.jpg"}
    files = sorted(glob.glob("data/provided/*.jpg"))
    if len(files) < 13:
        raise unittest.SkipTest("data/provided 사진 13장이 없음")
    table = io_utils.load_roi_table("data/roi.csv")
    wrong = []
    for path in files:
        name = os.path.basename(path)
        roi = table.get(name, {"top": C.ROI_TOP_DEFAULT, "bottom": C.ROI_BOTTOM_DEFAULT})
        band, _ = io_utils.crop_roi(io_utils.resize_width(io_utils.load_image(path)), roi["top"], roi["bottom"])
        ratio = analyze.measure_quality(cv.cvtColor(band, cv.COLOR_BGR2GRAY))["blur_ratio"]
        if (ratio < C.BLUR_RATIO_MAX) != (name in blurry):
            wrong.append(f"{name} {ratio:.2f}")
    assert not wrong, f"사람 판정과 다른 영상(blur_ratio): {wrong}"


def test_steps_to_text():
    """텍스트 변환 테스트"""
    steps = {"denoise": True, "tone": "gamma", "gamma": 0.53, "sharpen": True}
    text = analyze.steps_to_text(steps)
    assert "가우시안" in text and "감마 0.53" in text and "샤프닝" in text

# ==========================================
# 3. preprocess.py 및 통합 연동 테스트
# ==========================================
def test_preprocess_no_steps():
    """NO_STEPS 시 원본 동일 반환 테스트"""
    dummy_gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    out = preprocess.preprocess(dummy_gray, analyze.NO_STEPS)
    # 배열이 완전히 동일한지 확인
    np.testing.assert_array_equal(dummy_gray, out)

def test_gamma_direction():
    """어두운 노면에 고른 감마는 밝게, 밝은 노면에 고른 감마는 어둡게 만든다 (계산과 적용의 방향이 같음)."""
    rng = np.random.default_rng(0)
    for mean, brighter in ((40, True), (215, False)):
        gray = rng.normal(mean, 10, (60, 80)).clip(0, 255).astype(np.uint8)
        steps = analyze.choose_steps(analyze.measure_quality(gray))
        assert steps["tone"] == "gamma", steps
        out = preprocess.preprocess(gray, {**analyze.NO_STEPS, "tone": "gamma", "gamma": steps["gamma"]})
        moved_to_middle = abs(out.mean() - 127.5) < abs(gray.mean() - 127.5)
        assert (out.mean() > gray.mean()) == brighter and moved_to_middle, (gray.mean(), steps["gamma"], out.mean())


def test_preprocess_follows_config():
    """preprocess의 가우시안·CLAHE·샤프닝 세기는 config 값을 따른다."""
    rng = np.random.default_rng(1)
    # 대비가 낮고 충분히 큰 영상: CLAHE의 실제 한도는 clip × 타일 픽셀 수 / 256(정수)이라 작은 영상에선 clip 값이 같아짐
    gray = rng.normal(120, 8, (240, 320)).clip(0, 255).astype(np.uint8)
    for step, key, a, b in (({"denoise": True}, "GAUSS_SIGMA", 0.5, 2.0),
                            ({"tone": "clahe"}, "CLAHE_CLIP", 1.0, 4.0),
                            ({"sharpen": True}, "SHARPEN_AMOUNT", 0.5, 3.0)):
        steps = {**analyze.NO_STEPS, **step}
        with config_values(**{key: a}):
            out_a = preprocess.preprocess(gray, steps)
        with config_values(**{key: b}):
            out_b = preprocess.preprocess(gray, steps)
        assert not np.array_equal(out_a, out_b), f"{key}를 바꿔도 결과가 같음"


def test_pipeline_integration():
    """io_utils -> analyze -> preprocess 전체 파이프라인 흐름 연동 테스트"""
    # 1. 이미지 로드 대용 (임의의 3채널 노이즈 이미지 생성)
    dummy_img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
    
    # 2. 리사이즈 및 크롭
    resized = io_utils.resize_width(dummy_img, width=150)
    cropped, y0 = io_utils.crop_roi(resized, top=0.3, bottom=0.9)
    
    # 3. 흑백 변환 및 상태 측정
    gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    quality = analyze.measure_quality(gray)
    
    # 4. 스텝 선택 (강제로 모든 스텝을 켜서 오류가 안 나는지 테스트)
    steps = {
        "denoise": True, 
        "tone": "clahe", 
        "gamma": 1.0, 
        "sharpen": True
    }
    
    # 5. 전처리 적용
    processed = preprocess.preprocess(gray, steps)
    
    # 6. 결과 검증
    assert processed.shape == gray.shape
    assert processed.dtype == np.uint8
    # 필터가 적용되었으므로 원본 gray와는 값이 달라져야 함
    assert not np.array_equal(gray, processed)


def test_show_preprocess_pick_image():
    """show_preprocess를 인자 없이 실행하면 data 아래 사진(훈련 폴더 제외)을 번호로 고르고,
    잘못된 번호는 다시 묻고, 빈칸·입력 끝이면 None. 경로를 인자로 주는 방식도 그대로."""
    import io
    import tempfile
    from tools import show_preprocess as sp

    with tempfile.TemporaryDirectory() as root:
        for rel in ("a.jpg", "provided/b.jpg", "own/c.jpg", "train/img/d.jpg", "provided/note.txt"):
            path = os.path.join(root, *rel.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").close()
        listed = [os.path.relpath(p, root).replace(os.sep, "/") for p in sp.list_images(root)]
        assert listed == ["a.jpg", "own/c.jpg", "provided/b.jpg"], listed

        answers = iter(["x", "9", "3"])

        def raise_eof(prompt):
            raise EOFError

        with contextlib.redirect_stdout(io.StringIO()):
            assert sp.pick_image(root, lambda prompt: next(answers)) == os.path.join(root, "provided", "b.jpg")
            assert sp.pick_image(root, lambda prompt: "q") is None
            assert sp.pick_image(root, raise_eof) is None
    assert sp.parse_args([]).image is None
    assert sp.parse_args(["x.jpg"]).image == "x.jpg"


def test_shape_grid_small():
    """격자 탐색(보완 5-1): 손으로 만든 사진 3장(균열 정답·무손상·포트홀 정답)의 영역에서 조합마다 분류·τ 없음 지표가 손으로 센 값과 같다.
    지금 값(4, 2.5, 0.6): 균열(ci 6) 맞힘, 무손상의 영역(ci 3.2)은 어느 종류도 아님, 포트홀(ci 2.2) 맞힘 → 민감도 1, 특이도 1.
    (3.0, 2.0, 0.6): 무손상의 ci 3.2가 균열이 되어 특이도 0, 포트홀 ci 2.2는 상한 2.0 밖이라 놓쳐 민감도 0.5.
    고르기는 fp2 민감도 → fp1 민감도 → 지금 값에 가까운 쪽."""
    from tools import shape_scatter as ss
    region = lambda ci, sol, c, box: {"circ_inv": ci, "solidity": sol, "contrast": c, "area": 100, "bbox": box}  # noqa: E731
    regions = {"a.jpg": [region(6.0, 0.3, 5.0, (10, 10, 20, 20))], "b.jpg": [region(3.2, 0.8, 3.0, (300, 300, 20, 20))],
               "c.jpg": [region(2.2, 0.9, 4.0, (205, 205, 20, 20))]}
    gts = {"a.jpg": [(0, 0, 100, 100, "crack")], "b.jpg": [], "c.jpg": [(200, 200, 50, 50, "pothole")]}
    heights = {n: 480 for n in regions}
    with config_values(SCORE_FORMULA="margin"):
        rows = ss.grid_rows(regions, gts, heights, [(4.0, 2.5, 0.6), (3.0, 2.0, 0.6)])
    now, loose = rows
    assert (now["none_sensitivity"], now["none_specificity"], now["none_fp_per_image"]) == (1.0, 1.0, 0.0), now
    assert (loose["none_sensitivity"], loose["none_specificity"]) == (0.5, 0.0) and abs(loose["none_fp_per_image"] - 1 / 3) < 1e-9, loose
    assert ss.pick_grid(rows) is now
    assert all(c > p for c, p, _ in ss.grid_combos()), "격자에 균열 하한 ≤ 포트홀 상한인 조합이 있음"


def test_new_tools_args():
    """A의 새 도구 뼈대(노면 마스크 확인 그림·형태 산점도)가 import되고, 인자 없이 기본값으로, 고른 값은 그대로 읽힌다."""
    from tools import shape_scatter, show_road_mask

    args = show_road_mask.parse_args([])
    assert args.set == "dev39" and args.out == show_road_mask.OUT_DIR, args
    assert show_road_mask.parse_args(["--set", "diag35"]).set == "diag35"
    args = show_road_mask.parse_args(["--compare"])          # 보완 4-3: v1·v2 나란히, 기본 저장 폴더가 따로
    assert args.compare is True and args.out == show_road_mask.COMPARE_DIR and show_road_mask.parse_args([]).compare is False, args
    args = shape_scatter.parse_args([])
    assert args.y == "both" and args.grid is False and args.limit is None and args.out == shape_scatter.OUT_DIR, args
    args = shape_scatter.parse_args(["--y", "dt_width", "--grid", "--limit", "20"])
    assert (args.y, args.grid, args.limit) == ("dt_width", True, 20), args


def main():
    """이 파일의 test_ 함수를 모두 돌려 실패를 모아 보여 준다. 실패가 있으면 종료 코드 1.
    @todo로 표시한 시험(구현 전 기능)은 건너뛰고 몇 건인지 따로 보여 준다."""
    fails, skips = [], []
    for name, fn in list(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn()
        except unittest.SkipTest as e:
            skips.append(f"{name}: {e}")
        except Exception as e:
            line = traceback.extract_tb(e.__traceback__)[-1].line     # 메시지 없는 assert도 어느 줄인지 보이게
            fails.append(f"{name}: {type(e).__name__} {e} | {line}")
    if fails:
        print(f"A 시험 실패 {len(fails)}건")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("A 시험 통과" + (f" (건너뜀 {len(skips)}건: 구현 전 기능)" if skips else ""))
    for s in skips:
        print("  · 건너뜀", s)


if __name__ == "__main__":
    main()
