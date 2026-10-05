"""(파트 A) 전처리 파이프라인(io_utils, analyze, preprocess) 통합 및 단위 테스트
사용법: 맨 위 폴더에서  python tests/test_a.py   (pytest 없이 실행되고, pytest로 돌려도 됨)"""
import contextlib
import os
import sys
import traceback

import numpy as np
import cv2 as cv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import io_utils, analyze, preprocess  # noqa: E402
from src import config as C  # noqa: E402

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


def main():
    """이 파일의 test_ 함수를 모두 돌려 실패를 모아 보여 준다. 실패가 있으면 종료 코드 1."""
    fails = []
    for name, fn in list(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn()
        except Exception as e:
            line = traceback.extract_tb(e.__traceback__)[-1].line     # 메시지 없는 assert도 어느 줄인지 보이게
            fails.append(f"{name}: {type(e).__name__} {e} | {line}")
    if fails:
        print(f"A 시험 실패 {len(fails)}건")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("A 시험 통과")


if __name__ == "__main__":
    main()
