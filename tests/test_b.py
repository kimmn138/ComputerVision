"""파트 B의 후보 검출과 평가 함수를 검사한다.
사용법: 맨 위 폴더에서  python tests/test_b.py   (pytest로 돌려도 됨)"""
import contextlib
import functools
import inspect
import sys
import tempfile
import traceback
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.detect import describe_regions, detect_cracks, detect_potholes, extract_dark_regions  # noqa: E402
from src.evaluate import load_gt, iou, count_matches, count_center_hits, precision_recall  # noqa: E402

REGION_KEYS = {"area", "elong", "fill", "solidity", "bbox", "contour"}
NEW_REGION_KEYS = {"thickness", "dt_width", "circ_inv", "contrast", "score"}


def todo(reason):
    """아직 구현하지 않은 기능의 시험 표시: 기대값은 적어 두되 지금은 건너뛴다(unittest.SkipTest라 pytest도 건너뜀).
    구현한 사람이 이 줄(@todo)을 지우면 시험이 돈다. 이 시험이 그 작업의 통과 기준이다(docs/tasks/B.md 4장)."""
    def mark(fn):
        @functools.wraps(fn)
        def skipped(*args, **kwargs):
            raise unittest.SkipTest(reason)
        return skipped
    return mark


@contextlib.contextmanager
def config_values(**values):
    """config 값을 잠시 바꿨다가 시험이 끝나면(실패해도) 되돌린다. 예: DETECT_MODE="dark"로 새 방식만 시험."""
    old = {k: getattr(C, k) for k in values}
    for k, v in values.items():
        setattr(C, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(C, k, v)


def test_synthetic_crack():
    """합성 선에서 균열이 검출되고 포트홀은 검출되지 않는지 확인한다."""
    gray = np.full((400, 400), 200, dtype=np.uint8)

    points = np.array([
        [50, 200],
        [100, 195],
        [150, 205],
        [200, 198],
        [250, 207],
        [300, 196],
        [350, 202],
    ], dtype=np.int32)

    cv.polylines(
        gray,
        [points],
        False,
        30,
        thickness=2,
    )

    cracks, edges = detect_cracks(gray)

    assert isinstance(cracks, list)

    assert edges.dtype == np.uint8
    assert edges.shape == gray.shape

    # 합성 선의 경계는 Canny에서 검출되어야 한다.
    assert np.count_nonzero(edges) > 0

    # 거의 곧은 선이므로 균열 후보가 있어야 한다.
    assert len(cracks) >= 1

    # 후보의 계약 형식을 검사한다.
    for region in cracks:
        assert isinstance(region["area"], int)
        assert isinstance(region["elong"], float)
        assert isinstance(region["fill"], float)
        assert isinstance(region["solidity"], float)

        assert len(region["bbox"]) == 4
        assert region["contour"].dtype == np.int32




def asphalt(seed=0, shape=(240, 640)):
    """잔무늬가 있는 아스팔트 흉내 영상 (평균 120). shape은 (세로, 가로)."""
    rng = np.random.default_rng(seed)

    noise = rng.normal(120, 12, shape)

    return cv.GaussianBlur(
        noise.clip(0, 255).astype(np.uint8),
        (0, 0),
        1.2,
    )


def test_crack_shapes():
    """곧은 균열(가로·세로)·구불구불한 균열·갈래 균열을 모두 균열 후보로 찾는지 확인한다."""
    xs = np.arange(40, 600, 4)
    wave = np.stack([xs, 120 + 25 * np.sin(xs / 30)], 1).astype(np.int32)

    shapes = {
        "곧은 가로": [np.array([[40, 100], [600, 140]], np.int32)],
        "곧은 세로": [np.array([[300, 5], [330, 235]], np.int32)],
        "구불구불": [wave],
        "갈래": [
            np.array([[100, 60], [300, 180], [520, 90]], np.int32),
            np.array([[300, 180], [330, 235]], np.int32),
        ],
    }

    missed = []

    for name, lines in shapes.items():
        gray = asphalt()
        cv.polylines(gray, lines, False, 60, 2)

        cracks, _ = detect_cracks(gray)

        if not cracks:
            missed.append(name)

    assert not missed, f"균열을 놓침: {missed}"


def test_bright_lines_not_crack():
    """주변보다 밝은 선(가는 흰 선·넓은 흰 차선·횡단보도 점선)은 균열 후보에서 빠지고,
    진하거나 옅은 어두운 균열은 남는지 확인한다. 후보에는 darkness(주변보다 어두운 정도)가 붙는다."""
    bright = {}

    gray = asphalt()
    cv.line(gray, (40, 100), (600, 140), 230, 3)
    bright["가는 흰 선"] = gray

    gray = asphalt()
    cv.rectangle(gray, (0, 100), (639, 115), 230, -1)
    bright["넓은 흰 차선"] = gray

    gray = asphalt()
    for x in range(40, 600, 40):
        cv.rectangle(gray, (x, 90), (x + 12, 150), 230, -1)
    bright["횡단보도 점선"] = gray

    wrong = {name: len(detect_cracks(img)[0]) for name, img in bright.items()}

    assert not any(wrong.values()), f"밝은 선이 균열로 잡힘: {wrong}"

    for value, width in ((60, 2), (90, 1)):            # 진한 균열, 옅은 균열
        gray = asphalt()
        cv.line(gray, (40, 100), (600, 140), value, width)

        cracks, _ = detect_cracks(gray)

        assert len(cracks) >= 1, f"밝기 {value} 균열을 놓침"
        assert all(r["darkness"] >= C.CRACK_MIN_DARKNESS for r in cracks)


def test_pothole_not_crack():
    """둥근 포트홀과 무늬만 있는 노면은 균열 후보가 되지 않는지 확인한다."""
    for radius in (40, 90):
        gray = asphalt()
        cv.circle(gray, (320, 120), radius, 50, -1)

        cracks, _ = detect_cracks(gray)

        assert len(cracks) == 0, f"반지름 {radius} 포트홀이 균열 {len(cracks)}개로 잡힘"

    cracks, _ = detect_cracks(asphalt())

    assert len(cracks) == 0


def test_describe_crack_region():
    """가늘고 긴 이진 영역의 모양 지표가 균열 특성을 갖는지 확인한다."""
    mask = np.zeros((400, 400), dtype=np.uint8)

    cv.line(
        mask,
        (50, 50),
        (350, 300),
        255,
        thickness=2,
    )

    regions = describe_regions(
        mask,
        min_pixels=1,
    )

    assert len(regions) >= 1

    region = max(
        regions,
        key=lambda r: r["area"],
    )

    assert region["area"] > 0
    assert region["elong"] > 1.0
    assert 0.0 <= region["fill"] <= 1.0
    assert 0.0 <= region["solidity"] <= 1.0

def test_synthetic_pothole():
    """합성 원 영상에서 포트홀이 검출되는지 확인한다."""
    gray = np.zeros((400, 400), dtype=np.uint8)
    gray[:] = 200

    cv.circle(gray, (200, 200), 40, 30, -1)

    potholes, _ = detect_potholes(gray)

    assert len(potholes) >= 1


def test_blank_image():
    """빈 영상에서 균열과 포트홀이 검출되지 않는지 확인한다."""
    gray = np.full((400, 400), 200, dtype=np.uint8)

    cracks, _ = detect_cracks(gray)
    potholes, _ = detect_potholes(gray)

    assert len(cracks) == 0
    assert len(potholes) == 0

def test_load_gt():
    """정답 CSV의 정상·빈 파일·없는 파일 처리를 확인한다."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # 정상 CSV
        path = tmp_dir / "gt.csv"
        path.write_text(
            "x,y,w,h,kind\n"
            "10,20,30,40,crack\n"
            "50,60,70,80,pothole\n",
            encoding="utf-8-sig",
        )

        gt = load_gt(path)

        assert gt == [
            (10, 20, 30, 40, "crack"),
            (50, 60, 70, 80, "pothole"),
        ]

        # 헤더만 있는 CSV
        empty_path = tmp_dir / "empty.csv"
        empty_path.write_text(
            "x,y,w,h,kind\n",
            encoding="utf-8-sig",
        )

        assert load_gt(empty_path) == []

        # 존재하지 않는 CSV
        missing_path = tmp_dir / "missing.csv"

        assert load_gt(missing_path) is None

def test_load_gt_missing():
    """정답 CSV가 없으면 None인지 확인한다."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "missing.csv"

        assert load_gt(path) is None


def test_load_gt_empty():
    """헤더만 있는 CSV는 빈 리스트인지 확인한다."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "empty.csv"

        path.write_text(
            "x,y,w,h,kind\n",
            encoding="utf-8-sig",
        )

        assert load_gt(path) == []

def test_iou():
    """동일·비중첩 박스의 IoU를 확인한다."""
    box = (10, 10, 100, 100)

    assert iou(box, box) == 1.0
    assert iou(box, (200, 200, 50, 50)) == 0.0


def test_count_matches():
    """예측과 정답 박스의 일대일 매칭 결과를 확인한다."""
    pred = [
        (10, 10, 100, 100),
        (300, 300, 50, 50),
    ]

    gt = [
        (10, 10, 100, 100),
    ]

    counts = count_matches(pred, gt, 0.3)

    assert counts == {
        "tp": 1,
        "fp": 1,
        "found": 1,
        "fn": 0,
    }


def test_nan():
    """정밀도와 재현율의 분모가 0일 때 NaN인지 확인한다."""
    counts = {
        "tp": 0,
        "fp": 0,
        "found": 0,
        "fn": 0,
    }

    precision, recall = precision_recall(counts)

    assert np.isnan(precision)
    assert np.isnan(recall)


def test_count_center_hits():
    """검출 중심이 정답 박스 안이면 맞힘: tp·fp는 검출 기준(한 정답에 여러 개 가능), found·fn은 정답 기준."""
    gt = [
        (0, 0, 20, 20, "crack"),
        (50, 50, 10, 10, "crack"),
    ]

    pred = [
        (0, 0, 10, 10),       # 중심 (5, 5): 첫 정답 안
        (8, 8, 4, 4),         # 중심 (10, 10): 첫 정답 안 (같은 정답에 두 번째 조각)
        (100, 100, 10, 10),   # 중심이 어느 정답에도 없음
    ]

    counts = count_center_hits(pred, gt)

    assert counts == {"tp": 2, "fp": 1, "found": 1, "fn": 1}, counts

    assert count_center_hits([], gt) == {"tp": 0, "fp": 0, "found": 0, "fn": 2}
    assert count_center_hits(pred, []) == {"tp": 0, "fp": 3, "found": 0, "fn": 0}

    precision, recall = precision_recall(counts)

    assert abs(precision - 2 / 3) < 1e-9 and recall == 0.5


def test_evaluate_train_roi_coordinates():
    """evaluate_train이 ROI로 자른 노면에서 찾은 후보를 640px 전체 좌표(y + y0)로 옮겨 정답과 비교하는지 확인한다."""
    from tools import evaluate_train

    gray = np.vstack([asphalt(seed=1), asphalt(seed=2)])     # 480 × 640, 위·아래 절반 모두 노면 무늬
    cv.line(gray, (60, 360), (580, 380), 60, 2)                # 아래 절반(ROI 안)에 곧은 균열
    bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    gt = [(50, 350, 540, 40, "crack")]                         # 640px 전체 영상 좌표
    roi = {"top": 0.5, "bottom": 1.0, "condition": ""}

    result = evaluate_train.evaluate_one(bgr, gt, roi, False)

    assert result["gt_crack"] == 1 and result["pred_crack"] >= 1, result
    assert result["crack_ctr_found"] == 1 and result["crack_ctr_tp"] >= 1, result

    # 정답을 ROI 기준(y0를 빼지 않은 채 위로 240 올린 자리)에 두면 맞지 않아야 한다
    moved = evaluate_train.evaluate_one(bgr, [(50, 110, 540, 40, "crack")], roi, False)

    assert moved["crack_ctr_found"] == 0, moved


def test_label_gt_pick_image():
    """label_gt를 인자 없이 실행하면 data 아래 사진(훈련 폴더 제외)을 번호로 고르고,
    잘못된 번호는 다시 묻고, 빈칸·입력 끝이면 None. 경로를 인자로 주는 방식도 그대로."""
    import contextlib
    import io

    from tools import label_gt

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)

        for rel in ("a.jpg", "provided/b.jpg", "own/c.jpg", "train/img/d.jpg", "provided/note.txt"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(b"")

        listed = [p.relative_to(root).as_posix() for p in label_gt.list_images(root)]

        assert listed == ["a.jpg", "own/c.jpg", "provided/b.jpg"], listed

        answers = iter(["x", "9", "3"])

        def raise_eof(prompt):
            raise EOFError

        with contextlib.redirect_stdout(io.StringIO()):
            assert label_gt.pick_image(root, lambda prompt: next(answers)) == root / "provided" / "b.jpg"
            assert label_gt.pick_image(root, lambda prompt: "") is None
            assert label_gt.pick_image(root, raise_eof) is None

    assert label_gt.parse_args([]).image is None
    assert label_gt.parse_args(["x.jpg"]).image == "x.jpg"


# ---------------------------------------------------------
# 10/10 기반: road 인자·DETECT_MODE·후보 새 키 (지금 도는 시험)
# ---------------------------------------------------------

def summary_of(result):
    """(후보, 마스크) 결과를 비교하기 쉬운 값으로: 후보마다 (bbox, 넓이)와 마스크 바이트."""
    regions, mask = result
    return [(r["bbox"], r["area"]) for r in regions], mask.tobytes()


def test_detect_road_arg_legacy():
    """detect_*(gray, road=None): 기본값 None이면 지금과 같고, legacy는 road를 넘겨도 결과가 같다(Plan.md 3.4 (4) 마지막 줄).
    road와 gray는 바꾸지 않는다."""
    gray = asphalt()
    cv.line(gray, (40, 100), (600, 140), 60, 2)
    cv.circle(gray, (320, 170), 30, 50, -1)
    half = np.zeros_like(gray)
    half[:, :320] = 255
    before_gray, before_half = gray.copy(), half.copy()

    for fn in (detect_cracks, detect_potholes):
        param = inspect.signature(fn).parameters.get("road")
        assert param is not None and param.default is None, f"{fn.__name__}에 road=None 인자가 없음"
        base = summary_of(fn(gray))
        for road in (None, np.full_like(gray, 255), half):
            assert summary_of(fn(gray, road)) == base, f"{fn.__name__}: legacy인데 road를 넘기자 결과가 달라짐"
        assert summary_of(fn(gray, road=half)) == base, f"{fn.__name__}: road를 이름으로 넘기지 못함"

    assert np.array_equal(gray, before_gray) and np.array_equal(half, before_half), "detect가 입력 배열을 바꿈"


def test_detect_mode_invalid():
    """DETECT_MODE가 'legacy'·'dark'가 아니면 두 검출 함수 모두 ValueError로 바로 알린다."""
    for fn in (detect_cracks, detect_potholes):
        with config_values(DETECT_MODE="nosuchmode"):
            try:
                fn(asphalt())
                raise AssertionError(f"{fn.__name__}: 잘못된 DETECT_MODE인데 ValueError가 나지 않음")
            except ValueError:
                pass


def test_candidate_new_keys():
    """후보에 thickness·dt_width·circ_inv(양의 유한 실수)와 contrast·score(실수, 지금은 NaN 허용) 키가 있다(약속 표)."""
    gray = asphalt()
    cv.line(gray, (40, 100), (600, 140), 60, 2)
    cv.circle(gray, (320, 170), 30, 50, -1)

    regions = detect_cracks(gray)[0] + detect_potholes(gray)[0]

    assert regions, "시험 영상에서 후보가 하나도 없음"
    for r in regions:
        assert REGION_KEYS | NEW_REGION_KEYS <= r.keys(), sorted(r.keys())
        for key in ("thickness", "dt_width", "circ_inv"):
            assert isinstance(r[key], float) and np.isfinite(r[key]) and r[key] > 0, (key, r[key])
        for key in ("contrast", "score"):
            assert isinstance(r[key], float), (key, r[key])


def test_shape_metrics_rule():
    """합성 모양의 비원형도 c·두께 t가 출발 규칙대로 나온다(Plan.md 3.5 (5)): 선(폭 2·4)·갈래·구불구불은 c ≥ CRACK_MIN_CIRC_INV,
    원(반지름 10·40·90)은 c ≤ POT_MAX_CIRC_INV. t = 2A/P는 선이면 폭(±1), 원이면 반지름(±15%)에 가깝다."""
    def biggest(draw):
        mask = np.zeros((240, 640), np.uint8)
        draw(mask)
        return max(describe_regions(mask, min_pixels=1), key=lambda r: r["area"])

    for width in (2, 4):
        r = biggest(lambda m: cv.line(m, (40, 100), (600, 140), 255, width))
        assert r["circ_inv"] >= C.CRACK_MIN_CIRC_INV, (width, r["circ_inv"])
        assert abs(r["thickness"] - width) <= 1 and abs(r["dt_width"] - width) <= 1, (width, r["thickness"], r["dt_width"])

    for radius in (10, 40, 90):
        r = biggest(lambda m: cv.circle(m, (320, 120), radius, 255, -1))
        assert r["circ_inv"] <= C.POT_MAX_CIRC_INV, (radius, r["circ_inv"])
        assert abs(r["thickness"] - radius) <= 0.15 * radius, (radius, r["thickness"])

    xs = np.arange(40, 600, 4)
    wave = np.stack([xs, 120 + 25 * np.sin(xs / 30)], 1).astype(np.int32)
    branch = [np.array([[100, 60], [300, 180], [520, 90]], np.int32), np.array([[300, 180], [330, 235]], np.int32)]
    for name, lines in (("갈래", branch), ("구불구불", [wave])):
        r = biggest(lambda m: cv.polylines(m, lines, False, 255, 2))
        assert r["circ_inv"] >= C.CRACK_MIN_CIRC_INV, (name, r["circ_inv"])


# ---------------------------------------------------------
# dark 방식 시험 (B의 통과 기준). 구현 전이라 @todo로 건너뛴다.
# 'C 메모리 확인'은 Plan.md 3.4 (2)·3.5 (2) 명세를 그대로 옮긴 시험용 코드로 C가 10/10에 돌려 본 결과다(저장소에 없음).
# ---------------------------------------------------------

def centered_in(regions, center, radius):
    """bbox 중심이 원(center, radius) 안에 있는 후보만 고른다."""
    cx, cy = center
    return [r for r in regions
            if (r["bbox"][0] + r["bbox"][2] / 2 - cx) ** 2 + (r["bbox"][1] + r["bbox"][3] / 2 - cy) ** 2 <= radius ** 2]


def soft_tree_shadow(seed=2):
    """지름 200px, 가장자리가 흐린 나무 그림자(가운데를 40% 어둡게)를 드리운 아스팔트."""
    gray = asphalt(seed).astype(np.float32)
    shade = np.zeros(gray.shape, np.float32)
    cv.circle(shade, (320, 120), 100, 1.0, -1)
    shade = cv.GaussianBlur(shade, (0, 0), 15)
    return (gray * (1 - 0.4 * shade)).clip(0, 255).astype(np.uint8)


RECORDS = []   # 기대값 없이 결과만 남기는 시험(Plan.md 3.4 (4) '기록한다')의 기록. main이 마지막에 보여 준다


def record(name, text):
    """기대값을 두지 않는 시험의 결과를 남긴다. 시험은 통과로 두고, main이 끝에 '기록'으로 보여 준다."""
    RECORDS.append(f"{name}: {text}")


@todo("B WP2: extract_dark_regions 구현 뒤 (Plan.md 3.4 (2))")
def test_dark_extract_format():
    """extract_dark_regions: (영역 list, 같은 크기 0/255 마스크), 영역에 후보 키 + 새 키·darkness, contrast는 양수.
    잡음 σ가 0이 되지 않아 포트홀 하나만 한 영역이 된다(R1). 평평한 영상·노면 픽셀이 없는 road에서도 오류 없음.
    road가 전부 255면 road가 없을 때와 같다(σ를 띠 전체에서 잰 것과 같으므로). 입력은 바꾸지 않는다."""
    gray = asphalt(7)
    cv.circle(gray, (320, 120), 40, 50, -1)
    before = gray.copy()

    regions, mask = extract_dark_regions(gray)

    assert isinstance(regions, list) and mask.dtype == np.uint8 and mask.shape == gray.shape
    assert set(np.unique(mask).tolist()) <= {0, 255}
    assert len(regions) == 1, f"포트홀 하나가 영역 {len(regions)}개가 됨 (σ = 0이면 띠 전체가 한 영역)"
    r = regions[0]
    assert REGION_KEYS | {"thickness", "dt_width", "circ_inv", "contrast", "darkness"} <= r.keys(), sorted(r.keys())
    assert iou(r["bbox"], (280, 80, 81, 81)) >= 0.8 and r["contrast"] > 0, (r["bbox"], r["contrast"])
    assert np.array_equal(gray, before), "extract_dark_regions가 입력을 바꿈"

    assert extract_dark_regions(np.full((240, 640), 120, np.uint8))[0] == []          # C 메모리 확인: 0개, σ = 하한 1.0
    extract_dark_regions(gray, np.zeros_like(gray))                                    # 노면 픽셀이 없어도 오류 없음
    full = extract_dark_regions(gray, np.full_like(gray, 255))
    assert [x["bbox"] for x in full[0]] == [x["bbox"] for x in regions]


@todo("B WP2·WP3: dark 구현 뒤 (Plan.md 3.4 (4) 새 시험 1)")
def test_dark_big_pothole_one_piece():
    """반지름 90 포트홀(띠 320×640)은 조각나지 않고 포트홀 후보 1개가 원을 덮는다(두 창 배경). C 메모리 확인: 통과."""
    gray = asphalt(0, (320, 640))
    cv.circle(gray, (320, 160), 90, 50, -1)

    with config_values(DETECT_MODE="dark"):
        inside = centered_in(detect_potholes(gray)[0], (320, 160), 90)
        cracks_inside = centered_in(detect_cracks(gray)[0], (320, 160), 90)

    assert len(inside) == 1, f"원 안의 포트홀 후보가 {len(inside)}개"
    assert iou(inside[0]["bbox"], (230, 70, 181, 181)) >= 0.8, inside[0]["bbox"]
    assert not cracks_inside, "포트홀이 균열로도 잡힘"


@todo("B WP2·WP3: dark 구현 뒤 (Plan.md 3.4 (4) 새 시험 2)")
def test_dark_stripes_no_pothole():
    """흰 줄무늬(폭 12px, 40px 간격, 횡단보도 점선 시험과 같은 모양) 사이의 아스팔트가 포트홀이 되지 않는다(밝은 표시 억제).
    C 메모리 확인: 0개."""
    gray = asphalt(1)
    for x in range(40, 600, 40):
        cv.rectangle(gray, (x, 90), (x + 12, 150), 230, -1)

    with config_values(DETECT_MODE="dark"):
        potholes, _ = detect_potholes(gray)

    assert not potholes, f"줄무늬 사이가 포트홀 {len(potholes)}개로 잡힘"


@todo("B WP2 (Should): 밝은 표시 억제의 대안(기준 밝기를 띠·노면 중앙값으로) 뒤. 실패하면 한계로 보고 (Plan.md 3.4 (2) 1단계, 3.11)")
def test_dark_wide_stripes_no_pothole():
    """넓은 흰 줄무늬(폭 30px, 30px 간격)는 억제 창(약 44px)의 절반을 넘어 명세 그대로면 사이가 포트홀이 된다.
    C 메모리 확인: 명세 그대로 9개 → 실패. 대안을 넣어도 안 되면 한계로 보고하고 이 시험은 @todo로 둔다."""
    gray = asphalt(1)
    for x in range(20, 620, 60):
        cv.rectangle(gray, (x, 40), (x + 30, 200), 230, -1)

    with config_values(DETECT_MODE="dark"):
        potholes, _ = detect_potholes(gray)

    assert not potholes, f"넓은 줄무늬 사이가 포트홀 {len(potholes)}개로 잡힘"


@todo("B WP2 (Should): dark 구현 뒤 결과 기록. 후보가 남으면 한계로 보고 (Plan.md 3.4 (3)·(4) 새 시험 3)")
def test_dark_soft_shadow():
    """넓고 흐린 그림자(지름 200px, 가운데 40% 어둡게)에서 후보가 남는지 기록한다. 기대값은 두지 않는다(Plan.md 3.4 (4)).
    후보가 남으면 그림자 영상(US_004830·US_005996·직접 촬영)의 결과와 함께 한계로 보고한다.
    C 메모리 확인: 명세 그대로면 포트홀 후보 1개가 남음(c 2.0, solidity 0.94)."""
    gray = soft_tree_shadow()

    with config_values(DETECT_MODE="dark"):
        potholes, _ = detect_potholes(gray)
        cracks, _ = detect_cracks(gray)

    assert isinstance(potholes, list) and isinstance(cracks, list)
    record("test_dark_soft_shadow", f"흐린 그림자 → 포트홀 후보 {len(potholes)}개, 균열 후보 {len(cracks)}개 (기대값 없음)")


@todo("B WP2·WP3 (Should): 덩어리와 선을 나누는 단계를 넣은 뒤. 실패하면 한계로 보고 (Plan.md 3.4 (4) 새 시험 4)")
def test_dark_crack_touching_pothole():
    """포트홀(반지름 35)에 균열이 붙어 있어도 둘 다 검출한다: 원 안에 포트홀 후보 1개(폭이 원을 크게 넘지 않음), 선 쪽에 균열 후보.
    C 메모리 확인: 명세 그대로면 한 영역(c 16.1)으로 합쳐져 균열 1개만 나옴 → 덩어리와 선을 나누는 단계가 필요.
    넣어도 안 되면 한계로 보고하고 이 시험은 @todo로 둔다."""
    gray = asphalt(3)
    cv.circle(gray, (200, 120), 35, 50, -1)
    cv.line(gray, (235, 120), (600, 150), 60, 2)

    with config_values(DETECT_MODE="dark"):
        potholes = centered_in(detect_potholes(gray)[0], (200, 120), 35)
        cracks, _ = detect_cracks(gray)

    assert len(potholes) == 1 and potholes[0]["bbox"][2] <= 90, [p["bbox"] for p in potholes]
    assert any(r["bbox"][0] + r["bbox"][2] / 2 >= 300 for r in cracks), [r["bbox"] for r in cracks]


@todo("B WP2·WP3: dark 구현 뒤 (Plan.md 3.5 (5) 새 시험 1)")
def test_dark_wide_crack():
    """폭 6~8px 긴 균열은 두께가 커도 비원형도로 균열이다(첫 판 't ≤ 4px' 규칙의 실패를 막음). C 메모리 확인: 통과(c 22~28)."""
    for width in (6, 8):
        gray = asphalt(4)
        cv.line(gray, (40, 100), (600, 140), 60, width)

        with config_values(DETECT_MODE="dark"):
            cracks, _ = detect_cracks(gray)
            potholes, _ = detect_potholes(gray)

        assert len(cracks) >= 1 and not potholes, (width, len(cracks), len(potholes))


@todo("B WP2·WP3: 구멍 둘레를 포함한 뒤 (Plan.md 3.5 (1)·(5) 새 시험 2)")
def test_dark_alligator_net():
    """그물 모양 균열(거북등, 16px 칸·선 폭 2)은 균열이고 포트홀이 아니다. 둘레 P에 구멍 윤곽 길이를 더해야 한다.
    C 메모리 확인: 구멍 포함 c 224 → 균열, 바깥 윤곽만 c 3.6 → 버림."""
    gray = asphalt(5)
    for y in range(40, 201, 16):
        cv.line(gray, (220, y), (420, y), 60, 2)
    for x in range(220, 421, 16):
        cv.line(gray, (x, 40), (x, 200), 60, 2)

    with config_values(DETECT_MODE="dark"):
        cracks = centered_in(detect_cracks(gray)[0], (320, 120), 60)
        potholes = centered_in(detect_potholes(gray)[0], (320, 120), 100)

    assert len(cracks) >= 1 and not potholes, (len(cracks), [p["bbox"] for p in potholes])


@todo("B WP2·WP3: dark 구현 뒤, DETECT_MODE를 'dark'로 바꾸기 전에 (Plan.md 3.4 (4) 첫째 줄)")
def test_existing_tests_in_dark_mode():
    """기존 합성 시험(균열 모양·밝은 선·포트홀·빈 영상)을 DETECT_MODE='dark'에서도 모두 통과한다. C 메모리 확인: 통과."""
    with config_values(DETECT_MODE="dark"):
        for fn in (test_synthetic_crack, test_crack_shapes, test_bright_lines_not_crack, test_pothole_not_crack,
                   test_synthetic_pothole, test_blank_image, test_candidate_new_keys):
            fn()


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
            # 메시지 없는 assert도 어느 줄인지 보이게
            line = traceback.extract_tb(e.__traceback__)[-1].line
            fails.append(f"{name}: {type(e).__name__} {e} | {line}")

    if fails:
        print(f"파트 B 테스트 실패 {len(fails)}건")

        for f in fails:
            print("  -", f)

        sys.exit(1)

    print("파트 B 테스트 통과" + (f" (건너뜀 {len(skips)}건: 구현 전 기능)" if skips else ""))

    for s in skips:
        print("  · 건너뜀", s)

    for r in RECORDS:
        print("  · 기록", r)


if __name__ == "__main__":
    main()