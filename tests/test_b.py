"""파트 B의 후보 검출과 평가 함수를 검사한다.
사용법: 맨 위 폴더에서  python tests/test_b.py   (pytest로 돌려도 됨)"""
import sys
import tempfile
import traceback
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.detect import describe_regions, detect_cracks, detect_potholes  # noqa: E402
from src.evaluate import load_gt, iou, count_matches, precision_recall  # noqa: E402

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




def asphalt(seed=0):
    """잔무늬가 있는 아스팔트 흉내 영상 (평균 120)."""
    rng = np.random.default_rng(seed)

    noise = rng.normal(120, 12, (240, 640))

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


def main():
    """이 파일의 test_ 함수를 모두 돌려 실패를 모아 보여 준다. 실패가 있으면 종료 코드 1."""
    fails = []

    for name, fn in list(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue

        try:
            fn()
        except Exception as e:
            # 메시지 없는 assert도 어느 줄인지 보이게
            line = traceback.extract_tb(e.__traceback__)[-1].line
            fails.append(f"{name}: {type(e).__name__} {e} | {line}")

    if fails:
        print(f"파트 B 테스트 실패 {len(fails)}건")

        for f in fails:
            print("  -", f)

        sys.exit(1)

    print("파트 B 테스트 통과")


if __name__ == "__main__":
    main()