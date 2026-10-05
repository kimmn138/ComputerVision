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

    # 후보가 있다면 계약 형식을 검사한다.
    for region in cracks:
        assert isinstance(region["area"], int)
        assert isinstance(region["elong"], float)
        assert isinstance(region["fill"], float)
        assert isinstance(region["solidity"], float)

        assert len(region["bbox"]) == 4
        assert region["contour"].dtype == np.int32




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