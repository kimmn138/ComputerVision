"""(파트 B) 교수 제공 훈련 데이터 전체에서 검출 파라미터 후보를 정량 평가한다.
사용법: 맨 위 폴더에서  python tools/evaluate_train.py"""

import csv
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.detect import detect_cracks, detect_potholes  # noqa: E402
from src.evaluate import (  # noqa: E402
    count_matches,
    load_gt,
    precision_recall,
)


IMAGE_DIR = Path("data/train/img")
GT_DIR = Path("gt/train")
RESULT_DIR = Path("results/tuning")

DETAIL_PATH = RESULT_DIR / "train_detail.csv"
SUMMARY_PATH = RESULT_DIR / "train_summary.csv"


# ---------------------------------------------------------
# 파라미터 후보
#
# config.py 파일은 수정하지 않는다.
# 실행 중 C.xxx 값만 임시 변경한 뒤 반드시 원복한다.
# ---------------------------------------------------------

PARAM_SETS = [
    {
        "name": "base",

        "CANNY_LOW": 50,
        "CANNY_HIGH": 150,
        "CRACK_CLOSE_KSIZE": 5,
        "CRACK_MIN_AREA": 80,
        "CRACK_MIN_ELONG": 4.0,
        "CRACK_MAX_FILL": 0.10,

        "DOG_SIGMAS": (3, 12),
        "DOG_THRESH": 12,
        "POT_OPEN_KSIZE": 5,
        "POT_MIN_AREA": 150,
        "POT_MAX_AREA_RATIO": 0.20,
        "POT_MAX_ELONG": 3.0,
        "POT_MIN_SOLIDITY": 0.60,
    },

    {
        "name": "crack_loose",

        "CANNY_LOW": 50,
        "CANNY_HIGH": 150,
        "CRACK_CLOSE_KSIZE": 5,
        "CRACK_MIN_AREA": 60,
        "CRACK_MIN_ELONG": 3.0,
        "CRACK_MAX_FILL": 0.15,

        "DOG_SIGMAS": (3, 12),
        "DOG_THRESH": 12,
        "POT_OPEN_KSIZE": 5,
        "POT_MIN_AREA": 150,
        "POT_MAX_AREA_RATIO": 0.20,
        "POT_MAX_ELONG": 3.0,
        "POT_MIN_SOLIDITY": 0.60,
    },

    {
        "name": "crack_strict",

        "CANNY_LOW": 50,
        "CANNY_HIGH": 150,
        "CRACK_CLOSE_KSIZE": 5,
        "CRACK_MIN_AREA": 120,
        "CRACK_MIN_ELONG": 5.0,
        "CRACK_MAX_FILL": 0.08,

        "DOG_SIGMAS": (3, 12),
        "DOG_THRESH": 12,
        "POT_OPEN_KSIZE": 5,
        "POT_MIN_AREA": 150,
        "POT_MAX_AREA_RATIO": 0.20,
        "POT_MAX_ELONG": 3.0,
        "POT_MIN_SOLIDITY": 0.60,
    },

    {
        "name": "pothole_loose",

        "CANNY_LOW": 50,
        "CANNY_HIGH": 150,
        "CRACK_CLOSE_KSIZE": 5,
        "CRACK_MIN_AREA": 80,
        "CRACK_MIN_ELONG": 4.0,
        "CRACK_MAX_FILL": 0.10,

        "DOG_SIGMAS": (3, 12),
        "DOG_THRESH": 10,
        "POT_OPEN_KSIZE": 5,
        "POT_MIN_AREA": 100,
        "POT_MAX_AREA_RATIO": 0.25,
        "POT_MAX_ELONG": 4.0,
        "POT_MIN_SOLIDITY": 0.50,
    },

    {
        "name": "pothole_strict",

        "CANNY_LOW": 50,
        "CANNY_HIGH": 150,
        "CRACK_CLOSE_KSIZE": 5,
        "CRACK_MIN_AREA": 80,
        "CRACK_MIN_ELONG": 4.0,
        "CRACK_MAX_FILL": 0.10,

        "DOG_SIGMAS": (3, 12),
        "DOG_THRESH": 15,
        "POT_OPEN_KSIZE": 5,
        "POT_MIN_AREA": 200,
        "POT_MAX_AREA_RATIO": 0.15,
        "POT_MAX_ELONG": 2.5,
        "POT_MIN_SOLIDITY": 0.70,
    },
]


def resize_width(img, width=C.TARGET_WIDTH):
    """영상 비율을 유지하면서 프로젝트 기준 가로 크기로 변환한다."""
    h, w = img.shape[:2]

    if w == width:
        return img.copy()

    scale = width / w
    new_h = round(h * scale)

    return cv.resize(
        img,
        (width, new_h),
        interpolation=cv.INTER_AREA,
    )


def candidate_boxes(candidates):
    """검출 후보에서 평가에 필요한 bbox만 추출한다."""
    return [
        tuple(region["bbox"])
        for region in candidates
    ]


def split_gt(gt):
    """GT 목록을 균열과 포트홀 박스로 분리한다."""
    cracks = []
    potholes = []

    for x, y, w, h, kind in gt:
        box = (x, y, w, h)

        if kind == "crack":
            cracks.append(box)

        elif kind == "pothole":
            potholes.append(box)

    return cracks, potholes


def backup_params():
    """튜닝 중 변경되는 config 파라미터를 백업한다."""
    keys = {
        key
        for params in PARAM_SETS
        for key in params
        if key != "name"
    }

    return {
        key: getattr(C, key)
        for key in keys
    }


def apply_param_set(params):
    """한 파라미터 후보를 실행 중 config 객체에 적용한다."""
    for key, value in params.items():
        if key == "name":
            continue

        setattr(C, key, value)


def restore_params(backup):
    """실행 전 config 파라미터로 복원한다."""
    for key, value in backup.items():
        setattr(C, key, value)


def evaluate_one(image_path):
    """훈련 영상 한 장의 균열·포트홀 검출 결과를 GT와 비교한다."""
    bgr = cv.imread(str(image_path))

    if bgr is None:
        raise FileNotFoundError(image_path)

    bgr = resize_width(bgr)

    gray = cv.cvtColor(
        bgr,
        cv.COLOR_BGR2GRAY,
    )

    cracks, _ = detect_cracks(gray)
    potholes, _ = detect_potholes(gray)

    gt_path = GT_DIR / f"{image_path.stem}.csv"
    gt = load_gt(gt_path)

    if gt is None:
        return None

    gt_cracks, gt_potholes = split_gt(gt)

    crack_counts = count_matches(
        candidate_boxes(cracks),
        gt_cracks,
        C.IOU_THRESH,
    )

    pothole_counts = count_matches(
        candidate_boxes(potholes),
        gt_potholes,
        C.IOU_THRESH,
    )

    return {
        "gt_crack": len(gt_cracks),
        "pred_crack": len(cracks),

        "crack_tp": crack_counts["tp"],
        "crack_fp": crack_counts["fp"],
        "crack_fn": crack_counts["fn"],

        "gt_pothole": len(gt_potholes),
        "pred_pothole": len(potholes),

        "pothole_tp": pothole_counts["tp"],
        "pothole_fp": pothole_counts["fp"],
        "pothole_fn": pothole_counts["fn"],
    }


def add_counts(total, result, prefix):
    """영상별 TP/FP/FN을 파라미터 세트 전체 합계에 더한다."""
    total["tp"] += result[f"{prefix}_tp"]
    total["fp"] += result[f"{prefix}_fp"]
    total["fn"] += result[f"{prefix}_fn"]


def make_pr(tp, fp, fn):
    """합산 TP/FP/FN으로 precision과 recall을 계산한다."""
    counts = {
        "tp": tp,
        "fp": fp,
        "found": tp,
        "fn": fn,
    }

    return precision_recall(counts)


def metric_text(value):
    """NaN을 CSV에 읽기 쉬운 문자열로 바꾼다."""
    if np.isnan(value):
        return "NaN"

    return f"{value:.6f}"


def save_detail(rows):
    """영상별 세부 평가 결과를 CSV로 저장한다."""
    fieldnames = [
        "params",
        "image",

        "gt_crack",
        "pred_crack",
        "crack_tp",
        "crack_fp",
        "crack_fn",

        "gt_pothole",
        "pred_pothole",
        "pothole_tp",
        "pothole_fp",
        "pothole_fn",
    ]

    with DETAIL_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def save_summary(rows):
    """파라미터 세트별 전체 성능을 CSV로 저장한다."""
    fieldnames = [
        "params",
        "images",

        "crack_tp",
        "crack_fp",
        "crack_fn",
        "crack_precision",
        "crack_recall",

        "pothole_tp",
        "pothole_fp",
        "pothole_fn",
        "pothole_precision",
        "pothole_recall",
    ]

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    """훈련 데이터 전체에서 각 파라미터 후보의 정량 성능을 비교한다."""
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        IMAGE_DIR.glob("*.jpg")
    )

    if not image_paths:
        raise RuntimeError(
            f"훈련 이미지가 없습니다: {IMAGE_DIR}"
        )

    print("=" * 72)
    print("훈련 데이터 전체 평가")
    print("images =", len(image_paths))
    print("GT =", GT_DIR)
    print("=" * 72)

    original_params = backup_params()

    detail_rows = []
    summary_rows = []

    try:
        for params in PARAM_SETS:
            param_name = params["name"]

            apply_param_set(params)

            print()
            print("=" * 72)
            print("parameter set:", param_name)
            print("=" * 72)

            crack_total = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
            }

            pothole_total = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
            }

            evaluated_images = 0

            for index, image_path in enumerate(
                image_paths,
                start=1,
            ):
                result = evaluate_one(image_path)

                if result is None:
                    print(
                        "[GT 없음]",
                        image_path.name,
                    )
                    continue

                evaluated_images += 1

                add_counts(
                    crack_total,
                    result,
                    "crack",
                )

                add_counts(
                    pothole_total,
                    result,
                    "pothole",
                )

                detail_rows.append(
                    {
                        "params": param_name,
                        "image": image_path.name,
                        **result,
                    }
                )

                if index % 50 == 0:
                    print(
                        f"{index}/{len(image_paths)}"
                    )

            crack_precision, crack_recall = make_pr(
                crack_total["tp"],
                crack_total["fp"],
                crack_total["fn"],
            )

            pothole_precision, pothole_recall = make_pr(
                pothole_total["tp"],
                pothole_total["fp"],
                pothole_total["fn"],
            )

            summary = {
                "params": param_name,
                "images": evaluated_images,

                "crack_tp": crack_total["tp"],
                "crack_fp": crack_total["fp"],
                "crack_fn": crack_total["fn"],
                "crack_precision": metric_text(
                    crack_precision
                ),
                "crack_recall": metric_text(
                    crack_recall
                ),

                "pothole_tp": pothole_total["tp"],
                "pothole_fp": pothole_total["fp"],
                "pothole_fn": pothole_total["fn"],
                "pothole_precision": metric_text(
                    pothole_precision
                ),
                "pothole_recall": metric_text(
                    pothole_recall
                ),
            }

            summary_rows.append(summary)

            print()
            print("[crack]")
            print(
                "TP =", crack_total["tp"],
                "FP =", crack_total["fp"],
                "FN =", crack_total["fn"],
            )
            print(
                "precision =",
                metric_text(crack_precision),
            )
            print(
                "recall =",
                metric_text(crack_recall),
            )

            print()
            print("[pothole]")
            print(
                "TP =", pothole_total["tp"],
                "FP =", pothole_total["fp"],
                "FN =", pothole_total["fn"],
            )
            print(
                "precision =",
                metric_text(pothole_precision),
            )
            print(
                "recall =",
                metric_text(pothole_recall),
            )

    finally:
        restore_params(
            original_params
        )

    save_detail(
        detail_rows
    )

    save_summary(
        summary_rows
    )

    print()
    print("=" * 72)
    print("전체 평가 완료")
    print("상세:", DETAIL_PATH)
    print("요약:", SUMMARY_PATH)
    print("=" * 72)


if __name__ == "__main__":
    main()