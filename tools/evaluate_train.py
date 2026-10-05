"""(파트 B) 교수 제공 훈련 데이터 전체에서 검출 파라미터 후보를 정량 평가한다.
사용법: 맨 위 폴더에서  python tools/convert_json_gt.py (정답 CSV를 먼저 만듦) → python tools/evaluate_train.py
훈련 사진과 정답 CSV 폴더는 config B 구역의 TRAIN_IMAGE_DIR·TRAIN_GT_DIR.

run_experiment와 같은 조건으로 평가한다:
- pipeline.run_pipeline(640px → ROI 자르기 → 전처리 끔/켬 → 검출)을 그대로 쓰고, 끔·켬 두 줄을 모두 낸다.
- ROI는 run_experiment.get_roi와 같이 roi.csv에서 찾고, 없으면 config 기본값(ROI_TOP_DEFAULT·ROI_BOTTOM_DEFAULT).
- 후보 좌표는 run_experiment.to_full_boxes로 640px 전체 영상 좌표로 옮긴다(y + y0).
- ROI 밖에 있는 정답도 빼지 않고 놓친 것(FN)으로 센다.
판정은 두 가지를 함께 낸다:
- iou: IoU ≥ IOU_THRESH, 정답 하나에 검출 하나(evaluate.count_matches)
- ctr: 검출 중심이 정답 박스 안(evaluate.count_center_hits). 큰 정답 박스 안의 조각 검출도 맞힘으로 셈"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiment as rx  # noqa: E402
from src import config as C  # noqa: E402
from src import io_utils, pipeline  # noqa: E402
from src.evaluate import (  # noqa: E402
    count_center_hits,
    count_matches,
    load_gt,
    precision_recall,
)


RESULT_DIR = Path("results/tuning")

DETAIL_PATH = RESULT_DIR / "train_detail.csv"
SUMMARY_PATH = RESULT_DIR / "train_summary.csv"

ROI_PATH = Path("data/roi.csv")

KINDS = ("crack", "pothole")
MODES = (("off", False), ("on", True))     # 전처리 끔·켬 (run_experiment의 preprocess 열과 같은 이름)
CRITERIA = ("iou", "ctr")                  # iou: IoU 일대일, ctr: 검출 중심이 정답 박스 안
COUNT_KEYS = ("tp", "fp", "found", "fn")


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


def evaluate_one(bgr, gt, roi, use_preprocess):
    """훈련 영상 한 장을 run_experiment와 같은 조건(ROI·전처리)으로 검출해 정답과 두 기준으로 비교한다."""
    out = pipeline.run_pipeline(
        bgr,
        roi["top"],
        roi["bottom"],
        use_preprocess,
    )

    truth_by_kind = dict(zip(KINDS, split_gt(gt)))
    regions_by_kind = {
        "crack": out["cracks"],
        "pothole": out["potholes"],
    }

    result = {}

    for kind in KINDS:
        # 후보는 노면(ROI) 기준이므로 정답과 같은 640px 전체 영상 좌표로 옮긴다
        pred = rx.to_full_boxes(regions_by_kind[kind], out["y0"])
        truth = truth_by_kind[kind]

        result[f"gt_{kind}"] = len(truth)
        result[f"pred_{kind}"] = len(pred)

        counts_by_criterion = {
            "iou": count_matches(pred, truth, C.IOU_THRESH),
            "ctr": count_center_hits(pred, truth),
        }

        for criterion, counts in counts_by_criterion.items():
            for key in COUNT_KEYS:
                result[f"{kind}_{criterion}_{key}"] = counts[key]

    return result


def count_columns():
    """영상별 결과의 개수 열 이름: gt_·pred_ 개수와 종류·기준별 tp·fp·found·fn."""
    columns = []

    for kind in KINDS:
        columns += [f"gt_{kind}", f"pred_{kind}"]
        columns += [
            f"{kind}_{criterion}_{key}"
            for criterion in CRITERIA
            for key in COUNT_KEYS
        ]

    return columns


def metric_text(value):
    """NaN을 CSV에 읽기 쉬운 문자열로 바꾼다."""
    if np.isnan(value):
        return "NaN"

    return f"{value:.6f}"


def summarize(param_name, mode, total):
    """한 파라미터 세트·전처리 조건의 합계로 종류·기준별 정밀도·재현율을 계산한다(영상별 비율의 평균이 아님)."""
    row = {
        "params": param_name,
        "preprocess": mode,
        "images": total["images"],
    }

    for kind in KINDS:
        for criterion in CRITERIA:
            prefix = f"{kind}_{criterion}"
            counts = {key: total[f"{prefix}_{key}"] for key in COUNT_KEYS}
            precision, recall = precision_recall(counts)

            row.update({f"{prefix}_{key}": counts[key] for key in COUNT_KEYS})
            row[f"{prefix}_precision"] = metric_text(precision)
            row[f"{prefix}_recall"] = metric_text(recall)

    return row


def summary_fieldnames():
    """요약 표의 열: 파라미터·전처리·영상 수, 종류·기준별 개수와 정밀도·재현율."""
    columns = ["params", "preprocess", "images"]

    for kind in KINDS:
        for criterion in CRITERIA:
            prefix = f"{kind}_{criterion}"
            columns += [f"{prefix}_{key}" for key in COUNT_KEYS]
            columns += [f"{prefix}_precision", f"{prefix}_recall"]

    return columns


def save_csv(path, rows, fieldnames):
    """결과 행을 UTF-8(BOM) CSV로 저장한다."""
    with path.open(
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


def print_summary(summary_rows):
    """요약을 콘솔 표로 보여 준다: 종류마다 IoU 기준과 중심 기준의 정밀도 / 재현율."""
    header = " | ".join(
        f"{kind + ' ' + criterion + ' P / R':>15s}"
        for kind in KINDS
        for criterion in CRITERIA
    )

    print()
    print(f"{'params':15s} {'pre':4s} | {header}")

    for row in summary_rows:
        cells = []

        for kind in KINDS:
            for criterion in CRITERIA:
                precision = float(row[f"{kind}_{criterion}_precision"])
                recall = float(row[f"{kind}_{criterion}_recall"])
                cells.append(f"{precision:.3f} / {recall:.3f}")

        print(f"{row['params']:15s} {row['preprocess']:4s} | " + " | ".join(f"{c:>15s}" for c in cells))


def main():
    """훈련 데이터 전체에서 각 파라미터 후보의 정량 성능을 전처리 끔·켬과 두 판정 기준으로 비교한다."""
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_dir = Path(C.TRAIN_IMAGE_DIR)
    gt_dir = Path(C.TRAIN_GT_DIR)

    image_paths = sorted(
        image_dir.glob("*.jpg")
    )

    if not image_paths:
        raise RuntimeError(
            f"훈련 이미지가 없습니다: {image_dir}\n"
            "훈련 사진을 이 폴더에 두거나, 다른 곳에 있으면 "
            "src/config.py B 구역의 TRAIN_IMAGE_DIR을 그 경로로 바꾸세요."
        )

    # 정답 CSV가 하나도 없으면 사진마다 [GT 없음]만 찍히므로 먼저 멈춘다
    if not any(gt_dir.glob("*.csv")):
        raise RuntimeError(
            f"정답 CSV가 없습니다: {gt_dir}\n"
            "먼저 python tools/convert_json_gt.py 로 JSON 정답을 CSV로 바꾸세요 "
            "(JSON 폴더는 config B 구역의 TRAIN_ANN_DIR)."
        )

    roi_table = io_utils.load_roi_table(ROI_PATH)

    print("=" * 72)
    print("훈련 데이터 전체 평가 (run_experiment와 같은 조건)")
    print("images =", len(image_paths))
    print("GT =", gt_dir)
    print(f"ROI = roi.csv, 없으면 top {C.ROI_TOP_DEFAULT} · bottom {C.ROI_BOTTOM_DEFAULT} (ROI 밖 정답은 FN)")
    print(f"판정 = IoU ≥ {C.IOU_THRESH} 일대일(iou), 검출 중심이 정답 박스 안(ctr)")
    print("=" * 72)

    original_params = backup_params()

    totals = {
        (params["name"], mode): {"images": 0, **{col: 0 for col in count_columns()}}
        for params in PARAM_SETS
        for mode, _ in MODES
    }

    detail_rows = []
    n_default_roi = 0

    try:
        # 사진은 한 번만 읽고, 그 사진으로 파라미터 세트 × 전처리 끔·켬을 모두 돌린다
        for index, image_path in enumerate(
            image_paths,
            start=1,
        ):
            gt = load_gt(gt_dir / f"{image_path.stem}.csv")

            if gt is None:
                print(
                    "[GT 없음]",
                    image_path.name,
                )
                continue

            bgr = io_utils.load_image(str(image_path))
            roi, used_default = rx.get_roi(image_path.name, roi_table)
            n_default_roi += used_default

            for params in PARAM_SETS:
                apply_param_set(params)

                for mode, use_preprocess in MODES:
                    result = evaluate_one(
                        bgr,
                        gt,
                        roi,
                        use_preprocess,
                    )

                    detail_rows.append(
                        {
                            "params": params["name"],
                            "preprocess": mode,
                            "image": image_path.name,
                            **result,
                        }
                    )

                    total = totals[(params["name"], mode)]
                    total["images"] += 1

                    for key, value in result.items():
                        total[key] += value

            if index % 50 == 0:
                print(
                    f"{index}/{len(image_paths)}"
                )

    finally:
        restore_params(
            original_params
        )

    summary_rows = [
        summarize(param_name, mode, total)
        for (param_name, mode), total in totals.items()
    ]

    save_csv(
        DETAIL_PATH,
        detail_rows,
        ["params", "preprocess", "image"] + count_columns(),
    )

    save_csv(
        SUMMARY_PATH,
        summary_rows,
        summary_fieldnames(),
    )

    print_summary(summary_rows)

    print()
    print("=" * 72)
    print(f"전체 평가 완료 (config 기본 ROI를 쓴 사진 {n_default_roi}장)")
    print("상세:", DETAIL_PATH)
    print("요약:", SUMMARY_PATH)
    print("=" * 72)


if __name__ == "__main__":
    main()
