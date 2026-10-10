"""(파트 B) 교수 제공 훈련 데이터(RDD2022 804장)에서 검출 성능을 정량 평가한다. 평가 체계는 C가 더함(Plan.md 3.2).
사용법: 맨 위 폴더에서  python tools/convert_json_gt.py (정답 CSV를 먼저 만듦)
        → python tools/evaluate_train.py [--split dev|test|all] [--name 이름] [--param-set base|…|all]
                                         [--mask on|off] [--roi-top 0.3] [--detect legacy|dark] [--set NAME=VALUE]
훈련 사진과 정답 CSV 폴더는 config B 구역의 TRAIN_IMAGE_DIR·TRAIN_GT_DIR.

평가할 사진(--split): 국가별 이름순 순번 % 5 == 4이면 시험, 나머지는 개발(진단 35장은 개발에 고정).
기본은 개발 652장이고, 시험 152장은 파라미터를 확정한 뒤 한 번만(10/11 17:00) 잰다.

결과: results/<name>/ (--name을 빼면 results/latest. 1차 결과 폴더 base·tuning·diagnosis에는 쓰지 않는다)
- train_detail.csv: 영상별 개수와 영상 단위 판정(has_damage·n_pred·hit_same·hit_any·correct, IoU 기준 correct_iou)
- train_summary.csv: 합계. 종류·기준별 정밀도·재현율, 민감도·특이도·정답률·균형 정확도·기준선, 영상당 후보·오검출, 우연 대조군
- per_source.csv: 같은 요약을 국가별로
- config_used_train.txt: 이번 실행의 config 값, 파라미터 세트, 실행 옵션으로 바꾼 값
파라미터: 기본은 config.py 값을 그대로 한 번 돌린다(params 열 = config).
--param-set으로 1차 튜닝 세트를 고르면 그 값으로 config를 실행 중에만 덮는다.
base는 1차(10/06, results/tuning) 기준 값이라 E0 재현에 쓰고, all은 5세트를 모두 돈다.
--mask·--roi-top·--detect·--set도 config.py 파일은 고치지 않고 실행 중에만 값을 바꾼 뒤 끝나면 되돌린다.

run_experiment와 같은 조건으로 평가한다:
- pipeline.run_pipeline(640px → ROI 자르기 → 전처리 끔/켬 → 검출)을 그대로 쓰고, 끔·켬 두 줄을 모두 낸다.
- ROI는 run_experiment.get_roi와 같이 roi.csv에서 찾고, 없으면 config 기본값(ROI_TOP_DEFAULT·ROI_BOTTOM_DEFAULT).
- 후보 좌표는 run_experiment.to_full_boxes로 640px 전체 영상 좌표로 옮긴다(y + y0).
- ROI 밖에 있는 정답도 빼지 않고 놓친 것(FN)으로 센다.
판정은 두 가지를 함께 낸다:
- iou: IoU ≥ IOU_THRESH, 정답 하나에 검출 하나(evaluate.count_matches)
- ctr: 검출 중심이 정답 박스 안(evaluate.count_center_hits). 큰 정답 박스 안의 조각 검출도 맞힘으로 셈
영상 단위: 손상 영상은 같은 종류 후보가 하나라도 맞으면(ctr), 손상 없는 영상은 후보가 0개면 맞힘(evaluate.image_verdict).
우연 대조군: 640px 높이가 같은 사진끼리 이름순으로 묶고 i번 후보를 i+1번 정답과 비교한 재현율(사진이 1장뿐인 높이는 뺌)."""

import argparse
import ast
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
    image_verdict,
    load_gt,
    precision_recall,
    summarize_verdicts,
)


RESULTS_ROOT = Path("results")              # 결과는 results/<name>/ (이름 규칙은 run_experiment와 같음)

DETAIL_NAME = "train_detail.csv"
SUMMARY_NAME = "train_summary.csv"
CONFIG_NAME = "config_used_train.txt"       # run_experiment의 config_used.txt와 같은 폴더에서 겹치지 않게
SOURCE_NAME = "per_source.csv"

ROI_PATH = Path("data/roi.csv")

KINDS = ("crack", "pothole")
MODES = (("off", False), ("on", True))     # 전처리 끔·켬 (run_experiment의 preprocess 열과 같은 이름)
CRITERIA = ("iou", "ctr")                  # iou: IoU 일대일, ctr: 검출 중심이 정답 박스 안
COUNT_KEYS = ("tp", "fp", "found", "fn")

# 영상별 판정 열(train_detail.csv): 앞 5개는 evaluate.image_verdict(중심 기준), 뒤 2개는 같은 판정을 IoU 기준으로
VERDICT_COLUMNS = ["has_damage", "n_pred", "hit_same", "hit_any", "correct", "hit_same_iou", "correct_iou"]

# 요약 열(train_summary.csv·per_source.csv): 영상 단위 지표와 영상당 후보·오검출(Plan.md 3.2 (1) '보고할 값')
IMAGE_COLUMNS = [
    "n_damage", "n_none",
    "sensitivity", "specificity", "accuracy", "balanced", "baseline_none",
    "sensitivity_iou", "accuracy_iou", "sensitivity_any",
    "pred_per_image", "fp_per_image",
]

# 우연 대조군 열: 종류별 대조군 재현율과 '실제 − 대조군'(중심 기준, 채택 규칙 3번), 대조군에서 뺀 사진 수
CHANCE_COLUMNS = [
    f"{kind}_ctr_recall_{name}"
    for kind in KINDS
    for name in ("chance", "net")
] + ["chance_skipped"]


# ---------------------------------------------------------
# 개발/시험 나누기 (Plan.md 3.2 (3)·부록 B)
#
# 국가별로 이름순 정렬한 순번(0부터) % SPLIT_EVERY == SPLIT_TEST_INDEX이면 시험용이다.
# 진단 35장(부록 A)은 원인 분석에서 결과를 이미 본 영상이라 개발용에 고정한다.
# 영상 처리를 사진 이름으로 나누는 예외가 아니라 '평가 분할의 정의'다(검출·전처리에는 쓰지 않음).
# ---------------------------------------------------------

SPLIT_EVERY = 5
SPLIT_TEST_INDEX = 4

DIAG35 = (
    "China_Drone_000209", "China_Drone_000723", "China_Drone_001041", "China_Drone_001198", "China_Drone_001560",
    "China_MotorBike_000658", "China_MotorBike_000788", "China_MotorBike_000824", "China_MotorBike_001025", "China_MotorBike_001096",
    "Czech_001327", "Czech_001998", "Czech_002332", "Czech_003262", "Czech_003391",
    "India_001820", "India_002288", "India_003672", "India_007702", "India_009071",
    "Japan_001874", "Japan_008749", "Japan_010121", "Japan_011682", "Japan_012735",
    "Norway_001144", "Norway_001378", "Norway_004026", "Norway_004326", "Norway_005947",
    "United_States_000543", "United_States_001813", "United_States_002141", "United_States_002852", "United_States_003541",
)


# ---------------------------------------------------------
# 파라미터 후보
#
# 기본 실행은 config.py 값을 그대로 쓴다(CONFIG_SET).
# 아래 1차 튜닝 세트는 --param-set으로 고를 때만 쓴다.
# base는 1차(10/06, results/tuning) 기준 값을 숫자로 고정해 둔 것이라
# config B 구역이 바뀐 뒤에도 E0(1차 방법)를 다시 낼 수 있다.
#
# config.py 파일은 수정하지 않는다.
# 실행 중 C.xxx 값만 임시 변경한 뒤 반드시 원복한다.
# ---------------------------------------------------------

CONFIG_SET = {"name": "config"}            # 덮어쓰는 값이 없는 세트: config.py 값 그대로

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


def parse_args(argv=None):
    """평가할 사진(--split)·결과 폴더(--name)·1차 튜닝 세트(--param-set)와, 이번 실행에만 바꿀 config 값
    (--mask·--roi-top·--detect·--set)을 명령행에서 읽는다. 1차 결과 폴더 이름과 잘못된 값은 거부한다."""
    parser = argparse.ArgumentParser(description="훈련 데이터 전체 평가")

    parser.add_argument(
        "--split",
        choices=("dev", "test", "all"),
        default="dev",
        help="평가할 사진: dev 개발 652장(기본), test 시험 152장(파라미터 확정 뒤 한 번만), all 전체 804장",
    )

    parser.add_argument(
        "--name",
        default=rx.DEFAULT_NAME,
        help=f"결과 폴더 이름 (results/<name>/, 기본 {rx.DEFAULT_NAME})",
    )

    parser.add_argument(
        "--param-set",
        choices=[params["name"] for params in PARAM_SETS] + ["all"],
        help="1차 튜닝 세트로 돌림 (base = E0 재현용 1차 값, all = 5세트 모두). 빼면 config.py 값 그대로",
    )

    parser.add_argument(
        "--mask",
        choices=("on", "off"),
        help="노면 마스크(USE_ROAD_MASK)를 이번 실행에만 켬·끔",
    )

    parser.add_argument(
        "--roi-top",
        type=float,
        help="기본 띠의 위쪽 비율(ROI_TOP_DEFAULT)을 이번 실행에만 바꿈. 예: 0.3 (E1b)",
    )

    parser.add_argument(
        "--detect",
        choices=("legacy", "dark"),
        help="검출 방식(DETECT_MODE)을 이번 실행에만 바꿈",
    )

    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="config 값 하나를 이번 실행에만 바꿈(여러 번 가능). 예: --set DARK_SIGMA_IN_ROAD=False",
    )

    args = parser.parse_args(argv)
    rx.check_name(parser, args.name)

    if args.roi_top is not None and not 0 <= args.roi_top < 1:
        parser.error(f"--roi-top은 0 이상 1 미만이어야 함 ({args.roi_top})")

    try:
        args.overrides = runtime_overrides(args)
    except ValueError as e:
        parser.error(str(e))

    # 같은 값을 두 옵션이 바꾸면 어느 쪽이 적용됐는지 헷갈리므로 막는다
    swept = {
        key
        for params in select_param_sets(args.param_set)
        for key in params
        if key != "name"
    }
    clash = sorted(set(args.overrides) & swept)

    if clash:
        parser.error(f"--set과 --param-set이 같은 값을 바꿈: {clash}")

    return args


def parse_set(item):
    """'NAME=VALUE' 하나를 (이름, 값)으로 읽는다. 값은 파이썬 값(False·0.3·(3, 12))으로 읽고, 안 되면 글자 그대로(dark).
    config에 없는 이름은 오타일 수 있으므로 ValueError."""
    name, sep, text = item.partition("=")
    name = name.strip()

    if not sep or not name.isupper() or not hasattr(C, name):
        raise ValueError(f"--set은 config에 있는 대문자 이름=값이어야 함: {item!r}")

    try:
        value = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        value = text.strip()

    return name, value


def runtime_overrides(args):
    """명령행 옵션을 이번 실행에만 쓸 config 값 {이름: 값}으로 바꾼다(config.py 파일은 고치지 않음)."""
    overrides = {}

    if args.mask is not None:
        overrides["USE_ROAD_MASK"] = args.mask == "on"

    if args.roi_top is not None:
        overrides["ROI_TOP_DEFAULT"] = args.roi_top

    if args.detect is not None:
        overrides["DETECT_MODE"] = args.detect

    for item in args.set:
        name, value = parse_set(item)
        overrides[name] = value

    return overrides


def select_param_sets(name):
    """돌릴 파라미터 세트를 고른다: 없으면 config 값 그대로 한 세트, 'all'이면 1차 5세트, 아니면 그 이름의 세트."""
    if name is None:
        return [CONFIG_SET]

    if name == "all":
        return PARAM_SETS

    return [
        params
        for params in PARAM_SETS
        if params["name"] == name
    ]


def source_of(name):
    """사진 이름에서 국가(데이터 출처)를 뽑는다: 'Japan_000156.jpg' → 'Japan', 'China_Drone_000035.jpg' → 'China_Drone'."""
    return Path(name).stem.rsplit("_", 1)[0]


def assign_split(names):
    """사진 이름마다 개발('dev')·시험('test')을 정한다. 국가별 이름순 순번(0부터) % 5 == 4이면 시험이고,
    진단 35장은 개발에 고정한다. 무작위가 없어 누가 돌려도 같다(Plan.md 3.2 (3)·부록 B)."""
    by_source = {}

    for name in sorted(names):
        by_source.setdefault(source_of(name), []).append(name)

    diag = set(DIAG35)
    split = {}

    for members in by_source.values():
        for index, name in enumerate(members):
            is_test = index % SPLIT_EVERY == SPLIT_TEST_INDEX and Path(name).stem not in diag
            split[name] = "test" if is_test else "dev"

    return split


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


def backup_params(extra_keys=()):
    """튜닝 세트와 실행 옵션(--mask·--roi-top·--detect·--set)이 바꾸는 config 파라미터를 백업한다."""
    keys = {
        key
        for params in PARAM_SETS
        for key in params
        if key != "name"
    } | set(extra_keys)

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


def predict_one(bgr, roi, use_preprocess):
    """훈련 영상 한 장을 run_experiment와 같은 조건(ROI·전처리)으로 검출해 종류별 후보 상자(640px 전체 좌표)와 640px 영상 높이를 돌려준다.
    상자를 따로 돌려주는 것은 우연 대조군에서 다른 영상의 정답과 다시 비교하기 위해서다."""
    out = pipeline.run_pipeline(
        bgr,
        roi["top"],
        roi["bottom"],
        use_preprocess,
    )

    regions_by_kind = {
        "crack": out["cracks"],
        "pothole": out["potholes"],
    }

    # 후보는 노면(ROI) 기준이므로 정답과 같은 640px 전체 영상 좌표로 옮긴다
    pred_by_kind = {
        kind: rx.to_full_boxes(regions_by_kind[kind], out["y0"])
        for kind in KINDS
    }

    return pred_by_kind, out["small"].shape[0]


def score_one(pred_by_kind, gt):
    """후보 상자(640px 전체 좌표)를 정답과 두 기준(iou·ctr)으로 비교한 개수와, 영상 단위 판정(중심·IoU 기준)을 돌려준다."""
    truth_by_kind = dict(zip(KINDS, split_gt(gt)))

    result = {}

    for kind in KINDS:
        pred = pred_by_kind[kind]
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

    verdict = image_verdict(pred_by_kind, gt)

    # 같은 판정을 IoU 기준으로: 종류별 found는 같은 종류 정답과 맞춘 수라, 하나라도 0보다 크면 같은 종류로 맞힘
    hit_same_iou = any(result[f"{kind}_iou_found"] > 0 for kind in KINDS)

    result.update(verdict)
    result["hit_same_iou"] = hit_same_iou
    result["correct_iou"] = hit_same_iou if verdict["has_damage"] else verdict["n_pred"] == 0

    return result


def evaluate_one(bgr, gt, roi, use_preprocess):
    """훈련 영상 한 장을 run_experiment와 같은 조건(ROI·전처리)으로 검출해 정답과 두 기준으로 비교한다."""
    pred_by_kind, _ = predict_one(bgr, roi, use_preprocess)

    return score_one(pred_by_kind, gt)


def chance_found(names, heights, preds, gts):
    """우연 대조군(Plan.md 3.2 (3)·부록 B): 640px 높이가 같은 사진끼리 이름순으로 묶고, i번 사진의 후보를
    i+1번 사진의 같은 종류 정답과 중심 기준으로 비교해 찾은 정답 수를 센다(마지막 사진은 첫 사진과).
    정답 상자가 크면 후보 중심이 우연히 들어가므로, 실제 재현율에서 이 값을 빼 '우연을 뺀 재현율'을 낸다.
    사진이 1장뿐인 높이 묶음은 자기 자신과 비교하게 되므로 빼고, 뺀 사진 수를 함께 돌려준다.
    → ({종류: 찾은 정답 수}, {종류: 비교한 정답 수}, 뺀 사진 수)"""
    groups = {}

    for name in sorted(names):
        groups.setdefault(heights[name], []).append(name)

    found = {kind: 0 for kind in KINDS}
    truth = {kind: 0 for kind in KINDS}
    skipped = 0

    for members in groups.values():
        if len(members) < 2:
            skipped += len(members)
            continue

        for index, name in enumerate(members):
            other = members[(index + 1) % len(members)]
            truth_by_kind = dict(zip(KINDS, split_gt(gts[other])))

            for kind in KINDS:
                found[kind] += count_center_hits(preds[name][kind], truth_by_kind[kind])["found"]
                truth[kind] += len(truth_by_kind[kind])

    return found, truth, skipped


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


def ratio(numerator, denominator):
    """비율을 계산하며 분모가 0이면 NaN을 돌려준다."""
    return numerator / denominator if denominator else np.nan


def summarize(keys, rows, preds, gts):
    """영상별 행 묶음 하나(한 파라미터 세트·전처리 조건, 또는 그중 한 국가)를 요약 한 줄로 만든다.
    종류·기준별 개수와 정밀도·재현율(영상별 비율의 평균이 아니라 합계로), 영상 단위 지표, 영상당 후보·오검출, 우연 대조군.
    keys: 맨 앞 열(params·preprocess, 국가별이면 source), preds: {사진 이름: 종류별 후보 상자}, gts: {사진 이름: 정답}"""
    row = {
        **keys,
        "images": len(rows),
    }

    recall_ctr = {}

    for kind in KINDS:
        for criterion in CRITERIA:
            prefix = f"{kind}_{criterion}"
            counts = {key: sum(r[f"{prefix}_{key}"] for r in rows) for key in COUNT_KEYS}
            precision, recall = precision_recall(counts)

            row.update({f"{prefix}_{key}": counts[key] for key in COUNT_KEYS})
            row[f"{prefix}_precision"] = metric_text(precision)
            row[f"{prefix}_recall"] = metric_text(recall)

            if criterion == "ctr":
                recall_ctr[kind] = recall

    verdicts = summarize_verdicts(rows)
    damaged = [r for r in rows if r["has_damage"]]
    n_pred = sum(r["n_pred"] for r in rows)
    n_hit = sum(r[f"{kind}_ctr_tp"] for r in rows for kind in KINDS)

    image_values = {
        **{key: verdicts[key] for key in ("sensitivity", "specificity", "accuracy", "balanced", "baseline_none")},
        "sensitivity_iou": ratio(sum(r["hit_same_iou"] for r in damaged), len(damaged)),
        "accuracy_iou": ratio(sum(r["correct_iou"] for r in rows), len(rows)),
        "sensitivity_any": ratio(sum(r["hit_any"] for r in damaged), len(damaged)),
        "pred_per_image": ratio(n_pred, len(rows)),
        "fp_per_image": ratio(n_pred - n_hit, len(rows)),       # 중심 기준으로 안 맞은 후보
    }

    row["n_damage"] = verdicts["n_damage"]
    row["n_none"] = verdicts["n_none"]
    row.update({key: metric_text(value) for key, value in image_values.items()})

    found, truth, skipped = chance_found(
        [r["image"] for r in rows],
        {r["image"]: r["height"] for r in rows},
        preds,
        gts,
    )

    for kind in KINDS:
        chance = ratio(found[kind], truth[kind])
        row[f"{kind}_ctr_recall_chance"] = metric_text(chance)
        row[f"{kind}_ctr_recall_net"] = metric_text(recall_ctr[kind] - chance)

    row["chance_skipped"] = skipped

    return row


def summary_fieldnames(extra=()):
    """요약 표의 열: 파라미터·전처리(·국가)·영상 수, 종류·기준별 개수와 정밀도·재현율, 영상 단위 지표, 우연 대조군."""
    columns = ["params", "preprocess", *extra, "images"]

    for kind in KINDS:
        for criterion in CRITERIA:
            prefix = f"{kind}_{criterion}"
            columns += [f"{prefix}_{key}" for key in COUNT_KEYS]
            columns += [f"{prefix}_precision", f"{prefix}_recall"]

    return columns + IMAGE_COLUMNS + CHANCE_COLUMNS


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


def write_config(path, param_sets, overrides):
    """이번 실행의 config 값, 돌린 파라미터 세트 이름, 실행 옵션으로 바꾼 값을 남긴다(결과가 어떤 설정에서 나왔는지 알 수 있게).
    실행 옵션을 적용한 뒤에 부르므로 위쪽 config 값에도 바꾼 값이 들어 있다."""
    rx.write_config(path)

    names = ", ".join(params["name"] for params in param_sets)
    changed = ", ".join(f"{key}={value!r}" for key, value in overrides.items()) or "없음"

    with path.open("a", encoding="utf-8") as f:
        f.write(f"# 파라미터 세트: {names} (config가 아닌 세트는 PARAM_SETS 값이 위 값을 실행 중에 덮음)\n")
        f.write(f"# 실행 옵션으로 바꾼 값: {changed}\n")


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


def print_image_summary(summary_rows):
    """영상 단위 지표를 콘솔 표로 보여 준다: 민감도·특이도·정답률(기준선)·균형 정확도·영상당 오검출·우연을 뺀 균열 재현율(중심)."""
    keys = ("sensitivity", "specificity", "accuracy", "baseline_none", "balanced", "fp_per_image", "crack_ctr_recall_net")

    print()
    print(f"{'params':15s} {'pre':4s} |  sens   spec   acc  (base)    bal | FP/img | crack R-chance")

    for row in summary_rows:
        value = {key: float(row[key]) for key in keys}

        print(
            f"{row['params']:15s} {row['preprocess']:4s} | "
            f"{value['sensitivity']:.3f}  {value['specificity']:.3f}  {value['accuracy']:.3f} ({value['baseline_none']:.3f})  "
            f"{value['balanced']:.3f} | {value['fp_per_image']:6.2f} | {value['crack_ctr_recall_net']:.4f}"
        )


def main(argv=None):
    """훈련 데이터에서 고른 분할(기본 개발)을 고른 파라미터 세트(기본 config 값)와 실행 옵션으로 전처리 끔·켬 평가해 results/<name>/에 저장한다."""
    args = parse_args(argv)
    param_sets = select_param_sets(args.param_set)

    out_dir = RESULTS_ROOT / args.name
    detail_path = out_dir / DETAIL_NAME
    summary_path = out_dir / SUMMARY_NAME
    source_path = out_dir / SOURCE_NAME
    config_path = out_dir / CONFIG_NAME

    image_dir = Path(C.TRAIN_IMAGE_DIR)
    gt_dir = Path(C.TRAIN_GT_DIR)

    all_paths = sorted(
        image_dir.glob("*.jpg")
    )

    if not all_paths:
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

    # 나누기는 사진 전체 이름으로 정한다(고른 분할만으로 정하면 국가별 순번이 달라짐)
    split = assign_split([path.name for path in all_paths])

    image_paths = [
        path
        for path in all_paths
        if args.split == "all" or split[path.name] == args.split
    ]

    n_gt_files = sum(1 for _ in gt_dir.glob("*.csv"))

    # 사진이 일부만 있으면 순번이 밀려 계획의 나누기(시험 152·개발 652)와 달라진다
    if n_gt_files != len(all_paths):
        print(f"경고: 사진 {len(all_paths)}장과 정답 CSV {n_gt_files}개의 수가 달라 개발/시험 나누기가 계획과 다를 수 있습니다.")

    # 사진·정답을 확인한 뒤에 폴더를 만든다(입력이 없을 때 빈 결과 폴더가 남지 않게)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    roi_table = io_utils.load_roi_table(ROI_PATH)

    original_params = backup_params(args.overrides)

    detail_rows = []
    preds = {
        (params["name"], mode): {}
        for params in param_sets
        for mode, _ in MODES
    }
    gts = {}
    n_default_roi = 0

    try:
        # 실행 옵션은 실행 내내 적용하고, 튜닝 세트는 그 위에 덮는다(같은 값을 둘 다 바꾸는 것은 parse_args가 막음)
        apply_param_set(args.overrides)

        write_config(
            config_path,
            param_sets,
            args.overrides,
        )

        print("=" * 72)
        print("훈련 데이터 평가 (run_experiment와 같은 조건)")
        print(f"split = {args.split} ({len(image_paths)}/{len(all_paths)}장)")
        print("GT =", gt_dir)
        print("params =", ", ".join(params["name"] for params in param_sets))
        print("실행 옵션 =", ", ".join(f"{key}={value!r}" for key, value in args.overrides.items()) or "없음")
        print("결과 =", out_dir)
        print(f"ROI = roi.csv, 없으면 top {C.ROI_TOP_DEFAULT} · bottom {C.ROI_BOTTOM_DEFAULT} (ROI 밖 정답은 FN)")
        print(f"판정 = IoU ≥ {C.IOU_THRESH} 일대일(iou), 검출 중심이 정답 박스 안(ctr)")

        if args.split == "test":
            print("시험셋은 파라미터를 확정한 뒤 한 번만 평가하고, 이 결과를 보고 파라미터를 고치지 않습니다(Plan.md 3.10).")

        print("=" * 72)

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
            gts[image_path.name] = gt

            for params in param_sets:
                apply_param_set(params)

                for mode, use_preprocess in MODES:
                    pred_by_kind, height = predict_one(
                        bgr,
                        roi,
                        use_preprocess,
                    )

                    preds[(params["name"], mode)][image_path.name] = pred_by_kind

                    detail_rows.append(
                        {
                            "params": params["name"],
                            "preprocess": mode,
                            "image": image_path.name,
                            "split": split[image_path.name],
                            "source": source_of(image_path.name),
                            "height": height,
                            **score_one(pred_by_kind, gt),
                        }
                    )

            if index % 50 == 0:
                print(
                    f"{index}/{len(image_paths)}"
                )

    finally:
        restore_params(
            original_params
        )

    summary_rows = []
    source_rows = []

    for params in param_sets:
        for mode, _ in MODES:
            key = (params["name"], mode)
            keys = {"params": params["name"], "preprocess": mode}
            rows = [r for r in detail_rows if (r["params"], r["preprocess"]) == key]

            summary_rows.append(summarize(keys, rows, preds[key], gts))

            for source in sorted({r["source"] for r in rows}):
                part = [r for r in rows if r["source"] == source]
                source_rows.append(summarize({**keys, "source": source}, part, preds[key], gts))

    save_csv(
        detail_path,
        detail_rows,
        ["params", "preprocess", "image", "split", "source", "height"] + count_columns() + VERDICT_COLUMNS,
    )

    save_csv(
        summary_path,
        summary_rows,
        summary_fieldnames(),
    )

    save_csv(
        source_path,
        source_rows,
        summary_fieldnames(extra=("source",)),
    )

    print_summary(summary_rows)
    print_image_summary(summary_rows)

    print()
    print("=" * 72)
    print(f"평가 완료 (config 기본 ROI를 쓴 사진 {n_default_roi}장)")
    print("상세:", detail_path)
    print("요약:", summary_path)
    print("국가별:", source_path)
    print("설정:", config_path)
    print("=" * 72)


if __name__ == "__main__":
    main()
