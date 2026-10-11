"""(파트 C) 제공·직접 촬영 영상 전체를 전처리 끔·켬으로 처리해 지표 표·비교 그림·요약 표를 저장한다.
사용법: 맨 위 폴더에서  python run_experiment.py --name <이름>  [--only provided|own]
                        [--roi csv|auto] [--mask on|off] [--detect legacy|dark] [--roi-top 0.3]
결과: results/<name>/ (--name을 빼면 results/latest. 1차 결과 폴더 base·tuning·diagnosis에는 쓰지 않는다)
- metrics.csv: 영상마다 끔·켬 2행 (영상 정보, 상태 지표, row 9열, 노면 안 에지 수 edges_road·노면 마스크 시간 t_road_ms,
  정답 비교 tp·fp·found·fn(IoU), 영상 단위 판정 has_gt·has_damage·n_pred·hit_same·hit_any·correct(중심 기준).
  정답 파일이 없는 사진은 판정 칸(has_damage·hit_same·hit_any·correct)과 정답 비교 칸이 빈칸)
- figures/<영상 이름>.png: 위 줄 끔·아래 줄 켬 비교 그림
- table_by_image.csv, table_by_source.csv, table_by_condition.csv: 지표_off·지표_on·지표_diff(on − off) 요약 표.
  source·condition 표에는 정답 있는 영상의 민감도·특이도·영상 단위 정답률과, 정답률 바로 옆에 기준선('검출 없음'의 정답률)을 둔다.
  table_by_image에는 영상마다 판정(correct·hit_same·hit_any)의 끔·켬을 둔다(포트홀처럼 개수가 적은 것은 영상별 사례로 보고)
- config_used.txt: 이번 실행에 쓴 config.py의 대문자 값과, 맨 끝에 실행 옵션·옵션으로 바꾼 값
실행 옵션은 config.py 파일을 고치지 않고 실행 중에만 값을 바꾼 뒤 끝나면 되돌린다. 옵션을 주지 않으면 1차와 같은 조건이다.
- --roi csv(기본): roi.csv에 사진마다 정한 띠. auto: roi.csv의 범위를 쓰지 않고 모든 사진에 기본 띠(ROI_TOP_DEFAULT~ROI_BOTTOM_DEFAULT)
- --mask on|off: USE_ROAD_MASK, --detect legacy|dark: DETECT_MODE, --roi-top: ROI_TOP_DEFAULT(--roi auto와 함께만. csv에서는 39장에 효과가 없어 거부)
- --roi auto와 노면 마스크는 따로 켠다(한 단계씩 끄고 켜며 비교하려고). 둘 다 쓰려면 --roi auto --mask on
처리 시간(t_pre_ms·t_detect_ms·t_road_ms)은 끔·켬을 각각 C.TIME_REPEAT번 돌린 중앙값이고, 나머지 값은 첫 결과다.
영상 하나가 실패해도 나머지를 계속 처리하고, 실패가 있으면 끝에 같은 오류끼리 묶어 출력하고 종료 코드 1로 끝난다.
"""
import argparse
import glob
import math
import os
import statistics
import sys

import numpy as np
import pandas as pd

from src import analyze, io_utils, pipeline, visualize
from src import config as C

try:
    from src import evaluate  # B의 파일. 아직 없으면 정답 비교 열을 빈칸으로 둔다
except ImportError:
    evaluate = None

IMAGE_DIRS = {"provided": "data/provided", "own": "data/own"}
INFO_COLUMNS = ["file", "source", "condition", "preprocess", "steps"]
ROW_COLUMNS = ["edges", "harris", "n_crack", "area_crack", "n_pothole", "area_pothole",
               "roi_pixels", "t_pre_ms", "t_detect_ms"]
TIME_COLUMNS = ("t_pre_ms", "t_detect_ms")
KINDS = ("crack", "pothole")
COUNT_KEYS = ("tp", "fp", "found", "fn")
GT_COLUMNS = [f"{kind}_{key}" for kind in KINDS for key in COUNT_KEYS]
# row 9열 밖의 측정값: 노면 마스크 안의 에지 수, 노면 마스크 시간(반복 중앙값). row는 1차와 같은 뜻으로 두려고 따로 붙인다
EXTRA_COLUMNS = ["edges_road", "t_road_ms"]
# 영상 단위 판정(evaluate.image_verdict, 중심 기준 = evaluate_train과 같음). has_gt·n_pred는 정답과 상관없이 채운다
VERDICT_COLUMNS = ["has_gt", "has_damage", "n_pred", "hit_same", "hit_any", "correct"]
VERDICT_BOOL = ["has_damage", "hit_same", "hit_any", "correct"]     # 정답 파일이 없으면 빈칸이 되는 판정 칸
# table_by_image에 끔·켬으로 나란히 두는 판정(영상별 사례 보고용. 포트홀은 개수가 적어 비율보다 사례로 본다)
VERDICT_PAIRS = ["correct", "hit_same", "hit_any"]
# 요약 표에서 끔·켬을 비교하는 지표. 면적 %는 add_ratios가 더한다
SUMMARY_METRICS = ["edges", "edges_road", "harris", "n_crack", "area_crack", "area_crack_pct",
                   "n_pothole", "area_pothole", "area_pothole_pct", "t_pre_ms", "t_detect_ms", "t_road_ms"]
PR_METRICS = [f"{kind}_{name}" for kind in KINDS for name in ("precision", "recall")]
# source·condition 표의 영상 단위 지표(정답 있는 영상만). 정답률(accuracy) 바로 뒤에는 기준선 열 baseline_none을 둔다
IMAGE_METRICS = ["sensitivity", "specificity", "accuracy"]
ROI_MODES = ("csv", "auto")         # 노면 띠: csv = roi.csv에 사진마다 정한 범위, auto = 모든 사진에 config 기본 띠
DEFAULT_NAME = "latest"             # --name을 빼면 쓰는 결과 폴더(매번 덮어씀). run_matching·evaluate_train도 같은 기본값
# 1차 실험 결과 폴더: 계획서·Plan.md 기준 수치의 출처라 어느 도구도 덮어쓰지 않는다
# (base: Plan.md 1.3·1.4의 39장·매칭, tuning: 1.2·부록 B의 804장, diagnosis: 띠 밖 정답 비율·원인 분석)
PROTECTED_NAMES = ("base", "tuning", "diagnosis")


def check_name(parser, name):
    """결과 폴더가 1차 결과 폴더이거나 그 안이면 실행을 멈춘다. 덮어쓰면 계획서 숫자를 다시 셀 근거가 사라지기 때문이다.
    윈도·맥 파일 시스템은 대소문자를 가리지 않으므로 소문자로 비교한다."""
    target = os.path.abspath(os.path.join("results", name)).lower()
    for protected in PROTECTED_NAMES:
        root = os.path.abspath(os.path.join("results", protected)).lower()
        if target == root or target.startswith(root + os.sep):
            parser.error(f"results/{protected}: 1차 결과(기준 수치의 출처)라 쓰지 않습니다. --name으로 다른 이름을 주세요.")


def parse_args(argv=None):
    """결과 폴더 이름(--name)·사진 묶음(--only)·노면 띠 방식(--roi)과, 이번 실행에만 바꿀 config 값(--mask·--detect·--roi-top)을 읽는다.
    1차 결과 폴더 이름과, 효과가 없는 조합(--roi csv에서 --roi-top)·범위 밖 값은 거부한다(마지막 일괄 실행에서 조용히 안 먹는 옵션이 없게)."""
    parser = argparse.ArgumentParser(description="전처리 끔·켬 비교 실험")
    parser.add_argument("--name", default=DEFAULT_NAME, help=f"결과 폴더 이름 (results/<name>/, 기본 {DEFAULT_NAME})")
    parser.add_argument("--only", choices=list(IMAGE_DIRS), help="한쪽 사진만 처리")
    parser.add_argument("--roi", choices=ROI_MODES, default="csv",
                        help="노면 띠: csv = roi.csv에 사진마다 정한 범위(기본), auto = roi.csv의 범위 없이 모든 사진에 기본 띠")
    parser.add_argument("--mask", choices=("on", "off"), help="노면 마스크(USE_ROAD_MASK)를 이번 실행에만 켬·끔")
    parser.add_argument("--detect", choices=("legacy", "dark"), help="검출 방식(DETECT_MODE)을 이번 실행에만 바꿈")
    parser.add_argument("--roi-top", type=float,
                        help="기본 띠의 위쪽 비율(ROI_TOP_DEFAULT)을 이번 실행에만 바꿈. --roi auto와 함께만. 예: 0.3 (E1b)")
    args = parser.parse_args(argv)
    check_name(parser, args.name)
    if args.roi_top is not None:
        if args.roi != "auto":
            parser.error("--roi-top은 --roi auto와 함께만 쓴다: --roi csv에서는 39장 모두 roi.csv의 범위를 써서 효과가 없다")
        if not 0 <= args.roi_top < C.ROI_BOTTOM_DEFAULT:
            parser.error(f"--roi-top은 0 이상 {C.ROI_BOTTOM_DEFAULT}(ROI_BOTTOM_DEFAULT) 미만이어야 함 ({args.roi_top})")
    return args


def runtime_overrides(args):
    """명령행 옵션을 이번 실행에만 쓸 config 값 {이름: 값}으로 바꾼다. 옵션이 없으면 빈 dict라 1차와 같은 조건으로 돈다.
    --roi는 config 값이 아니라 get_roi의 방식이라 여기 들어가지 않는다."""
    overrides = {}
    if args.mask is not None:
        overrides["USE_ROAD_MASK"] = args.mask == "on"
    if args.detect is not None:
        overrides["DETECT_MODE"] = args.detect
    if args.roi_top is not None:
        overrides["ROI_TOP_DEFAULT"] = args.roi_top
    return overrides


def apply_overrides(overrides):
    """config 값을 실행 중에만 바꾸고, 되돌릴 때 쓸 원래 값 {이름: 값}을 돌려준다(config.py 파일은 고치지 않음)."""
    backup = {name: getattr(C, name) for name in overrides}
    for name, value in overrides.items():
        setattr(C, name, value)
    return backup


def restore_overrides(backup):
    """apply_overrides가 바꾼 config 값을 원래대로 되돌린다. main이 finally에서 불러 실패해도 되돌아가게 한다."""
    for name, value in backup.items():
        setattr(C, name, value)


def run_notes(args, overrides):
    """config_used.txt 끝에 붙일 실행 조건 두 줄: 준 옵션과, 그 옵션으로 바꾼 config 값. 결과 폴더만 보고도 조건을 알 수 있게 한다."""
    options = [f"--roi {args.roi}"] + [f"--{flag} {value}" for flag, value in
                                       (("only", args.only), ("mask", args.mask), ("detect", args.detect), ("roi-top", args.roi_top))
                                       if value is not None]
    changed = ", ".join(f"{name}={value!r}" for name, value in overrides.items()) or "없음"
    return [f"실행 옵션: {' '.join(options)}", f"실행 옵션으로 바꾼 config 값: {changed}"]


def list_images(only=None, image_dirs=IMAGE_DIRS):
    """처리할 사진 경로를 제공 → 직접 촬영 순서로, 폴더 안에서는 이름 순으로 모은다."""
    names = [only] if only else list(image_dirs)
    return [p for name in names for p in sorted(glob.glob(f"{image_dirs[name]}/*.jpg"))]


def get_roi(file_name, roi_table, mode="csv"):
    """사진의 노면 범위를 정한다. (ROI dict, 기본 띠를 썼는지)를 돌려준다.
    mode "csv": roi.csv에 있으면 그 값, 없으면 config 기본 띠. "auto": roi.csv의 범위는 쓰지 않고 모든 사진에 기본 띠
    (손으로 정한 ROI 없이 도는지 보는 실험용, Plan.md 3.7). 상태 태그(condition)는 사진의 성질이라 어느 방식이든 roi.csv에서 가져온다."""
    entry = roi_table.get(file_name)
    if mode == "csv" and entry is not None:
        return entry, False
    condition = entry["condition"] if entry else ""
    return {"top": C.ROI_TOP_DEFAULT, "bottom": C.ROI_BOTTOM_DEFAULT, "condition": condition}, True


def repeat_pipeline(bgr, roi, use_preprocess):
    """한 설정(끔 또는 켬)을 C.TIME_REPEAT번 돌린 결과 list. 처리 시간의 중앙값을 내려고 반복한다."""
    return [pipeline.run_pipeline(bgr, roi["top"], roi["bottom"], use_preprocess)
            for _ in range(max(1, C.TIME_REPEAT))]


def median_times(outs):
    """첫 결과를 쓰되 처리 시간만 반복 전체의 중앙값으로 바꾼 새 결과를 돌려준다.
    한 번 잰 시간은 캐시·다른 프로그램 때문에 튀므로 중앙값으로 튀는 값을 버린다(받은 결과는 바꾸지 않음).
    row의 두 시간 열과, row 밖 결과 키인 노면 마스크 시간 t_road_ms를 같은 규칙으로 바꾼다(인계 메모 4.4)."""
    row = dict(outs[0]["row"])
    for key in TIME_COLUMNS:
        row[key] = round(statistics.median(o["row"][key] for o in outs), 2)
    out = {**outs[0], "row": row}
    if all("t_road_ms" in o for o in outs):
        out["t_road_ms"] = round(statistics.median(o["t_road_ms"] for o in outs), 2)
    return out


def unstable_keys(outs):
    """반복마다 값이 달라진 row 열(시간 제외)을 찾는다. 같은 입력이면 같은 출력이어야 하므로 비어 있어야 정상이다."""
    first = outs[0]["row"]
    return sorted({k for o in outs[1:] for k, v in o["row"].items()
                   if k not in TIME_COLUMNS and v != first.get(k)})


def to_full_boxes(regions, y0):
    """노면 기준 후보 bbox를 640px 전체 영상 좌표로 옮긴다. 정답 박스와 같은 좌표에서 비교하려고 y에 y0를 더한다."""
    return [(x, y + y0, w, h) for x, y, w, h in (r["bbox"] for r in regions)]


def gt_counts(out, gt, count_fn, thr):
    """균열·포트홀을 종류별로 정답 박스와 맞춰 tp·fp·found·fn을 센다.
    정답 파일이나 비교 함수가 없으면 0이 아니라 NaN으로 두어 '정답 없음'과 '0개 맞춤'을 구별한다."""
    if gt is None or count_fn is None:
        return {col: math.nan for col in GT_COLUMNS}
    counts = {}
    for kind, regions in (("crack", out["cracks"]), ("pothole", out["potholes"])):
        truth = [(x, y, w, h) for x, y, w, h, k in gt if k == kind]
        c = count_fn(to_full_boxes(regions, out["y0"]), truth, thr)
        counts.update({f"{kind}_{key}": int(c[key]) for key in COUNT_KEYS})
    return counts


def load_gt_for(file_name):
    """사진 이름에 맞는 정답 파일 gt/<이름>.csv를 읽는다. evaluate가 아직 없거나 정답 파일이 없으면 None."""
    if evaluate is None:
        return None
    return evaluate.load_gt(f"gt/{os.path.splitext(file_name)[0]}.csv")


def edges_on_road(edges, road):
    """노면 마스크 안의 에지 픽셀 수. 띠 전체 에지 수(row의 edges)는 1차와 비교하려고 그대로 두고, 노면 밖 배경의 에지를 뺀 값을 따로 낸다.
    edges·road는 둘 다 노면 띠 좌표의 0/255 영상이다. 받은 영상은 바꾸지 않는다."""
    return int(np.count_nonzero((edges > 0) & (road > 0)))


def verdict_columns(out, gt):
    """영상 단위 판정 열: 후보를 640px 전체 좌표로 옮겨 evaluate.image_verdict(중심 기준, evaluate_train과 같음)로 정답과 견준다.
    정답 파일이 없으면 판정 칸(has_damage·hit_same·hit_any·correct)은 빈칸(NaN)으로 두어 '정답 없음'과 '틀림'을 구별한다.
    has_gt와 n_pred(모든 후보 수)는 정답과 상관없는 값이라 늘 채운다."""
    pred_by_kind = {kind: to_full_boxes(regions, out["y0"])
                    for kind, regions in (("crack", out["cracks"]), ("pothole", out["potholes"]))}
    if gt is None or evaluate is None:
        n_pred = sum(len(boxes) for boxes in pred_by_kind.values())
        return {"has_gt": False, "n_pred": n_pred, **{col: math.nan for col in VERDICT_BOOL}}
    return {"has_gt": True, **evaluate.image_verdict(pred_by_kind, gt)}


def make_row(file_name, info, roi, use_preprocess, out, quality, counts, verdict):
    """metrics.csv 한 행: 영상 정보 + 상태 지표 + row 9열 + 노면 안 에지 수·노면 마스크 시간 + 정답 비교 개수 + 영상 단위 판정."""
    return {"file": file_name, "source": info["source"],
            "condition": info.get("condition") or roi["condition"],
            "preprocess": "on" if use_preprocess else "off",
            "steps": analyze.steps_to_text(out["steps"]),
            **quality, **out["row"],
            "edges_road": edges_on_road(out["edges"], out["road"]), "t_road_ms": out["t_road_ms"],
            **counts, **verdict}


def save_figure(path, file_name, off, on):
    """끔(위 줄)·켬(아래 줄)마다 후보 그림·노면 흑백·캐니 에지 3칸을 한 장으로 저장한다.
    후보 그림에는 노면 마스크를 반투명 초록으로 먼저 칠하고 그 위에 후보 윤곽을 그린다.
    save_comparison은 6칸이라 노면 표시는 7번째 칸이 아니라 후보 칸에 겹쳐 그린다."""
    panels = []
    for label, out in (("끔", off), ("켬", on)):
        road = visualize.draw_road(out["small"], out["road"], out["y0"])
        drawn = visualize.draw_candidates(road, out["cracks"], out["potholes"], out["y0"])
        panels += [(f"{label}: 노면(초록)·후보", drawn), (f"{label}: 노면", out["proc"]), (f"{label}: 에지", out["edges"])]
    visualize.save_comparison(path, f"{file_name} | 켬: {analyze.steps_to_text(on['steps'])}", panels)


def process_image(path, roi_table, fig_dir, roi_mode="csv"):
    """사진 1장을 끔·켬으로 반복 처리해 metrics 2행을 만들고 비교 그림을 저장한다. 오류는 main이 잡는다.
    (행 2개, 기본 띠를 썼는지)를 돌려준다. --roi auto면 모든 사진이 일부러 기본 띠를 쓰므로 경고하지 않는다."""
    file_name = os.path.basename(path)
    roi, used_default = get_roi(file_name, roi_table, roi_mode)
    if used_default and roi_mode == "csv":
        print(f"경고: roi.csv에 없음 → 기본 ROI ({C.ROI_TOP_DEFAULT}, {C.ROI_BOTTOM_DEFAULT}): {file_name}")
    info = io_utils.parse_name(path)
    bgr = io_utils.load_image(path)

    runs = {use: repeat_pipeline(bgr, roi, use) for use in (False, True)}   # 끔 n번 → 켬 n번
    for use, outs in runs.items():
        bad = unstable_keys(outs)
        if bad:
            print(f"경고: 반복마다 값이 다름 ({'켬' if use else '끔'}: {', '.join(bad)}): {file_name}")
    off, on = median_times(runs[False]), median_times(runs[True])

    quality = on["quality"]          # 상태 지표는 입력 영상의 성질이라 끔 행도 켬에서 잰 값(전처리 전 노면)을 쓴다
    gt = load_gt_for(file_name)
    count_fn = evaluate.count_matches if evaluate else None
    rows = [make_row(file_name, info, roi, use, out, quality, gt_counts(out, gt, count_fn, C.IOU_THRESH), verdict_columns(out, gt))
            for use, out in ((False, off), (True, on))]
    save_figure(os.path.join(fig_dir, f"{os.path.splitext(file_name)[0]}.png"), file_name, off, on)
    return rows, used_default


def quality_columns(rows):
    """metrics.csv의 상태 지표 열 이름. measure_quality의 키 순서를 따라 사진이 0장이어도 머리글이 남고,
    A가 키를 더하면 열도 자동으로 늘어난다."""
    keys = list(analyze.measure_quality(np.zeros((32, 32), np.uint8)))
    known = set(INFO_COLUMNS + ROW_COLUMNS + EXTRA_COLUMNS + GT_COLUMNS + VERDICT_COLUMNS + keys)
    return keys + list(dict.fromkeys(k for r in rows for k in r if k not in known))


def metrics_frame(rows):
    """행 list를 열 순서가 정해진 표로 만든다: 영상 정보 → 상태 지표 → row 9열 → edges_road·t_road_ms → 정답 비교 개수 → 영상 단위 판정.
    개수는 빈칸을 허용하는 정수형(Int64)이라 1.0이 아닌 1로, 판정은 빈칸을 허용하는 boolean이라 정답 파일이 없으면 빈칸으로 저장된다."""
    columns = INFO_COLUMNS + quality_columns(rows) + ROW_COLUMNS + EXTRA_COLUMNS + GT_COLUMNS + VERDICT_COLUMNS
    types = {col: "Int64" for col in GT_COLUMNS + ["edges_road", "n_pred"]}
    types.update({col: "boolean" for col in ["has_gt"] + VERDICT_BOOL})
    return pd.DataFrame(rows, columns=columns).astype(types)


def add_ratios(df):
    """균열·포트홀 면적을 노면 픽셀 수로 나눈 %(area_*_pct) 열을 더한 새 표. 영상마다 노면 넓이가 달라 면적만으로는 비교가 어렵다."""
    out = df.copy()
    roi = pd.to_numeric(out["roi_pixels"]).where(lambda s: s > 0)     # 노면이 0픽셀이면 0으로 나누지 않고 빈칸
    for kind in KINDS:
        out[f"area_{kind}_pct"] = pd.to_numeric(out[f"area_{kind}"]) / roi * 100
    return out


def pr_from_counts(df, kind, pr_fn):
    """정답이 있는 행만 골라 tp·fp·found·fn을 먼저 더한 뒤 정밀도·재현율을 한 번 계산한다(영상별 비율의 평균이 아님).
    정답이 있는 행이 없거나 계산 함수가 없으면 (NaN, NaN)."""
    has_gt = df[f"{kind}_tp"].notna()
    if pr_fn is None or not has_gt.any():
        return math.nan, math.nan
    return pr_fn({key: int(df.loc[has_gt, f"{kind}_{key}"].sum()) for key in COUNT_KEYS})


def off_on_diff(name, off, on):
    """한 지표의 끔·켬 값과 차이(on − off)를 지표_off·지표_on·지표_diff 열로 만든다."""
    return {f"{name}_off": off, f"{name}_on": on, f"{name}_diff": on - off}


def triple_names(names):
    """지표 이름마다 _off·_on·_diff 열 이름을 차례로 늘어놓는다(요약 표 머리글용)."""
    return [f"{n}_{s}" for n in names for s in ("off", "on", "diff")]


def pr_columns(off, on, pr_fn):
    """끔 행 묶음과 켬 행 묶음에서 균열·포트홀 정밀도·재현율의 _off·_on·_diff 열을 만든다."""
    cols = {}
    for kind in KINDS:
        (p_off, r_off), (p_on, r_on) = pr_from_counts(off, kind, pr_fn), pr_from_counts(on, kind, pr_fn)
        cols.update(off_on_diff(f"{kind}_precision", p_off, p_on))
        cols.update(off_on_diff(f"{kind}_recall", r_off, r_on))
    return cols


def table_by_image(df, pr_fn):
    """영상 1장 = 1행: 켬의 steps와 상태 지표, 정답 유무·손상 유무와 영상 단위 판정(correct·hit_same·hit_any)의 끔·켬,
    지표마다 끔·켬·차이, 그 영상 하나의 정밀도·재현율. 판정은 영상별 사례 보고용이고 정답 파일이 없으면 빈칸이다."""
    df = add_ratios(df)
    known = set(INFO_COLUMNS + ROW_COLUMNS + GT_COLUMNS + VERDICT_COLUMNS + SUMMARY_METRICS)
    head = ["file", "source", "condition", "steps"] + [c for c in df.columns if c not in known] + ["roi_pixels"]
    verdict = ["has_gt", "has_damage"] + [f"{v}_{s}" for v in VERDICT_PAIRS for s in ("off", "on")]
    off_all, on_all = df[df["preprocess"] == "off"], df[df["preprocess"] == "on"]
    rows = []
    for file_name in on_all["file"]:
        off, on = off_all[off_all["file"] == file_name], on_all[on_all["file"] == file_name]
        row = {c: on[c].iloc[0] for c in head + ["has_gt", "has_damage"]}
        for v in VERDICT_PAIRS:
            row[f"{v}_off"], row[f"{v}_on"] = off[v].iloc[0], on[v].iloc[0]
        for m in SUMMARY_METRICS:
            row.update(off_on_diff(m, off[m].iloc[0], on[m].iloc[0]))
        row.update(pr_columns(off, on, pr_fn))
        rows.append(row)
    return pd.DataFrame(rows, columns=head + verdict + triple_names(SUMMARY_METRICS + PR_METRICS))


def verdict_summary(sub):
    """정답 있는 영상의 판정만 모아 evaluate.summarize_verdicts로 손상·무손상 수, 민감도·특이도·정답률·기준선을 낸다.
    정답 파일이 없는 영상은 빼고(판정이 빈칸), 정답 있는 영상이 없거나 evaluate가 없으면 비율은 NaN이다."""
    has_gt = sub["has_gt"].fillna(False).astype(bool)
    verdicts = [{"has_damage": bool(d), "correct": bool(c)}
                for d, c in zip(sub.loc[has_gt, "has_damage"], sub.loc[has_gt, "correct"])]
    if evaluate is None:
        return {"n_damage": 0, "n_none": 0, **{m: math.nan for m in IMAGE_METRICS + ["baseline_none"]}}
    return evaluate.summarize_verdicts(verdicts)


def table_by_group(df, key, pr_fn):
    """key(source 또는 condition)의 값마다 1행 + 전체(all) 1행: 영상 수, 정답 있는 영상 수와 그중 손상·무손상 수,
    정답 있는 영상의 민감도·특이도·영상 단위 정답률(끔·켬·차이)과 정답률 바로 옆의 기준선('검출 없음'의 정답률),
    지표 평균의 끔·켬·차이, 개수 합계로 낸 정밀도·재현율. 빈 조건은 unknown으로 묶는다."""
    df = add_ratios(df)
    df[key] = df[key].fillna("").replace("", "unknown")
    rows = []
    for name, sub in [(g, df[df[key] == g]) for g in pd.unique(df[key])] + [("all", df)]:
        off, on = sub[sub["preprocess"] == "off"], sub[sub["preprocess"] == "on"]
        v_off, v_on = verdict_summary(off), verdict_summary(on)
        row = {key: name, "n_images": on["file"].nunique(), "n_gt": int(on["crack_tp"].notna().sum()),
               "n_damage": v_on["n_damage"], "n_none": v_on["n_none"]}
        for m in IMAGE_METRICS:
            row.update(off_on_diff(m, v_off[m], v_on[m]))
        row["baseline_none"] = v_on["baseline_none"]                   # 정답(기준선)은 끔·켬이 같다
        for m in SUMMARY_METRICS:
            row.update(off_on_diff(m, pd.to_numeric(off[m]).mean(), pd.to_numeric(on[m]).mean()))
        row.update(pr_columns(off, on, pr_fn))
        rows.append(row)
    columns = ([key, "n_images", "n_gt", "n_damage", "n_none"] + triple_names(IMAGE_METRICS) + ["baseline_none"]
               + triple_names(SUMMARY_METRICS + PR_METRICS))
    return pd.DataFrame(rows, columns=columns)


def write_config(path, notes=()):
    """이번 실행에 쓴 config.py의 대문자 값을 config.py에 쓴 순서대로 저장하고, notes가 있으면 맨 끝에 '# ' 줄로 붙인다
    (결과가 어떤 설정·실행 옵션에서 나왔는지 결과 폴더만 보고 알 수 있게)."""
    lines = [f"{name} = {value!r}" for name, value in vars(C).items() if name.isupper()]
    lines += [f"# {note}" for note in notes]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def failure_lines(failures):
    """실패 목록 [(사진 이름, 오류)]을 같은 오류끼리 묶어 한 줄씩 만든다.
    모든 사진이 같은 이유로 실패할 때(예: 아직 구현되지 않은 dark 방식) 같은 줄이 수십 번 나오지 않게 한다."""
    groups = {}
    for name, msg in failures:
        groups.setdefault(msg, []).append(name)
    return [f"{msg} ({names[0] if len(names) == 1 else f'{len(names)}장: {names[0]} 외 {len(names) - 1}장'})"
            for msg, names in groups.items()]


def main(argv=None, image_dirs=IMAGE_DIRS, roi_path="data/roi.csv", results_dir="results"):
    """사진 전체를 처리해 results/<name>/에 표·그림을 저장한다. 실패한 사진이 있으면 1, 없으면 0을 돌려준다.
    실행 옵션은 config_used.txt를 쓰기 전에 적용하고(파일에 바뀐 값이 남게), 실패해도 finally에서 되돌린다.
    image_dirs·roi_path·results_dir는 시험에서 임시 폴더를 쓰려고 둔 인자이고, 명령행 실행은 기본값을 쓴다."""
    args = parse_args(argv)
    overrides = runtime_overrides(args)
    files = list_images(args.only, image_dirs)
    roi_table = io_utils.load_roi_table(roi_path)
    out_dir = f"{results_dir}/{args.name}"
    fig_dir = f"{out_dir}/figures"
    os.makedirs(fig_dir, exist_ok=True)
    backup = apply_overrides(overrides)
    try:
        write_config(f"{out_dir}/config_used.txt", run_notes(args, overrides))
        if evaluate is None:
            print("안내: src/evaluate.py가 아직 없어 정답 비교 열은 빈칸으로 둡니다.")

        rows, failures, n_default = [], [], 0
        for path in files:
            try:
                image_rows, used_default = process_image(path, roi_table, fig_dir, args.roi)
            except Exception as e:                      # 한 장이 실패해도 나머지는 계속 처리
                failures.append((os.path.basename(path), f"{type(e).__name__}: {e}"))
                continue
            rows += image_rows                          # 끔·켬 두 행이 모두 만들어진 사진만 넣는다
            n_default += used_default

        df = metrics_frame(rows)
        df.to_csv(f"{out_dir}/metrics.csv", encoding="utf-8-sig", index=False)
        pr_fn = evaluate.precision_recall if evaluate else None
        tables = {"table_by_image.csv": table_by_image(df, pr_fn),
                  "table_by_source.csv": table_by_group(df, "source", pr_fn),
                  "table_by_condition.csv": table_by_group(df, "condition", pr_fn)}
        for name, table in tables.items():
            table.to_csv(f"{out_dir}/{name}", encoding="utf-8-sig", index=False)
    finally:
        restore_overrides(backup)

    roi_text = "기본 띠" if args.roi == "auto" else "기본 ROI"
    print(f"{len(files) - len(failures)}/{len(files)}장 처리 (--roi {args.roi}, {roi_text} {n_default}장) -> {out_dir}/")
    if failures:
        print(f"실패 {len(failures)}장 (같은 오류끼리 묶음):")
        for line in failure_lines(failures):
            print(f"  - {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
