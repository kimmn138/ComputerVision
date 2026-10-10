"""(파트 C) 제공·직접 촬영 영상 전체를 전처리 끔·켬으로 처리해 지표 표·비교 그림·요약 표를 저장한다.
사용법: 맨 위 폴더에서  python run_experiment.py --name <이름>  [--only provided|own]
결과: results/<name>/ (--name을 빼면 results/latest. 1차 결과 폴더 base·tuning·diagnosis에는 쓰지 않는다)
- metrics.csv: 영상마다 끔·켬 2행 (영상 정보, 상태 지표, row 9열, 정답 비교 tp·fp·found·fn)
- figures/<영상 이름>.png: 위 줄 끔·아래 줄 켬 비교 그림
- table_by_image.csv, table_by_source.csv, table_by_condition.csv: 지표_off·지표_on·지표_diff(on − off) 요약 표
- config_used.txt: 이번 실행에 쓴 config.py의 대문자 값
처리 시간은 끔·켬을 각각 C.TIME_REPEAT번 돌린 중앙값이고, 나머지 값은 첫 결과다.
영상 하나가 실패해도 나머지를 계속 처리하고, 실패가 있으면 끝에 목록을 출력하고 종료 코드 1로 끝난다.
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
# 요약 표에서 끔·켬을 비교하는 지표. 면적 %는 add_ratios가 더한다
SUMMARY_METRICS = ["edges", "harris", "n_crack", "area_crack", "area_crack_pct",
                   "n_pothole", "area_pothole", "area_pothole_pct", "t_pre_ms", "t_detect_ms"]
PR_METRICS = [f"{kind}_{name}" for kind in KINDS for name in ("precision", "recall")]
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
    """결과 폴더 이름(--name)과 처리할 사진 묶음(--only)을 명령행에서 읽는다. 1차 결과 폴더 이름은 거부한다."""
    parser = argparse.ArgumentParser(description="전처리 끔·켬 비교 실험")
    parser.add_argument("--name", default=DEFAULT_NAME, help=f"결과 폴더 이름 (results/<name>/, 기본 {DEFAULT_NAME})")
    parser.add_argument("--only", choices=list(IMAGE_DIRS), help="한쪽 사진만 처리")
    args = parser.parse_args(argv)
    check_name(parser, args.name)
    return args


def list_images(only=None):
    """처리할 사진 경로를 제공 → 직접 촬영 순서로, 폴더 안에서는 이름 순으로 모은다."""
    names = [only] if only else list(IMAGE_DIRS)
    return [p for name in names for p in sorted(glob.glob(f"{IMAGE_DIRS[name]}/*.jpg"))]


def get_roi(file_name, roi_table):
    """roi.csv에서 사진의 노면 범위를 찾고, 없으면 config 기본 ROI를 쓴다. (ROI dict, 기본값을 썼는지)를 돌려준다."""
    if file_name in roi_table:
        return roi_table[file_name], False
    return {"top": C.ROI_TOP_DEFAULT, "bottom": C.ROI_BOTTOM_DEFAULT, "condition": ""}, True


def repeat_pipeline(bgr, roi, use_preprocess):
    """한 설정(끔 또는 켬)을 C.TIME_REPEAT번 돌린 결과 list. 처리 시간의 중앙값을 내려고 반복한다."""
    return [pipeline.run_pipeline(bgr, roi["top"], roi["bottom"], use_preprocess)
            for _ in range(max(1, C.TIME_REPEAT))]


def median_times(outs):
    """첫 결과를 쓰되 row의 처리 시간만 반복 전체의 중앙값으로 바꾼 새 결과를 돌려준다.
    한 번 잰 시간은 캐시·다른 프로그램 때문에 튀므로 중앙값으로 튀는 값을 버린다(받은 결과는 바꾸지 않음)."""
    row = dict(outs[0]["row"])
    for key in TIME_COLUMNS:
        row[key] = round(statistics.median(o["row"][key] for o in outs), 2)
    return {**outs[0], "row": row}


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


def make_row(file_name, info, roi, use_preprocess, out, quality, counts):
    """metrics.csv 한 행: 영상 정보 + 상태 지표 + row 9열 + 정답 비교 개수."""
    return {"file": file_name, "source": info["source"],
            "condition": info.get("condition") or roi["condition"],
            "preprocess": "on" if use_preprocess else "off",
            "steps": analyze.steps_to_text(out["steps"]),
            **quality, **out["row"], **counts}


def save_figure(path, file_name, off, on):
    """끔(위 줄)·켬(아래 줄)마다 후보 그림·노면 흑백·캐니 에지 3칸을 한 장으로 저장한다."""
    panels = []
    for label, out in (("끔", off), ("켬", on)):
        drawn = visualize.draw_candidates(out["small"], out["cracks"], out["potholes"], out["y0"])
        panels += [(f"{label}: 후보", drawn), (f"{label}: 노면", out["proc"]), (f"{label}: 에지", out["edges"])]
    visualize.save_comparison(path, f"{file_name} | 켬: {analyze.steps_to_text(on['steps'])}", panels)


def process_image(path, roi_table, fig_dir):
    """사진 1장을 끔·켬으로 반복 처리해 metrics 2행을 만들고 비교 그림을 저장한다. 오류는 main이 잡는다.
    (행 2개, 기본 ROI를 썼는지)를 돌려준다."""
    file_name = os.path.basename(path)
    roi, used_default = get_roi(file_name, roi_table)
    if used_default:
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
    rows = [make_row(file_name, info, roi, use, out, quality, gt_counts(out, gt, count_fn, C.IOU_THRESH))
            for use, out in ((False, off), (True, on))]
    save_figure(os.path.join(fig_dir, f"{os.path.splitext(file_name)[0]}.png"), file_name, off, on)
    return rows, used_default


def quality_columns(rows):
    """metrics.csv의 상태 지표 열 이름. measure_quality의 키 순서를 따라 사진이 0장이어도 머리글이 남고,
    A가 키를 더하면 열도 자동으로 늘어난다."""
    keys = list(analyze.measure_quality(np.zeros((32, 32), np.uint8)))
    known = set(INFO_COLUMNS + ROW_COLUMNS + GT_COLUMNS + keys)
    return keys + list(dict.fromkeys(k for r in rows for k in r if k not in known))


def metrics_frame(rows):
    """행 list를 열 순서가 정해진 표로 만든다. 정답 개수는 빈칸을 허용하는 정수형(Int64)이라 1.0이 아닌 1로 저장된다."""
    df = pd.DataFrame(rows, columns=INFO_COLUMNS + quality_columns(rows) + ROW_COLUMNS + GT_COLUMNS)
    return df.astype({col: "Int64" for col in GT_COLUMNS})


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
    """영상 1장 = 1행: 켬의 steps와 상태 지표, 지표마다 끔·켬·차이, 그 영상 하나의 정밀도·재현율."""
    df = add_ratios(df)
    known = set(INFO_COLUMNS + ROW_COLUMNS + GT_COLUMNS + SUMMARY_METRICS)
    head = ["file", "source", "condition", "steps"] + [c for c in df.columns if c not in known] + ["roi_pixels"]
    off_all, on_all = df[df["preprocess"] == "off"], df[df["preprocess"] == "on"]
    rows = []
    for file_name in on_all["file"]:
        off, on = off_all[off_all["file"] == file_name], on_all[on_all["file"] == file_name]
        row = {c: on[c].iloc[0] for c in head}
        for m in SUMMARY_METRICS:
            row.update(off_on_diff(m, off[m].iloc[0], on[m].iloc[0]))
        row.update(pr_columns(off, on, pr_fn))
        rows.append(row)
    return pd.DataFrame(rows, columns=head + triple_names(SUMMARY_METRICS + PR_METRICS))


def table_by_group(df, key, pr_fn):
    """key(source 또는 condition)의 값마다 1행 + 전체(all) 1행: 영상 수, 정답 있는 영상 수,
    지표 평균의 끔·켬·차이, 개수 합계로 낸 정밀도·재현율. 빈 조건은 unknown으로 묶는다."""
    df = add_ratios(df)
    df[key] = df[key].fillna("").replace("", "unknown")
    rows = []
    for name, sub in [(g, df[df[key] == g]) for g in pd.unique(df[key])] + [("all", df)]:
        off, on = sub[sub["preprocess"] == "off"], sub[sub["preprocess"] == "on"]
        row = {key: name, "n_images": on["file"].nunique(), "n_gt": int(on["crack_tp"].notna().sum())}
        for m in SUMMARY_METRICS:
            row.update(off_on_diff(m, pd.to_numeric(off[m]).mean(), pd.to_numeric(on[m]).mean()))
        row.update(pr_columns(off, on, pr_fn))
        rows.append(row)
    return pd.DataFrame(rows, columns=[key, "n_images", "n_gt"] + triple_names(SUMMARY_METRICS + PR_METRICS))


def write_config(path):
    """이번 실행에 쓴 config.py의 대문자 값을 config.py에 쓴 순서대로 저장한다(결과가 어떤 설정에서 나왔는지 남기려고)."""
    lines = [f"{name} = {value!r}" for name, value in vars(C).items() if name.isupper()]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    """사진 전체를 처리해 results/<name>/에 표·그림을 저장한다. 실패한 사진이 있으면 1, 없으면 0을 돌려준다."""
    args = parse_args(argv)
    files = list_images(args.only)
    roi_table = io_utils.load_roi_table("data/roi.csv")
    out_dir = f"results/{args.name}"
    fig_dir = f"{out_dir}/figures"
    os.makedirs(fig_dir, exist_ok=True)
    write_config(f"{out_dir}/config_used.txt")
    if evaluate is None:
        print("안내: src/evaluate.py가 아직 없어 정답 비교 열은 빈칸으로 둡니다.")

    rows, failures, n_default = [], [], 0
    for path in files:
        try:
            image_rows, used_default = process_image(path, roi_table, fig_dir)
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

    print(f"{len(files) - len(failures)}/{len(files)}장 처리 (기본 ROI {n_default}장) -> {out_dir}/")
    if failures:
        print(f"실패 {len(failures)}장:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
