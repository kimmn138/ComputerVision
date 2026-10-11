"""메인 실험: 제공·직접 촬영 사진 전체를 전처리 끔·켬으로 처리해 지표와 비교 그림을 저장한다.
실행: python run_experiment.py --name <이름>   (옵션은 --help 참고)
결과: results/<이름>/
- metrics.csv: 사진마다 끔·켬 2행 (상태 지표, 에지·코너·후보 수, 처리 시간, 정답 비교)
- figures/: 사진마다 위 줄 끔·아래 줄 켬 비교 그림
- table_by_image/source/condition.csv: 끔·켬·차이 요약 표
- config_used.txt: 이번 실행에 쓴 파라미터
옵션으로 바꾼 config 값은 실행 중에만 적용되고 config.py 파일은 바뀌지 않는다.
"""
import argparse
import ast
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
    from src import evaluate  # 없으면 정답 비교 열을 빈칸으로 둠
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
# row 9열 외에 더 기록하는 값
EXTRA_COLUMNS = ["edges_road", "t_road_ms", "t_band_ms"]
RESULT_TIMES = ("t_road_ms", "t_band_ms")   # row 밖의 처리 시간 (반복 중앙값으로 바꿈)
# 영상 단위 판정 열
VERDICT_COLUMNS = ["has_gt", "has_damage", "n_pred", "hit_same", "hit_any", "correct"]
VERDICT_BOOL = ["has_damage", "hit_same", "hit_any", "correct"]     # 정답 파일이 없으면 빈칸
# table_by_image에 끔·켬으로 나란히 두는 판정
VERDICT_PAIRS = ["correct", "hit_same", "hit_any"]
# 요약 표에서 끔·켬을 비교하는 지표
SUMMARY_METRICS = ["edges", "edges_road", "harris", "n_crack", "area_crack", "area_crack_pct",
                   "n_pothole", "area_pothole", "area_pothole_pct", "t_pre_ms", "t_detect_ms", "t_road_ms", "t_band_ms"]
PR_METRICS = [f"{kind}_{name}" for kind in KINDS for name in ("precision", "recall")]
# source·condition 표의 영상 단위 지표 (정답 있는 사진만)
IMAGE_METRICS = ["sensitivity", "specificity", "accuracy"]
ROI_MODES = ("csv", "auto")         # csv = roi.csv의 사진별 범위, auto = 모든 사진에 기본 띠
DEFAULT_NAME = "latest"             # --name을 빼면 쓰는 결과 폴더 (매번 덮어씀)
# 덮어쓰지 않도록 막아 둔 결과 폴더 이름 (1차 기준 결과)
PROTECTED_NAMES = ("base", "tuning", "diagnosis")


def check_name(parser, name):
    """결과 폴더 이름이 보호된 폴더(PROTECTED_NAMES)면 실행을 멈춘다."""
    target = os.path.abspath(os.path.join("results", name)).lower()
    for protected in PROTECTED_NAMES:
        root = os.path.abspath(os.path.join("results", protected)).lower()
        if target == root or target.startswith(root + os.sep):
            parser.error(f"results/{protected}: 1차 결과(기준 수치의 출처)라 쓰지 않습니다. --name으로 다른 이름을 주세요.")


def parse_args(argv=None):
    """명령행 옵션을 읽는다. 효과가 없는 조합이나 범위 밖 값은 거부한다."""
    parser = argparse.ArgumentParser(description="전처리 끔·켬 비교 실험")
    parser.add_argument("--name", default=DEFAULT_NAME, help=f"결과 폴더 이름 (results/<name>/, 기본 {DEFAULT_NAME})")
    parser.add_argument("--only", choices=list(IMAGE_DIRS), help="한쪽 사진만 처리")
    parser.add_argument("--roi", choices=ROI_MODES, default="csv",
                        help="노면 띠: csv = roi.csv에 사진마다 정한 범위(기본), auto = roi.csv의 범위 없이 모든 사진에 기본 띠")
    parser.add_argument("--mask", choices=("on", "off"), help="노면 마스크(USE_ROAD_MASK)를 이번 실행에만 켬·끔")
    parser.add_argument("--detect", choices=("legacy", "dark"), help="검출 방식(DETECT_MODE)을 이번 실행에만 바꿈")
    parser.add_argument("--roi-top", type=float,
                        help="기본 띠의 위쪽 비율(ROI_TOP_DEFAULT)을 이번 실행에만 바꿈. --roi auto와 함께만. 예: 0.3 (E1b)")
    parser.add_argument("--auto-band", choices=("on", "off"),
                        help="자동 띠 높이(AUTO_BAND_TOP)를 이번 실행에만 켬·끔. 켜면 노면만 이어지는 만큼 띠 위쪽 경계를 위로 넓힘")
    parser.add_argument("--band-texture", choices=("on", "off"),
                        help="자동 띠의 결 조건(AUTO_BAND_USE_TEXTURE)을 이번 실행에만 켬·끔. 켜면 결이 시작 띠 노면과 비슷한 구간만 넓힘")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="config 값 하나를 이번 실행에만 바꿈(여러 번 가능). 예: --set ROAD_KEEP_RULE=ratio")
    args = parser.parse_args(argv)
    check_name(parser, args.name)
    check_sets(parser, args.set)
    if args.roi_top is not None:
        if args.roi != "auto":
            parser.error("--roi-top은 --roi auto와 함께만 쓴다: --roi csv에서는 39장 모두 roi.csv의 범위를 써서 효과가 없다")
        if not 0 <= args.roi_top < C.ROI_BOTTOM_DEFAULT:
            parser.error(f"--roi-top은 0 이상 {C.ROI_BOTTOM_DEFAULT}(ROI_BOTTOM_DEFAULT) 미만이어야 함 ({args.roi_top})")
    return args


def parse_set(item):
    """'NAME=VALUE'를 (이름, 값)으로 읽는다. 값은 파이썬 값으로, 안 되면 문자열로 읽는다.
    config에 없는 이름은 ValueError."""
    name, sep, text = item.partition("=")
    name = name.strip()
    if not sep or not name.isupper() or not hasattr(C, name):
        raise ValueError(f"--set은 config에 있는 대문자 이름=값이어야 함: {item!r}")
    try:
        value = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        value = text.strip()
    return name, value


def check_sets(parser, items):
    """--set 값들을 미리 검사해, 잘못된 것이 있으면 사용법과 함께 멈춘다."""
    for item in items:
        try:
            parse_set(item)
        except ValueError as e:
            parser.error(str(e))


def runtime_overrides(args):
    """명령행 옵션을 이번 실행에만 바꿀 config 값 {이름: 값}으로 바꾼다."""
    overrides = {}
    if args.mask is not None:
        overrides["USE_ROAD_MASK"] = args.mask == "on"
    if args.detect is not None:
        overrides["DETECT_MODE"] = args.detect
    if args.roi_top is not None:
        overrides["ROI_TOP_DEFAULT"] = args.roi_top
    if getattr(args, "auto_band", None) is not None:      # demo의 args에는 이 옵션이 없을 수 있음
        overrides["AUTO_BAND_TOP"] = args.auto_band == "on"
    if getattr(args, "band_texture", None) is not None:
        overrides["AUTO_BAND_USE_TEXTURE"] = args.band_texture == "on"
    for item in getattr(args, "set", []):
        name, value = parse_set(item)
        overrides[name] = value
    return overrides


def apply_overrides(overrides):
    """config 값을 실행 중에만 바꾸고, 되돌릴 원래 값 {이름: 값}을 돌려준다."""
    backup = {name: getattr(C, name) for name in overrides}
    for name, value in overrides.items():
        setattr(C, name, value)
    return backup


def restore_overrides(backup):
    """apply_overrides가 바꾼 config 값을 원래대로 되돌린다."""
    for name, value in backup.items():
        setattr(C, name, value)


def run_notes(args, overrides):
    """config_used.txt 끝에 붙일 두 줄(준 옵션, 그로 인해 바뀐 config 값)을 만든다."""
    options = [f"--roi {args.roi}"] + [f"--{flag} {value}" for flag, value in
                                       (("only", args.only), ("mask", args.mask), ("detect", args.detect), ("roi-top", args.roi_top),
                                        ("auto-band", args.auto_band), ("band-texture", args.band_texture))
                                       if value is not None] + [f"--set {item}" for item in getattr(args, "set", [])]
    changed = ", ".join(f"{name}={value!r}" for name, value in overrides.items()) or "없음"
    return [f"실행 옵션: {' '.join(options)}", f"실행 옵션으로 바꾼 config 값: {changed}"]


def list_images(only=None, image_dirs=IMAGE_DIRS):
    """처리할 사진 경로를 제공 → 직접 촬영 순서, 폴더 안은 이름 순으로 모은다."""
    names = [only] if only else list(image_dirs)
    return [p for name in names for p in sorted(glob.glob(f"{image_dirs[name]}/*.jpg"))]


def get_roi(file_name, roi_table, mode="csv"):
    """사진의 노면 띠 범위를 정한다. (ROI dict, 기본 띠를 썼는지)를 돌려준다.
    "csv"는 roi.csv 값(없으면 기본 띠), "auto"는 모든 사진에 기본 띠를 쓴다."""
    entry = roi_table.get(file_name)
    if mode == "csv" and entry is not None:
        return entry, False
    condition = entry["condition"] if entry else ""
    return {"top": C.ROI_TOP_DEFAULT, "bottom": C.ROI_BOTTOM_DEFAULT, "condition": condition}, True


def band_overrides(used_default):
    """사람이 roi.csv에 정한 띠는 자동으로 넓히지 않도록, 그 사진에만 AUTO_BAND_TOP을 끈다."""
    return {} if used_default else {"AUTO_BAND_TOP": False}


def repeat_pipeline(bgr, roi, use_preprocess):
    """처리 시간의 중앙값을 내려고 같은 설정을 C.TIME_REPEAT번 돌린 결과 list를 돌려준다."""
    return [pipeline.run_pipeline(bgr, roi["top"], roi["bottom"], use_preprocess)
            for _ in range(max(1, C.TIME_REPEAT))]


def median_times(outs):
    """첫 결과를 쓰되 처리 시간만 반복 전체의 중앙값으로 바꾼 새 결과를 돌려준다."""
    row = dict(outs[0]["row"])
    for key in TIME_COLUMNS:
        row[key] = round(statistics.median(o["row"][key] for o in outs), 2)
    out = {**outs[0], "row": row}
    for key in RESULT_TIMES:
        if all(key in o for o in outs):
            out[key] = round(statistics.median(o[key] for o in outs), 2)
    return out


def unstable_keys(outs):
    """반복마다 값이 달라진 row 열(시간 제외)을 찾는다. 정상이면 비어 있다."""
    first = outs[0]["row"]
    return sorted({k for o in outs[1:] for k, v in o["row"].items()
                   if k not in TIME_COLUMNS and v != first.get(k)})


def to_full_boxes(regions, y0):
    """노면 띠 좌표의 후보 bbox를 640px 전체 영상 좌표로 옮긴다(y + y0)."""
    return [(x, y + y0, w, h) for x, y, w, h in (r["bbox"] for r in regions)]


def gt_counts(out, gt, count_fn, thr):
    """균열·포트홀 종류별로 정답 박스와 맞춰 tp·fp·found·fn을 센다. 정답이 없으면 NaN."""
    if gt is None or count_fn is None:
        return {col: math.nan for col in GT_COLUMNS}
    counts = {}
    for kind, regions in (("crack", out["cracks"]), ("pothole", out["potholes"])):
        truth = [(x, y, w, h) for x, y, w, h, k in gt if k == kind]
        c = count_fn(to_full_boxes(regions, out["y0"]), truth, thr)
        counts.update({f"{kind}_{key}": int(c[key]) for key in COUNT_KEYS})
    return counts


def load_gt_for(file_name):
    """사진에 맞는 정답 파일 gt/<이름>.csv를 읽는다. 없으면 None."""
    if evaluate is None:
        return None
    return evaluate.load_gt(f"gt/{os.path.splitext(file_name)[0]}.csv")


def edges_on_road(edges, road):
    """노면 마스크 안에 있는 에지 픽셀 수를 센다."""
    return int(np.count_nonzero((edges > 0) & (road > 0)))


def verdict_columns(out, gt):
    """영상 단위 판정 열을 만든다. 정답 파일이 없으면 판정 칸은 빈칸(NaN)."""
    pred_by_kind = {kind: to_full_boxes(regions, out["y0"])
                    for kind, regions in (("crack", out["cracks"]), ("pothole", out["potholes"]))}
    if gt is None or evaluate is None:
        n_pred = sum(len(boxes) for boxes in pred_by_kind.values())
        return {"has_gt": False, "n_pred": n_pred, **{col: math.nan for col in VERDICT_BOOL}}
    return {"has_gt": True, **evaluate.image_verdict(pred_by_kind, gt)}


def make_row(file_name, info, roi, use_preprocess, out, quality, counts, verdict):
    """metrics.csv 한 행을 만든다: 사진 정보 + 상태 지표 + 측정값 + 정답 비교 + 판정."""
    return {"file": file_name, "source": info["source"],
            "condition": info.get("condition") or roi["condition"],
            "preprocess": "on" if use_preprocess else "off",
            "steps": analyze.steps_to_text(out["steps"]),
            **quality, **out["row"],
            "edges_road": edges_on_road(out["edges"], out["road"]), "t_road_ms": out["t_road_ms"], "t_band_ms": out["t_band_ms"],
            **counts, **verdict}


def save_figure(path, file_name, off, on):
    """끔(위 줄)·켬(아래 줄)마다 후보(노면 초록 표시)·노면 흑백·에지 3칸을 한 그림으로 저장한다."""
    panels = []
    for label, out in (("끔", off), ("켬", on)):
        road = visualize.draw_road(out["small"], out["road"], out["y0"])
        drawn = visualize.draw_candidates(road, out["cracks"], out["potholes"], out["y0"])
        panels += [(f"{label}: 노면(초록)·후보", drawn), (f"{label}: 노면", out["proc"]), (f"{label}: 에지", out["edges"])]
    visualize.save_comparison(path, f"{file_name} | 켬: {analyze.steps_to_text(on['steps'])}", panels)


def process_image(path, roi_table, fig_dir, roi_mode="csv"):
    """사진 1장을 끔·켬으로 처리해 metrics 2행을 만들고 비교 그림을 저장한다.
    (행 2개, 기본 띠를 썼는지)를 돌려준다."""
    file_name = os.path.basename(path)
    roi, used_default = get_roi(file_name, roi_table, roi_mode)
    if used_default and roi_mode == "csv":
        print(f"경고: roi.csv에 없음 → 기본 ROI ({C.ROI_TOP_DEFAULT}, {C.ROI_BOTTOM_DEFAULT}): {file_name}")
    info = io_utils.parse_name(path)
    bgr = io_utils.load_image(path)

    backup = apply_overrides(band_overrides(used_default))
    try:
        runs = {use: repeat_pipeline(bgr, roi, use) for use in (False, True)}   # 끔 n번 → 켬 n번
    finally:
        restore_overrides(backup)
    for use, outs in runs.items():
        bad = unstable_keys(outs)
        if bad:
            print(f"경고: 반복마다 값이 다름 ({'켬' if use else '끔'}: {', '.join(bad)}): {file_name}")
    off, on = median_times(runs[False]), median_times(runs[True])

    quality = on["quality"]          # 상태 지표는 입력 사진의 성질이라 끔 행에도 같은 값을 씀
    gt = load_gt_for(file_name)
    count_fn = evaluate.count_matches if evaluate else None
    rows = [make_row(file_name, info, roi, use, out, quality, gt_counts(out, gt, count_fn, C.IOU_THRESH), verdict_columns(out, gt))
            for use, out in ((False, off), (True, on))]
    save_figure(os.path.join(fig_dir, f"{os.path.splitext(file_name)[0]}.png"), file_name, off, on)
    return rows, used_default


def quality_columns(rows):
    """metrics.csv의 상태 지표 열 이름을 measure_quality의 키 순서대로 돌려준다."""
    keys = list(analyze.measure_quality(np.zeros((32, 32), np.uint8)))
    known = set(INFO_COLUMNS + ROW_COLUMNS + EXTRA_COLUMNS + GT_COLUMNS + VERDICT_COLUMNS + keys)
    return keys + list(dict.fromkeys(k for r in rows for k in r if k not in known))


def metrics_frame(rows):
    """행 list를 열 순서가 정해진 표로 만든다. 개수·판정 열은 빈칸을 허용하는 형식으로 둔다."""
    columns = INFO_COLUMNS + quality_columns(rows) + ROW_COLUMNS + EXTRA_COLUMNS + GT_COLUMNS + VERDICT_COLUMNS
    types = {col: "Int64" for col in GT_COLUMNS + ["edges_road", "n_pred"]}
    types.update({col: "boolean" for col in ["has_gt"] + VERDICT_BOOL})
    return pd.DataFrame(rows, columns=columns).astype(types)


def add_ratios(df):
    """균열·포트홀 면적을 노면 넓이 대비 %(area_*_pct)로 나타낸 열을 더한 새 표를 돌려준다."""
    out = df.copy()
    roi = pd.to_numeric(out["roi_pixels"]).where(lambda s: s > 0)     # 0으로 나누지 않게 빈칸 처리
    for kind in KINDS:
        out[f"area_{kind}_pct"] = pd.to_numeric(out[f"area_{kind}"]) / roi * 100
    return out


def pr_from_counts(df, kind, pr_fn):
    """정답 있는 행의 개수를 모두 더한 뒤 정밀도·재현율을 계산한다. 정답 있는 행이 없으면 (NaN, NaN)."""
    has_gt = df[f"{kind}_tp"].notna()
    if pr_fn is None or not has_gt.any():
        return math.nan, math.nan
    return pr_fn({key: int(df.loc[has_gt, f"{kind}_{key}"].sum()) for key in COUNT_KEYS})


def off_on_diff(name, off, on):
    """한 지표의 끔·켬 값과 차이(on − off)를 _off·_on·_diff 열로 만든다."""
    return {f"{name}_off": off, f"{name}_on": on, f"{name}_diff": on - off}


def triple_names(names):
    """지표 이름마다 _off·_on·_diff 열 이름을 늘어놓는다."""
    return [f"{n}_{s}" for n in names for s in ("off", "on", "diff")]


def pr_columns(off, on, pr_fn):
    """균열·포트홀 정밀도·재현율의 끔·켬·차이 열을 만든다."""
    cols = {}
    for kind in KINDS:
        (p_off, r_off), (p_on, r_on) = pr_from_counts(off, kind, pr_fn), pr_from_counts(on, kind, pr_fn)
        cols.update(off_on_diff(f"{kind}_precision", p_off, p_on))
        cols.update(off_on_diff(f"{kind}_recall", r_off, r_on))
    return cols


def table_by_image(df, pr_fn):
    """사진 1장 = 1행인 요약 표: 고른 전처리, 판정, 지표마다 끔·켬·차이."""
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
    """정답 있는 사진의 판정을 모아 민감도·특이도·정답률을 낸다."""
    has_gt = sub["has_gt"].fillna(False).astype(bool)
    verdicts = [{"has_damage": bool(d), "correct": bool(c)}
                for d, c in zip(sub.loc[has_gt, "has_damage"], sub.loc[has_gt, "correct"])]
    if evaluate is None:
        return {"n_damage": 0, "n_none": 0, **{m: math.nan for m in IMAGE_METRICS + ["baseline_none"]}}
    return evaluate.summarize_verdicts(verdicts)


def table_by_group(df, key, pr_fn):
    """key(source 또는 condition) 값마다 1행 + 전체(all) 1행인 요약 표: 판정 지표와 지표 평균의 끔·켬·차이."""
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
        row["baseline_none"] = v_on["baseline_none"]                   # 기준선은 끔·켬이 같음
        for m in SUMMARY_METRICS:
            row.update(off_on_diff(m, pd.to_numeric(off[m]).mean(), pd.to_numeric(on[m]).mean()))
        row.update(pr_columns(off, on, pr_fn))
        rows.append(row)
    columns = ([key, "n_images", "n_gt", "n_damage", "n_none"] + triple_names(IMAGE_METRICS) + ["baseline_none"]
               + triple_names(SUMMARY_METRICS + PR_METRICS))
    return pd.DataFrame(rows, columns=columns)


def write_config(path, notes=()):
    """이번 실행에 쓴 config 값을 파일로 저장한다(결과가 어떤 설정에서 나왔는지 남기려고)."""
    lines = [f"{name} = {value!r}" for name, value in vars(C).items() if name.isupper()]
    lines += [f"# {note}" for note in notes]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def failure_lines(failures):
    """실패 목록 [(사진 이름, 오류)]을 같은 오류끼리 묶어 한 줄씩 만든다."""
    groups = {}
    for name, msg in failures:
        groups.setdefault(msg, []).append(name)
    return [f"{msg} ({names[0] if len(names) == 1 else f'{len(names)}장: {names[0]} 외 {len(names) - 1}장'})"
            for msg, names in groups.items()]


def main(argv=None, image_dirs=IMAGE_DIRS, roi_path="data/roi.csv", results_dir="results"):
    """사진 전체를 처리해 results/<이름>/에 표·그림을 저장한다. 실패한 사진이 있으면 1, 없으면 0을 돌려준다."""
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
            rows += image_rows
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
