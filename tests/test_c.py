"""파트 C 시험: features의 에지·해리스 점 세기와 SIFT 매칭, visualize의 후보 그리기·비교 그림 저장,
run_experiment의 좌표 변환·시간 중앙값·요약 계산, run_matching의 쌍 묶기·inlier 그림·결과 저장, demo의 ROI 선택·화면 맞춤·제목 띠,
run_all의 실험 목록·채택 비교·거부 조건이 경계 입력에서도 약속대로 동작하는지 확인한다.
사용법: 맨 위 폴더에서  python tests/test_c.py"""
import collections
import contextlib
import functools
import io
import math
import os
import struct
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import demo  # noqa: E402
import run_experiment as rx  # noqa: E402
import run_matching as rm  # noqa: E402
from src import analyze, detect, evaluate, features, io_utils, pipeline, visualize  # noqa: E402
from src import config as C  # noqa: E402

FAILS = []


def check(ok, msg):
    if not ok:
        FAILS.append(msg)


def todo(reason):
    """아직 구현하지 않은 기능의 시험 표시: 기대값은 적어 두되 지금은 건너뛴다(unittest.SkipTest).
    구현한 사람이 이 줄(@todo)을 지우면 시험이 돈다. 이 시험이 그 작업의 통과 기준이다(docs/tasks/C.md 4장)."""
    def mark(fn):
        @functools.wraps(fn)
        def skipped(*args, **kwargs):
            raise unittest.SkipTest(reason)
        return skipped
    return mark


@contextlib.contextmanager
def patched(obj, name, value):
    """obj.name을 잠시 value(시험용 가짜 함수나 config 값)로 바꿨다가, 끝나면(실패해도) 되돌린다."""
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def parse_quietly(parse, argv):
    """명령행 읽기 함수 parse(argv)를 부른다. 거부되면(argparse의 SystemExit) None. 오류 문구는 화면에 내지 않는다.
    SystemExit은 Exception이 아니라 main의 except가 못 잡아 시험 전체가 멈추므로 여기서 None으로 바꾼다."""
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            return parse(argv)
    except SystemExit:
        return None


def square(size, value):
    """검은 바탕에 회색 사각형 하나: 해리스 코너 4개, SIFT 특징점이 아주 적은 영상."""
    img = np.zeros((240, 640), np.uint8)
    y, x = 120 - size // 2, 320 - size // 2
    img[y:y + size, x:x + size] = value
    return img


def texture():
    """무늬가 많은 영상과 그것을 (5, 12)픽셀 민 영상: 매칭이 잘 되는 쌍."""
    rng = np.random.default_rng(0)
    t = cv.GaussianBlur(rng.integers(0, 256, (240, 640), dtype=np.uint8), (0, 0), 2)
    return t, np.roll(t, (5, 12), axis=(0, 1))


def test_harris():
    """빈 영상은 0, 사각형 하나는 모서리 4개."""
    blank = np.zeros((240, 640), np.uint8)
    check(features.count_harris(blank) == 0, "count_harris: 빈 영상이 0이 아님")
    n = features.count_harris(square(80, 255))
    check(n == 4, f"count_harris: 사각형 하나에서 4가 아님 ({n})")


def test_count_edges():
    """에지가 없으면 0, 가로 10픽셀 흰 선이면 10."""
    blank = np.zeros((240, 640), np.uint8)
    n = features.count_edges(blank)
    check(n == 0, f"count_edges: 빈 에지 영상이 0이 아님 ({n})")
    line = blank.copy()
    line[100, 50:60] = 255
    n = features.count_edges(line)
    check(n == 10, f"count_edges: 가로 10픽셀 흰 선이 10이 아님 ({n})")
    check(not blank.any(), "count_edges가 입력 배열을 직접 바꿈")


def test_draw_candidates():
    """균열 윤곽은 y0만큼 내려 빨강으로, 노면 시작선은 y0 줄에 노랑으로 그리고, 받은 영상은 그대로."""
    gray = np.full((300, 640, 3), 128, np.uint8)
    contour = np.array([[[10, 20]], [[200, 20]], [[200, 60]]], np.int32)
    out = visualize.draw_candidates(gray, [{"contour": contour, "bbox": (10, 20, 191, 41)}], [], 100)
    check((gray == 128).all(), "draw_candidates가 입력 영상을 직접 바꿈")
    for x, y in contour.reshape(-1, 2):
        check(out[y + 100, x].tolist() == [0, 0, 255],
              f"draw_candidates: 윤곽 점 ({x}, {y})이 (x, y + y0)에 빨강으로 그려지지 않음 ({out[y + 100, x].tolist()})")
        check(out[y, x].tolist() != [0, 0, 255],
              f"draw_candidates: 윤곽 점 ({x}, {y})이 y0를 더하지 않은 자리에 그려짐")
    check(out[100, 320].tolist() == [0, 255, 255],
          f"draw_candidates: y0 줄의 노면 시작선이 노랑이 아님 ({out[100, 320].tolist()})")


def test_match_blank():
    """빈 영상끼리: 특징점이 없어도 오류 없이 모두 0."""
    blank = np.zeros((240, 640), np.uint8)
    m = features.match_pair(blank, blank)
    check(m == {"kp1": 0, "kp2": 0, "good": 0, "inliers": 0, "inlier_ratio": 0.0,
                "k1": [], "k2": [], "matches": [], "mask": None},
          f"match_pair: 빈 영상 결과가 모두 0이 아님 ({m})")


def test_match_few():
    """good이 RANSAC_MIN_GOOD 미만이면 RANSAC을 건너뛰고 inliers 0, mask None."""
    img = square(40, 128)
    m = features.match_pair(img, np.zeros_like(img))
    check(m["good"] < C.RANSAC_MIN_GOOD, f"시험 영상의 good이 기준보다 많음 ({m['good']})")
    check(m["kp2"] == 0 and m["inliers"] == 0 and m["inlier_ratio"] == 0.0 and m["mask"] is None,
          "match_pair: good이 기준 미만인데 inlier가 0이 아님")
    check(m["good"] == len(m["matches"]), "match_pair: good 수와 matches 길이가 다름")


def test_match_boundary():
    """good이 정확히 RANSAC_MIN_GOOD이면 RANSAC을 돌려 mask가 good 길이의 list가 된다."""
    img = square(40, 128)
    m = features.match_pair(img, img)
    check(m["good"] == C.RANSAC_MIN_GOOD, f"시험 영상의 good이 기준과 같지 않음 ({m['good']})")
    check(isinstance(m["mask"], list) and len(m["mask"]) == m["good"],
          "match_pair: good이 기준과 같은데 RANSAC을 건너뜀")
    check(0 <= m["inlier_ratio"] <= 1, "match_pair: inlier_ratio가 0~1 밖")


def test_match_texture():
    """무늬 영상과 민 영상: inlier가 생기고 mask 길이는 good과 같다."""
    a, b = texture()
    m = features.match_pair(a, b)
    check(m["good"] >= C.RANSAC_MIN_GOOD and m["inliers"] > 0, f"match_pair: 민 영상 매칭 실패 ({m['good']})")
    check(isinstance(m["inliers"], int) and 0 <= m["inlier_ratio"] <= 1, "match_pair: inlier 형식 오류")
    check(isinstance(m["mask"], list) and len(m["mask"]) == m["good"],
          "match_pair: mask가 good 길이의 list가 아님")


def test_match_self():
    """무늬 영상과 그 자신: 모든 good이 inlier라 inlier_ratio 1.0."""
    a, _ = texture()
    m = features.match_pair(a, a)
    check(m["good"] >= C.RANSAC_MIN_GOOD, f"match_pair: 자기 자신과의 good이 기준보다 적음 ({m['good']})")
    check(m["inliers"] == m["good"] and m["inlier_ratio"] == 1.0,
          f"match_pair: 자기 자신과의 매칭에서 inlier가 good과 다름 (good {m['good']}, inliers {m['inliers']}, "
          f"inlier_ratio {m['inlier_ratio']})")


def test_no_inplace():
    """받은 영상을 바꾸지 않는다."""
    a, b = texture()
    a0, b0 = a.copy(), b.copy()
    features.count_harris(a)
    features.match_pair(a, b)
    check(np.array_equal(a, a0) and np.array_equal(b, b0), "features가 입력 배열을 직접 바꿈")


def test_save_comparison():
    """흑백 4장·컬러 2장을 한글 제목으로 저장: 파일 생성, 입력 그대로, 그림 닫힘·Agg, 글꼴 경고 없음, 6장 아니면 ValueError."""
    a, b = texture()
    color = cv.cvtColor(a, cv.COLOR_GRAY2BGR)
    imgs = [a, np.zeros_like(a), color, b, (b > 128).astype(np.uint8) * 255, color.copy()]
    names = ["끔: 노면", "끔: 에지", "끔: 후보", "켬: 노면", "켬: 에지", "켬: 후보"]
    before = [im.copy() for im in imgs]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub", "cmp.png")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            visualize.save_comparison(path, "전처리 전·후 비교", list(zip(names, imgs)))
        check(os.path.isfile(path) and os.path.getsize(path) > 0, "save_comparison: 파일이 저장되지 않음")
        glyph = [str(x.message) for x in w if "missing from font" in str(x.message)]
        check(not glyph, f"save_comparison: 한글 글꼴 경고 {len(glyph)}건 (예: {glyph[:1]})")
    check(all(np.array_equal(x, y) for x, y in zip(imgs, before)), "save_comparison이 입력 영상을 직접 바꿈")
    check(not plt.get_fignums(), "save_comparison: 저장 뒤 plt.close로 닫지 않아 그림이 남음")
    check(plt.get_backend().lower() == "agg", f"save_comparison: 백엔드가 Agg가 아님 ({plt.get_backend()})")
    try:
        visualize.save_comparison(os.path.join(tempfile.gettempdir(), "x.png"), "t", list(zip(names, imgs))[:5])
        check(False, "save_comparison: 5장인데 ValueError가 나지 않음")
    except ValueError:
        pass


def fake_pr(c):
    """evaluate.precision_recall 대신 쓰는 시험용 계산(분모가 0이면 NaN)."""
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else math.nan
    r = c["found"] / (c["found"] + c["fn"]) if c["found"] + c["fn"] else math.nan
    return p, r


def metric_row(file, source, condition, pre, edges, area_crack, roi, counts=None):
    """metrics.csv 한 행 모양의 시험용 dict. counts가 없으면 정답 열은 NaN(정답 없음)."""
    row = {"file": file, "source": source, "condition": condition, "preprocess": pre, "steps": "없음",
           "mean": 100.0, "std": 10.0, "lap_var": 0.0, "noise": 0.0,
           "edges": edges, "harris": 0, "n_crack": 1, "area_crack": area_crack, "n_pothole": 0, "area_pothole": 0,
           "roi_pixels": roi, "t_pre_ms": 1.0, "t_detect_ms": 2.0}
    row.update({col: math.nan for col in rx.GT_COLUMNS})
    if counts:
        row.update({f"crack_{k}": v for k, v in zip(rx.COUNT_KEYS, counts)})
        row.update({f"pothole_{k}": 0 for k in rx.COUNT_KEYS})
    return row


def test_get_roi():
    """roi.csv에 있으면 그 값, 없으면 config 기본 ROI와 '기본값 씀' 표시."""
    table = {"a.jpg": {"top": 0.3, "bottom": 0.9, "condition": "dark"}}
    check(rx.get_roi("a.jpg", table) == (table["a.jpg"], False), "get_roi: roi.csv 값을 쓰지 않음")
    roi, used = rx.get_roi("b.jpg", table)
    check(used and (roi["top"], roi["bottom"]) == (C.ROI_TOP_DEFAULT, C.ROI_BOTTOM_DEFAULT),
          "get_roi: 없는 사진에 기본 ROI를 쓰지 않음")


def test_median_times():
    """시간 두 열만 중앙값, 나머지는 첫 결과. 받은 결과는 그대로."""
    outs = [{"row": {"edges": e, "t_pre_ms": t, "t_detect_ms": d}} for e, t, d in ((10, 5.0, 9.0), (10, 1.0, 1.0), (10, 3.0, 4.0))]
    out = rx.median_times(outs)
    check(out["row"] == {"edges": 10, "t_pre_ms": 3.0, "t_detect_ms": 4.0}, f"median_times: 중앙값이 틀림 ({out['row']})")
    check(outs[0]["row"]["t_pre_ms"] == 5.0, "median_times가 받은 결과를 직접 바꿈")
    outs[1]["row"]["edges"] = 11
    check(rx.unstable_keys(outs) == ["edges"], "unstable_keys: 반복마다 달라진 열을 찾지 못함")


def test_gt_counts():
    """후보 bbox는 y + y0로 옮겨 종류별로 비교하고, 정답이 없으면 0이 아닌 NaN."""
    out = {"cracks": [{"bbox": (1, 2, 3, 4)}], "potholes": [], "y0": 100}
    check(rx.to_full_boxes(out["cracks"], 100) == [(1, 102, 3, 4)], "to_full_boxes: y0를 더하지 않음")
    full = rx.to_full_boxes([{"bbox": (10, 20, 30, 40)}], 100)
    check(full == [(10, 120, 30, 40)], f"to_full_boxes: (10, 20, 30, 40), y0 100이 (10, 120, 30, 40)이 아님 ({full})")
    calls = []

    def fake_count(pred, truth, thr):
        calls.append((pred, truth, thr))
        return {"tp": len(pred), "fp": 0, "found": len(truth), "fn": 0}

    gt = [(1, 102, 3, 4, "crack"), (5, 5, 5, 5, "pothole")]
    c = rx.gt_counts(out, gt, fake_count, 0.3)
    check(calls == [([(1, 102, 3, 4)], [(1, 102, 3, 4)], 0.3), ([], [(5, 5, 5, 5)], 0.3)],
          f"gt_counts: 종류별 좌표 변환·비교가 틀림 ({calls})")
    check(c["crack_tp"] == 1 and c["pothole_found"] == 1 and len(c) == 8, f"gt_counts: 개수 열이 틀림 ({c})")
    none = rx.gt_counts(out, None, fake_count, 0.3)
    check(all(math.isnan(v) for v in none.values()), "gt_counts: 정답이 없는데 NaN이 아님")


def test_ratios_and_pr():
    """면적 %는 노면 0픽셀이면 빈칸, 정밀도·재현율은 개수를 합한 뒤 한 번 계산(영상별 평균이 아님)."""
    rows = [metric_row("a.jpg", "own", "dark", "on", 10, 50, 1000, (1, 1, 1, 0)),
            metric_row("b.jpg", "own", "dark", "on", 10, 5, 0, (3, 0, 1, 1)),
            metric_row("c.jpg", "own", "dark", "on", 10, 5, 100)]
    df = rx.metrics_frame(rows)
    pct = rx.add_ratios(df)["area_crack_pct"]
    check(pct[0] == 5.0 and math.isnan(pct[1]) and pct[2] == 5.0, f"add_ratios: % 계산이 틀림 ({list(pct)})")
    check("area_crack_pct" not in df.columns, "add_ratios가 받은 표를 직접 바꿈")
    p, r = rx.pr_from_counts(df, "crack", fake_pr)
    check((p, r) == (4 / 5, 2 / 3), f"pr_from_counts: 합계 방식이 아님 ({p}, {r})")   # 영상별 평균이면 p = 0.75
    check(all(math.isnan(v) for v in rx.pr_from_counts(df.iloc[2:], "crack", fake_pr)),
          "pr_from_counts: 정답이 없는 묶음이 NaN이 아님")
    check(all(math.isnan(v) for v in rx.pr_from_counts(df, "crack", None)), "pr_from_counts: 계산 함수가 없는데 NaN이 아님")
    two = rx.metrics_frame([metric_row("d.jpg", "own", "dark", "on", 10, 5, 100, (1, 0, 1, 0)),
                            metric_row("e.jpg", "own", "dark", "on", 10, 5, 100, (0, 9, 0, 1))])
    p, _ = rx.pr_from_counts(two, "crack", fake_pr)
    check(p == 0.1, f"pr_from_counts: tp 1·fp 0 + tp 0·fp 9의 합계 정밀도가 0.1이 아님 ({p}, 영상별 평균이면 0.5)")


def test_summary_tables():
    """요약 표 3개: 지표_off·_on·_diff 열, 빈 조건은 unknown, 끝에 all 행, 0장이어도 머리글."""
    rows = [metric_row("a.jpg", "provided", "", "off", 10, 50, 1000), metric_row("a.jpg", "provided", "", "on", 16, 80, 1000),
            metric_row("b.jpg", "own", "dark", "off", 20, 0, 500, (0, 0, 0, 1)),
            metric_row("b.jpg", "own", "dark", "on", 30, 10, 500, (1, 0, 1, 0))]
    df = rx.metrics_frame(rows)
    img = rx.table_by_image(df, fake_pr)
    check(list(img["file"]) == ["a.jpg", "b.jpg"] and list(img["edges_diff"]) == [6, 10],
          "table_by_image: 영상별 끔·켬 차이가 틀림")
    check(math.isnan(img["crack_recall_off"][0]) and img["crack_recall_diff"][1] == 1.0,
          "table_by_image: 정답 없는 영상이 빈칸이 아니거나 재현율 차이가 틀림")
    src = rx.table_by_group(df, "source", fake_pr)
    check(list(src["source"]) == ["provided", "own", "all"], f"table_by_source: 행이 틀림 ({list(src['source'])})")
    total = src.iloc[2]
    check(total["n_images"] == 2 and total["n_gt"] == 1 and total["edges_on"] == 23 and total["area_crack_pct_on"] == 5.0,
          "table_by_source: all 행의 평균·개수가 틀림")
    cond = rx.table_by_group(df, "condition", fake_pr)
    check(list(cond["condition"]) == ["unknown", "dark", "all"], f"table_by_condition: 행이 틀림 ({list(cond['condition'])})")
    empty = rx.metrics_frame([])
    check("crack_tp" in empty.columns and "mean" in empty.columns, "metrics_frame: 0장일 때 머리글이 없음")
    check("edges_diff" in rx.table_by_image(empty, fake_pr).columns
          and list(rx.table_by_group(empty, "source", fake_pr)["source"]) == ["all"],
          "요약 표: 0장일 때 머리글이나 all 행이 없음")


def test_run_options():
    """run_experiment 실행 옵션: 아무것도 안 주면 roi.csv 띠에 바꾸는 config 값이 없다(1차와 같은 조건).
    --mask·--detect·--roi-top은 이번 실행에만 쓸 config 값이 되고, --roi auto는 마스크를 저절로 켜지 않는다(한 단계씩 비교하려고).
    --roi-top은 --roi auto와 함께만 받는다(--roi csv에서는 39장에 효과가 없음). 범위 밖·모르는 값도 거부하고, 바꾼 값은 되돌아온다."""
    args = parse_quietly(rx.parse_args, [])
    check(args is not None and args.roi == "csv" and rx.runtime_overrides(args) == {},
          f"run_experiment: 옵션이 없는데 roi.csv 띠가 아니거나 바꾸는 값이 있음 ({args})")
    auto = parse_quietly(rx.parse_args, ["--roi", "auto"])
    check(auto is not None and rx.runtime_overrides(auto) == {}, "run_experiment: --roi auto가 마스크 등 다른 값까지 바꿈")
    args = parse_quietly(rx.parse_args, ["--roi", "auto", "--mask", "on", "--detect", "dark", "--roi-top", "0.3"])
    expect = {"USE_ROAD_MASK": True, "DETECT_MODE": "dark", "ROI_TOP_DEFAULT": 0.3}
    check(args is not None and rx.runtime_overrides(args) == expect, f"run_experiment: 실행 옵션을 config 값으로 못 바꿈 ({args})")
    off = parse_quietly(rx.parse_args, ["--mask", "off", "--detect", "legacy"])
    check(off is not None and rx.runtime_overrides(off) == {"USE_ROAD_MASK": False, "DETECT_MODE": "legacy"},
          "run_experiment: --mask off·--detect legacy를 config 값으로 못 바꿈")
    for bad in (["--roi-top", "0.3"], ["--roi", "csv", "--roi-top", "0.3"], ["--roi", "auto", "--roi-top", "1.0"],
                ["--roi", "auto", "--roi-top", "-0.1"], ["--roi", "manual"], ["--detect", "fast"], ["--mask", "yes"]):
        check(parse_quietly(rx.parse_args, bad) is None, f"run_experiment: 잘못된 옵션을 받아들임 ({' '.join(bad)})")
    if args is not None:
        notes = rx.run_notes(args, expect)
        check(notes == ["실행 옵션: --roi auto --mask on --detect dark --roi-top 0.3",
                        "실행 옵션으로 바꾼 config 값: USE_ROAD_MASK=True, DETECT_MODE='dark', ROI_TOP_DEFAULT=0.3"],
              f"run_notes: 실행 조건 줄이 틀림 ({notes})")
    before = {name: getattr(C, name) for name in expect}
    backup = rx.apply_overrides(expect)
    try:
        check(C.USE_ROAD_MASK is True and C.DETECT_MODE == "dark" and C.ROI_TOP_DEFAULT == 0.3, "apply_overrides: config에 적용되지 않음")
    finally:
        rx.restore_overrides(backup)
    check({name: getattr(C, name) for name in expect} == before, "restore_overrides: 바꾼 config 값이 되돌아가지 않음")


def test_get_roi_auto():
    """--roi auto: roi.csv의 범위는 쓰지 않고 모든 사진에 config 기본 띠를 쓴다. 상태 태그(condition)는 사진의 성질이라 roi.csv에서 그대로 가져온다.
    기본 띠는 import 때가 아니라 부를 때 config를 읽어야 --roi-top이 먹는다(인계 메모 5.3)."""
    table = {"a.jpg": {"top": 0.3, "bottom": 0.9, "condition": "dark"}}
    roi, used = rx.get_roi("a.jpg", table, "auto")
    check(used and (roi["top"], roi["bottom"], roi["condition"]) == (C.ROI_TOP_DEFAULT, C.ROI_BOTTOM_DEFAULT, "dark"),
          f"get_roi auto: roi.csv 범위를 쓰거나 상태 태그를 잃음 ({roi})")
    roi, used = rx.get_roi("b.jpg", table, "auto")
    check(used and roi["condition"] == "", f"get_roi auto: roi.csv에 없는 사진의 상태 태그가 빈칸이 아님 ({roi})")
    check(rx.get_roi("a.jpg", table, "csv") == rx.get_roi("a.jpg", table) == (table["a.jpg"], False), "get_roi: 기본 방식이 csv가 아님")
    with patched(C, "ROI_TOP_DEFAULT", 0.3):
        check(rx.get_roi("a.jpg", table, "auto")[0]["top"] == 0.3, "get_roi auto: 실행 중에 바꾼 ROI_TOP_DEFAULT를 읽지 않음")


def test_median_road_time():
    """노면 마스크 시간 t_road_ms도 row의 두 시간 열처럼 반복의 중앙값으로 바꾼다(인계 메모 4.4). 받은 결과는 바꾸지 않는다."""
    outs = [{"row": {"edges": 1, "t_pre_ms": 1.0, "t_detect_ms": 1.0}, "t_road_ms": t} for t in (5.0, 1.0, 3.0)]
    out = rx.median_times(outs)
    check(out["t_road_ms"] == 3.0 and outs[0]["t_road_ms"] == 5.0,
          f"median_times: t_road_ms가 중앙값이 아니거나 받은 결과를 바꿈 ({out['t_road_ms']})")


def test_edges_on_road():
    """노면 안 에지 수: 에지(0/255) 중 노면(255) 위에 있는 것만 센다. 노면이 띠 전체(지금의 road_mask 자리)면 띠 전체 에지 수와 같다."""
    edges = np.zeros((10, 20), np.uint8)
    edges[:, ::2] = 255                                              # 에지 100개
    road = np.zeros((10, 20), np.uint8)
    road[:, :10] = 255                                               # 왼쪽 절반만 노면 → 에지 50개
    before = (edges.copy(), road.copy())
    full = np.full((10, 20), 255, np.uint8)
    check(rx.edges_on_road(edges, full) == features.count_edges(edges) == 100, "edges_on_road: 노면이 띠 전체인데 띠 전체 에지 수와 다름")
    check(rx.edges_on_road(edges, road) == 50, f"edges_on_road: 노면 안 에지만 세지 않음 ({rx.edges_on_road(edges, road)})")
    check(np.array_equal(edges, before[0]) and np.array_equal(road, before[1]), "edges_on_road가 받은 영상을 바꿈")


def test_verdict_columns():
    """영상 단위 판정 열: 정답이 있으면 image_verdict(중심 기준)로 채우고, 정답 파일이 없으면 판정 4칸은 빈칸(has_gt·n_pred는 채움).
    metrics.csv에서 판정은 빈칸을 허용하는 boolean, 개수는 Int64로 저장되고, 새 열은 상태 지표 열로 섞이지 않고 정한 자리에 온다."""
    out = {"cracks": [{"bbox": (40, 0, 10, 10)}], "potholes": [], "y0": 40}     # 640px 전체 좌표 (40, 40, 10, 10), 중심 (45, 45)
    v = rx.verdict_columns(out, [(0, 0, 100, 100, "crack")])
    check(v == {"has_gt": True, "has_damage": True, "n_pred": 1, "hit_same": True, "hit_any": True, "correct": True},
          f"verdict_columns: 정답 있는 사진의 판정이 틀림 ({v})")
    none = rx.verdict_columns(out, None)
    check(none["has_gt"] is False and none["n_pred"] == 1 and all(math.isnan(none[k]) for k in rx.VERDICT_BOOL),
          f"verdict_columns: 정답 파일이 없는데 판정 칸이 빈칸이 아님 ({none})")
    rows = [{**metric_row("a.jpg", "own", "dark", "off", 10, 5, 100, (1, 0, 1, 0)), "edges_road": 8, "t_road_ms": 0.5, **v},
            {**metric_row("b.jpg", "own", "dark", "off", 10, 5, 100), "edges_road": 9, "t_road_ms": 0.5, **none}]
    df = rx.metrics_frame(rows)
    types = {c: str(df[c].dtype) for c in ("has_gt", "correct", "n_pred", "edges_road")}
    check(types == {"has_gt": "boolean", "correct": "boolean", "n_pred": "Int64", "edges_road": "Int64"}, f"metrics_frame: 새 열의 형이 틀림 ({types})")
    check(list(df["has_gt"]) == [True, False] and df["correct"].isna().tolist() == [False, True],
          "metrics_frame: 정답 없는 사진의 판정이 빈칸이 아님")
    lines = df.to_csv(index=False).splitlines()
    head, b_row = lines[0].split(","), lines[2].split(",")
    check(b_row[head.index("correct")] == "" and b_row[head.index("n_pred")] == "1",
          f"metrics.csv: 정답 없는 사진의 판정 칸이 빈칸으로 저장되지 않음 (correct={b_row[head.index('correct')]!r})")
    check(not set(rx.quality_columns(rows)) & set(rx.VERDICT_COLUMNS + rx.EXTRA_COLUMNS), "quality_columns: 새 열이 상태 지표 열로 섞임")
    cols = list(df.columns)
    check(cols.index("t_road_ms") == cols.index("t_detect_ms") + 2 and cols[-len(rx.VERDICT_COLUMNS):] == rx.VERDICT_COLUMNS,
          "metrics_frame: 새 열이 row 9열 뒤(edges_road·t_road_ms)·정답 비교 열 뒤(판정)에 오지 않음")


def test_group_verdict_tables():
    """source·condition 표: 정답 있는 영상만으로 민감도·특이도·영상 단위 정답률의 끔·켬·차이를 내고, 정답률 바로 옆에 기준선을 둔다.
    정답 없는 영상(c)은 영상 수에는 들고 판정에서는 빠진다."""
    def row(file, source, pre, damage, correct):
        """damage None = 정답 파일 없음."""
        r = metric_row(file, source, "dark", pre, 10, 5, 100, None if damage is None else (1, 0, 1, 0))
        r.update({"edges_road": 10, "t_road_ms": 0.5, "has_gt": damage is not None, "n_pred": 1})
        r.update({k: (math.nan if damage is None else val) for k, val in
                  (("has_damage", damage), ("hit_same", correct and bool(damage)), ("hit_any", correct and bool(damage)), ("correct", correct))})
        return r

    rows = [row("a.jpg", "provided", "off", True, True), row("a.jpg", "provided", "on", True, False),
            row("b.jpg", "own", "off", False, True), row("b.jpg", "own", "on", False, True),
            row("c.jpg", "own", "off", None, None), row("c.jpg", "own", "on", None, None)]
    df = rx.metrics_frame(rows)
    src = rx.table_by_group(df, "source", fake_pr)
    cols = list(src.columns)
    check(cols[:5] == ["source", "n_images", "n_gt", "n_damage", "n_none"] and cols.index("baseline_none") == cols.index("accuracy_diff") + 1,
          f"table_by_source: 영상 단위 열이 앞에 없거나 정답률 바로 옆에 기준선이 없음 ({cols[:16]})")
    t = src.set_index("source")
    p, o, a = t.loc["provided"], t.loc["own"], t.loc["all"]
    check((p["sensitivity_off"], p["sensitivity_on"], p["sensitivity_diff"], p["accuracy_on"], p["baseline_none"]) == (1.0, 0.0, -1.0, 0.0, 0.0)
          and math.isnan(p["specificity_off"]), "table_by_source: 제공 행(손상 1장)의 민감도·정답률·기준선이 틀림")
    check((o["n_images"], o["n_gt"], o["n_none"], o["specificity_on"], o["accuracy_off"], o["baseline_none"]) == (2, 1, 1, 1.0, 1.0, 1.0)
          and math.isnan(o["sensitivity_on"]), "table_by_source: 직접 촬영 행(무손상 1장 + 정답 없음 1장)이 틀림")
    check((a["n_gt"], a["n_damage"], a["n_none"], a["accuracy_off"], a["accuracy_on"], a["accuracy_diff"], a["baseline_none"])
          == (2, 1, 1, 1.0, 0.5, -0.5, 0.5), "table_by_source: all 행의 정답률·기준선이 틀림")
    cond = rx.table_by_group(df, "condition", fake_pr)
    check(list(cond["condition"]) == ["dark", "all"] and cond.iloc[1]["accuracy_on"] == 0.5 and cond.iloc[1]["baseline_none"] == 0.5,
          "table_by_condition: 영상 단위 지표가 틀림")


def test_image_table_verdicts():
    """table_by_image: 영상마다 정답 유무·손상 유무와 판정(correct·hit_same·hit_any)의 끔·켬을 머리 열 바로 뒤에 둔다.
    정답 파일이 없는 영상은 판정 칸이 빈칸(CSV에서도 빈칸)이고, 있던 열(지표 끔·켬·차이)은 그대로다."""
    def row(file, pre, verdict):
        r = metric_row(file, "own", "blur", pre, 10, 5, 100, None if verdict is None else (1, 0, 1, 0))
        r.update({"edges_road": 10, "t_road_ms": 0.5, "n_pred": 2, "has_gt": verdict is not None})
        r.update({k: (math.nan if verdict is None else verdict[i]) for i, k in enumerate(("has_damage", "correct", "hit_same", "hit_any"))})
        return r

    rows = [row("a.jpg", "off", (True, True, True, True)), row("a.jpg", "on", (True, False, False, True)),
            row("b.jpg", "off", None), row("b.jpg", "on", None)]
    img = rx.table_by_image(rx.metrics_frame(rows), fake_pr)
    cols = list(img.columns)
    pairs = ["has_gt", "has_damage", "correct_off", "correct_on", "hit_same_off", "hit_same_on", "hit_any_off", "hit_any_on"]
    start = cols.index("roi_pixels") + 1
    check(cols[start:start + len(pairs)] == pairs and "edges_diff" in cols and "crack_recall_on" in cols,
          f"table_by_image: 판정 끔·켬 열이 머리 열 바로 뒤에 없거나 있던 열이 빠짐 ({cols[start:start + len(pairs)]})")
    a = img.iloc[0]
    check((bool(a["has_gt"]), bool(a["correct_off"]), bool(a["correct_on"]), bool(a["hit_same_on"]), bool(a["hit_any_on"]))
          == (True, True, False, False, True), "table_by_image: 정답 있는 영상의 끔·켬 판정이 틀림")
    b = img.iloc[1]
    check(not bool(b["has_gt"]) and all(pd.isna(b[c]) for c in pairs[1:]), "table_by_image: 정답 없는 영상의 판정이 빈칸이 아님")
    line = img.to_csv(index=False).splitlines()[2].split(",")
    check(line[cols.index("correct_on")] == "", "table_by_image: 정답 없는 영상의 판정이 CSV에서 빈칸이 아님")


def broken_detector(gray, road=None):
    """일부러 오류를 내는 가짜 검출기: 모든 사진이 같은 이유로 실패할 때 run_experiment·demo가 알기 쉽게 알리고 1로 끝나는지 보려고 쓴다."""
    raise NotImplementedError("시험용 가짜 검출기")


def test_run_main_options():
    """run_experiment.main을 임시 폴더의 합성 사진 2장(roi.csv에 있는 직접 촬영 1장, 없는 제공 1장)으로 끝까지 돌린다(TIME_REPEAT 1).
    옵션 없음과 기본값을 직접 준 실행은 시간 열 말고 같고, 실행 옵션·바꾼 값이 config_used.txt 끝에 남으며, 끝나면 config가 되돌아온다.
    정답 없는 사진은 판정 칸이 빈칸이고, 지금 노면 마스크(전부 255)에서는 edges_road = edges. --roi auto·--roi-top은 roi.csv 띠 대신 기본 띠를 쓴다.
    --detect dark는 2장을 모두 처리해 0으로 끝나고 결과 표를 낸다. 검출기가 오류를 내면(가짜 검출기) 같은 오류를 한 줄로 묶어
    알리고 1로 끝나며, 두 경우 모두 config는 되돌아온다."""
    rng = np.random.default_rng(1)
    times = ["t_pre_ms", "t_detect_ms", "t_road_ms", "t_band_ms"]
    before = (C.USE_ROAD_MASK, C.DETECT_MODE, C.ROI_TOP_DEFAULT)
    with tempfile.TemporaryDirectory() as tmp:
        dirs = {"provided": os.path.join(tmp, "provided"), "own": os.path.join(tmp, "own")}
        for folder, name in (("provided", "Japan_900001.jpg"), ("own", "own_90_crack_normal.jpg")):
            os.makedirs(dirs[folder])
            gray = cv.GaussianBlur(rng.integers(90, 160, (480, 640), dtype=np.uint8), (0, 0), 1.5)
            cv.line(gray, (60, 380), (580, 400), 40, 3)
            cv.imwrite(os.path.join(dirs[folder], name), cv.cvtColor(gray, cv.COLOR_GRAY2BGR))
        roi_path, res = os.path.join(tmp, "roi.csv"), os.path.join(tmp, "results")
        with open(roi_path, "w", encoding="utf-8") as f:
            f.write("file,roi_top,roi_bottom,condition\nown_90_crack_normal.jpg,0.2,1.0,normal\n")
        gts = {"own_90_crack_normal.jpg": [(50, 360, 540, 60, "crack")]}

        def run(argv):
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                code = rx.main(argv, dirs, roi_path, res)
            return code, printed.getvalue()

        def read(name):
            metrics = pd.read_csv(os.path.join(res, name, "metrics.csv"), encoding="utf-8-sig").set_index(["file", "preprocess"])
            with open(os.path.join(res, name, "config_used.txt"), encoding="utf-8") as f:
                return metrics, f.read().splitlines()

        # 띠 높이 옵션(--roi·--roi-top)을 보는 시험이라 자동 띠 높이는 끈다(자동 띠의 roi.csv 규칙은 test_auto_band_csv_rule)
        with patched(C, "TIME_REPEAT", 1), patched(rx, "load_gt_for", gts.get), patched(C, "AUTO_BAND_TOP", False):
            defaults = ["--mask", "on" if C.USE_ROAD_MASK else "off", "--detect", C.DETECT_MODE]  # 지금 config 기본값을 직접 줌
            codes = [run(argv)[0] for argv in ([], ["--name", "explicit", "--roi", "csv", *defaults],
                                               ["--name", "auto", "--roi", "auto", "--mask", "on", "--roi-top", "0.3"])]
            after = (C.USE_ROAD_MASK, C.DETECT_MODE, C.ROI_TOP_DEFAULT)
            dark_code, dark_out = run(["--name", "dark", "--detect", "dark"])
            after_dark = (C.USE_ROAD_MASK, C.DETECT_MODE, C.ROI_TOP_DEFAULT)
            with patched(detect, "detect_cracks", broken_detector):
                broken_code, broken_out = run(["--name", "broken", "--detect", "dark"])
            after_broken = (C.USE_ROAD_MASK, C.DETECT_MODE, C.ROI_TOP_DEFAULT)
        (m0, c0), (m1, c1), (m2, c2) = read("latest"), read("explicit"), read("auto")
        check(codes == [0, 0, 0] and after == before and after_dark == before and after_broken == before,
              f"main: 끝 코드({codes})이 0이 아니거나 실행 뒤 config가 되돌아오지 않음 ({after}, {after_dark}, {after_broken}, 원래 {before})")
        check(m0.drop(columns=times).equals(m1.drop(columns=times)), "main: 옵션 없음과 기본값을 직접 준 실행의 결과가 다름")
        check(c0[-2:] == ["# 실행 옵션: --roi csv", "# 실행 옵션으로 바꾼 config 값: 없음"], f"config_used: 옵션 없는 실행의 조건 줄이 틀림 ({c0[-2:]})")
        check("USE_ROAD_MASK = True" in c2 and "ROI_TOP_DEFAULT = 0.3" in c2
              and c2[-1] == "# 실행 옵션으로 바꾼 config 값: USE_ROAD_MASK=True, ROI_TOP_DEFAULT=0.3",
              f"config_used: 바꾼 값이 남지 않음 ({c2[-2:]})")
        own, provided = ("own_90_crack_normal.jpg", "off"), ("Japan_900001.jpg", "off")
        check(bool(m0.loc[own, "has_gt"]) and not pd.isna(m0.loc[own, "correct"])
              and not bool(m0.loc[provided, "has_gt"]) and pd.isna(m0.loc[provided, "correct"]) and m0.loc[provided, "n_pred"] >= 0,
              "main: 정답 있는 사진의 판정이 비었거나 정답 없는 사진의 판정이 빈칸이 아님")
        check((m0["edges_road"] == m0["edges"]).all() and (m0["t_road_ms"] >= 0).all(),
              "main: 노면 마스크가 띠 전체인데 edges_road가 edges와 다르거나 t_road_ms가 없음")
        check((m0.loc[own, "roi_pixels"], m0.loc[provided, "roi_pixels"]) == (384 * 640, 240 * 640)
              and (m2.loc[own, "roi_pixels"], m2.loc[provided, "roi_pixels"]) == (336 * 640, 336 * 640),
              "main: --roi csv(roi.csv 0.2·기본 0.5)와 --roi auto --roi-top 0.3의 띠 높이가 틀림")
        md, cd = read("dark") if dark_code == 0 else (None, [])
        check(dark_code == 0 and "2/2장 처리" in dark_out and md is not None and len(md) == 4
              and md[["n_crack", "n_pothole"]].notna().all().all() and (md[["n_crack", "n_pothole"]] >= 0).all().all()
              and cd[-1:] == ["# 실행 옵션으로 바꾼 config 값: DETECT_MODE='dark'"],
              f"main --detect dark: 2장을 끔·켬 모두 처리해 0으로 끝나지 않았거나 dark로 돈 기록이 없음 "
              f"(끝 코드 {dark_code}, {dark_out.strip().splitlines()[-2:]}, {cd[-1:]})")
        check(broken_code == 1 and broken_out.count("NotImplementedError") == 1 and "2장" in broken_out,
              f"main: 모든 사진이 같은 오류로 실패할 때 한 줄로 묶어 알리지 않음 ({broken_out.strip().splitlines()[-2:]})")


def test_demo_args_roi():
    """ROI는 명령행 → roi.csv → config 기본값 순서. 비율이 1개이거나 top >= bottom이면 멈춤."""
    table = {"a.jpg": {"top": 0.3, "bottom": 0.9, "condition": ""}}
    check(demo.choose_roi("x/a.jpg", [0.4, 1.0], table) == (0.4, 1.0, "명령행"), "choose_roi: 명령행 비율을 쓰지 않음")
    check(demo.choose_roi("x/a.jpg", [], table) == (0.3, 0.9, "roi.csv"), "choose_roi: roi.csv 값을 쓰지 않음")
    check(demo.choose_roi("x/b.jpg", [], table) == (C.ROI_TOP_DEFAULT, C.ROI_BOTTOM_DEFAULT, "config 기본값"),
          "choose_roi: 없는 사진에 config 기본값을 쓰지 않음")
    check(demo.parse_args(["a.jpg", "0.4", "1"]).roi == [0.4, 1.0], "demo.parse_args: ROI 비율 2개를 읽지 못함")
    for bad in (["a.jpg", "0.4"], ["a.jpg", "0.7", "0.3"], ["a.jpg", "0.2", "1.5"]):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                demo.parse_args(bad)
            check(False, f"demo.parse_args: 잘못된 인자 {bad[1:]}를 받아들임")
        except SystemExit:
            pass


def test_demo_pick_image():
    """영상 경로를 빼면(VS Code ▶ 버튼) data 아래 사진(훈련 폴더 제외)을 번호로 고르고, 잘못된 번호는 다시 묻고,
    빈칸·입력 끝이면 None. 경로·ROI를 인자로 주는 방식은 그대로."""
    with tempfile.TemporaryDirectory() as root:
        for rel in ("a.jpg", "provided/b.jpg", "own/c.jpg", "train/img/d.jpg", "provided/note.txt"):
            path = os.path.join(root, *rel.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").close()
        listed = [os.path.relpath(p, root).replace(os.sep, "/") for p in demo.list_images(root)]
        check(listed == ["a.jpg", "own/c.jpg", "provided/b.jpg"], f"demo.list_images: 목록이 틀림 ({listed})")
        answers = iter(["x", "9", "3"])

        def raise_eof(prompt):
            raise EOFError

        with contextlib.redirect_stdout(io.StringIO()):
            picked = demo.pick_image(root, lambda prompt: next(answers))
            quit_empty = demo.pick_image(root, lambda prompt: "")
            quit_eof = demo.pick_image(root, raise_eof)
        check(picked == os.path.join(root, "provided", "b.jpg"), f"demo.pick_image: 3번이 provided/b.jpg가 아님 ({picked})")
        check(quit_empty is None and quit_eof is None, "demo.pick_image: 빈칸·입력 끝인데 None이 아님")
    check(demo.parse_args([]).path is None and demo.parse_args([]).roi == [], "demo.parse_args: 인자 없이 실행할 수 없음")
    args = demo.parse_args(["a.jpg", "0.4", "1"])
    check(args.path == "a.jpg" and args.roi == [0.4, 1.0], "demo.parse_args: 경로·ROI를 주는 방식이 깨짐")


def test_demo_view():
    """세로 사진도 화면 안에 들어가게 같은 비율로 줄이고, 작은 영상은 키우지 않으며, 받은 영상은 그대로."""
    tall = np.full((1138, 640, 3), 128, np.uint8)
    small = demo.fit_to_screen(tall, 576, 824)
    check(small.shape == (824, 463, 3), f"fit_to_screen: 640×1138을 576×824 안에 맞추지 못함 ({small.shape})")
    same = demo.fit_to_screen(tall, 2000, 2000)
    check(same.shape == tall.shape and same is not tall, "fit_to_screen: 들어가는 영상을 키우거나 복사하지 않음")
    labeled = demo.add_label(tall, "전처리 켬", demo.has_unicode_text())
    check(labeled.shape == (1138 + C.DEMO_LABEL_HEIGHT, 640, 3), f"add_label: 제목 띠 높이가 틀림 ({labeled.shape})")
    check(labeled[:C.DEMO_LABEL_HEIGHT].any() and (labeled[C.DEMO_LABEL_HEIGHT:] == 128).all(),
          "add_label: 글자가 띠에 없거나 영상을 덮어 씀")
    check((tall == 128).all(), "add_label·fit_to_screen이 받은 영상을 바꿈")
    view = demo.build_view(tall, tall, ("끔", "켬"), (1707, 960), False)
    check(view.shape[1] <= 1707 * C.DEMO_SCREEN_MARGIN and view.shape[0] <= 960 * C.DEMO_SCREEN_MARGIN,
          f"build_view: 화면 1707×960의 {C.DEMO_SCREEN_MARGIN}배를 넘음 ({view.shape})")
    check(view.shape[1] % 2 == 0, "build_view: 끔·켬 두 장의 폭이 다름")


def test_demo_report():
    """콘솔 표는 약속된 9열 순서, 끔·켬·차이(켬 − 끔) 3줄."""
    off = {k: 1 for k in rx.ROW_COLUMNS}
    on = {**off, "edges": 4, "extra": 9}
    t = demo.report_table(off, on)
    check(list(t.columns) == rx.ROW_COLUMNS and list(t.index) == ["끔", "켬", "차이"], "report_table: 열·줄이 틀림")
    check(t.loc["차이", "edges"] == 3 and t.loc["차이", "n_crack"] == 0, "report_table: 차이가 켬 − 끔이 아님")


def test_demo_options():
    """demo 실행 옵션: 기본은 roi.csv 띠에 config 그대로(바꾸는 값 없음). --roi auto는 기본 띠, --mask·--detect는 이번 실행에만 쓸
    config 값이 된다(run_experiment와 같은 방식). ROI 비율과 --roi auto를 함께 주면 어느 띠를 쓸지 헷갈리므로 거부한다."""
    args = parse_quietly(demo.parse_args, ["a.jpg"])
    check(args is not None and args.roi_mode == "csv" and args.save is None and rx.runtime_overrides(args) == {},
          f"demo.parse_args: 옵션이 없는데 roi.csv 띠가 아니거나 바꾸는 값이 있음 ({args})")
    args = parse_quietly(demo.parse_args, ["a.jpg", "--roi", "auto", "--mask", "on", "--detect", "dark", "--save", "v.png"])
    check(args is not None and args.roi_mode == "auto" and args.save == "v.png"
          and rx.runtime_overrides(args) == {"USE_ROAD_MASK": True, "DETECT_MODE": "dark"},
          f"demo.parse_args: --roi·--mask·--detect·--save를 읽지 못함 ({args})")
    ratio = parse_quietly(demo.parse_args, ["a.jpg", "0.4", "1", "--mask", "off"])
    check(ratio is not None and ratio.roi == [0.4, 1.0] and rx.runtime_overrides(ratio) == {"USE_ROAD_MASK": False},
          "demo.parse_args: ROI 비율과 --mask를 함께 읽지 못함")
    check(parse_quietly(demo.parse_args, ["a.jpg", "0.4", "1", "--roi", "auto"]) is None,
          "demo.parse_args: ROI 비율과 --roi auto를 함께 받아들임")
    table = {"a.jpg": {"top": 0.3, "bottom": 0.9, "condition": ""}}
    check(demo.choose_roi("x/a.jpg", [], table, "auto") == (C.ROI_TOP_DEFAULT, C.ROI_BOTTOM_DEFAULT, "기본 띠, --roi auto"),
          "demo.choose_roi: --roi auto인데 roi.csv 띠를 씀")


def test_demo_footer():
    """시연 화면 아래 띠 두 줄: 상태 판정 값(아직 없는 값은 '-')과, 실행 옵션을 적용한 뒤의 설정(ROI·노면 마스크·검출 방식).
    한글을 못 쓰면 영문(ASCII)으로 쓰고, 아래 띠까지 화면 안에 들어가게 줄인다."""
    quality = {"mean": 116.04, "std": 28.33, "lap_var": 412.7, "noise": 2.31, "blur_ratio": math.nan}
    lines = demo.make_footer(quality, 0.5, 1.0, "roi.csv", True)
    check(len(lines) == 2 and "평균 116" in lines[0] and "표준편차 28.3" in lines[0] and "라플라시안 분산 413" in lines[0]
          and "재흐림 비율 -" in lines[0] and f"노면 마스크 {'켬' if C.USE_ROAD_MASK else '끔'}" in lines[1]
          and f"검출 {C.DETECT_MODE}" in lines[1] and "0.50~1.00" in lines[1], f"make_footer: 두 줄의 내용이 틀림 ({lines})")
    with patched(C, "USE_ROAD_MASK", True):
        check("노면 마스크 켬" in demo.make_footer(quality, 0.5, 1.0, "roi.csv", True)[1], "make_footer: 실행 중에 바꾼 마스크 설정을 읽지 않음")
    ascii_lines = demo.make_footer(quality, 0.5, 1.0, "roi.csv", False)
    check(all(line.isascii() for line in ascii_lines) and "mask" in ascii_lines[1], f"make_footer: 영문 줄이 ASCII가 아님 ({ascii_lines})")
    tall = np.full((1138, 640, 3), 128, np.uint8)
    view = demo.build_view(tall, tall, ("OFF", "ON"), (1707, 960), False, ascii_lines)
    check(view.shape[0] <= 960 * C.DEMO_SCREEN_MARGIN and view[-C.DEMO_LABEL_HEIGHT:].any() and view[-2 * C.DEMO_LABEL_HEIGHT:-C.DEMO_LABEL_HEIGHT].any(),
          f"build_view: 아래 띠 두 줄이 없거나 화면을 넘음 ({view.shape})")


def test_demo_main_save():
    """demo.main --save: 창을 띄우지 않고 화면(끔·켬 두 칸 + 위 제목 띠 + 아래 두 줄)을 그림 파일로 저장하고 0으로 끝난다.
    회색 합성 사진이라 띠(기본 0.5~1.0)는 초록이 섞이고 띠 위는 회색 그대로다. --mask on은 실행 중에만 적용되고 끝나면 되돌아오며,
    검출기가 NotImplementedError를 내면(가짜 검출기) 알기 쉬운 문구와 함께 1로 끝나고 그림을 저장하지 않으며 config는 되돌아온다."""
    rng = np.random.default_rng(5)
    before = (C.USE_ROAD_MASK, C.DETECT_MODE)
    with tempfile.TemporaryDirectory() as tmp:
        path, saved, dark = (os.path.join(tmp, name) for name in ("Japan_900001.jpg", "view.png", "dark.png"))
        gray = cv.GaussianBlur(rng.integers(90, 160, (480, 640), dtype=np.uint8), (0, 0), 1.5)
        cv.imwrite(path, cv.cvtColor(gray, cv.COLOR_GRAY2BGR))
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = demo.main([path, "--mask", "on", "--auto-band", "off", "--save", saved])   # 띠 0.5 위는 칠하지 않는지 봄
        after = (C.USE_ROAD_MASK, C.DETECT_MODE)
        view = cv.imdecode(np.fromfile(saved, np.uint8), cv.IMREAD_COLOR) if os.path.exists(saved) else None
        with contextlib.redirect_stdout(io.StringIO()) as dark_printed, patched(detect, "detect_cracks", broken_detector):
            dark_code = demo.main([path, "--detect", "dark", "--save", dark])
        after_dark, dark_saved = (C.USE_ROAD_MASK, C.DETECT_MODE), os.path.exists(dark)
    check(code == 0 and view is not None and "노면 마스크 켬" in printed.getvalue(), f"demo.main --save: 저장하지 못함 (끝 코드 {code})")
    if view is not None:
        panel_h = view.shape[0] - 3 * C.DEMO_LABEL_HEIGHT                        # 위 제목 띠 1줄 + 아래 띠 2줄을 뺀 사진 높이
        band = view[C.DEMO_LABEL_HEIGHT + int(panel_h * 0.8), : view.shape[1] // 2].astype(int)
        above = view[C.DEMO_LABEL_HEIGHT + int(panel_h * 0.2), : view.shape[1] // 2].astype(int)
        check(band[:, 1].mean() > band[:, 2].mean() + 20 and abs(above[:, 1].mean() - above[:, 2].mean()) < 3,
              "demo 화면: 띠에 초록이 섞이지 않았거나 띠 위까지 칠함")
    check(after == before and after_dark == before, f"demo.main: 실행 뒤 config가 되돌아오지 않음 ({after}, {after_dark}, 원래 {before})")
    check(dark_code == 1 and not dark_saved and "아직 구현되지 않은" in dark_printed.getvalue(),
          "demo.main: 검출기 오류(가짜 검출기)에 알기 쉬운 문구와 1로 끝나지 않았거나 그림을 저장함")


def test_demo_dialog_args():
    """선택 창 모드의 명령행: 인자가 없으면 --roi auto, --list·경로는 csv, 직접 준 --roi는 그대로.
    --list와 경로를 함께 주거나 선택 창에 --save를 주면(고를 때마다 덮어씀) 거부하고, --list와 --save는 함께 쓸 수 있다."""
    a = parse_quietly(demo.parse_args, [])
    check(a is not None and a.path is None and not a.list and a.roi_mode == "auto",
          f"demo.parse_args: 인자 없이 실행하면 선택 창(--roi auto)이 아님 ({a})")
    b = parse_quietly(demo.parse_args, ["--list"])
    check(b is not None and b.list and b.roi_mode == "csv", f"demo.parse_args: --list가 예전 방식(roi.csv 띠)이 아님 ({b})")
    c = parse_quietly(demo.parse_args, ["--roi", "csv", "--mask", "off"])
    check(c is not None and c.roi_mode == "csv" and rx.runtime_overrides(c) == {"USE_ROAD_MASK": False},
          "demo.parse_args: 선택 창에서 직접 준 --roi·--mask를 따르지 않음")
    for bad in (["a.jpg", "--list"], ["--save", "v.png"]):
        check(parse_quietly(demo.parse_args, bad) is None, f"demo.parse_args: 잘못된 조합 {bad}를 받아들임")
    check(parse_quietly(demo.parse_args, ["--list", "--save", "v.png"]) is not None, "demo.parse_args: --list --save를 거부함")


def test_demo_read_error_text():
    """못 읽는 사진의 안내: HEIC(대문자 확장자 포함)는 jpg로 바꾸는 방법, 없는 파일은 '없음', 깨진 파일은 jpg·png를 고르라는 문구."""
    heic = demo.read_error_text("x/IMG_0001.HEIC")
    check("HEIC" in heic and "JPG로 바꿔" in heic and "높은 호환성" in heic, f"read_error_text: HEIC 안내가 없음 ({heic})")
    check("파일이 없습니다" in demo.read_error_text("no/such_photo.jpg"), "read_error_text: 없는 파일 안내가 틀림")
    with tempfile.TemporaryDirectory() as tmp:
        broken = os.path.join(tmp, "broken.jpg")
        with open(broken, "wb") as f:
            f.write(b"not an image")
        check("깨진 파일" in demo.read_error_text(broken), "read_error_text: 깨진 파일 안내가 틀림")


def jpg_with_orientation(img, orientation):
    """jpg 바이트에 EXIF Orientation 태그 하나만 든 APP1 조각을 넣는다(폰 사진의 회전 정보 흉내)."""
    ifd = struct.pack("<H", 1) + struct.pack("<HHII", 0x0112, 3, 1, orientation) + struct.pack("<I", 0)
    body = b"Exif\x00\x00" + b"II*\x00" + struct.pack("<I", 8) + ifd
    data = cv.imencode(".jpg", img)[1].tobytes()
    return data[:2] + b"\xff\xe1" + struct.pack(">H", len(body) + 2) + body + data[2:]


def test_demo_dialog_loop():
    """선택 창 반복(가짜 선택 창·결과 창): roi.csv에 없는 대문자 .JPG, 한글·띄어쓰기 폴더의 .png, 회전 정보(6)가 든 가로 화소
    4000×3000 폰 사진은 --roi auto로 결과 창을 띄우고(폰 사진은 세로로 섬), HEIC와 깨진 jpg는 안내 창을 띄운 뒤 계속한다.
    처리 중 뜻밖의 오류도 안내 후 계속하고, 취소하면 0으로 끝나며 선택 창을 닫는다. 실행 뒤 config는 그대로."""
    rng = np.random.default_rng(7)
    before = (C.USE_ROAD_MASK, C.DETECT_MODE)
    with tempfile.TemporaryDirectory() as tmp:
        gray = cv.GaussianBlur(rng.integers(90, 160, (480, 640), dtype=np.uint8), (0, 0), 1.5)
        bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        ko_dir = os.path.join(tmp, "한글 폴더 1")
        os.makedirs(ko_dir)
        upper, png, phone, heic, broken = (os.path.join(tmp, "NEW_PHOTO_9.JPG"), os.path.join(ko_dir, "사진 A.png"),
                                           os.path.join(ko_dir, "폰 사진.jpg"), os.path.join(tmp, "IMG_0001.heic"),
                                           os.path.join(tmp, "broken.jpg"))
        cv.imencode(".jpg", bgr)[1].tofile(upper)
        cv.imencode(".png", bgr)[1].tofile(png)
        with open(phone, "wb") as f:
            f.write(jpg_with_orientation(cv.resize(bgr, (4000, 3000)), 6))
        with open(heic, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64)
        with open(broken, "wb") as f:
            f.write(b"not an image")

        def run(picks, pipeline_fn=None):
            order, shown, warned, closed = iter(picks + [None]), [], [], []
            tools = (lambda: next(order), warned.append, lambda: closed.append(True))
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                if pipeline_fn is None:
                    code = demo.main([], lambda: tools, shown.append)
                else:
                    with patched(pipeline, "run_pipeline", pipeline_fn):
                        code = demo.main([], lambda: tools, shown.append)
            return code, shown, warned, closed, printed.getvalue()

        code, shown, warned, closed, out = run([upper, heic, png, broken, phone])
        check(code == 0 and len(shown) == 3 and closed == [True],
              f"demo 선택 창: 3장을 띄우고 취소에서 0으로 끝나지 않음 (끝 코드 {code}, 띄운 {len(shown)}장, 닫음 {closed})")
        check(len(warned) == 2 and "HEIC" in warned[0] and "깨진 파일" in warned[1],
              f"demo 선택 창: HEIC·깨진 파일 안내 창이 틀림 ({warned})")
        check(out.count("기본 띠, --roi auto") == 3, "demo 선택 창: 고른 사진을 --roi auto로 돌리지 않음")
        if len(shown) == 3:
            panel_w, panel_h = shown[2].shape[1] // 2, shown[2].shape[0] - 3 * C.DEMO_LABEL_HEIGHT
            check(panel_h > panel_w, f"demo 선택 창: 회전 정보가 든 폰 사진이 세로로 서지 않음 (패널 {panel_w}×{panel_h})")

        def broken_pipeline(*args, **kwargs):
            raise RuntimeError("시험용 뜻밖의 오류")

        code, shown, warned, closed, out = run([upper, png], broken_pipeline)
        check(code == 0 and not shown and len(warned) == 2 and all("건너뜁니다" in w for w in warned) and closed == [True],
              f"demo 선택 창: 처리 중 오류에서 안내 후 계속하지 않음 (끝 코드 {code}, {warned})")
    check((C.USE_ROAD_MASK, C.DETECT_MODE) == before, "demo 선택 창: 실행 뒤 config가 되돌아오지 않음")


def test_demo_no_tkinter():
    """tkinter를 쓸 수 없는 컴퓨터: make_chooser는 None이고, main은 오류 없이 안내 문구를 낸 뒤 번호 목록(--list 방식)으로 넘어간다."""
    saved = {name: sys.modules.get(name) for name in ("tkinter", "tkinter.filedialog", "tkinter.messagebox")}
    sys.modules["tkinter"] = None                           # import tkinter가 ImportError를 내게
    try:
        check(demo.make_chooser() is None, "make_chooser: tkinter가 없는데 None이 아님")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    with patched(demo, "pick_image", lambda: None), contextlib.redirect_stdout(io.StringIO()) as printed:
        code = demo.main([], lambda: None)
    check(code == 0 and "번호 목록으로" in printed.getvalue() and "사진을 고르지 않아" in printed.getvalue(),
          f"demo.main: tkinter가 없을 때 안내 후 번호 목록으로 넘어가지 않음 ({printed.getvalue().strip()})")


def fake_pair_name(path):
    """A의 parse_name이 들어오기 전 시험용: pair01_B_dark.jpg → {source, pair, role, condition}, 그 밖은 쌍 정보 없음."""
    parts = os.path.splitext(os.path.basename(path))[0].split("_")
    if len(parts) == 3 and parts[0].startswith("pair"):
        if parts[1] not in C.PAIR_ROLES:
            raise ValueError(f"기호가 틀림: {parts[1]}")
        return {"source": "pair", "pair": parts[0][4:], "role": parts[1], "condition": parts[2]}
    return {"source": "provided", "damage": "", "condition": ""}


def test_match_group():
    """A가 없는 쌍·쌍 정보 없음·규칙 위반·같은 기호 두 장은 경고하고 건너뛰고, 나머지는 pair로 묶임."""
    paths = ["d/pair01_A_ref.jpg", "d/pair01_B_dark.jpg", "d/pair01_B_blur.jpg", "d/pair02_C_view.jpg",
             "d/Japan_1.jpg", "d/pair03_Z_ref.jpg"]
    groups, warns = rm.group_pairs(paths, fake_pair_name)
    check(list(groups) == ["01"] and groups["01"] == {"A": ("d/pair01_A_ref.jpg", "ref"), "B": ("d/pair01_B_dark.jpg", "dark")},
          f"group_pairs: 묶음이 틀림 ({groups})")
    check(len(warns) == 4, f"group_pairs: 경고가 4건(중복·쌍 정보 없음·규칙 위반·A 없음)이 아님 ({warns})")
    groups, warns = rm.group_pairs(["d/a.jpg"], lambda p: {"source": "provided", "damage": "", "condition": ""})
    check(groups == {} and len(warns) == 1, "group_pairs: 지금의 임시 parse_name 결과를 건너뛰지 않음")


def test_match_row_and_figure():
    """RANSAC을 못 하면 inlier_ratio는 NaN이고 선을 그리지 않음. 무늬 쌍은 inlier만 초록 선."""
    blank = np.zeros((240, 640), np.uint8)
    m, t = rm.timed_match(blank, blank, repeat=3)
    row = rm.make_row("01", "B", "dark", True, m, t)
    check(list(row) == rm.MATCH_COLUMNS and row["preprocess"] == "on", f"make_row: 열이 약속과 다름 ({list(row)})")
    check(math.isnan(row["inlier_ratio"]) and row["inliers"] == 0 and t >= 0, "make_row: 매칭 실패의 inlier_ratio가 NaN이 아님")
    green = lambda img: int(np.count_nonzero((img[:, :, 1] == 255) & (img[:, :, 0] == 0) & (img[:, :, 2] == 0)))
    check(green(rm.draw_inliers(blank, blank, m)) == 0, "draw_inliers: 매칭 실패인데 선이 그려짐")
    a, b = texture()
    m, _ = rm.timed_match(a, b, repeat=1)
    pic = rm.draw_inliers(a, b, m)
    check(m["inliers"] > 0 and pic.shape == (240, 1280, 3) and green(pic) > 0, "draw_inliers: 무늬 쌍의 inlier 선이 없음")
    none = rm.draw_inliers(a, b, {**m, "mask": [0] * len(m["matches"])})
    check(green(none) == 0, "draw_inliers: mask가 0인 매칭도 그려짐")


def test_match_main():
    """쌍 사진이 없으면 0쌍·머리글만·종료 코드 0. 합성 쌍 1개면 끔·켬 2행과 그림 2장."""
    with tempfile.TemporaryDirectory() as tmp:
        pair_dir, res = os.path.join(tmp, "pairs"), os.path.join(tmp, "results")
        os.makedirs(pair_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            code = rm.main(["--name", "t0"], pair_dir, res)
        df = pd.read_csv(os.path.join(res, "t0", "matching.csv"), encoding="utf-8-sig")
        check(code == 0 and df.empty and list(df.columns) == rm.MATCH_COLUMNS, "run_matching: 0쌍일 때 머리글만 있는 csv와 0이 아님")
        check(os.path.exists(os.path.join(res, "t0", "config_used_matching.txt")), "run_matching: config_used_matching.txt가 없음")

        a, b = texture()
        cv.imwrite(os.path.join(pair_dir, "pair01_A_ref.jpg"), cv.cvtColor(a, cv.COLOR_GRAY2BGR))
        cv.imwrite(os.path.join(pair_dir, "pair01_B_view.jpg"), cv.cvtColor(b, cv.COLOR_GRAY2BGR))
        with contextlib.redirect_stdout(io.StringIO()):
            code = rm.main(["--name", "t1"], pair_dir, res, fake_pair_name)
        df = pd.read_csv(os.path.join(res, "t1", "matching.csv"), encoding="utf-8-sig")
        check(code == 0 and list(df["preprocess"]) == ["off", "on"] and list(df["target"]) == ["B", "B"]
              and (df["inliers"] > 0).all(), f"run_matching: 합성 쌍의 끔·켬 행이 틀림 ({df.to_dict('list')})")
        figs = sorted(os.listdir(os.path.join(res, "t1", "matches")))
        check(figs == ["pair01_B_view_off.png", "pair01_B_view_on.png"], f"run_matching: 그림 이름이 틀림 ({figs})")


def road_scene():
    """파이프라인 시험용 480×640 컬러 영상: 아래 절반(노면 띠, y0 240)의 왼쪽에 균열 선, 오른쪽에 포트홀(어두운 원).
    지금 검출기(legacy)로 균열 후보 1개(중심 x 약 160)와 포트홀 후보 1개(중심 x 약 480)가 나온다."""
    rng = np.random.default_rng(3)
    gray = cv.GaussianBlur(rng.normal(120, 12, (480, 640)).clip(0, 255).astype(np.uint8), (0, 0), 1.2)
    cv.line(gray, (40, 300), (280, 330), 60, 2)
    cv.circle(gray, (480, 360), 14, 50, -1)
    return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)


def left_half(roi):
    """노면 띠의 왼쪽 절반만 노면(255)인 시험용 가짜 road_mask."""
    mask = np.zeros(roi.shape[:2], np.uint8)
    mask[:, :roi.shape[1] // 2] = 255
    return mask


def spy(fn, seen):
    """검출 함수를 감싸 pipeline이 넘긴 road를 seen에 적고 원래 함수를 그대로 부른다."""
    def wrapped(gray, road=None):
        seen.append(road)
        return fn(gray, road)
    return wrapped


def summary(out):
    """run_pipeline 결과를 비교하기 쉬운 값으로: row(시간 열 제외)와 균열·포트홀 후보 bbox."""
    return ({k: v for k, v in out["row"].items() if not k.startswith("t_")},
            [r["bbox"] for r in out["cracks"]], [r["bbox"] for r in out["potholes"]])


def test_pipeline_road():
    """run_pipeline 결과에 road(노면 띠와 같은 크기 uint8 0/255)와 t_road_ms(0 이상 실수)가 있고, 받은 영상은 그대로다."""
    bgr = road_scene()
    before = bgr.copy()
    out = pipeline.run_pipeline(bgr, 0.5, 1.0, False)
    road = out["road"]
    check(isinstance(road, np.ndarray) and road.dtype == np.uint8 and road.shape == out["gray"].shape
          and set(np.unique(road).tolist()) <= {0, 255}, "run_pipeline: road가 노면 띠 크기의 uint8 0/255가 아님")
    check(isinstance(out["t_road_ms"], float) and out["t_road_ms"] >= 0, f"run_pipeline: t_road_ms가 0 이상 실수가 아님 ({out['t_road_ms']})")
    check(np.array_equal(bgr, before), "run_pipeline이 받은 영상을 바꿈")


def test_road_mask_switch_off():
    """USE_ROAD_MASK가 꺼져 있으면 노면 마스크가 무엇이든 결과가 같고, 검출 함수에 road를 넘기지 않는다(1차 결과 보존)."""
    bgr, seen = road_scene(), []
    with patched(C, "USE_ROAD_MASK", False), patched(C, "AUTO_BAND_TOP", False):    # 띠 0.5 고정에서 노면 마스크만 봄
        base = summary(pipeline.run_pipeline(bgr, 0.5, 1.0, False))
        with patched(io_utils, "road_mask", left_half), \
                patched(detect, "detect_cracks", spy(detect.detect_cracks, seen)), \
                patched(detect, "detect_potholes", spy(detect.detect_potholes, seen)):
            half = summary(pipeline.run_pipeline(bgr, 0.5, 1.0, False))
    check(base[1] and base[2], f"시험 영상에서 균열·포트홀 후보가 나오지 않음 ({base})")
    check(half == base, f"USE_ROAD_MASK 끔인데 노면 마스크에 따라 결과가 바뀜 ({base} → {half})")
    check(seen == [None, None], f"USE_ROAD_MASK 끔인데 검출 함수에 road를 넘김 ({[type(s).__name__ for s in seen]})")


def test_road_mask_switch_on():
    """USE_ROAD_MASK를 켜면 road를 검출 함수에 넘기고, bbox 중심이 노면 밖인 후보만 지운 뒤 row를 센다.
    지금 임시 road_mask(전부 255)면 끈 것과 결과가 같다."""
    bgr, seen = road_scene(), []
    with patched(C, "USE_ROAD_MASK", False), patched(C, "AUTO_BAND_TOP", False):    # 띠 0.5 고정에서 노면 마스크만 봄
        off = pipeline.run_pipeline(bgr, 0.5, 1.0, False)
    with patched(C, "USE_ROAD_MASK", True), patched(C, "AUTO_BAND_TOP", False):
        full = pipeline.run_pipeline(bgr, 0.5, 1.0, False)
        with patched(io_utils, "road_mask", left_half), \
                patched(detect, "detect_cracks", spy(detect.detect_cracks, seen)), \
                patched(detect, "detect_potholes", spy(detect.detect_potholes, seen)):
            half = pipeline.run_pipeline(bgr, 0.5, 1.0, False)
    check(summary(full) == summary(off), "노면 마스크가 전부 255인데 켬·끔 결과가 다름")
    w = off["gray"].shape[1]
    on_left = lambda regions: [r["bbox"] for r in regions if int(r["bbox"][0] + r["bbox"][2] / 2) < w // 2]  # noqa: E731
    expect_c, expect_p = on_left(off["cracks"]), on_left(off["potholes"])
    check(expect_c and not expect_p and off["potholes"], "시험 영상: 왼쪽 균열은 남고 오른쪽 포트홀은 지워지는 장면이 아님")
    check(([r["bbox"] for r in half["cracks"]], [r["bbox"] for r in half["potholes"]]) == (expect_c, expect_p),
          f"노면 밖 후보만 지우지 않음 (남은 균열 {[r['bbox'] for r in half['cracks']]}, 포트홀 {[r['bbox'] for r in half['potholes']]})")
    row = half["row"]
    check((row["n_crack"], row["area_crack"], row["n_pothole"], row["area_pothole"])
          == (len(half["cracks"]), sum(r["area"] for r in half["cracks"]), len(half["potholes"]), sum(r["area"] for r in half["potholes"])),
          f"row가 지운 뒤의 후보로 세어지지 않음 ({row})")
    check(len(seen) == 2 and all(isinstance(s, np.ndarray) and np.array_equal(s, half["road"]) for s in seen),
          "USE_ROAD_MASK 켬인데 검출 함수에 road를 넘기지 않음")


def band_scene(sky_ratio):
    """자동 띠 높이 시험용 480×640 컬러 사진: 위 sky_ratio(비율)는 매끈한 하늘색, 나머지는 결이 있는 회색 노면.
    sky_ratio 0이면 노면만 찍힌 사진(드론·오토바이 시점)이고, 0.5면 위 절반이 하늘인 차량 시점이다."""
    rng = np.random.default_rng(5)
    gray = cv.GaussianBlur(rng.normal(115, 12, (480, 640)).clip(0, 255).astype(np.uint8), (0, 0), 1.2)
    bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    bgr[:int(480 * sky_ratio)] = (235, 200, 150)
    return bgr


def test_auto_band_switch_off():
    """AUTO_BAND_TOP이 꺼져 있으면 사진 전체 노면 마스크를 계산하지 않고 받은 top 그대로 자른다(지금 결과 그대로).
    결과의 top은 받은 값, t_band_ms는 0.0이고, 켰을 때 경계가 그대로이면(가짜 auto_band_top) 결과가 끈 것과 같다."""
    bgr = band_scene(0.0)

    def refuse(small, top):
        raise AssertionError("AUTO_BAND_TOP 끔인데 auto_band_top을 부름")

    with patched(C, "AUTO_BAND_TOP", False):
        with patched(pipeline, "auto_band_top", refuse):
            off = pipeline.run_pipeline(bgr, 0.5, 1.0, False)
    check(off["top"] == 0.5 and off["y0"] == 240 and off["gray"].shape[0] == 240 and off["t_band_ms"] == 0.0,
          f"AUTO_BAND_TOP 끔인데 띠가 받은 top 그대로가 아님 (top {off['top']}, y0 {off['y0']}, t_band_ms {off['t_band_ms']})")
    with patched(C, "AUTO_BAND_TOP", True):
        with patched(pipeline, "auto_band_top", lambda small, top: top):
            same = pipeline.run_pipeline(bgr, 0.5, 1.0, False)
    check(summary(same) == summary(off) and same["y0"] == off["y0"] and np.array_equal(same["road"], off["road"]),
          "AUTO_BAND_TOP 켬에서 경계가 그대로인데 결과가 끈 것과 다름 (자동 띠 말고 다른 것도 바뀜)")


def test_auto_band_road_only():
    """노면만 찍힌 합성 사진은 0.5에서 0.0(맨 위)까지 띠가 넓어지고, run_pipeline이 그 경계로 자른다(y0 = 0).
    받은 사진은 바꾸지 않는다."""
    bgr = band_scene(0.0)
    small = io_utils.resize_width(bgr)
    before = small.copy()
    top = pipeline.auto_band_top(small, 0.5)
    check(top == 0.0, f"auto_band_top: 노면만 있는 사진인데 맨 위까지 넓어지지 않음 ({top})")
    check(np.array_equal(small, before), "auto_band_top이 받은 사진을 바꿈")
    with patched(C, "AUTO_BAND_TOP", True):
        out = pipeline.run_pipeline(bgr, 0.5, 1.0, False)
    check(out["top"] == 0.0 and out["y0"] == 0 and out["gray"].shape[0] == 480 and out["road"].shape == out["gray"].shape
          and out["t_band_ms"] >= 0, f"run_pipeline: 자동 띠를 켰는데 맨 위부터 자르지 않음 (top {out['top']}, y0 {out['y0']})")


def test_auto_band_sky():
    """위 절반이 하늘인 합성 사진(차량 시점)은 바로 위 구간(0.4~0.5)이 하늘이라 0.5에 머문다.
    하늘이 위 25%뿐이면 0.3~0.4는 넓히고 하늘이 반쯤 걸친 0.2~0.3에서 멈춘다(구간마다 0.1씩 보고 멈추는지)."""
    top = pipeline.auto_band_top(io_utils.resize_width(band_scene(0.5)), 0.5)
    check(top == 0.5, f"auto_band_top: 위 절반이 하늘인데 띠가 넓어짐 ({top})")
    top = pipeline.auto_band_top(io_utils.resize_width(band_scene(0.25)), 0.5)
    check(top == 0.3, f"auto_band_top: 하늘이 위 25%인데 0.3에서 멈추지 않음 ({top})")
    with patched(C, "AUTO_BAND_ROAD_MIN", 0.4):
        top = pipeline.auto_band_top(io_utils.resize_width(band_scene(0.25)), 0.5)
    check(top == 0.2, f"auto_band_top: 기준값을 낮췄는데(0.4) 노면이 반인 0.2~0.3까지 넓어지지 않음 ({top})")


def cloudy_scene():
    """흐린 하늘 합성 사진 480×640: 위 절반은 아스팔트와 같은 회색(115)이지만 매끈하고(잡음 σ 1), 아래 절반은 결이 있는 회색 노면(σ 12).
    색·밝기가 같아 노면 마스크가 하늘까지 칠하므로, 결로만 하늘을 구별할 수 있다."""
    rng = np.random.default_rng(6)
    road = cv.GaussianBlur(rng.normal(115, 12, (480, 640)).clip(0, 255).astype(np.uint8), (0, 0), 1.2)
    sky = rng.normal(115, 1, (240, 640)).clip(0, 255).astype(np.uint8)
    gray = np.vstack([sky, road[240:]])
    return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)


def test_auto_band_cloudy_sky():
    """위쪽이 아스팔트와 같은 회색이지만 매끈한 '흐린 하늘'이면, 노면 마스크가 하늘을 칠해도 결 조건(AUTO_BAND_USE_TEXTURE)이 막아 0.5에 머문다.
    결 조건을 끄면 노면 비율만 보므로 맨 위까지 넓어진다(결 조건이 하는 일). 노면만 찍힌 사진은 결 조건을 켜도 맨 위까지 넓어진다."""
    small = io_utils.resize_width(cloudy_scene())
    road = io_utils.road_mask(small)
    check(np.count_nonzero(road[:240]) >= 0.95 * road[:240].size,
          "시험 영상: 노면 마스크가 흐린 하늘을 노면으로 칠하지 않음(결 조건을 시험할 수 없음)")
    with patched(C, "AUTO_BAND_USE_TEXTURE", True):
        top = pipeline.auto_band_top(small, 0.5)
        check(top == 0.5, f"auto_band_top: 결 조건을 켰는데 매끈한 흐린 하늘까지 넓어짐 ({top})")
        top = pipeline.auto_band_top(io_utils.resize_width(band_scene(0.0)), 0.5)
        check(top == 0.0, f"auto_band_top: 결 조건을 켰더니 노면만 있는 사진이 맨 위까지 넓어지지 않음 ({top})")
    with patched(C, "AUTO_BAND_USE_TEXTURE", False):
        top = pipeline.auto_band_top(small, 0.5)
        check(top == 0.0, f"auto_band_top: 결 조건을 껐는데 노면 비율만으로 넓어지지 않음 ({top})")
    tex = pipeline.local_std(np.full((20, 20), 7, np.uint8), C.AUTO_BAND_TEX_KSIZE)
    check(tex.shape == (20, 20) and float(tex.max()) < 1e-3, f"local_std: 평평한 영상의 결이 0이 아님 ({float(tex.max())})")


def test_auto_band_csv_rule():
    """자동 띠 높이는 기본 띠를 쓰는 사진에만: run_experiment.main을 노면만 찍힌 합성 사진 2장(roi.csv에 있는 직접 촬영 1장(0.5),
    없는 제공 1장)으로 돌리면, 자동 띠를 켰을 때 roi.csv 사진은 0.5 그대로(240행)이고 기본 띠 사진만 맨 위까지(480행) 넓어진다.
    --roi auto면 두 장 모두 넓어지고, 끝나면 AUTO_BAND_TOP이 원래 값으로 돌아온다."""
    assert rx.band_overrides(True) == {} and rx.band_overrides(False) == {"AUTO_BAND_TOP": False}
    with tempfile.TemporaryDirectory() as tmp:
        dirs = {"provided": os.path.join(tmp, "provided"), "own": os.path.join(tmp, "own")}
        for folder, name in (("provided", "Japan_900001.jpg"), ("own", "own_90_crack_normal.jpg")):
            os.makedirs(dirs[folder])
            cv.imwrite(os.path.join(dirs[folder], name), band_scene(0.0))
        roi_path, res = os.path.join(tmp, "roi.csv"), os.path.join(tmp, "results")
        with open(roi_path, "w", encoding="utf-8") as f:
            f.write("file,roi_top,roi_bottom,condition\nown_90_crack_normal.jpg,0.5,1.0,normal\n")
        rows = {}
        with patched(C, "TIME_REPEAT", 1), patched(C, "AUTO_BAND_TOP", False):
            for name, argv in (("csv", ["--auto-band", "on"]), ("auto", ["--roi", "auto", "--auto-band", "on"])):
                with contextlib.redirect_stdout(io.StringIO()):
                    rx.main(["--name", name] + argv, dirs, roi_path, res)
                m = pd.read_csv(os.path.join(res, name, "metrics.csv"), encoding="utf-8-sig").set_index(["file", "preprocess"])
                rows[name] = (m.loc[("own_90_crack_normal.jpg", "off"), "roi_pixels"] // 640, m.loc[("Japan_900001.jpg", "off"), "roi_pixels"] // 640)
            after = C.AUTO_BAND_TOP
    check(rows.get("csv") == (240, 480) and rows.get("auto") == (480, 480) and after is False,
          f"자동 띠: roi.csv 띠를 넓히거나 기본 띠를 넓히지 않음, 또는 config가 되돌아오지 않음 ({rows}, AUTO_BAND_TOP {after})")


def test_auto_band_options():
    """자동 띠 높이 실행 옵션: run_experiment와 demo의 --auto-band on|off가 이번 실행에만 쓸 AUTO_BAND_TOP 값이 되고,
    run_notes에 남는다. 옵션을 빼면 바꾸는 값이 없다(config 그대로)."""
    args = parse_quietly(rx.parse_args, ["--roi", "auto", "--auto-band", "on"])
    check(args is not None and rx.runtime_overrides(args) == {"AUTO_BAND_TOP": True},
          f"run_experiment: --auto-band on을 config 값으로 못 바꿈 ({args})")
    if args is not None:
        notes = rx.run_notes(args, {"AUTO_BAND_TOP": True})
        check(notes[0] == "실행 옵션: --roi auto --auto-band on", f"run_notes: 자동 띠 옵션이 빠짐 ({notes})")
    off = parse_quietly(rx.parse_args, ["--auto-band", "off"])
    check(off is not None and rx.runtime_overrides(off) == {"AUTO_BAND_TOP": False}, "run_experiment: --auto-band off를 못 읽음")
    check(parse_quietly(rx.parse_args, ["--auto-band", "yes"]) is None, "run_experiment: 잘못된 --auto-band 값을 받아들임")
    tex = parse_quietly(rx.parse_args, ["--roi", "auto", "--auto-band", "on", "--band-texture", "off"])
    check(tex is not None and rx.runtime_overrides(tex) == {"AUTO_BAND_TOP": True, "AUTO_BAND_USE_TEXTURE": False}
          and rx.run_notes(tex, {})[0] == "실행 옵션: --roi auto --auto-band on --band-texture off",
          f"run_experiment: --band-texture off를 config 값으로 못 바꾸거나 실행 조건 줄에 없음 ({tex})")
    d = parse_quietly(demo.parse_args, ["a.jpg", "--auto-band", "on", "--band-texture", "on"])
    check(d is not None and rx.runtime_overrides(d) == {"AUTO_BAND_TOP": True, "AUTO_BAND_USE_TEXTURE": True},
          f"demo: --band-texture on을 못 읽음 ({d})")
    d = parse_quietly(demo.parse_args, ["a.jpg", "--roi", "auto", "--auto-band", "on"])
    check(d is not None and rx.runtime_overrides(d) == {"AUTO_BAND_TOP": True}, f"demo: --auto-band on을 못 읽음 ({d})")
    d = parse_quietly(demo.parse_args, ["a.jpg"])
    check(d is not None and rx.runtime_overrides(d) == {}, "demo: 옵션이 없는데 자동 띠 값을 바꿈")


def test_keep_on_road():
    """keep_on_road('center'): bbox 중심 픽셀이 노면이면 남기고, 받은 list는 바꾸지 않으며, 모르는 규칙은 ValueError."""
    road = np.zeros((100, 200), np.uint8)
    road[:, :100] = 255
    regions = [{"bbox": (10, 10, 20, 20)}, {"bbox": (150, 10, 20, 20)}, {"bbox": (90, 50, 18, 4)}, {"bbox": (195, 95, 5, 5)}]
    kept = pipeline.keep_on_road(regions, road, "center")             # 중심 (20, 20)·(160, 20)·(99, 52)·(197, 97)
    check([r["bbox"] for r in kept] == [(10, 10, 20, 20), (90, 50, 18, 4)], f"keep_on_road: 남은 후보가 틀림 ({kept})")
    check(len(regions) == 4 and kept is not regions, "keep_on_road가 받은 list를 바꿈")
    check(pipeline.keep_on_road([], road) == [], "keep_on_road: 후보가 없을 때 빈 list가 아님")
    try:
        pipeline.keep_on_road(regions, road, "overlap50")
        check(False, "keep_on_road: 모르는 규칙인데 ValueError가 나지 않음")
    except ValueError:
        pass


def test_keep_on_road_ratio():
    """keep_on_road('ratio'): 후보 영역(윤곽 안쪽까지 채운 픽셀)의 ROAD_KEEP_RATIO_MIN 이상이 노면이면 남긴다.
    노면 경계가 비스듬하면 바깥 후보의 상자 중심이 노면에 떨어져 'center'로는 남는다(Japan_000107 켬). 그런 후보를 'ratio'는 지우고,
    상자 중심만 노면 밖인 L자 후보는 남긴다. 받은 list는 바꾸지 않고, 기본 규칙은 'ratio'(10/12 채택)다."""
    road = np.zeros((100, 200), np.uint8)
    cv.fillPoly(road, [np.array([[0, 99], [0, 0], [100, 0], [199, 99]], np.int32)], 255)  # 왼쪽 위가 노면, 경계는 오른쪽 아래로 비스듬

    def region(points):
        contour = np.array(points, np.int32).reshape(-1, 1, 2)
        return {"bbox": tuple(int(v) for v in cv.boundingRect(contour)), "contour": contour}

    outside = region([[95, 0], [199, 0], [199, 99], [194, 99], [194, 4], [95, 4]])     # 노면 밖을 도는 ㄱ자, 상자 중심 (147, 50)은 노면
    inside = region([[5, 5], [60, 5], [60, 95], [50, 95], [50, 15], [5, 15]])            # 노면 안 ㄱ자
    corner = region([[0, 0], [90, 0], [90, 10], [10, 10], [10, 99], [0, 99]])            # 상자 중심 (45, 50)은 노면, 영역 전부 노면
    regions = [outside, inside, corner]
    center = pipeline.keep_on_road(regions, road, "center")
    ratio = pipeline.keep_on_road(regions, road, "ratio")
    check(outside in center, "keep_on_road: 시험 전제가 틀림(비스듬한 경계 밖 후보의 상자 중심이 노면 밖)")
    check(pipeline.road_ratio(outside, road) < 0.5 <= pipeline.road_ratio(inside, road),
          f"road_ratio: 노면 안 비율이 틀림 ({pipeline.road_ratio(outside, road):.2f}, {pipeline.road_ratio(inside, road):.2f})")
    check(ratio == [inside, corner], f"keep_on_road('ratio'): 남은 후보가 틀림 ({[r['bbox'] for r in ratio]})")
    check(len(regions) == 3, "keep_on_road('ratio')가 받은 list를 바꿈")
    backup = rx.apply_overrides({"ROAD_KEEP_RULE": "center"})
    try:
        check(pipeline.keep_on_road(regions, road) == center, "keep_on_road: config ROAD_KEEP_RULE='center'를 읽지 않음")
    finally:
        rx.restore_overrides(backup)
    check(C.ROAD_KEEP_RULE == "ratio" and pipeline.keep_on_road(regions, road) == ratio,
          "keep_on_road: 기본 규칙이 'ratio'(10/12 채택)가 아님")


def test_set_option():
    """run_experiment·demo의 --set NAME=VALUE: 그 밖의 config 값(남기기 규칙·차선 스위치)을 이번 실행에만 바꾼다(evaluate_train --set과 같은 모양).
    값은 파이썬 값(True·0.5)이나 글자(ratio)로 읽고, config에 없는 이름·모양이 틀린 값은 거부하며, 실행 조건 줄에 남는다."""
    args = parse_quietly(rx.parse_args, ["--roi", "auto", "--set", "ROAD_KEEP_RULE=ratio", "--set", "ROAD_LANE_BRIDGE=True"])
    check(args is not None and rx.runtime_overrides(args) == {"ROAD_KEEP_RULE": "ratio", "ROAD_LANE_BRIDGE": True},
          f"run_experiment: --set을 config 값으로 못 바꿈 ({args})")
    if args is not None:
        check(rx.run_notes(args, {})[0] == "실행 옵션: --roi auto --set ROAD_KEEP_RULE=ratio --set ROAD_LANE_BRIDGE=True",
              f"run_notes: --set이 실행 조건 줄에 없음 ({rx.run_notes(args, {})})")
    for bad in (["--set", "NO_SUCH_NAME=1"], ["--set", "ROAD_KEEP_RULE"], ["--set", "road_keep_rule=ratio"]):
        check(parse_quietly(rx.parse_args, bad) is None, f"run_experiment: 잘못된 --set을 받아들임 ({' '.join(bad)})")
        check(parse_quietly(demo.parse_args, ["a.jpg"] + bad) is None, f"demo: 잘못된 --set을 받아들임 ({' '.join(bad)})")
    d = parse_quietly(demo.parse_args, ["a.jpg", "--set", "ROAD_CLOSE_BEFORE_SEED=True"])
    check(d is not None and rx.runtime_overrides(d) == {"ROAD_CLOSE_BEFORE_SEED": True}, f"demo: --set을 못 읽음 ({d})")
    check(parse_quietly(rx.parse_args, []).set == [] and demo.road_extra() == [("남기기 ratio", "keep ratio")],
          f"--set이 없는데 바꾸는 값이 있거나, 기본 설정의 시연 설정 줄이 남기기 규칙만이 아님 ({demo.road_extra()})")
    backup = rx.apply_overrides({"ROAD_KEEP_RULE": "ratio", "ROAD_LANE_BRIDGE": True})
    try:
        line = demo.make_footer({"mean": 100.0}, 0.5, 1.0, "roi.csv", True)[1]
        ascii_line = demo.make_footer({"mean": 100.0}, 0.5, 1.0, "roi.csv", False)[1]
        check("남기기 ratio" in line and "차선 (가)" in line and ascii_line.isascii() and "lane bright" in ascii_line,
              f"make_footer: 켠 보완(남기기 규칙·차선)이 설정 줄에 없음 ({line} / {ascii_line})")
    finally:
        rx.restore_overrides(backup)


def test_draw_road_copy():
    """draw_road는 같은 크기의 새 영상을 돌려주고 받은 영상은 바꾸지 않는다(약속 표). 칠하는 일은 WP5에서."""
    bgr = np.full((300, 640, 3), 128, np.uint8)
    road = np.zeros((200, 640), np.uint8)
    road[:, :320] = 255
    out = visualize.draw_road(bgr, road, 100)
    check(isinstance(out, np.ndarray) and out.shape == bgr.shape and out is not bgr and (bgr == 128).all(),
          "draw_road: 같은 크기의 새 영상을 돌려주지 않거나 받은 영상을 바꿈")


def test_result_folder_safety():
    """세 실행 도구(run_experiment·run_matching·evaluate_train)는 --name이 없으면 results/latest에 쓰고,
    1차 결과 폴더(base·tuning·diagnosis = 기준 수치의 출처)는 대소문자·하위 폴더까지 거부한다.
    evaluate_train의 기본 실행은 config 값 그대로 한 세트이고, 1차 튜닝 세트는 --param-set으로만 돈다."""
    from tools import evaluate_train as et
    parsers = {"run_experiment": rx.parse_args, "run_matching": rm.parse_args, "evaluate_train": et.parse_args}
    for tool, parse in parsers.items():
        args = parse_quietly(parse, [])
        check(args is not None and args.name == "latest", f"{tool}: --name 없이 실행하면 results/latest가 아님 ({args})")
        for name in ("base", "tuning", "diagnosis", "Base", "tuning/sub"):
            check(parse_quietly(parse, ["--name", name]) is None, f"{tool}: 1차 결과 폴더 이름도 받아들임 ({name})")
        args = parse_quietly(parse, ["--name", "e0_base"])
        check(args is not None and args.name == "e0_base", f"{tool}: 보통 이름(e0_base)을 받지 못함")
    args = parse_quietly(et.parse_args, [])
    default = et.select_param_sets(args.param_set) if args else []
    check(default == [{"name": "config"}],
          f"evaluate_train: 기본 세트가 config 값 그대로(덮는 값 없음)가 아님 ({[p['name'] for p in default]})")
    check([p["name"] for p in et.select_param_sets("base")] == ["base"], "evaluate_train: --param-set base로 base 세트를 고르지 못함")
    check(et.select_param_sets("all") == et.PARAM_SETS and len(et.PARAM_SETS) == 5, "evaluate_train: --param-set all이 1차 5세트가 아님")


def test_draw_road_paint():
    """노면(road 255)은 y0만큼 내린 자리에 초록이 섞이고, 노면 밖과 띠 위쪽은 그대로다."""
    bgr = np.full((300, 640, 3), 128, np.uint8)
    road = np.zeros((200, 640), np.uint8)
    road[:, :320] = 255
    out = visualize.draw_road(bgr, road, 100)
    b, g, r = (int(v) for v in out[150, 100])                        # 노면 안 (띠 좌표 y 50 → 전체 좌표 y 150)
    check(g > 128 and b <= 128 and r <= 128, f"draw_road: 노면이 초록으로 칠해지지 않음 ({b}, {g}, {r})")
    check((out[150, 500] == 128).all() and (out[50, 100] == 128).all(), "draw_road: 노면 밖이나 띠 위쪽이 바뀜")


def test_draw_road_short_band():
    """road 높이만큼만 칠한다: 띠 아래(roi_bottom < 1, 미국 3장 0.95)와 띠 위는 그대로이고,
    노면은 config C 구역의 색(ROAD_OVERLAY_COLOR)을 비율(ROAD_OVERLAY_ALPHA)만큼 섞은 값이다."""
    bgr = np.full((100, 64, 3), 128, np.uint8)
    road = np.full((50, 64), 255, np.uint8)                         # y0 40부터 50줄 → 40~89만 노면, 90~99는 띠 아래
    out = visualize.draw_road(bgr, road, 40)
    alpha, color = C.ROAD_OVERLAY_ALPHA, np.array(C.ROAD_OVERLAY_COLOR, float)
    expect = np.round((1 - alpha) * 128 + alpha * color).astype(np.uint8)
    check((out[40:90] == expect).all(), f"draw_road: 노면 색이 config의 색·비율로 섞이지 않음 ({out[60, 0]}, 기대 {expect})")
    check((out[90:] == 128).all() and (out[:40] == 128).all() and (bgr == 128).all(),
          "draw_road: 띠 아래·위를 칠했거나 받은 영상을 바꿈")


def test_save_figure_road():
    """run_experiment 비교 그림: 6칸 그대로이고, 끔·켬의 후보 칸에는 노면 마스크가 반투명 초록으로 깔린다(노면 밖·띠 위는 그대로)."""
    small = np.full((100, 64, 3), 128, np.uint8)
    road = np.zeros((50, 64), np.uint8)
    road[:, :32] = 255
    out = {"small": small, "road": road, "y0": 50, "cracks": [], "potholes": [],
           "proc": np.full((50, 64), 128, np.uint8), "edges": np.zeros((50, 64), np.uint8),
           "steps": {"denoise": False, "tone": None, "gamma": 1.0, "sharpen": False}}
    captured = []
    with patched(visualize, "save_comparison", lambda path, title, panels: captured.append(panels)):
        rx.save_figure("unused.png", "a.jpg", out, out)
    panels = captured[0] if captured else []
    check(len(panels) == 6, f"save_figure: 비교 그림이 6칸이 아님 ({len(panels)})")
    if len(panels) == 6:
        for name, img in (panels[0], panels[3]):
            check(int(img[70, 10, 1]) > int(img[70, 10, 0]) and (img[70, 50] == 128).all() and (img[20, 10] == 128).all()
                  and "노면" in name, f"save_figure: '{name}' 칸에 노면이 초록으로 깔리지 않았거나 노면 밖을 칠함")
        check((small == 128).all(), "save_figure가 받은 영상을 바꿈")


def test_image_verdict():
    """영상 단위 판정: 손상 영상은 같은 종류 후보의 중심이 정답 박스 안이면 맞힘(hit_same), 종류가 달라도 위치가 맞으면 hit_any.
    두 종류가 모두 있으면 한 종류만 맞혀도 맞힘. 손상 없는 영상은 최종 후보가 0개여야 맞힘. 좌표는 640px 전체 영상 기준."""
    gt = [(0, 0, 100, 100, "crack"), (200, 200, 50, 50, "pothole")]
    v = evaluate.image_verdict({"crack": [(40, 40, 10, 10)], "pothole": []}, gt)
    check(v == {"has_damage": True, "n_pred": 1, "hit_same": True, "hit_any": True, "correct": True}, f"image_verdict: 균열 맞힘 ({v})")
    v = evaluate.image_verdict({"crack": [], "pothole": [(40, 40, 10, 10)]}, gt)       # 위치는 균열 정답 안, 종류만 틀림
    check(v["hit_same"] is False and v["hit_any"] is True and v["correct"] is False, f"image_verdict: 종류만 틀린 경우 ({v})")
    v = evaluate.image_verdict({"crack": [], "pothole": []}, [])
    check(v["has_damage"] is False and v["n_pred"] == 0 and v["correct"] is True, f"image_verdict: 손상 없음·후보 없음 ({v})")
    v = evaluate.image_verdict({"crack": [(1, 1, 2, 2)], "pothole": []}, [])
    check(v["correct"] is False and v["n_pred"] == 1, f"image_verdict: 손상 없음·후보 1개 ({v})")


def test_summarize_verdicts():
    """민감도(손상 영상 중 맞힘)·특이도(손상 없는 영상 중 맞힘)·정답률·균형 정확도·기준선('검출 없음' 정답률). 분모가 0이면 NaN."""
    verdicts = [{"has_damage": True, "correct": True}, {"has_damage": True, "correct": False},
                {"has_damage": False, "correct": True}, {"has_damage": False, "correct": False},
                {"has_damage": False, "correct": True}]
    s = evaluate.summarize_verdicts(verdicts)
    check((s["n"], s["n_damage"], s["n_none"]) == (5, 2, 3), f"summarize_verdicts: 장수가 틀림 ({s})")
    expect = {"sensitivity": 0.5, "specificity": 2 / 3, "accuracy": 0.6, "balanced": (0.5 + 2 / 3) / 2, "baseline_none": 0.6}
    check(all(abs(s[k] - v) < 1e-9 for k, v in expect.items()), f"summarize_verdicts: 값이 틀림 ({s})")
    s0 = evaluate.summarize_verdicts([{"has_damage": False, "correct": True}])
    check(math.isnan(s0["sensitivity"]) and s0["specificity"] == 1.0, f"summarize_verdicts: 분모 0이 NaN이 아님 ({s0})")


def test_dev_test_split():
    """국가별 이름순 순번 % 5 == 4이면 시험, 진단 35장(DIAG35)은 개발에 고정. 804장이면 시험 152(손상 97·무손상 55)·개발 652.
    무작위가 없어 누가 돌려도 같다. 훈련 사진 804장이 없으면 건너뛴다. (C가 10/10에 이 규칙으로 세어 확인한 값)"""
    from tools import evaluate_train as et
    names = sorted(p.name for p in Path(C.TRAIN_IMAGE_DIR).glob("*.jpg"))
    if len(names) != 804:
        raise unittest.SkipTest("훈련 사진 804장이 없음")
    split = et.assign_split(names)
    test = [n for n in names if split[n] == "test"]
    damaged = sum(1 for n in test if evaluate.load_gt(Path(C.TRAIN_GT_DIR) / f"{Path(n).stem}.csv"))
    check((len(test), damaged, len(names) - len(test)) == (152, 97, 652), f"나누기 결과가 틀림 (시험 {len(test)}, 손상 {damaged})")
    check(len(et.DIAG35) == 35 and all(split[f"{n}.jpg"] == "dev" for n in et.DIAG35), "진단 35장이 시험셋에 들어감")
    check(et.assign_split(names) == split, "같은 입력인데 나누기가 달라짐")


def test_train_options():
    """evaluate_train 실행 옵션: --mask·--roi-top·--detect·--set은 이번 실행에만 쓸 config 값 {이름: 값}이 되고(--set 값은 파이썬 값으로,
    안 되면 글자로 읽음), config에 없는 이름·소문자 이름·범위 밖 띠·튜닝 세트와 겹치는 값은 거부한다. 백업으로 원래 값이 돌아온다."""
    from tools import evaluate_train as et
    args = parse_quietly(et.parse_args, ["--mask", "on", "--roi-top", "0.3", "--detect", "legacy",
                                         "--set", "DARK_SIGMA_IN_ROAD=False", "--set", "DETECT_MODE=dark"])
    expect = {"USE_ROAD_MASK": True, "ROI_TOP_DEFAULT": 0.3, "DETECT_MODE": "dark", "DARK_SIGMA_IN_ROAD": False}
    check(args is not None and args.overrides == expect, f"evaluate_train: 실행 옵션을 config 값으로 못 바꿈 ({args and args.overrides})")
    plain = parse_quietly(et.parse_args, [])
    check(plain.overrides == {} and plain.split == "dev", "evaluate_train: 옵션이 없는데 바꾸는 값이 있거나 기본 분할이 dev가 아님")
    for bad in (["--set", "NO_SUCH_NAME=1"], ["--set", "use_road_mask=True"], ["--set", "USE_ROAD_MASK"], ["--roi-top", "1.2"],
                ["--param-set", "base", "--set", "CANNY_LOW=10"], ["--split", "train"]):
        check(parse_quietly(et.parse_args, bad) is None, f"evaluate_train: 잘못된 옵션을 받아들임 ({' '.join(bad)})")
    check(parse_quietly(et.parse_args, ["--param-set", "base", "--set", "USE_ROAD_MASK=True"]) is not None,
          "evaluate_train: 튜닝 세트와 겹치지 않는 --set까지 거부함")
    if args is None:
        return
    before = {name: getattr(C, name) for name in expect}
    backup = et.backup_params(args.overrides)
    try:
        et.apply_param_set(args.overrides)
        check(C.USE_ROAD_MASK is True and C.ROI_TOP_DEFAULT == 0.3 and C.DETECT_MODE == "dark", "evaluate_train: 실행 옵션이 config에 적용되지 않음")
    finally:
        et.restore_params(backup)
    check({name: getattr(C, name) for name in expect} == before, "evaluate_train: 실행 옵션으로 바꾼 config 값이 되돌아가지 않음")


def test_chance_control():
    """우연 대조군: 640px 높이가 같은 사진끼리 이름순으로 묶어 i번 후보를 i+1번의 같은 종류 정답과 중심 기준으로 비교하고
    (마지막은 첫 사진과), 사진이 1장뿐인 높이 묶음은 자기 자신과 비교하게 되므로 뺀다."""
    from tools import evaluate_train as et
    heights = {"a.jpg": 640, "b.jpg": 640, "c.jpg": 640, "d.jpg": 358}
    gts = {"a.jpg": [(0, 0, 100, 100, "crack")],
           "b.jpg": [(300, 300, 50, 50, "crack"), (0, 0, 640, 640, "pothole")],
           "c.jpg": [],
           "d.jpg": [(0, 0, 10, 10, "crack")]}
    preds = {"a.jpg": {"crack": [(40, 40, 10, 10)], "pothole": []},   # → b: 균열 상자 밖, 포트홀 상자 안이지만 종류가 달라 안 셈
             "b.jpg": {"crack": [(10, 10, 20, 20)], "pothole": []},   # → c: 정답 없음
             "c.jpg": {"crack": [(40, 40, 10, 10)], "pothole": []},   # → a(마지막 → 첫 사진): 균열 상자 안이라 우연히 맞음
             "d.jpg": {"crack": [(0, 0, 4, 4)], "pothole": []}}       # 높이 358은 1장뿐이라 뺌(자기 정답과 맞아도 세지 않음)
    found, truth, skipped = et.chance_found(list(heights), heights, preds, gts)
    check((found, truth, skipped) == ({"crack": 1, "pothole": 0}, {"crack": 2, "pothole": 1}, 1),
          f"chance_found: 찾은 정답·비교한 정답·뺀 사진 수가 틀림 ({found}, {truth}, {skipped})")


def tau_scene():
    """τ 곡선 시험용 사진 다섯 장(손으로 센 값). a·b·c·d는 640px 높이 한 묶음(우연 대조군: a→b, b→c, c→d, d→a), e는 358px 1장이라
    우연 대조군에서 빠진다. 후보는 (640px 상자, 점수). a 균열 5.0은 정답 안, 2.0은 밖. b 균열 4.0은 포트홀 정답 안이지만 종류가 달라 오검출,
    b 포트홀 3.0은 정답 안. c 균열 1.0·d 포트홀 6.0은 무손상 사진의 오검출. e 균열 2.5는 정답 안."""
    heights = {"a.jpg": 640, "b.jpg": 640, "c.jpg": 640, "d.jpg": 640, "e.jpg": 358}
    gts = {"a.jpg": [(0, 0, 100, 100, "crack")], "b.jpg": [(200, 200, 100, 100, "pothole")], "c.jpg": [], "d.jpg": [],
           "e.jpg": [(0, 0, 10, 10, "crack")]}
    scored = {"a.jpg": {"crack": [((40, 40, 10, 10), 5.0), ((500, 500, 10, 10), 2.0)], "pothole": []},
              "b.jpg": {"crack": [((220, 220, 10, 10), 4.0)], "pothole": [((240, 240, 20, 20), 3.0)]},
              "c.jpg": {"crack": [((0, 0, 10, 10), 1.0)], "pothole": []},
              "d.jpg": {"crack": [], "pothole": [((300, 300, 20, 20), 6.0)]},
              "e.jpg": {"crack": [((0, 0, 4, 4), 2.5)], "pothole": []}}
    return heights, gts, scored


def test_tau_grid_small():
    """τ 격자 지표를 손으로 센 값과 견준다(사진 다섯 장, tau_scene). 점수 ≥ τ인 후보만 남고(같으면 남음), 종류마다 자기 τ를 쓴다.
    민감도·특이도·정답률·균형 정확도는 영상 단위, 영상당 오검출·후보와 재현율은 후보 단위다(evaluate_train.summarize와 같은 정의)."""
    from tools import evaluate_train as et
    heights, gts, scored = tau_scene()
    inf = math.inf
    table = et.tau_table(sorted(heights), heights, scored, gts, [-inf, 4.5, 5.0, inf], [-inf, 3.5, 6.0, inf])
    check(len(table) == 16 and list(table[["tau_crack", "tau_pothole"]].iloc[1]) == [-inf, 3.5],
          f"tau_table: 격자 행 수·순서가 틀림 ({len(table)})")
    row = {(r.tau_crack, r.tau_pothole): r for r in table.itertuples()}
    expect = {  # (τ균열, τ포트홀): 민감도, 특이도, 정답률, 영상당 오검출, 영상당 후보, 균열 재현율, 포트홀 재현율
        (-inf, -inf): (1.0, 0.0, 3 / 5, 4 / 5, 7 / 5, 1.0, 1.0),
        (4.5, -inf): (2 / 3, 0.5, 3 / 5, 1 / 5, 3 / 5, 0.5, 1.0),
        (4.5, 3.5): (1 / 3, 0.5, 2 / 5, 1 / 5, 2 / 5, 0.5, 0.0),
        (5.0, 6.0): (1 / 3, 0.5, 2 / 5, 1 / 5, 2 / 5, 0.5, 0.0),       # 점수 = τ인 a 균열 5.0과 d 포트홀 6.0은 남는다
        (inf, inf): (0.0, 1.0, 2 / 5, 0.0, 0.0, 0.0, 0.0)}
    for key, (sens, spec, acc, fp, pred, rc, rp) in expect.items():
        r = row[key]
        got = (r.sensitivity, r.specificity, r.accuracy, r.fp_per_image, r.pred_per_image, r.crack_ctr_recall, r.pothole_ctr_recall)
        check(all(math.isclose(g, e) for g, e in zip(got, (sens, spec, acc, fp, pred, rc, rp))),
              f"tau_table {key}: (민감도, 특이도, 정답률, 오검출, 후보, 균열 재현율, 포트홀 재현율) {got} ≠ {(sens, spec, acc, fp, pred, rc, rp)}")
    check(math.isclose(row[(-inf, -inf)].balanced, 0.5) and math.isclose(row[(-inf, -inf)].baseline_none, 2 / 5)
          and row[(-inf, -inf)].crack_ctr_recall_chance == 0.0 and math.isclose(row[(-inf, -inf)].crack_ctr_recall_net, 1.0),
          "tau_table: 균형 정확도·기준선·우연 대조군(d의 후보는 a의 균열 정답 밖, e는 1장뿐이라 빠짐)이 틀림")
    levels, q = et.tau_levels([1.0, 2.0, 3.0])
    check(len(levels) == 102 and levels[0] == -inf and levels[-1] == inf and levels[100] == 3.0 and q[0] == 0 and q[-1] == 1.01,
          f"tau_levels: 격자가 (−∞, 1~100% 분위, +∞)가 아님 ({levels[:2]}, {levels[-2:]}, {q[-1]})")
    check(list(et.tau_levels([])[0]) == [-inf, inf], "tau_levels: 점수가 없는데 (−∞, +∞)가 아님")


def test_tau_points_pick():
    """동작점 고르기: fp1·fp2는 영상당 오검출이 1·2 이하인 점 가운데 민감도 최대(같으면 특이도 → 균형 정확도 → 분위 합이 작은 쪽),
    rule1(규칙 2·3을 재는 점)은 10/13 사용자 결정으로 fp2와 같은 방법으로 고른다(처음의 '균형 정확도 최대'는 곡선이 50% 아래면
    모두 지우는 점으로 미끄러짐). none은 τ 없음. 없으면 NaN과 note."""
    from tools import evaluate_train as et
    inf = math.inf
    curve = pd.DataFrame([
        {"tau_crack": -inf, "tau_pothole": -inf, "q_crack": 0.0, "q_pothole": 0.0, "sensitivity": 0.9, "specificity": 0.1, "balanced": 0.50, "fp_per_image": 5.0},
        {"tau_crack": 1.0, "tau_pothole": 1.0, "q_crack": 0.3, "q_pothole": 0.3, "sensitivity": 0.6, "specificity": 0.3, "balanced": 0.45, "fp_per_image": 2.0},
        {"tau_crack": 1.0, "tau_pothole": 2.0, "q_crack": 0.3, "q_pothole": 0.5, "sensitivity": 0.6, "specificity": 0.4, "balanced": 0.50, "fp_per_image": 1.8},
        {"tau_crack": 2.0, "tau_pothole": 1.0, "q_crack": 0.5, "q_pothole": 0.3, "sensitivity": 0.5, "specificity": 0.6, "balanced": 0.55, "fp_per_image": 1.5},
        {"tau_crack": 3.0, "tau_pothole": 3.0, "q_crack": 0.8, "q_pothole": 0.8, "sensitivity": 0.3, "specificity": 0.6, "balanced": 0.45, "fp_per_image": 0.9},
        {"tau_crack": 4.0, "tau_pothole": 3.0, "q_crack": 0.9, "q_pothole": 0.8, "sensitivity": 0.3, "specificity": 0.6, "balanced": 0.45, "fp_per_image": 0.7}])
    points = et.tau_points(curve)
    pick = {p: (r["tau_crack"], r["tau_pothole"]) for p, r in points.items()}
    check(pick["none"] == (-inf, -inf) and pick["fp2"] == (1.0, 2.0) and pick["fp1"] == (3.0, 3.0) and pick["rule1"] == (1.0, 2.0),
          f"tau_points: 고른 점이 틀림 ({pick})")
    empty = et.tau_points(curve[curve["fp_per_image"] > 1.0])
    check(math.isnan(empty["fp1"]["sensitivity"]) and "불가" in empty["fp1"]["note"], "tau_points: 오검출 1 이하 점이 없는데 NaN·note가 아님")


def test_tau_image_rows():
    """τ 점 하나에서 영상별 판정(판정이 바뀐 사진을 세는 데 씀): tau_scene에서 τ 균열 4.5·포트홀 −∞이면
    a(균열 5.0 남음)·b(포트홀 3.0 남음) 맞힘, c(균열 1.0 지워져 후보 0개) 맞힘, d(포트홀 6.0 남음) 틀림, e(균열 2.5 지워짐) 놓침."""
    from tools import evaluate_train as et
    heights, gts, scored = tau_scene()
    rows = {r["image"]: (r["has_damage"], r["correct"]) for r in et.tau_image_rows(list(heights), scored, gts, 4.5, -math.inf)}
    check(rows == {"a.jpg": (True, True), "b.jpg": (True, True), "c.jpg": (False, True), "d.jpg": (False, False), "e.jpg": (True, False)},
          f"tau_image_rows: 영상별 판정이 틀림 ({rows})")


def test_tau_none_matches_summary():
    """τ 없음 점(모든 후보)의 지표가 evaluate_train.summarize(지금 요약 표)와 글자 하나까지 같다(metric_text 형식).
    같은 사진·후보·정답에서 두 계산이 어긋나면 τ 곡선의 다른 점도 믿을 수 없다."""
    from tools import evaluate_train as et
    heights, gts, scored = tau_scene()
    names = sorted(heights)
    preds = {n: {k: [box for box, _ in scored[n][k]] for k in et.KINDS} for n in names}
    filler = {name: 0 for name in et.MEAN_SOURCES if not name.startswith("pred_")}
    rows = [{"image": n, "height": heights[n], **filler, **et.score_one(preds[n], gts[n]), "judge_hit": None, "judge_base": None}
            for n in names]
    summary = et.summarize({"params": "config", "preprocess": "off", "degrade": "none"}, rows, preds, gts)
    none = et.tau_table(names, heights, scored, gts, [-math.inf], [-math.inf]).iloc[0]
    differ = [k for k in et.TAU_METRICS if et.metric_text(none[k]) != summary[k]]
    check(not differ and int(none["n_damage"]) == summary["n_damage"] and int(none["n_none"]) == summary["n_none"],
          f"τ 없음 점과 summarize가 다른 열: {[(k, et.metric_text(none[k]), summary[k]) for k in differ]}")


def test_degrade():
    """인위 저하(640px 컬러, ROI 자르기 전): none은 그대로, blur는 흐려지고, dark는 입력^2.2(128 → 56), bright는 입력^0.5(128 → 181),
    noise는 seed가 고정이라 두 번 해도 같다. 받은 영상은 바꾸지 않고 크기·형식(BGR uint8)은 같다. 값은 config C 구역 DEGRADE_*."""
    from tools import evaluate_train as et
    a, _ = texture()
    bgr = cv.cvtColor(a, cv.COLOR_GRAY2BGR)
    before = bgr.copy()
    out = {k: et.degrade(bgr, k) for k in ("none", "blur", "dark", "bright", "noise")}
    check(all(o.shape == bgr.shape and o.dtype == np.uint8 for o in out.values()), "degrade: 크기·형식이 바뀜")
    check(np.array_equal(out["none"], bgr) and np.array_equal(bgr, before), "degrade: none이 그대로가 아니거나 받은 영상을 바꿈")
    lap = lambda img: cv.Laplacian(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.CV_64F).var()  # noqa: E731
    check(lap(out["blur"]) < lap(bgr) / 2, "degrade: blur가 흐려지지 않음")
    flat = np.full((10, 10, 3), 128, np.uint8)
    check(abs(int(et.degrade(flat, "dark")[0, 0, 0]) - 57) <= 1 and abs(int(et.degrade(flat, "bright")[0, 0, 0]) - 181) <= 1,
          "degrade: 감마 방향이나 값이 틀림 (128 → dark 57, bright 181)")
    check(np.array_equal(et.degrade(bgr, "noise"), out["noise"]) and not np.array_equal(out["noise"], bgr),
          "degrade: noise가 seed 고정이 아니거나 잡음이 들어가지 않음")


def test_degrade_noise_per_photo():
    """noise 저하의 잡음 무늬는 사진마다 다르고(seed = DEGRADE_SEED + 사진 바이트 crc32), 같은 사진은 언제 불러도 같다.
    모든 사진에 같은 무늬가 들어가면 결과가 그 무늬 하나에 좌우될 수 있어서다. 0·255에 잘리지 않는 밝기(120·130)로 무늬만 비교한다."""
    from tools import evaluate_train as et
    a, b = np.full((64, 64, 3), 120, np.uint8), np.full((64, 64, 3), 130, np.uint8)
    noise_a = et.degrade(a, "noise").astype(int) - 120
    noise_b = et.degrade(b, "noise").astype(int) - 130
    check(np.array_equal(et.degrade(a, "noise"), et.degrade(a, "noise")), "degrade noise: 같은 사진인데 잡음이 달라짐")
    check((noise_a != noise_b).mean() > 0.5,
          f"degrade noise: 크기가 같은 두 사진에 같은 잡음 무늬가 들어감 (같은 칸 {float((noise_a == noise_b).mean()):.2f})")


def test_degrade_option_and_judge():
    """--degrade는 기본 none이고 다섯 가지만 받는다. 상태 판정 적중의 기준(blur → 샤프닝, dark → 감마 < 1, bright → 감마 > 1,
    noise → 잡티 제거)이 맞고, 끔 행(상태 판정을 하지 않음)과 저하 없음은 적중 칸이 빈칸이다. 모르는 저하는 오류."""
    from tools import evaluate_train as et
    check(parse_quietly(et.parse_args, []).degrade == "none" and parse_quietly(et.parse_args, ["--degrade", "dark"]).degrade == "dark",
          "evaluate_train: --degrade 기본값이 none이 아니거나 dark를 받지 못함")
    check(parse_quietly(et.parse_args, ["--degrade", "fog"]) is None, "evaluate_train: 모르는 저하(fog)를 받아들임")
    none = {"denoise": False, "tone": None, "gamma": 1.0, "sharpen": False}
    picks = {"blur": {**none, "sharpen": True}, "dark": {**none, "tone": "gamma", "gamma": 0.5},
             "bright": {**none, "tone": "gamma", "gamma": 2.0}, "noise": {**none, "denoise": True}}
    table = {kind: {other: et.EXPECTED_STEPS[kind][1](steps) for other, steps in picks.items()} for kind in picks}
    check(all(table[kind][other] == (kind == other) for kind in picks for other in picks)
          and not any(rule(none) for _, rule in et.EXPECTED_STEPS.values()),
          f"EXPECTED_STEPS: 저하마다 골라야 할 전처리 판정이 틀림 ({table})")
    check(et.judge_columns("dark", picks["dark"], none, True) == {"judge_hit": True, "judge_base": False}
          and et.judge_columns("dark", picks["dark"], none, False) == {"judge_hit": None, "judge_base": None}
          and et.judge_columns("none", picks["dark"], none, True) == {"judge_hit": None, "judge_base": None},
          "judge_columns: 켬 행만 적고 끔 행·저하 없음은 빈칸이어야 함")
    try:
        et.degrade(np.zeros((4, 4, 3), np.uint8), "fog")
        check(False, "degrade: 모르는 저하(fog)에 ValueError가 나지 않음")
    except ValueError:
        pass


def test_degrade_main():
    """evaluate_train.main을 임시 폴더의 합성 사진 4장으로 끝까지 돌린다(평균 밝기 약 125 → 원래는 감마를 고르지 않음).
    옵션 없음과 --degrade none은 처리 시간 열 말고 같고, --degrade dark는 켬에서 밝게 하는 감마를 고른 비율 1·원본 기준 0을 내며
    끔 행은 적중 칸이 빈칸이다. 저하 종류는 config_used_train.txt에 남고, 영상별 측정값 열(steps·에지·해리스·시간)이 있다."""
    from tools import evaluate_train as et
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        img_dir, gt_dir = os.path.join(tmp, "img"), os.path.join(tmp, "gt")
        os.makedirs(img_dir)
        os.makedirs(gt_dir)
        for i in range(4):
            gray = cv.GaussianBlur(np.clip(rng.normal(125, 20, (240, 640)), 0, 255).astype(np.uint8), (0, 0), 1.0)
            cv.imwrite(os.path.join(img_dir, f"Japan_90000{i}.jpg"), cv.cvtColor(gray, cv.COLOR_GRAY2BGR))
            with open(os.path.join(gt_dir, f"Japan_90000{i}.csv"), "w", encoding="utf-8", newline="") as f:
                f.write("x,y,w,h,kind\n" + ("60,180,520,40,crack\n" if i % 2 == 0 else ""))
        results = Path(tmp) / "results"

        def read(name, file):
            return pd.read_csv(results / name / file, encoding="utf-8-sig", keep_default_na=False)

        with patched(C, "TRAIN_IMAGE_DIR", img_dir), patched(C, "TRAIN_GT_DIR", gt_dir), patched(et, "RESULTS_ROOT", results):
            for name, argv in (("plain", []), ("none", ["--degrade", "none"]), ("dark", ["--degrade", "dark"])):
                with contextlib.redirect_stdout(io.StringIO()):
                    et.main(["--split", "all", "--name", name] + argv)
        times = ["t_pre_ms", "t_detect_ms", "t_road_ms", "t_band_ms"]
        plain, none = read("plain", et.DETAIL_NAME), read("none", et.DETAIL_NAME)
        check(plain.drop(columns=times).equals(none.drop(columns=times)), "evaluate_train: 옵션 없음과 --degrade none의 결과가 다름")
        check(all(c in plain.columns for c in ["degrade", "steps"] + et.MEASURE_COLUMNS + ["judge_hit", "judge_base"]),
              "evaluate_train: 영상별 측정값·적중 열이 없음")
        dark = read("dark", et.SUMMARY_NAME).set_index("preprocess")
        on, off = dark.loc["on"], dark.loc["off"]
        check((on["degrade"], on["judge_expect"], float(on["judge_hit_rate"]), float(on["judge_base_rate"])) == ("dark", "밝게 하는 감마(< 1)", 1.0, 0.0)
              and off["judge_hit_rate"] == "NaN" and float(on["edges_mean"]) >= 0 and float(on["t_pre_ms_mean"]) >= 0,
              f"evaluate_train --degrade dark: 적중률·원본 기준 비율·끔 행이 틀림 ({on['judge_hit_rate']}, {on['judge_base_rate']}, {off['judge_hit_rate']})")
        config = (results / "dark" / et.CONFIG_NAME).read_text(encoding="utf-8").splitlines()
        check(config[-1].startswith("# 인위 저하: dark"), f"config_used_train: 저하 종류가 남지 않음 ({config[-1]})")


def test_run_all_plan():
    """run_all의 실험 목록: 결과 폴더는 겹치지 않고(39장 최종과 매칭만 final을 함께 씀) 1차 결과 폴더가 아니며, 학습 데이터 실행은
    이름 끝이 데이터(_all·_dev·_test)이거나 test_once·final_test다. 명령은 각 도구가 그대로 받고, E0·E1·E1b는 1차 값(base 세트·legacy·
    띠 0.5)으로 고정, run_experiment에 없는 옵션(1차 legacy 값·σ)은 실행 중에만 덮는다. 최종·시험셋 최종은 config 그대로 돈다."""
    from tools import evaluate_train as et
    from tools import run_all as ra
    names = collections.Counter(e["name"] for e in ra.EXPERIMENTS)
    check(names.pop("final_ratio") == 2 and set(names.values()) == {1}, f"run_all: 결과 폴더 이름이 겹침 ({names.most_common(3)})")
    old_final = {"final", "final_auto", "e0", "e5_dev"} | {f"e6_{k}_final_dev" for k in et.DEGRADE_KINDS}
    check(not old_final & {e["name"] for e in ra.EXPERIMENTS if e["stage"] == "final"},
          "run_all: ratio 채택 뒤 최종 단계가 예전 최종 폴더(남기기 center)를 덮어씀")
    parsers = {"train": et.parse_args, "run39": rx.parse_args, "match": rm.parse_args}
    for e in ra.EXPERIMENTS:
        if e["tool"] == "train":
            check(e["name"].endswith(("_all", "_dev", "_test")) or e["name"] in ("test_once", "final_test", "test_autoband", "final_test_autoband",
                                                                                 "test_ratio", "final_test_ratio"),
                  f"run_all: 학습 데이터 폴더 이름 끝에 데이터가 없음 ({e['name']})")
        if e["tool"]:
            argv, _ = ra.command(e, ra.resolve(e, {"E0"}))
            args = parse_quietly(parsers[e["tool"]], argv)
            check(args is not None and args.name == e["name"], f"run_all: {e['id']} 명령을 도구가 받지 않음 ({' '.join(argv)})")

    def first(sid):
        return ra.command(ra.EXP[sid], ra.resolve(ra.EXP[sid], {"E0"}))

    e1 = parse_quietly(et.parse_args, first("E1")[0])
    check(e1.param_set == "base" and e1.overrides == {"USE_ROAD_MASK": True, "ROI_TOP_DEFAULT": 0.5, "DETECT_MODE": "legacy"},
          f"run_all: E1이 1차 값 + 노면 마스크가 아님 ({e1.param_set}, {e1.overrides})")
    e3s = parse_quietly(et.parse_args, first("E3s")[0])
    check(e3s.param_set is None and e3s.overrides == {"USE_ROAD_MASK": True, "ROI_TOP_DEFAULT": 0.5, "DETECT_MODE": "dark",
                                                      "DARK_SIGMA_IN_ROAD": False}, f"run_all: E3σ의 옵션이 틀림 ({e3s.overrides})")
    argv, pins = first("F39e0")
    base = {k: v for k, v in et.PARAM_SETS[0].items() if k != "name"}
    check(pins == {**base, "ROI_TOP_DEFAULT": 0.5, "AUTO_BAND_TOP": False} and "--roi-top" not in argv
          and argv[-2:] == ["--name", "e0_final_ratio"],
          f"run_all: 39장 1차 설정이 1차 값을 실행 중에만 덮지 않음 ({argv}, {sorted(pins)})")
    check(first("E3_39") == (["--roi", "auto", "--detect", "dark", "--mask", "on", "--roi-top", "0.5", "--name", "e3_dark_mask_auto"],
                             {"DARK_SIGMA_IN_ROAD": True, "AUTO_BAND_TOP": False, "ROAD_KEEP_RULE": "center"}),
          f"run_all: 39장 E3 자동 ROI의 명령·고정 값이 틀림 ({first('E3_39')})")
    check(first("TR") == (["--split", "test", "--name", "test_ratio"], {}) and first("F39") == (["--roi", "csv", "--name", "final_ratio"], {})
          and first("E5") == (["--split", "dev", "--name", "e5_ratio_dev"], {}) and first("RR")[1] == {},
          "run_all: 최종 설정 실행에 config를 바꾸는 옵션이 붙음")
    # 띠 고정 시절 시험셋(test_once·final_test)과 띠를 비율로 정한 실험은 자동 띠를 켠 뒤에도 띠 고정으로 돈다.
    # ratio 채택(10/12) 전에 돈 실험(비교·자동 띠·test_once·test_autoband)은 남기기 규칙도 처음처럼 center로 돈다
    center = {"ROAD_KEEP_RULE": "center"}
    check(first("T") == (["--split", "test", "--name", "test_once"], {"AUTO_BAND_TOP": False, **center})
          and first("R")[1] == {"AUTO_BAND_TOP": False, **center} and first("E3")[1] == {"AUTO_BAND_TOP": False, **center},
          f"run_all: 띠 고정 실험에 AUTO_BAND_TOP=False·center를 고정하지 않음 ({first('T')}, {first('R')}, {first('E3')})")
    check(first("TA") == (["--split", "test", "--name", "test_autoband"], center) and first("RA")[1] == center
          and all(first(sid)[1].get("ROAD_KEEP_RULE") == "center" for sid in ("E1", "E6_blur", "E3_39", "B1_39")),
          f"run_all: ratio 채택 전 실험에 남기기 center를 고정하지 않음 ({first('TA')})")
    m2 = parse_quietly(et.parse_args, first("M2")[0])
    m2_39 = parse_quietly(rx.parse_args, first("M2_39")[0])
    check(m2 is not None and m2.overrides == {"ROAD_KEEP_RULE": "center", "ROAD_LANE_BRIDGE": True} and first("M2")[1] == {}
          and m2_39 is not None and rx.runtime_overrides(m2_39) == {"ROAD_KEEP_RULE": "center", "ROAD_LANE_BRIDGE": True}
          and m2_39.roi == "auto" and parse_quietly(et.parse_args, first("M1")[0]).overrides == {"ROAD_KEEP_RULE": "ratio"},
          f"run_all: 노면 마스크 보완 비교(M1·M2)의 --set이 틀림 ({first('M2')}, {first('M2_39')})")
    b1 = parse_quietly(et.parse_args, first("B1")[0])
    check(b1 is not None and b1.overrides.get("AUTO_BAND_TOP") is True and b1.overrides.get("AUTO_BAND_USE_TEXTURE") is False
          and first("B1")[1] == {"ROAD_KEEP_RULE": "center"}, f"run_all: B1(자동 띠, 결 없음)의 옵션이 틀림 ({first('B1')})")
    b39 = parse_quietly(rx.parse_args, first("B1_39")[0])
    check(b39 is not None and rx.runtime_overrides(b39).get("AUTO_BAND_TOP") is True
          and rx.runtime_overrides(b39).get("AUTO_BAND_USE_TEXTURE") is False and b39.roi == "auto",
          f"run_all: B1_39(39장 자동 띠, 결 없음)의 옵션이 틀림 ({first('B1_39')})")
    args = parse_quietly(ra.parse_args, ["compare", "--accept", "E1b", "--reject", "E2,E3", "--force", "dark,mask"])
    check(args is not None and (args.accept, args.reject, args.force) == (["E1b"], ["E2", "E3"], {"dark", "mask"}),
          f"run_all: --accept·--reject·--force를 읽지 못함 ({args})")
    for bad in (["compare", "--accept", "E1", "--reject", "E1"], ["compare", "--accept", "E9"], ["compare", "--force", "score"]):
        check(parse_quietly(ra.parse_args, bad) is None, f"run_all: 잘못된 사람 결정·강제 옵션을 받아들임 ({' '.join(bad)})")


def test_run_all_resolve():
    """앞선 잠정 채택이 뒤 단계 설정을 정한다: E1b가 채택되면 마스크를 켜는 E3·E3σ·39장 E3는 띠 0.3(마스크를 끄는 E2는 0.5 그대로),
    E6 1회는 E1~E3 가운데 마지막 잠정 채택 단계의 설정에 저하 종류만 바꾸고, 아무것도 채택되지 않으면 E0 설정으로 돈다."""
    from tools import run_all as ra
    none, e1b = {"E0"}, {"E0", "E1", "E1b"}
    band = {sid: (ra.resolve(ra.EXP[sid], none)["roi_top"], ra.resolve(ra.EXP[sid], e1b)["roi_top"]) for sid in ("E2", "E3", "E3s", "E3_39")}
    check(band == {"E2": (0.5, 0.5), "E3": (0.5, 0.3), "E3s": (0.5, 0.3), "E3_39": (0.5, 0.3)}, f"resolve: E1b 채택에 따른 띠가 틀림 ({band})")
    e6 = ra.EXP["E6_blur"]
    check(ra.resolve(e6, none) == {**ra.FIRST, "degrade": "blur"}, f"resolve: 채택이 없을 때 E6 1회가 E0 설정이 아님 ({ra.resolve(e6, none)})")
    after = {"E0", "E1", "E1b", "E3"}
    check(ra.resolve(e6, after) == {**ra.resolve(ra.EXP["E3"], after), "degrade": "blur"} and ra.resolve(e6, after)["roi_top"] == 0.3,
          f"resolve: E6 1회가 마지막 잠정 채택(E3, 띠 0.3)의 설정이 아님 ({ra.resolve(e6, after)})")
    check(ra.last_adopted({"E0", "E1", "E2"}) == "E2" and ra.last_adopted({"E0"}) == "E0" and ra.last_adopted({"E0", "E3s", "E1"}) == "E3s",
          "last_adopted: 채택 순서로 마지막 단계를 고르지 않음")


def test_run_all_adoption():
    """채택 표: E1←E0, E1b←E1, E2←E0, E3←E1(E1b가 채택되면 E1b), E3σ←E3로 한 번에 하나만 바뀌게 견준다. 규칙 2(균형 정확도가 오름)·
    3(우연을 뺀 균열 재현율이 1%p 넘게 안 떨어짐)을 넘고 비교 기준이 채택된 단계일 때만 잠정 채택한다. 꼭 1%p 떨어지면 통과, 균형 정확도가
    같으면 못 넘음, 기준과 네 값이 같으면 '효과 없음'. E3는 E2와의 차이를 참고로 두고, 처리 시간(참고)을 함께 낸다.
    --reject로 E1b를 빼면 띠 0.3으로 돈 E3·E3σ는 '다시 돌려야 함'이 된다. 바꿀 config 값은 마지막 잠정 채택 설정과 지금 config의 차이다."""
    from tools import run_all as ra

    def row(sens, fp, balanced, net, ms=10.0):
        return {"sensitivity": sens, "fp_per_image": fp, "balanced": balanced, "crack_ctr_recall_net": net,
                "t_pre_ms_mean": 0.0, "t_detect_ms_mean": ms, "t_road_ms_mean": 1.0}

    def done(off, sid, adopted):
        return {"off": off, "status": "완료", "opts": ra.resolve(ra.EXP[sid], adopted)}

    a, a1b = {"E0"}, {"E0", "E1", "E1b"}
    results = {"E0": done(row(0.20, 8.0, 0.50, 0.016), "E0", a),
               "E1": done(row(0.18, 3.0, 0.55, 0.006), "E1", a),                    # E0와: 균형 ↑, 균열 꼭 −1%p → 채택
               "E1b": done(row(0.25, 4.0, 0.60, 0.010), "E1b", a),                  # E1과: 균형 ↑ → 채택
               "E2": done(row(0.30, 2.0, 0.60, -0.0041), "E2", a),                  # E0와: 균열 −2.01%p → 규칙 3
               "E3": done(row(0.30, 1.0, 0.62, 0.012, 14.0), "E3", a1b),            # 띠 0.3, E1b와: 채택
               "E3s": done(row(0.30, 1.0, 0.62, 0.012, 14.0), "E3s", a1b)}          # E3와 네 값이 같음 → 효과 없음
    table, adopted = ra.adoption(results)
    got = table.set_index("stage_id")
    verdicts = {sid: (got.loc[sid, "base"], got.loc[sid, "verdict"]) for sid in ("E1", "E1b", "E2", "E3", "E3s")}
    check(adopted == {"E0", "E1", "E1b", "E3"} and verdicts == {
        "E1": ("E0", "잠정 채택 (규칙 1은 사람 확인)"), "E1b": ("E1", "잠정 채택 (규칙 1은 사람 확인)"),
        "E2": ("E0", "되돌림: 규칙 3 못 넘음"), "E3": ("E1b", "잠정 채택 (규칙 1은 사람 확인)"),
        "E3s": ("E3", "되돌림: 규칙 2 못 넘음 (비교 기준과 같음: 효과 없음)")},
        f"adoption: 비교 기준이나 판정이 틀림 ({verdicts}, 채택 {adopted})")
    e3 = got.loc["E3"]
    check(e3["changed"] == "검출 legacy→dark" and e3["ref_base"] == "E2" and e3["ref_changed"] == "노면 마스크 끔→켬, 띠 위쪽 0.5→0.3"
          and e3["ref_rule2"] == "통과" and (e3["time_base"], e3["time"]) == (11.0, 15.0)
          and got.loc["E1", "rule1"] == "사람 확인 (민감도↓·오검출↓)",
          f"adoption: E3의 바뀐 설정·E2 참고·처리 시간이 틀림 ({e3['changed']}, {e3['ref_changed']}, {e3['time_base']}→{e3['time']})")

    dep = {**results, "E1": done(row(0.20, 8.0, 0.45, 0.016), "E1", a),        # E0보다 균형 ↓ → 되돌림
           "E1b": done(row(0.20, 8.0, 0.50, 0.016), "E1b", a),                  # E1보다는 나아도 E1이 채택되지 않음
           "E3": done(row(0.30, 1.0, 0.62, 0.012), "E3", a), "E3s": done(row(0.30, 1.0, 0.62, 0.012), "E3s", a)}
    table, adopted = ra.adoption(dep)
    got = table.set_index("stage_id")
    check(got.loc["E1b", "verdict"] == "채택 안 함: 비교 기준(E1)이 채택되지 않음" and got.loc["E3", "base"] == "E1"
          and got.loc["E3", "verdict"].startswith("채택 안 함") and adopted == {"E0"},
          f"adoption: 비교 기준이 채택되지 않은 단계를 채택함 ({dict(got['verdict'])})")

    table, adopted = ra.adoption(results, reject=["E1b"])
    got = table.set_index("stage_id")
    check(got.loc["E1b", "verdict"] == "되돌림: 사람이 뺌(--reject). 자동 판정: 잠정 채택 (규칙 1은 사람 확인)" and adopted == {"E0", "E1"}
          and got.loc["E3", "verdict"].startswith("다시 돌려야 함") and "띠 위쪽 0.3→0.5" in got.loc["E3", "verdict"]
          and got.loc["E3s", "verdict"].startswith("다시 돌려야 함"),
          f"adoption --reject E1b: 띠 0.3으로 돈 E3를 다시 돌리라고 하지 않음 ({dict(got['verdict'])})")

    # --accept: 사람이 정하면 규칙·비교 기준과 상관없이 채택하고(자동 판정은 판정 칸에 함께), 그 때문에 설정이 바뀐 실험은 다시 돌림
    table, adopted = ra.adoption(results, accept=["E2"])
    check(table.set_index("stage_id").loc["E2", "verdict"] == "채택: 사람이 정함(--accept). 자동 판정: 되돌림: 규칙 3 못 넘음"
          and "E2" in adopted, "adoption --accept E2: 규칙 3을 못 넘은 단계를 사람이 정해도 채택하지 않음")
    table, adopted = ra.adoption(dep, accept=["E1b"])                  # E1이 안 됐어도 사람이 E1b를 채택 → E3는 띠 0.3이어야 함
    got = table.set_index("stage_id")
    check(got.loc["E1b", "verdict"].startswith("채택: 사람이 정함(--accept). 자동 판정: 채택 안 함") and "E1b" in adopted
          and got.loc["E3", "verdict"].startswith("다시 돌려야 함") and "띠 위쪽 0.5→0.3" in got.loc["E3", "verdict"],
          f"adoption --accept E1b: 띠 0.5로 돈 E3를 다시 돌리라고 하지 않음 ({dict(got['verdict'])})")
    no_run = {**results, "E3s": {"off": None, "status": "아직 구현 전: x", "opts": None}}
    table, adopted = ra.adoption(no_run, accept=["E3s"])
    check(table.set_index("stage_id").loc["E3s", "verdict"] == "아직 구현 전: x (--accept는 결과가 있어야 반영)" and "E3s" not in adopted,
          "adoption --accept: 결과가 없는 단계를 채택함")
    forced = {**results, "E2": {**results["E2"], "forced": ["dark 검출(B)"]}}
    check(ra.adoption(forced)[0].set_index("stage_id").loc["E2", "verdict"] == "되돌림: 규칙 3 못 넘음 · 강제 실행(시험 미통과)",
          "adoption: --force로 돈 단계의 판정에 '강제 실행(시험 미통과)'이 없음")
    check(ra.passes(float("nan"), lambda d: d > 0) is None and ra.rule_text(None) == "계산 불가(값 없음)", "adoption: 값이 없을 때 계산 불가가 아님")

    with patched(C, "USE_ROAD_MASK", False), patched(C, "DETECT_MODE", "legacy"), patched(C, "DARK_SIGMA_IN_ROAD", True), \
            patched(C, "ROI_TOP_DEFAULT", 0.5):
        changes = ra.config_changes(ra.resolve(ra.EXP["E3"], a1b))
        same = ra.config_changes(ra.FIRST)
    check([(n, v) for n, v, _ in changes] == [("USE_ROAD_MASK", True), ("DETECT_MODE", "dark"), ("ROI_TOP_DEFAULT", 0.3)]
          and changes[0][2].startswith("C 구역") and changes[1][2].startswith("B 구역") and changes[2][2].startswith("공통")
          and {n for n, _, _ in same} <= set(ra.param_values("base")),
          f"config_changes: 바꿀 값이나 구역 주인이 틀림 ({changes}, E0 {same})")
    owners = ra.section_owners()
    check(owners["ROI_TOP_DEFAULT"].startswith("공통") and owners["DARK_SIGMA_IN_ROAD"].startswith("B")
          and owners["CANNY_LOW"] == owners["CANNY_HIGH"] == owners["DETECT_MODE"], "section_owners: config 구역을 잘못 읽음")


def test_run_all_grid():
    """채택 표 위의 네 칸: 검출기(1차·dark) × 노면 마스크(끔·켬)를 E0·E1·E2·E3 순서로, E3σ는 E3 아래 한 줄로 둔다.
    E1b가 채택되면 마스크를 켠 칸은 띠 0.3(1차 칸은 E1b, dark 칸은 E3)이다. 안 돈 칸은 상태만 적고 수는 비운다."""
    from tools import run_all as ra

    def row(sens, balanced):
        return {"sensitivity": sens, "specificity": 0.5, "balanced": balanced, "fp_per_image": 2.0, "crack_ctr_recall_net": 0.01,
                "t_pre_ms_mean": 0.0, "t_detect_ms_mean": 9.0, "t_road_ms_mean": 1.0}

    sids = ("E0", "E1", "E1b", "E2", "E3", "E3s")
    summaries = {sid: {"off": row(0.1 * (i + 1), 0.5)} for i, sid in enumerate(sids)}
    states = {sid: {"done": sid != "E3s", "status": "완료" if sid != "E3s" else "아직 구현 전: x"} for sid in sids}
    plain = ra.grid_rows(summaries, states, {"E0", "E1"})
    check(list(plain["stage_id"]) == ["E0", "E1", "E2", "E3", "E3s"] and list(plain["band"]) == [0.5] * 5
          and list(plain["detector"]) == ["1차(legacy)", "1차(legacy)", "dark", "dark", "dark"]
          and list(plain["mask"]) == ["끔", "켬", "끔", "켬", "켬 (σ 띠 전체)"],
          f"grid_rows: 네 칸과 E3σ 줄이 틀림 ({list(plain['stage_id'])}, {list(plain['band'])})")
    check(abs(plain.loc[2, "sensitivity"] - 0.4) < 1e-9 and plain.loc[2, "time"] == 10.0 and pd.isna(plain.loc[4, "fp_per_image"])
          and plain.loc[4, "status"] == "아직 구현 전: x", "grid_rows: 칸의 값·처리 시간이나 안 돈 칸의 상태가 틀림")
    e1b = ra.grid_rows(summaries, states, {"E0", "E1", "E1b"})
    check(list(e1b["stage_id"]) == ["E0", "E1b", "E2", "E3", "E3s"] and list(e1b["band"]) == [0.5, 0.3, 0.5, 0.3, 0.3],
          f"grid_rows: E1b가 채택됐는데 마스크를 켠 칸이 띠 0.3이 아님 ({list(e1b['stage_id'])}, {list(e1b['band'])})")


def test_run_all_ready():
    """구현 여부는 A·B 시험의 @todo가 지워졌는지로만 정하고, 합성 영상 검사는 어긋날 때 경고만 한다. 신호가 '구현됨'이면 합성 띠에서
    아무것도 안 지우는 마스크도 돌리고(E1 = E0이면 '효과 없음' 결과), 신호를 못 찾으면 건너뛰지 않고 돌린다.
    구현 전 기능이 필요한 실험만 그 이름을 돌려주고(매칭·1차 방법은 늘 돌고 E4는 자리), 최종 설정 실험은 지금 config로 판단한다."""
    from tools import run_all as ra
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_x.py")
        Path(path).write_text("def todo(reason):\n    return lambda fn: fn\n\n\n@todo('구현 뒤')\ndef test_a():\n    pass\n\n\n"
                              "def test_b():\n    pass\n", encoding="utf-8")
        signals = (ra.todo_removed(path, "test_a"), ra.todo_removed(path, "test_b"), ra.todo_removed(path, "test_c"),
                   ra.todo_removed(os.path.join(tmp, "none.py"), "test_a"))
        check(signals == (False, True, None, None), f"todo_removed: @todo 신호를 잘못 읽음 ({signals})")
        with patched(ra, "SIGNALS", {"mask": (path, "test_b"), "dark": (path, "test_a")}), \
                patched(io_utils, "road_mask", lambda bgr: np.full(bgr.shape[:2], 255, np.uint8)), \
                patched(detect, "detect_cracks", lambda gray, road=None: ([], np.zeros_like(gray))), \
                patched(detect, "detect_potholes", lambda gray, road=None: ([], np.zeros_like(gray))):
            ready = ra.readiness()              # 마스크: @todo는 지웠지만 합성 띠에서 0이 없음, dark: @todo가 남았지만 합성 영상에서 돎
        check(ready["mask"] is True and ready["dark"] is False and len(ready["notes"]) == 2
              and "효과 없음" in ready["notes"][0] and "@todo를 지우지 않았는지" in ready["notes"][1],
              f"readiness: @todo 신호로 정하지 않았거나 경고가 틀림 ({ready})")
        with patched(ra, "SIGNALS", {"mask": (path, "test_gone"), "dark": (path, "test_a")}):
            ready = ra.readiness()
        check(ready["mask"] is True and any("찾지 못해" in n for n in ready["notes"]),
              f"readiness: 신호를 못 찾았는데 건너뜀 ({ready})")
        with patched(ra, "SIGNALS", {"mask": (path, "test_a"), "dark": (path, "test_a")}):     # 둘 다 @todo가 남음
            forced = ra.readiness({"dark"})
        check(forced["dark"] is True and forced["mask"] is False and forced["forced"] == {"dark"}
              and any("--force" in n for n in forced["notes"]), f"readiness --force dark: 강제로 쓰지 않음 ({forced})")
        check(ra.forced_needs(ra.EXP["E2"], forced, ra.resolve(ra.EXP["E2"], {"E0"})) == ["dark 검출(B)"]
              and ra.forced_needs(ra.EXP["E0"], forced, ra.FIRST) == [] and ra.unmet(ra.EXP["E3"], forced) == ["노면 마스크(A)"],
              "forced_needs: 강제로 쓴 기능을 잘못 셈")
    none, both = {"dark": False, "mask": False, "notes": [], "forced": set()}, {"dark": True, "mask": True, "notes": [], "forced": set()}
    check(ra.unmet(ra.EXP["E3"], none) == ["dark 검출(B)", "노면 마스크(A)"] and ra.unmet(ra.EXP["E3"], both) == []
          and ra.unmet(ra.EXP["E0"], none) == [] and ra.unmet(ra.EXP["FM"], none) == []
          and ra.unmet(ra.EXP["E4"], both) == [ra.EXP["E4"]["todo"]], "run_all: 실험마다 필요한 기능을 잘못 셈")
    with patched(C, "DETECT_MODE", "dark"), patched(C, "USE_ROAD_MASK", True):
        check(ra.unmet(ra.EXP["F39"], none) == ["dark 검출(B)", "노면 마스크(A)"], "run_all: 최종 설정 실험이 config를 보지 않음")
    skips = [e["id"] for e, _ in ra.skip_list([e for e in ra.EXPERIMENTS if e["stage"] == "compare"], none)]
    check(skips == ["E1", "E1b", "E2", "E3", "E3s", "E4", "E1_39", "E3_39"], f"skip_list: 건너뛸 실험 목록이 틀림 ({skips})")


def test_run_all_refuse():
    """dry-run은 실험·명령·폴더·예상 시간만 보여 주고 결과 폴더를 만들지 않는다. 재현은 test_once 결과가 없으면, 시험은 test_once가
    이미 있으면(끝나지 않은 폴더라도) 아무것도 돌리지 않고 거부한다(끝 코드 2). 건너뛸 실험이 있으면 목록을 보여 주고 Enter를 받아야
    시작하며, 입력을 못 받으면 아무것도 돌리지 않고 멈춘다.
    (10/12 밤) 폴더 검사는 다음 시험 단계도 쓰는 장치라, 이미 잰 단계 목록(DONE_TEST_STAGES)을 비우고 그 장치만 본다."""
    from tools import run_all as ra

    def must_not_run(*args):
        raise AssertionError("거부·확인에서 멈춰야 하는데 더 감")      # 거부가 깨져도 실험·시험셋을 실제로 돌리지 않게

    def eof(prompt):
        raise EOFError

    with tempfile.TemporaryDirectory() as tmp, patched(ra, "precheck", must_not_run), patched(ra, "DONE_TEST_STAGES", {}):
        paths = {**ra.default_paths(), "results": os.path.join(tmp, "results")}
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            dry, repro = ra.main(["compare", "--dry-run"], paths), ra.main(["repro"], paths)
        text = printed.getvalue()
        check(dry == 0 and not os.path.exists(paths["results"]) and "e1_mask_dev" in text and "예상 합계" in text,
              "run_all --dry-run: 계획을 보여 주지 않았거나 결과 폴더를 만듦")
        check(repro == 2 and "시험 단계(test) 뒤에" in text, "run_all repro: test_once 없이도 돌았음")
        os.makedirs(os.path.join(paths["results"], "test_once"))
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = ra.main(["test", "--yes"], paths)
        check(code == 2 and "한 번만" in printed.getvalue() and "끝나지 않은 실행" in printed.getvalue()
              and os.listdir(paths["results"]) == ["test_once"], f"run_all test: test_once가 있는데 거부하지 않음 ({code})")
    with tempfile.TemporaryDirectory() as tmp, patched(ra, "precheck", lambda paths: [("확인", True, "", True)]), \
            patched(ra, "readiness", lambda force=(): {"mask": False, "dark": False, "notes": [], "forced": set()}), \
            patched(ra, "run_one", must_not_run):
        paths = {**ra.default_paths(), "results": os.path.join(tmp, "results")}
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = ra.main(["compare"], paths, ask=eof)
        check(code == 2 and "건너뛸 실험 8개" in printed.getvalue() and not os.path.exists(paths["results"]),
              f"run_all compare: 건너뛸 실험의 확인 없이 시작함 ({code})")


def test_run_all_retest_refuse():
    """retest(시험셋 자동 띠, 한 번만): 최종 단계(e5_dev) 결과가 없거나, 지금 config가 최종 단계를 돌린 config와 다르거나,
    test_autoband가 이미 있으면 아무것도 돌리지 않고 거부한다(끝 코드 2). 같으면 거부하지 않는다. retest_repro는 test_autoband가 없으면 거부.
    처리 시간 합은 t_band 열이 없는 예전 결과에서 0으로 읽는다.
    (10/12 밤) 폴더·config 검사는 다음 시험 단계도 쓰는 장치라, 이미 잰 단계 목록(DONE_TEST_STAGES)을 비우고 그 장치만 본다."""
    from tools import run_all as ra

    def must_not_run(*args):
        raise AssertionError("거부에서 멈춰야 하는데 더 감")

    with tempfile.TemporaryDirectory() as tmp, patched(ra, "precheck", must_not_run), patched(ra, "DONE_TEST_STAGES", {}):
        results = Path(tmp) / "results"
        paths = {**ra.default_paths(), "results": str(results)}
        check(ra.stage_refusal("retest", results) is not None and "final을 먼저" in ra.stage_refusal("retest", results),
              "run_all retest: 최종 단계 결과 없이 거부하지 않음")
        e5 = results / "e5_ratio_dev"
        e5.mkdir(parents=True)
        rx.write_config(e5 / "config_used_train.txt", ["# 끝 줄"])
        check(ra.stage_refusal("retest", results) is None, f"run_all retest: config가 같은데 거부함 ({ra.stage_refusal('retest', results)})")
        with patched(C, "AUTO_BAND_TEX_RATIO", 2.0):
            text = ra.stage_refusal("retest", results) or ""
        check("AUTO_BAND_TEX_RATIO 최종 1.5 / 지금 2.0" in text, f"run_all retest: config가 다른데 거부하지 않거나 값을 안 알림 ({text})")
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = ra.main(["retest_repro"], paths)
        check(code == 2 and "retest 뒤에" in printed.getvalue(), "run_all retest_repro: test_autoband 없이도 돌았음")
        (results / "test_autoband").mkdir()
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = ra.main(["retest", "--yes"], paths)
        check(code == 2 and "한 번만" in printed.getvalue(), f"run_all retest: test_autoband가 있는데 거부하지 않음 ({code})")
        check(ra.stage_refusal("retest_ratio", results) is None, "run_all retest_ratio: test_autoband만 있는데 거부함")
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = ra.main(["retest_ratio_repro"], paths)
        check(code == 2 and "retest_ratio 뒤에" in printed.getvalue(), "run_all retest_ratio_repro: test_ratio 없이도 돌았음")
        (results / "test_ratio").mkdir()
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = ra.main(["retest_ratio", "--yes"], paths)
        check(code == 2 and "한 번만" in printed.getvalue(), f"run_all retest_ratio: test_ratio가 있는데 거부하지 않음 ({code})")
    old = {"t_pre_ms_mean": 1.0, "t_detect_ms_mean": 2.0, "t_road_ms_mean": 3.0}
    check(ra.time_per_image(old) == 6.0 and ra.time_per_image({**old, "t_band_ms_mean": 4.0}) == 10.0,
          "run_all: 처리 시간 합에 t_band가 빠지거나 예전 결과(t_band 없음)를 0으로 읽지 않음")


def test_run_all_done_tests_blocked():
    """(10/12 밤) 10/12에 이미 잰 시험·재현 단계 여섯 개는 결과 폴더가 하나도 없어도(results/를 지운 상태), 그리고 예전 거부 조건을
    모두 비켜 가도(최종 단계 config가 지금과 같음) 아무것도 돌리지 않고 거부한다(끝 코드 2). dry-run도 거부 이유를 보여 준다.
    폴더 유무만 보던 예전 장치로는 test가 그대로 돌았다(docs/tasks/supplement.md 0-1)."""
    from tools import run_all as ra

    def must_not_run(*args):
        raise AssertionError("이미 잰 시험 단계인데 실행 전 확인까지 감")    # 거부가 깨져도 시험셋을 실제로 돌리지 않게

    stages = ("test", "repro", "retest", "retest_repro", "retest_ratio", "retest_ratio_repro")
    check(set(ra.DONE_TEST_STAGES) == set(stages), f"run_all: 이미 잰 시험 단계 목록이 다름 ({sorted(ra.DONE_TEST_STAGES)})")
    with tempfile.TemporaryDirectory() as tmp, patched(ra, "precheck", must_not_run), patched(ra, "run_one", must_not_run):
        results = Path(tmp) / "results"
        paths = {**ra.default_paths(), "results": str(results)}
        for stage in stages:
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                code = ra.main([stage, "--yes"], paths)
            check(code == 2 and "이미 쟀습니다" in printed.getvalue() and not results.exists(),
                  f"run_all {stage}: results/가 없는데 시험셋 단계를 거부하지 않음 ({code})")
        e5 = results / "e5_ratio_dev"              # 예전 retest 조건(최종 단계 결과가 있고 config가 같음)을 만족시켜도
        e5.mkdir(parents=True)
        rx.write_config(e5 / "config_used_train.txt", ["# 끝 줄"])
        for stage in ("retest", "retest_ratio"):
            text = ra.stage_refusal(stage, results) or ""
            check("이미 쟀습니다" in text, f"run_all {stage}: 최종 단계 config가 같으면 거부하지 않음 ({text})")
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = ra.main(["test", "--dry-run"], paths)
        check(code == 0 and "이미 쟀습니다" in printed.getvalue(), "run_all test --dry-run: 거부 이유를 보여 주지 않음")


def test_run_all_supp_plan():
    """보완 비교(supp) 단계: S0·S1a·S1b·S1c가 이 순서로 개발셋에 --tau-curve와 점수식(--set SCORE_FORMULA)을 주어 돌고,
    남기기 규칙은 config(ratio) 그대로다(실행 중에 center로 고정하지 않음). dry-run은 결과 폴더를 만들지 않는다."""
    from tools import evaluate_train as et
    from tools import run_all as ra
    exps = [e for e in ra.EXPERIMENTS if e["stage"] == "supp"]
    check([e["id"] for e in exps][:4] == ["S0", "S1a", "S1b", "S1c"], f"run_all supp: 실험 순서가 다름 ({[e['id'] for e in exps]})")
    formulas = {"S0": "none", "S1a": "margin", "S1b": "snr", "S1c": "shape"}
    for e in exps[:4]:
        argv, pins = ra.command(e, ra.resolve(e, {"E0"}))
        args = parse_quietly(et.parse_args, argv)
        check(args is not None and args.split == "dev" and args.tau_curve and args.overrides.get("SCORE_FORMULA") == formulas[e["id"]],
              f"run_all supp {e['id']}: 명령이 개발셋·--tau-curve·점수식이 아님 ({' '.join(argv)})")
        check("ROAD_KEEP_RULE" not in pins, f"run_all supp {e['id']}: 남기기 규칙을 center로 고정함 ({pins})")
    with tempfile.TemporaryDirectory() as tmp:
        paths = {**ra.default_paths(), "results": os.path.join(tmp, "results")}
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = ra.main(["supp", "--dry-run"], paths)
        check(code == 0 and "--tau-curve" in printed.getvalue() and "score_da_dev" in printed.getvalue()
              and not os.path.exists(paths["results"]), "run_all supp --dry-run: 계획을 보여 주지 않았거나 결과 폴더를 만듦")


def test_run_all_rule1_tau():
    """규칙 1(τ 곡선): fp1·fp2 민감도가 둘 다 안 떨어지고 하나 이상 오르면 통과, 둘 다 같으면 못 넘음, 하나라도 떨어지면 못 넘음,
    값이 없으면 None. 14번 표(supp_table)는 규칙 2·3을 각 실행 자신의 rule1 점으로 재고(E4처럼 judge가 tau면 기준의 τ 없음 점과),
    곡선 비교(curve) 행은 판정하지 않는다."""
    from tools import evaluate_train as et
    from tools import run_all as ra
    base = {"fp1": 0.30, "fp2": 0.40}
    cases = [({"fp1": 0.31, "fp2": 0.40}, True), ({"fp1": 0.30, "fp2": 0.40}, False), ({"fp1": 0.35, "fp2": 0.39}, False),
             ({"fp1": math.nan, "fp2": 0.50}, None), ({"fp1": 0.30 + 1e-9, "fp2": 0.40}, False)]
    for cur, expect in cases:
        check(ra.rule1_tau(base, cur) == expect, f"rule1_tau {cur}: {ra.rule1_tau(base, cur)} ≠ {expect}")

    def write_points(folder, rows):
        os.makedirs(folder)
        pd.DataFrame(rows).to_csv(os.path.join(folder, et.TAU_POINTS_NAME), encoding="utf-8-sig", index=False)

    def point(name, sens, spec, bal, fp, net, tau=1.0):
        return {"params": "config", "preprocess": "off", "degrade": "none", "point": name, "tau_crack": tau, "tau_pothole": tau,
                "sensitivity": sens, "specificity": spec, "balanced": bal, "fp_per_image": fp, "crack_ctr_recall_net": net}

    with tempfile.TemporaryDirectory() as tmp:
        results = os.path.join(tmp, "results")
        write_points(os.path.join(results, "sx_base_dev"), [point("none", 0.5, 0.1, 0.30, 5.0, 0.15, -math.inf),
                                                           point("fp1", 0.20, 0.5, 0.35, 1.0, 0.10), point("fp2", 0.30, 0.4, 0.35, 2.0, 0.12),
                                                           point("rule1", 0.30, 0.45, 0.375, 1.9, 0.12)])
        write_points(os.path.join(results, "sx_new_dev"), [point("none", 0.5, 0.1, 0.30, 5.0, 0.15, -math.inf),
                                                          point("fp1", 0.25, 0.5, 0.375, 1.0, 0.11), point("fp2", 0.30, 0.4, 0.35, 2.0, 0.12),
                                                          point("rule1", 0.32, 0.45, 0.385, 1.8, 0.115)])
        write_points(os.path.join(results, "sx_tau_dev"), [point("none", 0.5, 0.1, 0.30, 5.0, 0.15, -math.inf),
                                                          point("fp1", 0.2, 0.5, 0.35, 1.0, 0.0), point("fp2", 0.3, 0.4, 0.35, 2.0, 0.0),
                                                          point("rule1", 0.30, 0.45, 0.375, 1.9, 0.12)])
        exp = {"tool": "train", "stage": "supp", "judge_row": "off", "label": "시험"}
        fake = {"SX0": {**exp, "id": "SX0", "name": "sx_base_dev", "supp_base": None, "judge": "base"},
                "SX1": {**exp, "id": "SX1", "name": "sx_new_dev", "supp_base": "SX0", "judge": "rules"},
                "SX2": {**exp, "id": "SX2", "name": "sx_tau_dev", "supp_base": "SX0", "judge": "tau"},
                "SX3": {**exp, "id": "SX3", "name": "sx_new_dev", "supp_base": "SX0", "judge": "curve"}}
        states = {sid: {"done": True, "status": "완료"} for sid in fake}
        with patched(ra, "SUPP_IDS", tuple(fake)), patched(ra, "EXP", {**ra.EXP, **fake}):
            table = ra.supp_table({"results": results}, states).set_index("stage_id")
    r1, r2, r3 = table.loc["SX1"], table.loc["SX2"], table.loc["SX3"]
    check(r1["rule1"] == "통과" and r1["rule2"] == "통과" and r1["rule3"] == "통과" and math.isclose(r1["balanced_diff"], 0.01)
          and r1["auto"] == "규칙 통과", f"supp_table rules: 규칙 1~3 판정이 틀림 ({r1['rule1']}, {r1['rule2']}, {r1['rule3']}, {r1['balanced_diff']})")
    check(r2["rule1"].startswith("해당 없음") and math.isclose(r2["balanced_diff"], 0.075) and math.isclose(r2["crack_net_diff"], -0.03)
          and r2["rule3"] == "못 넘음" and r2["auto"] == "되돌림",
          f"supp_table tau: 기준의 τ 없음 점과 견주지 않음 ({r2['balanced_diff']}, {r2['crack_net_diff']}, {r2['rule3']})")
    check(pd.isna(r3["rule1"]) and pd.isna(r3["rule2"]) and math.isclose(r3["fp1_sens"], 0.25),
          "supp_table curve: 곡선 비교 행을 판정함 또는 값이 빠짐")


def test_run_all_only():
    """--only: 단계 안에서 고른 실험만 계획·실행한다. 단계에 없는 이름은 거부한다(끝 코드 2, argparse)."""
    from tools import run_all as ra
    with tempfile.TemporaryDirectory() as tmp:
        paths = {**ra.default_paths(), "results": os.path.join(tmp, "results")}
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = ra.main(["supp", "--dry-run", "--only", "E4s"], paths)
        text = printed.getvalue()
        check(code == 0 and "실험 1개" in text and "e4s_score_dev" in text and "score_ga_dev" not in text,
              "run_all --only E4s: 고른 실험만 보여 주지 않음")
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                ra.main(["supp", "--dry-run", "--only", "E1"], paths)
                bad = False
            except SystemExit as e:
                bad = e.code == 2
        check(bad, "run_all --only: 단계에 없는 실험(E1)을 거부하지 않음")


def test_run_all_report_empty():
    """결과 폴더가 하나도 없어도 표 만들기(1~8번, 채택 표, summary.md)가 끝까지 돌고, 실험마다 상태만 적으며 잠정 채택은 E0다."""
    from tools import run_all as ra
    with tempfile.TemporaryDirectory() as tmp:
        paths = {**ra.default_paths(), "results": os.path.join(tmp, "results")}
        report = ra.build_report(paths)
        out = Path(tmp) / "results" / ra.REPORT_DIR
        expect = {f"table{i}_{n}.csv" for i, n in enumerate(["stages", "by_country", "by_condition", "preprocess", "matching",
                                                            "provided_vs_own", "degrade", "39_images"], 1)}
        expect |= {"table9_auto_band.csv", "table9_auto_band_country.csv", "table9_auto_band_39.csv", "table10_mask2.csv",
                   "table10_mask2_changes.csv", "table10_mask_gt.csv", "table11_tests.csv", "table12_time.csv", "table13_judge.csv",
                   "table14_supp.csv", "table15_tau_points.csv"}
        check(expect | {"adoption.csv", "adoption_grid.csv", "summary.md", "accept.txt", "reject.txt"} == set(os.listdir(out)),
              f"build_report: 표 파일이 빠짐 ({os.listdir(out)})")
        text = (out / "summary.md").read_text(encoding="utf-8")
        check(report["last"] == "E0" and not any(s["done"] for s in report["states"].values())
              and "아직 안 돌림" in text and "아직 구현 전: 점수 τ" in text and "구현 여부(A·B 시험의 @todo 기준)" in text,
              "build_report: 결과가 없을 때 상태를 적지 않음")
        with contextlib.redirect_stdout(io.StringIO()):                # 사람의 결정은 저장되고, 이번에 준 결정이 저장된 반대 결정보다 앞선다
            ra.main(["tables", "--accept", "E1b,E2"], paths)
            ra.main(["tables", "--reject", "E1b"], paths)
        saved = [(out / f).read_text(encoding="utf-8") for f in ("accept.txt", "reject.txt")]
        check(saved == ["E2", "E1b"], f"run_all: 사람의 결정이 저장되지 않았거나 나중 결정이 앞서지 않음 ({saved})")


def test_run_all_judgement():
    """39장 상태 판정 검증: 켬 행의 상태 지표로 전처리를 다시 골라 사람 태그(blur → 샤프닝)에 맞는지 보고, 같은 장소 정상·흔들림 쌍은
    흔들림 쪽만 샤프닝이면 맞음. 다시 고른 전처리가 실행 때 적힌 글과 다르면(코드가 바뀜) 적중 칸을 비운다."""
    from tools import run_all as ra
    sharp = {**analyze.NO_STEPS, "sharpen": True}
    pick = lambda q: sharp if q["lap_var"] < 100 else dict(analyze.NO_STEPS)  # noqa: E731
    keys = list(analyze.measure_quality(np.zeros((32, 32), np.uint8)))
    rows = []
    for file, cond, lap, text in (("own_05_a_blur.jpg", "blur", 50, None), ("own_05_a_normal.jpg", "normal", 500, None),
                                  ("own_06_a_blur.jpg", "blur", 50, None), ("own_06_a_normal.jpg", "normal", 50, None),
                                  ("own_07_a_blur.jpg", "blur", 50, "가우시안"), ("own_08_a_blur.jpg", "blur", 500, None),
                                  ("Japan_1.jpg", math.nan, 50, None)):
        steps = sharp if lap < 100 else analyze.NO_STEPS
        quality = {k: 1.0 for k in keys} | {"lap_var": float(lap)}
        for mode in ("off", "on"):
            rows.append({"file": file, "condition": cond, "preprocess": mode, **quality,
                         "steps": text or analyze.steps_to_text(steps if mode == "on" else analyze.NO_STEPS)})
    with patched(analyze, "choose_steps", pick):
        judge = ra.judgements(pd.DataFrame(rows))
    check((judge["own_05_a_blur.jpg"]["hit"], judge["own_08_a_blur.jpg"]["hit"], judge["own_07_a_blur.jpg"]["hit"]) == (True, False, None)
          and judge["own_05_a_normal.jpg"]["expect"] == "" and judge["Japan_1.jpg"]["condition"] == "unknown",
          f"judgements: 태그에 맞는 전처리 적중이 틀림 ({ {f: j['hit'] for f, j in judge.items()} })")
    pairs = ra.blur_pairs(judge)                                    # 06은 정상 사진까지 샤프닝이라 틀림, 07·08은 짝이 없음
    check([(s, ok) for s, _, _, ok in pairs] == [("05", True), ("06", False)], f"blur_pairs: 정상·흔들림 쌍 판정이 틀림 ({pairs})")


def main():
    """시험 함수를 차례로 돌려 실패를 모아 보여 준다. @todo로 표시한 시험(구현 전 기능)은 건너뛰고 몇 건인지 따로 보여 준다."""
    skips = []
    for fn in (test_harris, test_count_edges, test_draw_candidates, test_match_blank, test_match_few,
               test_match_boundary, test_match_texture, test_match_self, test_no_inplace, test_save_comparison, test_get_roi, test_median_times, test_gt_counts, test_ratios_and_pr,
               test_summary_tables, test_run_options, test_get_roi_auto, test_median_road_time, test_edges_on_road,
               test_verdict_columns, test_group_verdict_tables, test_image_table_verdicts, test_run_main_options, test_demo_args_roi, test_demo_pick_image, test_demo_view, test_demo_report,
               test_demo_options, test_demo_footer, test_demo_main_save,
               test_demo_dialog_args, test_demo_read_error_text, test_demo_dialog_loop, test_demo_no_tkinter,
               test_match_group, test_match_row_and_figure, test_match_main,
               test_pipeline_road, test_road_mask_switch_off, test_road_mask_switch_on, test_keep_on_road, test_keep_on_road_ratio, test_set_option, test_draw_road_copy,
               test_auto_band_switch_off, test_auto_band_road_only, test_auto_band_sky, test_auto_band_cloudy_sky, test_auto_band_csv_rule, test_auto_band_options,
               test_result_folder_safety,
               test_draw_road_paint, test_draw_road_short_band, test_save_figure_road, test_image_verdict, test_summarize_verdicts, test_dev_test_split,
               test_train_options, test_chance_control, test_tau_grid_small, test_tau_points_pick, test_tau_image_rows, test_tau_none_matches_summary, test_degrade, test_degrade_noise_per_photo,
               test_degrade_option_and_judge, test_degrade_main,
               test_run_all_plan, test_run_all_resolve, test_run_all_adoption, test_run_all_grid, test_run_all_ready,
               test_run_all_refuse, test_run_all_retest_refuse, test_run_all_done_tests_blocked, test_run_all_supp_plan,
               test_run_all_rule1_tau, test_run_all_only, test_run_all_report_empty,
               test_run_all_judgement):
        try:
            fn()
        except unittest.SkipTest as e:
            skips.append(f"{fn.__name__}: {e}")
        except Exception as e:
            FAILS.append(f"{fn.__name__}: 실행 중 오류 {type(e).__name__}: {e}")
    if FAILS:
        print(f"C 시험 실패 {len(FAILS)}건")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("C 시험 통과" + (f" (건너뜀 {len(skips)}건: 구현 전 기능)" if skips else ""))
    for s in skips:
        print("  · 건너뜀", s)


if __name__ == "__main__":
    main()
