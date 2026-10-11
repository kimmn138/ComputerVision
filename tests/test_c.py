"""파트 C 시험: features의 에지·해리스 점 세기와 SIFT 매칭, visualize의 후보 그리기·비교 그림 저장,
run_experiment의 좌표 변환·시간 중앙값·요약 계산, run_matching의 쌍 묶기·inlier 그림·결과 저장, demo의 ROI 선택·화면 맞춤·제목 띠가 경계 입력에서도 약속대로 동작하는지 확인한다.
사용법: 맨 위 폴더에서  python tests/test_c.py"""
import contextlib
import functools
import io
import math
import os
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
from src import detect, evaluate, features, io_utils, pipeline, visualize  # noqa: E402
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


def test_run_main_options():
    """run_experiment.main을 임시 폴더의 합성 사진 2장(roi.csv에 있는 직접 촬영 1장, 없는 제공 1장)으로 끝까지 돌린다(TIME_REPEAT 1).
    옵션 없음과 기본값을 직접 준 실행은 시간 열 말고 같고, 실행 옵션·바꾼 값이 config_used.txt 끝에 남으며, 끝나면 config가 되돌아온다.
    정답 없는 사진은 판정 칸이 빈칸이고, 지금 노면 마스크(전부 255)에서는 edges_road = edges. --roi auto·--roi-top은 roi.csv 띠 대신 기본 띠를 쓴다.
    --detect dark(아직 구현 전)는 같은 오류를 한 줄로 묶어 알리고 1로 끝나며, 그래도 config는 되돌아온다."""
    rng = np.random.default_rng(1)
    times = ["t_pre_ms", "t_detect_ms", "t_road_ms"]
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

        with patched(C, "TIME_REPEAT", 1), patched(rx, "load_gt_for", gts.get):
            codes = [run(argv)[0] for argv in ([], ["--name", "explicit", "--roi", "csv", "--mask", "off", "--detect", "legacy"],
                                               ["--name", "auto", "--roi", "auto", "--mask", "on", "--roi-top", "0.3"])]
            after = (C.USE_ROAD_MASK, C.DETECT_MODE, C.ROI_TOP_DEFAULT)
            dark_code, dark_out = run(["--name", "dark", "--detect", "dark"])
            after_dark = (C.USE_ROAD_MASK, C.DETECT_MODE, C.ROI_TOP_DEFAULT)
        (m0, c0), (m1, c1), (m2, c2) = read("latest"), read("explicit"), read("auto")
        check(codes == [0, 0, 0] and after == before and after_dark == before,
              f"main: 끝 코드({codes})이 0이 아니거나 실행 뒤 config가 되돌아오지 않음 ({after}, {after_dark}, 원래 {before})")
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
        check(dark_code == 1 and dark_out.count("NotImplementedError") == 1 and "2장" in dark_out,
              f"main: dark의 '아직 구현 안 됨' 오류를 한 줄로 묶어 알리지 않음 ({dark_out.strip().splitlines()[-2:]})")


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
    with patched(C, "USE_ROAD_MASK", False):
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
    with patched(C, "USE_ROAD_MASK", False):
        off = pipeline.run_pipeline(bgr, 0.5, 1.0, False)
    with patched(C, "USE_ROAD_MASK", True):
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


@todo("C WP5: draw_road 구현 뒤 (Plan.md 3.7)")
def test_draw_road_paint():
    """노면(road 255)은 y0만큼 내린 자리에 초록이 섞이고, 노면 밖과 띠 위쪽은 그대로다."""
    bgr = np.full((300, 640, 3), 128, np.uint8)
    road = np.zeros((200, 640), np.uint8)
    road[:, :320] = 255
    out = visualize.draw_road(bgr, road, 100)
    b, g, r = (int(v) for v in out[150, 100])                        # 노면 안 (띠 좌표 y 50 → 전체 좌표 y 150)
    check(g > 128 and b <= 128 and r <= 128, f"draw_road: 노면이 초록으로 칠해지지 않음 ({b}, {g}, {r})")
    check((out[150, 500] == 128).all() and (out[50, 100] == 128).all(), "draw_road: 노면 밖이나 띠 위쪽이 바뀜")


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


@todo("C E6: tools/evaluate_train.py에 인위 저하(degrade)를 넣은 뒤 (Plan.md 3.2 (6))")
def test_degrade():
    """인위 저하(640px 컬러, ROI 자르기 전): none은 그대로, blur는 흐려지고, dark는 입력^2.2(128 → 57), bright는 입력^0.5(128 → 181),
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


def main():
    """시험 함수를 차례로 돌려 실패를 모아 보여 준다. @todo로 표시한 시험(구현 전 기능)은 건너뛰고 몇 건인지 따로 보여 준다."""
    skips = []
    for fn in (test_harris, test_count_edges, test_draw_candidates, test_match_blank, test_match_few,
               test_match_boundary, test_match_texture, test_match_self, test_no_inplace, test_save_comparison, test_get_roi, test_median_times, test_gt_counts, test_ratios_and_pr,
               test_summary_tables, test_run_options, test_get_roi_auto, test_median_road_time, test_edges_on_road,
               test_verdict_columns, test_group_verdict_tables, test_image_table_verdicts, test_run_main_options, test_demo_args_roi, test_demo_pick_image, test_demo_view, test_demo_report,
               test_match_group, test_match_row_and_figure, test_match_main,
               test_pipeline_road, test_road_mask_switch_off, test_road_mask_switch_on, test_keep_on_road, test_draw_road_copy,
               test_result_folder_safety,
               test_draw_road_paint, test_image_verdict, test_summarize_verdicts, test_dev_test_split,
               test_train_options, test_chance_control, test_degrade):
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
