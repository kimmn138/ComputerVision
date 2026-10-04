"""파트 C 시험: features의 해리스 점 세기와 SIFT 매칭, visualize의 비교 그림 저장,
run_experiment의 좌표 변환·시간 중앙값·요약 계산이 경계 입력에서도 약속대로 동작하는지 확인한다.
사용법: 맨 위 폴더에서  python tests/test_c.py"""
import math
import os
import sys
import tempfile
import warnings

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_experiment as rx  # noqa: E402
from src import features, visualize  # noqa: E402
from src import config as C  # noqa: E402

FAILS = []


def check(ok, msg):
    if not ok:
        FAILS.append(msg)


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


def main():
    for fn in (test_harris, test_match_blank, test_match_few, test_match_boundary, test_match_texture, test_no_inplace,
               test_save_comparison, test_get_roi, test_median_times, test_gt_counts, test_ratios_and_pr,
               test_summary_tables):
        try:
            fn()
        except Exception as e:
            FAILS.append(f"{fn.__name__}: 실행 중 오류 {type(e).__name__}: {e}")
    if FAILS:
        print(f"C 시험 실패 {len(FAILS)}건")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("C 시험 통과")


if __name__ == "__main__":
    main()
