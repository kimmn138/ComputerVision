"""매칭 실험: 쌍 사진(data/pairs)의 SIFT 매칭이 전처리로 좋아지는지 잰다.
실행: python run_matching.py --name <이름>
결과: results/<이름>/
- matching.csv: 기준 사진 A와 B·C·D를 전처리 끔·켬으로 매칭한 결과
- matches/: RANSAC inlier를 초록 선으로 그린 그림
"""
import argparse
import glob
import math
import os
import statistics
import sys
import time

import cv2 as cv
import pandas as pd

import run_experiment as rx
from src import analyze, features, io_utils, preprocess
from src import config as C

PAIR_DIR = "data/pairs"
REF_ROLE = "A"                      # 기준 사진 기호
MATCH_COLUMNS = ["pair", "target", "condition", "preprocess", "kp_ref", "kp_target",
                 "good", "inliers", "inlier_ratio", "t_match_ms"]
INLIER_COLOR = (0, 255, 0)          # inlier 선: 초록 (BGR)


def parse_args(argv=None):
    """명령행에서 결과 폴더 이름(--name)을 읽는다."""
    parser = argparse.ArgumentParser(description="쌍 사진 매칭의 전처리 끔·켬 비교 실험")
    parser.add_argument("--name", default=rx.DEFAULT_NAME, help=f"결과 폴더 이름 (results/<name>/, 기본 {rx.DEFAULT_NAME})")
    args = parser.parse_args(argv)
    rx.check_name(parser, args.name)
    return args


def group_pairs(paths, parse_fn=io_utils.parse_name):
    """사진을 pair 번호로 묶어 ({pair: {role: (경로, 조건)}}, 경고 목록)을 돌려준다.
    이름이 잘못됐거나 기준 사진 A가 없는 쌍은 경고만 남기고 건너뛴다."""
    groups, warnings = {}, []
    for path in paths:
        name = os.path.basename(path)
        try:
            info = parse_fn(path)
        except ValueError as e:
            warnings.append(f"이름 규칙 위반으로 건너뜀: {name} ({e})")
            continue
        if "pair" not in info or "role" not in info:
            warnings.append(f"쌍 정보(pair·role)가 없어 건너뜀: {name}")
            continue
        members = groups.setdefault(info["pair"], {})
        if info["role"] in members:
            warnings.append(f"pair {info['pair']}에 {info['role']}가 두 장이라 뒤의 것을 건너뜀: {name}")
            continue
        members[info["role"]] = (path, info.get("condition", ""))

    for pair in [p for p, members in groups.items() if REF_ROLE not in members]:
        warnings.append(f"pair {pair}에 기준 사진 {REF_ROLE}가 없어 건너뜀")
        del groups[pair]
    return dict(sorted(groups.items(), key=lambda kv: str(kv[0]))), warnings


def prepare(path, use_preprocess):
    """사진을 640px 흑백으로 읽고, 켬이면 상태에 맞는 전처리를 적용한다. (흑백 영상, steps)를 돌려준다."""
    gray = cv.cvtColor(io_utils.resize_width(io_utils.load_image(path)), cv.COLOR_BGR2GRAY)
    if not use_preprocess:
        return gray, dict(analyze.NO_STEPS)
    steps = analyze.choose_steps(analyze.measure_quality(gray))
    return preprocess.preprocess(gray, steps), steps


def timed_match(g_ref, g_target, repeat=C.TIME_REPEAT):
    """match_pair를 repeat번 돌려 (첫 결과, 시간 중앙값 ms)를 돌려준다."""
    results, times = [], []
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        results.append(features.match_pair(g_ref, g_target))
        times.append((time.perf_counter() - t0) * 1000)
    return results[0], round(statistics.median(times), 2)


def make_row(pair, target, condition, use_preprocess, m, t_ms):
    """matching.csv 한 행을 만든다. RANSAC을 못 했으면 inlier_ratio를 NaN으로 둬 평균에서 빠지게 한다."""
    ratio = math.nan if m["mask"] is None else m["inlier_ratio"]
    return {"pair": pair, "target": target, "condition": condition,
            "preprocess": "on" if use_preprocess else "off",
            "kp_ref": m["kp1"], "kp_target": m["kp2"], "good": m["good"],
            "inliers": m["inliers"], "inlier_ratio": ratio, "t_match_ms": t_ms}


def draw_inliers(g_ref, g_target, m):
    """RANSAC inlier만 초록 선으로 이은 그림을 만든다. RANSAC을 못 했으면 선을 그리지 않는다."""
    mask = m["mask"] if m["mask"] is not None else [0] * len(m["matches"])
    return cv.drawMatches(g_ref, m["k1"], g_target, m["k2"], m["matches"], None,
                          matchColor=INLIER_COLOR, matchesMask=mask,
                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def process_pair(pair, members, fig_dir):
    """한 쌍에서 기준 A와 나머지(B·C·D)를 끔·켬으로 매칭해 행 list를 만들고 그림을 저장한다."""
    ref_path = members[REF_ROLE][0]
    rows = []
    for use in (False, True):
        g_ref, _ = prepare(ref_path, use)               # 기준 사진은 끔·켬마다 한 번만 준비
        for role in sorted(r for r in members if r != REF_ROLE):
            path, condition = members[role]
            g_target, _ = prepare(path, use)
            m, t_ms = timed_match(g_ref, g_target)
            rows.append(make_row(pair, role, condition, use, m, t_ms))
            stem = os.path.splitext(os.path.basename(path))[0]
            cv.imwrite(os.path.join(fig_dir, f"{stem}_{'on' if use else 'off'}.png"), draw_inliers(g_ref, g_target, m))
    return rows


def main(argv=None, pair_dir=PAIR_DIR, results_dir="results", parse_fn=io_utils.parse_name):
    """쌍 사진 전체를 매칭해 결과를 저장한다. 실패한 쌍이 있으면 1, 없으면 0을 돌려준다."""
    args = parse_args(argv)
    out_dir = f"{results_dir}/{args.name}"
    fig_dir = f"{out_dir}/matches"
    os.makedirs(fig_dir, exist_ok=True)
    rx.write_config(f"{out_dir}/config_used_matching.txt")

    groups, warnings = group_pairs(sorted(glob.glob(f"{pair_dir}/*.jpg")), parse_fn)
    for w in warnings:
        print(f"경고: {w}")
    rows, failures = [], []
    for pair, members in groups.items():
        try:
            rows += process_pair(pair, members, fig_dir)
        except Exception as e:                          # 한 쌍이 실패해도 나머지는 계속 처리
            failures.append((pair, f"{type(e).__name__}: {e}"))

    pd.DataFrame(rows, columns=MATCH_COLUMNS).to_csv(f"{out_dir}/matching.csv", encoding="utf-8-sig", index=False)
    print(f"{len(groups) - len(failures)}/{len(groups)}쌍 처리 (건너뜀 경고 {len(warnings)}건) -> {out_dir}/")
    if failures:
        print(f"실패 {len(failures)}쌍:")
        for pair, msg in failures:
            print(f"  - pair {pair}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
