"""(파트 A) 자동 노면 마스크를 실제 영상 위에 겹쳐 확인하고 정답 유지율을 계산한다.

사용법:
    python tools/show_road_mask.py
    python tools/show_road_mask.py --set dev39
    python tools/show_road_mask.py --set diag35
    python tools/show_road_mask.py --out results/road_mask

출력:
- results/road_mask/<set>/<영상이름>.jpg
- results/road_mask/<set>/summary.csv

그림:
- 노란색 선: ROI 시작 위치
- 반투명 초록색: 자동으로 판단한 노면
- 빨간색 박스: 균열 GT
- 파란색 박스: 포트홀 GT

summary.csv:
- 마스크 면적 비율
- ROI 안 정답 수
- road mask 안에 남은 정답 수
"""
import argparse
import csv
import glob
import os
import sys

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as C
from src import io_utils
from src.evaluate import load_gt
from tools.evaluate_train import DIAG35

OUT_DIR = "results/road_mask"


COMPARE_DIR = "results/road_mask_compare"
# --compare 패널: (제목, 이 패널에서만 바꿀 config 값). v1 = v2 스위치를 모두 끈 마스크, 8a~8c = 그 스위치 하나만 켬, config = 지금 config.
# 8a의 배수 2는 보완 4-2의 개발셋 후보 1.5·2 가운데 큰 값(넓히는 효과가 잘 보이게, part_C 4-2)
V2_OFF = {"ROAD_SHADOW_L_K": 0, "ROAD_HOOD_CUT": False, "ROAD_EDGE_DENSITY": False}
COMPARE_PANELS = (("v1", V2_OFF), ("8a shadow K2", {**V2_OFF, "ROAD_SHADOW_L_K": 2}),
                  ("8b hood", {**V2_OFF, "ROAD_HOOD_CUT": True}), ("8c edge density", {**V2_OFF, "ROAD_EDGE_DENSITY": True}),
                  ("config", {}))


def parse_args(argv=None):
    """확인할 사진 묶음(--set)·저장 폴더(--out)와, v1·v2 마스크를 나란히 그릴지(--compare, 보완 4-3)를 읽는다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=("dev39", "diag35"), default="dev39")
    parser.add_argument("--out", default=None, help=f"저장 폴더. 기본 {OUT_DIR}, --compare면 {COMPARE_DIR}")
    parser.add_argument("--compare", action="store_true",
                        help="v1(v2 스위치 모두 끔)·8a·8b·8c(하나씩 켬)·config 마스크를 한 장에 나란히 저장 (보완 4-3, 채택 판정 없음)")
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = COMPARE_DIR if args.compare else OUT_DIR
    return args


def masks_by_panel(roi):
    """COMPARE_PANELS마다 config 값을 잠시 바꿔 노면 마스크를 만들고 되돌린다 → [(제목, 마스크)]. 받은 영상은 바꾸지 않는다."""
    out = []
    for title, values in COMPARE_PANELS:
        backup = {k: getattr(C, k) for k in values}
        try:
            for k, v in values.items():
                setattr(C, k, v)
            out.append((title, io_utils.road_mask(roi)))
        finally:
            for k, v in backup.items():
                setattr(C, k, v)
    return out


def compare_figure(small, roi, y0, gt):
    """한 사진의 패널들을 가로로 이어 붙인 그림과 패널마다 (제목, 노면 비율, 띠 안 정답, 마스크 안 정답)을 돌려준다."""
    panels, stats = [], []
    for title, road in masks_by_panel(roi):
        drawn = draw_gt(draw_road(small, road, y0), gt)
        cv.putText(drawn, title, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(drawn, title, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
        n_roi, n_road = count_gt(gt, road, y0)
        panels.append(drawn)
        stats.append((title, round(np.count_nonzero(road) / road.size, 4), n_roi, n_road))
    return np.hstack(panels), stats

def list_images(set_name):
    """dev39 또는 diag35에 해당하는 영상 경로를 반환한다."""

    if set_name == "dev39":
        return (
            sorted(glob.glob("data/provided/*.jpg"))
            + sorted(glob.glob("data/own/*.jpg"))
        )

    if set_name == "diag35":
        return [
            os.path.join(
                C.TRAIN_IMAGE_DIR,
                f"{name}.jpg"
            )
            for name in DIAG35
        ]

    raise ValueError(
        f"알 수 없는 set: {set_name}"
    )

def draw_road(img, road, y0):
    out = img.copy()

    roi = out[y0:y0 + road.shape[0]].copy()

    green = np.zeros_like(roi)
    green[:] = (0, 255, 0)

    blended = cv.addWeighted(
        roi,
        0.65,
        green,
        0.35,
        0
    )

    mask = road > 0
    roi[mask] = blended[mask]

    out[y0:y0 + road.shape[0]] = roi

    cv.line(
        out,
        (0, y0),
        (out.shape[1] - 1, y0),
        (0, 255, 255),
        2
    )

    return out


def draw_gt(img, gt):
    out = img.copy()

    if gt is None:
        return out

    for x, y, w, h, kind in gt:
        color = (0, 0, 255) if kind == "crack" else (255, 0, 0)

        cv.rectangle(
            out,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

    return out


def count_gt(gt, road, y0):
    if gt is None:
        return 0, 0

    in_roi = 0
    in_road = 0

    h, w = road.shape

    for x, y, bw, bh, kind in gt:
        cx = int(x + bw / 2)
        cy = int(y + bh / 2)

        if not (0 <= cx < w and y0 <= cy < y0 + h):
            continue

        in_roi += 1

        my = cy - y0

        if road[my, cx] > 0:
            in_road += 1

    return in_roi, in_road


def main(argv=None):
    args = parse_args(argv)

    roi_table = io_utils.load_roi_table("data/roi.csv")

    out_dir = os.path.join(
        args.out,
        args.set
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    rows = []

    for path in list_images(args.set):
        name = os.path.basename(path)

        bgr = io_utils.load_image(path)
        small = io_utils.resize_width(bgr)

        roi_info = roi_table.get(
            name,
            {
                "top": C.ROI_TOP_DEFAULT,
                "bottom": C.ROI_BOTTOM_DEFAULT
            }
        )

        roi, y0 = io_utils.crop_roi(
            small,
            roi_info["top"],
            roi_info["bottom"]
        )

        road = io_utils.road_mask(roi)

        stem = os.path.splitext(name)[0]

        if args.compare:
            gt_path = os.path.join(C.TRAIN_GT_DIR if args.set == "diag35" else "gt", f"{stem}.csv")
            figure, stats = compare_figure(small, roi, y0, load_gt(gt_path))
            cv.imwrite(os.path.join(out_dir, f"{stem}.jpg"), figure)
            rows += [{"file": name, "panel": t, "mask_ratio": r, "gt_in_roi": a, "gt_in_road": b} for t, r, a, b in stats]
            print(name, " | ".join(f"{t} road={r:.3f} GT={b}/{a}" for t, r, a, b in stats))
            continue

        if args.set == "diag35":
            gt_path = os.path.join(
                C.TRAIN_GT_DIR,
                f"{stem}.csv"
            )
        else:
            gt_path = os.path.join(
                "gt",
                f"{stem}.csv"
            )

        gt = load_gt(gt_path)

        n_roi, n_road = count_gt(
            gt,
            road,
            y0
        )

        mask_ratio = (
            np.count_nonzero(road)
            / road.size
        )

        drawn = draw_road(
            small,
            road,
            y0
        )

        drawn = draw_gt(
            drawn,
            gt
        )

        cv.imwrite(
            os.path.join(out_dir, name),
            drawn
        )

        rows.append({
            "file": name,
            "mask_ratio": round(mask_ratio, 4),
            "gt_in_roi": n_roi,
            "gt_in_road": n_road,
        })

        print(
            name,
            f"road={mask_ratio:.3f}",
            f"GT={n_road}/{n_roi}"
        )

    summary_path = os.path.join(
        out_dir,
        "summary.csv"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                *(["panel"] if args.compare else []),
                "mask_ratio",
                "gt_in_roi",
                "gt_in_road",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    if args.compare:                    # 패널마다 정답 유지율(띠 안 정답 중 마스크 안)
        for title, _ in COMPARE_PANELS:
            part = [r for r in rows if r["panel"] == title]
            n_roi, n_road = sum(r["gt_in_roi"] for r in part), sum(r["gt_in_road"] for r in part)
            if n_roi:
                print(f"{title} 정답 유지율: {n_road}/{n_roi} = {n_road / n_roi * 100:.2f}%")
        return

    total_roi = sum(
        r["gt_in_roi"]
        for r in rows
    )

    total_road = sum(
        r["gt_in_road"]
        for r in rows
    )

    if total_roi > 0:
        retention = total_road / total_roi

        print(
            f"전체 정답 유지율: "
            f"{total_road}/{total_roi} "
            f"= {retention * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
