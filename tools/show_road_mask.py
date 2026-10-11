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

OUT_DIR = "results/road_mask"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=("dev39", "diag35"), default="dev39")
    parser.add_argument("--out", default=OUT_DIR)
    return parser.parse_args(argv)


def list_images():
    return (
        sorted(glob.glob("data/provided/*.jpg"))
        + sorted(glob.glob("data/own/*.jpg"))
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

    if args.set != "dev39":
        raise NotImplementedError(
            "현재는 dev39 확인만 구현했습니다."
        )

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

    for path in list_images():
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

        gt_path = os.path.join(
            "gt",
            f"{os.path.splitext(name)[0]}.csv"
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
                "mask_ratio",
                "gt_in_roi",
                "gt_in_road",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

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
