"""(파트 A) 후보의 비원형도·두께 분포를 정답 종류별 산점도로 그린다.

사용법:
    python tools/shape_scatter.py
    python tools/shape_scatter.py --limit 50
    python tools/shape_scatter.py --y thickness
    python tools/shape_scatter.py --y dt_width
    python tools/shape_scatter.py --y both

출력:
    results/shape/candidates.csv
    results/shape/scatter_thickness.png
    results/shape/scatter_dt_width.png
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src import config as C
from src import io_utils, pipeline
from tools.convert_json_gt import convert_box
from tools.evaluate_train import assign_split


Y_CHOICES = ("thickness", "dt_width", "both")
OUT_DIR = "results/shape"

CLASS_MAP = {
    "longitudinal crack": "D00",
    "transverse crack": "D10",
    "alligator crack": "D20",
    "pothole": "D40",
}

PLOT_CLASSES = (
    "D00",
    "D10",
    "D20",
    "D40",
    "없음",
)


def parse_args(argv=None):
    """산점도의 세로축, 처리 장수, 저장 위치를 읽는다."""

    parser = argparse.ArgumentParser(
        description="후보 형태 특징 산점도"
    )

    parser.add_argument(
        "--y",
        choices=Y_CHOICES,
        default="both",
    )

    parser.add_argument(
        "--grid",
        action="store_true",
        help="격자 탐색은 S3에서 구현",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--out",
        default=OUT_DIR,
    )

    return parser.parse_args(argv)


def load_gt_json(image_name):
    """RDD2022 JSON에서 640px 기준 GT 박스와 세부 클래스를 읽는다."""

    json_path = Path(
        C.TRAIN_ANN_DIR
    ) / f"{image_name}.json"

    if not json_path.exists():
        return []

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    size = data.get("size")

    if not size:
        return []

    width = int(size["width"])

    if width <= 0:
        return []

    scale = C.TARGET_WIDTH / width

    boxes = []

    for obj in data.get("objects", []):

        if obj.get("geometryType") != "rectangle":
            continue

        title = obj.get(
            "classTitle",
            ""
        ).strip().lower()

        label = CLASS_MAP.get(title)

        if label is None:
            continue

        exterior = obj.get(
            "points",
            {}
        ).get(
            "exterior",
            []
        )

        try:
            x, y, w, h = convert_box(
                exterior,
                scale
            )
        except ValueError:
            continue

        boxes.append(
            (x, y, w, h, label)
        )

    return boxes


def point_in_box(cx, cy, box):
    """한 점이 GT 박스 안에 있는지 확인한다."""

    x, y, w, h, _ = box

    return (
        x <= cx <= x + w
        and
        y <= cy <= y + h
    )


def gt_class_for_region(region, y0, gt_boxes):
    """후보 bbox 중심이 포함된 GT의 세부 클래스를 반환한다."""

    x, y, w, h = region["bbox"]

    cx = x + w / 2
    cy = y + h / 2 + y0

    for box in gt_boxes:
        if point_in_box(
            cx,
            cy,
            box
        ):
            return box[4]

    return "없음"


def collect_candidates(image_paths):
    """개발셋 영상의 후보와 형태 특징을 모은다."""

    rows = []

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):
        name = image_path.name

        bgr = io_utils.load_image(
            str(image_path)
        )

        result = pipeline.run_pipeline(
            bgr,
            C.ROI_TOP_DEFAULT,
            C.ROI_BOTTOM_DEFAULT,
            False,
        )

        gt_boxes = load_gt_json(
            name
        )

        y0 = result["y0"]

        for candidate_type, regions in (
            ("crack", result["cracks"]),
            ("pothole", result["potholes"]),
        ):

            for region in regions:

                rows.append({
                    "image": name,
                    "candidate_type": candidate_type,
                    "circ_inv": region.get(
                        "circ_inv"
                    ),
                    "thickness": region.get(
                        "thickness"
                    ),
                    "dt_width": region.get(
                        "dt_width"
                    ),
                    "solidity": region.get(
                        "solidity"
                    ),
                    "gt_class": gt_class_for_region(
                        region,
                        y0,
                        gt_boxes
                    ),
                })

        print(
            f"[{index}/{len(image_paths)}] "
            f"{name} "
            f"candidates="
            f"{len(result['cracks']) + len(result['potholes'])}"
        )

    return rows


def save_candidates(path, rows):
    """후보별 형태 특징을 CSV로 저장한다."""

    fieldnames = [
        "image",
        "candidate_type",
        "circ_inv",
        "thickness",
        "dt_width",
        "solidity",
        "gt_class",
    ]

    with open(
        path,
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


def draw_scatter(rows, y_key, out_path):
    """circ_inv와 두께 특징의 산점도를 저장한다."""

    plt.figure(
        figsize=(9, 6)
    )

    for gt_class in PLOT_CLASSES:

        selected = [
            r
            for r in rows
            if r["gt_class"] == gt_class
            and r["circ_inv"] is not None
            and r[y_key] is not None
        ]

        if not selected:
            continue

        xs = [
            r["circ_inv"]
            for r in selected
        ]

        ys = [
            r[y_key]
            for r in selected
        ]

        plt.scatter(
            xs,
            ys,
            s=18,
            alpha=0.6,
            label=gt_class,
        )

    # 현재 B 파라미터 기준선
    plt.axvline(
        C.CRACK_MIN_CIRC_INV,
        linestyle="--",
        label=f"crack >= {C.CRACK_MIN_CIRC_INV}",
    )

    plt.axvline(
        C.POT_MAX_CIRC_INV,
        linestyle=":",
        label=f"pothole <= {C.POT_MAX_CIRC_INV}",
    )

    plt.xscale("log")

    plt.xlabel(
        "circ_inv = P² / (4πA)"
    )

    plt.ylabel(
        y_key
    )

    plt.title(
        f"Candidate shape distribution: {y_key}"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=150
    )

    plt.close()


def main(argv=None):
    """개발셋 후보를 모아 형태 특징 CSV와 산점도를 만든다."""

    args = parse_args(argv)

    if args.grid:
        print(
            "안내: --grid는 S3 항목이므로 "
            "현재 M6에서는 산점도만 생성합니다."
        )

    image_dir = Path(
        C.TRAIN_IMAGE_DIR
    )

    image_paths = sorted(
        image_dir.glob("*.jpg")
    )

    if not image_paths:
        raise RuntimeError(
            f"훈련 이미지가 없습니다: {image_dir}"
        )

    # 개발/시험 분할
    split = assign_split(
        [p.name for p in image_paths]
    )

    image_paths = [
        p
        for p in image_paths
        if split[p.name] == "dev"
    ]

    if args.limit is not None:
        image_paths = image_paths[
            :args.limit
        ]

    os.makedirs(
        args.out,
        exist_ok=True
    )

    print(
        "개발셋 영상 수 =",
        len(image_paths)
    )

    rows = collect_candidates(
        image_paths
    )

    csv_path = os.path.join(
        args.out,
        "candidates.csv"
    )

    save_candidates(
        csv_path,
        rows
    )

    if args.y in (
        "thickness",
        "both",
    ):
        draw_scatter(
            rows,
            "thickness",
            os.path.join(
                args.out,
                "scatter_thickness.png"
            )
        )

    if args.y in (
        "dt_width",
        "both",
    ):
        draw_scatter(
            rows,
            "dt_width",
            os.path.join(
                args.out,
                "scatter_dt_width.png"
            )
        )

    print()
    print(
        "후보 수 =",
        len(rows)
    )

    print(
        "CSV =",
        csv_path
    )

    if args.y in (
        "thickness",
        "both",
    ):
        print(
            "산점도 =",
            os.path.join(
                args.out,
                "scatter_thickness.png"
            )
        )

    if args.y in (
        "dt_width",
        "both",
    ):
        print(
            "산점도 =",
            os.path.join(
                args.out,
                "scatter_dt_width.png"
            )
        )


if __name__ == "__main__":
    main()
