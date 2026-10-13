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
from src import detect, io_utils, pipeline
from src.evaluate import load_gt
from tools import evaluate_train as et
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

# --grid(보완 5-1): 분류 문턱 격자. 지금 값(4, 2.5, 0.6)을 가운데 두고 양쪽으로 한 칸 이상. 균열 하한 > 포트홀 상한인 조합만 쓴다
GRID_CRACK_MIN = (3.0, 3.5, 4.0, 4.5, 5.0)     # CRACK_MIN_CIRC_INV 후보
GRID_POT_MAX = (2.0, 2.5, 3.0)                 # POT_MAX_CIRC_INV 후보
GRID_POT_SOL = (0.5, 0.6, 0.7)                 # POT_MIN_SOLIDITY 후보
GRID_KEYS = ("CRACK_MIN_CIRC_INV", "POT_MAX_CIRC_INV", "POT_MIN_SOLIDITY")

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


def collect_dark_regions(image_paths):
    """--grid용: 개발셋 사진마다 분류 전 어두운 영역을 모두 모은다(전처리 끔, 채택 규칙의 판정 행). run_pipeline과 같은 띠·노면 마스크·
    노면 밖 후보 제거(USE_ROAD_MASK·ROAD_KEEP_RULE)를 쓰고, 형태 분류(균열·포트홀 문턱)와 점수 문턱 τ만 적용하지 않는다.
    → ({사진: [영역(contour 뺌, bbox는 640px 전체 좌표)]}, {사진: 정답}, {사진: 640px 높이})"""
    roi_table = io_utils.load_roi_table(et.ROI_PATH)
    regions, gts, heights = {}, {}, {}
    for index, image_path in enumerate(image_paths, start=1):
        gt = load_gt(str(Path(C.TRAIN_GT_DIR) / f"{image_path.stem}.csv"))
        if gt is None:
            continue
        bgr = io_utils.load_image(str(image_path))
        roi = roi_table.get(image_path.name, {"top": C.ROI_TOP_DEFAULT, "bottom": C.ROI_BOTTOM_DEFAULT})
        out = pipeline.run_pipeline(bgr, roi["top"], roi["bottom"], False)
        found, _ = detect.extract_dark_regions(out["proc"], out["road"] if C.USE_ROAD_MASK else None)
        if C.USE_ROAD_MASK:
            found = pipeline.keep_on_road(found, out["road"])
        regions[image_path.name] = [{k: v for k, v in r.items() if k != "contour"} | {"bbox": (r["bbox"][0], r["bbox"][1] + out["y0"],
                                    r["bbox"][2], r["bbox"][3])} for r in found]
        gts[image_path.name] = gt
        heights[image_path.name] = out["small"].shape[0]
        if index % 50 == 0:
            print(f"[{index}/{len(image_paths)}] 영역 모음")
    return regions, gts, heights


def grid_combos():
    """격자 조합(균열 하한, 포트홀 상한, 포트홀 solidity 하한) 가운데 균열 하한 > 포트홀 상한인 것(한 영역이 두 종류가 되지 않게)."""
    return [(c, p, q) for c in GRID_CRACK_MIN for p in GRID_POT_MAX for q in GRID_POT_SOL if c > p]


def scored_for(regions, combo):
    """한 격자 조합으로 영역을 분류하고 점수(config의 SCORE_FORMULA, margin은 문턱에서의 여유라 조합마다 다시 계산)를 매긴다.
    → {사진: {"crack": [(상자, 점수)], "pothole": [...]}} (evaluate_train.tau_curve가 받는 모양)"""
    backup = {k: getattr(C, k) for k in GRID_KEYS}
    try:
        for k, v in zip(GRID_KEYS, combo):
            setattr(C, k, v)
        out = {}
        for name, found in regions.items():
            out[name] = {"crack": [], "pothole": []}
            for r in found:
                crack, pothole = detect.region_scores(r, float("nan"))
                if r["circ_inv"] >= C.CRACK_MIN_CIRC_INV:
                    out[name]["crack"].append((r["bbox"], crack))
                if r["circ_inv"] <= C.POT_MAX_CIRC_INV and r["solidity"] >= C.POT_MIN_SOLIDITY:
                    out[name]["pothole"].append((r["bbox"], pothole))
        return out
    finally:
        for k, v in backup.items():
            setattr(C, k, v)


def grid_rows(regions, gts, heights, combos):
    """격자 조합마다 τ 곡선(evaluate_train.tau_curve, 채택 규칙과 같은 정의)을 메모리에서 내고 동작점(τ 없음·fp1·fp2)의 지표를 모은다.
    → [dict] (조합 값, none·fp1·fp2의 민감도·특이도·균형 정확도·영상당 오검출, fp2의 τ·우연 뺀 균열 재현율)"""
    names = sorted(regions)
    rows = []
    for combo in combos:
        points = et.tau_points(et.tau_curve(names, heights, scored_for(regions, combo), gts))
        row = dict(zip(GRID_KEYS, combo))
        for point in ("none", "fp1", "fp2"):
            for col in ("sensitivity", "specificity", "accuracy", "balanced", "fp_per_image"):
                row[f"{point}_{col}"] = float(points[point][col])
        row.update({"fp2_tau_crack": float(points["fp2"]["tau_crack"]), "fp2_tau_pothole": float(points["fp2"]["tau_pothole"]),
                    "fp2_crack_net": float(points["fp2"]["crack_ctr_recall_net"])})
        rows.append(row)
    return rows


def pick_grid(rows):
    """3단계와 같은 고르기: fp2 민감도가 가장 높은 조합, 같으면 fp1 민감도, 그다음 지금 config 값에 가까운 쪽(각 문턱의 상대 차이 합)."""
    def distance(r):
        return sum(abs(r[k] - getattr(C, k)) / getattr(C, k) for k in GRID_KEYS)
    return sorted(rows, key=lambda r: (-round(r["fp2_sensitivity"], 6), -round(r["fp1_sensitivity"], 6), distance(r)))[0]


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

    if args.grid:                       # 보완 5-1: 분류 문턱 격자 탐색(산점도는 그리지 않음)
        if C.DETECT_MODE != "dark":
            raise RuntimeError("--grid는 DETECT_MODE 'dark'에서만 쓴다(분류 문턱이 dark 방식의 것)")
        regions, gts, heights = collect_dark_regions(image_paths)
        rows = grid_rows(regions, gts, heights, grid_combos())
        best = pick_grid(rows)
        grid_path = os.path.join(args.out, "grid.csv")
        with open(grid_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) + ["picked"])
            writer.writeheader()
            writer.writerows([{**r, "picked": r is best} for r in rows])
        print("격자 =", grid_path, f"({len(rows)}개 조합)")
        print("고른 값 =", ", ".join(f"{k} {best[k]}" for k in GRID_KEYS),
              f"(fp2 민감도 {best['fp2_sensitivity']:.4f}, fp1 {best['fp1_sensitivity']:.4f})")
        print("지금 값 =", ", ".join(f"{k} {getattr(C, k)}" for k in GRID_KEYS))
        return

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
