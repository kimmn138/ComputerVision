"""(파트 B) 검출 후보와 정답 박스를 비교하여 검출 성능을 평가한다."""

import csv
from pathlib import Path

import numpy as np


def load_gt(path):
    """정답 CSV를 읽어 (x, y, w, h, kind) 목록으로 반환한다."""
    path = Path(path)

    if not path.exists():
        return None

    gt = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            gt.append((
                int(row["x"]),
                int(row["y"]),
                int(row["w"]),
                int(row["h"]),
                row["kind"],
            ))

    return gt


def iou(a, b):
    """두 바운딩 박스의 IoU를 계산한다."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def count_matches(pred, gt, thr):
    """예측 박스와 정답 박스를 IoU 기준으로 일대일 대응시켜 개수를 센다."""
    matched_gt = set()
    tp = 0

    for pred_box in pred:
        best_iou = 0.0
        best_idx = None

        for i, gt_box in enumerate(gt):
            if i in matched_gt:
                continue

            score = iou(pred_box, gt_box)

            if score >= thr and score > best_iou:
                best_iou = score
                best_idx = i

        if best_idx is not None:
            matched_gt.add(best_idx)
            tp += 1

    fp = len(pred) - tp
    found = len(matched_gt)
    fn = len(gt) - found

    return {
        "tp": int(tp),
        "fp": int(fp),
        "found": int(found),
        "fn": int(fn),
    }


def count_center_hits(pred, gt):
    """검출 박스의 중심이 정답 박스 안에 있으면 맞힌 것으로 세어, IoU로는 놓치는 조각 검출도 평가한다.
    균열 정답은 균열 전체를 감싼 큰 박스이고 검출은 그 안의 조각이라 IoU가 낮게 나오므로 IoU 기준과 함께 본다.
    tp·fp는 검출 기준(중심이 어떤 정답 안에 들었는지, 한 정답에 여러 검출 가능),
    found·fn은 정답 기준(안에 검출 중심이 하나라도 있는지)이라 tp와 found가 다를 수 있다."""
    centers = [(x + w / 2, y + h / 2) for x, y, w, h in pred]

    def inside(point, box):
        bx, by, bw, bh = box[:4]
        return bx <= point[0] <= bx + bw and by <= point[1] <= by + bh

    tp = sum(any(inside(c, g) for g in gt) for c in centers)
    found = sum(any(inside(c, g) for c in centers) for g in gt)

    return {
        "tp": int(tp),
        "fp": int(len(pred) - tp),
        "found": int(found),
        "fn": int(len(gt) - found),
    }


def precision_recall(counts):
    """검출 개수에서 정밀도와 재현율을 계산하며 분모가 0이면 NaN을 반환한다."""
    tp = counts["tp"]
    fp = counts["fp"]
    found = counts["found"]
    fn = counts["fn"]

    precision_den = tp + fp
    recall_den = found + fn

    precision = (
        tp / precision_den
        if precision_den != 0
        else np.nan
    )

    recall = (
        found / recall_den
        if recall_den != 0
        else np.nan
    )

    return precision, recall