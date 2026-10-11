"""검출 결과를 정답 박스와 비교해 성능(정밀도·재현율, 영상 단위 판정)을 잰다."""

import csv
from pathlib import Path

import numpy as np


def load_gt(path):
    """정답 CSV를 읽어 [(x, y, w, h, kind)]로 돌려준다. 파일이 없으면 None."""
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
    """검출 박스와 정답 박스를 IoU 기준으로 일대일 대응시켜 tp·fp·found·fn을 센다."""
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
    """검출 박스의 중심이 정답 박스 안이면 맞힌 것으로 센다.
    큰 균열 정답 안의 조각 검출은 IoU가 낮아서, IoU 기준과 함께 본다."""
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
    """개수 dict로 (정밀도, 재현율)을 계산한다. 분모가 0이면 NaN."""
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


def image_verdict(pred_by_kind, gt):
    """영상 한 장의 판정이 맞았는지 본다.
    손상 영상: 같은 종류 후보의 중심이 정답 박스 안에 하나라도 있으면 맞힘. 손상 없는 영상: 후보가 0개면 맞힘."""
    truth_by_kind = {}

    for x, y, w, h, kind in gt:
        truth_by_kind.setdefault(kind, []).append((x, y, w, h))

    all_pred = [
        box
        for boxes in pred_by_kind.values()
        for box in boxes
    ]

    all_truth = [box[:4] for box in gt]

    hit_same = any(
        count_center_hits(pred_by_kind.get(kind, []), truth)["found"] > 0
        for kind, truth in truth_by_kind.items()
    )

    hit_any = count_center_hits(all_pred, all_truth)["found"] > 0

    has_damage = len(gt) > 0
    n_pred = len(all_pred)

    return {
        "has_damage": has_damage,
        "n_pred": n_pred,
        "hit_same": hit_same,
        "hit_any": hit_any,
        "correct": hit_same if has_damage else n_pred == 0,
    }


def summarize_verdicts(verdicts):
    """여러 장의 판정을 민감도·특이도·정답률·균형 정확도로 묶는다. 분모가 0이면 NaN.
    baseline_none = 아무것도 검출하지 않았을 때의 정답률."""
    n = len(verdicts)
    n_damage = sum(1 for v in verdicts if v["has_damage"])
    n_none = n - n_damage

    correct_damage = sum(1 for v in verdicts if v["has_damage"] and v["correct"])
    correct_none = sum(1 for v in verdicts if not v["has_damage"] and v["correct"])

    sensitivity = correct_damage / n_damage if n_damage else np.nan
    specificity = correct_none / n_none if n_none else np.nan

    return {
        "n": n,
        "n_damage": n_damage,
        "n_none": n_none,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": (correct_damage + correct_none) / n if n else np.nan,
        "balanced": (sensitivity + specificity) / 2,     # 한쪽이 NaN이면 NaN
        "baseline_none": n_none / n if n else np.nan,
    }