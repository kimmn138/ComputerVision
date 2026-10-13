"""(파트 C, 10/14) 보완 최종 설정에서 포트홀을 따로 잰다(1번 노면 방법을 정하기 전, 코드는 바꾸지 않음).
사용법(맨 위 폴더에서): python tools/pothole_v6.py → results/pothole_v6/
1. 측정 B 다시 보고: results/measure_v5/B_boxes_<셋>.csv(10/13 16:24, 보완 최종과 같은 config)를 39장·개발셋으로 나눠 ΔE·질감 비 AUC.
   39장 포트홀 상자는 지금 config로 다시 재서 CSV와 같은지 대조한다(measure_v5.analyze_image를 그대로 부름).
2. 39장 포트홀 정답 상자마다 안·바깥 고리의 평균 L·a·b(CIE, L 0~100), ΔE76, 질감 비 → box_stats_39.csv
3. 39장 포트홀 정답 상자 안의 어두운 영역(detect.extract_dark_regions, 자동 ROI, 전처리 미적용·적용)마다 어디서 걸러지는지:
   ROI 밖 → 모양(균열: 비원형도 ≥ CRACK_MIN_CIRC_INV, 포트홀: 비원형도 ≤ POT_MAX_CIRC_INV 그리고 solidity ≥ POT_MIN_SOLIDITY,
   둘 다 아님) → 점수 임계값(그 종류의 SCORE_TAU) → 노면 밖 후보 제거(영역의 노면 안 비율 < ROAD_KEEP_RATIO_MIN) → 남음.
   → candidates_39.csv(영역마다), boxes_39.csv(상자마다), summary.txt. 시험셋 사진은 열지 않는다."""
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src import detect, evaluate, io_utils, pipeline  # noqa: E402
from tools import measure_v5 as mv  # noqa: E402

OUT_DIR = Path("results/pothole_v6")
B_DIR = Path("results/measure_v5")


def separation(B, set_name):
    """측정 B 표 한 줄: 포트홀 정답(고리 측정된 것) vs 같은 크기 가짜(일반 노면) 상자의 ΔE·질감 비 AUC와 중앙값."""
    pos = B[(B["set"] == set_name) & (B["kind"] == "pothole") & B["ok"]]
    neg = B[(B["set"] == set_name) & (B["kind"] == "fake_pothole") & B["ok"]]
    return {"set": set_name, "pothole_boxes": int(((B["set"] == set_name) & (B["kind"] == "pothole")).sum()), "pothole_ok": len(pos),
            "fake_ok": len(neg), "dE_auc": mv.auc(pos["dE"].dropna(), neg["dE"].dropna()),
            "dE_median_pothole": pos["dE"].median(), "dE_median_fake": neg["dE"].median(),
            "tex_auc": mv.auc(pos["tex_ratio"].dropna(), neg["tex_ratio"].dropna()),
            "tex_median_pothole": pos["tex_ratio"].median(), "tex_median_fake": neg["tex_ratio"].median(),
            "dL_median_pothole": pos["dL"].median()}


def fate(region, kind, road):
    """어두운 영역 하나가 검출기에서 어디까지 가는지(모양 → 점수 임계값 → 노면 밖 제거)."""
    if kind == "none":
        return "모양 탈락(균열·포트홀 어느 쪽도 아님)"
    score = region[f"score_{kind}"]
    tau = C.SCORE_TAU_CRACK if kind == "crack" else C.SCORE_TAU_POTHOLE
    if C.USE_SCORE_TAU and not score >= tau:
        return f"점수 임계값 탈락({kind})"
    if C.USE_ROAD_MASK and pipeline.road_ratio(region, road) < C.ROAD_KEEP_RATIO_MIN:
        return f"노면 밖 제거({kind})"
    return f"남음({kind})"


def shape_kind(r):
    """detect_cracks·detect_potholes의 모양 조건(dark 방식, 보완 최종에서 밝은 자갈 포트홀 끔)."""
    if r["circ_inv"] >= C.CRACK_MIN_CIRC_INV:
        return "crack"
    if r["circ_inv"] <= C.POT_MAX_CIRC_INV and r["solidity"] >= C.POT_MIN_SOLIDITY:
        return "pothole"
    return "none"


def trace_image(item):
    """39장 사진 1장의 포트홀 정답 상자마다 (상자 행, 영역 행 list)를 전처리 미적용·적용으로 낸다."""
    name, path, gt_path, top0, bottom = item
    gt = [(i, g) for i, g in enumerate(evaluate.load_gt(str(gt_path)) or []) if g[4] == "pothole"]   # 번호는 정답 전체 순번(측정 B와 같게)
    if not gt:
        return [], []
    bgr = io_utils.load_image(path)
    box_rows, reg_rows = [], []
    for mode, use_pre in (("off", False), ("on", True)):
        out = pipeline.run_pipeline(bgr, top0, bottom, use_pre)
        y0, road = out["y0"], out["road"]
        regions, dark = detect.extract_dark_regions(out["proc"], road if C.USE_ROAD_MASK else None)
        h = road.shape[0]
        for i, (x, y, w, bh, _) in gt:
            ry0, ry1 = max(0, y - y0), min(h, y + bh - y0)        # 정답 상자의 ROI 안 부분(노면 띠 좌표)
            in_roi = ry1 > ry0
            box = {"image": name, "idx": i, "mode": mode, "x": x, "y": y, "w": w, "h": bh, "top": out["top"], "y0": y0,
                   "box_in_roi": float(max(0, ry1 - ry0) / bh), "center_in_roi": bool(y + bh / 2 >= y0),
                   "road_in_box": float((road[ry0:ry1, x:x + w] > 0).mean()) if in_roi else np.nan,
                   "dark_in_box": float((dark[ry0:ry1, x:x + w] > 0).mean()) if in_roi else np.nan,
                   "steps": str(out["steps"])}
            fates = []
            for r in regions:
                fill = np.zeros((h, road.shape[1]), np.uint8)
                cv.drawContours(fill, [r["contour"]], -1, 255, -1)
                inside = int(np.count_nonzero(fill[ry0:ry1, x:x + w])) if in_roi else 0
                if inside == 0:
                    continue
                kind = shape_kind(r)
                f = fate(r, kind, road)
                bx, by, bw2, bh2 = r["bbox"]
                fates.append(f)
                reg_rows.append({"image": name, "idx": i, "mode": mode, "area": r["area"], "in_box_px": inside,
                                 "in_box_frac": inside / max(1, r["area"]), "center_in_box": bool(x <= bx + bw2 / 2 <= x + w and y <= by + bh2 / 2 + y0 <= y + bh),
                                 "circ_inv": r["circ_inv"], "solidity": r["solidity"], "contrast": r["contrast"],
                                 "score_crack": r["score_crack"], "score_pothole": r["score_pothole"], "shape": kind,
                                 "road_ratio": pipeline.road_ratio(r, road), "fate": f})
            box.update({"n_dark_regions": len(fates), "kept_pothole": sum(f == "남음(pothole)" for f in fates),
                        "kept_crack": sum(f == "남음(crack)" for f in fates),
                        "fates": "; ".join(f"{k} {fates.count(k)}" for k in sorted(set(fates))) if fates else "어두운 영역 없음"})
            box_rows.append(box)
    return box_rows, reg_rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"보완 최종 config(점수 임계값 균열 {C.SCORE_TAU_CRACK}·포트홀 {C.SCORE_TAU_POTHOLE}), 39장은 자동 ROI. 시험셋 사진은 열지 않음.", ""]
    B = pd.concat([pd.read_csv(B_DIR / f"B_boxes_{s}.csv", encoding="utf-8-sig", dtype={"set": str}) for s in ("dev", "39")])
    sep = pd.DataFrame([separation(B, s) for s in ("39", "dev")])
    sep.to_csv(OUT_DIR / "separation.csv", encoding="utf-8-sig", index=False)
    lines += ["1. 측정 B(포트홀 정답 vs 같은 크기 가짜 노면 상자)", "| 셋 | 포트홀 상자(고리 측정) | 가짜 | ΔE AUC | ΔE 중앙값(포트홀/가짜) | 질감 비 AUC | 질감 비 중앙값(포트홀/가짜) | ΔL 중앙값 |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in sep.iterrows():
        lines.append(f"| {r['set']} | {r['pothole_boxes']}({r['pothole_ok']}) | {r['fake_ok']} | {r['dE_auc']:.3f} | {r['dE_median_pothole']:.2f} / "
                     f"{r['dE_median_fake']:.2f} | {r['tex_auc']:.3f} | {r['tex_median_pothole']:.2f} / {r['tex_median_fake']:.2f} | {r['dL_median_pothole']:+.2f} |")

    items = [it for it in mv.own_items() if any(g[4] == "pothole" for g in (evaluate.load_gt(str(it[2])) or []))]
    fresh = []
    for it in items:
        _, b_rows, _ = mv.analyze_image(it, "39", [])
        fresh += [r for r in b_rows if r["kind"] == "pothole"]
    fresh = pd.DataFrame(fresh)
    cols = ["image", "idx", "x", "y", "w", "h", "in_band", "L_in", "a_in", "b_in", "L_ring", "a_ring", "b_ring", "dL", "dE", "tex_ratio", "ring_n", "ok"]
    fresh[[c for c in cols if c in fresh.columns]].to_csv(OUT_DIR / "box_stats_39.csv", encoding="utf-8-sig", index=False)
    old = B[(B["set"] == "39") & (B["kind"] == "pothole")].set_index(["image", "idx"])
    same = sum(1 for _, r in fresh.iterrows() if (r["image"], r["idx"]) in old.index
               and (np.isclose(old.loc[(r["image"], r["idx"]), "dE"], r.get("dE", np.nan), equal_nan=True)))
    lines += ["", f"2. 39장 포트홀 정답 상자 {len(fresh)}개(사진 {len(items)}장), 지금 config로 다시 잰 값이 10/13 CSV와 같은 상자 {same}/{len(fresh)}",
              "| 사진 #상자 | 상자 안 L·a·b | 바깥 고리 L·a·b | ΔL | ΔE | 질감 비 |", "|---|---|---|---|---|---|"]
    for _, r in fresh.iterrows():
        if r["ok"]:
            lines.append(f"| {r['image'][:-4]} #{r['idx']} | {r['L_in']:.1f}·{r['a_in']:.1f}·{r['b_in']:.1f} | {r['L_ring']:.1f}·{r['a_ring']:.1f}·{r['b_ring']:.1f} "
                         f"| {r['dL']:+.1f} | {r['dE']:.2f} | {r['tex_ratio']:.2f} |")
        else:
            lines.append(f"| {r['image'][:-4]} #{r['idx']} | 고리 없음(주변 노면 마스크 픽셀 < {mv.RING_MIN_N}) | | | | |")

    boxes, regs = [], []
    for it in items:
        b, r = trace_image(it)
        boxes += b
        regs += r
    boxes, regs = pd.DataFrame(boxes), pd.DataFrame(regs)
    boxes.to_csv(OUT_DIR / "boxes_39.csv", encoding="utf-8-sig", index=False)
    regs.to_csv(OUT_DIR / "candidates_39.csv", encoding="utf-8-sig", index=False)
    lines += ["", "3. 39장 포트홀 정답 상자 안의 어두운 영역과 걸러지는 곳(자동 ROI)"]
    for mode in ("off", "on"):
        lines += ["", f"전처리 {'미적용' if mode == 'off' else '적용'}", "| 사진 #상자 | ROI 상단 | 상자 중 ROI 안 | 상자 안 노면 | 상자 안 어두운 픽셀 | 어두운 영역 수 | 결과 |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in boxes[boxes["mode"] == mode].iterrows():
            lines.append(f"| {r['image'][:-4]} #{r['idx']} | {r['top']:.1f} | {100 * r['box_in_roi']:.0f}% | "
                         + (f"{100 * r['road_in_box']:.0f}% | {100 * r['dark_in_box']:.0f}%" if r['box_in_roi'] > 0 else "- | -")
                         + f" | {r['n_dark_regions']} | {r['fates']} |")
        sub = regs[regs["mode"] == mode] if len(regs) else regs
        if len(sub):
            lines.append("영역별 결과 합계: " + ", ".join(f"{k} {v}" for k, v in sub["fate"].value_counts().items()))
            pot = sub[sub["shape"] == "pothole"]
            if len(pot):
                lines.append(f"포트홀 모양 영역 {len(pot)}개의 포트홀 점수: 최대 {pot['score_pothole'].max():.3f}, 중앙값 {pot['score_pothole'].median():.3f} "
                             f"(임계값 {C.SCORE_TAU_POTHOLE})")
            none = sub[sub["shape"] == "none"]
            if len(none):
                lines.append(f"모양 탈락 영역 {len(none)}개: 비원형도 중앙값 {none['circ_inv'].median():.2f}(포트홀 ≤ {C.POT_MAX_CIRC_INV}, 균열 ≥ {C.CRACK_MIN_CIRC_INV}), "
                             f"solidity 중앙값 {none['solidity'].median():.2f}(≥ {C.POT_MIN_SOLIDITY})")
    (OUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
