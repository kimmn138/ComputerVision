"""(파트 C, 10/14 3번 단계 1) 포트홀 색 차이(ΔE) 경로의 재현율 상한을 구현 전에 잰다(src는 바꾸지 않음, 지금 config = 노면 A2rA3B1B2).
사용법(맨 위 폴더에서): python tools/pothole_color.py --jobs 12 → results/pothole_color/
1. 주변 노면 대표 색: ROI 컬러(밝은 표시 보정·전처리 전)를 노면 마스크와 같은 σ로 흐린 CIE Lab(L 0~100)에서, CELL px 칸마다 노면 픽셀
   Lab 중앙값을 내고(노면이 CELL_ROAD_MIN 미만인 칸은 비움), 반경 = ROI 높이 × DARK_BG_FRAC / 2(어두운 영역 배경 창과 같은 크기)의
   이웃 칸 중앙값을 그 칸의 대표 색으로 한다 → 원래 크기로 늘림.
2. ΔE 지도 = |Lab − 대표 색|(CIE76), 부호 = L − 대표 L. 문턱은 개발셋 노면 픽셀(정답 상자 밖) ΔE 분포의 백분위:
   강 = Q_HI(99백분위), 약 = Q_LO(95백분위). 밝은 쪽(ΔL > 0)·어두운 쪽(ΔL < 0)을 따로, 약 영역 가운데 강 픽셀을 품은 것만(이중 문턱).
3. 닫힘(DARK_CLOSE_KSIZE) 뒤 넓이 DARK_MIN_AREA 미만, 노면 안 비율 ROAD_KEEP_RATIO_MIN 미만 영역을 버린다.
4. 차선·노면 표시 제외(밝은 쪽만): 가늘고 김(긴 변 ÷ 짧은 변 ≥ ROAD_LANE_ELONG) 또는 페인트 색(평균 채도 < PAINT_CHROMA이고 ΔL > PAINT_DL).
재는 것: 개발셋 ROI 안 포트홀 정답 상자 중 ΔE 영역 중심이 든 상자 비율(어두운 영역 = results/pothole_feat/regions_dev.csv와 합침),
정답 밖 ΔE 영역 수(영상당), 39장 포트홀 12개 각각. 시험셋 사진은 열지 않는다."""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src import evaluate, io_utils, pipeline  # noqa: E402
from tools import measure_v5 as mv  # noqa: E402

OUT = Path("results/pothole_color")
CELL = 16                 # px, 대표 색 칸
CELL_ROAD_MIN = 0.3       # 비율, 칸 안 노면 픽셀이 이보다 적으면 그 칸 중앙값을 쓰지 않음
Q_HI, Q_LO = 0.99, 0.95   # 노면 ΔE 분포 백분위(강·약 문턱)
SAMPLE_STEP = 37          # 문턱 분포를 잴 때 노면 픽셀 표본 간격
PAINT_CHROMA = 12         # CIE a·b 거리(L 0~100 단위), 이보다 무채색이고
PAINT_DL = 15             # ΔL이 이보다 밝으면 페인트(흰 표시)로 봄


def reference_lab(lab, road):
    """칸 중앙값 → 이웃 칸 중앙값으로 주변 노면 대표 색(H, W, 3)."""
    h, w = road.shape
    gh, gw = (h + CELL - 1) // CELL, (w + CELL - 1) // CELL
    cell = np.full((gh, gw, 3), np.nan, np.float32)
    for i in range(gh):
        for j in range(gw):
            r = road[i * CELL:(i + 1) * CELL, j * CELL:(j + 1) * CELL]
            if r.size and r.mean() >= CELL_ROAD_MIN:
                cell[i, j] = np.median(lab[i * CELL:(i + 1) * CELL, j * CELL:(j + 1) * CELL][r], axis=0)
    rad = max(1, int(round(h * C.DARK_BG_FRAC / 2 / CELL)))
    ref = np.empty_like(cell)
    whole = np.median(lab[road], axis=0) if road.any() else np.median(lab.reshape(-1, 3), axis=0)
    for i in range(gh):
        for j in range(gw):
            nb = cell[max(0, i - rad):i + rad + 1, max(0, j - rad):j + rad + 1].reshape(-1, 3)
            nb = nb[~np.isnan(nb[:, 0])]
            ref[i, j] = np.median(nb, axis=0) if len(nb) else whole
    return cv.resize(ref, (w, h), interpolation=cv.INTER_LINEAR)


def de_maps(roi, road):
    lab = cv.cvtColor(cv.GaussianBlur(roi, (0, 0), C.ROAD_BLUR_SIGMA).astype(np.float32) / 255.0, cv.COLOR_BGR2LAB)
    ref = reference_lab(lab, road)
    diff = lab - ref
    return lab, np.linalg.norm(diff, axis=2), diff[:, :, 0]


def regions_of(lab, de, dl, road, thr):
    """밝은 쪽·어두운 쪽 이중 문턱 영역 → [(쪽, 영역 bool 전체 크기, bbox, 특징 dict)]."""
    out = []
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (C.DARK_CLOSE_KSIZE, C.DARK_CLOSE_KSIZE))
    chroma = np.hypot(lab[:, :, 1], lab[:, :, 2])
    for side, sign in (("bright", dl > 0), ("dark", dl < 0)):
        weak = ((de >= thr["lo"]) & sign).astype(np.uint8)
        strong = (de >= thr["hi"]) & sign
        n, lab_w = cv.connectedComponents(weak, connectivity=8)
        keep = np.zeros(n, bool)
        keep[np.unique(lab_w[strong])] = True
        keep[0] = False
        m = cv.morphologyEx(keep[lab_w].astype(np.uint8), cv.MORPH_CLOSE, k)
        n, labels, stats, _ = cv.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            area = stats[i, cv.CC_STAT_AREA]
            if area < C.DARK_MIN_AREA:
                continue
            x, y, bw, bh = stats[i, :4]
            reg = labels[y:y + bh, x:x + bw] == i
            if road[y:y + bh, x:x + bw][reg].mean() < C.ROAD_KEEP_RATIO_MIN:
                continue
            pts = np.column_stack(np.nonzero(reg)[::-1]).astype(np.float32)
            (_, _), (rw, rh), _ = cv.minAreaRect(pts)
            elong = max(rw, rh) / max(1.0, min(rw, rh))
            hull = cv.contourArea(cv.convexHull(pts))
            f = {"side": side, "area": int(area), "x": int(x), "y": int(y), "w": int(bw), "h": int(bh), "elong": float(elong),
                 "solidity": float(area / hull) if hull > 0 else 1.0, "dE": float(de[y:y + bh, x:x + bw][reg].mean()),
                 "dL": float(dl[y:y + bh, x:x + bw][reg].mean()), "chroma": float(chroma[y:y + bh, x:x + bw][reg].mean())}
            f["lane_or_paint"] = bool(side == "bright" and (elong >= C.ROAD_LANE_ELONG or (f["chroma"] < PAINT_CHROMA and f["dL"] > PAINT_DL)))
            out.append(f)
    return out


def prepare(item):
    """사진 1장의 ROI·노면·y0·ΔE 지도."""
    name, path, gt_path, top0, bottom = item
    bgr = io_utils.load_image(path)
    o = pipeline.run_pipeline(bgr, top0, bottom, False)
    roi = io_utils.crop_roi(o["small"], o["top"], bottom)[0]
    road = o["road"] > 0
    lab, de, dl = de_maps(roi, road)
    gt = evaluate.load_gt(str(gt_path)) or []
    return name, o["y0"], road, lab, de, dl, gt


def sample_one(item):
    """문턱용: 정답 상자 밖 노면 픽셀의 ΔE 표본."""
    name, y0, road, _, de, _, gt = prepare(item)
    m = road.copy()
    for x, y, w, h, _ in gt:
        m[max(0, y - y0):max(0, y + h - y0), x:x + w] = False
    return de[m][::SAMPLE_STEP]


def measure_one(args):
    (set_name, item), thr = args
    name, y0, road, lab, de, dl, gt = prepare(item)
    rows = []
    for f in regions_of(lab, de, dl, road, thr):
        cx, cy = f["x"] + f["w"] / 2, f["y"] + f["h"] / 2 + y0
        inside = lambda k: [i for i, (gx, gy, gw, gh, kk) in enumerate(gt) if kk == k and gx <= cx <= gx + gw and gy <= cy <= gy + gh]  # noqa: E731
        pot, cr = inside("pothole"), inside("crack")
        rows.append({"set": set_name, "image": name, **f, "y": f["y"] + y0, "label": "pothole" if pot else ("crack" if cr else "none"),
                     "pothole_idx": pot[0] if pot else -1})
    boxes = [{"set": set_name, "image": name, "idx": i, "in_roi": y + h / 2 >= y0} for i, (x, y, w, h, k) in enumerate(gt) if k == "pothole"]
    return rows, boxes


def dark_hits(set_name):
    """어두운 영역(노면 안) 중심이 든 포트홀 상자 {(사진, 번호)} — results/pothole_feat/regions_<셋>.csv(지금 config)."""
    r = pd.read_csv(Path("results/pothole_feat") / f"regions_{set_name}.csv", encoding="utf-8-sig")
    r = r[r["kept_road"]]
    items = mv.dev_items() if set_name == "dev" else mv.own_items()
    hits = set()
    for name, _, gt_path, _, _ in items:
        sub = r[r["image"] == name]
        cx, cy = sub["x"] + sub["w"] / 2, sub["y"] + sub["h"] / 2
        for i, (gx, gy, gw, gh, k) in enumerate(evaluate.load_gt(str(gt_path)) or []):
            if k == "pothole" and ((cx >= gx) & (cx <= gx + gw) & (cy >= gy) & (cy <= gy + gh)).any():
                hits.add((name, i))
    return hits


def box_pixels(item):
    """--sweep: ROI 안 포트홀 정답 상자마다 상자 안 픽셀 ΔE(1차원)와 그 사진의 이름."""
    name, y0, road, lab, de, dl, gt = prepare(item)
    return [(name, i, de[max(0, y - y0):max(0, y + h - y0), x:x + w].ravel()) for i, (x, y, w, h, k) in enumerate(gt)
            if k == "pothole" and y + h / 2 >= y0]


def sweep(pool, dev, thr):
    """문턱을 낮춰 가며(노면 ΔE 백분위) ΔE 영역 상한과 정답 밖 영역 수. 포트홀 상자 안 픽셀 ΔE 분포도."""
    bx = [b for r in pool.map(box_pixels, [it for _, it in dev]) for b in r]
    allpx = np.concatenate([b[2] for b in bx])
    lines = [f"개발셋 ROI 안 포트홀 상자 {len(bx)}개 안 픽셀 ΔE: 중앙값 {np.median(allpx):.2f}, 90백분위 {np.quantile(allpx, 0.9):.2f}, "
             f"99백분위 {np.quantile(allpx, 0.99):.2f}. 상자별 최대 ΔE 중앙값 {np.median([b[2].max() for b in bx if b[2].size]):.2f}"]
    samples = np.concatenate(pool.map(sample_one, [it for _, it in dev]))
    lines += ["| 약·강 백분위 | 약·강 ΔE | ΔE 영역이 든 상자 | 정답 밖 ΔE 영역/장 |", "|---|---|---|---|"]
    for lo, hi in ((0.95, 0.99), (0.90, 0.97), (0.80, 0.95), (0.70, 0.90)):
        t = {"lo": float(np.quantile(samples, lo)), "hi": float(np.quantile(samples, hi))}
        res = pool.map(measure_one, [(p, t) for p in dev])
        regs = pd.DataFrame([r for rows, _ in res for r in rows])
        u = regs[~regs["lane_or_paint"]] if len(regs) else regs
        hit = len(set(zip(u[u["label"] == "pothole"]["image"], u[u["label"] == "pothole"]["pothole_idx"]))) if len(u) else 0
        lines.append(f"| {lo:.0%}·{hi:.0%} | {t['lo']:.2f}·{t['hi']:.2f} | {hit}/{len(bx)} | {(u['label'] == 'none').sum() / len(dev):.2f} |")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--sweep", action="store_true", help="문턱 백분위를 낮춰 가며 상한(개발셋)")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.sweep:
        with Pool(args.jobs, initializer=cv.setNumThreads, initargs=(1,)) as pool:
            lines = sweep(pool, [("dev", it) for it in mv.dev_items()], None)
        (OUT / "sweep.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 0
    dev = [("dev", it) for it in mv.dev_items()]
    own = [("39", it) for it in mv.own_items()]
    with Pool(args.jobs, initializer=cv.setNumThreads, initargs=(1,)) as pool:
        samples = np.concatenate(pool.map(sample_one, [it for _, it in dev]))
        thr = {"hi": float(np.quantile(samples, Q_HI)), "lo": float(np.quantile(samples, Q_LO)), "median": float(np.median(samples)),
               "n_samples": int(len(samples))}
        res = pool.map(measure_one, [(p, thr) for p in dev + own])
    (OUT / "threshold.json").write_text(json.dumps(thr, indent=1), encoding="utf-8")
    regs = pd.DataFrame([r for rows, _ in res for r in rows])
    boxes = pd.DataFrame([b for _, bl in res for b in bl])
    regs.to_csv(OUT / "regions.csv", encoding="utf-8-sig", index=False)
    lines = [f"ΔE 문턱(개발셋 정답 밖 노면 픽셀 {thr['n_samples']}개 표본): 중앙값 {thr['median']:.2f}, 약 {Q_LO:.0%} {thr['lo']:.2f}, 강 {Q_HI:.0%} {thr['hi']:.2f}"]
    use = regs[~regs["lane_or_paint"]]
    for set_name in ("dev", "39"):
        b = boxes[(boxes["set"] == set_name) & boxes["in_roi"]]
        key = set(zip(b["image"], b["idx"]))
        dh = dark_hits(set_name) & key
        u = use[use["set"] == set_name]
        eh = {k: set(zip(u[(u["label"] == "pothole") & (u["side"] == s if s else True)]["image"],
                          u[(u["label"] == "pothole") & (u["side"] == s if s else True)]["pothole_idx"])) & key for k, s in
              (("all", None), ("bright", "bright"), ("dark", "dark"))}
        n_img = len(dev) if set_name == "dev" else len(own)
        none = u[u["label"] == "none"]
        lines += ["", f"[{'개발셋' if set_name == 'dev' else '39장'}] ROI 안 포트홀 정답 {len(key)}개",
                  f"| 영역 | 상자가 든 수 | 비율 |", "|---|---|---|",
                  f"| 어두운 영역(지금 검출기) | {len(dh)} | {len(dh) / max(1, len(key)):.1%} |",
                  f"| ΔE 영역 전체 | {len(eh['all'])} | {len(eh['all']) / max(1, len(key)):.1%} |",
                  f"| ΔE 밝은 쪽 | {len(eh['bright'])} | {len(eh['bright']) / max(1, len(key)):.1%} |",
                  f"| ΔE 어두운 쪽 | {len(eh['dark'])} | {len(eh['dark']) / max(1, len(key)):.1%} |",
                  f"| 합침(어두운 영역 ∪ ΔE) | {len(dh | eh['all'])} | {len(dh | eh['all']) / max(1, len(key)):.1%} |",
                  f"정답 밖 ΔE 영역: {len(none)}개, 영상당 {len(none) / n_img:.2f}(밝은 쪽 {(none['side'] == 'bright').sum() / n_img:.2f}·"
                  f"어두운 쪽 {(none['side'] == 'dark').sum() / n_img:.2f}), 차선·페인트로 뺀 영역 {int(regs[regs['set'] == set_name]['lane_or_paint'].sum())}개"]
        if set_name == "39":
            lines += ["| 사진 #상자 | 어두운 영역 | ΔE 영역(밝은·어두운) | ΔE 영역 최대 ΔE·평균 ΔL |", "|---|---|---|---|"]
            for _, r in boxes[boxes["set"] == "39"].iterrows():
                s = u[(u["image"] == r["image"]) & (u["pothole_idx"] == r["idx"])]
                lines.append(f"| {r['image'][:-4]} #{r['idx']}{'' if r['in_roi'] else ' (ROI 위)'} | {'○' if (r['image'], r['idx']) in dh else '×'} | "
                             f"{len(s)}({(s['side'] == 'bright').sum()}·{(s['side'] == 'dark').sum()}) | "
                             + (f"{s['dE'].max():.1f} · {s.loc[s['dE'].idxmax(), 'dL']:+.1f}" if len(s) else "-") + " |")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
