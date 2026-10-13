"""(파트 C, 10/14 2번) 포트홀 분류·점수를 다시 정하려고 개발셋 어두운 영역의 특징 분포를 잰다(지금 config = 10/14 노면 A2rA3B1B2).
사용법(맨 위 폴더에서): python tools/pothole_feat.py [--jobs 12] [--set 39] → results/pothole_feat/regions_<셋>.csv, auc.csv, summary.txt
사진마다 pipeline.run_pipeline(전처리 미적용)의 ROI·노면 마스크·proc로 detect.extract_dark_regions를 불러 모든 어두운 영역(분류 전)을 모으고,
영역마다 모양(넓이·비원형도·solidity·두께·폭·길쭉함·채움), 대비(contrast·darkness), 주변 노면과의 차이(영역 안 − 바깥 고리:
흑백 ΔL, CIE Lab ΔL·ΔE, 질감 비)를 잰다. 바깥 고리 = 영역을 measure_v5.ring_width(bbox)만큼 넓힌 띠 중 노면 마스크 안·영역 밖.
이름표: pothole = 영역 bbox 중심이 포트홀 정답 상자 안, crack = 균열 정답 상자 안(포트홀 아님), none = 어느 정답 상자 밖(가짜).
노면 밖 후보 제거(영역의 노면 안 비율 < ROAD_KEEP_RATIO_MIN)에 걸리는 영역은 kept_road = False로 표시한다.
AUC = 포트홀 영역의 값이 가짜(none) 영역보다 클 확률(0.5 = 못 가름, 0.5보다 작으면 작을수록 포트홀). 39장은 결과를 보는 데만 쓴다."""
import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src import detect, evaluate, io_utils, pipeline  # noqa: E402
from tools import measure_v5 as mv  # noqa: E402
from tools import evaluate_train as et  # noqa: E402

OUT_DIR = Path("results/pothole_feat")
TEX_K = 7                         # px, 질감 = 흑백 7×7 지역 표준편차(측정 B와 같음)
FEATURES = ["area", "circ_inv", "solidity", "thickness", "dt_width", "elong", "fill", "contrast", "darkness",
            "dL_gray", "dL", "dE", "tex_ratio", "score_pothole", "score_crack"]
GRAY_FEATURES = ["area", "circ_inv", "solidity", "thickness", "dt_width", "elong", "fill", "contrast", "darkness", "dL_gray", "tex_ratio"]


def region_rows(item, set_name):
    """사진 1장의 어두운 영역마다 특징·이름표 행."""
    name, path, gt_path, top0, bottom = item
    bgr = io_utils.load_image(path)
    out = pipeline.run_pipeline(bgr, top0, bottom, False)
    y0, road, proc = out["y0"], out["road"] > 0, out["proc"]
    roi = io_utils.crop_roi(out["small"], out["top"], bottom)[0]
    regions, _ = detect.extract_dark_regions(proc, out["road"] if C.USE_ROAD_MASK else None)
    gt = evaluate.load_gt(str(gt_path)) or []
    lab = cv.cvtColor(roi.astype(np.float32) / 255.0, cv.COLOR_BGR2LAB)
    tex = pipeline.local_std(proc, TEX_K)
    g = proc.astype(np.float32)
    h, w = road.shape
    rows = []
    for r in regions:
        x, y, bw, bh = r["bbox"]
        cx, cy = x + bw / 2, y + bh / 2 + y0
        inside = lambda k: any(gx <= cx <= gx + gw and gy <= cy <= gy + gh for gx, gy, gw, gh, kk in gt if kk == k)   # noqa: E731
        label = "pothole" if inside("pothole") else ("crack" if inside("crack") else "none")
        d = mv.ring_width(bw, bh)
        X0, Y0, X1, Y1 = max(0, x - d), max(0, y - d), min(w, x + bw + d), min(h, y + bh + d)
        reg = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
        cv.drawContours(reg, [r["contour"]], -1, 1, -1, offset=(-X0, -Y0))
        reg = reg > 0
        ring = (cv.dilate(reg.astype(np.uint8), cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))) > 0) & ~reg & road[Y0:Y1, X0:X1]
        row = {"set": set_name, "image": name, "source": et.source_of(name), "label": label, "x": x, "y": y + y0, "w": bw, "h": bh,
               "kept_road": pipeline.road_ratio(r, out["road"]) >= C.ROAD_KEEP_RATIO_MIN, "ring_n": int(ring.sum()),
               **{k: r.get(k, np.nan) for k in ["area", "circ_inv", "solidity", "thickness", "dt_width", "elong", "fill", "contrast",
                                                 "darkness", "score_pothole", "score_crack"]}}
        if ring.sum() >= mv.RING_MIN_N and reg.any():
            li, lo = lab[Y0:Y1, X0:X1][reg].mean(0), lab[Y0:Y1, X0:X1][ring].mean(0)
            to = float(np.median(tex[Y0:Y1, X0:X1][ring]))
            row.update({"dL_gray": float(g[Y0:Y1, X0:X1][reg].mean() - g[Y0:Y1, X0:X1][ring].mean()),
                        "dL": float(li[0] - lo[0]), "dE": float(np.linalg.norm(li - lo)),
                        "tex_ratio": float(np.median(tex[Y0:Y1, X0:X1][reg])) / to if to > 0 else np.nan})
        rows.append(row)
    n_pot = sum(1 for gg in gt if gg[4] == "pothole")
    return rows, {"image": name, "set": set_name, "pothole_gt": n_pot, "y0": y0}


def _one(pair):
    return region_rows(pair[1], pair[0])


def auc_table(df):
    """특징마다 포트홀 영역 vs 가짜 영역 AUC. 두 묶음: 모든 영역 / 균열 모양이 아닌 영역(비원형도 < CRACK_MIN_CIRC_INV). 노면 밖 제거에 걸리는 영역은 뺌."""
    d = df[df["kept_road"]]
    out = []
    for group, sub in (("all", d), ("not_crack_shape", d[d["circ_inv"] < C.CRACK_MIN_CIRC_INV])):
        pos, neg = sub[sub["label"] == "pothole"], sub[sub["label"] == "none"]
        for f in FEATURES:
            p, n = pos[f].dropna(), neg[f].dropna()
            a = mv.auc(p, n) if len(p) and len(n) else np.nan
            out.append({"group": group, "feature": f, "n_pothole": len(p), "n_fake": len(n), "auc": a, "sep": abs(a - 0.5) * 2 if a == a else np.nan,
                        "median_pothole": p.median(), "median_fake": n.median(), "gray_only": f in GRAY_FEATURES})
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--set", default="dev", choices=["dev", "39"])
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = [(args.set, it) for it in (mv.dev_items() if args.set == "dev" else mv.own_items())]
    with Pool(args.jobs, initializer=cv.setNumThreads, initargs=(1,)) as pool:
        res = list(pool.imap(_one, items))
    df = pd.DataFrame([r for rows, _ in res for r in rows])
    imgs = pd.DataFrame([m for _, m in res])
    df.to_csv(OUT_DIR / f"regions_{args.set}.csv", encoding="utf-8-sig", index=False)
    imgs.to_csv(OUT_DIR / f"images_{args.set}.csv", encoding="utf-8-sig", index=False)
    if args.set == "dev":
        a = auc_table(df)
        a.to_csv(OUT_DIR / "auc.csv", encoding="utf-8-sig", index=False)
        print(a.round(3).to_string())
    print(df.groupby(["label", "kept_road"]).size())
    return 0


if __name__ == "__main__":
    sys.exit(main())
