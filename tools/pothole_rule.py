"""(파트 C, 10/14 2번) 포트홀 분류 규칙·점수식을 개발셋 어두운 영역(tools/pothole_feat.py 결과)으로 정하고, 포트홀 운영점과 균열 영향을 본다.
사용법: python tools/pothole_rule.py → results/pothole_feat/rule_summary.txt·weights.csv·curve_pothole.csv
- 점수식: 흑백으로 잴 수 있는 특징(detect가 흑백만 받음)의 표준화 값 가중합에 로지스틱(포트홀일 확률). 가중치는 개발셋 영역
  (포트홀 정답 안 = 1, 균열 정답 안·정답 밖 = 0)에 L2 정칙화 로지스틱 회귀로 맞춘다(numpy, 시드 없음·결정적).
  과적합을 보려고 사진 이름 crc32 짝·홀 2겹 교차 검증 점수(맞추지 않은 절반의 점수)로도 같은 표를 낸다.
- 분류: 노면 밖 제거를 통과한 모든 어두운 영역에 포트홀 점수를 매기고, 점수 ≥ τ_p면 포트홀(균열 후보에서는 빠짐),
  아니면 지금 균열 규칙(비원형도 ≥ CRACK_MIN_CIRC_INV, 균열 점수 ≥ τ_c). 지금 버려지는 '애매' 영역(비원형도 2.5~4)도 포트홀 후보가 된다.
- 포트홀 운영점: 영상당 포트홀 오검출 ≤ 0.5에서 포트홀 재현율(정답 상자 중 안에 포트홀 후보 중심이 든 비율) 최대.
- 균열 운영점: 영상당 균열 오검출 ≤ 1.5에서 균열 재현율 최대. 둘 다 중심 기준(evaluate.count_center_hits와 같은 뜻)."""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src import evaluate  # noqa: E402
from tools import measure_v5 as mv  # noqa: E402

OUT = Path("results/pothole_feat")
FEATS = ["log_area", "log_circ", "solidity", "fill", "thickness", "log_tex", "darkness", "contrast"]
L2 = 1.0                 # 정칙화 세기(표준화한 특징 기준)
POT_FP_MAX = 0.5         # 영상당 포트홀 오검출 상한(사용자 지정 운영점)
CRACK_FP_MAX = 1.5       # 영상당 균열 오검출 상한(사용자 지정 운영점)


def features(df, med_tex=None):
    X = pd.DataFrame({"log_area": np.log(df["area"]), "log_circ": np.log(df["circ_inv"]), "solidity": df["solidity"], "fill": df["fill"],
                      "thickness": df["thickness"], "log_tex": np.log(df["tex_ratio"].clip(lower=1e-3)), "darkness": df["darkness"],
                      "contrast": df["contrast"]})
    med_tex = X["log_tex"].median() if med_tex is None else med_tex
    X["log_tex"] = X["log_tex"].fillna(med_tex)                # 바깥 고리가 모자란 영역(질감 비 없음)은 중앙값
    return X, med_tex


def fit_logistic(X, y, l2=L2, iters=3000, lr=0.5):
    """표준화 → 경사 하강 로지스틱 회귀(L2). (평균, 표준편차, 가중치, 절편)."""
    mu, sd = X.mean(0), X.std(0).replace(0, 1)
    Z = ((X - mu) / sd).to_numpy()
    w, b = np.zeros(Z.shape[1]), 0.0
    pos_w = (len(y) - y.sum()) / max(1, y.sum())               # 양성이 아주 적어 무게를 맞춤
    sw = np.where(y == 1, pos_w, 1.0)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = (p - y) * sw
        w -= lr * (Z.T @ g / sw.sum() + l2 * w / len(y))
        b -= lr * g.sum() / sw.sum()
    return mu, sd, w, b


def score(X, model):
    mu, sd, w, b = model
    return 1 / (1 + np.exp(-(((X - mu) / sd).to_numpy() @ w + b)))


def gt_table():
    """개발셋 사진마다 (균열 상자 list, 포트홀 상자 list), 사진 수."""
    gts = {}
    for name, _, gt_path, _, _ in mv.dev_items():
        gt = evaluate.load_gt(str(gt_path)) or []
        gts[name] = ([g[:4] for g in gt if g[4] == "crack"], [g[:4] for g in gt if g[4] == "pothole"])
    return gts


def hit_table(d, gts):
    """영역마다 같은 종류 정답 상자 안 여부(중심), 정답 상자마다 중심이 든 영역 번호. 한 번만 계산해 τ마다 다시 쓴다."""
    cx, cy = (d["x"] + d["w"] / 2).to_numpy(), (d["y"] + d["h"] / 2).to_numpy()
    names = d["image"].to_numpy()
    idx_of = {n: np.flatnonzero(names == n) for n in gts}
    hit = {k: np.zeros(len(d), bool) for k in ("crack", "pothole")}
    boxes = {"crack": [], "pothole": []}
    for name, pair in gts.items():
        ii = idx_of[name]
        for kind, bl in zip(("crack", "pothole"), pair):
            for gx, gy, gw, gh in bl:
                m = ii[(cx[ii] >= gx) & (cx[ii] <= gx + gw) & (cy[ii] >= gy) & (cy[ii] <= gy + gh)]
                hit[kind][m] = True
                boxes[kind].append(m)
    return hit, boxes, len(gts)


def evaluate_rule(table, pot_mask, crack_mask):
    """포트홀·균열 후보(bool)로 종류별 재현율·영상당 오검출·정밀도·F1(중심 기준)."""
    hit, boxes, n = table
    res = {}
    for kind, mask in (("pothole", pot_mask), ("crack", crack_mask)):
        tp, fp = int((hit[kind] & mask).sum()), int((~hit[kind] & mask).sum())
        found, total = sum(bool(mask[m].any()) for m in boxes[kind]), len(boxes[kind])
        rec, prec = found / max(1, total), tp / (tp + fp) if tp + fp else np.nan
        res[kind] = {"recall": rec, "precision": prec, "f1": 2 * prec * rec / (prec + rec) if prec == prec and prec + rec > 0 else np.nan,
                     "fp_per_image": fp / n, "found": found, "total": total, "candidates": tp + fp}
    return res


def operating_point(table, s_pot, crack_ok, taus):
    """τ_p마다 포트홀(점수 ≥ τ_p) 지표. 포트홀 오검출 ≤ POT_FP_MAX에서 재현율 최대인 τ_p(같으면 오검출이 적은 쪽)."""
    rows = []
    for t in taus:
        pot = s_pot >= t
        r = evaluate_rule(table, pot, crack_ok & ~pot)
        rows.append({"tau_p": t, **{f"pot_{k}": v for k, v in r["pothole"].items()}, **{f"crack_{k}": v for k, v in r["crack"].items()}})
    curve = pd.DataFrame(rows)
    ok = curve[curve["pot_fp_per_image"] <= POT_FP_MAX]
    best = ok.sort_values(["pot_recall", "pot_fp_per_image"], ascending=[False, True]).iloc[0] if len(ok) else None
    return curve, best


def main():
    df = pd.read_csv(OUT / "regions_dev.csv", encoding="utf-8-sig")
    d = df[df["kept_road"]].reset_index(drop=True)
    gts = gt_table()
    X, med_tex = features(d)
    y = (d["label"] == "pothole").to_numpy().astype(float)
    model = fit_logistic(X, y)
    s_full = score(X, model)
    fold = d["image"].map(lambda n: zlib.crc32(n.encode()) % 2).to_numpy()
    s_cv = np.zeros(len(d))
    for k in (0, 1):
        m = fit_logistic(X[fold != k], y[fold != k])
        s_cv[fold == k] = score(X[fold == k], m)
    lines = [f"개발셋 어두운 영역(노면 안) {len(d)}개: 포트홀 정답 안 {int(y.sum())}, 균열 정답 안 {(d['label'] == 'crack').sum()}, 정답 밖 {(d['label'] == 'none').sum()}",
             f"포트홀 점수 AUC(포트홀 vs 나머지): 맞춘 그대로 {mv.auc(s_full[y == 1], s_full[y == 0]):.3f}, 2겹 교차 검증 {mv.auc(s_cv[y == 1], s_cv[y == 0]):.3f}, "
             f"지금 포트홀 점수 {mv.auc(d['score_pothole'][y == 1], d['score_pothole'][y == 0]):.3f}"]
    mu, sd, w, b = model
    wt = pd.DataFrame({"feature": FEATS, "weight_std": w, "mean": mu.values, "std": sd.values})
    wt.loc[len(wt)] = ["(절편)", b, np.nan, np.nan]
    wt.to_csv(OUT / "weights.csv", encoding="utf-8-sig", index=False)
    lines += ["가중치(표준화 특징 기준, +면 포트홀 쪽): " + ", ".join(f"{f} {v:+.2f}" for f, v in zip(FEATS, w)) + f", 절편 {b:+.2f}"]

    crack_ok = ((d["circ_inv"] >= C.CRACK_MIN_CIRC_INV) & (d["score_crack"] >= C.SCORE_TAU_CRACK)).to_numpy()
    now_pot = ((d["circ_inv"] <= C.POT_MAX_CIRC_INV) & (d["solidity"] >= C.POT_MIN_SOLIDITY)).to_numpy()
    table = hit_table(d, gts)
    now = evaluate_rule(table, now_pot & (d["score_pothole"] >= C.SCORE_TAU_POTHOLE).to_numpy(), crack_ok)
    lines += ["", f"지금 규칙(포트홀 비원형도 ≤ {C.POT_MAX_CIRC_INV}·solidity ≥ {C.POT_MIN_SOLIDITY}·점수 ≥ {C.SCORE_TAU_POTHOLE}, 균열 τ {C.SCORE_TAU_CRACK}):"]
    lines += [f"  {k}: 재현율 {v['recall']:.1%}({v['found']}/{v['total']}), 정밀도 {v['precision']:.1%}, F1 {v['f1']:.3f}, 오검출/장 {v['fp_per_image']:.2f}, 후보 {v['candidates']}"
              for k, v in now.items()]
    # 지금 분류에서 포트홀 점수 임계값만 운영점으로(포트홀 오검출 ≤ 0.5)
    sp = np.where(now_pot, d["score_pothole"].to_numpy(), -np.inf)
    taus = np.unique(np.quantile(sp[np.isfinite(sp)], np.linspace(0, 1, 201)))
    c0, b0 = operating_point(table, sp, crack_ok, np.r_[taus, np.inf])
    lines += [f"지금 분류 + 포트홀 점수 임계값만 포트홀 운영점으로: τ_p {b0['tau_p']:.4f}, 포트홀 재현율 {b0['pot_recall']:.1%}, 정밀도 {b0['pot_precision']:.1%}, "
              f"F1 {b0['pot_f1']:.3f}, 포트홀 오검출/장 {b0['pot_fp_per_image']:.2f}"]
    for tag, s in (("새 점수(맞춘 그대로)", s_full), ("새 점수(2겹 교차 검증)", s_cv)):
        taus = np.unique(np.quantile(s, np.linspace(0.5, 1, 251)))
        curve, best = operating_point(table, s, crack_ok, np.r_[taus, np.inf])
        if tag.endswith("그대로)"):
            curve.to_csv(OUT / "curve_pothole.csv", encoding="utf-8-sig", index=False)
            final = best
        lines += [f"{tag} 포트홀 운영점: τ_p {best['tau_p']:.4f}, 포트홀 재현율 {best['pot_recall']:.1%}({int(best['pot_found'])}/{int(best['pot_total'])}), "
                  f"정밀도 {best['pot_precision']:.1%}, F1 {best['pot_f1']:.3f}, 포트홀 오검출/장 {best['pot_fp_per_image']:.2f}, 후보 {int(best['pot_candidates'])}; "
                  f"그때 균열 재현율 {best['crack_recall']:.1%}, 균열 오검출/장 {best['crack_fp_per_image']:.2f}"]
    (OUT / "rule_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return final


if __name__ == "__main__":
    main()
