"""(파트 C) 보고서용 점수 임계값별 성능 표(균열·포트홀)와 그래프를 운영점 실행의 임계값별 성능 곡선(tau_curve.csv)에서 다시 만든다.
사용법: 맨 위 폴더에서  python tools/report_tables.py [--run e5_supp_dev] [--preprocess off]
최종 실행 뒤에는 --run에 그 실행 폴더를 주고 다시 돌린다. 실험은 돌리지 않고 결과 CSV만 읽는다.
결과: results/report/table18_tau_crack.csv, table19_tau_pothole.csv, fig_tau_tables.png,
      table20_threshold_basis.md(주요 임계값 근거 표: docs/part_C.md의 BASIS_HEADING 절을 그대로 옮김. 원본은 part_C 하나)
- 균열 표: 포트홀 임계값을 운영점 값으로 고정하고 균열 임계값만 바꾼다. 포트홀 표는 그 반대.
- 운영점 = 그 실행의 tau_points.csv 'config' 행(실행에 쓴 config 임계값)과 같은 후보를 남기는 격자 점(격자 값 ≤ config 값 중 가장 큰 값,
  반올림 오차 ROUND_TOL 허용). 표의 운영점 행이 tau_points.csv config 행과 같은지 확인해 다르면 멈춘다.
- 후보 정밀도 = 검출 성공 후보 ÷ 전체 후보. 검출 성공 = 후보 상자 중심이 같은 종류 정답 상자 안(evaluate_train의 ctr 판정).
  한 정답 상자 안의 후보 여러 개는 모두 성공으로 센다. 종류별 정밀도는 다른 종류 임계값을 +∞(후보 0개)로 둔 격자 행에서 낸다.
- 재현율 = 그 종류 정답 상자 중 안에 후보 중심이 든 비율, 우연 수준을 보정한 재현율 = 재현율 − 우연 대조군 재현율."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

REPORT_DIR = Path("results/report")
BASIS_SOURCE = Path("docs/part_C.md")
BASIS_HEADING = "### 주요 임계값 근거 표"   # 이 제목부터 다음 '### '/'## ' 제목 전까지를 옮긴다
ROUND_TOL = 1.5e-6              # CSV 반올림(소수 여섯째 자리, 최대 0.5e-6)과 config 내림(최대 1e-6)의 합
CRACK_Q = (0.0, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.01)    # 균열 임계값 대표 분위(+ 운영점 행). 0 = 임계값 없음, 1.01 = 후보 모두 지움
POT_Q = (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.01)            # 포트홀 임계값 대표 분위(+ 운영점 행). 포트홀 운영점은 최고 점수 근처라 위쪽을 촘촘히
COLORS = {"sensitivity": "#2a78d6", "specificity": "#eb6834", "balanced": "#1baf7a", "precision": "#6250d6"}   # dataviz 기준 팔레트 1·2·3·7번
LABELS = {"sensitivity": "민감도", "specificity": "특이도", "balanced": "균형 정확도", "precision": "후보 정밀도(그 종류)"}


def grid_value(levels, value):
    """격자 임계값 중 value 이하에서 가장 큰 값(같은 후보를 남김). value가 −∞면 −∞.
    CSV는 임계값을 소수 여섯째 자리로 반올림해 적고 config는 내림한 값이라(config.py 주석) 그 차이(1e-6)까지는 같은 값으로 본다."""
    levels = np.sort(np.asarray(levels, float))
    ok = levels[levels <= value + ROUND_TOL]
    return float(ok[-1]) if len(ok) else float(levels[0])


def kind_precision(curve, kind, tau):
    """그 종류 후보만 남긴 행(다른 종류 임계값 +∞)에서 (후보/영상, 오검출/영상, 정밀도)."""
    other = "tau_pothole" if kind == "crack" else "tau_crack"
    row = curve[(curve[other] == np.inf) & np.isclose(curve[f"tau_{kind}"], tau, rtol=0, atol=1e-9)]
    if not len(row):
        return np.nan, np.nan, np.nan
    pred, fp = float(row.pred_per_image.iloc[0]), float(row.fp_per_image.iloc[0])
    return pred, fp, (pred - fp) / pred if pred > 0 else np.nan


def build(curve, kind, fixed, qs, op_tau):
    """kind 임계값을 대표 분위와 운영점에서 바꾸고 다른 종류는 fixed로 고정한 표."""
    other = "pothole" if kind == "crack" else "crack"
    part = curve[np.isclose(curve[f"tau_{other}"], fixed, rtol=0, atol=1e-9) | ((curve[f"tau_{other}"] == fixed))]
    part = part.sort_values(f"q_{kind}")
    picks = [part.iloc[int(np.argmin(np.abs(part[f"q_{kind}"].to_numpy() - q)))] for q in qs]
    op = part[np.isclose(part[f"tau_{kind}"], op_tau, rtol=0, atol=1e-9) | (part[f"tau_{kind}"] == op_tau)]
    picks.append(op.iloc[0])
    rows = []
    for r in sorted({id(p): p for p in picks}.values(), key=lambda p: p[f"q_{kind}"]):
        k_pred, k_fp, k_prec = kind_precision(curve, kind, r[f"tau_{kind}"])
        rows.append({"임계값": r[f"tau_{kind}"], "분위": r[f"q_{kind}"], "운영점": "◀ 운영점" if r[f"tau_{kind}"] == op_tau else "",
                     "후보/영상(전체)": r.pred_per_image, "후보 정밀도(전체)": 1 - r.fp_per_image / r.pred_per_image if r.pred_per_image > 0 else np.nan,
                     f"{kind} 후보/영상": k_pred, f"{kind} 후보 정밀도": k_prec,
                     f"{kind} 재현율": r[f"{kind}_ctr_recall"], f"{kind} 우연 보정 재현율": r[f"{kind}_ctr_recall_net"],
                     "민감도": r.sensitivity, "특이도": r.specificity, "균형 정확도": r.balanced, "정답률": r.accuracy,
                     "영상당 오검출": r.fp_per_image})
    table = pd.DataFrame(rows).drop_duplicates(subset=["임계값"])
    return table, part


def draw(parts, ops, path):
    """종류마다 한 패널: x = 영상당 오검출, y = 민감도·특이도·균형 정확도·그 종류 후보 정밀도(%), 운영점 표시. 축은 하나(%)만 쓴다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = C.FONT_BY_PLATFORM.get(sys.platform, C.FONT_DEFAULT)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (kind, (part, curve)) in zip(axes, parts.items()):
        part = part.sort_values("fp_per_image")
        prec = [kind_precision(curve, kind, t)[2] for t in part[f"tau_{kind}"]]
        x = part.fp_per_image.to_numpy()
        for key in ("sensitivity", "specificity", "balanced"):
            ax.plot(x, 100 * part[key].to_numpy(), color=COLORS[key], lw=2, label=LABELS[key])
        ax.plot(x, 100 * np.asarray(prec, float), color=COLORS["precision"], lw=2, ls="--", label=LABELS["precision"])
        op = part[part[f"tau_{kind}"] == ops[kind]].iloc[0]
        ax.axvline(op.fp_per_image, color="#888888", lw=1, ls=":")
        ax.plot([op.fp_per_image], [100 * op.sensitivity], "o", ms=8, color=COLORS["sensitivity"], mec="white", mew=2)
        ax.annotate(f"운영점 τ={op[f'tau_{kind}']:.3f}\n민감도 {op.sensitivity:.1%}, 오검출 {op.fp_per_image:.2f}",
                    (op.fp_per_image, 100 * op.sensitivity), xytext=(10, 12), textcoords="offset points", fontsize=9, color="#333333")
        ax.set_title(f"{'균열' if kind == 'crack' else '포트홀'} 점수 임계값을 바꿀 때({'포트홀' if kind == 'crack' else '균열'}은 운영점 고정)", fontsize=10)
        ax.set_xlabel("영상당 오검출(개)")
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("%")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, frameon=False, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def copy_basis(source=BASIS_SOURCE, out=REPORT_DIR / "table20_threshold_basis.md"):
    """docs/part_C.md의 주요 임계값 근거 표 절을 results/report/로 옮긴다(표를 두 곳에서 따로 고치지 않게). 절이 없으면 None."""
    lines = source.read_text(encoding="utf-8").splitlines()
    if BASIS_HEADING not in lines:
        return None
    start = lines.index(BASIS_HEADING)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith(("### ", "## "))), len(lines))
    head = f"<!-- 출처: {source.as_posix()} '{BASIS_HEADING[4:]}' 절. 고칠 때는 원본을 고치고 이 도구를 다시 돌린다 -->"
    out.write_text("\n".join([head, *lines[start:end]]).rstrip() + "\n", encoding="utf-8")
    return out


def main(argv=None):
    """운영점 실행의 tau_curve.csv에서 균열·포트홀 임계값 표와 그래프를 results/report/에 저장한다."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="e5_supp_dev", help="임계값별 성능 곡선이 있는 결과 폴더(results/ 아래)")
    parser.add_argument("--preprocess", default="off", choices=("off", "on"), help="전처리 미적용(off, 운영점을 정한 행) 또는 적용(on)")
    args = parser.parse_args(argv)
    run = Path("results") / args.run
    curve = pd.read_csv(run / "tau_curve.csv", encoding="utf-8-sig")
    curve = curve[(curve.preprocess == args.preprocess) & (curve.params == "config")]
    points = pd.read_csv(run / "tau_points.csv", encoding="utf-8-sig")
    cfg = points[(points.preprocess == args.preprocess) & (points.point == "config")].iloc[0]
    ops = {k: grid_value(curve[f"tau_{k}"].unique(), cfg[f"tau_{k}"]) for k in ("crack", "pothole")}
    op_row = curve[(curve.tau_crack == ops["crack"]) & (curve.tau_pothole == ops["pothole"])].iloc[0]
    for col in ("sensitivity", "specificity", "balanced", "fp_per_image"):
        if not np.isclose(op_row[col], cfg[col], atol=1e-6):
            raise RuntimeError(f"운영점 격자 행의 {col} {op_row[col]}이 tau_points config 행 {cfg[col]}과 다름")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    crack, crack_part = build(curve, "crack", ops["pothole"], CRACK_Q, ops["crack"])
    pot, pot_part = build(curve, "pothole", ops["crack"], POT_Q, ops["pothole"])
    note = f"# 출처: results/{args.run}/tau_curve.csv, 전처리 {'미적용' if args.preprocess == 'off' else '적용'}, 개발셋"
    for table, name in ((crack, "table18_tau_crack.csv"), (pot, "table19_tau_pothole.csv")):
        with open(REPORT_DIR / name, "w", encoding="utf-8-sig", newline="") as f:
            f.write(note + "\n")
            table.to_csv(f, index=False, float_format="%.4f")
    draw({"crack": (crack_part, curve), "pothole": (pot_part, curve)}, ops, REPORT_DIR / "fig_tau_tables.png")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(note)
    print(crack.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pot.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("저장:", REPORT_DIR / "table18_tau_crack.csv", REPORT_DIR / "table19_tau_pothole.csv", REPORT_DIR / "fig_tau_tables.png")
    basis = copy_basis()
    print("근거 표:", basis if basis else f"{BASIS_SOURCE}에 '{BASIS_HEADING}' 절이 없어 건너뜀")


if __name__ == "__main__":
    main()
