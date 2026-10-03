"""합치기 전 약속 검사: 파트끼리 주고받는 형식이 약속대로인지 확인한다 (관리: 파트 C).
사용법: 맨 위 폴더에서  python tests/check_contract.py
통과하면 '약속 검사 통과', 실패하면 [담당 파트]와 이유를 모두 출력하고 종료 코드 1로 끝난다."""
import glob
import os
import sys

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import analyze, detect, features, io_utils, pipeline, preprocess, visualize  # noqa: E402
from src import config as C  # noqa: E402

QUALITY_KEYS = {"mean", "std", "lap_var", "noise"}
STEP_KEYS = {"denoise", "tone", "gamma", "sharpen"}
TONES = (None, "gamma", "equalize", "clahe")
REGION_KEYS = {"area", "elong", "fill", "solidity", "bbox", "contour"}
ROW_KEYS = {"edges", "harris", "n_crack", "area_crack", "n_pothole", "area_pothole",
            "roi_pixels", "t_pre_ms", "t_detect_ms"}
MATCH_KEYS = {"kp1", "kp2", "good", "inliers", "inlier_ratio", "k1", "k2", "matches", "mask"}
FAILS = []


def check(part, ok, msg):
    if not ok:
        FAILS.append(f"[{part}] {msg}")


def is_gray(a):
    return isinstance(a, np.ndarray) and a.dtype == np.uint8 and a.ndim == 2


def is_int(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def samples():
    """합성 노면 1장(항상) + 제공 영상 1장(있으면). 각 파트 검사는 이 영상에서 직접 만든 입력을 쓴다."""
    rng = np.random.default_rng(0)
    road = rng.normal(120, 6, (480, 640)).clip(0, 255).astype(np.uint8)
    cv.line(road, (60, 300), (580, 330), 50, 3)        # 가는 어두운 선 = 균열 흉내
    cv.circle(road, (320, 400), 25, 60, -1)            # 어두운 덩어리 = 포트홀 흉내
    out = [("합성 노면", cv.cvtColor(road, cv.COLOR_GRAY2BGR))]
    files = sorted(glob.glob("data/provided/*.jpg"))
    if files:
        img = cv.imread(files[0])
        if img is not None:
            out.append((os.path.basename(files[0]), img))
    return out


def gray_of(bgr):
    """검사용 흑백 노면: A의 함수를 쓰지 않고 직접 만든다 (A가 고장 나도 B·C 검사는 따로 돌게)."""
    h, w = bgr.shape[:2]
    small = cv.resize(bgr, (640, round(h * 640 / w)), interpolation=cv.INTER_AREA)
    return cv.cvtColor(small[small.shape[0] // 2:], cv.COLOR_BGR2GRAY)


def check_a(name, bgr):
    files = sorted(glob.glob("data/provided/*.jpg"))
    if files:
        img = io_utils.load_image(files[0])
        check("A", isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.ndim == 3
              and img.shape[2] == 3, "load_image: BGR uint8 (H, W, 3)을 돌려주지 않음")
    small = io_utils.resize_width(bgr)
    check("A", small.ndim == 3 and small.shape[1] == C.TARGET_WIDTH and small.dtype == np.uint8,
          "resize_width: 가로 640·uint8·컬러 3채널이 아님")
    roi, y0 = io_utils.crop_roi(small, 0.5, 1.0)
    check("A", is_int(y0) and roi.shape[0] == small.shape[0] - y0 and roi.shape[1] == small.shape[1],
          "crop_roi: (노면 영상, y0 정수)가 아니거나 크기가 맞지 않음")
    info = io_utils.parse_name("data/provided/Japan_000107.jpg")
    check("A", isinstance(info, dict) and info.get("source") in ("provided", "own", "pair")
          and isinstance(info.get("condition"), str), "parse_name: source·condition이 든 dict가 아님")
    gray = gray_of(bgr)
    before = gray.copy()
    q = analyze.measure_quality(gray)
    check("A", isinstance(q, dict) and QUALITY_KEYS <= q.keys()
          and all(np.isfinite(float(q[k])) for k in QUALITY_KEYS),
          "measure_quality: mean·std·lap_var·noise 숫자가 다 있지 않음")
    s = analyze.choose_steps(q)
    check("A", isinstance(s, dict) and STEP_KEYS <= s.keys()
          and isinstance(s["denoise"], (bool, np.bool_)) and isinstance(s["sharpen"], (bool, np.bool_))
          and s["tone"] in TONES and np.isfinite(s["gamma"]),
          f"choose_steps: steps 형식이 약속과 다름 (tone은 {TONES} 중 하나)")
    check("A", analyze.steps_to_text(dict(analyze.NO_STEPS)) == "없음"
          and isinstance(analyze.steps_to_text(s), str), "steps_to_text: NO_STEPS가 '없음'이 아님")
    check("A", np.array_equal(preprocess.preprocess(gray, dict(analyze.NO_STEPS)), gray),
          "preprocess: 전처리 끔(NO_STEPS)인데 영상이 바뀜")
    on = {"denoise": True, "tone": "gamma", "gamma": 0.5, "sharpen": True}
    for steps in (s, on, {**on, "tone": "equalize"}):
        out = preprocess.preprocess(gray, steps)
        check("A", is_gray(out) and out.shape == gray.shape,
              f"preprocess: 결과가 같은 크기 흑백 uint8이 아님 ({analyze.steps_to_text(steps)})")
    check("A", np.array_equal(gray, before), "analyze/preprocess가 입력 배열을 직접 바꿈")


def check_b(name, bgr):
    gray = gray_of(bgr)
    before = gray.copy()
    blank = np.zeros_like(gray)
    for fn, what in ((detect.detect_cracks, "에지"), (detect.detect_potholes, "마스크")):
        for img in (gray, blank):
            res = fn(img)
            if not (isinstance(res, tuple) and len(res) == 2):
                check("B", False, f"{fn.__name__}: (후보 list, {what}) 두 개를 돌려주지 않음")
                break
            regions, mask = res
            check("B", is_gray(mask) and mask.shape == img.shape and set(np.unique(mask)) <= {0, 255},
                  f"{fn.__name__}: {what}가 입력과 같은 크기의 uint8 0/255 영상이 아님")
            check("B", isinstance(regions, list), f"{fn.__name__}: 후보가 list가 아님")
            hh, ww = img.shape
            for r in regions:
                ok = isinstance(r, dict) and REGION_KEYS <= r.keys()
                if ok:
                    x, y, w, h = r["bbox"]
                    c = r["contour"]
                    ok = (is_int(r["area"]) and r["area"] > 0 and min(x, y) >= 0
                          and x + w <= ww and y + h <= hh and isinstance(c, np.ndarray)
                          and c.ndim == 3 and c.shape[1:] == (1, 2))
                if not ok:
                    check("B", False, f"{fn.__name__}: 후보 형식 오류 (키 {sorted(REGION_KEYS)}, "
                                      "area>0 정수, bbox가 노면 영상 안, contour (N,1,2))")
                    break
    check("B", np.array_equal(gray, before), "detect가 입력 배열을 직접 바꿈")
    try:
        from src import evaluate
    except ImportError:
        return                                         # evaluate.py는 아직 없어도 됨
    a, far = (10, 10, 20, 20), (300, 300, 5, 5)
    check("B", abs(evaluate.iou(a, a) - 1) < 1e-9 and evaluate.iou(a, far) == 0,
          "iou: 같은 박스가 1, 안 겹치는 박스가 0이 아님")
    cnt = evaluate.count_matches([a, far], [a], 0.3)
    check("B", {k: cnt.get(k) for k in ("tp", "fp", "found", "fn")}
          == {"tp": 1, "fp": 1, "found": 1, "fn": 0}, "count_matches: tp·fp·found·fn 개수가 틀림")
    p, r = evaluate.precision_recall({"tp": 0, "fp": 0, "found": 0, "fn": 0})
    check("B", np.isnan(p) and np.isnan(r), "precision_recall: 분모가 0이면 NaN이어야 함")


def check_c(name, bgr):
    gray = gray_of(bgr)
    edges = cv.Canny(gray, 50, 150)
    check("C", is_int(features.count_edges(edges)), "count_edges: 정수가 아님")
    for img in (gray, np.zeros_like(gray)):
        n = features.count_harris(img)
        check("C", is_int(n) and n >= 0, "count_harris: 0 이상 정수가 아님")
    m = features.match_pair(gray, gray)
    check("C", MATCH_KEYS <= m.keys() and is_int(m["inliers"]) and 0 <= m["inlier_ratio"] <= 1,
          "match_pair: 결과 키 또는 inlier 값이 약속과 다름")
    m0 = features.match_pair(np.zeros_like(gray), np.zeros_like(gray))
    check("C", m0["inliers"] == 0, "match_pair: 빈 영상에서 inlier가 0이 아님")
    rows = []
    for use in (False, True, True):
        out = pipeline.run_pipeline(bgr, 0.5, 1.0, use)
        check("C", set(out["row"].keys()) == ROW_KEYS, f"run_pipeline: row 열이 {sorted(ROW_KEYS)}와 다름")
        if not use:
            check("C", out["steps"] == analyze.NO_STEPS and np.array_equal(out["proc"], out["gray"]),
                  "run_pipeline: 전처리 끔인데 steps가 NO_STEPS가 아니거나 영상이 바뀜")
        drawn = visualize.draw_candidates(out["small"], out["cracks"], out["potholes"], out["y0"])
        check("C", drawn.shape == out["small"].shape and drawn is not out["small"],
              "draw_candidates: 같은 크기의 새 영상을 돌려주지 않음")
        rows.append({k: v for k, v in out["row"].items() if not k.startswith("t_")})
    check("C", rows[1] == rows[2], "run_pipeline: 같은 입력인데 결과가 달라짐 (무작위·전역 상태 확인)")


def main():
    for name, bgr in samples():
        for part, fn in (("A", check_a), ("B", check_b), ("C", check_c)):
            try:
                fn(name, bgr)
            except Exception as e:                     # 오류도 담당 파트 이름을 붙여 모은다
                FAILS.append(f"[{part}] {name}: 실행 중 오류 {type(e).__name__}: {' '.join(str(e).split())}")
    fails = list(dict.fromkeys(FAILS))                 # 같은 메시지는 한 번만
    if fails:
        print(f"약속 검사 실패 {len(fails)}건")
        for f in fails:
            print("  -", f)
        print("※ [C] 실패가 A·B 실패와 함께 나오면 A·B부터 고친다.")
        sys.exit(1)
    print("약속 검사 통과")


if __name__ == "__main__":
    main()
