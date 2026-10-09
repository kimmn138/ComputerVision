"""합치기 전 약속 검사: 파트끼리 주고받는 형식이 약속대로인지 확인한다 (관리: 파트 C).
사용법: 맨 위 폴더에서  python tests/check_contract.py
통과하면 '약속 검사 통과', 실패하면 [담당 파트]와 이유를 모두 출력하고 종료 코드 1로 끝난다."""
import glob
import inspect
import os
import sys

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import analyze, detect, features, io_utils, pipeline, preprocess, visualize  # noqa: E402
from src import config as C  # noqa: E402

QUALITY_KEYS = {"mean", "std", "lap_var", "noise"}          # 유한한 실수
STEP_KEYS = {"denoise", "tone", "gamma", "sharpen"}
TONES = (None, "gamma", "equalize", "clahe")
REGION_KEYS = {"area", "elong", "fill", "solidity", "bbox", "contour"}
NEW_REGION_KEYS = {"thickness", "dt_width", "circ_inv", "contrast", "score"}   # 10/10 약속 변경: 실수. contrast·score는 NaN 허용
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


def roi_of(bgr):
    """검사용 컬러 노면 띠(아래 절반): A의 함수를 쓰지 않고 직접 만든다 (A가 고장 나도 B·C 검사는 따로 돌게)."""
    h, w = bgr.shape[:2]
    small = cv.resize(bgr, (640, round(h * 640 / w)), interpolation=cv.INTER_AREA)
    return small[small.shape[0] // 2:].copy()


def gray_of(bgr):
    """검사용 흑백 노면: roi_of와 같은 띠를 흑백으로."""
    return cv.cvtColor(roi_of(bgr), cv.COLOR_BGR2GRAY)


def same_result(a, b):
    """detect_* 결과 두 개가 같은지: 후보마다 (bbox, 넓이)와 마스크가 같으면 같다."""
    return ([(r["bbox"], r["area"]) for r in a[0]] == [(r["bbox"], r["area"]) for r in b[0]]
            and np.array_equal(a[1], b[1]))


def center_on_road(regions, road):
    """후보 bbox의 중심 픽셀이 모두 노면(255) 안인지 (pipeline.keep_on_road의 'center' 규칙과 같은 계산)."""
    h, w = road.shape
    return all(road[min(int(y + bh / 2), h - 1), min(int(x + bw / 2), w - 1)] > 0
               for x, y, bw, bh in (r["bbox"] for r in regions))


def check_a_names():
    """parse_name이 약속 표대로 own·pair·제공 영상 이름을 읽고, 규칙을 어긴 own·pair 이름에 ValueError를 내는지 본다.
    태그는 config A 구역에서 가져와 A가 태그를 바꿔도 검사가 따라간다."""
    damage, cond = C.DAMAGE_TAGS[0], C.CONDITION_TAGS[0]
    role, pcond = C.PAIR_ROLES[0], C.PAIR_CONDITIONS[0]
    own = io_utils.parse_name(f"data/own/own_01_{damage}_{cond}.jpg")
    check("A", isinstance(own, dict) and own.get("source") == "own" and own.get("damage") == damage
          and own.get("condition") == cond,
          f"parse_name: own_01_{damage}_{cond}.jpg에서 {{source: own, damage, condition}}을 읽지 못함 ({own})")
    pair = io_utils.parse_name(f"data/pairs/pair01_{role}_{pcond}.jpg")
    check("A", isinstance(pair, dict) and pair.get("source") == "pair" and pair.get("pair") not in (None, "")
          and pair.get("role") == role and pair.get("condition") == pcond,
          f"parse_name: pair01_{role}_{pcond}.jpg에서 {{source: pair, pair, role, condition}}을 읽지 못함 ({pair})")
    multi = io_utils.parse_name("data/provided/United_States_004830.jpg")
    check("A", isinstance(multi, dict) and multi.get("source") == "provided" and "damage" in multi
          and isinstance(multi.get("condition"), str),
          f"parse_name: 밑줄이 여러 개인 제공 영상(United_States_004830.jpg)이 source provided가 아님 ({multi})")
    bad = [f"own_01_{damage}.jpg", f"own_01_{damage}_{cond}_x.jpg", f"own_01_nosuchtag_{cond}.jpg",
           f"own_01_{damage}_nosuchtag.jpg", f"pair01_{role}.jpg", f"pair01_nosuchrole_{pcond}.jpg",
           f"pair01_{role}_nosuchtag.jpg"]
    accepted = []
    for n in bad:
        try:
            io_utils.parse_name(n)
            accepted.append(n)
        except ValueError:
            pass
    check("A", not accepted, f"parse_name: 규칙을 어긴 own·pair 이름에 ValueError가 나지 않음 ({', '.join(accepted)})")


def check_a_tone(gray):
    """어두운·밝은 노면에서 choose_steps가 감마를 고르면, preprocess가 그 감마로 영상을 밝게·어둡게 하는지(방향) 본다.
    형식 검사만으로는 감마를 계산하는 쪽(analyze)과 적용하는 쪽(preprocess)의 방향이 엇갈려도 통과하므로 따로 확인한다."""
    dark = (gray * 0.3).astype(np.uint8)
    bright = (255 - (255 - gray.astype(np.float32)) * 0.3).astype(np.uint8)
    for img, what, brighter in ((dark, "어두운", True), (bright, "밝은", False)):
        s = analyze.choose_steps(analyze.measure_quality(img))
        if s["tone"] != "gamma":
            continue
        out = preprocess.preprocess(img, {**analyze.NO_STEPS, "tone": "gamma", "gamma": s["gamma"]})
        ok = out.mean() > img.mean() if brighter else out.mean() < img.mean()
        check("A", ok, f"choose_steps·preprocess: {what} 노면(평균 {img.mean():.0f})에 고른 감마({s['gamma']:.2f})를 "
                       f"적용해도 {'밝아지지' if brighter else '어두워지지'} 않음 (결과 평균 {out.mean():.0f}). "
                       "감마를 계산하는 식과 적용하는 식의 방향을 맞출 것")


def check_a_road(bgr):
    """road_mask가 약속대로인지 본다(10/10 약속 변경): 컬러 노면 띠 → 같은 크기 uint8 0/255, 입력을 바꾸지 않음,
    빈 영상에서도 오류 없음, 같은 입력이면 같은 출력. 지금 임시 버전(전부 255)도, A가 구현한 v1·v2도 이 검사를 통과해야 한다."""
    roi = roi_of(bgr)
    before = roi.copy()
    for img, what in ((roi, "노면 띠"), (np.zeros_like(roi), "빈 영상")):
        mask = io_utils.road_mask(img)
        check("A", is_gray(mask) and mask.shape == img.shape[:2] and set(np.unique(mask).tolist()) <= {0, 255},
              f"road_mask: {what}에서 같은 크기의 uint8 0/255 (H, W)를 돌려주지 않음")
    check("A", np.array_equal(roi, before), "road_mask가 입력 배열을 직접 바꿈")
    check("A", np.array_equal(io_utils.road_mask(roi), io_utils.road_mask(roi)),
          "road_mask: 같은 입력인데 결과가 달라짐 (무작위·전역 상태 확인)")


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
    check("A", isinstance(q.get("blur_ratio"), float),
          "measure_quality: blur_ratio 키(실수, 구현 전에는 NaN)가 없음 (약속 변경 10/10)")
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
    check_a_names()
    check_a_tone(gray)
    check_a_road(bgr)


def detect_form_ok(fn, res, shape, what):
    """detect_* 결과 하나가 약속 형식인지 보고, 어긋나면 [B]로 남긴 뒤 False를 돌려준다.
    후보 dict에는 10/10에 더한 키(thickness·dt_width·circ_inv는 유한한 실수, contrast·score는 실수이고 NaN 허용)도 있어야 한다."""
    if not (isinstance(res, tuple) and len(res) == 2):
        check("B", False, f"{fn.__name__}: (후보 list, {what}) 두 개를 돌려주지 않음")
        return False
    regions, mask = res
    check("B", is_gray(mask) and mask.shape == shape and set(np.unique(mask)) <= {0, 255},
          f"{fn.__name__}: {what}가 입력과 같은 크기의 uint8 0/255 영상이 아님")
    check("B", isinstance(regions, list), f"{fn.__name__}: 후보가 list가 아님")
    hh, ww = shape
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
            return False
        ok = (NEW_REGION_KEYS <= r.keys() and all(isinstance(r[k], float) for k in NEW_REGION_KEYS)
              and all(np.isfinite(r[k]) for k in ("thickness", "dt_width", "circ_inv")))
        if not ok:
            check("B", False, f"{fn.__name__}: 후보에 새 키 {sorted(NEW_REGION_KEYS)}가 없거나 실수가 아님 "
                              "(thickness·dt_width·circ_inv는 유한한 값, contrast·score는 NaN 허용, 약속 변경 10/10)")
            return False
    return True


def check_b_road(gray):
    """detect_*(gray, road=None)인지 본다(10/10 약속 변경): road 인자의 기본값이 None, road=None이면 인자 없이 부른 것과 같음,
    road를 줘도 약속 형식, road 배열을 바꾸지 않음, 같은 입력이면 같은 출력."""
    half = np.zeros_like(gray)
    half[:, :gray.shape[1] // 2] = 255
    before = half.copy()
    for fn, what in ((detect.detect_cracks, "에지"), (detect.detect_potholes, "마스크")):
        param = inspect.signature(fn).parameters.get("road")
        check("B", param is not None and param.default is None, f"{fn.__name__}: road=None 인자가 없음 (약속 변경 10/10)")
        if param is None:
            continue
        base = fn(gray)
        check("B", same_result(base, fn(gray, None)), f"{fn.__name__}: road=None 결과가 인자 없이 부른 결과와 다름")
        check("B", same_result(base, fn(gray)), f"{fn.__name__}: 같은 입력인데 결과가 달라짐 (무작위·전역 상태 확인)")
        detect_form_ok(fn, fn(gray, half), gray.shape, what)
    check("B", np.array_equal(half, before), "detect가 road 배열을 직접 바꿈")


def check_b(name, bgr):
    gray = gray_of(bgr)
    before = gray.copy()
    blank = np.zeros_like(gray)
    for fn, what in ((detect.detect_cracks, "에지"), (detect.detect_potholes, "마스크")):
        for img in (gray, blank):
            if not detect_form_ok(fn, fn(img), img.shape, what):
                break
    check_b_road(gray)
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
        road, t_road = out.get("road"), out.get("t_road_ms")
        check("C", is_gray(road) and road.shape == out["gray"].shape and set(np.unique(road).tolist()) <= {0, 255},
              "run_pipeline: road가 노면 띠와 같은 크기의 uint8 0/255가 아님 (약속 변경 10/10)")
        check("C", isinstance(t_road, (int, float)) and not isinstance(t_road, bool) and t_road >= 0,
              "run_pipeline: t_road_ms가 0 이상의 숫자가 아님 (약속 변경 10/10)")
        drawn = visualize.draw_candidates(out["small"], out["cracks"], out["potholes"], out["y0"])
        check("C", drawn.shape == out["small"].shape and drawn is not out["small"],
              "draw_candidates: 같은 크기의 새 영상을 돌려주지 않음")
        small = out["small"].copy()
        painted = visualize.draw_road(out["small"], road, out["y0"])
        check("C", isinstance(painted, np.ndarray) and painted.shape == small.shape and painted is not out["small"]
              and np.array_equal(out["small"], small), "draw_road: 같은 크기의 새 영상을 돌려주지 않거나 받은 영상을 바꿈")
        rows.append(({k: v for k, v in out["row"].items() if not k.startswith("t_")}, road.tobytes(),
                     [r["bbox"] for r in out["cracks"]], [r["bbox"] for r in out["potholes"]]))
    check("C", rows[1] == rows[2], "run_pipeline: 같은 입력인데 결과(row·road·후보)가 달라짐 (무작위·전역 상태 확인)")
    use_mask = C.USE_ROAD_MASK
    C.USE_ROAD_MASK = True                              # 노면 마스크를 켠 경로도 형식대로 도는지 (켜 둔 값은 꼭 되돌림)
    try:
        on = pipeline.run_pipeline(bgr, 0.5, 1.0, False)
    finally:
        C.USE_ROAD_MASK = use_mask
    check("C", center_on_road(on["cracks"] + on["potholes"], on["road"]) and on["row"]["n_crack"] == len(on["cracks"])
          and on["row"]["n_pothole"] == len(on["potholes"]),
          "run_pipeline: USE_ROAD_MASK를 켰는데 bbox 중심이 노면 밖인 후보가 남거나 row 개수가 후보 수와 다름")


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
