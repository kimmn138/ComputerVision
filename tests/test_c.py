"""파트 C 시험: features의 해리스 점 세기와 SIFT 매칭, visualize의 비교 그림 저장이 경계 입력에서도 약속대로 동작하는지 확인한다.
사용법: 맨 위 폴더에서  python tests/test_c.py"""
import os
import sys
import tempfile
import warnings

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import features, visualize  # noqa: E402
from src import config as C  # noqa: E402

FAILS = []


def check(ok, msg):
    if not ok:
        FAILS.append(msg)


def square(size, value):
    """검은 바탕에 회색 사각형 하나: 해리스 코너 4개, SIFT 특징점이 아주 적은 영상."""
    img = np.zeros((240, 640), np.uint8)
    y, x = 120 - size // 2, 320 - size // 2
    img[y:y + size, x:x + size] = value
    return img


def texture():
    """무늬가 많은 영상과 그것을 (5, 12)픽셀 민 영상: 매칭이 잘 되는 쌍."""
    rng = np.random.default_rng(0)
    t = cv.GaussianBlur(rng.integers(0, 256, (240, 640), dtype=np.uint8), (0, 0), 2)
    return t, np.roll(t, (5, 12), axis=(0, 1))


def test_harris():
    """빈 영상은 0, 사각형 하나는 모서리 4개."""
    blank = np.zeros((240, 640), np.uint8)
    check(features.count_harris(blank) == 0, "count_harris: 빈 영상이 0이 아님")
    n = features.count_harris(square(80, 255))
    check(n == 4, f"count_harris: 사각형 하나에서 4가 아님 ({n})")


def test_match_blank():
    """빈 영상끼리: 특징점이 없어도 오류 없이 모두 0."""
    blank = np.zeros((240, 640), np.uint8)
    m = features.match_pair(blank, blank)
    check(m == {"kp1": 0, "kp2": 0, "good": 0, "inliers": 0, "inlier_ratio": 0.0,
                "k1": [], "k2": [], "matches": [], "mask": None},
          f"match_pair: 빈 영상 결과가 모두 0이 아님 ({m})")


def test_match_few():
    """good이 RANSAC_MIN_GOOD 미만이면 RANSAC을 건너뛰고 inliers 0, mask None."""
    img = square(40, 128)
    m = features.match_pair(img, np.zeros_like(img))
    check(m["good"] < C.RANSAC_MIN_GOOD, f"시험 영상의 good이 기준보다 많음 ({m['good']})")
    check(m["kp2"] == 0 and m["inliers"] == 0 and m["inlier_ratio"] == 0.0 and m["mask"] is None,
          "match_pair: good이 기준 미만인데 inlier가 0이 아님")
    check(m["good"] == len(m["matches"]), "match_pair: good 수와 matches 길이가 다름")


def test_match_boundary():
    """good이 정확히 RANSAC_MIN_GOOD이면 RANSAC을 돌려 mask가 good 길이의 list가 된다."""
    img = square(40, 128)
    m = features.match_pair(img, img)
    check(m["good"] == C.RANSAC_MIN_GOOD, f"시험 영상의 good이 기준과 같지 않음 ({m['good']})")
    check(isinstance(m["mask"], list) and len(m["mask"]) == m["good"],
          "match_pair: good이 기준과 같은데 RANSAC을 건너뜀")
    check(0 <= m["inlier_ratio"] <= 1, "match_pair: inlier_ratio가 0~1 밖")


def test_match_texture():
    """무늬 영상과 민 영상: inlier가 생기고 mask 길이는 good과 같다."""
    a, b = texture()
    m = features.match_pair(a, b)
    check(m["good"] >= C.RANSAC_MIN_GOOD and m["inliers"] > 0, f"match_pair: 민 영상 매칭 실패 ({m['good']})")
    check(isinstance(m["inliers"], int) and 0 <= m["inlier_ratio"] <= 1, "match_pair: inlier 형식 오류")
    check(isinstance(m["mask"], list) and len(m["mask"]) == m["good"],
          "match_pair: mask가 good 길이의 list가 아님")


def test_no_inplace():
    """받은 영상을 바꾸지 않는다."""
    a, b = texture()
    a0, b0 = a.copy(), b.copy()
    features.count_harris(a)
    features.match_pair(a, b)
    check(np.array_equal(a, a0) and np.array_equal(b, b0), "features가 입력 배열을 직접 바꿈")


def test_save_comparison():
    """흑백 4장·컬러 2장을 한글 제목으로 저장: 파일 생성, 입력 그대로, 전역 그림 없음, 글꼴 경고 없음, 6장 아니면 ValueError."""
    a, b = texture()
    color = cv.cvtColor(a, cv.COLOR_GRAY2BGR)
    imgs = [a, np.zeros_like(a), color, b, (b > 128).astype(np.uint8) * 255, color.copy()]
    names = ["끔: 노면", "끔: 에지", "끔: 후보", "켬: 노면", "켬: 에지", "켬: 후보"]
    before = [im.copy() for im in imgs]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub", "cmp.png")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            visualize.save_comparison(path, "전처리 전·후 비교", list(zip(names, imgs)))
        check(os.path.isfile(path) and os.path.getsize(path) > 0, "save_comparison: 파일이 저장되지 않음")
        glyph = [str(x.message) for x in w if "missing from font" in str(x.message)]
        check(not glyph, f"save_comparison: 한글 글꼴 경고 {len(glyph)}건 (예: {glyph[:1]})")
    check(all(np.array_equal(x, y) for x, y in zip(imgs, before)), "save_comparison이 입력 영상을 직접 바꿈")
    check(not plt.get_fignums(), "save_comparison: pyplot 전역 그림이 남음")
    try:
        visualize.save_comparison(os.path.join(tempfile.gettempdir(), "x.png"), "t", list(zip(names, imgs))[:5])
        check(False, "save_comparison: 5장인데 ValueError가 나지 않음")
    except ValueError:
        pass


def main():
    for fn in (test_harris, test_match_blank, test_match_few, test_match_boundary, test_match_texture, test_no_inplace,
               test_save_comparison):
        try:
            fn()
        except Exception as e:
            FAILS.append(f"{fn.__name__}: 실행 중 오류 {type(e).__name__}: {e}")
    if FAILS:
        print(f"C 시험 실패 {len(FAILS)}건")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("C 시험 통과")


if __name__ == "__main__":
    main()
