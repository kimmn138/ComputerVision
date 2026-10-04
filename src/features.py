"""(파트 C) 전처리 전·후를 비교할 정량 지표(에지 수·해리스 코너 수·SIFT 매칭)를 잰다."""
import cv2 as cv
import numpy as np

from src import config as C


def count_edges(edges):
    """에지 영상(0/255)의 에지 픽셀 수를 센다. 전처리 전·후로 에지가 얼마나 달라지는지 보는 지표."""
    return int(cv.countNonZero(edges))


def count_harris(gray):
    """해리스 코너 점의 개수를 센다. 전처리로 특징점이 늘거나 주는지 보는 지표.
    R 계산 → R > 최댓값 × HARRIS_REL → HARRIS_NMS 창 비최대 억제 순서로, 남은 점 수를 돌려준다."""
    r = cv.cornerHarris(np.float32(gray), C.HARRIS_BLOCK, C.HARRIS_KSIZE, C.HARRIS_K)
    r_max = float(r.max())
    if r_max <= 0:                                      # 빈 영상: 임계값이 0이 되어 전부 세는 것을 막음
        return 0
    strong = r > C.HARRIS_REL * r_max
    local_max = r == cv.dilate(r, np.ones((C.HARRIS_NMS, C.HARRIS_NMS), np.uint8))
    return int(np.count_nonzero(strong & local_max))


def match_pair(g1, g2):
    """두 흑백 영상을 SIFT로 매칭해 맞는 점 수와 inlier 비율을 잰다. 매칭이 안 돼도 오류 없이 0.
    SIFT → 비율 검사(SIFT_RATIO) → RANSAC 호모그래피(RANSAC_THRESH). good이 RANSAC_MIN_GOOD 미만이면
    RANSAC을 건너뛰고 inliers 0, mask None으로 둔다(matches에는 good을 그대로 남겨 그림으로 보이게 함)."""
    sift = cv.SIFT_create()
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    out = {"kp1": len(k1), "kp2": len(k2), "good": 0, "inliers": 0, "inlier_ratio": 0.0,
           "k1": list(k1), "k2": list(k2), "matches": [], "mask": None}
    if d1 is None or d2 is None or len(d2) < 2:         # 비율 검사에 이웃 2개가 필요
        return out

    pairs = cv.BFMatcher(cv.NORM_L2).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < C.SIFT_RATIO * p[1].distance]
    out["good"], out["matches"] = len(good), good
    if len(good) < C.RANSAC_MIN_GOOD:
        return out

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv.findHomography(src, dst, cv.RANSAC, C.RANSAC_THRESH)
    if mask is None:
        return out
    out["inliers"] = int(mask.sum())
    out["inlier_ratio"] = out["inliers"] / len(good)
    out["mask"] = mask.ravel().astype(int).tolist()
    return out
