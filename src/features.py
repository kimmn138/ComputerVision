"""(파트 C) 전처리 전·후를 비교할 정량 지표(에지 수·해리스 코너 수·SIFT 매칭)를 잰다."""
import cv2 as cv


def count_edges(edges):
    """에지 영상(0/255)의 에지 픽셀 수를 센다. 전처리 전·후로 에지가 얼마나 달라지는지 보는 지표."""
    return int(cv.countNonZero(edges))


def count_harris(gray):
    """해리스 코너 점의 개수를 센다. 전처리로 특징점이 늘거나 주는지 보는 지표.
    임시 버전: 항상 0을 돌려준다."""
    # TODO(C): cornerHarris + 최댓값의 1% + 비최대 억제로 점 개수
    return 0


def match_pair(g1, g2):
    """두 흑백 영상을 SIFT로 매칭해 맞는 점 수와 inlier 비율을 잰다. 매칭이 안 돼도 오류 없이 0.
    임시 버전: 매칭이 하나도 없는 결과를 돌려준다."""
    # TODO(C): SIFT → 비율 검사 → RANSAC
    return {"kp1": 0, "kp2": 0, "good": 0, "inliers": 0, "inlier_ratio": 0.0,
            "k1": [], "k2": [], "matches": [], "mask": None}
