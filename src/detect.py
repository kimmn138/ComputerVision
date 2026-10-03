"""(파트 B) 전처리한 노면 영상에서 균열·포트홀 후보 영역을 찾는다."""
import numpy as np


def describe_regions(mask, min_pixels=30):
    """이진 마스크의 덩어리마다 모양 지표를 재서 균열·포트홀 공통 형식의 후보 dict로 만든다.
    임시 버전: 항상 빈 list를 돌려준다."""
    # TODO(B): 연결 요소·윤곽선으로 후보 dict(area, elong, fill, solidity, bbox, contour) 만들기
    return []


def detect_cracks(gray):
    """가늘고 긴 어두운 선을 균열 후보로 찾는다. (후보 list, 캐니 에지 uint8 0/255)를 돌려준다.
    임시 버전: 후보 없음과 빈 에지 영상을 돌려준다."""
    # TODO(B): 캐니 → 닫힘 → 모양으로 거르기, (후보, 캐니 에지) 반환
    return [], np.zeros_like(gray)


def detect_potholes(gray):
    """주변보다 어두운 덩어리를 포트홀 후보로 찾는다. (후보 list, 이진 마스크 uint8 0/255)를 돌려준다.
    임시 버전: 후보 없음과 빈 마스크를 돌려준다."""
    # TODO(B): DoG → 이진화 → 열림 → 모양으로 거르기, (후보, 마스크) 반환
    return [], np.zeros_like(gray)
