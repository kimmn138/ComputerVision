"""(파트 C) 검출 후보를 640px 영상 위에 그리고, 전처리 전·후 비교 그림을 만든다."""
import cv2 as cv

ROI_LINE_COLOR = (0, 255, 255)      # 노면 시작선: 노랑 (BGR)
CRACK_COLOR = (0, 0, 255)           # 균열 윤곽: 빨강
POTHOLE_COLOR = (255, 0, 0)         # 포트홀 윤곽: 파랑
ROI_LINE_THICKNESS = 1
CONTOUR_THICKNESS = 2


def draw_candidates(bgr, cracks, potholes, y0):
    """노면 시작선과 후보 윤곽을 640px 컬러 영상의 복사본에 그린다(받은 영상은 바꾸지 않음).
    후보 좌표는 노면 기준이므로 offset (0, y0)으로 전체 영상 좌표에 맞춰 그린다."""
    out = bgr.copy()
    cv.line(out, (0, y0), (out.shape[1] - 1, y0), ROI_LINE_COLOR, ROI_LINE_THICKNESS)
    for regions, color in ((cracks, CRACK_COLOR), (potholes, POTHOLE_COLOR)):
        cv.drawContours(out, [r["contour"] for r in regions], -1, color, CONTOUR_THICKNESS, offset=(0, y0))
    return out


def save_comparison(path, title, panels):
    """전처리 전·후 영상을 한 그림에 모아 path에 저장한다(src/에서 파일을 저장하는 유일한 함수).
    임시 버전: 아직 구현 전이라 NotImplementedError를 낸다."""
    # TODO(C): 전·후 6칸 비교 그림 저장
    raise NotImplementedError("TODO(C): save_comparison")
