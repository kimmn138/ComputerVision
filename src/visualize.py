"""(파트 C) 검출 후보를 640px 영상 위에 그리고, 전처리 전·후 비교 그림을 만든다."""
import os

import cv2 as cv
import matplotlib
import numpy as np
from matplotlib import font_manager
from matplotlib.figure import Figure

from src import config as C

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


def _korean_fonts():
    """FONT_CANDIDATES 중 이 컴퓨터에 설치된 글꼴만 고른다(없는 이름을 넣으면 findfont 경고가 쌓이므로)."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return [name for name in C.FONT_CANDIDATES if name in installed]


def save_comparison(path, title, panels):
    """전처리 끔(위 줄)·켬(아래 줄) 영상 6장을 2행 3열 한 그림으로 path에 저장한다(src/에서 파일을 저장하는 유일한 함수).
    panels는 (제목, 영상) 6개. 흑백은 0~255 고정 밝기로, 컬러는 BGR→RGB로 바꿔 그려 전·후를 같은 기준으로 비교한다.
    pyplot 대신 Figure를 직접 만들어 창을 띄우지 않고 전역 그림도 남기지 않는다."""
    if len(panels) != 6:
        raise ValueError(f"save_comparison: panels는 6개여야 함 ({len(panels)}개)")
    for name, img in panels:
        if img.dtype != np.uint8 or not (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 3)):
            raise ValueError(f"save_comparison: '{name}' 영상이 흑백 (H, W) 또는 컬러 (H, W, 3) uint8이 아님")

    fonts = _korean_fonts()
    rc = {"axes.unicode_minus": False}
    if fonts:
        rc["font.family"] = fonts
    with matplotlib.rc_context(rc):
        fig = Figure(figsize=C.COMPARE_FIGSIZE, layout="constrained")
        for ax, (name, img) in zip(fig.subplots(2, 3).flat, panels):
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            else:
                ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), interpolation="nearest")
            ax.set_title(name)
            ax.set_axis_off()
        fig.suptitle(title)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=C.COMPARE_DPI)
