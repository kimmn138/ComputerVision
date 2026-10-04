"""(파트 C) 검출 후보를 640px 영상 위에 그리고, 전처리 전·후 비교 그림을 만든다."""
import os
import sys

import cv2 as cv
import matplotlib
import numpy as np

matplotlib.use("Agg")  # 창 없이 파일로만 저장하는 백엔드. pyplot을 import하기 전에 정해야 함
import matplotlib.pyplot as plt  # noqa: E402

from src import config as C  # noqa: E402

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
    """전처리 끔(위 줄)·켬(아래 줄) 영상 6장을 2행 3열 한 그림으로 path에 저장한다(src/에서 파일을 저장하는 유일한 함수).
    panels는 (제목, 영상) 6개. 흑백은 0~255 고정 밝기로, 컬러는 BGR→RGB로 바꿔 그려 전·후를 같은 기준으로 비교한다.
    Agg 백엔드라 창을 띄우지 않고, 저장 뒤 plt.close로 닫아 여러 장을 연속 저장해도 그림이 쌓이지 않는다."""
    if len(panels) != 6:
        raise ValueError(f"save_comparison: panels는 6개여야 함 ({len(panels)}개)")
    for name, img in panels:
        if img.dtype != np.uint8 or not (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 3)):
            raise ValueError(f"save_comparison: '{name}' 영상이 흑백 (H, W) 또는 컬러 (H, W, 3) uint8이 아님")

    font = C.FONT_BY_PLATFORM.get(sys.platform, C.FONT_DEFAULT)
    with matplotlib.rc_context({"font.family": font, "axes.unicode_minus": False}):
        fig, axes = plt.subplots(2, 3, figsize=C.COMPARE_FIGSIZE, layout="constrained")
        try:
            for ax, (name, img) in zip(axes.flat, panels):
                if img.ndim == 2:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                else:
                    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), interpolation="nearest")
                ax.set_title(name)
                ax.set_axis_off()
            fig.suptitle(title)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            fig.savefig(path, dpi=C.COMPARE_DPI)
        finally:
            plt.close(fig)
