"""검출 결과를 640px 영상 위에 그리고, 전처리 전·후 비교 그림을 저장한다."""
import os
import sys

import cv2 as cv
import matplotlib
import numpy as np

matplotlib.use("Agg")  # 창 없이 파일로만 저장 (pyplot import 전에 정해야 함)
import matplotlib.pyplot as plt  # noqa: E402

from src import config as C  # noqa: E402

ROI_LINE_COLOR = (0, 255, 255)      # 노면 시작선: 노랑 (BGR)
CRACK_COLOR = (0, 0, 255)           # 균열 윤곽: 빨강
POTHOLE_COLOR = (255, 0, 0)         # 포트홀 윤곽: 파랑
ROI_LINE_THICKNESS = 1
CONTOUR_THICKNESS = 2


def draw_candidates(bgr, cracks, potholes, y0):
    """노면 시작선과 후보 윤곽을 그린 복사본을 돌려준다. 후보는 노면 띠 좌표라 y0만큼 내려 그린다."""
    out = bgr.copy()
    cv.line(out, (0, y0), (out.shape[1] - 1, y0), ROI_LINE_COLOR, ROI_LINE_THICKNESS)
    for regions, color in ((cracks, CRACK_COLOR), (potholes, POTHOLE_COLOR)):
        cv.drawContours(out, [r["contour"] for r in regions], -1, color, CONTOUR_THICKNESS, offset=(0, y0))
    return out


def draw_road(bgr, road, y0):
    """노면 마스크를 반투명 초록으로 칠한 복사본을 돌려준다. 마스크는 노면 띠 좌표라 y0만큼 내려 칠한다."""
    out = bgr.copy()
    h = max(0, min(road.shape[0], out.shape[0] - y0))
    w = min(road.shape[1], out.shape[1])
    band = out[y0:y0 + h, :w]                       # out의 일부 (여기에 칠하면 out에 반영됨)
    on_road = road[:h, :w] > 0
    color = np.array(C.ROAD_OVERLAY_COLOR, np.float64)
    alpha = C.ROAD_OVERLAY_ALPHA
    band[on_road] = np.round((1 - alpha) * band[on_road] + alpha * color).astype(np.uint8)
    return out


def save_comparison(path, title, panels):
    """(제목, 영상) 6개를 2행 3열 그림으로 저장한다. 위 줄은 전처리 끔, 아래 줄은 켬.
    흑백은 0~255 고정 밝기로 그려 전·후를 같은 기준으로 비교한다."""
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
