"""고른 전처리 단계(잡티 제거·밝기/대비 보정·샤프닝)를 흑백 노면 영상에 적용한다."""
import cv2 as cv
import numpy as np

from src import config as C


def preprocess(gray, steps):
    """steps에서 고른 단계만 적용한 새 흑백 영상을 돌려준다. NO_STEPS면 입력과 같은 영상.
    순서: 가우시안 → 감마·평활화·CLAHE 중 하나 → 샤프닝."""
    out = gray.copy()

    # 1. 잡티 제거
    if steps.get("denoise"):
        out = cv.GaussianBlur(out, C.GAUSS_KSIZE, C.GAUSS_SIGMA)

    # 2. 밝기·대비 보정
    tone = steps.get("tone")
    if tone == "gamma":
        g_val = steps.get("gamma", 1.0)
        # 결과 = 입력^감마 (밝기 0~1 기준). 표(LUT)로 한 번에 바꿈
        table = np.clip(np.round((np.arange(256) / 255.0) ** g_val * 255), 0, 255).astype(np.uint8)
        out = cv.LUT(out, table)

    elif tone == "equalize":
        out = cv.equalizeHist(out)

    elif tone == "clahe":
        clahe = cv.createCLAHE(clipLimit=C.CLAHE_CLIP, tileGridSize=C.CLAHE_TILE)
        out = clahe.apply(out)

    # 3. 샤프닝 (언샤프 마스크)
    if steps.get("sharpen"):
        blurred = cv.GaussianBlur(out, (0, 0), C.SHARPEN_SIGMA)
        # 입력 + AMOUNT × (입력 − 흐린 영상): 경계 성분을 강조
        out = cv.addWeighted(out, 1 + C.SHARPEN_AMOUNT, blurred, -C.SHARPEN_AMOUNT, 0)

    return out
