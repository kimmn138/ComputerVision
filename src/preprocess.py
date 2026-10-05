"""(파트 A) 고른 전처리 단계(잡티 제거·밝기/대비 보정·샤프닝)만 흑백 노면 영상에 적용한다."""
import cv2 as cv
import numpy as np

from src import config as C


def preprocess(gray, steps):
    """검출이 잘 되도록 고른 단계만 적용한 흑백 영상을 돌려준다. NO_STEPS면 입력과 똑같은 영상.
    순서: 가우시안(GAUSS_KSIZE·GAUSS_SIGMA) → 감마(결과 = 입력^감마, 감마 < 1이면 밝게)·평활화·CLAHE(CLAHE_CLIP·CLAHE_TILE)
    → 샤프닝(언샤프 마스크: 결과 = 입력 + SHARPEN_AMOUNT × (입력 − 가우시안(SHARPEN_SIGMA)))."""
    out = gray.copy()

    # 1. 잡티 제거 (Denoise)
    if steps.get("denoise"):
        out = cv.GaussianBlur(out, C.GAUSS_KSIZE, C.GAUSS_SIGMA)

    # 2. 톤/조도 보정 (Tone)
    tone = steps.get("tone")
    if tone == "gamma":
        g_val = steps.get("gamma", 1.0)
        # 0~1로 나눈 밝기를 감마 제곱. analyze.choose_steps가 이 방향(입력^감마)에 맞춰 감마를 고른다
        table = np.clip(np.round((np.arange(256) / 255.0) ** g_val * 255), 0, 255).astype(np.uint8)
        out = cv.LUT(out, table)

    elif tone == "equalize":
        out = cv.equalizeHist(out)

    elif tone == "clahe":
        clahe = cv.createCLAHE(clipLimit=C.CLAHE_CLIP, tileGridSize=C.CLAHE_TILE)
        out = clahe.apply(out)

    # 3. 샤프닝 (Sharpen)
    if steps.get("sharpen"):
        blurred = cv.GaussianBlur(out, (0, 0), C.SHARPEN_SIGMA)
        # 입력에 (입력 − 흐린 영상) 즉 경계 성분을 SHARPEN_AMOUNT배 더함. uint8 범위는 addWeighted가 0~255로 자름
        out = cv.addWeighted(out, 1 + C.SHARPEN_AMOUNT, blurred, -C.SHARPEN_AMOUNT, 0)

    return out
