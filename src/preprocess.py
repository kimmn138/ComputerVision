"""고른 전처리 단계(잡음 제거·밝기/대비 보정·샤프닝)만 흑백 ROI 영상에 적용한다."""
import cv2 as cv
import numpy as np

from src import config as C


def preprocess(gray, steps):
    """검출이 잘 되도록 고른 단계만 적용한 흑백 영상을 돌려준다(NO_STEPS면 입력과 같은 영상).
    순서: 잡음 제거(가우시안, 야간은 NLM) → 감마·평활화·CLAHE → 샤프닝(언샤프 마스크)."""
    out = gray.copy()

    # 1. 잡음 제거: 야간 영상(NIGHT_DENOISE를 켜고 밝게 하는 감마를 함께 고름)은 경계를 덜 뭉개는 NLM, 나머지는 가우시안
    if steps.get("denoise"):
        night = C.NIGHT_DENOISE and steps.get("tone") == "gamma" and steps.get("gamma", 1.0) < 1
        if night:
            out = cv.fastNlMeansDenoising(out, None, h=C.NIGHT_DENOISE_H, templateWindowSize=C.NIGHT_NLM_TEMPLATE,
                                          searchWindowSize=C.NIGHT_NLM_SEARCH)
        else:
            out = cv.GaussianBlur(out, C.GAUSS_KSIZE, C.GAUSS_SIGMA)

    # 2. 밝기·대비 보정
    tone = steps.get("tone")
    if tone == "gamma":
        g_val = steps.get("gamma", 1.0)
        # 결과 = 입력^감마 (밝기를 0~1로 보고 계산, 256칸 표로 한 번에 적용)
        table = np.clip(np.round((np.arange(256) / 255.0) ** g_val * 255), 0, 255).astype(np.uint8)
        out = cv.LUT(out, table)

    elif tone == "equalize":
        out = cv.equalizeHist(out)

    elif tone == "clahe":
        clahe = cv.createCLAHE(clipLimit=C.CLAHE_CLIP, tileGridSize=C.CLAHE_TILE)
        out = clahe.apply(out)

    # 3. 샤프닝
    if steps.get("sharpen"):
        blurred = cv.GaussianBlur(out, (0, 0), C.SHARPEN_SIGMA)
        # 언샤프 마스크: 입력 + AMOUNT × (입력 − 흐린 영상), 즉 경계 성분을 키움
        out = cv.addWeighted(out, 1 + C.SHARPEN_AMOUNT, blurred, -C.SHARPEN_AMOUNT, 0)

    return out
