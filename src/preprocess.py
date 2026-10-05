"""(파트 A) 고른 전처리 단계(잡티 제거·밝기/대비 보정·샤프닝)만 흑백 노면 영상에 적용한다."""
import cv2 as cv
import numpy as np


def preprocess(gray, steps):
    """검출이 잘 되도록 고른 단계만 적용한 흑백 영상을 돌려준다. NO_STEPS면 입력과 똑같은 영상."""
    out = gray.copy()

    # 1. 잡티 제거 (Denoise)
    if steps.get("denoise"):
        out = cv.GaussianBlur(out, (5, 5), 0)

    # 2. 톤/조도 보정 (Tone)
    tone = steps.get("tone")
    if tone == "gamma":
        g_val = steps.get("gamma", 1.0)
        # 0 나누기 방지를 위해 최소값 0.01로 제한
        inv_gamma = 1.0 / max(0.01, g_val)
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype(np.uint8)
        out = cv.LUT(out, table)

    elif tone == "equalize":
        out = cv.equalizeHist(out)

    elif tone == "clahe":
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(out)

    # 3. 샤프닝 (Sharpen)
    if steps.get("sharpen"):
        blurred = cv.GaussianBlur(out, (0, 0), 3)
        # 원본을 1.5배 강조하고 블러 성분을 -0.5배 빼서 경계를 날카롭게 만듦
        out = cv.addWeighted(out, 1.5, blurred, -0.5, 0)

    return out
