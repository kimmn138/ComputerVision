"""(파트 A) 노면 영상의 상태(밝기·대비·흐림·잡티)를 재고, 그 상태에 맞는 전처리 단계를 고른다."""
import cv2 as cv
import numpy as np

from src import config as C

# 전처리 끔: 아무 단계도 적용하지 않는다. 실험의 '전처리 끔'과 약속 검사가 이 값을 쓴다.
NO_STEPS = {"denoise": False, "tone": None, "gamma": 1.0, "sharpen": False}


def measure_quality(gray):
    """전처리를 고르는 근거로 노면 영상의 밝기(mean)·대비(std)·흐림(lap_var)·잡티(noise)를 잰다."""
    mean_val = float(gray.mean())
    std_val = float(gray.std())
    
    # 라플라시안 분산 (흐림도 측정)
    lap_var = float(cv.Laplacian(gray, cv.CV_64F).var())
    
    # Immerkær 잡음 추정 (영상 구조에 영향을 덜 받는 정밀 노이즈 측정법)
    kernel = np.array([[ 1, -2,  1],
                       [-2,  4, -2],
                       [ 1, -2,  1]], dtype=np.float32)
    filtered = cv.filter2D(gray.astype(np.float32), -1, kernel)
    noise = float(np.mean(np.abs(filtered)) * np.sqrt(0.5 * np.pi) / 6.0)

    return {"mean": mean_val, "std": std_val, "lap_var": lap_var, "noise": noise}


def choose_steps(q):
    """상태 지표를 config A 구역의 기준값과 비교해 적용할 전처리 단계를 고른다.
    잡티 > NOISE_SIGMA_MAX → 가우시안, 평균 < DARK_MEAN_MAX 또는 > BRIGHT_MEAN_MIN → 감마,
    (밝기는 괜찮고) 표준편차 < CONTRAST_STD_MIN → 평활화(USE_CLAHE면 CLAHE), 라플라시안 분산 < BLUR_VAR_MAX → 샤프닝.
    감마는 평균 밝기를 가운데(0.5)로 옮기는 값 log(0.5) / log(평균/255)이고, preprocess가 결과 = 입력^감마로 적용한다
    (감마 < 1이면 밝게, > 1이면 어둡게). GAMMA_RANGE 밖이면 끝값으로 자른다."""
    steps = dict(NO_STEPS)

    # 1. 노이즈 제거 여부
    steps["denoise"] = bool(q["noise"] > C.NOISE_SIGMA_MAX)

    # 2. 톤/조도 보정 (tone)
    if q["mean"] < C.DARK_MEAN_MAX or q["mean"] > C.BRIGHT_MEAN_MIN:
        steps["tone"] = "gamma"
        # 평균이 0이나 255이면 log가 -inf·0이 되므로 한 단계 안쪽(1~254)으로 제한
        norm_mean = np.clip(q["mean"], 1, 254) / 255.0
        steps["gamma"] = float(np.clip(np.log(0.5) / np.log(norm_mean), *C.GAMMA_RANGE))
    elif q["std"] < C.CONTRAST_STD_MIN:
        steps["tone"] = "clahe" if C.USE_CLAHE else "equalize"

    # 3. 샤프닝 여부
    steps["sharpen"] = bool(q["lap_var"] < C.BLUR_VAR_MAX)

    return steps


def steps_to_text(steps):
    """고른 전처리 단계를 결과 표·그림 제목에 쓸 짧은 글로 바꾼다."""
    if steps == NO_STEPS:
        return "없음"
        
    parts = []
    
    if steps.get("denoise"):
        parts.append("가우시안")
        
    tone = steps.get("tone")
    if tone == "gamma":
        g_val = steps.get("gamma", 1.0)
        parts.append(f"감마 {g_val:.2f}")
    elif tone == "equalize":
        parts.append("평활화")
    elif tone == "clahe":
        parts.append("CLAHE")
        
    if steps.get("sharpen"):
        parts.append("샤프닝")
        
    if not parts:
        return "없음"
        
    return "+".join(parts)
