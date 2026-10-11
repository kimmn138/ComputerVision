"""노면 영상의 상태(밝기·대비·흐림·잡티)를 재고, 그에 맞는 전처리 단계를 고른다."""
import cv2 as cv
import numpy as np

from src import config as C

# 전처리 끔: 아무 단계도 적용하지 않음
NO_STEPS = {"denoise": False, "tone": None, "gamma": 1.0, "sharpen": False}


def _estimate_noise(gray):
    """Immerkær 방식으로 잡음 표준편차를 어림한다.
    결·경계를 잡음으로 세지 않도록, 블록별로 잰 값 중 가장 평평한 블록들의 값을 쓴다."""
    kernel = np.array([[ 1, -2,  1],
                       [-2,  4, -2],
                       [ 1, -2,  1]], dtype=np.float32)
    # |응답| 평균에 √(π/2)/6을 곱하면 잡음 σ. 가장자리 1px은 뺌
    resp = np.abs(cv.filter2D(gray.astype(np.float32), -1, kernel))[1:-1, 1:-1] * (np.sqrt(0.5 * np.pi) / 6.0)
    if resp.size == 0:
        return 0.0

    b = C.NOISE_BLOCK
    h, w = (resp.shape[0] // b) * b, (resp.shape[1] // b) * b
    if h == 0 or w == 0:                                # 블록 하나보다 작은 영상은 전체 평균
        return float(resp.mean())

    def block_mean(a):
        return a[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))

    inner = gray[1:-1, 1:-1]
    est = block_mean(resp)
    saturated = block_mean(((inner == 0) | (inner == 255)).astype(np.float32))
    usable = est[saturated <= C.NOISE_SAT_MAX]
    return float(np.quantile(usable if usable.size else est, C.NOISE_FLAT_QUANTILE))

def _mean_gradient(gray):
    """Sobel로 영상의 평균 기울기 세기를 계산한다."""

    gray_f = gray.astype(np.float32)

    gx = cv.Sobel(
        gray_f,
        cv.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv.Sobel(
        gray_f,
        cv.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv.magnitude(gx, gy)

    return float(magnitude.mean())

def measure_quality(gray):
    """전처리를 고르는 근거가 되는 밝기·대비·흐림·잡티 지표를 잰다.
    blur_ratio = 원본 기울기 ÷ 한 번 더 흐린 영상의 기울기 (흐린 사진일수록 1에 가까움)."""

    mean_val = float(gray.mean())
    std_val = float(gray.std())

    # 흐림 지표 1: 라플라시안 분산
    lap_var = float(
        cv.Laplacian(
            gray,
            cv.CV_64F
        ).var()
    )

    # 잡음 추정
    noise = _estimate_noise(gray)

    # 흐림 지표 2: 재흐림 비율
    grad_original = _mean_gradient(gray)

    reblurred = cv.GaussianBlur(
        gray,
        (0, 0),
        C.BLUR_RATIO_SIGMA
    )

    grad_reblurred = _mean_gradient(reblurred)

    # 평평한 영상은 기울기가 0에 가까워 비율이 의미 없음
    if grad_reblurred <= 1e-6:
        blur_ratio = float("nan")
    else:
        blur_ratio = float(
            grad_original / grad_reblurred
        )

    return {
        "mean": mean_val,
        "std": std_val,
        "lap_var": lap_var,
        "noise": noise,
        "blur_ratio": blur_ratio,
    }

def choose_steps(q):
    """상태 지표를 config 기준값과 비교해 적용할 전처리 단계를 고른다.
    잡티 많음 → 가우시안, 너무 어둡거나 밝음 → 감마, 대비 부족 → 평활화, 흐림 → 샤프닝."""
    steps = dict(NO_STEPS)

    # 1. 잡티 제거
    steps["denoise"] = bool(q["noise"] > C.NOISE_SIGMA_MAX)

    # 2. 밝기·대비 보정 (감마는 평균 밝기를 가운데로 옮기는 값)
    if q["mean"] < C.DARK_MEAN_MAX or q["mean"] > C.BRIGHT_MEAN_MIN:
        steps["tone"] = "gamma"
        # 평균이 0·255면 log가 발산하므로 1~254로 제한
        norm_mean = np.clip(q["mean"], 1, 254) / 255.0
        steps["gamma"] = float(np.clip(np.log(0.5) / np.log(norm_mean), *C.GAMMA_RANGE))
    elif q["std"] < C.CONTRAST_STD_MIN:
        steps["tone"] = "clahe" if C.USE_CLAHE else "equalize"
    # 3. 샤프닝: 두 흐림 지표 중 하나라도 흐림이면 적용
    blur_by_laplacian = (
        q["lap_var"] < C.BLUR_VAR_MAX
    )
    blur_ratio = q.get(
        "blur_ratio",
        float("nan")
    )
    blur_by_ratio = (
        np.isfinite(blur_ratio)
        and blur_ratio < C.BLUR_RATIO_MAX
    )
    steps["sharpen"] = bool(
        blur_by_laplacian or blur_by_ratio
    )

    return steps


def steps_to_text(steps):
    """전처리 단계를 표·그림 제목에 쓸 짧은 글로 바꾼다(예: 가우시안+감마 0.53)."""
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
