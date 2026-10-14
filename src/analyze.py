"""ROI 영상의 화질(밝기·대비·흐림·잡음)을 재고, 그 상태에 맞는 전처리 단계를 고른다."""
import cv2 as cv
import numpy as np

from src import config as C

# 전처리 미적용: 아무 단계도 적용하지 않는 설정
NO_STEPS = {"denoise": False, "tone": None, "gamma": 1.0, "sharpen": False}


def _estimate_noise(gray):
    """잡음 표준편차를 어림한다(Immerkær 방식). 질감·경계를 잡음으로 세지 않도록 가장 평평한 블록들에서만 잰다.
    0·255에 붙은 픽셀이 많은 블록은 잡음이 잘려 낮게 나오므로 뺀다."""
    kernel = np.array([[ 1, -2,  1],
                       [-2,  4, -2],
                       [ 1, -2,  1]], dtype=np.float32)
    # 응답 크기에 √(π/2)/6을 곱하면 잡음 σ가 됨(가장자리 1px은 커널이 걸쳐 뺌)
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

def _direction_ratio(gray):
    """방향별 에지 세기의 최소 ÷ 최대(blur_dir)를 잰다. 흔들림은 한 방향의 경계만 뭉개서 이 값이 작아진다.
    평평해서 기울기가 거의 없는 영상은 NaN."""
    gray_f = gray.astype(np.float32)
    gx = cv.Sobel(gray_f, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray_f, cv.CV_32F, 0, 1, ksize=3)
    xx, yy, xy = float((gx * gx).mean()), float((gy * gy).mean()), float((gx * gy).mean())
    theta = np.deg2rad(np.asarray(C.BLUR_DIR_ANGLES, np.float64))
    energy = xx * np.cos(theta) ** 2 + yy * np.sin(theta) ** 2 + 2.0 * xy * np.sin(theta) * np.cos(theta)
    if energy.max() <= 1e-6:
        return float("nan")
    return float(energy.min() / energy.max())

def measure_quality(gray):
    """전처리를 고르는 근거가 되는 화질 지표를 잰다.
    mean(밝기)·std(대비)·lap_var(라플라시안 분산, 선명도)·noise(잡음)·blur_ratio(재흐림 비율, 흐림)·blur_dir(방향 비율, 흔들림)"""

    mean_val = float(gray.mean())
    std_val = float(gray.std())

    # 흐림 지표 1: 라플라시안 분산(작을수록 흐림)
    lap_var = float(
        cv.Laplacian(
            gray,
            cv.CV_64F
        ).var()
    )

    # 잡음 추정
    noise = _estimate_noise(gray)

    # 원본의 Sobel 평균
    grad_original = _mean_gradient(gray)

    # 흐림 지표 2: 일부러 한 번 더 흐려 봄(이미 흐린 사진은 기울기가 별로 줄지 않음)
    reblurred = cv.GaussianBlur(
        gray,
        (0, 0),
        C.BLUR_RATIO_SIGMA
    )

    grad_reblurred = _mean_gradient(reblurred)

    # 평평한 영상은 기울기가 거의 0이라 비율을 구할 수 없음
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
        "blur_dir": _direction_ratio(gray),
    }

def choose_steps(q):
    """화질 지표를 config 기준값과 비교해 적용할 전처리 단계를 고른다.
    잡음이 크면 가우시안, 너무 어둡거나 밝으면 감마, 대비가 낮으면 평활화(또는 CLAHE), 흐리면 샤프닝.
    감마는 평균 밝기를 가운데(0.5)로 옮기는 값이다(감마 < 1이면 밝게, > 1이면 어둡게)."""
    steps = dict(NO_STEPS)

    # 1. 잡음 제거 여부
    steps["denoise"] = bool(q["noise"] > C.NOISE_SIGMA_MAX)

    # 2. 밝기·대비 보정
    if q["mean"] < C.DARK_MEAN_MAX or q["mean"] > C.BRIGHT_MEAN_MIN:
        steps["tone"] = "gamma"
        # 평균이 0이나 255이면 log가 -inf·0이 되므로 한 단계 안쪽(1~254)으로 제한
        norm_mean = np.clip(q["mean"], 1, 254) / 255.0
        steps["gamma"] = float(np.clip(np.log(0.5) / np.log(norm_mean), *C.GAMMA_RANGE))
    elif q["std"] < C.CONTRAST_STD_MIN:
        steps["tone"] = "clahe" if C.USE_CLAHE else "equalize"
    # 3. 샤프닝 여부
    # 라플라시안 분산 기준
    blur_by_laplacian = (
        q["lap_var"] < C.BLUR_VAR_MAX
    )
    # 재흐림 비율 기준
    blur_ratio = q.get(
        "blur_ratio",
        float("nan")
    )
    blur_by_ratio = (
        np.isfinite(blur_ratio)
        and blur_ratio < C.BLUR_RATIO_MAX
    )
    # 방향별 흐림 기준(USE_BLUR_DIR일 때만): 흔들림은 한 방향만 뭉개 blur_dir이 작다
    blur_dir = q.get("blur_dir", float("nan"))
    blur_by_dir = bool(
        C.USE_BLUR_DIR
        and np.isfinite(blur_dir)
        and blur_dir < C.BLUR_DIR_MIN
    )
    # 하나라도 흐림이면 샤프닝
    steps["sharpen"] = bool(
        blur_by_laplacian or blur_by_ratio or blur_by_dir
    )

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
