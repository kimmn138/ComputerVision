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
    """상태 지표를 config A 구역의 기준값과 비교해 적용할 전처리 단계를 고른다."""
    steps = dict(NO_STEPS)
    
    # 1. 노이즈 제거 여부 (config에 없으면 기본값 5.0 사용)
    noise_th = getattr(C, 'NOISE_TH', 5.0)
    steps["denoise"] = bool(q["noise"] > noise_th)
    
    # 2. 톤/조도 보정 (tone)
    mean_low = getattr(C, 'MEAN_LOW_TH', 80.0)
    mean_high = getattr(C, 'MEAN_HIGH_TH', 170.0)
    std_low = getattr(C, 'STD_LOW_TH', 35.0)
    
    if q["mean"] < mean_low or q["mean"] > mean_high:
        steps["tone"] = "gamma"
        # 자동 감마 계산 (0 나누기 예외 처리를 위해 0.05 ~ 0.95 제한)
        norm_mean = max(0.05, min(0.95, q["mean"] / 255.0))
        steps["gamma"] = float(np.log(0.5) / np.log(norm_mean))
    elif q["std"] < std_low:
        steps["tone"] = "clahe"
        
    # 3. 샤프닝 여부 (config에 없으면 기본값 100.0 사용)
    lap_var_th = getattr(C, 'LAP_VAR_TH', 100.0)
    steps["sharpen"] = bool(q["lap_var"] < lap_var_th)
    
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
