"""(파트 A) 노면 영상의 상태(밝기·대비·흐림·잡티)를 재고, 그 상태에 맞는 전처리 단계를 고른다."""

# 전처리 끔: 아무 단계도 적용하지 않는다. 실험의 '전처리 끔'과 약속 검사가 이 값을 쓴다.
NO_STEPS = {"denoise": False, "tone": None, "gamma": 1.0, "sharpen": False}


def measure_quality(gray):
    """전처리를 고르는 근거로 노면 영상의 밝기(mean)·대비(std)·흐림(lap_var)·잡티(noise)를 잰다.
    임시 버전: mean·std만 재고 lap_var·noise는 0.0으로 둔다."""
    # TODO(A): 라플라시안 분산, Immerkær 잡음 추정
    return {"mean": float(gray.mean()), "std": float(gray.std()), "lap_var": 0.0, "noise": 0.0}


def choose_steps(q):
    """상태 지표를 config A 구역의 기준값과 비교해 적용할 전처리 단계를 고른다.
    임시 버전: 항상 NO_STEPS(전처리 없음)를 돌려준다."""
    # TODO(A): config A 구역의 기준값으로 결정
    return dict(NO_STEPS)


def steps_to_text(steps):
    """고른 전처리 단계를 결과 표·그림 제목에 쓸 짧은 글로 바꾼다.
    임시 버전: 항상 "없음"을 돌려준다."""
    # TODO(A): "가우시안+감마 0.53"처럼 표시, NO_STEPS면 "없음"
    return "없음"
