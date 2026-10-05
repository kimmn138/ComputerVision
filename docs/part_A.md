# 파트 A 기록

## 파라미터 변경

| 날짜 | 이름 | 전 | 후 | 이유 | 확인한 영상 |
|---|---|---|---|---|---|
| 2026-10-05 | NOISE_SIGMA_MAX, CONTRAST_STD_MIN, BLUR_VAR_MAX (실제 적용값) | 5.0, 35, 100 (choose_steps가 config에 없는 이름 NOISE_TH 등을 읽어 함수 안 기본값이 쓰임) | 3.0, 20, 50 (config 값) | config를 바꿔도 결과가 그대로이고 config_used.txt가 실제와 달랐음. config 값을 읽게 고침 | 합성 노면, test_a |
| 2026-10-05 | USE_CLAHE (대비 부족일 때 tone) | 항상 clahe | USE_CLAHE=False → equalize(평활화), True → clahe | config 주석(표준편차 < 20 → 평활화, True면 CLAHE)대로 | test_a |
| 2026-10-05 | 감마 적용 방향, GAMMA_RANGE | 결과 = 입력^(1/감마), 범위 자르기 없음(감마 3.88까지 나옴) | 결과 = 입력^감마, 감마를 (0.4, 2.5)로 자름 | 계산(감마 < 1 = 밝게)과 적용 방향이 반대라 어두운 노면이 평균 35 → 0으로 더 어두워졌음 | 합성 노면 평균 35·50·69·200·230 → 115·127·128·138·194 |
| 2026-10-05 | GAUSS_SIGMA, SHARPEN_AMOUNT (실제 적용값) | σ 0(커널 5에서 자동, 약 1.1), 0.5 | 1.0, 3.0 (config 값) | 함수 안 숫자 대신 config 값. 샤프닝 식은 언샤프 마스크(입력 + AMOUNT × (입력 − 가우시안(σ)))로 정하고 config 주석에 적음 | test_a |

## 알고리즘 설명과 설정 근거 (보고서·발표용)
