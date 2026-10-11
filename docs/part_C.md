# 파트 C 기록

## 파라미터 변경

| 날짜 | 이름 | 전 | 후 | 이유 | 확인한 영상 |
|---|---|---|---|---|---|
| 2026-10-04 | HARRIS_NMS | (없음) | 3 | 해리스 비최대 억제 창. 3×3이면 붙은 코너도 따로 남김 | 합성 사각형(코너 4개 → 4) |
| 2026-10-04 | RANSAC_MIN_GOOD | (없음, 수학적 최소 4) | 10 | good이 4~9개면 호모그래피가 거의 다 맞아 inlier 비율이 1.0 가까이 부풀려짐 | 합성 사각형(good 4 → inliers 0), 무늬 영상 |
| 2026-10-04 | RANSAC_MIN_GOOD | 10 | 4 | 호모그래피는 최소 4쌍이면 계산되므로 수학적 최소값으로 맞춤(발표 설명과 일치). 4~9쌍이면 비율이 1.0에 가깝게 나오므로 결과 표에서 good 열을 같이 봄 | 합성 사각형(good 0 → inliers 0, good 4 → RANSAC 실행) |
| 2026-10-04 | COMPARE_FIGSIZE, COMPARE_DPI | (없음) | (19.2, 9.0), 100 | 비교 그림 가로 1920px = 3열 × 640px, 축소 보간으로 1px 캐니 에지가 뭉개지지 않게 | 합성 어두운 노면 6칸(에지 선 끊김 없음) |
| 2026-10-04 | FONT_CANDIDATES | (없음) | Malgun Gothic, AppleGothic, NanumGothic | 한글 제목이 □로 깨지지 않게 윈도·맥·리눅스 글꼴 후보. 설치된 것만 씀 | 윈도(맑은 고딕): 글꼴 경고 0건 |
| 2026-10-04 | FONT_CANDIDATES → FONT_BY_PLATFORM, FONT_DEFAULT | Malgun Gothic, AppleGothic, NanumGothic | win32: Malgun Gothic, darwin: AppleGothic, 그 밖: DejaVu Sans | 팀 기준: sys.platform으로 글꼴을 하나로 정함(그 밖 OS는 한글이 □로 나옴) | 윈도: 30장 연속 저장, 글꼴 경고 0건 |
| 2026-10-04 | DEMO_SCREEN_FALLBACK, DEMO_SCREEN_MARGIN | (없음) | (1280, 720), 0.9 | 시연 창이 화면보다 크면 잘리므로 화면의 90% 안에 끔·켬 두 장을 같은 비율로 줄여 넣음. 화면 크기를 못 읽으면 1280×720으로 가정 | 합성 세로 영상 640×1200, 화면 1707×960 → 창 878×864 |
| 2026-10-04 | DEMO_LABEL_HEIGHT, DEMO_FONT_SIZE | (없음) | 40, 22 | 패널 위 제목 띠(영상을 가리지 않음). 가장 좁은 패널 463px에도 '전처리 켬: 가우시안+감마 0.53'이 한 줄에 들어감 | 합성 회색 패널 463px |
| 2026-10-10 | USE_ROAD_MASK, ROAD_KEEP_RULE | (없음) | False, "center" | pipeline의 노면 마스크 켬·끔과 노면 안 후보 규칙(Plan.md 3.3 (7)). 끄면 road를 검출 함수에 넘기지 않고 후보도 지우지 않아 1차와 같음. E1에서 채택되면 True로 바꾸고 이 표에 적는다 | 39장(끔·켬, 5번 반복)·804장(끔·켬) 메모리 비교: row·후보·요약이 main과 같음. True로 켜도(임시 마스크 전부 255) 같음 |
| 2026-10-10 | DEGRADE_BLUR_SIGMA, DEGRADE_DARK_GAMMA, DEGRADE_BRIGHT_GAMMA, DEGRADE_NOISE_SIGMA, DEGRADE_SEED | (없음) | 2.0, 2.2, 0.5, 10.0, 0 | 인위 저하 실험 E6 값(Plan.md 3.2 (6)). evaluate_train --degrade에서 쓸 값(C 작업) | 아직 쓰이지 않음. 참고 구현으로 128 → dark 57·bright 181 확인(test_c의 @todo 시험) |
| 2026-10-11 | DEGRADE_SEED (쓰는 방식) | 0 (모든 사진에 같은 seed) | 0 + 사진 바이트의 crc32 (값 0은 그대로) | 크기가 같은 사진에 모두 같은 잡음 무늬가 들어가 결과가 그 무늬 하나에 좌우될 수 있었음. 같은 사진은 늘 같은 무늬 | 학습 사진 Japan 2장(640×640): 같은 사진 두 번 같음, 두 사진의 잡음 값이 같은 픽셀 2.8%(독립인 잡음의 우연 일치 수준), 잡음 표준편차 10.01. dark는 감마 표 반올림으로 128 → 56(시험 기대 57±1 안) |
| 2026-10-11 | ROAD_OVERLAY_COLOR, ROAD_OVERLAY_ALPHA | (없음) | (0, 255, 0), 0.35 | 시연·비교 그림에서 노면 마스크를 반투명 초록으로 칠함(draw_road, Plan.md 3.7). 초록은 균열(빨강)·포트홀(파랑)·노면 시작선(노랑)과 헷갈리지 않고, 0.35는 노면 결과 후보 윤곽이 비쳐 보이는 옅기 | 회색 128 → (83, 172, 83). demo 화면 저장으로 Japan_013055·United_States_004830(띠 아래 0.95~1.0은 칠하지 않음)·own_03_pothole_blur(세로)·--roi auto --mask on, run_experiment 비교 그림 United_States_004952를 눈으로 확인 |
| 2026-10-12 | USE_ROAD_MASK | False | True | 비교 단계에서 E3(dark + 노면 마스크, 띠 0.5, σ 노면 안)를 사람이 채택(--accept E3). 마스크만 바꾼 E1←E0: 균형 정확도 9.5%→11.7%, 영상당 오검출 8.08→5.00, 우연 뺀 균열 재현율 +1.9%p→+2.0%p. dark에서 마스크만 바꾼 E3←E2(참고): 균형 정확도 27.4%→30.3%, 오검출 6.89→5.21 | 개발셋 652장(전처리 끔, results/report/summary.md) |
| 2026-10-12 | DETECT_MODE (B 구역, B 동의) | "legacy" | "dark" | E3 채택(--accept E3). 검출만 바꾼 E3←E1: 민감도 11.4%→50.6%, 영상당 오검출 5.00→5.21(거의 같음), 균형 정확도 11.7%→30.3%, 우연 뺀 균열 재현율 +2.0%p→+12.9%p. E0(1차)와 견주면 민감도 11.9%→50.6%, 오검출 8.08→5.21로 둘 다 나아져 규칙 1도 통과로 봄. DARK_SIGMA_IN_ROAD는 True 그대로(E3σ 되돌림) | 개발셋 652장(전처리 끔, results/report/summary.md) |

## 알고리즘 설명과 설정 근거 (보고서·발표용)
