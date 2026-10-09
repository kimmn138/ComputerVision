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

## 알고리즘 설명과 설정 근거 (보고서·발표용)
