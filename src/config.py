"""모든 파라미터. 구역마다 주인이 있고, 자기 구역 안에서만 고친다.
구역 사이의 빈 줄과 주석 줄은 지우지 않는다(Git이 구역별 수정을 자동으로 합치게 해 줌)."""

# ===== 공통 (C 관리 · 바꾸려면 세 명 합의) =====
TARGET_WIDTH = 640          # 모든 영상을 가로 640px로
ROI_TOP_DEFAULT = 0.5       # roi.csv에 없는 사진의 노면 시작 비율
ROI_BOTTOM_DEFAULT = 1.0
TIME_REPEAT = 5             # 처리 시간은 5번 재서 중앙값

# ===== 화질 분석·전처리 =====
DAMAGE_TAGS = ("crack", "pothole", "both", "none")      # 직접 촬영 사진 이름의 '손상' 칸에 쓸 수 있는 값
CONDITION_TAGS = ("normal", "dark", "night", "bright", "shadow", "blur", "wet", "clutter", "none")   # '조건' 칸 값
PAIR_ROLES = ("A", "B", "C", "D")                       # 매칭 쌍 사진의 기호(A = 기준 사진)
PAIR_CONDITIONS = ("ref", "dark", "blur", "view")       # 매칭 쌍 사진의 조건 칸 값
BLUR_VAR_MAX = 50           # 라플라시안 분산이 이보다 작으면 흐림 → 샤프닝
NOISE_SIGMA_MAX = 3.0       # 잡음 σ가 이보다 크면 가우시안으로 잡음 제거
CONTRAST_STD_MIN = 20       # 밝기 표준편차가 이보다 작으면 대비 부족 → 평활화
DARK_MEAN_MAX = 80          # 평균 밝기가 이보다 작으면 감마로 밝게
BRIGHT_MEAN_MIN = 170       # 평균 밝기가 이보다 크면 감마로 어둡게
GAUSS_KSIZE, GAUSS_SIGMA = (5, 5), 1.0      # 잡음 제거 가우시안의 커널 크기(px)·σ
GAMMA_RANGE = (0.4, 2.5)    # 감마 값을 이 범위로 자름(결과 = 입력^감마, 1보다 작으면 밝게)
SHARPEN_AMOUNT, SHARPEN_SIGMA = 3.0, 3.0    # 언샤프 마스크: 결과 = 입력 + AMOUNT × (입력 − 가우시안(σ px))
USE_CLAHE = False           # True면 평활화 대신 CLAHE(구역별 평활화)
CLAHE_CLIP, CLAHE_TILE = 2.0, (8, 8)        # CLAHE 대비 제한·격자 크기
NOISE_BLOCK = 16            # px, 잡음을 잴 블록 크기
NOISE_FLAT_QUANTILE = 0.05  # 가장 평평한 하위 5% 블록으로 잡음을 잼(질감·경계를 잡음으로 세지 않게)
NOISE_SAT_MAX = 0.05        # 비율, 0·255 픽셀이 이보다 많은 블록은 잡음이 낮게 재져서 뺌
# 노면 마스크
ROAD_BLUR_SIGMA = 3         # px, 색을 비교하기 전 가우시안 σ(골재 질감이 색 비교를 흔들지 않게)
ROAD_SEED_BOX = (0.30, 0.40, 0.70, 0.80)    # 비율 (가로 시작, 세로 시작, 가로 끝, 세로 끝), ROI 안 시드 영역(맨 아래 보닛은 피함)
ROAD_CHROMA_MAX = 10        # Lab a·b 거리, 시드 색과 이보다 가까우면 노면 색
ROAD_L_K = 4                # 배, 시드 밝기 ± ROAD_L_K × (1.4826 × MAD + 2) 안이면 노면 밝기
ROAD_OPEN_KSIZE = 5         # px, 노면 후보의 작은 점을 지우는 열림 커널
ROAD_CLOSE_KSIZE = 31       # px, 노면 마스크의 틈(균열·차선 조각)을 메우는 닫힘 커널
BLUR_RATIO_MAX = 1.5        # 재흐림 비율이 이보다 작으면 흐림 → 샤프닝
BLUR_RATIO_SIGMA = 1.5      # px, 재흐림 비율을 잴 때 일부러 거는 가우시안 σ
# 노면 마스크: 차선 처리
ROAD_LANE_BRIDGE = False    # True면 밝고 가는 흰·노란 줄(차선)을 모두 노면으로 다룸(차선이 노면을 둘로 가르지 않게)
ROAD_LANE_KSIZE = 41        # px, 가로나 세로로 이 폭보다 가는 밝은 줄을 차선으로 봄(넓은 하늘·벽은 빠짐)
ROAD_LANE_DELTA = 20        # 밝기 단계(0~255), 주변보다 이만큼 이상 밝아야 차선
ROAD_LANE_WHITE_CHROMA = 12 # Lab a·b 거리, 무채색과 이보다 가까우면 흰 차선 색
ROAD_LANE_YELLOW_B = 20     # Lab b − 128, 이 값 이상이면 노란 중앙선 색
ROAD_CLOSE_BEFORE_SEED = False  # True면 시드와 이어진 영역을 고르기 전에 닫힘 연산으로 차선 건너편 노면을 이음
# 흔들림 판정·야간 잡음 제거
USE_BLUR_DIR = False        # True면 방향별 에지 비율(blur_dir)로 흔들림도 흐림으로 보고 샤프닝
BLUR_DIR_MIN = 0.5          # 비율(가장 약한 방향 ÷ 가장 강한 방향), 이보다 작으면 흔들림
BLUR_DIR_ANGLES = (0, 45, 90, 135)  # 도, 에지 세기를 잴 방향(0 = 가로, 90 = 세로)
NIGHT_DENOISE = False       # True면 야간 영상(잡음 제거와 밝게 하는 감마를 함께 고름)에 가우시안 대신 NLM
NIGHT_DENOISE_H = 10        # 밝기 단계, NLM 잡음 제거 세기(잡음 σ와 비슷하게)
NIGHT_NLM_TEMPLATE, NIGHT_NLM_SEARCH = 7, 21    # px, NLM 비교 조각·탐색 창 크기(OpenCV 기본값)
# 노면 마스크: 그림자·보닛·건물 처리
ROAD_SHADOW_L_K = 0         # 배, 0보다 크면 색이 시드와 아주 가까운 픽셀의 밝기 허용 폭을 이 배수로 넓힘(그림자·햇빛)
ROAD_SHADOW_CHROMA = 5      # Lab a·b 거리, 위에서 '색이 아주 가까움'의 기준
ROAD_HOOD_CUT = False       # True면 ROI 아래 테두리의 매끈한 행(보닛·와이퍼)을 노면에서 뺌
ROAD_HOOD_TEX_KSIZE = 7     # px, 보닛 판정에 쓸 질감(국소 표준편차) 창 크기
ROAD_HOOD_TEX_RATIO = 0.5   # 배, 행의 질감이 시드 영역 질감 × 이 값보다 작으면 매끈한 행
ROAD_EDGE_DENSITY = False   # True면 에지가 시드 영역보다 훨씬 빽빽한 곳(건물·인도)을 노면에서 뺌
ROAD_EDGE_KSIZE = 15        # px, 에지 밀도를 잴 창 크기
ROAD_EDGE_QUANTILE = 0.95   # 분위, 시드 영역 에지 밀도의 이 분위보다 크면 노면 아님
ROAD_EDGE_CANNY = (50, 150) # 에지 밀도에 쓸 Canny 낮은·높은 임계값

# 노면 마스크: 시드 영역 정리·끊긴 노면 잇기
ROAD_SEED_FILTER = True     # True면 시드 영역에서 노면답지 않은 픽셀(채도가 유독 높은 풀·차, 흰 선, 노란 선)을 뺌
ROAD_SEED_CHROMA_REL = 8.3  # Lab 채도 차, ROI 아래쪽 채도 중앙값보다 이만큼 넘게 높은 픽셀을 시드에서 뺌
ROAD_SEED_REF_BOX = (0.05, 0.35, 0.95, 0.90)    # 비율 (가로 시작, 세로 시작, 가로 끝, 세로 끝), 위 채도 중앙값을 잴 ROI 아래쪽 범위
ROAD_SEED_KEEP_MIN = 0.25   # 비율, 빼고 남은 시드가 이보다 적으면 빼지 않음(시드가 사라지지 않게)
ROAD_L_RANGE_MAX = 107.9    # 밝기 단계(0~255), 노면 밝기 허용 폭의 상한(0이면 상한 없음)
ROAD_KEEP_BLOBS = True      # True면 시드와 떨어진 덩어리도 크거나 ROI 아래쪽에 닿으면 노면으로 남김(흰 선에 막힌 건너편)
ROAD_BLOB_SMALL = 0.005     # ROI 넓이 비율, 아래쪽에 닿은 덩어리를 남길 최소 넓이
ROAD_BLOB_BIG = 0.05        # ROI 넓이 비율, 위치와 상관없이 남길 덩어리 넓이
ROAD_BLOB_BOTTOM = 0.25     # 높이 비율, 'ROI 아래쪽에 닿음'으로 볼 범위
ROAD_LANE_AS_ROAD = True    # True면 노면 위의 가늘고 긴 흰·노란 줄(차선)을 노면에 더함
ROAD_LANE_MIN_AREA = 30     # px, 차선으로 볼 줄 덩어리의 최소 넓이
ROAD_LANE_ELONG = 3.0       # 배, 긴 변 ÷ 짧은 변이 이 이상이어야 가늘고 긴 줄
ROAD_LANE_RING = 9          # px, 줄 양옆의 노면을 확인할 둘레 고리 폭
ROAD_LANE_RING_ROAD = 0.5   # 비율, 둘레 고리의 이 이상이 노면 색이어야 노면 위의 차선
ROAD_LANE_EDGE = 6          # px, 더한 줄을 넓혀 흐린 줄 가장자리의 틈을 메우는 폭

# ===== B: 균열·포트홀·평가 =====
CANNY_LOW, CANNY_HIGH = 50, 150
CRACK_CLOSE_KSIZE = 5
CRACK_MIN_AREA = 80         # 픽셀 수
CRACK_MIN_ELONG = 4.0       # 길쭉함(긴 변 / 짧은 변). 이 값 이상이거나
CRACK_MAX_FILL = 0.1        # 채움 정도(윤곽 넓이 / 회전 사각형 넓이)가 이 값 이하면 균열 (곧은 균열은 길쭉함, 구불·갈래 균열은 채움으로 잡음)
DOG_SIGMAS = (3, 12)        # (작은 σ, 큰 σ)
DOG_THRESH = 12
POT_OPEN_KSIZE = 5
POT_MIN_AREA = 150
POT_MAX_AREA_RATIO = 0.2    # ROI 넓이 대비
POT_MAX_ELONG = 3.0
POT_MIN_SOLIDITY = 0.6
IOU_THRESH = 0.3            # 정답 비교 기준
TRAIN_IMAGE_DIR = "data/train/img"  # 경로(맨 위 폴더 기준), 교수 제공 훈련 사진 폴더 (convert_json_gt·evaluate_train)
TRAIN_ANN_DIR = "data/train/ann"    # 경로, 훈련 사진의 JSON 정답(<사진 이름>.jpg.json) 폴더 (convert_json_gt)
TRAIN_GT_DIR = "gt/train"           # 경로, JSON을 바꾼 정답 CSV 폴더 (convert_json_gt가 쓰고 evaluate_train이 읽음)
DARK_RING_PX = 4            # px, 후보의 어둡기를 잴 주변 띠 두께. 어둡기 = 띠의 어두운 쪽 밝기 − 후보 안 평균
DARK_RING_QUANTILE = 0.25   # 비율, 띠에서 어두운 쪽(하위 25%)과 비교. 평균을 쓰면 넓은 흰 차선의 가장자리가 걸러지지 않음
CRACK_MIN_DARKNESS = -5     # 밝기 단계(0~255), 균열 후보의 어둡기 하한. 주변보다 밝은 선(흰 차선·횡단보도 점선)을 거름 (훈련 804장 crack_loose 끔: 후보 3300→723, IoU 정밀도 0.015→0.039, 우연을 뺀 IoU 재현율 0.022 유지)
# 아래는 10/10 기반 작업(C)이 Plan.md 3.4·3.5 값으로 넣은 출발값. B가 WP2·WP3에서 확정하고 docs/part_B.md에 적는다
DETECT_MODE = "dark"        # 검출 방식: "legacy" = 1차 방식(Canny 균열 + DoG 포트홀), "dark" = 어두운 영역 한 번 추출 + 형태 분류. E3 채택(10/12)으로 "dark", B 동의를 받아 C가 바꿈 (Plan.md 3.10, docs/part_C.md)
DARK_BG_DOWNSCALE = 4       # 배, 배경을 잴 때 흑백 띠를 1/4로 줄여 중앙값을 빠르게 구함 (Plan.md 3.4 (2) 1·2단계)
DARK_BG_KSIZE_SMALL = 11    # px(1/4 영상 기준), 밝은 표시 억제용 거친 배경의 중앙값 창. 원래 크기로 약 44px (1단계)
DARK_BRIGHT_DELTA = 25      # 밝기 단계(0~255), 거친 배경보다 이만큼 밝은 픽셀(흰 차선·횡단보도)을 배경값으로 메움 (1단계)
DARK_BG_FRAC = 0.5          # 비율, 작은 배경 창 = 띠 높이 × 0.5 (2단계)
DARK_BG_FRAC_LARGE = 1.0    # 비율, 큰 배경 창 = 띠 높이 × 1.0. 두 창의 큰 값을 써서 큰 포트홀 가운데가 비지 않게 (2단계, 합성 반지름 90: 한 창 43% → 두 창 100%)
DARK_SIGMA_MIN = 1.0        # 밝기 단계, 잡음 σ의 하한. σ = 0이면 문턱도 0이 되어 띠 전체가 한 영역이 됨 (3단계, Plan.md R1)
DARK_SIGMA_IN_ROAD = True   # True면 road를 받았을 때 잡음 σ를 노면 안에서만 잼. E3에서 켬·끔을 비교해 B가 정함 (Plan.md 3.4 (1))
DARK_K_HI = 4               # σ의 배수, 강 문턱 (4단계 이중 문턱)
DARK_K_LO = 2               # σ의 배수, 약 문턱 (4단계)
DARK_ABS_MIN = 8            # 밝기 단계, 강 문턱의 하한. σ가 작은 영상에서 잡음이 강 픽셀이 되지 않게 (4단계)
DARK_CLOSE_KSIZE = 5        # px, 끊긴 조각을 잇는 닫힘 타원 커널 (5단계)
DARK_MIN_AREA = 40          # 픽셀 수, 이보다 작은 영역(골재 무늬)은 버림 (5단계)
CRACK_MIN_CIRC_INV = 4      # 비원형도 P²/(4πA), 이 값 이상이면 균열. 폭의 약 12.6배보다 긴 선 (Plan.md 3.5 (2) 출발값, 산점도를 보고 B가 확정)
POT_MAX_CIRC_INV = 2.5      # 비원형도, 이 값 이하이고 solidity ≥ POT_MIN_SOLIDITY면 포트홀. 경계가 거친 원(c 약 3)도 들어가게 여유 (3.5 (2) 출발값)
# 아래는 10/12 밤 보완 1번 점수(C가 B 동의로 넣음, docs/tasks/supplement.md 1-1). SCORE_FORMULA "none"·USE_SCORE_TAU False면 v9와 같다
SCORE_FORMULA = "margin"    # 후보 점수식: "none"(NaN), "margin"(대비 × 분류 문턱 여유), "snr"(대비 × √넓이), "shape"(대비 상한 × 길이·넓이). 10/13 사용자가 margin을 고름: S1a~c 개발셋에서 fp1·fp2 민감도 23.8%·42.3%(shape 14.8%·32.1%, snr 10.2%·30.7%)
SCORE_MARGIN_SCALE = 3.5    # 비원형도 단위, margin 식 균열 여유 1 − e^(−(ci − 4)/k)의 k. 개발셋 균열 후보 (ci − 4) 중앙값 3.531(10/12 밤 측정). 중앙값 거리에서 0.63
SCORE_C_CAP = 4.0           # contrast(σ의 배수), shape 식에서 대비를 이 값에서 자름. 개발셋 균열 후보 contrast 90백분위 4.0(10/12 밤 측정). 진한 그림자·얼룩이 대비만으로 높은 점수를 받지 않게 (Plan.md 1.5)
SCORE_LEN_REF = 90          # px, shape 식 균열 길이(둘레 ÷ 2)가 이 값 이상이면 길이 항 1. 개발셋 균열 후보 길이 중앙값 91.1px
SCORE_AREA_REF = 80         # 픽셀 수, shape 식 포트홀 넓이가 이 값 이상이면 넓이 항 1. 개발셋 포트홀 후보 넓이 중앙값 79
SCORE_MARGIN_SCALE_POT = 0.72   # 비원형도 단위, margin 식 포트홀 여유 1 − e^(−(2.5 − ci)/k)의 k. 개발셋 포트홀 후보 (2.5 − ci) 중앙값 0.716(10/12 밤 측정). 균열과 범위가 달라 따로 둠
USE_SCORE_TAU = True        # 점수 문턱 켬·끔. 10/13 E4 예외 채택(규칙 3 미달, docs/part_C.md)으로 켬. True면 dark 방식에서 score ≥ SCORE_TAU_<종류>인 후보만 남김(SCORE_FORMULA가 none이면 ValueError)
SCORE_TAU_CRACK = 0.304508  # 균열 점수 임계값. v6 운영점 ETY(e6_road6_tau_dev, 노면 A2rA3B1B2 뒤 설정) 개발셋 전처리 미적용 임계값별 성능 곡선의 fp2 점(영상당 오검출 ≤ 2에서 민감도 최대), tau_points 값을 소수 여섯째 자리로 내림. 보완 최종(ET) 0.255937, E4 때 1.537689
SCORE_TAU_POTHOLE = 7.822985  # 포트홀 점수 임계값. 정하는 법은 위와 같음(ETY fp2 점). 보완 최종(ET) 5.136871, E4 때 6.002551. 포트홀은 어느 임계값에서도 재현율 0이라 fp2 점에서 포트홀 후보를 거의 다 지움(docs/part_C.md '10/14 결정 2')
DARK_SIGMA_LOCAL = True     # 보완 2-1(12번) 켬·끔. True면 잡음 σ를 위치마다 주변 창에서 밝은 쪽 잔차로 재어(국소 σ, 하한 = 전역 σ) 결이 거친 노면에서 문턱을 올림. 10/13 양쪽 잔차 판은 큰 포트홀 시험 실패로 되돌리고, 밝은 쪽 잔차 판(안 A)을 채택: 개발셋 fp2 점 E4s 대비 규칙 1~3 통과(균형 33.6 → 38.5%, 우연 뺀 균열 +12.2 → +17.1%p)
DARK_SIGMA_LOCAL_FRAC = 0.25    # 비율, 국소 σ 창 = 띠 높이 × 0.25(배경 창 0.5의 절반). 골재 결 조각(넓이 40~100px)보다 충분히 크고 노면 한 구역의 결을 잼
DARK_SIGMA_LOCAL_MIN_N = 50 # 개, 국소 σ 창 안 밝은 쪽 표본이 이보다 적으면 전역 σ. MAD로 잰 σ의 상대 표준오차 ≈ 1.17/√n이라 50개면 약 17%. 높이 240px 띠의 창(4px 간격 15×15 = 225개, 밝은 쪽 약 106개)에서는 창의 절반 이상이 어두운 손상일 때만 전역 σ로 돌아감
DARK_BRIGHT_DILATE = 6      # px, 보완 2-2(13번). 0이면 끔. 10/13 채택(S12 대비 규칙 1~3 통과: 끔 행 균형 +0.21%p, 켬 행 +1.8%p). 밝은 표시 마스크를 이만큼 팽창해 샤프닝이 밝은 선 옆에 만든 어두운 테두리까지 메움(후보 값 6: preprocess 샤프닝(σ 3, 세기 3)이 밝은 선 옆에 배경보다 8 이상 어둡게 만드는 폭, 선 폭 5·8·12px 모두 6px)
DARK_BRIGHT_DILATE_DELTA = 80   # 밝기 단계, DARK_BRIGHT_DILATE의 팽창 씨앗 = 기준보다 이만큼 밝은 픽셀. 샤프닝한 차선(노면 +60 → +93 이상)과 균열 옆 오버슈트(깊이 60·폭 5 균열에서도 +72 이하, 10/13 측정)를 가름
DARK_BRIGHT_SEED_KSIZE = 5  # px, 팽창 씨앗을 고르기 전 중앙값 필터 창. 샤프닝이 키운 골재 잡무늬 점(1~2px)은 지우고 폭 5px 이상 차선은 남김
DARK_SPLIT_BLOB = False     # 보완 2-3(9c) 켬·끔. True면 원판 열림으로 덩어리(포트홀)와 선(균열)을 나눠 균열이 붙은 포트홀을 두 후보로
DARK_SPLIT_KSIZE = 15       # px, 덩어리를 남길 원판 지름. 균열 폭(대부분 2~8px, 넓은 균열 시험 6~8px)의 약 두 배라 선은 지워지고, 지름 15px 이상 덩어리만 남음
SCORE_USE_EDGE = False      # 보완 2-4(9b) 켬·끔. True면 점수에 경계 날카로움 항 min(1, edge / SCORE_EDGE_REF)를 곱함. 그림자 경계는 흐리고 포트홀·균열 경계는 날카로움(Plan.md 3.4)
SCORE_EDGE_REF = 0.06       # 경계 날카로움(경계 고리 평균 기울기 ÷ 안쪽 평균 어둡기)의 기준. 합성 원(깊이 70)에서 가장자리 흐림 σ 0·1·2·4·8·15px일 때 0.39·0.10·0.06·0.045·0.032·0.026(10/13 측정) → σ 2px까지는 깎지 않고 σ 15px 그림자는 약 0.43배
DARK_BRIGHT_REF = "coarse"  # 보완 2-2(9a). "coarse" = 지금(거친 배경 Bc와 비교·Bc로 메움), "road_median" = 노면 밝기 중앙값 하나와 비교·그 값으로 메움(억제 창보다 넓은 흰 줄무늬)
# 아래는 10/13 보완 5단계(C가 B 동의로 넣음, docs/tasks/supplement.md 5-2~5-4). 스위치를 모두 끄면 그 전과 같다
SCORE_USE_HESSIAN = False   # 5-2 켬·끔. True면 균열 점수에 영역 안 헤시안 선 강조(0~1)를 곱함(가는 어두운 선 > 둥근 골재 무늬)
HESSIAN_SIGMA = 1.5         # px, 5-2 2차 미분 전에 거는 가우시안 σ(계획 출발값, 균열 폭 3~5px의 절반쯤)
DARK_AREA_PERSPECTIVE = False   # 5-3 켬·끔. True면 최소 넓이를 영역 중심의 행 위치로 선형으로 줄임(띠 맨 위 DARK_MIN_AREA_TOP, 맨 아래 DARK_MIN_AREA)
DARK_MIN_AREA_TOP = 20      # 픽셀 수, 5-3 띠 맨 위 행의 최소 넓이(계획 출발값 = DARK_MIN_AREA 40의 절반)
BRIGHT_POTHOLE = False      # 5-4 켬·끔. True면 결(국소 표준편차)이 노면의 BRIGHT_POT_TEX_RATIO배 넘는 포트홀 모양 덩어리를 포트홀 후보로 더함(own_18 자갈 포트홀)
BRIGHT_POT_TEX_RATIO = 2.0  # 배, 5-4 결 비율 문턱(계획 출발값)
BRIGHT_POT_TEX_KSIZE = 7    # px, 5-4 결을 잴 창(노면 마스크·자동 띠와 같은 7×7)

# ===== C: 정량 측정 =====
HARRIS_BLOCK, HARRIS_KSIZE, HARRIS_K, HARRIS_REL = 2, 3, 0.04, 0.01
SIFT_RATIO = 0.7
RANSAC_THRESH = 5.0
HARRIS_NMS = 3              # 픽셀, 3×3 이웃 중 최댓값인 곳만 점 하나로 셈
RANSAC_MIN_GOOD = 4         # 개, 호모그래피 계산에 필요한 최소 대응점 수. 이보다 적으면 RANSAC 생략
COMPARE_FIGSIZE = (19.2, 9.0)   # 인치, 비교 그림 크기. 3열 × 640px을 dpi 100에서 거의 1:1로 보여 1px 에지가 뭉개지지 않게
COMPARE_DPI = 100               # 점/인치, 19.2 × 100 = 가로 1920px
FONT_BY_PLATFORM = {"win32": "Malgun Gothic", "darwin": "AppleGothic"}  # sys.platform별 한글 글꼴(윈도 맑은 고딕, 맥 기본 탑재)
FONT_DEFAULT = "DejaVu Sans"    # 그 밖 운영체제: matplotlib 기본 글꼴(한글은 □로 나옴)
DEMO_SCREEN_FALLBACK = (1280, 720)  # px (가로, 세로), 시연 창: 화면 크기를 못 읽을 때(윈도 밖) 가정하는 가장 흔한 노트북·프로젝터 해상도
DEMO_SCREEN_MARGIN = 0.9        # 비율, 시연 창이 화면의 90%까지만 쓰게. 작업 표시줄·창 제목 줄 몫
DEMO_LABEL_HEIGHT = 40          # px, 시연 창 패널 위 제목 띠 높이. 영상 위에 글자를 덮어 쓰지 않게 띠를 따로 붙임
DEMO_FONT_SIZE = 22             # px, 제목 글자 크기. 세로 사진이 463px까지 줄어도 '전처리 켬: 가우시안+감마 0.53'이 한 줄에 들어가게
USE_ROAD_MASK = True            # 노면 마스크 켬·끔. True면 pipeline이 road를 검출 함수에 넘기고 노면 밖 후보를 지운 뒤 row를 셈. E3 채택(10/12)으로 켬 (Plan.md 3.3 (7), docs/part_C.md)
ROAD_KEEP_RULE = "ratio"        # 노면 안 후보를 고르는 규칙. "center" = bbox 중심 픽셀이 노면 안, "ratio" = 후보 영역(윤곽 안쪽까지 채운 픽셀)의 ROAD_KEEP_RATIO_MIN 이상이 노면 안. 10/12 "ratio" 채택: 개발셋 규칙 2 미달(31.17%→31.05%, 652장 중 3장)이지만 영역 92%가 노면 밖인 후보를 노면 안으로 세던 한계를 고치는 수정이라 예외 채택 (3.3 (7), docs/part_C.md)
DEGRADE_BLUR_SIGMA = 2.0        # px, 인위 저하 blur: 640px 컬러에 ROI 자르기 전 가우시안 σ. 상태 판정이 샤프닝을 골라야 함 (Plan.md 3.2 (6), E6)
DEGRADE_DARK_GAMMA = 2.2        # 인위 저하 dark: 결과 = 입력^2.2 (0~1 밝기, 어두워짐). 감마 < 1(밝게)을 골라야 함
DEGRADE_BRIGHT_GAMMA = 0.5      # 인위 저하 bright: 결과 = 입력^0.5 (밝아짐, 과노출 흉내). 감마 > 1(어둡게)을 골라야 함
DEGRADE_NOISE_SIGMA = 10.0      # 밝기 단계(0~255), 인위 저하 noise: 가우시안 잡음 σ. 가우시안(잡티 제거)을 골라야 함
DEGRADE_SEED = 0                # 인위 저하 잡음의 seed. 사진마다 이 값에 사진 바이트의 crc32를 더해 씀(사진마다 다른 무늬, 같은 사진은 늘 같음. CONTRIBUTING 3장 5번)
ROAD_OVERLAY_COLOR = (0, 255, 0)    # BGR, 노면 마스크를 칠할 색: 초록. 균열(빨강)·포트홀(파랑)·노면 시작선(노랑) 윤곽과 헷갈리지 않는 색 (Plan.md 3.7)
ROAD_OVERLAY_ALPHA = 0.35           # 비율(0~1), 노면 픽셀에 초록을 섞는 정도. 절반보다 옅게 해 노면 결과 후보 윤곽이 비쳐 보이게
AUTO_BAND_TOP = True                # 자동 띠 높이 켬·끔. True면 pipeline이 사진 전체 노면 마스크를 보고 띠 위쪽 경계를 받은 top에서 위로 넓힘. '자동 띠 + 결 1.5'를 최종 방법으로 채택(10/12)해 켬. 끄면 띠 고정(E3) (docs/part_C.md)
AUTO_BAND_STEP = 0.1                # 비율(사진 높이 대비), 한 번에 살펴보고 넓히는 구간 높이. 0.5에서 0.0까지 다섯 구간
AUTO_BAND_ROAD_MIN = 0.95           # 비율, 바로 위 구간에서 노면 마스크 픽셀이 이 비율 이상이면 그 구간까지 넓힘. 후보 0.8·0.9·0.95 중 개발셋 652장에서 채택 규칙 2·3을 모두 넘은 값 (docs/part_C.md)
AUTO_BAND_USE_TEXTURE = True       # 자동 띠의 결 조건 켬·끔. True면 넓힐 구간 노면 픽셀의 결이 시작 띠 노면과 비슷할 때만 넓힘(흐린 하늘·나무가 노면 마스크에 든 차량 시점을 막으려고). AUTO_BAND_TOP이 꺼져 있으면 효과 없음 (docs/part_C.md)
AUTO_BAND_TEX_KSIZE = 7             # px, 결 = 흑백 640px 사진의 이 크기 창 안 표준편차(국소 표준편차). Plan.md 3.3의 보닛 규칙과 같은 7×7
AUTO_BAND_TEX_RATIO = 1.5           # 배, 구간 결 중앙값 ÷ 시작 띠 노면 결 중앙값이 1/1.5 ~ 1.5 안이면 비슷함. 후보 1.25·1.5·2.0을 개발셋에서만 견줌 (docs/part_C.md)
ROAD_KEEP_RATIO_MIN = 0.5           # 비율, ROAD_KEEP_RULE = "ratio"일 때 후보 영역 픽셀 중 노면 안 비율이 이 값 이상이면 남김. 경계가 비스듬해 상자 중심만 노면에 떨어지는 바깥 후보(Japan_000107 켬: 노면 안 8%)를 거름 (Plan.md 3.3 (7))
AUTO_BAND_TEX_PERSPECTIVE = False   # 보완 4-4(14번) 켬·끔. True면 위로 한 구간 갈 때마다 결 비율의 아래 한계(1/AUTO_BAND_TEX_RATIO)에 AUTO_BAND_TEX_DECAY를 곱함. 원근 때문에 먼 노면은 결이 매끈해 보이므로. 위 한계(나무·건물)는 그대로
AUTO_BAND_TEX_DECAY = 0.8          # 배, 4-4 한 구간(AUTO_BAND_STEP)마다 아래 한계에 곱하는 값(계획 7장 4-4 출발값)
DRAW_TAU_REJECTED = False           # 보완 6단계 켬·끔. True면 시연 그림(demo)에 점수 문턱 τ로 걸러진 후보를 옅은 점선으로 함께 그림. 평가 숫자(row·요약)에는 쓰이지 않음
REJECTED_ALPHA = 0.5               # 비율, 걸러진 후보 점선을 영상과 섞는 비율(0 = 안 보임, 1 = 진한 색)
REJECTED_DASH = 4                  # px, 걸러진 후보 점선의 한 칸 길이(대각 줄무늬 주기의 절반)
REJECTED_THICKNESS = 2             # px, 걸러진 후보 점선 두께(시연 창은 화면에 맞춰 줄어들어 1px는 거의 안 보임)
