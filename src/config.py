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

# ===== 균열·포트홀 검출·평가 =====
CANNY_LOW, CANNY_HIGH = 50, 150     # Canny 에지의 낮은·높은 임계값
CRACK_CLOSE_KSIZE = 5       # px, legacy 균열: 끊긴 에지를 잇는 닫힘 커널
CRACK_MIN_AREA = 80         # 픽셀 수, legacy 균열 후보의 최소 넓이
CRACK_MIN_ELONG = 4.0       # 길쭉함(긴 변 ÷ 짧은 변)이 이 값 이상이면 균열(곧은 균열)
CRACK_MAX_FILL = 0.1        # 또는 채움(윤곽 넓이 ÷ 회전 사각형 넓이)이 이 값 이하면 균열(구불구불·갈래 균열)
DOG_SIGMAS = (3, 12)        # px, legacy 포트홀: DoG(두 가우시안의 차)의 작은·큰 σ
DOG_THRESH = 12             # 밝기 단계, DoG 응답이 이보다 크면 어두운 덩어리
POT_OPEN_KSIZE = 5          # px, 포트홀 마스크의 작은 점을 지우는 열림 커널
POT_MIN_AREA = 150          # 픽셀 수, 포트홀 후보의 최소 넓이
POT_MAX_AREA_RATIO = 0.2    # ROI 넓이 대비, legacy 포트홀 후보의 최대 넓이
POT_MAX_ELONG = 3.0         # legacy 포트홀: 길쭉함이 이 값 이하여야 포트홀
POT_MIN_SOLIDITY = 0.6      # 속이 찬 정도(윤곽 넓이 ÷ 볼록 껍질 넓이)가 이 값 이상이어야 포트홀
IOU_THRESH = 0.3            # 정답 비교: IoU가 이 값 이상이면 같은 상자로 봄
DARK_RING_PX = 4            # px, 후보의 어둡기를 잴 둘레 고리 두께
DARK_RING_QUANTILE = 0.25   # 분위, 고리에서 어두운 쪽(하위 25%)과 비교(넓은 흰 차선의 가장자리를 거르려고)
CRACK_MIN_DARKNESS = -5     # 밝기 단계(0~255), legacy 균열 후보의 어둡기 하한(주변보다 밝은 흰 선을 거름)
# dark 방식: 배경보다 어두운 영역을 뽑아 모양으로 균열·포트홀을 가름
DETECT_MODE = "dark"        # 검출 방식: "dark" = 어두운 영역을 뽑아 모양으로 분류, "legacy" = Canny 균열 + DoG 포트홀
DARK_BG_DOWNSCALE = 4       # 배, 배경을 잴 때 영상을 1/4로 줄여 중앙값을 빠르게 구함
DARK_BG_KSIZE_SMALL = 11    # px(1/4 영상 기준), 밝은 표시를 찾을 거친 배경의 중앙값 창(원래 크기로 약 44px)
DARK_BRIGHT_DELTA = 25      # 밝기 단계(0~255), 거친 배경보다 이만큼 밝은 픽셀(흰 차선·횡단보도)을 배경값으로 메움
DARK_BG_FRAC = 0.5          # 비율, 작은 배경 창 = ROI 높이 × 0.5
DARK_BG_FRAC_LARGE = 1.0    # 비율, 큰 배경 창 = ROI 높이 × 1.0(두 창의 큰 값을 써서 큰 포트홀 가운데가 비지 않게)
DARK_SIGMA_MIN = 1.0        # 밝기 단계, 잡음 σ의 하한(σ = 0이면 임계값도 0이 되어 ROI 전체가 한 영역이 됨)
DARK_SIGMA_IN_ROAD = True   # True면 노면 마스크가 있을 때 잡음 σ를 노면 안에서만 잼
DARK_K_HI = 4               # σ의 배수, 이중 임계값의 강한 임계값
DARK_K_LO = 2               # σ의 배수, 이중 임계값의 약한 임계값
DARK_ABS_MIN = 8            # 밝기 단계, 강한 임계값의 하한(σ가 아주 작은 영상에서 잡음이 후보가 되지 않게)
DARK_CLOSE_KSIZE = 5        # px, 끊긴 조각을 잇는 닫힘 커널
DARK_MIN_AREA = 40          # 픽셀 수, 이보다 작은 영역(골재 무늬)은 버림
CRACK_MIN_CIRC_INV = 4      # 비원형도(둘레² ÷ 4π넓이), 이 값 이상이면 균열(가늘고 긴 선)
POT_MAX_CIRC_INV = 2.5      # 비원형도, 이 값 이하이고 속이 차 있으면 포트홀(둥근 덩어리)
# 후보 점수: 점수가 임계값보다 낮은 후보를 버려 오검출을 줄임
SCORE_FORMULA = "margin"    # 점수 식: "margin" = 대비 × 분류 경계에서의 여유, "snr" = 대비 × √넓이, "shape" = 대비 × 길이·넓이, "none" = 점수 없음
SCORE_MARGIN_SCALE = 3.5    # margin 식 균열 여유 1 − e^(−(비원형도 − 4) / k)의 k
SCORE_C_CAP = 4.0           # σ의 배수, shape 식에서 대비의 상한(진한 그림자가 대비만으로 높은 점수를 받지 않게)
SCORE_LEN_REF = 90          # px, shape 식: 균열 길이가 이 값 이상이면 길이 항 1
SCORE_AREA_REF = 80         # 픽셀 수, shape 식: 포트홀 넓이가 이 값 이상이면 넓이 항 1
SCORE_MARGIN_SCALE_POT = 0.72   # margin 식 포트홀 여유 1 − e^(−(2.5 − 비원형도) / k)의 k
USE_SCORE_TAU = True        # True면 점수가 임계값보다 낮은 후보를 버림(dark 방식에서만, 점수 식이 none이면 오류)
SCORE_TAU_CRACK = 0.304508  # 균열 점수 임계값: 개발용 데이터에서 영상당 오검출 2개 이하로 민감도가 가장 높은 운영점
SCORE_TAU_POTHOLE = 7.822985  # 포트홀 점수 임계값: 정한 방법은 균열과 같음(값이 높아 포트홀 후보는 거의 남지 않음)
# dark 방식 보완: 지역 잡음 추정·밝은 표시 주변 보정·덩어리 나누기·경계 날카로움
DARK_SIGMA_LOCAL = True     # True면 지역 잡음 추정: 잡음 σ를 위치마다 주변에서 재, 질감이 거친 노면에서 임계값을 올림
DARK_SIGMA_LOCAL_FRAC = 0.25    # 비율, 지역 잡음을 잴 창 = ROI 높이 × 0.25
DARK_SIGMA_LOCAL_MIN_N = 50 # 개, 창 안 표본이 이보다 적으면 전역 σ를 씀
DARK_BRIGHT_DILATE = 6      # px, 밝은 표시를 이만큼 넓혀 샤프닝이 그 옆에 만든 어두운 테두리까지 메움(0이면 끔)
DARK_BRIGHT_DILATE_DELTA = 80   # 밝기 단계, 넓힐 '아주 밝은 픽셀'의 기준(기준 밝기 + 이 값, 균열 옆 밝은 테두리는 빠짐)
DARK_BRIGHT_SEED_KSIZE = 5  # px, 아주 밝은 픽셀을 고르기 전 중앙값 필터 창(작은 점은 빼고 차선은 남김)
DARK_SPLIT_BLOB = False     # True면 원판 열림으로 굵은 덩어리(포트홀)와 가는 선(균열)을 나눔
DARK_SPLIT_KSIZE = 15       # px, 덩어리로 남길 원판 지름(균열 폭의 약 두 배)
SCORE_USE_EDGE = False      # True면 경계가 흐린 후보(그림자)의 점수를 깎음
SCORE_EDGE_REF = 0.06       # 경계 날카로움 기준, 이보다 흐린 후보는 비례해서 점수를 깎음
DARK_BRIGHT_REF = "coarse"  # 밝은 표시의 기준 밝기: "coarse" = 거친 배경, "road_median" = 노면 밝기 중앙값 하나
# 선택 기능: 헤시안 선 강조·원근 보정·자갈 포트홀
SCORE_USE_HESSIAN = False   # True면 균열 점수에 헤시안 선 강조(가는 어두운 선일수록 1)를 곱함
HESSIAN_SIGMA = 1.5         # px, 2차 미분 전에 거는 가우시안 σ(균열 폭 3~5px의 절반쯤)
DARK_AREA_PERSPECTIVE = False   # True면 ROI 위쪽(먼 곳)일수록 최소 넓이를 줄임(원근 보정)
DARK_MIN_AREA_TOP = 20      # 픽셀 수, 원근 보정 때 ROI 맨 위 행의 최소 넓이
BRIGHT_POTHOLE = False      # True면 질감이 노면보다 훨씬 거친 포트홀 모양 덩어리(자갈 포트홀)를 후보에 더함
BRIGHT_POT_TEX_RATIO = 2.0  # 배, 질감이 노면의 이 배수를 넘으면 거친 영역
BRIGHT_POT_TEX_KSIZE = 7    # px, 질감(국소 표준편차)을 잴 창 크기

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
