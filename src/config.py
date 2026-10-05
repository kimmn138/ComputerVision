"""모든 파라미터. 구역마다 주인이 있고, 자기 구역 안에서만 고친다.
구역 사이의 빈 줄과 주석 줄은 지우지 않는다(Git이 구역별 수정을 자동으로 합치게 해 줌)."""

# ===== 공통 (C 관리 · 바꾸려면 세 명 합의) =====
TARGET_WIDTH = 640          # 모든 영상을 가로 640px로
ROI_TOP_DEFAULT = 0.5       # roi.csv에 없는 사진의 노면 시작 비율
ROI_BOTTOM_DEFAULT = 1.0
TIME_REPEAT = 5             # 처리 시간은 5번 재서 중앙값

# ===== A: 데이터·상태 분석·전처리 =====
DAMAGE_TAGS = ("crack", "pothole", "both", "none")
CONDITION_TAGS = ("normal", "dark", "night", "bright", "shadow", "blur", "wet", "clutter", "none")
PAIR_ROLES = ("A", "B", "C", "D")
PAIR_CONDITIONS = ("ref", "dark", "blur", "view")
BLUR_VAR_MAX = 50           # 라플라시안 분산 < 50 -> 흐림 -> 샤프닝
NOISE_SIGMA_MAX = 3.0       # 잡티 지표 > 3 -> 가우시안
CONTRAST_STD_MIN = 20       # 표준편차 < 20 -> 대비 부족 -> 평활화
DARK_MEAN_MAX = 80          # 평균 < 80 -> 감마(밝게)
BRIGHT_MEAN_MIN = 170       # 평균 > 170 -> 감마(어둡게)
GAUSS_KSIZE, GAUSS_SIGMA = (5, 5), 1.0
GAMMA_RANGE = (0.4, 2.5)    # 결과 = 입력^감마 (감마 < 1 밝게). 감마 = log0.5 / log(평균/255)를 이 범위로 자름
SHARPEN_AMOUNT, SHARPEN_SIGMA = 3.0, 3.0    # 언샤프 마스크: 결과 = 입력 + AMOUNT × (입력 − 가우시안(σ px))
USE_CLAHE = False           # True면 평활화 대신 CLAHE
CLAHE_CLIP, CLAHE_TILE = 2.0, (8, 8)
NOISE_BLOCK = 16            # px, 잡티는 이 크기 블록마다 재서
NOISE_FLAT_QUANTILE = 0.1   # 가장 평평한 하위 10% 블록 값을 씀. 전체로 재면 결·경계를 잡티로 세어 깨끗한 사진도 흐려짐(제공 사진 13장: 깨끗 최대 1.5, σ3 잡음 2.7 이상)
NOISE_SAT_MAX = 0.05        # 비율, 0·255 픽셀이 5%보다 많은 블록은 잡티가 잘려 낮게 나오므로 뺌

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
