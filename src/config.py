"""모든 파라미터를 한곳에 모은다. 함수 안에서는 C.이름으로 읽는다."""

# ===== 공통 =====
TARGET_WIDTH = 640          # px, 모든 영상을 이 가로 크기로 맞춤
ROI_TOP_DEFAULT = 0.5       # 비율, roi.csv에 없는 사진의 노면 띠 시작
ROI_BOTTOM_DEFAULT = 1.0    # 비율, 노면 띠 끝
TIME_REPEAT = 5             # 회, 처리 시간을 여러 번 재서 중앙값을 씀

# ===== A: 데이터·상태 분석·전처리 =====
DAMAGE_TAGS = ("crack", "pothole", "both", "none")
CONDITION_TAGS = ("normal", "dark", "night", "bright", "shadow", "blur", "wet", "clutter", "none")
PAIR_ROLES = ("A", "B", "C", "D")
PAIR_CONDITIONS = ("ref", "dark", "blur", "view")
BLUR_VAR_MAX = 50           # 라플라시안 분산이 이보다 작으면 흐림 → 샤프닝
NOISE_SIGMA_MAX = 3.0       # 잡티 지표가 이보다 크면 → 가우시안
CONTRAST_STD_MIN = 20       # 밝기 표준편차가 이보다 작으면 대비 부족 → 평활화
DARK_MEAN_MAX = 80          # 평균 밝기가 이보다 작으면 → 감마(밝게)
BRIGHT_MEAN_MIN = 170       # 평균 밝기가 이보다 크면 → 감마(어둡게)
GAUSS_KSIZE, GAUSS_SIGMA = (5, 5), 1.0
GAMMA_RANGE = (0.4, 2.5)    # 감마 = log0.5 / log(평균/255)를 이 범위로 자름 (감마 < 1이면 밝아짐)
SHARPEN_AMOUNT, SHARPEN_SIGMA = 3.0, 3.0    # 언샤프 마스크: 입력 + AMOUNT × (입력 − 가우시안(σ))
USE_CLAHE = False           # True면 평활화 대신 CLAHE
CLAHE_CLIP, CLAHE_TILE = 2.0, (8, 8)
NOISE_BLOCK = 16            # px, 잡티를 재는 블록 크기
NOISE_FLAT_QUANTILE = 0.1   # 가장 평평한 하위 10% 블록으로 잡티를 잼 (결·경계를 잡티로 세지 않게)
NOISE_SAT_MAX = 0.05        # 비율, 0·255 픽셀이 이보다 많은 블록은 잡티 측정에서 뺌
ROAD_BLUR_SIGMA = 3         # px, 노면 마스크: 색을 재기 전 가우시안 σ
ROAD_SEED_BOX = (0.30, 0.40, 0.70, 0.80)    # 비율 (x 시작, y 시작, x 끝, y 끝), 노면 색을 배우는 씨앗 영역
ROAD_CHROMA_MAX = 10        # Lab a·b 거리, 씨앗 색과 이보다 가까우면 노면 색
ROAD_L_K = 4                # 배, 씨앗 밝기에서 이 배수(× 강건 표준편차) 안이면 노면 밝기
ROAD_OPEN_KSIZE = 5         # px, 노면 마스크의 작은 점을 지우는 열림 커널
ROAD_CLOSE_KSIZE = 31       # px, 노면 마스크의 틈(균열·차선)을 메우는 닫힘 커널
BLUR_RATIO_MAX = 1.5        # 재흐림 비율이 이보다 작으면 흐림 → 샤프닝
BLUR_RATIO_SIGMA = 1.5      # px, 재흐림 비율을 잴 때 쓰는 가우시안 σ
ROAD_LANE_BRIDGE = False    # True면 차선(밝고 가는 흰·노란 줄)도 노면으로 다룸
ROAD_LANE_KSIZE = 41        # px, 이 폭보다 가는 밝은 줄을 차선으로 봄 (탑햇 커널)
ROAD_LANE_DELTA = 20        # 밝기, 주변보다 이만큼 밝으면 밝은 줄
ROAD_LANE_WHITE_CHROMA = 12 # Lab a·b 거리, 무채색과 이보다 가까우면 흰 차선
ROAD_LANE_YELLOW_B = 20     # Lab b − 128, 이 값 이상이면 노란 중앙선
ROAD_CLOSE_BEFORE_SEED = False  # True면 씨앗 영역을 고르기 전에 닫힘을 먼저 해 차선 건너편 노면을 이음

# ===== B: 균열·포트홀·평가 =====
CANNY_LOW, CANNY_HIGH = 50, 150
CRACK_CLOSE_KSIZE = 5
CRACK_MIN_AREA = 80         # 픽셀 수
CRACK_MIN_ELONG = 4.0       # 길쭉함(긴 변 / 짧은 변)이 이 값 이상이거나
CRACK_MAX_FILL = 0.1        # 채움 정도(윤곽 넓이 / 회전 사각형 넓이)가 이 값 이하면 균열
DOG_SIGMAS = (3, 12)        # (작은 σ, 큰 σ)
DOG_THRESH = 12
POT_OPEN_KSIZE = 5
POT_MIN_AREA = 150
POT_MAX_AREA_RATIO = 0.2    # 노면 띠 넓이 대비
POT_MAX_ELONG = 3.0
POT_MIN_SOLIDITY = 0.6
IOU_THRESH = 0.3            # 정답 박스와 겹친 것으로 볼 IoU 기준
TRAIN_IMAGE_DIR = "data/train/img"  # 학습 사진 폴더
TRAIN_ANN_DIR = "data/train/ann"    # 학습 사진의 JSON 정답 폴더
TRAIN_GT_DIR = "gt/train"           # JSON을 CSV로 바꾼 정답 폴더
DARK_RING_PX = 4            # px, 후보의 어둡기를 잴 주변 띠 두께
DARK_RING_QUANTILE = 0.25   # 비율, 주변 띠의 어두운 쪽(하위 25%) 밝기와 비교
CRACK_MIN_DARKNESS = -5     # 밝기, 균열 후보의 어둡기 하한 (주변보다 밝은 흰 차선을 거름)
DETECT_MODE = "dark"        # "legacy" = Canny 균열 + DoG 포트홀, "dark" = 어두운 영역 추출 후 형태로 분류
DARK_BG_DOWNSCALE = 4       # 배, 배경을 1/4 크기에서 계산해 빠르게
DARK_BG_KSIZE_SMALL = 11    # px(1/4 영상), 밝은 표시를 누르기 위한 거친 배경의 중앙값 창
DARK_BRIGHT_DELTA = 25      # 밝기, 배경보다 이만큼 밝은 픽셀(차선·횡단보도)을 배경값으로 메움
DARK_BG_FRAC = 0.5          # 비율, 작은 배경 창 = 띠 높이 × 이 값
DARK_BG_FRAC_LARGE = 1.0    # 비율, 큰 배경 창 = 띠 높이 × 이 값 (큰 포트홀 가운데가 비지 않게)
DARK_SIGMA_MIN = 1.0        # 밝기, 잡음 σ의 하한
DARK_SIGMA_IN_ROAD = True   # True면 잡음 σ를 노면 안에서만 잼
DARK_K_HI = 4               # σ의 배수, 강 문턱
DARK_K_LO = 2               # σ의 배수, 약 문턱
DARK_ABS_MIN = 8            # 밝기, 강 문턱의 하한
DARK_CLOSE_KSIZE = 5        # px, 끊긴 조각을 잇는 닫힘 커널
DARK_MIN_AREA = 40          # 픽셀 수, 이보다 작은 영역은 버림
CRACK_MIN_CIRC_INV = 4      # 비원형도 P²/(4πA)가 이 값 이상이면 균열 (가늘고 긴 선)
POT_MAX_CIRC_INV = 2.5      # 비원형도가 이 값 이하이고 solidity가 충분하면 포트홀 (둥근 모양)

# ===== C: 정량 측정·실행 =====
HARRIS_BLOCK, HARRIS_KSIZE, HARRIS_K, HARRIS_REL = 2, 3, 0.04, 0.01
SIFT_RATIO = 0.7
RANSAC_THRESH = 5.0
HARRIS_NMS = 3              # px, 이 이웃 안의 최댓값만 코너 하나로 셈
RANSAC_MIN_GOOD = 4         # 개, 호모그래피에 필요한 최소 대응점 수
COMPARE_FIGSIZE = (19.2, 9.0)   # 인치, 비교 그림 크기
COMPARE_DPI = 100               # 비교 그림 해상도 (가로 1920px)
FONT_BY_PLATFORM = {"win32": "Malgun Gothic", "darwin": "AppleGothic"}  # 운영체제별 한글 글꼴
FONT_DEFAULT = "DejaVu Sans"    # 그 밖 운영체제의 글꼴 (한글은 깨짐)
DEMO_SCREEN_FALLBACK = (1280, 720)  # px, 시연 창: 화면 크기를 못 읽을 때 가정하는 해상도
DEMO_SCREEN_MARGIN = 0.9        # 비율, 시연 창이 화면에서 차지하는 최대 비율
DEMO_LABEL_HEIGHT = 40          # px, 시연 창 패널 위 제목 띠 높이
DEMO_FONT_SIZE = 22             # px, 시연 창 제목 글자 크기
USE_ROAD_MASK = True            # True면 노면 마스크를 검출에 쓰고 노면 밖 후보를 지움
ROAD_KEEP_RULE = "ratio"        # 노면 안 후보 판정: "center" = 상자 중심이 노면 안, "ratio" = 후보 영역의 일정 비율 이상이 노면 안
DEGRADE_BLUR_SIGMA = 2.0        # px, 인위 저하 blur의 가우시안 σ
DEGRADE_DARK_GAMMA = 2.2        # 인위 저하 dark: 입력^2.2 (어두워짐)
DEGRADE_BRIGHT_GAMMA = 0.5      # 인위 저하 bright: 입력^0.5 (밝아짐)
DEGRADE_NOISE_SIGMA = 10.0      # 밝기, 인위 저하 noise의 가우시안 잡음 σ
DEGRADE_SEED = 0                # 인위 저하 잡음의 seed (사진마다 다르지만 같은 사진은 늘 같게)
ROAD_OVERLAY_COLOR = (0, 255, 0)    # BGR, 노면 마스크를 칠할 색 (초록)
ROAD_OVERLAY_ALPHA = 0.35           # 0~1, 노면에 초록을 섞는 정도
AUTO_BAND_TOP = True                # True면 노면 마스크를 보고 띠 위쪽 경계를 자동으로 넓힘
AUTO_BAND_STEP = 0.1                # 비율(사진 높이 대비), 한 번에 넓혀 보는 구간 높이
AUTO_BAND_ROAD_MIN = 0.95           # 비율, 구간의 노면 픽셀이 이 이상이면 그 구간까지 넓힘
AUTO_BAND_USE_TEXTURE = True       # True면 결(질감)이 노면과 비슷한 구간만 넓힘 (하늘·나무 제외)
AUTO_BAND_TEX_KSIZE = 7             # px, 결(국소 표준편차)을 재는 창 크기
AUTO_BAND_TEX_RATIO = 1.5           # 배, 구간 결 ÷ 노면 결이 1/1.5 ~ 1.5 안이면 비슷함
ROAD_KEEP_RATIO_MIN = 0.5           # 비율, "ratio" 규칙에서 후보 영역 중 노면 안 비율의 하한
