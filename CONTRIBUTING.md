# 협업 규칙

이 저장소는 파트 A(데이터·전처리), B(후보 검출·평가), C(측정·실행·통합) 세 명이 나눠 구현한다.
한 파일은 한 사람만 고치고, 함수 약속을 지키며, PR로만 main에 합친다.

## 1. 파트와 파일 주인

| 파트 | 담당 파일 |
|---|---|
| A · 데이터·전처리 | src/io_utils.py, src/analyze.py, src/preprocess.py, data/roi.csv, data/shot_log.csv, tools/check_data.py, tools/show_preprocess.py, tests/test_a.py, docs/part_A.md, docs/ai_log_A.md |
| B · 후보 검출·평가 | src/detect.py, src/evaluate.py, tools/tune_detect.py, tools/label_gt.py, gt/, tests/test_b.py, docs/part_B.md, docs/ai_log_B.md |
| C · 측정·실행·통합 | src/config.py(공통 구역과 파일 관리), src/features.py, src/visualize.py, src/pipeline.py, run_experiment.py, run_matching.py, demo.py, tests/check_contract.py, tests/test_c.py, README.md, CONTRIBUTING.md, CLAUDE.md, requirements.txt, .gitignore, .github/, docs/part_C.md, docs/ai_log_C.md |

- src/config.py는 구역(공통·A·B·C)마다 주인이 있다. 자기 구역 안에서만 고치고, 새 값은 자기 구역 맨 아래에 추가한다.
- gt/의 정답 박스는 기준과 검수를 B가 맡고, 표시 작업은 세 명이 영상을 나눠 한다. 영상마다 파일이 따로라 충돌이 없다.
- TODO(A)·TODO(B)·TODO(C)가 붙은 함수는 형식만 맞춘 임시 버전이다. 각 파트가 가이드 문서 3장의 코드에서 시작해 자기 브랜치에서 구현한다.

## 2. 함수 약속

함수 이름·입력·출력은 아래와 같이 고정한다. tests/check_contract.py가 이 형식을 자동으로 검사하고, 실패하면 담당 파트 이름([A]·[B]·[C])을 붙여 알려 준다.

| 함수 | 주인 | 입력 | 출력 | 꼭 지킬 점 |
|---|---|---|---|---|
| io_utils.load_image(path) | A | 파일 경로 | BGR uint8 (H, W, 3) | 못 읽으면 FileNotFoundError. None을 돌려주지 않음 |
| io_utils.resize_width(img, width=640) | A | 컬러 또는 흑백 영상 | 가로 640, 채널 유지 | 세로는 비율대로 |
| io_utils.crop_roi(img, top, bottom) | A | 영상, 비율 2개(0~1) | (노면 영상, y0: int) | y0 = 위에서 잘라 낸 픽셀 수 |
| io_utils.load_roi_table(path) | A | csv 경로 | {파일명: {top, bottom, condition}} | 파일이 없으면 빈 dict |
| io_utils.parse_name(path) | A | 사진 경로 | {source, damage, condition} 또는 {source, pair, role, condition} | 규칙을 어긴 own·pair 이름은 ValueError |
| analyze.measure_quality(gray) | A | 흑백 uint8 | {mean, std, lap_var, noise} (float) | 노면 영상 기준 |
| analyze.choose_steps(q) | A | 위 dict | {denoise: bool, tone: None·"gamma"·"equalize"·"clahe", gamma: float, sharpen: bool} | tone 값을 늘리려면 약속 변경 절차 |
| analyze.steps_to_text(steps) | A | steps | 예: "가우시안+감마 0.53" | NO_STEPS면 "없음" |
| preprocess.preprocess(gray, steps) | A | 흑백 uint8, steps | 같은 크기 흑백 uint8 | NO_STEPS면 입력과 똑같은 영상 |
| detect.detect_cracks(gray) | B | 흑백 uint8 | (후보 list, 캐니 에지 uint8 0/255) | 에지 영상은 C가 '에지 수'로 그대로 셈 |
| detect.detect_potholes(gray) | B | 흑백 uint8 | (후보 list, 이진 마스크 uint8 0/255) | 균열과 같은 후보 형식 |
| 후보 dict (두 함수 공통) | B | — | area: int, elong·fill·solidity: float, bbox: (x, y, w, h) int, contour: (N, 1, 2) int32 | 좌표는 노면 영상 기준 |
| evaluate.load_gt(path) | B | 정답 csv 경로 | [(x, y, w, h, kind)] | kind는 "crack"·"pothole". 파일이 없으면 None, 헤더만 있으면 [] |
| evaluate.count_matches(pred, gt, thr) | B | 박스 list 2개, IoU 기준 | {tp, fp, found, fn} (int) | 박스는 640px 전체 영상 좌표 |
| evaluate.precision_recall(counts) | B | 위 dict(여러 장 합계 가능) | (정밀도, 재현율) | 분모가 0이면 NaN |
| features.count_edges(edges), count_harris(gray) | C | 에지 영상, 흑백 | int | |
| features.match_pair(g1, g2) | C | 흑백 2장 | {kp1, kp2, good, inliers, inlier_ratio, …} | 매칭이 안 돼도 오류 없이 0 |
| visualize.draw_candidates(bgr, cracks, potholes, y0) | C | 640px 컬러, 후보, y0 | 그린 복사본 | 받은 영상은 바꾸지 않음 |
| pipeline.run_pipeline(bgr, top, bottom, use_preprocess) | C | 원본 컬러, ROI 비율, 끔/켬 | {small, y0, gray, proc, edges, cracks, potholes, steps, quality, row} | 끔/켬의 차이는 steps 하나뿐 |

row의 열은 edges, harris, n_crack, area_crack, n_pothole, area_pothole, roi_pixels, t_pre_ms, t_detect_ms 9개로 고정한다.

## 3. 데이터 규칙

1. 단계 사이 영상은 흑백 uint8 2차원 (H, W), 값 0~255다. float나 컬러를 넘기지 않는다. 컬러(BGR)는 읽기와 그림 그리기에서만 쓴다.
2. 받은 배열을 제자리에서 바꾸지 않는다. 고칠 때는 새 배열을 만들어 돌려준다(out = img.copy()).
3. 좌표는 두 가지뿐이다. 후보의 bbox·contour는 노면(ROI) 기준, 정답 박스와 그림은 640px 전체 영상 기준이다. 바꾸는 식은 y + y0 하나이고, C(visualize, run_experiment)만 바꾼다.
4. src/ 안의 함수는 print, cv.imshow, 파일 저장을 하지 않는다. 확인용 출력은 tools/와 tests/에서 하고, 파일 저장은 visualize.save_comparison만 한다.
5. 같은 입력이면 같은 출력이다. 전역 변수에 상태를 남기지 않고, 무작위를 쓰면 seed를 고정한다.
6. dict(steps, quality, 후보, row)는 키를 더할 수만 있다. 지우거나 이름을 바꾸지 않는다.

## 4. 구현 규칙

환경
- Python 3.10 이상을 세 명이 같은 버전으로 쓰고, 패키지는 requirements.txt로만 설치한다.
- 라이브러리는 opencv-python, numpy, matplotlib, pandas 네 가지다. 다른 것은 먼저 합의한다. opencv-contrib-python이나 opencv-python-headless와 섞어 설치하지 않는다.
- 가상환경(.venv)은 각자 만들고 Git에 올리지 않는다.

코드
- 파라미터 숫자는 함수 안에 쓰지 않고 src/config.py 자기 구역에 둔다. 함수 안에서는 C.이름으로 읽는다.
- 함수마다 한국어 docstring 첫 줄에 무엇을 왜 하는지 쓴다. config 값 옆 주석에는 단위와 근거를 쓴다.
- 함수·변수 이름은 snake_case, config 값은 대문자. import는 from src import detect처럼 맨 위 폴더 기준이고, 실행도 맨 위 폴더에서 한다.
- 픽셀마다 도는 파이썬 이중 for 문 대신 OpenCV·NumPy 연산을 쓴다.
- 영상별 예외(파일 이름으로 분기)는 쓰지 않는다. 값은 모든 영상에 하나다.

데이터·경로
- 경로는 맨 위 폴더 기준 상대 경로만 쓴다.
- 사진 이름은 영어·숫자·밑줄만, 확장자는 소문자 .jpg. 직접 촬영은 own_번호_손상_조건.jpg, 매칭용 쌍은 pair번호_기호_조건.jpg.
- 원본 사진은 덮어쓰지 않는다. 사진은 Git에 올리지 않고 드라이브로 나눈다.
- CSV는 UTF-8로 저장하고 encoding="utf-8-sig"로 읽는다(엑셀이 붙이는 BOM 대비).
- results/의 숫자와 그림은 손으로 고치지 않는다. 바꾸려면 다시 실행한다.

기록
- 파라미터를 바꿀 때마다 docs/part_X.md 표에 날짜·이름·전·후·이유·확인한 영상을 한 줄 쓴다.
- AI 도구를 쓰면 docs/ai_log_X.md에 날짜·도구·프롬프트·결과를 어떻게 고쳤는지·디버깅 과정을 남긴다.
- 처리 시간은 C의 컴퓨터 한 대에서 잰 값만 보고서에 쓴다.

## 5. Git 작업 순서

- 브랜치: A는 a-preprocess, B는 b-detect, C는 c-integrate. main에 직접 push하지 않는다.
- 처음 한 번: git config --global pull.rebase false

```bash
git switch a-preprocess            # 내 브랜치로
git pull origin main               # 다른 사람이 합친 최신 main 받기
# ... 내 파일만 수정 ...
python tests/check_contract.py     # 약속 검사
python tests/test_a.py             # 내 시험
git add src/analyze.py             # 내 파일만 골라서 (git add . 금지)
git commit -m "[A] 무엇을 바꿨는지"
git push -u origin a-preprocess
```

- PR: GitHub에서 내 브랜치 → main. 제목 앞에 [A]·[B]·[C], 양식을 채우고 전·후 그림을 붙인다.
- 리뷰: 1명이 확인하고 승인한다. A·B의 PR은 C가, C의 PR은 A나 B가 본다. 바뀐 파일이 모두 작성자 것인지, config는 자기 구역만 바뀌었는지, 약속 표 형식이 그대로인지 본다.
- 병합: 기본 버튼(Create a merge commit)을 쓴다. Squash는 쓰지 않는다.
- 관문 날에는 A → B → C 순서로 병합하고, 병합할 때마다 C가 main에서 약속 검사와 python run_experiment.py --name gN 을 돌려 결과를 공유한다.
- 충돌이 남의 파일에서 나면 직접 풀지 않고 주인에게 맡긴다. 충돌 표시(<<<<<<<, =======, >>>>>>>)를 남긴 채 커밋하지 않는다.
- 병합 뒤 main이 실행되지 않으면 고치려 하지 말고 그 PR을 Revert한 뒤 원인을 찾는다.

## 6. 통합 관문

| 관문 | 시점(예시) | 통과 기준 |
|---|---|---|
| G1 뼈대 동작 | 1일차 | 세 명 모두 clone → 약속 검사 통과, python run_experiment.py --name g1 실행 |
| G2 1차 통합 | 4일차 | 각 파트가 가이드 코드를 옮기고 새로 작성할 함수를 완성, 직접 촬영 사진·roi.csv 반영, --name g2 전체 실행 (2차 계획서 실험 결과) |
| G3 전처리 고정 | 6일차 | config A 구역 동결, 정답 박스 기준·도구 완성, run_matching 완성 |
| G4 코드 동결 | 8일차 | 검출 파라미터 확정, 정답 박스 검수 완료, C의 컴퓨터에서 --name final 실행. 이후는 버그 수정만 |

## 7. 약속을 바꾸는 절차

1. 바꾸려는 사람이 단톡방에 무엇을, 왜, 누가 영향을 받는지 올린다.
2. 세 명이 모두 동의하면 C가 이 문서와 tests/check_contract.py를 먼저 고쳐 main에 올린다.
3. 영향받는 사람이 각자 자기 파일을 고쳐 PR을 올리고, 같은 관문 안에 끝낸다.
