# pbl1_road

PBL 모듈 1 (3조) · 저품질 도로 영상을 상태에 맞게 전처리한 뒤 균열·포트홀 후보 영역을 찾고, 전처리 적용·미적용 결과를 정량 비교한다.

## 설치

Python 3.12 이상 (최종 실행 환경: Python 3.13.2, Windows 11)

```bash
python -m venv .venv            # 윈도: .venv\Scripts\activate   맥·리눅스: source .venv/bin/activate
pip install -r requirements.txt
```

## 사진 넣기

사진은 이 압축 파일에 들어 있지 않다. 아래 폴더에 넣는다(확장자는 소문자 .jpg).

| 폴더 | 내용 | 이름 규칙 |
|---|---|---|
| data/provided | 제공 영상 13장 | 받은 이름 그대로 (예: Japan_000107.jpg) |
| data/own | 직접 촬영 26장 | own_번호_손상_조건.jpg (예: own_10_crack_normal.jpg) |
| data/pairs | 매칭 실험용 5쌍 10장 | pair번호_기호_조건.jpg (예: pair01_A_ref.jpg) |

- 손상: crack · pothole · both · none / 조건: normal · dark · night · bright · shadow · blur · wet · clutter · none
- 쌍 기호: A(기준) · B · C · D / 쌍 조건: ref · dark · blur · view
- data/roi.csv: 사진마다 ROI(노면 띠)의 위·아래 비율(0~1)과 촬영 조건. 여기에 없는 사진은 기본 ROI(0.5~1.0)에서 시작해, 노면이 이어지는 만큼 ROI 상단 경계를 자동으로 올린다.
- gt/<사진 이름>.csv: 정답 상자(x, y, w, h, kind), 가로 640px로 줄인 영상의 좌표. 정답 파일이 없는 사진은 정답 비교 칸이 빈칸이다.

## 실행

모든 명령은 맨 위 폴더(이 README가 있는 폴더)에서 실행한다. 옵션 없이 돌리면 src/config.py의 최종 설정을 쓴다.

```bash
python run_experiment.py --name final              # data/provided·own의 사진을 전처리 미적용·적용으로 처리
python run_matching.py --name final                # data/pairs 쌍 사진의 SIFT 매칭을 전처리 미적용·적용으로 비교
python demo.py data/own/own_10_crack_normal.jpg    # 사진 1장의 결과를 창에 띄움(아무 키나 누르면 닫힘)
```

- 결과는 results/<name>/에 저장된다(--name을 빼면 results/latest).
- demo.py: 경로를 빼면 파일 선택 창이 뜬다. --save <그림 경로>를 주면 창 없이 같은 화면을 그림 파일로 저장한다.
- 그 밖의 실행 옵션은 --help로 본다.

| 결과 파일 | 내용 |
|---|---|
| metrics.csv | 영상마다 2행(preprocess 열 off = 전처리 미적용, on = 적용): 화질 지표, 고른 전처리, 균열·포트홀 후보 수·면적, 에지·해리스 코너 수, 처리 시간(ms), 정답 비교 |
| table_by_image.csv · table_by_source.csv · table_by_condition.csv | 영상별·출처별·조건별 요약(미적용·적용 값과 그 차이) |
| figures/<영상 이름>.png | 위 줄 미적용·아래 줄 적용 비교 그림(노면 마스크 초록, 균열 빨강, 포트홀 파랑, ROI 시작선 노랑) |
| matching.csv · matches/ | 쌍 매칭 결과 표와 RANSAC inlier 그림 |
| config_used.txt · config_used_matching.txt | 이번 실행에 쓴 설정 값 |

## 코드 구성

한 장의 처리 순서: 가로 640px 맞춤 → ROI 자르기 → 노면 마스크 → 화질 측정·전처리 선택 → 전처리 → 균열·포트홀 후보 검출 → 노면 밖 후보 제거

| 파일 | 역할 |
|---|---|
| src/config.py | 모든 파라미터 값(단위·근거 주석) |
| src/io_utils.py | 사진 읽기, 가로 640px 맞춤, ROI 자르기, 노면 마스크, 파일 이름 해석 |
| src/analyze.py | 화질 지표(밝기·대비·흐림·잡음)를 재고 적용할 전처리 단계를 고름 |
| src/preprocess.py | 고른 단계만 적용: 잡음 제거(가우시안, 야간은 NLM), 밝기 보정(감마·히스토그램 평활화·CLAHE), 샤프닝 |
| src/detect.py | 배경보다 어두운 영역을 뽑아 모양(가늘고 긺 → 균열, 둥긂 → 포트홀)과 점수 임계값으로 후보를 고름 |
| src/evaluate.py | 정답 상자와 비교(IoU·중심 기준)해 정밀도·재현율, 영상 단위 민감도·특이도를 계산 |
| src/features.py | 에지 수·해리스 코너 수, SIFT 매칭(RANSAC inlier) |
| src/pipeline.py | 위 처리 순서를 한 장에 대해 실행 |
| src/visualize.py | 후보 윤곽·노면 마스크를 그린 비교 그림 |
| run_experiment.py · run_matching.py · demo.py | 실험 · 매칭 실험 · 시연 실행 파일 |
