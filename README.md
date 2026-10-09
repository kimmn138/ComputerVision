# pbl1_road

PBL 모듈 1 · 저품질 도로 영상을 전처리한 뒤 균열·포트홀 후보 영역을 찾고, 전처리 전·후를 정량 비교한다.

## 시작하기

```bash
git config --global pull.rebase false     # 처음 한 번만
git clone <저장소 주소>
cd ComputerVision
git switch feat/전처리                    # 자기 브랜치 (B: feat/후보검출평가, C: feat/측정실행통합)
python -m venv .venv                      # 윈도: .venv\Scripts\activate   맥: source .venv/bin/activate
pip install -r requirements.txt
python tests/check_contract.py            # "약속 검사 통과"가 나오면 준비 끝
python run_experiment.py --name check
```

사진은 Git에 올리지 않는다. 구글 드라이브 공유 폴더의 사진을 data/provided, data/own, data/pairs에 복사한다.

## 폴더

| 경로 | 내용 | 주인 |
|---|---|---|
| src/ | 파이프라인 모듈 (io_utils·analyze·preprocess = A, detect = B, config·features·visualize·pipeline = C) | 파일별 |
| data/ | 사진 폴더(provided·own·pairs·train, Git 제외), roi.csv, shot_log.csv | A |
| gt/ | 정답 박스 CSV | B |
| tools/ | 파트별 도구 스크립트 | A·B |
| tests/ | 약속 검사(check_contract.py)와 파트별 시험 | C·각자 |
| docs/ | 파트별 파라미터 기록과 AI 활용 기록, 작업 지시서(docs/tasks) | 각자 (docs/tasks는 C) |
| Plan.md | 2차 개선 계획: 기준 수치, 실험 설계, 일정 | C |
| run_experiment.py · run_matching.py · demo.py | 실험·매칭 실험·시연 실행 파일 | C |

## 2차 개선 (10/10 ~ 10/12)

할 일은 팀원별 작업 지시서 docs/tasks/(README.md를 먼저)에, 근거와 숫자는 Plan.md에 있다.
일을 시작하기 전에 GitHub Desktop에서 Branch → Update from main을 한다.

## 규칙

협업 규칙은 CONTRIBUTING.md, AI 코딩 도구(Claude Code 등)에 주는 규칙은 CLAUDE.md에 있다.
시험과 도구는 모두 맨 위 폴더에서 파일 이름으로 실행한다. 예: python tests/test_a.py, python tools/check_data.py, python tools/label_gt.py [영상 경로]
영상 경로가 필요한 label_gt·show_preprocess·demo는 경로를 빼면(VS Code ▶ 버튼 포함) data 아래 사진 목록을 번호로 보여 주고 고르게 한다.
