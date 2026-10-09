# A 작업 지시서 — 배한원 (데이터·전처리)

먼저 [README.md](README.md)를 읽는다. 브랜치는 `feat/전처리`. 계획서 확정본은 [docs/plan_2nd.md](../plan_2nd.md).

---

## 1. 목표와 계획서 연결

**목표**: 사진마다 도로 면(노면)만 자동으로 골라내는 **노면 마스크**를 만들어, 건물·차·풀숲·보닛에서 나오는 오검출을 줄인다. 흔들려 찍은 사진을 노면 결과 상관없이 찾는 **흐림 판정**을 보완한다. B가 균열·포트홀 기준값을 정할 수 있게 후보 모양을 그림으로 보여 주는 **산점도 도구**를 만든다.

| 계획서 확정본 | 이 지시서에서 |
|---|---|
| 5장 (1) 자동 노면 ROI 생성과 보정 | M2 road_mask v1, M5 확인 그림, S2 v2 |
| 5장 (5) 영상 상태 판정과 선택적 전처리 보완 | M3 흐림 판정(재흐림 비율), S1 감마 상한·CLAHE·야간 처리·밝기·잡음 판정 |
| 5장 (7) 데이터 분리와 보완 | M1 정답 표시(제공 12장)·제공 13장 상태 태그, M4 교차 검수 |
| 5장 (4) 형태 특징 기반 분류 (B 담당. A는 분포 확인 도구) | M6 형태 산점도, S3 격자 탐색 |
| 4.2 학습과제 1번 색 공간과 자동 노면 분할 | M2 전에 학습하고 공유 |
| 4.2 학습과제 5번 영상 상태 판정 | M3 전에 학습하고 공유 |

확정본 6장대로 각 단계를 구현하기 전에 맡은 학습과제를 먼저 공부하고 단톡방에 3~5줄로 공유한다.

---

## 2. 맡은 파일·함수·config

| 파일 | 할 일 | 지금 상태 (기반) |
|---|---|---|
| `src/io_utils.py` | `road_mask(bgr_roi)` → uint8 (H, W), 노면 255·아님 0 | 전부 255를 돌려주는 자리 (TODO(A)) |
| `src/analyze.py` | `measure_quality`의 `blur_ratio`, `choose_steps`가 blur_ratio를 씀 (Must) | blur_ratio = NaN |
| `src/config.py` A 구역 | `ROAD_BLUR_SIGMA` 3, `ROAD_SEED_BOX` (0.30, 0.40, 0.70, 0.80), `ROAD_CHROMA_MAX` 10, `ROAD_L_K` 4, `ROAD_OPEN_KSIZE` 5, `ROAD_CLOSE_KSIZE` 31, `BLUR_RATIO_MAX` 1.5(임시) | 출발값이 들어 있음 |
| `tools/show_road_mask.py` | 노면 마스크 확인 그림 + 정답 유지율 | 인자·입출력 설명만 있는 뼈대 |
| `tools/shape_scatter.py` | 형태 산점도(+ 격자 탐색) | 인자·입출력 설명만 있는 뼈대 |
| `tests/test_a.py` | @todo 시험 5개를 통과시킴(v2 시험은 Should) | 형식 시험은 이미 통과 |
| `data/roi.csv` | 제공 13장 condition 칸(상태 태그) | 비어 있음 |
| `docs/part_A.md`, `docs/ai_log_A.md` | 파라미터 변경, AI 활용 기록 | |

**함수 약속** (CONTRIBUTING 2장, 바꾸지 않는다)

```python
def road_mask(bgr_roi):        # crop_roi가 돌려준 컬러 띠 BGR uint8 (H, W, 3)
    ...                        # → 같은 크기 uint8 (H, W), 값은 0 또는 255
```

- 입력을 바꾸지 않는다(복사해서 쓴다). 빈 영상(모든 픽셀이 같은 값)에서도 오류가 나지 않는다. 같은 입력이면 같은 출력.
- 사진 이름으로 나누지 않는다. 숫자는 config A 구역 값만 쓴다.
- quality dict에는 키를 더할 수만 있다. `blur_ratio`는 float(평평한 영상처럼 계산할 수 없으면 NaN 허용).

---

## 3. 할 일 (Must → Should → Could)

### Must

**M1. 정답 박스 12장 + 제공 13장 상태 태그** (10/10 13:00) — README 4장. 제공 사진 12장. 끝나면 `gt/` 자기 몫 CSV만 커밋.
- 상태 태그는 **단톡방에서 A가 동의하면** 한다. 정답을 그리면서 제공 13장의 상태를 보고 `data/roi.csv`의 condition 칸(지금 비어 있음)에 `normal`·`blur`·`dark`·`bright` 중 하나를 적는다. Plan.md 1.3의 '사람 판정' 표(흐림 3, 과노출 1, 어두움 2, 정상 7)를 참고하되 직접 보고 정한다. ROI 값(roi_top·roi_bottom)은 바꾸지 않는다.
- 이 태그는 E5의 상태 판정 검증('사람이 판정한 제공 13장')과 조건별 표에 쓰인다.

**M2. road_mask v1** (10/10 18:00 PR) — Plan.md 3.3 (2), 참고 코드는 Plan.md 부록 C. 먼저 학습과제 1번을 단톡방에 공유한다.

- 단계: 컬러 띠를 흐림(`ROAD_BLUR_SIGMA`) → Lab 색 공간 → 아래 가운데 씨앗 영역(`ROAD_SEED_BOX`)의 색·밝기 중앙값 → 색 거리 < `ROAD_CHROMA_MAX`이고 밝기 차 < `ROAD_L_K` × (1.4826 × MAD + 2)인 픽셀 → 열림(`ROAD_OPEN_KSIZE`) → 씨앗과 이어진 연결 요소만 → 닫힘(타원 `ROAD_CLOSE_KSIZE`) → 구멍 메우기.
  - **Lab 색 공간**: 밝기(L)와 색(a·b)을 나눠 둔 색 표현. 그림자는 주로 L만 바뀌어 색(a·b)으로 노면을 잇기 쉽다.
  - **씨앗 영역**: '여기는 분명히 노면'이라고 보는 띠의 아래 가운데. 여기와 비슷하고 이어진 곳을 노면으로 넓혀 간다.
  - **연결 요소**: 서로 붙어 있는 픽셀 덩어리. 씨앗과 붙은 덩어리만 남긴다.
  - **구멍 메우기**: 노면 안의 포트홀·차선·맨홀처럼 노면에 둘러싸인 빈 곳을 채워, 손상이 마스크에서 빠지지 않게 한다.
- 부록 C 코드를 옮길 때: 숫자 → `C.ROAD_*`, 한국어 docstring 첫 줄에 무엇을 왜, print 금지, `fill_holes`는 io_utils 안의 `_fill_holes` 같은 작은 도우미로.
- 빈 영상: 씨앗 MAD가 0이어도 `+ 2`가 있어 나눗셈 문제는 없다. 그래도 `test_road_mask_format`(빈 영상 포함)을 꼭 돌린다.
- 시험: `test_road_mask_v1_synthetic`의 `@todo` 줄을 지우고 통과시킨다. C가 부록 C 코드로 이 시험이 통과하는 것을 미리 확인했다(노면 0.99, 포트홀 1.0, 차선 조각 1.0, 풀 0.0, 건물 0.001).
- `python tests/check_contract.py` — `[A] road_mask ...` 실패가 없어야 한다. pipeline에 연결하는 일은 기반에서 끝났다(C).

**M3. 흐림 판정(재흐림 비율)** (10/10 22:00 PR, E6 1회 전) — Plan.md 3.6 1번. 확정본 8장대로 v1 바로 뒤에 한다. 먼저 학습과제 5번을 단톡방에 공유한다.
- `blur_ratio` = Sobel 기울기 크기 평균 ÷ (가우시안 σ 1.5를 건 뒤의 같은 값). 이미 흐린 사진은 더 흐려도 덜 변해 1에 가깝고, 선명한 사진은 크다. 비율이라 노면 결의 세기와 상관이 없다. σ 1.5는 config A 구역 맨 아래에 `BLUR_RATIO_SIGMA`로 둔다.
- `choose_steps`: `lap_var < BLUR_VAR_MAX` **또는** `blur_ratio < BLUR_RATIO_MAX`이면 샤프닝. blur_ratio가 NaN이거나 키가 없으면 지금처럼 lap_var만 본다.
- 시험: `test_blur_ratio_synthetic`, `test_choose_steps_blur_ratio`, `test_blur_ratio_provided`의 `@todo`를 지운다.
- C가 같은 식으로 미리 재 본 값(참고): 제공 13장 중 흐린 3장 1.13~1.18, 나머지 1.87 이상 → 임시값 1.5. 같은 장소 5쌍(흔들림 vs 정상): own_03 2.31 < 3.77, own_09 2.16 < 3.65, own_10 3.61 < 4.29, own_11 2.09 < 2.48, own_19 3.398 ≈ 3.395(거의 같음). 5쌍은 시험이 아니라 E5 상태 판정 검증에서 C가 보고한다.
- 이 값이 E6 1회째(10/10 밤)의 상태 판정 적중률에 들어가고, 그 결과로 G3에서 전처리 규칙을 정한다. 전처리 보완을 줄여도 흐림 판정은 남긴다.

**M4. 교차 검수** (10/10 22:00) — B가 표시한 10장(README 4장).

**M5. 확인 그림 v1** (10/11 09:00) — `tools/show_road_mask.py`, Plan.md 3.3 (5)·(8)

- `--set dev39`(제공 13 + 직접 촬영 26)부터. 사진마다 640px 사진 위에 띠 시작선(노랑), 노면(반투명 초록), 정답 박스를 그려 `results/road_mask/dev39/`에 저장. `summary.csv`에 마스크 넓이 / 띠 넓이, 띠 안 정답 수, 노면 안에 남은 정답 수.
- 정답 유지율 = 노면 안에 남은 정답 / 띠 안 정답. 정답 상자 중심(640px 좌표)에서 y0를 빼서 마스크 좌표로 바꾼다.
- 반투명 칠하기는 `cv.addWeighted`로 직접 해도 되고, C가 `visualize.draw_road`를 구현하면(10/11) 그것을 써도 된다.
- `--set diag35`는 C가 `tools/evaluate_train.py`에 `DIAG35`(35장 이름)를 넣은 뒤(10/10 17:00) `from tools.evaluate_train import DIAG35`로 쓴다. 그 전에는 Plan.md 부록 A를 보고 같은 이름을 임시로 둔다.
- 마스크 채택은 10/10 21:00에 E1 수치로 정하고, 이 그림으로 G3 전에 다시 확인한다. 눈으로 볼 것(Plan.md 3.3 (4)): United_States_004830(그림자 노면이 빠지는가), United_States_005996(햇빛 노면), China_MotorBike_002209(차선 건너 차로), Japan_013055(경찰차), Japan_000156(인도), Norway_008315(흙 비탈), Japan_000107(자갈). 결과를 몇 장 골라 단톡방에 올린다.

**M6. 형태 산점도** (10/11 09:00 → B) — `tools/shape_scatter.py`, Plan.md 3.5 (2), 학습과제 4번의 분포 확인 도구

- 개발셋 사진마다 `pipeline.run_pipeline(bgr, C.ROI_TOP_DEFAULT, C.ROI_BOTTOM_DEFAULT, False)`(기본 띠, 전처리 끔)의 후보를 모으고, 후보마다 `circ_inv`(가로축, 로그 눈금)와 `thickness`·`dt_width`(세로축)를 찍는다.
- 점 색 = 후보 상자 중심(y + y0)이 든 정답의 종류. JSON(`data/train/ann/<이름>.jpg.json`)의 classTitle: longitudinal crack = D00, transverse crack = D10, alligator crack = D20, pothole = D40, 어느 정답에도 안 들면 '없음'. 좌표는 `convert_json_gt.convert_box`처럼 640px로 바꾼다.
- 세로선으로 지금 `CRACK_MIN_CIRC_INV`(4)·`POT_MAX_CIRC_INV`(3.5)를 그린다.
- 지금(legacy) 후보에도 세 값이 있어서 B를 기다리지 않고 도구를 만들 수 있다. B의 dark 1차가 합쳐지면(10/10 22:00) `DETECT_MODE`를 실행 중에만 "dark"로 바꿔 다시 그린다(config 파일은 고치지 않는다).
- 개발셋 나누기는 C의 `assign_split`을 쓴다(10/10 17:00). 시험셋은 쓰지 않는다. 빨리 볼 때는 `--limit 50`.

### Should (G3 전에 끝난 것만 반영, G3 뒤에는 A 구역을 고치지 않음)

**S1. 나머지 전처리 보완** — Plan.md 3.6 2~4번, 확정본 5장 (5)
- 감마 상한(`GAMMA_RANGE` 2.5 → 예: 1.5), 평활화를 CLAHE로 바꾸거나 조건 좁히기, 야간 영상은 잡티를 먼저 줄이고 밝히기, 밝기·잡음 판정이 노면 색·골재 결에 덜 반응하게 보완.
- 감마와 평활화는 E6 1회째(B의 dark 1차) 결과를 보고 G3에서 정한다.
- **시간이 모자라면 CLAHE·야간 처리부터 뺀다.** 흐림 판정(M3)은 빼지 않는다.

**S2. 노면 마스크 v2** — Plan.md 3.3 (5)
- 차선이 노면을 가르는 문제부터: `test_road_mask_v2_lane_split`(띠를 위아래로 가르는 폭 8px 차선 → 건너편 노면 0.9 이상). v1은 건너편을 통째로 뺀다(C 확인: 0%). 방법 (가) 밝고 색이 없는 픽셀(차선)을 연결 판정에서 노면처럼 다룸, (나) 연결 요소를 고르기 전에 닫힘.
- 그다음 그림자·햇빛 경계, 보닛, 인도·흙·자갈(결·에지 밀도, 확정본 5장 (1)). 새 숫자는 A 구역 맨 아래에.
- 확인 그림에서 v1·v2를 나란히(config 값만 실행 중에 바꿔 두 번 부르고 되돌림).

**S3. 격자 탐색** — `shape_scatter --grid`: (CRACK_MIN_CIRC_INV, POT_MAX_CIRC_INV) 격자마다 종류별 F1·균형 정확도를 `grid.csv`로.

### Could

- 흔들린 방향을 보는 방향별 에지 세기(확정본 5장 (5)).
- 촬영 기록 정리(확정본 7장 A 업무): `data/shot_log.csv`가 실제 26장과 맞지 않음(옛 이름), `data/roi.csv`의 쌍 행이 실제 파일과 다름(Plan.md 3.8).

---

## 4. 통과 기준

**시험** (`python tests/test_a.py`, `python tests/check_contract.py`)

| 시험 | 언제 | 지금 |
|---|---|---|
| `test_road_mask_format`, `test_blur_ratio_key`, `test_new_tools_args` | 항상 | 통과 |
| `test_road_mask_v1_synthetic` | M2 | @todo |
| `test_blur_ratio_synthetic`, `test_choose_steps_blur_ratio`, `test_blur_ratio_provided` | M3 | @todo |
| `test_road_mask_v2_lane_split` | S2 (Should) | @todo |
| check_contract의 `[A]` 검사(road_mask 형식·입력 불변·빈 영상·같은 출력, blur_ratio 키) | 항상 | 통과 |

**수치**
- 노면 마스크(Plan.md 3.3 (8), E1에서 C가 개발셋으로 잰다): 정답 유지율 95% 이상, 후보 35% 이상 감소. 93% 미만이면 되돌린다(`USE_ROAD_MASK = False`, 띠 ROI 그대로).
- 흐림 판정: 사람이 흐리다고 본 제공 3장만 흐림(`test_blur_ratio_provided`). E6 blur 저하의 상태 판정 적중률과 E5의 5쌍 검증은 C가 잰다.

**눈으로 볼 그림**: M5의 일곱 장. 보닛·건물·풀숲이 빠졌는지, 그림자·햇빛 노면과 차선 건너 차로가 남았는지. 검은 차체는 색으로 가를 수 없어 한계로 둔다.

---

## 5. 의존 관계

| 받는 것 | 누구에게서 | 언제 | 없을 때 |
|---|---|---|---|
| 기반(자리·config·시험) | C | 10/10 10:00 | 시작하지 않는다 |
| 정답 표시 기준 | B | 10/10 09:00 | Plan.md 3.8 기준으로 표시 |
| `DIAG35`·`assign_split` | C | 10/10 17:00 | 부록 A로 임시 목록, 산점도는 `--limit`으로 도구만 먼저 |
| dark 1차 후보 | B | 10/10 22:00 | legacy 후보로 도구를 완성해 둔다 |
| E6 1회째(B의 dark 1차) 결과 | C | 10/11 09:00 | 전처리 규칙은 지금 그대로 둔다 |

| 넘기는 것 | 누구에게 | 언제 |
|---|---|---|
| 정답 CSV 12장, 제공 13장 상태 태그(roi.csv, A 동의 시) | C | 10/10 13:00 |
| `road_mask` v1 병합 | C (E1·E1b) | 10/10 18:00 |
| 흐림 판정(`blur_ratio`) 병합 | C (E6 1회) | 10/10 22:00 |
| 교차 검수 결과(B 표시분) | B | 10/10 22:00 |
| 확인 그림 v1 몇 장 + 정답 유지율 | 전원 | 10/11 09:00 |
| 형태 산점도 | B | 10/11 09:00 |
| 전처리 규칙 확정(A 구역 동결) | 전원 | G3 (10/11 10:00) |

---

## 6. 마감과 21:00 점검 보고

- 10/10 21:00: 정답 12장·상태 태그 끝?, road_mask v1 PR(병합 여부, 시험 결과), 흐림 판정 진행, 학습과제 1·5번 공유 여부, 막힌 것.
- 10/11 21:00(G4): 확인 그림·산점도 전달, S1·S2 중 G3 전에 반영한 것, 남은 버그.
- 보고 모양은 README 3장.

---

## 7. 커밋·PR 규칙

- 시작 전에: GitHub Desktop **Branch → Update from main** (Changes가 비어 있는지 먼저 확인).
- 내 파일만: `src/io_utils.py`, `src/analyze.py`, `src/preprocess.py`, `src/config.py`(A 구역만), `tools/show_road_mask.py`, `tools/shape_scatter.py`, `tools/check_data.py`, `tools/show_preprocess.py`, `tests/test_a.py`, `data/roi.csv`, `data/shot_log.csv`, `docs/part_A.md`, `docs/ai_log_A.md`, 그리고 내가 표시한 `gt/<사진>.csv`.
- GitHub Desktop Changes에서 위 파일만 체크한다. `src/config.py`는 오른쪽 diff에서 A 구역만 바뀌었는지 본다. 남의 파일·계획서 docx는 체크를 푼다.
- 메시지: `[A] road_mask v1: 씨앗 색·밝기 + 연결 요소 + 구멍 메우기` 처럼 무엇을 바꿨는지.
- config 값을 바꾸면 `docs/part_A.md` 표에 한 줄. AI 도구를 쓰면 `docs/ai_log_A.md`에 한 줄.
- PR 제목 앞에 [A], 양식 채우기, 확인 그림 몇 장 붙이기. 리뷰는 C.

---

## 8. AI 도구 시작 프롬프트

아래를 그대로 붙여 넣는다.

```
너는 3조 PBL 저장소에서 파트 A(데이터·전처리) 담당 배한원을 돕는다.
1. CLAUDE.md, CONTRIBUTING.md, docs/tasks/README.md, docs/tasks/A.md를 끝까지 읽고,
   계획서 확정본 docs/plan_2nd.md의 4.2(학습과제 1·5번)와 5장 (1)·(5),
   Plan.md의 3.3(노면 마스크), 3.6(전처리 보완), 부록 C(노면 마스크 참고 코드)를 읽어라.
2. git branch --show-current로 지금 브랜치가 feat/전처리인지 확인하라. 아니면 멈추고 알려 줘.
   main에서는 파일을 고치지 마라.
3. git log --oneline -12에 '[C] A 파일, 약속 변경 묶음' 커밋이 있는지 확인하라.
   없으면 GitHub Desktop에서 Branch → Update from main을 하라고 알려 주고 멈춰라.
4. python tests/check_contract.py와 python tests/test_a.py가 통과(건너뜀 표시 포함)하는지 확인하라.
5. docs/tasks/A.md 3장에서 아직 안 끝난 첫 번째 Must부터 하라. 구현 전에 그 단계의 학습과제를
   3~5줄로 요약해 보여 줘(내가 단톡방에 공유한다).
   A의 파일만 고치고, 함수 이름·입력·출력(CONTRIBUTING 2장)은 바꾸지 마라.
   숫자는 src/config.py A 구역 맨 아래에 두고, 한국어 docstring 첫 줄에 무엇을 왜 하는지 써라.
6. 기능을 구현하면 그 기능의 @todo 줄을 지우고 시험을 통과시켜라. 시험의 기대값을 고쳐서 통과시키지 마라.
   통과하지 않으면 이유와 측정값을 알려 줘.
7. 끝나면 check_contract와 test_a를 다시 돌려 결과를 보여 주고, 바꾼 파일을 요약하고,
   docs/ai_log_A.md 표에 한 줄 추가하라('받은 결과를 어떻게 고쳤나' 칸은 비워 둔다).
   커밋은 A의 파일만 골라 '[A] ...'로 하라. push와 PR은 내가 한다.
```
