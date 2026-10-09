# C 작업 지시서 — 김민기 (측정·실행·통합)

먼저 [README.md](README.md)를 읽는다. 브랜치는 `feat/측정실행통합`(기반은 `c-foundation`).

---

## 1. 목표와 계획서 연결

**목표**: 모든 개선을 같은 자로 잴 수 있게 **평가 체계**(영상 단위 판정, 개발/시험 분리, 우연 대조군)를 만들고, 실험 E0~E6을 돌려 **표·그림**을 만든다. 일부러 화질을 떨어뜨린 사진으로 **전처리 효과**를 정답과 함께 잰다(인위 저하 E6). 약속 문서·검사를 관리하고, 마지막에 C 컴퓨터 한 대에서 **최종 실행과 처리 시간**을 잰다.

| 계획서 | 이 지시서에서 |
|---|---|
| 5장 (1) 평가 체계 | 할 일 M2, M7, M8 |
| 5장 (5) 중 인위 저하 실험과 전처리 전·후 비교 지표(에지·특징점·매칭 점 수) | 할 일 M3, M8, M10 |
| 5장 (2) 중 노면 밖 후보 제거 | 기반에서 끝남(`USE_ROAD_MASK`), 켤지는 E1로 결정 |
| 4.2 학습과제 6번 성능 평가·전처리 효과 측정 | 민감도·특이도·균형 정확도·기준선, 우연 대조군, 개발/시험 분리, 인위 저하 |

계획서 5장 번호는 Plan.md 4장 초안 기준이다. 확정본에서 번호가 바뀌었으면 확정본 번호로 읽는다.

---

## 2. 맡은 파일·함수·config

**C 파일**

| 파일 | 할 일 | 지금 상태 (기반) |
|---|---|---|
| `src/pipeline.py` | (끝남) road·t_road_ms, `USE_ROAD_MASK`면 road를 넘기고 노면 밖 후보 삭제(`keep_on_road`) | 구현됨 |
| `src/visualize.py` | `draw_road(bgr, road, y0)` 칠하기 (Should) | 복사본만 돌려주는 자리 |
| `src/config.py` C 구역 | `USE_ROAD_MASK` False, `ROAD_KEEP_RULE` "center", `DEGRADE_*` 5개 | 출발값 |
| `run_experiment.py` | `--roi csv\|auto`, `--mask`, `--detect`, `--roi-top`, metrics 새 열, 표에 민감도·특이도·정답률 | 1차 그대로 |
| `run_matching.py`, `demo.py` | 최종 실행 / (Could) 노면 겹치기 | 1차 그대로 |
| `tests/test_c.py`, `tests/check_contract.py` | @todo 5개, 약속 검사 관리 | 기반에서 새 검사·시험 추가 |
| `CONTRIBUTING.md`, `README.md`, `Plan.md`, `docs/tasks/`, `docs/part_C.md`, `docs/ai_log_C.md` | 문서 | 기반에서 고침 |

**B 파일 (B 동의를 받고 C가 고침. 커밋 규칙은 7장)**

| 파일 | 할 일 |
|---|---|
| `src/evaluate.py` | `image_verdict(pred_by_kind, gt)`, `summarize_verdicts(verdicts)` 추가 (기존 함수는 그대로) |
| `tools/evaluate_train.py` | `DIAG35`, `assign_split`, `--split`, 영상 단위 요약, 국가별 표, 우연 대조군, `--mask`·`--roi-top`·`--detect`·`--degrade`·`--name`, 마스크 정답 유지율, 상태 판정 적중률, (Should) τ 곡선 |

**새 함수의 약속** (Plan.md 3.2 (2). 시험 기대값은 `tests/test_c.py`의 @todo 시험에 있다)

```python
# src/evaluate.py — 640px 전체 좌표
def image_verdict(pred_by_kind, gt):
    """pred_by_kind: {"crack": [(x, y, w, h)], "pothole": [...]}, gt: [(x, y, w, h, kind)]
    → {"has_damage", "n_pred", "hit_same", "hit_any", "correct"}"""
def summarize_verdicts(verdicts):
    """→ {"n", "n_damage", "n_none", "sensitivity", "specificity", "accuracy", "balanced", "baseline_none"}, 분모 0이면 NaN"""

# tools/evaluate_train.py
DIAG35 = (...)                  # 진단 35장 이름(확장자 없음, Plan.md 부록 A). '평가 분할 정의'이지 영상 처리 분기가 아님을 주석에
def assign_split(names):        # → {사진 이름: "dev" | "test"}. 국가별 이름순 순번 % 5 == 4면 test, DIAG35는 dev
def degrade(bgr, kind):         # kind: none·blur·dark·bright·noise → 같은 크기 BGR uint8. 값은 config C 구역 DEGRADE_*
```

- `image_verdict`·`summarize_verdicts`를 약속 표에 넣을지는 B와 정한다(넣으면 CONTRIBUTING 2장과 check_contract를 C가 고침).
- 실행 중에만 config를 바꾸는 옵션(`--mask`·`--roi-top`·`--detect`)은 evaluate_train의 `apply_param_set`·`restore_params`처럼 `finally`에서 되돌리고, 바꾼 값은 결과 폴더의 config_used에 남긴다. config 파일은 고치지 않는다.

---

## 3. 할 일 (Must → Should → Could)

### Must

**M0. 기반 병합** (10/10 10:00)
- 09:00 단톡방에 약속 변경 묶음 동의 요청(보고 메시지). 세 명이 동의하면 `c-foundation` → main PR, Create a merge commit.
- 병합 뒤 main에서 `python tests/check_contract.py`와 `python run_experiment.py --name foundation`을 돌려 1차와 같은지 확인하고 단톡방에 알린다(CONTRIBUTING 5장 '병합할 때마다').

**M1. 정답 박스 11장 + own_15 커밋** (10/10 13:00) — README 4장. `gt/own_15_pothole_dark.csv`는 C 컴퓨터에만 있어 같이 커밋한다.

**M2. WP0 평가 체계 + E0 재현** (10/10 17:00 PR) — Plan.md 3.2
- `evaluate.image_verdict`, `summarize_verdicts` → `test_image_verdict`, `test_summarize_verdicts`의 @todo 지우기.
- `evaluate_train`: `DIAG35`·`assign_split`·`--split dev|test|all`(기본 dev) → `test_dev_test_split`의 @todo 지우기. 결과 폴더 `--name`(results/<name>/), 영상 단위 요약 열, 국가별 표(`per_source.csv`).
  - 지금 evaluate_train은 파라미터 5세트를 모두 돈다. 기본을 config 값 한 세트로 하고 5세트는 옵션으로 둘지 B와 정한다(시간 5배 차이).
- **우연 대조군**(Plan.md 부록 B): 640px 높이가 같은 사진끼리 이름순으로 묶고, i번 사진의 후보를 i+1번 사진의 정답과 비교(마지막은 첫 사진과). 대조군 재현율 = 그렇게 센 찾은 정답 / 정답 수. 균열·포트홀 따로 '실제 − 대조군'.
- 실행 옵션 `--mask on|off`, `--roi-top 0.3`, `--detect legacy|dark`.
- **E0 재현** (Plan.md 3.2 (5)): `--split all`, 전처리 끔에서 정답률 10.4%, 찾은 정답 중심 67·IoU 21, 후보 6,588. 기반에서 메모리로 같은 값을 확인했으니, 다르면 새 코드 탓이다.
- PR이 병합되면 A에게 `DIAG35`·`assign_split`을 쓰라고 알린다.

**M3. 인위 저하 E6 도구** (10/10 저녁) — Plan.md 3.2 (6)
- `degrade(bgr, kind)`: 640px 컬러에, 띠를 자르기 전에. blur = 가우시안 σ 2, dark = 입력^2.2, bright = 입력^0.5, noise = 가우시안 잡음 σ 10(`DEGRADE_SEED` 고정). → `test_degrade`의 @todo 지우기.
- `--degrade none|blur|dark|bright|noise`: 저하마다 끔·켬.
- **상태 판정 적중률**: 넣은 저하에 맞는 전처리를 고른 사진의 비율(blur → sharpen, dark → 감마 < 1, bright → 감마 > 1, noise → denoise).
- 잴 것: 검출 지표, 에지 수, 해리스 코너 수, 후보 수·면적, 처리 시간, 상태 판정 적중률.

**M4. E1·E1b** (10/10 21:00) — A의 road_mask v1이 병합되면
- E1: `--split dev --mask on`. 마스크 정답 유지율(띠 안 정답 중 상자 중심이 노면 안인 비율)과 후보 감소율. 통과 기준 95% 이상·35% 이상, 93% 미만이면 되돌림(Plan.md 3.3 (8)).
- E1b: `--mask on --roi-top 0.3`. 띠 밖 정답(드론·오토바이)을 되찾는지, 후보가 얼마나 느는지. 21:00에 E0·E1·E1b 표를 단톡방에 올리고 띠 값·마스크 켬을 정한다. 마스크를 켜기로 하면 C 구역 `USE_ROAD_MASK = True`, 띠 값이 바뀌면 공통 구역 `ROI_TOP_DEFAULT`(세 명 합의) — 둘 다 `docs/part_C.md`에 기록.

**M5. 교차 검수** (10/10 22:00) — A가 표시한 12장(README 4장).

**M6. E2·E3·E6 1회 → G3** (10/10 밤 ~ 10/11 10:00) — B의 dark 1차가 병합되면
- E2 `--detect dark`, E3 `--detect dark --mask on`(B의 `DARK_SIGMA_IN_ROAD` 켬·끔 비교 포함), 채택 규칙(Plan.md 3.10)으로 판정.
- E6 1회째를 이 새 검출기 코드로 돌린다(10/10 개정). 결과로 A가 G3에서 전처리 규칙을 정한다.
- G3에서 결정한 것(dark 채택, 마스크 켬, 띠 값, 전처리 규칙)을 단톡방과 Plan.md 맨 위 개정 내역(날짜 10/11)에 남긴다.

**M7. run_experiment 개선** (10/11 오전) — Plan.md 3.2 (4), 3.7
- `--roi csv|auto`(auto = roi.csv를 쓰지 않고 기본 띠 + 노면 마스크), `--mask`, `--detect`, `--roi-top`.
- metrics.csv 새 열: has_gt, has_damage, n_pred, hit_same, hit_any, correct, edges_road(노면 안 에지 수), t_road_ms. 정답 파일이 없으면 빈칸(NaN). row 9열은 그대로.
- table_by_source·table_by_condition에 민감도·특이도·정답률(정답 있는 영상만).

**M8. 시험셋 1회·E5·E6 2회·39장 최종** (10/11 17:00~)
- 15:00 B의 파라미터 확정 → **17:00 `--split test` 한 번만**. 그 뒤 파라미터를 고치지 않는다.
- E5: 최종 검출기의 전처리 끔 vs 켬(개발, 39장, 시험) + **상태 판정 검증**: 같은 장소 정상·흔들림 5쌍(own_03·09·10·11·19)에서 흔들림 쪽만 흐림으로 잡는가(A의 blur_ratio가 있으면 그것도), 사람이 판정한 제공 13장(A가 roi.csv에 넣은 태그)과 analyze 판정이 맞는가.
- E6 2회째(최종 코드), 39장 최종 판정(roi.csv·auto).
- 보고서 표 일곱 가지 초안(Plan.md 3.7): 단계별(E0~E6), 국가별, 조건별, 전처리 끔·켬, 특징점·매칭 전·후, 제공 vs 직접 촬영, 인위 저하.

**M9. G4** (10/11 21:00) — 코드 동결. 데모 영상 녹화.

**M10. 최종 실행** (10/12 오전, C 컴퓨터) — `python tests/check_contract.py`, `python run_experiment.py --name final`, `python tools/evaluate_train.py --split test`(10/11 숫자와 같아야 함, 다르면 원인만 찾음), `python run_matching.py --name final`. 처리 시간은 이 값만 보고서에 쓴다.

### Should

- **S1. draw_road**: 노면을 반투명 초록으로(y0만큼 내려), run_experiment `save_figure`의 후보 칸에 함께 → `test_draw_road_paint`의 @todo 지우기. 섞는 비율 같은 숫자는 C 구역에.
- **S2. `ROAD_KEEP_RULE` 50% 규칙**: 윤곽 픽셀의 50% 이상이 노면 안이면 남김. 'center'와 정답 유지율·후보 수로 비교.
- **S3. τ 곡선**(B의 E4 지원): 후보에 score가 있으면 τ마다 민감도·특이도·정답률·균형 정확도 CSV와 민감도-(1 − 특이도) 그림, 같은 동작점(영상당 오검출 1·2개)의 민감도.
- **S4. E1의 39장 auto 실행.**

### Could

- demo.py 노면 겹치기.

---

## 4. 통과 기준

**시험** (`python tests/test_c.py`, `python tests/check_contract.py`)

| 시험 | 할 일 | 지금 |
|---|---|---|
| 기존 22개 + `test_pipeline_road`, `test_road_mask_switch_off`, `test_road_mask_switch_on`, `test_keep_on_road`, `test_draw_road_copy` | — | 통과 |
| `test_image_verdict`, `test_summarize_verdicts`, `test_dev_test_split` | M2 | @todo |
| `test_degrade` | M3 | @todo |
| `test_draw_road_paint` | S1 | @todo |

기반에서 이 다섯 @todo 시험이 간단한 참고 구현으로 통과하는 것을 확인했다(기대값이 맞음).

**수치**
- E0 재현: 정답률 10.4%, 찾은 정답 중심 67·IoU 21, 후보 6,588(`--split all`, 전처리 끔). 켬은 후보 9,365, 정답률 9.7%.
- 나누기: 시험 152장(손상 97·무손상 55), 개발 652장. 고정 규칙이 없으면 시험 159장(부록 A의 굵은 7장이 시험으로 감).
- 기준선: 균형 정확도 50%, 정답률 36.8%(804장)·36.2%(시험 55/152). 정답률이 기준선을 넘는 조건: 민감도 > 0.58 × (1 − 특이도)(시험셋 0.57).

**눈으로 볼 그림**: run_experiment 비교 그림(끔·켬), A의 노면 마스크 확인 그림, B의 dark 후보 그림(Plan.md 부록 D).

---

## 5. 의존 관계

| 받는 것 | 누구에게서 | 언제 |
|---|---|---|
| 약속 변경 묶음 동의 | A·B | 10/10 09:00 |
| 정답 CSV, 제공 13장 상태 태그 | 전원, A | 10/10 13:00 |
| `road_mask` v1 | A | 10/10 18:00 |
| dark 1차 | B | 10/10 22:00 |
| 교차 검수 결과(C 표시분) | B | 10/10 22:00 |
| 확정 파라미터 | B | 10/11 15:00 |

| 넘기는 것 | 누구에게 | 언제 |
|---|---|---|
| 기반 병합 | 전원 | 10/10 10:00 |
| `DIAG35`·`assign_split`·실행 옵션 | A·B | 10/10 17:00 |
| E0·E1·E1b 결과 | 전원 | 10/10 21:00 |
| 교차 검수 결과(A 표시분) | A | 10/10 22:00 |
| E2·E3·E6 1회 결과 → G3 | 전원 | 10/11 10:00 |
| 시험셋 결과, E5·E6 표 | 전원 | 10/11 17:00~21:00 |
| final 결과 | 전원 | 10/12 오전 |

---

## 6. 마감과 21:00 점검 보고

- 10/10 21:00: 기반 병합·동의, 정답 11장, WP0 PR(시험 결과), E0 재현 값, E1·E1b 표.
- 10/11 21:00(G4): G3 결정 기록, 시험셋 결과, E5·E6 표, 표 일곱 가지 진행.
- 보고 모양은 README 3장.

---

## 7. 커밋·PR 규칙

- 시작 전에: **Branch → Update from main**. 기반이 병합된 뒤에는 `feat/측정실행통합`에서 일한다.
- C 파일만 체크해서 `[C] ...`.
- **B 파일(evaluate·evaluate_train)**: 그 파일만 담은 커밋을 따로 만들고 `[C] B 파일, B 동의: evaluate에 영상 단위 판정 추가`처럼 쓴다. AI 활용 기록은 `docs/ai_log_C.md`. PR 리뷰는 B.
- config C 구역·공통 구역을 바꾸면 `docs/part_C.md`에 한 줄. 공통 구역(`ROI_TOP_DEFAULT`)은 세 명 합의 뒤에만.
- 결과 파일(results/)은 올리지 않는다. 표·그림은 단톡방·드라이브로.

---

## 8. AI 도구 시작 프롬프트

```
너는 3조 PBL 저장소에서 파트 C(측정·실행·통합) 담당 김민기를 돕는다.
1. CLAUDE.md, CONTRIBUTING.md, docs/tasks/README.md, docs/tasks/C.md를 끝까지 읽고,
   Plan.md의 3.2(평가 체계), 3.7(통합·표), 3.10(실험 설계), 부록 B(재현 방법)를 읽어라.
2. git branch --show-current로 지금 브랜치가 feat/측정실행통합인지 확인하라. 아니면 멈추고 알려 줘.
   main에서는 파일을 고치지 마라.
3. git log --oneline -10에 c-foundation 병합 커밋이 있는지 확인하라.
   없으면 GitHub Desktop에서 Branch → Update from main을 하라고 알려 주고 멈춰라.
4. python tests/check_contract.py와 python tests/test_c.py가 통과(건너뜀 표시 포함)하는지 확인하라.
5. docs/tasks/C.md 3장에서 아직 안 끝난 첫 번째 Must부터 하라. C의 파일과, C.md에 C가 맡는다고 적힌
   B 파일(src/evaluate.py, tools/evaluate_train.py)만 고친다. B 파일을 고친 커밋은 그 파일만 담아
   '[C] B 파일, B 동의: ...'로 따로 만든다. 기존 함수의 이름·입력·출력은 바꾸지 마라.
6. 기능을 구현하면 그 기능의 @todo 줄을 지우고 시험을 통과시켜라. 시험의 기대값을 고쳐서 통과시키지 마라.
7. 확인용 실행은 메모리에서만 하고, results/에 쓰는 실행은 내가 하라고 할 때만 한다.
8. 끝나면 check_contract와 test_c를 다시 돌려 결과를 보여 주고, 바꾼 파일을 요약하고,
   docs/ai_log_C.md 표에 한 줄 추가하라('받은 결과를 어떻게 고쳤나' 칸은 비워 둔다).
   커밋까지만 하고 push와 PR은 내가 한다.
```
