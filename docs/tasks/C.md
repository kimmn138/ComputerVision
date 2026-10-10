# C 작업 지시서 — 김민기 (측정·실행·통합)

먼저 [README.md](README.md)를 읽는다. 브랜치는 `feat/측정실행통합`(기반은 `c-foundation`). 계획서 확정본은 [docs/plan_2nd.md](../plan_2nd.md).

---

## 1. 목표와 계획서 연결

**목표**: 모든 개선을 같은 자로 잴 수 있게 **평가 체계**(영상 단위 판정, 개발/시험 분리, 우연 대조군)를 만들고, 실험 E0·E1·E1b·E5·E6과 시험셋·39장 실행을 돌려 **표·그림**을 만든다. 일부러 화질을 떨어뜨린 사진으로 **전처리 효과**를 정답과 함께 잰다(인위 저하 E6). 최종 시연에서 노면 마스크를 보여 주고, 약속 문서·검사를 관리하며, 마지막에 C 컴퓨터 한 대에서 **최종 실행과 처리 시간**을 잰다.

| 계획서 확정본 | 이 지시서에서 |
|---|---|
| 5장 (6) 성능 평가 방법 보완 | M2 영상 단위 판정·우연 대조군, M7 표, M9 시험셋 |
| 5장 (7) 데이터 분리와 보완 | M2 개발/시험 분리, M6 인위 저하, M1·M5 정답 표시·교차 검수 |
| 5장 (1) 자동 노면 ROI 생성과 보정 | 노면 밖 후보 제거(기반에서 끝남), M3·M4 39장 자동 ROI·띠 넓히기(E1b), M8 시연의 노면 표시 |
| 5장 (5) 영상 상태 판정과 선택적 전처리 보완 | 전처리 전·후의 에지·특징점(해리스 코너·SIFT 매칭 점) 수·후보·시간 비교, E5 상태 판정 검증, E6 상태 판정 적중률 |
| 4.2 학습과제 6번 성능 평가와 전처리 효과 측정 | M2 전에 학습하고 공유 |

특징점·매칭은 학습과제가 아니다(수업에서 다룸). 전처리 전·후 비교 지표로만 쓴다(`run_matching`, row의 harris).

---

## 2. 맡은 파일·함수·config

**C 파일**

| 파일 | 할 일 | 지금 상태 (기반) |
|---|---|---|
| `src/pipeline.py` | (끝남) road·t_road_ms, `USE_ROAD_MASK`면 road를 넘기고 노면 밖 후보 삭제(`keep_on_road`) | 구현됨 |
| `src/visualize.py` | `draw_road(bgr, road, y0)` 칠하기 (Must, 시연) | 복사본만 돌려주는 자리 |
| `src/config.py` C 구역 | `USE_ROAD_MASK` False, `ROAD_KEEP_RULE` "center", `DEGRADE_*` 5개 | 출발값 |
| `run_experiment.py` | `--roi csv\|auto`, `--mask`, `--detect`, `--roi-top`, metrics 새 열, 표에 민감도·특이도·정답률(기준선과 함께) | 1차 그대로 |
| `demo.py` | 상태 판정 값·고른 전처리·노면 마스크·후보를 끔·켬으로 (Must, 시연) | 노면 표시 없음 |
| `run_matching.py` | 최종 실행 | 1차 그대로 |
| `tests/test_c.py`, `tests/check_contract.py` | @todo 5개, 약속 검사 관리 | 기반에서 새 검사·시험 추가 |
| `CONTRIBUTING.md`, `README.md`, `Plan.md`, `docs/plan_2nd.md`, `docs/tasks/`, `docs/review/`, `docs/part_C.md`, `docs/ai_log_C.md` | 문서 | 기반에서 고침 |

**B 파일 (B 동의를 받고 C가 고침. 커밋 규칙은 7장)**

| 파일 | 할 일 |
|---|---|
| `src/evaluate.py` | `image_verdict(pred_by_kind, gt)`, `summarize_verdicts(verdicts)` 추가 (기존 함수는 그대로) |
| `tools/evaluate_train.py` | `DIAG35`, `assign_split`, `--split`, `--name`, 영상 단위 요약, 국가별 표, 우연 대조군, `--mask`·`--roi-top`·`--detect`·`--set NAME=VALUE`·`--degrade`, 마스크 정답 유지율, 상태 판정 적중률, (Should) τ 곡선 |

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
- 실행 중에만 config를 바꾸는 옵션(`--mask`·`--roi-top`·`--detect`·`--set`)은 evaluate_train의 `apply_param_set`·`restore_params`처럼 `finally`에서 되돌리고, 바꾼 값은 결과 폴더의 config_used에 남긴다. config 파일은 고치지 않는다. `--set`은 B가 E3에서 `DARK_SIGMA_IN_ROAD=False`처럼 값 하나를 바꿔 비교할 때 쓴다.

---

## 3. 할 일 (Must → Should → Could)

### Must

**M0. 기반 병합** (10/10 10:00)
- 09:00 단톡방에 약속 변경 묶음 동의 요청(보고 메시지). 세 명이 동의하면 `c-foundation` → main PR, Create a merge commit.
- 병합 뒤 main에서 `python tests/check_contract.py`와 `python run_experiment.py --name foundation`을 돌려 1차와 같은지 확인하고 단톡방에 알린다(CONTRIBUTING 5장 '병합할 때마다').

**M1. 정답 박스 11장** (10/10 13:00) — README 4장. (own_15는 기반에서 커밋함)

**M2. WP0 평가 체계 + E0 재현** (10/10 17:00 PR) — Plan.md 3.2. 먼저 학습과제 6번을 단톡방에 공유한다.
- `evaluate.image_verdict`, `summarize_verdicts` → `test_image_verdict`, `test_summarize_verdicts`의 @todo 지우기.
- `evaluate_train`: `DIAG35`·`assign_split`·`--split dev|test|all`(기본 dev) → `test_dev_test_split`의 @todo 지우기. 결과 폴더 `--name`(results/<name>/), 영상 단위 요약 열, 국가별 표(`per_source.csv`).
  - (10/11 구현) 기본 실행은 config 값 한 세트이고, 1차 튜닝 5세트는 `--param-set base|…|all`로만 돈다(base = 1차 기준 값, E0 재현 확인용). 결과는 `results/<--name>/`, 이름을 빼면 `results/latest`. B의 확인은 PR 리뷰에서 받는다.
- **우연 대조군 (E0부터 Must)** (Plan.md 부록 B): 640px 높이가 같은 사진끼리 이름순으로 묶고, i번 사진의 후보를 i+1번 사진의 정답과 비교(마지막은 첫 사진과). 대조군 재현율 = 그렇게 센 찾은 정답 / 정답 수. 균열·포트홀 따로 '실제 − 대조군'. 채택 규칙 3번이 이 값을 쓴다.
- 실행 옵션 `--mask on|off`, `--roi-top 0.3`, `--detect legacy|dark`, `--set NAME=VALUE`.
- **E0 재현** (Plan.md 3.2 (5)): `python tools/evaluate_train.py --split all --name e0_all`, 전처리 끔에서 정답률 10.4%, 찾은 정답 중심 67·IoU 21, 후보 6,588. 기반에서 메모리로 같은 값을 확인했으니, 다르면 새 코드 탓이다.
- PR이 병합되면 A에게 `DIAG35`·`assign_split`을, B에게 `--detect`·`--mask`·`--set`을 쓰라고 알린다.

**M3. run_experiment의 39장 자동 ROI** (10/10 저녁, E1 전) — Plan.md 3.2 (4), 3.7. E1에서 39장을 손으로 정한 ROI 없이도 돌려야 한다(교수님 질문 'ROI 자동 설정', 확정본 6장 E1).
- `--roi csv|auto`: auto는 roi.csv를 쓰지 않고 기본 띠(`ROI_TOP_DEFAULT`, `--roi-top`으로 바꿀 수 있음) + 노면 마스크로 돈다.
- `--mask on|off`, `--detect legacy|dark`. 바꾼 값은 config_used.txt에 남긴다.

**M4. E1·E1b** (10/10 21:00) — A의 road_mask v1이 병합되면
- E1: 개발셋 `evaluate_train --split dev --mask on --name e1_mask_dev`, 39장 `run_experiment --mask on --name e1_mask`(roi.csv)과 `--mask on --roi auto --name e1_mask_auto`. 마스크 정답 유지율(띠 안 정답 중 상자 중심이 노면 안인 비율)과 후보 감소율. 통과 기준 95% 이상·35% 이상, 93% 미만이면 되돌림(Plan.md 3.3 (8)).
- E1b: `--mask on --roi-top 0.3 --name e1b_top03_dev`. 띠 밖 정답(드론·오토바이)을 되찾는지, 후보가 얼마나 느는지. 21:00에 E0·E1·E1b 표를 단톡방에 올리고 띠 값·마스크 켬을 정한다. 마스크를 켜기로 하면 C 구역 `USE_ROAD_MASK = True`, 띠 값이 바뀌면 공통 구역 `ROI_TOP_DEFAULT`(세 명 합의) — 둘 다 `docs/part_C.md`에 기록.

**M5. 교차 검수** (10/10 22:00) — A가 표시한 12장(README 4장).

**M6. 인위 저하 E6 → G3** (도구 10/10 저녁, 1회 실행 10/10 밤~10/11 09:00) — Plan.md 3.2 (6)
- `degrade(bgr, kind)`: 640px 컬러에, 띠를 자르기 전에. blur = 가우시안 σ 2, dark = 입력^2.2, bright = 입력^0.5, noise = 가우시안 잡음 σ 10(`DEGRADE_SEED` 고정). → `test_degrade`의 @todo 지우기.
- `--degrade none|blur|dark|bright|noise`: 저하마다 끔·켬.
- **상태 판정 적중률**: 넣은 저하에 맞는 전처리를 고른 사진의 비율(blur → sharpen, dark → 감마 < 1, bright → 감마 > 1, noise → denoise). A의 흐림 판정(10/10 22:00)이 들어간 뒤에 잰다.
- 잴 것: 검출 지표, 에지 수, 해리스 코너 수, 후보 수·면적, 처리 시간, 상태 판정 적중률.
- **1회째는 B의 어두운 영역 1차(10/10 22:00 병합)로 돌린다.** 결과를 10/11 09:00에 올리고, G3(10:00)에서 B의 E2·E3과 함께 dark 채택·전처리 규칙·마스크·띠 값 결정을 정리해 단톡방과 Plan.md 맨 위 개정 내역(날짜 10/11)에 남긴다. B의 1차가 늦어지면 전처리 확정만 E5 전까지 미룬다.

**M7. run_experiment 열과 표** (10/11 오전) — Plan.md 3.2 (4)
- metrics.csv 새 열: has_gt, has_damage, n_pred, hit_same, hit_any, correct, edges_road(노면 안 에지 수), t_road_ms. 정답 파일이 없으면 빈칸(NaN). row 9열은 그대로.
- table_by_source·table_by_condition에 민감도·특이도와 영상 단위 정답률(정답 있는 영상만, **항상 기준선과 같이**).

**M8. 시연의 노면 마스크** (10/11 오후, G4 전) — 확정본 6장 최종 시연
- `draw_road`: 노면을 반투명 초록으로(y0만큼 내려) → `test_draw_road_paint`의 @todo 지우기. 섞는 비율 같은 숫자는 C 구역에. run_experiment `save_figure`의 후보 칸에도 함께 그린다.
- `demo.py`: 영상을 넣으면 상태 판정 값(quality)과 고른 전처리, 노면 마스크, 균열·포트홀 후보를 전처리 끔·켬으로 나란히 보여 준다.

**M9. E5·E6 최종·39장 자동 ROI → 시험셋** (10/11 15:00~17:00) — 확정본 8장 순서
- 15:00 B의 파라미터 확정 → E5: 최종 검출기의 전처리 끔 vs 켬(개발, 39장) + **상태 판정 검증**: 같은 장소 정상·흔들림 5쌍(own_03·09·10·11·19)에서 흔들림 쪽만 흐림으로 잡는가, 사람이 판정한 제공 13장(A가 roi.csv에 넣은 태그)과 analyze 판정이 맞는가. E6 2회째(최종 코드), 39장 자동 ROI 실행.
- **17:00 `python tools/evaluate_train.py --split test --name test_once` 한 번만.** 그 뒤 파라미터를 고치지 않는다. 39장 최종 판정(roi.csv·auto).
- 보고서 표 초안(Plan.md 3.7): 단계별(E0~E6), 국가별, 조건별, 전처리 끔·켬, 특징점·매칭 전·후, 제공 vs 직접 촬영, 인위 저하.

**M10. G4** (10/11 21:00) — 코드 동결. 같은 화면을 데모 영상으로 녹화.

**M11. 최종 실행** (10/12 오전, C 컴퓨터) — `python tests/check_contract.py`, `python run_experiment.py --name final`, `python tools/evaluate_train.py --split test --name final_test`(10/11 숫자와 같아야 함, 다르면 원인만 찾음. run_experiment의 `final`과 폴더를 나눔), `python run_matching.py --name final`. 처리 시간은 이 값만 보고서에 쓰고, **검출 시간에 후보 형태 값 계산(두께·거리 변환 폭·비원형도)이 들어 있다고 적는다**(기반에서 legacy 기준 15~23% 늘어남, 팀이 받아들임).

### Should

- **S1. `ROAD_KEEP_RULE` 50% 규칙**: 윤곽 픽셀의 50% 이상이 노면 안이면 남김. 'center'와 정답 유지율·후보 수로 비교.
- **S2. τ 곡선**(B의 E4 지원): 후보에 score가 있으면 τ마다 민감도·특이도·정답률·균형 정확도 CSV와 민감도-(1 − 특이도) 그림, 같은 동작점(영상당 오검출 1·2개)의 민감도.

### Could

- 비교 그림에 상태 판정 값 적기, 결과 표 꾸미기.

---

## 4. 통과 기준

**시험** (`python tests/test_c.py`, `python tests/check_contract.py`)

| 시험 | 할 일 | 지금 |
|---|---|---|
| 기존 22개 + `test_pipeline_road`, `test_road_mask_switch_off`, `test_road_mask_switch_on`, `test_keep_on_road`, `test_draw_road_copy` | — | 통과 |
| `test_image_verdict`, `test_summarize_verdicts`, `test_dev_test_split` | M2 | @todo |
| `test_degrade` | M6 | @todo |
| `test_draw_road_paint` | M8 | @todo |

기반에서 이 다섯 @todo 시험이 간단한 참고 구현으로 통과하는 것을 확인했다(기대값이 맞음).

**수치**
- E0 재현: 정답률 10.4%, 찾은 정답 중심 67·IoU 21, 후보 6,588(`--split all`, 전처리 끔). 켬은 후보 9,365, 정답률 9.7%.
- 나누기: 시험 152장(손상 97·무손상 55), 개발 652장. 고정 규칙이 없으면 시험 159장(부록 A의 굵은 7장이 시험으로 감).
- 보고 지표(확정본 6장): 민감도·특이도·정밀도·재현율·균형 정확도·영상당 오검출, 영상 단위 정답률은 **항상 기준선과 같이**: 804장 36.8%, 시험셋 36.2%(55/152), 39장은 정답 표시 뒤 확정. 균형 정확도의 기준선은 50%. 정답률이 기준선을 넘는 조건: 민감도 > 0.58 × (1 − 특이도)(시험셋 0.57).
- 목표(확정본 5장): 시험셋 영상당 오검출 2개 이하·특이도 50% 이상·균열 재현율(중심) 4.9% 이상 유지, 39장 영상 단위 정답률 60% 이상(도전).

**눈으로 볼 그림**: run_experiment 비교 그림(끔·켬, 노면 겹침), A의 노면 마스크 확인 그림, B의 dark 후보 그림(Plan.md 부록 D), 시연 화면.

---

## 5. 의존 관계

| 받는 것 | 누구에게서 | 언제 |
|---|---|---|
| 약속 변경 묶음 동의 | A·B | 10/10 09:00 |
| 정답 CSV, 제공 13장 상태 태그(A 동의 시) | 전원, A | 10/10 13:00 |
| `road_mask` v1 | A | 10/10 18:00 |
| 흐림 판정(`blur_ratio`) | A | 10/10 22:00 |
| dark 1차 | B | 10/10 22:00 |
| 교차 검수 결과(C 표시분) | B | 10/10 22:00 |
| E2·E3 결과 | B | 10/11 09:00 |
| 확정 파라미터 | B | 10/11 15:00 |

| 넘기는 것 | 누구에게 | 언제 |
|---|---|---|
| 기반 병합 | 전원 | 10/10 10:00 |
| `DIAG35`·`assign_split`·실행 옵션 | A·B | 10/10 17:00 |
| E0·E1(39장 auto 포함)·E1b 결과 | 전원 | 10/10 21:00 |
| 교차 검수 결과(A 표시분) | A | 10/10 22:00 |
| E6 1회 결과 → G3 정리 | 전원 | 10/11 09:00 → 10:00 |
| E5·E6 최종, 시험셋 결과, 표 초안 | 전원 | 10/11 17:00~21:00 |
| final 결과 | 전원 | 10/12 오전 |

---

## 6. 마감과 21:00 점검 보고

- 10/10 21:00: 기반 병합·동의, 정답 11장, 학습과제 6번 공유, WP0 PR(시험 결과), E0 재현 값, E1(39장 auto 포함)·E1b 표.
- 10/11 21:00(G4): G3 결정 기록, 시험셋 결과, E5·E6 표, 시연 화면, 표 일곱 가지 진행.
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
   계획서 확정본 docs/plan_2nd.md의 4.2(학습과제 6번), 5장 (6)·(7), 6장(실행 계획),
   Plan.md의 3.2(평가 체계), 3.7(통합·표), 3.10(실험 설계), 부록 B(재현 방법)를 읽어라.
2. git branch --show-current로 지금 브랜치가 feat/측정실행통합인지 확인하라. 아니면 멈추고 알려 줘.
   main에서는 파일을 고치지 마라.
3. git log --oneline -15에 c-foundation 병합 커밋이 있는지 확인하라.
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
