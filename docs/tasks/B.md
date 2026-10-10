# B 작업 지시서 — 서민영 (후보 검출·평가)

먼저 [README.md](README.md)를 읽는다. 브랜치는 `feat/후보검출평가`. 계획서 확정본은 [docs/plan_2nd.md](../plan_2nd.md).

---

## 1. 목표와 계획서 연결

**목표**: 지금은 균열(Canny 에지)과 포트홀(DoG)을 서로 모르는 두 검출기로 따로 찾는다. 그래서 정답의 89%가 균열인데 후보의 95%가 포트홀이고, 손상 없는 사진 296장 중 275장에서 후보가 나왔다. 이것을 **배경보다 어두운 영역을 한 번에 뽑고(WP2), 모양으로 균열과 포트홀을 나누는(WP3)** 방식으로 바꿔 오검출을 줄인다. 또 정답 박스 기준을 공지하고 C의 표시분을 검수한다.

| 계획서 확정본 | 이 지시서에서 |
|---|---|
| 5장 (2) 공통 손상 후보 추출 | M3 (밝은 표시 억제, 두 크기 창 배경, 닫힘) |
| 5장 (3) 잡음 크기에 맞춘 적응형 임계값 | M3 (부호 있는 잔차의 MAD로 σ, 하한, 이중 문턱, σ를 노면 안에서 재기 비교) |
| 5장 (4) 형태 특징 기반 균열·포트홀 분류 | M4·M7, S1 (비원형도 주 기준, 구멍 둘레, 점수와 동작점) |
| 5장 (7) 데이터 분리와 보완 | M1 정답 기준 공지, M2 정답 10장, M5 교차 검수 |
| 4.2 학습과제 2번 형태학적 연산과 지역 배경 추정 | M3 전에 학습하고 공유 |
| 4.2 학습과제 3번 잡음 추정과 적응형 임계값 | M3 전에 학습하고 공유 |
| 4.2 학습과제 4번 형태 특징 기반 손상 분류 | M4 전에 학습하고 공유 (분포 확인 도구는 A가 만듦) |

확정본 6장대로 각 단계를 구현하기 전에 맡은 학습과제를 먼저 공부하고 단톡방에 3~5줄로 공유한다. 실험은 E2·E3·E4를 B가 돌린다(확정본 8장).

---

## 2. 맡은 파일·함수·config

| 파일 | 할 일 | 지금 상태 (기반) |
|---|---|---|
| `src/detect.py` | `extract_dark_regions(gray, road=None)`, `detect_cracks`·`detect_potholes`의 dark 갈래, 둘레에 구멍 윤곽 더하기 | dark는 '구현 안 함' 오류. legacy는 그대로. 후보 새 키 thickness·dt_width·circ_inv는 `shape_metrics`가 계산(바깥 윤곽만) |
| `src/config.py` B 구역 | `DETECT_MODE`, `DARK_*` 12개, `CRACK_MIN_CIRC_INV` 4, `POT_MAX_CIRC_INV` 3.5 | 출발값이 들어 있음, `DETECT_MODE = "legacy"` |
| `tests/test_b.py` | @todo 시험 9개를 통과시킴(3개는 Should) | 지금 도는 새 시험 4개는 통과 |
| `gt/` | 정답 기준 공지, C 표시분 검수, 직접 촬영 10장 표시 | |
| `docs/part_B.md`, `docs/ai_log_B.md` | 파라미터 변경, AI 활용 기록 | |

`src/evaluate.py`·`tools/evaluate_train.py`도 B 파일이지만, 평가 체계(WP0)·실행 옵션·인위 저하(E6)는 C가 B 동의를 받아 고친다(Plan.md 3.1). 그 PR은 B가 리뷰하고, B는 그 도구로 E2·E3·E4를 돌린다.

**함수 약속** (CONTRIBUTING 2장, 바꾸지 않는다)

```python
def detect_cracks(gray, road=None):    # → (균열 후보 list, Canny 에지 uint8 0/255)
def detect_potholes(gray, road=None):  # → (포트홀 후보 list, 이진 마스크 uint8 0/255)
# 후보 dict: area(int), elong·fill·solidity(float), bbox (x, y, w, h), contour (N, 1, 2) int32,
#            thickness·dt_width·circ_inv·contrast·score(float). 좌표는 노면 띠 기준
```

- `road`는 노면 마스크(띠와 같은 크기 0/255) 또는 None. **None이면 지금과 같아야 한다.** road로 후보를 지우지 않는다(지우는 일은 C의 pipeline). dark에서는 잡음 σ를 노면 안에서 잴 때만 쓴다(`DARK_SIGMA_IN_ROAD`).
- dark에서도 `detect_cracks`의 두 번째 값은 지금과 똑같은 Canny 에지다. row의 edges(에지 수)를 1차 결과와 비교하기 때문이다.
- `extract_dark_regions`는 B만 쓰는 내부 함수라 약속 표에 없다. docstring의 입출력을 지키면 된다.

---

## 3. 할 일 (Must → Should → Could)

### Must

**M1. 정답 표시 기준 공지** (10/10 09:00) — README 4장의 기준을 단톡방에 올린다. 예시 사진 한 장(균열 줄기마다 상자)을 붙이면 좋다.

**M2. 정답 박스 10장** (10/10 13:00) — 직접 촬영 10장(README 4장). 자기 몫 CSV만 커밋.

**M3. 어두운 영역 후보 `extract_dark_regions`** (10/10 오후) — Plan.md 3.4 (2). docstring에 6단계가 다 있다. 먼저 학습과제 2·3번을 단톡방에 공유한다.

1. **밝은 표시 억제**: 흑백 띠를 1/4로 줄여 중앙값 필터(`DARK_BG_KSIZE_SMALL`)로 거친 배경 Bc. `gray > Bc + DARK_BRIGHT_DELTA`인 픽셀(흰 차선·횡단보도)을 Bc로 바꾼 복사본 g2. 입력은 그대로 둔다.
2. **배경 B = 두 크기 창 중앙값의 큰 값**: 창 = 띠 높이 × `DARK_BG_FRAC`(0.5)와 × `DARK_BG_FRAC_LARGE`(1.0). 한 창이면 큰 포트홀의 가운데가 배경으로 흡수되어 빈다.
   - **중앙값 배경**: 넓은 범위의 밝기 중앙값. 작은 손상은 중앙값을 바꾸지 못하므로 '손상이 없을 때의 밝기'를 어림할 수 있다.
3. **잔차와 잡음 σ**: r = B − g2(부호 있는 실수), σ = max(1.4826 × MAD(r), `DARK_SIGMA_MIN`), D = max(r, 0).
   - **MAD**(중앙값 절대 편차): 값들이 중앙값에서 얼마나 떨어져 있는지의 중앙값. 1.4826을 곱하면 정규 잡음의 표준편차가 된다. **D의 MAD를 쓰면 σ = 0이 된다**(D = 0이 절반을 넘음). 꼭 부호 있는 r로 잰다.
   - road가 있고 `DARK_SIGMA_IN_ROAD`면 노면 안의 r만으로 잰다. 노면 픽셀이 하나도 없으면 띠 전체로.
4. **이중 문턱**: 강 D ≥ max(`DARK_K_HI`·σ, `DARK_ABS_MIN`), 약 D ≥ `DARK_K_LO`·σ. 약 영역 중 강 픽셀을 품은 것만 남긴다(`np.unique(labels[strong])`). 옅게 이어지는 균열을 진한 부분과 한 덩어리로 묶는다.
5. **닫힘 뒤 작은 영역 제거**: 닫힘(타원 `DARK_CLOSE_KSIZE`), 넓이 `DARK_MIN_AREA` 미만 삭제(골재 무늬).
6. **영역마다 값**: 기존 `describe_regions(mask, DARK_MIN_AREA, gray)`로 기본 키·darkness를 얻고, 둘레 P를 바깥 + 구멍 윤곽 길이로(`cv.findContours(..., cv.RETR_CCOMP, ...)` + `cv.arcLength`) 다시 재서 thickness = 2A/P, circ_inv = P²/(4πA)를 고친다. contrast = 영역 안 D의 평균 / σ.

- 반환: (영역 list, 어두운 영역 마스크 0/255).
- 시험: `test_dark_extract_format`의 `@todo`를 지운다(포트홀 하나가 한 영역, 평평한 영상 0개, road 전부 0이어도 오류 없음, road 전부 255 = road 없음).

**M4. dark 갈래와 분류 출발 규칙** (10/10 22:00 PR) — Plan.md 3.5 (2). 먼저 학습과제 4번을 단톡방에 공유한다.

| 판정 | 출발 규칙 |
|---|---|
| 균열 | circ_inv ≥ `CRACK_MIN_CIRC_INV`(4) |
| 포트홀 | circ_inv ≤ `POT_MAX_CIRC_INV`(3.5) 그리고 solidity ≥ `POT_MIN_SOLIDITY`(0.6) |
| 버림 | 그 사이 |

- **비원형도**(circ_inv) = 둘레² ÷ (4π × 넓이). 원은 1, 가늘고 길거나 갈래가 많을수록 커진다. 균열·포트홀을 가르는 주 기준이다.
- **두께**(thickness = 2A/P)는 넓은 균열에서 커지므로 보조로만 쓴다(첫 판의 '균열 t ≤ 4px'는 넓은 균열을 모두 버렸다).
- `detect_cracks`·`detect_potholes`의 `if _detect_mode() == "dark":` 아래 `raise NotImplementedError`를 지우고 위 규칙으로 채운다. 두 함수가 `extract_dark_regions`를 각자 한 번씩 불러도 된다(640px에서 몇 ms. 전역 캐시는 쓰지 않는다).
- **config의 `DETECT_MODE`는 아직 "legacy"로 둔다.** 실험은 `--detect dark`로 실행 중에만 바꾼다. 채택 규칙을 넘으면 G3에서 "dark"로 바꾼다.
- 시험: `test_dark_big_pothole_one_piece`, `test_dark_stripes_no_pothole`, `test_dark_wide_crack`, `test_dark_alligator_net`, `test_existing_tests_in_dark_mode`의 `@todo`를 지운다.
- **그림자 확인** (Plan.md 3.4 (3)): 첫 구현 때 United_States_004830, United_States_005996, 직접 촬영 그림자 사진을 본다. 그림을 보는 방법: B의 도구 `tools/tune_detect.py`의 `PARAM_SETS`에 `"DETECT_MODE": "dark"`인 세트를 더해 돌리거나(이 도구는 실행 중에만 config 값을 바꾸고 되돌린다), C에게 `run_experiment --detect dark` 그림을 받는다.

**M5. 교차 검수** (10/10 22:00) — C가 표시한 11장(README 4장). 엇갈리는 판정은 B가 정한다.

**M6. E2·E3 실행** (10/11 09:00 결과 공유, G3 근거) — 확정본 8장. C의 evaluate_train 옵션(10/10 17:00 병합)으로 개발셋에서 돌린다.

```bash
python tools/evaluate_train.py --split dev --detect dark --name e2_dark_dev                         # E2: 마스크 없음
python tools/evaluate_train.py --split dev --detect dark --mask on --name e3_dark_mask_dev           # E3: 노면 마스크
python tools/evaluate_train.py --split dev --detect dark --mask on --set DARK_SIGMA_IN_ROAD=False --name e3_sigma_band_dev   # E3: σ를 띠 전체에서
```

- 결과는 `results/<--name>/`에 쌓인다. `--name`을 빼면 `results/latest`에 써서 다음 실행이 덮어쓰므로 실험마다 이름을 준다. 1차 결과 폴더 이름(`base`·`tuning`·`diagnosis`)은 거부된다.
- 볼 파일: `train_summary.csv`(끔·켬 두 줄. `sensitivity`·`specificity`·`accuracy`·`baseline_none`·`balanced`·`fp_per_image`, 우연을 뺀 균열 재현율 `crack_ctr_recall_net`), `per_source.csv`(같은 요약을 국가별로), `train_detail.csv`(영상별), `config_used_train.txt`(돈 config 값과 실행 옵션).
- 옵션은 config.py 파일을 고치지 않고 실행 중에만 값을 바꾼 뒤 되돌린다. `--set`은 config에 있는 대문자 이름만 받는다(오타 방지).

- E1(C가 21:00에 공유)과 채택 규칙(4장)으로 비교해 단톡방에 표를 올린다. 옵션 이름은 C의 PR에서 확인한다.

**M7. 문턱 조정** (10/11 12:00) — A의 산점도(10/11 09:00)를 보고 `CRACK_MIN_CIRC_INV`·`POT_MAX_CIRC_INV`(필요하면 버림 구간)를 고친다. 개발셋만 쓴다. 바꾸면 `docs/part_B.md`에 전·후·이유.

**M8. 검출 파라미터 확정** (10/11 15:00) — G3 결정대로 `DETECT_MODE`를 정하고(채택이면 "dark"), 확정값을 병합한다. 이후 C가 E5·E6 최종과 시험셋을 돌린다. 그 뒤에는 고치지 않는다.

### Should — 실패하면 @todo로 두고 한계로 보고한다 (10/10 팀 결정)

- **S1. 점수와 문턱 τ** (E4, G3 뒤, Plan.md 3.5 (3)): score = contrast × 형태 항(문턱에서 멀수록 1에 가까움). τ 목록마다의 민감도·특이도 곡선은 C의 evaluate_train이 그린다. 기준선을 넘지 못하면 '영상당 오검출 ≤ 2'를 만족하는 τ를 고른다.
- **S2. 균열이 붙은 포트홀** (`test_dark_crack_touching_pothole`): 명세 그대로면 한 영역으로 합쳐져 균열 1개만 나온다(C 확인: c 16.1). 예: 선 폭보다 큰 원판으로 열림 → 덩어리만 남김 → 포트홀, 원래 영역 − 덩어리 → 균열.
- **S3. 넓은 흰 줄무늬** (`test_dark_wide_stripes_no_pothole`): 줄무늬가 억제 창(약 44px)의 절반을 넘으면 사이가 포트홀이 된다(C 확인: 9개). 대안: 기준 밝기를 Bc 대신 띠(또는 노면)의 중앙값으로.
- **S4. 흐린 그림자 기록** (`test_dark_soft_shadow`): 기대값이 없는 시험이다. `@todo`를 지우면 후보 수를 '기록'으로 출력한다(C 확인: 포트홀 후보 1개가 남음). 후보가 남으면 그림자 사진의 결과와 함께 한계로 보고한다.
- **S5. 그림자 경계 기울기**: 그림자 경계는 흐리고 포트홀 경계는 날카롭다(Plan.md 3.4 (3)).

### Could

- 헤시안 선 강조(Frangi), 원근 보정(Plan.md 3.5 (4)).

---

## 4. 통과 기준

**시험** (`python tests/test_b.py`, `python tests/check_contract.py`)

| 시험 | 등급 | C가 명세 그대로 메모리에서 돌려 본 결과 |
|---|---|---|
| `test_detect_road_arg_legacy`, `test_detect_mode_invalid`, `test_candidate_new_keys`, `test_shape_metrics_rule` | 항상 | 지금 통과 |
| `test_dark_extract_format` | Must | 통과 |
| `test_dark_big_pothole_one_piece` (반지름 90 → 1개) | Must | 통과 |
| `test_dark_stripes_no_pothole` (폭 12px 줄무늬 → 0개) | Must | 통과 |
| `test_dark_wide_crack` (폭 6~8px → 균열) | Must | 통과 (c 22~28) |
| `test_dark_alligator_net` (그물 → 균열) | Must | 구멍 둘레를 더하면 통과 (c 224), 바깥만이면 버림 (c 3.6) |
| `test_existing_tests_in_dark_mode` (기존 시험 전부) | Must | 통과 |
| `test_dark_soft_shadow` (기대값 없이 기록) | Should | 기록: 포트홀 후보 1개 (c 2.0) → 한계 후보 |
| `test_dark_crack_touching_pothole` | Should | 실패 — 한 영역으로 합쳐짐 → 안 되면 한계로 보고 |
| `test_dark_wide_stripes_no_pothole` | Should | 실패 — 포트홀 9개 → 안 되면 한계로 보고 |

check_contract의 `[B]` 검사: road=None 인자와 기본값, road를 줘도 약속 형식, road 배열을 바꾸지 않음, 후보 새 키(thickness·dt_width·circ_inv 유한한 실수, contrast·score 실수), 같은 입력이면 같은 출력.

**수치 — 채택 규칙** (Plan.md 3.10, 확정본 6장. 개발셋에서 세 가지를 모두 만족해야 dark를 채택)
1. 같은 동작점에서 비교한다(E3까지는 각 방법 그대로 비교하고 영상당 오검출을 함께 적음. 점수가 생기면 영상당 오검출 1·2개일 때의 민감도).
2. 균형 정확도가 오른다(E3 vs E1).
3. 우연을 뺀 균열 재현율(중심 기준)이 1%p 넘게 떨어지지 않는다.
- 최종 방법을 고를 때는 처리 시간도 함께 본다.

**목표** (확정본 5장): 시험용 152장에서 영상당 오검출 8.1 → 2개 이하, 특이도 7% → 50% 이상, 균열 재현율(중심) 4.9% 이상 유지. 영상 단위 정답률은 기준선(36.2%)과 함께 보고한다.

**눈으로 볼 그림** (Plan.md 1.6): United_States_004830 그림자, Japan_000107 자갈 경계(균열로 잡히는가), Japan_010993 넓은 균열(버려지는가), China_MotorBike_001096 점자 블록(사이가 포트홀·그물 균열로 잡히는가).

---

## 5. 의존 관계

| 받는 것 | 누구에게서 | 언제 | 없을 때 |
|---|---|---|---|
| 기반 | C | 10/10 10:00 | 시작하지 않는다 |
| 개발셋 나누기·`--detect`·`--mask`·`--set` | C | 10/10 17:00 | 합성 시험으로만 개발 |
| `road_mask` v1 | A | 10/10 18:00 | road 없이(σ를 띠 전체에서) 개발 |
| E1 결과 | C | 10/10 21:00 | E2와 E0으로만 비교 |
| 형태 산점도 | A | 10/11 09:00 | 출발 규칙 그대로 |

| 넘기는 것 | 누구에게 | 언제 |
|---|---|---|
| 정답 표시 기준 | 전원 | 10/10 09:00 |
| 교차 검수 결과(C 표시분) | C | 10/10 22:00 |
| dark 1차 병합(WP2 + 출발 규칙) | C(E6 1회)·A(산점도) | 10/10 22:00 |
| E2·E3 결과 | 전원 (G3) | 10/11 09:00 |
| 확정 파라미터 | C (E5·E6 최종, 시험셋) | 10/11 15:00 |

---

## 6. 마감과 21:00 점검 보고

- 10/10 21:00: 정답 10장·기준 공지 끝?, 학습과제 2·3·4번 공유, extract_dark_regions 진행(통과한 @todo 시험 수), 그림자 확인 결과, 막힌 것.
- 10/11 21:00(G4): 확정 파라미터, dark 채택 여부, Should 중 된 것·안 된 것(한계로 보고할 것).
- 보고 모양은 README 3장.

---

## 7. 커밋·PR 규칙

- 시작 전에: GitHub Desktop **Branch → Update from main** (Changes가 비어 있는지 먼저 확인).
- 내 파일만: `src/detect.py`, `src/config.py`(B 구역만), `tests/test_b.py`, `tools/tune_detect.py`, `tools/label_gt.py`, `tools/convert_json_gt.py`, `docs/part_B.md`, `docs/ai_log_B.md`, 내가 표시한 `gt/<사진>.csv`. (`src/evaluate.py`·`tools/evaluate_train.py`는 지금 C가 고치므로 같은 시간에 고치지 않는다. 고쳐야 하면 C와 먼저 말한다.)
- GitHub Desktop Changes에서 위 파일만 체크한다. `src/config.py`는 오른쪽 diff에서 B 구역만 바뀌었는지 본다.
- 메시지: `[B] extract_dark_regions: 두 창 배경·부호 있는 잔차 σ·이중 문턱` 처럼.
- config 값을 바꾸면 `docs/part_B.md`에 한 줄, AI 도구를 쓰면 `docs/ai_log_B.md`에 한 줄.
- PR 제목 앞에 [B]. 리뷰는 C. C가 B 파일을 고친 PR('B 파일, B 동의')은 B가 리뷰한다.

---

## 8. AI 도구 시작 프롬프트

```
너는 3조 PBL 저장소에서 파트 B(후보 검출·평가) 담당 서민영을 돕는다.
1. CLAUDE.md, CONTRIBUTING.md, docs/tasks/README.md, docs/tasks/B.md를 끝까지 읽고,
   계획서 확정본 docs/plan_2nd.md의 4.2(학습과제 2·3·4번)와 5장 (2)~(4),
   Plan.md의 1.6(첫 명세 재현), 3.4(어두운 영역 후보), 3.5(형태 분류), 3.10(채택 규칙)을 읽어라.
2. git branch --show-current로 지금 브랜치가 feat/후보검출평가인지 확인하라. 아니면 멈추고 알려 줘.
   main에서는 파일을 고치지 마라.
3. git log --oneline -12에 '[C] B 파일, 약속 변경 묶음' 커밋이 있는지 확인하라.
   없으면 GitHub Desktop에서 Branch → Update from main을 하라고 알려 주고 멈춰라.
4. python tests/check_contract.py와 python tests/test_b.py가 통과(건너뜀 표시 포함)하는지 확인하라.
5. docs/tasks/B.md 3장에서 아직 안 끝난 첫 번째 Must부터 하라. 구현 전에 그 단계의 학습과제를
   3~5줄로 요약해 보여 줘(내가 단톡방에 공유한다).
   src/detect.py의 extract_dark_regions docstring에 적힌 단계를 그대로 구현하라. B의 파일만 고치고,
   detect_cracks·detect_potholes의 이름·입력·출력은 바꾸지 마라. legacy 갈래(지금 코드)는 고치지 마라.
   숫자는 src/config.py B 구역의 DARK_* 값을 쓰고, 새 값은 B 구역 맨 아래에 둔다.
   config의 DETECT_MODE는 "legacy"로 두고, 시험은 tests/test_b.py처럼 실행 중에만 "dark"로 바꿔 확인하라.
6. 기능을 구현하면 그 기능의 @todo 줄을 지우고 시험을 통과시켜라. 시험의 기대값을 고쳐서 통과시키지 마라.
   Should 시험이 통과하지 않으면 @todo로 두고, 영역 수·비원형도 같은 측정값과 함께 한계로 정리해 줘.
7. 끝나면 check_contract와 test_b를 다시 돌려 결과를 보여 주고, 바꾼 파일을 요약하고,
   docs/ai_log_B.md 표에 한 줄 추가하라('받은 결과를 어떻게 고쳤나' 칸은 비워 둔다).
   커밋은 B의 파일만 골라 '[B] ...'로 하라. push와 PR은 내가 한다.
```
