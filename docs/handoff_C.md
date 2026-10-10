# C 인계 메모 (측정·실행·통합)

> 주인: C(김민기). 기준 시각: 2026-10-10 17:30. 이 시각 뒤의 상태는 `git log`·GitHub·단톡방이 원본이다.
> 지난 대화를 볼 수 없는 새 Claude(Claude Code·채팅 Claude)가 저장소만 읽고 C의 일을 이어받도록 쓴 메모다.
> 프로젝트 전체 요약은 `docs/project_brief.md`, 계획은 `docs/plan_2nd.md`, C의 할 일은 `docs/tasks/C.md`에 있다. 문서끼리 다르면 7장을 따른다.

## 0. 새 세션에서 먼저 할 일

1. `CLAUDE.md`, `CONTRIBUTING.md`, `docs/tasks/README.md`, `docs/tasks/C.md`, 이 문서를 읽는다.
2. `git branch --show-current`가 `feat/측정실행통합`인지, `git status`가 깨끗한지 본다. main에서는 파일을 고치지 않는다.
3. GitHub Desktop에서 Fetch(또는 `git fetch`)를 하고, 1장 이후 새 커밋·PR이 있는지 본다. 있으면 1장 표보다 그쪽이 맞다.
4. 시험을 돌린다(4.1의 환경 참고).

   ```bash
   .venv/Scripts/python.exe tests/check_contract.py    # "약속 검사 통과"
   .venv/Scripts/python.exe tests/test_c.py            # "C 시험 통과 (건너뜀 5건: 구현 전 기능)"
   ```

5. `data/train/img`에 사진 804장이 있는지 본다. 10/10 17:30에는 비어 있었다(1.3).
6. 3장 표에서 끝나지 않은 첫 작업부터 한다. 끝나면 `docs/ai_log_C.md`에 한 줄 남기고 C 파일만 골라 `[C] ...`로 커밋한다. push·PR은 사용자가 한다.

---

## 1. 지금 상태 (10/10 17:30)

### 1.1 브랜치

| 브랜치 | 로컬 | 원격 | 설명 |
|---|---|---|---|
| main | 69504d2 | 69504d2 | PR #4(기반) 병합 커밋, 10/10 05:55 |
| feat/측정실행통합 (C, 지금 브랜치) | 69504d2 + 이 인계 커밋 | 69504d2 | 인계 커밋은 사용자가 GitHub Desktop으로 push한다 |
| feat/전처리 (A) | 69504d2 | 69504d2 | 병합 뒤 새 커밋 없음 |
| feat/후보검출평가 (B) | 69504d2 | 69504d2 | 병합 뒤 새 커밋 없음 |
| c-foundation | d0b1d54 | 지워짐 | PR #4로 병합 끝. 로컬 브랜치는 지워도 된다 |

PR은 #1(feat/측정실행통합, 10/04)·#2(feat/전처리, 10/05)·#3(feat/후보검출평가, 10/05)·#4(c-foundation, 10/10) 네 개이고 모두 병합됐다. 열린 PR은 없다.

### 1.2 main에 합쳐진 것과 아닌 것

**합쳐진 것**

- 1차 구현(PR #1~#3)과 10/05 병합 뒤 점검 수정(감마 방향, 잡티 지표, 균열 판정 '또는', 밝은 선 거름, evaluate_train 두 판정 기준, 39장 ROI 지정, 사진 이름 정리).
- 10/10 기반(PR #4). 동작은 1차와 같고 자리만 만들었다.
  - 약속 변경 묶음 8개(CONTRIBUTING 7장 기록): `road_mask` 자리(전부 255), quality의 `blur_ratio`(NaN), `detect_*(gray, road=None)`·`DETECT_MODE = "legacy"`·`extract_dark_regions` 자리, 후보 새 키(thickness·dt_width·circ_inv는 계산, contrast·score는 NaN), pipeline의 `road`·`t_road_ms`·`USE_ROAD_MASK = False`·`keep_on_road("center")`, `draw_road` 자리(복사본만 돌려줌).
  - config 각 구역 맨 아래의 출발값, check_contract 새 검사, @todo로 건너뛰는 시험(test_a 5건, test_b 9건, test_c 5건).
  - 문서: Plan.md 10/10 개정, `docs/plan_2nd.md`, `docs/tasks/`, `docs/review/` 그림 2장, CONTRIBUTING·README.
  - 정답 CSV 6장(Japan_013055, own_01_none_blur, own_03_pothole_blur, own_03_pothole_normal, own_10_crack_blur, own_15_pothole_dark)과 `gt/train` 804개.
  - 1차와 같음을 확인한 값: 39장 row·후보, 804장 요약(후보 6,588, 찾은 정답 중심 67·IoU 21, 정답률 10.4%, 켬 9,365·9.7%). 검출 시간만 영상당 3.3~5.0ms(15~23%) 늘었다.

**아직 없는 것** (마감은 확정본 8장·`docs/tasks/README.md` 2장)

| 무엇 | 누구 | 마감 | 10/10 17:30 GitHub 상태 |
|---|---|---|---|
| 정답 표시 기준 공지 | B | 10/10 09:00 | 단톡방 일이라 저장소로는 알 수 없음 |
| 정답 CSV 33장 | A 12·B 10·C 11 | 10/10 13:00 | 없음 |
| 제공 13장 상태 태그(`data/roi.csv` condition 칸) | A(동의하면) | 10/10 13:00 | 비어 있음 |
| WP0 평가 체계·실행 옵션 | C | 10/10 17:00 | 시작 전. `src/evaluate.py`·`tools/evaluate_train.py`는 1차 그대로 |
| `road_mask` v1 | A | 10/10 18:00 | 자리(전부 255) |
| `blur_ratio` | A | 10/10 22:00 | NaN |
| dark 1차 | B | 10/10 22:00 | NotImplementedError |
| `draw_road` 칠하기·demo 노면 표시 | C | 10/11 21:00 | 자리 |

팀원 컴퓨터에 올리지 않은 작업이 있을 수 있다. 단톡방으로 확인한다.

### 1.3 지금 브랜치의 내용과 남은 확인 사항

**이 인계 커밋에 든 것**: `docs/project_brief.md`(사용자가 넣은 인계서를 저장소와 대조해 고침), `docs/handoff_C.md`(이 문서), `CLAUDE.md`(맨 아래 배경 문서 목록), `CONTRIBUTING.md`(파일 주인 표 C 줄에 두 문서), `docs/ai_log_C.md`.

이 커밋은 아직 `feat/측정실행통합`에만 있다. main에 PR로 합쳐야(리뷰는 A나 B) 팀원 브랜치와 그들의 AI 도구가 CLAUDE.md의 배경 문서 목록을 본다.

**확인할 것**

1. **C 컴퓨터의 `data/train/img`·`data/train/ann`이 비어 있다**(폴더만 있고 파일 0개). 정답 CSV(`gt/train` 804개)는 Git에 있지만 사진과 JSON이 없으면 E0 재현·E1·E1b·E6·시험셋·우연 대조군(사진 높이가 필요)·`test_dev_test_split`을 돌릴 수 없다. 드라이브에서 사진 804장을 `data/train/img`에, JSON(`<사진 이름>.jpg.json`)을 `data/train/ann`에 다시 넣는다. 비운 이유는 기록에 없다.
2. GitHub에 13:00 정답·팀원 작업이 없다. 단톡방에서 진행 상황을 묻는다.
3. **드라이브 사진 이름**: 10/05에 C가 로컬 사진 이름을 바꿨다(own_06_none_night → own_09_none_night, pair01_A_dark → pair01_B_dark, pair01_B_ref → pair01_A_ref, own_15_pothole_night·own_18_pothole_night·own_19_none_night → *_dark). `docs/ai_log_A.md`에 "드라이브의 파일 이름은 따로 바꿔야 함"이라고만 있고, 바꿨다는 기록은 없다. 팀원이 옛 이름을 받으면 roi.csv를 못 찾아 기본 ROI로 돌고(경고가 나옴) 정답 CSV 이름과도 어긋난다.
4. 단톡방 동의: 약속 변경 묶음(기반은 이미 병합됨), A의 상태 태그.

---

## 2. 기반 작업 보고 (10/10 04:40~05:55, c-foundation → PR #4)

대화에서 보고한 원문은 남아 있지 않다. 저장소의 기록(`docs/ai_log_C.md`의 10/10 다섯 줄, Plan.md 10/10 개정 내역)으로 다시 정리했다.

### 2.1 한 일

| 커밋 | 내용 |
|---|---|
| bb6c575 `[C] A 파일, 약속 변경 묶음` | road_mask·blur_ratio 자리, config A 구역 ROAD_* 6개·BLUR_RATIO_MAX, show_road_mask·shape_scatter 뼈대, test_a 새 시험 |
| b37f2ed `[C] B 파일, 약속 변경 묶음` | detect_*(gray, road=None)·DETECT_MODE, extract_dark_regions 자리, shape_metrics, config B 구역 DARK_* 등, test_b 새 시험 |
| 2956aab `[C]` | pipeline road·t_road_ms·USE_ROAD_MASK·keep_on_road, draw_road 자리, config C 구역, check_contract·test_c, CONTRIBUTING |
| 19c81b1 `[C]` | Plan.md 10/10 개정, `docs/tasks/` |
| afd7a31 `[C]` | 정답 박스 own_15_pothole_dark |
| 4bee1d3 `[C] B 파일` | 흐린 그림자를 기대값 없는 기록 시험으로, 그림자·붙은 포트홀·넓은 줄무늬 시험을 Should로 |
| 845da69 `[C]` | `docs/plan_2nd.md`, `docs/review/` |
| 60a500b `[C] A 파일` | test_a 흐림 판정 시험 3개의 이유를 Must로 |
| d0b1d54 `[C]` | 팀 답 반영: 확정본 번호, Should 조정, 정답률 보고, E6·일정, 처리 시간 원칙 |

**동작 보존을 확인한 방법** (다음에 '결과가 그대로인가'를 볼 때 같은 방법을 쓴다)

- main 코드를 임시 폴더에 꺼내고, 두 버전을 각각 subprocess로 돌려 결과를 메모리에서 비교했다(39장 끔·켬 78번의 row·후보·steps, 804장 1,608개 결과와 요약). 파일은 저장하지 않았다.
- check_contract는 일부러 14가지로 망가뜨려(형식, 입력 바꿈, 무작위, 빈 영상 오류, 키 없음, 음수 시간, 켬인데 안 지움 등) 맞는 파트 이름으로 실패하는지 봤다.

### 2.2 새 일정: 제안과 확정

- 제안(10/10 새벽): 구현 시작이 10/09에서 10/10으로 밀려, G3를 10/11 10:00으로(계획서의 '늦어지면 10/11 오전' 조항), 시험셋 평가를 15:00에서 17:00으로 옮김.
- 확정(10/10 세 명 합의, 확정본 6·8장): G3 10/11 10:00, 시험셋 10/11 17:00, G4 10/11 21:00, 10/12 오전 C 컴퓨터에서 최종 실행. E6 1회째는 B의 dark 1차로 10/10 밤~10/11 09:00에 돌리고, 그 결과로 G3에서 전처리 규칙을 정한다. 시각별 넘겨주기는 `docs/tasks/README.md` 2장.

### 2.3 Should로 내린 항목 (최종)

- 처음 제안: 격자 탐색, 노면 마스크 v1·v2 나란히 그림, draw_road 겹치기, E1의 39장 auto 실행, 넓은 줄무늬·균열이 붙은 포트홀 시험.
- 팀 답으로 조정한 결과(Plan.md 3.1 'Should 조정'):
  - Must로 되돌림: E1의 39장 auto 실행(교수님이 'ROI 자동 설정'을 물었음), E0의 우연 대조군.
  - 확정본에 맞춰 Must: 흐림 판정(blur_ratio), 시연의 노면 표시(draw_road·demo).
  - Should: 격자 탐색, v1·v2 나란히, 넓은 줄무늬·붙은 포트홀·흐린 그림자(실패하면 한계로 보고, 흐린 그림자는 기대값 없이 기록), 점수 τ(E4), 감마 상한·CLAHE·야간 처리, 노면 마스크 v2, ROAD_KEEP_RULE 50% 규칙, τ 곡선.
  - 시간이 모자라면 CLAHE·야간 처리부터 뺀다. 흐림 판정·우연 대조군·39장 auto 실행은 빼지 않는다.

### 2.4 단톡방 메시지

기반 때 보낸 메시지의 원문은 저장소에 없다(10/08판 예시는 Plan.md 3.9). 아래는 저장소 내용으로 다시 쓴 판이다. 이미 보냈다면 기록으로만 둔다.

```
[C] 2차 개선 기반을 main에 합쳤습니다 (PR #4).
1) 약속 변경 묶음 (CONTRIBUTING 7장에 기록)
 - A: io_utils.road_mask(bgr_roi) → 노면 0/255 (지금은 전부 255인 자리), quality에 blur_ratio (지금 NaN)
 - B: detect_cracks·detect_potholes(gray, road=None), DETECT_MODE("legacy" 그대로), 후보 새 키 5개
 - C: run_pipeline에 road·t_road_ms, USE_ROAD_MASK(기본 끔, 켜면 노면 밖 후보 삭제), visualize.draw_road 자리
 - ROI_TOP_DEFAULT 0.5 → 0.3은 E1b 결과를 보고 따로 정합니다.
2) 결과는 1차와 같습니다(39장 row·후보, 804장 후보 6,588·찾은 정답 67/21·정답률 10.4%).
   후보 형태 값 계산 때문에 검출 시간만 영상당 3~5ms 늘었습니다.
3) 각자: GitHub Desktop Branch → Update from main → check_contract와 자기 시험('건너뜀 n건'이면 정상)
   → docs/tasks/README.md와 자기 지시서 → 지시서 8장의 시작 프롬프트.
4) 13:00까지 정답 박스: A 제공 12장, B 직접 촬영 10장, C 직접 촬영 11장 (기준은 B 공지, README 4장).
5) A님, 정답을 그리면서 제공 13장 상태 태그(roi.csv condition 칸)를 넣어 주실 수 있는지 답 부탁드립니다.
```

### 2.5 남은 질문 (답이 저장소에 없음)

1. `image_verdict`·`summarize_verdicts`를 약속 표(CONTRIBUTING 2장)에 넣을지: B와 정한다. 넣으면 C가 CONTRIBUTING·check_contract를 먼저 고친다(7장 절차).
2. evaluate_train의 기본을 config 값 한 세트로 하고 5세트는 옵션으로 둘지: (10/11) 사용자 결정으로 C가 기본 = config 한 세트, 1차 5세트 = `--param-set base|…|all`로 구현했다. B의 확인은 PR 리뷰에서 받는다(5.2).
3. 제공 13장 상태 태그를 A가 넣을지: A의 동의가 필요하다. 안 되면 E5 검증과 조건별 표에 Plan.md 1.3의 '사람 판정' 표를 그대로 쓸지 팀이 정한다.

나머지 정해지지 않은 것은 6장에 있다.

---

## 3. C의 다음 작업 (`docs/tasks/C.md` 순서)

| 순서 | 작업 | 마감 | 손댈 파일·함수 | 끝났다는 기준 |
|---|---|---|---|---|
| 1 | M1 정답 박스 11장 | 10/10 13:00 (지남) | `gt/<사진 이름>.csv`를 `tools/label_gt.py`로: own_11_pothole_blur·normal, own_12_crack_normal, own_13_none_normal, own_14_none_blur, own_16_pothole_blur, own_17_crack_blur, own_18_pothole_dark, own_19_none_blur·dark·normal | CSV 11개(손상 없으면 머리글만), `[C] 정답 박스 11장` |
| 2 | M2 WP0 평가 체계 + E0 재현 | 10/10 17:00 (지남) | `src/evaluate.py`(B 파일): `image_verdict`, `summarize_verdicts`. `tools/evaluate_train.py`(B 파일): `DIAG35`, `assign_split`, `--split dev\|test\|all`(기본 dev), `--name`, 영상 단위 요약, `per_source.csv`, 우연 대조군, `--mask`·`--roi-top`·`--detect`·`--set NAME=VALUE` | test_image_verdict·test_summarize_verdicts·test_dev_test_split의 @todo를 지우고 통과. E0(`--split all --name e0_all`, 끔): 정답률 10.4%, 찾은 정답 67·21, 후보 6,588 |
| 3 | M3 run_experiment 39장 자동 ROI | 10/10 저녁 | `run_experiment.py`: `parse_args`(`--roi csv\|auto`, `--mask`, `--detect`, `--roi-top`), `main`, `get_roi` 호출부, `write_config` 순서 | `--roi auto`면 roi.csv를 쓰지 않음, 바꾼 값이 config_used.txt에 남음 |
| 4 | M4 E1·E1b | 10/10 21:00 | 실행. 채택되면 config C 구역 `USE_ROAD_MASK`, 합의되면 공통 구역 `ROI_TOP_DEFAULT`, 둘 다 `docs/part_C.md` | 정답 유지율 95% 이상·후보 35% 이상 감소(93% 미만이면 되돌림). 21:00 표를 단톡방에 |
| 5 | M5 교차 검수 | 10/10 22:00 | 없음(`label_gt`로 A의 12장을 열어 봄) | 고칠 곳을 A에게 알림 |
| 6 | M6 E6 인위 저하 | 도구 10/10 저녁, 1회 10/11 09:00 | `tools/evaluate_train.py`: `degrade(bgr, kind)`, `--degrade`, 상태 판정 적중률. 값은 config C 구역 `DEGRADE_*` | test_degrade 통과. 1회 결과로 G3 정리(단톡방, Plan.md 개정 내역 10/11) |
| 7 | M7 run_experiment 새 열·표 | 10/11 오전 | `run_experiment.py`: `process_image`·`make_row`(has_gt, has_damage, n_pred, hit_same, hit_any, correct, edges_road, t_road_ms), `metrics_frame`, `table_by_group`(민감도·특이도·정답률과 기준선) | row 9열 그대로, 정답 파일이 없으면 빈칸 |
| 8 | M8 시연의 노면 표시 | 10/11 오후 | `src/visualize.py` `draw_road`, config C 구역(섞는 비율·색), `run_experiment.save_figure`, `demo.py` | test_draw_road_paint 통과. demo가 quality·고른 전처리·노면·후보를 끔·켬으로 보여 줌 |
| 9 | M9 E5·E6 최종, 39장 자동 ROI, 시험셋 | 10/11 15:00~17:00 | 실행 | 17:00 `--split test --name test_once` 한 번만. 그 뒤 파라미터를 고치지 않음 |
| 10 | M10 G4, 데모 녹화 | 10/11 21:00 | 없음 | 같은 화면을 녹화 |
| 11 | M11 최종 실행 | 10/12 오전 | 없음 | 4.2의 최종 명령, 처리 시간은 4.5 규칙 |

Should: S1 `ROAD_KEEP_RULE` 50% 규칙(`pipeline.keep_on_road`에 규칙 추가. 지금은 "center" 말고는 ValueError), S2 τ 곡선(`evaluate_train`).

**순서 제안** (13:00·17:00이 지났으므로): 남이 기다리는 것부터 한다. ① M2 중 `DIAG35`·`assign_split`·`--split`·`--detect`·`--mask`·`--set`(A의 산점도와 B의 E2·E3이 기다림) → ② M1 정답 11장(39장 E0와 교차 검수가 기다림) → ③ M2 나머지(영상 단위 요약·국가별 표·우연 대조군, E0 재현) → ④ M3 → 표의 순서. M2의 E0 재현은 학습 사진 804장을 먼저 되살려야 한다(1.3). 순서는 21:00 점검에서 팀과 맞춘다.

**작업별 메모**

- M2 `image_verdict(pred_by_kind, gt)`: 종류별로 `count_center_hits(pred_by_kind[kind], 그 종류 정답)["found"] > 0`이면 그 종류를 맞힌 것이다. hit_same = 정답에 있는 종류 중 하나라도 맞힘, hit_any = 종류와 상관없이 어떤 후보 중심이든 어떤 정답 안, n_pred = 모든 후보 수, correct = 손상 영상이면 hit_same, 아니면 n_pred == 0. 기대값은 `tests/test_c.py`에 있다.
- M2 `summarize_verdicts`: sensitivity = 손상 영상 중 correct 비율, specificity = 무손상 영상 중 correct 비율, accuracy = correct / n, balanced = (민감도 + 특이도) / 2, baseline_none = 무손상 수 / n. 분모가 0이면 NaN.
- M2 `assign_split(names)`: 시험은 `"xxx.jpg"` 이름으로 부르고 `DIAG35`는 확장자 없는 이름이다. 국가는 `Path(name).stem.rsplit("_", 1)[0]`, 국가별 이름순 순번 % 5 == 4이면 test, DIAG35는 dev. 저장소의 `gt/train` 이름으로 다시 세어 시험 152(손상 97·무손상 55)·개발 652, 고정 규칙이 없으면 159·645, 옮겨지는 7장이 Plan.md 부록 A의 굵은 7장과 같음을 확인했다(10/10 17:30).
- M2 B 파일 커밋은 파일마다 따로: `[C] B 파일, B 동의: evaluate에 영상 단위 판정 추가`, `[C] B 파일, B 동의: evaluate_train 개발/시험 나누기·우연 대조군·실행 옵션`. PR 리뷰는 B.
- M4 마스크 정답 유지율 = 띠 안 정답 중 상자 중심이 노면 안인 비율. 정답 중심(640px 좌표)의 y에서 y0를 빼 road 좌표로 바꾸고, `0 <= y - y0 < road.shape[0]`이면 띠 안이다.
- M6 `degrade`: none은 복사본, blur는 `cv.GaussianBlur(bgr, (0, 0), DEGRADE_BLUR_SIGMA)`, dark·bright는 0~1 밝기에 감마(2.2·0.5)를 건 LUT, noise는 함수 안에서 `np.random.default_rng(DEGRADE_SEED)`를 만들어 더한다(그래야 두 번 불러도 같다). 상태 판정 적중률: blur → steps의 sharpen, dark → tone "gamma"이고 gamma < 1, bright → gamma > 1, noise → denoise.
- M8 `draw_road`: `out = bgr.copy()`, `out[y0:y0 + road.shape[0]]`에서 road > 0인 픽셀만 초록과 섞는다. 회색 128에 초록 (0, 255, 0)을 비율 a로 섞으면 b·r ≤ 128, g > 128이라 시험 기대값과 맞는다.

---

## 4. 실행 방법

### 4.1 환경

- 맨 위 폴더에서 파일 이름으로 실행한다(`python tests/test_c.py`).
- Python은 `.venv`(3.13.2)를 쓴다. PATH의 시스템 Python(`C:\Python313`)에는 pandas가 없다. PowerShell이면 `.venv\Scripts\Activate.ps1`, Git Bash면 `.venv/Scripts/python.exe`를 직접 부른다.
- Git Bash에서는 한글 출력이 깨진다(Python이 cp949로 출력). `export PYTHONIOENCODING=utf-8`을 먼저 한다. PowerShell·VS Code 터미널은 그대로 된다.
- 패키지는 `requirements.txt`로만 설치한다(opencv-python 5.0.0.93, numpy 2.5.3, pandas 3.0.6, matplotlib 3.11.2).

### 4.2 자주 쓰는 명령

지금 있는 것:

```bash
python tests/check_contract.py                      # 약속 검사
python tests/test_c.py                              # C 시험 (A는 test_a, B는 test_b)
python run_experiment.py --name <이름> [--only provided|own]   # 39장 끔·켬 → results/<이름>/
python run_matching.py --name <이름>                # 쌍 사진 5쌍 매칭 → results/<이름>/matching.csv
python demo.py [사진 경로 [roi_top roi_bottom]]     # 경로를 빼면 사진 목록에서 번호로 고름
python tools/label_gt.py [사진 경로]                # 정답 박스 (c 균열, p 포트홀, Enter, q 저장)
python tools/evaluate_train.py --name <이름> [--split dev|test|all] [옵션]   # 학습 사진 평가 → results/<이름>/ (기본 개발셋, 5.2)
python tools/convert_json_gt.py                     # data/train/ann JSON → gt/train CSV (이미 커밋됨)
python tools/check_data.py                          # 사진·CSV 이름 검사
```

실험 명령(Plan.md 5장). evaluate_train의 옵션(`--split`·`--name`·`--param-set`·`--mask`·`--roi-top`·`--detect`·`--set`)은 M2에서 만들었고(10/11), run_experiment의 `--mask`·`--roi`·`--detect`는 M3, `--degrade`는 M6에서 만든다:

```bash
python tools/evaluate_train.py --split all --name e0_all                      # E0 재현, 804장 (C)
python tools/evaluate_train.py --split dev --name e0_dev                      # E0 개발셋, 채택 규칙의 비교 기준 (C)
python tools/evaluate_train.py --split dev --mask on --name e1_mask_dev       # E1 (C)
python tools/evaluate_train.py --split dev --mask on --roi-top 0.3 --name e1b_top03_dev   # E1b (C)
python run_experiment.py --name e1_mask --mask on                            # E1 39장 roi.csv (C)
python run_experiment.py --name e1_mask_auto --mask on --roi auto            # E1 39장 자동 ROI (C)
python tools/evaluate_train.py --split dev --detect dark --name e2_dark_dev                  # E2 (B)
python tools/evaluate_train.py --split dev --detect dark --mask on --name e3_dark_mask_dev   # E3 (B)
python tools/evaluate_train.py --split dev --degrade blur --name e6_blur_dev   # E6 (C, dark·bright·noise도 이름을 바꿔)
python tools/evaluate_train.py --split test --name test_once                  # 시험셋, 10/11 17:00 한 번만 (C)
```

최종(10/12 오전, C 컴퓨터):

```bash
python tests/check_contract.py
python run_experiment.py --name final
python tools/evaluate_train.py --split test --name final_test   # test_once와 같은지만 확인, 다르면 원인만 찾음 (run_experiment의 final과 폴더를 나눔)
python run_matching.py --name final
```

확인용 실행은 메모리에서만 하고, `results/`에 쓰는 실행은 사용자가 하라고 할 때만 한다(`docs/tasks/C.md` 8장 7번).

### 4.3 결과 폴더

- `run_experiment`: `results/<이름>/metrics.csv`(영상마다 끔·켬 2행), `figures/<사진>.png`, `table_by_image.csv`·`table_by_source.csv`·`table_by_condition.csv`, `config_used.txt`.
- `run_matching`: 같은 폴더에 `matching.csv`, `matches/<사진>_off.png`·`_on.png`, `config_used_matching.txt`.
- `evaluate_train`: `results/<이름>/`에 `train_detail.csv`(영상별 개수·판정, split·source·height 열), `train_summary.csv`(끔·켬 요약: 종류·기준별 정밀도·재현율, 민감도·특이도·정답률·균형 정확도·기준선, 영상당 후보·오검출, 우연 대조군), `per_source.csv`(같은 요약을 국가별로), `config_used_train.txt`(config 값, 파라미터 세트, 실행 옵션으로 바꾼 값). 파일 이름이 run_experiment·run_matching과 겹치지 않는다.
- 세 도구 모두 `--name`을 빼면 `results/latest`에 쓰고(매번 덮어씀), 1차 결과 폴더 이름(`base`·`tuning`·`diagnosis`)은 대소문자·하위 폴더까지 거부한다(`run_experiment.check_name`).
- 쓸 이름(Plan.md 3.7): `e0_base`, `e1_mask`, `e1_mask_auto`, `e1b_top03`, `e2_dark`, `e3_dark_mask`, `e4_score`, `e6_degrade`, `final`. evaluate_train은 이름 끝에 데이터를 붙여(`e0_all`·`e1_mask_dev` 등) 39장 폴더와 나누고, 시험셋은 `test_once`(10/11)·`final_test`(10/12 재현 확인)로 쓴다(4.2).
- 지금 있는 것(C 컴퓨터에만, Git에 없음):

| 폴더 | 무엇 | 쓰임 |
|---|---|---|
| `results/base` | 10/05 18:18, 39장 끔·켬 + 매칭 | 계획서 3장·Plan.md 1.3·1.4 숫자의 출처 |
| `results/tuning` | 10/06, 804장 × 5세트 × 끔·켬 (`train_detail.csv`, `train_summary.csv`) | Plan.md 1.2·부록 B의 출처. 덮어쓰지 않는다 |
| `results/diagnosis` | 35장 시트, `train_compare/pre_off/`(gt_boxes·per_image·pred_boxes CSV, 국가별 시트) | 띠 밖 정답 비율, 원인 분석 그림 |
| `results/check`, `results/check_vis` | 10/04 점검용 | 버려도 됨 |

- `results/`는 Git에 올리지 않는다. 표·그림은 단톡방·드라이브로 나누고, 손으로 고치지 않는다(다시 실행).

### 4.4 처리 시간 측정 규칙

- 보고서에는 C 컴퓨터 한 대에서 `--name final`로 잰 값만 쓴다(CONTRIBUTING 4장).
- `run_experiment`는 끔·켬을 각각 `TIME_REPEAT`(5)번 돌린 중앙값, `run_matching`의 `t_match_ms`도 5번 중앙값이다.
- `t_pre_ms` = 상태 분석(analyze) + 전처리. `t_detect_ms` = `detect_cracks` + `detect_potholes`이고, 후보 형태 값(두께·거리 변환 폭·비원형도) 계산이 들어 있다. 보고서에 이 점을 적는다.
- `t_road_ms` = `road_mask`만. row 밖의 결과 키다. 노면 밖 후보 지우기(`keep_on_road`)는 어느 시간에도 넣지 않는다(row 시간 열의 뜻을 1차와 같게).
- 해리스 코너 수와 에지 수는 시간을 다 잰 뒤 계산하므로 시간에 들어가지 않는다.
- 최종 실행 때는 노트북 전원을 연결하고 다른 무거운 프로그램을 닫는다(권장).

---

## 5. 함정과 결정의 이유 (Plan.md·project_brief.md에 없는 것)

### 5.1 환경·데이터

- 학습 사진이 비어 있어도 `test_dev_test_split`은 실패하지 않고 '훈련 사진 804장이 없음'으로 **건너뛴다**. @todo를 지운 뒤 '통과'만 보고 끝내지 말고, 건너뜀 목록에 이 시험이 없는지 본다.
- `data/shot_log.csv`는 어느 코드도 읽지 않는다(기록용). 실행에 쓰이는 것은 `data/roi.csv`뿐이다.
- 제공 13장은 `parse_name`의 condition이 빈칸이고 roi.csv condition 칸도 비어 있어, `table_by_condition`에서 'unknown'으로 묶인다. A가 상태 태그를 넣으면 풀린다.

### 5.2 evaluate_train (B 파일, C가 M2에서 고침, M6에서 `--degrade`를 더함)

- (10/11) 결과 폴더는 `--name`으로 정하고(기본 `latest`), 1차 결과 폴더 이름은 거부한다. 10/06 기준 파일(`results/tuning`, Plan.md 1.2·부록 B의 출처)은 덮이지 않는다.
- (10/11) 기본 실행은 config 값 한 세트(params 열 `config`)라 사진 한 장을 끔·켬 2번 돈다. 1차 튜닝 5세트는 `--param-set base|…|all`로만 돈다(all이면 10번). `base`는 config B 구역의 legacy 검출 값 13개를 숫자로 고정해 둔 것이라 B가 config를 바꾼 뒤에도 1차 값을 다시 낼 수 있다. 다만 `DETECT_MODE`·`USE_ROAD_MASK`·`ROI_TOP_DEFAULT`는 들어 있지 않으므로, 그때 E0를 다시 내려면 `--detect legacy --mask off`도 함께 준다.
- 실행 옵션(`--mask`·`--roi-top`·`--detect`·`--set`)은 실행 중에만 config 값을 바꾸고 `finally`로 되돌린다. `--set`은 config에 있는 대문자 이름만 받고, `--param-set`과 같은 값을 바꾸면 거부한다. 개발/시험 나누기는 고른 분할이 아니라 사진 전체 이름으로 정한다(고른 분할만으로 정하면 국가별 순번이 달라짐).
- (10/11 검증, 임시 폴더) 학습 사진·JSON 804개씩, 확장자 소문자 `.jpg`, 사진·JSON·`gt/train` 이름이 하나씩 맞음. config 실행과 `--param-set base` 실행의 결과 파일 3개가 params 열 말고 모두 같았다. `--split all` 실행이 10/06 `results/tuning` base 행과 영상별 개수 20열 × 1,608행(804장 × 끔·켬)에서 모두 같았고, Plan.md 1.2 표(35장·804장, 끔·켬, 띠 밖 정답 비율 포함)도 모두 같았다. 실행 시간(C 컴퓨터, legacy·마스크 끔, 한 세트 × 끔·켬): `--split all` 56초(사진을 처음 읽을 때 81초), `--split dev` 46초. 사진·끔켬 1회에 약 35ms다.
- 우연 대조군: 640px 높이가 같은 사진끼리 이름순으로 묶어 i번 사진의 후보를 i+1번 사진의 같은 종류 정답과 중심 기준으로 비교한다(마지막은 첫 사진과). 1장뿐인 높이 묶음은 자기 자신과 비교하게 되므로 빼고 `chance_skipped` 열에 그 수를 적는다. 학습 데이터의 묶음은 아래 넷이고 1장뿐인 묶음이 없어 **빠진 사진은 0장**이다(전체·개발 모두). Norway는 원본 크기가 셋이다(10/10 메모의 '둘'을 고침). 358과 359는 1px 차이지만 규칙대로 다른 묶음이다. 시험셋의 묶음은 시험셋 실행의 `chance_skipped`로 나온다.

| 640px 높이 | 전체 804장 | 개발 652장 | 원본 크기 |
|---|---|---|---|
| 640 | 629 (China_Drone 43, China_MotorBike 46, Czech 42, India 167, Japan 236, United_States 95) | 510 | 정사각형 512·540·600·640·720·1024·1080 |
| 322 | 94 (Norway) | 75 | 4040×2035 |
| 359 | 62 (Norway) | 54 | 3643×2041 |
| 358 | 19 (Norway) | 13 | 3650×2044 |

- (10/11 검증 실행의 값, 1차 방법·804장. 보고서 숫자는 마지막 실행에서 다시 낸다) 중심 기준 재현율:

| 종류 | 전처리 | 실제 | 대조군 | 실제 − 대조군 |
|---|---|---|---|---|
| 균열 | 끔 | 55/1,113 = 4.94% | 37/1,113 = 3.32% | +1.62%p |
| 포트홀 | 끔 | 12/133 = 9.02% | 14/133 = 10.53% | −1.50%p |
| 균열 | 켬 | 58/1,113 = 5.21% | 40/1,113 = 3.59% | +1.62%p |
| 포트홀 | 켬 | 16/133 = 12.03% | 16/133 = 12.03% | 0.00%p |

- 포트홀은 끔·켬 모두 대조군과 같은 수준이라 Plan.md 1.2의 '우연 수준'과 맞는다. 균열은 대조군보다 1.6%p 높다(다른 사진의 정답과 비교해도 균열 37개가 '찾은' 것으로 세어짐).

### 5.3 run_experiment·pipeline (C 파일)

- `metrics.csv`의 정답 열(`crack_tp` 등)은 IoU 0.3 일대일(`count_matches`)이다. 계획서의 주 기준(중심)이 아니다. 39장의 영상 단위 판정은 M7에서 `image_verdict`(중심)로 새 열을 만든다. `evaluate_train`은 이미 iou·ctr 두 기준을 낸다.
- `write_config`가 `main` 맨 처음에 불린다. 실행 중 바꾸는 값(`--mask` 등)을 먼저 적용한 뒤 써야 config_used.txt가 맞다. 끝나면 `finally`로 되돌린다.
- 실행 중 config 바꾸기는 `C.이름`을 그때그때 읽는 곳에만 먹힌다. `io_utils.crop_roi(top=C.ROI_TOP_DEFAULT)`·`resize_width(width=C.TARGET_WIDTH)`·`run_matching.timed_match(repeat=C.TIME_REPEAT)`의 기본값은 import 때 굳는다. 지금 경로는 `get_roi`가 그때 값을 읽어 `run_pipeline`에 top을 넘기므로 괜찮다.
- `--roi-top`은 roi.csv에 없는 사진에만 먹힌다. 39장의 E1b는 `--roi auto`와 같이 쓴다. 개발셋은 모두 roi.csv에 없어 그대로 먹힌다.
- `metrics_frame`은 row dict의 모르는 키를 `quality_columns()`로 상태 지표 뒤에 붙이고, 그 열은 `table_by_image`의 머리 열에도 들어간다. M7의 새 열은 열 목록을 따로 두고 위치를 정한다. 정답이 없을 때 빈칸이 되도록 형(Int64·boolean)도 정한다.
- `median_times`는 row의 두 시간 열만 중앙값으로 바꾼다. `t_road_ms`는 결과의 맨 위 키라 따로 처리해야 한다.
- `save_comparison`은 패널이 정확히 6개여야 한다. 노면 표시는 7번째 칸이 아니라 '후보' 칸에 겹쳐 그린다(`draw_candidates(draw_road(...), ...)`).
- `draw_road`는 `road.shape[0]`만큼만 칠한다. roi_bottom이 1보다 작으면(미국 3장 0.95) 띠 높이가 '사진 높이 − y0'보다 작다.
- `USE_ROAD_MASK`가 꺼져 있으면 pipeline은 road를 검출 함수에 넘기지 않는다. E2('마스크 없음')에서 dark의 σ가 노면 안에서 재지지 않게 하고, 꺼진 상태가 어느 `DETECT_MODE`에서도 1차와 같게 하려는 결정이다.
- `keep_on_road`는 "center"만 받고 다른 규칙은 ValueError다. S1에서 규칙을 더한다.

### 5.4 매칭·그림·시연

- `RANSAC_MIN_GOOD = 4`(팀 기준, 호모그래피의 수학적 최소)다. good이 4~9개면 inlier 비율이 1.0 가까이 부풀려지므로, 매칭 표에서는 good 열을 같이 본다(`docs/part_C.md`).
- `run_matching`은 ROI 없이 640px 전체 흑백을 쓰고, 켬이면 기준 A를 포함한 모든 사진이 자기 상태에 맞는 전처리를 거친다.
- 그림의 한글 글꼴은 `sys.platform`으로 고른다(윈도 맑은 고딕, 맥 AppleGothic, 그 밖은 □로 나옴). 최종 실행 컴퓨터가 윈도·맥이 아니면 그림 제목이 깨진다.
- 데모 창 제목은 ASCII(`demo`)다. 윈도 HighGUI에서 한글 창 제목이 깨지기 때문이고, 패널의 한글은 OpenCV 5 내장 글꼴 `FontFace("sans")`로 그린다.
- `src/`에서 파일을 저장하는 함수는 `visualize.save_comparison` 하나다(Agg 백엔드, 저장 뒤 `plt.close`).

### 5.5 흩어진 판단의 이유

- `BLUR_RATIO_MAX = 1.5`는 임시값이다. 기반 때 같은 식으로 재 보니 제공 13장 중 흐린 3장이 1.13~1.18, 나머지가 1.87 이상이었다. 같은 장소 5쌍 중 own_19는 3.398(흔들림)과 3.395(정상)로 거의 같아 시험에서 빼고, C가 E5 상태 판정 검증에서 보고하기로 했다(`docs/tasks/A.md` M3).
- `CLAUDE.md`의 배경 문서 목록은 경로만 적고 `@` 가져오기를 쓰지 않았다. `@docs/...`로 쓰면 모든 세션이 그 문서를 자동으로 읽어(Plan.md만 134KB) 문맥을 크게 차지한다. 필요할 때 열어 읽게 했다.

---

## 6. 아직 정해지지 않은 것과 누가 정하나

| 무엇 | 누가 정함 | 언제·근거 | 정해지면 |
|---|---|---|---|
| 노면 마스크 켬(`USE_ROAD_MASK`) | 전원(E1 결과) | 10/10 21:00, Plan.md 3.3 (8) | C가 config C 구역, `docs/part_C.md` |
| 띠 값(`ROI_TOP_DEFAULT` 0.5 → 0.3?) | 세 명 합의 | E1b 뒤 | C가 공통 구역, `docs/part_C.md` |
| `ROAD_KEEP_RULE` center vs 50% | C (Should) | 10/11 18:00 | C |
| dark 채택(`DETECT_MODE`) | 전원(G3, 채택 규칙) | 10/11 10:00, Plan.md 3.10 | B가 config B 구역 |
| `DARK_SIGMA_IN_ROAD` | B | E3 | B |
| 균열·포트홀 문턱(`CRACK_MIN_CIRC_INV`·`POT_MAX_CIRC_INV`) | B | 10/11 12:00, A의 산점도 | B, `docs/part_B.md` |
| 점수 τ | B (Should) | E4 | B |
| 검출 파라미터 확정 | B | 10/11 15:00 | 이후 고치지 않음 |
| 전처리 규칙(감마 상한, CLAHE, 야간 처리, `BLUR_RATIO_MAX` 확정) | A(G3에서 전원 확인) | 10/11 10:00, E6 1회 | A 구역 동결 |
| 39장 정답과 무손상 수 | 표시한 사람, 엇갈리면 B | 교차 검수 10/10 22:00 → G3 | 39장 기준선 확정 |
| 제공 13장 상태 태그 | A(동의 여부) | 10/10 13:00 | A가 roi.csv |
| `image_verdict`·`summarize_verdicts`를 약속 표에 | B·C | M2 PR 때 | C가 CONTRIBUTING 2장·check_contract |
| evaluate_train 기본 세트(한 세트 vs 5세트) | B·C | M2 PR 리뷰 | (10/11) 사용자 결정으로 C가 config 한 세트로 구현, 1차 5세트는 `--param-set`. B 확인 대기 |
| 우연 대조군에서 사진 1장뿐인 크기 묶음 | C | (10/11 정함) | 빼고 `chance_skipped`에 수를 적음. 학습 804장은 1장뿐인 묶음이 없어 0장(5.2), 보고서에 적음 |
| `--roi auto`가 노면 마스크를 저절로 켤지 | C | M3 | Plan.md 명령은 둘을 함께 줌 |
| 최종 방법을 고를 때 처리 시간을 어떻게 볼지 | 전원 | G4 전, 확정본 6장 | |
| 계획서 3장 라플라시안 표기(3~5 vs 3.0~3.9) | 전원(확정본은 합의 뒤 수정) | 제출 전 | C가 `docs/plan_2nd.md` |
| 계획서 1장 조 정보, 제출 | 전원 | 10/12 | 양식에 직접 |
| 보고서·발표 분담 | 전원 | 기록 없음 | |
| 데모 영상에 쓸 사진과 녹화 방법 | C | G4(10/11 21:00) | |
| 드라이브 사진 이름 맞추기 | A(데이터 주인)·C | 지금(1.3) | |

---

## 7. 어떤 문서가 무엇의 기준인가

| 문서 | 기준이 되는 것 | 고치는 사람 |
|---|---|---|
| `docs/plan_2nd.md` | 계획 내용: 문제 재정의, 개선 7항목, 목표, 실험 순서 E0~E6, 채택 규칙, 역할, 일정·관문(8장). 지시서의 '5장 (1)', '4.2 학습과제 6번' 같은 번호 | 세 명 합의 뒤 C |
| `Plan.md` | 근거와 숫자: 기준 수치(1장, 재현 방법은 부록 B), 정오표, 알고리즘 명세의 세부값(3.2~3.6), 실험 설계 세부(3.10), 위험(3.11). 4장은 초안이라 참고용 | C |
| `docs/tasks/` | 누가 언제 무엇을: Must·Should·Could, 마감, 넘겨주기, 통과 기준(시험), AI 시작 프롬프트 | C |
| `CONTRIBUTING.md` | 규칙: 파일 주인(1장), 함수 약속(2장), 데이터 규칙(3장), 구현 규칙(4장), Git(5장), 약속 변경 절차(7장) | C(약속 변경은 세 명 합의) |
| `tests/check_contract.py` | 함수 약속의 자동 검사. 코드 형식의 실제 기준 | C |
| `CLAUDE.md` | AI 도구 규칙 | C |
| `docs/part_X.md` / `docs/ai_log_X.md` | 파라미터 변경 기록 / AI 활용 기록(그룹 평가·AI 활용 평가의 근거) | 각 파트 |
| `docs/project_brief.md` | 처음 보는 사람을 위한 요약. 원본이 아니다 | C |
| `docs/handoff_C.md` (이 문서) | C의 상태·다음 작업·함정. 원본이 아니고, 상태는 기준 시각의 것이다 | C |
| `results/` | 숫자의 실제 출처. 손으로 고치지 않고 다시 실행한다 | 실행한 사람 |

**서로 다르면**

- 규칙과 함수 약속은 `CONTRIBUTING.md`(그리고 그것을 검사하는 `check_contract.py`)를 따른다.
- 계획 내용과 일정은 `docs/plan_2nd.md` → `docs/tasks/` → `Plan.md` → `project_brief.md`·이 문서 순으로 따른다.
- 숫자는 `results/` 파일과 Plan.md 부록 B의 방법으로 다시 센 값을 따른다.
- 코드가 실제로 어떻게 도는지는 코드와 시험이 기준이다. 문서와 코드가 다르면 문서를 고치거나 알린다. 다른 파트의 파일이면 고치지 않고 주인에게 알린다.
