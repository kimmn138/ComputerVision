"""(파트 C) E0~E6 실험을 단계마다 명령 하나로 돌리고, 보고서 표(CSV)와 요약(summary.md)을 만든다(Plan.md 3.7·3.10).
사용법: 맨 위 폴더에서  python tools/run_all.py compare|band|mask2|final|test|repro|retest|retest_repro|retest_ratio|retest_ratio_repro|tables
                                              [--dry-run] [--yes]
                                              [--accept E1b] [--reject E1,E3] [--force dark,mask]
순서: compare(비교) → 사용자가 채택을 정하고 구역 주인이 config를 고침 → final(최종) → test(시험, 한 번만) → (10/12) repro(재현)
- compare: 개발셋에서 E0(804장 재현 포함)·E1·E1b·E2·E3·E3σ를 차례로 돌리며, 끝날 때마다 채택 규칙 2·3으로 잠정 판정해 뒤 단계 설정에 반영한다.
  비교는 한 번에 하나만 바뀌게: E1은 E0와, E1b는 E1과, E2는 E0와, E3는 마스크를 켠 1차 설정(E1, E1b가 채택되면 E1b)과, E3σ는 E3와.
  E1b가 채택되면 마스크를 켜는 E3·E3σ·39장 E3는 띠 0.3으로 돈다(마스크를 끄는 E2는 0.5 그대로). E3는 E2와의 차이(마스크만 다름)도 참고로 낸다.
  E6 1회(저하 다섯 가지)는 E1~E3 가운데 마지막으로 잠정 채택된 단계의 설정으로(없으면 E0), 39장 E1·E3 자동 ROI는 채택 판단에 쓰지 않는 참고로 돈다.
  E0·E1·E1b는 1차 값(--param-set base, legacy, 띠 0.5)으로 고정해 config가 바뀐 뒤에 다시 돌려도 같은 조건이다.
  채택 표(규칙 1은 민감도와 영상당 오검출을 나란히 둔 사람 확인, 처리 시간은 단일 실행 참고값)와 바꿀 config 값 목록(구역 주인과 함께)을 낸다.
  채택 표 위에는 검출기(1차·dark) × 노면 마스크(끔·켬) 네 칸(E0·E1·E2·E3, E1b가 채택되면 마스크를 켠 칸은 띠 0.3)과 E3σ 한 줄을 나란히 둔다.
  자동 판정은 길잡이이고 최종 채택은 사람이 정한다: --accept(채택)·--reject(되돌림)는 자동 판정보다 앞서고 results/report/에 남아 유지되며,
  그 때문에 뒤 단계 설정이 바뀌면 표가 '다시 돌려야 함'으로 알린다. config 파일은 고치지 않는다.
- final: config 그대로(채택 결과를 반영한 최종 설정)로 39장(최종 roi.csv·최종 자동 ROI·1차 설정), 매칭, E5(개발셋 끔·켬), E6 2회.
  처리 시간은 이 단계 값만 보고서에 쓴다. 시작할 때 전원 연결·다른 프로그램 닫기를 알리고 Enter를 기다린다(--yes면 건너뜀).
- test: 시험셋을 한 번만 잰다. results/test_once가 있으면 거부한다. 같은 시험셋에서 1차 방법(e0_test)을 먼저 돌려 전·후를 견준다.
  시작할 때 지금 config를 보여 주고 Enter를 기다린다(--yes면 건너뜀). 1차 방법이 실패하면 test_once를 돌리지 않는다.
- repro: (10/12) 같은 코드로 시험셋을 다시 돌려(final_test) test_once와 같은지 본다. 다르면 원인만 찾고 다시 튜닝하지 않는다.
- band: (10/12) 자동 띠 비교의 참고 실행. 결 조건 없는 자동 띠(B1 개발셋, B1_39 39장 자동 ROI). E3(띠 고정)·E5·F39a(자동 띠 + 결 = 최종)와
  나란히 9번 표를 만든다.
- retest: (10/12, 자동 띠 채택 뒤) 시험셋을 최종 설정(자동 띠 + 결)으로 한 번만 잰다(test_autoband). 이미 있거나, 지금 config가
  최종 단계(e5_dev)를 돌린 config와 다르면 거부한다. test_once(띠 고정)는 그대로 둔다.
- retest_repro: 같은 코드로 시험셋을 다시 돌려(final_test_autoband) test_autoband와 같은지 본다.
- mask2: (10/12) 노면 마스크 보완 비교의 참고 실행. 자동 띠 + 결(남기기 center, 이전 최종) → + 남기기 ratio → + 차선 (가)·(나)
  (개발셋 M0~M3, 39장 자동 ROI M0_39~M3_39). ratio는 '규칙 2 미달, 예외 채택', 차선은 '채택 안 함'(사람이 정함, MASK2_DECISIONS).
- final은 ratio 채택(10/12) 뒤 새 폴더(final_ratio·final_auto_ratio·e0_final_ratio·e5_ratio_dev·e6_*_final_ratio_dev)에 쓴다.
  예전 최종 폴더(final·final_auto·e0·e5_dev·e6_*_final_dev, 남기기 center)는 그대로 둔다.
- retest_ratio: (10/12, ratio 채택 뒤) 시험셋을 최종 설정(자동 띠 + 결 + ratio)으로 한 번만 잰다(test_ratio). 거부 조건은 retest와 같다.
- retest_ratio_repro: 같은 코드로 시험셋을 다시 돌려(final_test_ratio) test_ratio와 같은지 본다.
- (10/12 밤) 위 시험·재현 단계 여섯 개(test·repro·retest·retest_repro·retest_ratio·retest_ratio_repro)는 10/12에 이미 잰 단계라
  결과 폴더와 상관없이 거부한다(DONE_TEST_STAGES). results/가 지워져 폴더 검사만으로는 다시 도는 것을 막을 수 없었다.
남기기 규칙(config ROAD_KEEP_RULE): ratio 채택 전에 돈 실험(비교·자동 띠·시험 단계)은 실행 중에만 'center'로 고정해, 다시 돌려도 처음과 같은 조건이다.
- supp: (10/12 밤, 보완) 개발셋에서 보완 항목을 하나씩 더하며 --tau-curve로 돌리고, 채택 규칙을 τ 곡선의 동작점으로 판정한다
  (docs/tasks/supplement.md 4.1). 규칙 1 = fp1·fp2 민감도가 둘 다 안 떨어지고 하나 이상 오름, 규칙 2·3 = 각 실행 자신의 rule1 점
  (10/13부터 fp2 점: 영상당 오검출 2개 이하 중 민감도 최대)의 균형 정확도·우연 뺀 균열 재현율. 표 14(table14_supp)·15(table15_tau_points).
  --only S12,E5s처럼 단계 안의 실험만 골라 돌릴 수 있다(앞 행을 다시 돌리지 않으려고).
- tables: 실험은 돌리지 않고 있는 결과 폴더로 표만 다시 만든다.
자동 띠 높이(config AUTO_BAND_TOP): 띠 위쪽 비율을 옵션으로 정한 실험(E0~E3, 39장 1차 설정, 시험셋 1차)과 띠 고정 시절의 시험셋 실행
(test_once·final_test)은 실행 중에만 AUTO_BAND_TOP=False로 고정해, 자동 띠를 켠 뒤에 다시 돌려도 처음과 같은 조건이다.
--dry-run은 돌릴 실험·명령·결과 폴더·예상 시간만 보여 주고 아무것도 쓰지 않는다.
실행 전 확인: 39장 정답 파일(빠지면 이름을 알리고 39장 표에 '미완'), 학습 사진 804장, 약속 검사, 커밋 안 된 변경. 뒤의 셋은 어기면 멈춘다.
구현 여부: 노면 마스크·dark 검출은 A·B 시험의 해당 @todo가 지워졌는지(각 파트의 통과 기준)로 정하고, 합성 영상 검사는 어긋날 때 경고만 한다.
구현 전 기능이 필요한 실험은 건너뛰고 표에 '아직 구현 전'으로 적되, 시작 전에 목록을 크게 보여 주고 Enter를 기다린다(--yes면 그대로 진행).
--force dark,mask면 @todo가 남은 기능도 쓰고, 그 기능을 쓴 실험은 표에 '강제 실행(시험 미통과)'으로 남긴다(구현했지만 시험 하나를 아직 못 넘긴 경우).
결과: 실험마다 results/<폴더>/ (도구의 결과·config_used*.txt와, run_all이 남기는 run_info*.txt: 커밋 해시·명령·옵션·고정한 값)
      표는 results/report/: table1_stages … table8_39_images.csv, table9_auto_band*.csv(자동 띠 비교), table10_*(노면 마스크 보완·마스크 안 정답), table11_tests(시험셋 세 번), table12_time(1차 vs 최종 시간), table13_judge(상태 판정), adoption.csv·adoption_grid.csv, summary.md(보고서용 요약),
      accept.txt·reject.txt(사람이 정한 채택·되돌림)."""
import argparse
import ast
import collections
import contextlib
import datetime
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiment as rx  # noqa: E402
import run_matching as rm  # noqa: E402
from src import analyze, detect, evaluate, io_utils, pipeline  # noqa: E402
from src import config as C  # noqa: E402
from tools import evaluate_train as et  # noqa: E402

STAGES = {"compare": "비교", "band": "자동 띠 비교", "mask2": "노면 마스크 보완 비교", "supp": "보완 비교", "final": "최종", "test": "시험",
          "repro": "재현",
          "retest": "시험(자동 띠)", "retest_repro": "재현(자동 띠)", "retest_ratio": "시험(ratio)", "retest_ratio_repro": "재현(ratio)",
          "tables": "표만"}
TEST_STAGES = ("test", "retest", "retest_ratio")    # 시험셋을 한 번만 재는 단계: 실패하면 다음 실험을 돌리지 않는다
SET_TEST_STAGES = ("test", "repro", "retest", "retest_repro", "retest_ratio", "retest_ratio_repro")   # 시험셋을 쓰는 단계
RETESTS = {"retest": "TA", "retest_ratio": "TR"}    # 다시 재는 시험 단계 → 한 번만 재는 실험(최종 단계 config와 같아야 돈다)
RETEST_REPROS = {"retest_repro": "retest", "retest_ratio_repro": "retest_ratio"}    # 재현 단계 → 그 앞 시험 단계
RATIO_STAGES = ("final", "retest_ratio", "retest_ratio_repro", "mask2", "supp")   # 남기기 규칙을 config(또는 opts sets)대로 쓰는 단계. 나머지는 'center' 고정
# 10/12에 이미 잰 시험셋 단계(재현 포함). results/가 지워져(10/12 밤) 폴더 유무로는 막을 수 없으므로 단계 이름으로 거부한다.
# 다음 시험셋은 보완 뒤 retest_supp 한 번뿐이다(docs/tasks/supplement.md 0-1)
DONE_TEST_STAGES = {"test": "test_once, 10/12 00:49", "repro": "final_test",
                    "retest": "test_autoband, 10/12 03:00", "retest_repro": "final_test_autoband",
                    "retest_ratio": "test_ratio, 10/12 06:38", "retest_ratio_repro": "final_test_ratio"}
KEEP_BEFORE_RATIO = "center"        # ratio 채택(10/12) 전의 남기기 규칙
SCORE_STAGES = ("supp", "final")   # 점수 문턱(E4, 10/13 채택)을 config대로 쓰는 단계(final은 늘 config 그대로). 나머지(E4 전에 돈 단계)는 실행 중에만 점수를 끔
SCORE_BEFORE_E4 = {"SCORE_FORMULA": "none", "USE_SCORE_TAU": False}
FINAL_CONFIG_EXP = "E5"                 # retest 전에 지금 config를 대조할 최종 단계 실험(config 그대로 돈 개발셋)
REPORT_DIR = "report"               # 표는 results/report/
DECISION_FILES = {"accept": "accept.txt", "reject": "reject.txt"}   # 사람이 정한 채택·되돌림(다음 단계에서 표를 다시 만들어도 유지되게)
FORCED_MARK = "강제 실행(시험 미통과)"   # --force로 A·B 시험의 @todo가 남은 기능을 쓴 실험의 표시
N_TRAIN = 804                       # 학습 사진 수. 일부만 있으면 개발/시험 나누기가 계획(652·152)과 달라진다
PLANNED_SPLIT = {"all": 804, "dev": 652, "test": 152}   # 학습 사진이 없을 때 예상 시간에 쓰는 장수 (Plan.md 3.2 (3))

# 1차 방법(E0)의 고정값. config가 채택 결과로 바뀐 뒤에도 E0·E1·E1b를 같은 조건으로 다시 낼 수 있게 숫자로 둔다
FIRST_ROI_TOP = 0.5                 # 1차 기본 띠의 위쪽 비율 (Plan.md 1.2)
E1B_ROI_TOP = 0.3                   # E1b 띠의 위쪽 비율 (Plan.md 3.10). E1b가 채택되면 마스크를 켜는 E3 쪽도 이 값
FIRST = {"params": "base", "detect": "legacy", "mask": "off", "roi_top": FIRST_ROI_TOP}
DARK = {"detect": "dark", "mask": "off", "roi_top": FIRST_ROI_TOP}         # E2: 마스크 없이 dark 검출만

RULE3_DROP = 0.01                   # 채택 규칙 3: 우연을 뺀 균열 재현율이 1%p 넘게 떨어지면 되돌림 (Plan.md 3.10)
# E0 재현 통과 기준 (Plan.md 3.2 (5), 804장·전처리 끔)
E0_EXPECT = {"정답률 %": 10.4, "찾은 정답(중심)": 67, "찾은 정답(IoU)": 21, "후보 수": 6588}

# dry-run 예상 시간. C 컴퓨터(10/11)에서 legacy·마스크 끔으로 잰 값이라 dark·노면 마스크는 더 걸릴 수 있다
SEC_TRAIN = 0.036                   # evaluate_train 사진 1장의 끔 또는 켬 한 번 (개발 652장 끔·켬 46초)
SEC_PHOTO39 = 1.15                  # run_experiment 사진 1장 (끔·켬 × TIME_REPEAT 5번 + 비교 그림, 39장 45초)
SEC_PAIR_PHOTO = 1.5                # run_matching 쌍 사진 1장 (10장 15초)
SEC_CHECK = 2                       # 약속 검사

DEGRADE_NAMES = {"none": "저하 없음", "blur": "흐림", "dark": "어두움", "bright": "과노출", "noise": "잡음"}
NEEDS = {"dark": "dark 검출(B)", "mask": "노면 마스크(A)"}
# 구현 여부의 신호: 각 파트가 기능을 구현하면 지우는 시험의 @todo (docs/tasks/A.md·B.md 4장의 통과 기준)
SIGNALS = {"mask": ("tests/test_a.py", "test_road_mask_v1_synthetic"),
           "dark": ("tests/test_b.py", "test_dark_big_pothole_one_piece")}
PROBE_FAIL = {"mask": "왼쪽이 풀색인 합성 띠에서 노면 아님(0)을 하나도 내지 않는다. 그대로 돌리고, E1이 E0와 같으면 '효과 없음' 결과로 남는다",
              "dark": "합성 영상에서 NotImplementedError가 난다. 그대로 돌리면 실패로 남는다"}
PROBE_PASS = {"mask": "합성 띠에서는 노면 아님(0)이 나온다. A가 구현하고 @todo를 지우지 않았는지 확인",
              "dark": "합성 영상에서는 dark 검출이 돈다. B가 구현하고 @todo를 지우지 않았는지 확인"}
TOOL_SCRIPT = {"train": "tools/evaluate_train.py", "run39": "run_experiment.py", "match": "run_matching.py"}
RESULT_FILE = {"train": et.SUMMARY_NAME, "run39": "metrics.csv", "match": "matching.csv"}       # 있으면 그 실험이 끝난 것
RUN_INFO = {"train": "run_info_train.txt", "run39": "run_info.txt", "match": "run_info_matching.txt"}
CONFIG_FILE = {"train": et.CONFIG_NAME, "run39": "config_used.txt", "match": "config_used_matching.txt"}
DATA_NAMES = {"all": "학습 전체", "dev": "개발", "test": "시험"}


def _exp(id_, stage, tool, name, label, opts=None, **extra):
    """실험 하나의 정의. opts는 도구 옵션으로 줄 값(params·detect·mask·roi_top·sigma_in_road·degrade)이고, 없는 값은 config 그대로 돈다.
    adopt는 채택 규칙으로 견주는 단계(목록 순서가 판정 순서), reference는 채택 판단에 쓰지 않는 참고 실행이다.
    follow_band는 E1b가 채택되면 띠 0.3, follow_last는 E1~E3 가운데 마지막 잠정 채택 단계의 설정을 따른다(resolve)."""
    return {"id": id_, "stage": stage, "tool": tool, "name": name, "label": label, "opts": dict(opts or {}), **extra}


# 노면 마스크 보완 비교(mask2)의 실행: (개발셋 폴더, 39장 폴더, 이름, 실행 중에만 바꿀 config 값). 10/12에 run_all 밖에서 같은 옵션으로 먼저 돌렸다
MASK2_RUNS = (("lane_base_dev", "lane39_base", "자동 띠 + 결, 남기기 center(이전 최종)", {"ROAD_KEEP_RULE": "center"}),
              ("keep_ratio_dev", "lane39_ratio", "+ 남기기 ratio", {"ROAD_KEEP_RULE": "ratio"}),
              ("lane_ga_dev", "lane39_ga", "+ 차선 (가)", {"ROAD_KEEP_RULE": "center", "ROAD_LANE_BRIDGE": True}),
              ("lane_na_dev", "lane39_na", "+ 차선 (나)", {"ROAD_KEEP_RULE": "center", "ROAD_CLOSE_BEFORE_SEED": True}))
# 사람이 정한 판정(10/12, 사용자). 자동 판정(규칙 2·3)과 나란히 적는다
MASK2_DECISIONS = {"M1": "규칙 2 미달, 예외 채택: 판정이 바뀐 사진이 652장 중 3장(우연 범위), 전처리 켬 +0.2%p, "
                         "영역 대부분이 노면 밖인 후보(Japan_000107: 노면 안 8%)를 노면 안으로 세던 한계를 고치는 수정",
                   "M2": "채택 안 함(스위치 끔 유지, 시연 기본값도 끔)", "M3": "채택 안 함(스위치 끔 유지, 시연 기본값도 끔)"}
MASK_GT_IDS = ("E3", "M0", "M2", "M3")
# 보완 비교(supp, 10/12 밤~, docs/tasks/supplement.md 4.1): (id, 개발셋 폴더, 이름, 실행 중에만 바꿀 config 값, 비교 기준 id, 판정 행, 판정 방식)
# 판정 방식: "base" = 비교 기준 없음, "curve" = 곡선만 비교(채택 판정 없음), "tau" = τ 없음 점 대비 rule1 점(E4), "rules" = 규칙 1~3
SUPP_RUNS = (("S0", "supp_base_dev", "보완 기준: v9 최종 설정(점수 없음)", {"SCORE_FORMULA": "none", "USE_SCORE_TAU": False},
              None, "off", "base"),
             ("S1a", "score_ga_dev", "점수식 margin(대비 × 분류 문턱 여유)", {"SCORE_FORMULA": "margin", "USE_SCORE_TAU": False},
              "S0", "off", "curve"),
             ("S1b", "score_na_dev", "점수식 snr(대비 × √넓이)", {"SCORE_FORMULA": "snr", "USE_SCORE_TAU": False}, "S0", "off", "curve"),
             ("S1c", "score_da_dev", "점수식 shape(대비 상한 × 길이·넓이)", {"SCORE_FORMULA": "shape", "USE_SCORE_TAU": False},
              "S0", "off", "curve"),
             ("E4s", "e4s_score_dev", "E4 점수 문턱: margin, fp2 점 τ(config)", {}, "S0", "off", "tau"),
             ("E5s", "e5s_score_dev", "E5 E4 설정의 전처리 끔·켬(참고)", {}, "E4s", "on", "curve"),
             ("S12", "s12_localsigma_dev", "2-1(12번) 국소 σ", {"DARK_SIGMA_LOCAL": True}, "E4s", "off", "rules"),
             ("S13", "s13_halo_dev", "2-2(13번) 밝은 줄 옆 테두리 메움(팽창 6px)", {"DARK_BRIGHT_DILATE": 6}, "S12", "off", "rules"),
             ("S9a", "s09a_stripe_dev", "2-2(9a) 밝은 표시 기준을 노면 중앙값으로", {"DARK_BRIGHT_REF": "road_median"}, "S13", "off", "rules"))
SUPP39_RUNS = (("E5s_39", "e5s_score39", "E5 E4 설정 39장 자동 ROI(상태 판정 검증, 참고)", {}),
               ("S12_39", "s12_localsigma39", "2-1(12번) 국소 σ 39장 자동 ROI(참고)", {"DARK_SIGMA_LOCAL": True}),
               ("S13_39", "s13_halo39", "2-2(13번) 테두리 메움 39장 자동 ROI(참고)", {"DARK_BRIGHT_DILATE": 6}),
               ("S9a_39", "s09a_stripe39", "2-2(9a) 노면 중앙값 기준 39장 자동 ROI(참고)", {"DARK_BRIGHT_REF": "road_median"}))   # 39장 참고 실행(채택 판단에 쓰지 않음)
SUPP_IDS = tuple(r[0] for r in SUPP_RUNS)
SUPP_DECISIONS = {"S1a": "점수식 margin 선택(사용자, 10/13): 같은 동작점(fp1·fp2) 민감도가 셋 중 가장 높음",
                  "S1b": "곡선 비교(고르지 않음)", "S1c": "곡선 비교(고르지 않음)",
                  "E4s": "예외 채택(사용자, 10/13): 규칙 3은 후보를 만드는 방법을 바꿀 때 균열 맞힘을 잃지 않게 하려는 규칙인데, "
                         "E4는 같은 후보에서 동작점을 고르는 일이고 Plan.md 3.5 (3)이 E4의 목적을 '오검출 2개 이하 τ'로 정함. "
                         "손실(우연 뺀 균열 재현율 감소)은 보고서에 그대로 적음",
                  "E5s": "참고(전처리 끔·켬, 판정 행 켬)",
                  "S12": "채택(규칙 1~3 통과, 밝은 쪽 잔차 판 안 A, 10/13. 양쪽 잔차 판은 큰 포트홀 시험 실패로 되돌림)",
                  "S13": "채택(규칙 1~3 통과. 끔 행 차이는 작고 켬 행에서 뚜렷함)"}   # 사람이 정한 판정. 자동 판정과 나란히 적는다
SWITCHES_BEFORE = {"AUTO_BAND_TOP": False, "ROAD_LANE_BRIDGE": False, "ROAD_CLOSE_BEFORE_SEED": False}   # 생기기 전 실행의 값  # 마스크 안 정답 비율을 잴 개발셋 실행(노면 마스크가 다른 것만. 남기기 규칙은 마스크를 바꾸지 않음)

EXPERIMENTS = [
    # 비교 단계 (개발셋). 이 순서로 돌고, adopt 단계는 끝날 때마다 잠정 판정해 뒤 단계 설정에 반영한다
    _exp("E0_all", "compare", "train", "e0_all", "E0 재현: 1차 방법 804장", FIRST, split="all"),
    _exp("E0", "compare", "train", "e0_dev", "E0 1차 방법", FIRST, split="dev"),
    _exp("E1", "compare", "train", "e1_mask_dev", "E1 + 노면 마스크", {**FIRST, "mask": "on"}, split="dev", adopt=True),
    _exp("E1b", "compare", "train", "e1b_top03_dev", "E1b + 띠 0.3", {**FIRST, "mask": "on", "roi_top": E1B_ROI_TOP},
         split="dev", adopt=True),
    _exp("E2", "compare", "train", "e2_dark_dev", "E2 어두운 영역 검출(마스크 없음)", DARK, split="dev", adopt=True),
    _exp("E3", "compare", "train", "e3_dark_mask_dev", "E3 어두운 영역 + 노면 마스크(σ 노면 안)",
         {**DARK, "mask": "on", "sigma_in_road": True}, split="dev", adopt=True, follow_band=True),
    _exp("E3s", "compare", "train", "e3_sigma_band_dev", "E3σ 어두운 영역 + 노면 마스크(σ 띠 전체)",
         {**DARK, "mask": "on", "sigma_in_road": False}, split="dev", adopt=True, follow_band=True),
    # E4 자리 (Should, Plan.md 3.2 (3)·3.10): B가 후보 score를 채우고 evaluate_train에 τ 곡선(영상당 오검출 1·2개의 민감도)이 생기면
    # todo를 지우고 tool·opts(E3 설정 + τ 옵션)를 주며, rule1()을 같은 동작점의 민감도 비교로 바꾼다
    _exp("E4", "compare", None, "e4_score_dev", "E4 점수 문턱 τ (Should)", split="dev", adopt=True,
         todo="점수 τ: 보완 비교(supp) 단계의 E4s에서 돔(docs/tasks/supplement.md 1-3)"),
    # 39장 자동 ROI (참고): 손으로 정한 ROI 없이 도는지(확정본 6장 E1, 교수님 질문 'ROI 자동 설정'). 채택 판단에는 쓰지 않는다
    _exp("E1_39", "compare", "run39", "e1_mask_auto", "E1 39장 자동 ROI (참고)", {**FIRST, "mask": "on"}, roi="auto", reference=True),
    _exp("E3_39", "compare", "run39", "e3_dark_mask_auto", "E3 39장 자동 ROI (참고)", {**DARK, "mask": "on", "sigma_in_road": True},
         roi="auto", reference=True, follow_band=True),
    *[_exp(f"E6_{kind}", "compare", "train", f"e6_{kind}_dev", f"E6 1회 {DEGRADE_NAMES[kind]}", {"degrade": kind},
           split="dev", follow_last=True) for kind in et.DEGRADE_KINDS],
    # 최종 단계 (config 그대로). 처리 시간을 보고서에 쓰는 39장 최종과 매칭을 앞에 둔다
    # (10/12 ratio 채택 뒤) 새 폴더에 쓴다. 예전 최종(남기기 center)은 final·final_auto·e0·e5_dev·e6_*_final_dev에 그대로 둔다
    _exp("F39", "final", "run39", "final_ratio", "39장 최종 설정(roi.csv)", roi="csv"),
    _exp("FM", "final", "match", "final_ratio", "특징점·매칭 최종 설정"),
    _exp("F39a", "final", "run39", "final_auto_ratio", "39장 최종 설정(자동 ROI)", roi="auto"),
    _exp("F39e0", "final", "run39", "e0_final_ratio", "39장 1차 설정", FIRST, roi="csv"),
    _exp("E5", "final", "train", "e5_ratio_dev", "E5 최종 설정 전처리 끔·켬", split="dev"),
    *[_exp(f"F6_{kind}", "final", "train", f"e6_{kind}_final_ratio_dev", f"E6 2회(최종) {DEGRADE_NAMES[kind]}", {"degrade": kind},
           split="dev") for kind in et.DEGRADE_KINDS],
    # 시험 단계: 1차 방법을 먼저(실패하면 test_once 전에 멈춤), 그다음 최종 설정을 한 번만
    _exp("T0", "test", "train", "e0_test", "시험셋 1차 방법", FIRST, split="test"),
    _exp("T", "test", "train", "test_once", "시험셋 최종 설정(띠 고정, 한 번만)", split="test", fixed_band=True),
    # 재현 단계 (10/12): test_once와 같은 코드·설정으로 다시
    _exp("R", "repro", "train", "final_test", "시험셋 재현 확인(띠 고정)", split="test", fixed_band=True),
    # 자동 띠 비교 (10/12, 참고): 결 조건 없는 자동 띠. 띠 고정(E3·E3_39)과 자동 띠 + 결(E5·F39a, 최종)은 다른 단계의 결과를 쓴다
    _exp("B1", "band", "train", "band_auto_dev", "자동 띠(결 없음) 개발셋 (참고)",
         {**DARK, "mask": "on", "sigma_in_road": True, "auto_band": "on", "band_texture": "off"}, split="dev", reference=True),
    _exp("B1_39", "band", "run39", "band_auto_auto", "자동 띠(결 없음) 39장 자동 ROI (참고)",
         {**DARK, "mask": "on", "sigma_in_road": True, "auto_band": "on", "band_texture": "off"}, roi="auto", reference=True),
    # 시험셋 다시 재기 (10/12, 자동 띠 채택 뒤): 최종 설정(config 그대로)으로 한 번만, 그다음 같은 코드로 재현 확인
    _exp("TA", "retest", "train", "test_autoband", "시험셋 최종 설정(자동 띠 + 결, 한 번만)", split="test"),
    _exp("RA", "retest_repro", "train", "final_test_autoband", "시험셋 재현 확인(자동 띠 + 결)", split="test"),
    # 노면 마스크 보완 비교 (10/12, 참고): 이전 최종(자동 띠 + 결, 남기기 center)에 하나씩 더함. 차선은 남기기 판정(center 유지 상태) 위에
    *[e for i, (folder, folder39, label, sets) in enumerate(MASK2_RUNS)
      for e in (_exp(f"M{i}", "mask2", "train", folder, label, {"sets": sets}, split="dev", reference=True),
                _exp(f"M{i}_39", "mask2", "run39", folder39, f"{label} 39장 자동 ROI", {"sets": sets}, roi="auto", reference=True))],
    # 보완 비교 (10/12 밤~, 개발셋): 모든 실행에 τ 곡선. 판정은 τ 곡선의 동작점(supp_table)
    *[_exp(sid, "supp", "train", folder, label, {"sets": sets, "tau_curve": True}, split="dev", reference=True,
           supp_base=base, judge_row=row, judge=judge) for sid, folder, label, sets, base, row, judge in SUPP_RUNS],
    *[_exp(sid, "supp", "run39", folder, label, {"sets": sets}, roi="auto", reference=True)
      for sid, folder, label, sets in SUPP39_RUNS],
    # 시험셋 세 번째 재기 (10/12, ratio 채택 뒤): 최종 설정(config 그대로)으로 한 번만, 그다음 같은 코드로 재현 확인
    _exp("TR", "retest_ratio", "train", "test_ratio", "시험셋 최종 설정(자동 띠 + 결 + ratio, 한 번만)", split="test"),
    _exp("RR", "retest_ratio_repro", "train", "final_test_ratio", "시험셋 재현 확인(자동 띠 + 결 + ratio)", split="test"),
]
EXP = {e["id"]: e for e in EXPERIMENTS}
ADOPT_ORDER = [e["id"] for e in EXPERIMENTS if e.get("adopt")]       # E1, E1b, E2, E3, E3s, E4

METRICS = ["images", "n_damage", "n_none", "sensitivity", "specificity", "accuracy", "baseline_none", "balanced",
           "fp_per_image", "pred_per_image",
           "crack_ctr_recall", "crack_ctr_recall_net", "crack_ctr_precision",
           "pothole_ctr_recall", "pothole_ctr_recall_net", "pothole_ctr_precision",
           "crack_iou_recall", "crack_iou_precision", "pothole_iou_recall", "pothole_iou_precision", "chance_skipped"]
STAGE_IDS = ["E0_all", "E0", "E1", "E1b", "E2", "E3", "E3s", "E4", "B1", "M0", "M1", "M2", "M3", "E5", "T0", "T", "TA", "TR"]   # 1번 단계별 표의 행 순서
STAGE_HEAD = ["stage_id", "label", "data", "folder", "status", "commit"]
COUNTRY_RUNS = (("E0", "1차"), ("E3", "띠 고정(E3)"), ("B1", "자동 띠(결 없음)"), ("M0", "자동 띠 + 결(center)"),
                ("E5", "최종(자동 띠 + 결 + ratio)"),
                ("T0", "1차"), ("T", "띠 고정"), ("TA", "자동 띠 + 결(center)"), ("TR", "최종(자동 띠 + 결 + ratio)"))
RUNS39 = (("F39e0", "1차 설정"), ("F39", "최종(roi.csv)"), ("F39a", "최종(자동 ROI, 자동 띠 + 결 + ratio)"),
          ("E1_39", "E1 자동 ROI(참고)"), ("E3_39", "E3 자동 ROI(참고, 띠 고정)"), ("B1_39", "자동 띠(결 없음) 자동 ROI(참고)"))
# 9번 자동 띠 비교: (단계, 이름). 개발셋은 전처리 끔(채택 규칙의 조건), 39장은 자동 ROI 끔·켬
BAND_DEV = (("E3", "띠 고정(E3)"), ("B1", "자동 띠"), ("M0", "자동 띠 + 결"))      # 모두 남기기 center (ratio는 10번 표)
BAND_39 = (("E3_39", "띠 고정(E3)"), ("B1_39", "자동 띠"), ("M0_39", "자동 띠 + 결"))
MASK2_IDS = ("M0", "M1", "M2", "M3")    # 10번 노면 마스크 보완 표 (개발셋, 전처리 끔). 규칙은 M0와 견준다
# 시험셋 세 번 (10/12 보고): (단계, 이름, 설정 글)
TEST_RUNS = (("T", "띠 고정", "E3(dark + 노면 마스크) + 띠 고정 0.5 + 남기기 center"),
             ("TA", "자동 띠", "E3 + 자동 띠 + 결 1.5 + 남기기 center"),
             ("TR", "자동 띠 + ratio(최종)", "E3 + 자동 띠 + 결 1.5 + 남기기 ratio"))
IMAGE_RUNS = (("first", "F39e0"), ("final", "F39"), ("auto", "F39a"), ("e1_auto", "E1_39"), ("e3_auto", "E3_39"))
GROUP39 = ["n_images", "n_gt", "n_damage", "n_none", "sensitivity_off", "sensitivity_on", "specificity_off", "specificity_on",
           "accuracy_off", "accuracy_on", "baseline_none"]
SOURCE39 = GROUP39 + ["n_crack_off", "n_crack_on", "n_pothole_off", "n_pothole_on", "edges_off", "edges_on"]
PRE39 = [("에지 수", "edges"), ("노면 안 에지 수", "edges_road"), ("해리스 코너 수", "harris"), ("균열 후보 수", "n_crack"),
         ("균열 면적 % (노면 대비)", "area_crack_pct"), ("포트홀 후보 수", "n_pothole"), ("포트홀 면적 % (노면 대비)", "area_pothole_pct"),
         ("전처리 시간 ms", "t_pre_ms"), ("검출 시간 ms", "t_detect_ms"), ("노면 마스크 시간 ms", "t_road_ms"),
         ("자동 띠 경계 시간 ms", "t_band_ms")]
TIME39 = ("t_pre_ms", "t_detect_ms", "t_road_ms", "t_band_ms")     # 39장 처리 시간 합(4번 표 '처리 시간 합')
OPTIONAL_TIMES = ("t_band_ms",)     # 자동 띠 전에 돈 결과에는 없는 시간 열: 그때는 띠 경계를 계산하지 않았으므로 0으로 읽는다
PRE_TRAIN = [("에지 수", "edges_mean"), ("해리스 코너 수", "harris_mean"), ("균열 후보 수", "pred_crack_mean"),
             ("균열 면적 (픽셀)", "area_crack_mean"), ("포트홀 후보 수", "pred_pothole_mean"), ("포트홀 면적 (픽셀)", "area_pothole_mean")]
MATCH_ITEMS = [("SIFT 특징점 수(기준 A)", "kp_ref"), ("SIFT 특징점 수(대상)", "kp_target"), ("good 매칭 점 수", "good"),
               ("inlier 수", "inliers"), ("inlier 비율", "inlier_ratio")]
DEGRADE_ROUNDS = (("1회(비교)", "E6_"), ("2회(최종 설정)", "F6_"))
DEGRADE_COLS = ["sensitivity", "specificity", "accuracy", "baseline_none", "balanced", "fp_per_image",
                "crack_ctr_recall", "crack_ctr_recall_net", "pothole_ctr_recall", "pothole_ctr_recall_net",
                "edges_mean", "harris_mean", "pred_crack_mean", "pred_pothole_mean", "area_crack_mean", "area_pothole_mean",
                "judge_expect", "judge_hit_rate", "judge_base_rate"]
TIME_MEANS = ["t_pre_ms_mean", "t_detect_ms_mean", "t_road_ms_mean", "t_band_ms_mean"]   # 학습 데이터 처리 시간(사진마다 한 번 잰 값의 평균): 참고용
COMPARE_KEYS = (("sens", "sensitivity"), ("fp", "fp_per_image"), ("balanced", "balanced"), ("crack_net", "crack_ctr_recall_net"))
# 채택 표 위의 네 칸 (검출기, 노면 마스크, 단계): 마스크를 켠 1차 칸은 E1b가 채택되면 E1b(띠 0.3). E3σ는 E3 아래 한 줄
GRID_CELLS = (("1차(legacy)", "끔", "E0"), ("1차(legacy)", "켬", "E1"), ("dark", "끔", "E2"), ("dark", "켬", "E3"),
              ("dark", "켬 (σ 띠 전체)", "E3s"))
GRID_METRICS = ["sensitivity", "specificity", "balanced", "fp_per_image", "crack_ctr_recall_net"]
GRID_COLUMNS = ["detector", "mask", "band", "stage_id"] + GRID_METRICS + ["time", "status"]
ADOPT_COLUMNS = (["stage_id", "label", "status", "base", "changed"]
                 + [f"{k}{s}" for k, _ in COMPARE_KEYS for s in ("_base", "", "_diff")]
                 + ["time_base", "time", "time_diff", "rule1", "rule2", "rule3", "verdict",
                    "ref_base", "ref_changed", "ref_sens_diff", "ref_fp_diff", "ref_balanced", "ref_balanced_diff",
                    "ref_crack_net", "ref_crack_net_diff", "ref_rule2", "ref_rule3"])
JUDGE_TAGS = ("blur", "dark", "bright", "noise")   # 사람 태그 가운데 골라야 할 전처리가 정해진 것(evaluate_train.EXPECTED_STEPS)


@contextlib.contextmanager
def patched(obj, values):
    """obj의 속성(config 값이나 도구의 경로 상수)을 잠시 values로 바꿨다가, 끝나면(실패해도) 되돌린다."""
    backup = {name: getattr(obj, name) for name in values}
    for name, value in values.items():
        setattr(obj, name, value)
    try:
        yield
    finally:
        for name, value in backup.items():
            setattr(obj, name, value)


def default_paths():
    """실제 실행의 경로(맨 위 폴더 기준)와 학습 사진 수. 시험에서는 임시 폴더를 가리키는 dict를 대신 넘긴다."""
    return {"results": "results", "image_dirs": rx.IMAGE_DIRS, "roi": "data/roi.csv", "gt": "gt", "pairs": rm.PAIR_DIR,
            "train_img": C.TRAIN_IMAGE_DIR, "train_gt": C.TRAIN_GT_DIR, "n_train": N_TRAIN}


def parse_args(argv=None):
    """단계(compare·final·test·repro·tables)와 --dry-run·--yes·--accept·--reject·--force를 읽는다.
    --accept·--reject는 채택을 견주는 단계(E1~E4) 이름만 받고 같은 단계를 둘 다에 넣으면 거부한다. --force는 dark·mask만 받는다."""
    parser = argparse.ArgumentParser(description="E0~E6 일괄 실행과 보고서 표")
    parser.add_argument("stage", choices=list(STAGES),
                        help="compare(비교) → final(최종) → test(시험, 한 번만) → repro(재현, 10/12). tables는 표만 다시")
    parser.add_argument("--dry-run", action="store_true", help="돌릴 실험·명령·결과 폴더·예상 시간만 보여 주고 끝냄(아무것도 쓰지 않음)")
    parser.add_argument("--yes", action="store_true", help="시작 전 확인(건너뛸 실험·전원·시험셋 Enter)을 건너뜀")
    parser.add_argument("--accept", help="사람이 채택하기로 정한 단계를 쉼표로(예: E1b). 자동 판정보다 앞선다. 빈 값('')이면 지움. "
                                         "주지 않으면 지난번 값(results/report/accept.txt)")
    parser.add_argument("--reject", help="사람이 되돌리기로 정한 단계(예: 규칙 1에서 뺀 것)를 쉼표로(예: E1,E3). 빈 값('')이면 지움. "
                                         "주지 않으면 지난번 값(results/report/reject.txt)")
    parser.add_argument("--only", help="단계 안에서 이 실험만 쉼표로(예: S12,E5s). 앞 행을 다시 돌리지 않으려고 쓴다")
    parser.add_argument("--force", help="A·B 시험의 @todo가 남아 있어도 쓸 기능을 쉼표로: dark, mask. "
                                        f"그 기능을 쓴 실험은 표에 '{FORCED_MARK}'으로 남는다")
    args = parser.parse_args(argv)
    for flag in ("accept", "reject"):
        value = getattr(args, flag)
        if value is not None:
            ids = [s.strip() for s in value.split(",") if s.strip()]
            choices = ADOPT_ORDER + list(SUPP_IDS)
            bad = [s for s in ids if s not in choices]
            if bad:
                parser.error(f"--{flag}는 {', '.join(choices)} 가운데서만 고른다: {', '.join(bad)}")
            setattr(args, flag, ids)
    both = set(args.accept or []) & set(args.reject or [])
    if both:
        parser.error(f"--accept와 --reject에 같은 단계가 있음: {', '.join(sorted(both))}")
    force = [s.strip() for s in (args.force or "").split(",") if s.strip()]
    bad = [s for s in force if s not in SIGNALS]
    if bad:
        parser.error(f"--force는 {', '.join(SIGNALS)} 가운데서만 고른다: {', '.join(bad)}")
    args.force = set(force)
    args.only = [s.strip() for s in (args.only or "").split(",") if s.strip()] or None
    if args.only:
        known = {e["id"] for e in EXPERIMENTS if e["stage"] == args.stage}
        bad = [s for s in args.only if s not in known]
        if bad:
            parser.error(f"--only는 {args.stage} 단계의 실험({', '.join(sorted(known))}) 가운데서만 고른다: {', '.join(bad)}")
    return args


# ---------------------------------------------------------
# 실험 설정과 명령
# ---------------------------------------------------------

def param_values(name):
    """evaluate_train의 1차 튜닝 세트 하나(base 등)를 config 값 {이름: 값}으로."""
    return {key: value for params in et.PARAM_SETS if params["name"] == name for key, value in params.items() if key != "name"}


def last_adopted(adopted):
    """E1~E3 가운데 채택 순서로 마지막에 잠정 채택된 단계. 없으면 E0. E6 1회의 설정과 바꿀 config 값의 기준이다."""
    return next((sid for sid in reversed(ADOPT_ORDER) if sid in adopted), "E0")


def resolve(exp, adopted):
    """실험이 실제로 쓸 옵션: 정해진 옵션에 비교 단계의 앞선 채택 결과를 반영한다. adopted는 잠정 채택된 단계 집합(E0 포함).
    follow_band(마스크를 켜는 E3 쪽)는 E1b가 채택되면 띠 0.3, 아니면 0.5. 띠 0.3은 마스크를 켤 때만 뜻이 있어 E2는 0.5 그대로다.
    follow_last(E6 1회)는 E1~E3 가운데 마지막 잠정 채택 단계의 설정(없으면 E0)에 저하 종류만 바꾼다."""
    opts = dict(exp["opts"])
    if exp.get("follow_band"):
        opts["roi_top"] = E1B_ROI_TOP if "E1b" in adopted else FIRST_ROI_TOP
    if exp.get("follow_last"):
        opts = {**resolve(EXP[last_adopted(adopted)], adopted), "degrade": exp["opts"]["degrade"]}
    return opts


def command(exp, opts):
    """실험의 도구 명령행 인자와, 명령행으로 줄 수 없어 실행 중에만 고정할 config 값 {이름: 값}을 만든다(opts는 resolve의 결과).
    run_experiment에는 --param-set·--set이 없어 1차 legacy 값(evaluate_train의 base 세트)과 DARK_SIGMA_IN_ROAD를 실행 중에만 덮고,
    --roi csv에서는 --roi-top을 받지 않아 띠 위쪽 비율도 실행 중에만 덮는다(39장은 모두 roi.csv에 있어 결과는 같다).
    자동 띠(opts auto_band·band_texture)는 도구 옵션으로 주고, 띠 위쪽 비율을 정한 실험과 fixed_band 실험은 AUTO_BAND_TOP=False로 고정한다.
    opts sets(그 밖의 config 값)는 두 도구 모두 --set으로 준다. ratio 채택 전에 돈 단계(RATIO_STAGES 밖)의 실험은 남기기 규칙을
    sets로 정하지 않았으면 실행 중에만 ROAD_KEEP_RULE='center'로 고정해 처음과 같은 조건으로 돈다."""
    argv, pins = [], {}
    if exp["tool"] == "train":
        argv += ["--split", exp["split"]]
        if opts.get("params"):
            argv += ["--param-set", opts["params"]]
    elif exp["tool"] == "run39":
        argv += ["--roi", exp["roi"]]
        if opts.get("params"):
            pins.update(param_values(opts["params"]))
    for key in ("detect", "mask"):
        if opts.get(key):
            argv += [f"--{key}", opts[key]]
    if "roi_top" in opts:
        if exp["tool"] == "run39" and exp["roi"] == "csv":
            pins["ROI_TOP_DEFAULT"] = opts["roi_top"]
        else:
            argv += ["--roi-top", str(opts["roi_top"])]
    if "sigma_in_road" in opts:
        if exp["tool"] == "train":
            argv += ["--set", f"DARK_SIGMA_IN_ROAD={opts['sigma_in_road']}"]
        else:
            pins["DARK_SIGMA_IN_ROAD"] = opts["sigma_in_road"]
    if opts.get("auto_band"):
        if exp["tool"] == "train":
            argv += ["--set", f"AUTO_BAND_TOP={opts['auto_band'] == 'on'}"]
            if opts.get("band_texture"):
                argv += ["--set", f"AUTO_BAND_USE_TEXTURE={opts['band_texture'] == 'on'}"]
        else:
            argv += ["--auto-band", opts["auto_band"]] + (["--band-texture", opts["band_texture"]] if opts.get("band_texture") else [])
    elif "roi_top" in opts or exp.get("fixed_band"):
        pins["AUTO_BAND_TOP"] = False           # 띠를 비율로 정한 실험·띠 고정 시절 시험셋: 자동 띠를 켠 뒤에도 처음과 같은 조건
    if opts.get("degrade"):
        argv += ["--degrade", opts["degrade"]]
    if opts.get("tau_curve") and exp["tool"] == "train":
        argv += ["--tau-curve"]
    sets = opts.get("sets") or {}
    for name, value in sets.items():
        argv += ["--set", f"{name}={value}"]
    if exp["stage"] not in RATIO_STAGES and "ROAD_KEEP_RULE" not in sets and exp["tool"] != "match":
        pins["ROAD_KEEP_RULE"] = KEEP_BEFORE_RATIO
    if exp["stage"] not in SCORE_STAGES and exp["tool"] != "match":
        pins.update({k: v for k, v in SCORE_BEFORE_E4.items() if k not in sets})   # E4 전에 돈 단계는 다시 돌려도 처음과 같은 조건
    return argv + ["--name", exp["name"]], pins


def pins_text(pins):
    """실행 중에만 고정한 config 값을 한 줄로. 1차 legacy 값 묶음은 이름만 줄여 쓴다."""
    if not pins:
        return "없음"
    base = param_values("base")
    rest = {k: v for k, v in pins.items() if not (k in base and base[k] == v)}
    parts = [f"1차 legacy 값 {len(pins) - len(rest)}개(evaluate_train base 세트)"] if len(rest) < len(pins) else []
    return ", ".join(parts + [f"{k}={v!r}" for k, v in rest.items()])


def settings_text(opts):
    """옵션을 사람이 읽는 설정 글로: '검출 dark, 노면 마스크 켬, 띠 0.3, σ 노면 안'. 옵션에 없는 값은 config."""
    onoff = {"on": "켬", "off": "끔"}
    parts = [f"검출 {opts.get('detect') or 'config'}", f"노면 마스크 {onoff.get(opts.get('mask'), 'config')}",
             f"띠 {opts['roi_top']}" if "roi_top" in opts else "띠 config"]
    if "sigma_in_road" in opts:
        parts.append(f"σ {'노면 안' if opts['sigma_in_road'] else '띠 전체'}")
    if opts.get("params"):
        parts.append(f"legacy 값 {opts['params']}(1차)")
    if opts.get("auto_band"):
        parts.append(f"자동 띠 {onoff[opts['auto_band']]}" + (f"(결 {onoff[opts['band_texture']]})" if opts.get("band_texture") else ""))
    parts += [f"{name}={value}" for name, value in (opts.get("sets") or {}).items()]
    if opts.get("tau_curve"):
        parts.append("τ 곡선")
    return ", ".join(parts)


# ---------------------------------------------------------
# 구현 여부
# ---------------------------------------------------------

def needs(opts):
    """옵션이 쓰는 (검출 방식, 노면 마스크 켬?): 옵션으로 준 값, 없으면 지금 config 값."""
    mode = opts.get("detect") or C.DETECT_MODE
    mask = opts["mask"] == "on" if opts.get("mask") else bool(C.USE_ROAD_MASK)
    return mode, mask


def todo_removed(path, test_name):
    """A·B 시험 파일에서 그 시험 함수의 @todo 표시가 지워졌는지. 각 파트가 기능을 구현하면 지우는 통과 기준이라
    (docs/tasks/A.md·B.md 4장) 구현 여부의 분명한 신호로 쓴다. 파일이나 시험 함수를 찾지 못하면 None."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == test_name:
            return not any(isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "todo"
                           for d in node.decorator_list)
    return None


def dark_runs():
    """합성 영상 검사(경고용): 작은 영상을 DETECT_MODE "dark"로 검출해 자리의 NotImplementedError가 나지 않는지."""
    gray = np.full((64, 64), 128, np.uint8)
    gray[28:36, 8:56] = 60                                  # 어두운 띠 하나
    with patched(C, {"DETECT_MODE": "dark"}):
        try:
            detect.detect_cracks(gray)
            detect.detect_potholes(gray)
        except NotImplementedError:
            return False
        except Exception:                                   # 구현은 됐다. 오류는 실제 실행에서 그대로 드러난다
            return True
    return True


def mask_removes_something():
    """합성 영상 검사(경고용): 왼쪽이 풀색인 합성 띠에서 노면 마스크가 노면 아님(0)을 하나라도 내는지. 자리(전부 255)면 False."""
    band = np.full((120, 240, 3), 120, np.uint8)
    band[:, :100] = (40, 160, 60)                           # 채도가 높은 풀색(BGR)
    try:
        road = io_utils.road_mask(band)
    except NotImplementedError:
        return False
    except Exception:                                       # 구현은 됐다. 오류는 실제 실행에서 그대로 드러난다
        return True
    return bool(np.any(np.asarray(road) == 0))


def readiness(force=()):
    """노면 마스크·dark 검출의 구현 여부와 경고 {"mask": bool, "dark": bool, "notes": [경고], "forced": {강제로 쓴 기능}}.
    실행 한 번에 한 번만 잰다. 구현 여부는 A·B 시험의 @todo가 지워졌는지(SIGNALS)로만 정한다. 합성 영상 검사는 신호와 어긋날 때
    경고만 하므로, 진짜 마스크가 합성 띠에서 우연히 아무것도 안 지워도 E1을 건너뛰지 않는다(E1 = E0이면 '효과 없음' 결과로 남음).
    신호를 찾지 못하면(시험 이름이 바뀜 등) 조용히 건너뛰지 않도록 구현된 것으로 보고 돌린다.
    force(--force)에 든 기능은 @todo가 남아 있어도 쓰고 forced에 넣는다(그 기능을 쓴 실험은 표에 '강제 실행(시험 미통과)')."""
    probes = {"mask": mask_removes_something, "dark": dark_runs}
    ready, notes, forced = {}, [], set()
    for key, (path, test) in SIGNALS.items():
        signal, probe = todo_removed(path, test), probes[key]()
        ready[key] = True if signal is None else signal
        where = f"{path}의 {test}"
        if signal is None:
            notes.append(f"{NEEDS[key]}: {where}를 찾지 못해 구현된 것으로 보고 돌린다(시험 이름이 바뀌었는지 확인)")
        elif signal and not probe:
            notes.append(f"{NEEDS[key]}: {where}의 @todo는 지워졌지만 {PROBE_FAIL[key]}")
        elif not signal and probe:
            notes.append(f"{NEEDS[key]}: {where}에 @todo가 남아 구현 전으로 보지만, {PROBE_PASS[key]}")
        if not ready[key] and key in force:
            ready[key] = True
            forced.add(key)
            notes.append(f"{NEEDS[key]}: {where}에 @todo가 남아 있지만 --force로 쓴다. 이 기능을 쓴 실험은 표에 '{FORCED_MARK}'으로 남는다"
                         + ("" if probe else f". 다만 {PROBE_FAIL[key]}"))
    return {**ready, "notes": notes, "forced": forced}


def unmet(exp, ready, opts=None):
    """실험에 필요한데 아직 구현되지 않은 것의 이름 list(비면 돌릴 수 있음). opts를 주면 그 설정으로(resolve의 결과) 판단한다.
    매칭은 ROI·검출을 쓰지 않아 늘 돌린다."""
    if exp.get("todo"):
        return [exp["todo"]]
    if exp["tool"] == "match":
        return []
    mode, mask = needs(opts if opts is not None else exp["opts"])
    return ([NEEDS["dark"]] if mode == "dark" and not ready["dark"] else []) + \
           ([NEEDS["mask"]] if mask and not ready["mask"] else [])


def forced_needs(exp, ready, opts):
    """실험이 쓰는 기능 가운데 --force로 쓴 것(A·B 시험의 @todo가 아직 남은 기능)의 이름 list. 표에 '강제 실행(시험 미통과)'으로 남긴다."""
    if exp.get("todo") or exp["tool"] == "match":
        return []
    mode, mask = needs(opts)
    return ([NEEDS["dark"]] if mode == "dark" and "dark" in ready["forced"] else []) + \
           ([NEEDS["mask"]] if mask and "mask" in ready["forced"] else [])


def skip_list(exps, ready):
    """시작 전에 보일 건너뛸 실험 [(실험, 빠진 기능)]. E3 쪽 띠는 앞선 채택으로 바뀌어도 필요한 기능은 같다.
    E6 1회는 이미 돌아 채택된 단계의 설정만 따르므로 구현 전 기능 때문에 건너뛸 일이 없어 넣지 않는다."""
    return [(exp, missing) for exp in exps if not exp.get("follow_last")
            for missing in [unmet(exp, ready, resolve(exp, {"E0"}))] if missing]


def print_skips(skips, ready):
    """건너뛸 실험 목록을 눈에 띄게 보여 준다: 실험마다 빠진 기능, 판단 근거(@todo 신호), 합성 영상 경고."""
    bar = "!" * 78
    print(f"\n{bar}\n!! 건너뛸 실험 {len(skips)}개: 필요한 기능이 아직 구현 전으로 보임 (표에는 '아직 구현 전'으로 남음)")
    for exp, missing in skips:
        print(f"!!   {exp['id']:<6} {exp['label']}  ← {', '.join(missing)}")
    for key, (path, test) in SIGNALS.items():
        if not ready[key]:
            print(f"!! 근거: {NEEDS[key]} = {path}의 {test}에 @todo가 남아 있음")
    for note in ready["notes"]:
        print(f"!! 경고: {note}")
    print(bar)


# ---------------------------------------------------------
# 실행 전 확인
# ---------------------------------------------------------

def git(*args):
    """git 명령의 출력(앞뒤 공백을 뺀 글). git이 없거나 실패하면 None."""
    try:
        done = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_changes():
    """커밋 안 된 변경(고친 파일·새 파일) 목록. git을 쓸 수 없으면 None. 결과 폴더에 남기는 커밋 해시가 실제 코드와 맞으려면 비어야 한다."""
    out = git("-c", "core.quotepath=off", "status", "--porcelain")
    return None if out is None else [line for line in out.splitlines() if line.strip()]


def contract_check():
    """약속 검사(tests/check_contract.py)를 따로 돌려 (통과?, 마지막 몇 줄)."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        done = subprocess.run([sys.executable, "tests/check_contract.py"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", env=env, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"{type(e).__name__}: {e}"
    lines = (done.stdout + done.stderr).strip().splitlines()
    return done.returncode == 0, " / ".join(lines[-3:]) if done.returncode else (lines[-1] if lines else "")


def missing_gt(paths):
    """39장 가운데 정답 파일(gt/<사진 이름>.csv)이 없는 사진 이름과 39장 수. 정답은 마지막에 그리므로 멈추지 않고 39장 표에 '미완'으로 적는다."""
    files = rx.list_images(None, paths["image_dirs"])
    missing = [os.path.basename(p) for p in files if not os.path.exists(os.path.join(paths["gt"], f"{Path(p).stem}.csv"))]
    return missing, len(files)


def precheck(paths):
    """실행 전 확인 네 가지를 [(항목, 통과?, 내용, 어기면 멈춤?)]으로 돌려준다. 39장 정답만 멈추지 않는다."""
    missing, n39 = missing_gt(paths)
    note = f"{n39 - len(missing)}/{n39}장 있음"
    if missing:
        note += f". 없음 {len(missing)}장: {', '.join(missing)} → 39장 표에 '미완'"
    n_img = len(list(Path(paths["train_img"]).glob("*.jpg")))
    n_gt = len(list(Path(paths["train_gt"]).glob("*.csv")))
    ok, last = contract_check()
    changes = git_changes()
    if changes is None:
        change_note = "git을 쓸 수 없음"
    else:
        change_note = "없음" if not changes else f"{len(changes)}개: " + ", ".join(changes[:10]) + (" …" if len(changes) > 10 else "")
    return [("39장 정답 파일", not missing, note, False),
            ("학습 사진", n_img == n_gt == paths["n_train"], f"사진 {n_img}장, 정답 CSV {n_gt}개 (필요 {paths['n_train']}개씩)", True),
            ("약속 검사", ok, last, True),
            ("커밋 안 된 변경", changes == [], change_note, True)]


def print_checks(checks, ready):
    """실행 전 확인 결과를 한 줄씩 보여 준다: 통과, 멈춤(어기면 실행하지 않음), 알림(39장 정답). 구현 여부와 그 경고도 함께."""
    print("실행 전 확인")
    for name, ok, note, blocking in checks:
        print(f"  [{'통과' if ok else ('멈춤' if blocking else '알림')}] {name}: {note}")
    print("  [구현 여부] " + ", ".join(f"{NEEDS[k]} {readiness_word(ready, k)}" for k in SIGNALS) + " (A·B 시험의 @todo 기준)")
    for note in ready["notes"]:
        print(f"  [경고] {note}")


def readiness_word(ready, key):
    """구현 여부 한 낱말: 구현됨, 구현 전, 강제(--force, @todo가 남은 채 씀)."""
    return "강제(--force)" if key in ready.get("forced", ()) else ("구현됨" if ready[key] else "구현 전")


def confirm(message, yes, ask=input):
    """시작 전 확인: 문구를 보여 주고 Enter를 기다린다. --yes면 건너뛰고, 입력을 받을 수 없거나 Ctrl+C면 멈춘다."""
    print(message)
    if yes:
        print("(--yes: 확인을 건너뜀)")
        return True
    try:
        ask("계속하려면 Enter, 그만두려면 Ctrl+C: ")
    except (EOFError, KeyboardInterrupt):
        print("\n멈춤: 확인을 받지 못함 (입력 없이 돌리려면 --yes)")
        return False
    return True


# ---------------------------------------------------------
# 실행
# ---------------------------------------------------------

def run_context(stage, command_line):
    """run_info에 남길 실행 정보: 단계와 명령, 커밋 해시·브랜치, 환경(버전)."""
    return {"단계": f"{STAGES[stage]} ({command_line})",
            "커밋": f"{git('rev-parse', 'HEAD') or '알 수 없음'} (브랜치 {git('branch', '--show-current') or '알 수 없음'})",
            "환경": f"Python {platform.python_version()}, OpenCV {cv.__version__}, NumPy {np.__version__}, "
                    f"pandas {pd.__version__}, {platform.platform()}"}


def write_run_info(path, exp, run):
    """결과 폴더에 실행 기록을 남긴다(결과가 어떤 코드·명령·설정에서 나왔는지 폴더만 보고 알게). '옵션' 줄은 표를 만들 때
    앞선 채택과 실제로 돈 설정이 맞는지 보는 데 쓴다. config 값은 도구가 따로 쓴다(config_used*.txt)."""
    lines = [f"단계: {run['context']['단계']}", f"실험: {exp['id']} {exp['label']}",
             f"명령: python {TOOL_SCRIPT[exp['tool']]} {' '.join(run['argv'])}", f"옵션: {run['opts']!r}",
             f"설정: {settings_text(run['opts'])}", f"설정 출처: {run['source'] or '-'}",
             f"강제 실행: {'·'.join(run['forced']) or '없음'}",
             f"실행 중 고정한 config 값: {pins_text(run['pins'])}", f"상태: {run['status']}", f"커밋: {run['context']['커밋']}",
             f"시작: {run['started']:%Y-%m-%d %H:%M:%S}", f"걸린 시간: {run['seconds']:.1f}초", f"환경: {run['context']['환경']}",
             f"config 값: {CONFIG_FILE[exp['tool']]} (같은 폴더, 도구가 씀)"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_run_info(path):
    """run_info를 {항목: 값}으로 읽는다. 없으면 빈 dict."""
    if not path.exists():
        return {}
    pairs = (line.split(": ", 1) for line in path.read_text(encoding="utf-8").splitlines() if ": " in line)
    return dict(pairs)


def parse_opts(text):
    """run_info의 '옵션' 줄을 dict로. 없거나 읽을 수 없으면 None(run_all 밖에서 돌린 결과 등)."""
    try:
        value = ast.literal_eval(text) if text else None
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, dict) else None


def run_tool(exp, argv, paths):
    """도구의 main을 이 프로세스에서 부르고 끝 코드(0 = 성공)를 돌려준다. 오류는 부른 쪽이 잡는다.
    evaluate_train은 결과·학습 사진 폴더를 모듈 상수와 config에서 읽어서, paths 값으로 잠시 바꾼다(실제 실행에서는 같은 값)."""
    if exp["tool"] == "train":
        with patched(et, {"RESULTS_ROOT": Path(paths["results"])}), \
                patched(C, {"TRAIN_IMAGE_DIR": paths["train_img"], "TRAIN_GT_DIR": paths["train_gt"]}):
            et.main(argv)
        return 0
    if exp["tool"] == "run39":
        return rx.main(argv, paths["image_dirs"], paths["roi"], paths["results"])
    return rm.main(argv, paths["pairs"], paths["results"])


def run_one(exp, opts, paths, ready, context, source=""):
    """실험 하나를 opts(resolve의 결과)로 돌려 (상태, 걸린 초, 강제로 쓴 기능 list)를 돌려준다. 필요한 기능이 아직 없으면 돌리지 않고
    '아직 구현 전'. 끝나면 결과 폴더에 run_info를 남긴다(실패해서 폴더가 없으면 남기지 않음). source는 설정을 따온 단계(E6 1회)."""
    missing = unmet(exp, ready, opts)
    if missing:
        return "아직 구현 전: " + "·".join(missing), 0.0, []
    forced = forced_needs(exp, ready, opts)
    argv, pins = command(exp, opts)
    started, start = datetime.datetime.now(), time.perf_counter()
    try:
        with patched(C, pins):
            code = run_tool(exp, argv, paths)
        status = "완료" if code == 0 else f"실패: 끝 코드 {code} (실패한 사진은 위 출력)"
    except NotImplementedError as e:
        status = f"실패: {'강제로 썼지만' if forced else '구현 신호와 달리'} NotImplementedError ({e})"
    except (Exception, SystemExit) as e:                    # 명령행 오류(SystemExit)도 이 실험의 실패로 적고 다음으로
        status = f"실패: {type(e).__name__}: {e}"
    seconds = time.perf_counter() - start
    folder = Path(paths["results"]) / exp["name"]
    if folder.is_dir():
        write_run_info(folder / RUN_INFO[exp["tool"]], exp, {"context": context, "argv": argv, "opts": opts, "source": source,
                                                              "pins": pins, "forced": forced, "status": status,
                                                              "started": started, "seconds": seconds})
    return status, seconds, forced


def config_used_values(path):
    """도구가 남긴 config_used*.txt에서 'NAME = 값' 줄만 {이름: 값 글}로 읽는다('#' 줄은 뺌). 없으면 None."""
    if not path.exists():
        return None
    pairs = (line.split(" = ", 1) for line in path.read_text(encoding="utf-8").splitlines()
             if " = " in line and not line.startswith("#"))
    return {name: value for name, value in pairs}


def final_config_diff(results):
    """지금 config와 최종 단계(FINAL_CONFIG_EXP)를 돌린 config(config_used_train.txt)가 다른 값 [(이름, 최종, 지금)].
    최종 단계 결과가 없으면 None. retest가 시험셋을 최종 설정으로 재는지 확인하려고 쓴다."""
    used = config_used_values(results / EXP[FINAL_CONFIG_EXP]["name"] / CONFIG_FILE["train"])
    if used is None:
        return None
    now = {name: repr(value) for name, value in vars(C).items() if name.isupper()}
    return [(name, used.get(name, "(없음)"), now.get(name, "(없음)")) for name in sorted(set(used) | set(now))
            if used.get(name) != now.get(name)]


def stage_refusal(stage, results):
    """단계를 거부할 이유(없으면 None): 시험은 test_once가 이미 있으면(한 번만), 재현은 test_once 결과가 없으면.
    retest(test_autoband)·retest_ratio(test_ratio)는 그 폴더가 이미 있거나, 최종 단계 결과가 없거나, 지금 config가 최종 단계를 돌린
    config와 다르면. retest_repro·retest_ratio_repro는 앞 시험 단계의 결과가 없으면.
    (10/12 밤) 이미 잰 시험셋 단계(DONE_TEST_STAGES)는 결과 폴더가 있든 없든 먼저 거부한다. 아래 폴더 검사는 다음 시험 단계도 같이 쓴다."""
    if stage in DONE_TEST_STAGES:
        return (f"{stage} 단계의 시험셋은 10/12에 이미 쟀습니다({DONE_TEST_STAGES[stage]}). results/가 지워져 폴더로는 막을 수 없어 "
                "단계 이름으로 거부합니다. 다음 시험셋 평가는 보완을 마친 뒤 한 번뿐입니다(docs/tasks/supplement.md 0-1).")
    if stage in RETESTS:
        again = results / EXP[RETESTS[stage]]["name"]
        if again.exists():
            return f"{again.as_posix()}/가 이미 있습니다. 시험셋은 한 번만 잽니다(Plan.md 3.10)."
        diff = final_config_diff(results)
        if diff is None:
            return f"최종 단계 결과({EXP[FINAL_CONFIG_EXP]['name']})가 없습니다. final을 먼저 돌립니다(시험셋은 최종 설정으로 잼)."
        if diff:
            return "지금 config가 최종 단계를 돌린 config와 다릅니다: " + ", ".join(f"{n} 최종 {a} / 지금 {b}" for n, a, b in diff)
        return None
    if stage in RETEST_REPROS:
        once = results / EXP[RETESTS[RETEST_REPROS[stage]]]["name"]
        if not (once / et.SUMMARY_NAME).exists():
            return f"{once.as_posix()}/{et.SUMMARY_NAME}가 없습니다. 재현은 {RETEST_REPROS[stage]} 뒤에 돌립니다."
        return None
    once = results / "test_once"
    if stage == "test" and once.exists():
        hint = "" if (once / et.SUMMARY_NAME).exists() else \
            f" {et.SUMMARY_NAME}가 없어 끝나지 않은 실행으로 보입니다. 숫자를 보지 않았다면 폴더를 지운 뒤 다시 돌리세요."
        return f"{once.as_posix()}/가 이미 있습니다. 시험셋은 한 번만 잽니다(Plan.md 3.10).{hint}"
    if stage == "repro" and not (once / et.SUMMARY_NAME).exists():
        return f"{once.as_posix()}/{et.SUMMARY_NAME}가 없습니다. 재현은 시험 단계(test) 뒤에 돌립니다."
    return None


def duration(seconds):
    """초를 '9분 12초'처럼 읽기 쉬운 글로."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}초"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}분 {sec}초"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}시간 {minutes}분"


def split_counts(paths):
    """학습 사진의 분할별 장수 {all, dev, test}. 사진이 없으면 계획의 장수(예상 시간용)."""
    names = [p.name for p in Path(paths["train_img"]).glob("*.jpg")]
    if not names:
        return dict(PLANNED_SPLIT)
    split = et.assign_split(names)
    return {"all": len(names), "dev": sum(v == "dev" for v in split.values()), "test": sum(v == "test" for v in split.values())}


def estimate(exp, counts, n39, n_pair_photos):
    """실험 하나의 예상 초(C 컴퓨터에서 잰 값 기준)."""
    if exp["tool"] == "train":
        return counts[exp["split"]] * 2 * SEC_TRAIN
    if exp["tool"] == "run39":
        return n39 * SEC_PHOTO39
    if exp["tool"] == "match":
        return n_pair_photos * SEC_PAIR_PHOTO
    return 0.0


def plan_note(exp, adopted):
    """실험 줄에 붙일 설명: 앞선 채택으로 설정이 정해지는 실험과 참고 실험."""
    if exp.get("follow_last"):
        source = last_adopted(adopted)
        return f"설정: {source} ({'E1~E3 가운데 마지막 잠정 채택' if source != 'E0' else '잠정 채택 없음'})"
    if exp.get("follow_band"):
        note = f"띠 {E1B_ROI_TOP if 'E1b' in adopted else FIRST_ROI_TOP} ({'E1b 잠정 채택' if 'E1b' in adopted else 'E1b 채택 안 됨'})"
        return note + (", 참고: 채택 판단에 쓰지 않음" if exp.get("reference") else "")
    if exp.get("judge"):                    # 보완 비교: 비교 단계(compare) 채택과 따로, τ 곡선의 동작점으로 판정(14번 표)
        return {"base": "보완 비교 기준", "curve": f"{exp['supp_base']}와 곡선만 비교(채택 판정 없음)",
                "tau": f"{exp['supp_base']}의 τ 없음 점 대비 rule1 점", "rules": f"{exp['supp_base']}와 규칙 1~3(τ 곡선)"}[exp["judge"]]
    if exp.get("reference"):
        return "참고: 채택 판단에 쓰지 않음"
    return ""


def human_text(human):
    """사람이 정한 채택·되돌림을 한 줄로: '채택 E1b, 되돌림 E2'. 없으면 빈 글."""
    parts = [f"{word} {', '.join(human[key])}" for key, word in (("accept", "채택"), ("reject", "되돌림")) if human.get(key)]
    return ", ".join(parts)


def print_plan(stage, exps, paths, ready, refusal, skips, human):
    """--dry-run: 돌릴 실험·명령·결과 폴더·예상 시간을 보여 준다. 아무것도 쓰거나 돌리지 않는다."""
    counts = split_counts(paths)
    n39 = len(rx.list_images(None, paths["image_dirs"]))
    n_pairs = len(list(Path(paths["pairs"]).glob("*.jpg")))
    results = Path(paths["results"])
    skipped = {exp["id"] for exp, _ in skips}
    print(f"[{STAGES[stage]} 단계] 실험 {len(exps)}개 (dry-run: 돌리지 않음)")
    if refusal:
        print(f"  ※ 지금 실제로 돌리면 거부됨: {refusal}")
    if stage == "compare":
        print("  ※ E1·E1b·E2·E3·E3σ는 끝날 때마다 잠정 판정해 E3 쪽 띠(E1b 채택 → 0.3)와 E6 1회 설정(마지막 잠정 채택)에 반영한다. "
              "아래 명령은 아무것도 채택되지 않았을 때의 모양이다")
        if human_text(human):
            print(f"  ※ 사람이 정한 것(지난번 값 포함, 자동 판정보다 앞섬): {human_text(human)}")
    total, everything = SEC_CHECK, SEC_CHECK
    for i, exp in enumerate(exps, 1):
        opts = resolve(exp, {"E0"})
        folder = results / exp["name"]
        sec = estimate(exp, counts, n39, n_pairs)
        everything += sec
        note = plan_note(exp, {"E0"})
        print(f"{i:2d}. {exp['id']}  {exp['label']}" + (f"  [{note}]" if note else ""))
        print(f"    폴더: {folder.as_posix()}/" + ("  (이미 있음: 덮어씀)" if folder.exists() else ""))
        if exp["tool"]:
            argv, pins = command(exp, opts)
            print(f"    명령: python {TOOL_SCRIPT[exp['tool']]} {' '.join(argv)}")
            if pins:
                print(f"    실행 중 고정: {pins_text(pins)}")
        if exp["id"] in skipped:
            print(f"    건너뜀: 아직 구현 전 ({', '.join(unmet(exp, ready, opts))})")
        else:
            total += sec
            forced = forced_needs(exp, ready, opts)
            print(f"    예상: {duration(sec)}" + (f"  ({FORCED_MARK}: {', '.join(forced)})" if forced else ""))
    if skips:
        print_skips(skips, ready)
        print("실제 실행에서는 시작 전에 위 목록을 보여 주고 Enter를 기다린다(--yes면 그대로 진행)")
    else:
        for note in ready["notes"]:
            print(f"경고: {note}")
    if stage == "final":
        print("시작할 때: 전원 연결·다른 프로그램 닫기 안내 후 Enter를 기다림(--yes면 건너뜀)")
    if stage == "test":
        print("시작할 때: 지금 config를 보여 주고 Enter를 기다림(--yes면 건너뜀). 1차 방법이 실패하면 test_once를 돌리지 않음")
    print(f"예상 합계: 약 {duration(total)} (실행 전 확인 포함"
          + (f", 건너뛸 실험 {len(skips)}개 빼고), 모두 구현되면 약 {duration(everything)}" if skips else ")")
          + ". legacy·마스크 끔으로 잰 값이라 dark·노면 마스크는 더 걸릴 수 있음")
    print(f"표: {(results / REPORT_DIR).as_posix()}/ (table1_stages … table13_judge.csv, adoption.csv, summary.md)")


FINAL_PROMPT = "처리 시간을 잽니다: 노트북 전원을 연결하고 다른 프로그램을 닫아 주세요."


def test_prompt(ready):
    """시험 단계 시작 확인 문구: 한 번만 잰다는 것, 지금 config의 채택 관련 값, --force로 쓰는 기능."""
    forced = [NEEDS[k] for k in SIGNALS if k in ready.get("forced", ())]
    return ("시험셋은 한 번만 잽니다. 이 결과를 보고 파라미터를 고치지 않습니다(Plan.md 3.10).\n"
            f"지금 config: DETECT_MODE={C.DETECT_MODE!r}, USE_ROAD_MASK={C.USE_ROAD_MASK}, "
            f"ROI_TOP_DEFAULT={C.ROI_TOP_DEFAULT}, DARK_SIGMA_IN_ROAD={C.DARK_SIGMA_IN_ROAD}, AUTO_BAND_TOP={C.AUTO_BAND_TOP}, "
            f"AUTO_BAND_USE_TEXTURE={C.AUTO_BAND_USE_TEXTURE}, AUTO_BAND_ROAD_MIN={C.AUTO_BAND_ROAD_MIN}, AUTO_BAND_TEX_RATIO={C.AUTO_BAND_TEX_RATIO}"
            + (f"\n--force로 쓰는 기능(A·B 시험 미통과, 표에 '{FORCED_MARK}'): {', '.join(forced)}" if forced else ""))


def run_stage(stage, args, paths, human, command_line, ask=input):
    """한 단계를 돈다: 거부 조건 → 실행 전 확인 → 건너뛸 실험·시작 확인 → 실험을 차례로 → 표 만들기. 끝 코드는 main과 같다.
    비교 단계는 E0와 채택 단계가 끝날 때마다 잠정 판정(adoption, 사람이 정한 --accept·--reject가 앞섬)을 다시 내어
    뒤 실험의 설정(resolve)에 반영한다. human = {"accept": [...], "reject": [...]}"""
    exps = [e for e in EXPERIMENTS if e["stage"] == stage and (not getattr(args, "only", None) or e["id"] in args.only)]
    ready = readiness(args.force)
    results_dir = Path(paths["results"])
    refusal = stage_refusal(stage, results_dir)
    skips = skip_list(exps, ready)
    if args.dry_run:
        print_plan(stage, exps, paths, ready, refusal, skips, human)
        return 0
    if refusal:
        print(f"멈춤: {refusal}")
        return 2
    checks = precheck(paths)
    print_checks(checks, ready)
    if any(blocking and not ok for _, ok, _, blocking in checks):
        print("멈춤: 위 확인을 통과하지 못함 (39장 정답 말고는 모두 통과해야 돌림)")
        return 2
    if stage in SET_TEST_STAGES and skips:
        print(f"멈춤: 시험셋은 설정에 필요한 기능을 다 갖춘 뒤 돌린다. 아직 구현 전: "
              f"{', '.join(sorted({m for _, missing in skips for m in missing}))}")
        return 2
    if skips:
        print_skips(skips, ready)
        if not confirm("위 실험을 건너뛰고 나머지를 돌립니다.", args.yes, ask):
            return 2
    if stage == "final" and not confirm(FINAL_PROMPT, args.yes, ask):
        return 2
    if stage in TEST_STAGES and not confirm(test_prompt(ready), args.yes, ask):
        return 2
    if stage == "compare" and human_text(human):
        print(f"사람이 정한 것: {human_text(human)} (자동 판정보다 앞서고 뒤 단계 설정에 반영)")

    context = run_context(stage, command_line)
    start, done, results, adopted = time.perf_counter(), [], {}, {"E0"}
    for i, exp in enumerate(exps, 1):
        opts = resolve(exp, adopted)
        source = last_adopted(adopted) if exp.get("follow_last") else ""
        note = plan_note(exp, adopted)
        print(f"\n===== [{STAGES[stage]} {i}/{len(exps)}] {exp['id']} {exp['label']} -> {(results_dir / exp['name']).as_posix()}/"
              + (f"  [{note}]" if note else ""))
        status, seconds, forced = run_one(exp, opts, paths, ready, context, source)
        shown = status + (f" · {FORCED_MARK}: {'·'.join(forced)}" if forced else "")
        print(f"----- {exp['id']}: {shown} ({duration(seconds)})")
        done.append((exp, shown, seconds))
        if exp["id"] == "E0" or exp.get("adopt"):
            summary = train_summary(paths, exp, {"done": status == "완료"})
            results[exp["id"]] = {"off": (summary or {}).get("off"), "status": shown, "opts": opts, "forced": forced}
            table, adopted = adoption(results, human["reject"], human["accept"])
            if exp.get("adopt"):
                row = table.set_index("stage_id").loc[exp["id"]]
                print(f"      → 판정: {exp['id']} (비교 기준 {row['base']}) {row['verdict']}")
        if stage in TEST_STAGES and status != "완료":
            print("멈춤: 시험 단계는 실패하면 다음 실험을 돌리지 않는다(시험셋을 한 번만 재려고)")
            break

    print(f"\n===== {STAGES[stage]} 단계 끝 ({duration(time.perf_counter() - start)})")
    for exp, shown, seconds in done:
        print(f"  {exp['id']:<9} {exp['name']:<22} {shown} ({duration(seconds)})")
    report = build_report(paths, human, command_line)
    print_report(report, stage)
    return 1 if any(shown.startswith("실패") for _, shown, _ in done) else 0


# ---------------------------------------------------------
# 결과 읽기
# ---------------------------------------------------------

def read_csv(path):
    """CSV를 읽는다(utf-8-sig, 'NaN' 글은 NaN 수로). 없으면 None."""
    return pd.read_csv(path, encoding="utf-8-sig") if Path(path).exists() else None


def exp_state(exp, paths, ready):
    """실험의 지금 상태 {"done", "status", "commit", "opts", "source", "forced"}: 결과 파일이 있으면 run_info의 상태·커밋·옵션,
    없으면 구현 전·실패·안 돌림. opts는 실제로 돈 옵션(run_info '옵션' 줄)이고 앞선 채택과 맞는지 볼 때 쓴다.
    --force로 돈 실험은 forced(쓴 기능 이름)가 있고 상태 뒤에 '강제 실행(시험 미통과)'이 붙는다."""
    if exp["tool"] is None:
        return {"done": False, "status": "아직 구현 전: " + exp["todo"], "commit": "", "opts": None, "source": "", "forced": "",
                "started": ""}
    folder = Path(paths["results"]) / exp["name"]
    info = read_run_info(folder / RUN_INFO[exp["tool"]])
    forced = "" if info.get("강제 실행", "없음") == "없음" else info["강제 실행"]
    state = {"commit": info.get("커밋", "")[:7], "opts": parse_opts(info.get("옵션")), "source": info.get("설정 출처", "-"),
             "forced": forced, "started": info.get("시작", "")}
    if (folder / RESULT_FILE[exp["tool"]]).exists():
        return {**state, "done": True,
                "status": info.get("상태", "완료(run_all 밖에서 돌림)") + (f" · {FORCED_MARK}: {forced}" if forced else "")}
    missing = unmet(exp, ready)
    if missing:
        return {**state, "done": False, "status": "아직 구현 전: " + "·".join(missing), "commit": ""}
    return {**state, "done": False, "status": info.get("상태", "아직 안 돌림")}


def train_summary(paths, exp, state):
    """evaluate_train 요약을 {"off": 행, "on": 행}으로. 끝나지 않았으면 None."""
    df = read_csv(Path(paths["results"]) / exp["name"] / et.SUMMARY_NAME) if state["done"] else None
    return None if df is None else {row["preprocess"]: row for _, row in df.iterrows()}


def run39(paths, exp, state):
    """run_experiment 결과(metrics, 영상별·출처별·조건별 표)를 dict로. 끝나지 않았으면 None."""
    if not state["done"]:
        return None
    folder = Path(paths["results"]) / exp["name"]
    return {key: read_csv(folder / file) for key, file in (("metrics", "metrics.csv"), ("image", "table_by_image.csv"),
                                                          ("source", "table_by_source.csv"),
                                                          ("condition", "table_by_condition.csv"))}


def gt_status(metrics):
    """39장 실행의 정답 상태: 모두 있으면 '완료', 빠지면 '미완 (정답 n/N장)'. 정답은 마지막에 그리므로 빠진 채로도 표를 만든다."""
    off = metrics[metrics["preprocess"] == "off"]
    has = int(off["has_gt"].fillna(False).astype(bool).sum())
    return "완료" if has == len(off) else f"미완 (정답 {has}/{len(off)}장)"


def judgements(metrics):
    """39장 켬 행의 상태 지표로 상태 판정(analyze.choose_steps)을 다시 내어, 사람 태그(condition)에 맞는 전처리를 골랐는지 본다.
    다시 낸 전처리 글이 실행 때 적힌 steps와 다르면(그 사이 코드가 바뀜) 적중 칸을 비운다.
    → {사진: {"steps": 고른 전처리 글, "expect": 골라야 할 전처리(태그가 blur·dark·bright·noise일 때), "hit", "sharpen"}}"""
    keys = list(analyze.measure_quality(np.zeros((32, 32), np.uint8)))
    out = {}
    for _, row in metrics[metrics["preprocess"] == "on"].iterrows():
        steps = analyze.choose_steps({key: float(row[key]) for key in keys})
        same = analyze.steps_to_text(steps) == row["steps"]
        tag = row["condition"] if isinstance(row["condition"], str) else ""
        expect = et.EXPECTED_STEPS.get(tag) if tag in JUDGE_TAGS else None
        out[row["file"]] = {"steps": row["steps"], "condition": tag or "unknown", "expect": expect[0] if expect else "",
                            "hit": expect[1](steps) if expect and same else None,
                            "sharpen": bool(steps["sharpen"]) if same else None}
    return out


def blur_pairs(judge):
    """E5 상태 판정 검증: 같은 장소의 정상·흔들림 쌍(직접 촬영 own_번호가 같고 condition이 normal·blur)에서
    흔들림 쪽만 흐림(샤프닝)으로 잡았는지. 이름의 번호로 묶는 것은 평가의 정의이고 영상 처리에는 쓰지 않는다(Plan.md 3.10 E5).
    → [(번호, 흔들림 사진, 정상 사진, 맞음? 또는 None)]"""
    scenes = {}
    for file, j in judge.items():
        parts = Path(file).stem.split("_")
        if parts[0] == "own" and len(parts) > 1 and j["condition"] in ("blur", "normal"):
            scenes.setdefault(parts[1], {})[j["condition"]] = (file, j["sharpen"])
    pairs = []
    for scene, members in sorted(scenes.items()):
        if "blur" in members and "normal" in members:
            (blur, s_blur), (normal, s_normal) = members["blur"], members["normal"]
            ok = None if s_blur is None or s_normal is None else (s_blur and not s_normal)
            pairs.append((scene, blur, normal, ok))
    return pairs


# ---------------------------------------------------------
# 채택 비교
# ---------------------------------------------------------

def base_of(sid, adopted):
    """채택 비교의 기준(한 번에 하나만 바뀌게): E1←E0, E1b←E1, E2←E0, E3←마스크를 켠 1차 설정(E1b가 채택되면 E1b, 아니면 E1),
    E3σ←E3, E4←E3. E3와 기준은 검출 방식만 다르다(E3의 띠도 E1b 채택을 따르므로)."""
    if sid == "E3":
        return "E1b" if "E1b" in adopted else "E1"
    return {"E1": "E0", "E1b": "E1", "E2": "E0", "E3s": "E3", "E4": "E3"}[sid]


def changed_text(a, b):
    """두 설정(opts)의 차이를 글로: 검출 legacy→dark, 노면 마스크 끔→켬, 띠 위쪽 0.5→0.3, σ 노면 안→띠 전체, legacy 값, 저하."""
    onoff, sigma = {"on": "켬", "off": "끔", None: "config"}, {True: "노면 안", False: "띠 전체", None: "config"}
    parts = []
    if a.get("detect") != b.get("detect"):
        parts.append(f"검출 {a.get('detect')}→{b.get('detect')}")
    if a.get("mask") != b.get("mask"):
        parts.append(f"노면 마스크 {onoff[a.get('mask')]}→{onoff[b.get('mask')]}")
    if a.get("roi_top") != b.get("roi_top"):
        parts.append(f"띠 위쪽 {a.get('roi_top')}→{b.get('roi_top')}")
    both_dark_mask = all(o.get("detect") == "dark" and o.get("mask") == "on" for o in (a, b))
    if both_dark_mask and a.get("sigma_in_road") != b.get("sigma_in_road"):
        parts.append(f"σ {sigma[a.get('sigma_in_road')]}→{sigma[b.get('sigma_in_road')]}")
    if a.get("detect") == b.get("detect") == "legacy" and a.get("params") != b.get("params"):
        parts.append(f"legacy 값 {a.get('params') or 'config'}→{b.get('params') or 'config'}")
    if a.get("degrade") != b.get("degrade"):
        parts.append(f"저하 {a.get('degrade')}→{b.get('degrade')}")
    return ", ".join(parts) or "없음"


def passes(diff, rule):
    """차이 값이 규칙을 넘는지(소수 여섯째 자리로 반올림해 비교). 값이 없으면(NaN) None."""
    return None if pd.isna(diff) else bool(rule(round(float(diff), 6)))


def rule_text(ok):
    """규칙 판정 글: 통과·못 넘음·계산 불가(값 없음)."""
    return {True: "통과", False: "못 넘음", None: "계산 불가(값 없음)"}[ok]


def rule1(values):
    """채택 규칙 1(같은 동작점): 점수 문턱이 없는 단계(E0~E3)는 민감도와 영상당 오검출을 나란히 두고 사람이 본다(Plan.md 3.10).
    어느 쪽으로 움직였는지만 덧붙인다. E4(점수)가 생기면 영상당 오검출 1·2개에서의 민감도 비교로 바꾼다."""
    ds, df = values["sens_diff"], values["fp_diff"]
    if pd.isna(ds) or pd.isna(df):
        return "사람 확인(값 없음)"
    arrow = lambda d: "↑" if round(d, 6) > 0 else ("↓" if round(d, 6) < 0 else "=")  # noqa: E731
    return f"사람 확인 (민감도{arrow(ds)}·오검출{arrow(df)})"


def time_value(row, col):
    """처리 시간 한 칸: 자동 띠 전에 돈 결과에 없는 t_band 열은 0(그때는 띠 경계를 계산하지 않음), 그 밖에 없는 열은 NaN."""
    if col in row and not pd.isna(row[col]):
        return float(row[col])
    return 0.0 if col.startswith(OPTIONAL_TIMES) else float("nan")


def time_per_image(row):
    """영상 1장 처리 시간 ms(전처리 + 검출 + 노면 마스크 + 자동 띠 경계의 평균 합). 사진마다 한 번 잰 값이라 채택 표의 참고값으로만 쓴다."""
    return sum(time_value(row, c) for c in TIME_MEANS)


def compare_values(base, cur):
    """두 요약 행(전처리 끔)의 채택 비교 값: 민감도·영상당 오검출·균형 정확도·우연을 뺀 균열 재현율의 기준·이번·차이와
    규칙 1~3, 처리 시간(단일 실행 참고값)."""
    values = {}
    for key, col in COMPARE_KEYS:
        values.update({f"{key}_base": base[col], key: cur[col], f"{key}_diff": cur[col] - base[col]})
    values.update({"time_base": time_per_image(base), "time": time_per_image(cur)})
    values["time_diff"] = values["time"] - values["time_base"]
    values["rule1"] = rule1(values)
    values["rule2"] = rule_text(passes(values["balanced_diff"], lambda d: d > 0))
    values["rule3"] = rule_text(passes(values["crack_net_diff"], lambda d: d >= -RULE3_DROP))
    return values


def auto_verdict(row, base, adopted):
    """자동 판정(길잡이): (판정 글, 채택?). 규칙 2·3을 넘고 비교 기준이 E0이거나 채택된 단계일 때만 채택이다(채택된 것 위에만 쌓아,
    마스크를 채택하지 않았는데 마스크 + 띠 0.3이 채택되는 일이 없게). 기준과 네 값이 모두 같으면 '효과 없음'을 덧붙인다(구현 전이 아니라 결과)."""
    failed = [f"규칙 {n} {row[f'rule{n}']}" for n in (2, 3) if row[f"rule{n}"] != "통과"]
    if failed:
        text, ok = "되돌림: " + ", ".join(failed), False
    elif base not in adopted:
        text, ok = f"채택 안 함: 비교 기준({base})이 채택되지 않음", False
    else:
        text, ok = "잠정 채택 (규칙 1은 사람 확인)", True
    diffs = [row[f"{key}_diff"] for key, _ in COMPARE_KEYS]
    if not any(pd.isna(d) for d in diffs) and all(round(float(d), 6) == 0 for d in diffs):
        text += " (비교 기준과 같음: 효과 없음)"
    return text, ok


def adoption(results, reject=(), accept=()):
    """채택 표: ADOPT_ORDER 순서로 각 단계를 정해진 비교 기준(base_of)과 개발셋·전처리 끔으로 견준다(Plan.md 3.10, 검출 튜닝 조건).
    규칙 2(균형 정확도가 오름)·3(우연을 뺀 균열 재현율이 1%p 넘게 안 떨어짐)은 자동, 규칙 1은 rule1(사람 확인)이고 자동 판정은 auto_verdict.
    자동 판정은 길잡이이고 사람이 정한 것이 앞선다: accept(--accept)에 든 단계는 결과가 있으면 규칙·비교 기준과 상관없이 채택하고,
    reject(--reject)에 든 단계는 채택하지 않는다. 둘 다 판정 칸에 자동 판정을 함께 적는다. 강제 실행(--force)한 단계는 판정 끝에 표시한다.
    앞선 채택으로 정해지는 설정(E3 쪽 띠)과 실제로 돈 설정(opts)이 다르면 '다시 돌려야 함'으로 두고 판단에 쓰지 않는다.
    E3는 E2와의 차이(마스크만 다름)도 ref_ 열에 참고로 둔다(판단에는 쓰지 않음).
    results: {단계: {"off": 전처리 끔 요약 행, "status": 상태, "opts": 실제로 돈 옵션, "forced": 강제로 쓴 기능}} (없는 단계는 아직 안 돌림)
    → (표, 채택된 단계 집합(E0 포함))"""
    adopted, invalid, rows = {"E0"}, set(), []
    for sid in ADOPT_ORDER:
        exp, base = EXP[sid], base_of(sid, adopted)
        cur, expect = results.get(sid, {}), resolve(exp, adopted)
        row = {"stage_id": sid, "label": exp["label"], "status": cur.get("status", "아직 안 돌림"), "base": base,
               "changed": "-" if exp.get("todo") else changed_text(resolve(EXP[base], adopted), expect)}
        cur_off = cur.get("off")
        base_off = None if base in invalid else results.get(base, {}).get("off")
        if cur_off is not None and cur.get("opts") is not None and cur["opts"] != expect:
            invalid.add(sid)
            row["verdict"] = f"다시 돌려야 함: 지금 채택으로는 {changed_text(cur['opts'], expect)} (compare를 다시)"
        elif cur_off is None:
            row["verdict"] = row["status"] + (" (--accept는 결과가 있어야 반영)" if sid in accept else "")
        else:
            if base_off is None:
                auto, ok = f"비교 기준 {base}의 결과 없음", False
            else:
                row.update(compare_values(base_off, cur_off))
                auto, ok = auto_verdict(row, base, adopted)
            if sid in reject:
                row["verdict"] = f"되돌림: 사람이 뺌(--reject). 자동 판정: {auto}"
            elif sid in accept:
                row["verdict"] = f"채택: 사람이 정함(--accept). 자동 판정: {auto}"
                adopted.add(sid)
            else:
                row["verdict"] = auto
                if ok:
                    adopted.add(sid)
            e2 = None if "E2" in invalid else results.get("E2", {}).get("off")
            if sid == "E3" and e2 is not None:
                ref = compare_values(e2, cur_off)
                row.update({"ref_base": "E2", "ref_changed": changed_text(resolve(EXP["E2"], adopted), expect),
                            "ref_sens_diff": ref["sens_diff"], "ref_fp_diff": ref["fp_diff"],
                            "ref_balanced": ref["balanced_base"], "ref_balanced_diff": ref["balanced_diff"],
                            "ref_crack_net": ref["crack_net_base"], "ref_crack_net_diff": ref["crack_net_diff"],
                            "ref_rule2": ref["rule2"], "ref_rule3": ref["rule3"]})
        if cur.get("forced"):
            row["verdict"] += f" · {FORCED_MARK}"
        rows.append(row)
    return pd.DataFrame(rows, columns=ADOPT_COLUMNS), adopted


def grid_rows(summaries, states, adopted):
    """채택 표 위의 네 칸: 검출기(1차·dark) × 노면 마스크(끔·켬)와 E3σ 한 줄(개발셋, 전처리 끔). 자동 판정은 길잡이라
    네 조합을 한 표에 나란히 두어 사람이 바로 견주게 한다(예: E1은 떨어졌는데 E3가 E1보다 나은 경우, E1·E2가 둘 다 채택된 경우).
    E1b가 채택되면 마스크를 켠 칸은 띠 0.3이다(1차 칸은 E1b, dark 칸은 띠 0.3으로 돈 E3). '다시 돌려야 함'이거나 아직 안 돈 칸은
    상태만 적는다. 처리 시간은 단일 실행 참고값이다."""
    rows = []
    for detector, mask, sid in GRID_CELLS:
        if sid == "E1" and "E1b" in adopted:
            sid = "E1b"
        state = states[sid]
        off = (summaries.get(sid) or {}).get("off") if state["done"] else None
        row = {"detector": detector, "mask": mask, "band": resolve(EXP[sid], adopted).get("roi_top"), "stage_id": sid,
               "status": state["status"]}
        if off is not None:
            row.update({c: off[c] for c in GRID_METRICS})
            row["time"] = time_per_image(off)
        rows.append(row)
    return pd.DataFrame(rows, columns=GRID_COLUMNS)


def section_owners(path="src/config.py"):
    """config.py의 '# ===== 구역 =====' 줄을 따라 값 이름마다 구역 주인을 찾는다(구역마다 고칠 사람이 달라서)."""
    owner_text = {"공통": "공통 구역: 세 명 합의 뒤 C", "A": "A 구역: A에게 부탁", "B": "B 구역: B에게 부탁",
                  "C": "C 구역: C가 고치고 docs/part_C.md에 기록"}
    owners, section = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        head = re.match(r"#\s*=+\s*(공통|[ABC])\b", line)
        if head:
            section = head.group(1)
            continue
        names = re.match(r"([A-Z][A-Z0-9_]*(?:\s*,\s*[A-Z][A-Z0-9_]*)*)\s*=", line)
        if names and section:
            for name in names.group(1).split(","):
                owners[name.strip()] = owner_text[section]
    return owners


def config_changes(opts, config_path="src/config.py"):
    """잠정 채택 방법의 설정(opts)으로 가려면 바꿀 config 값 [(이름, 값, 구역 주인)]. config는 고치지 않고 목록만 낸다
    (구역마다 주인이 있어서: DETECT_MODE·DARK_SIGMA_IN_ROAD는 B, USE_ROAD_MASK는 C, ROI_TOP_DEFAULT는 세 명 합의)."""
    target = {"USE_ROAD_MASK": opts.get("mask") == "on", "DETECT_MODE": opts.get("detect"), "ROI_TOP_DEFAULT": opts.get("roi_top")}
    if opts.get("detect") == "dark" and opts.get("mask") == "on":
        target["DARK_SIGMA_IN_ROAD"] = opts.get("sigma_in_road")
    if opts.get("params"):
        target.update(param_values(opts["params"]))
    owners = section_owners(config_path)
    return [(name, value, owners.get(name, "구역 모름")) for name, value in target.items() if getattr(C, name, None) != value]


def e0_reproduction(summary):
    """E0 재현 확인(Plan.md 3.2 (5)): 804장·전처리 끔의 정답률·찾은 정답(중심·IoU)·후보 수를 기준값과 견준다.
    → [(항목, 기준, 실제, 같음?)]. 결과가 없거나 사진이 804장이 아니면 []."""
    off = summary.get("off") if summary else None
    if off is None or int(off["images"]) != N_TRAIN:
        return []
    actual = {"정답률 %": round(float(off["accuracy"]) * 100, 1),
              "찾은 정답(중심)": int(off["crack_ctr_found"] + off["pothole_ctr_found"]),
              "찾은 정답(IoU)": int(off["crack_iou_found"] + off["pothole_iou_found"]),
              "후보 수": int(round(float(off["pred_per_image"]) * int(off["images"])))}
    return [(key, expect, actual[key], actual[key] == expect) for key, expect in E0_EXPECT.items()]


REPRO_PAIRS = (("T", "R"), ("TA", "RA"), ("TR", "RR"))     # 재현 확인: (한 번만 잰 시험셋, 같은 코드로 다시 돈 것)


def repro_pair(once, again):
    """두 시험셋 결과 폴더가 같은지: 요약과 영상별 결과를 처리 시간 열(t_로 시작)을 빼고 견준다(같은 코드면 같아야 함).
    다시 돈 쪽에만 새 열(예: 나중에 생긴 t_band_ms)이 있으면 그 열은 빼고 견주고 그 사실을 적는다. 둘 중 하나가 없으면 빈 list."""
    lines = []
    for file in (et.SUMMARY_NAME, et.DETAIL_NAME):
        a, b = read_csv(once / file), read_csv(again / file)
        if a is None or b is None:
            return []
        extra = [c for c in b.columns if c not in a.columns]
        keep = [c for c in a.columns if not c.startswith("t_")]
        if [c for c in b.columns if c in a.columns] != list(a.columns) or len(a) != len(b):
            lines.append(f"{file}: 다름 (열 또는 행 수가 다름: {len(a)}행 / {len(b)}행)")
            continue
        diff = [c for c in keep if not a[c].equals(b[c])]
        note = f", 새 열 {', '.join(extra)}는 빼고" if extra else ""
        lines.append(f"{file}: " + (f"같음 (처리 시간 열{note} 빼고 {len(a)}행)" if not diff else f"다름: {', '.join(diff[:10])}"))
    return lines


def repro_check(paths):
    """재현 확인: final_test가 test_once와, final_test_autoband가 test_autoband와, final_test_ratio가 test_ratio와 같은지(repro_pair).
    → 설명 줄 list ('폴더 vs 폴더 파일: 같음'). 결과가 없는 쌍은 빼고, 하나도 없으면 빈 list"""
    results, lines = Path(paths["results"]), []
    for once_id, again_id in REPRO_PAIRS:
        once, again = EXP[once_id]["name"], EXP[again_id]["name"]
        lines += [f"{again} vs {once} {line}" for line in repro_pair(results / once, results / again)]
    return lines


# ---------------------------------------------------------
# 보고서 표
# ---------------------------------------------------------

def stages_table(paths, states):
    """1번 단계별 표: E0~E5와 시험셋(1차·최종)의 요약을 끔·켬 두 줄로. 끝나지 않은 단계는 상태만 한 줄."""
    rows = []
    for sid in STAGE_IDS:
        exp, state = EXP[sid], states[sid]
        head = dict(zip(STAGE_HEAD, (sid, exp["label"], DATA_NAMES[exp["split"]], exp["name"], state["status"], state["commit"])))
        summary = train_summary(paths, exp, state)
        if not summary:
            rows.append({**head, "preprocess": ""})
            continue
        rows += [{**head, "preprocess": mode, **{m: summary[mode].get(m) for m in METRICS}} for mode in ("off", "on") if mode in summary]
    return pd.DataFrame(rows, columns=STAGE_HEAD + ["preprocess"] + METRICS)


def country_table(paths, states):
    """2번 국가별 표: 개발셋(E0 1차·E5 최종)과 시험셋(1차·최종)의 국가별 요약(per_source.csv)을 끔·켬으로."""
    rows = []
    for sid, method in COUNTRY_RUNS:
        exp = EXP[sid]
        df = read_csv(Path(paths["results"]) / exp["name"] / et.SOURCE_NAME) if states[sid]["done"] else None
        for _, r in (df.iterrows() if df is not None else []):
            rows.append({"data": DATA_NAMES[exp["split"]], "method": method, "folder": exp["name"], "preprocess": r["preprocess"],
                         "source": r["source"], **{m: r.get(m) for m in METRICS}})
    return pd.DataFrame(rows, columns=["data", "method", "folder", "preprocess", "source"] + METRICS)


def run_labels(states):
    """39장 실행의 표 이름 {단계: 이름}. --force로 돈 실행은 이름 뒤에 '[강제 실행(시험 미통과)]'을 붙인다."""
    return {sid: method + (f" [{FORCED_MARK}]" if states[sid]["forced"] else "") for sid, method in RUNS39}


def condition_table(runs, labels):
    """3번 조건별 표: 39장 실행마다 조건(사람이 붙인 태그: 직접 촬영은 이름, 제공 13장은 roi.csv)별 영상 단위 지표와
    상태 판정(analyze)이 고른 전처리를 나란히(Plan.md 3.7 3번). 태그가 blur·dark·bright·noise면 맞는 전처리를 고른 수도 센다."""
    rows = []
    for sid, _ in RUNS39:
        data = runs[sid]
        if data is None:
            continue
        judge, status = judgements(data["metrics"]), gt_status(data["metrics"])
        for _, r in data["condition"].iterrows():
            cond = r["condition"]
            picked = [j for j in judge.values() if cond == "all" or j["condition"] == cond]
            judged = [j for j in picked if j["hit"] is not None]
            chosen = collections.Counter(j["steps"] for j in picked).most_common()
            rows.append({"method": labels[sid], "folder": EXP[sid]["name"], "condition": cond, "gt_status": status,
                         **{c: r.get(c) for c in GROUP39},
                         "steps_chosen": "; ".join(f"{s} {n}장" for s, n in chosen),
                         "judge_expect": judged[0]["expect"] if judged and cond != "all" else "",
                         "judge_hit": f"{sum(j['hit'] for j in judged)}/{len(judged)}" if judged else ""})
    return pd.DataFrame(rows, columns=["method", "folder", "condition", "gt_status"] + GROUP39
                        + ["steps_chosen", "judge_expect", "judge_hit"])


def preprocess_table(paths, states, runs):
    """4번 전처리 끔·켬 표: 영상 1장 평균의 에지 수·후보 수·면적·처리 시간(1차 계획의 항목). 39장 최종(roi.csv·자동 ROI)과 개발·시험셋 최종.
    처리 시간은 최종 단계 39장(TIME_REPEAT번 중앙값)의 값만 쓰고, 맨 끝에 처리 시간 합(전처리 + 검출 + 노면 마스크 + 자동 띠 경계)을 둔다.
    자동 띠는 기본 띠를 쓰는 사진에만 쓰므로 roi.csv 줄의 띠 경계 시간은 0이다. 학습 데이터의 시간은 한 번 잰 값이라 넣지 않는다."""
    rows = []
    for sid, name in (("F39", "39장 최종(roi.csv)"), ("F39a", "39장 최종(자동 ROI, 자동 띠 + 결 + ratio)")):
        data = runs[sid]
        if data is None:
            continue
        whole = data["source"][data["source"]["source"] == "all"].iloc[0]
        value = {mode: {m: time_value(whole, f"{m}_{mode}") if m in TIME39 else whole[f"{m}_{mode}"] for _, m in PRE39}
                 for mode in ("off", "on")}
        rows += [{"data": name, "folder": EXP[sid]["name"], "item": item, "off": value["off"][m], "on": value["on"][m],
                  "diff": value["on"][m] - value["off"][m]} for item, m in PRE39]
        total = {mode: sum(value[mode][m] for m in TIME39) for mode in ("off", "on")}
        rows.append({"data": name, "folder": EXP[sid]["name"], "item": "처리 시간 합 ms (전처리+검출+노면+띠 경계)",
                     "off": total["off"], "on": total["on"], "diff": total["on"] - total["off"]})
    for sid in ("E5", "TR"):
        summary = train_summary(paths, EXP[sid], states[sid])
        if summary and {"off", "on"} <= set(summary):
            rows += [{"data": f"{DATA_NAMES[EXP[sid]['split']]} 최종", "folder": EXP[sid]["name"], "item": item,
                      "off": summary["off"][c], "on": summary["on"][c], "diff": summary["on"][c] - summary["off"][c]}
                     for item, c in PRE_TRAIN]
    return pd.DataFrame(rows, columns=["data", "folder", "item", "off", "on", "diff"])


def matching_table(paths, states, runs):
    """5번 특징점·매칭 전·후 표: 해리스 코너 수(39장·개발셋 영상 평균)와 쌍 사진마다 SIFT 특징점 수·good 매칭 점 수·inlier 수,
    맨 끝에 쌍 평균. inlier 비율은 천장(0.97 이상)이라 점의 수를 중심으로 본다(Plan.md 1.4·3.7 5번)."""
    rows = []
    if runs["F39"] is not None:
        whole = runs["F39"]["source"][runs["F39"]["source"]["source"] == "all"].iloc[0]
        rows.append({"item": "해리스 코너 수", "target": "39장 영상 평균", "off": whole["harris_off"], "on": whole["harris_on"],
                     "diff": whole["harris_diff"]})
    summary = train_summary(paths, EXP["E5"], states["E5"])
    if summary and {"off", "on"} <= set(summary):
        off, on = summary["off"]["harris_mean"], summary["on"]["harris_mean"]
        rows.append({"item": "해리스 코너 수", "target": "개발셋 영상 평균", "off": off, "on": on, "diff": on - off})
    match = read_csv(Path(paths["results"]) / EXP["FM"]["name"] / "matching.csv") if states["FM"]["done"] else None
    if match is not None and len(match):
        for (pair, target), g in match.groupby(["pair", "target"], sort=True):
            off, on = g[g["preprocess"] == "off"].iloc[0], g[g["preprocess"] == "on"].iloc[0]
            rows += [{"item": item, "target": f"쌍 {pair} {target} ({off['condition']})", "off": off[c], "on": on[c],
                      "diff": on[c] - off[c]} for item, c in MATCH_ITEMS]
        mean = {mode: match[match["preprocess"] == mode] for mode in ("off", "on")}
        rows += [{"item": item, "target": "쌍 평균", "off": mean["off"][c].mean(), "on": mean["on"][c].mean(),
                  "diff": mean["on"][c].mean() - mean["off"][c].mean()} for item, c in MATCH_ITEMS]
    return pd.DataFrame(rows, columns=["item", "target", "off", "on", "diff"])


def source_table(runs, labels):
    """6번 제공 vs 직접 촬영 표: 39장 실행마다 출처(제공·직접 촬영·전체)별 영상 단위 지표와 영상당 후보·에지 수(끔·켬).
    E1·E3 자동 ROI 줄은 비교 단계의 참고 실행이다(채택 판단에 쓰지 않음)."""
    rows = []
    for sid, _ in RUNS39:
        data = runs[sid]
        if data is None:
            continue
        status = gt_status(data["metrics"])
        rows += [{"method": labels[sid], "folder": EXP[sid]["name"], "source": r["source"], "gt_status": status,
                  **{c: r.get(c) for c in SOURCE39}} for _, r in data["source"].iterrows()]
    return pd.DataFrame(rows, columns=["method", "folder", "source", "gt_status"] + SOURCE39)


def degrade_table(paths, states):
    """7번 인위 저하 표: 회차(1회 비교·2회 최종) × 저하 × 끔·켬의 검출 지표, 에지·해리스·후보 수·면적, 상태 판정 적중률과 원본 기준.
    1회는 E1~E3 가운데 마지막 잠정 채택 설정으로 돈다(settings_from). 처리 시간은 최종 단계(2회) 값만 넣는다(한 번 잰 값의 평균, 참고)."""
    rows = []
    for round_name, prefix in DEGRADE_ROUNDS:
        for kind in et.DEGRADE_KINDS:
            exp, state = EXP[prefix + kind], states[prefix + kind]
            head = {"round": round_name, "settings_from": state["source"] if prefix == "E6_" else "config", "degrade": kind,
                    "folder": exp["name"], "status": state["status"]}
            summary = train_summary(paths, exp, state)
            if not summary:
                rows.append({**head, "preprocess": ""})
                continue
            for mode in ("off", "on"):
                times = {c: summary[mode].get(c) for c in TIME_MEANS} if prefix == "F6_" else {}
                rows.append({**head, "preprocess": mode, **{c: summary[mode].get(c) for c in DEGRADE_COLS}, **times})
    return pd.DataFrame(rows, columns=["round", "settings_from", "degrade", "folder", "status", "preprocess"] + DEGRADE_COLS + TIME_MEANS)


def band_rows(detail, gt_dir, gt_cache):
    """영상별 결과(전처리 끔 행)마다 띠가 기본 띠(0.5)보다 넓어졌는지와, 정답 상자 가운데 중심이 띠 안에 든 수·전체 수를 더한 새 표.
    띠 위쪽 y0 = 640px 높이 − 띠 행 수(roi_pixels / 가로 640). 정답은 gt_dir/<사진 이름>.csv(evaluate.load_gt)이고 gt_cache에 모아 둔다."""
    out = detail.copy()
    rows = pd.to_numeric(out["roi_pixels"]) / C.TARGET_WIDTH
    out["y0"] = pd.to_numeric(out["height"]) - rows
    out["widened"] = out["y0"] < (pd.to_numeric(out["height"]) * FIRST_ROI_TOP).astype(int)
    gin, gn = [], []
    for image, y0 in zip(out["image"], out["y0"]):
        if image not in gt_cache:
            gt_cache[image] = et.load_gt(Path(gt_dir) / f"{Path(image).stem}.csv") or []
        gn.append(len(gt_cache[image]))
        gin.append(sum(1 for x, y, w, h, _ in gt_cache[image] if y + h / 2 >= y0))
    out["gt_in"], out["gt_n"] = gin, gn
    return out


def band_dev_tables(paths, states, summaries):
    """9번 자동 띠 비교(개발셋, 전처리 끔 = 채택 규칙의 조건): 띠 고정(E3) → 자동 띠(B1, 결 없음) → 자동 띠 + 결(M0). 모두 남기기 center.
    전체 표: 민감도·특이도·균형 정확도·영상당 오검출·우연 뺀 균열 재현율, 띠 안 정답 비율(중심), 넓어진 사진 비율, 처리 시간 합(참고)과
    그중 띠 경계 시간, E3와 견준 채택 규칙 1~3. 국가별 표: 넓어진 사진 비율, 균형 정확도(무손상 사진이 없는 국가는 민감도도), 띠 안 정답 비율.
    → (전체 표, 국가별 표). 결과가 없는 단계는 상태만."""
    results, gt_cache, rows, country = Path(paths["results"]), {}, [], []
    base = (summaries.get("E3") or {}).get("off")
    for sid, name in BAND_DEV:
        state, summary = states[sid], summaries.get(sid)
        detail = read_csv(results / EXP[sid]["name"] / et.DETAIL_NAME) if state["done"] else None
        head = {"run": name, "stage_id": sid, "folder": EXP[sid]["name"], "status": state["status"]}
        if not summary or "off" not in summary or detail is None:
            rows.append(head)
            continue
        off = summary["off"]
        band = band_rows(detail[detail["preprocess"] == "off"], paths["train_gt"], gt_cache)
        row = {**head, **{m: off[m] for m in GRID_METRICS}, "gt_in_band": band["gt_in"].sum() / max(1, band["gt_n"].sum()),
               "widened": band["widened"].mean(), "time": time_per_image(off), "t_band": time_value(off, "t_band_ms_mean")}
        if sid != "E3" and base is not None:
            v = compare_values(base, off)
            row.update({"rule1": f"민감도 {pp(v['sens_diff'])}·오검출 {v['fp_diff']:+.2f}", "rule2": v["rule2"], "rule3": v["rule3"]})
        rows.append(row)
        per = read_csv(results / EXP[sid]["name"] / et.SOURCE_NAME)
        per = per[per["preprocess"] == "off"].set_index("source") if per is not None else None
        for source, part in band.groupby("source", sort=True):
            p = per.loc[source] if per is not None and source in per.index else {}
            country.append({"run": name, "stage_id": sid, "source": source, "images": len(part), "widened": part["widened"].mean(),
                            "balanced": p.get("balanced"), "sensitivity": p.get("sensitivity"),
                            "gt_in_band": part["gt_in"].sum() / max(1, part["gt_n"].sum())})
    columns = ["run", "stage_id", "folder", "status"] + GRID_METRICS + ["gt_in_band", "widened", "time", "t_band", "rule1", "rule2", "rule3"]
    return (pd.DataFrame(rows, columns=columns),
            pd.DataFrame(country, columns=["run", "stage_id", "source", "images", "widened", "balanced", "sensitivity", "gt_in_band"]))


def band_39_table(runs, states):
    """9번 자동 띠 비교(39장 자동 ROI): 띠 고정(E3_39) → 자동 띠(B1_39) → 자동 띠 + 결(M0_39)의 전처리 끔·켬 정답률·기준선·
    민감도·특이도, 띠 고정보다 넓어진 사진 수, 처리 시간 합(끔·켬 각 TIME_REPEAT번 중앙값의 평균)과 그중 띠 경계 시간."""
    rows = []
    fixed = runs["E3_39"]["image"].set_index("file") if runs["E3_39"] is not None else None
    for sid, name in BAND_39:
        data = runs[sid]
        if data is None:
            rows.append({"run": name, "stage_id": sid, "status": states[sid]["status"]})
            continue
        whole = data["source"][data["source"]["source"] == "all"].iloc[0]
        image = data["image"].set_index("file")
        widened = int((image["roi_pixels"] > fixed["roi_pixels"].reindex(image.index)).sum()) if fixed is not None else None
        for mode in ("off", "on"):
            rows.append({"run": name, "stage_id": sid, "status": states[sid]["status"], "preprocess": mode,
                         "accuracy": whole[f"accuracy_{mode}"], "baseline_none": whole["baseline_none"],
                         "sensitivity": whole[f"sensitivity_{mode}"], "specificity": whole[f"specificity_{mode}"],
                         "widened": widened, "n_images": int(whole["n_images"]),
                         "time": sum(time_value(whole, f"{m}_{mode}") for m in TIME39), "t_band": time_value(whole, f"t_band_ms_{mode}")})
    return pd.DataFrame(rows, columns=["run", "stage_id", "status", "preprocess", "accuracy", "baseline_none", "sensitivity",
                                       "specificity", "widened", "n_images", "time", "t_band"])


def config_from_used(path):
    """도구가 남긴 config_used*.txt의 값을 {이름: 파이썬 값}으로(config_used_values의 글을 literal_eval). 읽을 수 없는 값은 뺀다."""
    out = {}
    for name, text in (config_used_values(path) or {}).items():
        try:
            out[name] = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            pass
    return out


def mask_gt_rows(paths, exp, state, out_dir):
    """개발셋 실행 하나의 사진마다 띠·노면 마스크 안 정답 수. 그 실행의 config(config_used_train.txt)의 띠·노면 마스크 값(ROAD_·AUTO_BAND_·ROI_)으로
    pipeline과 같게 띠 위쪽 경계(자동 띠)와 띠 안 노면 마스크를 다시 만들어 센다. 마스크 안 정답 = 띠 안 정답 중 상자 중심이 노면 안(handoff_C M4).
    검출은 하지 않는다(남기기 규칙은 마스크를 바꾸지 않음). 결과는 out_dir/mask_gt_<폴더>.csv에 두고, 실행 결과가 더 새로우면 다시 만든다.
    → DataFrame(image, source, top, widened, gt_n, gt_in_band, gt_in_mask). 끝나지 않았거나 사진·config가 없으면 None."""
    folder = Path(paths["results"]) / exp["name"]
    cache = Path(out_dir) / f"mask_gt_{exp['name']}.csv"
    summary = folder / et.SUMMARY_NAME
    if not state["done"] or not summary.exists():
        return None
    if cache.exists() and cache.stat().st_mtime >= summary.stat().st_mtime:
        return read_csv(cache)
    used = {k: v for k, v in config_from_used(folder / CONFIG_FILE["train"]).items()
            if k.startswith(("ROAD_", "AUTO_BAND_", "ROI_")) or k == "TARGET_WIDTH"}
    used = {**SWITCHES_BEFORE, **used} if used else used     # 기록에 없는 스위치는 그 기능이 생기기 전에 돈 것(꺼짐)
    image_dir = Path(paths["train_img"])
    names = sorted(p.name for p in image_dir.glob("*.jpg"))
    if not used or not names:
        return None
    split = et.assign_split(names)
    rows = []
    with patched(C, used):
        for name in (n for n in names if split[n] == exp["split"]):
            small = io_utils.resize_width(io_utils.load_image(str(image_dir / name)))
            top = pipeline.auto_band_top(small, C.ROI_TOP_DEFAULT) if C.AUTO_BAND_TOP else C.ROI_TOP_DEFAULT
            roi, y0 = io_utils.crop_roi(small, top, C.ROI_BOTTOM_DEFAULT)
            road = io_utils.road_mask(roi)
            gt = evaluate.load_gt(str(Path(paths["train_gt"]) / f"{Path(name).stem}.csv")) or []
            band = [(int(x + w / 2), int(y + h / 2) - y0) for x, y, w, h, _ in gt if 0 <= int(y + h / 2) - y0 < road.shape[0]]
            rows.append({"image": name, "source": et.source_of(name), "top": top,
                         "widened": y0 < int(small.shape[0] * C.ROI_TOP_DEFAULT), "gt_n": len(gt), "gt_in_band": len(band),
                         "gt_in_mask": sum(1 for cx, cy in band if road[cy, min(cx, road.shape[1] - 1)] > 0)})
    df = pd.DataFrame(rows, columns=["image", "source", "top", "widened", "gt_n", "gt_in_band", "gt_in_mask"])
    df.to_csv(cache, encoding="utf-8-sig", index=False)
    return df


def mask_gt_ratios(df):
    """mask_gt_rows의 표 → (띠 안 정답 비율 = 띠 안 / 전체, 마스크 안 정답 비율 = 마스크 안 / 띠 안, 넓어진 사진 비율). 없으면 NaN 셋."""
    if df is None or not len(df):
        return math.nan, math.nan, math.nan
    return (df["gt_in_band"].sum() / max(1, df["gt_n"].sum()), df["gt_in_mask"].sum() / max(1, df["gt_in_band"].sum()),
            float(df["widened"].astype(bool).mean()))


def mask2_table(summaries, states, mask_gt):
    """10번 노면 마스크 보완 채택 표(개발셋, 전처리 끔): M0(자동 띠 + 결, 남기기 center) → + 남기기 ratio(M1) → + 차선 (가)(M2)·(나)(M3).
    영상 단위 지표와 반올림 전 균형 정확도 차이, 마스크 안·띠 안 정답, 넓어진 사진, 영상당 후보, 노면 마스크·띠 경계 시간(사진마다 한 번 잰 평균),
    M0와 견준 규칙 1~3, 자동 판정(규칙 2·3)과 사람이 정한 판정(MASK2_DECISIONS). M1은 마스크가 M0와 같아 M0의 정답 비율을 쓴다.
    전처리 켬 균형 정확도도 참고로 둔다."""
    rows, base = [], (summaries.get("M0") or {}).get("off")
    for sid in MASK2_IDS:
        summary = summaries.get(sid)
        head = {"stage_id": sid, "label": EXP[sid]["label"], "folder": EXP[sid]["name"], "status": states[sid]["status"]}
        if not summary or "off" not in summary:
            rows.append(head)
            continue
        off = summary["off"]
        band, mask, widened = mask_gt_ratios(mask_gt.get("M0" if sid == "M1" else sid))
        row = {**head, **{m: off[m] for m in GRID_METRICS}, "balanced_on": summary["on"]["balanced"] if "on" in summary else math.nan,
               "gt_in_mask": mask, "gt_in_band": band, "widened": widened, "pred_per_image": off["pred_per_image"],
               "t_road": time_value(off, "t_road_ms_mean"), "t_band": time_value(off, "t_band_ms_mean"),
               "decision": MASK2_DECISIONS.get(sid, "비교 기준")}
        if sid != "M0" and base is not None:
            v = compare_values(base, off)
            row.update({"rule1": f"민감도 {pp(v['sens_diff'])}·오검출 {v['fp_diff']:+.2f}", "rule2": v["rule2"], "rule3": v["rule3"],
                        "balanced_diff": v["balanced_diff"],
                        "auto": "규칙 2·3 통과" if v["rule2"] == "통과" and v["rule3"] == "통과" else
                        "되돌림: " + ", ".join(f"규칙 {n} {v[f'rule{n}']}" for n in (2, 3) if v[f"rule{n}"] != "통과")})
        rows.append(row)
    columns = (["stage_id", "label", "folder", "status"] + GRID_METRICS + ["balanced_diff", "balanced_on", "gt_in_mask", "gt_in_band",
               "widened", "pred_per_image", "t_road", "t_band", "rule1", "rule2", "rule3", "auto", "decision"])
    return pd.DataFrame(rows, columns=columns)


def tau_points_of(paths, exp, state):
    """evaluate_train --tau-curve의 동작점 표(tau_points.csv)를 {(전처리, 점 이름): 행}으로. 없으면 None."""
    df = read_csv(Path(paths["results"]) / exp["name"] / et.TAU_POINTS_NAME) if state["done"] else None
    return None if df is None else {(row["preprocess"], row["point"]): row for _, row in df.iterrows()}


def tau_images_of(paths, exp, state):
    """evaluate_train --tau-curve의 영상별 판정(tau_images.csv)을 {(전처리, 점): {사진: 맞힘}}으로. 없으면 None."""
    df = read_csv(Path(paths["results"]) / exp["name"] / et.TAU_IMAGES_NAME) if state["done"] else None
    if df is None:
        return None
    return {key: dict(zip(part["image"], part["correct"].astype(str) == "True")) for key, part in df.groupby(["preprocess", "point"])}


def rule1_tau(base, cur):
    """채택 규칙 1(같은 동작점, 결정 10/12 밤): 영상당 오검출 1·2개 동작점(fp1·fp2)의 민감도가 둘 다 떨어지지 않고 하나 이상 오르면 통과.
    소수 여섯째 자리에서 반올림해 견준다(passes와 같음). 한쪽이라도 값이 없으면(그 동작점이 없음) None. base·cur는 {"fp1": 민감도, "fp2": 민감도}."""
    diffs = [cur[p] - base[p] for p in ("fp1", "fp2")]
    if any(pd.isna(d) for d in diffs):
        return None
    diffs = [round(float(d), 6) for d in diffs]
    return all(d >= 0 for d in diffs) and any(d > 0 for d in diffs)


def supp_table(paths, states):
    """14번 보완 비교 채택 표(개발셋): 실행마다 판정 행(끔·켬)의 τ 없음·fp1·fp2·rule1 점 지표와, 비교 기준과 견준 규칙 1~3.
    규칙 1 = rule1_tau(fp1·fp2 민감도), 규칙 2·3 = 각 실행 자신의 rule1 점(비교 기준도 자기 rule1 점)의 균형 정확도·우연 뺀 균열 재현율.
    E4처럼 judge가 "tau"면 비교 기준의 τ 없음 점과 이번 rule1 점을 견주고 규칙 1은 해당 없음. "curve"는 값만 적고 판정하지 않는다."""
    rows = []
    for sid in SUPP_IDS:
        exp, state = EXP[sid], states[sid]
        mode, judge = exp["judge_row"], exp["judge"]
        head = {"stage_id": sid, "label": exp["label"], "folder": exp["name"], "status": state["status"], "row": mode,
                "base": exp["supp_base"] or "-"}
        points = tau_points_of(paths, exp, state)
        if points is None:
            rows.append(head)
            continue
        get = lambda pts, name, col: pts[(mode, name)].get(col, math.nan) if (mode, name) in pts else math.nan  # noqa: E731
        row = {**head, "none_sens": get(points, "none", "sensitivity"), "none_spec": get(points, "none", "specificity"),
               "none_balanced": get(points, "none", "balanced"), "none_fp": get(points, "none", "fp_per_image"),
               "fp1_sens": get(points, "fp1", "sensitivity"), "fp1_spec": get(points, "fp1", "specificity"),
               "fp2_sens": get(points, "fp2", "sensitivity"), "fp2_spec": get(points, "fp2", "specificity"),
               **{f"rule1_{c}": get(points, "rule1", c) for c in ("tau_crack", "tau_pothole", "sensitivity", "specificity",
                                                                   "balanced", "fp_per_image", "crack_ctr_recall_net")},
               "rule1_accuracy": get(points, "rule1", "accuracy"),
               "decision": SUPP_DECISIONS.get(sid, "비교 기준" if judge == "base" else "")}
        base_exp = EXP.get(exp["supp_base"]) if exp["supp_base"] else None
        base_pts = tau_points_of(paths, base_exp, states[base_exp["id"]]) if base_exp else None
        if base_pts is not None and judge in ("tau", "rules"):
            ref = "none" if judge == "tau" else "rule1"
            d_bal = row["rule1_balanced"] - get(base_pts, ref, "balanced")
            d_net = row["rule1_crack_ctr_recall_net"] - get(base_pts, ref, "crack_ctr_recall_net")
            r1 = None if judge == "tau" else rule1_tau({p: get(base_pts, p, "sensitivity") for p in ("fp1", "fp2")},
                                                       {p: get(points, p, "sensitivity") for p in ("fp1", "fp2")})
            r2, r3 = passes(d_bal, lambda d: d > 0), passes(d_net, lambda d: d >= -RULE3_DROP)
            mine, theirs = tau_images_of(paths, exp, state), tau_images_of(paths, base_exp, states[base_exp["id"]])
            if mine and theirs and (mode, "rule1") in mine and (mode, ref) in theirs:
                a, b = theirs[(mode, ref)], mine[(mode, "rule1")]
                changed = sorted(n for n in a if n in b and a[n] != b[n])
                row.update({"changed_images": len(changed), "changed_list": ", ".join(changed[:8]) + (" …" if len(changed) > 8 else "")})
            row.update({"rule1": "해당 없음(같은 후보에서 τ만 고름)" if judge == "tau" else rule_text(r1),
                        "rule2": rule_text(r2), "rule3": rule_text(r3), "balanced_diff": d_bal, "crack_net_diff": d_net,
                        "auto": "규칙 통과" if r2 and r3 and (judge == "tau" or r1) else "되돌림"})
        rows.append(row)
    columns = (["stage_id", "label", "folder", "status", "row", "base", "none_sens", "none_spec", "none_balanced", "none_fp",
                "fp1_sens", "fp1_spec", "fp2_sens", "fp2_spec"]
               + [f"rule1_{c}" for c in ("tau_crack", "tau_pothole", "sensitivity", "specificity", "balanced", "fp_per_image",
                                          "crack_ctr_recall_net")]
               + ["rule1_accuracy", "changed_images", "changed_list", "balanced_diff", "crack_net_diff", "rule1", "rule2", "rule3",
                  "auto", "decision"])
    return pd.DataFrame(rows, columns=columns)


def supp_points_table(paths, states):
    """15번 보완 실행의 동작점 표: 실행마다 끔·켬 × (none, fp1, fp2, rule1, config) 행을 tau_points.csv 그대로 모은다."""
    frames = []
    for sid in SUPP_IDS:
        exp = EXP[sid]
        df = read_csv(Path(paths["results"]) / exp["name"] / et.TAU_POINTS_NAME) if states[sid]["done"] else None
        if df is not None:
            frames.append(df.assign(stage_id=sid, label=exp["label"], folder=exp["name"]))
    if not frames:
        return pd.DataFrame(columns=["stage_id", "label", "folder", "preprocess", "point"])
    df = pd.concat(frames, ignore_index=True)
    return df[["stage_id", "label", "folder"] + [c for c in df.columns if c not in ("stage_id", "label", "folder")]]


def keep_rule_changes(paths, states):
    """남기기 규칙 center(M0) → ratio(M1)로 영상 판정이 바뀐 사진(개발셋, 끔·켬)과, 그 사진에서 한쪽에만 남은 후보의 노면 안 비율.
    두 실행의 영상별 결과(correct)를 견주고, 바뀐 사진만 지금 config에서 남기기 규칙만 바꿔 다시 검출한다(사진 몇 장이라 짧다).
    → DataFrame(preprocess, image, has_damage, correct_center, correct_ratio, kind, kept_by, bbox(640px), area, center_on_road, road_ratio)"""
    columns = ["preprocess", "image", "has_damage", "correct_center", "correct_ratio", "kind", "kept_by", "bbox", "area",
               "center_on_road", "road_ratio"]
    if not (states["M0"]["done"] and states["M1"]["done"]):
        return pd.DataFrame(columns=columns)
    a = read_csv(Path(paths["results"]) / EXP["M0"]["name"] / et.DETAIL_NAME)
    b = read_csv(Path(paths["results"]) / EXP["M1"]["name"] / et.DETAIL_NAME)
    rows = []
    for mode, use_pre in (("off", False), ("on", True)):
        x, y = a[a["preprocess"] == mode].set_index("image"), b[b["preprocess"] == mode].set_index("image")
        for image in x.index[x["correct"] != y["correct"].reindex(x.index)]:
            path = Path(paths["train_img"]) / image
            if not path.exists():
                continue
            bgr, found = io_utils.load_image(str(path)), {}
            for rule in ("center", "ratio"):
                with patched(C, {"ROAD_KEEP_RULE": rule}):
                    out = pipeline.run_pipeline(bgr, C.ROI_TOP_DEFAULT, C.ROI_BOTTOM_DEFAULT, use_pre)
                found[rule] = {(kind, r["bbox"]): r for kind in ("cracks", "potholes") for r in out[kind]}
            road, y0 = out["road"], out["y0"]
            for key in sorted(set(found["center"]) ^ set(found["ratio"])):
                rule = "center" if key in found["center"] else "ratio"
                r = found[rule][key]
                bx, by, bw, bh = r["bbox"]
                rows.append({"preprocess": mode, "image": image, "has_damage": bool(x.loc[image, "has_damage"]),
                             "correct_center": bool(x.loc[image, "correct"]), "correct_ratio": bool(y.loc[image, "correct"]),
                             "kind": "균열" if key[0] == "cracks" else "포트홀", "kept_by": f"{rule}만 남김",
                             "bbox": (bx, by + y0, bw, bh), "area": r["area"],
                             "center_on_road": bool(road[min(int(by + bh / 2), road.shape[0] - 1),
                                                         min(int(bx + bw / 2), road.shape[1] - 1)] > 0),
                             "road_ratio": pipeline.road_ratio(r, road)})
    return pd.DataFrame(rows, columns=columns)


def test_table(summaries, states):
    """11번 시험셋 세 번(TEST_RUNS: 띠 고정 test_once → 자동 띠 test_autoband → 자동 띠 + ratio test_ratio)과 같은 시험셋의 1차 방법(T0)을
    폴더·잰 시각(run_info의 시작)·설정과 함께 끔·켬 두 줄로. 결과가 없는 실행은 상태만."""
    rows = []
    for sid, name, setting in (("T0", "1차 방법", "1차(legacy, 마스크 없음, 띠 0.5)"),) + TEST_RUNS:
        head = {"stage_id": sid, "name": name, "folder": EXP[sid]["name"], "started": states[sid].get("started", ""),
                "settings": setting, "status": states[sid]["status"]}
        summary = summaries.get(sid)
        if not summary:
            rows.append(head)
            continue
        rows += [{**head, "preprocess": mode, **{m: summary[mode].get(m) for m in METRICS}} for mode in ("off", "on") if mode in summary]
    return pd.DataFrame(rows, columns=["stage_id", "name", "folder", "started", "settings", "status", "preprocess"] + METRICS)


def time_table(runs):
    """12번 처리 시간 비교: 최종 실행의 39장 기록에서 1차 설정(F39e0)과 최종 설정(roi.csv F39, 자동 ROI F39a)의 영상 1장 처리 시간
    (끔·켬 각 TIME_REPEAT번 중앙값의 39장 평균). 항목은 전처리·검출·노면 마스크·띠 경계와 그 합."""
    rows = []
    for sid, name in (("F39e0", "1차 설정(roi.csv)"), ("F39", "최종(roi.csv)"), ("F39a", "최종(자동 ROI, 자동 띠 + 결 + ratio)")):
        data = runs.get(sid)
        if data is None:
            continue
        whole = data["source"][data["source"]["source"] == "all"].iloc[0]
        for mode in ("off", "on"):
            value = {m: time_value(whole, f"{m}_{mode}") for m in TIME39}
            rows.append({"run": name, "stage_id": sid, "folder": EXP[sid]["name"], "preprocess": mode, **value,
                         "total": sum(value.values())})
    return pd.DataFrame(rows, columns=["run", "stage_id", "folder", "preprocess"] + list(TIME39) + ["total"])


def judge_table(runs):
    """13번 상태 판정(보고서 6.2·6.4): 39장 최종 실행(자동 ROI F39a, roi.csv F39)의 켬 상태 지표로 다시 낸 판정에서
    같은 장소 정상·흔들림 쌍 가운데 흔들림 쪽만 샤프닝을 고른 쌍 수, 흐림(blur) 사진 가운데 샤프닝을 고른 장 수, own_18_pothole_dark가 고른 전처리.
    → DataFrame(item, auto, csv)"""
    items = {}
    for col, sid in (("auto", "F39a"), ("csv", "F39")):
        if runs.get(sid) is None:
            continue
        judge = judgements(runs[sid]["metrics"])
        pairs = blur_pairs(judge)
        blur = [j for j in judge.values() if j["condition"] == "blur"]
        items.setdefault("같은 장소 쌍: 흔들림 쪽만 샤프닝", {})[col] = (
            f"{sum(1 for p in pairs if p[3])}/{len(pairs)}쌍 (" + ", ".join(f"{p[0]} {mark(p[3])}" for p in pairs) + ")")
        items.setdefault("흐림 사진 중 샤프닝을 고른 장", {})[col] = f"{sum(1 for j in blur if j['sharpen'])}/{len(blur)}장"
        example = judge.get("own_18_pothole_dark.jpg")
        items.setdefault("own_18_pothole_dark가 고른 전처리", {})[col] = example["steps"] if example else "-"
    return pd.DataFrame([{"item": k, **v} for k, v in items.items()], columns=["item", "auto", "csv"])


def image_table(runs):
    """8번 39장 영상별 최종 판정: 사진마다 정답 유무(없으면 '미완'), 손상 유무, 상태 판정이 고른 전처리와 사람 태그에 맞는지,
    1차 설정·최종(roi.csv)·최종(자동 ROI)과 E1·E3 자동 ROI(참고)의 끔·켬 판정(correct), 최종 켬의 hit_same·hit_any·후보 수.
    포트홀은 이 표로 사례를 본다."""
    columns = (["file", "source", "condition", "gt", "has_damage", "steps", "judge_expect", "judge_hit"]
               + [f"{prefix}_correct_{mode}" for prefix, _ in IMAGE_RUNS for mode in ("off", "on")]
               + ["final_hit_same_on", "final_hit_any_on", "final_n_crack_on", "final_n_pothole_on"])
    base = next((runs[sid] for _, sid in sorted(IMAGE_RUNS, key=lambda p: p[0] != "final") if runs[sid] is not None), None)
    if base is None:
        return pd.DataFrame(columns=columns)
    judge = judgements(base["metrics"])
    tables = {prefix: runs[sid]["image"].set_index("file") if runs[sid] is not None else None for prefix, sid in IMAGE_RUNS}
    rows = []
    for _, r in base["image"].iterrows():
        f = r["file"]
        row = {"file": f, "source": r["source"], "condition": r["condition"], "gt": "있음" if bool(r["has_gt"]) else "없음(미완)",
               "has_damage": r["has_damage"], "steps": judge[f]["steps"], "judge_expect": judge[f]["expect"],
               "judge_hit": judge[f]["hit"]}
        for prefix, table in tables.items():
            for mode in ("off", "on"):
                row[f"{prefix}_correct_{mode}"] = table.loc[f, f"correct_{mode}"] if table is not None and f in table.index else None
        final = tables["final"]
        if final is not None and f in final.index:
            row.update({"final_hit_same_on": final.loc[f, "hit_same_on"], "final_hit_any_on": final.loc[f, "hit_any_on"],
                        "final_n_crack_on": final.loc[f, "n_crack_on"], "final_n_pothole_on": final.loc[f, "n_pothole_on"]})
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def saved_ids(paths, kind):
    """지난번에 저장한 사람의 결정(kind: accept 채택, reject 되돌림)의 단계 list. 없으면 빈 list."""
    path = Path(paths["results"]) / REPORT_DIR / DECISION_FILES[kind]
    return [s for s in path.read_text(encoding="utf-8").strip().split(",") if s] if path.exists() else []


def build_report(paths, human=None, command_line="python tools/run_all.py tables"):
    """있는 결과 폴더로 보고서 표 CSV(1~8번, 채택 표, 네 칸 표)와 summary.md를 results/report/에 만든다. 돌리지 않은 실험은 상태만 적는다.
    사람이 정한 채택·되돌림(human = {"accept", "reject"})이 자동 판정보다 앞서고, accept.txt·reject.txt에 남아 다음에도 쓰인다.
    앞선 채택으로 설정이 정해지는 실험(E3 쪽 띠, E6 1회)이 지금 채택과 다른 설정으로 돌았으면 '다시 돌려야 함'으로 표에서 뺀다.
    화면에 보여 줄 값(상태, 채택 표, 네 칸 표, 채택 방법, 바꿀 config 값, E0 재현, 재현 확인, 39장 정답 상태, 경고)을 돌려준다."""
    human = {"accept": [], "reject": [], **(human or {})}
    out = Path(paths["results"]) / REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    ready = readiness()
    states = {e["id"]: exp_state(e, paths, ready) for e in EXPERIMENTS}
    summaries = {sid: train_summary(paths, EXP[sid], states[sid])
                 for sid in ["E0", "E0_all", "T0", "T", "TA", "TR", "B1", "E5"] + list(MASK2_IDS) + ADOPT_ORDER}
    results = {sid: {"off": (summaries[sid] or {}).get("off"), "status": states[sid]["status"], "opts": states[sid]["opts"],
                     "forced": states[sid]["forced"]} for sid in ["E0"] + ADOPT_ORDER}
    adopt, adopted = adoption(results, human["reject"], human["accept"])
    for exp in EXPERIMENTS:
        state = states[exp["id"]]
        if (exp.get("follow_band") or exp.get("follow_last")) and state["done"] and state["opts"] is not None:
            expect = resolve(exp, adopted)
            if state["opts"] != expect:
                state.update(done=False, status=f"다시 돌려야 함: 지금 채택으로는 {changed_text(state['opts'], expect)} (compare를 다시)")
    last = last_adopted(adopted)
    notes = list(ready["notes"])
    if last == "E2" and {"E1", "E1b"} & adopted:
        notes.append("채택 방법은 순서상 마지막인 E2(마스크 없음)지만, 마스크를 켠 E1·E1b도 채택됐고 둘은 직접 견주지 않았다. "
                     "위 네 칸 표를 보고 사람이 정한다(--accept·--reject)")
    stale = [sid for sid in states if states[sid]["status"].startswith("다시 돌려야 함")]
    if stale:
        notes.append(f"지금 채택과 다른 설정으로 돈 실험: {', '.join(stale)}. python tools/run_all.py compare를 다시 돌린다"
                     "(사람이 정한 --accept·--reject는 저장돼 있어 그대로 반영)")
    forced = [sid for sid in states if states[sid]["forced"] and states[sid]["done"]]
    if forced:
        notes.append(f"{FORCED_MARK}: {', '.join(forced)}. A·B 시험의 @todo가 남은 기능을 --force로 썼다")
    runs = {e["id"]: run39(paths, e, states[e["id"]]) for e in EXPERIMENTS if e["tool"] == "run39"}
    labels = run_labels(states)
    missing, n39 = missing_gt(paths)
    grid = grid_rows(summaries, states, adopted)
    tables = {"table1_stages.csv": stages_table(paths, states), "table2_by_country.csv": country_table(paths, states),
              "table3_by_condition.csv": condition_table(runs, labels), "table4_preprocess.csv": preprocess_table(paths, states, runs),
              "table5_matching.csv": matching_table(paths, states, runs), "table6_provided_vs_own.csv": source_table(runs, labels),
              "table7_degrade.csv": degrade_table(paths, states), "table8_39_images.csv": image_table(runs), "adoption.csv": adopt,
              "adoption_grid.csv": grid}
    band_dev, band_country = band_dev_tables(paths, states, summaries)
    tables.update({"table9_auto_band.csv": band_dev, "table9_auto_band_country.csv": band_country,
                   "table9_auto_band_39.csv": band_39_table(runs, states)})
    mask_gt = {sid: mask_gt_rows(paths, EXP[sid], states[sid], out) for sid in MASK_GT_IDS}
    tables.update({"table10_mask2.csv": mask2_table(summaries, states, mask_gt),
                   "table10_mask2_changes.csv": keep_rule_changes(paths, states),
                   "table10_mask_gt.csv": pd.DataFrame(
                       [{"stage_id": sid, "folder": EXP[sid]["name"], "label": EXP[sid]["label"],
                         **dict(zip(("gt_in_band", "gt_in_mask", "widened"), mask_gt_ratios(mask_gt[sid])))} for sid in MASK_GT_IDS],
                       columns=["stage_id", "folder", "label", "gt_in_band", "gt_in_mask", "widened"]),
                   "table11_tests.csv": test_table(summaries, states), "table12_time.csv": time_table(runs),
                   "table13_judge.csv": judge_table(runs), "table14_supp.csv": supp_table(paths, states),
                   "table15_tau_points.csv": supp_points_table(paths, states)})
    for name, table in tables.items():
        table.to_csv(out / name, encoding="utf-8-sig", index=False)
    for kind, file in DECISION_FILES.items():
        (out / file).write_text(",".join(human[kind]), encoding="utf-8")
    method = resolve(EXP[last], adopted)
    report = {"out": out, "states": states, "adopt": adopt, "adopted": adopted, "last": last, "method": method, "grid": grid,
              "human": human, "labels": labels, "changes": config_changes(method), "e0": e0_reproduction(summaries["E0_all"]),
              "repro": repro_check(paths), "gt": (n39 - len(missing), n39), "runs": runs, "tables": tables, "notes": notes,
              "ready": ready, "summaries": summaries, "created": f"{datetime.datetime.now():%Y-%m-%d %H:%M}", "command": command_line}
    (out / "summary.md").write_text("\n".join(summary_lines(report)) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------
# summary.md
# ---------------------------------------------------------

def pct(v):
    """비율을 '12.3%'로(없으면 -)."""
    return "-" if pd.isna(v) else f"{float(v) * 100:.1f}%"


def pp(v):
    """비율 차이를 '+1.2%p'로(없으면 -)."""
    return "-" if pd.isna(v) else f"{float(v) * 100:+.1f}%p"


def num(v, digits=2):
    """수를 천 단위 쉼표와 정한 자리수로(없으면 -)."""
    return "-" if pd.isna(v) else f"{float(v):,.{digits}f}"


def mark(v):
    """판정 칸: 맞음 O, 틀림 X, 정답 없음·값 없음 -."""
    return "-" if v is None or pd.isna(v) else ("O" if str(v) == "True" else "X")


def md_table(header, rows):
    """마크다운 표 줄 list. 칸 안의 '|'와 줄바꿈은 표가 깨지지 않게 바꾼다."""
    clean = lambda v: str(v).replace("|", "/").replace("\n", " ")  # noqa: E731
    return (["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(clean(v) for v in row) + " |" for row in rows] + [""])


def metric_cells(r):
    """요약 행의 핵심 칸: 민감도, 특이도, 정답률(기준선), 균형 정확도, 영상당 오검출, 균열·포트홀 재현율(우연 뺌)."""
    return [pct(r["sensitivity"]), pct(r["specificity"]), f"{pct(r['accuracy'])} ({pct(r['baseline_none'])})",
            pct(r["balanced"]), num(r["fp_per_image"]),
            f"{pct(r['crack_ctr_recall'])} ({pp(r['crack_ctr_recall_net'])})",
            f"{pct(r['pothole_ctr_recall'])} ({pp(r['pothole_ctr_recall_net'])})"]


METRIC_HEAD = ["민감도", "특이도", "정답률 (기준선)", "균형 정확도", "영상당 오검출", "균열 재현율 (우연 뺌)", "포트홀 재현율 (우연 뺌)"]


def grid_lines(grid):
    """채택 표 위의 네 칸 표 줄: 검출기(1차·dark) × 노면 마스크(끔·켬)와 E3σ 한 줄. 아직 안 돈 칸·다시 돌려야 할 칸은 상태만."""
    rows = []
    for _, g in grid.iterrows():
        head = [g["detector"], g["mask"], g["band"], g["stage_id"]]
        if pd.isna(g["fp_per_image"]):
            rows.append(head + ["-"] * 6 + [g["status"]])
        else:
            rows.append(head + [pct(g["sensitivity"]), pct(g["specificity"]), pct(g["balanced"]), num(g["fp_per_image"]),
                                pp(g["crack_ctr_recall_net"]), num(g["time"], 1), g["status"]])
    return (["### 검출기 × 노면 마스크 (개발셋, 전처리 끔)", "",
             "네 조합을 자동 판정과 상관없이 나란히 둔다. 최종 채택은 이 표와 아래 판정을 보고 사람이 정한다(--accept·--reject). "
             "E1b가 채택되면 마스크를 켠 칸은 띠 0.3이다. E3σ는 E3 아래 한 줄이다.", ""]
            + md_table(["검출기", "노면 마스크", "띠 위쪽", "단계", "민감도", "특이도", "균형 정확도", "영상당 오검출",
                        "우연 뺀 균열 재현율", "처리 시간 ms/장 (참고)", "상태"], rows))


def adoption_lines(r):
    """채택 비교 절의 줄: 네 칸 표, 단계별 판정 표, E3와 E2의 차이(참고), 사람이 정한 것, 채택 방법과 바꿀 config 값, 경고,
    39장 자동 ROI(참고), E0 재현."""
    lines = ["## 채택 비교 (개발셋, 전처리 끔)", "",
             "비교는 한 번에 하나만 바뀌게 한다: E1←E0, E1b←E1, E2←E0, E3←마스크를 켠 1차 설정(E1b가 채택되면 E1b, 아니면 E1), "
             "E3σ←E3. E1b가 채택되면 E3 쪽은 띠 0.3으로 돈다. 규칙 2·3은 자동으로 보고, 규칙 1(같은 오검출 수준의 민감도)은 "
             "점수 문턱이 없어 민감도와 영상당 오검출을 나란히 두고 사람이 본다(Plan.md 3.10). 자동 판정은 비교 기준이 채택되지 않은 "
             "단계를 규칙을 넘어도 채택하지 않는다. 자동 판정은 길잡이이고, 사람이 정한 --accept·--reject가 앞선다.", ""]
    lines += grid_lines(r["grid"])
    lines += ["### 단계별 판정", ""]
    rows = []
    for _, a in r["adopt"].iterrows():
        if pd.isna(a["rule2"]):
            rows.append([a["stage_id"], a["base"], a["changed"], "-", "-", "-", "-", a["verdict"]])
            continue
        rows.append([a["stage_id"], a["base"], a["changed"],
                     f"{pct(a['sens_base'])}→{pct(a['sens'])}, {num(a['fp_base'])}→{num(a['fp'])} · {a['rule1']}",
                     f"{pct(a['balanced_base'])}→{pct(a['balanced'])} ({pp(a['balanced_diff'])}) {a['rule2']}",
                     f"{pp(a['crack_net_base'])}→{pp(a['crack_net'])} ({pp(a['crack_net_diff'])}) {a['rule3']}",
                     f"{num(a['time_base'], 1)}→{num(a['time'], 1)}", a["verdict"]])
    lines += md_table(["단계", "비교 기준", "바뀐 설정", "규칙 1: 민감도, 영상당 오검출", "규칙 2: 균형 정확도",
                       "규칙 3: 우연 뺀 균열 재현율", "처리 시간 ms/장 (참고)", "판정"], rows)
    lines += ["- 처리 시간은 단일 실행 참고값(사진마다 한 번 잰 전처리·검출·노면 마스크 시간의 평균 합)이다. 보고서에는 최종 단계 값만 쓴다."]
    e3 = r["adopt"].set_index("stage_id").loc["E3"]
    if not pd.isna(e3["ref_rule2"]):
        lines += [f"- E3를 E2와 견줌 (참고, 판단에 쓰지 않음, {e3['ref_changed']}): 균형 정확도 {pct(e3['ref_balanced'])}→{pct(e3['balanced'])} "
                  f"({pp(e3['ref_balanced_diff'])}) {e3['ref_rule2']}, 우연 뺀 균열 재현율 {pp(e3['ref_crack_net'])}→{pp(e3['crack_net'])} "
                  f"({pp(e3['ref_crack_net_diff'])}) {e3['ref_rule3']}, 민감도 {pp(e3['ref_sens_diff'])}, 영상당 오검출 {num(e3['ref_fp_diff'])}"]
    last = EXP[r["last"]]
    if human_text(r["human"]):
        lines += [f"- 사람이 정한 것(자동 판정보다 앞섬, results/report/accept.txt·reject.txt에 남음): {human_text(r['human'])}"]
    lines += [f"- 채택 방법: {last['id']} {last['label']} ({settings_text(r['method'])}). 자동 판정과 사람이 정한 것을 합쳐 "
              "E1~E3 가운데 마지막으로 채택된 단계이고, E6 1회도 이 설정으로 돈다",
              "- 그 방법으로 가려면 바꿀 config 값 (run_all은 config를 고치지 않는다. 구역 주인이 고친다):"]
    lines += [f"  - `{name} = {value!r}` ({owner})" for name, value, owner in r["changes"]] or ["  - 없음 (지금 config 그대로)"]
    lines += [f"- 경고: {note}" for note in r["notes"]] + [""]

    rows = []
    for sid in ("E1_39", "E3_39"):
        state, data = r["states"][sid], r["runs"][sid]
        label = EXP[sid]["label"] + (f" [{FORCED_MARK}]" if state["forced"] and state["done"] else "")
        if data is None:
            rows.append([label, "-", "-", "-", "-", "-", "-", state["status"]])
            continue
        s = data["source"][data["source"]["source"] == "all"].iloc[0]
        rows.append([label, settings_text(state["opts"] or EXP[sid]["opts"]), int(s["n_images"]), int(s["n_gt"]),
                     f"{pct(s['sensitivity_off'])} / {pct(s['sensitivity_on'])}", f"{pct(s['specificity_off'])} / {pct(s['specificity_on'])}",
                     f"{pct(s['accuracy_off'])} / {pct(s['accuracy_on'])} ({pct(s['baseline_none'])})", gt_status(data["metrics"])])
    lines += ["39장 자동 ROI (참고: 손으로 정한 ROI 없이 도는지 보는 것이고 채택 판단에는 쓰지 않는다)", ""]
    lines += md_table(["실행", "설정", "장수", "정답 있는", "민감도 끔/켬", "특이도 끔/켬", "정답률 끔/켬 (기준선)", "정답"], rows)
    if r["e0"]:
        lines += ["E0 재현 확인 (Plan.md 3.2 (5), 804장·전처리 끔)", ""] + md_table(
            ["항목", "기준", "이번", "같음"], [[k, f"{e:,}", f"{a:,}", "O" if ok else "X"] for k, e, a, ok in r["e0"]])
    elif r["summaries"].get("E0_all"):
        lines += [f"- E0 재현 확인은 학습 사진 {N_TRAIN}장일 때만 한다(이번 e0_all은 "
                  f"{int(r['summaries']['E0_all']['off']['images'])}장).", ""]
    return lines


def summary_lines(r):
    """보고서용 요약(summary.md) 줄: 결과 폴더 상태, 채택 비교와 바꿀 config 값, E0 재현, 표 1~8의 핵심 열, 시험셋 전·후, 재현 확인."""
    t, states = r["tables"], r["states"]
    has39, n39 = r["gt"]
    gt_note = "모두 있음" if has39 == n39 else f"{has39}/{n39}장 → 39장 표(3·6·8번)는 미완"
    ready = r["ready"]
    lines = ["# 실험 결과 요약 (tools/run_all.py)", "",
             f"- 만든 시각: {r['created']} ({r['command']})",
             "- 같은 폴더에 table1~8과 adoption.csv(모든 열)가 있고, 아래는 핵심 열만 옮겼다.",
             "- 읽는 법: 정답률 옆 괄호는 기준선('검출 없음'의 정답률)이고 균형 정확도의 기준선은 50%다. 재현율은 중심 기준이고 "
             "옆 괄호는 우연을 뺀 값(실제 − 우연 대조군)이다. 영상당 오검출은 중심 기준으로 맞지 않은 후보 수다.",
             "- 처리 시간은 최종 단계 값만 보고서에 쓴다(4번 표의 39장 최종 = 끔·켬 각 5번 중앙값). 채택 표의 시간은 한 번 잰 값의 평균이라 참고.",
             f"- 39장 정답 파일: {gt_note}",
             "- 구현 여부(A·B 시험의 @todo 기준): " + ", ".join(f"{NEEDS[k]} {'구현됨' if ready[k] else '구현 전'}" for k in SIGNALS)]
    forced = [sid for sid, state in states.items() if state["forced"] and state["done"]]
    if forced:
        lines += [f"- {FORCED_MARK}: {', '.join(forced)} (A·B 시험의 @todo가 남은 기능을 --force로 썼다. 표의 상태·판정에 표시)"]
    lines += [""]

    lines += ["## 결과 폴더", ""] + md_table(
        ["단계", "실험", "폴더", "상태", "커밋"],
        [[STAGES[e["stage"]], f"{e['id']} {e['label']}", e["name"], states[e["id"]]["status"], states[e["id"]]["commit"] or "-"]
         for e in EXPERIMENTS])

    lines += adoption_lines(r)

    lines += ["## 1. 단계별 표 (E0~E5, 시험셋)", "",
              "E6는 7번 표. 끝나지 않은 단계는 상태만 적는다. M0~M3는 노면 마스크 보완 비교(10번 표): M0 자동 띠 + 결(남기기 center, 이전 최종), "
              "M1 + 남기기 ratio(예외 채택), M2 + 차선 (가), M3 + 차선 (나)(둘 다 채택 안 함). E5·TR은 최종 설정(자동 띠 + 결 + ratio)이다.", ""]
    rows = []
    for _, s in t["table1_stages.csv"].iterrows():
        if s["preprocess"] not in ("off", "on"):
            rows.append([s["stage_id"], s["data"], "-", "-"] + ["-"] * len(METRIC_HEAD) + [s["status"]])
        else:
            rows.append([s["stage_id"], s["data"], "끔" if s["preprocess"] == "off" else "켬", int(s["images"])]
                        + metric_cells(s) + [s["status"]])
    lines += md_table(["단계", "데이터", "전처리", "장수"] + METRIC_HEAD + ["상태"], rows)

    lines += test_lines(t["table11_tests.csv"], r["summaries"])

    lines += ["## 2. 국가별 표 (전처리 끔)", "", "켬 행은 table2_by_country.csv에 있다.", ""]
    rows = [[c["data"], c["method"], c["source"], int(c["images"])] + metric_cells(c)
            for _, c in t["table2_by_country.csv"].iterrows() if c["preprocess"] == "off"]
    lines += md_table(["데이터", "방법", "국가", "장수"] + METRIC_HEAD, rows) if rows else ["아직 결과 없음 (E0·E5·시험 단계 뒤)", ""]

    lines += condition_lines(t["table3_by_condition.csv"], r["runs"])
    lines += ["## 4. 전처리 끔·켬 표 (영상 1장 평균)", ""]
    rows = [[p["data"], p["item"], num(p["off"]), num(p["on"]), num(p["diff"])] for _, p in t["table4_preprocess.csv"].iterrows()]
    lines += md_table(["데이터", "항목", "끔", "켬", "차이"], rows) if rows else ["아직 결과 없음 (최종 단계 뒤)", ""]

    lines += ["## 5. 특징점·매칭 전·후 표", "", "쌍마다의 값은 table5_matching.csv에 있다. inlier 비율은 천장이라 점의 수를 본다.", ""]
    rows = [[m["item"], m["target"], num(m["off"]), num(m["on"]), num(m["diff"])]
            for _, m in t["table5_matching.csv"].iterrows() if not str(m["target"]).startswith("쌍 ") or m["target"] == "쌍 평균"]
    lines += md_table(["항목", "대상", "끔", "켬", "차이"], rows) if rows else ["아직 결과 없음 (최종 단계 뒤)", ""]

    lines += ["## 6. 제공 vs 직접 촬영 (39장)", "", "'(참고)' 줄은 비교 단계의 자동 ROI 실행이고 채택 판단에 쓰지 않는다.", ""]
    rows = [[s["method"], s["source"], int(s["n_images"]), int(s["n_gt"]), f"{pct(s['sensitivity_off'])} / {pct(s['sensitivity_on'])}",
             f"{pct(s['specificity_off'])} / {pct(s['specificity_on'])}",
             f"{pct(s['accuracy_off'])} / {pct(s['accuracy_on'])} ({pct(s['baseline_none'])})",
             f"{num(s['n_crack_off'])} / {num(s['n_crack_on'])}", f"{num(s['n_pothole_off'])} / {num(s['n_pothole_on'])}", s["gt_status"]]
            for _, s in t["table6_provided_vs_own.csv"].iterrows()]
    lines += md_table(["방법", "출처", "장수", "정답 있는", "민감도 끔/켬", "특이도 끔/켬", "정답률 끔/켬 (기준선)",
                       "균열 후보 끔/켬", "포트홀 후보 끔/켬", "정답"], rows) if rows else ["아직 결과 없음 (비교·최종 단계 뒤)", ""]

    lines += ["## 7. 인위 저하 표 (개발셋)", "", "1회는 E1~E3 가운데 마지막 잠정 채택 설정, 2회는 최종 설정(config)으로 돈다.", ""]
    rows = []
    for _, d in t["table7_degrade.csv"].iterrows():
        round_name = f"{d['round']}, {d['settings_from']} 설정" if d["round"].startswith("1회") else d["round"]
        if d["preprocess"] not in ("off", "on"):
            rows.append([round_name, DEGRADE_NAMES[d["degrade"]], "-"] + ["-"] * 7 + [d["status"]])
            continue
        judge = "-" if pd.isna(d["judge_hit_rate"]) else f"{pct(d['judge_hit_rate'])} ({pct(d['judge_base_rate'])})"
        rows.append([round_name, DEGRADE_NAMES[d["degrade"]], "끔" if d["preprocess"] == "off" else "켬",
                     pct(d["sensitivity"]), pct(d["specificity"]), f"{pct(d['accuracy'])} ({pct(d['baseline_none'])})",
                     num(d["fp_per_image"]), f"{pct(d['crack_ctr_recall'])} ({pp(d['crack_ctr_recall_net'])})",
                     num(d["edges_mean"], 0), judge, d["status"]])
    lines += md_table(["회차", "저하", "전처리", "민감도", "특이도", "정답률 (기준선)", "영상당 오검출", "균열 재현율 (우연 뺌)",
                       "에지 수", "상태 판정 적중 (원본 기준)", "상태"], rows)

    lines += image_lines(t["table8_39_images.csv"], gt_note)
    lines += band_lines(t)
    lines += mask2_lines(t)
    lines += time_lines(t["table12_time.csv"])
    lines += judge_lines(t["table13_judge.csv"])
    lines += supp_lines(t["table14_supp.csv"])
    if r["repro"]:
        lines += ["## 시험셋 재현 확인 (같은 코드로 다시 돌린 결과가 한 번만 잰 결과와 같은지)", ""] + [f"- {line}" for line in r["repro"]] + [""]
    return lines


def supp_lines(table):
    """14번 보완 비교 줄: 실행마다 τ 없음·fp1·fp2 민감도, rule1 점(τ·민감도·특이도·균형 정확도·오검출·우연 뺀 균열), 규칙 판정."""
    lines = ["## 14. 보완 비교 (개발셋, τ 곡선의 동작점, docs/tasks/supplement.md 4.1)", "",
             "fp1·fp2 = 영상당 오검출 1·2개 이하에서 민감도 최대인 점(규칙 1), rule1 = 규칙 2·3을 재는 점(10/13부터 fp2 점과 같음). "
             "동작점마다의 모든 지표는 table15_tau_points.csv에 있다.", ""]
    done = table[table["none_sens"].notna()] if len(table) and "none_sens" in table else table.iloc[0:0]
    if not len(done):
        return lines + ["아직 결과 없음 (supp 단계 뒤)", ""]

    def tau(v):
        return "-" if pd.isna(v) else ("−∞" if v == -math.inf else ("+∞" if v == math.inf else f"{float(v):.3g}"))
    rows = [[r["stage_id"], r["label"], r["row"], f"{pct(r['none_sens'])} / {pct(r['none_spec'])} / {num(r['none_fp'])}",
             f"{pct(r['fp1_sens'])} / {pct(r['fp1_spec'])}", f"{pct(r['fp2_sens'])} / {pct(r['fp2_spec'])}",
             f"{tau(r['rule1_tau_crack'])}·{tau(r['rule1_tau_pothole'])}",
             f"{pct(r['rule1_sensitivity'])} / {pct(r['rule1_specificity'])} / {pct(r['rule1_balanced'])} / {num(r['rule1_fp_per_image'])}",
             pp(r["rule1_crack_ctr_recall_net"]), r["rule1"] if isinstance(r["rule1"], str) else "-",
             r["rule2"] if isinstance(r["rule2"], str) else "-", r["rule3"] if isinstance(r["rule3"], str) else "-",
             r["decision"] if isinstance(r["decision"], str) and r["decision"] else "-"] for _, r in done.iterrows()]
    return lines + md_table(["단계", "이름", "판정 행", "τ 없음 민감도/특이도/오검출", "fp1 민감도/특이도", "fp2 민감도/특이도",
                             "rule1 τ 균열·포트홀", "rule1 민감도/특이도/균형/오검출", "rule1 우연 뺀 균열", "규칙 1", "규칙 2", "규칙 3",
                             "판정"], rows)


def band_lines(t):
    """9번 자동 띠 비교 줄: 개발셋(전처리 끔) 전체·국가별과 39장 자동 ROI. E3(띠 고정) → 자동 띠 → 자동 띠 + 결(모두 남기기 center)."""
    lines = ["## 9. 자동 띠 비교: 띠 고정(E3) → 자동 띠 → 자동 띠 + 결 (모두 남기기 center, ratio는 10번 표)", "",
             "자동 띠: 사진 전체 노면 마스크를 보고 띠 위쪽 경계를 0.5에서 0.1씩 위로(구간 노면 비율 ≥ 0.95). 결: 넓힐 구간 노면의 7×7 국소 "
             "표준편차 중앙값이 시작 띠 노면의 1/1.5~1.5배일 때만. 띠 안 정답은 정답 상자 중심이 띠 안에 든 비율, 처리 시간은 사진마다 한 번 잰 "
             "값의 평균 합(참고, 띠 경계 포함). 규칙은 E3와 견준다.", ""]
    dev = t["table9_auto_band.csv"]
    rows = []
    for _, d in dev.iterrows():
        if pd.isna(d.get("sensitivity")):
            rows.append([d["run"], d["status"]] + ["-"] * 11)
            continue
        rows.append([d["run"], d["status"], pct(d["sensitivity"]), pct(d["specificity"]), pct(d["balanced"]), num(d["fp_per_image"]),
                     pp(d["crack_ctr_recall_net"]), pct(d["gt_in_band"]), pct(d["widened"]), f"{num(d['time'], 1)} ({num(d['t_band'], 1)})",
                     d["rule1"] if isinstance(d["rule1"], str) else "-", d["rule2"] if isinstance(d["rule2"], str) else "-",
                     d["rule3"] if isinstance(d["rule3"], str) else "-"])
    lines += ["### 개발셋 (전처리 끔)", ""] + md_table(
        ["방법", "상태", "민감도", "특이도", "균형 정확도", "영상당 오검출", "균열 재현율 (우연 뺌)", "띠 안 정답", "넓어진 사진",
         "시간 ms/장 (띠 경계)", "규칙 1", "규칙 2", "규칙 3"], rows)
    country = t["table9_auto_band_country.csv"]
    if len(country):
        runs = list(dict.fromkeys(country["run"]))
        cell = {(c["source"], c["run"]): c for _, c in country.iterrows()}
        view = {"China_Drone": "드론", "China_MotorBike": "오토바이"}
        for title, fn in (("넓어진 사진 비율", lambda c: pct(c["widened"])),
                          ("균형 정확도 (무손상 사진이 없는 국가는 민감도)",
                           lambda c: pct(c["balanced"]) if not pd.isna(c["balanced"]) else f"민감도 {pct(c['sensitivity'])}"),
                          ("띠 안 정답 비율", lambda c: pct(c["gt_in_band"]))):
            rows = [[source, view.get(source, "차량")] + [fn(cell[(source, run)]) if (source, run) in cell else "-" for run in runs]
                    for source in dict.fromkeys(country["source"])]
            lines += [f"### 국가별 {title} (개발셋, 전처리 끔)", ""] + md_table(["국가", "시점"] + runs, rows)
    b39 = t["table9_auto_band_39.csv"]
    rows = []
    for _, b in b39.iterrows():
        if pd.isna(b.get("accuracy")):
            rows.append([b["run"], "-", "-", "-", "-", "-", "-", b["status"]])
            continue
        rows.append([b["run"], "끔" if b["preprocess"] == "off" else "켬", f"{pct(b['accuracy'])} ({pct(b['baseline_none'])})",
                     pct(b["sensitivity"]), pct(b["specificity"]),
                     "-" if pd.isna(b["widened"]) else f"{int(b['widened'])}/{int(b['n_images'])}",
                     f"{num(b['time'], 1)} ({num(b['t_band'], 1)})", b["status"]])
    lines += ["### 39장 자동 ROI", "", "처리 시간은 끔·켬 각 5번 중앙값의 39장 평균(띠 경계 포함). 넓어진 사진은 띠 고정보다 띠가 넓은 사진 수.", ""]
    lines += md_table(["방법", "전처리", "정답률 (기준선)", "민감도", "특이도", "넓어진 사진", "시간 ms/장 (띠 경계)", "상태"], rows)
    return lines


def test_lines(table, summaries):
    """1번 표 아래 시험셋 세 번(table11): 목표 출발값(804장 1차, 전처리 끔)과 같은 시험셋의 1차 방법·띠 고정·자동 띠·자동 띠 + ratio(최종)를
    폴더·잰 시각(run_info 시작)·설정과 함께 나란히. 시험 결과가 하나도 없으면 빈 list."""
    done = table[table["preprocess"].isin(["off", "on"])] if len(table) else table
    if not len(done):
        return []
    cols = []
    e0 = (summaries.get("E0_all") or {}).get("off")
    if e0 is not None:
        cols.append(("804장 1차 끔", e0, {"folder": EXP["E0_all"]["name"], "started": "-", "settings": "1차(legacy, 마스크 없음, 띠 0.5)"}))
    for _, row in done.iterrows():
        cols.append((f"{row['name']} {'끔' if row['preprocess'] == 'off' else '켬'}", row, row))
    items = [("영상당 오검출", lambda x: num(x["fp_per_image"])), ("특이도", lambda x: pct(x["specificity"])),
             ("균열 재현율 (우연 뺌)", lambda x: f"{pct(x['crack_ctr_recall'])} ({pp(x['crack_ctr_recall_net'])})"),
             ("민감도", lambda x: pct(x["sensitivity"])),
             ("정답률 (기준선)", lambda x: f"{pct(x['accuracy'])} ({pct(x['baseline_none'])})"),
             ("균형 정확도", lambda x: pct(x["balanced"]))]
    head = [("폴더", lambda m: m["folder"]), ("잰 시각", lambda m: m["started"] or "-"), ("설정", lambda m: m["settings"])]
    rows = [[item] + [fn(meta) for _, _, meta in cols] for item, fn in head]
    rows += [[item] + [fn(x) for _, x, _ in cols] for item, fn in items]
    return (["### 시험셋 세 번 (table11_tests.csv, 같은 시험셋 152장)", "",
             "시험셋은 설정을 정한 뒤 한 번씩만 쟀다: 띠 고정(test_once), 자동 띠 채택 뒤(test_autoband), ratio 채택 뒤(test_ratio, 최종). "
             "잰 시각은 각 폴더 run_info의 시작 시각이다. 목표의 출발값(영상당 오검출 8.1, 특이도 7%, 균열 재현율 4.9%)은 804장 1차 값이라 "
             "같은 시험셋의 1차 방법(e0_test)도 함께 둔다. 시험셋 결과를 보고 설정을 다시 바꾸지 않았다.", ""]
            + md_table(["값"] + [name for name, _, _ in cols], rows))


def mask2_lines(t):
    """10번 노면 마스크 보완 줄: 채택 표(개발셋, 전처리 끔, M0와 견줌), 판정이 바뀐 사진과 후보의 노면 안 비율, 마스크 안 정답 비율과 정의."""
    lines = ["## 10. 노면 마스크 보완: 남기기 규칙·차선 (채택 표, 개발셋, 전처리 끔)", "",
             "한 번에 하나씩 더했다: 자동 띠 + 결(M0, 남기기 center = 이전 최종) → + 남기기 ratio(M1). 차선 (가)·(나)는 남기기 판정을 내린 뒤의 "
             "설정(그때 center 유지) 위에 더했다. 규칙은 M0와 견주고, 판정은 사람이 정한 것(10/12)이다. 균형 정확도 차이는 반올림 전 값이다. "
             "시간은 사진마다 한 번 잰 평균(참고).", ""]
    rows = []
    for _, d in t["table10_mask2.csv"].iterrows():
        if pd.isna(d.get("sensitivity")):
            rows.append([f"{d['stage_id']} {d['label']}", d["status"]] + ["-"] * 15)
            continue
        diff = "-" if pd.isna(d["balanced_diff"]) else f"{d['balanced_diff'] * 100:+.2f}%p"
        rows.append([f"{d['stage_id']} {d['label']}", pct(d["sensitivity"]), pct(d["specificity"]),
                     f"{d['balanced'] * 100:.2f}% ({diff})", pct(d["balanced_on"]), num(d["fp_per_image"]),
                     pp(d["crack_ctr_recall_net"]), pct(d["gt_in_mask"]), pct(d["gt_in_band"]), pct(d["widened"]),
                     num(d["pred_per_image"]), f"{num(d['t_road'], 1)} / {num(d['t_band'], 1)}",
                     d["rule1"] if isinstance(d["rule1"], str) else "-", d["rule2"] if isinstance(d["rule2"], str) else "-",
                     d["rule3"] if isinstance(d["rule3"], str) else "-", d["auto"] if isinstance(d["auto"], str) else "-",
                     d["decision"]])
    lines += md_table(["방법", "민감도", "특이도", "균형 정확도 (M0와 차이)", "켬 균형 정확도", "영상당 오검출", "균열 재현율 (우연 뺌)",
                       "마스크 안 정답", "띠 안 정답", "넓어진 사진", "영상당 후보", "노면 마스크 / 띠 경계 ms", "규칙 1", "규칙 2", "규칙 3",
                       "자동 판정", "판정 (사람)"], rows)
    changes = t["table10_mask2_changes.csv"]
    if len(changes):
        lines += ["### 남기기 ratio로 영상 판정이 바뀐 사진", "",
                  "한쪽 규칙에만 남은 후보와 그 후보 영역(윤곽 안쪽까지 채운 픽셀) 중 노면 안 비율. 상자는 640px 좌표(x, y, 가로, 세로).", ""]
        rows = [["끔" if c["preprocess"] == "off" else "켬", c["image"], "있음" if c["has_damage"] else "없음",
                 f"{mark(c['correct_center'])} → {mark(c['correct_ratio'])}", f"{c['kept_by']}: {c['kind']} {c['bbox']}, 넓이 {c['area']}",
                 "노면 안" if c["center_on_road"] else "노면 밖", f"{c['road_ratio']:.2f}"] for _, c in changes.iterrows()]
        lines += md_table(["전처리", "사진", "손상", "판정 center → ratio", "바뀐 후보", "상자 중심", "노면 안 비율"], rows)
    gt = t["table10_mask_gt.csv"]
    lines += ["### 마스크 안 정답 비율 (개발셋)", "",
              "정의: 띠 안 정답(정답 상자 중심이 띠 안) 가운데 상자 중심이 노면 마스크 안인 비율(handoff_C M4). 띠 안 정답 비율 = 띠 안 정답 / 전체 정답. "
              "각 실행의 config(띠·노면 마스크 값)로 마스크를 다시 만들어 센다. 남기기 규칙은 마스크를 바꾸지 않으므로 최종(E5, ratio)은 M0와 같다.", ""]
    lines += md_table(["실행", "폴더", "띠 안 정답", "마스크 안 정답", "넓어진 사진"],
                      [[f"{g['stage_id']} {g['label']}", g["folder"], pct(g["gt_in_band"]), pct(g["gt_in_mask"]), pct(g["widened"])]
                       for _, g in gt.iterrows()])
    return lines


def time_lines(table):
    """12번 처리 시간 줄: 최종 실행의 39장 기록에서 1차 설정과 최종 설정(roi.csv·자동 ROI)."""
    lines = ["## 12. 처리 시간: 1차 설정 vs 최종 설정 (최종 실행의 39장)", "",
             "끔·켬 각 5번 잰 중앙값의 39장 평균(ms/장). 같은 실행(같은 컴퓨터·같은 때)에서 잰 값끼리 견준다. 1차 설정은 노면 마스크를 검출에 쓰지 않지만 pipeline이 늘 계산하므로 노면 마스크 시간이 들어 있고, roi.csv 띠라 자동 띠 경계 시간은 0이다.", ""]
    if not len(table):
        return lines + ["아직 결과 없음 (최종 단계 뒤)", ""]
    rows = [[d["run"], d["folder"], "끔" if d["preprocess"] == "off" else "켬"] + [num(d[m]) for m in TIME39] + [num(d["total"])]
            for _, d in table.iterrows()]
    return lines + md_table(["설정", "폴더", "전처리", "전처리 ms", "검출 ms", "노면 마스크 ms", "띠 경계 ms", "합 ms"], rows)[:-1] + [
        "", "최종 실행은 두 번 돌렸다: 첫 실행에서 코드가 같은 1차 설정도 40.5 → 121.8 ms로 약 3배 느려져, 컴퓨터 상태 문제로 보고 다시 쟀다"
            "(위 표는 두 번째 실행, docs/part_C.md).", ""]


def judge_lines(table):
    """13번 상태 판정 줄(보고서 6.2·6.4): 자동 ROI(최종 시연과 같은 띠)와 roi.csv 기준."""
    lines = ["## 13. 상태 판정 (보고서 6.2·6.4, 39장 최종 실행)", "",
             "39장 켬 행의 상태 지표로 다시 낸 판정. 자동 ROI는 기본 띠 + 자동 띠 + 결(최종 시연과 같은 띠), roi.csv는 사람이 정한 띠다. "
             "같은 장소 쌍은 own 번호가 같은 정상·흔들림 사진이다.", ""]
    if not len(table):
        return lines + ["아직 결과 없음 (최종 단계 뒤)", ""]
    return lines + md_table(["항목", "자동 ROI", "roi.csv"],
                            [[j["item"], j["auto"] if isinstance(j["auto"], str) else "-", j["csv"] if isinstance(j["csv"], str) else "-"]
                             for _, j in table.iterrows()])


def condition_lines(table, runs):
    """3번 조건별 표 줄(최종 roi.csv만, 나머지는 CSV)과 E5 상태 판정 검증(같은 장소 정상·흔들림 쌍)."""
    lines = ["## 3. 조건별 표 (39장 최종 roi.csv)", "",
             "조건은 사람이 붙인 태그(직접 촬영은 이름, 제공 13장은 roi.csv)이고, '고른 전처리'는 상태 판정(analyze)의 결과다. "
             "1차 설정·자동 ROI·참고 실행의 행은 table3_by_condition.csv에 있다.", ""]
    rows = [[c["condition"], int(c["n_images"]), int(c["n_gt"]), f"{pct(c['sensitivity_off'])} / {pct(c['sensitivity_on'])}",
             f"{pct(c['specificity_off'])} / {pct(c['specificity_on'])}",
             f"{pct(c['accuracy_off'])} / {pct(c['accuracy_on'])} ({pct(c['baseline_none'])})",
             c["steps_chosen"], f"{c['judge_expect']} {c['judge_hit']}".strip() or "-", c["gt_status"]]
            for _, c in table.iterrows() if str(c["method"]).startswith("최종(roi.csv)")]
    if not rows:
        return lines + ["아직 결과 없음 (최종 단계 뒤)", ""]
    lines += md_table(["조건", "장수", "정답 있는", "민감도 끔/켬", "특이도 끔/켬", "정답률 끔/켬 (기준선)", "고른 전처리",
                       "맞는 전처리를 고른 수", "정답"], rows)
    pairs = blur_pairs(judgements(runs["F39"]["metrics"])) if runs["F39"] is not None else []
    if pairs:
        ok = sum(1 for p in pairs if p[3])
        lines += [f"E5 상태 판정 검증: 같은 장소 정상·흔들림 {len(pairs)}쌍 가운데 흔들림 쪽만 흐림(샤프닝)으로 잡은 쌍 {ok}쌍", ""]
        lines += md_table(["장소", "흔들림 사진", "정상 사진", "흔들림만 샤프닝"], [[s, b, n, mark(o)] for s, b, n, o in pairs])
    return lines


def image_lines(table, gt_note):
    """8번 39장 영상별 최종 판정 표 줄. 정답이 없는 사진은 '없음(미완)'이고 판정은 -. E1·E3 자동 ROI(참고)의 판정은 CSV에 있다."""
    lines = [f"## 8. 39장 영상별 최종 판정 (정답 파일: {gt_note})", "",
             "O 맞힘, X 틀림, - 정답 없음. 손상 영상은 같은 종류 후보가 맞으면, 손상 없는 영상은 후보가 0개면 맞힘(중심 기준). "
             "E1·E3 자동 ROI(참고)의 판정은 table8_39_images.csv에 있다.", ""]
    if not len(table):
        return lines + ["아직 결과 없음 (최종 단계 뒤)", ""]
    rows = [[i["file"], i["condition"] if isinstance(i["condition"], str) else "unknown", i["gt"], mark(i["has_damage"]),
             i["steps"], f"{i['judge_expect']} {mark(i['judge_hit'])}" if i["judge_expect"] else "-",
             f"{mark(i['first_correct_off'])} / {mark(i['first_correct_on'])}",
             f"{mark(i['final_correct_off'])} / {mark(i['final_correct_on'])}",
             f"{mark(i['auto_correct_off'])} / {mark(i['auto_correct_on'])}"] for _, i in table.iterrows()]
    return lines + md_table(["사진", "조건", "정답", "손상", "고른 전처리", "맞는 전처리?", "1차 끔/켬", "최종 끔/켬",
                             "자동 ROI 끔/켬"], rows)


def print_report(report, stage):
    """표를 만든 뒤 화면에 핵심만: 표 위치, (비교·표만) 네 칸 표·판정·사람이 정한 것·바꿀 config 값·E0 재현, 경고, (시험) 전·후,
    (재현) 같은지."""
    print(f"\n표: {report['out'].as_posix()}/ (summary.md, table1~13, adoption.csv, adoption_grid.csv)")
    has39, n39 = report["gt"]
    if has39 < n39:
        print(f"39장 정답 파일 {has39}/{n39}장: 39장 표는 '미완'")
    if stage in ("compare", "tables"):
        print("검출기 × 노면 마스크 (개발셋, 전처리 끔. 처리 시간은 단일 실행 참고값):")
        for _, g in report["grid"].iterrows():
            head = f"  {g['detector']:<11} {g['mask']:<12} 띠 {g['band']} {g['stage_id']:<4}"
            if pd.isna(g["fp_per_image"]):
                print(f"{head} {g['status']}")
            else:
                print(f"{head} 민감도 {pct(g['sensitivity'])} 특이도 {pct(g['specificity'])} 균형 {pct(g['balanced'])} "
                      f"오검출 {num(g['fp_per_image'])} 균열(우연 뺌) {pp(g['crack_ctr_recall_net'])} 시간 {num(g['time'], 1)}ms"
                      + (f"  [{FORCED_MARK}]" if FORCED_MARK in g["status"] else ""))
        print("단계별 판정 (자동 판정은 길잡이, 사람이 정한 것이 앞섬):")
        for _, a in report["adopt"].iterrows():
            print(f"  {a['stage_id']:<4} 비교 기준 {a['base']:<4} {a['verdict']}")
        if human_text(report["human"]):
            print(f"사람이 정한 것: {human_text(report['human'])}")
        last = EXP[report["last"]]
        print(f"채택 방법: {last['id']} {last['label']} ({settings_text(report['method'])})")
        print("바꿀 config 값 (config는 고치지 않음):" if report["changes"] else "바꿀 config 값: 없음")
        for name, value, owner in report["changes"]:
            print(f"  {name} = {value!r}  ({owner})")
        for key, expect, actual, ok in report["e0"]:
            print(f"E0 재현 {key}: 기준 {expect:,} / 이번 {actual:,} {'같음' if ok else '다름'}")
    for note in report["notes"]:
        print(f"경고: {note}")
    if stage in TEST_STAGES:
        print("\n".join(test_lines(report["tables"]["table11_tests.csv"], report["summaries"])))
    if stage in ("repro",) + tuple(RETEST_REPROS):
        for line in report["repro"] or ["test_once·final_test 결과가 없어 견주지 못함"]:
            print(f"재현 확인 {line}")


def main(argv=None, paths=None, ask=input):
    """명령행의 단계를 돈다. 끝 코드: 0 = 끝남(구현 전이라 건너뛴 실험 포함), 1 = 실패한 실험이 있음, 2 = 확인에서 멈춤·거부.
    paths·ask는 시험에서 임시 폴더와 가짜 입력을 쓰려고 둔 인자이고, 명령행 실행은 기본값을 쓴다."""
    args = parse_args(argv)
    paths = paths or default_paths()
    human = {kind: getattr(args, kind) if getattr(args, kind) is not None else saved_ids(paths, kind) for kind in DECISION_FILES}
    for kind, other in (("accept", "reject"), ("reject", "accept")):   # 이번에 준 결정이 지난번에 저장한 반대 결정보다 앞선다
        if getattr(args, kind) is not None:
            human[other] = [s for s in human[other] if s not in human[kind]]
    command_line = "python tools/run_all.py " + " ".join(argv if argv is not None else sys.argv[1:])
    if args.stage == "tables":
        if args.dry_run:
            print(f"dry-run: {Path(paths['results']).as_posix()}/{REPORT_DIR}/에 표를 다시 만든다(실험은 돌리지 않음)")
            return 0
        print_report(build_report(paths, human, command_line), "tables")
        return 0
    return run_stage(args.stage, args, paths, human, command_line, ask)


if __name__ == "__main__":
    sys.exit(main())
