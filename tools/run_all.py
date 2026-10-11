"""(파트 C) E0~E6 실험을 단계마다 명령 하나로 돌리고, 보고서 표(CSV)와 요약(summary.md)을 만든다(Plan.md 3.7·3.10).
사용법: 맨 위 폴더에서  python tools/run_all.py compare|final|test|repro|tables [--dry-run] [--yes] [--reject E1,E3]
순서: compare(비교) → 사용자가 채택을 정하고 구역 주인이 config를 고침 → final(최종) → test(시험, 한 번만) → (10/12) repro(재현)
- compare: 개발셋에서 E0(804장 재현 포함)·E1·E1b·E2·E3(σ 노면 안·띠 전체)와 E6 1회(저하 다섯 가지, E2 설정 = B의 dark 1차).
  E0·E1·E1b는 1차 값(--param-set base, legacy, 띠 0.5)으로 고정해 config가 바뀐 뒤에 다시 돌려도 같은 조건이다.
  각 단계를 직전 채택 단계와 채택 규칙 세 개로 견준 표와, 채택하면 바꿀 config 값 목록(구역 주인과 함께)을 낸다.
  config 파일은 고치지 않는다. 규칙 1은 점수 문턱이 없어 사람이 보고, 뺄 단계는 --reject로 준다(표만 다시: tables --reject).
- final: config 그대로(채택 결과를 반영한 최종 설정)로 39장(최종 roi.csv·최종 자동 ROI·1차 설정), 매칭, E5(개발셋 끔·켬), E6 2회.
  처리 시간은 이 단계 값만 표에 쓴다. 시작할 때 전원 연결·다른 프로그램 닫기를 알리고 Enter를 기다린다(--yes면 건너뜀).
- test: 시험셋을 한 번만 잰다. results/test_once가 있으면 거부한다. 같은 시험셋에서 1차 방법(e0_test)을 먼저 돌려 전·후를 견준다.
  시작할 때 지금 config를 보여 주고 Enter를 기다린다(--yes면 건너뜀). 1차 방법이 실패하면 test_once를 돌리지 않는다.
- repro: (10/12) 같은 코드로 시험셋을 다시 돌려(final_test) test_once와 같은지 본다. 다르면 원인만 찾고 다시 튜닝하지 않는다.
- tables: 실험은 돌리지 않고 있는 결과 폴더로 표만 다시 만든다.
--dry-run은 돌릴 실험·명령·결과 폴더·예상 시간만 보여 주고 아무것도 쓰지 않는다.
실행 전 확인: 39장 정답 파일(빠지면 이름을 알리고 39장 표에 '미완'), 학습 사진 804장, 약속 검사, 커밋 안 된 변경. 뒤의 셋은 어기면 멈춘다.
dark 검출·노면 마스크가 아직 자리(구현 전)이면 그것이 필요한 실험은 '아직 구현 전'으로 건너뛰고 표에도 그렇게 적는다.
결과: 실험마다 results/<폴더>/ (도구의 결과·config_used*.txt와, run_all이 남기는 run_info*.txt: 커밋 해시·명령·시각·고정한 값)
      표는 results/report/: table1_stages … table8_39_images.csv, adoption.csv, summary.md(보고서용 요약), reject.txt."""
import argparse
import collections
import contextlib
import datetime
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
from src import analyze, detect, io_utils  # noqa: E402
from src import config as C  # noqa: E402
from tools import evaluate_train as et  # noqa: E402

STAGES = {"compare": "비교", "final": "최종", "test": "시험", "repro": "재현", "tables": "표만"}
REPORT_DIR = "report"               # 표는 results/report/
REJECT_FILE = "reject.txt"          # 사람 확인에서 뺀 단계(다음 단계에서 표를 다시 만들어도 유지되게)
N_TRAIN = 804                       # 학습 사진 수. 일부만 있으면 개발/시험 나누기가 계획(652·152)과 달라진다
PLANNED_SPLIT = {"all": 804, "dev": 652, "test": 152}   # 학습 사진이 없을 때 예상 시간에 쓰는 장수 (Plan.md 3.2 (3))

# 1차 방법(E0)의 고정값. config가 채택 결과로 바뀐 뒤에도 E0·E1·E1b를 같은 조건으로 다시 낼 수 있게 숫자로 둔다
FIRST_ROI_TOP = 0.5                 # 1차 기본 띠의 위쪽 비율 (Plan.md 1.2)
E1B_ROI_TOP = 0.3                   # E1b 띠의 위쪽 비율 (Plan.md 3.10)
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
TOOL_SCRIPT = {"train": "tools/evaluate_train.py", "run39": "run_experiment.py", "match": "run_matching.py"}
RESULT_FILE = {"train": et.SUMMARY_NAME, "run39": "metrics.csv", "match": "matching.csv"}       # 있으면 그 실험이 끝난 것
RUN_INFO = {"train": "run_info_train.txt", "run39": "run_info.txt", "match": "run_info_matching.txt"}
CONFIG_FILE = {"train": et.CONFIG_NAME, "run39": "config_used.txt", "match": "config_used_matching.txt"}
DATA_NAMES = {"all": "학습 전체", "dev": "개발", "test": "시험"}


def _exp(id_, stage, tool, name, label, opts=None, **extra):
    """실험 하나의 정의. opts는 도구 옵션으로 줄 값(params·detect·mask·roi_top·sigma_in_road·degrade)이고, 없는 값은 config 그대로 돈다.
    adopt=True인 실험은 채택 규칙으로 견주는 단계이고, 목록 순서가 '직전 채택 단계'를 찾는 순서다."""
    return {"id": id_, "stage": stage, "tool": tool, "name": name, "label": label, "opts": dict(opts or {}), **extra}


EXPERIMENTS = [
    # 비교 단계 (개발셋)
    _exp("E0_all", "compare", "train", "e0_all", "E0 재현: 1차 방법 804장", FIRST, split="all"),
    _exp("E0", "compare", "train", "e0_dev", "E0 1차 방법", FIRST, split="dev", adopt=True),
    _exp("E1", "compare", "train", "e1_mask_dev", "E1 + 노면 마스크", {**FIRST, "mask": "on"}, split="dev", adopt=True),
    _exp("E1b", "compare", "train", "e1b_top03_dev", "E1b + 띠 0.3", {**FIRST, "mask": "on", "roi_top": E1B_ROI_TOP},
         split="dev", adopt=True),
    _exp("E2", "compare", "train", "e2_dark_dev", "E2 어두운 영역 검출(마스크 없음)", DARK, split="dev", adopt=True),
    _exp("E3", "compare", "train", "e3_dark_mask_dev", "E3 + 노면 마스크(σ 노면 안)",
         {**DARK, "mask": "on", "sigma_in_road": True}, split="dev", adopt=True),
    _exp("E3s", "compare", "train", "e3_sigma_band_dev", "E3 + 노면 마스크(σ 띠 전체)",
         {**DARK, "mask": "on", "sigma_in_road": False}, split="dev", adopt=True),
    # E4 자리 (Should, Plan.md 3.2 (3)·3.10): B가 후보 score를 채우고 evaluate_train에 τ 곡선(영상당 오검출 1·2개의 민감도)이 생기면
    # todo를 지우고 tool·opts(E3 설정 + τ 옵션)를 주며, rule1()을 같은 동작점의 민감도 비교로 바꾼다
    _exp("E4", "compare", None, "e4_score_dev", "E4 점수 문턱 τ (Should)", split="dev", adopt=True, todo="점수 τ(B, Should)"),
    *[_exp(f"E6_{kind}", "compare", "train", f"e6_{kind}_dev", f"E6 1회 {DEGRADE_NAMES[kind]}", {**DARK, "degrade": kind},
           split="dev") for kind in et.DEGRADE_KINDS],
    # 최종 단계 (config 그대로). 처리 시간을 보고서에 쓰는 39장 최종과 매칭을 앞에 둔다
    _exp("F39", "final", "run39", "final", "39장 최종 설정(roi.csv)", roi="csv"),
    _exp("FM", "final", "match", "final", "특징점·매칭 최종 설정"),
    _exp("F39a", "final", "run39", "final_auto", "39장 최종 설정(자동 ROI)", roi="auto"),
    _exp("F39e0", "final", "run39", "e0", "39장 1차 설정", FIRST, roi="csv"),
    _exp("E5", "final", "train", "e5_dev", "E5 최종 설정 전처리 끔·켬", split="dev"),
    *[_exp(f"F6_{kind}", "final", "train", f"e6_{kind}_final_dev", f"E6 2회(최종) {DEGRADE_NAMES[kind]}", {"degrade": kind},
           split="dev") for kind in et.DEGRADE_KINDS],
    # 시험 단계: 1차 방법을 먼저(실패하면 test_once 전에 멈춤), 그다음 최종 설정을 한 번만
    _exp("T0", "test", "train", "e0_test", "시험셋 1차 방법", FIRST, split="test"),
    _exp("T", "test", "train", "test_once", "시험셋 최종 설정(한 번만)", split="test"),
    # 재현 단계 (10/12): test_once와 같은 코드·설정으로 다시
    _exp("R", "repro", "train", "final_test", "시험셋 재현 확인", split="test"),
]
EXP = {e["id"]: e for e in EXPERIMENTS}
ADOPT_IDS = [e["id"] for e in EXPERIMENTS if e.get("adopt")]          # E0, E1, E1b, E2, E3, E3s, E4

METRICS = ["images", "n_damage", "n_none", "sensitivity", "specificity", "accuracy", "baseline_none", "balanced",
           "fp_per_image", "pred_per_image",
           "crack_ctr_recall", "crack_ctr_recall_net", "crack_ctr_precision",
           "pothole_ctr_recall", "pothole_ctr_recall_net", "pothole_ctr_precision",
           "crack_iou_recall", "crack_iou_precision", "pothole_iou_recall", "pothole_iou_precision", "chance_skipped"]
STAGE_IDS = ["E0_all", "E0", "E1", "E1b", "E2", "E3", "E3s", "E4", "E5", "T0", "T"]      # 1번 단계별 표의 행 순서
STAGE_HEAD = ["stage_id", "label", "data", "folder", "status", "commit"]
COUNTRY_RUNS = (("E0", "1차"), ("E5", "최종"), ("T0", "1차"), ("T", "최종"))
RUNS39 = (("F39e0", "1차 설정"), ("F39", "최종(roi.csv)"), ("F39a", "최종(자동 ROI)"))
GROUP39 = ["n_images", "n_gt", "n_damage", "n_none", "sensitivity_off", "sensitivity_on", "specificity_off", "specificity_on",
           "accuracy_off", "accuracy_on", "baseline_none"]
SOURCE39 = GROUP39 + ["n_crack_off", "n_crack_on", "n_pothole_off", "n_pothole_on", "edges_off", "edges_on"]
PRE39 = [("에지 수", "edges"), ("노면 안 에지 수", "edges_road"), ("해리스 코너 수", "harris"), ("균열 후보 수", "n_crack"),
         ("균열 면적 % (노면 대비)", "area_crack_pct"), ("포트홀 후보 수", "n_pothole"), ("포트홀 면적 % (노면 대비)", "area_pothole_pct"),
         ("전처리 시간 ms", "t_pre_ms"), ("검출 시간 ms", "t_detect_ms"), ("노면 마스크 시간 ms", "t_road_ms")]
PRE_TRAIN = [("에지 수", "edges_mean"), ("해리스 코너 수", "harris_mean"), ("균열 후보 수", "pred_crack_mean"),
             ("균열 면적 (픽셀)", "area_crack_mean"), ("포트홀 후보 수", "pred_pothole_mean"), ("포트홀 면적 (픽셀)", "area_pothole_mean")]
MATCH_ITEMS = [("SIFT 특징점 수(기준 A)", "kp_ref"), ("SIFT 특징점 수(대상)", "kp_target"), ("good 매칭 점 수", "good"),
               ("inlier 수", "inliers"), ("inlier 비율", "inlier_ratio")]
DEGRADE_ROUNDS = (("1회(비교, E2 설정)", "E6_"), ("2회(최종 설정)", "F6_"))
DEGRADE_COLS = ["sensitivity", "specificity", "accuracy", "baseline_none", "balanced", "fp_per_image",
                "crack_ctr_recall", "crack_ctr_recall_net", "pothole_ctr_recall", "pothole_ctr_recall_net",
                "edges_mean", "harris_mean", "pred_crack_mean", "pred_pothole_mean", "area_crack_mean", "area_pothole_mean",
                "judge_expect", "judge_hit_rate", "judge_base_rate"]
TIME_MEANS = ["t_pre_ms_mean", "t_detect_ms_mean", "t_road_ms_mean"]   # 학습 데이터 처리 시간(한 번 잰 값의 평균): 2회만, 참고용
ADOPT_COLUMNS = ["stage_id", "label", "status", "base", "changed",
                 "sens_base", "sens", "sens_diff", "fp_base", "fp", "fp_diff", "rule1",
                 "balanced_base", "balanced", "balanced_diff", "rule2",
                 "crack_net_base", "crack_net", "crack_net_diff", "rule3", "verdict"]
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
    """단계(compare·final·test·repro·tables)와 --dry-run·--yes·--reject를 읽는다. --reject는 채택을 견주는 단계(E1~E4) 이름만 받는다."""
    parser = argparse.ArgumentParser(description="E0~E6 일괄 실행과 보고서 표")
    parser.add_argument("stage", choices=list(STAGES),
                        help="compare(비교) → final(최종) → test(시험, 한 번만) → repro(재현, 10/12). tables는 표만 다시")
    parser.add_argument("--dry-run", action="store_true", help="돌릴 실험·명령·결과 폴더·예상 시간만 보여 주고 끝냄(아무것도 쓰지 않음)")
    parser.add_argument("--yes", action="store_true", help="시작 전 확인(Enter)을 건너뜀")
    parser.add_argument("--reject", help="사람 확인(채택 규칙 1)에서 뺀 단계를 쉼표로(예: E1,E3). 빈 값('')이면 지움. "
                                         "주지 않으면 지난번 값(results/report/reject.txt)")
    args = parser.parse_args(argv)
    if args.reject is not None:
        ids = [s.strip() for s in args.reject.split(",") if s.strip()]
        bad = [s for s in ids if s not in ADOPT_IDS[1:]]
        if bad:
            parser.error(f"--reject는 {', '.join(ADOPT_IDS[1:])} 가운데서만 고른다: {', '.join(bad)}")
        args.reject = ids
    return args


# ---------------------------------------------------------
# 실험 명령과 필요한 기능
# ---------------------------------------------------------

def param_values(name):
    """evaluate_train의 1차 튜닝 세트 하나(base 등)를 config 값 {이름: 값}으로."""
    return {key: value for params in et.PARAM_SETS if params["name"] == name for key, value in params.items() if key != "name"}


def command(exp):
    """실험의 도구 명령행 인자와, 명령행으로 줄 수 없어 실행 중에만 고정할 config 값 {이름: 값}을 만든다.
    run_experiment에는 --param-set이 없어 1차 legacy 값(evaluate_train의 base 세트)을 실행 중에만 덮고, --roi csv에서는
    --roi-top을 받지 않아 띠 위쪽 비율도 실행 중에만 덮는다(39장은 모두 roi.csv에 있어 결과는 같고, 빠진 사진이 생겨도 1차 값을 쓰게)."""
    opts, argv, pins = exp["opts"], [], {}
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
        argv += ["--set", f"DARK_SIGMA_IN_ROAD={opts['sigma_in_road']}"]
    if opts.get("degrade"):
        argv += ["--degrade", opts["degrade"]]
    return argv + ["--name", exp["name"]], pins


def pins_text(pins):
    """실행 중에만 고정한 config 값을 한 줄로. 1차 legacy 값 묶음은 이름만 줄여 쓴다."""
    if not pins:
        return "없음"
    base = param_values("base")
    rest = {k: v for k, v in pins.items() if not (k in base and base[k] == v)}
    parts = [f"1차 legacy 값 {len(pins) - len(rest)}개(evaluate_train base 세트)"] if len(rest) < len(pins) else []
    return ", ".join(parts + [f"{k}={v!r}" for k, v in rest.items()])


def needs(exp):
    """실험이 쓰는 (검출 방식, 노면 마스크 켬?): 옵션으로 준 값, 없으면 지금 config 값."""
    opts = exp["opts"]
    mode = opts.get("detect") or C.DETECT_MODE
    mask = opts["mask"] == "on" if opts.get("mask") else bool(C.USE_ROAD_MASK)
    return mode, mask


def dark_ready():
    """dark 검출이 구현됐는지: 작은 합성 영상으로 두 검출 함수를 DETECT_MODE "dark"로 불러 자리의 NotImplementedError가 나는지 본다."""
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


def mask_ready():
    """노면 마스크가 자리(띠 전체 = 노면 255)에서 바뀌었는지: 왼쪽이 풀색인 합성 띠에서 노면 아님(0)이 하나라도 나오는지 본다.
    자리와, 구현했지만 아무것도 빼지 못하는 마스크는 결과가 같아(E1 = E0) 둘 다 구현 전으로 본다."""
    band = np.full((120, 240, 3), 120, np.uint8)
    band[:, :100] = (40, 160, 60)                           # 채도가 높은 풀색(BGR)
    try:
        road = io_utils.road_mask(band)
    except NotImplementedError:
        return False
    except Exception:                                       # 구현은 됐다. 오류는 실제 실행에서 그대로 드러난다
        return True
    return bool(np.any(np.asarray(road) == 0))


def readiness():
    """dark 검출과 노면 마스크가 구현됐는지 {"dark": bool, "mask": bool}. 실행 한 번에 한 번만 잰다."""
    return {"dark": dark_ready(), "mask": mask_ready()}


def unmet(exp, ready):
    """실험에 필요한데 아직 구현되지 않은 것의 이름 list(비면 돌릴 수 있음). 매칭은 ROI·검출을 쓰지 않아 늘 돌린다."""
    if exp.get("todo"):
        return [exp["todo"]]
    if exp["tool"] == "match":
        return []
    mode, mask = needs(exp)
    return ([NEEDS["dark"]] if mode == "dark" and not ready["dark"] else []) + \
           ([NEEDS["mask"]] if mask and not ready["mask"] else [])


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


def print_checks(checks):
    """실행 전 확인 결과를 한 줄씩 보여 준다: 통과, 멈춤(어기면 실행하지 않음), 알림(39장 정답)."""
    print("실행 전 확인")
    for name, ok, note, blocking in checks:
        print(f"  [{'통과' if ok else ('멈춤' if blocking else '알림')}] {name}: {note}")


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


def write_run_info(path, exp, argv, pins, status, started, seconds, context):
    """결과 폴더에 실행 기록을 남긴다(결과가 어떤 코드·명령·설정에서 나왔는지 폴더만 보고 알게). config 값은 도구가 따로 쓴다."""
    lines = [f"단계: {context['단계']}", f"실험: {exp['id']} {exp['label']}",
             f"명령: python {TOOL_SCRIPT[exp['tool']]} {' '.join(argv)}", f"실행 중 고정한 config 값: {pins_text(pins)}",
             f"상태: {status}", f"커밋: {context['커밋']}", f"시작: {started:%Y-%m-%d %H:%M:%S}", f"걸린 시간: {seconds:.1f}초",
             f"환경: {context['환경']}", f"config 값: {CONFIG_FILE[exp['tool']]} (같은 폴더, 도구가 씀)"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_run_info(path):
    """run_info를 {항목: 값}으로 읽는다. 없으면 빈 dict."""
    if not path.exists():
        return {}
    pairs = (line.split(": ", 1) for line in path.read_text(encoding="utf-8").splitlines() if ": " in line)
    return dict(pairs)


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


def run_one(exp, paths, ready, context):
    """실험 하나를 돌려 (상태, 걸린 초)를 돌려준다. 필요한 기능이 아직 없으면 돌리지 않고 '아직 구현 전'.
    끝나면 결과 폴더에 run_info를 남긴다(실패해서 폴더가 없으면 남기지 않음)."""
    missing = unmet(exp, ready)
    if missing:
        return "아직 구현 전: " + "·".join(missing), 0.0
    argv, pins = command(exp)
    started, start = datetime.datetime.now(), time.perf_counter()
    try:
        with patched(C, pins):
            code = run_tool(exp, argv, paths)
        status = "완료" if code == 0 else f"실패: 끝 코드 {code} (실패한 사진은 위 출력)"
    except NotImplementedError as e:
        status = f"아직 구현 전: {e}"
    except (Exception, SystemExit) as e:                    # 명령행 오류(SystemExit)도 이 실험의 실패로 적고 다음으로
        status = f"실패: {type(e).__name__}: {e}"
    seconds = time.perf_counter() - start
    folder = Path(paths["results"]) / exp["name"]
    if folder.is_dir():
        write_run_info(folder / RUN_INFO[exp["tool"]], exp, argv, pins, status, started, seconds, context)
    return status, seconds


def stage_refusal(stage, results):
    """단계를 거부할 이유(없으면 None): 시험은 test_once가 이미 있으면(한 번만), 재현은 test_once 결과가 없으면."""
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


def print_plan(stage, exps, paths, ready, refusal):
    """--dry-run: 돌릴 실험·명령·결과 폴더·예상 시간을 보여 준다. 아무것도 쓰거나 돌리지 않는다."""
    counts = split_counts(paths)
    n39 = len(rx.list_images(None, paths["image_dirs"]))
    n_pairs = len(list(Path(paths["pairs"]).glob("*.jpg")))
    results = Path(paths["results"])
    print(f"[{STAGES[stage]} 단계] 실험 {len(exps)}개 (dry-run: 돌리지 않음)")
    if refusal:
        print(f"  ※ 지금 실제로 돌리면 거부됨: {refusal}")
    total, everything, skipped = SEC_CHECK, SEC_CHECK, 0
    for i, exp in enumerate(exps, 1):
        missing = unmet(exp, ready)
        folder = results / exp["name"]
        sec = estimate(exp, counts, n39, n_pairs)
        everything += sec
        print(f"{i:2d}. {exp['id']}  {exp['label']}")
        print(f"    폴더: {folder.as_posix()}/" + ("  (이미 있음: 덮어씀)" if folder.exists() else ""))
        if exp["tool"]:
            argv, pins = command(exp)
            print(f"    명령: python {TOOL_SCRIPT[exp['tool']]} {' '.join(argv)}")
            if pins:
                print(f"    실행 중 고정: {pins_text(pins)}")
        if missing:
            skipped += 1
            print(f"    건너뜀: 아직 구현 전 ({', '.join(missing)})")
        else:
            total += sec
            print(f"    예상: {duration(sec)}")
    if stage == "final":
        print("시작할 때: 전원 연결·다른 프로그램 닫기 안내 후 Enter를 기다림(--yes면 건너뜀)")
    if stage == "test":
        print("시작할 때: 지금 config를 보여 주고 Enter를 기다림(--yes면 건너뜀). 1차 방법이 실패하면 test_once를 돌리지 않음")
    print(f"예상 합계: 약 {duration(total)} (실행 전 확인 포함"
          + (f", 건너뛸 실험 {skipped}개 빼고), 모두 구현되면 약 {duration(everything)}" if skipped else ")")
          + ". legacy·마스크 끔으로 잰 값이라 dark·노면 마스크는 더 걸릴 수 있음")
    print(f"표: {(results / REPORT_DIR).as_posix()}/ (table1_stages … table8_39_images.csv, adoption.csv, summary.md)")


FINAL_PROMPT = "처리 시간을 잽니다: 노트북 전원을 연결하고 다른 프로그램을 닫아 주세요."


def test_prompt():
    """시험 단계 시작 확인 문구: 한 번만 잰다는 것과 지금 config의 채택 관련 값."""
    return ("시험셋은 한 번만 잽니다. 이 결과를 보고 파라미터를 고치지 않습니다(Plan.md 3.10).\n"
            f"지금 config: DETECT_MODE={C.DETECT_MODE!r}, USE_ROAD_MASK={C.USE_ROAD_MASK}, "
            f"ROI_TOP_DEFAULT={C.ROI_TOP_DEFAULT}, DARK_SIGMA_IN_ROAD={C.DARK_SIGMA_IN_ROAD}")


def run_stage(stage, args, paths, reject, command_line, ask=input):
    """한 단계를 돈다: 거부 조건 → 실행 전 확인 → 시작 확인 → 실험을 차례로 → 표 만들기. 끝 코드는 main과 같다."""
    exps = [e for e in EXPERIMENTS if e["stage"] == stage]
    ready = readiness()
    results = Path(paths["results"])
    refusal = stage_refusal(stage, results)
    if args.dry_run:
        print_plan(stage, exps, paths, ready, refusal)
        return 0
    if refusal:
        print(f"멈춤: {refusal}")
        return 2
    checks = precheck(paths)
    print_checks(checks)
    if any(blocking and not ok for _, ok, _, blocking in checks):
        print("멈춤: 위 확인을 통과하지 못함 (39장 정답 말고는 모두 통과해야 돌림)")
        return 2
    if stage in ("test", "repro"):
        missing = sorted({m for e in exps for m in unmet(e, ready)})
        if missing:
            print(f"멈춤: 시험셋은 설정에 필요한 기능을 다 갖춘 뒤 돌린다. 아직 구현 전: {', '.join(missing)}")
            return 2
    if stage == "final" and not confirm(FINAL_PROMPT, args.yes, ask):
        return 2
    if stage == "test" and not confirm(test_prompt(), args.yes, ask):
        return 2

    context = run_context(stage, command_line)
    start, done = time.perf_counter(), []
    for i, exp in enumerate(exps, 1):
        print(f"\n===== [{STAGES[stage]} {i}/{len(exps)}] {exp['id']} {exp['label']} -> {(results / exp['name']).as_posix()}/")
        status, seconds = run_one(exp, paths, ready, context)
        print(f"----- {exp['id']}: {status} ({duration(seconds)})")
        done.append((exp, status, seconds))
        if stage == "test" and status != "완료":
            print("멈춤: 시험 단계는 실패하면 다음 실험을 돌리지 않는다(test_once를 한 번만 재려고)")
            break

    print(f"\n===== {STAGES[stage]} 단계 끝 ({duration(time.perf_counter() - start)})")
    for exp, status, seconds in done:
        print(f"  {exp['id']:<7} {exp['name']:<20} {status} ({duration(seconds)})")
    report = build_report(paths, reject, command_line)
    print_report(report, stage)
    return 1 if any(status.startswith("실패") for _, status, _ in done) else 0


# ---------------------------------------------------------
# 결과 읽기
# ---------------------------------------------------------

def read_csv(path):
    """CSV를 읽는다(utf-8-sig, 'NaN' 글은 NaN 수로). 없으면 None."""
    return pd.read_csv(path, encoding="utf-8-sig") if Path(path).exists() else None


def exp_state(exp, paths, ready):
    """실험의 지금 상태 {"done", "status", "commit"}: 결과 파일이 있으면 run_info의 상태·커밋, 없으면 구현 전·실패·안 돌림."""
    if exp["tool"] is None:
        return {"done": False, "status": "아직 구현 전: " + exp["todo"], "commit": ""}
    folder = Path(paths["results"]) / exp["name"]
    info = read_run_info(folder / RUN_INFO[exp["tool"]])
    commit = info.get("커밋", "")[:7]
    if (folder / RESULT_FILE[exp["tool"]]).exists():
        return {"done": True, "status": info.get("상태", "완료(run_all 밖에서 돌림)"), "commit": commit}
    missing = unmet(exp, ready)
    if missing:
        return {"done": False, "status": "아직 구현 전: " + "·".join(missing), "commit": ""}
    return {"done": False, "status": info.get("상태", "아직 안 돌림"), "commit": commit}


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

def changed_text(a, b):
    """두 단계 설정(opts)의 차이를 글로: 검출 legacy→dark, 노면 마스크 끔→켬, 띠 위쪽 0.5→0.3, σ 노면 안→띠 전체, legacy 값."""
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


def adoption(off_rows, states, reject=()):
    """채택 표: 비교 단계의 각 단계를 직전 채택 단계(E0 → E1 → E1b → E2 → E3 → E3 σ 띠 전체 순서에서 그때까지 잠정 채택된
    마지막 단계, 처음은 E0)와 개발셋·전처리 끔으로 견준다(Plan.md 3.10 채택 규칙, 검출 튜닝은 전처리를 끈 상태).
    규칙 2(균형 정확도가 오름)·3(우연을 뺀 균열 재현율이 1%p 넘게 떨어지지 않음)은 자동, 규칙 1은 rule1(사람 확인).
    규칙 2·3을 넘으면 '잠정 채택'으로 다음 단계의 비교 기준이 되고, reject에 든 단계(사람이 규칙 1에서 뺀 것)는 채택하지 않는다.
    off_rows: {단계: 전처리 끔 요약 행 또는 None} → (표, 마지막 잠정 채택 단계)"""
    champion, rows = ADOPT_IDS[0], []
    for sid in ADOPT_IDS[1:]:
        exp, base, cur = EXP[sid], off_rows.get(champion), off_rows.get(sid)
        row = {"stage_id": sid, "label": exp["label"], "status": states[sid]["status"], "base": champion,
               "changed": "-" if exp.get("todo") else changed_text(EXP[champion]["opts"], exp["opts"])}
        if cur is None or base is None:
            row["verdict"] = states[sid]["status"] if cur is None else f"비교 기준 {champion}의 결과 없음"
            rows.append(row)
            continue
        for key, col in (("sens", "sensitivity"), ("fp", "fp_per_image"), ("balanced", "balanced"),
                         ("crack_net", "crack_ctr_recall_net")):
            row.update({f"{key}_base": base[col], key: cur[col], f"{key}_diff": cur[col] - base[col]})
        row["rule1"] = rule1(row)
        row["rule2"] = rule_text(passes(row["balanced_diff"], lambda d: d > 0))
        row["rule3"] = rule_text(passes(row["crack_net_diff"], lambda d: d >= -RULE3_DROP))
        failed = [f"규칙 {n} {row[f'rule{n}']}" for n in (2, 3) if row[f"rule{n}"] != "통과"]
        if sid in reject:
            row["verdict"] = "되돌림: 사람 확인(규칙 1)에서 뺌"
        elif failed:
            row["verdict"] = "되돌림: " + ", ".join(failed)
        else:
            row["verdict"] = "잠정 채택 (규칙 1은 사람 확인)"
            champion = sid
        rows.append(row)
    return pd.DataFrame(rows, columns=ADOPT_COLUMNS), champion


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


def repro_check(paths):
    """재현 확인: final_test가 test_once와 같은지. 요약과 영상별 결과를 처리 시간 열(t_로 시작)을 빼고 견준다(같은 코드면 같아야 함).
    → 설명 줄 list (둘 중 하나가 없으면 빈 list)"""
    once, again = Path(paths["results"]) / "test_once", Path(paths["results"]) / "final_test"
    lines = []
    for file in (et.SUMMARY_NAME, et.DETAIL_NAME):
        a, b = read_csv(once / file), read_csv(again / file)
        if a is None or b is None:
            return []
        keep = [c for c in a.columns if not c.startswith("t_")]
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            lines.append(f"{file}: 다름 (열 또는 행 수가 다름: {len(a)}행 / {len(b)}행)")
            continue
        diff = [c for c in keep if not a[c].equals(b[c])]
        lines.append(f"{file}: " + (f"같음 (처리 시간 열 빼고 {len(a)}행)" if not diff else f"다름: {', '.join(diff[:10])}"))
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


def condition_table(runs):
    """3번 조건별 표: 39장 실행마다 조건(사람이 붙인 태그: 직접 촬영은 이름, 제공 13장은 roi.csv)별 영상 단위 지표와
    상태 판정(analyze)이 고른 전처리를 나란히(Plan.md 3.7 3번). 태그가 blur·dark·bright·noise면 맞는 전처리를 고른 수도 센다."""
    rows = []
    for sid, method in RUNS39:
        data = runs[sid]
        if data is None:
            continue
        judge, status = judgements(data["metrics"]), gt_status(data["metrics"])
        for _, r in data["condition"].iterrows():
            cond = r["condition"]
            picked = [j for j in judge.values() if cond == "all" or j["condition"] == cond]
            judged = [j for j in picked if j["hit"] is not None]
            chosen = collections.Counter(j["steps"] for j in picked).most_common()
            rows.append({"method": method, "folder": EXP[sid]["name"], "condition": cond, "gt_status": status,
                         **{c: r.get(c) for c in GROUP39},
                         "steps_chosen": "; ".join(f"{s} {n}장" for s, n in chosen),
                         "judge_expect": judged[0]["expect"] if judged and cond != "all" else "",
                         "judge_hit": f"{sum(j['hit'] for j in judged)}/{len(judged)}" if judged else ""})
    return pd.DataFrame(rows, columns=["method", "folder", "condition", "gt_status"] + GROUP39
                        + ["steps_chosen", "judge_expect", "judge_hit"])


def preprocess_table(paths, states, runs):
    """4번 전처리 끔·켬 표: 영상 1장 평균의 에지 수·후보 수·면적·처리 시간(1차 계획의 항목). 39장 최종(roi.csv)과 개발·시험셋 최종.
    처리 시간은 최종 단계 39장(TIME_REPEAT번 중앙값)의 값만 쓴다. 학습 데이터의 시간은 한 번 잰 값이라 넣지 않는다."""
    rows = []
    data = runs["F39"]
    if data is not None:
        whole = data["source"][data["source"]["source"] == "all"].iloc[0]
        rows += [{"data": "39장 최종(roi.csv)", "folder": EXP["F39"]["name"], "item": item,
                  "off": whole[f"{m}_off"], "on": whole[f"{m}_on"], "diff": whole[f"{m}_diff"]} for item, m in PRE39]
    for sid in ("E5", "T"):
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


def source_table(runs):
    """6번 제공 vs 직접 촬영 표: 39장 실행마다 출처(제공·직접 촬영·전체)별 영상 단위 지표와 영상당 후보·에지 수(끔·켬)."""
    rows = []
    for sid, method in RUNS39:
        data = runs[sid]
        if data is None:
            continue
        status = gt_status(data["metrics"])
        rows += [{"method": method, "folder": EXP[sid]["name"], "source": r["source"], "gt_status": status,
                  **{c: r.get(c) for c in SOURCE39}} for _, r in data["source"].iterrows()]
    return pd.DataFrame(rows, columns=["method", "folder", "source", "gt_status"] + SOURCE39)


def degrade_table(paths, states):
    """7번 인위 저하 표: 회차(1회 비교·2회 최종) × 저하 × 끔·켬의 검출 지표, 에지·해리스·후보 수·면적, 상태 판정 적중률과 원본 기준.
    처리 시간은 최종 단계(2회) 값만 넣는다(한 번 잰 값의 평균이라 참고). 끝나지 않은 실험은 상태만 한 줄."""
    rows = []
    for round_name, prefix in DEGRADE_ROUNDS:
        for kind in et.DEGRADE_KINDS:
            exp, state = EXP[prefix + kind], states[prefix + kind]
            head = {"round": round_name, "degrade": kind, "folder": exp["name"], "status": state["status"]}
            summary = train_summary(paths, exp, state)
            if not summary:
                rows.append({**head, "preprocess": ""})
                continue
            for mode in ("off", "on"):
                times = {c: summary[mode].get(c) for c in TIME_MEANS} if prefix == "F6_" else {}
                rows.append({**head, "preprocess": mode, **{c: summary[mode].get(c) for c in DEGRADE_COLS}, **times})
    return pd.DataFrame(rows, columns=["round", "degrade", "folder", "status", "preprocess"] + DEGRADE_COLS + TIME_MEANS)


def image_table(runs):
    """8번 39장 영상별 최종 판정: 사진마다 정답 유무(없으면 '미완'), 손상 유무, 상태 판정이 고른 전처리와 사람 태그에 맞는지,
    1차 설정·최종(roi.csv)·최종(자동 ROI)의 끔·켬 판정(correct)과 최종 켬의 hit_same·hit_any·후보 수. 포트홀은 이 표로 사례를 본다."""
    columns = ["file", "source", "condition", "gt", "has_damage", "steps", "judge_expect", "judge_hit",
               "first_correct_off", "first_correct_on", "final_correct_off", "final_correct_on",
               "auto_correct_off", "auto_correct_on", "final_hit_same_on", "final_hit_any_on", "final_n_crack_on", "final_n_pothole_on"]
    base = next((runs[sid] for sid in ("F39", "F39e0", "F39a") if runs[sid] is not None), None)
    if base is None:
        return pd.DataFrame(columns=columns)
    judge = judgements(base["metrics"])
    tables = {prefix: runs[sid]["image"].set_index("file") if runs[sid] is not None else None
              for prefix, sid in (("first", "F39e0"), ("final", "F39"), ("auto", "F39a"))}
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


def saved_reject(paths):
    """지난번에 저장한 '사람 확인에서 뺀 단계'. 없으면 빈 list."""
    path = Path(paths["results"]) / REPORT_DIR / REJECT_FILE
    return [s for s in path.read_text(encoding="utf-8").strip().split(",") if s] if path.exists() else []


def build_report(paths, reject=(), command_line="python tools/run_all.py tables"):
    """있는 결과 폴더로 보고서 표 CSV(1~8번, 채택 표)와 summary.md를 results/report/에 만든다. 돌리지 않은 실험은 상태만 적는다.
    화면에 보여 줄 값(상태, 채택 표, 잠정 채택 방법, 바꿀 config 값, E0 재현, 재현 확인, 39장 정답 상태)을 돌려준다."""
    out = Path(paths["results"]) / REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    ready = readiness()
    states = {e["id"]: exp_state(e, paths, ready) for e in EXPERIMENTS}
    summaries = {sid: train_summary(paths, EXP[sid], states[sid]) for sid in ADOPT_IDS + ["E0_all", "T0", "T"]}
    adopt, champion = adoption({sid: (s or {}).get("off") for sid, s in summaries.items()}, states, reject)
    runs = {sid: run39(paths, EXP[sid], states[sid]) for sid, _ in RUNS39}
    missing, n39 = missing_gt(paths)
    tables = {"table1_stages.csv": stages_table(paths, states), "table2_by_country.csv": country_table(paths, states),
              "table3_by_condition.csv": condition_table(runs), "table4_preprocess.csv": preprocess_table(paths, states, runs),
              "table5_matching.csv": matching_table(paths, states, runs), "table6_provided_vs_own.csv": source_table(runs),
              "table7_degrade.csv": degrade_table(paths, states), "table8_39_images.csv": image_table(runs), "adoption.csv": adopt}
    for name, table in tables.items():
        table.to_csv(out / name, encoding="utf-8-sig", index=False)
    (out / REJECT_FILE).write_text(",".join(reject), encoding="utf-8")
    report = {"out": out, "states": states, "adopt": adopt, "champion": champion, "reject": list(reject),
              "changes": config_changes(EXP[champion]["opts"]), "e0": e0_reproduction(summaries["E0_all"]),
              "repro": repro_check(paths), "gt": (n39 - len(missing), n39), "runs": runs, "tables": tables,
              "summaries": summaries, "created": f"{datetime.datetime.now():%Y-%m-%d %H:%M}", "command": command_line}
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


def summary_lines(r):
    """보고서용 요약(summary.md) 줄: 결과 폴더 상태, 채택 비교와 바꿀 config 값, E0 재현, 표 1~8의 핵심 열, 시험셋 전·후, 재현 확인."""
    t, states = r["tables"], r["states"]
    has39, n39 = r["gt"]
    gt_note = "모두 있음" if has39 == n39 else f"{has39}/{n39}장 → 39장 표(3·6·8번)는 미완"
    lines = ["# 실험 결과 요약 (tools/run_all.py)", "",
             f"- 만든 시각: {r['created']} ({r['command']})",
             "- 같은 폴더에 table1~8과 adoption.csv(모든 열)가 있고, 아래는 핵심 열만 옮겼다.",
             "- 읽는 법: 정답률 옆 괄호는 기준선('검출 없음'의 정답률)이고 균형 정확도의 기준선은 50%다. 재현율은 중심 기준이고 "
             "옆 괄호는 우연을 뺀 값(실제 − 우연 대조군)이다. 영상당 오검출은 중심 기준으로 맞지 않은 후보 수다.",
             "- 처리 시간은 최종 단계 값만 쓴다(4번 표의 39장 최종 = 끔·켬 각 5번 중앙값. 7번 표의 2회는 한 번 잰 값의 평균이라 참고).",
             f"- 39장 정답 파일: {gt_note}", ""]

    lines += ["## 결과 폴더", ""] + md_table(
        ["단계", "실험", "폴더", "상태", "커밋"],
        [[STAGES[e["stage"]], f"{e['id']} {e['label']}", e["name"], states[e["id"]]["status"], states[e["id"]]["commit"] or "-"]
         for e in EXPERIMENTS])

    lines += ["## 채택 비교 (개발셋, 전처리 끔)", "",
              "각 단계를 직전 채택 단계(E0 → E1 → E1b → E2 → E3 → E3 σ 띠 전체 순서에서 그때까지 잠정 채택된 마지막 단계)와 견준다. "
              "규칙 2·3은 자동으로 보고, 규칙 1(같은 오검출 수준의 민감도)은 점수 문턱이 없어 민감도와 영상당 오검출을 나란히 두고 사람이 본다"
              "(Plan.md 3.10). 비교 기준과 설정이 둘 이상 다르면 '바뀐 설정'에 모두 적힌다.", ""]
    rows = []
    for _, a in r["adopt"].iterrows():
        if pd.isna(a["rule2"]):
            rows.append([a["stage_id"], a["base"], a["changed"], "-", "-", "-", a["verdict"]])
            continue
        rows.append([a["stage_id"], a["base"], a["changed"],
                     f"{pct(a['sens_base'])}→{pct(a['sens'])}, {num(a['fp_base'])}→{num(a['fp'])} · {a['rule1']}",
                     f"{pct(a['balanced_base'])}→{pct(a['balanced'])} ({pp(a['balanced_diff'])}) {a['rule2']}",
                     f"{pp(a['crack_net_base'])}→{pp(a['crack_net'])} ({pp(a['crack_net_diff'])}) {a['rule3']}", a["verdict"]])
    lines += md_table(["단계", "비교 기준", "바뀐 설정", "규칙 1: 민감도, 영상당 오검출", "규칙 2: 균형 정확도",
                       "규칙 3: 우연 뺀 균열 재현율", "판정"], rows)
    champion = EXP[r["champion"]]
    lines += [f"- 잠정 채택 방법: {champion['id']} {champion['label']}"
              + (f" (사람 확인에서 뺀 단계: {', '.join(r['reject'])})" if r["reject"] else ""),
              "- 그 방법으로 가려면 바꿀 config 값 (run_all은 config를 고치지 않는다. 구역 주인이 고친다):"]
    lines += [f"  - `{name} = {value!r}` ({owner})" for name, value, owner in r["changes"]] or ["  - 없음 (지금 config 그대로)"]
    lines += [""]
    if r["e0"]:
        lines += ["E0 재현 확인 (Plan.md 3.2 (5), 804장·전처리 끔)", ""] + md_table(
            ["항목", "기준", "이번", "같음"], [[k, f"{e:,}", f"{a:,}", "O" if ok else "X"] for k, e, a, ok in r["e0"]])
    elif r["summaries"].get("E0_all"):
        lines += [f"- E0 재현 확인은 학습 사진 {N_TRAIN}장일 때만 한다(이번 e0_all은 "
                  f"{int(r['summaries']['E0_all']['off']['images'])}장).", ""]

    lines += ["## 1. 단계별 표 (E0~E5, 시험셋)", "", "E6는 7번 표. 끝나지 않은 단계는 상태만 적는다.", ""]
    rows = []
    for _, s in t["table1_stages.csv"].iterrows():
        if s["preprocess"] not in ("off", "on"):
            rows.append([s["stage_id"], s["data"], "-", "-"] + ["-"] * len(METRIC_HEAD) + [s["status"]])
        else:
            rows.append([s["stage_id"], s["data"], "끔" if s["preprocess"] == "off" else "켬", int(s["images"])]
                        + metric_cells(s) + [s["status"]])
    lines += md_table(["단계", "데이터", "전처리", "장수"] + METRIC_HEAD + ["상태"], rows)

    lines += target_lines(r["summaries"])

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

    lines += ["## 6. 제공 vs 직접 촬영 (39장)", ""]
    rows = [[s["method"], s["source"], int(s["n_images"]), int(s["n_gt"]), f"{pct(s['sensitivity_off'])} / {pct(s['sensitivity_on'])}",
             f"{pct(s['specificity_off'])} / {pct(s['specificity_on'])}",
             f"{pct(s['accuracy_off'])} / {pct(s['accuracy_on'])} ({pct(s['baseline_none'])})",
             f"{num(s['n_crack_off'])} / {num(s['n_crack_on'])}", f"{num(s['n_pothole_off'])} / {num(s['n_pothole_on'])}", s["gt_status"]]
            for _, s in t["table6_provided_vs_own.csv"].iterrows()]
    lines += md_table(["방법", "출처", "장수", "정답 있는", "민감도 끔/켬", "특이도 끔/켬", "정답률 끔/켬 (기준선)",
                       "균열 후보 끔/켬", "포트홀 후보 끔/켬", "정답"], rows) if rows else ["아직 결과 없음 (최종 단계 뒤)", ""]

    lines += ["## 7. 인위 저하 표 (개발셋)", ""]
    rows = []
    for _, d in t["table7_degrade.csv"].iterrows():
        if d["preprocess"] not in ("off", "on"):
            rows.append([d["round"], DEGRADE_NAMES[d["degrade"]], "-"] + ["-"] * 7 + [d["status"]])
            continue
        judge = "-" if pd.isna(d["judge_hit_rate"]) else f"{pct(d['judge_hit_rate'])} ({pct(d['judge_base_rate'])})"
        rows.append([d["round"], DEGRADE_NAMES[d["degrade"]], "끔" if d["preprocess"] == "off" else "켬",
                     pct(d["sensitivity"]), pct(d["specificity"]), f"{pct(d['accuracy'])} ({pct(d['baseline_none'])})",
                     num(d["fp_per_image"]), f"{pct(d['crack_ctr_recall'])} ({pp(d['crack_ctr_recall_net'])})",
                     num(d["edges_mean"], 0), judge, d["status"]])
    lines += md_table(["회차", "저하", "전처리", "민감도", "특이도", "정답률 (기준선)", "영상당 오검출", "균열 재현율 (우연 뺌)",
                       "에지 수", "상태 판정 적중 (원본 기준)", "상태"], rows)

    lines += image_lines(t["table8_39_images.csv"], gt_note)
    if r["repro"]:
        lines += ["## 시험셋 재현 확인 (final_test vs test_once)", ""] + [f"- {line}" for line in r["repro"]] + [""]
    return lines


def target_lines(summaries):
    """목표 출발값(804장 1차, 전처리 끔)과 같은 시험셋의 1차·최종(전·후) 비교 표 줄. 시험 결과가 없으면 804장 값만."""
    cols = [("804장 1차 끔", summaries.get("E0_all"), "off"), ("시험 1차 끔", summaries.get("T0"), "off"),
            ("시험 1차 켬", summaries.get("T0"), "on"), ("시험 최종 끔", summaries.get("T"), "off"),
            ("시험 최종 켬", summaries.get("T"), "on")]
    cols = [(name, s[mode]) for name, s, mode in cols if s and mode in s]
    if not cols:
        return []
    items = [("영상당 오검출", lambda x: num(x["fp_per_image"])), ("특이도", lambda x: pct(x["specificity"])),
             ("균열 재현율 (우연 뺌)", lambda x: f"{pct(x['crack_ctr_recall'])} ({pp(x['crack_ctr_recall_net'])})"),
             ("민감도", lambda x: pct(x["sensitivity"])),
             ("정답률 (기준선)", lambda x: f"{pct(x['accuracy'])} ({pct(x['baseline_none'])})"),
             ("균형 정확도", lambda x: pct(x["balanced"]))]
    return (["### 목표 출발값과 시험셋 전·후", "",
             "목표의 출발값(영상당 오검출 8.1, 특이도 7%, 균열 재현율 4.9%)은 804장 값이라, 같은 시험셋에서 1차 방법(e0_test)과 최종(test_once)을 견준다.", ""]
            + md_table(["값"] + [name for name, _ in cols], [[item] + [fn(row) for _, row in cols] for item, fn in items]))


def condition_lines(table, runs):
    """3번 조건별 표 줄(최종 roi.csv만, 나머지는 CSV)과 E5 상태 판정 검증(같은 장소 정상·흔들림 쌍)."""
    lines = ["## 3. 조건별 표 (39장 최종 roi.csv)", "",
             "조건은 사람이 붙인 태그(직접 촬영은 이름, 제공 13장은 roi.csv)이고, '고른 전처리'는 상태 판정(analyze)의 결과다. "
             "1차 설정·자동 ROI 행은 table3_by_condition.csv에 있다.", ""]
    rows = [[c["condition"], int(c["n_images"]), int(c["n_gt"]), f"{pct(c['sensitivity_off'])} / {pct(c['sensitivity_on'])}",
             f"{pct(c['specificity_off'])} / {pct(c['specificity_on'])}",
             f"{pct(c['accuracy_off'])} / {pct(c['accuracy_on'])} ({pct(c['baseline_none'])})",
             c["steps_chosen"], f"{c['judge_expect']} {c['judge_hit']}".strip() or "-", c["gt_status"]]
            for _, c in table.iterrows() if c["method"] == "최종(roi.csv)"]
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
    """8번 39장 영상별 최종 판정 표 줄. 정답이 없는 사진은 '없음(미완)'이고 판정은 -."""
    lines = [f"## 8. 39장 영상별 최종 판정 (정답 파일: {gt_note})", "",
             "O 맞힘, X 틀림, - 정답 없음. 손상 영상은 같은 종류 후보가 맞으면, 손상 없는 영상은 후보가 0개면 맞힘(중심 기준).", ""]
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
    """표를 만든 뒤 화면에 핵심만: 표 위치, (비교·표만) 채택 판정과 바꿀 config 값·E0 재현, (재현) 같은지."""
    print(f"\n표: {report['out'].as_posix()}/ (summary.md, table1~8, adoption.csv)")
    has39, n39 = report["gt"]
    if has39 < n39:
        print(f"39장 정답 파일 {has39}/{n39}장: 39장 표는 '미완'")
    if stage in ("compare", "tables"):
        print("채택 비교 (개발셋, 전처리 끔):")
        for _, a in report["adopt"].iterrows():
            print(f"  {a['stage_id']:<4} 비교 기준 {a['base']:<4} {a['verdict']}")
        champion = EXP[report["champion"]]
        print(f"잠정 채택 방법: {champion['id']} {champion['label']}")
        print("바꿀 config 값 (config는 고치지 않음):" if report["changes"] else "바꿀 config 값: 없음")
        for name, value, owner in report["changes"]:
            print(f"  {name} = {value!r}  ({owner})")
        for key, expect, actual, ok in report["e0"]:
            print(f"E0 재현 {key}: 기준 {expect:,} / 이번 {actual:,} {'같음' if ok else '다름'}")
    if stage == "test":
        print("\n".join(target_lines(report["summaries"])))
    if stage == "repro":
        for line in report["repro"] or ["test_once·final_test 결과가 없어 견주지 못함"]:
            print(f"재현 확인 {line}")


def main(argv=None, paths=None, ask=input):
    """명령행의 단계를 돈다. 끝 코드: 0 = 끝남(구현 전이라 건너뛴 실험 포함), 1 = 실패한 실험이 있음, 2 = 확인에서 멈춤·거부.
    paths·ask는 시험에서 임시 폴더와 가짜 입력을 쓰려고 둔 인자이고, 명령행 실행은 기본값을 쓴다."""
    args = parse_args(argv)
    paths = paths or default_paths()
    reject = args.reject if args.reject is not None else saved_reject(paths)
    command_line = "python tools/run_all.py " + " ".join(argv if argv is not None else sys.argv[1:])
    if args.stage == "tables":
        if args.dry_run:
            print(f"dry-run: {Path(paths['results']).as_posix()}/{REPORT_DIR}/에 표를 다시 만든다(실험은 돌리지 않음)")
            return 0
        print_report(build_report(paths, reject, command_line), "tables")
        return 0
    return run_stage(args.stage, args, paths, reject, command_line, ask)


if __name__ == "__main__":
    sys.exit(main())
