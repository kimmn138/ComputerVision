"""(파트 C) 보고서 그림 1~6과 발표용 그림을 최종 설정으로 다시 저장한다(demo.py와 같은 그리기 함수).

사용법 (맨 위 폴더에서):
    python tools/report_figures.py              # results/demo/ 에 전부
    python tools/report_figures.py --only fig4  # 이름이 fig4로 시작하는 것만

demo.py --save와 같은 화면에 보고서용으로 두 가지를 더한다(결과 숫자는 바꾸지 않음).
- 정답 상자(그림 2): gt/<사진>.csv의 상자를 흰색 2px로 640px 패널에 그린다(v9 그림 2와 같은 픽셀).
- 번호판 가림(그림 4): 640px 사진 좌표의 상자를 모자이크로 가린다(v9 그림 4와 같은 세 곳).
세로 사진은 아래 띠 줄이 그림 폭을 넘으므로 나눠 쓴다. 그림 4는 v9와 같은 크기(가로 1050px)가 되게 화면 크기를 정한다.
results/의 그림은 손으로 고치지 않고 이 도구로 다시 만든다(CONTRIBUTING 4장).
"""
import argparse
import contextlib
import io
import os
import sys

import cv2 as cv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import demo  # noqa: E402
from src.evaluate import load_gt  # noqa: E402

OUT_DIR = "results/demo"
P, O = "data/provided", "data/own"
GT_COLOR = (255, 255, 255)                          # v9 그림 2의 정답 상자(흰색)
GT_OFFSETS = ((0, 0), (1, -1))                      # 1px 상자를 이만큼 옮겨 두 번 그림 → 2px. v9와 같은 픽셀(왼·오른 변 x·x+1, 위 변 y−1·y)
PLATE_BLOCK = 6                                       # px(640px 사진 기준), 번호판 모자이크 한 칸. 번호판 높이 11px가 두 칸이 되어 글자가 남지 않음
# own_19_none_normal의 번호판(640px 사진 좌표 x, y, w, h): 가까운 차 1곳(글자 x 46~86, y 26~36에 여유) + 먼 차 2곳. v9 그림 4와 같은 곳
OWN19_PLATES = ((44, 24, 46, 15), (278, 0, 26, 12), (362, 0, 26, 12))
V9_FIG4_SCREEN = (1050 / demo.C.DEMO_SCREEN_MARGIN, 1400)   # 패널 가로 525px(1050 / 2) = v9 그림 4와 같은 크기
# (이름, 사진, demo 옵션, 정답 상자?, 번호판 상자, 화면 크기 또는 None, 아래 줄 나누기?)
FIGURES = (
    ("fig1_China_MotorBike_002209", f"{P}/China_MotorBike_002209.jpg", ["--roi", "auto"], False, (), None, False),
    ("fig2_Japan_000156", f"{P}/Japan_000156.jpg", ["--roi", "auto"], True, (), None, False),
    ("fig3a_China_MotorBike_002159_band_fixed", f"{P}/China_MotorBike_002159.jpg", ["--roi", "auto", "--auto-band", "off"],
     False, (), None, False),
    ("fig3b_China_MotorBike_002159_band_auto", f"{P}/China_MotorBike_002159.jpg", ["--roi", "auto"], False, (), None, False),
    ("fig4_own_19_none_normal", f"{O}/own_19_none_normal.jpg", ["--roi", "auto"], False, OWN19_PLATES, V9_FIG4_SCREEN, True),
    ("fig5a_China_MotorBike_002209_lane_before", f"{P}/China_MotorBike_002209.jpg", ["--roi", "auto", "--set", "ROAD_KEEP_RULE=center"],
     False, (), None, False),
    ("fig5b_China_MotorBike_002209_lane_ga", f"{P}/China_MotorBike_002209.jpg",
     ["--roi", "auto", "--set", "ROAD_KEEP_RULE=center", "--set", "ROAD_LANE_BRIDGE=True"], False, (), None, False),
    ("fig6a_Japan_000107_keep_center", f"{P}/Japan_000107.jpg", ["--roi", "auto", "--set", "ROAD_KEEP_RULE=center"], False, (), None, False),
    ("fig6b_Japan_000107_keep_ratio", f"{P}/Japan_000107.jpg", ["--roi", "auto"], False, (), None, False),
    ("talk_own_16_pothole_blur_tau_rejected", f"{O}/own_16_pothole_blur.jpg", ["--roi", "auto", "--set", "DRAW_TAU_REJECTED=True"],
     False, (), None, True),
    ("talk_own_19_none_normal_tau_rejected", f"{O}/own_19_none_normal.jpg", ["--roi", "auto", "--set", "DRAW_TAU_REJECTED=True"],
     False, OWN19_PLATES, V9_FIG4_SCREEN, True),
)


def mosaic(img, boxes, block=PLATE_BLOCK):
    """상자마다 block px 칸의 평균색으로 뭉갠 새 영상을 돌려준다(받은 영상은 바꾸지 않음). 번호판 글자가 남지 않게 가린다."""
    out = img.copy()
    for x, y, w, h in boxes:
        part = out[y:y + h, x:x + w]
        if not part.size:
            continue
        small = cv.resize(part, (max(1, part.shape[1] // block), max(1, part.shape[0] // block)), interpolation=cv.INTER_AREA)
        out[y:y + h, x:x + w] = cv.resize(small, (part.shape[1], part.shape[0]), interpolation=cv.INTER_NEAREST)
    return out


def draw_gt(img, gt):
    """정답 상자(640px 전체 좌표)를 흰색 2px로 그린 새 영상(받은 영상은 바꾸지 않음). OpenCV 두께 2는 선 양쪽으로 퍼져 3px가 되므로
    1px 상자를 GT_OFFSETS만큼 옮겨 두 번 그린다."""
    out = img.copy()
    for x, y, w, h, _ in gt or []:
        for dx, dy in GT_OFFSETS:
            cv.rectangle(out, (x + dx, y + dy), (x + w + dx, y + h + dy), GT_COLOR, 1)
    return out


def decorator(path, with_gt, plates):
    """demo.demo_one에 넘길 패널 꾸미기 함수(번호판 가림 → 정답 상자). 꾸밀 것이 없으면 None(demo 화면 그대로)."""
    if not with_gt and not plates:
        return None
    gt = load_gt(f"gt/{os.path.splitext(os.path.basename(path))[0]}.csv") if with_gt else None
    return lambda img, out: draw_gt(mosaic(img, plates), gt)


def save_figure(name, path, options, with_gt, plates, screen, wrap, out_dir=OUT_DIR):
    """그림 하나를 demo와 같은 함수로 그려 out_dir/name.png로 저장하고, 콘솔 출력은 name.txt에 둔다. demo의 끝 코드를 돌려준다."""
    target = os.path.join(out_dir, f"{name}.png")
    args = demo.parse_args([path, *options, "--save", target])
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        code = demo.demo_one(path, args, decorate=decorator(path, with_gt, plates), screen=screen, wrap=wrap)
    with open(os.path.join(out_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
        f.write(printed.getvalue())
    return code


def main(argv=None):
    """FIGURES를 차례로 저장한다. --only면 이름이 그 글자로 시작하는 것만. 실패가 있으면 1."""
    parser = argparse.ArgumentParser(description="보고서 그림 1~6과 발표용 그림을 최종 설정으로 다시 저장")
    parser.add_argument("--only", help="이름 앞부분(예: fig4)")
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    codes = []
    for name, path, options, with_gt, plates, screen, wrap in FIGURES:
        if args.only and not name.startswith(args.only):
            continue
        code = save_figure(name, path, options, with_gt, plates, screen, wrap, args.out)
        codes.append(code)
        print(f"{'저장' if code == 0 else '실패'}: {os.path.join(args.out, name)}.png")
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
