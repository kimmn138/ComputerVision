"""(파트 C) 발표 시연: 영상 1장을 전처리 끔·켬으로 처리해 검출 결과를 한 창에 나란히 띄운다.
사용법: 맨 위 폴더에서  python demo.py <영상 경로> [roi_top roi_bottom]
- ROI 비율을 빼면 data/roi.csv의 값, 거기에도 없으면 config 기본값을 쓴다.
- 콘솔에 고른 전처리와 끔·켬의 row(9열)를 출력하고, 창은 아무 키나 누르거나 X로 닫는다.
화면 표시는 이 파일에서만 한다(src/ 함수는 창을 띄우지 않음).
"""
import argparse
import os
import sys

import cv2 as cv
import numpy as np
import pandas as pd

import run_experiment as rx
from src import analyze, io_utils, pipeline, visualize
from src import config as C

ROI_PATH = "data/roi.csv"
WINDOW_NAME = "demo"            # 윈도 HighGUI에서 한글 창 제목이 깨질 수 있어 ASCII로
KEY_POLL_MS = 100               # 키·창 닫힘을 확인하는 간격
FONT_NAME = "sans"              # OpenCV 5 내장 유니코드 글꼴
LABEL_COLOR = (255, 255, 255)   # 제목 띠 글자: 흰색 (BGR)
LABEL_X = 8


def parse_args(argv=None):
    """영상 경로와 선택 ROI 비율 2개를 읽고, 비율이 1개이거나 0 <= top < bottom <= 1이 아니면 사용법과 함께 멈춘다."""
    parser = argparse.ArgumentParser(description="전처리 끔·켬 검출 결과를 한 창에 띄우는 발표 시연")
    parser.add_argument("path", help="영상 경로")
    parser.add_argument("roi", nargs="*", type=float, metavar="roi_top roi_bottom",
                        help="노면 비율 2개(0~1). 빼면 roi.csv, 없으면 config 기본값")
    args = parser.parse_args(argv)
    if len(args.roi) not in (0, 2):
        parser.error(f"ROI 비율은 0개 또는 2개여야 함 ({len(args.roi)}개)")
    if args.roi and not 0 <= args.roi[0] < args.roi[1] <= 1:
        parser.error(f"ROI 비율은 0 <= top < bottom <= 1 이어야 함 ({args.roi[0]}, {args.roi[1]})")
    return args


def choose_roi(path, roi_args, roi_table):
    """명령행 비율 → roi.csv → config 기본값 순서로 노면 범위를 정한다. (top, bottom, 출처 글)을 돌려준다."""
    if roi_args:
        return roi_args[0], roi_args[1], "명령행"
    roi, used_default = rx.get_roi(os.path.basename(path), roi_table)
    return roi["top"], roi["bottom"], "config 기본값" if used_default else "roi.csv"


def has_unicode_text():
    """창 안에 한글을 쓸 수 있는지 확인한다. OpenCV 5의 FontFace가 있고, 실제로 '한글'을 그린 결과가
    '??'(그릴 수 없는 글자의 대체)와 다를 때만 참이다. 4.x는 Hershey 글꼴뿐이라 거짓."""
    if not hasattr(cv, "FontFace"):
        return False
    try:
        face = cv.FontFace(FONT_NAME)
        korean, question = np.zeros((40, 120), np.uint8), np.zeros((40, 120), np.uint8)
        cv.putText(korean, "한글", (4, 30), 255, face, C.DEMO_FONT_SIZE)
        cv.putText(question, "??", (4, 30), 255, face, C.DEMO_FONT_SIZE)
    except cv.error:
        return False
    return not np.array_equal(korean, question)


def screen_size():
    """화면 크기(가로, 세로 px)를 읽는다. 윈도는 표준 라이브러리 ctypes로 읽고, 그 밖이나 실패하면 config 기본값."""
    if sys.platform == "win32":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            w, h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
            if w > 0 and h > 0:
                return w, h
        except (AttributeError, OSError):
            pass
    return C.DEMO_SCREEN_FALLBACK


def fit_to_screen(img, max_w, max_h):
    """영상이 max_w × max_h 안에 들어가게 비율을 지켜 줄인 새 영상을 돌려준다. 이미 들어가면 키우지 않고 복사본만."""
    h, w = img.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    if scale >= 1.0:
        return img.copy()
    return cv.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv.INTER_AREA)


def add_label(img, text, unicode_ok):
    """영상 위에 검은 제목 띠를 붙인 새 영상을 돌려준다. 영상을 가리지 않게 글자는 띠에만 쓴다.
    한글을 못 쓰는 OpenCV면 text 대신 같은 뜻의 영문을 받아 Hershey 글꼴로 쓴다."""
    band = np.zeros((C.DEMO_LABEL_HEIGHT, img.shape[1], 3), np.uint8)
    y = (C.DEMO_LABEL_HEIGHT + C.DEMO_FONT_SIZE) // 2 - 2       # 글자 기준선을 띠 가운데쯤에
    if unicode_ok:
        cv.putText(band, text, (LABEL_X, y), LABEL_COLOR, cv.FontFace(FONT_NAME), C.DEMO_FONT_SIZE)
    else:
        cv.putText(band, text, (LABEL_X, y), cv.FONT_HERSHEY_SIMPLEX, C.DEMO_FONT_SIZE / 30, LABEL_COLOR, 1, cv.LINE_AA)
    return cv.vconcat([band, img])


def build_view(off_img, on_img, labels, screen, unicode_ok):
    """끔·켬 그림을 같은 비율로 화면에 맞게 줄이고 각각 제목 띠를 붙여 가로로 잇는다.
    라벨은 줄인 뒤에 붙여 화면 크기와 상관없이 글자 크기가 같다."""
    max_w = screen[0] * C.DEMO_SCREEN_MARGIN / 2
    max_h = screen[1] * C.DEMO_SCREEN_MARGIN - C.DEMO_LABEL_HEIGHT
    panels = [add_label(fit_to_screen(img, max_w, max_h), text, unicode_ok)
              for img, text in zip((off_img, on_img), labels)]
    return cv.hconcat(panels)


def make_labels(steps_text, unicode_ok):
    """끔·켬 패널 제목 2개. 한글을 못 쓰면 영문으로(고른 전처리 이름은 콘솔에서 봄)."""
    if unicode_ok:
        return "전처리 끔", f"전처리 켬: {steps_text}"
    return "Preprocess OFF", "Preprocess ON"


def report_table(row_off, row_on):
    """끔·켬 row를 약속된 9열 순서로 놓고 차이(켬 − 끔) 줄을 더한 표.
    차이 줄을 열마다 따로 계산해 붙여야 개수 열이 정수로 남는다(.loc로 한 줄을 넣으면 시간 열 때문에 모두 실수가 됨)."""
    diff = {k: row_on[k] - row_off[k] for k in rx.ROW_COLUMNS}
    return pd.DataFrame([row_off, row_on, diff], index=["끔", "켬", "차이"])[rx.ROW_COLUMNS]


def show(img):
    """창을 띄우고 아무 키나 누르거나 X로 닫을 때까지 기다린다.
    waitKey(0)만 쓰면 X로 닫았을 때 윈도에서 영원히 기다리므로, 짧게 기다리며 창이 보이는지 같이 본다."""
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
    cv.imshow(WINDOW_NAME, img)
    while cv.getWindowProperty(WINDOW_NAME, cv.WND_PROP_VISIBLE) >= 1:
        if cv.waitKey(KEY_POLL_MS) != -1:
            break
    cv.destroyAllWindows()


def main(argv=None):
    """영상 1장을 끔·켬으로 처리해 콘솔에 전처리와 row를 출력하고 결과 두 장을 한 창에 띄운다. 종료 코드를 돌려준다."""
    args = parse_args(argv)
    try:
        bgr = io_utils.load_image(args.path)
        top, bottom, roi_source = choose_roi(args.path, args.roi, io_utils.load_roi_table(ROI_PATH))
    except FileNotFoundError:
        print(f"파일을 읽을 수 없음: {args.path}")
        return 1
    except ValueError as e:
        print(f"{ROI_PATH} 형식 오류: {e}")
        return 1

    outs = {use: pipeline.run_pipeline(bgr, top, bottom, use) for use in (False, True)}
    steps_text = analyze.steps_to_text(outs[True]["steps"])
    print(f"영상: {args.path}")
    print(f"ROI: top {top}, bottom {bottom} ({roi_source})")
    print(f"고른 전처리: {steps_text}")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(report_table(outs[False]["row"], outs[True]["row"]))

    drawn = [visualize.draw_candidates(o["small"], o["cracks"], o["potholes"], o["y0"]) for o in outs.values()]
    unicode_ok = has_unicode_text()
    show(build_view(drawn[0], drawn[1], make_labels(steps_text, unicode_ok), screen_size(), unicode_ok))
    return 0


if __name__ == "__main__":
    sys.exit(main())
