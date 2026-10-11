"""발표 시연: 사진 1장을 전처리 끔·켬으로 처리해 노면 마스크와 검출 결과를 한 창에 나란히 띄운다.
실행: python demo.py                → 파일 선택 창에서 사진을 고름 (창을 닫으면 다시 고르고, 취소하면 끝)
      python demo.py <사진 경로>    → 그 사진을 바로 띄움   (옵션은 --help 참고)
창에는 노면(초록)·균열(빨강)·포트홀(파랑)이 표시되고, 아무 키나 누르거나 X로 닫는다.
"""
import argparse
import glob
import math
import os
import sys

import cv2 as cv
import numpy as np
import pandas as pd

import run_experiment as rx
from src import analyze, io_utils, pipeline, visualize
from src import config as C

ROI_PATH = "data/roi.csv"
DEFAULT_SOURCES = ("config 기본값", "기본 띠, --roi auto")    # 기본 띠를 쓴 경우의 출처 글 (이때만 자동 띠를 씀)
WINDOW_NAME = "demo"            # 한글 창 제목은 깨질 수 있어 영문으로
KEY_POLL_MS = 100               # ms, 키 입력·창 닫힘을 확인하는 간격
FONT_NAME = "sans"              # OpenCV 5 내장 유니코드 글꼴
LABEL_COLOR = (255, 255, 255)   # 제목 띠 글자: 흰색 (BGR)
LABEL_X = 8
DATA_DIR = "data"
EXCLUDE_DIRS = ("train",)       # 학습 사진 폴더는 목록에서 뺌
DIALOG_TITLE = "시연할 사진 고르기 (취소하면 끝)"
DIALOG_TYPES = [("사진", "*.jpg *.jpeg *.png *.bmp *.heic *.heif"), ("모든 파일", "*.*")]
HEIC_EXTS = (".heic", ".heif")  # 아이폰 사진 형식 (OpenCV로 못 읽음)
# 화면 아래에 적을 상태 지표 이름 (한글, 영문)
QUALITY_NAMES = {"mean": ("평균", "mean"), "std": ("표준편차", "std"), "lap_var": ("라플라시안 분산", "lap_var"),
                 "noise": ("잡티", "noise"), "blur_ratio": ("재흐림 비율", "blur_ratio")}


def parse_args(argv=None):
    """명령행 옵션을 읽는다. 잘못된 조합이나 범위 밖 ROI 비율은 사용법과 함께 멈춘다."""
    parser = argparse.ArgumentParser(description="전처리 끔·켬의 노면 마스크와 검출 결과를 한 창에 띄우는 발표 시연")
    parser.add_argument("path", nargs="?", help="영상 경로. 빼면 파일 선택 창(--list면 번호 목록)에서 고름")
    parser.add_argument("roi", nargs="*", type=float, metavar="roi_top roi_bottom",
                        help="노면 비율 2개(0~1). 빼면 roi.csv, 없으면 config 기본값")
    parser.add_argument("--roi", dest="roi_mode", choices=rx.ROI_MODES,
                        help="노면 띠: csv = roi.csv에 사진마다 정한 범위, auto = roi.csv 없이 기본 띠(run_experiment와 같음). "
                             "빼면 경로·--list는 csv, 선택 창은 auto")
    parser.add_argument("--mask", choices=("on", "off"), help="노면 마스크(USE_ROAD_MASK)를 이번 실행에만 켬·끔. 빼면 config")
    parser.add_argument("--detect", choices=("legacy", "dark"), help="검출 방식(DETECT_MODE)을 이번 실행에만 바꿈. 빼면 config")
    parser.add_argument("--auto-band", choices=("on", "off"),
                        help="자동 띠 높이(AUTO_BAND_TOP)를 이번 실행에만 켬·끔. 켜면 노면만 이어지는 만큼 띠 위쪽 경계를 위로 넓힘. 빼면 config")
    parser.add_argument("--band-texture", choices=("on", "off"),
                        help="자동 띠의 결 조건(AUTO_BAND_USE_TEXTURE)을 이번 실행에만 켬·끔. 켜면 결이 시작 띠 노면과 비슷한 구간만 넓힘")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="config 값 하나를 이번 실행에만 바꿈(여러 번 가능). 예: --set ROAD_LANE_BRIDGE=True. 빼면 config")
    parser.add_argument("--save", metavar="그림 경로", help="창을 띄우지 않고 같은 화면을 그림 파일(.png)로 저장")
    parser.add_argument("--list", action="store_true", help="선택 창 대신 data 아래 사진 목록에서 번호로 고름(예전 방식)")
    parser.set_defaults(roi_top=None)       # run_experiment.runtime_overrides가 읽는 값
    args = parser.parse_args(argv)
    rx.check_sets(parser, args.set)
    if args.list and args.path is not None:
        parser.error("--list는 사진 경로 없이 쓸 것(경로를 주면 목록이 필요 없음)")
    if args.save and args.path is None and not args.list:
        parser.error("--save는 사진 경로나 --list와 함께 쓸 것(선택 창에서는 고를 때마다 같은 파일을 덮어씀)")
    if args.roi_mode is None:
        args.roi_mode = "auto" if args.path is None and not args.list else "csv"
    if len(args.roi) not in (0, 2):
        parser.error(f"ROI 비율은 0개 또는 2개여야 함 ({len(args.roi)}개)")
    if args.roi and not 0 <= args.roi[0] < args.roi[1] <= 1:
        parser.error(f"ROI 비율은 0 <= top < bottom <= 1 이어야 함 ({args.roi[0]}, {args.roi[1]})")
    if args.roi and args.roi_mode == "auto":
        parser.error("ROI 비율과 --roi auto는 함께 줄 수 없음(어느 띠를 쓸지 헷갈림). 둘 중 하나만 줄 것")
    return args


def list_images(root=DATA_DIR):
    """root 아래 사진(.jpg) 경로를 이름순으로 모은다(EXCLUDE_DIRS 폴더 제외)."""
    paths = glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
    return sorted(p for p in paths if os.path.relpath(p, root).split(os.sep)[0] not in EXCLUDE_DIRS)


def pick_image(root=DATA_DIR, input_fn=input):
    """사진 목록을 번호로 보여 주고 고른 사진의 경로를 돌려준다. 빈칸이나 q면 None."""
    paths = list_images(root)
    if not paths:
        print(f"{root} 아래에 사진(.jpg)이 없음. 맨 위 폴더에서 실행했는지 확인")
        return None
    for number, path in enumerate(paths, start=1):
        print(f"{number:3d}. {path.replace(os.sep, '/')}")
    while True:
        try:
            answer = input_fn(f"번호 입력 (1~{len(paths)}, 빈칸이나 q는 종료): ").strip()
        except EOFError:
            return None
        if answer in ("", "q"):
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(paths):
            return paths[int(answer) - 1]
        print("목록에 있는 번호를 입력")


def make_chooser(root_dir=DATA_DIR):
    """파일 선택 창 함수 (choose, warn, close)를 만든다. tkinter를 쓸 수 없으면 None.
    choose()는 고른 경로(취소면 None), warn(글)은 안내 창, close()는 창 정리."""
    try:
        import tkinter
        from tkinter import filedialog, messagebox
        root = tkinter.Tk()
    except Exception:                                   # tkinter 없음, 화면 없음 등
        return None
    root.withdraw()
    root.attributes("-topmost", True)                  # 선택 창이 다른 창 뒤에 숨지 않게
    folder = {"dir": os.path.abspath(root_dir) if os.path.isdir(root_dir) else os.getcwd()}

    def choose():
        root.lift()
        root.focus_force()
        path = filedialog.askopenfilename(parent=root, title=DIALOG_TITLE, initialdir=folder["dir"], filetypes=DIALOG_TYPES)
        if not path:                                    # 취소면 "" 또는 ()
            return None
        folder["dir"] = os.path.dirname(path)
        return path

    def warn(text):
        messagebox.showwarning("시연", text, parent=root)

    return choose, warn, root.destroy


def read_error_text(path):
    """사진을 읽지 못했을 때 보여 줄 안내 문구를 만든다."""
    name = os.path.basename(path)
    if os.path.splitext(path)[1].lower() in HEIC_EXTS:
        return (f"아이폰 HEIC 사진은 읽을 수 없습니다: {name}\n"
                "JPG로 바꿔서 골라 주세요. (아이폰 설정 → 카메라 → 포맷 → '높은 호환성'으로 찍거나, "
                "사진을 메일·메신저로 보내면 jpg로 바뀝니다)")
    if not os.path.exists(path):
        return f"파일이 없습니다: {path}"
    return f"사진 파일이 아니거나 깨진 파일이라 읽을 수 없습니다: {name}\n(jpg·png 사진을 골라 주세요)"


def choose_roi(path, roi_args, roi_table, mode="csv"):
    """명령행 비율 → roi.csv → 기본 띠 순서로 노면 띠 범위를 정한다. (top, bottom, 출처 글)을 돌려준다."""
    if roi_args:
        return roi_args[0], roi_args[1], "명령행"
    roi, used_default = rx.get_roi(os.path.basename(path), roi_table, mode)
    if mode == "auto":
        return roi["top"], roi["bottom"], "기본 띠, --roi auto"
    return roi["top"], roi["bottom"], "config 기본값" if used_default else "roi.csv"


def has_unicode_text():
    """OpenCV 창에 한글을 그릴 수 있는지 확인한다(OpenCV 5의 FontFace가 필요)."""
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
    """화면 크기(가로, 세로 px)를 읽는다. 못 읽으면 config 기본값."""
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
    """영상이 max_w × max_h 안에 들어가도록 비율을 지켜 줄인다(키우지는 않음)."""
    h, w = img.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    if scale >= 1.0:
        return img.copy()
    return cv.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv.INTER_AREA)


def text_band(width, text, unicode_ok):
    """글자 한 줄을 쓴 검은 띠를 만든다. 한글을 못 쓰면 기본 영문 글꼴로 쓴다."""
    band = np.zeros((C.DEMO_LABEL_HEIGHT, width, 3), np.uint8)
    y = (C.DEMO_LABEL_HEIGHT + C.DEMO_FONT_SIZE) // 2 - 2       # 글자를 띠 가운데쯤에
    if unicode_ok:
        cv.putText(band, text, (LABEL_X, y), LABEL_COLOR, cv.FontFace(FONT_NAME), C.DEMO_FONT_SIZE)
    else:
        cv.putText(band, text, (LABEL_X, y), cv.FONT_HERSHEY_SIMPLEX, C.DEMO_FONT_SIZE / 30, LABEL_COLOR, 1, cv.LINE_AA)
    return band


def add_label(img, text, unicode_ok):
    """영상 위에 제목 띠를 붙인 새 영상을 돌려준다."""
    return cv.vconcat([text_band(img.shape[1], text, unicode_ok), img])


def build_view(off_img, on_img, labels, screen, unicode_ok, footer=()):
    """끔·켬 그림을 화면에 맞게 줄여 제목 띠와 함께 가로로 잇고, 아래에 footer 줄을 붙인다."""
    max_w = screen[0] * C.DEMO_SCREEN_MARGIN / 2
    max_h = screen[1] * C.DEMO_SCREEN_MARGIN - C.DEMO_LABEL_HEIGHT * (1 + len(footer))
    panels = [add_label(fit_to_screen(img, max_w, max_h), text, unicode_ok)
              for img, text in zip((off_img, on_img), labels)]
    view = cv.hconcat(panels)
    return cv.vconcat([view] + [text_band(view.shape[1], line, unicode_ok) for line in footer])


def make_labels(steps_text, unicode_ok):
    """끔·켬 패널 제목 2개를 만든다. 한글을 못 쓰면 영문."""
    if unicode_ok:
        return "전처리 끔", f"전처리 켬: {steps_text}"
    return "Preprocess OFF", "Preprocess ON"


def format_value(value):
    """상태 지표 값을 짧은 글로 바꾼다(NaN은 '-', 100 이상은 정수, 그 밖은 소수 한 자리)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.0f}" if abs(value) >= 100 else f"{value:.1f}"


def make_footer(quality, top, bottom, roi_source, unicode_ok):
    """화면 아래 두 줄을 만든다: 상태 지표(전처리를 고른 근거)와 이번 시연의 설정."""
    lang = 0 if unicode_ok else 1
    values = [f"{QUALITY_NAMES.get(key, (key, key))[lang]} {format_value(value)}" for key, value in quality.items()]
    mask = C.USE_ROAD_MASK
    extra = road_extra()
    if unicode_ok:
        return ["상태 판정: " + " · ".join(values),
                f"설정: ROI {top:.2f}~{bottom:.2f}({roi_source}) · 노면 마스크 {'켬' if mask else '끔'} · 검출 {C.DETECT_MODE}"
                + "".join(f" · {ko}" for ko, _ in extra)]
    return ["quality: " + " / ".join(values),
            f"ROI {top:.2f}-{bottom:.2f}{' (auto band)' if C.AUTO_BAND_TOP else ''} / mask {'on' if mask else 'off'} / detect {C.DETECT_MODE}"
            + "".join(f" / {en}" for _, en in extra)]


def road_extra():
    """설정 줄에 덧붙일 노면 마스크 옵션 [(한글, 영문)]을 만든다(남기기 규칙, 켜진 차선 옵션)."""
    extra = []
    if C.USE_ROAD_MASK:
        extra.append((f"남기기 {C.ROAD_KEEP_RULE}", f"keep {C.ROAD_KEEP_RULE}"))
    lanes = [name for name, on in (("가", C.ROAD_LANE_BRIDGE), ("나", C.ROAD_CLOSE_BEFORE_SEED)) if on]
    if lanes:
        extra.append(("차선 " + "+".join(f"({n})" for n in lanes),
                      "lane " + "+".join({"가": "bright", "나": "preclose"}[n] for n in lanes)))
    return extra


def save_image(path, img):
    """화면 그림을 파일로 저장한다(한글 경로도 지원)."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    ok, data = cv.imencode(os.path.splitext(path)[1] or ".png", img)
    if not ok:
        raise ValueError(f"그림으로 바꿀 수 없음: {path}")
    data.tofile(path)


def report_table(row_off, row_on):
    """끔·켬 row와 차이(켬 − 끔)를 3줄 표로 만든다."""
    diff = {k: row_on[k] - row_off[k] for k in rx.ROW_COLUMNS}
    return pd.DataFrame([row_off, row_on, diff], index=["끔", "켬", "차이"])[rx.ROW_COLUMNS]


def show(img):
    """창을 맨 앞에 띄우고, 아무 키나 누르거나 X로 닫을 때까지 기다린다."""
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
    cv.imshow(WINDOW_NAME, img)
    try:
        cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_TOPMOST, 1)
    except (AttributeError, cv.error):
        pass
    while cv.getWindowProperty(WINDOW_NAME, cv.WND_PROP_VISIBLE) >= 1:     # X로 닫아도 빠져나오게 창 상태를 같이 봄
        if cv.waitKey(KEY_POLL_MS) != -1:
            break
    cv.destroyAllWindows()


def demo_one(path, args, show_fn=None, warn=None):
    """사진 1장을 끔·켬으로 처리해 결과를 콘솔에 출력하고 한 창에 띄운다(--save면 파일로 저장).
    사진을 못 읽으면 안내 문구를 내고 1, 정상이면 0을 돌려준다."""
    def report(text):
        print(text)
        if warn is not None:
            warn(text)

    try:
        bgr = io_utils.load_image(path)
    except FileNotFoundError:
        report(read_error_text(path))
        return 1
    try:
        top, bottom, roi_source = choose_roi(path, args.roi, io_utils.load_roi_table(ROI_PATH), args.roi_mode)
    except ValueError as e:
        report(f"{ROI_PATH} 형식 오류: {e}")
        return 1

    used_default = roi_source in DEFAULT_SOURCES             # 직접 정한 띠는 자동으로 넓히지 않음
    backup = rx.apply_overrides({**rx.runtime_overrides(args), **rx.band_overrides(used_default)})
    try:
        try:
            outs = {use: pipeline.run_pipeline(bgr, top, bottom, use) for use in (False, True)}
        except NotImplementedError as e:
            report(f"아직 구현되지 않은 설정이라 시연할 수 없음: {e}")
            return 1
        unicode_ok = has_unicode_text()
        steps_text = analyze.steps_to_text(outs[True]["steps"])
        if C.AUTO_BAND_TOP:                         # 실제로 쓴 띠 경계를 표시 (끔·켬이 같음)
            texture = "+결" if C.AUTO_BAND_USE_TEXTURE else ""
            roi_source, top = f"{roi_source}, 자동 띠{texture} {top:.2f}에서", outs[True]["top"]
        footer = make_footer(outs[True]["quality"], top, bottom, roi_source, unicode_ok)
        print(f"영상: {path}")
        print(f"ROI: top {top}, bottom {bottom} ({roi_source})")
        band_text = f"켬(결 조건 {'켬' if C.AUTO_BAND_USE_TEXTURE else '끔'})" if C.AUTO_BAND_TOP else "끔"
        print(f"설정: 노면 마스크 {'켬' if C.USE_ROAD_MASK else '끔'}, 검출 {C.DETECT_MODE}, 자동 띠 높이 {band_text}"
              + "".join(f", {ko}" for ko, _ in road_extra()))
        print(make_footer(outs[True]["quality"], top, bottom, roi_source, True)[0])
        print(f"고른 전처리: {steps_text}")
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(report_table(outs[False]["row"], outs[True]["row"]))

        drawn = [visualize.draw_candidates(visualize.draw_road(o["small"], o["road"], o["y0"]), o["cracks"], o["potholes"], o["y0"])
                 for o in outs.values()]
        view = build_view(drawn[0], drawn[1], make_labels(steps_text, unicode_ok), screen_size(), unicode_ok, footer)
    finally:
        rx.restore_overrides(backup)

    if args.save:
        save_image(args.save, view)
        print(f"화면을 저장함: {args.save}")
    else:
        (show_fn or show)(view)
    return 0


def dialog_loop(args, tools, show_fn=None):
    """선택 창으로 사진을 고를 때마다 시연하고, 취소하면 0으로 끝난다.
    오류가 나도 시연이 멈추지 않도록 안내 창만 띄우고 다음 사진을 고르게 한다."""
    choose, warn, close = tools
    try:
        while True:
            path = choose()
            if path is None:
                print("사진 고르기를 취소해 시연을 끝냄")
                return 0
            try:
                demo_one(path, args, show_fn, warn)
            except Exception as e:                  # 어떤 오류든 시연은 계속
                text = f"처리 중 오류가 나서 이 사진은 건너뜁니다: {os.path.basename(path)}\n({type(e).__name__}: {e})"
                print(text)
                warn(text)
    finally:
        close()


def main(argv=None, chooser_factory=None, show_fn=None):
    """사진 고르는 방식을 정해 시연한다: 경로를 주면 그 사진, --list면 번호 목록, 아니면 파일 선택 창.
    선택 창을 쓸 수 없으면 번호 목록으로 넘어간다."""
    args = parse_args(argv)
    if args.path is not None:
        return demo_one(args.path, args, show_fn)
    if not args.list:
        tools = (chooser_factory or make_chooser)()
        if tools is not None:
            return dialog_loop(args, tools, show_fn)
        print("파일 선택 창(tkinter)을 이 컴퓨터에서 쓸 수 없어 번호 목록으로 고릅니다")
    path = pick_image()
    if path is None:
        print("사진을 고르지 않아 종료")
        return 0
    return demo_one(path, args, show_fn)


if __name__ == "__main__":
    sys.exit(main())
