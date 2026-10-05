"""(Tools) 단일 이미지에 대한 전처리 파이프라인 결과를 시각적으로 확인합니다.
사용법: 맨 위 폴더에서  python tools/show_preprocess.py [영상 경로]
영상 경로를 빼면(VS Code ▶ 버튼 등) data 아래 사진 목록을 번호로 보여 주고 고르게 합니다."""
import argparse
import glob
import os
import sys
import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import io_utils, analyze, preprocess  # noqa: E402

DATA_DIR = "data"
EXCLUDE_DIRS = ("train",)     # data/train은 B의 훈련 사진 804장이라 목록에서 뺌


def parse_args(argv=None):
    """확인할 영상 경로를 명령행에서 읽는다. 빼면 None(목록에서 고름)."""
    parser = argparse.ArgumentParser(description="사진 1장의 상태 지표·고른 전처리·전후 영상을 창에 띄웁니다")
    parser.add_argument("image", nargs="?", help="영상 경로. 빼면 data 아래 사진 목록에서 번호로 고름")
    return parser.parse_args(argv)


def list_images(root=DATA_DIR):
    """root 아래 사진(.jpg) 경로를 이름순으로 모은다. EXCLUDE_DIRS 폴더는 뺀다."""
    paths = glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
    return sorted(p for p in paths if os.path.relpath(p, root).split(os.sep)[0] not in EXCLUDE_DIRS)


def pick_image(root=DATA_DIR, input_fn=input):
    """사진 목록을 번호로 보여 주고 입력받은 번호의 경로를 돌려준다. 빈칸·q·입력 끝이면 None."""
    paths = list_images(root)
    if not paths:
        print(f"{root} 아래에 사진(.jpg)이 없습니다. 맨 위 폴더에서 실행했는지 확인하세요.")
        return None
    for number, path in enumerate(paths, start=1):
        print(f"{number:3d}. {path.replace(os.sep, '/')}")
    while True:
        try:
            answer = input_fn(f"번호를 입력하세요 (1~{len(paths)}, 빈칸이나 q는 종료): ").strip()
        except EOFError:
            return None
        if answer in ("", "q"):
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(paths):
            return paths[int(answer) - 1]
        print("목록에 있는 번호를 입력하세요.")


def show_preprocessing_result(image_path, roi_csv_path="data/roi.csv"):
    if not os.path.exists(image_path):
        print(f"오류: 파일을 찾을 수 없습니다 -> {image_path}")
        return

    filename = os.path.basename(image_path)
    print(f"[{filename}] 전처리 테스트 시작...")

    # 1. 이미지 로드 및 리사이즈
    img = io_utils.load_image(image_path)
    resized = io_utils.resize_width(img, width=640)

    # 2. ROI 크롭 (CSV에 있으면 해당 값, 없으면 기본값 사용)
    roi_table = io_utils.load_roi_table(roi_csv_path)
    roi_info = roi_table.get(filename, {"top": 0.35, "bottom": 1.0})
    cropped, y0 = io_utils.crop_roi(resized, top=roi_info["top"], bottom=roi_info["bottom"])

    # 3. 흑백 변환 및 상태 진단
    gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
    quality = analyze.measure_quality(gray)
    
    # 4. 전처리 단계 선택 및 실행
    steps = analyze.choose_steps(quality)
    steps_text = analyze.steps_to_text(steps)
    processed_gray = preprocess.preprocess(gray, steps)

    # 5. 시각화용 이미지 구성
    # 원본(리사이즈) 복사 및 크롭 선 그리기
    vis_orig = resized.copy()
    cv.line(vis_orig, (0, y0), (vis_orig.shape[1], y0), (0, 0, 255), 2)
    cv.putText(vis_orig, "Original & ROI Line", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 크롭된 흑백 원본
    vis_gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    cv.putText(vis_gray, "Cropped Gray", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.putText(vis_gray, f"M:{quality['mean']:.1f} S:{quality['std']:.1f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # 전처리 결과
    vis_processed = cv.cvtColor(processed_gray, cv.COLOR_GRAY2BGR)
    cv.putText(vis_processed, "Preprocessed", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
    cv.putText(vis_processed, f"Steps: {steps_text}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)

    # 높이를 맞추기 위해 검은색 패딩 추가 (원본과 크롭 이미지 간의 높이 차이 보정)
    h_orig = vis_orig.shape[0]
    h_crop = vis_gray.shape[0]
    
    pad_h = h_orig - h_crop
    if pad_h > 0:
        padding = np.zeros((pad_h, vis_gray.shape[1], 3), dtype=np.uint8)
        vis_gray = np.vstack([vis_gray, padding])
        vis_processed = np.vstack([vis_processed, padding])

    # 가로로 이어 붙이기
    combined = np.hstack([vis_orig, vis_gray, vis_processed])

    # 결과 출력
    print(f"▶ 품질 지표: {quality}")
    print(f"▶ 적용 단계: {steps_text}")
    
    cv.imshow(f"Preprocess Result - {filename}", combined)
    print("아무 키나 누르면 창이 닫힙니다...")
    cv.waitKey(0)
    cv.destroyAllWindows()

def main(argv=None):
    """명령행으로 받은(없으면 목록에서 고른) 사진의 전처리 결과를 띄운다."""
    target_image = parse_args(argv).image or pick_image()
    if target_image is None:
        print("사진을 고르지 않아 종료합니다.")
        return
    show_preprocessing_result(target_image)


if __name__ == "__main__":
    # 예: python tools/show_preprocess.py data/provided/Japan_000107.jpg   (경로를 빼면 목록에서 고름)
    main()
