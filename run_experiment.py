"""(파트 C) 제공·직접 촬영 영상 전체를 전처리 끔·켬으로 한 번씩 처리해 지표 표를 저장한다.
사용법: 맨 위 폴더에서  python run_experiment.py --name check
결과: results/<name>/metrics.csv

TODO(C): 아직 하지 않은 것
- 5회 중앙값 시간
- 상태 지표 열
- 비교 그림
- 요약 표 3개
- config_used.txt
- 정답 비교 열
- 영상 하나가 실패해도 계속 돌리기
"""
import argparse
import glob
import os

import pandas as pd

from src import analyze, io_utils, pipeline
from src import config as C

# metrics.csv의 열 순서: 영상 정보 5열 + run_pipeline의 row 9열 (사진이 0장이어도 머리글은 남김)
COLUMNS = ["file", "source", "condition", "preprocess", "steps",
           "edges", "harris", "n_crack", "area_crack", "n_pothole", "area_pothole",
           "roi_pixels", "t_pre_ms", "t_detect_ms"]


def main():
    parser = argparse.ArgumentParser(description="전처리 끔·켬 비교 실험")
    parser.add_argument("--name", default="base", help="결과 폴더 이름 (results/<name>/)")
    args = parser.parse_args()

    files = sorted(glob.glob("data/provided/*.jpg") + glob.glob("data/own/*.jpg"))
    roi_table = io_utils.load_roi_table("data/roi.csv")
    rows = []
    for path in files:
        file_name = os.path.basename(path)
        roi = roi_table.get(file_name, {"top": C.ROI_TOP_DEFAULT, "bottom": C.ROI_BOTTOM_DEFAULT, "condition": ""})
        info = io_utils.parse_name(path)
        bgr = io_utils.load_image(path)
        for use in (False, True):
            out = pipeline.run_pipeline(bgr, roi["top"], roi["bottom"], use)
            rows.append({"file": file_name, "source": info["source"],
                         "condition": info["condition"] or roi["condition"],
                         "preprocess": "on" if use else "off",
                         "steps": analyze.steps_to_text(out["steps"]) if use else "",
                         **out["row"]})

    out_dir = f"results/{args.name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/metrics.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(out_path, encoding="utf-8-sig", index=False)
    print(f"{len(files)}장 처리 -> {out_path}")


if __name__ == "__main__":
    main()
