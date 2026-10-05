"""(파트 B) 제공된 JSON annotation을 프로젝트 GT CSV 형식으로 변환한다.
사용법: 맨 위 폴더에서  python tools/convert_json_gt.py
읽는 폴더(훈련 사진·JSON)와 쓰는 폴더(정답 CSV)는 config B 구역의 TRAIN_IMAGE_DIR·TRAIN_ANN_DIR·TRAIN_GT_DIR."""

import csv
import json
import sys
from pathlib import Path

import cv2 as cv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402


# ---------------------------------------------------------
# 교수님 제공 데이터 위치
#
# config B 구역의 TRAIN_IMAGE_DIR(사진)·TRAIN_ANN_DIR(JSON)·TRAIN_GT_DIR(정답 CSV)에서 바꾼다.
# JSON과 JPG가 같은 폴더에 있으면 TRAIN_IMAGE_DIR·TRAIN_ANN_DIR을 같은 경로로 두면 된다.
# evaluate_train도 같은 값을 읽으므로 두 도구의 경로가 어긋나지 않는다.
# ---------------------------------------------------------


def class_to_kind(class_title):
    """제공 데이터의 손상 클래스를 프로젝트 crack/pothole로 통합한다."""
    title = class_title.strip().lower()

    if "crack" in title:
        return "crack"

    if "pothole" in title:
        return "pothole"

    return None


def find_image(json_path):
    """JSON 파일명에서 .json만 제거해 대응 JPG를 찾는다."""
    image_name = json_path.name.removesuffix(".json")

    image_path = Path(C.TRAIN_IMAGE_DIR) / image_name

    if image_path.exists():
        return image_path

    return None

def check_image_size(image_path, json_width, json_height):
    """실제 JPG 크기와 JSON에 기록된 원본 크기가 같은지 확인한다."""
    bgr = cv.imread(str(image_path))

    if bgr is None:
        raise FileNotFoundError(image_path)

    image_height, image_width = bgr.shape[:2]

    if image_width != json_width or image_height != json_height:
        raise ValueError(
            f"이미지와 JSON 크기가 다릅니다: {image_path.name}\n"
            f"image = {image_width}x{image_height}, "
            f"json = {json_width}x{json_height}"
        )


def convert_box(exterior, scale):
    """JSON의 두 꼭짓점을 640px 기준 (x, y, w, h)로 변환한다."""
    if len(exterior) != 2:
        raise ValueError(
            f"rectangle exterior는 점 2개여야 합니다: {exterior}"
        )

    x1, y1 = exterior[0]
    x2, y2 = exterior[1]

    left = min(x1, x2)
    top = min(y1, y2)

    right = max(x1, x2)
    bottom = max(y1, y2)

    x = round(left * scale)
    y = round(top * scale)

    w = round((right - left) * scale)
    h = round((bottom - top) * scale)

    return x, y, w, h


def convert_json(json_path):
    """JSON 한 개를 읽어 프로젝트 GT 행 목록으로 변환한다."""
    with json_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    size = data.get("size")

    if not size:
        raise ValueError(
            f"{json_path.name}: size 정보가 없습니다."
        )

    json_width = int(size["width"])
    json_height = int(size["height"])

    if json_width <= 0 or json_height <= 0:
        raise ValueError(
            f"{json_path.name}: 잘못된 이미지 크기입니다."
        )

    image_path = find_image(json_path)

    if image_path is not None:
        check_image_size(
            image_path,
            json_width,
            json_height,
        )
    else:
        print(
            "[주의] 대응 JPG 없음:",
            f"{json_path.stem}.jpg",
        )

    # 프로젝트 전체 영상은 가로 TARGET_WIDTH로 통일
    scale = C.TARGET_WIDTH / json_width

    rows = []

    for obj in data.get("objects", []):
        geometry_type = obj.get("geometryType")

        # 현재 프로젝트 평가는 bbox 기준이므로 rectangle만 사용
        if geometry_type != "rectangle":
            print(
                "[건너뜀]",
                json_path.name,
                "geometryType =",
                geometry_type,
            )
            continue

        class_title = obj.get(
            "classTitle",
            "",
        )

        kind = class_to_kind(
            class_title,
        )

        # crack/pothole이 아닌 클래스는 평가 대상에서 제외
        if kind is None:
            print(
                "[건너뜀]",
                json_path.name,
                "classTitle =",
                class_title,
            )
            continue

        points = obj.get(
            "points",
            {},
        )

        exterior = points.get(
            "exterior",
            [],
        )

        x, y, w, h = convert_box(
            exterior,
            scale,
        )

        # 넓이나 높이가 0인 잘못된 박스 제외
        if w <= 0 or h <= 0:
            print(
                "[건너뜀] 크기가 0인 박스:",
                json_path.name,
                class_title,
            )
            continue

        rows.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "kind": kind,
            }
        )

    return rows


def save_csv(json_path, rows):
    """변환된 GT를 evaluate.load_gt와 호환되는 CSV로 저장한다."""
    gt_dir = Path(C.TRAIN_GT_DIR)

    gt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_name = json_path.name.removesuffix(".json")
    image_stem = Path(image_name).stem

    csv_path = gt_dir / f"{image_stem}.csv"

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "x",
                "y",
                "w",
                "h",
                "kind",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return csv_path

def main():
    """JSON 전체를 찾아 GT CSV로 일괄 변환한다."""
    json_dir = Path(C.TRAIN_ANN_DIR)

    json_paths = sorted(
        json_dir.glob("*.json")
    )

    if not json_paths:
        raise RuntimeError(
            f"JSON 파일이 없습니다: {json_dir}\n"
            "훈련 데이터를 이 폴더에 두거나, 다른 곳에 있으면 "
            "src/config.py B 구역의 TRAIN_ANN_DIR(JSON)·TRAIN_IMAGE_DIR(사진)을 그 경로로 바꾸세요."
        )

    print("=" * 70)
    print("JSON -> GT CSV 변환")
    print("JSON:", json_dir)
    print("IMAGE:", C.TRAIN_IMAGE_DIR)
    print("GT:", C.TRAIN_GT_DIR)
    print("TARGET_WIDTH:", C.TARGET_WIDTH)
    print("=" * 70)

    total_objects = 0

    for json_path in json_paths:
        rows = convert_json(
            json_path,
        )

        csv_path = save_csv(
            json_path,
            rows,
        )

        total_objects += len(rows)

        print(
            json_path.name,
            "->",
            csv_path,
            f"({len(rows)} objects)",
        )

    print()
    print("=" * 70)
    print("변환 완료")
    print("JSON 수:", len(json_paths))
    print("GT 객체 수:", total_objects)
    print("=" * 70)


if __name__ == "__main__":
    main()