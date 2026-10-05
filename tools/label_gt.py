"""(파트 B) 도로 손상 정답 박스를 지정하여 GT CSV로 저장한다."""

import csv
from pathlib import Path

import cv2 as cv


IMAGE_PATH = "data/provided/United_States_004830.jpg"
GT_DIR = Path("gt")
DISPLAY_WIDTH = 640


def resize_width(img, width=DISPLAY_WIDTH):
    """영상 비율을 유지하면서 GT 기준 가로 640px로 맞춘다."""
    h, w = img.shape[:2]

    if w == width:
        return img.copy()

    scale = width / w
    new_h = round(h * scale)

    return cv.resize(
        img,
        (width, new_h),
        interpolation=cv.INTER_AREA,
    )


def gt_path_from_image(image_path):
    """영상 파일명과 같은 이름의 GT CSV 경로를 만든다."""
    image_path = Path(image_path)

    return GT_DIR / f"{image_path.stem}.csv"


def load_existing_gt(path):
    """기존 GT CSV가 있으면 박스 목록으로 읽는다."""
    path = Path(path)

    if not path.exists():
        return []

    boxes = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            boxes.append(
                {
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                    "w": int(row["w"]),
                    "h": int(row["h"]),
                    "kind": row["kind"],
                }
            )

    return boxes


def save_gt(path, boxes):
    """GT 박스 목록을 evaluate.load_gt와 호환되는 CSV로 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
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
        writer.writerows(boxes)


def draw_gt(img, boxes):
    """현재 GT 박스를 영상 복사본 위에 그린다."""
    out = img.copy()

    for box in boxes:
        x = box["x"]
        y = box["y"]
        w = box["w"]
        h = box["h"]
        kind = box["kind"]

        if kind == "crack":
            color = (0, 255, 0)

        else:
            color = (0, 0, 255)

        cv.rectangle(
            out,
            (x, y),
            (x + w, y + h),
            color,
            2,
        )

        cv.putText(
            out,
            kind,
            (x, max(20, y - 5)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv.LINE_AA,
        )

    return out


def select_kind():
    """라벨 종류를 터미널에서 선택한다."""
    while True:
        value = input(
            "\n[c] crack  [p] pothole  [q] 저장 후 종료\n"
            "선택: "
        ).strip().lower()

        if value == "c":
            return "crack"

        if value == "p":
            return "pothole"

        if value == "q":
            return None

        print("c, p, q 중 하나를 입력하세요.")


def main():
    image_path = Path(IMAGE_PATH)

    bgr = cv.imread(str(image_path))

    if bgr is None:
        raise FileNotFoundError(image_path)

    # GT는 가로 640px 전체 영상 좌표를 사용
    bgr = resize_width(bgr)

    gt_path = gt_path_from_image(image_path)
    boxes = load_existing_gt(gt_path)

    print("=" * 60)
    print("GT 라벨링")
    print("=" * 60)
    print("image:", image_path)
    print("GT:", gt_path)
    print("기존 박스 수:", len(boxes))

    print()
    print("[사용 방법]")
    print("1. 터미널에서 c 또는 p를 선택")
    print("2. 이미지 창에서 마우스로 박스를 드래그")
    print("3. Enter 또는 Space로 박스 확정")
    print("4. c 키를 누르면 현재 박스 선택 취소")
    print("5. 터미널에서 q를 입력하면 저장 후 종료")

    while True:
        # 먼저 터미널 입력을 받음.
        # 이 시점에는 OpenCV 창을 띄우지 않으므로 응답없음 문제가 없음.
        kind = select_kind()

        if kind is None:
            break

        preview = draw_gt(
            bgr,
            boxes,
        )

        print()
        print(f"{kind} 영역을 마우스로 지정하세요.")

        # selectROI가 자체적으로 GUI 이벤트 루프를 처리함
        x, y, w, h = cv.selectROI(
            "GT ROI - Enter/Space: confirm, C: cancel",
            preview,
            showCrosshair=True,
            fromCenter=False,
        )

        cv.destroyAllWindows()

        # ROI 선택 취소
        if w == 0 or h == 0:
            print("박스 선택을 취소했습니다.")
            continue

        box = {
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "kind": kind,
        }

        boxes.append(box)

        print(
            "추가 완료:",
            kind,
            f"(x={x}, y={y}, w={w}, h={h})",
        )

        # 매번 저장해서 중간에 프로그램이 종료돼도 작업 보존
        save_gt(
            gt_path,
            boxes,
        )

        print("현재 GT 수:", len(boxes))

    save_gt(
        gt_path,
        boxes,
    )

    cv.destroyAllWindows()

    print()
    print("=" * 60)
    print("GT 저장 완료")
    print("경로:", gt_path)
    print("박스 수:", len(boxes))
    print("=" * 60)


if __name__ == "__main__":
    main()