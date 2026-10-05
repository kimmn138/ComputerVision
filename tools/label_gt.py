"""(파트 B) 도로 손상 정답 박스를 지정하여 GT CSV로 저장한다.
사용법: 맨 위 폴더에서  python tools/label_gt.py [영상 경로]
예: python tools/label_gt.py data/provided/United_States_004830.jpg  -> gt/United_States_004830.csv
영상 경로를 빼면(VS Code ▶ 버튼 등) data 아래 사진 목록을 번호로 보여 주고 고르게 한다.
세 명이 영상을 나눠 표시하므로 이 파일은 고치지 않고 영상 경로만 바꿔 실행한다."""

import argparse
import csv
from pathlib import Path

import cv2 as cv


GT_DIR = Path("gt")
DISPLAY_WIDTH = 640

DATA_DIR = Path("data")
# data/train은 훈련 사진 804장이고 정답은 JSON에서 변환(convert_json_gt)하므로 목록에서 뺀다
EXCLUDE_DIRS = ("train",)


def parse_args(argv=None):
    """정답 박스를 표시할 영상 경로를 명령행에서 읽는다. 빼면 None(목록에서 고름)."""
    parser = argparse.ArgumentParser(
        description="정답 박스(GT) 표시 도구: 가로 640px 전체 영상 좌표로 gt/<영상 이름>.csv에 저장",
    )

    parser.add_argument(
        "image",
        nargs="?",
        help="영상 경로 (예: data/provided/United_States_004830.jpg). 빼면 data 아래 사진 목록에서 번호로 고름",
    )

    return parser.parse_args(argv)


def list_images(root=DATA_DIR):
    """root 아래 사진(.jpg)을 경로 이름순으로 모은다. EXCLUDE_DIRS 폴더는 뺀다."""
    root = Path(root)

    return sorted(
        path
        for path in root.rglob("*.jpg")
        if path.relative_to(root).parts[0] not in EXCLUDE_DIRS
    )


def gt_status(image_path):
    """목록에 함께 보일 정답 상태: 'GT n개' 또는 'GT 없음'. 세 명이 나눠 표시할 때 남은 사진을 찾기 쉽게."""
    gt_path = gt_path_from_image(image_path)

    if not gt_path.exists():
        return "GT 없음"

    return f"GT {len(load_existing_gt(gt_path))}개"


def pick_image(root=DATA_DIR, input_fn=input):
    """사진 목록을 번호로 보여 주고 입력받은 번호의 경로를 돌려준다. 빈칸·q·입력 끝이면 None."""
    paths = list_images(root)

    if not paths:
        print(f"{root} 아래에 사진(.jpg)이 없습니다. 맨 위 폴더에서 실행했는지 확인하세요.")
        return None

    for number, path in enumerate(paths, start=1):
        print(f"{number:3d}. {path.as_posix():50s} [{gt_status(path)}]")

    while True:
        try:
            answer = input_fn(
                f"번호를 입력하세요 (1~{len(paths)}, 빈칸이나 q는 종료): "
            ).strip()
        except EOFError:
            return None

        if answer in ("", "q"):
            return None

        if answer.isdigit() and 1 <= int(answer) <= len(paths):
            return paths[int(answer) - 1]

        print("목록에 있는 번호를 입력하세요.")


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


def main(argv=None):
    """명령행으로 받은(없으면 목록에서 고른) 영상에 정답 박스를 하나씩 표시하고, 박스를 더할 때마다 GT CSV에 저장한다."""
    image = parse_args(argv).image

    if image is None:
        image = pick_image()

        if image is None:
            print("사진을 고르지 않아 종료합니다.")
            return

    image_path = Path(image)

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