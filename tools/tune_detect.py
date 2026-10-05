"""(파트 B) 튜닝 세트에서 검출 파라미터 후보를 반복 비교하고 결과를 저장한다.
사용법: 맨 위 폴더에서  python tools/tune_detect.py"""

import csv
import sys
from pathlib import Path

import cv2 as cv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.detect import detect_cracks, detect_potholes  # noqa: E402


PROVIDED_DIR = Path("data/provided")
OWN_DIR = Path("data/own")
RESULT_DIR = Path("results/tuning")

PROVIDED_COUNT = 4
OWN_COUNT = 2


# ---------------------------------------------------------
# 파라미터 후보
#
# 한 번에 균열/포트홀 값을 전부 바꾸면 원인 분석이 어려우므로
# base를 기준으로 균열과 포트홀을 각각 느슨/엄격하게 비교한다.
# ---------------------------------------------------------

PARAM_SETS = [
    {
        "name": "base",

        "CANNY_LOW": 50,
        "CANNY_HIGH": 150,
        "CRACK_CLOSE_KSIZE": 5,
        "CRACK_MIN_AREA": 80,
        "CRACK_MIN_ELONG": 4.0,
        "CRACK_MAX_FILL": 0.10,

        "DOG_SIGMAS": (3, 12),
        "DOG_THRESH": 12,
        "POT_OPEN_KSIZE": 5,
        "POT_MIN_AREA": 150,
        "POT_MAX_AREA_RATIO": 0.20,
        "POT_MAX_ELONG": 3.0,
        "POT_MIN_SOLIDITY": 0.60,
    },

    {
        "name": "crack_loose",

        "CANNY_LOW": 50,
        "CANNY_HIGH": 150,
        "CRACK_CLOSE_KSIZE": 5,
        "CRACK_MIN_AREA": 60,
        "CRACK_MIN_ELONG": 3.0,
        "CRACK_MAX_FILL": 0.15,

        # 포트홀은 base 유지
        "DOG_SIGMAS": (3, 12),
        "DOG_THRESH": 12,
        "POT_OPEN_KSIZE": 5,
        "POT_MIN_AREA": 150,
        "POT_MAX_AREA_RATIO": 0.20,
        "POT_MAX_ELONG": 3.0,
        "POT_MIN_SOLIDITY": 0.60,
    },

    {
        "name": "crack_strict",

        "CANNY_LOW": 50,
        "CANNY_HIGH": 150,
        "CRACK_CLOSE_KSIZE": 5,
        "CRACK_MIN_AREA": 120,
        "CRACK_MIN_ELONG": 5.0,
        "CRACK_MAX_FILL": 0.08,

        # 포트홀은 base 유지
        "DOG_SIGMAS": (3, 12),
        "DOG_THRESH": 12,
        "POT_OPEN_KSIZE": 5,
        "POT_MIN_AREA": 150,
        "POT_MAX_AREA_RATIO": 0.20,
        "POT_MAX_ELONG": 3.0,
        "POT_MIN_SOLIDITY": 0.60,
    },

    {
        "name": "pothole_loose",

        # 균열은 base 유지
        "CANNY_LOW": 50,
        "CANNY_HIGH": 150,
        "CRACK_CLOSE_KSIZE": 5,
        "CRACK_MIN_AREA": 80,
        "CRACK_MIN_ELONG": 4.0,
        "CRACK_MAX_FILL": 0.10,

        "DOG_SIGMAS": (3, 12),
        "DOG_THRESH": 10,
        "POT_OPEN_KSIZE": 5,
        "POT_MIN_AREA": 100,
        "POT_MAX_AREA_RATIO": 0.25,
        "POT_MAX_ELONG": 4.0,
        "POT_MIN_SOLIDITY": 0.50,
    },

    {
        "name": "pothole_strict",

        # 균열은 base 유지
        "CANNY_LOW": 50,
        "CANNY_HIGH": 150,
        "CRACK_CLOSE_KSIZE": 5,
        "CRACK_MIN_AREA": 80,
        "CRACK_MIN_ELONG": 4.0,
        "CRACK_MAX_FILL": 0.10,

        "DOG_SIGMAS": (3, 12),
        "DOG_THRESH": 15,
        "POT_OPEN_KSIZE": 5,
        "POT_MIN_AREA": 200,
        "POT_MAX_AREA_RATIO": 0.15,
        "POT_MAX_ELONG": 2.5,
        "POT_MIN_SOLIDITY": 0.70,
    },
]


def resize_width(img, width=C.TARGET_WIDTH):
    """영상 비율을 유지하면서 공통 기준 가로 크기로 맞춘다."""
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


def collect_images(directory, count):
    """폴더에서 이름순으로 JPG를 지정 개수만 선택한다."""
    paths = sorted(directory.glob("*.jpg"))

    if len(paths) < count:
        raise RuntimeError(
            f"{directory}에 JPG가 {count}장 필요하지만 "
            f"{len(paths)}장만 있습니다."
        )

    return paths[:count]


def backup_params():
    """튜닝에서 변경할 B config 값을 원래 상태로 백업한다."""
    keys = {
        key
        for params in PARAM_SETS
        for key in params
        if key != "name"
    }

    backup = {}

    for key in keys:
        if not hasattr(C, key):
            raise AttributeError(
                f"src.config에 {key}가 없습니다."
            )

        backup[key] = getattr(C, key)

    return backup


def apply_param_set(params):
    """파라미터 후보를 실행 중인 config 객체에만 임시 적용한다."""
    for key, value in params.items():
        if key == "name":
            continue

        setattr(C, key, value)


def restore_params(backup):
    """튜닝 종료 후 B config 값을 실행 전 상태로 복구한다."""
    for key, value in backup.items():
        setattr(C, key, value)


def draw_candidates(bgr, cracks, potholes):
    """검출 후보를 초록색 균열, 빨간색 포트홀 박스로 표시한다."""
    out = bgr.copy()

    for region in cracks:
        x, y, w, h = region["bbox"]

        cv.rectangle(
            out,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

    for region in potholes:
        x, y, w, h = region["bbox"]

        cv.rectangle(
            out,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2,
        )

    return out


def add_title(img, text):
    """비교 결과 상단에 파라미터 세트와 후보 개수를 표시한다."""
    out = img.copy()

    cv.rectangle(
        out,
        (0, 0),
        (out.shape[1], 36),
        (0, 0, 0),
        -1,
    )

    cv.putText(
        out,
        text,
        (10, 25),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    return out


def make_comparison(original, result):
    """원본과 후보 검출 결과를 좌우 비교 영상으로 만든다."""
    left = original.copy()
    right = result.copy()

    cv.putText(
        left,
        "original",
        (10, 25),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    return cv.hconcat([left, right])


def run_one(image_path, param_name):
    """영상 한 장에서 균열과 포트홀 후보를 검출한다."""
    bgr = cv.imread(str(image_path))

    if bgr is None:
        raise FileNotFoundError(image_path)

    bgr = resize_width(bgr)

    gray = cv.cvtColor(
        bgr,
        cv.COLOR_BGR2GRAY,
    )

    cracks, edges = detect_cracks(gray)
    potholes, mask = detect_potholes(gray)

    candidate_img = draw_candidates(
        bgr,
        cracks,
        potholes,
    )

    title = (
        f"{param_name} | "
        f"crack={len(cracks)} "
        f"pothole={len(potholes)}"
    )

    candidate_img = add_title(
        candidate_img,
        title,
    )

    comparison = make_comparison(
        bgr,
        candidate_img,
    )

    return {
        "original": bgr,
        "candidates": candidate_img,
        "comparison": comparison,
        "edges": edges,
        "mask": mask,
        "cracks": cracks,
        "potholes": potholes,
    }


def save_outputs(image_path, source, param_name, outputs):
    """튜닝 결과와 중간 영상을 results/tuning 아래 저장한다."""
    save_dir = (
        RESULT_DIR
        / param_name
        / source
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = image_path.stem

    cv.imwrite(
        str(save_dir / f"{stem}_comparison.jpg"),
        outputs["comparison"],
    )

    cv.imwrite(
        str(save_dir / f"{stem}_candidates.jpg"),
        outputs["candidates"],
    )

    cv.imwrite(
        str(save_dir / f"{stem}_edges.jpg"),
        outputs["edges"],
    )

    cv.imwrite(
        str(save_dir / f"{stem}_pothole_mask.jpg"),
        outputs["mask"],
    )


def make_summary_row(source, image_path, param_name, outputs):
    """한 번의 튜닝 결과를 CSV 요약 행으로 만든다."""
    return {
        "source": source,
        "image": image_path.name,
        "params": param_name,
        "n_crack": len(outputs["cracks"]),
        "area_crack": sum(
            region["area"]
            for region in outputs["cracks"]
        ),
        "n_pothole": len(outputs["potholes"]),
        "area_pothole": sum(
            region["area"]
            for region in outputs["potholes"]
        ),
    }


def save_summary(rows):
    """전체 파라미터 비교 결과를 CSV로 저장한다."""
    path = RESULT_DIR / "summary.csv"

    fieldnames = [
        "source",
        "image",
        "params",
        "n_crack",
        "area_crack",
        "n_pothole",
        "area_pothole",
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    """6장 영상과 여러 파라미터 후보를 반복 실행하여 비교 결과를 저장한다."""
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    provided_images = collect_images(
        PROVIDED_DIR,
        PROVIDED_COUNT,
    )

    own_images = collect_images(
        OWN_DIR,
        OWN_COUNT,
    )

    image_sets = [
        ("provided", provided_images),
        ("own", own_images),
    ]

    print("=" * 70)
    print("튜닝 세트")
    print("=" * 70)

    print("\n[provided]")

    for path in provided_images:
        print("-", path.name)

    print("\n[own]")

    for path in own_images:
        print("-", path.name)

    print()

    original_params = backup_params()
    summary_rows = []

    try:
        for params in PARAM_SETS:
            param_name = params["name"]

            apply_param_set(params)

            print()
            print("=" * 70)
            print("parameter set:", param_name)
            print("=" * 70)

            for source, image_paths in image_sets:
                for image_path in image_paths:
                    outputs = run_one(
                        image_path,
                        param_name,
                    )

                    save_outputs(
                        image_path,
                        source,
                        param_name,
                        outputs,
                    )

                    row = make_summary_row(
                        source,
                        image_path,
                        param_name,
                        outputs,
                    )

                    summary_rows.append(row)

                    print(
                        source,
                        image_path.name,
                        "| cracks =",
                        row["n_crack"],
                        "potholes =",
                        row["n_pothole"],
                    )

    finally:
        # 중간에 오류가 발생해도 실행 전 config 상태로 복원한다.
        restore_params(original_params)

    save_summary(summary_rows)

    print()
    print("=" * 70)
    print("튜닝 완료")
    print("결과:", RESULT_DIR)
    print("요약:", RESULT_DIR / "summary.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()