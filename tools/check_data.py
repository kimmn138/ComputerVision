"""(Tools) 데이터 폴더의 이미지 파일명 규칙, CSV 연동, 로드 가능 여부를 일괄 검증합니다.
사용법: 맨 위 폴더에서  python tools/check_data.py"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import io_utils  # noqa: E402

# 하위 폴더 → 그 폴더 사진의 parse_name source (폴더와 이름 규칙이 어긋나면 실패)
IMAGE_DIRS = {"provided": "provided", "own": "own", "pairs": "pair"}
NEEDS_ROI = ("provided", "own")     # run_experiment가 처리하는 폴더만 roi.csv가 필요 (pairs는 영상 전체로 매칭)


def check_dataset(data_dir="data"):
    """data/provided·own·pairs의 사진마다 확장자·이름 규칙·폴더 일치·읽기·roi.csv 등록을 검사해 결과를 출력한다."""
    print(f"[{data_dir}] 폴더 데이터 검증 시작...\n")

    # 1. CSV 파일 로드 테스트
    roi_path = os.path.join(data_dir, "roi.csv")

    try:
        roi_table = io_utils.load_roi_table(roi_path)
        print(f"[OK] roi.csv 로드 완료 (총 {len(roi_table)}개 항목)")
    except Exception as e:
        print(f"[FAIL] roi.csv 로드 실패: {e}")
        roi_table = {}

    # 2. 하위 폴더의 파일 탐색 (.gitkeep 제외)
    files = []
    for sub in IMAGE_DIRS:
        for path in sorted(glob.glob(os.path.join(data_dir, sub, "*"))):
            if os.path.isfile(path) and os.path.basename(path) != ".gitkeep":
                files.append((sub, path))

    if not files:
        print(f"\n[WARN] 검사할 파일이 없습니다 ({', '.join(IMAGE_DIRS)} 폴더).")
        return

    print(f"\n총 {len(files)}개의 파일 검사를 시작합니다.")
    print("-" * 50)

    success_count = 0
    error_count = 0

    for sub, path in files:
        filename = os.path.basename(path)
        errors = []
        parsed = None

        # A. 확장자 검사 (소문자 .jpg만 처리 대상)
        if not filename.endswith(".jpg"):
            errors.append("확장자가 소문자 .jpg가 아닙니다")

        # B. 파일명 규칙 검사 + 폴더와 이름이 맞는지
        try:
            parsed = io_utils.parse_name(path)
            if parsed["source"] != IMAGE_DIRS[sub]:
                errors.append(f"{sub} 폴더인데 이름이 {parsed['source']} 규칙으로 읽힙니다")
        except ValueError as e:
            errors.append(f"파일명 규칙 위반 ({e})")

        # C. 이미지 읽기 검사
        try:
            io_utils.load_image(path)
        except FileNotFoundError as e:
            errors.append(f"이미지 로드 실패 ({e})")

        # D. roi.csv 등록 여부 확인
        if sub in NEEDS_ROI and filename not in roi_table:
            errors.append("roi.csv에 항목이 없습니다.")

        # 결과 출력
        if errors:
            print(f"[FAIL] {sub}/{filename}: {', '.join(errors)}")
            error_count += 1
        else:
            print(f"[OK] {sub}/{filename} (규칙: {parsed.get('source')})")
            success_count += 1

    print("-" * 50)
    print(f"검사 완료: 성공 {success_count}건, 실패 {error_count}건")

if __name__ == "__main__":
    check_dataset()
