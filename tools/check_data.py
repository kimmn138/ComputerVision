"""(Tools) 데이터 폴더의 이미지 파일명 규칙, CSV 연동, 로드 가능 여부를 일괄 검증합니다."""
import os
import glob
from src import io_utils

def check_dataset(data_dir="data"):
    print(f"[{data_dir}] 폴더 데이터 검증 시작...\n")
    
    # 1. CSV 파일 로드 테스트
    roi_path = os.path.join(data_dir, "roi.csv")
    log_path = os.path.join(data_dir, "shot_log.csv")
    
    try:
        roi_table = io_utils.load_roi_table(roi_path)
        print(f"[OK] roi.csv 로드 완료 (총 {len(roi_table)}개 항목)")
    except Exception as e:
        print(f"[FAIL] roi.csv 로드 실패: {e}")
        roi_table = {}

    # 2. 이미지 파일 탐색 및 검증
    img_paths = glob.glob(os.path.join(data_dir, "*.jpg")) + glob.glob(os.path.join(data_dir, "*.png"))
    
    if not img_paths:
        print("\n[WARN] 검사할 이미지 파일이 없습니다.")
        return

    print(f"\n총 {len(img_paths)}개의 이미지 검사를 시작합니다.")
    print("-" * 50)
    
    success_count = 0
    error_count = 0

    for path in img_paths:
        filename = os.path.basename(path)
        errors = []

        # A. 파일명 규칙 검사
        try:
            parsed = io_utils.parse_name(path)
        except ValueError as e:
            errors.append(f"파일명 규칙 위반 ({e})")

        # B. 이미지 읽기 검사
        try:
            img = io_utils.load_image(path)
        except FileNotFoundError as e:
            errors.append(f"이미지 로드 실패 ({e})")

        # C. roi.csv 등록 여부 확인
        if filename not in roi_table:
            errors.append("roi.csv에 항목이 없습니다.")

        # 결과 출력
        if errors:
            print(f"[FAIL] {filename}: {', '.join(errors)}")
            error_count += 1
        else:
            print(f"[OK] {filename} (규칙: {parsed.get('source')})")
            success_count += 1

    print("-" * 50)
    print(f"검사 완료: 성공 {success_count}건, 실패 {error_count}건")

if __name__ == "__main__":
    check_dataset()
