"""(파트 A) 형태 산점도·격자 탐색: 후보의 비원형도와 두께를 정답 종류별 색으로 찍어, B가 균열·포트홀 기준값을 정하게 돕는다.
사용법: 맨 위 폴더에서  python tools/shape_scatter.py [--y thickness|dt_width|both] [--grid] [--limit N] [--out results/shape]
근거: Plan.md 3.5 (2) 산점도·격자 탐색, 계획서 4.2 학습과제 4번(형태 분류)의 분포 확인 도구.

입력
- 학습 데이터(config B 구역 TRAIN_IMAGE_DIR) 중 개발셋만 쓴다. 개발/시험 나누기는 C가 WP0에서 tools/evaluate_train.py에 넣는
  규칙을 그대로 가져다 쓴다. 시험셋은 기준값을 정하는 데 쓰지 않는다(Plan.md 3.10). --limit N이면 앞에서 N장만(빨리 확인할 때).
- 후보: pipeline.run_pipeline(전처리 끔, config 기본 띠)의 cracks + potholes. 후보 dict의 circ_inv·thickness·dt_width를 쓴다
  (CONTRIBUTING 2장 후보 키). DETECT_MODE가 "dark"면 어두운 영역 후보, "legacy"면 지금 후보(legacy 후보에도 세 키가 있다).
- 점 색: 후보 bbox 중심(640px 전체 좌표 = y + y0)이 든 정답 박스의 클래스. JSON(config B 구역 TRAIN_ANN_DIR/<이름>.jpg.json)의
  classTitle로 가른다: longitudinal crack = D00, transverse crack = D10, alligator crack = D20, pothole = D40, 어느 정답에도 안 들면 '없음'.
  정답 CSV(gt/train)에는 crack·pothole만 있어 D00·D10·D20을 가를 수 없으므로 JSON을 읽는다(박스 좌표는 convert_json_gt와 같게 640px로).

출력 (tools는 파일을 저장해도 된다. results/는 Git에 올리지 않는다)
- <out>/scatter_thickness.png, <out>/scatter_dt_width.png: x = 비원형도 circ_inv(로그 눈금), y = 두께(2A/P 또는 거리 변환 폭),
  색 = D00·D10·D20·D40·없음. 지금 config의 CRACK_MIN_CIRC_INV·POT_MAX_CIRC_INV를 세로선으로 함께 그린다.
- <out>/candidates.csv: 후보 1개 = 1행 (영상, 후보 종류, circ_inv, thickness, dt_width, solidity, 정답 클래스)
- --grid: <out>/grid.csv에 (CRACK_MIN_CIRC_INV, POT_MAX_CIRC_INV) 격자마다 종류별 F1과 균형 정확도를 쓰고, 콘솔에 가장 좋은 값을 출력

TODO(A): 아직 뼈대다. 인자와 입출력 설명만 있다(docs/tasks/A.md 3장)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

Y_CHOICES = ("thickness", "dt_width", "both")
OUT_DIR = "results/shape"


def parse_args(argv=None):
    """세로축 지표(--y), 격자 탐색 여부(--grid), 처리할 장수(--limit), 저장 폴더(--out)를 명령행에서 읽는다."""
    parser = argparse.ArgumentParser(description="후보 형태(비원형도·두께)의 정답 종류별 산점도와 문턱 격자 탐색")
    parser.add_argument("--y", choices=Y_CHOICES, default="both",
                        help="세로축: thickness = 2A/P, dt_width = 거리 변환 폭, both = 두 장 (기본 both)")
    parser.add_argument("--grid", action="store_true", help="문턱 격자 탐색 표(grid.csv)도 만든다")
    parser.add_argument("--limit", type=int, default=None, help="개발셋에서 앞의 N장만 처리 (기본: 전부)")
    parser.add_argument("--out", default=OUT_DIR, help=f"그림·표를 저장할 폴더 (기본 {OUT_DIR})")
    return parser.parse_args(argv)


def main(argv=None):
    """개발셋 후보의 형태 산점도(와 --grid면 격자 탐색 표)를 저장한다. TODO(A): 아직 구현 전."""
    parse_args(argv)
    raise NotImplementedError("TODO(A): tools/shape_scatter.py는 아직 뼈대입니다 (docs/tasks/A.md 참고)")


if __name__ == "__main__":
    main()
