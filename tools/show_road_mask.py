"""(파트 A) 노면 마스크 확인 그림: v1·v2 노면 마스크를 나란히 겹쳐 그려, 노면을 제대로 골랐는지 눈으로 확인하고 정답 유지율을 잰다.
사용법: 맨 위 폴더에서  python tools/show_road_mask.py [--set dev39|diag35] [--out results/road_mask]
근거: Plan.md 3.3 (5) 확인 그림, (8) 통과 기준. docs/review/road_mask_v1_check.jpg와 같은 그림을 이 도구로 다시 만든다.

입력
- --set dev39: data/provided 13장 + data/own 26장. 띠는 roi.csv(없으면 config 기본 띠), 정답 박스는 gt/<이름>.csv(있을 때만)
- --set diag35: 학습 데이터(config B 구역 TRAIN_IMAGE_DIR)의 진단 시트 35장(Plan.md 부록 A). 띠는 config 기본 띠, 정답은 gt/train/<이름>.csv
  35장 이름은 C가 WP0에서 tools/evaluate_train.py에 DIAG35로 넣는다. 합쳐지기 전에는 부록 A를 보고 이 파일에 임시로 둔다.
- 노면 마스크는 io_utils.road_mask(컬러 노면 띠)로 만든다. v1·v2를 나란히 그릴 때는 config A 구역 값만 실행 중에 바꿔
  두 번 부르고 되돌린다(evaluate_train의 apply_param_set·restore_params와 같은 방법). 영상 이름으로 나누지 않는다.

출력 (tools는 파일을 저장해도 된다. results/는 Git에 올리지 않는다)
- <out>/<set>/<영상 이름>.jpg: 왼쪽 v1, 오른쪽 v2. 640px 영상에 띠 시작선(노랑), 노면(반투명 초록), 정답 박스(균열 빨강·포트홀 파랑)
- <out>/<set>/summary.csv: 영상마다 마스크 넓이 / 띠 넓이, 띠 안 정답 수, 노면 안에 남은 정답 수
  (띠 안 정답 = 정답 박스 중심이 띠 안, 남은 정답 = 그 중심 픽셀이 노면 마스크 안. 좌표는 y − y0로 맞춘다)
- 콘솔: 전체 정답 유지율 = 남은 정답 / 띠 안 정답 (통과 기준 95% 이상, Plan.md 3.3 (8))

TODO(A): 아직 뼈대다. 인자와 입출력 설명만 있다(docs/tasks/A.md 3장)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SETS = ("dev39", "diag35")
OUT_DIR = "results/road_mask"


def parse_args(argv=None):
    """확인할 영상 묶음(--set)과 그림·표를 저장할 폴더(--out)를 명령행에서 읽는다."""
    parser = argparse.ArgumentParser(description="노면 마스크 v1·v2를 나란히 겹친 확인 그림과 정답 유지율")
    parser.add_argument("--set", choices=SETS, default="dev39",
                        help="dev39 = 제공 13장 + 직접 촬영 26장, diag35 = 진단 시트 35장 (기본 dev39)")
    parser.add_argument("--out", default=OUT_DIR, help=f"그림·표를 저장할 폴더 (기본 {OUT_DIR})")
    return parser.parse_args(argv)


def main(argv=None):
    """영상마다 v1·v2 마스크를 겹친 그림과 summary.csv를 저장하고 정답 유지율을 출력한다. TODO(A): 아직 구현 전."""
    parse_args(argv)
    raise NotImplementedError("TODO(A): tools/show_road_mask.py는 아직 뼈대입니다 (docs/tasks/A.md 참고)")


if __name__ == "__main__":
    main()
