"""(파트 A) 영상 읽기·가로 크기 맞추기·노면(ROI) 자르기, roi.csv와 사진 이름 해석."""
import cv2 as cv

from src import config as C


def load_image(path):
    """사진을 BGR uint8 컬러로 읽는다. 못 읽으면 None 대신 FileNotFoundError를 내서 바로 알아채게 한다.
    임시 버전: cv.imread로 한 번만 시도한다."""
    img = cv.imread(path)
    # TODO(A): 윈도 한글 경로 대비 cv.imdecode(np.fromfile(path, np.uint8), cv.IMREAD_COLOR)로 한 번 더 시도
    if img is None:
        raise FileNotFoundError(path)
    return img


def resize_width(img, width=C.TARGET_WIDTH):
    """영상마다 크기가 달라도 파라미터가 같은 뜻을 갖도록 가로를 width로 맞춘다.
    세로는 비율대로 바꾸고 채널 수는 그대로 둔다."""
    h, w = img.shape[:2]
    interp = cv.INTER_AREA if width < w else cv.INTER_LINEAR    # 줄일 때 AREA, 키울 때 LINEAR
    return cv.resize(img, (width, round(h * width / w)), interpolation=interp)


def crop_roi(img, top=C.ROI_TOP_DEFAULT, bottom=C.ROI_BOTTOM_DEFAULT):
    """노면 밖(하늘·차·건물)을 빼려고 위·아래 비율(0~1)로 잘라 노면만 남긴다.
    (노면 영상, 위에서 잘라 낸 픽셀 수 y0)를 돌려준다."""
    h = img.shape[0]
    y0, y1 = int(h * top), int(h * bottom)
    return img[y0:y1], y0


def load_roi_table(path):
    """사진마다 다른 노면 범위(top·bottom 비율)와 촬영 조건을 roi.csv에서 읽는다.
    임시 버전: 항상 빈 dict를 돌려준다(모든 사진이 config 기본값을 씀)."""
    # TODO(A): utf-8-sig로 읽어 {파일명: {"top": float, "bottom": float, "condition": str}}, 잘못된 행은 ValueError
    return {}


def parse_name(path):
    """사진 파일 이름에서 출처·손상·촬영 조건을 읽어 결과 표에 붙인다.
    임시 버전: 모든 사진을 손상·조건 정보가 없는 제공 사진으로 본다."""
    # TODO(A): own_번호_손상_조건.jpg와 pair번호_기호_조건.jpg 해석, 규칙 위반은 ValueError
    return {"source": "provided", "damage": "", "condition": ""}
