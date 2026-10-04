
"""(파트 A) 영상 읽기·가로 크기 맞추기·노면(ROI) 자르기, roi.csv와 사진 이름 해석."""
import os
import csv
import numpy as np
import cv2 as cv

from src import config as C


def load_image(path):
    """사진을 BGR uint8 컬러로 읽는다. 못 읽으면 None 대신 FileNotFoundError를 내서 바로 알아채게 한다.
    윈도우 한글 경로 대비 cv.imdecode(np.fromfile(path, np.uint8), cv.IMREAD_COLOR)로 한 번 더 시도한다."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    # 1차 시도: 기본 imread
    img = cv.imread(path, cv.IMREAD_COLOR)
    
    # 2차 시도: 윈도우 환경 한글 경로 인식 오류 대비
    if img is None:
        try:
            img_array = np.fromfile(path, np.uint8)
            img = cv.imdecode(img_array, cv.IMREAD_COLOR)
        except Exception:
            pass

    if img is None:
        raise FileNotFoundError(f"이미지 파일을 읽을 수 없습니다: {path}")
        
    return img


def resize_width(img, width=C.TARGET_WIDTH):
    """영상마다 크기가 달라도 파라미터가 같은 뜻을 갖도록 가로를 width로 맞춘다.
    세로는 비율대로 바꾸고 채널 수는 그대로 둔다."""
    h, w = img.shape[:2]
    if w == width:
        return img.copy()

    interp = cv.INTER_AREA if width < w else cv.INTER_LINEAR    # 줄일 때 AREA, 키울 때 LINEAR
    return cv.resize(img, (width, round(h * width / w)), interpolation=interp)


def crop_roi(img, top=C.ROI_TOP_DEFAULT, bottom=C.ROI_BOTTOM_DEFAULT):
    """노면 밖(하늘·차·건물)을 빼려고 위·아래 비율(0~1)로 잘라 노면만 남긴다.
    (노면 영상, 위에서 잘라 낸 픽셀 수 y0)를 돌려준다."""
    h = img.shape[0]
    y0, y1 = int(h * top), int(h * bottom)
    
    # 안전하게 인덱스 범위 제한
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))
    
    return img[y0:y1].copy(), y0


def load_roi_table(path):
    """사진마다 다른 노면 범위(top·bottom 비율)와 촬영 조건을 roi.csv에서 읽는다.
    utf-8-sig로 읽어 {파일명: {"top": float, "bottom": float, "condition": str}} 형태로 반환하며, 
    파일이 없으면 빈 dict를 반환하고 데이터 형식이 잘못된 행은 ValueError를 낸다."""
    if not os.path.exists(path):
        return {}

    roi_dict = {}
    try:
        # utf-8-sig를 사용하여 BOM(Byte Order Mark) 문제 해결
        with open(path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"].strip()
                top_val = float(row["top"])
                bottom_val = float(row["bottom"])
                condition = row.get("condition", "").strip()
                
                roi_dict[filename] = {
                    "top": top_val,
                    "bottom": bottom_val,
                    "condition": condition
                }
    except KeyError as e:
        raise ValueError(f"roi.csv에 필수 컬럼이 누락되었습니다: {e}")
    except ValueError as e:
        raise ValueError(f"roi.csv의 숫자 형식이 잘못되었습니다: {e}")

    return roi_dict


def parse_name(path):
    """사진 파일 이름에서 출처·손상·촬영 조건을 읽어 결과 표에 붙인다.
    own_번호_손상_조건.jpg와 pair번호_기호_조건.jpg 해석, 규칙 위반은 ValueError."""
    filename = os.path.basename(path)
    name_without_ext = os.path.splitext(filename)[0]
    tokens = name_without_ext.split("_")

    # 1. 'own' 규칙: own_번호_손상_조건 (예: own_01_crack_low)
    if name_without_ext.startswith("own_"):
        if len(tokens) < 4:
            raise ValueError(f"own 이름 규칙 위반 (토큰 부족): {filename}")
        return {
            "source": tokens[0],
            "pair": tokens[1],
            "role": tokens[2],
            "condition": tokens[3]
        }
        
    # 2. 'pair' 규칙: pair번호_기호_조건 (예: pair01_A_low -> ['pair01', 'A', 'low'])
    elif name_without_ext.startswith("pair"):
        if len(tokens) < 3:
            raise ValueError(f"pair 이름 규칙 위반 (토큰 부족): {filename}")
            
        pair_token = tokens[0]
        pair_num = pair_token.replace("pair", "")
        if not pair_num:
            raise ValueError(f"pair 번호가 누락되었습니다: {filename}")
            
        return {
            "source": "pair",
            "pair": pair_num,
            "role": tokens[1],
            "condition": tokens[2]
        }
        
    # 3. 그 외 일반 제공 데이터: 출처_손상_조건
    else:
        if len(tokens) >= 3:
            return {
                "source": tokens[0],
                "damage": tokens[1],
                "condition": tokens[2]
            }
        else:
            return {"source": "provided", "damage": "", "condition": ""}
            
