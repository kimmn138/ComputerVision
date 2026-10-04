# 파트 C AI 활용 기록

| 날짜 | 도구 | 프롬프트(요약) | 받은 결과를 어떻게 고쳤나 | 디버깅 과정 |
|---|---|---|---|---|
| 2026-10-03 | Claude Code | 저장소 초기 세팅: 폴더 구조·규칙 문서·임시 함수·약속 검사 | | 없음 |
| 2026-10-03 | Claude Code | README·CONTRIBUTING을 실제 저장소에 맞게 수정: clone 폴더 이름(ComputerVision), 브랜치 이름(feat/…), Python 3.12 이상 (바꾼 파일: README.md, CONTRIBUTING.md) | | 없음 |
| 2026-10-04 | Claude Code | features.count_harris(R → 임계값 → 비최대 억제)·match_pair(SIFT → 비율 검사 → RANSAC) 실제 구현, config C 구역에 HARRIS_NMS·RANSAC_MIN_GOOD 추가, C 시험 작성 (바꾼 파일: src/features.py, src/config.py, tests/test_c.py, docs/part_C.md, docs/ai_log_C.md) | | 빈 영상에서 R >= 임계값이면 모든 픽셀이 세어지는 것, knnMatch가 이웃 1개만 주는 것, 일직선 4점에서도 RANSAC 비율이 1.0인 것을 OpenCV 5.0으로 직접 확인해 가드를 넣음 |
