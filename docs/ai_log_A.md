# 파트 A AI 활용 기록

| 날짜 | 도구 | 프롬프트(요약) | 받은 결과를 어떻게 고쳤나 | 디버깅 과정 |
|---|---|---|---|---|
| 2026-10-05 | Claude Code | (C의 병합 점검 후 C가 요청해 수정) preprocess.py import 누락 보충, parse_name을 약속대로(own은 damage 키, config 태그·번호·칸 수 검사, 밑줄이 여러 개인 제공 영상 이름은 provided), test_a를 pytest 없이 python tests/test_a.py로 실행되게 하고 parse_name 시험을 약속에 맞춤, check_data가 provided·own·pairs 하위 폴더와 확장자·폴더-이름 일치를 검사, tools가 파일 이름으로 실행되게 sys.path 추가, roi.csv·shot_log의 own·pair 이름을 config 태그에 맞춤(day→normal, 손상 normal→none, 쌍 기준→ref, pair02_B_night→pair02_B_blur) (바꾼 파일: src/preprocess.py, src/io_utils.py, tests/test_a.py, tools/check_data.py, tools/show_preprocess.py, data/roi.csv, data/shot_log.csv, docs/ai_log_A.md) | | |
