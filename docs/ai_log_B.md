# 파트 B AI 활용 기록

| 날짜 | 도구 | 프롬프트(요약) | 받은 결과를 어떻게 고쳤나 | 디버깅 과정 |
|---|---|---|---|---|
| 2026-10-05 | ChatGPT | 제공 4장과 직접 촬영 2장에서 여러 검출 파라미터를 반복 비교하도록 튜닝 도구 확장 | 단일 이미지 하드코딩 방식에서 여러 이미지와 여러 파라미터 세트를 반복 실행하는 방식으로 변경하고, 변경한 config 값은 실행 종료 후 복원하도록 수정함 | 실험 중 변경된 파라미터가 다음 실험에 남을 가능성을 확인하여, 실행 전 원래 값을 백업하고 `finally`에서 복구하도록 수정함 |
| 2026-10-05 | ChatGPT | `*.jpg.json` 파일명 처리 오류 수정 | `.json` 확장자만 제거한 뒤 이미지명과 CSV명을 생성하도록 수정함 | `Norway_008146.jpg.json`에 `stem`을 사용하자 `Norway_008146.jpg.jpg`가 생성됨. `removesuffix(".json")` 방식으로 수정함 |
| 2026-10-05 | ChatGPT | 훈련 데이터 전체 GT 변환 결과 확인 | 훈련 이미지와 annotation을 별도 폴더로 분리하고, 실제 이미지 크기와 JSON 크기가 다른 경우 오류를 내도록 보완함 | 전체 804개 JSON 변환 후 총 1246개 GT 객체가 생성되는지 확인함. 잘못된 좌표 생성을 막기 위해 이미지 크기 검증을 추가함 |
| 2026-10-05 | ChatGPT | 훈련 데이터 804장 전체 정량 평가 | 모든 결과 이미지를 저장하지 않고 세부 결과와 요약 결과를 CSV로 저장하도록 변경함 | 804장 × 여러 파라미터 세트를 돌리면 결과 이미지가 너무 많이 생성될 수 있어, 정량 결과만 `train_detail.csv`, `train_summary.csv`에 저장하도록 함 |
| 2026-10-05 | Claude Code | (C의 병합 점검 후 C가 요청해 수정) test_b를 python tests/test_b.py로 실행되게 sys.path와 실패를 모아 보여 주는 main 추가, tmp_path 시험 2개를 tempfile로 바꿔 직접 실행에서도 돌게 함, tools(tune_detect·convert_json_gt·evaluate_train)가 파일 이름으로 실행되게 sys.path 추가, label_gt의 영상 경로를 명령행 인자로 받음, 이 표의 머리와 행 사이 빈 줄을 지워 표가 깨지지 않게 함 (바꾼 파일: tests/test_b.py, tools/tune_detect.py, tools/convert_json_gt.py, tools/evaluate_train.py, tools/label_gt.py, docs/ai_log_B.md) | | |
