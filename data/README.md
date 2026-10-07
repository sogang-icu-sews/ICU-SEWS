# 데이터 디렉터리

담당: 정상민

- `raw/`: 다운로드한 PhysioNet PSV 원본. 수정하지 않고 Git에 커밋하지 않는다.
- `interim/`: 정렬·정제 등 중간 처리 결과. 재생성 가능해야 한다.
- `processed/`: 모델 입력용 Parquet, NPZ 등. 생성 코드와 메타데이터만 공유한다.

실제 환자 데이터 대신 `.gitkeep`만 저장소에 포함한다. 데이터 다운로드 출처, 버전, 체크섬, 생성 명령은 향후 이 문서에 기록한다.

## 로지스틱 회귀 기준선의 공통 분할

`scripts/train_logistic.py`는 `processed/patient_split.parquet`를 처음에 생성하고
이후 실행에서 재사용한다. 열은 `patient_id`, `split`이며 분할 값은
`train`, `validation`, `test`다. 동일 환자의 시간 행은 모두 한 분할에 속한다.
원본 환자 집합이 바뀌면 기존 분할을 임의로 갱신하지 않고 오류로 중단한다.
공통 분할을 변경할 때는 데이터·모델 담당자가 함께 검토한다.

현재 로컬 원본은 `raw/training_setA/training`과
`raw/training_setB/training_setB`에 있다. 각 실행의 원본 내용 SHA-256 요약과
분할 해시는 모델의 `metadata.json`에 기록한다.
자세한 실행 절차는 [`LOGISTIC_BASELINE.md`](../docs/LOGISTIC_BASELINE.md)를 참고한다.
