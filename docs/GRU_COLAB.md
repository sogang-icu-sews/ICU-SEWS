# Colab에서 GRU 학습하기

학습 후 v2/v3 확률 보정·경보 성능 비교는 [별도 비교 노트북 안내](GRU_COMPARISON.md)를 따른다.
비교에는 GPU·GRU 재학습·원본 ZIP이 필요 없다.

담당: 이택훈. 데이터 담당의 v4 전처리, 환자 dev 85% / test 15%, dev 5-fold를 사용한다.

[Colab에서 노트북 열기](https://colab.research.google.com/github/sogang-icu-sews/ICU-SEWS/blob/codex/gru-colab/notebooks/ICU_SEWS_GRU_Colab.ipynb)

## 실행 순서

1. 노트북을 열고 런타임 유형을 GPU로 선택한다. GPU 할당은 Colab 이용 상황에 따라 달라진다.
2. Drive 연결을 승인하고 `SOURCE_ROOT`를 데이터 담당의 자료 위치로 설정한다. 기본은 `MyDrive/capston`이다.
3. 환경 설치, 입력 준비, 학습, 결과 확인 셀을 차례로 실행한다.
4. 기본은 fold 0이다. 확인 후 `FOLDS=[0,1,2,3,4]`로 바꾸면 전체 CV를 실행한다.
5. 중단 후 같은 셀들을 실행하면 Drive의 마지막 완료 epoch부터 재개한다.

Colab 노트북 커널과 분리하여 uv로 Python 3.10 환경을 만들고, 학습은 그 Python으로 실행한다.
PyTorch 2.8.0 CUDA 12.6 휠과 `requirements-gru-colab.txt`의 버전을 사용한다.
로컬 Python 3.10 CPU에서 테스트했으며 Colab GPU 실행 및 전체 데이터 성능은 아직 미검증이다.
설치된 라이브러리가 다르거나 CUDA 드라이버가 맞지 않으면 GPU 확인 셀에서 중단한다.

## 입력 두 가지 방법

**기존 전처리 자료 사용:** `SOURCE_ROOT/data/` 아래에 `seq_values.npy`,
`seq_row_start.npy`, `eval_meta.parquet`, `features.parquet`를 둔다.
`features.parquet` 대신 정적 열만 담은 `gru_static.parquet`도 지원한다.
`SOURCE_ROOT/artifacts/`에는 `patient_split.csv`, `preprocess_config.json`,
`feature_registry.json`, `seq_meta.json`, `preprocess_params/fold0.json`~`fold4.json`이 필요하다.
전처리 결과와 함께 생성된 파라미터를 사용하고 서로 다른 버전의 자료를 섞지 않는다.

**원본 ZIP만 있는 경우:** `SOURCE_ROOT/training.zip`을 둔다. ZIP의 `training_setA`,
`training_setB` 폴더 안에 PSV 파일이 있어야 하며 중첩 폴더도 지원한다.
`scripts/prepare_gru_colab.py`가 A/B 순서와 파일명 정렬로 기존 Patient_ID를 재현한다.
고정 정제 함수와 공개된 fold 파라미터를 재사용하고 모델용 통계를 새로 학습하지 않는다.
환자 수·행 수·병원·환자별 패혈증 여부·시간 간격을 검사하고 원본 파일별 SHA-256을 기록한다.
원본 식별의 전제는 데이터 담당이 사용한 동일한 공식 A/B 파일 집합이다.
파일명-Patient_ID 매핑이 없는 기존 명세의 한계 때문에 행 수·결과 일치만으로 원본 동일성을
완전히 증명하지는 못한다. `gru_source_files.csv`로 이후 실행의 원본을 추적한다.
로지스틱 회귀나 EDA 전체를 실행하지 않고 GRU에 필요한 시계열·정적 입력만 생성한다.
완료된 입력은 `MyDrive/SEWS_GRU/prepared`에 보관해 다음 런타임에서 재사용한다.

입력은 Colab 로컬 디스크에서 읽고 결과만 Drive에 저장한다. 원본·생성 데이터·모델은 Git에 올리지 않는다.

## 모델과 검증

- 시계열 입력: 값 24 + 측정 여부 21 + 측정 간격 21 = 66차원, 최대 24시간.
- 정적 입력: Age, Gender, Unit_MICU, Unit_SICU, Unit_Unknown.
- 단방향 GRU 1층, hidden 64, 분류층 앞 dropout 0.2, 배치 128, Adam 0.001.
- 첫 행 이전의 padding은 실제 기록 길이로 제거한다. 기록 안에서 측정이 없는 시간은 유지한다.
  데이터 모듈의 왼쪽 padding을 오른쪽으로 옮긴 뒤 packed sequence로 GRU에 전달한다.
  기록 길이는 padding 처리에만 사용하며 별도의 숫자 특징으로 넣지 않는다. 짧은 창의 길이 자체는
  모델 동작에 영향을 줄 수 있으므로 ICULOS<=60 결과도 함께 확인한다.
- 정답은 마지막 행의 공식 SepsisLabel, 추가 시프트 없음.
- 각 fold는 나머지 dev fold로만 학습하고 그 fold의 전처리 파라미터를 사용한다.
- 기본 노트북은 BCEWithLogitsLoss의 양성 가중치를 학습 행의 음성/양성 비율로 계산한다.
  가중치 없는 비교 실험은 `POS_WEIGHT=False`와 새 `RUN_NAME`을 사용한다.
- 검증 AUPRC로 최적 epoch 선택, patience 5. AUROC·AUPRC·보정 전 Brier 및 A/B·60시간 이내 지표 저장.
- 최대 epochs를 늘리는 것은 허용하지만, 데이터·코드·설정이 바뀐 체크포인트 재개는 거부한다.
- GPU가 없을 때 CPU로 자동 전환하지 않는다. CPU 테스트는 `--device cpu`를 명시한다.

검증 fold로 early stopping과 설정을 선택하므로 OOF 결과는 개발 성능이다. 독립적인 최종 성능이 아니다.
이 단계에서는 test 예측·평가, 확률 보정, 경보 임계값 선택, API 배포를 수행하지 않는다.
test 행의 파일 구조·환자 분할 일치 여부만 검증한다. 전체 5-fold OOF를 확보한 뒤 보정·임계값
설계를 별도로 결정하고, 확정한 모델과 전처리로 test를 최종 평가해야 한다.

## 저장과 재개

`MyDrive/SEWS_GRU/results/<RUN_NAME>/foldK/`에 `latest.pt`, `best.pt`, `history.json`,
`metrics.json`, `oof.parquet`를 저장한다. 마지막 완료 epoch의 모델·optimizer·난수 상태를 보존한다.
완료 전 끊긴 epoch는 다시 실행한다. `latest.pt`에는 재개용 Python 상태가 있으므로 본인이 생성한
체크포인트만 읽는다. 이 파일의 `weights_only=False` 로드는 외부 출처 체크포인트에 사용하지 않는다.

통합 `oof_predictions.parquet`와 `summary.json`은 현재 명령에 지정한 fold만 포함한다.
모든 fold를 합치려면 마지막에 `--folds 0 1 2 3 4 --resume`으로 실행한다.
`complete_5fold`가 true인지 확인한다. 계산된 점수는 보정 전이며 0.5를 경보 기준으로 확정하지 않는다.

## 참고

검증 기록 (2026-11-10): 로컬 Python 3.10.13 / PyTorch 2.14.1 CPU에서 전체 pytest 19개 통과.
신규 GRU Python 파일의 Ruff 검사·포맷 검사와 노트북 5개 코드 셀의 문법 검사를 통과했다.
기존 데이터 파트 코드에는 저장소 전체 Ruff 검사 오류 443건과 포맷 불일치가 남아 있다.
이번 작업에서 공통 전처리 모듈과 그 SHA-256 계약을 변경하지 않았다.
이는 Colab CUDA 실행이나 전체 실데이터 학습 성능을 검증했다는 뜻은 아니다.

- [Colab 런타임과 GPU 제약](https://research.google.com/colaboratory/faq.html)
- [PyTorch 공식 이전 버전 설치 명령](https://pytorch.org/get-started/previous-versions/)
- [PyTorch 가변 길이 시퀀스](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html)
- [uv Python 설치](https://docs.astral.sh/uv/guides/install-python/)
