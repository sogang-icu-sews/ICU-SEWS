"""Load the trained logistic baseline and its input feature order."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


@dataclass
class LoadedLogisticModel:
    """A trained pipeline and the features it expects."""

    pipeline: Pipeline
    features: list[str]


def load_logistic_model(model_dir: str) -> LoadedLogisticModel:
    """Load files produced by train_logistic.py."""

    directory = Path(model_dir)
    pipeline_path = directory / "logistic_pipeline.joblib"
    metadata_path = directory / "metadata.json"

    # 두 파일이 모두 있어야 모델을 사용할 수 있다.
    if not pipeline_path.is_file():
        raise FileNotFoundError(f"Model file not found: {pipeline_path}")

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # 학습할 때 사용한 입력 항목과 순서를 읽는다.
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata["model_config"]["features"]

    if not isinstance(features, list) or not features:
        raise ValueError("Model features must be a nonempty list")

    if any(not isinstance(name, str) or not name.strip() for name in features):
        raise ValueError("Each model feature must be a nonempty string")

    if len(features) != len(set(features)):
        raise ValueError("Model features must be unique")

    # 프로젝트에서 학습하거나 팀원이 전달한 신뢰할 수 있는 파일만 읽는다.
    pipeline = joblib.load(pipeline_path)

    if not isinstance(pipeline, Pipeline):
        raise ValueError("Model file must contain a scikit-learn Pipeline")

    required_steps = ["imputer", "scaler", "classifier"]
    if list(pipeline.named_steps) != required_steps:
        raise ValueError("Unexpected logistic pipeline steps")

    # 전처리와 분류기가 학습을 마친 상태인지 확인한다.
    for step_name in required_steps:
        check_is_fitted(pipeline.named_steps[step_name])

    if not callable(getattr(pipeline, "predict_proba", None)):
        raise ValueError("Model must support probability prediction")

    # 메타데이터의 입력 순서가 실제 학습된 모델과 일치해야 한다.
    trained_features = getattr(pipeline, "feature_names_in_", None)
    if trained_features is None or list(trained_features) != features:
        raise ValueError("Metadata features do not match the trained model")

    # 현재 모델은 0과 1로 분류하며, 1이 SepsisLabel 양성을 의미한다.
    classes = list(pipeline.classes_)
    if classes != [0, 1]:
        raise ValueError("Model classes must be [0, 1]")

    return LoadedLogisticModel(
        pipeline=pipeline,
        features=features,
    )
