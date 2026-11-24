"""Prepare patient observations and run the frozen GRU model."""

import math
from numbers import Real

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit

from icu_sews.models.train_gru import module, pack_inputs


class GruInputError(ValueError):
    """Invalid patient observation."""

    def __init__(self, message, field, observation_index=None):
        super().__init__(message)
        self.field = field
        self.observation_index = observation_index


class GruPredictor:
    def __init__(self, bundle):
        self.bundle = bundle
        self.features = module("sews_features")
        self.preprocess = module("sews_preprocess")

    def prepare(self, observations):
        """Convert one patient's consecutive hourly records to tensors."""
        if not 1 <= len(observations) <= 24:
            raise GruInputError(
                "Provide between 1 and 24 observations.",
                "observations",
            )

        sequence_vars = self.bundle["preprocessing"]["sequence"]["vars"]
        columns = list(dict.fromkeys(sequence_vars + ["Age", "Gender", "Unit1", "Unit2"]))
        records = []
        previous_hour = None

        for index, observation in enumerate(observations):
            hour = observation.get("ICULOS")
            if (
                isinstance(hour, bool)
                or not isinstance(hour, Real)
                or not math.isfinite(float(hour))
                or hour < 1
                or int(hour) != hour
            ):
                raise GruInputError(
                    "ICULOS must be a positive integer.",
                    "ICULOS",
                    index,
                )

            if previous_hour is not None and hour != previous_hour + 1:
                raise GruInputError(
                    "Observations must be consecutive hourly records.",
                    "ICULOS",
                    index,
                )
            previous_hour = hour

            record = {}
            for column in columns:
                value = observation.get(column)
                if value is None:
                    record[column] = np.nan
                    continue

                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                ):
                    raise GruInputError(
                        f"{column} must be a finite number or null.",
                        column,
                        index,
                    )
                record[column] = float(value)

            records.append(record)

        frame = pd.DataFrame(records, columns=columns)
        cleaned, _ = self.features.clean(frame, self.bundle["config"])

        windows = self.preprocess.seq_window_from_frame(
            cleaned,
            self.bundle["preprocessing"]["sequence"],
            self.bundle["config"],
        )
        lengths = np.array([len(cleaned)], dtype=np.int64)
        sequence = pack_inputs(windows, lengths)

        static = cleaned[["Age", "Gender"]].astype(np.float32).copy()
        for unit in self.features.UNITS:
            static[f"Unit_{unit}"] = (cleaned["Unit"] == unit).astype(np.float32)

        static_values = self.preprocess.apply_tabular(
            static.tail(1),
            self.bundle["preprocessing"]["tabular"],
            self.bundle["static_columns"],
        )

        if not np.isfinite(sequence).all() or not np.isfinite(static_values).all():
            raise RuntimeError("Preprocessing produced non-finite values.")

        return (
            torch.from_numpy(sequence),
            torch.from_numpy(static_values),
            torch.from_numpy(lengths),
        )

    def predict(self, observations):
        sequence, static, lengths = self.prepare(observations)

        with torch.inference_mode():
            logits = self.bundle["model"](sequence, static, lengths)
            raw_probability = float(torch.sigmoid(logits).item())

        if not math.isfinite(raw_probability):
            raise RuntimeError("GRU produced a non-finite probability.")

        calibration = self.bundle["platt"]
        clip = float(calibration["clip"])
        clipped = np.clip(raw_probability, clip, 1.0 - clip)
        score = float(
            expit(float(calibration["slope"]) * logit(clipped) + float(calibration["intercept"]))
        )
        threshold = float(self.bundle["threshold"]["threshold"])

        return {
            "risk_score": score,
            "alarm": score >= threshold,
            "threshold": threshold,
            "model_version": self.bundle["model_version"],
            "is_mock": False,
        }
