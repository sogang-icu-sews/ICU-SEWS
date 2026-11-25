"""Hourly inference using the frozen ICU-7 LSTM and logit-weighted ensemble."""

import math
from numbers import Real

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit

from icu_sews.api.gru_predictor import GruInputError
from icu_sews.models.train_gru import module, pack_inputs


class EnsemblePredictor:
    def __init__(self, bundle):
        self.bundle = bundle
        self.features = module("sews_features")
        self.preprocess = module("sews_preprocess")

    def prepare(self, observations):
        if not 1 <= len(observations) <= 24:
            raise GruInputError("Provide between 1 and 24 observations.", "observations")
        cfg = self.bundle["config"]
        columns = list(
            dict.fromkeys(
                cfg["model_prep"]["sequence"]["vars"]
                + cfg["feature_vars"]["vitals_window"]
                + cfg["feature_vars"]["labs"]
                + ["Calcium", "Age", "Gender", "Unit1", "Unit2"]
            )
        )
        records, previous_hour = [], None
        for index, observation in enumerate(observations):
            hour = observation.get("ICULOS")
            if (
                isinstance(hour, bool)
                or not isinstance(hour, Real)
                or not math.isfinite(float(hour))
                or hour < 1
                or int(hour) != hour
            ):
                raise GruInputError("ICULOS must be a positive integer.", "ICULOS", index)
            if previous_hour is not None and hour != previous_hour + 1:
                raise GruInputError(
                    "Observations must be consecutive hourly records.", "ICULOS", index
                )
            previous_hour = hour
            record = {"ICULOS": hour, "Patient_ID": 0}
            for column in columns:
                value = observation.get(column)
                if value is None:
                    record[column] = np.nan
                elif (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                ):
                    raise GruInputError(f"{column} must be a finite number or null.", column, index)
                else:
                    record[column] = float(value)
            records.append(record)
        cleaned, _ = self.features.clean(pd.DataFrame(records), cfg)
        windows = self.preprocess.seq_window_from_frame(
            cleaned, self.bundle["preprocessing"]["sequence"], cfg
        )
        left = np.concatenate([windows[k] for k in ("x", "mask", "delta")], axis=2)
        lengths = np.array([len(cleaned)], dtype=np.int64)
        packed = pack_inputs(windows, lengths)
        static = cleaned[["Age", "Gender"]].copy()
        for unit in self.features.UNITS:
            static[f"Unit_{unit}"] = (cleaned["Unit"] == unit).astype(float)
        static = self.preprocess.apply_tabular(
            static.tail(1), self.bundle["preprocessing"]["tabular"], self.bundle["static_columns"]
        )
        if any(not np.isfinite(x).all() for x in (left, packed, static)):
            raise RuntimeError("Preprocessing produced non-finite neural-network inputs")
        tabular = None
        if self.bundle["mode"] != "lstm":
            # Same vectorized formula as training, restricted to this patient's <=24 hours.
            with np.errstate(divide="ignore", invalid="ignore"):
                features = self.features.build_features(cleaned, cfg)
            tabular = features[self.bundle["feature_names"]].tail(1).to_numpy(np.float32)
            if np.isinf(tabular).any():
                raise RuntimeError("Preprocessing produced infinite LightGBM inputs")
        return left, packed, static, lengths, tabular

    def predict(self, observations):
        left, packed, static, lengths, tabular = self.prepare(observations)
        models = self.bundle["models"]
        with torch.inference_mode():
            st = torch.from_numpy(static)
            probs = {
                "lstm": float(torch.sigmoid(models["lstm"](torch.from_numpy(left), st)).item())
            }
            if self.bundle["mode"] != "lstm":
                probs["gru"] = float(
                    torch.sigmoid(
                        models["gru"](torch.from_numpy(packed), st, torch.from_numpy(lengths))
                    ).item()
                )
                probs["lgbm"] = float(models["lgbm"].predict(tabular, num_threads=1)[0])
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()):
            raise RuntimeError("Model returned an invalid probability")
        clip = 1e-7  # ICU-7 train_ensemble.py CLIP and final_gru.apply_platt default.
        if self.bundle["mode"] == "lstm":
            raw = probs["lstm"]
        else:
            raw = float(
                expit(
                    sum(
                        self.bundle["weights"][k] * logit(np.clip(probs[k], clip, 1 - clip))
                        for k in ("lgbm", "lstm", "gru")
                    )
                )
            )
        cal = self.bundle["platt"]
        score = float(expit(cal["slope"] * logit(np.clip(raw, clip, 1 - clip)) + cal["intercept"]))
        if not math.isfinite(score):
            raise RuntimeError("Calibration returned an invalid probability")
        threshold = self.bundle["threshold"]
        return {
            "risk_score": score,
            "alarm": score >= threshold,
            "threshold": threshold,
            "model_version": self.bundle["model_version"],
            "is_mock": False,
            "raw_probability": raw,
            "component_probabilities": probs,
        }
