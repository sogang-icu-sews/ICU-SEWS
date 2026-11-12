"""데이터 파트 테스트 준비물: 설정 파일과 가짜 환자 데이터 (실제 환자 데이터는 쓰지 않는다)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES = Path(__file__).parent / "fixtures"

# PhysioNet 2019 원본 열 순서 (master.parquet와 같음)
RAW_COLS = [
    "Patient_ID", "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
    "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets", "Age",
    "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS", "SepsisLabel", "set",
]  # fmt: skip
STATIC = {"Patient_ID", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime"}
STATIC |= {"ICULOS", "SepsisLabel", "set"}


def load_cfg() -> dict:
    """ICU-3에서 확정한 preprocess_config.json (v4, 기준값만 있고 환자 데이터 없음)."""
    with open(FIXTURES / "preprocess_config.json", encoding="utf-8") as f:
        return json.load(f)


def make_patients(cfg: dict, n: int = 40, seed: int = 0) -> pd.DataFrame:
    """가짜 환자 n명: 길이 1~60시간, 생리적 범위 안팎의 값, 변수마다 다른 결측률.

    일부러 범위 밖 값(정제 규칙 확인용), 패혈증 환자, 24시간 미만 기록을 섞는다.
    """
    rng = np.random.default_rng(seed)
    ranges = cfg["plausible_ranges"]
    pieces = []
    for pid in range(1, n + 1):
        T = int(rng.integers(1, 61))
        df = pd.DataFrame(np.nan, index=range(T), columns=RAW_COLS)
        df["Patient_ID"] = pid
        df["ICULOS"] = np.arange(1, T + 1)
        df["Age"] = rng.choice([45.0, 70.0, 100.0])  # 100 = 재코딩 대상
        df["Gender"] = float(rng.integers(0, 2))
        unit = rng.integers(0, 3)  # MICU, SICU, 둘 다 NaN(Unknown)
        df["Unit1"] = 1.0 if unit == 0 else (0.0 if unit == 1 else np.nan)
        df["Unit2"] = 1.0 if unit == 1 else (0.0 if unit == 0 else np.nan)
        df["HospAdmTime"] = -float(rng.integers(0, 100))
        df["set"] = "A"
        for c in RAW_COLS:
            if c in STATIC:
                continue
            lo, hi = ranges.get(c, [0.0, 100.0])
            p_obs = rng.choice([0.0, 0.05, 0.3, 0.9])
            obs = rng.random(T) < p_obs
            vals = rng.uniform(lo, hi, T)
            out = rng.random(T) < 0.03  # 범위 밖 값
            vals[out] = hi * 1.5 + 1
            df.loc[obs, c] = vals[obs]
        df.loc[rng.random(T) < 0.3, "Calcium"] = 1.1  # 이온화칼슘 분리 규칙
        lab = np.zeros(T, int)
        if pid % 5 == 0 and T > 8:  # 패혈증 환자: 공식 라벨처럼 어느 시점부터 끝까지 1
            lab[int(rng.integers(0, T - 3)) :] = 1
        df["SepsisLabel"] = lab
        pieces.append(df)
    return pd.concat(pieces, ignore_index=True)
