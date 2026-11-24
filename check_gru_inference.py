from pathlib import Path

from icu_sews.api.gru_loader import load_gru_bundle
from icu_sews.api.gru_predictor import GruInputError, GruPredictor

model_dir = Path.home() / "Projects" / "SEWS-models" / "gru_v2_platt_final"
predictor = GruPredictor(load_gru_bundle(str(model_dir)))

observations = [
    {
        "ICULOS": hour,
        "HR": 85.0,
        "O2Sat": 97.0,
        "Temp": 37.0,
        "SBP": 120.0,
        "MAP": 80.0,
        "Resp": 18.0,
        "Age": 65.0,
        "Gender": 1,
        "Unit1": 1,
        "Unit2": 0,
    }
    for hour in range(1, 25)
]

for count in (1, 24):
    result = predictor.predict(observations[:count])
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["alarm"] == (result["risk_score"] >= result["threshold"])
    print(f"{count}-hour prediction:", result)

without_label = predictor.predict(observations)
with_label = predictor.predict([dict(row, SepsisLabel=1) for row in observations])
assert without_label == with_label
print("SepsisLabel excluded: OK")

try:
    predictor.predict([observations[0], observations[2]])
except GruInputError as exc:
    print("Time gap rejected: OK", exc)
else:
    raise AssertionError("Time gap was not rejected")

print("GRU inference smoke test: OK")
