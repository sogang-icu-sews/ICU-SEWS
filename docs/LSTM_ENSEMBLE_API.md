# LSTM and ens_v1 API integration

Owner: 김진호

Supported SEWS_MODEL_TYPE values: logistic, gru, lstm, ensemble (Optuna weights), ensemble_equal.
The existing logistic and final-GRU bundle formats remain unchanged. LSTM and ensembles
use the frozen ens_v1/oof/*_fold4 model files. They do not import train_ensemble.py,
which executes training and data-loading operations at module level.

## Inference contract

- 1–24 consecutive hourly records; invalid used numeric fields are rejected.
- Fixed published cleaning rules and saved fold4 scaling parameters; no fitting on requests.
- LSTM: 24-hour left-padded values/mask/delta window and static covariates.
- GRU in ensemble: packed observed window and saved ens_v1 GRU weights (not the separate final GRU).
- LightGBM: original 125 registered nonoptional features, float32, with missing values retained.
- Ensemble: sigmoid(sum(weight * logit(clip(raw_probability, 1e-7, 1-1e-7)))).
- Apply candidate-specific Platt calibration once, then compare score >= saved threshold.
- No per-component calibration before blending. No cooldown or new threshold selection.
- Response uses existing fields: risk_score, alarm, threshold, model_version, is_mock=false;
  risk_level is alert or below_threshold; top_features is empty (no fabricated SHAP values).
- Startup verifies hashes and shapes; missing/corrupt files leave readiness false and predict returns 503.
- ens_v1_manifest.json pins this specific supplied release. Other releases require explicit review.

## Windows PowerShell

Run in ICU-SEWS-lstm-ensemble:

```powershell
uv sync --group dev --group data
$env:SEWS_USE_MOCK_MODEL = "false"
$env:SEWS_MODEL_TYPE = "ensemble"
$env:SEWS_MODEL_DIR = "$HOME\Projects\SEWS-models\ens_v1"
$env:SEWS_PREPROCESS_DIR = "$PWD\artifacts"
$env:SEWS_TEST_ENSEMBLE_DIR = $env:SEWS_MODEL_DIR
uv run pytest tests/test_ensemble_api.py tests/test_api.py tests/test_model_integration.py -q
uv run uvicorn icu_sews.api.main:app --reload
```

SEWS_PREPROCESS_DIR must contain preprocess_config.json, feature_registry.json,
and preprocess_params/fold4.json matching the pinned release. The supplied repository's
artifacts files matched those hashes during verification. Do not regenerate them.
To use standalone LSTM, set SEWS_MODEL_TYPE=lstm and restart.

## Validation

Linux / Python 3.11 / PyTorch 2.8.0 CPU: 35 targeted API/model tests passed, including
real artifact loading, 1/6/24-hour inference, missing observations, invalid values,
tamper detection, and missing-model HTTP 503. The original training Windows class and
SepsisLSTM class were extracted without executing training; window/forward outputs
matched within 1e-6 for synthetic 1/6/24-hour inputs with missing values.
This verifies inference plumbing; it does not repeat clinical/test-set performance evaluation.
Windows / Python 3.10 must still be checked on the user's machine.

Full suite: 88 passed initially; four pre-existing training/plot tests failed because
matplotlib was absent from the verification environment. After installing matplotlib,
all four passed on retry (92 tests passed across those runs). Changed Python files
passed Ruff checks and formatting. Patch installer apply/reapply/conflict guards passed.
