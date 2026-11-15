"""Entry point for patient-fold GRU training; no editable package install required."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icu_sews.models.train_gru import main  # noqa: E402

if __name__ == "__main__":
    main()
