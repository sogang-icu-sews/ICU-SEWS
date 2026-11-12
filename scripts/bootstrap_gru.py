"""Compute confidence intervals from frozen predictions without loading torch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icu_sews.evaluation.bootstrap_gru import main  # noqa: E402

if __name__ == "__main__":
    main()
