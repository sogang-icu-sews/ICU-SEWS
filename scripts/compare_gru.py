"""Compare saved GRU OOF probabilities without loading torch or retraining GRU."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icu_sews.evaluation.compare_gru import main  # noqa: E402

if __name__ == "__main__":
    main()
