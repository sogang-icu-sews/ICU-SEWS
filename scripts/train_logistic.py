"""Run from the repository root without installing the full application."""

import sys
from pathlib import Path

if sys.version_info[:3] != (3, 10, 13):
    raise SystemExit(
        "This project requires Python 3.10.13. "
        f"Current interpreter: {sys.version.split()[0]}. "
        "Use the project .venv after setting it up with Python 3.10.13."
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icu_sews.models.train_logistic import main  # noqa: E402

if __name__ == "__main__":
    main()
