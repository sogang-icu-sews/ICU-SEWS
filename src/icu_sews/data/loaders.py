"""Load PhysioNet Challenge 2019 patient PSV files.

Owner: 정상민
Completed in scaffold:
- one-file-per-patient loading;
- stable source-prefixed patient IDs;
- long-format concatenation and chronological sorting.

Next work:
- validate all 40 clinical columns and physiological ranges;
- write EDA summaries for set A and set B;
- add chunked/streaming processing for the complete dataset.
"""

from collections.abc import Iterator, Mapping
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"ICULOS", "SepsisLabel"}


def iter_patient_files(
    raw_dir: Path,
    sources: Mapping[str, str],
) -> Iterator[tuple[str, Path]]:
    """Yield ``(source, path)`` pairs in deterministic order."""
    for source, directory_name in sorted(sources.items()):
        source_dir = raw_dir / directory_name
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {source_dir}")
        for path in sorted(source_dir.glob("*.psv")):
            yield source, path


def load_patient(path: Path, source: str) -> pd.DataFrame:
    """Load one patient file and add stable identification columns."""
    patient = pd.read_csv(path, sep="|")
    missing = REQUIRED_COLUMNS.difference(patient.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

    patient_id = f"{source}_{path.stem}"
    patient.insert(0, "patient_id", patient_id)
    patient.insert(1, "source", source)
    patient.insert(2, "record_file", path.name)

    return patient.sort_values("ICULOS", kind="stable").reset_index(drop=True)


def load_all_patients(raw_dir: Path, sources: Mapping[str, str]) -> pd.DataFrame:
    """Vertically concatenate all patient files into one long-format table."""
    frames = [load_patient(path, source) for source, path in iter_patient_files(raw_dir, sources)]
    if not frames:
        raise ValueError(f"No PSV files found under {raw_dir}")

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["patient_id", "ICULOS"], kind="stable").reset_index(drop=True)


def build_patient_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Create one row per patient for stratified patient-level splitting."""
    required = {"patient_id", "source", "ICULOS", "SepsisLabel"}
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"Records are missing required columns: {sorted(missing)}")

    return (
        records.groupby("patient_id", as_index=False)
        .agg(
            source=("source", "first"),
            total_hours=("ICULOS", "max"),
            is_sepsis=("SepsisLabel", "max"),
        )
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
