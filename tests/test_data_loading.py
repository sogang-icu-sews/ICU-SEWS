"""Tests for patient identity and chronological loading. Owner: 정상민."""

from pathlib import Path

from icu_sews.data.loaders import load_patient


def test_load_patient_adds_source_prefixed_id(tmp_path: Path) -> None:
    path = tmp_path / "p000001.psv"
    path.write_text("ICULOS|HR|SepsisLabel\n2|90|0\n1|80|0\n", encoding="utf-8")

    result = load_patient(path, "A")

    assert result["patient_id"].unique().tolist() == ["A_p000001"]
    assert result["ICULOS"].tolist() == [1, 2]

