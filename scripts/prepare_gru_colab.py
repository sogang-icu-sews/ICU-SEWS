"""Build only GRU inputs from training.zip using the published split and parameters.

No EDA, logistic regression, re-splitting, scaler fitting, or test evaluation is run.
Archive members are read directly, never extracted. Output must be a new directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data/src"))
import sews_features as sf  # noqa: E402


def archive_members(archive):
    """Sort A then B, matching the data team's sequential Patient_ID assignment."""
    members = {"A": [], "B": []}
    for name in archive.namelist():
        path = PurePosixPath(name)
        if path.suffix.lower() != ".psv" or "__MACOSX" in path.parts:
            continue
        sites = [site for site in members if f"training_set{site}" in path.parts]
        if len(sites) != 1:
            raise ValueError(f"Cannot determine A/B source: {name}")
        members[sites[0]].append(name)
    result = []
    for site, names in members.items():
        names.sort(key=lambda name: PurePosixPath(name).name)
        if len({PurePosixPath(n).name for n in names}) != len(names):
            raise ValueError(f"Duplicate filenames in set {site}")
        result.extend((site, name) for name in names)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output is not empty. Reuse completed output, or choose a new output path.")
    art = REPO / "artifacts"
    cfg = json.loads((art / "preprocess_config.json").read_text("utf-8"))
    seq_meta = json.loads((art / "seq_meta.json").read_text("utf-8"))
    registry = json.loads((art / "feature_registry.json").read_text("utf-8"))
    source = (REPO / "data/src/sews_features.py").read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(source).hexdigest() != registry["module_sha256"]:
        raise ValueError("Feature module differs from published preprocessing")
    manifest = pd.read_csv(art / "patient_split.csv").sort_values("Patient_ID")
    if not np.array_equal(manifest.Patient_ID, np.arange(1, len(manifest) + 1)):
        raise ValueError("Expected the published sequential Patient_ID manifest")
    variables = cfg["model_prep"]["sequence"]["vars"]
    n = seq_meta["n_rows"]
    data_dir, art_dir = args.out / "data", args.out / "artifacts"
    data_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    values = np.lib.format.open_memmap(
        data_dir / "seq_values.npy", mode="w+", dtype=np.float32, shape=(n, len(variables))
    )
    starts = np.lib.format.open_memmap(
        data_dir / "seq_row_start.npy", mode="w+", dtype=np.int64, shape=(n,)
    )
    offset, meta_chunks, static_chunks, file_map = 0, [], [], []
    meta_writer = static_writer = None
    try:
        with zipfile.ZipFile(args.archive) as archive:
            members = archive_members(archive)
            if len(members) != len(manifest):
                raise ValueError(f"Expected {len(manifest)} patients; found {len(members)}")
            for i, ((site, name), patient) in enumerate(
                zip(members, manifest.itertuples(index=False), strict=True)
            ):
                raw_bytes = archive.read(name)
                # Read from bytes once so the per-patient checksum covers exactly the parsed input.
                import io

                raw = pd.read_csv(io.BytesIO(raw_bytes), sep="|")
                if raw.empty or not raw.SepsisLabel.isin([0, 1]).all():
                    raise ValueError(f"Invalid labels or empty record: {name}")
                if site != patient.set or int(raw.SepsisLabel.max()) != patient.sepsis:
                    raise ValueError(f"Patient identity/outcome mismatch at {name}")
                if not (np.diff(raw.ICULOS) == 1).all():
                    raise ValueError(f"Non-hourly observations: {name}")
                count = len(raw)
                end = offset + count
                if end > n:
                    raise ValueError("More rows than the published sequence metadata")
                clean, _ = sf.clean(raw, cfg)
                values[offset:end] = clean[variables].to_numpy(np.float32)
                starts[offset:end] = offset
                labels = raw.SepsisLabel.to_numpy(dtype=np.int64)
                hours = raw.ICULOS.to_numpy(dtype=np.int64)
                positives = np.flatnonzero(labels)
                onset = float(hours[positives[0]] + 6) if len(positives) else np.nan
                rel = hours - onset
                ids = np.arange(offset, end, dtype=np.int64)
                meta_chunks.append(
                    pd.DataFrame(
                        {
                            "row_id": ids,
                            "Patient_ID": int(patient.Patient_ID),
                            "ICULOS": hours,
                            "set": site,
                            "split": patient.split,
                            "fold": int(patient.fold),
                            "SepsisLabel": labels,
                            "t_sepsis": onset,
                            "rel": rel,
                            "label_at_start": int(labels[0] == 1),
                            "iculos_le60": (hours <= 60).astype(int),
                            "early_window": ((rel >= -12) & (rel < -6)).astype(int),
                            "obs_hours": np.minimum(np.arange(count) + 1, 24),
                        }
                    )
                )
                static = clean[["Age", "Gender"]].astype(np.float32).copy()
                static.insert(0, "row_id", ids)
                for unit in sf.UNITS:
                    static[f"Unit_{unit}"] = (clean.Unit == unit).to_numpy(dtype=np.float32)
                static_chunks.append(static)
                file_map.append(
                    {
                        "Patient_ID": int(patient.Patient_ID),
                        "source": name,
                        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                        "rows": count,
                    }
                )
                offset = end
                if len(meta_chunks) >= 500 or i == len(members) - 1:
                    mt = pa.Table.from_pandas(
                        pd.concat(meta_chunks, ignore_index=True), preserve_index=False
                    )
                    st = pa.Table.from_pandas(
                        pd.concat(static_chunks, ignore_index=True), preserve_index=False
                    )
                    if meta_writer is None:
                        meta_writer = pq.ParquetWriter(data_dir / "eval_meta.parquet", mt.schema)
                        static_writer = pq.ParquetWriter(data_dir / "gru_static.parquet", st.schema)
                    meta_writer.write_table(mt)
                    static_writer.write_table(st)
                    meta_chunks.clear()
                    static_chunks.clear()
                    print(
                        f"Prepared {i + 1:,}/{len(members):,} patients; {offset:,} rows", flush=True
                    )
        if offset != n:
            raise ValueError(f"Expected {n} rows, found {offset}")
    finally:
        if meta_writer is not None:
            meta_writer.close()
            static_writer.close()
        values.flush()
        starts.flush()
    for filename in [
        "preprocess_config.json",
        "feature_registry.json",
        "seq_meta.json",
        "patient_split.csv",
    ]:
        shutil.copy2(art / filename, art_dir / filename)
    shutil.copytree(art / "preprocess_params", art_dir / "preprocess_params")
    pd.DataFrame(file_map).to_csv(art_dir / "gru_source_files.csv", index=False)
    (art_dir / "gru_preparation_complete.json").write_text(
        json.dumps(
            {
                "rows": offset,
                "patients": len(manifest),
                "config_version": cfg["version"],
                "note": "Published split/parameters reused; no fitting or test evaluation.",
            },
            indent=2,
        ),
        "utf-8",
    )
    print(f"GRU inputs ready: {args.out}", flush=True)


if __name__ == "__main__":
    main()
