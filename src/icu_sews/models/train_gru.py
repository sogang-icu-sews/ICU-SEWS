"""Train GRU on the data team's fixed dev folds, with epoch-level recovery.

Only dev predictions are exported. CV scores used for early stopping are development
estimates, not independent test estimates. Checkpoints are trusted local files only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn

from icu_sews.models.gru import GruSepsisModel

REPO = Path(__file__).resolve().parents[3]
STATIC = ["Age", "Gender", "Unit_MICU", "Unit_SICU", "Unit_Unknown"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def module(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "data/src" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    temp.replace(path)


def atomic_checkpoint(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def validate_alignment(em, starts, manifest):
    """Reject patient leakage, reordered arrays, invalid labels and time gaps."""
    n = len(em)
    if not np.array_equal(em.row_id, np.arange(n)):
        raise ValueError("eval_meta.row_id must be contiguous and start at zero")
    if not em.SepsisLabel.isin([0, 1]).all():
        raise ValueError("SepsisLabel must be binary")
    if not manifest.Patient_ID.is_unique:
        raise ValueError("Duplicate patients in split manifest")
    pts = em.groupby("Patient_ID", sort=False)
    if (pts[["split", "fold", "set"]].nunique() != 1).any().any():
        raise ValueError("A patient's rows cross splits/folds/sites")
    observed = pts[["split", "fold", "set"]].first().sort_index()
    expected = manifest.set_index("Patient_ID")[["split", "fold", "set"]].sort_index()
    if not observed.equals(expected):
        raise ValueError("Metadata does not match the supplied patient split")
    if (
        not pts.SepsisLabel.max()
        .sort_index()
        .equals(manifest.set_index("Patient_ID").sepsis.sort_index().astype(em.SepsisLabel.dtype))
    ):
        raise ValueError("Patient outcomes do not match the split manifest")
    if not em.split.isin(["dev", "test"]).all():
        raise ValueError("Only dev/test splits are supported")
    if set(em.loc[em.split == "dev", "fold"]) != set(range(5)):
        raise ValueError("Expected all five development folds")
    if not (em.loc[em.split == "test", "fold"] == -1).all():
        raise ValueError("Test fold must be -1")
    pid = em.Patient_ID.to_numpy()
    changes = np.r_[True, pid[1:] != pid[:-1]]
    if changes.sum() != em.Patient_ID.nunique():
        raise ValueError("Patient rows must be contiguous")
    expected_starts = np.maximum.accumulate(np.where(changes, np.arange(n), 0))
    if not np.array_equal(starts, expected_starts):
        raise ValueError("seq_row_start does not match patient boundaries")
    if not (np.diff(em.ICULOS.to_numpy())[~changes[1:]] == 1).all():
        raise ValueError("Patient observations must be one hour apart")


def load_data(root):
    art, data = root / "artifacts", root / "data"
    cfg = read_json(art / "preprocess_config.json")
    registry = read_json(art / "feature_registry.json")
    for name, expected in [
        ("sews_features", registry["module_sha256"]),
        ("sews_preprocess", registry["preprocess"]["module_sha256"]),
    ]:
        # Git text files are normalized to LF; Windows checkout must not alter identity.
        source = (REPO / "data/src" / f"{name}.py").read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError(f"{name}: preprocessing module hash mismatch")
    if cfg["version"] != 4 or cfg["window"]["hours"] != 24:
        raise ValueError("Expected preprocessing config v4 and a 24-hour window")
    em = pd.read_parquet(data / "eval_meta.parquet")
    values = np.load(data / "seq_values.npy", mmap_mode="r", allow_pickle=False)
    starts = np.load(data / "seq_row_start.npy", mmap_mode="r", allow_pickle=False)
    spec = cfg["model_prep"]["sequence"]
    seq_meta = read_json(art / "seq_meta.json")
    if seq_meta["vars"] != spec["vars"] or seq_meta["row_order"] != "row_id":
        raise ValueError("Sequence metadata does not match the configured variable order")
    if values.shape != (len(em), len(spec["vars"])) or values.dtype != np.float32:
        raise ValueError("Sequence values must be float32 with the expected shape")
    validate_alignment(em, starts, pd.read_csv(art / "patient_split.csv"))
    static_path = data / "gru_static.parquet"
    if not static_path.exists():
        static_path = data / "features.parquet"
    static = pd.read_parquet(static_path, columns=["row_id", *STATIC])
    if not np.array_equal(static.row_id, em.row_id):
        raise ValueError("Static features and metadata row_id differ")
    paths = [data / f for f in ["seq_values.npy", "seq_row_start.npy", "eval_meta.parquet"]]
    paths += [static_path, *sorted(art.glob("*.json")), art / "patient_split.csv"]
    paths += sorted((art / "preprocess_params").glob("fold*.json"))
    return {
        "cfg": cfg,
        "em": em,
        "values": values,
        "starts": starts,
        "static": static,
        "labels": em.SepsisLabel.to_numpy(dtype=np.float32),
        "hashes": {str(p.relative_to(root)): digest(p) for p in paths},
        "pp": module("sews_preprocess"),
        "root": root,
    }


def pack_inputs(windows, lengths):
    """Move real left-padded observations to the front for PyTorch packing."""
    x = np.concatenate([windows[k] for k in ["x", "mask", "delta"]], axis=-1)
    steps = np.arange(x.shape[1])[None, :]
    idx = np.minimum(x.shape[1] - lengths[:, None] + steps, x.shape[1] - 1)
    out = np.take_along_axis(x, idx[..., None], axis=1)
    out[steps >= lengths[:, None]] = 0
    return np.ascontiguousarray(out, dtype=np.float32)


def batches(d, rows, params, static, batch_size, device, seed=None):
    pp = d["pp"]
    for ids, win in pp.iter_batches(
        rows,
        batch_size,
        d["values"],
        d["starts"],
        params["sequence"],
        d["cfg"],
        shuffle=seed is not None,
        seed=0 if seed is None else seed,
    ):
        lengths = np.minimum(ids - d["starts"][ids] + 1, 24).astype(np.int64)
        x = torch.from_numpy(pack_inputs(win, lengths)).to(device)
        s = torch.from_numpy(static[ids]).to(device)
        y = torch.from_numpy(d["labels"][ids]).to(device)
        yield ids, x, s, torch.from_numpy(lengths), y


def metrics(em, p):
    def pair(frame, probs):
        y = frame.SepsisLabel.to_numpy()
        if len(np.unique(y)) < 2:
            return {"AUROC": None, "AUPRC": None, "rows": len(y)}
        return {
            "AUROC": float(roc_auc_score(y, probs)),
            "AUPRC": float(average_precision_score(y, probs)),
            "rows": len(y),
        }

    result = {"all": pair(em, p), "Brier_uncalibrated": float(brier_score_loss(em.SepsisLabel, p))}
    for name, mask in [
        ("ICULOS_le60", em.ICULOS <= 60),
        ("set_A", em["set"] == "A"),
        ("set_B", em["set"] == "B"),
    ]:
        result[name] = pair(em[mask], p[mask.to_numpy()])
    return result


@torch.no_grad()
def predict(model, d, rows, params, static, batch_size, device):
    model.eval()
    preds = np.empty(len(rows), dtype=np.float32)
    offset = 0
    for _, x, s, lengths, _ in batches(d, rows, params, static, batch_size, device):
        p = model(x, s, lengths).sigmoid().cpu().numpy()
        preds[offset : offset + len(p)] = p
        offset += len(p)
    return preds


def run_fold(d, args, fold, device):
    seed = args.seed + fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    em = d["em"]
    tr = np.flatnonzero((em.split == "dev") & (em.fold != fold))
    va = np.flatnonzero((em.split == "dev") & (em.fold == fold))
    params = read_json(d["root"] / f"artifacts/preprocess_params/fold{fold}.json")
    if params["sequence"]["vars"] != d["cfg"]["model_prep"]["sequence"]["vars"]:
        raise ValueError("Parameter variable order mismatch")
    if any(params[k]["n_rows"] != len(tr) for k in ["sequence", "tabular"]):
        raise ValueError("Fold preprocessing training row count mismatch")
    static = d["pp"].apply_tabular(d["static"], params["tabular"], STATIC)
    y = em.SepsisLabel.to_numpy()[tr]
    if len(np.unique(y)) != 2:
        raise ValueError("Both classes must be present in training")
    weight = float((y == 0).sum() / (y == 1).sum()) if args.pos_weight else 1.0
    cfg = d["cfg"]["model_prep"]["sequence"]
    model_cfg = {
        "input_size": len(cfg["vars"]) + 2 * len(cfg["mask_delta_vars"]),
        "hidden_size": args.hidden_size,
        "num_layers": args.layers,
        "dropout": args.dropout,
        "static_size": len(STATIC),
    }
    model = GruSepsisModel(**model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    directory = args.out / f"fold{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "latest.pt"
    identity = {
        "data": d["hashes"],
        "model": model_cfg,
        "fold": fold,
        "seed": seed,
        "batch_size": args.batch_size,
        "lr": args.learning_rate,
        "pos_weight": weight,
        "patience": args.patience,
        "code": {p.name: digest(p) for p in [Path(__file__), Path(__file__).with_name("gru.py")]},
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    start, best_ap, bad, history, best_state, best_epoch = 0, -1.0, 0, [], None, 0
    if checkpoint.exists():
        if not args.resume:
            raise ValueError(f"{checkpoint} exists; use --resume or a new --out")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Resume rejected: data, code or training settings changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best_ap, bad = saved["epoch"], saved["best_ap"], saved["bad"]
        history, best_state, best_epoch = saved["history"], saved["best_state"], saved["best_epoch"]
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    for epoch in range(start, args.epochs):
        if bad >= args.patience:
            break
        before = time.perf_counter()
        model.train()
        total_loss = 0.0
        for _, x, s, lengths, labels in batches(
            d, tr, params, static, args.batch_size, device, seed + epoch
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x, s, lengths), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
        p = predict(model, d, va, params, static, args.batch_size, device)
        ap = float(average_precision_score(em.SepsisLabel.iloc[va], p))
        if ap > best_ap:
            best_ap, bad, best_epoch = ap, 0, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        record = {
            "epoch": epoch + 1,
            "loss": total_loss / len(tr),
            "val_auprc": ap,
            "seconds": time.perf_counter() - before,
        }
        history.append(record)
        atomic_checkpoint(
            checkpoint,
            {
                "identity": identity,
                "epoch": epoch + 1,
                "best_ap": best_ap,
                "bad": bad,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "history": history,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            },
        )
        atomic_json(directory / "history.json", history)
        print(json.dumps({"fold": fold, **record}), flush=True)
    if best_state is None:
        raise RuntimeError("No completed training epoch")
    model.load_state_dict(best_state)
    p = predict(model, d, va, params, static, args.batch_size, device)
    atomic_checkpoint(
        directory / "best.pt",
        {
            "state_dict": best_state,
            "model_config": model_cfg,
            "epoch": best_epoch,
            "preprocessing": params,
            "config": d["cfg"],
            "static_columns": STATIC,
            "identity": identity,
            "probabilities": "uncalibrated",
        },
    )
    report = metrics(em.iloc[va], p)
    report.update({"fold": fold, "best_epoch": best_epoch, "test_evaluated": False})
    atomic_json(directory / "metrics.json", report)
    oof = em.iloc[va][["row_id", "Patient_ID", "ICULOS", "set", "fold", "SepsisLabel"]].copy()
    oof["p_gru"] = p
    oof.to_parquet(directory / "oof.parquet", index=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=[0])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos-weight", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.patience, args.batch_size, args.hidden_size, args.layers) < 1:
        parser.error("epochs, patience, batch size and model dimensions must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("GPU unavailable: select a Colab GPU runtime, or explicitly use --device cpu")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    d = load_data(args.root)
    reports = [run_fold(d, args, k, torch.device(args.device)) for k in dict.fromkeys(args.folds)]
    # Aggregate only folds explicitly requested in this invocation; never stale files.
    frames = [pd.read_parquet(args.out / f"fold{r['fold']}/oof.parquet") for r in reports]
    oof = pd.concat(frames).sort_values("row_id")
    if oof.row_id.duplicated().any():
        raise ValueError("Duplicate OOF rows")
    oof.to_parquet(args.out / "oof_predictions.parquet", index=False)
    atomic_json(
        args.out / "summary.json",
        {
            "folds": reports,
            "complete_5fold": len(reports) == 5,
            "pooled": metrics(oof, oof.p_gru.to_numpy()),
            "test_evaluated": False,
            "note": "Development scores; epochs selected using validation AUPRC. Uncalibrated.",
        },
    )


if __name__ == "__main__":
    main()
