"""SEWS LSTM: 5-fold 교차검증, 보정(Calibration), 경보 임계값, 결과 저장.

ICU 패혈증 조기 경보 시스템(SEWS)의 시계열 모델이다.
노트북 ICU-6_lstm과 같은 과정을 하나의 스크립트로 재현한다.
GPU(T4 등)가 있으면 자동으로 쓴다.

모델을 만든 방법
----------------
- 데이터: PhysioNet/CinC Challenge 2019 (setA + setB, 40,336명).
  환자 단위로 test 15%를 먼저 떼어 두고, dev 85%(34,285명)를 5-fold로 나눴다
  (set x 패혈증 여부로 층화, SEED 42). 이 스크립트는 dev 행만 쓴다.
  test는 전체 모델링이 끝난 뒤 한 번만 평가한다.
- 입력: 예측 시점마다 최근 24시간 창 (24 x 66채널) + 정적 입력 5개.
  66채널 = 값 24개(활력징후 6 + 검사값 18,
  창 안 forward fill 후 fold별 중앙값·IQR 스케일)
  + 측정 여부(mask) 21개 + 마지막 측정 이후 시간(delta) 21개.
  pH·PaCO2·PTT는 병원마다 측정 방식이 달라 mask·delta를 만들지 않는다.
  정적 입력은 Age, Gender, Unit(MICU, SICU, Unknown).
  ICULOS 같은 시간 정보는 쓰지 않는다.
- 창은 GPU에서 만든다.
  4단계 함수(sews_preprocess.seq_windows)와 같은 값인지 매번 확인한다.
- 모델: LSTM(은닉 64, 1층) -> 마지막 시점 은닉 상태 + 정적 입력 -> Linear(32) -> ReLU
  -> Dropout(0.2) -> Linear(1). 파라미터 36,065개.
- 학습: AdamW(lr 3e-4), batch 1024, gradient clipping 1.0.
  클래스 불균형은 BCE pos_weight = sqrt(음성/양성) (약 7.4배)로 처리한다.
  early stopping은 학습 fold 환자 10%를 따로 떼어 AUPRC로 판단한다
  (patience 3, 최대 20 epoch).
  평가 fold는 학습과 선택에 쓰지 않는다.
- 기본 설정 근거 (fold 0, early stopping용 내부 검증 AUPRC, 평가 fold는 보지 않음):
  lr 1e-3·가중치 54배 0.0725 (첫 epoch 뒤 과적합), lr 3e-4·54배 0.0709,
  lr 3e-4·54배·weight decay 1e-2 0.0709, lr 3e-4·제곱근 가중치 0.0814 (선택).
- 보정: Platt scaling과 Isotonic regression을 비교한다. fold k의 보정값은 나머지 fold의
  OOF 예측으로 학습한 보정기로 계산한다.
  OOF Brier가 낮은 쪽을 운영용으로 쓴다(결과: Platt).
- 임계값: PhysioNet 공식 Utility를 최대화하는 임계값.
  fold k의 임계값은 나머지 fold로 고른다.
- 최종 모델: dev 전체를 5-fold best epoch의 중앙값만큼 학습한다(final_dev 파라미터).

결과 (dev 5-fold OOF, 행 단위, CPU 실행)
---------------------------------------
AUROC 0.776 +- 0.015, AUPRC 0.071, ICULOS<=60 AUROC 0.762, 조기 구간 AUROC 0.760,
특이도 95%에서 민감도 0.272, 환자 단위 AUROC 0.807 / AUPRC 0.291,
Utility 0.282 (교차 적합),
Brier 0.0173 (Platt).
로지스틱 회귀(AUROC 0.769, AUPRC 0.064, Utility 0.273)보다 모든 지표가 높다.
GPU에서는 cuDNN 계산 순서 때문에 best epoch와 소수 셋째 자리가 조금 다를 수 있다.

실행
----
    python models/train_lstm.py --root /content/drive/MyDrive/capston
    python models/train_lstm.py --root <데이터 폴더> --smoke     # 5분 시험 실행

--root 아래에 다음이 있어야 한다 (전처리 단계 산출물).
    data/seq_values.npy, data/seq_row_start.npy, data/eval_meta.parquet,
    data/features.parquet (정적 입력 5개만 읽음)
    artifacts/preprocess_config.json (v4), artifacts/feature_registry.json
    artifacts/preprocess_params/fold0~4.json, final_dev.json
공용 모듈은 저장소의 src/sews_preprocess.py, src/sews_eval.py를 쓴다
(--src-dir로 바꿀 수 있음).

출력 (--out-dir, 기본 <root>/results/lstm, --smoke면 results/lstm_smoke)
    summary.json, cv_metrics_by_fold.csv, training_history.csv, calibration.csv,
    threshold_table.csv, comparison_with_logreg.csv, oof_predictions.parquet,
    figures/lstm_curves.png, folds/ (fold별 예측·모델, 이어 하기용),
    model/lstm_final.pt, model/lstm_final_meta.json, settings.json
중간에 멈추면 같은 명령으로 다시 실행한다. 끝난 fold는 folds/에서 읽어 건너뛴다.

지키는 원칙
-----------
환자 ID 기준 분리,
스케일러·보정기·임계값·early stopping은 학습(또는 지정 검증) 데이터에서만,
예측 시각 이후 데이터 미사용(최근 24행만), 공식 SepsisLabel 재이동 금지.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

LOG = logging.getLogger("train_lstm")

# ---- 학습 설정 (노트북 ICU-6 [6-0]과 같은 값) ----
HIDDEN, LAYERS, DROPOUT = 64, 1, 0.2
BATCH, LR, MAX_EPOCHS, PATIENCE = 1024, 3e-4, 20, 3
WEIGHT_DECAY = 0.0
POS_WEIGHT = "sqrt"  # 'ratio': 음성/양성 비율(약 54배), 'sqrt': 그 제곱근(약 7.4배)
INNER_VAL = 0.10
SEED = 42
N_FOLDS = 5
SMOKE_ROWS, SMOKE_EPOCHS = 60_000, 2
BASELINE_NAME = "baseline: constant positive rate"
FOLD_METRICS = [
    "AUROC",
    "AUPRC",
    "AUROC_ICULOS<=60",
    "AUROC_early(rel -6~-1 vs neg)",
    "Sens@95%Spec",
    "patient_AUROC(max)",
    "patient_AUPRC(max)",
    "best_epoch",
]


# --------------------------------------------------------------------------- 입력
def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    ap = argparse.ArgumentParser(description="SEWS LSTM (5-fold CV).")
    ap.add_argument("--root", required=True, help="데이터 루트 (data/, artifacts/)")
    ap.add_argument("--src-dir", default=None, help="sews_*.py 위치 (기본: 저장소 src/)")
    ap.add_argument("--out-dir", default=None, help="결과 폴더 (기본: <root>/results/lstm)")
    ap.add_argument("--local-dir", default=None, help="입력 배열 복사 위치 (기본: 임시 폴더)")
    ap.add_argument("--smoke", action="store_true", help="fold마다 6만 행, 2 epoch 시험 실행")
    return ap.parse_args()


def load_json(path: str) -> dict:
    """Read a UTF-8 JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def settings(smoke: bool) -> dict:
    """Training settings recorded with the results (checked when resuming)."""
    return {
        "SMOKE": smoke,
        "HIDDEN": HIDDEN,
        "LAYERS": LAYERS,
        "DROPOUT": DROPOUT,
        "BATCH": BATCH,
        "LR": LR,
        "MAX_EPOCHS": MAX_EPOCHS,
        "PATIENCE": PATIENCE,
        "INNER_VAL": INNER_VAL,
        "SEED": SEED,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "POS_WEIGHT": POS_WEIGHT,
    }


def check_settings(out_dir: str, smoke: bool) -> None:
    """Refuse to resume from fold results made with different settings."""
    cur = settings(smoke)
    path = f"{out_dir}/settings.json"
    if os.path.exists(path):
        saved = load_json(path)
        if saved != cur:
            raise RuntimeError(
                f"{out_dir}에 다른 설정의 결과가 있다: {saved}. "
                "설정을 되돌리거나 그 폴더를 지우고 다시 실행하세요."
            )
    with open(path, "w") as f:
        json.dump(cur, f, indent=1)


def load_inputs(pp: ModuleType, root: str, src_dir: str, local_dir: str) -> dict:
    """Load config, fold parameters, eval_meta, static inputs and sequence arrays."""
    art, data = f"{root}/artifacts", f"{root}/data"
    cfg = load_json(f"{art}/preprocess_config.json")
    registry = load_json(f"{art}/feature_registry.json")
    if cfg["version"] != 4:
        raise RuntimeError(f"config v{cfg['version']}: 4단계 v4가 필요하다")
    from sews_check import module_matches

    how = module_matches(
        f"{src_dir}/sews_preprocess.py",
        registry["preprocess"]["module_sha256"],
        ref_dir=f"{root}/src",
    )
    LOG.info("[OK] sews_preprocess.py: 4단계 때와 %s", how)

    em_all = pd.read_parquet(f"{data}/eval_meta.parquet")
    static_cols = cfg["model_prep"]["sequence"]["static"]
    f_static = pd.read_parquet(f"{data}/features.parquet", columns=["row_id", *static_cols])
    if not (f_static["row_id"].to_numpy() == em_all["row_id"].to_numpy()).all():
        raise RuntimeError("features와 eval_meta의 row_id 순서가 다르다")
    params = {}
    for k in [*range(N_FOLDS), "final"]:
        fn = f"fold{k}.json" if k != "final" else "final_dev.json"
        params[k] = load_json(f"{art}/preprocess_params/{fn}")

    os.makedirs(local_dir, exist_ok=True)
    for fn in ("seq_values.npy", "seq_row_start.npy"):
        src, dst = f"{data}/{fn}", f"{local_dir}/{fn}"
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
            shutil.copy(src, dst)  # Drive보다 빠른 로컬 디스크
    seq_values = np.load(f"{local_dir}/seq_values.npy")
    row_start = np.load(f"{local_dir}/seq_row_start.npy")
    spec = pp.seq_spec(cfg)
    if seq_values.shape != (len(em_all), len(spec["vars"])):
        raise RuntimeError("seq_values.npy 크기가 eval_meta와 맞지 않는다")
    dev = (em_all["split"] == "dev").to_numpy()
    return {
        "cfg": cfg,
        "registry": registry,
        "params": params,
        "em_all": em_all,
        "f_static": f_static,
        "static_cols": static_cols,
        "seq_values": seq_values,
        "row_start": row_start,
        "spec": spec,
        "dev_rows": np.flatnonzero(dev),
        "y_all": em_all["SepsisLabel"].to_numpy().astype(np.float32),
    }


# --------------------------------------------------------------------------- 창
class WindowBuilder:
    """Build 24h LSTM input windows on the device (same as seq_windows)."""

    def __init__(self, d: dict, device: torch.device) -> None:
        self.device = device
        self.vals = torch.from_numpy(d["seq_values"]).to(device)
        self.start = torch.from_numpy(d["row_start"]).to(device)
        self.steps = torch.arange(24, device=device)
        self.mask_idx = torch.tensor(d["spec"]["mask_idx"], device=device)

    def __call__(self, rows_t: torch.Tensor, sp: dict) -> torch.Tensor:
        """Return a (B, 24, V + 2M) tensor for row ids rows_t."""
        dev = self.device
        c = torch.tensor(sp["center"], dtype=torch.float32, device=dev)
        s = torch.tensor(sp["scale"], dtype=torch.float32, device=dev)
        idx = rows_t[:, None] - (23 - self.steps)[None, :]
        valid = idx >= self.start[rows_t][:, None]
        raw = self.vals[idx.clamp(min=0)]
        raw = torch.where(valid[..., None], raw, torch.full_like(raw, float("nan")))
        obs = ~torch.isnan(raw)
        neg = torch.full_like(raw, -1, dtype=torch.long)
        li = torch.where(obs, self.steps[None, :, None], neg).cummax(dim=1).values
        ff = torch.gather(raw, 1, li.clamp(min=0))
        ff = torch.where(li >= 0, ff, torch.full_like(ff, float("nan")))
        delta = torch.where(li >= 0, self.steps[None, :, None] - li, torch.full_like(li, 24))
        x = torch.nan_to_num(torch.clamp((ff - c) / s, -sp["clip"], sp["clip"]), nan=0.0)
        m = self.mask_idx
        return torch.cat([x, obs[..., m].float(), delta.float()[..., m] / 24], dim=2)


def check_windows(pp: ModuleType, wb: WindowBuilder, d: dict) -> None:
    """Device windows must equal pp.seq_windows on sampled dev rows (incl. <24h)."""
    rng = np.random.default_rng(0)
    dev_rows = d["dev_rows"]
    pos = d["em_all"]["row_id"].to_numpy() - d["row_start"]
    pick = np.sort(
        np.unique(
            np.concatenate(
                [
                    rng.choice(dev_rows, 2000, replace=False),
                    rng.choice(dev_rows[pos[dev_rows] < 23], 500, replace=False),
                ]
            )
        )
    )
    sp = d["params"][0]["sequence"]
    ref = pp.seq_windows(d["seq_values"], d["row_start"], pick, sp, d["cfg"])
    ref_cat = np.concatenate([ref["x"], ref["mask"], ref["delta"]], axis=2)
    got = wb(torch.from_numpy(pick).to(wb.device), sp).cpu().numpy()
    if got.shape != ref_cat.shape or not np.allclose(got, ref_cat, rtol=1e-6, atol=1e-6):
        raise RuntimeError("GPU 창이 4단계 seq_windows와 다르다")
    LOG.info(
        "[OK] 장치 창 = 4단계 seq_windows (%d개 창, 최대 차이 %.1e)",
        len(pick),
        float(np.abs(got - ref_cat).max()),
    )


# --------------------------------------------------------------------------- 모델·학습
class SepsisLSTM(nn.Module):
    """LSTM over the 24h window, last hidden state concatenated with static inputs."""

    def __init__(self, n_in: int, n_static: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_in,
            HIDDEN,
            num_layers=LAYERS,
            batch_first=True,
            dropout=DROPOUT if LAYERS > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN + n_static, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, st: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B,)."""
        out, _ = self.lstm(x)
        return self.head(torch.cat([out[:, -1], st], dim=1)).squeeze(1)


class Trainer:
    """Training, prediction and inner-validation split for one dataset."""

    def __init__(self, d: dict, wb: WindowBuilder) -> None:
        self.d, self.wb, self.dev = d, wb, wb.device
        self.y_t = torch.from_numpy(d["y_all"]).to(self.dev)
        self.n_in = len(d["spec"]["vars"]) + 2 * len(d["spec"]["mask_vars"])

    def static_tensor(self, pp: ModuleType, tab_params: dict) -> torch.Tensor:
        """Scale the static inputs once per fold and keep them on the device."""
        cols = self.d["static_cols"]
        arr = pp.apply_tabular(self.d["f_static"], tab_params, cols)
        return torch.from_numpy(arr).to(self.dev)

    @torch.no_grad()
    def predict(
        self,
        model: nn.Module,
        rows: np.ndarray,
        sp: dict,
        st_t: torch.Tensor,
        bs: int = 4096,
    ) -> np.ndarray:
        """Sigmoid probabilities for the given row ids."""
        model.eval()
        out = []
        for i in range(0, len(rows), bs):
            r = torch.from_numpy(rows[i : i + bs]).to(self.dev)
            out.append(torch.sigmoid(model(self.wb(r, sp), st_t[r])).float().cpu())
        return torch.cat(out).numpy()

    def split_inner(self, train_rows: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Hold out INNER_VAL of the training patients for early stopping."""
        pids = self.d["em_all"]["Patient_ID"].to_numpy()
        up = np.unique(pids[train_rows])
        hold = np.random.default_rng(seed).choice(
            up, int(round(INNER_VAL * len(up))), replace=False
        )
        m = np.isin(pids[train_rows], hold)
        return train_rows[~m], train_rows[m]

    def train(
        self,
        train_rows: np.ndarray,
        sp: dict,
        st_t: torch.Tensor,
        seed: int,
        max_epochs: int,
        inner_rows: np.ndarray | None = None,
        prefix: str = "",
    ) -> tuple[nn.Module, pd.DataFrame, int]:
        """Train with early stopping on inner_rows (AUPRC) or for max_epochs."""
        y_all = self.d["y_all"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SepsisLSTM(self.n_in, len(self.d["static_cols"])).to(self.dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        n_pos = float(y_all[train_rows].sum())
        ratio = (len(train_rows) - n_pos) / n_pos
        pw = ratio if POS_WEIGHT == "ratio" else np.sqrt(ratio)
        lossf = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pw, dtype=torch.float32, device=self.dev)
        )
        rng = np.random.default_rng(seed)
        best, best_state, best_ep, wait, hist = -1.0, None, 0, 0, []
        for ep in range(1, max_epochs + 1):
            model.train()
            t0 = time.time()
            order = rng.permutation(train_rows)
            tot = 0.0
            for i in range(0, len(order), BATCH):
                r = torch.from_numpy(np.sort(order[i : i + BATCH])).to(self.dev)
                opt.zero_grad(set_to_none=True)
                loss = lossf(model(self.wb(r, sp), st_t[r]), self.y_t[r])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(r)
            row = {
                "epoch": ep,
                "train_loss": tot / len(order),
                "sec": round(time.time() - t0, 1),
            }
            msg = f"{prefix} epoch {ep:2d} | loss {row['train_loss']:.4f}"
            if inner_rows is not None:
                p = self.predict(model, inner_rows, sp, st_t)
                row["inner_AUPRC"] = average_precision_score(y_all[inner_rows], p)
                row["inner_AUROC"] = roc_auc_score(y_all[inner_rows], p)
                msg += f" | inner AUPRC {row['inner_AUPRC']:.4f} AUROC {row['inner_AUROC']:.4f}"
                if row["inner_AUPRC"] > best:
                    best, best_ep, wait = row["inner_AUPRC"], ep, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    wait += 1
            hist.append(row)
            LOG.info("%s | %.0fs", msg, row["sec"])
            if inner_rows is not None and wait >= PATIENCE:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, pd.DataFrame(hist), best_ep


# --------------------------------------------------------------------------- 교차검증
def run_cv(
    pp: ModuleType, ev: ModuleType, d: dict, tr: Trainer, ckpt_dir: str, smoke: bool
) -> dict:
    """5-fold CV with per-fold checkpoints; returns OOF predictions and tables."""
    em = d["em_all"].loc[d["em_all"]["split"] == "dev"].reset_index(drop=True)
    folds = em["fold"].to_numpy()
    dev_rows = d["dev_rows"]
    oof = np.full(len(em), np.nan)
    fold_rows, histories, best_epochs = [], [], {}
    t_all = time.time()
    for k in range(N_FOLDS):
        path = f"{ckpt_dir}/fold{k}.npz"
        va_local = np.flatnonzero(folds == k)
        va_rows = dev_rows[va_local]
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            oof[va_local] = z["p"]
            best_epochs[k] = int(z["best_epoch"])
            histories.append(pd.DataFrame(json.loads(str(z["history"]))).assign(fold=k))
            LOG.info("[fold %d] 저장된 결과 사용 (best epoch %d)", k, best_epochs[k])
        else:
            tr_rows, inner = tr.split_inner(dev_rows[folds != k], SEED + k)
            if smoke:
                tr_rows = np.random.default_rng(k).choice(tr_rows, SMOKE_ROWS, replace=False)
            st_t = tr.static_tensor(pp, d["params"][k]["tabular"])
            sp = d["params"][k]["sequence"]
            t0 = time.time()
            model, hist, best_ep = tr.train(
                tr_rows,
                sp,
                st_t,
                SEED + k,
                SMOKE_EPOCHS if smoke else MAX_EPOCHS,
                inner_rows=inner,
                prefix=f"[fold {k}]",
            )
            p = tr.predict(model, va_rows, sp, st_t)
            oof[va_local] = p
            best_epochs[k] = best_ep
            np.savez(path, p=p, best_epoch=best_ep, history=hist.to_json(orient="records"))
            torch.save(model.state_dict(), f"{ckpt_dir}/fold{k}_model.pt")
            histories.append(hist.assign(fold=k))
            LOG.info(
                "[fold %d] 완료 %.0f초, 학습 %s행 / 내부 검증 %s행 / best epoch %d",
                k,
                time.time() - t0,
                f"{len(tr_rows):,}",
                f"{len(inner):,}",
                best_ep,
            )
        r = ev.evaluate(em[folds == k], oof[va_local])
        r.update(fold=k, best_epoch=best_epochs[k])
        fold_rows.append(r)
    if np.isnan(oof).any():
        raise RuntimeError("OOF 예측에 빈 값이 있다")
    LOG.info("5-fold 완료: %.1f분", (time.time() - t_all) / 60)
    ftbl = pd.DataFrame(fold_rows).set_index("fold")[FOLD_METRICS]
    ftbl.loc["mean"] = ftbl.mean()
    ftbl.loc["std"] = ftbl.iloc[:N_FOLDS].std()
    return {
        "em": em,
        "oof": oof,
        "fold_tbl": ftbl,
        "best_epochs": best_epochs,
        "history": pd.concat(histories, ignore_index=True),
    }


# ------------------------------------------------------------------ 보정·임계값
def calibrate(ev: ModuleType, em: pd.DataFrame, p_raw: np.ndarray) -> dict:
    """Cross-fitted Platt and isotonic calibration with Brier, ECE and Brier skill."""
    y = em["SepsisLabel"].to_numpy()
    p_platt = ev.crossfit_calibrate(em, p_raw, "platt")
    p_iso = ev.crossfit_calibrate(em, p_raw, "isotonic")
    rows = []
    for name, p in [
        ("raw (pos_weight)", p_raw),
        ("Platt", p_platt),
        ("Isotonic", p_iso),
        (BASELINE_NAME, np.full(len(y), y.mean())),
    ]:
        rows.append(
            {
                "model": name,
                "Brier": np.mean((p - y) ** 2),
                "ECE(10 bins)": ev.ece(y, p),
                "AUROC": roc_auc_score(y, p) if name != BASELINE_NAME else 0.5,
                "mean_pred": p.mean(),
                "max_pred": p.max(),
            }
        )
    cal = pd.DataFrame(rows).set_index("model")
    cal["Brier_skill"] = 1 - cal["Brier"] / cal.loc[BASELINE_NAME, "Brier"]
    use = "Platt" if cal.loc["Platt", "Brier"] <= cal.loc["Isotonic", "Brier"] else "Isotonic"
    return {
        "table": cal,
        "p_platt": p_platt,
        "p_iso": p_iso,
        "selected": use,
        "p_cal": p_platt if use == "Platt" else p_iso,
    }


def thresholds(ev: ModuleType, em: pd.DataFrame, p_cal: np.ndarray) -> dict:
    """Utility-optimal threshold (cross-fitted and final) plus reference thresholds."""
    y = em["SepsisLabel"].to_numpy()
    pred_cf, thr_by_fold = ev.crossfit_threshold(em, p_cal)
    u_cf = ev.normalized_utility(em, pred_cf)
    thr_u, u_all = ev.best_utility_threshold(em, p_cal)
    fpr, tpr, thr = ev.roc_curve(y, p_cal)
    ok = fpr <= 0.05
    sens_thr = float(thr[ok][np.argmax(tpr[ok])])
    idx = ev._time_index(em)
    tbl = pd.DataFrame(
        [
            dict(name="utility_optimal", **ev.threshold_report(em, p_cal, thr_u, idx)),
            dict(name="specificity_95", **ev.threshold_report(em, p_cal, sens_thr, idx)),
            dict(name="spec_warning_0.5", **ev.threshold_report(em, p_cal, 0.5, idx)),
            dict(name="spec_alert_0.75", **ev.threshold_report(em, p_cal, 0.75, idx)),
        ]
    ).set_index("name")
    return {
        "table": tbl,
        "u_cf": u_cf,
        "thr_by_fold": thr_by_fold,
        "thr_u": thr_u,
        "u_all": u_all,
    }


def plot_curves(ev: ModuleType, cv: dict, cal: dict, path: str) -> None:
    """Learning curves, ROC (all rows and ICULOS<=60) and reliability diagram."""
    em, history, p_raw = cv["em"], cv["history"], cv["oof"]
    y = em["SepsisLabel"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for k, g in history.groupby("fold"):
        axes[0].plot(g["epoch"], g["inner_AUPRC"], marker="o", ms=3, label=f"fold {k}")
    axes[0].set(
        xlabel="Epoch",
        ylabel="Inner-validation AUPRC",
        title="Learning curves (early stopping set)",
    )
    le60 = em["iculos_le60"].to_numpy() == 1
    for mask, name, col in [
        (np.ones(len(y), bool), "all rows", "#1f5fae"),
        (le60, "ICULOS<=60", "#e76f51"),
    ]:
        fpr, tpr, _ = ev.roc_curve(y[mask], p_raw[mask])
        auc = roc_auc_score(y[mask], p_raw[mask])
        axes[1].plot(fpr, tpr, color=col, lw=1.8, label=f"{name}, pooled (AUROC {auc:.3f})")
    axes[1].plot([0, 1], [0, 1], color="gray", ls="--", lw=0.8)
    axes[1].set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC (5-fold OOF, LSTM)",
    )
    lim = 0.0
    for p, name, col in [
        (cal["p_platt"], "Platt", "#1f5fae"),
        (cal["p_iso"], "Isotonic", "#2a9d8f"),
    ]:
        q = pd.qcut(pd.Series(p).rank(method="first"), 20, labels=False)
        g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").mean()
        axes[2].plot(g["p"], g["y"], marker="o", ms=3, color=col, label=name)
        lim = max(lim, float(g[["p", "y"]].max().max()) * 1.1)
    axes[2].plot([0, lim], [0, lim], color="black", ls="--", lw=0.8)
    axes[2].set(
        xlim=(0, lim),
        ylim=(0, lim),
        xlabel="Mean predicted probability",
        ylabel="Observed positive rate",
        title="Reliability (20 quantile bins)",
    )
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- 저장
def fit_final(
    pp: ModuleType, d: dict, tr: Trainer, best_epochs: dict, model_dir: str, smoke: bool
) -> int:
    """Train the final model on all dev rows for the median best epoch."""
    n_ep = max(1, int(np.median(list(best_epochs.values()))))
    path = f"{model_dir}/lstm_final.pt"
    if os.path.exists(path):
        LOG.info("[최종 모델] 저장된 모델 사용")
        return n_ep
    st_t = tr.static_tensor(pp, d["params"]["final"]["tabular"])
    rows = d["dev_rows"]
    if smoke:
        rows = np.random.default_rng(9).choice(rows, SMOKE_ROWS, replace=False)
    t0 = time.time()
    model, _, _ = tr.train(
        rows, d["params"]["final"]["sequence"], st_t, SEED, n_ep, prefix="[final]"
    )
    torch.save(model.state_dict(), path)
    LOG.info("[최종 모델] %d epoch, %.0f초", n_ep, time.time() - t0)
    return n_ep


def export(
    ev: ModuleType,
    d: dict,
    cv: dict,
    cal: dict,
    th: dict,
    n_final: int,
    root: str,
    out_dir: str,
    device: torch.device,
    smoke: bool,
) -> None:
    """Write summary, OOF predictions, model metadata and the comparison table."""
    em, p_raw, ftbl = cv["em"], cv["oof"], cv["fold_tbl"]
    y = em["SepsisLabel"].to_numpy()
    ctab = cal["table"]
    meta = {
        "model": "LSTM",
        "hidden": HIDDEN,
        "layers": LAYERS,
        "dropout": DROPOUT,
        "batch": BATCH,
        "lr": LR,
        "epochs": n_final,
        "seed": SEED,
        "input": (
            f"{len(d['spec']['vars']) + 2 * len(d['spec']['mask_vars'])} channels"
            f" x 24h (x, mask, delta) + static {d['static_cols']}"
        ),
        "loss": f"BCEWithLogits, pos_weight = {POS_WEIGHT}(neg/pos of training rows)",
        "preprocess_params": "artifacts/preprocess_params/final_dev.json",
        "calibration": {
            "selected": cal["selected"],
            "platt_a_b": ev.fit_platt(p_raw, y),
            "formula": "p_cal = 1 / (1 + exp(-(a * logit(p_raw) + b)))",
        },
        "thresholds": {
            "utility_optimal": th["thr_u"],
            "spec_warning": 0.5,
            "spec_alert": 0.75,
        },
        "trained_on": "dev 전체 (test 미사용)",
        "torch_version": torch.__version__,
        "device": str(device),
        "module_sha256": {"sews_preprocess": d["registry"]["preprocess"]["module_sha256"]},
    }
    with open(f"{out_dir}/model/lstm_final_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    pd.DataFrame(
        {
            "row_id": em["row_id"],
            "Patient_ID": em["Patient_ID"],
            "fold": em["fold"],
            "SepsisLabel": y,
            "p_raw": p_raw,
            "p_platt": cal["p_platt"],
            "p_isotonic": cal["p_iso"],
        }
    ).to_parquet(f"{out_dir}/oof_predictions.parquet", index=False)
    mean, std = ftbl.loc["mean"], ftbl.loc["std"]
    summary = {
        "model": (
            f"LSTM (hidden {HIDDEN}, {LAYERS} layer, pos_weight {POS_WEIGHT}, "
            "early stopping on inner 10% patients)"
        ),
        "smoke_test": smoke,
        "data": {
            "split": "dev 5-fold CV (환자 단위, SEED 42), test 미사용",
            "rows": int(len(em)),
            "positive_row_rate": float(y.mean()),
        },
        "cv_mean_std": {k: [float(mean[k]), float(std[k])] for k in ftbl.columns},
        "calibration": ctab.drop(columns=["mean_pred", "max_pred"])
        .round(5)
        .to_dict(orient="index"),
        "selected_calibration": cal["selected"],
        "utility": {
            "crossfit": th["u_cf"],
            "final_threshold": th["thr_u"],
            "final_threshold_oof_utility": th["u_all"],
        },
        "thresholds": th["table"].round(5).to_dict(orient="index"),
        "training_minutes_total": float(cv["history"]["sec"].sum() / 60),
        "device": str(device),
    }
    with open(f"{out_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    keys = [
        "AUROC",
        "AUPRC",
        "AUROC_ICULOS<=60",
        "AUROC_early(rel -6~-1 vs neg)",
        "Sens@95%Spec",
    ]
    rows = [
        {
            "model": "LSTM",
            **{k: mean[k] for k in keys},
            "Utility": th["u_cf"],
            "Brier(cal)": ctab.loc[cal["selected"], "Brier"],
        }
    ]
    lr_sum = f"{root}/results/logreg/summary.json"
    if os.path.exists(lr_sum):
        s_lr = load_json(lr_sum)
        rows.append(
            {
                "model": "Logistic regression",
                **{k: s_lr["cv_mean_std"][k][0] for k in keys},
                "Utility": s_lr["utility"]["crossfit"],
                "Brier(cal)": s_lr["calibration"][s_lr["selected_calibration"]]["Brier"],
            }
        )
    cmp_df = pd.DataFrame(rows).set_index("model")
    cmp_df.to_csv(f"{out_dir}/comparison_with_logreg.csv")
    LOG.info("[모델 비교 (dev 5-fold 평균, 행 단위)]\n%s", cmp_df.round(4).to_string())


def main() -> None:
    """Run cross-validation, calibration, thresholding, final model and export."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    root = args.root.rstrip("/")
    src_dir = args.src_dir or str(Path(__file__).resolve().parents[1] / "src")
    out_dir = args.out_dir or f"{root}/results/lstm" + ("_smoke" if args.smoke else "")
    local_dir = args.local_dir or f"{tempfile.gettempdir()}/sews_local"
    fig_dir, ckpt_dir, model_dir = (
        f"{out_dir}/figures",
        f"{out_dir}/folds",
        f"{out_dir}/model",
    )
    for p in (fig_dir, ckpt_dir, model_dir):
        os.makedirs(p, exist_ok=True)
    check_settings(out_dir, args.smoke)
    sys.path.insert(0, src_dir)
    import sews_eval as ev
    import sews_preprocess as pp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info(
        "장치: %s %s",
        device,
        torch.cuda.get_device_name(0) if device.type == "cuda" else "(GPU 없음)",
    )
    d = load_inputs(pp, root, src_dir, local_dir)
    LOG.info(
        "dev %s행, 입력 %d채널 x 24시간 + 정적 %d, SMOKE=%s",
        f"{len(d['dev_rows']):,}",
        len(d["spec"]["vars"]) + 2 * len(d["spec"]["mask_vars"]),
        len(d["static_cols"]),
        args.smoke,
    )
    wb = WindowBuilder(d, device)
    check_windows(pp, wb, d)
    tr = Trainer(d, wb)

    cv = run_cv(pp, ev, d, tr, ckpt_dir, args.smoke)
    cv["fold_tbl"].to_csv(f"{out_dir}/cv_metrics_by_fold.csv")
    cv["history"].to_csv(f"{out_dir}/training_history.csv", index=False)
    LOG.info("[fold별 성능]\n%s", cv["fold_tbl"].round(4).to_string())

    cal = calibrate(ev, cv["em"], cv["oof"])
    cal["table"].to_csv(f"{out_dir}/calibration.csv")
    th = thresholds(ev, cv["em"], cal["p_cal"])
    th["table"].to_csv(f"{out_dir}/threshold_table.csv")
    LOG.info(
        "[보정] 운영용 %s | Utility 교차 적합 %.3f | 최종 임계값 %.4f",
        cal["selected"],
        th["u_cf"],
        th["thr_u"],
    )
    plot_curves(ev, cv, cal, f"{fig_dir}/lstm_curves.png")

    n_final = fit_final(pp, d, tr, cv["best_epochs"], model_dir, args.smoke)
    export(ev, d, cv, cal, th, n_final, root, out_dir, device, args.smoke)
    mean, std = cv["fold_tbl"].loc["mean"], cv["fold_tbl"].loc["std"]
    LOG.info(
        "[요약] AUROC %.3f +- %.3f | AUPRC %.3f | ICULOS<=60 %.3f | Utility %.3f",
        mean["AUROC"],
        std["AUROC"],
        mean["AUPRC"],
        mean["AUROC_ICULOS<=60"],
        th["u_cf"],
    )


if __name__ == "__main__":
    main()
