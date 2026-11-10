"""SEWS 로지스틱 회귀: 5-fold 교차검증, 보정(Calibration), 경보 임계값, 결과 저장.

ICU 패혈증 조기 경보 시스템(SEWS)의 선형 기준선 모델이다.
노트북 ICU-5_logreg와 같은 과정을 하나의 스크립트로 재현한다.

모델을 만든 방법
----------------
- 데이터: PhysioNet/CinC Challenge 2019 (setA + setB, 40,336명).
  환자 단위로 test 15%를 먼저 떼어 두고, dev 85%(34,285명)를 5-fold로 나눴다
  (set x 패혈증 여부로 층화, SEED 42).
  이 스크립트는 dev 행만 쓴다. test는 전체 모델링이 끝난 뒤 한 번만 평가한다.
- 입력 특징: 최근 24시간 창에서 만든 125개
  (활력징후 통계 66, 검사값 36, 파생 4, 측정 패턴 6, SOFA 5, qSOFA 3, 정적 5).
  ICULOS 같은 시간 특징은 쓰지 않는다(시간 지름길).
- 전처리: fold k는 fold k의 학습 행으로만 만든 파라미터(중앙값, IQR)로 스케일링한다.
  (x - 중앙값) / IQR, +-10에서 자르고, 결측은 0(= 학습 행 중앙값)으로 대치한다.
- 모델: scikit-learn LogisticRegression (L2, lbfgs, max_iter 1000).
  클래스 불균형은 class_weight='balanced'로 처리한다.
  C는 {0.001, 0.01, 0.1, 1.0} 중 5-fold 평균 AUPRC가 가장 높은 값을 고른다
  (결과: C = 0.01).
  비교용으로 class_weight=None도 같이 학습한다.
- 보정: Platt scaling과 Isotonic regression을 비교한다. fold k의 보정값은 나머지 fold의
  OOF 예측으로 학습한 보정기로 계산한다(평가 fold의 정답은 보정기 학습에 쓰지 않는다).
  OOF Brier가 낮은 쪽을 운영용으로 쓴다(결과: Isotonic).
- 임계값: PhysioNet 공식 Utility를 최대화하는 임계값.
  fold k의 임계값은 나머지 fold로 고른다.
  명세의 Warning 0.5 / Alert 0.75와 특이도 95% 지점도 같이 보고한다.
- 최종 모델: 고른 설정으로 dev 전체를 학습한다(final_dev 파라미터).
  보정기는 dev OOF 예측으로 학습한다.

결과 (dev 5-fold OOF, 행 단위)
------------------------------
AUROC 0.769 +- 0.008, AUPRC 0.064 +- 0.003, ICULOS<=60 AUROC 0.757,
조기 구간 AUROC 0.740, 특이도 95%에서 민감도 0.245, 환자 단위 AUROC 0.810,
Utility 0.273 (교차 적합), Brier 0.190(보정 전) -> 0.0172(Isotonic)
(항상 1.8%로 예측하면 0.0177).
Utility 최적 임계값 0.0286에서 경보 행 19.5%, 명세 0.5 / 0.75에서는 경보 0건.

실행
----
    python models/train_logreg.py --root /content/drive/MyDrive/capston
    python models/train_logreg.py --root <데이터 폴더> --no-mlflow

--root 아래에 다음이 있어야 한다 (전처리 단계 산출물).
    data/features.parquet, data/eval_meta.parquet
    artifacts/preprocess_config.json (v4), artifacts/feature_registry.json
    artifacts/preprocess_params/fold0~4.json, final_dev.json
공용 모듈은 저장소의 src/sews_preprocess.py, src/sews_eval.py를 쓴다
(--src-dir로 바꿀 수 있음).

출력 (--out-dir, 기본 <root>/results/logreg)
    summary.json, cv_grid.csv, cv_metrics_by_fold.csv, calibration.csv,
    threshold_table.csv, coefficients.csv, oof_predictions.parquet,
    figures/*.png, model/logreg_final.joblib, model/logreg_final_meta.json, mlflow/

지키는 원칙
-----------
환자 ID 기준 분리, 스케일러·보정기·임계값은 학습(또는 지정 검증) 데이터에서만 학습,
예측 시각 이후 데이터 미사용, 공식 SepsisLabel 재이동 금지.

Utility 정의는 PhysioNet/CinC Challenge 2019 공식 평가 코드
(https://github.com/physionetchallenges/evaluation-2019, BSD 2-Clause)를 따른다
(src/sews_eval.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

LOG = logging.getLogger("train_logreg")
SEED = 42
C_GRID = [0.001, 0.01, 0.1, 1.0]
CW_GRID = ["balanced", None]
N_FOLDS = 5
BASELINE_NAME = "baseline: constant positive rate"
SHOW_METRICS = ["AUROC", "AUPRC", "AUROC_ICULOS<=60", "AUROC_early(rel -6~-1 vs neg)"]
FOLD_METRICS = SHOW_METRICS + [
    "Sens@95%Spec",
    "patient_AUROC(max)",
    "patient_AUPRC(max)",
]


# --------------------------------------------------------------------------- 입력
def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    ap = argparse.ArgumentParser(description="SEWS logistic regression (5-fold CV).")
    ap.add_argument("--root", required=True, help="데이터 루트 (data/, artifacts/)")
    ap.add_argument(
        "--src-dir", default=None, help="sews_*.py 위치 (기본: 저장소 src/)"
    )
    ap.add_argument(
        "--out-dir", default=None, help="결과 폴더 (기본: <root>/results/logreg)"
    )
    ap.add_argument("--no-mlflow", action="store_true", help="MLflow 기록 생략")
    return ap.parse_args()


def sha256(path: str) -> str:
    """Return the SHA-256 hex digest of a file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_json(path: str) -> dict:
    """Read a UTF-8 JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_inputs(root: str, src_dir: str) -> dict:
    """Load dev rows, features, config and fold parameters; verify module versions."""
    art, data = f"{root}/artifacts", f"{root}/data"
    cfg = load_json(f"{art}/preprocess_config.json")
    registry = load_json(f"{art}/feature_registry.json")
    if cfg["version"] != 4:
        raise RuntimeError(f"config v{cfg['version']}: 4단계 v4가 필요하다")
    expected = {
        "sews_features.py": registry["module_sha256"],
        "sews_preprocess.py": registry["preprocess"]["module_sha256"],
    }
    for mod, digest in expected.items():
        if sha256(f"{src_dir}/{mod}") != digest:
            raise RuntimeError(f"{mod}가 전처리 단계 때와 다르다 (sha256 불일치)")

    model_cols = [r["name"] for r in registry["features"] if not r["optional"]]
    f_all = pd.read_parquet(f"{data}/features.parquet")
    em_all = pd.read_parquet(f"{data}/eval_meta.parquet")
    if not (f_all["row_id"].to_numpy() == em_all["row_id"].to_numpy()).all():
        raise RuntimeError("features와 eval_meta의 row_id 순서가 다르다")
    dev = (em_all["split"] == "dev").to_numpy()
    feats = f_all.loc[dev, model_cols].reset_index(drop=True)
    em = em_all.loc[dev].reset_index(drop=True)
    if set(np.unique(em["fold"])) != set(range(N_FOLDS)):
        raise RuntimeError("dev fold가 0~4가 아니다")

    params = {}
    for k in [*range(N_FOLDS), "final"]:
        fn = f"fold{k}.json" if k != "final" else "final_dev.json"
        params[k] = load_json(f"{art}/preprocess_params/{fn}")
    LOG.info(
        "dev %s행 / 환자 %s명 / 특징 %d개 / 양성 행 %.2f%%",
        f"{len(em):,}",
        f"{em['Patient_ID'].nunique():,}",
        len(model_cols),
        100 * em["SepsisLabel"].mean(),
    )
    return {
        "cfg": cfg,
        "registry": registry,
        "model_cols": model_cols,
        "F": feats,
        "em": em,
        "params": params,
    }


# --------------------------------------------------------------------------- 검사
def check_utility(ev: ModuleType, em: pd.DataFrame) -> None:
    """Vectorized utility must equal the official per-patient function exactly."""
    rng = np.random.default_rng(0)
    pick = rng.choice(em["Patient_ID"].unique(), 400, replace=False)
    sub = em[em["Patient_ID"].isin(pick)].reset_index(drop=True)
    worst = 0.0
    for rate in (0.02, 0.3, 0.9):
        pred = rng.random(len(sub)) < rate
        u_vec = ev.row_utility(sub, pred)
        for _, g in sub.groupby("Patient_ID", sort=False):
            ix = g.index.to_numpy()
            ref = ev.official_utility(g["SepsisLabel"].to_numpy(), pred[ix])
            worst = max(worst, abs(ref - u_vec[ix].sum()))
    _, d, septic = ev._time_index(sub)
    best = septic & (d >= -12) & (d <= 3)
    ok_norm = (
        ev.normalized_utility(sub, best) == 1.0
        and ev.normalized_utility(sub, np.zeros(len(sub), bool)) == 0.0
    )
    if worst != 0 or not ok_norm:
        raise RuntimeError(f"Utility 검사 실패 (최대 오차 {worst}, 정규화 {ok_norm})")
    LOG.info("[OK] 벡터화 Utility = 공식 함수 (환자 400명, 오차 0)")


# --------------------------------------------------------------------------- 교차검증
def run_cv(pp: ModuleType, ev: ModuleType, d: dict) -> tuple[pd.DataFrame, dict]:
    """Fit every (C, class_weight) per fold; return fold metrics and OOF preds."""
    em, y, folds = (
        d["em"],
        d["em"]["SepsisLabel"].to_numpy(),
        d["em"]["fold"].to_numpy(),
    )
    oofs = {(c, str(cw)): np.zeros(len(em)) for c in C_GRID for cw in CW_GRID}
    rows = []
    t_all = time.time()
    for k in range(N_FOLDS):
        tr, va = folds != k, folds == k
        x = pp.apply_tabular(d["F"], d["params"][k]["tabular"], d["model_cols"])
        for c in C_GRID:
            for cw in CW_GRID:
                t0 = time.time()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    m = LogisticRegression(
                        C=c, class_weight=cw, max_iter=1000, random_state=SEED
                    ).fit(x[tr], y[tr])
                p = m.predict_proba(x[va])[:, 1]
                oofs[(c, str(cw))][va] = p
                r = ev.evaluate(em[va], p)
                r.update(
                    fold=k,
                    C=c,
                    class_weight=str(cw),
                    n_iter=int(m.n_iter_[0]),
                    converged=not any(
                        "converge" in str(x_.message).lower() for x_ in w
                    ),
                    sec=round(time.time() - t0, 1),
                )
                rows.append(r)
        del x
        LOG.info("fold %d 완료 (%.0f초 경과)", k, time.time() - t_all)
    grid = pd.DataFrame(rows)
    if not grid["converged"].all():
        raise RuntimeError("수렴하지 않은 학습이 있다")
    return grid, oofs


def select_c(grid: pd.DataFrame) -> float:
    """Pick C with the highest mean AUPRC among class_weight='balanced'."""
    bal = grid[grid["class_weight"] == "balanced"]
    return float(bal.groupby("C")["AUPRC"].mean().idxmax())


def fold_table(grid: pd.DataFrame, best_c: float) -> pd.DataFrame:
    """Per-fold metrics of the selected setting with mean and std rows."""
    sel = grid[(grid["C"] == best_c) & (grid["class_weight"] == "balanced")]
    tbl = sel.set_index("fold")[FOLD_METRICS]
    tbl.loc["mean"] = tbl.mean()
    tbl.loc["std"] = tbl.iloc[:N_FOLDS].std()
    return tbl


# ------------------------------------------------------------------ 보정·임계값
def calibration(ev: ModuleType, em: pd.DataFrame, p_raw: np.ndarray) -> dict:
    """Cross-fitted Platt and isotonic calibration with Brier, ECE and Brier skill."""
    y = em["SepsisLabel"].to_numpy()
    p_platt = ev.crossfit_calibrate(em, p_raw, "platt")
    p_iso = ev.crossfit_calibrate(em, p_raw, "isotonic")
    base = np.full(len(y), y.mean())
    rows = []
    for name, p in [
        ("raw (balanced weights)", p_raw),
        ("Platt", p_platt),
        ("Isotonic", p_iso),
        (BASELINE_NAME, base),
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
    use = (
        "Platt"
        if cal.loc["Platt", "Brier"] <= cal.loc["Isotonic", "Brier"]
        else "Isotonic"
    )
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
    fpr, tpr, thr = roc_curve(y, p_cal)
    ok = fpr <= 0.05
    sens_thr = float(thr[ok][np.argmax(tpr[ok])])
    idx = ev._time_index(em)
    tbl = pd.DataFrame(
        [
            dict(name="utility_optimal", **ev.threshold_report(em, p_cal, thr_u, idx)),
            dict(
                name="specificity_95", **ev.threshold_report(em, p_cal, sens_thr, idx)
            ),
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
        "idx": idx,
    }


# --------------------------------------------------------------------------- 최종 모델
def fit_final(
    pp: ModuleType, ev: ModuleType, d: dict, best_c: float, p_raw: np.ndarray
) -> dict:
    """Fit the final model on all dev rows and calibrators on dev OOF predictions."""
    y = d["em"]["SepsisLabel"].to_numpy()
    x_dev = pp.apply_tabular(d["F"], d["params"]["final"]["tabular"], d["model_cols"])
    final = LogisticRegression(
        C=best_c, class_weight="balanced", max_iter=1000, random_state=SEED
    ).fit(x_dev, y)
    del x_dev
    group_of = {r["name"]: r["group"] for r in d["registry"]["features"]}
    coef = pd.DataFrame({"feature": d["model_cols"], "coef": final.coef_[0]})
    coef["group"] = coef["feature"].map(group_of)
    coef = coef.reindex(coef["coef"].abs().sort_values(ascending=False).index)
    return {
        "model": final,
        "coef": coef.reset_index(drop=True),
        "platt_ab": ev.fit_platt(p_raw, y),
        "isotonic": ev.fit_isotonic(p_raw, y),
    }


# --------------------------------------------------------------------------- 그림
def plot_roc_pr(ev: ModuleType, em: pd.DataFrame, p: np.ndarray, path: str) -> None:
    """ROC and precision-recall curves for all rows and ICULOS<=60 rows."""
    y = em["SepsisLabel"].to_numpy()
    le60 = em["iculos_le60"].to_numpy() == 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for mask, name, col in [
        (np.ones(len(y), bool), "all rows", "#1f5fae"),
        (le60, "ICULOS<=60", "#e76f51"),
    ]:
        fpr, tpr, _ = roc_curve(y[mask], p[mask])
        axes[0].plot(
            fpr,
            tpr,
            color=col,
            lw=1.8,
            label=f"{name} (AUROC {roc_auc_score(y[mask], p[mask]):.3f})",
        )
        pr, rc, _ = precision_recall_curve(y[mask], p[mask])
        axes[1].plot(
            rc,
            pr,
            color=col,
            lw=1.8,
            label=f"{name} (AUPRC {average_precision_score(y[mask], p[mask]):.3f})",
        )
    axes[0].plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--")
    axes[0].set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC (5-fold OOF, logistic regression)",
    )
    axes[1].axhline(
        y.mean(), color="gray", lw=0.8, ls="--", label=f"base rate {y.mean():.3f}"
    )
    axes[1].set(
        xlabel="Recall", ylabel="Precision", title="Precision-Recall (5-fold OOF)"
    )
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(
    y: np.ndarray, curves: list[tuple[np.ndarray, str, str]], path: str
) -> None:
    """Reliability diagram with 20 equal-count bins (full range and 0-0.3 zoom)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for p, name, col in curves:
        q = pd.qcut(pd.Series(p).rank(method="first"), 20, labels=False)
        g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").mean()
        for ax in axes:
            ax.plot(g["p"], g["y"], marker="o", ms=3, lw=1.5, color=col, label=name)
    axes[0].plot([0, 1], [0, 1], color="black", lw=0.8, ls="--")
    axes[0].set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive rate",
        title="Reliability (20 quantile bins)",
    )
    axes[1].plot([0, 0.3], [0, 0.3], color="black", lw=0.8, ls="--")
    axes[1].set(
        xlim=(0, 0.3),
        ylim=(0, 0.3),
        xlabel="Mean predicted probability",
        title="Zoom: 0-0.3",
    )
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_utility(
    ev: ModuleType,
    em: pd.DataFrame,
    p: np.ndarray,
    thr_u: float,
    idx: tuple,
    cal_name: str,
    path: str,
) -> None:
    """Normalized utility against the share of alerted rows."""
    grid_t = ev.threshold_grid(p, 80)
    us = [ev.normalized_utility(em, p >= t, idx) for t in grid_t]
    rates = [100 * (p >= t).mean() for t in grid_t]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rates, us, color="#1f5fae", lw=1.8)
    ax.axvline(
        100 * (p >= thr_u).mean(),
        color="crimson",
        ls="--",
        lw=1,
        label=f"utility-optimal (thr {thr_u:.3f})",
    )
    ax.set_xscale("log")
    ax.set(
        xlabel="Rows alerted (%)",
        ylabel="Normalized utility",
        title=f"Utility vs alert rate ({cal_name}-calibrated LR, 5-fold OOF)",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coef(coef: pd.DataFrame, path: str) -> None:
    """Top-20 coefficients of the final model."""
    top = coef.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(
        top["feature"],
        top["coef"],
        color=np.where(top["coef"] > 0, "#e76f51", "#1f5fae"),
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set(
        xlabel="Coefficient (log-odds per 1 IQR)",
        title="Logistic regression: top 20 coefficients (final, dev)",
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- 저장
def log_mlflow(
    grid: pd.DataFrame,
    best_c: float,
    n_feat: int,
    u_cf: float,
    brier_sel: float,
    cal_name: str,
    fig_dir: str,
    out_dir: str,
) -> None:
    """Record every setting in MLflow (SQLite in /tmp, copied to out_dir/mlflow)."""
    import mlflow

    local = "/tmp/sews_mlflow"
    shutil.rmtree(local, ignore_errors=True)
    os.makedirs(local)
    mlflow.set_tracking_uri(f"sqlite:///{local}/mlflow.db")
    exp_id = mlflow.create_experiment(
        "sews_logreg", artifact_location=f"{local}/artifacts"
    )
    names = {
        "AUROC": "AUROC",
        "AUPRC": "AUPRC",
        "AUROC_ICULOS<=60": "AUROC_ICULOS_le60",
        "AUROC_early(rel -6~-1 vs neg)": "AUROC_early",
    }
    for (c, cw), g in grid.groupby(["C", "class_weight"]):
        with mlflow.start_run(experiment_id=exp_id, run_name=f"logreg_C{c}_{cw}"):
            mlflow.log_params(
                {
                    "model": "LogisticRegression",
                    "C": c,
                    "class_weight": cw,
                    "seed": SEED,
                    "n_features": n_feat,
                    "cv": "5-fold patient",
                }
            )
            for _, r in g.iterrows():
                mlflow.log_metrics(
                    {"AUROC": r["AUROC"], "AUPRC": r["AUPRC"]}, step=int(r["fold"])
                )
            mlflow.log_metrics(
                {f"mean_{v}": float(g[k].mean()) for k, v in names.items()}
            )
            if (c, cw) == (best_c, "balanced"):
                mlflow.set_tag("selected", "true")
                mlflow.log_metrics(
                    {"utility_crossfit": u_cf, f"brier_{cal_name}": brier_sel}
                )
                mlflow.log_artifacts(fig_dir, artifact_path="figures")
    shutil.copytree(local, f"{out_dir}/mlflow", dirs_exist_ok=True)
    LOG.info(
        "[OK] MLflow run %d개 기록", len(mlflow.search_runs(experiment_ids=[exp_id]))
    )


def main() -> None:
    """Run cross-validation, calibration, thresholding and export."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    root = args.root.rstrip("/")
    src_dir = args.src_dir or str(Path(__file__).resolve().parents[1] / "src")
    out_dir = args.out_dir or f"{root}/results/logreg"
    fig_dir, model_dir = f"{out_dir}/figures", f"{out_dir}/model"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    sys.path.insert(0, src_dir)
    import sews_eval as ev
    import sews_preprocess as pp

    d = load_inputs(root, src_dir)
    em, y = d["em"], d["em"]["SepsisLabel"].to_numpy()
    check_utility(ev, em)

    grid, oofs = run_cv(pp, ev, d)
    grid.to_csv(f"{out_dir}/cv_grid.csv", index=False)
    best_c = select_c(grid)
    p_raw = oofs[(best_c, "balanced")]
    ftbl = fold_table(grid, best_c)
    ftbl.to_csv(f"{out_dir}/cv_metrics_by_fold.csv")
    LOG.info(
        "[선택] class_weight=balanced, C=%s\n%s",
        best_c,
        grid.groupby(["class_weight", "C"])[SHOW_METRICS].mean().round(4),
    )
    plot_roc_pr(ev, em, p_raw, f"{fig_dir}/roc_pr.png")

    cal = calibration(ev, em, p_raw)
    cal["table"].to_csv(f"{out_dir}/calibration.csv")
    plot_calibration(
        y,
        [
            (p_raw, "raw (balanced)", "#9a9a9a"),
            (cal["p_platt"], "Platt", "#1f5fae"),
            (cal["p_iso"], "Isotonic", "#2a9d8f"),
        ],
        f"{fig_dir}/calibration.png",
    )

    th = thresholds(ev, em, cal["p_cal"])
    th["table"].to_csv(f"{out_dir}/threshold_table.csv")
    plot_utility(
        ev,
        em,
        cal["p_cal"],
        th["thr_u"],
        th["idx"],
        cal["selected"],
        f"{fig_dir}/utility_threshold.png",
    )

    fin = fit_final(pp, ev, d, best_c, p_raw)
    fin["coef"].to_csv(f"{out_dir}/coefficients.csv", index=False)
    plot_coef(fin["coef"], f"{fig_dir}/coef_top20.png")
    joblib.dump(
        {"model": fin["model"], "isotonic": fin["isotonic"]},
        f"{model_dir}/logreg_final.joblib",
    )
    meta = {
        "model": "LogisticRegression",
        "C": best_c,
        "class_weight": "balanced",
        "seed": SEED,
        "features": d["model_cols"],
        "preprocess_params": "artifacts/preprocess_params/final_dev.json",
        "calibration": {
            "selected": cal["selected"],
            "platt_a_b": fin["platt_ab"],
            "formula": "p_cal = 1 / (1 + exp(-(a * logit(p_raw) + b)))",
        },
        "thresholds": {
            "utility_optimal": th["thr_u"],
            "spec_warning": 0.5,
            "spec_alert": 0.75,
        },
        "trained_on": "dev 전체 (test 미사용)",
        "sklearn_version": sklearn.__version__,
        "module_sha256": {
            "sews_features": d["registry"]["module_sha256"],
            "sews_preprocess": d["registry"]["preprocess"]["module_sha256"],
        },
    }
    with open(f"{model_dir}/logreg_final_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    folds = em["fold"].to_numpy()
    pd.DataFrame(
        {
            "row_id": em["row_id"],
            "Patient_ID": em["Patient_ID"],
            "fold": folds,
            "SepsisLabel": y,
            "p_raw": p_raw,
            "p_platt": cal["p_platt"],
            "p_isotonic": cal["p_iso"],
        }
    ).to_parquet(f"{out_dir}/oof_predictions.parquet", index=False)
    mean, std = ftbl.loc["mean"], ftbl.loc["std"]
    ctab = cal["table"]
    summary = {
        "model": f"LogisticRegression (L2, C={best_c}, class_weight=balanced)",
        "data": {
            "split": "dev 5-fold CV (환자 단위, SEED 42), test 미사용",
            "rows": int(len(em)),
            "patients": int(em["Patient_ID"].nunique()),
            "positive_row_rate": float(y.mean()),
            "features": len(d["model_cols"]),
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
        "spec_targets": {
            "AUROC>=0.82": bool(mean["AUROC"] >= 0.82),
            "AUPRC>=0.40": bool(mean["AUPRC"] >= 0.40),
            "Brier<=0.15 (calibrated)": bool(
                ctab.loc[cal["selected"], "Brier"] <= 0.15
            ),
        },
        "notes": [
            "행 단위 지표 기준",
            "test 평가는 전체 모델링 종료 후 1회",
            "보정기·임계값은 dev OOF 예측으로만 학습",
        ],
    }
    with open(f"{out_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    if not args.no_mlflow:
        log_mlflow(
            grid,
            best_c,
            len(d["model_cols"]),
            th["u_cf"],
            float(ctab.loc[cal["selected"], "Brier"]),
            cal["selected"],
            fig_dir,
            out_dir,
        )
    LOG.info(
        "[요약] AUROC %.3f +- %.3f | AUPRC %.3f +- %.3f | ICULOS<=60 AUROC %.3f | "
        "Utility(교차 적합) %.3f | Brier(%s) %.4f",
        mean["AUROC"],
        std["AUROC"],
        mean["AUPRC"],
        std["AUPRC"],
        mean["AUROC_ICULOS<=60"],
        th["u_cf"],
        cal["selected"],
        ctab.loc[cal["selected"], "Brier"],
    )


if __name__ == "__main__":
    main()
