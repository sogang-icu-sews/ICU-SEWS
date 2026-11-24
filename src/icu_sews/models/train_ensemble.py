"""ICU-7: LightGBM·LSTM·GRU Optuna 탐색 → dev 5-fold OOF → 가중 앙상블 → (선택) test 평가.

notebooks/ICU-7_ensemble.ipynb와 같은 계산을 하는 스크립트다. 셀(# %%) 단위로 VS Code·Jupyter에서
실행할 수도 있고, 저장소 루트에서 한 번에 실행할 수도 있다.

    # Colab: Drive를 먼저 연결하고, 저장소를 받은 뒤
    pip install optuna==5.1.0
    python data/models/train_ensemble.py                      # 탐색·5-fold·앙상블 고정까지
    SEWS_ENS_OPEN_TEST=1 python data/models/train_ensemble.py  # 고정 뒤 test 한 번 평가

환경 변수: SEWS_ROOT(데이터 파트 산출물, 기본 /content/drive/MyDrive/capston),
SEWS_ENS_OUT(결과, 기본 /content/drive/MyDrive/SEWS_ENSEMBLE),
SEWS_ENS_RUN(실행 이름, 기본 ens_script_v1),
SEWS_ENS_OPEN_TEST=1(test 평가), SEWS_ENS_SMOKE=1(빠른 점검, 실제 test 환자는 쓰지 않음).
끊겨도 같은 명령을 다시 실행하면 끝난 trial·fold·고정값을 Drive에서 읽어 이어 간다.
환자 역할은 팀 최종 GRU(src/icu_sews/models/final_gru.py)의 roles_for를 그대로 쓴다.
"""

# %%
import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.special import expit, logit  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch import nn  # noqa: E402

REPO_DIR = Path(__file__).resolve().parents[2]  # 저장소 루트 (data/models/의 두 단계 위)
for p in [REPO_DIR / "src", REPO_DIR / "data" / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from icu_sews.evaluation import compare_gru as CG  # noqa: E402  팀 평가 함수
from icu_sews.models import final_gru as FG  # noqa: E402  팀 최종 GRU (환자 역할)
from icu_sews.models import train_gru as T  # noqa: E402  팀 GRU 데이터 로드·창 정렬
from icu_sews.models.gru import GruSepsisModel  # noqa: E402  팀 GRU 모델

# ===== [7-0] 설정 =====
# 탐색 범위나 trial 수를 바꾸면 RUN_NAME도 바꾼다.
# (같은 RUN_NAME으로 다른 설정을 이어 돌리면 멈춘다.)

RUN_NAME = os.environ.get("SEWS_ENS_RUN", "ens_script_v1")
DATA_ROOT = os.environ.get("SEWS_ROOT", "/content/drive/MyDrive/capston")  # 데이터 파트 산출물
OUT_ROOT = os.environ.get("SEWS_ENS_OUT", "/content/drive/MyDrive/SEWS_ENSEMBLE")
GRU_FINAL_DIR = os.environ.get(
    "SEWS_GRU_FINAL", "/content/drive/MyDrive/SEWS_GRU/final/gru_v2_platt_final"
)  # 팀 최종 GRU 결과 폴더 (있으면 환자 역할이 같은지 확인)


SEED = 42
TUNE_VAL_FOLD = 3  # 탐색: fold 0~2 학습 → fold 3 검증
OBJECTIVE = "AUPRC"  # 탐색·가중치 목표 ("AUPRC" 또는 "AUROC"). 팀 GRU도 검증 AUPRC로 골랐다.
N_TRIALS = {"lgbm": 25, "lstm": 15, "gru": 15}
TIMEOUT_MIN = {"lgbm": 120, "lstm": 120, "gru": 150}  # 한 번 실행에서 모델별 최대 탐색 시간
MAX_EPOCHS, PATIENCE = 10, 2  # 신경망 탐색: 검증 점수가 2 epoch 연속 안 오르면 중단
LGBM_MAX_ROUNDS, LGBM_EARLY_STOP = 2000, 100
ENSEMBLE_TRIALS = 200  # 앙상블 가중치 탐색 (fold마다)
TARGET_FALSE_ALARM = 0.10  # 비패혈증 환자 중 한 번이라도 경보를 받는 비율 목표 (팀 GRU와 같음)

OPEN_TEST = os.environ.get("SEWS_ENS_OPEN_TEST") == "1"  # 앙상블을 확인·고정한 뒤에만 1로 실행

# 빠른 점검용 (코드가 끝까지 도는지만 확인). 실제 test 환자는 쓰지 않는다.
SMOKE = os.environ.get("SEWS_ENS_SMOKE") == "1"
if SMOKE:
    RUN_NAME += "_smoke"
    N_TRIALS = {"lgbm": 3, "lstm": 3, "gru": 3}
    MAX_EPOCHS, LGBM_MAX_ROUNDS, LGBM_EARLY_STOP = 2, 60, 20
    ENSEMBLE_TRIALS = 30
    SMOKE_FRAC = 0.06  # 역할별 환자 비율
print("RUN:", RUN_NAME, "| SMOKE" if SMOKE else "")


# %%
# ===== [7-1] 환경 =====
IN_COLAB = Path("/content").exists()  # Colab에서는 입력을 로컬 디스크로 복사해 쓴다


def git(*args):
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_DIR), *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


REPO_COMMIT = git("rev-parse", "HEAD")
# 탐색을 이어 갈지 판단할 때는 커밋이 아니라 계산에 쓰는 팀 코드 파일 내용을 본다.
# (대시보드 커밋 등은 무관하다.)
CODE_FILES = [
    "src/icu_sews/models/gru.py",
    "src/icu_sews/models/train_gru.py",
    "src/icu_sews/models/final_gru.py",
    "src/icu_sews/evaluation/compare_gru.py",
    "data/src/sews_preprocess.py",
    "data/src/sews_eval.py",
    "data/models/train_ensemble.py",
]
CODE_SHA = hashlib.sha256(
    json.dumps(
        {f: hashlib.sha256((REPO_DIR / f).read_bytes()).hexdigest() for f in CODE_FILES}
    ).encode()
).hexdigest()

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
warnings.filterwarnings("ignore", message="Overriding the parameters from Reference Dataset")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu" and not SMOKE:
    raise RuntimeError("GPU가 없다. 런타임 유형을 T4 GPU로 바꾼다.")
torch.set_num_threads(min(torch.get_num_threads(), 4))
N_THREADS = os.cpu_count() or 2

OUT = Path(OUT_ROOT) / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
VERSIONS = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "lightgbm": lgb.__version__,
    "optuna": optuna.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "repo_commit": REPO_COMMIT,
}
print("장치:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "")
print("버전:", VERSIONS)
print("결과 폴더:", OUT)


# %%
# ===== [7-2] 데이터와 환자 역할 =====
# 팀 GRU의 load_data로 읽는다.
# 행 정렬, 환자 경계, 분할 파일, 전처리 모듈 sha256, 설정 v4를 모두 검사한다.
DATA_FILES = ["seq_values.npy", "seq_row_start.npy", "eval_meta.parquet", "features.parquet"]
ART_FILES = [
    "preprocess_config.json",
    "feature_registry.json",
    "seq_meta.json",
    "patient_split.csv",
]
ART_FILES += [f"preprocess_params/fold{k}.json" for k in range(5)]

if IN_COLAB:  # Drive에서 직접 읽으면 느리므로 Colab 로컬 디스크로 복사한다
    local = Path("/content/sews_ens_inputs")
    for folder, names in [("data", DATA_FILES), ("artifacts", ART_FILES)]:
        for name in names:
            src, dst = Path(DATA_ROOT, folder, name), local / folder / name
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst.with_name(dst.name + ".partial"))
                dst.with_name(dst.name + ".partial").replace(dst)
    INPUT_ROOT = local
else:
    INPUT_ROOT = Path(DATA_ROOT)

t0 = time.time()
d = T.load_data(INPUT_ROOT)
em, cfg, pp = d["em"], d["cfg"], d["pp"]
Y = em["SepsisLabel"].to_numpy(dtype=np.int64)
PID = em["Patient_ID"].to_numpy()
FOLD = em["fold"].to_numpy()
print(
    f"[OK] load_data 검사 통과: {len(em):,}행, {em.Patient_ID.nunique():,}명",
    f"({time.time() - t0:.0f}초)",
)

# --- 환자 역할: 팀 최종 GRU와 같은 함수 ---
patients, roles = FG.roles_for(em)
dev_train = np.flatnonzero(
    ((em.split == "dev") & (em.fold != FG.PLAN["held_out_dev_fold"])).to_numpy()
)
assert np.array_equal(roles["train"], dev_train)
assert (em.split.iloc[roles["test"]] == "test").all()
gru_roles = Path(GRU_FINAL_DIR) / "patient_roles.csv"
if gru_roles.exists():
    ref = pd.read_csv(gru_roles).set_index("Patient_ID")["role"].sort_index()
    same = ref.equals(patients.set_index("Patient_ID")["role"].sort_index())
    assert same, "팀 최종 GRU의 patient_roles.csv와 환자 역할이 다르다"
    print("[OK] 팀 최종 GRU patient_roles.csv와 환자 역할 일치")
else:
    print(
        "(팀 최종 GRU 결과 폴더가 없어 patient_roles.csv 비교는 건너뜀.",
        "같은 함수로 나눴으므로 같다)",
    )

tr = roles["train"]
R = {
    "tune_train": tr[FOLD[tr] != TUNE_VAL_FOLD],
    "tune_val": tr[FOLD[tr] == TUNE_VAL_FOLD],
    "train": tr,
    "cal": roles["calibration"],
    "thr": roles["threshold"],
    "test": roles["test"],
}

if (
    SMOKE
):  # 역할별로 환자 일부만 쓴다. test 자리에는 실제 test가 아니라 dev 환자로 만든 가짜 test를 둔다.
    rng = np.random.default_rng(SEED)
    sepsis_pt = em.groupby("Patient_ID")["SepsisLabel"].max()

    def sample_patients(rows, frac):
        ids = np.unique(PID[rows])
        keep = []
        for flag in (0, 1):  # 패혈증 여부로 층화
            group = ids[sepsis_pt.loc[ids].to_numpy() == flag]
            keep.append(rng.choice(group, max(3, int(round(frac * len(group)))), replace=False))
        return rows[np.isin(PID[rows], np.concatenate(keep))]

    fake_test = sample_patients(tr[FOLD[tr] == 0], 0.08)
    rest = np.setdiff1d(tr, fake_test)
    train_s = sample_patients(rest, SMOKE_FRAC)
    R = {
        "tune_train": train_s[FOLD[train_s] != TUNE_VAL_FOLD],
        "tune_val": train_s[FOLD[train_s] == TUNE_VAL_FOLD],
        "train": train_s,
        "cal": sample_patients(roles["calibration"], SMOKE_FRAC * 2),
        "thr": sample_patients(roles["threshold"], SMOKE_FRAC * 2),
        "test": fake_test,
    }
    assert not (em.split.iloc[R["test"]] == "test").any()

# --- 역할 점검: 환자가 겹치지 않고, 두 클래스가 모두 있고, test는 학습·탐색에 없다 ---
assert np.array_equal(np.sort(np.r_[R["tune_train"], R["tune_val"]]), R["train"])
main_roles = ["train", "cal", "thr", "test"]
for i, a in enumerate(main_roles):
    assert len(np.unique(Y[R[a]])) == 2, f"{a}: 두 클래스가 모두 필요하다"
    for b in main_roles[i + 1 :]:
        assert not np.intersect1d(PID[R[a]], PID[R[b]]).size, f"{a}와 {b}의 환자가 겹친다"
assert not np.intersect1d(PID[R["tune_train"]], PID[R["tune_val"]]).size

role_table = pd.DataFrame(
    [
        {
            "역할": k,
            "환자": len(np.unique(PID[v])),
            "패혈증 환자": int(pd.Series(Y[v]).groupby(PID[v]).max().sum()),
            "행": len(v),
            "양성 행 비율(%)": round(100 * Y[v].mean(), 2),
        }
        for k, v in R.items()
    ]
).set_index("역할")
print(role_table)
role_rows = pd.concat(
    [pd.DataFrame({"row_id": v, "Patient_ID": PID[v], "role": k}) for k, v in R.items()],
    ignore_index=True,
)
role_rows.to_parquet(
    OUT / "row_roles.parquet", index=False
)  # 행마다 역할 (tune_train·tune_val은 train의 일부)

# --- LightGBM 입력: 데이터 파트 특징 125개 (최근 24시간, ICULOS 없음) ---
registry = json.loads(Path(INPUT_ROOT, "artifacts/feature_registry.json").read_text())
FEATURES = [f["name"] for f in registry["features"] if not f["optional"]]
assert len(FEATURES) == registry["n_model_input"] == 125 and "ICULOS" not in FEATURES
feat = pd.read_parquet(Path(INPUT_ROOT, "data/features.parquet"), columns=["row_id", *FEATURES])
assert np.array_equal(feat["row_id"].to_numpy(), em["row_id"].to_numpy())
X_TAB = feat[FEATURES].to_numpy(np.float32)
del feat
print(f"[OK] LightGBM 특징 {X_TAB.shape[1]}개, 행 {X_TAB.shape[0]:,}")


# %%
# ===== [7-3] 스케일링 통계 (학습 행에서만) =====
# 탐색: fold 0~2 행, 재학습: fold 0~3 행. 팀 최종 GRU의 fit_preprocessing과 같은 함수다.
# LightGBM은 결측을 직접 다루므로 스케일링·대치를 하지 않는다.
PARAMS = {
    "tune": FG.fit_preprocessing(d, R["tune_train"]),
    "final": FG.fit_preprocessing(d, R["train"]),
}
for key, rows in [("tune", R["tune_train"]), ("final", R["train"])]:
    assert PARAMS[key]["sequence"]["n_rows"] == len(rows)
    assert PARAMS[key]["sequence"]["vars"] == cfg["model_prep"]["sequence"]["vars"]

# 정적 변수 5개(Age, Gender, Unit)는 행마다 한 번 스케일해서 GPU에 올려 둔다.
ST_T = {
    k: torch.from_numpy(pp.apply_tabular(d["static"], PARAMS[k]["tabular"], T.STATIC)).to(DEVICE)
    for k in PARAMS
}
Y_T = torch.from_numpy(Y.astype(np.float32)).to(DEVICE)
print("정적 변수:", T.STATIC, "| 시퀀스 변수:", len(cfg["model_prep"]["sequence"]["vars"]))


# %%
# ===== [7-4] 24시간 창 (GPU에서 생성) =====
# 데이터 파트 seq_windows와 같은 계산을 GPU에서 한다: 창 안 forward fill, (x-중앙값)/IQR, ±clip,
# 측정 mask, 마지막 측정 이후 시간 delta. 같은 환자의 현재·과거 행만 쓴다.
# LSTM: 왼쪽 빈 행을 채운 24칸 창 (ICU-6과 같음).
# GRU : 팀 train_gru.pack_inputs처럼 실제 관측을 앞으로 당기고 길이를 넘겨 패딩을 무시한다.
SEQ = cfg["model_prep"]["sequence"]
MASK_IDX = [SEQ["vars"].index(v) for v in SEQ["mask_delta_vars"]]
N_IN = len(SEQ["vars"]) + 2 * len(MASK_IDX)
WIN = 24


class Windows:
    def __init__(self, values, starts, device):
        self.vals = torch.from_numpy(np.array(values, dtype=np.float32)).to(device)  # 복사본
        self.start = torch.from_numpy(np.array(starts, dtype=np.int64)).to(device)
        self.steps = torch.arange(WIN, device=device)
        self.mask_idx = torch.tensor(MASK_IDX, device=device)
        self.device = device

    def scaler(self, sp):
        f = lambda v: torch.tensor(v, dtype=torch.float32, device=self.device)  # noqa: E731
        return {"c": f(sp["center"]), "s": f(sp["scale"]), "clip": float(sp["clip"])}

    def left(self, rows_t, sc):
        """(B, 24, V+2M): 데이터 파트 seq_windows의 x, mask, delta를 이어 붙인 것과 같다."""
        idx = rows_t[:, None] - (WIN - 1 - self.steps)[None, :]
        valid = idx >= self.start[rows_t][:, None]
        raw = self.vals[idx.clamp(min=0)]
        raw = torch.where(valid[..., None], raw, torch.full_like(raw, float("nan")))
        obs = ~torch.isnan(raw)
        neg = torch.full(raw.shape, -1, dtype=torch.long, device=raw.device)
        li = torch.where(obs, self.steps[None, :, None].expand_as(neg), neg).cummax(dim=1).values
        ff = torch.gather(raw, 1, li.clamp(min=0))
        ff = torch.where(li >= 0, ff, torch.full_like(ff, float("nan")))
        delta = torch.where(li >= 0, self.steps[None, :, None] - li, torch.full_like(li, WIN))
        x = torch.nan_to_num(
            torch.clamp((ff - sc["c"]) / sc["s"], -sc["clip"], sc["clip"]), nan=0.0
        )
        m = self.mask_idx
        return torch.cat([x, obs[..., m].float(), delta[..., m].float() / WIN], dim=2)

    def packed(self, rows_t, sc):
        """GRU 입력: 실제 관측(길이 L ≤ 24)을 앞으로 당긴 창과 길이."""
        w = self.left(rows_t, sc)
        lengths = (rows_t - self.start[rows_t] + 1).clamp(max=WIN)
        idx = (WIN - lengths[:, None] + self.steps[None, :]).clamp(max=WIN - 1)
        out = torch.gather(w, 1, idx[..., None].expand(-1, -1, w.shape[2]))
        out = torch.where(self.steps[None, :, None] >= lengths[:, None, None], 0.0, out)
        return out, lengths


WINS = Windows(d["values"], d["starts"], DEVICE)
SC = {k: WINS.scaler(PARAMS[k]["sequence"]) for k in PARAMS}

# --- 점검: 데이터 파트 seq_windows, 팀 pack_inputs와 같은지 ---
# (dev 행 표본, 24시간 미만 기록 포함)
rng = np.random.default_rng(0)
dev_rows = np.r_[R["train"], R["cal"], R["thr"]]
pos = dev_rows - d["starts"][dev_rows]
pick = np.unique(
    np.r_[
        rng.choice(dev_rows, min(3000, len(dev_rows)), replace=False),
        rng.choice(dev_rows[pos < 23], min(1000, int((pos < 23).sum())), replace=False),
    ]
)
for key in PARAMS:
    sp = PARAMS[key]["sequence"]
    ref = pp.seq_windows(d["values"], d["starts"], pick, sp, cfg)
    ref_left = np.concatenate([ref["x"], ref["mask"], ref["delta"]], axis=2)
    lengths_np = np.minimum(pick - d["starts"][pick] + 1, WIN).astype(np.int64)
    ref_packed = T.pack_inputs(ref, lengths_np)
    rows_t = torch.from_numpy(pick).to(DEVICE)
    got_left = WINS.left(rows_t, SC[key]).cpu().numpy()
    got_packed, got_len = WINS.packed(rows_t, SC[key])
    assert got_left.shape == ref_left.shape == (len(pick), WIN, N_IN)
    assert np.allclose(got_left, ref_left, rtol=1e-6, atol=1e-6), "LSTM 창이 seq_windows와 다르다"
    assert np.array_equal(got_len.cpu().numpy(), lengths_np)
    assert np.allclose(got_packed.cpu().numpy(), ref_packed, rtol=1e-6, atol=1e-6), (
        "GRU 창이 다르다"
    )
print(f"[OK] GPU 창 = seq_windows (LSTM), = pack_inputs (GRU): {len(pick):,}개 창, 통계 2종")


# %%
# ===== [7-5] 모델과 학습 함수 =====
def score(y, p):
    """탐색·가중치 목표 점수 (행 단위)."""
    return average_precision_score(y, p) if OBJECTIVE == "AUPRC" else roc_auc_score(y, p)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


class SepsisLSTM(nn.Module):
    """ICU-6 LSTM: 24시간 창 → 마지막 은닉 상태 + 정적 변수 → 2층 분류기."""

    def __init__(self, n_in, n_static, hidden, layers, dropout, head):
        super().__init__()
        self.lstm = nn.LSTM(
            n_in,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head, 1)
        )

    def forward(self, x, st):
        out, _ = self.lstm(x)
        return self.head(torch.cat([out[:, -1], st], dim=1)).squeeze(1)


def make_net(kind, hp):
    if kind == "lstm":
        return SepsisLSTM(
            N_IN, len(T.STATIC), hp["hidden"], hp["layers"], hp["dropout"], hp["head"]
        ).to(DEVICE)
    return GruSepsisModel(  # 팀 GRU 모델 그대로
        input_size=N_IN,
        hidden_size=hp["hidden"],
        num_layers=hp["layers"],
        dropout=hp["dropout"],
        static_size=len(T.STATIC),
    ).to(DEVICE)


def net_logits(kind, model, rows_t, key):
    if kind == "lstm":
        return model(WINS.left(rows_t, SC[key]), ST_T[key][rows_t])
    x, lengths = WINS.packed(rows_t, SC[key])
    return model(x, ST_T[key][rows_t], lengths)


@torch.no_grad()
def predict_net(kind, model, rows, key, bs=4096):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        r = torch.from_numpy(rows[i : i + bs]).to(DEVICE)
        out.append(torch.sigmoid(net_logits(kind, model, r, key)).double().cpu())
    return torch.cat(out).numpy()


def pos_weight(rows, how):
    n_pos = float(Y[rows].sum())
    ratio = (len(rows) - n_pos) / n_pos
    return {"ratio": ratio, "sqrt": math.sqrt(ratio), "none": 1.0}[how]


def train_net(kind, hp, train_rows, key, epochs, val_rows=None, trial=None):
    """val_rows가 있으면 epoch마다 검증하고 가장 좋은 epoch의 가중치로 돌려준다(탐색).
    없으면 epochs만큼 학습한 마지막 가중치를 돌려준다(재학습)."""
    set_seed(SEED)
    model = make_net(kind, hp)
    if kind == "lstm":
        opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    else:  # 팀 GRU와 같은 Adam
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    pw = torch.tensor(pos_weight(train_rows, hp["pos_weight"]), device=DEVICE)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    rng = np.random.default_rng(SEED)
    best, best_state, best_ep, wait, hist = -np.inf, None, 0, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        order = rng.permutation(train_rows)
        for i in range(0, len(order), hp["batch"]):
            r = torch.from_numpy(order[i : i + hp["batch"]]).to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = lossf(net_logits(kind, model, r, key), Y_T[r])
            if not torch.isfinite(loss):
                raise optuna.TrialPruned("손실이 발산했다") if trial else RuntimeError("손실 발산")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(r)
        rec = {"epoch": ep, "loss": total / len(order), "sec": round(time.time() - t0, 1)}
        if val_rows is not None:
            p = predict_net(kind, model, val_rows, key)
            rec["val"] = float(score(Y[val_rows], p))
            rec["val_AUROC"] = float(roc_auc_score(Y[val_rows], p))
            if rec["val"] > best:
                best, best_ep, wait = rec["val"], ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
        hist.append(rec)
        shown = ", ".join(f"{k} {v:.4f}" for k, v in rec.items() if k not in ("epoch", "sec"))
        print(f"   {kind} epoch {ep}: {shown}, {rec['sec']:.0f}초")
        if trial is not None:
            trial.set_user_attr("history", hist)
            trial.report(rec["val"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if val_rows is not None and wait >= PATIENCE:
            break
    if val_rows is not None:
        model.load_state_dict(best_state)
    return model, hist, best_ep, best


# --- LightGBM ---
def lgbm_params(hp, rows):
    params = {
        "objective": "binary",
        "metric": "average_precision" if OBJECTIVE == "AUPRC" else "auc",
        "learning_rate": hp["learning_rate"],
        "num_leaves": hp["num_leaves"],
        "min_data_in_leaf": hp["min_data_in_leaf"],
        "feature_fraction": hp["feature_fraction"],
        "bagging_fraction": hp["bagging_fraction"],
        "bagging_freq": 1,
        "lambda_l1": hp["lambda_l1"],
        "lambda_l2": hp["lambda_l2"],
        "seed": SEED,
        "num_threads": N_THREADS,
        "force_col_wise": True,
        "verbose": -1,
    }
    if hp["pos_weight"] != "none":  # 양성 가중치는 그 단계의 학습 행에서 계산한다
        params["scale_pos_weight"] = pos_weight(rows, hp["pos_weight"])
    return params


def lgbm_dataset(rows, reference=None):
    return lgb.Dataset(
        X_TAB[rows],
        Y[rows],
        reference=reference,
        params={"feature_pre_filter": False, "max_bin": 255, "verbose": -1},
        free_raw_data=False,
    ).construct()


print("모델 입력: 시퀀스", N_IN, "채널 + 정적", len(T.STATIC), "| LightGBM 특징", len(FEATURES))


# %%
# ===== [7-6] Optuna 준비: 탐색 범위, 기본 설정(trial 0), 저장·재개 =====
# 탐색 범위. ("float"|"int", 최소, 최대, 로그 척도) 또는 ("cat", [후보])
SPACES = {
    "lgbm": {
        "learning_rate": ("float", 0.02, 0.15, True),
        "num_leaves": ("int", 15, 255, True),
        "min_data_in_leaf": ("int", 50, 2000, True),
        "feature_fraction": ("float", 0.3, 1.0, False),
        "bagging_fraction": ("float", 0.5, 1.0, False),
        "lambda_l1": ("float", 1e-8, 10.0, True),
        "lambda_l2": ("float", 1e-8, 10.0, True),
        "pos_weight": ("cat", ["none", "sqrt"]),
    },
    "lstm": {
        "hidden": ("cat", [32, 64, 128]),
        "layers": ("int", 1, 2, False),
        "dropout": ("float", 0.0, 0.5, False),
        "head": ("cat", [16, 32, 64]),
        "lr": ("float", 1e-4, 3e-3, True),
        "weight_decay": ("float", 1e-6, 1e-2, True),
        "batch": ("cat", [512, 1024, 2048]),
        "pos_weight": ("cat", ["sqrt", "ratio"]),
    },
    "gru": {
        "hidden": ("cat", [32, 64, 128]),
        "layers": ("int", 1, 2, False),
        "dropout": ("float", 0.0, 0.5, False),
        "lr": ("float", 1e-4, 3e-3, True),
        "batch": ("cat", [128, 256, 512, 1024]),
        "pos_weight": ("cat", ["ratio", "sqrt"]),
    },
}
# trial 0 = 지금까지 쓴 설정. Optuna가 이보다 나은지 바로 비교할 수 있다.
DEFAULTS = {
    "lgbm": {  # ICU-3 [3-5] 기준선 점검 설정 (λ≈0)
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "lambda_l1": 1e-8,
        "lambda_l2": 1e-8,
        "pos_weight": "none",
    },
    "lstm": {  # ICU-6 설정 (weight decay 0 ≈ 1e-6)
        "hidden": 64,
        "layers": 1,
        "dropout": 0.2,
        "head": 32,
        "lr": 3e-4,
        "weight_decay": 1e-6,
        "batch": 1024,
        "pos_weight": "sqrt",
    },
    "gru": {  # 팀 최종 GRU v2 설정
        "hidden": 64,
        "layers": 1,
        "dropout": 0.2,
        "lr": 3e-4,
        "batch": 128,
        "pos_weight": "ratio",
    },
}
MODELS = ["lgbm", "lstm", "gru"]


def suggest(trial, name):
    hp = {}
    for k, spec in SPACES[name].items():
        if spec[0] == "cat":
            hp[k] = trial.suggest_categorical(k, spec[1])
        elif spec[0] == "int":
            hp[k] = trial.suggest_int(k, spec[1], spec[2], log=spec[3])
        else:
            hp[k] = trial.suggest_float(k, spec[1], spec[2], log=spec[3])
    return hp


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def file_sha(path):
    return T.digest(path)


ROWS_SHA = {
    k: hashlib.sha256(np.ascontiguousarray(v, dtype=np.int64).tobytes()).hexdigest()
    for k, v in R.items()
}


def study_identity(name):
    """탐색 조건. 같은 RUN_NAME에서 이것이 바뀌면 이어서 돌리지 않는다 (trial 수는 늘려도 된다)."""
    base = {
        "space": SPACES.get(name),
        "default": DEFAULTS.get(name),
        "objective": OBJECTIVE,
        "seed": SEED,
        "rows": ROWS_SHA,
        "data": d["hashes"],
        "code": CODE_SHA,
    }
    if name in ("lstm", "gru"):
        base.update(max_epochs=MAX_EPOCHS, patience=PATIENCE)
    if name == "lgbm":
        base.update(max_rounds=LGBM_MAX_ROUNDS, early_stop=LGBM_EARLY_STOP)
    return sha(base)


LOCAL_DB = Path("/content/sews_optuna" if IN_COLAB else OUT / "_local") / f"{RUN_NAME}.db"
DRIVE_DB = OUT / "optuna.db"
LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)
if DRIVE_DB.exists() and not LOCAL_DB.exists():  # 새 런타임: Drive 백업에서 이어 간다
    shutil.copyfile(DRIVE_DB, LOCAL_DB)
STORAGE = optuna.storages.RDBStorage(
    f"sqlite:///{LOCAL_DB}",
    heartbeat_interval=60,
    # 끊긴 런타임에서 '실행 중'으로 남은 trial은 실패 처리 후 같은 설정으로 한 번 다시 돈다
    grace_period=180,
    heartbeat_stale_trial_callback=optuna.storages.RetryHeartbeatStaleTrialCallback(max_retry=1),
)


def backup(study=None, trial=None):
    """trial이 끝날 때마다 Drive로 안전하게 복사 (sqlite backup API)."""
    tmp = DRIVE_DB.with_name("optuna.db.partial")
    src, dst = sqlite3.connect(LOCAL_DB), sqlite3.connect(tmp)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
    tmp.replace(DRIVE_DB)


def open_study(name, n_startup, pruner):
    study = optuna.create_study(
        study_name=name,
        storage=STORAGE,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=n_startup),
        pruner=pruner,
        load_if_exists=True,
    )
    ident = study_identity(name)
    saved = study.user_attrs.get("identity")
    if saved is None:
        study.set_user_attr("identity", ident)
    elif saved != ident:
        raise RuntimeError(f"{name}: 같은 RUN_NAME의 탐색 조건이 바뀌었다. SEWS_ENS_RUN을 바꾼다.")
    return study


FINISHED = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)


def run_study(name, objective, n_trials, timeout_min, n_startup, pruner):
    study = open_study(name, n_startup, pruner)
    if not study.trials and name in DEFAULTS:
        study.enqueue_trial(DEFAULTS[name])
    left = n_trials - sum(t.state in FINISHED for t in study.trials)
    if left > 0:
        print(f"{name}: trial {left}개 실행 (이미 끝난 trial {n_trials - left}개)")
        study.optimize(
            objective,
            n_trials=left,
            timeout=timeout_min * 60,
            callbacks=[backup],
            gc_after_trial=True,
        )
    else:
        print(f"{name}: trial {n_trials}개 모두 끝남. 저장된 결과를 쓴다.")
    backup()
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(
        f"{name}: 완료 {len(done)}개,",
        f"가지치기 {sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)}개",
    )
    if done:
        print(f"  최고 {OBJECTIVE} {study.best_value:.4f} | {study.best_params}")
    return study


print("탐색 저장소:", LOCAL_DB, "→ Drive 백업", DRIVE_DB)


# %%
# ===== [7-7] LightGBM 탐색 (CPU): fold 0~2 학습 → fold 3 검증 =====
# 검증 점수가 LGBM_EARLY_STOP번 연속 오르지 않으면 멈추고, 그때까지 가장 좋은 반복 수를 기록한다.
# 50회마다 중간 점수를 보고해 다른 trial보다 확실히 나쁘면 일찍 끝낸다(가지치기).
DTRAIN = lgbm_dataset(R["tune_train"])
DVAL = lgbm_dataset(R["tune_val"], reference=DTRAIN)
X_TUNE_VAL = X_TAB[R["tune_val"]]


def objective_lgbm(trial):
    hp = suggest(trial, "lgbm")
    t0 = time.time()

    def report(env):
        if (env.iteration + 1) % 50 == 0:
            trial.report(env.evaluation_result_list[0][2], env.iteration + 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

    booster = lgb.train(
        lgbm_params(hp, R["tune_train"]),
        DTRAIN,
        num_boost_round=LGBM_MAX_ROUNDS,
        valid_sets=[DVAL],
        callbacks=[lgb.early_stopping(LGBM_EARLY_STOP, verbose=False), report],
    )
    p = booster.predict(X_TUNE_VAL, num_iteration=booster.best_iteration)
    yv = Y[R["tune_val"]]
    trial.set_user_attr("best_iteration", int(booster.best_iteration))
    trial.set_user_attr("val_AUROC", float(roc_auc_score(yv, p)))
    trial.set_user_attr("val_AUPRC", float(average_precision_score(yv, p)))
    trial.set_user_attr("sec", round(time.time() - t0, 1))
    print(
        f"   lgbm trial {trial.number}: {OBJECTIVE} {score(yv, p):.4f},",
        f"반복 {booster.best_iteration}, {time.time() - t0:.0f}초",
    )
    return float(score(yv, p))


STUDIES = {}
STUDIES["lgbm"] = run_study(
    "lgbm",
    objective_lgbm,
    N_TRIALS["lgbm"],
    TIMEOUT_MIN["lgbm"],
    n_startup=min(8, N_TRIALS["lgbm"]),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=300),
)
DTRAIN = DVAL = X_TUNE_VAL = None  # 메모리 해제


# %%
# ===== [7-8] LSTM 탐색 (GPU): fold 0~2 학습 → fold 3 검증 =====
# epoch마다 fold 3 점수를 재고, 가장 좋은 epoch를 기록한다(재학습 때 이 epoch 수로 고정).
# 2번째 epoch부터 다른 trial의 같은 epoch 중앙값보다 낮으면 일찍 끝낸다.


def objective_net(kind):
    def objective(trial):
        hp = suggest(trial, kind)
        t0 = time.time()
        print(f" {kind} trial {trial.number}: {hp}")
        model, hist, best_ep, best = train_net(
            kind, hp, R["tune_train"], "tune", MAX_EPOCHS, R["tune_val"], trial
        )
        rec = hist[best_ep - 1]
        trial.set_user_attr("best_epoch", int(best_ep))
        trial.set_user_attr("val_AUROC", rec["val_AUROC"])
        trial.set_user_attr("sec", round(time.time() - t0, 1))
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return float(best)

    return objective


NET_PRUNER = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1)
STUDIES["lstm"] = run_study(
    "lstm",
    objective_net("lstm"),
    N_TRIALS["lstm"],
    TIMEOUT_MIN["lstm"],
    min(5, N_TRIALS["lstm"]),
    NET_PRUNER,
)


# %%
# ===== [7-9] GRU 탐색 (GPU): fold 0~2 학습 → fold 3 검증 =====
# 모델은 팀 GruSepsisModel, 최적화는 팀과 같은 Adam, trial 0은 팀 최종 GRU v2 설정이다.
STUDIES["gru"] = run_study(
    "gru",
    objective_net("gru"),
    N_TRIALS["gru"],
    TIMEOUT_MIN["gru"],
    min(5, N_TRIALS["gru"]),
    NET_PRUNER,
)


# %%
# ===== [7-10] 단일 모델 Optuna 결과 (fold 3 검증, 행 단위) =====
# '기본 설정' = trial 0(지금까지 쓴 설정), '최적' = Optuna가 찾은 설정.
# 같은 학습·검증 행에서 비교한다.
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
rows_summary, trial_tables = [], {}
for m in MODELS:
    st = STUDIES.get(m) or open_study(m, 1, optuna.pruners.NopPruner())
    done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise RuntimeError(f"{m}: 완료된 trial이 없다. [7-7]~[7-9]를 다시 실행한다.")
    df = pd.DataFrame(
        [
            {
                "trial": t.number,
                "state": t.state.name,
                OBJECTIVE: t.value,
                "AUROC": t.user_attrs.get("val_AUROC"),
                "epoch/반복": t.user_attrs.get("best_epoch", t.user_attrs.get("best_iteration")),
                "초": t.user_attrs.get("sec"),
                **t.params,
            }
            for t in st.trials
        ]
    )
    trial_tables[m] = df
    df.to_csv(OUT / f"tuning_trials_{m}.csv", index=False)
    default = [t for t in done if t.params == DEFAULTS[m]]
    best = st.best_trial
    rows_summary.append(
        {
            "모델": m,
            f"기본 설정 {OBJECTIVE}": default[0].value if default else np.nan,
            f"최적 {OBJECTIVE}": best.value,
            "향상": best.value - default[0].value if default else np.nan,
            "최적 AUROC": best.user_attrs["val_AUROC"],
            "최적 trial": best.number,
            "epoch/반복": best.user_attrs.get("best_epoch", best.user_attrs.get("best_iteration")),
            "완료/가지치기": f"{len(done)}/"
            f"{sum(t.state == optuna.trial.TrialState.PRUNED for t in st.trials)}",
        }
    )
TUNE_SUMMARY = pd.DataFrame(rows_summary).set_index("모델")
TUNE_SUMMARY.to_csv(OUT / "tuning_summary.csv")
print(TUNE_SUMMARY.round(4))
for m in MODELS:
    print(f"\n[{m}] 최적 설정:", STUDIES[m].best_params if m in STUDIES else "")

fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
for ax, m in zip(axes, MODELS, strict=True):
    df = trial_tables[m]
    ok = df[df.state == "COMPLETE"]
    ax.scatter(ok.trial, ok[OBJECTIVE], s=18, label="completed trial")
    ax.plot(ok.trial, ok[OBJECTIVE].cummax(), "r-", label="best so far")
    ax.set(title=f"{m} (fold {TUNE_VAL_FOLD} validation)", xlabel="trial", ylabel=OBJECTIVE)
    ax.legend()
fig.savefig(FIG / "tuning_history.png", dpi=150)
plt.close(fig)


# %%
# ===== [7-11] 5-fold OOF: 최적 설정으로 fold마다 학습 → 빠진 fold 예측 =====
# 지금까지(로지스틱 회귀 ICU-5, LSTM ICU-6)와 같은 dev 5-fold다.
# fold k 모델은 dev에서 fold k를 뺀 환자로 학습하고
# fold k를 예측한다. 스케일링 통계도 fold마다 그 학습 행에서만 구한다(= 데이터 파트 fold{k}.json).
# 학습 길이는 탐색에서 고른 값으로 고정한다 (신경망: 최적 epoch, LightGBM: 최적 반복 수).
# fold 4 모델은 fold 0~3(최종 학습 역할)으로 학습한 모델이므로 그대로 최종 모델이 된다.
OOF_DIR = OUT / "oof"
OOF_DIR.mkdir(exist_ok=True)
DEV_ROWS = np.sort(np.r_[R["train"], R["cal"], R["thr"]])
FOLDS = list(range(5))
FOLD_ROWS = {k: (DEV_ROWS[FOLD[DEV_ROWS] != k], DEV_ROWS[FOLD[DEV_ROWS] == k]) for k in FOLDS}
assert np.array_equal(FOLD_ROWS[4][0], R["train"])  # fold 4 모델 = 최종 학습 역할 그대로
assert np.array_equal(FOLD_ROWS[4][1], np.sort(np.r_[R["cal"], R["thr"]]))
assert not (em.split.iloc[DEV_ROWS] == "test").any()

for k in FOLDS:  # fold별 스케일링 통계 (학습 행만)
    key = f"fold{k}"
    PARAMS[key] = FG.fit_preprocessing(d, FOLD_ROWS[k][0])
    if not SMOKE:  # 데이터 파트가 저장한 fold{k}.json과 같은지 확인
        saved = json.loads(
            Path(INPUT_ROOT, f"artifacts/preprocess_params/fold{k}.json").read_text()
        )
        assert PARAMS[key]["sequence"] == saved["sequence"], f"fold {k} 시퀀스 통계가 다르다"
        for part in ("center", "scale"):
            assert all(
                PARAMS[key]["tabular"][part][c] == saved["tabular"][part][c] for c in T.STATIC
            )
    SC[key] = WINS.scaler(PARAMS[key]["sequence"])
    ST_T[key] = torch.from_numpy(
        pp.apply_tabular(d["static"], PARAMS[key]["tabular"], T.STATIC)
    ).to(DEVICE)
assert PARAMS["fold4"] == PARAMS["final"]
print(
    "[OK] fold별 스케일링 통계" + ("" if SMOKE else " = 데이터 파트 preprocess_params/fold{k}.json")
)


def fold_identity(m, k):
    best = STUDIES[m].best_trial
    return sha(
        {
            "study": study_identity(m),
            "params": best.params,
            "length": best.user_attrs.get("best_epoch", best.user_attrs.get("best_iteration")),
            "train_rows": hashlib.sha256(FOLD_ROWS[k][0].tobytes()).hexdigest(),
            "val_rows": hashlib.sha256(FOLD_ROWS[k][1].tobytes()).hexdigest(),
        }
    )


def model_path(m, k=4):
    return OOF_DIR / (f"{m}_fold{k}.txt" if m == "lgbm" else f"{m}_fold{k}.pt")


def load_model(m, k=4):
    if m == "lgbm":
        return lgb.Booster(model_file=str(model_path(m, k)))
    saved = torch.load(model_path(m, k), map_location=DEVICE, weights_only=False)
    model = make_net(m, saved["hp"])
    model.load_state_dict(saved["state_dict"])
    return model


def predict_model(m, model, rows, key="fold4"):
    if m == "lgbm":
        return model.predict(X_TAB[rows]).astype(np.float64)
    return predict_net(m, model, rows, key)


OOF = {m: np.full(len(DEV_ROWS), np.nan) for m in MODELS}  # DEV_ROWS 순서
OOF_META = {m: {} for m in MODELS}
for m in MODELS:
    best = STUDIES[m].best_trial
    hp = dict(best.params)
    for k in FOLDS:
        tr_k, va_k = FOLD_ROWS[k]
        key = f"fold{k}"
        meta_path, pred_path = OOF_DIR / f"{m}_fold{k}.json", OOF_DIR / f"{m}_fold{k}_pred.parquet"
        ident = fold_identity(m, k)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if meta is not None and meta["identity"] != ident:
            if (OUT / "frozen.json").exists():
                raise RuntimeError(
                    f"{m} fold {k}: 고정된 모델과 지금 최적 설정이 다르다. "
                    "고정 뒤에는 바꾸지 않는다."
                )
            print(f"{m} fold {k}: 최적 설정이 바뀌어 다시 학습한다.")
            meta = None
        if meta is not None:
            if (
                file_sha(model_path(m, k)) != meta["model_sha256"]
                or file_sha(pred_path) != meta["pred_sha256"]
            ):
                raise RuntimeError(f"{m} fold {k}: 저장된 파일이 바뀌었다.")
            pred = pd.read_parquet(pred_path)
        else:
            t0 = time.time()
            if m == "lgbm":
                n_iter = best.user_attrs["best_iteration"]
                dtr = lgbm_dataset(tr_k)
                model = lgb.train(lgbm_params(hp, tr_k), dtr, num_boost_round=n_iter)
                del dtr
                model.save_model(str(model_path(m, k)))
                length = {"best_iteration": n_iter}
            else:
                n_ep = best.user_attrs["best_epoch"]
                model, hist, _, _ = train_net(m, hp, tr_k, key, n_ep)
                torch.save(
                    {"state_dict": model.state_dict(), "hp": hp, "epochs": n_ep, "history": hist},
                    model_path(m, k),
                )
                length = {"epochs": n_ep}
            pred = pd.DataFrame({"row_id": va_k, "p": predict_model(m, model, va_k, key)})
            pred.to_parquet(pred_path, index=False)
            meta = {
                "identity": ident,
                "fold": k,
                "params": hp,
                **length,
                "train_rows": len(tr_k),
                "seconds": round(time.time() - t0, 1),
                "model_sha256": file_sha(model_path(m, k)),
                "pred_sha256": file_sha(pred_path),
            }
            T.atomic_json(meta_path, meta)
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            print(f"{m} fold {k}: 학습·예측 끝 ({meta['seconds']:.0f}초)")
        assert np.array_equal(pred.row_id.to_numpy(), va_k)
        OOF[m][np.searchsorted(DEV_ROWS, va_k)] = pred.p.to_numpy(np.float64)
        OOF_META[m][k] = meta
    assert np.isfinite(OOF[m]).all() and ((OOF[m] >= 0) & (OOF[m] <= 1)).all()
print(
    f"[OK] 3개 모델 5-fold OOF 예측: dev {len(np.unique(PID[DEV_ROWS])):,}명, {len(DEV_ROWS):,}행"
)


# %%
# ===== [7-12] 5-fold 비교: 단일 3개 + 앙상블 2개 (+ 기존 로지스틱·LSTM) =====
# 앙상블 점수 = Σ w_k · logit(p_k), w ≥ 0, Σw = 1. 세 모델의 확률 척도가 달라 logit 공간에서 더한다.
# Optuna 가중치는 교차 적합한다: fold k를 평가할 때 가중치는 나머지 4개 fold의 OOF로만 찾는다.
# 따라서 표의 모든 점수는 그 fold를 학습·가중치 선택에 쓰지 않은 점수다.
# (행 단위, fold별 평균 ± 표준편차)
# 단, 탐색 검증에 fold 3을 썼으므로 fold 3 점수는 조금 낙관적일 수 있다.

CLIP = 1e-7
y_dev, fold_dev = Y[DEV_ROWS], FOLD[DEV_ROWS]
le60_dev = em["ICULOS"].to_numpy()[DEV_ROWS] <= 60


def lg(p):
    return logit(np.clip(p, CLIP, 1 - CLIP))


def blend(weights, logits):
    """weights: {모델: w}, logits: {모델: logit 배열} → 앙상블 확률 expit(Σ w·logit)."""
    return expit(sum(weights[m] * logits[m] for m in MODELS))


def normalize(raw):
    total = sum(raw.values())
    return {m: raw[m] / total for m in MODELS}


def find_weights(logits, y, n_trials):
    """Optuna(TPE)로 Σw·logit의 목표 점수를 최대화하는 가중치. 첫 trial은 같은 가중치."""
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=20)
    )
    study.enqueue_trial({f"w_{m}": 1.0 for m in MODELS})

    def objective(trial):
        raw = {m: trial.suggest_float(f"w_{m}", 0.0, 1.0) for m in MODELS}
        if sum(raw.values()) < 1e-9:
            return 0.0
        return float(score(y, blend(normalize(raw), logits)))

    study.optimize(objective, n_trials=n_trials)
    return normalize({m: study.best_params[f"w_{m}"] for m in MODELS}), study.best_value


L_DEV = {m: lg(OOF[m]) for m in MODELS}
W_EQ = {m: 1 / 3 for m in MODELS}
cv_path = OUT / "cv_weights.json"
cv_ident = sha(
    {
        "oof": {m: [OOF_META[m][k]["pred_sha256"] for k in FOLDS] for m in MODELS},
        "trials": ENSEMBLE_TRIALS,
        "objective": OBJECTIVE,
        "seed": SEED,
    }
)
CV_W = json.loads(cv_path.read_text()) if cv_path.exists() else None
if CV_W is not None and CV_W["identity"] != cv_ident and (OUT / "frozen.json").exists():
    raise RuntimeError("고정 뒤에 OOF 예측이 바뀌었다.")
if CV_W is None or CV_W["identity"] != cv_ident:
    CV_W = {"identity": cv_ident, "weights": {}, "scores": {}}
    for k in FOLDS:
        other = fold_dev != k
        w, s = find_weights({m: L_DEV[m][other] for m in MODELS}, y_dev[other], ENSEMBLE_TRIALS)
        CV_W["weights"][str(k)], CV_W["scores"][str(k)] = w, s
        print(
            f"fold {k} 평가용 가중치 (나머지 fold OOF로 찾음):",
            {m: round(v, 3) for m, v in w.items()},
        )
    T.atomic_json(cv_path, CV_W)

CV_SCORES = {
    "LightGBM": lambda k, idx: OOF["lgbm"][idx],
    "LSTM": lambda k, idx: OOF["lstm"][idx],
    "GRU": lambda k, idx: OOF["gru"][idx],
    "앙상블 (같은 가중치)": lambda k, idx: blend(W_EQ, {m: L_DEV[m][idx] for m in MODELS}),
    "앙상블 (Optuna 가중치)": lambda k, idx: blend(
        CV_W["weights"][str(k)], {m: L_DEV[m][idx] for m in MODELS}
    ),
}
# 지금까지 만든 결과 (같은 dev 5-fold). 파일이 있으면 같은 행에서 다시 계산해 함께 보여 준다.
REFERENCE_OOF = {
    "기존 로지스틱 회귀 (ICU-5)": Path(DATA_ROOT, "results/logreg/oof_predictions.parquet"),
    "기존 LSTM (ICU-6)": Path(DATA_ROOT, "results/lstm/oof_predictions.parquet"),
}
for name, path in REFERENCE_OOF.items():
    if not path.exists():
        print(f"(참고 결과 없음: {path})")
        continue
    ref = pd.read_parquet(path, columns=["row_id", "fold", "p_raw"]).set_index("row_id")
    if not ref.index.is_unique or not np.isin(DEV_ROWS, ref.index).all():
        print(f"(참고 결과의 행이 dev와 맞지 않아 건너뜀: {path})")
        continue
    sub = ref.loc[DEV_ROWS]
    assert (sub["fold"].to_numpy() == fold_dev).all(), f"{name}: fold 배정이 다르다"
    arr = sub["p_raw"].to_numpy(np.float64)
    CV_SCORES[name] = lambda k, idx, arr=arr: arr[idx]

cv_rows = []
for name, fn in CV_SCORES.items():
    for k in FOLDS:
        idx = np.flatnonzero(fold_dev == k)
        p, y = fn(k, idx), y_dev[idx]
        cv_rows.append(
            {
                "모델": name,
                "fold": k,
                "AUROC": roc_auc_score(y, p),
                "AUPRC": average_precision_score(y, p),
                "AUROC (ICULOS≤60)": roc_auc_score(y[le60_dev[idx]], p[le60_dev[idx]]),
            }
        )
CV_FOLDS = pd.DataFrame(cv_rows)
CV_FOLDS.to_csv(OUT / "cv5_by_fold.csv", index=False)
agg = CV_FOLDS.groupby("모델", sort=False)[["AUROC", "AUPRC", "AUROC (ICULOS≤60)"]].agg(
    ["mean", "std"]
)
CV_TABLE = pd.DataFrame(
    {
        c: [f"{agg.loc[n, (c, 'mean')]:.4f} ± {agg.loc[n, (c, 'std')]:.4f}" for n in agg.index]
        for c in ["AUROC", "AUPRC", "AUROC (ICULOS≤60)"]
    },
    index=agg.index,
)
CV_TABLE.to_csv(OUT / "cv5_summary.csv")
print(f"\ndev 5-fold OOF (행 단위, fold별 평균 ± 표준편차, {len(np.unique(PID[DEV_ROWS])):,}명)")
print(CV_TABLE)

fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
names = {
    "앙상블 (같은 가중치)": "Ensemble (equal)",
    "앙상블 (Optuna 가중치)": "Ensemble (Optuna, cross-fit)",
    "기존 로지스틱 회귀 (ICU-5)": "LogReg (ICU-5)",
    "기존 LSTM (ICU-6)": "LSTM (ICU-6)",
}
for name in CV_SCORES:
    sub = CV_FOLDS[CV_FOLDS["모델"] == name]
    ax.plot(sub["fold"], sub["AUROC"], "o-", label=names.get(name, name))
ax.set(xlabel="fold", ylabel="AUROC", title="dev 5-fold OOF AUROC by fold", xticks=FOLDS)
ax.legend(fontsize=8)
fig.savefig(FIG / "cv5_auroc_by_fold.png", dpi=150)
plt.close(fig)


# %%
# ===== [7-13] 최종 앙상블 =====
# 가중치(fold 0~3 OOF) → Platt 보정(보정 그룹) → 경보 임계값(임계값 그룹)
# 최종 모델 = fold 4 모델 (fold 0~3으로 학습, 팀 최종 GRU와 같은 학습 환자·같은 스케일링 통계).
#   가중치: fold 0~3의 OOF (패혈증 환자 약 2,000명). fold 4 라벨은 쓰지 않는다.
#   Platt 보정: 보정 그룹 / 경보 임계값: 임계값 그룹
#   (임계값은 팀 GRU와 같은 '비패혈증 환자 오경보율 10%' 규칙)
#   임계값 그룹의 AUROC·AUPRC·Brier는 가중치·보정에 쓰지 않은 환자의 점수다.
#   경보 지표(감지율·Utility 등)는 이 그룹에 임계값을 맞춘 값이므로 참고용이다.
#   최종 경보 성능은 test로 본다.
KEYS = FG.KEYS
VAL_ROLES = ["cal", "thr"]
P = {m: {role: OOF[m][np.searchsorted(DEV_ROWS, R[role])] for role in VAL_ROLES} for m in MODELS}

# fold 4를 평가할 때 쓴 교차 적합 가중치가 바로 'fold 0~3 OOF로 찾은 가중치'다.
# 같은 값을 최종 가중치로 쓴다.
w_rows = np.searchsorted(DEV_ROWS, R["train"])  # fold 0~3 행
assert np.array_equal(np.flatnonzero(fold_dev != 4), w_rows)
W_OPT = CV_W["weights"]["4"]
w_score = float(score(Y[R["train"]], blend(W_OPT, {m: L_DEV[m][w_rows] for m in MODELS})))
final_oof = cv_ident
print(
    "최종 Optuna 가중치:",
    {m: round(w, 3) for m, w in W_OPT.items()},
    f"| fold 0~3 OOF {OBJECTIVE} {w_score:.4f}",
)

CANDIDATES = {
    "LightGBM": {"lgbm": 1.0, "lstm": 0.0, "gru": 0.0},
    "LSTM": {"lgbm": 0.0, "lstm": 1.0, "gru": 0.0},
    "GRU": {"lgbm": 0.0, "lstm": 0.0, "gru": 1.0},
    "앙상블 (같은 가중치)": W_EQ,
    "앙상블 (Optuna 가중치)": W_OPT,
}


def candidate_prob(name, probs):
    """probs: {모델: 확률 배열} → 후보의 보정 전 확률 (단일 모델은 원래 확률 그대로)."""
    w = CANDIDATES[name]
    used = [m for m in MODELS if w[m] > 0]
    return probs[used[0]] if len(used) == 1 else blend(w, {m: lg(probs[m]) for m in MODELS})


def evaluate(frame, p_raw, p_cal, alarms, baseline):
    y = frame.SepsisLabel.to_numpy()
    le60 = (frame.ICULOS <= 60).to_numpy()
    out = {
        "AUROC": roc_auc_score(y, p_raw),
        "AUPRC": average_precision_score(y, p_raw),
        "AUROC (ICULOS≤60)": roc_auc_score(y[le60], p_raw[le60]),
    }
    prob = CG.probability_metrics(y, p_cal, np.full(len(y), baseline))
    out.update({"Brier (보정)": prob["Brier"], "ECE (보정)": prob["ECE_10_equal_width"]})
    alarm = CG.alarm_metrics(frame, alarms)
    out.update(
        {
            "비패혈증 환자 오경보율": alarm["nonsepsis_patient_false_alarm_rate"],
            "발병 전 0~12h 감지율": alarm["pre_onset_recall_0to12h"],
            "선행 시간 중앙값(h)": alarm["median_lead_hours_detected"],
            "Utility": alarm["Utility"],
            "행 민감도": alarm["row_sensitivity"],
            "행 PPV": alarm["row_PPV"],
        }
    )
    return out


FRAME = {role: em.iloc[R[role]][KEYS].reset_index(drop=True) for role in VAL_ROLES}
frozen_path = OUT / "frozen.json"
if frozen_path.exists():  # 이미 고정됨: 저장된 가중치·보정·임계값을 그대로 쓴다
    FROZEN = json.loads(frozen_path.read_text())
    for name, h in FROZEN["files"].items():
        assert file_sha(OUT / name) == h, f"고정 후 파일이 바뀌었다: {name}"
    assert FROZEN["oof"] == final_oof, "고정 후 OOF 예측이 바뀌었다"
    assert FROZEN["weights"]["optuna"] == W_OPT, "고정 후 가중치가 바뀌었다"
    print("이미 고정된 앙상블·보정·임계값을 쓴다:", frozen_path)
else:
    FROZEN = None

PLATT, THRESH, VAL_ROWS = {}, {}, []
for name in CANDIDATES:
    p_cal_raw = candidate_prob(name, {m: P[m]["cal"] for m in MODELS})
    p_thr_raw = candidate_prob(name, {m: P[m]["thr"] for m in MODELS})
    if FROZEN:
        coef, thr = FROZEN["platt"][name], FROZEN["threshold"][name]
    else:
        fitted = CG.fit_calibrator(p_cal_raw, Y[R["cal"]], "platt")
        coef = {"slope": float(fitted.coef_[0, 0]), "intercept": float(fitted.intercept_[0])}
        if coef["slope"] <= 0:
            raise RuntimeError(f"{name}: Platt 기울기가 0 이하다")
        p_thr_cal = FG.apply_platt(p_thr_raw, coef)
        pt = (
            pd.DataFrame({"pid": PID[R["thr"]], "y": Y[R["thr"]], "p": p_thr_cal})
            .groupby("pid")
            .max()
        )
        thr = CG.rate_threshold(pt.loc[pt.y == 0, "p"], TARGET_FALSE_ALARM)
    PLATT[name], THRESH[name] = coef, thr
    p_thr_cal = FG.apply_platt(p_thr_raw, coef)
    VAL_ROWS.append(
        {
            "모델": name,
            **evaluate(FRAME["thr"], p_thr_raw, p_thr_cal, p_thr_cal >= thr, Y[R["cal"]].mean()),
            "임계값": thr,
        }
    )
VAL_TABLE = pd.DataFrame(VAL_ROWS).set_index("모델")
VAL_TABLE.to_csv(OUT / "validation_threshold_group.csv")
print(
    "\n임계값 그룹 (최종 모델, 가중치·보정에 쓰지 않은 fold 4 환자).",
    "경보 지표는 이 그룹에 임계값을 맞춘 값이다.",
)
print(VAL_TABLE.round(4).T)

if not FROZEN:  # 최종 모델·가중치·보정·임계값을 고정한다. test는 이 파일과 일치할 때만 평가한다.
    weights = {"equal": W_EQ, "optuna": W_OPT, "optuna_fold0to3_oof_score": w_score}
    T.atomic_json(OUT / "ensemble_weights.json", weights)
    T.atomic_json(OUT / "platt.json", PLATT)
    T.atomic_json(
        OUT / "thresholds.json", {"target_false_alarm": TARGET_FALSE_ALARM, "thresholds": THRESH}
    )
    files = [
        "ensemble_weights.json",
        "platt.json",
        "thresholds.json",
        "validation_threshold_group.csv",
    ]
    files += ["cv_weights.json", "cv5_summary.csv", "cv5_by_fold.csv", "row_roles.parquet"]
    files += [f"oof/{m}_fold4.json" for m in MODELS] + [
        str(model_path(m).relative_to(OUT)) for m in MODELS
    ]
    FROZEN = {
        "run": RUN_NAME,
        "smoke": SMOKE,
        "oof": final_oof,
        "weights": weights,
        "platt": PLATT,
        "threshold": THRESH,
        "settings": {"objective": OBJECTIVE, "seed": SEED, "rows": ROWS_SHA, "versions": VERSIONS},
        "files": {f: file_sha(OUT / f) for f in files},
        "test_evaluated": False,
    }
    T.atomic_json(frozen_path, FROZEN)
    print("\n[고정] frozen.json 저장. 이후 test 결과를 보고 이 값들을 바꾸지 않는다.")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
y = Y[R["thr"]]
for name in CANDIDATES:
    p = candidate_prob(name, {m: P[m]["thr"] for m in MODELS})
    prec, rec, _ = precision_recall_curve(y, p)
    label = {
        "앙상블 (같은 가중치)": "Ensemble (equal)",
        "앙상블 (Optuna 가중치)": "Ensemble (Optuna)",
    }.get(name, name)
    axes[0].plot(rec, prec, label=f"{label} AP {average_precision_score(y, p):.3f}")
axes[0].set(
    xlabel="Recall",
    ylabel="Precision",
    title="Threshold group: precision-recall",
    xlim=(0, 1),
    ylim=(0, 0.4),
)
axes[0].legend(fontsize=8)
axes[1].bar(MODELS, [W_OPT[m] for m in MODELS])
axes[1].set(title="Final Optuna weights (logit space, fold 0-3 OOF)", ylim=(0, 1))
fig.savefig(FIG / "validation_pr_weights.png", dpi=150)
plt.close(fig)


# %%
# ===== [7-14] test 최종 평가 (한 번만) =====
# [7-13]에서 고정한 최종 모델(fold 4 모델)·가중치·보정·임계값을 그대로 적용한다.
# 이 결과를 보고 설정을 바꾸거나
# 다른 RUN_NAME으로 다시 골라 같은 test로 보고하지 않는다.
# 참고: 팀 최종 GRU(v2)가 이미 같은 test를 한 번 평가했다.
# 이 스크립트의 GRU 탐색 기본값은 그 v2 설정이다.
TEAM_GRU_TEST = {  # docs/GRU_EXPERIMENT_REPORT_2026-11-12.md (팀 최종 GRU v2 + Platt, 같은 test)
    "AUROC": 0.76316,
    "AUPRC": 0.07032,
    "Brier (보정)": 0.01718,
    "Utility": 0.12141,
    "비패혈증 환자 오경보율": 0.09303,
    "발병 전 0~12h 감지율": 0.27632,
    "선행 시간 중앙값(h)": 8.0,
}
TEST_TABLE = None
if not (OPEN_TEST or SMOKE):
    print("test는 아직 열지 않았다. 위 결과를 확인하고 SEWS_ENS_OPEN_TEST=1로 다시 실행한다.")
else:
    if SMOKE:
        print("SMOKE: 실제 test가 아니라 dev 환자로 만든 가짜 test로 코드만 점검한다.")
    frozen_sha = file_sha(frozen_path)
    for name, h in FROZEN["files"].items():
        assert file_sha(OUT / name) == h, f"고정 후 파일이 바뀌었다: {name}"
    started = OUT / "test_started.json"
    if started.exists():
        if json.loads(started.read_text())["frozen_sha256"] != frozen_sha:
            raise RuntimeError("test가 다른 고정 결과로 이미 열렸다")
    else:
        T.atomic_json(
            started, {"frozen_sha256": frozen_sha, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        )
    summary_path = OUT / "test_summary.json"
    if summary_path.exists():  # 이미 평가함: 저장된 결과만 보여 준다
        TEST_SUMMARY = json.loads(summary_path.read_text())
        assert TEST_SUMMARY["frozen_sha256"] == frozen_sha
        for name, h in TEST_SUMMARY["files"].items():
            assert file_sha(OUT / name) == h, f"test 결과 파일이 바뀌었다: {name}"
        print("이미 평가한 test 결과를 불러온다 (다시 계산하지 않음).")
        TEST_TABLE = pd.read_csv(OUT / "test_metrics.csv").set_index(["범위", "모델"])
    else:
        rows_test = R["test"]
        p_test = {m: predict_model(m, load_model(m), rows_test) for m in MODELS}
        frame = em.iloc[rows_test][KEYS].reset_index(drop=True)
        scopes = {
            "전체": np.ones(len(frame), bool),
            "병원 A": (frame["set"] == "A").to_numpy(),
            "병원 B": (frame["set"] == "B").to_numpy(),
        }
        out_pred = frame.copy()
        results = []
        for name in CANDIDATES:
            p_raw = candidate_prob(name, p_test)
            p_cal = FG.apply_platt(p_raw, FROZEN["platt"][name])
            alarm = p_cal >= FROZEN["threshold"][name]
            short = {
                "앙상블 (같은 가중치)": "ens_equal",
                "앙상블 (Optuna 가중치)": "ens_optuna",
            }.get(name, name)
            out_pred[f"p_raw_{short}"], out_pred[f"p_cal_{short}"] = p_raw, p_cal
            for scope, mask in scopes.items():
                part = frame[mask].reset_index(drop=True)
                if part.empty or part.SepsisLabel.nunique() < 2:
                    continue
                results.append(
                    {
                        "범위": scope,
                        "모델": name,
                        **evaluate(part, p_raw[mask], p_cal[mask], alarm[mask], Y[R["cal"]].mean()),
                    }
                )
        results.append({"범위": "전체", "모델": "참고: 팀 최종 GRU v2 (보고서)", **TEAM_GRU_TEST})
        TEST_TABLE = pd.DataFrame(results).set_index(["범위", "모델"])
        TEST_TABLE.to_csv(OUT / "test_metrics.csv")
        out_pred.to_parquet(OUT / "test_predictions.parquet", index=False)
        files = ["test_metrics.csv", "test_predictions.parquet"]
        TEST_SUMMARY = {
            "complete": True,
            "smoke": SMOKE,
            "frozen_sha256": frozen_sha,
            "patients": int(frame.Patient_ID.nunique()),
            "rows": len(frame),
            "files": {f: file_sha(OUT / f) for f in files},
            "limitations": [
                "단일 분할·단일 seed 결과다. 신뢰구간은 포함하지 않았다.",
                "개발 데이터(fold 0~4)가 탐색·가중치·보정·임계값 선택에 쓰였다.",
                "팀 최종 GRU가 같은 test를 이미 한 번 평가했다.",
                "경보는 매시간 플래그이며 cooldown을 적용하지 않았다.",
            ],
        }
        T.atomic_json(summary_path, TEST_SUMMARY)
    print(f"\ntest {TEST_SUMMARY['patients']:,}명, {TEST_SUMMARY['rows']:,}행")
    for scope in TEST_TABLE.index.get_level_values("범위").unique():
        print(f"\n[{scope}]")
        print(TEST_TABLE.xs(scope, level="범위").round(4).T)


# %%
# ===== [7-15] 결과 정리 =====
SUMMARY = {
    "run": RUN_NAME,
    "smoke": SMOKE,
    "repo_commit": REPO_COMMIT,
    "versions": VERSIONS,
    "objective": OBJECTIVE,
    "roles": role_table.to_dict(orient="index"),
    "tuning": TUNE_SUMMARY.reset_index().to_dict(orient="records"),
    "best_params": {m: STUDIES[m].best_params for m in MODELS},
    "oof_training": {
        m: {k: {a: b for a, b in OOF_META[m][k].items() if a != "identity"} for k in FOLDS}
        for m in MODELS
    },
    "cv5_summary": CV_TABLE.reset_index().to_dict(orient="records"),
    "cv_weights_by_fold": CV_W["weights"],
    "ensemble_weights": {"equal": W_EQ, "optuna": W_OPT},
    "validation_threshold_group": VAL_TABLE.reset_index().to_dict(orient="records"),
    "test": None if TEST_TABLE is None else TEST_TABLE.reset_index().to_dict(orient="records"),
}
T.atomic_json(OUT / "summary.json", FG.json_safe(SUMMARY))
print("저장한 파일:")
for f in sorted(OUT.rglob("*")):
    if f.is_file() and "_local" not in f.parts:
        print(f"  {f.relative_to(OUT)}  ({f.stat().st_size / 1e6:.2f} MB)")
