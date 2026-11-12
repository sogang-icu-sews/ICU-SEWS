"""SEWS 특징 모듈: 정제 + 24시간 창 특징. 학습(노트북)과 API가 같은 파일을 import 한다.

- clean(df, cfg)                  : 고정 규칙 정제 (데이터에서 학습하는 값 없음)
- build_features(df, cfg)         : 학습용. 전체 행을 한 번에 계산
                                    (환자·시간 순 정렬, 1시간 간격 필요)
- build_features_window(win, cfg) : 참조용(API용). 환자 1명의 최근 최대 24행을 받아
                                    마지막 행의 특징 계산
- feature_names(cfg), feature_registry(cfg)

두 계산 방식은 일부러 서로 다른 코드로 작성했다. 3단계 검사에서 두 결과가 같으면
(1) 학습 특징이 최근 24시간만 쓰고 (2) 학습과 API가 같은 값을 만든다는 것이
함께 확인된다.
창 앞쪽을 값이 전부 비어 있는 행으로 채워도 특징은 바뀌지 않는다
(통계는 측정된 값만 사용).
"""

import re
import warnings

import numpy as np
import pandas as pd

WIN = 24
UNITS = ["MICU", "SICU", "Unknown"]
SOFA_COMPONENTS = ["coagulation", "liver", "renal", "cardiovascular"]


# ------------------------------------------------------------ 정제
def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply fixed cleaning rules from config. Returns (cleaned copy, change log)."""
    out = df.copy()
    log = []
    thr = cfg["transforms"]["calcium_ionized_threshold"]
    ion = out["Calcium"] < thr  # 이온화칼슘(다른 검사) → 별도 컬럼
    out["Calcium_ionized"] = out["Calcium"].where(ion)
    out.loc[ion, "Calcium"] = np.nan
    log.append(("Calcium<3 → Calcium_ionized", int(ion.sum())))
    for v, (lo, hi) in cfg["plausible_ranges"].items():  # 생리적 범위 밖 → NaN (clip 아님)
        if v not in out.columns:
            continue
        bad = out[v].notna() & ((out[v] < lo) | (out[v] > hi))
        out.loc[bad, v] = np.nan
        log.append((f"{v} out of [{lo}, {hi}]", int(bad.sum())))
    ar = cfg["transforms"]["age_recode"]
    old = out["Age"] == ar["from"]
    out.loc[old, "Age"] = ar["to"]
    log.append((f"Age {ar['from']}→{ar['to']}", int(old.sum())))
    unit = np.select([out["Unit1"] == 1, out["Unit2"] == 1], ["MICU", "SICU"], "Unknown")
    out["Unit"] = pd.Categorical(unit, categories=UNITS)
    log.append(("Unit = Unknown", int((unit == "Unknown").sum())))
    return out, pd.DataFrame(log, columns=["rule", "rows_changed"])


# ------------------------------------------------------------ 이름·목록
def _vars(cfg):
    fv = cfg["feature_vars"]
    return fv["vitals_window"], fv["vital_stats"], fv["labs"], fv["lab_stats"]


def feature_registry(cfg: dict) -> list[dict]:
    """One dict per feature: name, group, source, stat, model_input, optional."""
    vitals, vstats, labs, lstats = _vars(cfg)
    reg = []

    def add(name, group, source, stat, optional=False):
        reg.append(
            {
                "name": name,
                "group": group,
                "source": source,
                "stat": stat,
                "model_input": True,
                "optional": optional,
            }
        )

    for v in vitals:
        for s in vstats:
            add(f"{v}_{s}", "vital_window", [v], s)
    for v in labs:
        for s in lstats:
            add(f"{v}_{s}", "lab", [v], s)
    for v in cfg["derived_features"]["deviation"]:
        add(f"{v}_dev", "derived", [v], "abs_deviation")
    for name, (a, b) in cfg["derived_features"]["ratio"].items():
        add(f"{name}_ratio", "derived", [a, b], "ratio_last")
    for name, (a, b) in cfg["derived_features"]["optional"].items():
        add(name, "derived", [a, b], "ratio_last", optional=True)
    for g, members in cfg["measurement_indicator_groups"].items():
        add(f"meas_{g}_n24h", "measurement", members, "n_hours_measured")
        add(f"meas_{g}_since_h", "measurement", members, "hours_since_last")
    comps = cfg["sofa"]["components"]
    for c in SOFA_COMPONENTS:
        add(f"sofa_{c}", "sofa", [comps[c]["var"]], comps[c]["type"])
    add(
        "sofa_delta",
        "sofa",
        [comps[c]["var"] for c in SOFA_COMPONENTS],
        "per_component_sum",
    )
    add("qsofa_resp", "qsofa", ["Resp"], "resp_ge")
    add("qsofa_sbp", "qsofa", ["SBP"], "sbp_le")
    add("qsofa_partial", "qsofa", ["Resp", "SBP"], "sum_available")
    add("Age", "static", ["Age"], "value")
    add("Gender", "static", ["Gender"], "value")
    for u in UNITS:
        add(f"Unit_{u}", "static", ["Unit1", "Unit2"], "one_hot")
    return reg


def feature_names(cfg: dict, include_optional: bool = True) -> list[str]:
    return [r["name"] for r in feature_registry(cfg) if include_optional or not r["optional"]]


def banned_name_hits(names, cfg: dict) -> list[str]:
    """Feature names that match a banned column or name pattern."""
    b = cfg["banned_time_features"]
    bad_cols = set(b["columns"]) | set(cfg["eval_only_columns"])
    return [n for n in names if n in bad_cols or any(re.search(p, n) for p in b["name_patterns"])]


# ------------------------------------------------------------ 점수 규칙
def _score(x, comp: dict):
    """SOFA component score for array x (NaN stays NaN)."""
    x = np.asarray(x, dtype=float)
    if comp["type"] == "partial_0_or_1":
        s = (x < comp["threshold_lt"]).astype(float)
    elif "cuts_desc" in comp:
        s = (x[..., None] < np.asarray(comp["cuts_desc"])).sum(-1).astype(float)
    else:
        s = (x[..., None] >= np.asarray(comp["cuts_asc"])).sum(-1).astype(float)
    return np.where(np.isnan(x), np.nan, s)


# ------------------------------------------------------------ 학습용 (전체 행)
def _lag_matrix(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """(n, WIN) matrix: column j = value at offset j-23 (same patient, else NaN)."""
    n = len(x)
    M = np.full((n, WIN), np.nan)
    for k in range(WIN):
        col = WIN - 1 - k
        if k == 0:
            M[:, col] = x
        else:
            M[k:, col] = x[:-k]
            M[pos < k, col] = np.nan
    return M


def _last_idx(obs: np.ndarray) -> np.ndarray:
    """Column index of the last True per row (-1 if none)."""
    idx = obs.shape[1] - 1 - np.argmax(obs[:, ::-1], axis=1)
    return np.where(obs.any(axis=1), idx, -1)


def _take(M, idx):
    r = np.arange(len(M))
    return np.where(idx >= 0, M[r, np.clip(idx, 0, None)], np.nan)


def _vital_stats(M: np.ndarray, cfg: dict) -> dict:
    w = cfg["window"]
    obs = ~np.isnan(M)
    n = obs.sum(axis=1)
    has1, has2, has3 = n >= 1, n >= w["min_obs_dispersion"], n >= w["min_obs_skew"]
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        last = _take(M, _last_idx(obs))
        mean = np.where(has1, np.nansum(M, axis=1) / np.maximum(n, 1), np.nan)
        S = np.sort(M, axis=1)  # NaN은 뒤로 감 → 앞 n개가 측정값
        r = np.arange(len(M))

        def quant(
            p,
        ):  # np.percentile(method='linear')와 같은 정의, 행마다 측정값 n개 기준
            at = (n - 1) * p
            lo, hi = np.floor(at).astype(int), np.ceil(at).astype(int)
            vlo, vhi = S[r, np.clip(lo, 0, None)], S[r, np.clip(hi, 0, None)]
            return np.where(has1, vlo + (vhi - vlo) * (at - lo), np.nan)

        mn, mx = S[:, 0], S[r, np.clip(n - 1, 0, None)]
        const = mx == mn
        med, q25, q75 = quant(0.5), quant(0.25), quant(0.75)
        d = M - mean[:, None]
        ss = np.nansum(d**2, axis=1)
        std = np.where(const, 0.0, np.sqrt(ss / np.maximum(n - 1, 1)))
        m2, m3 = ss / np.maximum(n, 1), np.nansum(d**3, axis=1) / np.maximum(n, 1)
        skew = np.where(const, 0.0, m3 / m2**1.5)
        cv = np.where(mean != 0, std / np.abs(mean), np.nan)
        x = np.where(obs, np.arange(WIN, dtype=float), np.nan)
        dx = x - (np.nansum(x, axis=1) / np.maximum(n, 1))[:, None]
        slope = np.where(const, 0.0, np.nansum(dx * d, axis=1) / np.nansum(dx**2, axis=1))
        asof6 = _take(M, _last_idx(obs[:, : WIN - 6]))  # t-6 시점에 알고 있던 마지막 값
    return {
        "last": last,
        "mean": mean,
        "median": med,
        "std": np.where(has2, std, np.nan),
        "min": np.where(has1, mn, np.nan),
        "max": np.where(has1, mx, np.nan),
        "iqr": np.where(has2, np.where(const, 0.0, q75 - q25), np.nan),
        "skew": np.where(has3, skew, np.nan),
        "cv": np.where(has2, cv, np.nan),
        "slope": np.where(has2, slope, np.nan),
        "delta6h": np.where(has2, last - asof6, np.nan),
    }


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Vectorized features for every row (cleaned, sorted by Patient_ID, ICULOS)."""
    pid = df["Patient_ID"].to_numpy()
    assert (np.diff(pid) >= 0).all(), "Patient_ID 순으로 정렬 필요"
    same = pid[1:] == pid[:-1]
    assert (np.diff(df["ICULOS"].to_numpy())[same] == 1).all(), (
        "환자 안에서 ICULOS가 1시간 간격이어야 함"
    )
    pos = df.groupby("Patient_ID", sort=False).cumcount().to_numpy()
    vitals, vstats, labs, _ = _vars(cfg)
    F = {}
    lastv = {}
    for v in vitals:
        st = _vital_stats(_lag_matrix(df[v].to_numpy(float), pos), cfg)
        for s in vstats:
            F[f"{v}_{s}"] = st[s]
        lastv[v] = st["last"]
    comps = cfg["sofa"]["components"]
    sofa_vars = {comps[c]["var"] for c in SOFA_COMPONENTS}
    sofa_min = {}
    ind_vars = {m for g in cfg["measurement_indicator_groups"].values() for m in g}
    obs_store = {}
    for v in labs:
        M = _lag_matrix(df[v].to_numpy(float), pos)
        obs = ~np.isnan(M)
        i_last = _last_idx(obs)
        last = _take(M, i_last)
        obs_prev = obs.copy()
        obs_prev[np.arange(len(M)), np.clip(i_last, 0, None)] = False
        prev = _take(M, _last_idx(obs_prev))
        F[f"{v}_last"] = last
        F[f"{v}_delta_prev"] = last - prev  # 측정 2회 미만이면 NaN
        lastv[v] = last
        if v in ind_vars:
            obs_store[v] = obs
        if v in sofa_vars:
            for c in SOFA_COMPONENTS:
                if comps[c]["var"] == v:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        sofa_min[c] = np.nanmin(_score(M, comps[c]), axis=1)
    for c in SOFA_COMPONENTS:  # MAP처럼 활력징후인 항목
        if c not in sofa_min:
            M = _lag_matrix(df[comps[c]["var"]].to_numpy(float), pos)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sofa_min[c] = np.nanmin(_score(M, comps[c]), axis=1)

    for v, center in cfg["derived_features"]["deviation"].items():
        F[f"{v}_dev"] = np.abs(lastv[v] - center)
    for name, (a, b) in cfg["derived_features"]["ratio"].items():
        F[f"{name}_ratio"] = lastv[a] / lastv[b]
    for name, (a, b) in cfg["derived_features"]["optional"].items():
        F[name] = lastv[a] / lastv[b]
    for g, members in cfg["measurement_indicator_groups"].items():
        G = np.zeros((len(df), WIN), dtype=bool)
        for m_ in members:
            G |= obs_store[m_]
        i_last = _last_idx(G)
        F[f"meas_{g}_n24h"] = G.sum(axis=1).astype(float)
        F[f"meas_{g}_since_h"] = np.where(
            i_last >= 0, WIN - 1 - i_last, cfg["window"]["since_cap_h"]
        ).astype(float)
    deltas = []
    for c in SOFA_COMPONENTS:
        cur = _score(lastv[comps[c]["var"]], comps[c])
        F[f"sofa_{c}"] = cur
        deltas.append(cur - sofa_min[c])
    F["sofa_delta"] = pd.DataFrame(np.column_stack(deltas)).sum(axis=1, min_count=1).to_numpy()
    q = cfg["qsofa"]
    F["qsofa_resp"] = np.where(
        np.isnan(lastv["Resp"]), np.nan, (lastv["Resp"] >= q["resp_ge"]).astype(float)
    )
    F["qsofa_sbp"] = np.where(
        np.isnan(lastv["SBP"]), np.nan, (lastv["SBP"] <= q["sbp_le"]).astype(float)
    )
    F["qsofa_partial"] = (
        pd.DataFrame({"a": F["qsofa_resp"], "b": F["qsofa_sbp"]})
        .sum(axis=1, min_count=1)
        .to_numpy()
    )
    F["Age"] = df["Age"].to_numpy(float)
    F["Gender"] = df["Gender"].to_numpy(float)
    unit = df["Unit"].astype(str).to_numpy()
    for u in UNITS:
        F[f"Unit_{u}"] = (unit == u).astype(float)
    out = pd.DataFrame(F, index=df.index)
    return out[feature_names(cfg)]


# ------------------------------------------------ 참조용 (환자 1명, 24시간 창)
def _ref_vital(vals: np.ndarray, hours: np.ndarray, cfg: dict) -> dict:
    w = cfg["window"]
    ok = ~np.isnan(vals)
    v, h = vals[ok], hours[ok]
    out = dict.fromkeys(cfg["feature_vars"]["vital_stats"], np.nan)
    if len(v) == 0:
        return out
    const = v.max() == v.min()
    out.update(last=v[-1], mean=v.mean(), median=float(np.median(v)), min=v.min(), max=v.max())
    if len(v) >= w["min_obs_dispersion"]:
        sd = 0.0 if const else float(np.std(v, ddof=1))
        out["std"] = sd
        out["iqr"] = 0.0 if const else float(np.percentile(v, 75) - np.percentile(v, 25))
        out["cv"] = sd / abs(v.mean()) if v.mean() != 0 else np.nan
        out["slope"] = 0.0 if const else float(np.polyfit(h, v, 1)[0])
        before = v[h <= -6]  # t-6 시점에 알고 있던 값들
        out["delta6h"] = v[-1] - before[-1] if len(before) else np.nan
    if len(v) >= w["min_obs_skew"]:
        out["skew"] = (
            0.0 if const else float(np.mean(((v - v.mean()) / np.std(v)) ** 3))
        )  # 모집단 기준 g1
    return out


def build_features_window(win: pd.DataFrame, cfg: dict) -> pd.Series:
    """Features at the last row of one cleaned window (only the last 24 rows used)."""
    win = win.tail(WIN)
    hours = np.arange(-len(win) + 1, 1)  # 마지막 행 = 0시
    vitals, vstats, labs, _ = _vars(cfg)
    f, last = {}, {}
    for v in vitals:
        st = _ref_vital(win[v].to_numpy(float), hours, cfg)
        for s in vstats:
            f[f"{v}_{s}"] = st[s]
        last[v] = st["last"]
    for v in labs:
        x = win[v].dropna().to_numpy(float)
        f[f"{v}_last"] = x[-1] if len(x) else np.nan
        f[f"{v}_delta_prev"] = x[-1] - x[-2] if len(x) >= 2 else np.nan
        last[v] = f[f"{v}_last"]
    for v, center in cfg["derived_features"]["deviation"].items():
        f[f"{v}_dev"] = abs(last[v] - center)
    for name, (a, b) in cfg["derived_features"]["ratio"].items():
        f[f"{name}_ratio"] = last[a] / last[b]
    for name, (a, b) in cfg["derived_features"]["optional"].items():
        f[name] = last[a] / last[b]
    for g, members in cfg["measurement_indicator_groups"].items():
        measured_hours = hours[win[members].notna().any(axis=1).to_numpy()]
        f[f"meas_{g}_n24h"] = float(len(measured_hours))
        f[f"meas_{g}_since_h"] = (
            float(-measured_hours[-1])
            if len(measured_hours)
            else float(cfg["window"]["since_cap_h"])
        )
    comps = cfg["sofa"]["components"]
    deltas = []
    for c in SOFA_COMPONENTS:
        comp = comps[c]
        cur = float(_score(last[comp["var"]], comp))
        f[f"sofa_{c}"] = cur
        measured = win[comp["var"]].dropna().to_numpy(float)
        if len(measured):
            deltas.append(cur - min(float(_score(m, comp)) for m in measured))
    f["sofa_delta"] = float(sum(deltas)) if deltas else np.nan
    q = cfg["qsofa"]
    f["qsofa_resp"] = np.nan if np.isnan(last["Resp"]) else float(last["Resp"] >= q["resp_ge"])
    f["qsofa_sbp"] = np.nan if np.isnan(last["SBP"]) else float(last["SBP"] <= q["sbp_le"])
    parts = [p for p in (f["qsofa_resp"], f["qsofa_sbp"]) if not np.isnan(p)]
    f["qsofa_partial"] = float(sum(parts)) if parts else np.nan
    cur_row = win.iloc[-1]
    f["Age"] = float(cur_row["Age"])
    f["Gender"] = float(cur_row["Gender"])
    for u in UNITS:
        f[f"Unit_{u}"] = float(str(cur_row["Unit"]) == u)
    return pd.Series(f)[feature_names(cfg)]
