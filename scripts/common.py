from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize

SEED = 123
CV_SEED = 20260903
RESAMPLE_SEED = 20260907
LAMBDA_GRID = np.round(np.arange(0.0, 1.0001, 0.025), 3)

PEDSQL_ORDINAL = [
    "pedsql_core_score_tot_scaled_label",
    "pedsql_core_score_phys_scaled_label",
    "pedsql_core_score_emo_scaled_label",
    "pedsql_core_score_soc_scaled_label",
    "pedsql_core_score_sch_scaled_label",
    "pedsql_core_score_psy_scaled_label",
    "pedsql_cardiac_score_treat_scaled_label",
    "pedsql_cardiac_score_ppa_scaled_label",
    "pedsql_cardiac_score_ta_scaled_label",
    "pedsql_cardiac_score_cp_scaled_label",
    "pedsql_cardiac_score_com_scaled_label",
]
DASS_ORDINAL = ["dass_dep_class", "dass_anx_class", "dass_stress_class"]
PRO_ORDINAL = PEDSQL_ORDINAL + DASS_ORDINAL

PEDSQL_CONTINUOUS = [
    "pedsql_core_score_tot_scaled",
    "pedsql_core_score_phys_scaled",
    "pedsql_core_score_emo_scaled",
    "pedsql_core_score_soc_scaled",
    "pedsql_core_score_sch_scaled",
    "pedsql_core_score_psy_scaled",
    "pedsql_cardiac_score_treat_scaled",
    "pedsql_cardiac_score_ppa_scaled",
    "pedsql_cardiac_score_ta_scaled",
    "pedsql_cardiac_score_cp_scaled",
    "pedsql_cardiac_score_com_scaled",
]
DASS_CONTINUOUS = ["dass_dep_score", "dass_anx_score", "dass_stress_score"]
PRO_CONTINUOUS = PEDSQL_CONTINUOUS + DASS_CONTINUOUS

PEDSQL_ORDER = {"very low": 0.0, "low": 1.0, "moderate": 2.0, "high": 3.0, "very high": 4.0}
DASS_ORDER = {"normal": 0.0, "mild": 1.0, "moderate": 2.0, "severe": 3.0, "extremely severe": 4.0}

DEMOCLIN_CONTEXT = [
    "sex", "age_band", "remoteness", "num_meds_label", "num_comorb_label",
    "diabetes_label", "hbp_label", "asthma_label",
]
CHD_CONTEXT = ["chd_complexity", "chd_category"]
CONTEXT_FIELDS = DEMOCLIN_CONTEXT + CHD_CONTEXT

BASE_NUMERIC = ["age"]
BASE_CATEGORICAL = ["sex", "remoteness", "chd_complexity", "chd_category"]

CONTINUOUS_OUTCOMES = {
    "Cardiac anxiety": "caq_tot_score",
    "Resilience": "cdrisc_score",
    "Trauma symptoms": "iesr_mscore",
    "Sleep disturbance": "promis_tscore",
    "6-minute walk distance": "smwt_distance",
}
BINARY_OUTCOMES = {
    "NYHA class > I": "nyha_binary",
    "Negative employment impact": "employment_impact_binary",
    "Employment changes": "employment_change_binary",
    "Adjusted educational assessment": "education_assessment_binary",
    "Education changes": "education_change_binary",
    "Fell behind in education": "education_behind_binary",
    "Negative education impact": "education_impact_binary",
    "Learning disruption": "education_learning_binary",
    "Reduced social participation": "participation_binary",
    "Emergency-department use": "ed_use_binary",
    "Multiple-specialist burden": "specialist_burden_binary",
}
CONVERGENT_OUTCOMES = {"Cardiac anxiety", "Resilience", "Trauma symptoms", "Sleep disturbance"}

RAW_BINARY_COLUMNS = [
    "nyha_class", "rdq_emp_impact", "rdq_emp_changes", "rdq_edu_assess", "rdq_edu_change",
    "rdq_edu_behind", "rdq_edu_impact", "rdq_edu_learn", "rdq_involv",
    "rdq_ed_use_bucket", "rdq_dr_num_bucket",
]

KEEP_AS_OWN_FAMILY = {
    "Tetralogy of Fallot",
    "Aortic Valve Disorders",
    "Coarctation of the Aorta",
    "Ventricular Septal Defect",
    "Transposition of the Great Arteries",
    "Fontan",
    "Pulmonary Valve Disorders",
    "Univentricular Heart (not Fontan)",
    "Atrial Septal Defect",
    "Systemic Right Ventricle – TGA After Atrial Switch",
}
OTHER_RARE_FAMILY = "Other / rare"


def normalise_ids(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def canonical_text(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.casefold()


def one_hot_encoder(*, drop: str | None = None, dense: bool = False) -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore", "drop": drop}
    try:
        return OneHotEncoder(sparse_output=not dense, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=not dense, **kwargs)


def write_json(obj: dict, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> pd.DataFrame:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def map_ordinal_checked(series: pd.Series, mapping: dict[str, float], column: str) -> pd.Series:
    cleaned = canonical_text(series)
    mapped = cleaned.map(mapping).astype(float)
    bad = series.notna() & mapped.isna()
    if bad.any():
        values = sorted(series.loc[bad].astype(str).str.strip().unique().tolist())
        raise ValueError(f"Unrecognised ordinal level(s) in {column}: {values}")
    return mapped


def ordinal_pro_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in PEDSQL_ORDINAL:
        out[c] = map_ordinal_checked(df[c], PEDSQL_ORDER, c)
    for c in DASS_ORDINAL:
        out[c] = map_ordinal_checked(df[c], DASS_ORDER, c)
    return out[PRO_ORDINAL]


def derive_binary_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def from_levels(series, positive, negative):
        s = canonical_text(series)
        pos = {str(x).casefold() for x in positive}; neg = {str(x).casefold() for x in negative}
        y = pd.Series(np.nan, index=series.index, dtype=float)
        y.loc[s.isin(pos)] = 1.0; y.loc[s.isin(neg)] = 0.0
        return y

    if "nyha_class" in out:
        s = out["nyha_class"].astype("string").str.strip()
        out["nyha_binary"] = np.nan
        observed = s.notna()
        out.loc[observed & s.eq("SA-NYHA Class I"), "nyha_binary"] = 0.0
        out.loc[observed & ~s.eq("SA-NYHA Class I"), "nyha_binary"] = 1.0

    impact_pos = {"Negative impact", "Very negative impact"}
    impact_neg = {"No impact", "Positive impact", "Very positive impact"}
    yes, no = {"Yes"}, {"No"}
    recodes = {
        "employment_impact_binary": ("rdq_emp_impact", impact_pos, impact_neg),
        "employment_change_binary": ("rdq_emp_changes", yes, no),
        "education_assessment_binary": ("rdq_edu_assess", yes, no),
        "education_change_binary": ("rdq_edu_change", yes, no),
        "education_behind_binary": ("rdq_edu_behind", yes, no),
        "education_impact_binary": ("rdq_edu_impact", impact_pos, impact_neg),
        "education_learning_binary": ("rdq_edu_learn", yes, no),
        "participation_binary": ("rdq_involv", yes, no),
        "ed_use_binary": ("rdq_ed_use_bucket", {"occasional", "frequent"}, {"none"}),
        "specialist_burden_binary": ("rdq_dr_num_bucket", {"2-3", "4+"}, {"0-1"}),
    }
    for dest, (src, pos, neg) in recodes.items():
        if src in out:
            out[dest] = from_levels(out[src], pos, neg)
    return out


def outcome_specs() -> list[tuple[str, str, bool, str]]:
    rows = []
    for name, col in CONTINUOUS_OUTCOMES.items():
        rows.append((name, col, False, "construct-convergent" if name in CONVERGENT_OUTCOMES else "construct-distinct"))
    for name, col in BINARY_OUTCOMES.items():
        rows.append((name, col, True, "construct-distinct"))
    return rows


def add_lesion_family_11(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    exact = out["chd_category"].astype("string").str.strip()
    if exact.isna().any() or exact.eq("").any():
        raise ValueError("chd_category must be observed for lesion-family construction")
    out["lesion_family_11"] = np.where(exact.isin(KEEP_AS_OWN_FAMILY), exact, OTHER_RARE_FAMILY)
    return out


def l2_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected a matrix; found shape {x.shape}")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(~np.isfinite(n)) or np.any(n <= 0):
        raise ValueError("Zero or non-finite row norm")
    return x / n


def combine_views(context: np.ndarray, pro: np.ndarray, lam: float) -> np.ndarray:
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must lie in [0,1]")
    c, p = l2_rows(context), l2_rows(pro)
    if len(c) != len(p):
        raise ValueError("Context and PRO views have different row counts")
    # Manuscript notation: z_i(lambda) = [sqrt(1-lambda)c_i, sqrt(lambda)p_i].
    return np.hstack([np.sqrt(1.0 - lam) * c, np.sqrt(lam) * p]).astype(np.float32)


def mean_pairwise_sq_distance_unit(x: np.ndarray) -> float:
    x = l2_rows(x)
    n = len(x)
    mean_vec = x.mean(axis=0)
    return float(2.0 * n / (n - 1.0) * (1.0 - float(mean_vec @ mean_vec)))


def lesion_coherence(labels: np.ndarray, lesion: Sequence) -> float:
    labels = np.asarray(labels, dtype=int)
    lesion = pd.Series(lesion, dtype="string").fillna("__MISSING__").astype(str).to_numpy()
    assigned = labels >= 0
    if not assigned.any(): return np.nan
    work = pd.DataFrame({"cluster": labels[assigned], "lesion": lesion[assigned]})
    dominant = work.groupby("cluster")["lesion"].value_counts().groupby(level=0).max()
    return float(dominant.sum() / len(work))


def diagnosis_fragmentation(labels: np.ndarray, lesion: Sequence) -> float:
    labels = np.asarray(labels, dtype=int)
    lesion = pd.Series(lesion, dtype="string").fillna("__MISSING__").astype(str).to_numpy()
    assigned = labels >= 0
    if not assigned.any(): return np.nan
    la, ll = labels[assigned], lesion[assigned]
    clusters = sorted(set(la.tolist())); total = len(la); score = 0.0
    for category in sorted(set(ll)):
        mask = ll == category; n = int(mask.sum())
        meaningful = sum(np.sum(mask & (la == k)) / n >= 0.10 for k in clusters)
        score += (n / total) * max(0, meaningful - 1)
    return float(score)


def fit_hdbscan(x: np.ndarray, *, min_cluster_size=40, min_samples=10, method="leaf", metric="euclidean"):
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError("Install the hdbscan package to run density-based clustering") from exc
    model = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size), min_samples=int(min_samples), metric=metric,
        cluster_selection_method=method, gen_min_span_tree=True, core_dist_n_jobs=-1,
    )
    labels = model.fit_predict(np.asarray(x, dtype=np.float32) if metric != "precomputed" else np.asarray(x, dtype=float)).astype(int)
    probs = np.asarray(model.probabilities_, dtype=float)
    persistence = np.asarray(model.cluster_persistence_, dtype=float)
    return model, labels, probs, persistence


def weighted_cluster_persistence(labels: np.ndarray, persistence: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int); persistence = np.asarray(persistence, dtype=float)
    clusters = sorted(int(c) for c in np.unique(labels) if c >= 0)
    if not clusters or len(clusters) != len(persistence): return np.nan
    sizes = np.asarray([(labels == c).sum() for c in clusters], dtype=float)
    return float(np.average(persistence, weights=sizes))


def cluster_metrics(labels: np.ndarray, persistence: np.ndarray, lesions: Sequence) -> dict:
    labels = np.asarray(labels, dtype=int); assigned = labels >= 0
    clusters = sorted(int(c) for c in np.unique(labels) if c >= 0)
    sizes = np.asarray([(labels == c).sum() for c in clusters], dtype=int)
    return {
        "n_clusters": len(clusters), "n_assigned": int(assigned.sum()), "n_noise": int((labels == -1).sum()),
        "assigned_fraction": float(assigned.mean()), "noise_fraction": float((labels == -1).mean()),
        "cluster_size_min": int(sizes.min()) if len(sizes) else 0,
        "cluster_size_median": float(np.median(sizes)) if len(sizes) else np.nan,
        "cluster_size_max": int(sizes.max()) if len(sizes) else 0,
        "weighted_cluster_persistence": weighted_cluster_persistence(labels, persistence),
        "lesion_coherence": lesion_coherence(labels, lesions),
        "diagnosis_fragmentation": diagnosis_fragmentation(labels, lesions),
    }


def lesion_stratified_resamples(lesions: Sequence, n_resamples: int, fraction: float = 0.80, seed: int = RESAMPLE_SEED) -> list[np.ndarray]:
    y = pd.Series(lesions, dtype="string").fillna("__MISSING__").to_numpy()
    groups = {key: np.where(y == key)[0] for key in sorted(np.unique(y))}
    out = []
    for r in range(n_resamples):
        rng = np.random.default_rng(seed + r); chosen = []
        for idx in groups.values():
            n_take = min(len(idx), max(2, int(round(fraction * len(idx)))))
            chosen.extend(rng.choice(idx, size=n_take, replace=False).tolist())
        out.append(np.asarray(sorted(chosen), dtype=int))
    return out


def partition_agreement(ref: np.ndarray, other: np.ndarray) -> dict:
    ref, other = np.asarray(ref, int), np.asarray(other, int)
    both = (ref >= 0) & (other >= 0)
    return {
        "ari_all": float(adjusted_rand_score(ref, other)),
        "nmi_all": float(normalized_mutual_info_score(ref, other)),
        "assigned_noise_agreement": float(np.mean((ref >= 0) == (other >= 0))),
        "n_both_assigned": int(both.sum()),
        "ari_both_assigned": float(adjusted_rand_score(ref[both], other[both])) if both.sum() >= 2 else np.nan,
        "nmi_both_assigned": float(normalized_mutual_info_score(ref[both], other[both])) if both.sum() >= 2 else np.nan,
    }


def minmax(x: Sequence[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float); lo, hi = float(np.min(x)), float(np.max(x))
    return np.ones_like(x) if np.isclose(lo, hi) else (x - lo) / (hi - lo)


def pareto_mask(points: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=float); keep = np.ones(len(x), dtype=bool); tol = 1e-12
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j: continue
            if np.all(x[j] >= x[i] - tol) and np.any(x[j] > x[i] + tol):
                keep[i] = False; break
    return keep


def select_lambda(metrics: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    d = metrics.copy().sort_values("lambda", kind="mergesort").reset_index(drop=True)
    required = ["assigned_fraction", "stability_ari_all_mean", "weighted_cluster_persistence"]
    d["admissible"] = (d["n_clusters"] >= 2) & np.isfinite(d[required]).all(axis=1)
    d["pareto_optimal"] = False; d["ideal_distance"] = np.nan
    a = d[d["admissible"]].copy()
    if a.empty: raise RuntimeError("No admissible lambda values")
    for m in required:
        a[m + "_norm"] = minmax(a[m])
    pts = a[[m + "_norm" for m in required]].to_numpy(float)
    a["pareto_optimal"] = pareto_mask(pts)
    a["ideal_distance"] = np.sqrt(np.sum((1.0 - pts) ** 2, axis=1))
    for c in [m + "_norm" for m in required] + ["pareto_optimal", "ideal_distance"]:
        d.loc[a.index, c] = a[c]
    candidates = a[a["pareto_optimal"]].sort_values(
        ["ideal_distance", "stability_ari_all_mean", "assigned_fraction", "weighted_cluster_persistence", "lambda"],
        ascending=[True, False, False, False, True], kind="mergesort",
    )
    selected = float(candidates.iloc[0]["lambda"])
    d["selected"] = np.isclose(d["lambda"], selected)
    return selected, d


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float); out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any(): return out
    vals = p[ok]; order = np.argsort(vals); ranked = vals[order]
    adj = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]; adj = np.minimum(adj, 1.0)
    restored = np.empty_like(adj); restored[order] = adj; out[np.where(ok)[0]] = restored
    return out


def fit_tabular_block(df: pd.DataFrame, *, categorical: list[str], include_pro: bool, max_components: int = 50):
    work = df.copy(); transformers = []
    if categorical:
        for c in categorical:
            s = work[c].astype("string").str.strip(); work[c] = s.astype(object).where(s.notna(), np.nan)
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot_encoder())]), categorical))
    pro_cols = []
    if include_pro:
        pro = ordinal_pro_frame(work)
        for c in PRO_ORDINAL:
            nc = "__PRO__" + c; work[nc] = pro[c].to_numpy(); pro_cols.append(nc)
        transformers.append(("pro", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), pro_cols))
    if not transformers: raise ValueError("No tabular feature block requested")
    prep = ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)
    encoded = prep.fit_transform(work)
    reducer = None
    if encoded.shape[1] > max_components:
        ncomp = min(max_components, encoded.shape[0] - 1, encoded.shape[1] - 1)
        reducer = TruncatedSVD(n_components=ncomp, random_state=CV_SEED).fit(encoded)
        reduced = reducer.transform(encoded)
    else:
        reduced = encoded.toarray() if sparse.issparse(encoded) else np.asarray(encoded)
    return normalize(np.asarray(reduced, float), norm="l2"), (prep, reducer, categorical, include_pro)


def transform_tabular_block(df: pd.DataFrame, fitted) -> np.ndarray:
    prep, reducer, categorical, include_pro = fitted; work = df.copy()
    for c in categorical:
        s = work[c].astype("string").str.strip(); work[c] = s.astype(object).where(s.notna(), np.nan)
    if include_pro:
        pro = ordinal_pro_frame(work)
        for c in PRO_ORDINAL: work["__PRO__" + c] = pro[c].to_numpy()
    encoded = prep.transform(work)
    reduced = reducer.transform(encoded) if reducer is not None else (encoded.toarray() if sparse.issparse(encoded) else np.asarray(encoded))
    return normalize(np.asarray(reduced, float), norm="l2")


def gower_distance(df: pd.DataFrame, *, categorical: list[str], include_pro: bool) -> np.ndarray:
    n = len(df); total = np.zeros((n, n), float); weight = np.zeros((n, n), float)
    for c in categorical:
        source_series = df[c]; missing = source_series.isna().to_numpy(); values = canonical_text(source_series).fillna("__MISSING__").to_numpy(str)
        valid = ~(missing[:, None] | missing[None, :]); diff = (values[:, None] != values[None, :]).astype(float)
        total += np.where(valid, diff, 0.0); weight += valid.astype(float)
    if include_pro:
        pro = ordinal_pro_frame(df)
        for c in PRO_ORDINAL:
            values = pro[c].to_numpy(float); valid_vec = np.isfinite(values); valid = valid_vec[:, None] & valid_vec[None, :]
            finite = values[valid_vec]; r = float(np.max(finite) - np.min(finite)) if len(finite) else 0.0
            diff = np.zeros((n, n), float) if r == 0 else np.abs(values[:, None] - values[None, :]) / r
            total += np.where(valid, diff, 0.0); weight += valid.astype(float)
    if np.any(weight == 0): raise RuntimeError("A participant pair has no jointly observed Gower features")
    d = total / weight; np.fill_diagonal(d, 0.0); return d
