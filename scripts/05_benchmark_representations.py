from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from common import (
    CHD_CONTEXT, CONTEXT_FIELDS, DEMOCLIN_CONTEXT, PRO_ORDINAL,
    combine_views, fit_hdbscan, fit_tabular_block, transform_tabular_block,
    gower_distance, cluster_metrics, lesion_stratified_resamples,
    partition_agreement, read_json, normalise_ids, ordinal_pro_frame,
)

CONTENT_SETS = ["all-summary", "context-only", "demographic/clinical-only", "CHD-only", "PRO-only"]
PRIMARY_MCS = 40
PRIMARY_MS = 10
PRIMARY_METHOD = "leaf"
K_VALUES = range(2, 13)
KMEANS_N_INIT = 20
KMEANS_SEED = 123


@dataclass
class Rep:
    family: str
    content_set: str
    matrix: np.ndarray
    metric: str = "euclidean"


def load_data(path: str, labels_path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"record_id": "string"}, low_memory=False)
    labels = pd.read_csv(labels_path, dtype={"record_id": "string"})
    df["record_id"] = normalise_ids(df["record_id"]); labels["record_id"] = normalise_ids(labels["record_id"])
    if df["record_id"].duplicated().any() or labels["record_id"].duplicated().any():
        raise ValueError("record_id must be unique")
    overlap = [c for c in PRO_ORDINAL if c in df]
    if overlap: df = df.drop(columns=overlap)
    df = df.merge(labels, on="record_id", how="inner", validate="one_to_one")
    if "age_band" not in df:
        if "age" not in df: raise ValueError("age or age_band is required")
        age = pd.to_numeric(df["age"], errors="coerce")
        df["age_band"] = pd.cut(age, [-np.inf,29,39,49,59,np.inf], labels=["18-29","30-39","40-49","50-59","60+"]).astype("string")
    required = {"record_id", "chd_category", *CONTEXT_FIELDS, *PRO_ORDINAL}
    missing = sorted(required - set(df.columns))
    if missing: raise ValueError("Missing fields: " + ", ".join(missing))
    return df.sort_values("record_id", kind="mergesort").reset_index(drop=True)


def load_semantic(directory: Path, ids: pd.Series) -> dict[str, np.ndarray]:
    source_ids = [x.strip() for x in (directory.parent / "embedding_ids.txt").read_text().splitlines() if x.strip()]
    index = {rid:i for i,rid in enumerate(source_ids)}
    order = [index[str(x)] for x in ids]
    mapping = {
        "context-only":"context.npy", "demographic/clinical-only":"demographic_clinical.npy",
        "CHD-only":"chd.npy", "PRO-only":"pro.npy",
    }
    out = {k: np.load(directory / fn)[order] for k,fn in mapping.items()}
    return out


def tabular_rep(df: pd.DataFrame, content: str, lam: float) -> tuple[np.ndarray, object]:
    if content == "all-summary":
        c, cf = fit_tabular_block(df, categorical=CONTEXT_FIELDS, include_pro=False)
        p, pf = fit_tabular_block(df, categorical=[], include_pro=True)
        return combine_views(c, p, lam), ("mixed", cf, pf, lam)
    if content == "context-only": return (*fit_tabular_block(df, categorical=CONTEXT_FIELDS, include_pro=False),)
    if content == "demographic/clinical-only": return (*fit_tabular_block(df, categorical=DEMOCLIN_CONTEXT, include_pro=False),)
    if content == "CHD-only": return (*fit_tabular_block(df, categorical=CHD_CONTEXT, include_pro=False),)
    if content == "PRO-only": return (*fit_tabular_block(df, categorical=[], include_pro=True),)
    raise KeyError(content)


def tabular_transform(df: pd.DataFrame, fitted) -> np.ndarray:
    if isinstance(fitted, tuple) and fitted and fitted[0] == "mixed":
        _, cf, pf, lam = fitted
        return combine_views(transform_tabular_block(df, cf), transform_tabular_block(df, pf), lam)
    return transform_tabular_block(df, fitted)


def gower_rep(df: pd.DataFrame, content: str, lam: float) -> np.ndarray:
    if content == "all-summary":
        dc = gower_distance(df, categorical=CONTEXT_FIELDS, include_pro=False)
        dp = gower_distance(df, categorical=[], include_pro=True)
        return (1.0-lam)*dc + lam*dp
    if content == "context-only": return gower_distance(df, categorical=CONTEXT_FIELDS, include_pro=False)
    if content == "demographic/clinical-only": return gower_distance(df, categorical=DEMOCLIN_CONTEXT, include_pro=False)
    if content == "CHD-only": return gower_distance(df, categorical=CHD_CONTEXT, include_pro=False)
    if content == "PRO-only": return gower_distance(df, categorical=[], include_pro=True)
    raise KeyError(content)


def fit_partition(rep: Rep) -> tuple[np.ndarray, np.ndarray]:
    _, labels, _, persistence = fit_hdbscan(rep.matrix, min_cluster_size=PRIMARY_MCS, min_samples=PRIMARY_MS,
                                            method=PRIMARY_METHOD, metric=rep.metric)
    return labels, persistence


def fixed_k_labels(x: np.ndarray, algorithm: str, k: int) -> np.ndarray:
    if algorithm == "K-means":
        return KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=KMEANS_SEED, algorithm="lloyd").fit_predict(x)
    z = linkage(np.asarray(x, float), method="ward", metric="euclidean")
    return fcluster(z, t=k, criterion="maxclust").astype(int) - 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare semantic, direct-tabular and Gower representations under matched information and clustering conditions.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--ordinal-labels", required=True)
    ap.add_argument("--semantic-dir", required=True, help="Directory containing adapted semantic *.npy files from script 03.")
    ap.add_argument("--selection-dir", required=True, help="Output directory from script 04.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stability-resamples", type=int, default=50)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data, args.ordinal_labels)
    selection = read_json(Path(args.selection_dir) / "selected_operating_point.json")
    lam = float(selection["selected_lambda"])
    semantic = load_semantic(Path(args.semantic_dir), df["record_id"])
    semantic["all-summary"] = combine_views(semantic["context-only"], semantic["PRO-only"], lam)

    reps: dict[tuple[str,str], Rep] = {}
    tabular_fitted = {}
    for content in CONTENT_SETS:
        reps[("Semantic", content)] = Rep("Semantic", content, semantic[content])
        tx, fitted = tabular_rep(df, content, lam); tabular_fitted[content] = fitted
        reps[("Tabular", content)] = Rep("Tabular", content, tx)
        reps[("Gower", content)] = Rep("Gower", content, gower_rep(df, content, lam), "precomputed")

    metric_rows, assignment_rows = [], []
    full_labels = {}
    for key, rep in reps.items():
        labels, persistence = fit_partition(rep); full_labels[key] = labels
        row = {"family":rep.family, "content_set":rep.content_set, **cluster_metrics(labels, persistence, df["chd_category"])}
        metric_rows.append(row)
        for rid, lab in zip(df["record_id"], labels): assignment_rows.append({"record_id":rid,"family":rep.family,"content_set":rep.content_set,"cluster":int(lab)})
    pd.DataFrame(metric_rows).to_csv(out / "matched_hdbscan_metrics.csv", index=False)
    pd.DataFrame(assignment_rows).to_csv(out / "matched_hdbscan_assignments.csv.gz", index=False, compression="gzip")

    # Representation-specific stability. Transparent transformations are refitted in each resample;
    # frozen semantic embeddings are subset directly.
    resamples = lesion_stratified_resamples(df["chd_category"], args.stability_resamples)
    stability_rows = []
    for r, idx in enumerate(resamples, start=1):
        sub = df.iloc[idx].reset_index(drop=True)
        for content in CONTENT_SETS:
            # semantic
            srep = Rep("Semantic", content, semantic[content][idx])
            sl, _ = fit_partition(srep)
            ag = partition_agreement(full_labels[("Semantic",content)][idx], sl)
            stability_rows.append({"resample":r,"family":"Semantic","content_set":content,**ag})
            # tabular refit
            tx, _ = tabular_rep(sub, content, lam); tl, _ = fit_partition(Rep("Tabular",content,tx))
            ag = partition_agreement(full_labels[("Tabular",content)][idx], tl)
            stability_rows.append({"resample":r,"family":"Tabular","content_set":content,**ag})
            # gower refit, including ranges
            gd = gower_rep(sub, content, lam); gl, _ = fit_partition(Rep("Gower",content,gd,"precomputed"))
            ag = partition_agreement(full_labels[("Gower",content)][idx], gl)
            stability_rows.append({"resample":r,"family":"Gower","content_set":content,**ag})
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(out / "matched_hdbscan_stability.csv.gz", index=False, compression="gzip")
    stability.groupby(["family","content_set"], as_index=False).agg(
        mean_ari=("ari_all","mean"), mean_nmi=("nmi_all","mean"), mean_assigned_noise_agreement=("assigned_noise_agreement","mean")
    ).to_csv(out / "matched_hdbscan_stability_summary.csv", index=False)

    # Fixed-K comparison uses only Euclidean semantic and tabular spaces.
    fixed_rows = []
    fixed_full = {}
    for family in ["Semantic","Tabular"]:
        for content in CONTENT_SETS:
            x = reps[(family,content)].matrix
            for algorithm in ["K-means","Ward"]:
                for k in K_VALUES:
                    labels = fixed_k_labels(x, algorithm, k); fixed_full[(family,content,algorithm,k)] = labels
                    fixed_rows.append({"family":family,"content_set":content,"algorithm":algorithm,"k":k,
                                       "lesion_coherence":cluster_metrics(labels,np.ones(k),df["chd_category"])["lesion_coherence"],
                                       "diagnosis_fragmentation":cluster_metrics(labels,np.ones(k),df["chd_category"])["diagnosis_fragmentation"]})
    pd.DataFrame(fixed_rows).to_csv(out / "fixed_k_metrics.csv", index=False)

    fixed_stability = []
    for r, idx in enumerate(resamples, start=1):
        sub = df.iloc[idx].reset_index(drop=True)
        for family in ["Semantic","Tabular"]:
            for content in CONTENT_SETS:
                x_sub = semantic[content][idx] if family == "Semantic" else tabular_rep(sub, content, lam)[0]
                for algorithm in ["K-means","Ward"]:
                    labels = fixed_k_labels(x_sub, algorithm, 6)
                    ref = fixed_full[(family,content,algorithm,6)][idx]
                    fixed_stability.append({"resample":r,"family":family,"content_set":content,"algorithm":algorithm,
                                            "ari":adjusted_rand_score(ref,labels),"nmi":normalized_mutual_info_score(ref,labels)})
    pd.DataFrame(fixed_stability).to_csv(out / "fixed_k6_stability.csv.gz", index=False, compression="gzip")
    print(f"Matched representation benchmark complete; selected lambda={lam:.3f}; outputs: {out}")


if __name__ == "__main__": main()
