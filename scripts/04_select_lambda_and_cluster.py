from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    LAMBDA_GRID, RESAMPLE_SEED, cluster_metrics, combine_views, fit_hdbscan,
    lesion_stratified_resamples, partition_agreement, select_lambda, write_json,
)

PRIMARY = {"min_cluster_size": 40, "min_samples": 10, "method": "leaf"}
MCS_GRID = [30, 40, 50]
MS_GRID = [5, 10, 15]
METHOD_GRID = ["leaf", "eom"]


def load_embeddings(directory: Path, cohort_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    source_ids = [x.strip() for x in (directory / "embedding_ids.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    pos = {rid: i for i, rid in enumerate(source_ids)}
    missing = [rid for rid in cohort_ids if rid not in pos]
    if missing: raise ValueError(f"Semantic representation is missing {len(missing)} cohort IDs")
    order = [pos[rid] for rid in cohort_ids]
    context = np.load(directory / "adapted" / "context.npy")[order]
    pro = np.load(directory / "adapted" / "pro.npy")[order]
    return np.asarray(context, float), np.asarray(pro, float)


def stability(x: np.ndarray, full_labels: np.ndarray, resamples: list[np.ndarray], config: dict) -> pd.DataFrame:
    rows = []
    for r, idx in enumerate(resamples, start=1):
        _, labels, _, _ = fit_hdbscan(
            x[idx], min_cluster_size=config["min_cluster_size"], min_samples=config["min_samples"], method=config["method"]
        )
        rows.append({"resample": r, "n_sampled": len(idx), **partition_agreement(full_labels[idx], labels),
                     "n_clusters_resample": int(len(set(labels[labels >= 0].tolist()))), "noise_fraction_resample": float(np.mean(labels == -1))})
    return pd.DataFrame(rows)


def run_grid(context: np.ndarray, pro: np.ndarray, lesions: pd.Series, resamples: list[np.ndarray], config: dict,
             keep_assignments: bool = False):
    metric_rows, stability_rows, labels_by_lambda, probability_by_lambda = [], [], {}, {}
    for lam in LAMBDA_GRID:
        x = combine_views(context, pro, float(lam))
        _, labels, probs, persistence = fit_hdbscan(
            x, min_cluster_size=config["min_cluster_size"], min_samples=config["min_samples"], method=config["method"]
        )
        labels_by_lambda[float(lam)] = labels
        if keep_assignments: probability_by_lambda[float(lam)] = probs
        row = {"lambda": float(lam), **cluster_metrics(labels, persistence, lesions)}
        metric_rows.append(row)
        stab = stability(x, labels, resamples, config); stab.insert(0, "lambda", float(lam)); stability_rows.append(stab)
    metrics = pd.DataFrame(metric_rows)
    stabs = pd.concat(stability_rows, ignore_index=True)
    summary = stabs.groupby("lambda", as_index=False).agg(
        stability_ari_all_mean=("ari_all", "mean"), stability_ari_all_sd=("ari_all", "std"),
        stability_nmi_all_mean=("nmi_all", "mean"),
        stability_assigned_noise_agreement_mean=("assigned_noise_agreement", "mean"),
    )
    metrics = metrics.merge(summary, on="lambda", validate="one_to_one")
    selected, selection_table = select_lambda(metrics)
    return selected, selection_table, stabs, labels_by_lambda, probability_by_lambda


def pairwise_partition_table(labels_by_lambda: dict[float, np.ndarray]) -> pd.DataFrame:
    rows = []; vals = sorted(labels_by_lambda)
    for i, a in enumerate(vals):
        for b in vals[i:]:
            rows.append({"lambda_a": a, "lambda_b": b, **partition_agreement(labels_by_lambda[a], labels_by_lambda[b])})
    return pd.DataFrame(rows)


def assignment_classification(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # matrix is n_participants x n_solutions, True if assigned.
    freq = matrix.mean(axis=1)
    status = np.where(freq == 1.0, "always assigned", np.where(freq == 0.0, "never assigned", "variably assigned"))
    return freq, status


def main() -> None:
    ap = argparse.ArgumentParser(description="Select the relative context-PRO weight without outcomes and assess HDBSCAN sensitivity.")
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--primary-resamples", type=int, default=50)
    ap.add_argument("--sensitivity-resamples", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(args.cohort, dtype={"record_id": "string"}, low_memory=False)
    if "record_id" not in cohort or "chd_category" not in cohort: raise ValueError("cohort requires record_id and chd_category")
    cohort["record_id"] = cohort["record_id"].astype("string").str.strip()
    if cohort.record_id.isna().any() or cohort.record_id.duplicated().any(): raise ValueError("record_id must be complete and unique")
    ids = cohort.record_id.astype(str).tolist(); lesions = cohort["chd_category"].astype("string")
    context, pro = load_embeddings(Path(args.embeddings), ids)

    primary_samples = lesion_stratified_resamples(lesions, args.primary_resamples, seed=RESAMPLE_SEED)
    selected, selection, primary_stability, primary_labels, primary_probs = run_grid(
        context, pro, lesions, primary_samples, PRIMARY, keep_assignments=True
    )
    selection.to_csv(out / "lambda_selection_table.csv", index=False)
    primary_stability.to_csv(out / "lambda_resampling_stability.csv.gz", index=False, compression="gzip")
    pairwise_partition_table(primary_labels).to_csv(out / "lambda_pairwise_partition_agreement.csv.gz", index=False, compression="gzip")

    assignment_rows = []
    for lam in sorted(primary_labels):
        for rid, label, prob in zip(ids, primary_labels[lam], primary_probs[lam]):
            assignment_rows.append({"record_id": rid, "lambda": lam, "cluster": int(label), "probability": float(prob)})
    pd.DataFrame(assignment_rows).to_csv(out / "lambda_full_assignments.csv.gz", index=False, compression="gzip")

    sensitivity_samples = lesion_stratified_resamples(lesions, args.sensitivity_resamples, seed=RESAMPLE_SEED)
    config_rows, config_tables, selected_config_labels = [], [], []
    for mcs, ms, method in product(MCS_GRID, MS_GRID, METHOD_GRID):
        config = {"min_cluster_size": mcs, "min_samples": ms, "method": method}
        sel, table, _, labels_grid, _ = run_grid(context, pro, lesions, sensitivity_samples, config, keep_assignments=False)
        table.insert(0, "cluster_selection_method", method); table.insert(0, "min_samples", ms); table.insert(0, "min_cluster_size", mcs)
        config_tables.append(table)
        selected_row = table[np.isclose(table["lambda"], sel)].iloc[0]
        config_rows.append({
            "min_cluster_size": mcs, "min_samples": ms, "cluster_selection_method": method, "selected_lambda": sel,
            "selected_n_clusters": int(selected_row["n_clusters"]), "selected_assigned_fraction": float(selected_row["assigned_fraction"]),
            "selected_stability_ari": float(selected_row["stability_ari_all_mean"]),
            "selected_weighted_persistence": float(selected_row["weighted_cluster_persistence"]),
            "selected_lesion_coherence": float(selected_row["lesion_coherence"]),
            "selected_diagnosis_fragmentation": float(selected_row["diagnosis_fragmentation"]),
        })
        selected_config_labels.append(labels_grid[sel])
    sensitivity = pd.DataFrame(config_rows)
    sensitivity.to_csv(out / "hdbscan_sensitivity_selected_operating_points.csv", index=False)
    pd.concat(config_tables, ignore_index=True).to_csv(out / "hdbscan_sensitivity_full_grid.csv.gz", index=False, compression="gzip")

    agreement_rows = []
    for i in range(len(selected_config_labels)):
        for j in range(i + 1, len(selected_config_labels)):
            agreement_rows.append({
                "configuration_a": i + 1, "configuration_b": j + 1,
                **partition_agreement(selected_config_labels[i], selected_config_labels[j]),
            })
    selected_agreement = pd.DataFrame(agreement_rows)
    selected_agreement.to_csv(out / "hdbscan_sensitivity_selected_partition_agreement.csv", index=False)

    supported_min, supported_max = float(sensitivity.selected_lambda.min()), float(sensitivity.selected_lambda.max())
    region_lambdas = [float(x) for x in LAMBDA_GRID if supported_min - 1e-12 <= x <= supported_max + 1e-12]
    region_matrix = np.column_stack([primary_labels[x] >= 0 for x in region_lambdas])
    region_freq, region_status = assignment_classification(region_matrix)
    selected_matrix = np.column_stack([x >= 0 for x in selected_config_labels])
    selected_freq, selected_status = assignment_classification(selected_matrix)
    uncertainty = pd.DataFrame({
        "record_id": ids,
        "supported_lambda_assignment_fraction": region_freq,
        "supported_lambda_assignment_status": region_status,
        "selected_configuration_assignment_fraction": selected_freq,
        "selected_configuration_assignment_status": selected_status,
    })
    uncertainty.to_csv(out / "assignment_uncertainty.csv", index=False)

    primary_row = selection[np.isclose(selection["lambda"], selected)].iloc[0]
    write_json({
        "selected_lambda": selected,
        "selection_metrics": ["assigned_fraction", "stability_ari_all_mean", "weighted_cluster_persistence"],
        "selection_uses_lesion_coherence_or_composition_metric": False,
        "selection_resampling_stratified_by_exact_chd_category": True,
        "selection_uses_burden_labels": False,
        "selection_uses_validation_outcomes": False,
        "tie_break_order": ["minimum ideal_distance", "higher stability", "higher coverage", "higher weighted persistence", "smaller lambda"],
        "primary_hdbscan": PRIMARY,
        "primary_resampling": {"n_resamples": args.primary_resamples, "fraction": 0.80, "stratification": "exact chd_category", "seed_base": RESAMPLE_SEED},
        "selected_primary_metrics": {
            "n_clusters": int(primary_row.n_clusters), "n_assigned": int(primary_row.n_assigned), "n_noise": int(primary_row.n_noise),
            "assigned_fraction": float(primary_row.assigned_fraction), "stability_ari_all_mean": float(primary_row.stability_ari_all_mean),
            "weighted_cluster_persistence": float(primary_row.weighted_cluster_persistence),
            "lesion_coherence": float(primary_row.lesion_coherence), "diagnosis_fragmentation": float(primary_row.diagnosis_fragmentation),
        },
        "hdbscan_sensitivity": {
            "min_cluster_size": MCS_GRID, "min_samples": MS_GRID, "cluster_selection_method": METHOD_GRID,
            "n_resamples_per_configuration_lambda": args.sensitivity_resamples,
            "supported_lambda_min": supported_min, "supported_lambda_max": supported_max,
            "selected_partition_ari_min": float(selected_agreement["ari_all"].min()),
            "selected_partition_ari_median": float(selected_agreement["ari_all"].median()),
            "selected_partition_ari_max": float(selected_agreement["ari_all"].max()),
        },
    }, out / "selected_operating_point.json")
    print(f"Selected lambda: {selected:.3f}; supported sensitivity interval: {supported_min:.3f}–{supported_max:.3f}")


if __name__ == "__main__":
    main()
