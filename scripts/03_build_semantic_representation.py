from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import LAMBDA_GRID, combine_views, l2_rows, mean_pairwise_sq_distance_unit, read_jsonl, write_json

MAX_SEQ_LENGTH = 384
BATCH_SIZE = 32
PAIRWISE_SAMPLE_N = 100_000
GEOMETRY_SEED = 20260904
NEIGHBOUR_K = (5, 10, 25, 50)


def load_model(spec: str, device: str | None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("This script requires sentence-transformers") from exc
    model = SentenceTransformer(spec, device=device); model.max_seq_length = MAX_SEQ_LENGTH
    return model


def token_audit(model, ids: list[str], texts: dict[str, list[str]], label: str) -> pd.DataFrame:
    rows = []
    for view, values in texts.items():
        for rid, text in zip(ids, values):
            n = len(model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
            rows.append({"encoder": label, "record_id": rid, "view": view, "n_tokens": n})
            if n > MAX_SEQ_LENGTH:
                raise RuntimeError(f"{label}/{view}: record {rid} has {n} tokens, exceeding {MAX_SEQ_LENGTH}; truncation is not permitted")
    return pd.DataFrame(rows)


def encode(model, values: list[str], name: str) -> np.ndarray:
    x = model.encode(values, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return l2_rows(np.asarray(x, dtype=np.float32)).astype(np.float32)


def geometry_summary(base: np.ndarray, adapted: np.ndarray, view: str) -> dict:
    base, adapted = l2_rows(base), l2_rows(adapted)
    same = np.sum(base * adapted, axis=1)
    rng = np.random.default_rng(GEOMETRY_SEED); n = len(base)
    i = rng.integers(0, n, size=PAIRWISE_SAMPLE_N * 2); j = rng.integers(0, n, size=PAIRWISE_SAMPLE_N * 2)
    keep = i != j; i, j = i[keep][:PAIRWISE_SAMPLE_N], j[keep][:PAIRWISE_SAMPLE_N]
    db = 1.0 - np.sum(base[i] * base[j], axis=1); da = 1.0 - np.sum(adapted[i] * adapted[j], axis=1)
    pearson = float(np.corrcoef(db, da)[0, 1]) if np.std(db) > 0 and np.std(da) > 0 else np.nan
    spearman = float(pd.Series(db).corr(pd.Series(da), method="spearman")) if np.std(db) > 0 and np.std(da) > 0 else np.nan
    sb, sa = base @ base.T, adapted @ adapted.T; np.fill_diagonal(sb, -np.inf); np.fill_diagonal(sa, -np.inf)
    overlaps = {}
    for k in NEIGHBOUR_K:
        nb = np.argpartition(-sb, kth=k - 1, axis=1)[:, :k]; na = np.argpartition(-sa, kth=k - 1, axis=1)[:, :k]
        vals = [len(set(nb[r]).intersection(set(na[r]))) / k for r in range(n)]
        overlaps[f"top{k}_mean_overlap"] = float(np.mean(vals)); overlaps[f"top{k}_median_overlap"] = float(np.median(vals))
    return {
        "view": view, "n": n, "n_features": base.shape[1],
        "mean_same_patient_cosine": float(np.mean(same)), "sd_same_patient_cosine": float(np.std(same, ddof=1)),
        "min_same_patient_cosine": float(np.min(same)), "pairwise_cosine_distance_pearson": pearson,
        "pairwise_cosine_distance_spearman": spearman, **overlaps,
    }


def encode_all(model, texts: dict[str, list[str]]) -> dict[str, np.ndarray]:
    return {name: encode(model, values, name) for name, values in texts.items()}


def save_payload(payload: dict[str, np.ndarray], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, x in payload.items(): np.save(directory / f"{name}.npy", x.astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build semantic views and quantify two-view geometry.")
    ap.add_argument("--summaries", required=True)
    ap.add_argument("--adapted-model", required=True, help="Path to the frozen schema-adapted encoder.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--base-model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-base-ablation", action="store_true")
    args = ap.parse_args()

    df = read_jsonl(args.summaries)
    required = {"record_id", "context_text", "pro_text", "demographic_clinical_text", "chd_text", "vignette_canonical"}
    missing = required - set(df.columns)
    if missing: raise ValueError("Summary JSONL missing: " + ", ".join(sorted(missing)))
    if df["record_id"].duplicated().any(): raise ValueError("Duplicate record IDs in summaries")
    ids = df["record_id"].astype(str).tolist()
    texts = {
        "context": df["context_text"].astype(str).tolist(),
        "pro": df["pro_text"].astype(str).tolist(),
        "demographic_clinical": df["demographic_clinical_text"].astype(str).tolist(),
        "chd": df["chd_text"].astype(str).tolist(),
        "joint": df["vignette_canonical"].astype(str).tolist(),
    }
    if any(not x.strip() for values in texts.values() for x in values):
        raise ValueError("Analytic representation construction requires all five text views to be non-empty")

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    adapted_model = load_model(args.adapted_model, args.device)
    token_frames = [token_audit(adapted_model, ids, texts, "adapted")]
    adapted = encode_all(adapted_model, texts)
    adapted["separate_equal"] = combine_views(adapted["context"], adapted["pro"], 0.5)
    save_payload(adapted, out / "adapted")
    (out / "embedding_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    d2_context = mean_pairwise_sq_distance_unit(adapted["context"])
    d2_pro = mean_pairwise_sq_distance_unit(adapted["pro"])
    equal_lambda = d2_context / (d2_context + d2_pro)
    geo = []
    for lam in LAMBDA_GRID:
        denom = (1.0 - lam) * d2_context + lam * d2_pro
        geo.append({
            "lambda": float(lam),
            "mean_context_sqdistance_contribution": float((1.0 - lam) * d2_context),
            "mean_pro_sqdistance_contribution": float(lam * d2_pro),
            "mean_total_sqdistance": float(denom),
            "realised_pro_share": float(lam * d2_pro / denom) if denom > 0 else np.nan,
        })
    pd.DataFrame(geo).to_csv(out / "geometric_weighting.csv", index=False)

    diagnostics = []
    if not args.skip_base_ablation:
        base_model = load_model(args.base_model, args.device)
        token_frames.append(token_audit(base_model, ids, texts, "base"))
        base = encode_all(base_model, texts); base["separate_equal"] = combine_views(base["context"], base["pro"], 0.5)
        save_payload(base, out / "base")
        for view in ["context", "pro", "joint", "separate_equal"]:
            diagnostics.append(geometry_summary(base[view], adapted[view], view))
        pd.DataFrame(diagnostics).to_csv(out / "base_adapted_geometry_diagnostics.csv", index=False)

    pd.concat(token_frames, ignore_index=True).to_csv(out / "token_length_audit.csv", index=False)
    write_json({
        "n_participants": len(ids), "adapted_model": args.adapted_model, "base_model": args.base_model,
        "max_sequence_length": MAX_SEQ_LENGTH, "embedding_dimension_each_view": int(adapted["context"].shape[1]),
        "mean_pairwise_squared_distance_context": d2_context, "mean_pairwise_squared_distance_pro": d2_pro,
        "context_to_pro_distance_ratio": d2_context / d2_pro,
        "lambda_for_equal_mean_squared_distance_contribution": equal_lambda,
        "representation": "z_i(lambda)=[sqrt(1-lambda)c_i, sqrt(lambda)p_i], with c_i and p_i separately L2-normalised",
        "base_ablation_generated": not args.skip_base_ablation,
    }, out / "representation_manifest.json")
    print(f"Saved semantic representations to {out}")


if __name__ == "__main__":
    main()
