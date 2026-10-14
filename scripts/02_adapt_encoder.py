from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from common import read_jsonl, write_json

HEADERS = ["DEMOGRAPHICS:", "CHD_CONTEXT:", "COMORBIDITY_TREATMENT:", "PRO_PROFILE:"]
DEFAULTS = {
    "seed": 123, "batch_size": 32, "epochs": 1, "lr": 5e-6, "warmup_frac": 0.05,
    "max_seq_length": 384, "train_frac": 0.70, "val_frac": 0.15,
    "num_views_per_record": 1, "anchor_repeats": 6, "unfreeze_last_n_layers": 1,
}


def stable_seed(base_seed: int, record_id: str, salt: int = 0) -> int:
    h = hashlib.md5(f"{base_seed}|{record_id}|{salt}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def split_sections(text: str) -> dict[str, str]:
    text = str(text).strip(); out = {h: "" for h in HEADERS}
    hits = sorted((m.start(), m.group(0)) for m in re.finditer("|".join(re.escape(h) for h in HEADERS), text))
    for i, (pos, header) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        out[header] = text[pos:end].strip()
    return out


def split_semicolon_section(section: str) -> tuple[str, list[str]]:
    if not section or ":" not in section: return "", []
    header, payload = section.split(":", 1); payload = payload.strip().rstrip(".")
    return header.strip(), [x.strip() for x in payload.split(";") if x.strip()]


def render_semicolon_section(header: str, items: list[str]) -> str:
    return "" if not items else f"{header}: " + "; ".join(items) + "."


def mask_value(item: str) -> str:
    words = item.strip().rstrip(".").split(" ", 1)
    return (words[0] + " [MASK]") if len(words) == 1 else (words[0] + " " + (words[1].split(" ", 1)[0] if " " in words[1] else words[1]) + " [MASK]")


def mask_one_semicolon_item(section: str, rng: np.random.Generator) -> str:
    header, items = split_semicolon_section(section)
    if not items: return section
    idx = int(rng.integers(0, len(items))); items = items.copy()
    # Preserve the fixed field label and replace only its value.
    item = items[idx]
    labels = [
        "sex", "age band", "remoteness", "medication burden", "comorbidity burden",
        "diabetes", "high blood pressure", "asthma",
    ]
    matched = next((lab for lab in sorted(labels, key=len, reverse=True) if item.lower().startswith(lab + " ")), None)
    items[idx] = f"{matched} [MASK]" if matched else mask_value(item)
    return render_semicolon_section(header, items)


def mask_one_pro_item(section: str, rng: np.random.Generator) -> str:
    if not section or ":" not in section: return section
    _, payload = section.split(":", 1)
    blocks = [b.strip() for b in re.split(r"(?<=\.)\s+", payload.strip()) if b.strip()]
    candidates = []
    for bi, block in enumerate(blocks):
        if "—" not in block: continue
        prefix, values = block.split("—", 1)
        items = [x.strip() for x in values.strip().rstrip(".").split(",") if x.strip()]
        if items: candidates.append((bi, prefix, items))
    if not candidates: return section
    bi, prefix, items = candidates[int(rng.integers(0, len(candidates)))]
    ii = int(rng.integers(0, len(items))); item = items[ii]
    labels = [
        "physical appearance", "cognitive problems", "treatment anxiety", "communication", "treatment",
        "school/work", "psychosocial", "emotional", "physical", "social", "total",
        "depression", "anxiety", "stress",
    ]
    matched = next((lab for lab in sorted(labels, key=len, reverse=True) if item.lower().startswith(lab + " ")), None)
    items[ii] = f"{matched} [MASK]" if matched else mask_value(item)
    blocks[bi] = prefix + "—" + ", ".join(items) + "."
    return "PRO_PROFILE: " + " ".join(blocks)


def make_augmented_view(text: str, rng: np.random.Generator) -> str:
    """CHD-preserving augmentation used for schema familiarisation."""
    sec = split_sections(text); new = dict(sec); u = float(rng.random())
    if u < 0.30 and new["PRO_PROFILE:"]:
        new["PRO_PROFILE:"] = mask_one_pro_item(new["PRO_PROFILE:"], rng)
    else:
        choices = [h for h in ["DEMOGRAPHICS:", "COMORBIDITY_TREATMENT:"] if new[h]]
        if choices:
            target = choices[int(rng.integers(0, len(choices)))]
            new[target] = mask_one_semicolon_item(new[target], rng)
        elif new["PRO_PROFILE:"]:
            new["PRO_PROFILE:"] = mask_one_pro_item(new["PRO_PROFILE:"], rng)
    # CHD_CONTEXT is never modified.
    aug = " ".join(new[h] for h in HEADERS if new[h]).strip()
    return aug if aug else text


def derangement(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); base = np.arange(n)
    if n < 2: return base
    for _ in range(100):
        p = rng.permutation(n)
        if np.all(p != base): return p
    return np.roll(base, 1)


def freeze_conservatively(model, unfreeze_last_n_layers: int = 1) -> None:
    first = model._first_module()
    if not hasattr(first, "auto_model"): raise RuntimeError("Could not locate transformer auto_model")
    auto = first.auto_model
    for p in auto.parameters(): p.requires_grad = False
    if hasattr(auto, "encoder") and hasattr(auto.encoder, "layer"):
        layers = auto.encoder.layer; start = max(0, len(layers) - max(1, int(unfreeze_last_n_layers)))
        for i in range(start, len(layers)):
            for p in layers[i].parameters(): p.requires_grad = True
    if hasattr(auto, "pooler") and auto.pooler is not None:
        for p in auto.pooler.parameters(): p.requires_grad = True
    for name, p in auto.named_parameters():
        lname = name.lower()
        if "layernorm" in lname or lname.endswith(".ln.weight") or lname.endswith(".ln.bias") or "norm.weight" in lname or "norm.bias" in lname:
            p.requires_grad = True


def main() -> None:
    ap = argparse.ArgumentParser(description="Conservative schema adaptation of BAAI/bge-large-en-v1.5.")
    ap.add_argument("--summaries", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--base-model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--warmup-frac", type=float, default=DEFAULTS["warmup_frac"])
    ap.add_argument("--max-seq-length", type=int, default=DEFAULTS["max_seq_length"])
    ap.add_argument("--anchor-repeats", type=int, default=DEFAULTS["anchor_repeats"])
    ap.add_argument("--unfreeze-last-n-layers", type=int, default=DEFAULTS["unfreeze_last_n_layers"])
    args = ap.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from sentence_transformers.evaluation import BinaryClassificationEvaluator
    except ImportError as exc:
        raise ImportError("This script requires torch and sentence-transformers") from exc

    df = read_jsonl(args.summaries)
    if not {"record_id", "vignette_canonical"}.issubset(df): raise ValueError("Summary JSONL lacks required fields")
    df = df[df["vignette_canonical"].astype(str).str.strip().ne("")].drop_duplicates("record_id").reset_index(drop=True)
    ids = df["record_id"].astype(str).tolist()
    train_ids, temp_ids = train_test_split(ids, test_size=0.30, random_state=args.seed, shuffle=True)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=args.seed, shuffle=True)
    train = df[df.record_id.astype(str).isin(train_ids)].reset_index(drop=True)
    val = df[df.record_id.astype(str).isin(val_ids)].reset_index(drop=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = SentenceTransformer(args.base_model, device=device); base.max_seq_length = args.max_seq_length
    student = SentenceTransformer(args.base_model, device=device); student.max_seq_length = args.max_seq_length
    freeze_conservatively(student, args.unfreeze_last_n_layers)

    cosine_examples = []
    for rid, text in zip(train.record_id.astype(str), train.vignette_canonical.astype(str)):
        rng = np.random.default_rng(stable_seed(args.seed, rid, 10000))
        aug = make_augmented_view(text, rng)
        if aug != text: cosine_examples.append(InputExample(texts=[text, aug], label=1.0))
    base_embeddings = base.encode(train.vignette_canonical.astype(str).tolist(), batch_size=max(64, args.batch_size), normalize_embeddings=True, show_progress_bar=True)
    mse_examples = []
    for text, emb in zip(train.vignette_canonical.astype(str), np.asarray(base_embeddings, np.float32)):
        for _ in range(args.anchor_repeats): mse_examples.append(InputExample(texts=[text], label=emb.tolist()))

    val_texts = val.vignette_canonical.astype(str).tolist(); val_ids_s = val.record_id.astype(str).tolist(); neg = derangement(len(val), args.seed + 999)
    s1, s2, labels = [], [], []
    for i, (rid, text) in enumerate(zip(val_ids_s, val_texts)):
        aug = make_augmented_view(text, np.random.default_rng(stable_seed(args.seed, rid, 90000)))
        s1 += [text, text]; s2 += [aug, val_texts[int(neg[i])]]; labels += [1, 0]
    evaluator = BinaryClassificationEvaluator(s1, s2, labels, name="schema_pairs_val")

    loader_cos = DataLoader(cosine_examples, shuffle=True, batch_size=args.batch_size, drop_last=False, pin_memory=(device == "cuda"))
    loader_mse = DataLoader(mse_examples, shuffle=True, batch_size=args.batch_size, drop_last=False, pin_memory=(device == "cuda"))
    steps_per_epoch = max(1, min(len(loader_cos), len(loader_mse)))
    warmup_steps = max(1, int(steps_per_epoch * args.epochs * args.warmup_frac))
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    student.fit(
        train_objectives=[(loader_cos, losses.CosineSimilarityLoss(student)), (loader_mse, losses.MSELoss(student))],
        evaluator=evaluator, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
        optimizer_params={"lr": args.lr}, warmup_steps=warmup_steps,
        output_path=str(out), save_best_model=True, show_progress_bar=True,
    )

    (out / "train_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (out / "val_ids.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    (out / "test_ids.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    write_json({
        "base_model": args.base_model, "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr, "warmup_fraction": args.warmup_frac, "max_sequence_length": args.max_seq_length,
        "anchor_repeats": args.anchor_repeats, "unfreeze_last_n_layers": args.unfreeze_last_n_layers,
        "trainable_scope": "final transformer block plus transformer normalisation/pooler parameters; remaining transformer parameters frozen",
        "augmentation": "CHD-preserving; 30% single PRO descriptor mask, otherwise one demographic or treatment/comorbidity descriptor mask",
        "n_records_total": len(df), "n_train": len(train_ids), "n_validation": len(val_ids), "n_test": len(test_ids),
        "training_objectives": ["positive-pair cosine similarity", "MSE anchoring to frozen base-model embeddings"],
    }, out / "adaptation_manifest.json")
    print(f"Saved adapted encoder to {out}")


if __name__ == "__main__":
    main()
