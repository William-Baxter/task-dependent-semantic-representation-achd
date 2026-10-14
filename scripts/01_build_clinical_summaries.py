from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import write_jsonl

PEDS_MAP = [
    ("pedsql_core_score_phys_scaled", "pedsql_core_score_phys_scaled_label", "physical"),
    ("pedsql_core_score_emo_scaled", "pedsql_core_score_emo_scaled_label", "emotional"),
    ("pedsql_core_score_soc_scaled", "pedsql_core_score_soc_scaled_label", "social"),
    ("pedsql_core_score_sch_scaled", "pedsql_core_score_sch_scaled_label", "school/work"),
    ("pedsql_core_score_psy_scaled", "pedsql_core_score_psy_scaled_label", "psychosocial"),
    ("pedsql_core_score_tot_scaled", "pedsql_core_score_tot_scaled_label", "total"),
    ("pedsql_cardiac_score_cp_scaled", "pedsql_cardiac_score_cp_scaled_label", "cognitive problems"),
    ("pedsql_cardiac_score_com_scaled", "pedsql_cardiac_score_com_scaled_label", "communication"),
    ("pedsql_cardiac_score_ta_scaled", "pedsql_cardiac_score_ta_scaled_label", "treatment anxiety"),
    ("pedsql_cardiac_score_ppa_scaled", "pedsql_cardiac_score_ppa_scaled_label", "physical appearance"),
    ("pedsql_cardiac_score_treat_scaled", "pedsql_cardiac_score_treat_scaled_label", "treatment"),
]
DASS_MAP = [
    ("dass_dep_score", "dass_dep_class", "depression", (10, 14, 21, 28)),
    ("dass_anx_score", "dass_anx_class", "anxiety", (8, 10, 15, 20)),
    ("dass_stress_score", "dass_stress_class", "stress", (15, 19, 26, 34)),
]
DASS_LEVELS = ["normal", "mild", "moderate", "severe", "extremely severe"]
MISSING = {"", "na", "n/a", "nan", "none", "null", "missing", "unknown"}


def is_missing(x: Any) -> bool:
    if x is None or pd.isna(x): return True
    return isinstance(x, str) and x.strip().lower() in MISSING


def clean_token(x: Any) -> str | None:
    if is_missing(x): return None
    return " ".join(str(x).strip().lower().split())


def age_band(age: Any) -> str | None:
    if is_missing(age): return None
    x = float(age)
    if x < 18: return None
    if x < 30: return "18-29"
    if x < 40: return "30-39"
    if x < 50: return "40-49"
    if x < 60: return "50-59"
    return "≥60"


def pedsql_descriptor(value: Any, row: pd.Series) -> str | None:
    """Apply the exact deterministic boundary rules used by add_quantile_bands()."""
    if is_missing(value): return None
    x = float(value)
    if str(row["scheme"]) == "3-band fallback":
        q25, q75 = float(row["q25"]), float(row["q75"])
        if x <= q25: return "low"
        if x < q75: return "moderate"
        return "high"
    q10, q30, q70, q90 = map(float, [row["q10"], row["q30"], row["q70"], row["q90"]])
    if x <= q10: return "very low"
    if x <= q30: return "low"
    if x < q70: return "moderate"
    if x < q90: return "high"
    return "very high"


def dass_descriptor(value: Any, cuts: tuple[int, int, int, int]) -> str | None:
    if is_missing(value): return None
    x = float(value)
    if x < cuts[0]: return DASS_LEVELS[0]
    if x < cuts[1]: return DASS_LEVELS[1]
    if x < cuts[2]: return DASS_LEVELS[2]
    if x < cuts[3]: return DASS_LEVELS[3]
    return DASS_LEVELS[4]


def section(header: str, items: list[str]) -> str:
    return "" if not items else f"{header}: " + "; ".join(items) + "."


def render_row(row: pd.Series) -> dict:
    demo = []
    for field, label in [("sex", "sex"), ("age_band", "age band"), ("remoteness", "remoteness")]:
        value = clean_token(row.get(field))
        if value is not None: demo.append(f"{label} {value}")

    chd = []
    for field, label in [("chd_complexity", "complexity"), ("chd_category", "category")]:
        value = clean_token(row.get(field))
        if value is not None: chd.append(f"{label} {value}")

    clinical = []
    for field, label in [
        ("num_meds_label", "medication burden"), ("num_comorb_label", "comorbidity burden"),
        ("diabetes_label", "diabetes"), ("hbp_label", "high blood pressure"), ("asthma_label", "asthma"),
    ]:
        value = clean_token(row.get(field))
        if value is not None: clinical.append(f"{label} {value}")

    core_items = []
    for _, label_col, label in PEDS_MAP[:6]:
        value = clean_token(row.get(label_col))
        if value is not None: core_items.append(f"{label} {value}")
    cardiac_items = []
    for _, label_col, label in PEDS_MAP[6:]:
        value = clean_token(row.get(label_col))
        if value is not None: cardiac_items.append(f"{label} {value}")
    distress_items = []
    for _, label_col, label, _ in DASS_MAP:
        value = clean_token(row.get(label_col))
        if value is not None: distress_items.append(f"{label} {value}")

    demographics = section("DEMOGRAPHICS", demo)
    chd_text = section("CHD_CONTEXT", chd)
    clinical_text = section("COMORBIDITY_TREATMENT", clinical)
    pro_blocks = []
    if core_items: pro_blocks.append("pedsql core—" + ", ".join(core_items) + ".")
    if cardiac_items: pro_blocks.append("pedsql cardiac—" + ", ".join(cardiac_items) + ".")
    if distress_items: pro_blocks.append("distress—" + ", ".join(distress_items) + ".")
    pro_text = "" if not pro_blocks else "PRO_PROFILE: " + " ".join(pro_blocks)

    context = " ".join(x for x in [demographics, chd_text, clinical_text] if x).strip()
    democlin = " ".join(x for x in [demographics, clinical_text] if x).strip()
    canonical = " ".join(x for x in [demographics, chd_text, clinical_text, pro_text] if x).strip()
    return {
        "demographics_text": demographics,
        "chd_text": chd_text,
        "clinical_text": clinical_text,
        "demographic_clinical_text": democlin,
        "context_text": context,
        "pro_text": pro_text,
        "vignette_canonical": canonical,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess construction variables and render deterministic clinical summaries.")
    ap.add_argument("--input", required=True, help="Participant-level CSV with continuous PedsQL/DASS variables and context descriptors.")
    ap.add_argument("--thresholds", required=True, help="Frozen PedsQL rendering-threshold CSV.")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, dtype={"record_id": "string"}, low_memory=False)
    if "record_id" not in df: raise ValueError("record_id is required")
    df["record_id"] = df["record_id"].astype("string").str.strip()
    if df["record_id"].isna().any() or df["record_id"].duplicated().any():
        raise ValueError("record_id must be complete and unique")

    thresholds = pd.read_csv(args.thresholds).set_index("column")
    required_thresholds = {x[0] for x in PEDS_MAP}
    missing_thresholds = sorted(required_thresholds - set(thresholds.index))
    if missing_thresholds: raise ValueError("Threshold table missing: " + ", ".join(missing_thresholds))

    if "age_band" not in df:
        if "age" not in df: raise ValueError("age or age_band is required")
        df["age_band"] = [age_band(x) for x in df["age"]]
    else:
        missing_band = df["age_band"].isna() & df.get("age", pd.Series(np.nan, index=df.index)).notna()
        if missing_band.any(): df.loc[missing_band, "age_band"] = [age_band(x) for x in df.loc[missing_band, "age"]]

    for raw_col, label_col, _ in PEDS_MAP:
        if raw_col not in df: raise ValueError(f"Missing PedsQL construction variable: {raw_col}")
        df[label_col] = [pedsql_descriptor(v, thresholds.loc[raw_col]) for v in df[raw_col]]
    for raw_col, label_col, _, cuts in DASS_MAP:
        if raw_col not in df: raise ValueError(f"Missing DASS construction variable: {raw_col}")
        df[label_col] = [dass_descriptor(v, cuts) for v in df[raw_col]]

    rows = []
    for _, row in df.iterrows():
        text = render_row(row)
        # Rendering rule: CHD context plus at least one other major section.
        major_present = [bool(text["demographics_text"]), bool(text["chd_text"]), bool(text["clinical_text"]), bool(text["pro_text"])]
        if not major_present[1] or sum(major_present) < 2:
            continue
        rows.append({"record_id": str(row["record_id"]), **text})

    label_cols = ["record_id"] + [x[1] for x in PEDS_MAP] + [x[1] for x in DASS_MAP]
    df[label_cols].to_csv(out / "construction_pro_ordinal_labels.csv", index=False)
    write_jsonl(rows, out / "clinical_summaries.jsonl")
    pd.DataFrame({"record_id": df["record_id"], "rendered": df["record_id"].astype(str).isin({r["record_id"] for r in rows})}).to_csv(out / "rendering_audit.csv", index=False)
    print(f"Rendered {len(rows)} / {len(df)} participant summaries")


if __name__ == "__main__":
    main()
