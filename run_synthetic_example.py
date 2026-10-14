from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "synthetic_data" / "synthetic_cohort.csv"
THRESHOLDS = ROOT / "config" / "pedsql_rendering_thresholds.csv"


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *map(str, args)]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the public workflow on the artificial example cohort.")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--adapted-model", default=None, help="Optional existing compatible adapted encoder. If omitted, one is fitted on the artificial cohort.")
    ap.add_argument("--device", default=None, help="Optional SentenceTransformers device, e.g. cuda or cpu.")
    ap.add_argument("--quick", action="store_true", help="Software smoke test with fewer resamples and CV repeats.")
    args = ap.parse_args()

    work = Path(args.workdir).resolve(); work.mkdir(parents=True, exist_ok=True)
    p1, p2, p3, p4 = [work / x for x in ["01_summaries","02_encoder","03_semantic","04_lambda_hdbscan"]]
    p5, p6, p7, p8 = [work / x for x in ["05_representation_benchmark","06_pro_burden","07_clinical_validation","08_sensitivities"]]

    run("01_build_clinical_summaries.py", "--input", DATA, "--thresholds", THRESHOLDS, "--outdir", p1)

    adapted = Path(args.adapted_model).resolve() if args.adapted_model else p2
    if args.adapted_model is None:
        command = ["--summaries", p1 / "clinical_summaries.jsonl", "--outdir", p2]
        if args.device: command += ["--device", args.device]
        run("02_adapt_encoder.py", *command)

    command = ["--summaries", p1 / "clinical_summaries.jsonl", "--adapted-model", adapted, "--outdir", p3]
    if args.device: command += ["--device", args.device]
    run("03_build_semantic_representation.py", *command)

    primary_resamples, sensitivity_resamples = (3, 1) if args.quick else (50, 10)
    run("04_select_lambda_and_cluster.py", "--cohort", DATA, "--embeddings", p3, "--outdir", p4,
        "--primary-resamples", primary_resamples, "--sensitivity-resamples", sensitivity_resamples)

    stability = 3 if args.quick else 50
    run("05_benchmark_representations.py", "--data", DATA, "--ordinal-labels", p1 / "construction_pro_ordinal_labels.csv",
        "--semantic-dir", p3 / "adapted", "--selection-dir", p4, "--outdir", p5, "--stability-resamples", stability)

    repeats = 1 if args.quick else 10
    run("06_construct_pro_burden.py", "--data", DATA, "--ordinal-labels", p1 / "construction_pro_ordinal_labels.csv",
        "--semantic-dir", p3 / "adapted", "--outdir", p6, "--repeats", repeats)

    run("07_validate_clinically.py", "--data", DATA, "--ordinal-labels", p1 / "construction_pro_ordinal_labels.csv",
        "--burden-dir", p6, "--semantic-root", p3, "--outdir", p7, "--repeats", repeats)

    sensitivity_args = ["--data", DATA, "--ordinal-labels", p1 / "construction_pro_ordinal_labels.csv",
        "--cluster-dir", p4, "--benchmark-dir", p5, "--outdir", p8, "--repeats", repeats]
    if not args.quick: sensitivity_args += ["--include-full-lambda-grid"]
    run("08_run_sensitivity_analyses.py", *sensitivity_args)

    print(f"\nSynthetic workflow complete. Outputs: {work}")


if __name__ == "__main__":
    main()
