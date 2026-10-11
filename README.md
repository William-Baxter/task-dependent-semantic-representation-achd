# Task-dependent semantic representation in adult congenital heart disease

This repository contains the code and synthetic example data accompanying the manuscript:

**Task-dependent value of transformer representations for phenotyping and patient-reported burden in adult congenital heart disease**

William Baxter et al.

*npj Digital Medicine*

---

The repository provides a registry-independent implementation of the computational analyses used to evaluate semantic patient representation, phenotype structure, patient-reported burden, and held-out clinical relevance in adults with congenital heart disease.

No participant-level National Congenital Heart Disease Registry data are distributed. The included synthetic cohort is entirely artificial and is provided only to demonstrate the workflow and required input schema. Numerical results reported in the manuscript require access to the governed study data.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── LICENSE_NOTICE.md
├── run_synthetic_example.py
├── config/
│   └── pedsql_rendering_thresholds.csv
├── synthetic_data/
│   ├── README.md
│   └── synthetic_cohort.csv
└── scripts/
    ├── common.py
    ├── 01_build_clinical_summaries.py
    ├── 02_adapt_encoder.py
    ├── 03_build_semantic_representation.py
    ├── 04_select_lambda_and_cluster.py
    ├── 05_benchmark_representations.py
    ├── 06_construct_pro_burden.py
    ├── 07_validate_clinically.py
    └── 08_run_sensitivity_analyses.py
```

## Methods-to-code crosswalk

| Methods component | Public implementation |
| --- | --- |
| Preprocessing and deterministic clinical summaries | `01_build_clinical_summaries.py` |
| Sentence encoder and conservative schema adaptation | `02_adapt_encoder.py` |
| Two-view semantic representation and geometric calibration | `03_build_semantic_representation.py` |
| Outcome-independent selection of the relative context–PRO balance | `04_select_lambda_and_cluster.py` |
| HDBSCAN specification sensitivity and assignment uncertainty | `04_select_lambda_and_cluster.py` |
| Matched semantic, tabular and Gower representation benchmarks | `05_benchmark_representations.py` |
| Fixed-K K-means and Ward benchmarks | `05_benchmark_representations.py` |
| Full-cohort within-lesion PRO burden construction | `06_construct_pro_burden.py` |
| External characteristics of burden | `06_construct_pro_burden.py` |
| Construction-PRO versus validator Spearman audit | `07_validate_clinically.py` |
| Adjusted higher- versus lower-burden clinical associations | `07_validate_clinically.py` |
| Repeated out-of-fold burden and full-summary prediction | `07_validate_clinically.py` |
| Lesion-conditioned median-split benchmark | `08_run_sensitivity_analyses.py` |
| Clinical evaluation of the matched HDBSCAN partitions | `08_run_sensitivity_analyses.py` |
| Robustness across the supported λ interval | `08_run_sensitivity_analyses.py` |

## Installation

The reported computational stack used Python 3.12.10. A clean environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first use of `BAAI/bge-large-en-v1.5` may download model weights from the model provider. GPU acceleration is optional but strongly recommended for encoder adaptation and embedding.

## Synthetic worked example

The end-to-end driver uses only the artificial cohort included here:

```bash
python run_synthetic_example.py --workdir synthetic_run
```

This executes the full scientific defaults. It includes encoder adaptation and can therefore take substantial time on CPU. For a software smoke test with fewer resamples/repeats:

```bash
python run_synthetic_example.py --workdir synthetic_run_quick --quick
```

An existing compatible adapted encoder may instead be supplied:

```bash
python run_synthetic_example.py --workdir synthetic_run --adapted-model /path/to/model
```

Individual scripts expose their arguments with `--help` and can be run independently.

## Expected study-data schema

The synthetic CSV documents the expected column names. The representation-construction fields comprise:

- ten context descriptors: sex, age band, remoteness, medication-burden category, comorbidity-burden category, diabetes, hypertension, asthma, CHD complexity and exact CHD category;
- eleven continuous PedsQL construction scores and their deterministic ordinal descriptors;
- DASS-21 depression, anxiety and stress scores/classes.

Validation outcomes are carried in separate columns and are never used upstream of the clinical-evaluation scripts. `record_id` is used only for deterministic alignment across artefacts.

## Reproducibility notes

Analysis defaults used for the manuscript are encoded directly in the numbered scripts. Command-line options intended for synthetic-data smoke testing are explicitly labelled and do not alter the manuscript specification unless invoked.

## Data availability

The public code is designed so that the complete methodological workflow can be inspected and executed without access to Registry participant data. Governed patient-level data are not included and must not be added to this repository. The synthetic cohort contains no real participant records and no transformed, perturbed or resampled patient-level observations.
