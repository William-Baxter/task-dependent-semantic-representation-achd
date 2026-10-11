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

## Core implementation choices

The clinical summary is deterministic: source variables are mapped to fixed descriptors and rendered in a fixed field order. Missing fields are omitted rather than numerically imputed during text rendering. The eleven PedsQL construction domains use the fixed thresholds in `config/pedsql_rendering_thresholds.csv`; DASS-21 depression, anxiety and stress use ordered severity classes. These descriptors are representation-construction labels, not clinical severity categories.

Encoder adaptation is outcome-blind and is used only for conservative schema familiarisation. Participants are split before augmentation (seed 123; 70%/15%/15%), CHD descriptors are never masked, and training combines positive-pair cosine similarity with mean-squared-error anchoring to the frozen base-model embedding. Training runs for one epoch with the final transformer block, transformer normalisation parameters and pooler (where present) trainable; other transformer parameters remain frozen. The unadapted encoder is retained for the adaptation ablation.

The semantic representation uses `BAAI/bge-large-en-v1.5`. Context and PRO summaries are embedded separately and L2-normalised before being combined as

```text
z_i(lambda) = [sqrt(1-lambda) c_i, sqrt(lambda) p_i].
```

The complete 41-point grid from 0 to 1 in increments of 0.025 is evaluated. The primary HDBSCAN specification uses minimum cluster size 40, minimum samples 10 and leaf extraction. λ is selected without using lesion coherence/composition, burden labels or validation outcomes as optimisation criteria: coverage, exact-lesion-stratified resampling stability and cluster-size-weighted persistence are min-max normalised, Pareto-nondominated candidates are identified, and the candidate closest to the ideal point is selected. Exact CHD category is used only to stratify the stability resamples. The complete selection procedure is also repeated over 18 HDBSCAN specifications.

The matched representation benchmark compares Semantic, direct Tabular and Gower constructions using the same five information sets. Transparent preprocessing is refitted in each stability resample. A separate fixed-K analysis compares Semantic and Tabular spaces using K-means and Ward clustering for K=2–12.

Patient-reported burden is constructed for all eligible participants within an 11-family lesion scaffold. Four formulations are evaluated independently: Semantic PRO, direct ordinal Tabular PRO, a continuous PedsQL-minus-DASS composite and PC1. K=2 burden construction and continuous burden-axis orientation are fitted within lesion family. Repeated out-of-fold reconstruction uses five folds repeated ten times with all direct preprocessing fitted on training participants only.

Clinical outcomes are withheld from summary construction, encoder adaptation, λ selection, clustering and burden derivation. Clinical evaluation uses available outcomes only. Binary performance is AUROC and continuous performance is R², calculated after pooling the five held-out folds within each repeat. The principal full-summary probes compare joint semantic encoding, separately encoded semantic context and PRO views with equal squared-distance weighting, an additive tabular representation of the same ten context plus 14 ordinal PRO descriptors, and anatomy plus the 14 raw continuous PRO scores. Base and adapted semantic encoders are both evaluated as an adaptation sensitivity. Inner five-fold cross-validation selects L2-logistic C or ridge α over 10^-4 to 10^4.

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
