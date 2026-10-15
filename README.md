# Task-dependent semantic representation in adult congenital heart disease

This repository contains the code and synthetic example data accompanying the manuscript:

**Transformer representations have task-dependent value for anatomy and patient-reported burden in adult congenital heart disease**

William Baxter et al.

*npj Digital Medicine*

## Overview

The repository provides a Registry-independent implementation of the computational analyses used to evaluate semantic patient representation, phenotype structure and patient-reported burden, together with held-out clinical validation in adults with congenital heart disease.

## Code

The analysis is organised in the same sequence as the Methods:

| Script | Analysis |
| --- | --- |
| `01_build_clinical_summaries.py` | Deterministic clinical-summary construction |
| `02_adapt_encoder.py` | Conservative encoder schema adaptation |
| `03_build_semantic_representation.py` | Context/PRO semantic representation |
| `04_select_lambda_and_cluster.py` | Relative view weighting and HDBSCAN selection |
| `05_benchmark_representations.py` | Semantic, tabular and Gower benchmarks |
| `06_construct_pro_burden.py` | Within-lesion patient-reported burden |
| `07_validate_clinically.py` | Held-out clinical validation |
| `08_run_sensitivity_analyses.py` | Sensitivity analyses |

## Installation

The reported computational stack used Python 3.12.10:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

The first use of [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5) may download the base-model weights from the model provider.

## Synthetic example

A complete demonstration using only artificial data can be run with:

```bash
python run_synthetic_example.py --workdir synthetic_run
```

For a shorter software smoke test:

```bash
python run_synthetic_example.py --workdir synthetic_run_quick --quick
```

The synthetic example performs encoder adaptation using the artificial summaries included with this repository before running the downstream analyses. It therefore demonstrates the reported adaptation procedure and wider computational workflow, but does not reproduce the study-trained encoder or the numerical results of the study.

Individual scripts expose their arguments with `--help` and can be run independently.

## Data availability

Participant-level study data are governed by the National Congenital Heart Disease Registry and are not included in this repository. The adapted encoder used for the study analyses was fitted using summaries derived from these governed data and is not distributed. The public implementation instead provides the complete adaptation procedure together with an entirely artificial cohort on which it can be executed independently. No real participant records, transformed participant records or resampled participant-level observations are contained in the synthetic dataset.
