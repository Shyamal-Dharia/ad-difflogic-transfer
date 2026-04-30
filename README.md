# AD DiffLogic Transfer

This repository contains code for lightweight EEG-based cross-dataset transfer learning using Differentiable Logic Gate Networks (DiffLogic).

The project trains a binary Alzheimer's disease (AD) versus cognitively normal control model on a clinical resting-state EEG dataset, then applies the trained model directly to the PEARL dataset without retraining. In PEARL, the model output is interpreted as an AD-like EEG score and compared across genetic-risk groups.

This code is intended for research reproducibility and method development. The model outputs should not be interpreted as clinical diagnoses.

## Overview

The pipeline includes:

- common EEG preprocessing for ALZ_FTD and PEARL;
- Welch power spectral density (PSD) feature extraction;
- relative bandpower conversion and thermometer encoding;
- subject-stratified DiffLogic training on AD versus control;
- direct transfer inference on PEARL;
- PEARL group-level summaries and covariate checks;
- model interpretation with Grad x Input and Integrated Gradients.

## Repository Structure

```text
src/
  preprocessing.py                 # EEG preprocessing
  feature_extraction_psd.py         # Welch PSD extraction
  statistical_analysis_psd.py       # subject-level PSD statistics
  train_helpers.py                  # feature loading, folds, encoding
  difflogic_model.py                # DiffLogic model definitions
  train_difflogic.py                # model training and PEARL transfer inference
  summarize_difflogic_runs.py       # multi-seed summary tables
  analyze_pearl_covariates.py       # age/sex adjusted PEARL analysis
  interpret_difflogic_gradients.py  # model interpretation

notes/
  PREPROCESSING_RATIONALE.md
  TRAINING_RATIONALE.md
  DIFFLOGIC_MEDIUM_RESULTS.md
  DIFFLOGIC_INTERPRETATION.md
```

Large local data and generated outputs are not tracked by git:

```text
datasets/
outputs/
```

## Installation

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

DiffLogic uses PyTorch. For GPU support, install the PyTorch build that matches the local CUDA version before installing the remaining requirements.

## Data Layout

Place the datasets in the following structure:

```text
datasets/
  ALZ_FTD/
  PEARL/
```

The scripts expect BIDS-like subject folders under each dataset. Large raw data files are not included in this repository.

The preprocessing uses the strict 15-channel overlap between datasets:

```text
Fp1, Fp2, F7, F3, Fz, F4, F8, C3, Cz, C4, P3, Pz, P4, O1, O2
```

Both datasets are restricted to these channels before average referencing.

## Running the Pipeline

Run commands from the repository root. Use `PYTHONPATH=src` so the scripts can import local modules.

### 1. Preprocess EEG

```bash
PYTHONPATH=src python src/preprocessing.py
```

This creates:

```text
datasets/processed/ALZ_FTD/
datasets/processed/PEARL/
```

### 2. Extract PSD Features

```bash
PYTHONPATH=src python src/feature_extraction_psd.py
```

This creates:

```text
datasets/features_psd/ALZ_FTD/
datasets/features_psd/PEARL/
```

### 3. Run PSD Statistics

```bash
PYTHONPATH=src python src/statistical_analysis_psd.py
```

This creates:

```text
datasets/statistics_psd/
```

### 4. Train DiffLogic Models

Example medium-size model with 5 independent seeds:

```bash
for seed in 1 2 3 4 5; do
  PYTHONPATH=src python src/train_difflogic.py \
    --model-size medium \
    --output-name medium_interpretable \
    --seed $seed \
    --epochs 200 \
    --patience 50
done
```

This creates:

```text
outputs/difflogic/medium_interpretable/
```

### 5. Summarize Multi-Seed Results

```bash
PYTHONPATH=src python src/summarize_difflogic_runs.py \
  --model-size medium_interpretable
```

This creates:

```text
outputs/difflogic/medium_interpretable/summary/
```

### 6. Run Age/Sex Adjustment

```bash
PYTHONPATH=src python src/analyze_pearl_covariates.py \
  --summary-dir outputs/difflogic/medium_interpretable/summary
```

### 7. Interpret the Trained Model

Grad x Input:

```bash
PYTHONPATH=src python src/interpret_difflogic_gradients.py \
  --run-name medium_interpretable \
  --model-size medium \
  --method grad_x_input \
  --logic-mode soft \
  --batch-size 512
```

Integrated Gradients:

```bash
PYTHONPATH=src python src/interpret_difflogic_gradients.py \
  --run-name medium_interpretable \
  --model-size medium \
  --method integrated_gradients \
  --logic-mode soft \
  --ig-steps 16 \
  --batch-size 512
```

Interpretation outputs are saved under:

```text
outputs/difflogic/medium_interpretable/interpretation/
```

## Main Outputs

Key generated files include:

```text
outputs/difflogic/<run_name>/summary/pearl_subject_predictions_seed_average.csv
outputs/difflogic/<run_name>/summary/pearl_group_summary_seed_average.csv
outputs/difflogic/<run_name>/summary/pearl_age_sex_adjusted_ols.csv
outputs/difflogic/<run_name>/interpretation/soft_grad_x_input/
outputs/difflogic/<run_name>/interpretation/soft_integrated_gradients/
```

## Notes

The analysis is designed around subject-level evaluation to avoid epoch-level leakage. PEARL transfer scores are interpreted as AD-like EEG scores for genetic-risk stratification, not as diagnostic labels.

