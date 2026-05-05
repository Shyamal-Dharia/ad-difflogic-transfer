# Cross-Cohort AD EEG DiffLogic Project Details

This document summarizes the current repository state, hypotheses, processing pipeline, model design, controls, and main local results for:

**Cross-Cohort Transfer of Alzheimer's Disease EEG Signatures Using Differentiable Logic Gate Networks for Preclinical Screening**

The summary is based on the local repository at `/data/s.dharia-ra/difflogic_AD_PEARL` and the currently generated outputs under `datasets/` and `outputs/`.

## High-Level Hypothesis

The central hypothesis is that a lightweight classifier trained to distinguish clinical Alzheimer's disease (AD) from cognitively normal controls (CN) using harmonized resting-state EEG can transfer directly to a preclinical genetic-risk cohort and produce a meaningful continuous AD-like EEG score.

Specific working hypotheses:

1. A DiffLogic model trained on clinical AD vs CN EEG will learn spectral signatures associated with clinical AD, especially AD-related oscillatory slowing.
2. When transferred to PEARL-Neuro without retraining, dual-risk participants (`A+P+`) will show higher AD-like EEG scores than neutral-risk (`N`) and single-risk (`A+P-`) participants.
3. The transfer effect should depend on true clinical supervision. Shuffling AD/CN labels should destroy both clinical performance and meaningful PEARL group separation.
4. PSD-based log-relative bandpower should be the primary interpretable signal, because AD EEG changes are commonly expressed as shifts in oscillatory power.
5. HFD features provide a nonlinear feature-family control, but should be treated as secondary to PSD unless they reproduce the same transfer pattern.
6. DiffLogic feature interpretation should identify channel-band features contributing to the AD-like score, with theta-band features expected to be prominent.

## Repository Layout

Core source files:

| Path | Role |
| --- | --- |
| `src/preprocessing.py` | Common EEG preprocessing for ALZ_FTD and PEARL. |
| `src/feature_extraction_psd.py` | Welch PSD and log-relative bandpower feature extraction. |
| `src/feature_extraction_hfd.py` | Higuchi fractal dimension feature extraction. |
| `src/train_helpers.py` | Dataset loading, labels, subject folds, scaling, thermometer encoding. |
| `src/difflogic_model.py` | DiffLogic model sizes and GroupSum architecture. |
| `src/train_difflogic.py` | Clinical training and PEARL transfer inference. |
| `src/summarize_difflogic_runs.py` | Multi-seed summaries. |
| `src/analyze_pearl_covariates.py` | PEARL age/sex balance and adjusted OLS. |
| `src/interpret_difflogic_gradients.py` | Grad x Input and Integrated Gradients attribution. |
| `src/model_relevance_statistics.py` | Statistical tests over model relevance outputs. |
| `src/focused_psd_statistics.py` | PSD follow-up tests on model-relevant features. |
| `src/theta_psd_group_statistics.py` | Focused theta-band pairwise tests. |
| `src/make_publication_figures_v2.py` | Publication figure/table generation. |
| `src/plot_common_channel_montage.py` | MNE-based 15-channel montage figure. |

Major output locations:

| Path | Contents |
| --- | --- |
| `datasets/processed/` | Cleaned 4-s EEG epochs per subject. |
| `datasets/features_psd/` | PSD feature files per subject. |
| `datasets/features_hfd/` | HFD feature files per subject. |
| `datasets/statistics_psd/` | Broad PSD subject-level statistics. |
| `outputs/difflogic/medium_interpretable/` | Primary PSD DiffLogic run used for interpretation. |
| `outputs/difflogic/medium_shuffled_alz_labels/` | Shuffled-label negative control. |
| `outputs/difflogic/hfd_kmax16_medium/` | HFD control with `kmax=16`. |
| `outputs/difflogic/hfd_kmaxnone_medium/` | HFD control with `kmax=None`. |
| `outputs/model_relevance_statistics/` | Integrated Gradient relevance tests. |
| `outputs/focused_psd_statistics/` | PSD statistics for attribution-selected features. |
| `outputs/theta_psd_group_statistics/` | Theta-specific pairwise PSD tests. |
| `outputs/publication_figures_v2/` | Current summary table and figure assets. |

## Datasets

### Clinical ALZ_FTD Source Dataset

The clinical dataset contains closed-eyes resting-state EEG from:

| Group | Subjects | Age, mean +/- SD | Sex | Clean epochs |
| --- | ---: | ---: | --- | ---: |
| AD (`A`) | 36 | 66.4 +/- 7.9 | F: 24, M: 12 | 6,329 |
| CN (`C`) | 29 | 67.9 +/- 5.4 | F: 11, M: 18 | 5,538 |
| FTD (`F`) | 23 | 63.7 +/- 8.2 | F: 9, M: 14 | 3,464 |
| Total | 88 | 66.2 +/- 7.4 | F: 44, M: 44 | 15,331 |

Only AD and CN subjects are used for supervised model training and clinical test evaluation. FTD subjects are excluded from the binary AD vs CN task.

Clinical model sample:

| Group | Subjects | Clean epochs | Epochs per subject |
| --- | ---: | ---: | ---: |
| AD | 36 | 6,329 | 175.8 +/- 45.1 |
| CN | 29 | 5,538 | 191.0 +/- 17.2 |
| Total | 65 | 11,867 | - |

### PEARL-Neuro Target Dataset

PEARL-Neuro contains cognitively healthy, middle-aged participants grouped by genetic risk. In this project the groups are:

| Group | Meaning in analysis |
| --- | --- |
| `N` | Neutral risk group. |
| `A+P-` | APOE-related risk present, PICALM risk absent. |
| `A+P+` | APOE-related risk present, PICALM risk present. |

Analyzed PEARL sample:

| Group | Subjects | Age, mean +/- SD | Sex coding in local metadata | Clean epochs |
| --- | ---: | ---: | --- | ---: |
| `N` | 31 | 54.8 +/- 2.9 | 0: 16, 1: 15 | 2,676 |
| `A+P-` | 26 | 55.5 +/- 3.2 | 0: 13, 1: 13 | 2,274 |
| `A+P+` | 20 | 55.6 +/- 3.4 | 0: 10, 1: 10 | 1,727 |
| Total | 77 | 55.2 +/- 3.1 | 0: 39, 1: 38 | 6,677 |

The PEARL output is never interpreted as a clinical diagnosis. It is treated as a continuous AD-like EEG score obtained by direct transfer from the clinical model.

## Preprocessing and Harmonization

The preprocessing principle is strict cross-cohort comparability.

Common EEG channels:

```python
COMMON_CHANNELS = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "C3", "Cz", "C4",
    "P3", "Pz", "P4",
    "O1", "O2",
]
```

Important harmonization choices:

- Select the strict 15-channel overlap first.
- Apply common average reference only after channel selection.
- Compute the average reference over the same 15 channels in both cohorts.
- Do not use dataset-specific electrodes for referencing or feature extraction.
- Do not remap missing temporal channels such as `T7/T8/P7/P8` to old `T3/T4/T5/T6` labels.

Preprocessing parameters:

| Step | Parameter |
| --- | --- |
| Segment | Closed-eyes resting-state EEG |
| PEARL crop | Eyes-closed segment isolated from events |
| Resampling | 250 Hz |
| Notch filter | 50 Hz |
| Bandpass | 1-45 Hz |
| Reference | Common average over the 15 retained channels |
| Epoching | 4-s non-overlapping epochs |
| Epoch rejection | Peak-to-peak amplitude > 150 uV in any retained channel |

Rationale:

- Selecting common channels before average reference makes both cohorts undergo the same linear channel transformation.
- Relative power features and common referencing reduce dataset-specific amplitude differences from montage, amplifier, and impedance effects.
- The goal is not perfect hardware equivalence, but a standardized spatial and spectral representation suitable for zero-shot transfer.

## Feature Extraction

### Primary PSD Features

PSD features are the main representation because AD resting-state EEG is commonly associated with spectral slowing: increased low-frequency power and altered alpha/beta activity.

Parameters:

| Feature step | Value |
| --- | --- |
| PSD method | Welch |
| Window | 2-s Hann |
| Overlap | 1 s |
| Frequency range | 1-45 Hz |
| Normalization | Relative PSD per epoch/channel, normalized by total 1-45 Hz power |
| Band features | delta, theta, alpha, beta, gamma |
| Transform | `log10(relative bandpower + EPS)` |
| EPS in code | `1e-20` |
| Feature shape | epochs x 15 channels x 5 bands |

Band definitions:

| Band | Range |
| --- | --- |
| delta | 1-4 Hz |
| theta | 4-8 Hz |
| alpha | 8-13 Hz |
| beta | 13-30 Hz |
| gamma | 30-45 Hz |

Rationale for relative PSD:

- It reduces dependence on absolute signal amplitude.
- It makes features represent the distribution of power across frequency rather than raw magnitude.
- It is better suited to cross-cohort transfer across different EEG systems.

Rationale for log transform:

- EEG power is right-skewed.
- Log scaling compresses large multiplicative differences.
- It stabilizes feature scale before thermometer encoding and DiffLogic training.

### HFD Feature-Family Controls

Higuchi fractal dimension (HFD) is used as a nonlinear signal-complexity control. HFD checks whether the transfer pattern is specific to interpretable spectral oscillations or can also be reproduced by a complexity-based representation.

HFD runs:

- `hfd_kmax16_medium`: HFD with `kmax=16`.
- `hfd_kmaxnone_medium`: HFD with `kmax=None`.

PSD remains the primary analysis because it is directly interpretable in terms of canonical EEG bands and AD-related spectral slowing.

## Thermometer Encoding

DiffLogic expects binary inputs. Continuous log-relative bandpower features are converted to binary thermometer codes.

Procedure:

1. Fit min-max scaling on clinical training epochs only.
2. Apply the same scaler to validation, clinical test, and all PEARL epochs.
3. Clip scaled values to `[0, 1]`.
4. Expand each continuous feature into 15 thermometer bins.

Implementation details:

- Number of original PSD features per epoch: `15 channels x 5 bands = 75`.
- Thermometer bins per feature: `15`.
- Encoded input dimension: `75 x 15 = 1125`.
- Thresholds in code: `np.linspace(1.0 / n_bins, 1.0, n_bins)`, so for 15 bins the thresholds are `1/15, 2/15, ..., 1`.

Rationale:

- Thermometer encoding preserves feature order.
- Adjacent feature values activate similar binary patterns.
- This is more appropriate than one-hot encoding for continuous EEG features.

## DiffLogic Model

Primary model:

| Component | Value |
| --- | --- |
| Model size | Medium |
| Input dimension | 1125 |
| Logic layers | 4 |
| Width | 1600 |
| Output | GroupSum with 2 classes |
| Classes | CN and AD |
| Temperature | `tau=30` |
| Optimizer | Adam |
| Learning rate | 0.01 |
| Batch size | 256 |
| Max epochs | 100 |
| Early stopping | 15 epochs without validation loss improvement |
| Loss | Class-weighted cross entropy |

Training setup:

- Train only on clinical AD vs CN subjects.
- Exclude FTD from training and test.
- Use 10 subject-stratified folds.
- Use 80/20 train/validation split inside each training fold.
- Repeat across 5 independent seeds.
- Evaluate at subject level by averaging epoch-level probabilities.
- Apply each trained model directly to all PEARL subjects without retraining.

Important interpretation of model output:

- Clinical evaluation: binary AD vs CN classification.
- PEARL inference: continuous AD-like EEG score, not diagnosis.

## AD-Like EEG Score

For each PEARL epoch, the clinical model returns an AD probability. For each PEARL subject, epoch probabilities are averaged:

```text
subject score = mean(epoch AD probabilities)
```

This is repeated across fold-specific models and seeds. The final score is the seed/fold-averaged subject-level AD-like EEG score.

Interpretation:

- Higher score means the subject's EEG features are more similar to the clinical AD pattern learned from ALZ_FTD.
- It does not mean the subject has AD.
- It is a transfer-derived screening/research score.

## Primary Results

### Clinical AD vs CN Performance and PEARL Transfer Scores

From `outputs/publication_figures_v2/performance_summary_table.csv`:

| Analysis | Balanced accuracy | AUROC | Median PEARL N | Median PEARL A+P- | Median PEARL A+P+ |
| --- | ---: | ---: | ---: | ---: | ---: |
| PSD | 0.807 +/- 0.155 | 0.862 +/- 0.152 | 0.308 | 0.307 | 0.430 |
| PSD without Fp1/Fp2 | 0.773 +/- 0.165 | 0.844 +/- 0.167 | 0.307 | 0.277 | 0.421 |
| PSD shuffled labels | 0.459 +/- 0.164 | 0.426 +/- 0.231 | 0.514 | 0.512 | 0.515 |
| HFD kmax=16 | 0.631 +/- 0.200 | 0.744 +/- 0.209 | 0.524 | 0.480 | 0.542 |
| HFD kmax=None | 0.585 +/- 0.183 | 0.713 +/- 0.226 | 0.530 | 0.492 | 0.514 |

Primary interpretation:

- PSD DiffLogic performs well on clinical AD vs CN.
- PEARL `A+P+` has the highest median AD-like score.
- Shuffled labels reduce clinical performance to near chance and flatten PEARL group medians.
- HFD controls are weaker and do not reproduce the same clean PSD pattern.
- The no-Fp1/Fp2 run exists as a sensitivity output and shows some performance reduction but a similar PEARL median ordering. If the manuscript is not including the no-Fp1/Fp2 sensitivity analysis, omit it from the paper table.

### Primary PSD Run: Detailed PEARL Group Summary

From `outputs/difflogic/medium_interpretable/summary/pearl_group_summary_seed_average.csv`:

| Group | Subjects | Mean AD-like score | SD | Median | Predicted AD-rate at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A+P+` | 20 | 0.417 | 0.154 | 0.430 | 0.350 |
| `A+P-` | 26 | 0.319 | 0.123 | 0.307 | 0.115 |
| `N` | 31 | 0.331 | 0.136 | 0.308 | 0.097 |

Group tests from `pearl_group_tests_seed_average.csv`:

| Comparison | Test | p-value | Medians |
| --- | --- | ---: | --- |
| `N` vs `A+P-` vs `A+P+` | Kruskal-Wallis | 0.0738 | - |
| `N` vs `A+P-` | Mann-Whitney U | 0.8039 | 0.308 vs 0.307 |
| `N` vs `A+P+` | Mann-Whitney U | 0.0600 | 0.308 vs 0.430 |
| `A+P-` vs `A+P+` | Mann-Whitney U | 0.0325 | 0.307 vs 0.430 |

Interpretation:

- The strongest unadjusted pairwise separation is `A+P+` vs `A+P-`.
- `A+P+` vs `N` is borderline in the unadjusted Mann-Whitney test.
- The global Kruskal-Wallis test is trend-level rather than conventionally significant.

### Age/Sex Adjusted OLS

From `outputs/difflogic/medium_interpretable/summary/pearl_age_sex_adjusted_ols.csv`:

| Contrast | Coefficient | 95% CI | p-value |
| --- | ---: | --- | ---: |
| `A+P+` vs `N` | 0.0885 | [0.0031, 0.1738] | 0.0423 |
| `A+P-` vs `N` | -0.0095 | [-0.0776, 0.0586] | 0.7851 |
| `A+P+` vs `A+P-` | 0.0979 | [0.0123, 0.1835] | 0.0250 |
| Age centered | -0.0018 | [-0.0116, 0.0080] | 0.7202 |
| Sex | -0.0565 | [-0.1201, 0.0070] | 0.0814 |

Interpretation:

- After age/sex adjustment, `A+P+` has significantly higher AD-like scores than both `N` and `A+P-`.
- Age is not associated with the AD-like score in this model.
- Sex is not conventionally significant but has a trend-level coefficient.

## Negative and Feature-Family Controls

### Shuffled Clinical Labels

From `outputs/difflogic/medium_shuffled_alz_labels/summary/`:

| Metric | Value |
| --- | ---: |
| Balanced accuracy | 0.459 +/- 0.164 |
| AUROC | 0.426 +/- 0.231 |
| PEARL median `N` | 0.514 |
| PEARL median `A+P-` | 0.512 |
| PEARL median `A+P+` | 0.515 |

Interpretation:

- Shuffling labels collapses clinical performance toward chance.
- PEARL scores cluster around 0.5 and lose meaningful group ordering.
- This supports that the PSD transfer pattern depends on real clinical AD/CN supervision.

### HFD Controls

HFD `kmax=16`:

| Metric | Value |
| --- | ---: |
| Balanced accuracy | 0.631 +/- 0.200 |
| AUROC | 0.744 +/- 0.209 |
| PEARL median `N` | 0.524 |
| PEARL median `A+P-` | 0.480 |
| PEARL median `A+P+` | 0.542 |

HFD `kmax=None`:

| Metric | Value |
| --- | ---: |
| Balanced accuracy | 0.585 +/- 0.183 |
| AUROC | 0.713 +/- 0.226 |
| PEARL median `N` | 0.530 |
| PEARL median `A+P-` | 0.492 |
| PEARL median `A+P+` | 0.514 |

Interpretation:

- HFD models show weaker clinical performance than PSD.
- HFD transfer medians do not reproduce the same clear `A+P+` elevation seen with PSD.
- This supports the choice of log-relative PSD as the primary interpretable representation.

## DiffLogic Feature Interpretation

Primary attribution method:

- Integrated Gradients.
- Run directory: `outputs/difflogic/medium_interpretable/interpretation/soft_integrated_gradients/`.
- Target score: AD evidence, defined as `logit_AD - logit_CN`.
- Thermometer-bin relevance is collapsed back to channel x band by summing across the 15 bins for each original PSD feature.
- Epoch relevance maps are averaged within subject and summarized by PEARL group.

### Top Integrated Gradient Contrasts

Largest signed relevance differences for `A+P+ - N`:

| Channel | Band | A+P+ | N | Difference |
| --- | --- | ---: | ---: | ---: |
| F7 | theta | -0.192 | -0.217 | 0.0248 |
| F8 | theta | -0.0099 | -0.0311 | 0.0211 |
| P3 | theta | -0.0204 | -0.0410 | 0.0206 |
| O2 | alpha | -0.193 | -0.211 | 0.0182 |
| P4 | theta | -0.0276 | -0.0446 | 0.0170 |
| F4 | theta | 0.0586 | 0.0429 | 0.0158 |
| F3 | theta | 0.0624 | 0.0470 | 0.0154 |
| C4 | theta | 0.0786 | 0.0649 | 0.0137 |
| Fz | theta | 0.0799 | 0.0663 | 0.0136 |
| C3 | theta | 0.0985 | 0.0858 | 0.0128 |

Largest signed relevance differences for `A+P+ - A+P-`:

| Channel | Band | A+P+ | A+P- | Difference |
| --- | --- | ---: | ---: | ---: |
| O2 | delta | 0.311 | 0.274 | 0.0374 |
| O2 | alpha | -0.193 | -0.216 | 0.0238 |
| F7 | theta | -0.192 | -0.215 | 0.0235 |
| Fz | theta | 0.0799 | 0.0588 | 0.0211 |
| P4 | theta | -0.0276 | -0.0457 | 0.0181 |
| Fp2 | alpha | -0.124 | -0.141 | 0.0170 |
| F8 | theta | -0.0099 | -0.0264 | 0.0165 |
| F3 | theta | 0.0624 | 0.0473 | 0.0152 |
| F4 | theta | 0.0586 | 0.0442 | 0.0144 |
| F4 | alpha | -0.0332 | -0.0470 | 0.0138 |

Interpretation:

- Theta features dominate the largest attribution contrasts.
- Frontal, central, and parietal theta channels recur across `A+P+ - N` and `A+P+ - A+P-`.
- Some alpha/delta features, especially O2 alpha/delta, also appear in attribution contrasts.

### Relevance Statistical Testing

From `outputs/model_relevance_statistics/integrated_gradient_relevance_tests.csv`:

- Total tests: 225.
- Uncorrected `p < 0.05`: 2.
- FDR-significant at 0.05: 0.

Top uncorrected finding:

| Feature | Comparison | Median difference | p uncorrected | p FDR |
| --- | --- | ---: | ---: | ---: |
| Pz theta | `A+P+` vs `A+P-` | 0.0175 | 0.0275 | 0.7321 |

Interpretation:

- Attribution statistics are exploratory.
- No relevance feature survives FDR correction.
- The strongest attribution pattern is still biologically coherent because theta features recur across the highest-ranked contrasts.

## Model-Relevant PSD Follow-Up

The focused PSD analysis uses model-relevant channel-band features selected from integrated-gradient contrasts.

Feature list from `outputs/focused_psd_statistics/model_relevant_channel_band_features.csv`:

```text
C3 theta
C4 theta
Cz beta
F3 theta
F4 alpha
F4 theta
F7 gamma
F7 theta
F8 theta
Fp1 beta
Fp2 alpha
Fz alpha
Fz theta
O1 alpha
O2 alpha
O2 delta
O2 theta
P3 theta
P4 theta
Pz theta
```

Focused PSD statistical results from `outputs/focused_psd_statistics/focused_psd_tests.csv`:

- Total tests: 140.
- Uncorrected `p < 0.05`: 0.
- FDR-significant at 0.05: 0.

Top global Kruskal-Wallis tests:

| Feature | p uncorrected | p FDR |
| --- | ---: | ---: |
| O2 delta | 0.1499 | 0.6334 |
| Fp1 beta | 0.1583 | 0.6334 |
| Fp2 alpha | 0.1731 | 0.6334 |
| Pz theta | 0.2425 | 0.6334 |
| O2 alpha | 0.2430 | 0.6334 |

Interpretation:

- Independent PSD tests on attribution-selected features do not reach significance.
- Directionally, several theta features show higher log-relative theta values in `A+P+` than other groups, but this remains exploratory.

## Theta-Band Interpretation

Theta was examined because:

- AD EEG slowing often includes increased low-frequency activity.
- Integrated Gradients repeatedly highlighted theta features in `A+P+` contrasts.
- Theta features appeared across frontal, central, and parietal channels.

Theta pairwise tests from `outputs/theta_psd_group_statistics/theta_psd_pairwise_tests.csv`:

- Total theta pairwise tests: 45.
- Uncorrected `p < 0.05`: 0.
- FDR-significant at 0.05: 0.

Best `A+P+` vs `N` theta differences:

| Channel | Median `A+P+` | Median `N` | Difference | p uncorrected | p FDR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pz | -0.935 | -1.033 | 0.098 | 0.0952 | 0.4559 |
| C3 | -0.981 | -1.035 | 0.054 | 0.1800 | 0.4559 |
| F3 | -0.946 | -1.023 | 0.076 | 0.1863 | 0.4559 |
| P3 | -0.995 | -1.047 | 0.053 | 0.1863 | 0.4559 |
| C4 | -0.995 | -1.032 | 0.036 | 0.2205 | 0.4559 |

Best `A+P+` vs `A+P-` theta differences:

| Channel | Median `A+P+` | Median `A+P-` | Difference | p uncorrected | p FDR |
| --- | ---: | ---: | ---: | ---: | ---: |
| F4 | -0.928 | -0.987 | 0.0587 | 0.2106 | 0.8007 |
| F3 | -0.946 | -0.960 | 0.0135 | 0.2926 | 0.8007 |
| C3 | -0.981 | -1.047 | 0.0663 | 0.3028 | 0.8007 |
| Fz | -0.817 | -0.859 | 0.0418 | 0.3134 | 0.8007 |
| Pz | -0.935 | -1.002 | 0.0673 | 0.3695 | 0.8007 |

Theta interpretation:

- Model attribution points toward theta-band relevance.
- Direct subject-level theta PSD tests do not show statistically significant group differences after correction.
- The evidence should be framed as model-guided, hypothesis-generating theta involvement rather than a standalone significant theta biomarker.

## Statistical Analysis Summary

Clinical evaluation:

- Epoch probabilities are averaged within subject.
- Subject-level predictions are used for accuracy, balanced accuracy, sensitivity, specificity, and AUROC.
- This avoids epoch-level leakage and inflated performance.

PEARL transfer analysis:

- Each clinical fold model is applied to all PEARL subjects.
- Epoch-level probabilities are averaged within subject.
- Scores are averaged across folds and seeds.
- Group tests compare subject-level AD-like scores.

Covariate analysis:

- PEARL OLS models adjust for age and sex.
- Main contrasts are `A+P+` vs `N` and `A+P+` vs `A+P-`.

Multiple testing:

- Relevance and PSD follow-up tests include FDR correction.
- Current theta and model-relevance tests are not FDR-significant.

## Figures and Tables

Current generated figure/table assets:

| Path | Description |
| --- | --- |
| `outputs/publication_figures_v2/performance_summary_table.csv` | Compact performance and PEARL median summary. |
| `outputs/publication_figures_v2/common_15_channel_montage.png` | MNE-based 15-channel montage PNG at 300 dpi. |
| `outputs/publication_figures_v2/common_15_channel_montage.pdf` | Vector montage figure. |

The montage generation script is `src/plot_common_channel_montage.py`.

## Reproducibility Commands

Commands are run from the repository root with `PYTHONPATH=src`.

Preprocessing:

```bash
PYTHONPATH=src python src/preprocessing.py
```

PSD extraction:

```bash
PYTHONPATH=src python src/feature_extraction_psd.py
```

HFD extraction:

```bash
PYTHONPATH=src python src/feature_extraction_hfd.py --k-max 16 --num-k 16
PYTHONPATH=src python src/feature_extraction_hfd.py --k-max none --num-k 16
```

Primary medium PSD model, one seed:

```bash
PYTHONPATH=src python src/train_difflogic.py \
  --model-size medium \
  --output-name medium_interpretable \
  --seed 1
```

Interpretation with Integrated Gradients:

```bash
PYTHONPATH=src python src/interpret_difflogic_gradients.py \
  --run-name medium_interpretable \
  --model-size medium \
  --method integrated_gradients \
  --logic-mode soft
```

Covariate analysis:

```bash
PYTHONPATH=src python src/analyze_pearl_covariates.py \
  --summary-dir outputs/difflogic/medium_interpretable/summary
```

Focused PSD and theta follow-up:

```bash
PYTHONPATH=src python src/focused_psd_statistics.py
PYTHONPATH=src python src/theta_psd_group_statistics.py
```

Publication summary figures/tables:

```bash
PYTHONPATH=src python src/make_publication_figures_v2.py
PYTHONPATH=src python src/plot_common_channel_montage.py
```

## Current Scientific Takeaways

1. The PSD DiffLogic model learns a transferable clinical AD EEG signature with subject-level balanced accuracy around 0.81 and AUROC around 0.86.
2. In PEARL-Neuro, the transferred AD-like EEG score is highest in the dual-risk `A+P+` group.
3. Age/sex-adjusted OLS supports higher AD-like scores in `A+P+` relative to both `N` and `A+P-`.
4. Shuffled-label controls collapse both clinical performance and PEARL group structure, supporting label-dependent transfer rather than artifact.
5. HFD controls are weaker and less specific than PSD, supporting the primary PSD representation.
6. Integrated Gradients highlights a recurring theta-band contribution, especially across frontal, central, and parietal channels.
7. Direct theta PSD tests are not significant after correction, so theta should be described as model-guided and exploratory rather than independently confirmed.

## Manuscript Framing Recommendation

Strong claim:

> A clinical AD vs CN DiffLogic model trained on harmonized log-relative PSD features transfers to PEARL-Neuro and yields higher AD-like EEG scores in the dual genetic-risk group.

Careful claim:

> Integrated Gradients suggest theta-band features contribute to the transferred AD-like score, but direct theta-band group tests do not survive multiple-comparison correction.

Avoid:

> PEARL `A+P+` participants are diagnosed as AD-like or preclinical AD cases.

Use instead:

> PEARL outputs are transfer-derived AD-like EEG scores and should be interpreted as research-level risk/signature measures, not clinical diagnoses.
