# Project Report

## Problem Statement

This project builds a real ECG arrhythmia classification research system around the MIT-BIH Arrhythmia Database. The goal is not to claim clinical reliability. The goal is to demonstrate a complete, reproducible ML engineering platform for ECG signal ingestion, model training, evaluation, diagnostics, inference serving, and release-quality documentation.

## Dataset

The system uses local MIT-BIH WFDB records stored under:

```text
datasets/mit-bih/
```

Each complete record requires `.dat`, `.hea`, and `.atr` files. The current local run detects 48 complete usable records. Dataset files are ignored by Git and must be provided locally by the user.

## System Architecture

The system is organized as a modular Python ML stack:

- `src/config`: YAML config loading and validation.
- `src/data`: MIT-BIH loading, inventory validation, and split metadata.
- `src/preprocessing`: signal validation, filtering, windowing, and normalization.
- `src/models`: neural model registry.
- `src/training`: reproducible training loop and checkpoint metadata.
- `src/evaluation`: metrics, threshold tuning, leakage checks, robustness, and error analysis.
- `src/inference`: safe checkpoint loading and prediction pipeline.
- `src/api`: FastAPI serving layer.
- `src/monitoring`: inference logs.
- `scripts`: runnable CLI workflows.

## Pipeline

```text
WFDB records
  -> inventory validation
  -> signal loading
  -> annotation-centered beat/window extraction
  -> preprocessing and normalization
  -> record-level split
  -> training
  -> evaluation and threshold tuning
  -> inference/API/UI
  -> monitoring and reports
```

## Training Methodology

Training is config-driven through `configs/default.yaml`. The current release supports:

- Deterministic seed control.
- Model selection through the registry.
- Class weighting.
- Weighted sampling.
- ECG-specific augmentations.
- Gradient clipping.
- Scheduler support.
- Macro-F1 monitoring.
- Checkpoint metadata with class order, preprocessing settings, model type, sampling rate, timestamp, and git commit when available.

The default baseline is `baseline_cnn`. Research experiments also evaluate `resnet1d`, `inceptiontime`, and `cnn_lstm`.

## Evaluation Methodology

Evaluation reports include:

- Accuracy.
- Macro F1.
- Weighted F1.
- ROC-AUC OvR macro.
- Per-class precision, recall, sensitivity, specificity, and F1.
- Confusion matrix.
- Prediction distribution and anti-collapse checks.
- False-negative summary.
- Low-confidence predictions.
- ROC and precision-recall plots.

Reports are written under `reports/evaluation/` and `reports/experiments/`.

## Leakage Prevention

The split strategy groups by `record_id`. This prevents beat/window leakage where beats from the same MIT-BIH record appear in multiple splits. Leakage reports are generated with:

```bash
python -m scripts.leakage_check --config configs/default.yaml
```

## Experiments

The v0.2.0 research pass ran capped high-budget experiments:

```bash
python -m scripts.run_high_budget_experiments --config configs/default.yaml --epochs 5 --cap 2500
```

The experiment runner stores candidate checkpoints under ignored `artifacts/models/candidates/`, evaluates each candidate, and only copies a candidate to `artifacts/models/best_model_research.pt` if it passes the promotion rule.

Promotion required:

- Macro F1 above the baseline value `0.2824`.
- Non-zero `A`, `L`, `R`, and `V` F1.
- Non-degenerate prediction distribution.
- At least four predicted classes.

## Results

Baseline:

| Model | Accuracy | Macro F1 | Weighted F1 | N F1 | V F1 | A F1 | L F1 | R F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_cnn` | 0.6284 | 0.2824 | 0.6057 | 0.8975 | 0.4527 | 0.0014 | 0.0000 | 0.0606 |

Best research candidate:

| Model | Accuracy | Macro F1 | Weighted F1 | N F1 | V F1 | A F1 | L F1 | R F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cnn_lstm_maxabs_focal_balanced` | 0.5520 | 0.3295 | 0.5890 | 0.8390 | 0.7493 | 0.0153 | 0.0026 | 0.0414 |

The research candidate improves macro F1 but remains weak on minority morphology classes.

## Limitations

- The model is not clinically reliable.
- `A`, `L`, and `R` remain poor.
- Capped training is a local reproducibility compromise.
- The record-level split is more realistic than random beat-level splits, but it exposes substantial generalization difficulty.
- No external dataset validation is included.
- No live hosted demo URL is claimed.

## Future Work

- Beat segmentation validation and R-peak alignment review.
- Uncapped training and repeated grouped-fold validation.
- Per-class threshold tuning and calibration.
- Minority-class training methods that preserve waveform morphology.
- Model registry, artifact versioning, and experiment tracking.
- Hosted API and dashboard only after verified deployment.

## Medical Disclaimer

This project is for research and education only. It is not for diagnosis, treatment, monitoring, triage, patient care, or medical decision-making.
