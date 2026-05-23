# Model Card

## Model Details

| Field | Value |
| :--- | :--- |
| Project | ECG AI Platform |
| Release | `v0.2.0-real-mitbih-research` |
| Primary artifact | `artifacts/models/best_model.pt` |
| Research artifact | `artifacts/models/best_model_research.pt` |
| Primary model | `baseline_cnn` |
| Research model | `cnn_lstm` |
| Runtime | PyTorch local training/evaluation/inference; small ONNX fallback for API container cold start |
| Input | 360-sample single-channel ECG beat/window |
| Dataset | MIT-BIH Arrhythmia Database from local WFDB files |
| Status | Complete ML engineering research system with weak held-out model quality |

## Intended Use

This project is intended for research, education, portfolio review, and ML systems engineering demonstration. It demonstrates a full ECG ML workflow: WFDB ingestion, leakage-safe splitting, preprocessing, model training, evaluation, diagnostics, threshold tuning, explainability, API serving, Streamlit UI, Docker packaging, and CI tests.

## Non-Intended Use

This model is not intended for clinical diagnosis, treatment, monitoring, triage, patient care, or medical decision-making. It is not a medical device and has not been clinically validated.

## Dataset

The project uses local MIT-BIH Arrhythmia Database WFDB records under:

```text
datasets/mit-bih/
```

Each usable record requires:

```text
<record>.dat
<record>.hea
<record>.atr
```

The local run uses 48 complete records. An incomplete `102-0.atr` sidecar is skipped because matching `.dat` and `.hea` files are absent.

## Labels And Class Mapping

| Label | Index | Project class |
| :--- | ---: | :--- |
| `N` | 0 | Normal / normal-like class mapping |
| `V` | 1 | Premature ventricular contraction |
| `A` | 2 | Atrial premature beat |
| `L` | 3 | Left bundle branch block beat |
| `R` | 4 | Right bundle branch block beat |

Class order and mapping are stored in checkpoint metadata and validated before inference.

## Preprocessing

- Sampling rate: 360 Hz.
- Window length: 360 samples.
- Filtering: configured bandpass preprocessing in `configs/default.yaml`.
- Default normalization: max-absolute scaling.
- Training/evaluation windows are beat-centered around MIT-BIH annotations.
- API file inference validates input length, NaN/Inf values, and checkpoint metadata before prediction.

## Evaluation Split

- Split strategy: record-level grouping by `record_id`.
- Goal: prevent beat/window leakage across train, validation, and test sets.
- Leakage report: `reports/evaluation/leakage_check.json`.
- Dataset validation: `reports/evaluation/dataset_validation.json`.

## Baseline Metrics

Current verified held-out MIT-BIH test metrics for `artifacts/models/best_model.pt`:

| Metric | Value |
| :--- | ---: |
| Accuracy | 0.6284 |
| Macro F1 | 0.2824 |
| Weighted F1 | 0.6057 |
| ROC-AUC OvR macro | 0.6663 |

Per-class F1:

| Class | F1 |
| :--- | ---: |
| N | 0.8975 |
| V | 0.4527 |
| A | 0.0014 |
| L | 0.0000 |
| R | 0.0606 |

## Research Candidate Metrics

Best capped high-budget research candidate: `cnn_lstm_maxabs_focal_balanced`.

| Metric | Value |
| :--- | ---: |
| Accuracy | 0.5520 |
| Macro F1 | 0.3295 |
| Weighted F1 | 0.5890 |
| ROC-AUC OvR macro | 0.6171 |

Per-class F1:

| Class | F1 |
| :--- | ---: |
| N | 0.8390 |
| V | 0.7493 |
| A | 0.0153 |
| L | 0.0026 |
| R | 0.0414 |

The research candidate passes a narrow research promotion rule because macro F1 improves over `0.2824`, `A`/`L`/`R`/`V` F1 are non-zero, and the prediction distribution is not degenerate. It is still weak and not clinically reliable.

## Limitations

- Current model quality is not adequate for clinical use.
- `A`, `L`, and `R` remain the main failure classes.
- Capped local training is useful for reproducibility but not a final performance setting.
- Record-level generalization remains unstable.
- Saliency outputs are technical debugging artifacts, not medical reasoning.
- Docker does not include MIT-BIH files or PyTorch `.pt` checkpoints.
- No external validation cohort has been evaluated.

## Ethical And Clinical Warning

This project must not be used to diagnose, treat, monitor, or make decisions about any person. Any future clinical use would require expert clinical review, stronger validation, regulatory analysis, privacy/security review, calibrated deployment controls, and prospective evaluation.

## Future Validation Needed

- Repeated grouped-record validation and external dataset validation.
- Beat segmentation and R-peak alignment audit.
- Uncapped training on stronger hardware.
- Minority-class methods for `A`, `L`, and `R`.
- Per-class threshold calibration with confidence review.
- Independent review by ECG/domain experts.
