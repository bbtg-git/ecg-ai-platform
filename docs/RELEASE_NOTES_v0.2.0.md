# Release Notes: v0.2.0-real-mitbih-research

Release name: `v0.2.0-real-mitbih-research`

## Summary

Real MIT-BIH ECG ML system with modular training, evaluation, inference, API, Streamlit UI, Docker support, diagnostics, and capped high-budget research experiments. Includes a CNN-LSTM research checkpoint that improves macro F1 but remains non-clinical.

## What Changed

- Added robust real MIT-BIH WFDB ingestion and inventory validation.
- Added leakage-safe record-level train/validation/test splitting.
- Added a model registry with `baseline_cnn`, `resnet1d`, `inceptiontime`, and `cnn_lstm`.
- Added reproducible training with checkpoint metadata, class weighting, weighted sampling, augmentation, and macro-F1 monitoring.
- Added evaluation reports with accuracy, macro F1, weighted F1, ROC-AUC, per-class metrics, confusion matrices, false negatives, and prediction distribution checks.
- Added threshold tuning, robustness tests, error analysis, explainability, and inference benchmarking.
- Added a FastAPI inference API and Streamlit research UI.
- Added Docker API image support and GitHub Actions CI.
- Added capped high-budget research experiments under `reports/experiments/high_budget/`.
- Added release documentation, reproducibility checklist, project report, roadmap, and release checklist.

## Real MIT-BIH Status

- Source: local MIT-BIH WFDB files.
- Real MIT-BIH: true.
- Complete records used locally: 48.
- Split strategy: grouped by `record_id`.
- Dataset files are intentionally not committed.
- Model checkpoints are intentionally not committed.

## Baseline Metrics

Checkpoint: `artifacts/models/best_model.pt`

| Metric | Value |
| :--- | ---: |
| Accuracy | 0.6284 |
| Macro F1 | 0.2824 |
| Weighted F1 | 0.6057 |
| ROC-AUC OvR macro | 0.6663 |

| Class | F1 |
| :--- | ---: |
| N | 0.8975 |
| V | 0.4527 |
| A | 0.0014 |
| L | 0.0000 |
| R | 0.0606 |

## Research Candidate Metrics

Candidate: `cnn_lstm_maxabs_focal_balanced`

Checkpoint: `artifacts/models/best_model_research.pt`

| Metric | Value |
| :--- | ---: |
| Accuracy | 0.5520 |
| Macro F1 | 0.3295 |
| Weighted F1 | 0.5890 |
| ROC-AUC OvR macro | 0.6171 |

| Class | F1 |
| :--- | ---: |
| N | 0.8390 |
| V | 0.7493 |
| A | 0.0153 |
| L | 0.0026 |
| R | 0.0414 |

The research candidate improves macro F1 and avoids zero `A`, `L`, `R`, and `V` F1, but class quality remains weak.

## Known Limitations

- Not clinically reliable.
- `A`, `L`, and `R` performance remains poor.
- Training uses a local per-class cap for reproducibility.
- No external validation dataset has been evaluated.
- No live hosted URL is claimed.
- Clinical compliance, privacy regulation, and medical-device validation are out of scope.

## Reproducibility Commands

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.prepare_mitbih --config configs/default.yaml
python -m scripts.validate_dataset --config configs/default.yaml
python -m scripts.leakage_check --config configs/default.yaml
python -m scripts.train_model --config configs/default.yaml
python -m scripts.evaluate_model --config configs/default.yaml --checkpoint artifacts/models/best_model.pt
python -m scripts.run_high_budget_experiments --config configs/default.yaml --epochs 5 --cap 2500
docker build -t ecg-ai-system .
```

## Intentionally Not Included

- MIT-BIH dataset files.
- PyTorch `.pt` checkpoints.
- Candidate model weights.
- Private environment variables.
- Fake deployment URLs.
- Clinical-performance claims.

## Next Roadmap

- Recover `A`, `L`, and `R` classes.
- Review record-level split strategy and fold stability.
- Validate beat segmentation and annotation alignment.
- Run longer uncapped training on stronger hardware.
- Explore safe minority-class methods and per-class thresholds.
- Add experiment tracking and model registry/versioning.
