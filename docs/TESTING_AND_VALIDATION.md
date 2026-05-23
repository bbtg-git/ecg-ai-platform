# Testing And Validation

This document summarizes the v0.2.0 test strategy. Commands that depend on local MIT-BIH data or local `.pt` checkpoints require ignored files that are not committed to Git.

## Unit Tests

Run:

```bash
python -m pytest -q
```

Coverage areas:

- Config loading and validation.
- Signal validation and preprocessing.
- Dataset loading utilities and small leakage-safe split checks.
- Model forward pass for registered architectures.
- Checkpoint metadata validation.
- Inference output schema and threshold behavior.
- API health and core route behavior.
- Edge cases for malformed, short, NaN/Inf, and corrupted inputs.

## Smoke Tests

```bash
python -m scripts.smoke_test_pipeline --config configs/default.yaml
```

CI runs smoke tests with explicit synthetic/no-checkpoint environment flags. Local real MIT-BIH smoke paths should use the real dataset and local checkpoints.

## Dataset Validation

```bash
python -m scripts.prepare_mitbih --config configs/default.yaml
python -m scripts.validate_dataset --config configs/default.yaml
```

Expected reports:

- `reports/evaluation/mitbih_preparation.json`
- `reports/evaluation/dataset_validation.json`
- `artifacts/evaluation/dataset_summary.json`

Known local behavior: incomplete sidecar `102-0.atr` is skipped because matching `.dat` and `.hea` files are missing.

## Leakage Checks

```bash
python -m scripts.leakage_check --config configs/default.yaml
```

The project uses record-level grouping by `record_id`, so train/validation/test splits should not share the same MIT-BIH record.

Expected report:

```text
reports/evaluation/leakage_check.json
```

## Evaluation Validation

```bash
python -m scripts.evaluate_model --config configs/default.yaml --checkpoint artifacts/models/best_model.pt
```

Expected outputs:

- `reports/evaluation/metrics_summary.json`
- `reports/evaluation/classification_report.json`
- `reports/evaluation/confusion_matrix.csv`
- `reports/evaluation/confusion_matrix.png`
- `reports/evaluation/prediction_distribution.json`
- `reports/evaluation/false_negative_summary.json`

The current baseline model is not collapsed to one class, but `A`, `L`, and `R` performance is weak.

## Robustness Tests

```bash
python -m scripts.robustness_test --config configs/default.yaml
```

Covered cases include:

- Gaussian noise.
- Baseline wander.
- Amplitude scaling.
- Short signal rejection.
- NaN/Inf signal rejection.
- Corrupted CSV handling.
- Wrong sampling-rate metadata.
- Missing annotation behavior.

Expected report:

```text
reports/robustness/robustness_summary.json
```

## API Health Checks

Run:

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8010
```

Verify:

```text
GET http://127.0.0.1:8010/health
GET http://127.0.0.1:8010/model-info
POST http://127.0.0.1:8010/predict
POST http://127.0.0.1:8010/predict-file
```

The API must not silently fall back to an incompatible model. Checkpoint metadata mismatches should fail clearly unless an explicit demo fallback is enabled.

## Streamlit Verification

Run:

```bash
python -m streamlit run streamlit_app.py --server.headless true --server.port 8502
```

Verify:

- Page loads.
- Medical/research disclaimer is visible.
- CSV upload path is available.
- Prediction probabilities and confidence render.
- Backend URL environment override works through `ECG_API_URL`.

## Docker Verification

Build:

```bash
docker build -t ecg-ai-system .
```

Run:

```bash
docker run -p 8001:8000 ecg-ai-system
```

Verify:

```text
GET http://127.0.0.1:8001/health
```

The image intentionally excludes MIT-BIH files and `.pt` checkpoints. It is API-ready with the bundled small ONNX fallback for cold-start demonstration.

## Known Failure Modes

- Missing MIT-BIH files: real training/evaluation commands fail with a clear dataset error unless synthetic fallback is explicitly enabled.
- Missing `.pt` checkpoint: PyTorch inference fails clearly unless fallback is explicitly enabled.
- Checkpoint/config mismatch: metadata guard rejects mismatched model type, class order, input size, or preprocessing.
- Weak model quality: `A`, `L`, and `R` remain unreliable despite the completed system.
- Local CPU/GPU constraints: uncapped training and cross-validation can be slow.
