# ECG AI Platform

[![CI](https://github.com/pr6thv3/ecg-ai-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/pr6thv3/ecg-ai-platform/actions/workflows/ci.yml)

Real MIT-BIH ECG arrhythmia classification research system with reproducible training, evaluation, inference, API serving, Streamlit UI, Docker packaging, diagnostics, and honest model-quality reporting.

> Not for clinical use. This project is for research, education, portfolio review, and ML systems engineering demonstration only. It must not be used for diagnosis, treatment, monitoring, triage, or medical decision-making.

Release status: `v0.2.0-real-mitbih-research`

## Current Status

- Real MIT-BIH WFDB files are loaded locally from `datasets/mit-bih/`.
- The loader auto-detects complete `.dat`, `.hea`, and `.atr` record triples.
- The local run uses 48 complete MIT-BIH records; an incomplete `102-0.atr` sidecar is skipped because matching signal/header files are absent.
- Training, validation, and test splits are leakage-safe at record level.
- The ML system is functionally complete, but the model is not clinically reliable.
- `.pt` checkpoints and datasets are ignored and intentionally not committed.

## Key Features

- WFDB-based MIT-BIH ingestion with clear missing-file validation.
- Config-driven preprocessing, class mapping, splitting, training, evaluation, and inference.
- Model registry: `baseline_cnn`, `resnet1d`, `inceptiontime`, `cnn_lstm`.
- Training with class weights, weighted sampling, augmentations, checkpoint metadata, and macro-F1 monitoring.
- Evaluation artifacts: metrics, classification report, confusion matrix, ROC/PR curves, false negatives, prediction distribution, and anti-collapse checks.
- Threshold tuning, error analysis, robustness testing, explainability, and benchmarking.
- FastAPI inference API and Streamlit research dashboard.
- Docker API image and GitHub Actions CI smoke checks.
- High-budget capped research experiments under `reports/experiments/high_budget/`.

## Architecture

```text
MIT-BIH WFDB files / CSV input
  -> signal validation
  -> filtering and beat/window preprocessing
  -> grouped record-level split
  -> model training and checkpoint metadata
  -> evaluation, threshold tuning, diagnostics
  -> inference pipeline
  -> FastAPI / Streamlit / CLI
  -> monitoring logs and report artifacts
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full Mermaid diagram and artifact flow.

## Folder Structure

```text
configs/                  YAML configuration
src/config/               config loading and validation
src/data/                 MIT-BIH and synthetic demo data utilities
src/preprocessing/        signal validation, filtering, segmentation
src/models/               model registry and neural architectures
src/training/             training loop and checkpoint metadata
src/evaluation/           metrics, threshold tuning, robustness, error analysis
src/explainability/       saliency/explanation outputs
src/inference/            checkpoint loading and prediction pipeline
src/api/                  FastAPI app
src/monitoring/           inference logging
scripts/                  CLI entry points
tests/                    pytest suite
reports/                  human-readable evaluation/research outputs
artifacts/                local generated artifacts, with model weights ignored
docs/                     release, reproducibility, and project documentation
```

## Quickstart

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.smoke_test_pipeline --config configs/default.yaml
```

The smoke test can run in demo mode. Real MIT-BIH training/evaluation requires local WFDB files and local checkpoint files that are intentionally not committed.

## MIT-BIH Dataset Setup

Place MIT-BIH Arrhythmia Database files here:

```text
datasets/mit-bih/
  100.dat
  100.hea
  100.atr
  ...
```

Each usable record must have matching `.dat`, `.hea`, and `.atr` files.

Validate the local dataset:

```bash
python -m scripts.prepare_mitbih --config configs/default.yaml
python -m scripts.validate_dataset --config configs/default.yaml
python -m scripts.leakage_check --config configs/default.yaml
```

Synthetic fallback is only for CI/demo smoke paths when explicitly enabled. Synthetic output must not be cited as MIT-BIH performance.

## Training

Select a model in `configs/default.yaml`:

```yaml
model:
  type: baseline_cnn  # baseline_cnn | resnet1d | inceptiontime | cnn_lstm
```

Train:

```bash
python -m scripts.train_model --config configs/default.yaml
```

Expected local outputs:

```text
artifacts/models/best_model.pt
artifacts/metrics/training_history.csv
artifacts/metrics/training_curves.png
artifacts/metrics/training_summary.json
```

`artifacts/models/*.pt` is ignored by Git.

## Evaluation

```bash
python -m scripts.evaluate_model --config configs/default.yaml --checkpoint artifacts/models/best_model.pt
python -m scripts.tune_thresholds --config configs/default.yaml --checkpoint artifacts/models/best_model.pt
python -m scripts.error_analysis --config configs/default.yaml
python -m scripts.robustness_test --config configs/default.yaml
```

Generated outputs include:

```text
reports/evaluation/metrics_summary.json
reports/evaluation/classification_report.json
reports/evaluation/confusion_matrix.csv
reports/evaluation/confusion_matrix.png
reports/evaluation/prediction_distribution.csv
reports/evaluation/false_negatives.csv
reports/evaluation/roc_curves.png
reports/evaluation/pr_curves.png
```

## Inference

CLI:

```bash
python -m scripts.run_inference --input artifacts/samples/demo_ecg.csv --config configs/default.yaml
```

The prediction response includes predicted class, confidence, probabilities, thresholds used, timing, checkpoint metadata, warnings, and the research-only disclaimer.

## FastAPI

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8010
```

Endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict-file`

OpenAPI docs are available locally at:

```text
http://127.0.0.1:8010/docs
```

## Streamlit

```bash
python -m streamlit run streamlit_app.py --server.headless true --server.port 8502
```

Set `ECG_API_URL=https://your-api-host` to make the dashboard call a deployed backend. Do not add hosted URLs to this README until they are verified live.

## Docker

Build:

```bash
docker build -t ecg-ai-system .
```

Run:

```bash
docker run -p 8001:8000 ecg-ai-system
```

The Docker image excludes MIT-BIH data and `.pt` checkpoints. It is configured for API serving with the bundled small ONNX fallback. For PyTorch checkpoint serving, mount/export a compatible artifact explicitly and validate checkpoint metadata.

## Tests

```bash
python -m pytest -q
python -m scripts.smoke_test_pipeline --config configs/default.yaml
```

CI uses explicit synthetic/no-checkpoint flags and does not require MIT-BIH data or `.pt` checkpoints.

## Real MIT-BIH Metrics

Current baseline checkpoint: `artifacts/models/best_model.pt`

| Metric | Value |
| :--- | ---: |
| Accuracy | 0.6284 |
| Macro F1 | 0.2824 |
| Weighted F1 | 0.6057 |
| ROC-AUC OvR macro | 0.6663 |

Baseline per-class F1:

| Class | Meaning | F1 |
| :--- | :--- | ---: |
| N | Normal / bundle branch / escape group mapping | 0.8975 |
| V | Premature ventricular contraction | 0.4527 |
| A | Atrial premature beat | 0.0014 |
| L | Left bundle branch block beat | 0.0000 |
| R | Right bundle branch block beat | 0.0606 |

## Baseline vs Research Candidate

Best capped research candidate: `cnn_lstm_maxabs_focal_balanced`

| Model | Accuracy | Macro F1 | Weighted F1 | N F1 | V F1 | A F1 | L F1 | R F1 | Status |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `baseline_cnn` | 0.6284 | 0.2824 | 0.6057 | 0.8975 | 0.4527 | 0.0014 | 0.0000 | 0.0606 | Stable baseline |
| `cnn_lstm_maxabs_focal_balanced` | 0.5520 | 0.3295 | 0.5890 | 0.8390 | 0.7493 | 0.0153 | 0.0026 | 0.0414 | Research checkpoint |

The research candidate improves macro F1 and avoids zero F1 for `A`, `L`, `R`, and `V`, but minority-class performance remains weak. It is not a clinical-quality model.

## Screenshots And Demo Assets

See [docs/assets/README.md](docs/assets/README.md).

Expected release assets:

- `docs/assets/streamlit_ui.png`
- `docs/assets/api_docs.png`
- `docs/assets/confusion_matrix.png`
- `docs/assets/evaluation_summary.png`
- `docs/assets/architecture_diagram.png`
- `docs/assets/demo.gif`

Existing generated visuals and report images can be used until final screenshots are captured.

## Limitations

- Not clinically validated and not suitable for medical decision-making.
- Held-out split is record-level and leakage-safe, but model quality remains weak.
- `A`, `L`, and `R` recovery is still the central research problem.
- Local training uses per-class caps for reproducibility on commodity hardware.
- `.pt` checkpoints and MIT-BIH data are local-only ignored assets.
- No public deployment URL is claimed in this repository.
- HIPAA, GDPR, medical-device validation, hospital deployment, and production clinical monitoring are out of scope.

## Future Work

- Recover `A`, `L`, and `R` classes with better segmentation and minority-class training strategy.
- Review record-level split strategy and repeated grouped-fold validation.
- Validate beat-centered windows and R-peak alignment.
- Run longer uncapped training on stronger hardware.
- Explore safe oversampling alternatives and per-class threshold optimization.
- Add model registry/versioning, experiment tracking, and verified hosted demos.

## Release Documentation

- [Release Notes v0.2.0](docs/RELEASE_NOTES_v0.2.0.md)
- [Project Report](docs/PROJECT_REPORT.md)
- [Model Card](docs/MODEL_CARD.md)
- [Testing And Validation](docs/TESTING_AND_VALIDATION.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [GitHub Release Checklist](docs/GITHUB_RELEASE_CHECKLIST.md)
