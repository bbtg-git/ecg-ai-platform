# Reproducibility

This project is reproducible in two modes:

- CI/demo mode without MIT-BIH or `.pt` files.
- Real MIT-BIH mode with local ignored dataset and checkpoint files.

## Install

```bash
python -m pip install -r requirements.txt
```

## Test

```bash
python -m pytest -q
```

## Real MIT-BIH Preparation

Requires local files:

```text
datasets/mit-bih/
  <record>.dat
  <record>.hea
  <record>.atr
```

Commands:

```bash
python -m scripts.prepare_mitbih --config configs/default.yaml
python -m scripts.validate_dataset --config configs/default.yaml
python -m scripts.leakage_check --config configs/default.yaml
```

## Train

```bash
python -m scripts.train_model --config configs/default.yaml
```

Expected local-only output:

```text
artifacts/models/best_model.pt
```

This file is ignored and must not be committed.

## Evaluate

```bash
python -m scripts.evaluate_model --config configs/default.yaml --checkpoint artifacts/models/best_model.pt
```

Expected reports:

```text
reports/evaluation/metrics_summary.json
reports/evaluation/classification_report.json
reports/evaluation/confusion_matrix.png
```

## High-Budget Capped Research Run

```bash
python -m scripts.run_high_budget_experiments --config configs/default.yaml --epochs 5 --cap 2500
```

Expected local-only checkpoints:

```text
artifacts/models/candidates/
artifacts/models/best_model_research.pt
```

Expected tracked reports:

```text
reports/experiments/high_budget/
artifacts/evaluation/high_budget/
artifacts/metrics/high_budget/
```

## Docker Build

```bash
docker build -t ecg-ai-system .
```

## Local API Check

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8010
```

Open:

```text
http://127.0.0.1:8010/health
http://127.0.0.1:8010/docs
```

## Streamlit Check

```bash
python -m streamlit run streamlit_app.py --server.headless true --server.port 8502
```

## Expected Local-Only Files

These files are required for full real-data runs but are intentionally ignored:

```text
datasets/mit-bih/
artifacts/models/best_model.pt
artifacts/models/best_model_research.pt
artifacts/models/candidates/
```

Do not commit datasets or `.pt` model files.
