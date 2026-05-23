# Portfolio Summary

## Short Description

Built a real MIT-BIH ECG arrhythmia classification ML engineering system with reproducible data ingestion, leakage-safe training/evaluation, model diagnostics, FastAPI inference, Streamlit UI, Docker support, CI, and honest model-quality reporting.

## Tech Stack

- Python, PyTorch, NumPy, Pandas, scikit-learn.
- WFDB for physiological waveform and annotation loading.
- FastAPI and Uvicorn for inference serving.
- Streamlit for the research UI.
- Docker for API packaging.
- GitHub Actions for CI.
- Matplotlib for evaluation and diagnostic plots.

## Engineering Highlights

- Modular `src/` architecture with config-driven behavior.
- Robust CLI workflows for dataset validation, training, evaluation, leakage checks, threshold tuning, error analysis, robustness, explainability, benchmarking, and research comparisons.
- Checkpoint metadata guards for model type, class order, input size, sampling rate, and preprocessing compatibility.
- Local-only data/model handling through `.gitignore` and reproducibility docs.
- Release-ready documentation package for GitHub presentation.

## ML Highlights

- Real MIT-BIH WFDB ingestion from 48 complete records.
- Record-level leakage-safe splitting.
- Model registry with CNN, ResNet1D, InceptionTime, and CNN-LSTM variants.
- Evaluation with macro F1, weighted F1, per-class sensitivity/specificity, confusion matrices, ROC/PR curves, and false-negative analysis.
- Capped high-budget research pass with explicit promotion criteria.

## MLOps Highlights

- Dockerized API.
- CI smoke path that does not require private/local data.
- Reproducibility checklist for local real-data runs.
- Generated reports for evaluation, robustness, diagnostics, and model-quality research.
- Roadmap for experiment tracking, model registry, and hosted demo.

## Honest Model-Performance Note

The system is strong as an ML engineering project, but the current model is not clinically reliable. The baseline macro F1 is `0.2824`; the best capped research candidate improves macro F1 to `0.3295`, but `A`, `L`, and `R` remain weak.

## Suggested Resume Bullets

- Built an end-to-end ECG arrhythmia classification platform using PyTorch, WFDB, FastAPI, Streamlit, Docker, and GitHub Actions.
- Implemented leakage-safe MIT-BIH training/evaluation with record-level splitting, checkpoint metadata validation, threshold tuning, and diagnostic reporting.
- Developed model-quality research workflows comparing CNN, ResNet1D, InceptionTime, and CNN-LSTM architectures with honest real-data metrics and non-clinical limitations.
- Productized ML inference with CLI, API, UI, monitoring logs, robustness checks, explainability outputs, and reproducible release documentation.
