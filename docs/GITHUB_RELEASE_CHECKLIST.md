# GitHub Release Checklist

## Pre-Release Checks

- [ ] `git status` is clean.
- [ ] No `datasets/` files are committed.
- [ ] No `.pt` checkpoint files are committed.
- [ ] `artifacts/models/candidates/` is ignored.
- [ ] `python -m pytest -q` passes.
- [ ] `python -m scripts.validate_dataset --config configs/default.yaml` passes when local MIT-BIH files are present.
- [ ] `python -m scripts.leakage_check --config configs/default.yaml` passes when local MIT-BIH files are present.
- [ ] `python -m scripts.evaluate_model --config configs/default.yaml --checkpoint artifacts/models/best_model.pt` passes when local checkpoint exists.
- [ ] `docker build -t ecg-ai-system .` passes.
- [ ] README is updated.
- [ ] Release notes are ready.
- [ ] Model card is updated.
- [ ] Screenshots are added if available.
- [ ] No fake hosted URLs are included.

## Tag Command

```bash
git tag -a v0.2.0-real-mitbih-research -m "v0.2.0 real MIT-BIH research release"
git push origin v0.2.0-real-mitbih-research
```

If the tag already exists and must be moved to a newer final release commit, delete or update it intentionally and document that action.

## Release Description

```text
Real MIT-BIH ECG ML system with modular training, evaluation, inference, API, Streamlit UI, Docker support, diagnostics, and capped high-budget research experiments. Includes a CNN-LSTM research checkpoint that improves macro F1 but remains non-clinical.
```

## Assets

Recommended release assets:

- README screenshot.
- Streamlit screenshot.
- API docs screenshot.
- Confusion matrix image.
- Short demo GIF/video.

Do not attach MIT-BIH data, `.pt` model checkpoints, private files, or fake screenshots.
