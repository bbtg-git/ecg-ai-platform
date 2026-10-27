"""Inference backend for the separately trained 187-point Kaggle ECG CNN."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


CLASS_NAMES = ["N", "S", "V", "F", "Q"]
MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "models"
    / "kaggle_heartbeat_cnn.pt"
)


class ECG1DCNN(nn.Module):
    """Architecture must match scripts/train_kaggle_ecg.py."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class KaggleHeartbeatPredictor:
    """Loads the Kaggle model without modifying the original inference pipeline."""

    def __init__(self, checkpoint_path=MODEL_PATH):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Kaggle ECG checkpoint not found: {self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint.get("input_length") != 187:
            raise ValueError("Expected a checkpoint trained on 187-point inputs.")
        if checkpoint.get("num_classes") != 5:
            raise ValueError("Expected a five-class checkpoint.")

        self.class_mapping = checkpoint.get("class_mapping", {})
        expected_mapping = {
            "0": "N", "1": "S", "2": "V", "3": "F", "4": "Q"
        }
        if self.class_mapping != expected_mapping:
            raise ValueError("Checkpoint class mapping does not match this backend.")

        self.model = ECG1DCNN()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, signal):
        """Predict one already-preprocessed heartbeat of exactly 187 values."""
        signal = np.asarray(signal, dtype=np.float32)

        if signal.shape != (187,):
            raise ValueError(
                f"Expected exactly 187 signal values; received shape {signal.shape}."
            )
        if not np.isfinite(signal).all():
            raise ValueError("Signal contains NaN or infinite values.")

        tensor = torch.from_numpy(signal.copy()).reshape(1, 1, 187)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].numpy()

        predicted_index = int(np.argmax(probabilities))
        return {
            "predicted_class": CLASS_NAMES[predicted_index],
            "class_index": predicted_index,
            "confidence": float(probabilities[predicted_index]),
            "probabilities": {
                name: float(probabilities[i])
                for i, name in enumerate(CLASS_NAMES)
            },
            "model": "kaggle_heartbeat_cnn",
            "input_length": 187,
            "disclaimer": (
                "Research and education only; not validated for clinical use."
            ),
        }
