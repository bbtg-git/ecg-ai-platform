"""Synthetic ECG-like signals for software testing only."""

import numpy as np


def synthetic_beat(class_index=0, length=360, seed=42):
    """Generate one ECG-like window. This is not real physiological data."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, length, endpoint=False)

    def pulse(center, width, amplitude):
        return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)

    beat = (
        pulse(0.20, 0.035, 0.12)
        - pulse(0.385, 0.012, 0.15)
        + pulse(0.405, 0.009, 0.95)
        - pulse(0.425, 0.014, 0.25)
        + pulse(0.68, 0.065, 0.28)
    )

    # Vary the synthetic morphology by class to exercise the software.
    if class_index == 1:
        beat += pulse(0.405, 0.035, 0.45)
    elif class_index == 2:
        beat += pulse(0.25, 0.07, 0.25)
    elif class_index == 3:
        beat -= pulse(0.405, 0.025, 0.30)
    elif class_index == 4:
        beat += pulse(0.44, 0.025, 0.25)

    beat += 0.025 * np.sin(2 * np.pi * 0.7 * t)
    beat += rng.normal(0.0, 0.012, length)

    return beat.astype(np.float32)


def synthetic_long_signal(length=3600, sampling_rate=360, seed=42):
    """Generate a longer synthetic signal for preprocessing/API smoke tests."""
    rng = np.random.default_rng(seed)
    length = int(length)
    if length <= 0:
        raise ValueError("length must be positive")

    signal = rng.normal(0.0, 0.015, length).astype(np.float32)
    beat_length = max(1, int(sampling_rate))
    template = synthetic_beat(length=beat_length, seed=seed)

    for start in range(0, length, beat_length):
        end = min(start + beat_length, length)
        signal[start:end] += template[:end - start]

    return signal.astype(np.float32)
