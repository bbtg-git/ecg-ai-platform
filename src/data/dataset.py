"""MIT-BIH dataset loading, synthetic demo fallback, and leakage-safe splits."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.config import resolve_path
from src.data.synthetic import synthetic_beat


@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    record_ids: np.ndarray
    source: str
    warnings: list


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    record_ids_train: np.ndarray
    record_ids_val: np.ndarray
    record_ids_test: np.ndarray


def mitbih_inventory(path, records=None, annotation_extension="atr"):
    """Find complete WFDB records with matching .hea, .dat, and annotation files."""
    root = Path(resolve_path(path))
    requested = [str(r) for r in (records or [])]
    candidates = requested or sorted(p.stem for p in root.glob("*.hea")) if root.exists() else requested

    complete = []
    incomplete = []

    for record in candidates:
        # Accept either a record name or a path-like record entry.
        name = Path(record).stem
        required = [
            root / f"{name}.hea",
            root / f"{name}.dat",
            root / f"{name}.{annotation_extension.lstrip('.')}",
        ]
        if all(p.is_file() for p in required):
            complete.append(name)
        else:
            incomplete.append({
                "record": name,
                "missing": [p.name for p in required if not p.is_file()],
            })

    return {
        "path": str(root),
        "ready": bool(complete),
        "complete_records": complete,
        "complete_record_count": len(complete),
        "incomplete_records": incomplete,
    }


def _synthetic_bundle(config):
    dataset_cfg = config["dataset"]
    class_mapping = dataset_cfg["class_mapping"]
    class_count = len(config["model"]["class_names"])
    window_size = int(config["preprocessing"]["window_size"])
    samples_per_class = int(dataset_cfg.get("synthetic_samples_per_class", 30))
    records_per_class = max(
        1, int(dataset_cfg.get("synthetic_records_per_class", 6))
    )

    xs, ys, groups = [], [], []

    for class_index in range(class_count):
        for sample_index in range(samples_per_class):
            record_number = sample_index % records_per_class
            record_id = f"synthetic_c{class_index}_r{record_number}"
            xs.append(
                synthetic_beat(
                    class_index,
                    length=window_size,
                    seed=42 + class_index * 10000 + sample_index,
                )
            )
            ys.append(class_index)
            groups.append(record_id)

    if not xs:
        raise ValueError("Synthetic dataset configuration produced no samples.")

    return DatasetBundle(
        X=np.asarray(xs, dtype=np.float32),
        y=np.asarray(ys, dtype=np.int64),
        record_ids=np.asarray(groups),
        source="synthetic",
        warnings=[
            "Synthetic demo data only; not real MIT-BIH data and not suitable "
            "for reporting clinical or real-world model performance."
        ],
    )


def load_dataset(config):
    """Load annotated MIT-BIH beats, or use explicitly enabled demo data."""
    dataset_cfg = config["dataset"]
    root = resolve_path(dataset_cfg["mitbih_path"])
    extension = str(dataset_cfg.get("annotation_extension", "atr"))
    inventory = mitbih_inventory(
        root, dataset_cfg.get("records", []), extension
    )

    if not inventory["ready"]:
        if dataset_cfg.get("allow_synthetic_fallback", False):
            return _synthetic_bundle(config)
        raise FileNotFoundError(
            f"No complete MIT-BIH records found in {root}. Each record needs "
            f".hea, .dat, and .{extension.lstrip('.')} files. For software "
            "smoke tests only, explicitly enable dataset.allow_synthetic_fallback."
        )

    try:
        import wfdb
    except ImportError as exc:
        raise ImportError(
            "Reading MIT-BIH records requires the 'wfdb' package."
        ) from exc

    class_mapping = dataset_cfg["class_mapping"]
    window_size = int(config["preprocessing"]["window_size"])
    half = window_size // 2
    channel = int(dataset_cfg.get("signal_channel", 0))
    xs, ys, groups, warnings = [], [], [], []

    for record_name in inventory["complete_records"]:
        record_path = root / record_name
        try:
            record = wfdb.rdrecord(str(record_path))
            annotation = wfdb.rdann(
                str(record_path), extension.lstrip(".")
            )
            if record.p_signal is None or record.p_signal.shape[1] <= channel:
                warnings.append(f"{record_name}: requested ECG channel unavailable.")
                continue

            signal = np.asarray(record.p_signal[:, channel], dtype=np.float32)
            for sample, symbol in zip(annotation.sample, annotation.symbol):
                if symbol not in class_mapping:
                    continue

                start = int(sample) - half
                end = start + window_size
                if start < 0 or end > len(signal):
                    continue

                window = signal[start:end]
                if not np.all(np.isfinite(window)):
                    continue
                scale = float(np.max(np.abs(window)))
                if scale > 1e-8:
                    window = window / scale

                xs.append(window.astype(np.float32))
                ys.append(int(class_mapping[symbol]))
                groups.append(record_name)

        except Exception as exc:
            warnings.append(f"{record_name}: could not read record ({exc}).")

    if not xs:
        raise ValueError(
            "MIT-BIH files were found, but no usable annotated beat windows "
            "were extracted. Check the annotations and class_mapping."
        )

    return DatasetBundle(
        X=np.asarray(xs, dtype=np.float32),
        y=np.asarray(ys, dtype=np.int64),
        record_ids=np.asarray(groups),
        source="mitbih",
        warnings=warnings,
    )


def split_dataset(bundle, config):
    """Split by record ID so windows from one record cannot cross splits."""
    split_cfg = config["dataset"]["split"]
    train_ratio = float(split_cfg["train_ratio"])
    val_ratio = float(split_cfg["val_ratio"])
    test_ratio = float(split_cfg["test_ratio"])

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Each split ratio must be greater than zero.")

    groups = np.asarray(bundle.record_ids)
    indices = np.arange(len(bundle.y))
    if len(np.unique(groups)) < 3:
        raise ValueError("At least three distinct records are needed for grouped splitting.")

    seed = int(split_cfg.get("seed", 42))
    first = GroupShuffleSplit(
        n_splits=1, test_size=1.0 - train_ratio, random_state=seed
    )
    train_idx, remainder_idx = next(
        first.split(indices, bundle.y, groups)
    )

    remainder_groups = groups[remainder_idx]
    remainder_test_fraction = test_ratio / (val_ratio + test_ratio)
    second = GroupShuffleSplit(
        n_splits=1,
        test_size=remainder_test_fraction,
        random_state=seed + 1,
    )
    val_local, test_local = next(
        second.split(
            remainder_idx,
            bundle.y[remainder_idx],
            remainder_groups,
        )
    )
    val_idx = remainder_idx[val_local]
    test_idx = remainder_idx[test_local]

    return DatasetSplits(
        X_train=bundle.X[train_idx],
        y_train=bundle.y[train_idx],
        X_val=bundle.X[val_idx],
        y_val=bundle.y[val_idx],
        X_test=bundle.X[test_idx],
        y_test=bundle.y[test_idx],
        record_ids_train=groups[train_idx],
        record_ids_val=groups[val_idx],
        record_ids_test=groups[test_idx],
    )


def leakage_report(splits):
    """Report whether record IDs overlap between the three data splits."""
    train = set(splits.record_ids_train.tolist())
    val = set(splits.record_ids_val.tolist())
    test = set(splits.record_ids_test.tolist())

    train_val = sorted(train & val)
    train_test = sorted(train & test)
    val_test = sorted(val & test)
    passed = not (train_val or train_test or val_test)

    return {
        "status": "passed" if passed else "failed",
        "strategy": "record_id",
        "train_record_count": len(train),
        "validation_record_count": len(val),
        "test_record_count": len(test),
        "train_validation_overlap": train_val,
        "train_test_overlap": train_test,
        "validation_test_overlap": val_test,
    }


def class_weights(y, num_classes):
    """Return inverse-frequency class weights for a multiclass target."""
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=int(num_classes)).astype(np.float64)
    weights = np.zeros(int(num_classes), dtype=np.float32)
    present = counts > 0
    if np.any(present):
        weights[present] = len(y) / (int(num_classes) * counts[present])
    return weights
