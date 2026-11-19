"""Shared loading and preprocessing for ASL landmark data."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).resolve().parent / "data" / "asl_landmarks_final.csv"
EXCLUDED_LABELS = ("J", "Z", "del", "space")
LANDMARK_COUNT = 21
VALUES_PER_LANDMARK = 3
FEATURE_COUNT = LANDMARK_COUNT * VALUES_PER_LANDMARK


def normalize_landmarks(landmarks):
    """Center one hand on its wrist and normalize its 2D size."""
    points = np.asarray(landmarks, dtype=np.float32)
    if points.size != FEATURE_COUNT:
        raise ValueError(
            f"Expected {FEATURE_COUNT} landmark values, received {points.size}."
        )

    points = points.reshape(LANDMARK_COUNT, VALUES_PER_LANDMARK)
    points = points - points[:1]

    scale = np.max(np.linalg.norm(points[:, :2], axis=1))
    if scale > 0:
        points = points / scale

    return points.reshape(FEATURE_COUNT)


def load_dataset(path=DATA_PATH):
    """Load, filter, and normalize the static-letter landmark dataset."""
    data = pd.read_csv(path)
    if "label" not in data.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    feature_columns = [
        f"{axis}{landmark}"
        for landmark in range(LANDMARK_COUNT)
        for axis in ("x", "y", "z")
    ]
    actual_feature_columns = [column for column in data.columns if column != "label"]
    if actual_feature_columns != feature_columns:
        raise ValueError(
            "Dataset feature columns must be ordered x0, y0, z0 through "
            "x20, y20, z20."
        )

    data = data[~data["label"].isin(EXCLUDED_LABELS)]
    raw_features = data[feature_columns].to_numpy(dtype=np.float32)
    features = np.stack([normalize_landmarks(row) for row in raw_features])
    labels = data["label"].to_numpy()
    return features, labels


def split_dataset(features, labels, test_size=0.20, random_state=42):
    """Create a reproducible, stratified training/test split."""
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
