from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "data" / "asl_landmarks_final.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Remove signs that need movement or aren't letters
df = df[~df["label"].isin(["J", "Z", "del", "space"])]

# Get only the landmark numbers
X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)

# Change 63 numbers into 21 landmarks with x, y, z
X = X.reshape(-1, 21, 3)

# Make the wrist (landmark 0) the starting point
X = X - X[:, :1, :]

# Find the size of each hand
scale = np.max(
    np.linalg.norm(X[:, :, :2], axis=2),
    axis=1,
)

# Prevent division by zero
scale[scale == 0] = 1

# Scale every hand to roughly the same size
X = X / scale[:, None, None]

# Turn the 21 landmarks back into 63 numbers
X = X.reshape(-1, 63)

# Correct answers
y = df["label"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst normalized example:")
print(X[0])

print("\nCorrect answer:")
print(y[0])

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
