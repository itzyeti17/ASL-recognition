from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "asl_landmarks_final.csv"

df = pd.read_csv(DATA_PATH)

print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nLabels:")
print(df["label"].value_counts())
