from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "row" / "twcs.csv"
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATES ==========")
print("Duplicate rows:", df.duplicated().sum())

if "name" in df.columns:
    print("\n========== TOP BRANDS ==========")
    print(df["name"].value_counts().head(20))

if "inbound" in df.columns:
    print("\n========== INBOUND ==========")
    print(df["inbound"].value_counts())

print("\n========== FIRST 10 ROWS ==========")
print(df.head(10).to_string())