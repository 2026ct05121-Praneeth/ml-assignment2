import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

FEATURE_MAP = {
    "X1": "limit_bal", "X2": "sex", "X3": "education", "X4": "marriage", "X5": "age",
    "X6": "pay_0", "X7": "pay_2", "X8": "pay_3", "X9": "pay_4", "X10": "pay_5", "X11": "pay_6",
    "X12": "bill_amt1", "X13": "bill_amt2", "X14": "bill_amt3", "X15": "bill_amt4", "X16": "bill_amt5", "X17": "bill_amt6",
    "X18": "pay_amt1", "X19": "pay_amt2", "X20": "pay_amt3", "X21": "pay_amt4", "X22": "pay_amt5", "X23": "pay_amt6"
}

def prepare_dataset():
    print("Fetching Default of Credit Card Clients dataset (UCI ID: 350)...")
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features.rename(columns=FEATURE_MAP).copy()
    y = dataset.data.targets.iloc[:, 0].astype(int).rename("default_next_month")

    df = pd.concat([X, y], axis=1)
    initial_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Initial instances: {initial_rows} | Duplicates removed: {initial_rows - len(df)} | Final instances: {len(df)}")
    print(f"Total missing values: {df.isnull().sum().sum()}")

    counts = df["default_next_month"].value_counts()
    pcts = df["default_next_month"].value_counts(normalize=True) * 100
    print("\n=== TARGET CLASS DISTRIBUTION ===")
    print(f"No Default (0): {counts[0]} ({pcts[0]:.2f}%)")
    print(f"Default (1):    {counts[1]} ({pcts[1]:.2f}%)")

    X_clean = df.drop(columns=["default_next_month"])
    y_clean = df["default_next_month"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.20, random_state=42, stratify=y_clean
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    os.makedirs("model/artifacts", exist_ok=True)
    train_df.to_csv("model/artifacts/train_data.csv", index=False)
    test_df.to_csv("test_data.csv", index=False)
    print(f"\nInput features: {X_clean.shape[1]} | Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print("Saved stratified test_data.csv to project root.")

if __name__ == "__main__":
    prepare_dataset()
