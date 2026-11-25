import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, roc_curve
)

NOMINAL_COLS = ["sex", "education", "marriage"]

def run_benchmark():
    train_df = pd.read_csv("model/artifacts/train_data.csv")
    test_df = pd.read_csv("test_data.csv")

    X_train = train_df.drop(columns=["default_next_month"])
    y_train = train_df["default_next_month"]
    X_test = test_df.drop(columns=["default_next_month"])
    y_test = test_df["default_next_month"]

    numeric_cols = [c for c in X_train.columns if c not in NOMINAL_COLS]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, NOMINAL_COLS)
    ])

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=11, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=10, random_state=42, n_jobs=-1
        )
    }

    os.makedirs("model/trained_artifacts", exist_ok=True)
    results = []
    roc_data = {}

    for name, clf in classifiers.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        results.append({
            "ML Model Name": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "AUC": round(roc_auc_score(y_test, y_prob), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_test, y_prob))

        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
        joblib.dump(pipe, os.path.join("model/trained_artifacts", slug))

    results_df = pd.DataFrame(results)
    results_df.to_csv("model/results_summary.csv", index=False)
    print("\n=== MODEL EVALUATION COMPARISON TABLE ===")
    print(results_df.to_string(index=False))

    metrics = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    x = np.arange(len(metrics))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, row in results_df.iterrows():
        vals = [row[m] for m in metrics]
        ax.bar(x + (i - 2) * width, vals, width, label=row["ML Model Name"])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Classification Benchmark Across 6 Evaluation Metrics (Credit Card Default)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("model/metric_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8.5, 6))
    for name, (fpr, tpr, auc_val) in roc_data.items():
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Baseline (AUC = 0.5000)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison Across All 5 Classification Models")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig("model/roc_comparison.png", dpi=200)
    plt.close()
    print("\nSaved models to model/trained_artifacts/ and plots to model/.")

if __name__ == "__main__":
    run_benchmark()
