"""
STEP 1 — Fairness Metrics Engine (core of the whole project)

What this does:
1. Loads a dataset with: features, true labels (loan approved: 0/1),
   a protected attribute (e.g. gender), and trains a simple model.
2. Computes fairness metrics using `fairlearn`.
3. Prints a bias report to the console.

HOW TO USE WITH YOUR REAL DATA:
Replace the `generate_demo_data()` function's output with your actual
credit-scoring CSV. Your CSV just needs these columns:
    - feature columns (income, credit_history, etc.)
    - 'label'              -> the TRUE outcome (0 = rejected, 1 = approved)
    - 'protected_attribute' -> e.g. gender, age_group, etc.
Then load it with: df = pd.read_csv("your_file.csv")
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    selection_rate,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score


def generate_demo_data(n=2000, seed=42):
    """Creates a synthetic loan-approval dataset with a built-in bias,
    so you can see the tool actually catch something."""
    rng = np.random.default_rng(seed)

    income = rng.normal(50000, 15000, n)
    credit_score = rng.normal(650, 80, n)
    protected_attribute = rng.choice(["Group_A", "Group_B"], size=n, p=[0.5, 0.5])

    # Intentionally bias the true approval outcome against Group_B
    bias_penalty = np.where(protected_attribute == "Group_B", -40, 0)
    approval_score = (income / 1000) + (credit_score / 10) + bias_penalty
    label = (approval_score > np.percentile(approval_score, 50)).astype(int)

    df = pd.DataFrame({
        "income": income,
        "credit_score": credit_score,
        "protected_attribute": protected_attribute,
        "label": label,
    })
    return df


def train_model(df):
    X = df[["income", "credit_score"]]
    y = df["label"]
    sensitive = df["protected_attribute"]

    X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred, sf_test


def compute_fairness_report(y_test, y_pred, sensitive_features):
    print("=" * 60)
    print("FAIRNESS AUDIT REPORT")
    print("=" * 60)

    overall_acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall model accuracy: {overall_acc:.3f}")

    # Per-group breakdown
    metric_frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "precision": precision_score,
            "recall": recall_score,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    print("\nPer-group metrics:")
    print(metric_frame.by_group)

    dp_diff = demographic_parity_difference(
        y_test, y_pred, sensitive_features=sensitive_features
    )
    dp_ratio = demographic_parity_ratio(
        y_test, y_pred, sensitive_features=sensitive_features
    )
    eo_diff = equalized_odds_difference(
        y_test, y_pred, sensitive_features=sensitive_features
    )

    print(f"\nDemographic Parity Difference: {dp_diff:.3f}  (0 = perfectly fair)")
    print(f"Demographic Parity Ratio:      {dp_ratio:.3f}  (1 = perfectly fair)")
    print(f"Equalized Odds Difference:     {eo_diff:.3f}  (0 = perfectly fair)")

    print("\nInterpretation:")
    if abs(dp_diff) > 0.1:
        disadvantaged = metric_frame.by_group["selection_rate"].idxmin()
        print(f"⚠️  Potential bias detected — '{disadvantaged}' has a "
              f"noticeably lower approval rate than the other group.")
    else:
        print("✅ No strong demographic parity violation detected.")

    print("=" * 60)
    return metric_frame


if __name__ == "__main__":
    df = generate_demo_data()
    model, X_test, y_test, y_pred, sf_test = train_model(df)
    compute_fairness_report(y_test, y_pred, sf_test)
