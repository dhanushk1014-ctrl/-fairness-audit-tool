"""
STEP 3 — FastAPI Backend + SHAP Explainability

Adds a second endpoint on top of Step 2:
    POST /explain -> upload a CSV, get back fairness metrics
                      PLUS which features drive the model's decisions,
                      broken down per protected group.

HOW TO RUN:
    uvicorn step3_api_with_shap:app --reload

Then visit: http://127.0.0.1:8000/docs
Try both /analyze (Step 2) and /explain (new, Step 3).

WHY THIS MATTERS:
Fairness metrics tell you THAT a model is biased.
SHAP tells you WHY -- which features are responsible, and whether they
affect one group more than another. This is the difference between
"the model is unfair" and "the model over-relies on credit_score for
Group_B in a way it doesn't for Group_A" -- the second is something
you can actually act on.
"""

import io
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI(title="Bias & Fairness Audit API")
app.add_middleware( CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"], )


@app.get("/")
def home():
    return {"message": "Fairness Audit API is running. Go to /docs to try it."}


def _load_and_validate_csv(contents: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as a CSV.")

    required_cols = {"label", "protected_attribute"}
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required column(s): {missing}. "
                   f"Your CSV needs 'label' and 'protected_attribute' columns."
        )
    return df


def _train_and_split(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in {"label", "protected_attribute"}]
    if not feature_cols:
        raise HTTPException(status_code=400, detail="No feature columns found besides label/protected_attribute.")

    X = df[feature_cols]
    y = df["label"]
    sensitive = df["protected_attribute"]

    X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_test, sf_test, feature_cols


@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    """Fairness metrics only (same as Step 2)."""
    contents = await file.read()
    df = _load_and_validate_csv(contents)
    model, X_train, X_test, y_test, sf_test, feature_cols = _train_and_split(df)
    y_pred = model.predict(X_test)

    metric_frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "precision": precision_score,
            "recall": recall_score,
        },
        y_true=y_test, y_pred=y_pred, sensitive_features=sf_test,
    )
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sf_test)
    dp_ratio = demographic_parity_ratio(y_test, y_pred, sensitive_features=sf_test)
    eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sf_test)
    disadvantaged_group = metric_frame.by_group["selection_rate"].idxmin()
    bias_flag = abs(dp_diff) > 0.1

    return {
        "overall_accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
        "per_group_metrics": metric_frame.by_group.round(3).to_dict(),
        "fairness_metrics": {
            "demographic_parity_difference": round(float(dp_diff), 3),
            "demographic_parity_ratio": round(float(dp_ratio), 3),
            "equalized_odds_difference": round(float(eo_diff), 3),
        },
        "bias_detected": bool(bias_flag),
        "most_disadvantaged_group": disadvantaged_group if bias_flag else None,
    }


@app.post("/explain")
async def explain_csv(file: UploadFile = File(...)):
    """
    Fairness metrics PLUS SHAP feature importance, broken down per
    protected group -- shows WHICH features drive bias.
    """
    contents = await file.read()
    df = _load_and_validate_csv(contents)
    model, X_train, X_test, y_test, sf_test, feature_cols = _train_and_split(df)

    # --- SHAP explainability ---
    # Use a small background sample for speed (standard SHAP practice)
    background = shap.sample(X_train, min(100, len(X_train)), random_state=42)
    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer.shap_values(X_test)

    # Average absolute SHAP value per feature, split by group
    shap_df = pd.DataFrame(shap_values, columns=feature_cols, index=X_test.index)
    shap_df["protected_attribute"] = sf_test.values

    per_group_importance = (
        shap_df.groupby("protected_attribute")[feature_cols]
        .apply(lambda g: g.abs().mean())
        .round(3)
        .to_dict(orient="index")
    )

    overall_importance = (
        shap_df[feature_cols].abs().mean().round(3).sort_values(ascending=False).to_dict()
    )

    # Find the feature with the biggest gap in importance between groups
    importance_df = pd.DataFrame(per_group_importance)
    feature_gaps = (importance_df.max(axis=1) - importance_df.min(axis=1)).sort_values(ascending=False)
    top_gap_feature = feature_gaps.index[0]
    top_gap_value = round(float(feature_gaps.iloc[0]), 3)

    return {
        "overall_feature_importance": overall_importance,
        "feature_importance_by_group": per_group_importance,
        "most_unevenly_weighted_feature": {
            "feature": top_gap_feature,
            "importance_gap_between_groups": top_gap_value,
            "explanation": (
                f"'{top_gap_feature}' influences the model's decision very differently "
                f"depending on the group -- this is often a root cause of unfair outcomes."
            )
        }
    }
