"""
STEP 2 — FastAPI Backend

What this does:
Turns step1's fairness logic into a real web API with one endpoint:
    POST /analyze  -> upload a CSV, get back a fairness report as JSON

HOW TO RUN THIS:
    uvicorn step2_api:app --reload

Then open in your browser:
    http://127.0.0.1:8000/docs

That gives you an interactive page where you can upload a CSV and test
the API immediately -- no frontend needed yet.

YOUR CSV MUST HAVE THESE COLUMNS:
    - one or more feature columns (numeric) -> e.g. income, credit_score
    - 'label'                -> true outcome (0 or 1)
    - 'protected_attribute'  -> the group to check for bias (e.g. gender)

If your real dataset uses different column names, either rename them
in your CSV before uploading, or tell me the real names and I'll adjust
the code.
"""

import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
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


@app.get("/")
def home():
    """Quick check that the server is running."""
    return {"message": "Fairness Audit API is running. Go to /docs to try it."}


@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    """
    Upload a CSV with feature columns, 'label', and 'protected_attribute'.
    Returns fairness metrics comparing groups in 'protected_attribute'.
    """
    # --- Read the uploaded CSV into a DataFrame ---
    contents = await file.read()
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

    # --- Everything except label/protected_attribute is treated as a feature ---
    feature_cols = [c for c in df.columns if c not in required_cols]
    if not feature_cols:
        raise HTTPException(status_code=400, detail="No feature columns found besides label/protected_attribute.")

    X = df[feature_cols]
    y = df["label"]
    sensitive = df["protected_attribute"]

    # --- Train a simple model and get predictions ---
    X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Compute fairness metrics (same logic as Step 1) ---
    metric_frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "precision": precision_score,
            "recall": recall_score,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sf_test,
    )

    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sf_test)
    dp_ratio = demographic_parity_ratio(y_test, y_pred, sensitive_features=sf_test)
    eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sf_test)

    disadvantaged_group = metric_frame.by_group["selection_rate"].idxmin()
    bias_flag = abs(dp_diff) > 0.1

    # --- Build the JSON response ---
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
        "note": "0 = perfectly fair for difference metrics; 1 = perfectly fair for ratio metrics."
    }
