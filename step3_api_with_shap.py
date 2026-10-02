"""
STEP 3 — FastAPI Backend + SHAP Explainability + PDF Report Export

Endpoints:
    POST /analyze -> fairness metrics only
    POST /explain -> fairness metrics + SHAP feature importance by group
    POST /report  -> everything above, rendered as a downloadable PDF

HOW TO RUN:
    uvicorn step3_api_with_shap:app --reload

Then visit: http://127.0.0.1:8000/docs

WHY THIS MATTERS:
Fairness metrics tell you THAT a model is biased.
SHAP tells you WHY -- which features are responsible, and whether they
affect one group more than another. This is the difference between
"the model is unfair" and "the model over-relies on credit_score for
Group_B in a way it doesn't for Group_A" -- the second is something
you can actually act on. The PDF report packages both into something
you can hand to a non-technical stakeholder.
"""

import io
from datetime import datetime
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

app = FastAPI(title="Bias & Fairness Audit API")

# Allow the React frontend (running on localhost:5173 by default with Vite)
# to call this API from the browser. Without this, the browser blocks the
# request with a CORS error even though the server itself is working fine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _compute_fairness(df: pd.DataFrame):
    """Shared logic used by /analyze, /explain, and /report."""
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

    fairness_result = {
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
    return fairness_result, model, X_train, X_test, sf_test, feature_cols


def _compute_explainability(model, X_train, X_test, sf_test, feature_cols):
    """Shared SHAP logic used by /explain and /report."""
    background = shap.sample(X_train, min(100, len(X_train)), random_state=42)
    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer.shap_values(X_test)

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


@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    """Fairness metrics only (same as Step 2)."""
    contents = await file.read()
    df = _load_and_validate_csv(contents)
    fairness_result, *_ = _compute_fairness(df)
    return fairness_result


@app.post("/explain")
async def explain_csv(file: UploadFile = File(...)):
    """
    Fairness metrics PLUS SHAP feature importance, broken down per
    protected group -- shows WHICH features drive bias.
    """
    contents = await file.read()
    df = _load_and_validate_csv(contents)
    _, model, X_train, X_test, sf_test, feature_cols = _compute_fairness(df)
    return _compute_explainability(model, X_train, X_test, sf_test, feature_cols)


def _build_pdf(fairness, explain, filename_hint: str) -> io.BytesIO:
    """Builds a formatted PDF report from the fairness + explainability results."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], fontSize=20, spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], textColor=colors.grey, spaceAfter=18
    )
    heading_style = ParagraphStyle(
        "HeadingCustom", parent=styles["Heading2"], spaceBefore=18, spaceAfter=8,
        textColor=colors.HexColor("#4f46e5")
    )
    callout_style = ParagraphStyle(
        "Callout", parent=styles["Normal"], backColor=colors.HexColor("#f0f0fb"),
        borderPadding=10, spaceAfter=10, leading=15,
    )

    elements = []
    elements.append(Paragraph("Bias &amp; Fairness Audit Report", title_style))
    elements.append(Paragraph(
        f"Source file: {filename_hint} &nbsp;|&nbsp; Generated: "
        f"{datetime.now().strftime('%d %b %Y, %I:%M %p')}",
        subtitle_style
    ))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#e2e2ea")))

    # --- Summary ---
    elements.append(Paragraph("Summary", heading_style))
    bias_text = (
        f'<font color="#b3261e"><b>Bias Detected:</b></font> the '
        f"<b>{fairness['most_disadvantaged_group']}</b> group shows a meaningfully "
        f"lower approval rate than others."
        if fairness["bias_detected"] else
        '<font color="#1a7a3c"><b>No Bias Detected:</b></font> no strong demographic '
        "parity violation was found in this run."
    )
    elements.append(Paragraph(
        f"Overall model accuracy: <b>{fairness['overall_accuracy']}</b><br/>{bias_text}",
        callout_style
    ))

    # --- Fairness metrics table ---
    elements.append(Paragraph("Fairness Metrics", heading_style))
    fm = fairness["fairness_metrics"]
    fm_data = [
        ["Metric", "Value", "Interpretation"],
        ["Demographic Parity Difference", fm["demographic_parity_difference"], "0 = perfectly fair"],
        ["Demographic Parity Ratio", fm["demographic_parity_ratio"], "1 = perfectly fair"],
        ["Equalized Odds Difference", fm["equalized_odds_difference"], "0 = perfectly fair"],
    ]
    elements.append(_styled_table(fm_data, col_widths=[7 * cm, 3 * cm, 5 * cm]))

    # --- Per-group breakdown table ---
    elements.append(Paragraph("Per-Group Breakdown", heading_style))
    pg = fairness["per_group_metrics"]
    groups = list(pg["accuracy"].keys())
    pg_data = [["Group", "Accuracy", "Selection Rate", "Precision", "Recall"]]
    for g in groups:
        pg_data.append([
            g, pg["accuracy"][g], pg["selection_rate"][g], pg["precision"][g], pg["recall"][g]
        ])
    elements.append(_styled_table(pg_data, col_widths=[4 * cm] + [3.5 * cm] * 4))

    # --- Feature importance ---
    if explain:
        elements.append(Paragraph("Feature Importance by Group (SHAP)", heading_style))
        gap = explain["most_unevenly_weighted_feature"]
        elements.append(Paragraph(
            f"<b>{gap['feature']}</b> shows the largest importance gap between groups "
            f"({gap['importance_gap_between_groups']}) — {gap['explanation']}",
            callout_style
        ))
        by_group = explain["feature_importance_by_group"]
        features = list(next(iter(by_group.values())).keys())
        fi_data = [["Feature"] + list(by_group.keys())]
        for feat in features:
            fi_data.append([feat] + [by_group[g][feat] for g in by_group])
        elements.append(_styled_table(fi_data, col_widths=[6 * cm] + [4 * cm] * len(by_group)))

    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#e2e2ea")))
    elements.append(Paragraph(
        "Generated by the Bias &amp; Fairness Audit Tool — a project built to detect and "
        "explain demographic bias in classification models using Fairlearn and SHAP.",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey, spaceBefore=10)
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer


def _styled_table(data, col_widths):
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e2ea")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7fb")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 1), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
    ]))
    return table


@app.post("/report")
async def report_csv(file: UploadFile = File(...)):
    """
    Runs the full fairness + explainability audit and returns a
    downloadable PDF report instead of raw JSON.
    """
    contents = await file.read()
    df = _load_and_validate_csv(contents)
    fairness_result, model, X_train, X_test, sf_test, feature_cols = _compute_fairness(df)
    explain_result = _compute_explainability(model, X_train, X_test, sf_test, feature_cols)

    pdf_buffer = _build_pdf(fairness_result, explain_result, filename_hint=file.filename or "uploaded_data.csv")

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=fairness_audit_report.pdf"},
    )
