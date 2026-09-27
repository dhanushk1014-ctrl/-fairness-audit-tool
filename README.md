# Bias & Fairness Audit Tool

A full-stack tool that audits machine learning classifiers for demographic bias and explains *which features* drive that bias — built on top of a credit-scoring use case.

Most fairness tools stop at "is this model biased?" This one goes further: it uses **SHAP explainability** to show *why*, breaking down feature importance per demographic group so the bias is actionable, not just flagged.

## Why This Project Exists

Machine learning models used in high-stakes decisions — loan approvals, hiring, lending — can unintentionally discriminate against protected groups even when the developer never intended it. Standard accuracy metrics don't catch this. This tool audits a model's predictions across multiple fairness metrics and surfaces the specific features responsible for any disparity.

## Features

- **Fairness metrics engine** — computes Demographic Parity Difference/Ratio and Equalized Odds Difference using [Fairlearn](https://fairlearn.org/)
- **Explainability layer** — uses [SHAP](https://shap.readthedocs.io/) to show per-group feature importance, and automatically flags the feature with the largest importance gap between groups
- **REST API** — FastAPI backend with `/analyze` and `/explain` endpoints, interactive Swagger docs at `/docs`
- **Dashboard UI** — React frontend to upload a CSV and view results as stat cards, tables, and a comparison bar chart
- **Bring-your-own-data** — works with any binary classification dataset that includes a protected attribute column

## Tech Stack

| Layer | Tools |
|---|---|
| Fairness metrics | Python, Fairlearn |
| Explainability | SHAP |
| Backend | FastAPI, scikit-learn, pandas |
| Frontend | React (Vite), Recharts |

## Project Structure

```
.
├── step1_fairness_metrics.py   # Core fairness metrics logic (standalone script)
├── step3_api_with_shap.py      # FastAPI backend (fairness metrics + SHAP explainability)
├── sample_data.csv             # Synthetic demo dataset with a built-in bias
├── frontend/                   # React dashboard
│   ├── src/
│   │   ├── App.jsx
│   │   └── App.css
│   └── package.json
└── README.md
```

## Getting Started

### Backend

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install fastapi uvicorn pandas scikit-learn fairlearn shap python-multipart

uvicorn step3_api_with_shap:app --reload
```

Backend runs at `http://127.0.0.1:8000` — interactive docs at `http://127.0.0.1:8000/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`.

### Try It

1. Start both servers (backend and frontend, in separate terminals)
2. Open `http://localhost:5173`
3. Upload `sample_data.csv`
4. Click **Run Audit**

## Input Data Format

Your CSV needs:
- One or more numeric feature columns (e.g. `income`, `credit_score`)
- `label` — the true outcome (0 or 1)
- `protected_attribute` — the group to check for bias (e.g. gender, age group)

## Example Output

On the included synthetic dataset, the tool detects that while overall approval rates look balanced between groups (Demographic Parity Difference: 0.002), the model's *error rates* differ sharply between groups (Equalized Odds Difference: 0.446) — a bias that a naive fairness check would miss entirely.

## What's Next

- [ ] PDF report export
- [ ] Bias mitigation suggestions (reweighing / threshold adjustment via Fairlearn)
- [ ] Support for multiple protected attributes simultaneously
- [ ] Deploy live demo (Render/Railway + Vercel)

## Author

Dhanush — B.Tech Information Technology (Data Science), Sri Sairam Engineering College
