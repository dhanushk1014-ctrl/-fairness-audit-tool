import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";
import "./App.css";

// Change this if your FastAPI backend runs on a different port
const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analyzeResult, setAnalyzeResult] = useState(null);
  const [explainResult, setExplainResult] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setAnalyzeResult(null);
    setExplainResult(null);
    setError(null);
  };

  const runAudit = async () => {
    if (!file) {
      setError("Please choose a CSV file first.");
      return;
    }
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const [analyzeRes, explainRes] = await Promise.all([
        fetch(`${API_BASE}/analyze`, { method: "POST", body: formData }),
        fetch(`${API_BASE}/explain`, { method: "POST", body: formData }),
      ]);

      if (!analyzeRes.ok || !explainRes.ok) {
        const errBody = !analyzeRes.ok ? await analyzeRes.json() : await explainRes.json();
        throw new Error(errBody.detail || "Something went wrong on the server.");
      }

      setAnalyzeResult(await analyzeRes.json());
      setExplainResult(await explainRes.json());
    } catch (err) {
      setError(
        err.message.includes("Failed to fetch")
          ? "Could not reach the backend. Is your FastAPI server running on port 8000?"
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  // Reshape feature_importance_by_group for recharts:
  // [{ feature: "income", Group_A: 0.69, Group_B: 0.64 }, ...]
  const chartData = () => {
    if (!explainResult) return [];
    const byGroup = explainResult.feature_importance_by_group;
    const groups = Object.keys(byGroup);
    const features = Object.keys(byGroup[groups[0]]);

    return features.map((feature) => {
      const row = { feature };
      groups.forEach((g) => {
        row[g] = byGroup[g][feature];
      });
      return row;
    });
  };

  const groupNames = explainResult
    ? Object.keys(explainResult.feature_importance_by_group)
    : [];

  return (
    <div className="page">
      <header className="header">
        <h1>Bias & Fairness Audit Tool</h1>
        <p className="subtitle">
          Upload a CSV of model predictions to check for demographic bias and see which features drive it.
        </p>
      </header>

      <section className="upload-card">
        <input
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          className="file-input"
        />
        <button onClick={runAudit} disabled={loading} className="run-btn">
          {loading ? "Running Audit…" : "Run Audit"}
        </button>
        <p className="hint">
          CSV must include a <code>label</code> column, a <code>protected_attribute</code> column,
          and one or more numeric feature columns.
        </p>
        {error && <div className="error-box">{error}</div>}
      </section>

      {analyzeResult && (
        <section className="results">
          <div className="summary-row">
            <div className="stat-card">
              <span className="stat-label">Overall Accuracy</span>
              <span className="stat-value">{analyzeResult.overall_accuracy}</span>
            </div>
            <div className={`stat-card ${analyzeResult.bias_detected ? "bias-yes" : "bias-no"}`}>
              <span className="stat-label">Bias Detected</span>
              <span className="stat-value">
                {analyzeResult.bias_detected ? "⚠️ Yes" : "✅ No"}
              </span>
            </div>
            {analyzeResult.most_disadvantaged_group && (
              <div className="stat-card">
                <span className="stat-label">Disadvantaged Group</span>
                <span className="stat-value">{analyzeResult.most_disadvantaged_group}</span>
              </div>
            )}
          </div>

          <div className="panel">
            <h2>Fairness Metrics</h2>
            <table className="metrics-table">
              <tbody>
                <tr>
                  <td>Demographic Parity Difference</td>
                  <td>{analyzeResult.fairness_metrics.demographic_parity_difference}</td>
                  <td className="note-col">0 = perfectly fair</td>
                </tr>
                <tr>
                  <td>Demographic Parity Ratio</td>
                  <td>{analyzeResult.fairness_metrics.demographic_parity_ratio}</td>
                  <td className="note-col">1 = perfectly fair</td>
                </tr>
                <tr>
                  <td>Equalized Odds Difference</td>
                  <td>{analyzeResult.fairness_metrics.equalized_odds_difference}</td>
                  <td className="note-col">0 = perfectly fair</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="panel">
            <h2>Per-Group Breakdown</h2>
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Group</th>
                  <th>Accuracy</th>
                  <th>Selection Rate</th>
                  <th>Precision</th>
                  <th>Recall</th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(analyzeResult.per_group_metrics.accuracy).map((group) => (
                  <tr key={group}>
                    <td>{group}</td>
                    <td>{analyzeResult.per_group_metrics.accuracy[group]}</td>
                    <td>{analyzeResult.per_group_metrics.selection_rate[group]}</td>
                    <td>{analyzeResult.per_group_metrics.precision[group]}</td>
                    <td>{analyzeResult.per_group_metrics.recall[group]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {explainResult && (
            <div className="panel">
              <h2>Feature Importance by Group (SHAP)</h2>
              <p className="callout">
                <strong>{explainResult.most_unevenly_weighted_feature.feature}</strong> shows the
                largest importance gap between groups
                ({explainResult.most_unevenly_weighted_feature.importance_gap_between_groups}) —
                {" "}{explainResult.most_unevenly_weighted_feature.explanation}
              </p>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={chartData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="feature" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {groupNames.map((g, i) => (
                    <Bar key={g} dataKey={g} fill={i === 0 ? "#4f46e5" : "#e11d48"} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>
      )}
    </div>
  );
}

export default App;
