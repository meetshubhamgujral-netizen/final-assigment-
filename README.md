# 🪐 Saturn Insight Studio

A dynamic **descriptive + diagnostic** analytics dashboard for any tabular dataset.
Upload a CSV / Excel / JSON file and the app automatically detects column types
(IDs, categories, Likert/scale items, continuous numbers), then adapts its
charts, statistical tests, and a plain-language **auto-insights** narrative to fit.

Built to run on **Streamlit Community Cloud** from a GitHub repo.

---

## What it does

| Section | What you get |
|---|---|
| **Overview** | Record/variable counts, data-quality KPIs, detected column types, quick insights, data preview. |
| **Descriptive** | Likert item ranking (mean, Top-2-Box / Bottom-2-Box), distributions, categorical mix, correlation matrix. |
| **Diagnostic** | Group comparison with the **correct test auto-selected** (Kruskal–Wallis for ordinal/Likert, ANOVA for continuous), cross-tab association (chi-square + Cramér's V), and correlation-based driver analysis — each with effect sizes and a plain-English verdict. |
| **Modeling** | Trains and compares **KNN, Decision Tree, Random Forest, and Gradient Boosting** on a target you pick. Task type (classification / regression) auto-detected; reports accuracy/F1 or R²/RMSE vs a baseline, feature importance, and a confusion matrix or actual-vs-predicted plot. |
| **Summary & Insights** | Auto-generated narrative that regenerates for whatever you load, plus a downloadable Markdown report. |

The app is **survey-aware**: if your Excel workbook has a question-key sheet
(a small sheet mapping `Q1, Q2, …` to question text), it's detected automatically
and used for labels. Otherwise everything still works generically.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run saturn_dashboard.py
```

Then click **✨ Load sample survey** in the sidebar to explore immediately, or upload your own file.

---

## Deploy on Streamlit Community Cloud (the 3 fixes that unblock it)

Your previous deploy failed for three reasons. All are fixed in this repo:

1. **Missing `requirements.txt`** — Cloud installs *only* Streamlit unless this
   file exists, which is why `plotly` / `scipy` / `matplotlib` threw
   `ModuleNotFoundError`. It's now included.
2. **Python 3.14** — bleeding-edge; heavy packages often have no wheels yet, so
   installs fail. Use **Python 3.12** (the Cloud default) — see step 4.
3. **A stray broken file** (`gemini-code-*.py`, the `r_sel = st.` SyntaxError).
   **Delete it from your repo** — keep a single entrypoint.

### Steps

1. Put these files in your GitHub repo (root):
   ```
   saturn_dashboard.py
   requirements.txt
   .streamlit/config.toml      (optional theme)
   README.md
   ```
   and **delete** `gemini-code-1782632970684.py` (and any other duplicate app file).
2. Go to <https://share.streamlit.io> → **Create app** → pick your repo + branch.
3. Set **Main file path** to `saturn_dashboard.py`.
4. Open **Advanced settings** → set **Python version = 3.12** (do *not* use 3.14).
5. Click **Deploy**. First build takes a few minutes while dependencies install.

> Already deployed on 3.14? Streamlit can't change an app's Python version in
> place — **delete the app and redeploy**, selecting 3.12 in Advanced settings.

---

## Files

```
saturn_dashboard.py     # the app (single file)
requirements.txt        # pinned dependencies (fixes ModuleNotFoundError)
.streamlit/config.toml  # theme + upload limit
README.md               # this file
```

## Notes

- Diagnostic tests report **effect sizes** (η² / ε² / Cramér's V), not just
  p-values, so "significant" never gets mistaken for "large."
- The **Modeling** page needs `scikit-learn` (already in requirements.txt). This is
  exactly the kind of package that has no wheels on Python 3.14 — another reason to
  deploy on **Python 3.12**.
- Correlation is association, not causation — the app says so where it matters.
- No data leaves your session; analysis runs entirely on the uploaded file.
