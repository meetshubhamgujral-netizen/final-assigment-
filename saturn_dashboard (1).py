"""
Saturn Insight Studio
A dynamic descriptive + diagnostic analytics dashboard for any tabular dataset.
Built for Streamlit Community Cloud. Single-file app.

Run locally:   streamlit run saturn_dashboard.py
Dependencies:  see requirements.txt
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

# scikit-learn is optional: the Modeling page degrades gracefully if it's absent.
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                                  GradientBoostingClassifier, GradientBoostingRegressor)
    from sklearn.metrics import (accuracy_score, f1_score, r2_score,
                                 mean_squared_error, mean_absolute_error, confusion_matrix)
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ============================================================================
# PAGE CONFIG + THEME
# ============================================================================
st.set_page_config(
    page_title="Saturn Insight Studio",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#1f6f78", "#3a9aa3", "#6fb7bd", "#e07a5f", "#f2cc8f", "#81a4cd", "#3d5a80"]
PX_TEMPLATE = "plotly_white"

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1300px; }
      .kpi {
          background: #ffffff; border: 1px solid #ECECEC; border-radius: 12px;
          padding: 1.1rem 1.2rem; box-shadow: 0 1px 3px rgba(16,24,40,.04);
          border-left: 4px solid #1f6f78; height: 100%;
      }
      .kpi .label { color:#667085; font-size:.72rem; font-weight:700;
          text-transform:uppercase; letter-spacing:.05em; margin:0; }
      .kpi .value { color:#101828; font-size:1.7rem; font-weight:700; margin:.15rem 0 0 0; }
      .kpi .sub   { color:#98A2B3; font-size:.78rem; margin:.1rem 0 0 0; }
      .insight {
          background:#F7FAFA; border:1px solid #E3EDED; border-left:4px solid #1f6f78;
          border-radius:10px; padding:.7rem 1rem; margin-bottom:.6rem; font-size:.95rem;
      }
      h1, h2, h3 { color:#1d2939; }
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi(col, label, value, sub=""):
    col.markdown(
        f'<div class="kpi"><p class="label">{label}</p>'
        f'<p class="value">{value}</p><p class="sub">{sub}</p></div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# SESSION STATE
# ============================================================================
for key in ("df", "profile", "labels", "source_name"):
    if key not in st.session_state:
        st.session_state[key] = None


# ============================================================================
# DATA PROFILING (validated against real data)
# ============================================================================
def is_likert(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    vals = series.dropna()
    if vals.empty:
        return False
    uniq = vals.unique()
    if not np.all(np.equal(np.mod(uniq, 1), 0)):
        return False
    mn, mx, k = float(vals.min()), float(vals.max()), len(uniq)
    return (mn >= 0) and (mx <= 10) and (k <= 11) and (mx - mn >= 1)


def is_id(series: pd.Series, n_rows: int) -> bool:
    name = str(series.name).lower()
    if any(t in name for t in ("respondent", "uuid", "guid")) or name == "id" or name.endswith("_id"):
        return True
    if series.nunique(dropna=True) >= 0.95 * n_rows and not pd.api.types.is_numeric_dtype(series):
        return True
    return False


def profile_columns(df: pd.DataFrame) -> dict:
    n = len(df)
    id_cols, likert_cols, numeric_cols, categorical_cols, datetime_cols = [], [], [], [], []
    for col in df.columns:
        s = df[col]
        if is_id(s, n):
            id_cols.append(col); continue
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col); continue
        if pd.api.types.is_numeric_dtype(s):
            if is_likert(s):
                likert_cols.append(col)
            elif s.nunique(dropna=True) <= 10:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
            continue
        try:
            parsed = pd.to_datetime(s.dropna().astype(str).head(50), errors="coerce")
            if len(parsed) and parsed.notna().mean() > 0.8:
                datetime_cols.append(col); continue
        except Exception:
            pass
        categorical_cols.append(col)
    return {
        "n_rows": n, "n_cols": df.shape[1],
        "id_cols": id_cols, "likert_cols": likert_cols, "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols, "datetime_cols": datetime_cols,
        "group_factors": categorical_cols,
        "metrics": likert_cols + numeric_cols,
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


def lbl(col: str) -> str:
    """Return a human label for a column if a key/label map was detected."""
    labels = st.session_state.get("labels") or {}
    txt = labels.get(col)
    return f"{col} — {txt}" if txt else col


# ============================================================================
# ANALYTICS
# ============================================================================
def likert_summary(df: pd.DataFrame, cols, scale_max=None) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        smax = scale_max or int(s.max())
        top2 = (s >= smax - 1).mean()
        bot2 = (s <= 2).mean()
        rows.append({
            "Item": c, "Mean": s.mean(), "Std": s.std(),
            "Top-2-Box %": top2 * 100, "Neutral %": (1 - top2 - bot2) * 100,
            "Bottom-2-Box %": bot2 * 100, "n": int(s.count()),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Mean", ascending=False).reset_index(drop=True)


def group_difference(df, factor, metric, ordinal=True):
    sub = df[[factor, metric]].dropna()
    grouped = list(sub.groupby(factor))
    groups = [g[metric].values for _, g in grouped if len(g) >= 2]
    labels = [name for name, g in grouped if len(g) >= 2]
    if len(groups) < 2:
        return None
    if ordinal:
        stat, p = stats.kruskal(*groups)
        test = "Kruskal–Wallis H"
        N = sum(len(g) for g in groups)
        eff = (stat - len(groups) + 1) / (N - len(groups)) if N - len(groups) > 0 else np.nan
        eff_name = "ε² (epsilon-squared)"
    else:
        stat, p = stats.f_oneway(*groups)
        test = "One-way ANOVA F"
        grand = sub[metric].mean()
        ss_b = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
        ss_t = ((sub[metric] - grand) ** 2).sum()
        eff = ss_b / ss_t if ss_t > 0 else np.nan
        eff_name = "η² (eta-squared)"
    means = sub.groupby(factor)[metric].agg(["mean", "std", "count"]).reset_index()
    return {"test": test, "stat": float(stat), "p": float(p), "effect": float(eff),
            "effect_name": eff_name, "labels": labels, "means": means}


def crosstab_chi2(df, a, b):
    ct = pd.crosstab(df[a], df[b])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    mn = min(ct.shape)
    v = np.sqrt(chi2 / (n * (mn - 1))) if mn > 1 else np.nan
    return {"table": ct, "chi2": float(chi2), "p": float(p), "dof": int(dof), "cramers_v": float(v)}


def correlation_drivers(df, target, cols):
    others = [c for c in cols if c != target]
    rows = []
    for c in others:
        sub = df[[target, c]].dropna()
        if len(sub) < 3:
            continue
        rows.append({"Variable": c,
                     "Pearson r": sub[target].corr(sub[c], method="pearson"),
                     "Spearman ρ": sub[target].corr(sub[c], method="spearman")})
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    return d.reindex(d["Pearson r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# ----------------------------- predictive modeling -----------------------------
def build_models(task):
    if task == "classification":
        return {
            "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
            "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
    return {
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7)),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }


def run_models(df, target, features, task):
    data = df[features + [target]].dropna()
    X = pd.get_dummies(data[features], drop_first=True)
    y = data[target]
    strat = y if (task == "classification" and y.value_counts().min() >= 2) else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)

    results, fitted = [], {}
    for name, model in build_models(task).items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        fitted[name] = model
        if task == "classification":
            results.append({"Model": name, "Accuracy": accuracy_score(y_te, pred),
                            "Macro F1": f1_score(y_te, pred, average="macro", zero_division=0)})
        else:
            results.append({"Model": name, "R2": r2_score(y_te, pred),
                            "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
                            "MAE": float(mean_absolute_error(y_te, pred))})
    sort_key = "Accuracy" if task == "classification" else "R2"
    res_df = pd.DataFrame(results).sort_values(sort_key, ascending=False).reset_index(drop=True)
    baseline = y_te.value_counts(normalize=True).max() if task == "classification" else 0.0

    best_name = res_df.iloc[0]["Model"]
    best = fitted[best_name]
    est = best.steps[-1][1] if hasattr(best, "steps") else best
    importances = None
    if hasattr(est, "feature_importances_"):
        importances = pd.DataFrame({"Feature": X.columns, "Importance": est.feature_importances_}
                                   ).sort_values("Importance", ascending=False).reset_index(drop=True)
    return {"results": res_df, "baseline": baseline, "best_name": best_name,
            "y_test": y_te, "y_pred": best.predict(X_te), "importances": importances,
            "classes": sorted(y.unique().tolist()) if task == "classification" else None}


def effect_word(eff):
    a = abs(eff)
    if np.isnan(a):
        return "n/a"
    if a < 0.01:
        return "negligible"
    if a < 0.06:
        return "small"
    if a < 0.14:
        return "moderate"
    return "large"


def generate_insights(df, prof):
    out, n = [], prof["n_rows"]
    out.append(f"The dataset contains **{n:,} records** across **{prof['n_cols']} variables** "
               f"({len(prof['categorical_cols'])} categorical, {len(prof['likert_cols'])} scale/Likert, "
               f"{len(prof['numeric_cols'])} continuous numeric).")
    miss_pct = prof["missing"] / max(n * prof["n_cols"], 1) * 100
    q = " The data is clean and analysis-ready." if (miss_pct < 1 and prof["duplicates"] == 0) else ""
    out.append(f"**Data quality —** {prof['missing']:,} missing cells ({miss_pct:.2f}%) and "
               f"{prof['duplicates']:,} duplicate rows.{q}")
    for c in prof["categorical_cols"][:4]:
        vc = df[c].value_counts(normalize=True)
        if len(vc):
            out.append(f"**{c}:** largest group is **{vc.index[0]}** "
                       f"({vc.iloc[0]*100:.1f}%); {df[c].nunique()} distinct values.")
    if prof["likert_cols"]:
        ls = likert_summary(df, prof["likert_cols"])
        if not ls.empty:
            hi, lo = ls.iloc[0], ls.iloc[-1]
            out.append(f"**Highest-rated:** {lbl(hi['Item'])} (mean {hi['Mean']:.2f}, {hi['Top-2-Box %']:.0f}% agree). "
                       f"**Lowest:** {lbl(lo['Item'])} (mean {lo['Mean']:.2f}, {lo['Bottom-2-Box %']:.0f}% disagree).")
            comp = df[prof["likert_cols"]].mean(axis=1)
            out.append(f"**Composite sentiment** (mean of all scale items) is "
                       f"**{comp.mean():.2f}** on a {int(df[prof['likert_cols']].max().max())}-point scale.")
    nums = prof["likert_cols"] + prof["numeric_cols"]
    if len(nums) >= 2:
        corr = df[nums].corr().abs()
        cv = corr.values.copy(); np.fill_diagonal(cv, np.nan)
        if np.isfinite(cv).any():
            i, j = np.unravel_index(np.nanargmax(cv), cv.shape)
            r = df[nums].corr().iloc[i, j]
            out.append(f"**Strongest relationship:** {nums[i]} ↔ {nums[j]} (r = {r:+.2f}).")
    if prof["likert_cols"] and prof["categorical_cols"]:
        comp = df.copy(); comp["_composite_"] = df[prof["likert_cols"]].mean(axis=1)
        best = None
        for f in prof["categorical_cols"]:
            res = group_difference(comp, f, "_composite_", ordinal=False)
            if res and (best is None or res["p"] < best["p"]):
                best = {"factor": f, **res}
        if best:
            verdict = "a statistically significant" if best["p"] < 0.05 else "no statistically significant"
            out.append(f"**Driver scan:** among demographics, **{best['factor']}** has {verdict} association with "
                       f"overall sentiment (p = {best['p']:.3g}, {effect_word(best['effect'])} effect).")
    return out


# ============================================================================
# DATA LOADING
# ============================================================================
def detect_label_map(xls: pd.ExcelFile, data_cols) -> dict:
    """Look for a 2-3 column sheet whose first column matches the data's column names."""
    data_cols = set(map(str, data_cols))
    for sheet in xls.sheet_names:
        try:
            d = pd.read_excel(xls, sheet)
        except Exception:
            continue
        if d.shape[1] < 2 or d.shape[1] > 4:
            continue
        key_col = d.columns[0]
        keys = set(d[key_col].astype(str))
        if len(keys & data_cols) >= max(3, 0.5 * len(data_cols)):
            desc_col = d.columns[-1]
            return {str(k): str(v) for k, v in zip(d[key_col], d[desc_col]) if pd.notna(v)}
    return {}


def set_active_data(df, source_name, labels=None):
    st.session_state.df = df
    st.session_state.profile = profile_columns(df)
    st.session_state.labels = labels or {}
    st.session_state.source_name = source_name


def make_sample_survey(n=1500, seed=42):
    """Generate a realistic Likert survey so the demo works without any upload."""
    rng = np.random.default_rng(seed)
    age = rng.choice(["25-34", "35-44", "45-54", "55+"], n, p=[.35, .38, .19, .08])
    gender = rng.choice(["Male", "Female"], n, p=[.82, .18])
    income = rng.choice(["<20k", "20k-40k", "40k-70k", "70k-120k", "120k+"], n, p=[.18, .25, .29, .19, .09])
    emirate = rng.choice(["Dubai", "Abu Dhabi", "Sharjah", "Other"], n, p=[.74, .17, .06, .03])
    data = {"Respondent_ID": [f"R{i:04d}" for i in range(1, n + 1)],
            "Age_Group": age, "Gender": gender, "Monthly_Income_AED": income, "Emirate": emirate}
    sections = {1: "Climate & Comfort", 2: "Product Utility", 3: "Fabric Performance",
                4: "Purchase Intent", 5: "Brand & Value"}
    key_rows = []
    for q in range(1, 26):
        base = rng.uniform(3.0, 4.0)
        vals = np.clip(np.round(rng.normal(base, 1.0, n)), 1, 5).astype(int)
        data[f"Q{q}"] = vals
        sec = (q - 1) // 5 + 1
        key_rows.append({"Code": f"Q{q}", "Section": f"Sec {sec}: {sections[sec]}",
                         "Question (abbreviated)": f"Survey statement {q} ({sections[sec]})"})
    df = pd.DataFrame(data)
    labels = {r["Code"]: r["Question (abbreviated)"] for r in key_rows}
    return df, labels


# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown("## 🪐 Saturn Insight Studio")
st.sidebar.caption("Descriptive · Diagnostic · Auto-insights")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Descriptive", "🔬 Diagnostic", "🤖 Modeling", "💡 Summary & Insights"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown("### Load data")
up = st.sidebar.file_uploader("CSV, Excel, or JSON", type=["csv", "xlsx", "xls", "json"])

if up is not None:
    try:
        if up.name.lower().endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(up)
            sheet = (st.sidebar.selectbox("Worksheet", xls.sheet_names)
                     if len(xls.sheet_names) > 1 else xls.sheet_names[0])
            df = pd.read_excel(xls, sheet)
            labels = detect_label_map(xls, df.columns)
            if st.sidebar.button("Use this sheet", width='stretch'):
                set_active_data(df, f"{up.name} · {sheet}", labels)
                st.rerun()
        elif up.name.lower().endswith(".json"):
            df = pd.read_json(up)
            if st.sidebar.button("Load file", width='stretch'):
                set_active_data(df, up.name); st.rerun()
        else:
            df = pd.read_csv(up)
            if st.sidebar.button("Load file", width='stretch'):
                set_active_data(df, up.name); st.rerun()
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")

if st.sidebar.button("✨ Load sample survey", width='stretch'):
    sdf, slabels = make_sample_survey()
    set_active_data(sdf, "Sample survey (synthetic)", slabels)
    st.rerun()

if st.session_state.df is not None:
    st.sidebar.success(f"Loaded: {st.session_state.source_name}")

df = st.session_state.df
prof = st.session_state.profile


def needs_data():
    st.info("👈 Upload a CSV/Excel/JSON file, or click **Load sample survey** to explore instantly.")


# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "🏠 Overview":
    st.title("Saturn Insight Studio")
    st.caption("Upload any dataset. The studio detects its structure and adapts descriptive, "
               "diagnostic, and narrative analysis automatically.")

    if df is None:
        needs_data()
    else:
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "Records", f"{prof['n_rows']:,}")
        kpi(c2, "Variables", f"{prof['n_cols']}",
            f"{len(prof['likert_cols'])} scale · {len(prof['categorical_cols'])} categorical")
        kpi(c3, "Missing cells", f"{prof['missing']:,}",
            f"{prof['missing']/max(prof['n_rows']*prof['n_cols'],1)*100:.2f}% of data")
        kpi(c4, "Duplicate rows", f"{prof['duplicates']:,}")

        st.markdown("### Detected structure")
        a, b = st.columns([1, 1])
        with a:
            st.markdown("**Column types**")
            st.json({
                "Identifier": prof["id_cols"],
                "Categorical / group factors": prof["categorical_cols"],
                "Scale / Likert items": prof["likert_cols"],
                "Continuous numeric": prof["numeric_cols"],
                "Datetime": prof["datetime_cols"],
            }, expanded=False)
        with b:
            st.markdown("**Quick read**")
            for line in generate_insights(df, prof)[:4]:
                st.markdown(f'<div class="insight">{line}</div>', unsafe_allow_html=True)

        if st.session_state.labels:
            with st.expander("📖 Variable reference (detected labels)"):
                ref = pd.DataFrame(
                    [{"Code": k, "Description": v} for k, v in st.session_state.labels.items()]
                )
                st.dataframe(ref, width='stretch', hide_index=True)

        st.markdown("### Data preview")
        st.dataframe(df.head(20), width='stretch')


# ============================================================================
# PAGE: DESCRIPTIVE
# ============================================================================
elif page == "📊 Descriptive":
    st.title("📊 Descriptive Analysis")
    if df is None:
        needs_data()
    else:
        tabs = st.tabs(["Scale items", "Distributions", "Categorical mix", "Correlation"])

        # --- Scale items ---
        with tabs[0]:
            if prof["likert_cols"]:
                ls = likert_summary(df, prof["likert_cols"])
                st.markdown("#### Item ratings — ranked by mean")
                disp = ls.copy()
                disp.insert(1, "Question", [st.session_state.labels.get(i, "") for i in ls["Item"]])
                st.dataframe(
                    disp.style.format({"Mean": "{:.2f}", "Std": "{:.2f}", "Top-2-Box %": "{:.1f}",
                                       "Neutral %": "{:.1f}", "Bottom-2-Box %": "{:.1f}"})
                        .bar(subset=["Top-2-Box %"], color="#9fd8c8")
                        .bar(subset=["Bottom-2-Box %"], color="#f4b6a8"),
                    width='stretch', hide_index=True,
                )

                fig = px.bar(ls.sort_values("Mean"), x="Mean", y="Item", orientation="h",
                             color="Mean", color_continuous_scale="Teal",
                             title="Mean rating by item", template=PX_TEMPLATE)
                fig.update_layout(height=max(380, 22 * len(ls)), coloraxis_showscale=False)
                st.plotly_chart(fig, width='stretch')

                st.markdown("#### Response distribution (Top-2-Box vs Bottom-2-Box)")
                stacked = ls.sort_values("Mean")
                f2 = go.Figure()
                f2.add_bar(y=stacked["Item"], x=stacked["Bottom-2-Box %"], name="Disagree (1-2)",
                           orientation="h", marker_color="#e07a5f")
                f2.add_bar(y=stacked["Item"], x=stacked["Neutral %"], name="Neutral (3)",
                           orientation="h", marker_color="#cbd5d8")
                f2.add_bar(y=stacked["Item"], x=stacked["Top-2-Box %"], name="Agree (4-5)",
                           orientation="h", marker_color="#1f6f78")
                f2.update_layout(barmode="stack", template=PX_TEMPLATE,
                                 height=max(380, 22 * len(ls)), xaxis_title="% of respondents",
                                 legend_orientation="h", legend_y=1.06)
                st.plotly_chart(f2, width='stretch')
            else:
                st.info("No Likert/scale columns detected. See the **Distributions** tab for numeric summaries.")

        # --- Distributions ---
        with tabs[1]:
            choices = prof["metrics"] + prof["categorical_cols"]
            if not choices:
                st.info("No analyzable columns detected.")
            else:
                col = st.selectbox("Choose a variable", choices, format_func=lbl)
                if col in prof["metrics"]:
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    m1, m2, m3, m4 = st.columns(4)
                    kpi(m1, "Mean", f"{s.mean():.2f}")
                    kpi(m2, "Median", f"{s.median():.2f}")
                    kpi(m3, "Std dev", f"{s.std():.2f}")
                    kpi(m4, "Range", f"{s.min():.0f}–{s.max():.0f}")
                    fig = px.histogram(df, x=col, nbins=min(30, int(s.nunique())),
                                       marginal="box", color_discrete_sequence=[PALETTE[0]],
                                       template=PX_TEMPLATE, title=f"Distribution of {col}")
                    st.plotly_chart(fig, width='stretch')
                else:
                    vc = df[col].value_counts().reset_index()
                    vc.columns = [col, "Count"]
                    fig = px.bar(vc, x=col, y="Count", color="Count",
                                 color_continuous_scale="Teal", template=PX_TEMPLATE,
                                 title=f"Frequency of {col}")
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig, width='stretch')

        # --- Categorical mix ---
        with tabs[2]:
            if prof["categorical_cols"]:
                cols = st.columns(min(2, len(prof["categorical_cols"])))
                for i, c in enumerate(prof["categorical_cols"][:4]):
                    vc = df[c].value_counts().reset_index()
                    vc.columns = [c, "Count"]
                    fig = px.pie(vc, names=c, values="Count", hole=.5,
                                 color_discrete_sequence=PALETTE, title=c)
                    fig.update_layout(template=PX_TEMPLATE, showlegend=True, height=330,
                                      margin=dict(t=40, b=10))
                    cols[i % len(cols)].plotly_chart(fig, width='stretch')
            else:
                st.info("No categorical columns detected.")

        # --- Correlation ---
        with tabs[3]:
            nums = prof["likert_cols"] + prof["numeric_cols"]
            if len(nums) > 1:
                corr = df[nums].corr()
                fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                template=PX_TEMPLATE, title="Correlation matrix")
                fig.update_layout(height=max(450, 26 * len(nums)))
                st.plotly_chart(fig, width='stretch')
                st.caption("Values are Pearson correlations. Blue = positive, red = negative.")
            else:
                st.info("Need at least two numeric/scale columns to compute correlations.")


# ============================================================================
# PAGE: DIAGNOSTIC
# ============================================================================
elif page == "🔬 Diagnostic":
    st.title("🔬 Diagnostic Analysis")
    st.caption("Test *why* values differ: group comparisons, associations, and drivers — "
               "with the appropriate statistical test chosen automatically.")
    if df is None:
        needs_data()
    else:
        tabs = st.tabs(["Group comparison", "Cross-tab (association)", "Drivers"])

        # --- Group comparison ---
        with tabs[0]:
            if prof["categorical_cols"] and prof["metrics"]:
                c1, c2 = st.columns(2)
                factor = c1.selectbox("Compare across (group)", prof["categorical_cols"])
                metric = c2.selectbox("Metric to compare", prof["metrics"], format_func=lbl)
                default_ord = metric in prof["likert_cols"]
                ordinal = st.radio(
                    "Test type", ["Auto (ordinal → Kruskal–Wallis)", "Continuous (ANOVA)"],
                    index=0 if default_ord else 1, horizontal=True,
                ).startswith("Auto")

                res = group_difference(df, factor, metric, ordinal=ordinal)
                if res is None:
                    st.warning("Not enough groups with ≥2 observations to run a test.")
                else:
                    k1, k2, k3 = st.columns(3)
                    kpi(k1, res["test"], f"{res['stat']:.3f}")
                    kpi(k2, "p-value", f"{res['p']:.3g}",
                        "significant (p<.05)" if res["p"] < 0.05 else "not significant")
                    kpi(k3, res["effect_name"], f"{res['effect']:.3f}", effect_word(res["effect"]) + " effect")

                    if res["p"] < 0.05:
                        st.success(f"**{factor}** groups differ significantly on **{lbl(metric)}** "
                                   f"(effect size: {effect_word(res['effect'])}). The group means below are "
                                   f"unlikely to be equal by chance.")
                    else:
                        st.info(f"No significant difference in **{lbl(metric)}** across **{factor}** "
                                f"(p = {res['p']:.3g}). Differences in group means are within chance variation.")

                    fig = px.box(df.dropna(subset=[factor, metric]), x=factor, y=metric,
                                 color=factor, points="outliers",
                                 color_discrete_sequence=PALETTE, template=PX_TEMPLATE,
                                 title=f"{metric} by {factor}")
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, width='stretch')

                    means = res["means"].copy()
                    st.markdown("##### Group means")
                    st.dataframe(means.style.format({"mean": "{:.2f}", "std": "{:.2f}"}),
                                 width='stretch', hide_index=True)
            else:
                st.info("Need at least one categorical column and one numeric/scale column.")

        # --- Cross-tab ---
        with tabs[1]:
            if len(prof["categorical_cols"]) >= 2:
                c1, c2 = st.columns(2)
                a = c1.selectbox("Row variable", prof["categorical_cols"], index=0)
                b = c2.selectbox("Column variable", prof["categorical_cols"],
                                 index=min(1, len(prof["categorical_cols"]) - 1))
                if a == b:
                    st.warning("Choose two different variables.")
                else:
                    res = crosstab_chi2(df, a, b)
                    if res is None:
                        st.warning("Not enough categories to test association.")
                    else:
                        k1, k2, k3 = st.columns(3)
                        kpi(k1, "Chi-square", f"{res['chi2']:.2f}", f"dof = {res['dof']}")
                        kpi(k2, "p-value", f"{res['p']:.3g}",
                            "associated (p<.05)" if res["p"] < 0.05 else "independent")
                        kpi(k3, "Cramér's V", f"{res['cramers_v']:.3f}", effect_word(res["cramers_v"]) + " strength")

                        if res["p"] < 0.05:
                            st.success(f"**{a}** and **{b}** are statistically associated "
                                       f"(Cramér's V = {res['cramers_v']:.3f}, {effect_word(res['cramers_v'])}).")
                        else:
                            st.info(f"**{a}** and **{b}** appear independent (p = {res['p']:.3g}).")

                        ct_pct = pd.crosstab(df[a], df[b], normalize="index") * 100
                        fig = px.imshow(ct_pct, text_auto=".1f", aspect="auto",
                                        color_continuous_scale="Teal", template=PX_TEMPLATE,
                                        title=f"{b} distribution within each {a} (row %)")
                        st.plotly_chart(fig, width='stretch')
                        with st.expander("Raw counts"):
                            st.dataframe(res["table"], width='stretch')
            else:
                st.info("Need at least two categorical columns for a cross-tabulation.")

        # --- Drivers ---
        with tabs[2]:
            nums = prof["likert_cols"] + prof["numeric_cols"]
            if len(nums) >= 2:
                target = st.selectbox("Target variable", nums, format_func=lbl)
                drivers = correlation_drivers(df, target, nums)
                if drivers.empty:
                    st.info("Not enough data to compute drivers.")
                else:
                    top = drivers.head(15).sort_values("Pearson r")
                    fig = px.bar(top, x="Pearson r", y="Variable", orientation="h",
                                 color="Pearson r", color_continuous_scale="RdBu_r",
                                 range_color=[-1, 1], template=PX_TEMPLATE,
                                 title=f"Variables most correlated with {target}")
                    fig.update_layout(height=max(380, 24 * len(top)), coloraxis_showscale=False)
                    st.plotly_chart(fig, width='stretch')
                    st.dataframe(
                        drivers.style.format({"Pearson r": "{:+.3f}", "Spearman ρ": "{:+.3f}"}),
                        width='stretch', hide_index=True,
                    )
                    st.caption("Correlation is association, not proof of causation.")
            else:
                st.info("Need at least two numeric/scale columns for driver analysis.")


# ============================================================================
# PAGE: MODELING
# ============================================================================
elif page == "🤖 Modeling":
    st.title("🤖 Predictive Modeling")
    st.caption("Train and compare four models — KNN, Decision Tree, Random Forest, "
               "Gradient Boosting — on a target you choose. Task type is auto-detected.")
    if df is None:
        needs_data()
    elif not SKLEARN_OK:
        st.error("scikit-learn isn't available in this environment. Add **scikit-learn** to "
                 "requirements.txt and redeploy to enable modeling.")
    else:
        predictable = [c for c in df.columns if c not in prof["id_cols"]]
        if not predictable:
            st.info("No predictable columns detected.")
        else:
            target = st.selectbox("Target to predict", predictable, format_func=lbl)

            if target in prof["categorical_cols"]:
                task = "classification"
                st.caption("Categorical target detected → **classification**.")
            elif target in prof["likert_cols"]:
                choice = st.radio("Task type", ["Classification (predict the rating as a class)",
                                                "Regression (predict the rating as a number)"],
                                  horizontal=True)
                task = "classification" if choice.startswith("Classification") else "regression"
            else:
                task = "regression"
                st.caption("Continuous numeric target detected → **regression**.")

            candidates = [c for c in predictable if c != target]
            features = st.multiselect("Predictor features", candidates, default=candidates)

            if not features:
                st.warning("Select at least one predictor feature.")
            elif st.button("🚀 Train & compare models", width='stretch'):
                with st.spinner("Training KNN, Decision Tree, Random Forest, and Gradient Boosting…"):
                    try:
                        out = run_models(df, target, features, task)
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        st.stop()

                res = out["results"]
                metric = "Accuracy" if task == "classification" else "R2"
                best = res.iloc[0]

                k1, k2, k3 = st.columns(3)
                kpi(k1, "Best model", out["best_name"])
                if task == "classification":
                    kpi(k2, "Accuracy", f"{best['Accuracy']*100:.1f}%",
                        f"baseline {out['baseline']*100:.1f}%")
                    kpi(k3, "Macro F1", f"{best['Macro F1']:.3f}")
                else:
                    kpi(k2, "R²", f"{best['R2']:.3f}", "baseline 0.00")
                    kpi(k3, "RMSE", f"{best['RMSE']:.3f}")

                st.markdown("#### Model comparison")
                fmt = ({"Accuracy": "{:.3f}", "Macro F1": "{:.3f}"} if task == "classification"
                       else {"R2": "{:.3f}", "RMSE": "{:.3f}", "MAE": "{:.3f}"})
                st.dataframe(res.style.format(fmt).bar(subset=[metric], color="#9fd8c8"),
                             width='stretch', hide_index=True)

                fig = px.bar(res, x=metric, y="Model", orientation="h", color=metric,
                             color_continuous_scale="Teal", template=PX_TEMPLATE,
                             title=f"{metric} by model")
                fig.update_layout(coloraxis_showscale=False,
                                  yaxis={"categoryorder": "total ascending"})
                if task == "classification":
                    fig.add_vline(x=out["baseline"], line_dash="dash", line_color="#e07a5f",
                                  annotation_text="baseline")
                st.plotly_chart(fig, width='stretch')

                lift = best[metric] - out["baseline"]
                if task == "classification":
                    if lift <= 0.02:
                        st.info(f"The best model barely beats the majority-class baseline "
                                f"({out['baseline']*100:.1f}%) — these features carry little signal "
                                f"for **{lbl(target)}**.")
                    else:
                        st.success(f"**{out['best_name']}** beats the baseline by "
                                   f"{lift*100:.1f} points — the features hold real predictive "
                                   f"signal for **{lbl(target)}**.")
                else:
                    if best["R2"] < 0.1:
                        st.info(f"Low R² ({best['R2']:.2f}) — the chosen features explain little "
                                f"variance in **{lbl(target)}**.")
                    else:
                        st.success(f"**{out['best_name']}** explains {best['R2']*100:.0f}% of the "
                                   f"variance in **{lbl(target)}** on held-out data.")

                if out["importances"] is not None:
                    st.markdown("#### What drives the prediction (best tree-based model)")
                    imp = out["importances"].head(15).sort_values("Importance")
                    fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                                 color="Importance", color_continuous_scale="Teal",
                                 template=PX_TEMPLATE)
                    fig.update_layout(coloraxis_showscale=False, height=max(360, 24 * len(imp)))
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.caption("Feature importance isn't available because the best model was KNN.")

                if task == "classification":
                    cm = confusion_matrix(out["y_test"], out["y_pred"], labels=out["classes"])
                    cmdf = pd.DataFrame(cm, index=[str(c) for c in out["classes"]],
                                        columns=[str(c) for c in out["classes"]])
                    fig = px.imshow(cmdf, text_auto=True, color_continuous_scale="Teal",
                                    template=PX_TEMPLATE, aspect="auto",
                                    title=f"Confusion matrix — {out['best_name']}",
                                    labels=dict(x="Predicted", y="Actual", color="Count"))
                    st.plotly_chart(fig, width='stretch')
                else:
                    avp = pd.DataFrame({"Actual": out["y_test"].values, "Predicted": out["y_pred"]})
                    fig = px.scatter(avp, x="Actual", y="Predicted", opacity=0.5,
                                     template=PX_TEMPLATE, color_discrete_sequence=[PALETTE[0]],
                                     title=f"Actual vs predicted — {out['best_name']}")
                    lo, hi = float(avp.min().min()), float(avp.max().max())
                    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                                  line=dict(dash="dash", color="#e07a5f"))
                    st.plotly_chart(fig, width='stretch')

                st.caption("Models trained on a 75/25 split; metrics are computed on the held-out 25%. "
                           "Categorical predictors are one-hot encoded; KNN features are standardized.")


# ============================================================================
# PAGE: SUMMARY & INSIGHTS
# ============================================================================
elif page == "💡 Summary & Insights":
    st.title("💡 Summary & Insights")
    st.caption("Auto-generated narrative read of the dataset — regenerates for whatever you load.")
    if df is None:
        needs_data()
    else:
        insights = generate_insights(df, prof)
        for line in insights:
            st.markdown(f'<div class="insight">{line}</div>', unsafe_allow_html=True)

        # Build a downloadable markdown report
        report = io.StringIO()
        report.write(f"# Saturn Insight Report\n\nSource: {st.session_state.source_name}\n\n")
        report.write("## Key findings\n\n")
        for line in insights:
            report.write(f"- {line.replace('**','')}\n")
        if prof["likert_cols"]:
            report.write("\n## Item summary (scale items)\n\n")
            report.write(likert_summary(df, prof["likert_cols"]).round(2).to_markdown(index=False))
        st.download_button(
            "⬇️ Download report (Markdown)", report.getvalue(),
            file_name="saturn_insight_report.md", mime="text/markdown",
            width='stretch',
        )

        if prof["likert_cols"]:
            st.markdown("### Full item summary")
            st.dataframe(
                likert_summary(df, prof["likert_cols"]).style.format(
                    {"Mean": "{:.2f}", "Std": "{:.2f}", "Top-2-Box %": "{:.1f}",
                     "Neutral %": "{:.1f}", "Bottom-2-Box %": "{:.1f}"}),
                width='stretch', hide_index=True,
            )
