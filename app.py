"""
Claim Settlement Bias Analysis Dashboard
=========================================
A Streamlit application for investigating possible bias in death-claim
settlement decisions, and for benchmarking supervised classification
models (KNN, Decision Tree, Random Forest, Gradient Boosting) that try
to predict the claim outcome.

Run locally:
    streamlit run app.py

Deploy:
    Push this repo to GitHub, then deploy on https://share.streamlit.io
    pointing at app.py. The bundled sample dataset lives in data/Insurance.csv,
    or upload your own CSV with the same column layout from the sidebar.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_processing import (
    load_raw_data,
    clean_data,
    crosstab_counts,
    crosstab_percent,
    chi_square_test,
    approval_rate_by_group,
    numeric_distribution_test,
    build_model_dataframe,
    TARGET_COL,
    APPROVED_LABEL,
    REJECTED_LABEL,
    AGE_LABELS,
    INCOME_LABELS,
)
from utils.modeling import (
    make_preprocessor,
    split_data,
    get_model_zoo,
    train_and_evaluate,
    metrics_summary_table,
    feature_importance,
)

st.set_page_config(
    page_title="Claim Settlement Bias Dashboard",
    page_icon="📊",
    layout="wide",
)

PRIMARY_COLOR = "#1f6feb"
APPROVED_COLOR = "#2e7d32"
REJECTED_COLOR = "#c62828"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_data(uploaded_bytes):
    if uploaded_bytes is not None:
        import io
        raw = load_raw_data(io.BytesIO(uploaded_bytes))
    else:
        raw = load_raw_data("data/Insurance.csv")
    return clean_data(raw)


with st.sidebar:
    st.title("📊 Claim Bias Dashboard")
    st.caption("Claim settlement analytics & fairness audit")

    uploaded_file = st.file_uploader("Upload claims CSV (optional)", type=["csv"])
    uploaded_bytes = uploaded_file.getvalue() if uploaded_file is not None else None

    page = st.radio(
        "Navigate",
        [
            "1. Overview",
            "2. Descriptive Analysis",
            "3. Diagnostic Bias Analysis",
            "4. ML Classification",
            "5. Findings",
        ],
    )

    st.markdown("---")
    st.caption(
        "Built for claim settlement officers to audit potential bias in "
        "claim approvals and benchmark predictive models."
    )

df = get_data(uploaded_bytes)

CATEGORICAL_OPTIONS = {
    "Gender": "PI_GENDER",
    "Team / Zone": "TEAM",
    "Payment Mode": "PAYMENT_MODE",
    "Early / Non-Early Claim": "EARLY_NON",
    "Medical / Non-Medical": "MEDICAL_NONMED",
    "Age Group": "AGE_GROUP",
    "Income Group": "INCOME_GROUP",
    "State": "PI_STATE",
    "Occupation": "PI_OCCUPATION",
}


# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================
if page.startswith("1"):
    st.header("Overview")
    st.write(
        "This dashboard reviews the claim settlement dataset for potential "
        "bias patterns and builds predictive models of the claim outcome."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Claims", f"{len(df):,}")
    c2.metric("Approved", f"{(df[TARGET_COL] == APPROVED_LABEL).sum():,}")
    c3.metric("Repudiated", f"{(df[TARGET_COL] == REJECTED_LABEL).sum():,}")
    c4.metric(
        "Overall Approval Rate",
        f"{(df[TARGET_COL] == APPROVED_LABEL).mean() * 100:.1f}%",
    )

    col1, col2 = st.columns([1, 1.4])
    with col1:
        status_counts = df[TARGET_COL].value_counts().reset_index()
        status_counts.columns = ["Policy Status", "Count"]
        fig = px.pie(
            status_counts,
            names="Policy Status",
            values="Count",
            color="Policy Status",
            color_discrete_map={APPROVED_LABEL: APPROVED_COLOR, REJECTED_LABEL: REJECTED_COLOR},
            title="Claim Outcome Distribution",
            hole=0.45,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sample Data")
        st.dataframe(df.head(15), use_container_width=True)

    st.subheader("Missing Values")
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if len(miss):
        st.dataframe(miss.rename("Missing Count").to_frame(), use_container_width=True)
    else:
        st.success("No missing values after cleaning.")

    with st.expander("Data quality notes"):
        st.markdown(
            """
- `SUM_ASSURED` and `PI_ANNUAL_INCOME` were stored as text with thousands
  separators (e.g. `"1,981,818"`) and were converted to numeric.
- `ZONE` contained case/whitespace duplicates (e.g. `South`, `SOUTH`,
  `South 2`) which were normalized into a single `TEAM` column for
  team-wise comparisons.
- Missing `PI_OCCUPATION` and `REASON_FOR_CLAIM` were labeled `"Unknown"` /
  `"Not Specified"` rather than dropped, to avoid silently removing claims
  that might be disproportionately of one outcome.
"""
        )


# ===========================================================================
# PAGE 2 — DESCRIPTIVE ANALYSIS (Cross-tabulation)
# ===========================================================================
elif page.startswith("2"):
    st.header("Descriptive Analysis — Cross Tabulation vs. Policy Status")
    st.write(
        "Cross-tabulate any claim attribute against the final policy status "
        "to see how claim volume and outcomes are distributed."
    )

    choice_label = st.selectbox("Cross-tabulate by:", list(CATEGORICAL_OPTIONS.keys()))
    col = CATEGORICAL_OPTIONS[choice_label]

    counts = crosstab_counts(df, col)
    pct = crosstab_percent(df, col)

    tab1, tab2, tab3 = st.tabs(["Counts", "Row %", "Chart"])
    with tab1:
        st.dataframe(counts, use_container_width=True)
    with tab2:
        st.dataframe(pct, use_container_width=True)
    with tab3:
        plot_df = counts.reset_index().melt(id_vars=col, var_name="Policy Status", value_name="Count")
        fig = px.bar(
            plot_df,
            x=col,
            y="Count",
            color="Policy Status",
            barmode="stack",
            color_discrete_map={APPROVED_LABEL: APPROVED_COLOR, REJECTED_LABEL: REJECTED_COLOR},
            title=f"Claim Volume by {choice_label} and Outcome",
        )
        fig.update_layout(xaxis_title=choice_label, xaxis={"categoryorder": "total descending"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Summary statistics")
    num_cols = st.columns(3)
    num_cols[0].write("**Age**")
    num_cols[0].dataframe(df.groupby(TARGET_COL)["PI_AGE"].describe().T, use_container_width=True)
    num_cols[1].write("**Annual Income**")
    num_cols[1].dataframe(df.groupby(TARGET_COL)["PI_ANNUAL_INCOME"].describe().T, use_container_width=True)
    num_cols[2].write("**Sum Assured**")
    num_cols[2].dataframe(df.groupby(TARGET_COL)["SUM_ASSURED"].describe().T, use_container_width=True)


# ===========================================================================
# PAGE 3 — DIAGNOSTIC BIAS ANALYSIS
# ===========================================================================
elif page.startswith("3"):
    st.header("Diagnostic Analysis — Where Does Bias Show Up?")
    st.write(
        "For each attribute, we test whether approval rate differs more than "
        "chance would predict (chi-square test of independence) and rank "
        "groups by approval rate to surface gaps worth investigating."
    )

    bias_dims = {
        "Gender": "PI_GENDER",
        "Team": "TEAM",
        "Age Group": "AGE_GROUP",
        "Income Group": "INCOME_GROUP",
        "Occupation": "PI_OCCUPATION",
        "Early vs Non-Early Claim": "EARLY_NON",
        "Medical vs Non-Medical": "MEDICAL_NONMED",
        "State": "PI_STATE",
    }

    st.subheader("Statistical significance summary")
    rows = []
    for label, col in bias_dims.items():
        try:
            chi2, p, dof, cramers_v = chi_square_test(df, col)
            rows.append(
                {
                    "Attribute": label,
                    "Chi-square": round(chi2, 2),
                    "p-value": round(p, 5),
                    "Significant (p<0.05)": "Yes" if p < 0.05 else "No",
                    "Cramér's V (effect size)": round(cramers_v, 3),
                }
            )
        except Exception:
            continue
    sig_df = pd.DataFrame(rows).sort_values("Cramér's V (effect size)", ascending=False)
    st.dataframe(sig_df, use_container_width=True, hide_index=True)
    st.caption(
        "A significant p-value (< 0.05) means approval rate genuinely differs across "
        "groups of that attribute rather than by chance. Cramér's V indicates how "
        "strong that association is (>0.1 small, >0.3 moderate, >0.5 large)."
    )

    st.markdown("---")
    st.subheader("Approval rate by group — drill down")
    drill_label = st.selectbox("Choose attribute to drill into:", list(bias_dims.keys()), key="drill")
    drill_col = bias_dims[drill_label]

    rate_df = approval_rate_by_group(df, drill_col).reset_index()
    min_claims = st.slider("Minimum claims to include a group (filters noisy small groups)", 1, 50, 5)
    rate_df = rate_df[rate_df["total_claims"] >= min_claims].sort_values("approval_rate", ascending=False)

    fig = px.bar(
        rate_df,
        x=drill_col,
        y="approval_rate",
        color="approval_rate",
        color_continuous_scale="RdYlGn",
        text="total_claims",
        title=f"Approval Rate (%) by {drill_label} (label = claim count)",
    )
    fig.add_hline(
        y=(df[TARGET_COL] == APPROVED_LABEL).mean() * 100,
        line_dash="dash",
        annotation_text="Overall average",
        line_color="black",
    )
    fig.update_layout(yaxis_title="Approval Rate (%)", xaxis_title=drill_label)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(rate_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Numeric attributes: Age & Income distribution by outcome")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(
            df, x=TARGET_COL, y="PI_AGE", color=TARGET_COL,
            color_discrete_map={APPROVED_LABEL: APPROVED_COLOR, REJECTED_LABEL: REJECTED_COLOR},
            title="Age distribution: Approved vs Repudiated",
        )
        st.plotly_chart(fig, use_container_width=True)
        stat, p, med_a, med_r = numeric_distribution_test(df, "PI_AGE")
        st.caption(
            f"Mann-Whitney U test p-value = {p:.5f} "
            f"({'significant' if p < 0.05 else 'not significant'}). "
            f"Median age — Approved: {med_a:.0f}, Repudiated: {med_r:.0f}."
        )
    with c2:
        fig = px.box(
            df, x=TARGET_COL, y="PI_ANNUAL_INCOME", color=TARGET_COL,
            color_discrete_map={APPROVED_LABEL: APPROVED_COLOR, REJECTED_LABEL: REJECTED_COLOR},
            title="Annual income distribution: Approved vs Repudiated",
        )
        fig.update_yaxes(range=[0, df["PI_ANNUAL_INCOME"].quantile(0.95)])
        st.plotly_chart(fig, use_container_width=True)
        stat, p, med_a, med_r = numeric_distribution_test(df, "PI_ANNUAL_INCOME")
        st.caption(
            f"Mann-Whitney U test p-value = {p:.5f} "
            f"({'significant' if p < 0.05 else 'not significant'}). "
            f"Median income — Approved: {med_a:,.0f}, Repudiated: {med_r:,.0f}."
        )

    st.info(
        "💡 **Reading caution:** a statistical association between an attribute "
        "(e.g. age, income, team) and approval rate is evidence worth investigating "
        "further — it does not by itself prove unfair intent. Differences could "
        "also stem from legitimate underwriting factors (claim reason mix, policy "
        "tenure, documentation quality) not captured here. Use this as a starting "
        "point for case-level audits of the lowest-approval groups."
    )


# ===========================================================================
# PAGE 4 — ML CLASSIFICATION
# ===========================================================================
elif page.startswith("4"):
    st.header("Predictive Modeling — Supervised Classification")
    st.write(
        "We train four classifiers to predict claim outcome from claim "
        "attributes. If models can predict the outcome accurately from "
        "factors like team, age, income or occupation alone, that is further "
        "evidence those factors are driving decisions — by policy or by bias."
    )

    with st.expander("Feature engineering applied", expanded=False):
        st.markdown(
            """
- **Numeric features** (`PI_AGE`, `PI_ANNUAL_INCOME`, `SUM_ASSURED`): cleaned
  to numeric, then standardized (zero mean / unit variance) inside the
  pipeline so distance-based models like KNN aren't dominated by income's
  large scale.
- **Categorical features** (`PI_GENDER`, `TEAM`, `PAYMENT_MODE`, `EARLY_NON`,
  `MEDICAL_NONMED`, `PI_STATE`, `PI_OCCUPATION`, `REASON_FOR_CLAIM`):
  high-cardinality columns (state, occupation, reason, team) are reduced to
  their top-N most frequent categories + an `"Other"` bucket, then one-hot
  encoded.
- **Target**: `POLICY_STATUS` binarized as `1 = Approved`, `0 = Repudiated`.
- **Train/test split**: stratified, so both sets keep the same approve/reject
  ratio as the full dataset.
- Preprocessing is fit only on the training fold (inside a single
  `Pipeline`) to avoid leaking test information.
"""
        )

    st.sidebar.markdown("### Model settings")
    test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.25, 0.05)
    top_n_rare = st.sidebar.slider("Top-N categories before 'Other' bucket", 5, 20, 10)
    knn_k = st.sidebar.slider("KNN: n_neighbors", 3, 25, 7, 2)
    rf_trees = st.sidebar.slider("Random Forest: n_estimators", 50, 500, 200, 50)
    gb_trees = st.sidebar.slider("Gradient Boosting: n_estimators", 50, 500, 150, 50)

    model_df = build_model_dataframe(df, top_n_rare=top_n_rare)
    X_train, X_test, y_train, y_test = split_data(model_df, test_size=test_size)

    st.write(
        f"Training rows: **{len(X_train)}**  |  Testing rows: **{len(X_test)}**  |  "
        f"Features after one-hot encoding will expand from {X_train.shape[1]} raw columns."
    )

    run = st.button("🚀 Train all 4 models", type="primary")

    if run or "results" in st.session_state:
        if run:
            with st.spinner("Training KNN, Decision Tree, Random Forest, Gradient Boosting..."):
                preprocessor = make_preprocessor()
                models = get_model_zoo(knn_k=knn_k, rf_trees=rf_trees, gb_trees=gb_trees)
                results = train_and_evaluate(models, preprocessor, X_train, X_test, y_train, y_test)
                st.session_state["results"] = results
                st.session_state["y_test"] = y_test

        results = st.session_state["results"]
        y_test = st.session_state["y_test"]

        st.subheader("Performance summary")
        summary = metrics_summary_table(results)
        st.dataframe(
            summary.style.background_gradient(cmap="Greens", subset=["Test Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]),
            use_container_width=True,
        )

        st.subheader("Train vs Test Accuracy (overfitting check)")
        acc_plot = summary.reset_index().melt(
            id_vars="Model", value_vars=["Train Accuracy", "Test Accuracy"],
            var_name="Split", value_name="Accuracy",
        )
        fig = px.bar(acc_plot, x="Model", y="Accuracy", color="Split", barmode="group",
                     title="Train vs Test Accuracy by Model")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Precision / Recall / F1 comparison")
        prf_plot = summary.reset_index().melt(
            id_vars="Model", value_vars=["Precision", "Recall", "F1 Score"],
            var_name="Metric", value_name="Score",
        )
        fig = px.bar(prf_plot, x="Model", y="Score", color="Metric", barmode="group",
                     title="Precision, Recall & F1 Score by Model")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("ROC Curves (model stability check)")
        fig = go.Figure()
        for name, r in results.items():
            fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines",
                                      name=f"{name} (AUC={r['auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(dash="dash", color="gray"), name="Random guess"))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                           title="ROC Curves — All Models")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Curves that hug the top-left corner and stay well above the diagonal "
            "indicate a model that separates approved vs repudiated claims reliably "
            "across thresholds, not just at the default cutoff — i.e. a stable model."
        )

        st.subheader("Confusion Matrices")
        cm_cols = st.columns(4)
        for i, (name, r) in enumerate(results.items()):
            cm = r["confusion_matrix"]
            fig = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                x=["Pred: Repudiated", "Pred: Approved"],
                y=["Actual: Repudiated", "Actual: Approved"],
                title=name,
            )
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=0, l=0, r=0))
            cm_cols[i % 4].plotly_chart(fig, use_container_width=True)

        st.subheader("What's driving predictions? (tree-based feature importance)")
        fi_cols = st.columns(2)
        tree_models = ["Decision Tree", "Random Forest", "Gradient Boosting"]
        for i, name in enumerate(tree_models):
            fi = feature_importance(results[name]["pipeline"])
            if fi is not None:
                fig = px.bar(
                    fi.sort_values().tail(12), orientation="h",
                    title=f"Top features — {name}",
                    labels={"value": "Importance", "index": "Feature"},
                )
                fi_cols[i % 2].plotly_chart(fig, use_container_width=True)
    else:
        st.info("Adjust settings in the sidebar if desired, then click **Train all 4 models**.")


# ===========================================================================
# PAGE 5 — FINDINGS
# ===========================================================================
elif page.startswith("5"):
    st.header("Findings & Recommendations")

    overall_rate = (df[TARGET_COL] == APPROVED_LABEL).mean() * 100
    st.write(
        f"Overall approval rate across **{len(df):,}** claims is "
        f"**{overall_rate:.1f}%**. Below is an auto-generated summary of "
        f"where approval rates deviate most from this baseline, based on "
        f"the statistical tests in the Diagnostic Analysis page."
    )

    bias_dims = {
        "Gender": "PI_GENDER",
        "Team": "TEAM",
        "Age Group": "AGE_GROUP",
        "Income Group": "INCOME_GROUP",
        "Occupation": "PI_OCCUPATION",
        "Early vs Non-Early Claim": "EARLY_NON",
        "Medical vs Non-Medical": "MEDICAL_NONMED",
        "State": "PI_STATE",
    }

    findings = []
    for label, col in bias_dims.items():
        try:
            chi2, p, dof, cramers_v = chi_square_test(df, col)
        except Exception:
            continue
        if p < 0.05:
            rate_df = approval_rate_by_group(df, col)
            rate_df = rate_df[rate_df["total_claims"] >= 5]
            if rate_df.empty:
                continue
            best = rate_df.iloc[0]
            worst = rate_df.iloc[-1]
            findings.append(
                {
                    "attribute": label,
                    "p_value": p,
                    "cramers_v": cramers_v,
                    "best_group": best.name,
                    "best_rate": best["approval_rate"],
                    "worst_group": worst.name,
                    "worst_rate": worst["approval_rate"],
                    "gap": best["approval_rate"] - worst["approval_rate"],
                }
            )

    findings = sorted(findings, key=lambda x: x["cramers_v"], reverse=True)

    if findings:
        st.subheader("Statistically significant disparities (p < 0.05), ranked by effect size")
        for f in findings:
            st.markdown(
                f"- **{f['attribute']}**: approval rate ranges from "
                f"**{f['worst_rate']:.1f}%** (`{f['worst_group']}`) to "
                f"**{f['best_rate']:.1f}%** (`{f['best_group']}`) — a "
                f"**{f['gap']:.1f} point** gap (Cramér's V = {f['cramers_v']:.3f}, "
                f"p = {f['p_value']:.4f})."
            )
    else:
        st.success("No statistically significant disparities detected at p < 0.05.")

    st.markdown("---")
    st.subheader("Model-based evidence")
    if "results" in st.session_state:
        summary = metrics_summary_table(st.session_state["results"])
        best_model = summary["ROC-AUC"].idxmax()
        best_auc = summary.loc[best_model, "ROC-AUC"]
        st.write(
            f"The best performing model was **{best_model}** with ROC-AUC = "
            f"**{best_auc:.3f}** and test accuracy = "
            f"**{summary.loc[best_model, 'Test Accuracy']:.3f}**."
        )
        st.write(
            "If a model can predict claim outcome well above a 50% baseline "
            "using attributes such as team, occupation, income or age — and "
            "those same attributes showed significant disparities above — "
            "that is converging evidence the settlement process may be "
            "influenced by factors beyond claim merit alone."
        )
        st.dataframe(summary, use_container_width=True)
    else:
        st.info("Run the models on the **ML Classification** page to see model-based evidence here.")

    st.markdown("---")
    st.subheader("Recommended next steps")
    st.markdown(
        """
1. **Case-level audit** of the lowest-approval teams/occupations/age bands
   flagged above — pull a sample of repudiated claims and verify the stated
   rejection reason against policy documentation.
2. **Standardize team naming** in source systems (`ZONE` had case and
   whitespace duplicates) so reporting doesn't fragment the same team into
   multiple buckets.
3. **Track approval rate by claim handler/officer**, not just by team or
   geography, if that data can be added — individual-level bias is often
   masked at the team level.
4. **Re-run this analysis periodically** as new claims are settled, and
   monitor whether disparities narrow after any process changes.
5. Treat the statistical associations here as **leads for investigation**,
   not as proof of intentional bias — confirm with underwriting/claims
   documentation before drawing conclusions about individuals or teams.
"""
    )
