from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Supplier Quality Risk Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = Path(__file__).with_name("Project_supplier_quality_cleaned.xlsx")

RISK_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
RISK_COLORS = {
    "CRITICAL": "#991B1B",
    "HIGH": "#DC2626",
    "MEDIUM": "#D97706",
    "LOW": "#16A34A",
}

# ---------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_prepare(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.name}")

    df = pd.read_excel(path, engine="openpyxl")

    required = [
        "QN #",
        "Disp",
        "Rewrk hrs",
        "Vendor name",
        "Cause",
        "Vendor Response time ",
        "Total Cost",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    quality = {
        "source_rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_vendor": int(df["Vendor name"].isna().sum()),
    }

    data = df.copy()

    data["Vendor name"] = data["Vendor name"].astype("string").str.strip()
    data["Disp_clean"] = data["Disp"].astype("string").str.strip().str.lower()
    data["Cause"] = data["Cause"].astype("string").str.strip()

    data["Rewrk hrs"] = pd.to_numeric(
        data["Rewrk hrs"], errors="coerce"
    ).fillna(0)

    data["Vendor Response time "] = pd.to_numeric(
        data["Vendor Response time "], errors="coerce"
    ).clip(lower=0)

    data["Total Cost"] = pd.to_numeric(
        data["Total Cost"], errors="coerce"
    ).fillna(0)

    data["is_scrap_rework"] = (
        data["Disp_clean"]
        .str.contains("scrap|rework", na=False)
        .astype(int)
    )

    data = data[data["Vendor name"].notna()].copy()

    vendor = (
        data.groupby("Vendor name", observed=True)
        .agg(
            qn_count=("QN #", "count"),
            total_cost_sum=("Total Cost", "sum"),
            avg_cost_per_qn=("Total Cost", "mean"),
            max_cost_qn=("Total Cost", "max"),
            rework_hrs_sum=("Rewrk hrs", "sum"),
            avg_response_time=("Vendor Response time ", "mean"),
            max_response_time=("Vendor Response time ", "max"),
            scrap_rework_count=("is_scrap_rework", "sum"),
        )
        .reset_index()
    )

    vendor["scrap_rework_rate"] = (
        vendor["scrap_rework_count"] / vendor["qn_count"].replace(0, np.nan)
    ).fillna(0)

    # Composite risk score follows the logic used in the notebook:
    # 30% total cost, 25% rework hours, 25% response time,
    # 20% scrap/rework rate.
    vendor["rank_cost"] = vendor["total_cost_sum"].rank(pct=True) * 100
    vendor["rank_rework"] = vendor["rework_hrs_sum"].rank(pct=True) * 100
    vendor["rank_response_time"] = (
        vendor["avg_response_time"].rank(pct=True) * 100
    )
    vendor["rank_scrap_rework"] = (
        vendor["scrap_rework_rate"].rank(pct=True) * 100
    )

    vendor["risk_score"] = (
        vendor["rank_cost"] * 0.30
        + vendor["rank_rework"] * 0.25
        + vendor["rank_response_time"] * 0.25
        + vendor["rank_scrap_rework"] * 0.20
    )

    threshold = float(vendor["risk_score"].quantile(0.67))
    vendor["high_risk"] = (vendor["risk_score"] >= threshold).astype(int)

    return data, vendor, quality, threshold


FEATURES = [
    "qn_count",
    "avg_cost_per_qn",
    "max_cost_qn",
    "total_cost_sum",
    "rework_hrs_sum",
    "avg_response_time",
    "max_response_time",
    "scrap_rework_rate",
]


@st.cache_resource(show_spinner=False)
def train_model(vendor: pd.DataFrame):
    X = vendor[FEATURES].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = vendor["high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc": roc_auc_score(y_test, prob),
        "confusion": confusion_matrix(y_test, pred),
    }

    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})

    importance = (
        pd.DataFrame(
            {
                "Feature": FEATURES,
                "Importance": model.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return model, metrics, roc_df, importance


def risk_tier(prob: float) -> str:
    if prob >= 0.75:
        return "CRITICAL"
    if prob >= 0.50:
        return "HIGH"
    if prob >= 0.30:
        return "MEDIUM"
    return "LOW"


def money(value: float) -> str:
    return f"${value:,.0f}"


# ---------------------------------------------------------
# LOAD DATA + TRAIN MODEL
# ---------------------------------------------------------
try:
    raw, vendor, data_quality, risk_threshold = load_and_prepare(DATA_FILE)
    model, model_metrics, roc_df, importance_df = train_model(vendor)
except Exception as exc:
    st.error(f"Unable to load or process the project data: {exc}")
    st.stop()

X_all = vendor[FEATURES].replace([np.inf, -np.inf], np.nan)
X_all = X_all.fillna(X_all.median(numeric_only=True))

vendor["predicted_risk_prob"] = model.predict_proba(X_all)[:, 1]
vendor["predicted_risk_label"] = model.predict(X_all)
vendor["risk_tier"] = vendor["predicted_risk_prob"].apply(risk_tier)

# ---------------------------------------------------------
# STYLE
# ---------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
      [data-testid="stMetric"] {
          background:#FFFFFF;
          border:1px solid #E2E8F0;
          border-radius:14px;
          padding:14px 16px;
          box-shadow:0 2px 8px rgba(15,23,42,.05);
      }
      [data-testid="stSidebar"] {background:#F8FAFC;}
      .hero {
          padding:1.1rem 1.25rem;
          border-radius:16px;
          background:linear-gradient(120deg,#0F172A,#1D4ED8);
          color:white;
          margin-bottom:1rem;
      }
      .hero h1 {margin:0; font-size:2rem;}
      .hero p {margin:.35rem 0 0; opacity:.9;}
      .riskbox {
          border-left:5px solid #DC2626;
          background:#FEF2F2;
          padding:.9rem 1rem;
          border-radius:8px;
          margin:.4rem 0;
      }
      .actionbox {
          border-left:5px solid #2563EB;
          background:#EFF6FF;
          padding:.9rem 1rem;
          border-radius:8px;
          margin:.4rem 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Supplier Quality Risk Dashboard</h1>
      <p>Interactive supplier risk prioritization using quality notifications, cost,
      rework, response time, and disposition severity.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
with st.sidebar:
    st.header("Dashboard Filters")

    vendor_options = sorted(vendor["Vendor name"].dropna().unique().tolist())
    selected_vendors = st.multiselect(
        "Supplier",
        vendor_options,
        default=vendor_options,
    )

    tier_options = RISK_ORDER
    selected_tiers = st.multiselect(
        "Risk tier",
        tier_options,
        default=tier_options,
    )

    causes = sorted(raw["Cause"].dropna().unique().tolist())
    selected_causes = st.multiselect(
        "Defect cause",
        causes,
        default=causes,
    )

    st.divider()
    with st.expander("Data quality"):
        st.write(f"Source QN rows: **{data_quality['source_rows']:,}**")
        st.write(f"Unique suppliers: **{vendor['Vendor name'].nunique():,}**")
        st.write(f"Exact duplicate rows: **{data_quality['duplicate_rows']:,}**")
        st.write(f"Missing supplier names: **{data_quality['missing_vendor']:,}**")

filtered_vendor = vendor[
    vendor["Vendor name"].isin(selected_vendors)
    & vendor["risk_tier"].isin(selected_tiers)
].copy()

filtered_raw = raw[
    raw["Vendor name"].isin(selected_vendors)
    & raw["Cause"].isin(selected_causes)
].copy()

if filtered_vendor.empty:
    st.warning("No suppliers match the selected filters.")
    st.stop()

# ---------------------------------------------------------
# KPIs
# ---------------------------------------------------------
total_suppliers = filtered_vendor["Vendor name"].nunique()
high_suppliers = filtered_vendor["risk_tier"].isin(["CRITICAL", "HIGH"]).sum()
total_qns = int(filtered_vendor["qn_count"].sum())
total_cost = float(filtered_vendor["total_cost_sum"].sum())
avg_response = float(filtered_vendor["avg_response_time"].mean())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Suppliers", f"{total_suppliers:,}")
k2.metric("Critical / High Risk", f"{high_suppliers:,}")
k3.metric("Quality Notifications", f"{total_qns:,}")
k4.metric("Total Quality Cost", money(total_cost))
k5.metric("Avg. Response Time", f"{avg_response:.1f} days")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
overview_tab, ranking_tab, model_tab, actions_tab = st.tabs(
    [
        "Executive Overview",
        "Supplier Risk Ranking",
        "Model Performance",
        "Recommended Actions",
    ]
)

# ---------------------------------------------------------
# EXECUTIVE OVERVIEW
# ---------------------------------------------------------
with overview_tab:
    c1, c2 = st.columns(2)

    tier_summary = (
        filtered_vendor["risk_tier"]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
        .rename_axis("Risk Tier")
        .reset_index(name="Suppliers")
    )

    fig_tier = px.bar(
        tier_summary,
        x="Risk Tier",
        y="Suppliers",
        color="Risk Tier",
        color_discrete_map=RISK_COLORS,
        title="Supplier Risk Tier Distribution",
        category_orders={"Risk Tier": RISK_ORDER},
        text="Suppliers",
    )
    fig_tier.update_layout(showlegend=False, height=390)
    c1.plotly_chart(fig_tier, use_container_width=True)

    risk_map = filtered_vendor.copy()
    fig_map = px.scatter(
        risk_map,
        x="qn_count",
        y="total_cost_sum",
        size="predicted_risk_prob",
        color="risk_tier",
        hover_name="Vendor name",
        color_discrete_map=RISK_COLORS,
        category_orders={"risk_tier": RISK_ORDER},
        title="Supplier Risk Map: QN Count vs. Total Cost",
        labels={
            "qn_count": "Quality Notification Count",
            "total_cost_sum": "Total Cost ($)",
            "risk_tier": "Risk Tier",
        },
    )
    fig_map.update_layout(height=390)
    c2.plotly_chart(fig_map, use_container_width=True)

    c3, c4 = st.columns(2)

    cause_summary = (
        filtered_raw["Cause"]
        .value_counts()
        .head(12)
        .sort_values(ascending=True)
        .rename_axis("Cause")
        .reset_index(name="QN Count")
    )
    fig_cause = px.bar(
        cause_summary,
        x="QN Count",
        y="Cause",
        orientation="h",
        title="Top Quality Notification Causes",
        text="QN Count",
    )
    fig_cause.update_layout(height=410)
    c3.plotly_chart(fig_cause, use_container_width=True)

    top_cost = (
        filtered_vendor.nlargest(12, "total_cost_sum")
        .sort_values("total_cost_sum")
    )
    fig_cost = px.bar(
        top_cost,
        x="total_cost_sum",
        y="Vendor name",
        orientation="h",
        color="risk_tier",
        color_discrete_map=RISK_COLORS,
        category_orders={"risk_tier": RISK_ORDER},
        title="Top Suppliers by Total Quality Cost",
        labels={
            "total_cost_sum": "Total Cost ($)",
            "Vendor name": "",
            "risk_tier": "Risk Tier",
        },
    )
    fig_cost.update_layout(height=410)
    c4.plotly_chart(fig_cost, use_container_width=True)

# ---------------------------------------------------------
# SUPPLIER RISK RANKING
# ---------------------------------------------------------
with ranking_tab:
    st.subheader("Supplier Risk Priority List")

    ranking = filtered_vendor[
        [
            "Vendor name",
            "qn_count",
            "total_cost_sum",
            "rework_hrs_sum",
            "avg_response_time",
            "scrap_rework_rate",
            "risk_score",
            "predicted_risk_prob",
            "risk_tier",
        ]
    ].sort_values("predicted_risk_prob", ascending=False)

    ranking_display = ranking.copy()
    ranking_display["total_cost_sum"] = ranking_display["total_cost_sum"].map(
        lambda x: f"${x:,.0f}"
    )
    ranking_display["scrap_rework_rate"] = ranking_display[
        "scrap_rework_rate"
    ].map(lambda x: f"{x:.1%}")
    ranking_display["predicted_risk_prob"] = ranking_display[
        "predicted_risk_prob"
    ].map(lambda x: f"{x:.1%}")
    ranking_display["risk_score"] = ranking_display["risk_score"].map(
        lambda x: f"{x:.1f}"
    )
    ranking_display["avg_response_time"] = ranking_display[
        "avg_response_time"
    ].map(lambda x: f"{x:.1f}")

    ranking_display = ranking_display.rename(
        columns={
            "Vendor name": "Supplier",
            "qn_count": "QN Count",
            "total_cost_sum": "Total Cost",
            "rework_hrs_sum": "Rework Hrs",
            "avg_response_time": "Avg Response Days",
            "scrap_rework_rate": "Scrap/Rework Rate",
            "risk_score": "Composite Risk Score",
            "predicted_risk_prob": "Predicted Risk Probability",
            "risk_tier": "Risk Tier",
        }
    )

    st.dataframe(
        ranking_display,
        use_container_width=True,
        hide_index=True,
    )

    csv = ranking.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download supplier risk ranking",
        data=csv,
        file_name="supplier_risk_ranking.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------
with model_tab:
    st.subheader("Random Forest Model Performance")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{model_metrics['accuracy']:.1%}")
    m2.metric("Precision", f"{model_metrics['precision']:.1%}")
    m3.metric("Recall", f"{model_metrics['recall']:.1%}")
    m4.metric("F1 Score", f"{model_metrics['f1']:.1%}")
    m5.metric("ROC-AUC", f"{model_metrics['auc']:.3f}")

    c1, c2 = st.columns(2)

    fig_roc = px.line(
        roc_df,
        x="FPR",
        y="TPR",
        title="ROC Curve",
        labels={
            "FPR": "False Positive Rate",
            "TPR": "True Positive Rate",
        },
    )
    fig_roc.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(dash="dash"),
    )
    fig_roc.update_layout(height=400)
    c1.plotly_chart(fig_roc, use_container_width=True)

    fig_imp = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance",
    )
    fig_imp.update_layout(height=400)
    c2.plotly_chart(fig_imp, use_container_width=True)

    st.caption(
        "The model classifies suppliers against a high-risk label derived from the "
        "composite risk score. The high-risk cutoff is the 67th percentile of the "
        f"vendor risk score ({risk_threshold:.1f})."
    )

# ---------------------------------------------------------
# RECOMMENDED ACTIONS
# ---------------------------------------------------------
with actions_tab:
    top_supplier = (
        filtered_vendor.sort_values("predicted_risk_prob", ascending=False)
        .iloc[0]
    )

    top_cause = (
        filtered_raw["Cause"].value_counts().index[0]
        if not filtered_raw.empty
        else "Not available"
    )

    st.markdown(
        f"""
        <div class="riskbox">
        <b>Highest current supplier risk:</b>
        {top_supplier['Vendor name']} has a predicted risk probability of
        <b>{top_supplier['predicted_risk_prob']:.1%}</b> and is classified as
        <b>{top_supplier['risk_tier']}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="actionbox">
        <b>Defect focus:</b> The most frequent selected defect cause is
        <b>{top_cause}</b>. Use Pareto review and corrective-action follow-up
        to reduce recurrence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Suggested Supplier Quality Actions
        1. **Audit CRITICAL and HIGH suppliers first** using the risk ranking as the priority list.
        2. **Target recurring defect causes** with focused corrective actions and 8D follow-up.
        3. **Reduce response-time risk** by setting supplier response targets and escalating overdue actions.
        4. **Track cost and rework exposure** as leading indicators of supplier performance deterioration.
        5. **Review the model periodically** as new QN data is added so risk rankings remain current.
        """
    )

st.caption(
    f"Data source: {DATA_FILE.name} | "
    f"{len(raw):,} QN records | "
    f"{vendor['Vendor name'].nunique():,} suppliers"
)
