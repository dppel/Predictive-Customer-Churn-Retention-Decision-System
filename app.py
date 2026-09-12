"""
Enterprise Customer Retention & Revenue Protection System
==========================================================
Designed by Lead Decision Scientist & Customer Analytics Specialist
Targeting: C-Suite / Head of Customer Retention / Chief Commercial Officer
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.models.train_xgboost import load_model
from src.preprocessing.pipeline import ChurnPreprocessingPipeline
from src.explainability.shap_explainer import ChurnShapExplainer
from src.decision.retention_engine import RETENTION_CATALOG, evaluate_retention_actions
from src.decision.sensitivity_analysis import run_monte_carlo_sensitivity
from src.monitoring.drift_detector import monitor_feature_drift
from src.decision.ab_testing_design import calculate_ab_sample_size

#-----------------------------------------------------------------------------
#Page Configuration & High-Contrast Professional Palette
#-----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Retention & Revenue Protection Portal",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

#Comprehensive High-Contrast CSS Theme Overrides
st.markdown("""
<style>
    /* Main App Background & Typography */
    .stApp {
        background-color: #0A0E17 !important;
        color: #E2E8F0 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0D1322 !important;
        border-right: 1px solid #1E293B !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #F8FAFC !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
    }

    /* Multiselect Tag Chips - Override Streamlit Red BaseWeb */
    span[data-baseweb="tag"] {
        background-color: #1E293B !important;
        border: 1px solid #818CF8 !important;
        border-radius: 6px !important;
        padding: 2px 8px !important;
    }
    span[data-baseweb="tag"] span {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }
    span[data-baseweb="tag"] svg {
        fill: #818CF8 !important;
    }
    span[data-baseweb="tag"] svg:hover {
        fill: #FFFFFF !important;
    }

    /* Dropdown Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: #131B2E !important;
        border: 1px solid #25334D !important;
        color: #E2E8F0 !important;
        border-radius: 6px !important;
    }

    /* Radio Buttons & Sliders Accents */
    div[role="radiogroup"] label {
        color: #CBD5E1 !important;
    }
    .stSlider [data-baseweb="slider"] {
        color: #818CF8 !important;
    }
    div[data-testid="stWidgetLabel"] p {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
    }

    /* Executive Top Bar */
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.25rem 1.75rem;
        background: linear-gradient(135deg, #161F33 0%, #0D1527 100%);
        border: 1px solid #25334D;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
    }
    .top-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .top-subtitle {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-top: 0.25rem;
    }
    .status-pill {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        border: 1px solid rgba(16, 185, 129, 0.4);
        letter-spacing: 0.03em;
    }

    /* Executive Metric Cards */
    .metric-card {
        background-color: #131B2E;
        border: 1px solid #25334D;
        border-radius: 10px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-num {
        font-size: 1.75rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0.3rem 0;
    }
    .metric-foot {
        font-size: 0.8rem;
        font-weight: 500;
    }
    .foot-green { color: #10B981; }
    .foot-indigo { color: #818CF8; }
    .foot-cyan { color: #38BDF8; }
    .foot-muted { color: #64748B; }

    /* Actionable Strategy Case Study Box */
    .strategy-card {
        background-color: #131B2E;
        border-radius: 10px;
        border: 1px solid #25334D;
        padding: 1.3rem;
        height: 100%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
    }
    .strategy-card-positive {
        border-left: 4px solid #10B981;
    }
    .strategy-card-negative {
        border-left: 4px solid #F43F5E;
    }
    .strategy-head {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .strategy-row {
        display: flex;
        justify-content: space-between;
        padding: 0.45rem 0;
        border-bottom: 1px solid #1E293B;
        font-size: 0.88rem;
    }
    .strategy-label { color: #8B949E; }
    .strategy-val { font-weight: 600; color: #F8FAFC; }

    /* Action Badges with High Contrast */
    .badge-approve {
        margin-top: 1rem;
        padding: 0.65rem;
        background-color: #10B981;
        color: #022C22;
        font-weight: 700;
        text-align: center;
        font-size: 0.85rem;
        border-radius: 6px;
        letter-spacing: 0.02em;
    }
    .badge-reject {
        margin-top: 1rem;
        padding: 0.65rem;
        background-color: #F43F5E;
        color: #4C0519;
        font-weight: 700;
        text-align: center;
        font-size: 0.85rem;
        border-radius: 6px;
        letter-spacing: 0.02em;
    }

    /* Table Fixes */
    .stDataFrame {
        border: 1px solid #25334D !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

#-----------------------------------------------------------------------------
#Data & Model Pipeline Ingestion
#-----------------------------------------------------------------------------
@st.cache_data
def load_decision_data():
    csv_path = "reports/customer_retention_decisions.csv"
    if not os.path.exists(csv_path):
        st.error(f"Decision records missing at '{csv_path}'. Please run `python main.py` first.")
        st.stop()
    return pd.read_csv(csv_path)

@st.cache_resource
def load_model_artifacts():
    try:
        artifacts = load_model("models/xgboost_model.pkl")
        preprocessor = ChurnPreprocessingPipeline.load("models/preprocessor.pkl")
        explainer = ChurnShapExplainer(artifacts["base_model"], artifacts["feature_names"])
        return artifacts, preprocessor, explainer
    except Exception as e:
        return None, None, None

df_decisions = load_decision_data()
artifacts, preprocessor, explainer = load_model_artifacts()

#-----------------------------------------------------------------------------
#Header Component
#-----------------------------------------------------------------------------
st.markdown("""
<div class="top-header">
    <div>
        <div class="top-title">Customer Retention & Revenue Protection System</div>
        <div class="top-subtitle">Value-Based Churn Optimization, Prescriptive Action Routing & Unit Economics</div>
    </div>
    <div>
        <span class="status-pill">● Production Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

#-----------------------------------------------------------------------------
#Global Filter Toolbar
#-----------------------------------------------------------------------------
st.sidebar.markdown("### Portfolio Filters")

selected_risk = st.sidebar.multiselect(
    "Risk Tiers",
    options=["Low Risk", "Medium Risk", "High Risk", "Critical Risk"],
    default=["Medium Risk", "High Risk", "Critical Risk"]
)

selected_contract = st.sidebar.multiselect(
    "Contract Type",
    options=df_decisions['ContractType'].unique().tolist(),
    default=df_decisions['ContractType'].unique().tolist()
)

min_clv, max_clv = int(df_decisions['CLV'].min()), int(df_decisions['CLV'].max())
selected_clv_range = st.sidebar.slider(
    "Customer Lifetime Value Range ($)",
    min_value=min_clv,
    max_value=max_clv,
    value=(min_clv, max_clv)
)

#Apply global filters
filtered_df = df_decisions[
    (df_decisions['Risk_Tier'].isin(selected_risk)) &
    (df_decisions['ContractType'].isin(selected_contract)) &
    (df_decisions['CLV'] >= selected_clv_range[0]) &
    (df_decisions['CLV'] <= selected_clv_range[1])
]

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Select Portal View",
    [
        "Portfolio Retention Overview",
        "Account Risk Diagnostics",
        "Targeting Matrix & Directory",
        "Retention Economics Simulator",
        "Governance & Trial Setup"
    ],
    label_visibility="collapsed"
)

#CRM Campaign Download Action in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Export Campaign Audience")
csv_data = filtered_df[filtered_df['Recommended_Action'] != "No Intervention"].to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download CRM Action Cohort (.csv)",
    data=csv_data,
    file_name="crm_retention_campaign_audience.csv",
    mime="text/csv",
    use_container_width=True
)

#High-Contrast Plotly Dark Theme
HIGH_CONTRAST_PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94A3B8', family='Inter, Segoe UI, sans-serif'),
    xaxis=dict(gridcolor='#1E293B', zerolinecolor='#1E293B'),
    yaxis=dict(gridcolor='#1E293B', zerolinecolor='#1E293B')
)

#-----------------------------------------------------------------------------
#TAB 1: PORTFOLIO RETENTION OVERVIEW
#-----------------------------------------------------------------------------
if page == "Portfolio Retention Overview":
    
    # Financial KPI Bar
    tot_eval = len(filtered_df)
    targeted = (filtered_df['Recommended_Action'] != "No Intervention").sum()
    budget_spent = filtered_df['Intervention_Cost'].sum()
    net_saved = filtered_df['Expected_Net_Gain'].sum()
    net_roi = (net_saved / budget_spent * 100.0) if budget_spent > 0 else 0.0
    
    k1, k2, k3, k4, k5 = st.columns(5)
    
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Evaluated Portfolio</div>
            <div class="metric-num">{tot_eval:,}</div>
            <div class="metric-foot foot-muted">Active Accounts</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Targeted Accounts</div>
            <div class="metric-num" style="color:#818CF8;">{targeted:,}</div>
            <div class="metric-foot foot-indigo">{targeted/tot_eval if tot_eval > 0 else 0:.1%} Campaign Rate</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Required Budget</div>
            <div class="metric-num">${budget_spent:,.0f}</div>
            <div class="metric-foot foot-muted">Optimized Spend</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Net Revenue Saved</div>
            <div class="metric-num" style="color:#10B981;">${net_saved:,.0f}</div>
            <div class="metric-foot foot-green">Preserved Net Value</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Campaign Net ROI</div>
            <div class="metric-num" style="color:#38BDF8;">{net_roi:.1f}%</div>
            <div class="metric-foot foot-cyan">Return on Investment</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Executive Briefing Callout Card
    st.markdown("""
    <div style="background-color: #131B2E; border-left: 4px solid #818CF8; padding: 1rem 1.25rem; border-radius: 8px; border: 1px solid #25334D; margin-bottom: 1.5rem; font-size: 0.9rem; color: #CBD5E1;">
        <strong style="color: #F8FAFC;">Executive Summary:</strong> 
        Unifying calibrated XGBoost probabilities with Customer Lifetime Value (CLV) prevents capital destruction. 
        Across the selected cohort, the engine prescribes <strong>${:,.2f}</strong> in net preserved revenue with a <strong>{:.1f}% Net ROI</strong> while eliminating wasted retention offers on non-viable accounts.
    </div>
    """.format(net_saved, net_roi), unsafe_allow_html=True)

    # Side-by-Side Customer Case Contrast
    st.markdown("##### Strategic Targeting Logic: Value-Ignorant vs Value-Optimized")
    
    c_case1, c_case2 = st.columns(2)
    
    with c_case1:
        st.markdown("""
        <div class="strategy-card strategy-card-positive">
            <div class="strategy-head" style="color: #10B981;">
                Case Study A: High Risk + High Account Value
            </div>
            <div class="strategy-row">
                <span class="strategy-label">Customer Profile</span>
                <span class="strategy-val">Enterprise Account #1042</span>
            </div>
            <div class="strategy-row">
                <span class="strategy-label">Calibrated Churn Risk P(Churn)</span>
                <span class="strategy-val">87.0%</span>
            </div>
            <div class="strategy-row">
                <span class="strategy-label">Annual Customer Value (CLV)</span>
                <span class="strategy-val">$1,200.00</span>
            </div>
            <div class="strategy-row">
                <span class="strategy-label">Retention Offer Cost</span>
                <span class="strategy-val">$50.00 (Dedicated VIP Concierge)</span>
            </div>
            <div class="strategy-row">
                <span class="strategy-label">Expected Retention Recovery</span>
                <span class="strategy-val">$420.00</span>
            </div>
            <div class="strategy-row" style="border-bottom:none; margin-top:0.4rem;">
                <span style="color:#F8FAFC; font-weight:700;">Expected Net Benefit</span>
                <span style="color:#10B981; font-weight:700;">+$370.00 (ROI: 740%)</span>
            </div>
            <div class="badge-approve">
                DECISION: APPROVE VIP CONCIERGE OUTREACH
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c_case2:
        st.markdown("""
        <div class="strategy-card strategy-card-negative">
            <div class="strategy-head" style="color: #F43F5E;">
                Case Study B: High Risk + Low Account Value
            </div>
            <div class="strategy-row">
                <span class="strategy-label">Customer Profile</span>
                <span class="strategy-val">Basic Tier Account #8491</span>
            </div>
            <div class="strategy-row">
                <span class="strategy-label">Calibrated Churn Risk P(Churn)</span>
                <span class="strategy-val">82.0%</span>
            </div>
            <div class="strategy-row">
                <span class="strategy-label">Annual Customer Value (CLV)</span>
                <span class="strategy-val">$80.00</span>
            </div>
            <div class="strategy-row">
                <span class="strategy-label">Retention Offer Cost</span>
                <span class="strategy-val">$50.00 (Discount Voucher)</span>
            </div>
            <div class="strategy-row">
                <span class="strategy-label">Expected Retention Recovery</span>
                <span class="strategy-val">$28.00</span>
            </div>
            <div class="strategy-row" style="border-bottom:none; margin-top:0.4rem;">
                <span style="color:#F8FAFC; font-weight:700;">Expected Net Benefit</span>
                <span style="color:#F43F5E; font-weight:700;">-$22.00 (Negative ROI)</span>
            </div>
            <div class="badge-reject">
                DECISION: REJECT INTERVENTION (PRESERVE CAPITAL)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    
    # High-Contrast Visual Charts Grid
    g1, g2 = st.columns(2)
    
    with g1:
        fig_budget = px.pie(
            filtered_df, 
            names='Recommended_Action', 
            values='Intervention_Cost',
            title="Retention Budget Allocation by Action Type",
            hole=0.5,
            color_discrete_sequence=['#818CF8', '#10B981', '#F59E0B', '#475569']
        )
        fig_budget.update_traces(
            textposition='inside',
            textinfo='percent',
            marker=dict(line=dict(color='#0A0E17', width=2)),
            domain=dict(y=[0.1, 0.95])
        )
        fig_budget.update_layout(
            height=420,
            title=dict(text="Retention Budget Allocation by Action Type", x=0.5, xanchor='center', y=0.96),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                title=dict(text="")
            ),
            margin=dict(l=40, r=40, t=50, b=80),
            **HIGH_CONTRAST_PLOTLY_THEME
        )
        st.plotly_chart(fig_budget, use_container_width=True)
        
    with g2:
        fig_dist = px.histogram(
            filtered_df, 
            x='Churn_Probability', 
            color='Risk_Tier',
            nbins=30,
            title="Calibrated Churn Probability Distribution",
            color_discrete_map={
                'Low Risk': '#10B981',
                'Medium Risk': '#F59E0B',
                'High Risk': '#818CF8',
                'Critical Risk': '#F43F5E'
            }
        )
        fig_dist.update_layout(
            height=420,
            title=dict(text="Calibrated Churn Probability Distribution", x=0.5, xanchor='center', y=0.96),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                title=dict(text="")
            ),
            margin=dict(l=40, r=40, t=50, b=80),
            **HIGH_CONTRAST_PLOTLY_THEME
        )
        st.plotly_chart(fig_dist, use_container_width=True)

#-----------------------------------------------------------------------------
#TAB 2: ACCOUNT RISK DIAGNOSTICS & SHAP
#-----------------------------------------------------------------------------
elif page == "Account Risk Diagnostics":
    st.markdown("##### Account Risk Attribution & Root Cause Analysis")
    st.markdown("<span style='color:#8B949E; font-size:0.88rem;'>Deep-dive individual accounts to inspect underlying drivers of churn risk using TreeSHAP feature attributions.</span>", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)
    
    lookup_id = st.selectbox("Search Account ID", filtered_df['CustomerID'])
    account = filtered_df[filtered_df['CustomerID'] == lookup_id].iloc[0]
    
    ac1, ac2, ac3, ac4, ac5 = st.columns(5)
    ac1.metric("Account ID", f"#{account['CustomerID']}")
    ac2.metric("Contract Type", account['ContractType'])
    ac3.metric("Churn Risk P(Churn)", f"{account['Churn_Probability']:.1%}")
    ac4.metric("Risk Tier", account['Risk_Tier'])
    ac5.metric("Account Value (CLV)", f"${account['CLV']:,.2f}")
    
    st.markdown("---")
    
    if explainer is not None and preprocessor is not None:
        acc_df = pd.DataFrame([account])
        X_acc = preprocessor.transform(acc_df)
        shap_info = explainer.explain_single_customer(X_acc[0])
        
        col_push, col_anchor = st.columns(2)
        
        with col_push:
            st.markdown("<strong style='color:#F43F5E; font-size:0.95rem;'>Top Friction Drivers (Increasing Churn Probability)</strong>", unsafe_allow_html=True)
            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
            for item in shap_info['top_churn_drivers']:
                st.markdown(f"""
                <div style="background-color: #131B2E; border-left: 3px solid #F43F5E; border: 1px solid #25334D; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 0.5rem; display: flex; justify-content: space-between; font-size:0.88rem;">
                    <div><strong style="color:#F8FAFC;">{item['feature']}</strong> <span style="color:#8B949E;">(Observed: {item['value']:.2f})</span></div>
                    <div style="color:#F43F5E; font-weight:600;">+{item['shap_impact']:.3f} SHAP</div>
                </div>
                """, unsafe_allow_html=True)
                
        with col_anchor:
            st.markdown("<strong style='color:#10B981; font-size:0.95rem;'>Top Loyalty Anchors (Decreasing Churn Probability)</strong>", unsafe_allow_html=True)
            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
            for item in shap_info['top_retention_anchors']:
                st.markdown(f"""
                <div style="background-color: #131B2E; border-left: 3px solid #10B981; border: 1px solid #25334D; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 0.5rem; display: flex; justify-content: space-between; font-size:0.88rem;">
                    <div><strong style="color:#F8FAFC;">{item['feature']}</strong> <span style="color:#8B949E;">(Observed: {item['value']:.2f})</span></div>
                    <div style="color:#10B981; font-weight:600;">{item['shap_impact']:.3f} SHAP</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #131B2E; border: 1px solid #25334D; padding: 1.2rem; border-radius: 8px;">
        <strong style="color:#F8FAFC;">Prescribed Action Plan for Account #{}:</strong><br/>
        <span style="color:#818CF8; font-weight:600; font-size:1.05rem;">▶ {}</span><br/>
        <span style="color:#8B949E; font-size:0.85rem;">Allocated Cost: <strong>${:,.2f}</strong> | Expected Net Financial Benefit: <strong style="color:#10B981;">+${:,.2f}</strong></span>
    </div>
    """.format(account['CustomerID'], account['Recommended_Action'], account['Intervention_Cost'], account['Expected_Net_Gain']), unsafe_allow_html=True)

#-----------------------------------------------------------------------------
#TAB 3: TARGETING MATRIX & DIRECTORY
#-----------------------------------------------------------------------------
elif page == "Targeting Matrix & Directory":
    st.markdown("##### Portfolio Targeting Density & Action Routing")
    st.markdown("<span style='color:#8B949E; font-size:0.88rem;'>2D segmentation matrix mapping Churn Risk Tiers against Customer Lifetime Value (CLV) Tiers.</span>", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)
    
    pivot_gain = filtered_df.pivot_table(
        index='Risk_Tier', 
        columns='Value_Tier', 
        values='Expected_Net_Gain', 
        aggfunc='sum'
    ).fillna(0)
    
    fig_matrix = px.imshow(
        pivot_gain,
        text_auto=",.0f",
        aspect="auto",
        labels=dict(x="Customer Lifetime Value (CLV) Tier", y="Churn Risk Tier", color="Net Preserved Gain ($)"),
        color_continuous_scale="Tealgrn",
        title="Net Preserved Revenue Density Matrix ($)"
    )
    fig_matrix.update_layout(**HIGH_CONTRAST_PLOTLY_THEME)
    st.plotly_chart(fig_matrix, use_container_width=True)
    
    st.markdown("##### Campaign Action Routing Directory")
    cols_to_show = ['CustomerID', 'Risk_Tier', 'Value_Tier', 'ContractType', 'Churn_Probability', 'CLV', 'Recommended_Action', 'Intervention_Cost', 'Expected_Net_Gain']
    st.dataframe(
        filtered_df[cols_to_show].sort_values(by='Expected_Net_Gain', ascending=False),
        use_container_width=True,
        height=380
    )

#-----------------------------------------------------------------------------
#TAB 4: RETENTION ECONOMICS SIMULATOR
#-----------------------------------------------------------------------------
elif page == "Retention Economics Simulator":
    st.markdown("##### Commercial Strategy & Unit Economics Simulator")
    st.markdown("<span style='color:#8B949E; font-size:0.88rem;'>Adjust offer costs and customer acceptance probabilities to simulate portfolio net profitability.</span>", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)
    
    col_inputs, col_outputs = st.columns([1, 2])
    
    with col_inputs:
        st.markdown("<h6>Commercial Offer Costs ($)</h6>")
        c_disc = st.slider("15% Renewal Discount ($)", 10, 100, 35)
        c_upg = st.slider("Free Speed Upgrade ($)", 20, 150, 65)
        c_vip = st.slider("Dedicated VIP Concierge ($)", 50, 300, 150)
        
        st.markdown("<br/><h6>Base Acceptance Rates (%)</h6>", unsafe_allow_html=True)
        a_disc = st.slider("Discount Acceptance (%)", 10, 90, 45) / 100.0
        a_upg = st.slider("Upgrade Acceptance (%)", 10, 90, 65) / 100.0
        a_vip = st.slider("VIP Concierge Acceptance (%)", 10, 95, 82) / 100.0
        
    custom_catalog = {
        "NO_ACTION": {"name": "No Intervention", "cost": 0.0, "base_acceptance": 0.00},
        "DISCOUNT_VOUCHER": {"name": "15% Discount Voucher", "cost": float(c_disc), "base_acceptance": float(a_disc)},
        "FEATURE_UPGRADE": {"name": "Free Service & Speed Upgrade", "cost": float(c_upg), "base_acceptance": float(a_upg)},
        "DEDICATED_VIP": {"name": "Dedicated VIP Concierge", "cost": float(c_vip), "base_acceptance": float(a_vip)}
    }
    
    sim_costs, sim_gains = [], []
    for _, row in filtered_df.iterrows():
        res = evaluate_retention_actions(
            churn_prob=row['Churn_Probability'],
            clv=row['CLV'],
            monthly_charges=row['MonthlyCharges'],
            risk_tier=row['Risk_Tier'],
            catalog=custom_catalog
        )
        sim_costs.append(res['intervention_cost'])
        sim_gains.append(res['expected_net_gain'])
        
    tot_sim_cost = sum(sim_costs)
    tot_sim_gain = sum(sim_gains)
    sim_roi = (tot_sim_gain / tot_sim_cost * 100.0) if tot_sim_cost > 0 else 0.0
    
    with col_outputs:
        st.markdown("<h6>Simulation Results Across Cohort</h6>")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Simulated Budget", f"${tot_sim_cost:,.2f}")
        m2.metric("Simulated Net Revenue Saved", f"${tot_sim_gain:,.2f}")
        m3.metric("Simulated Campaign Net ROI", f"{sim_roi:.1f}%")
        
        st.markdown("<br/>", unsafe_allow_html=True)
        st.latex(r"\text{Expected Net Gain}_i = P(\text{Churn}_i) \cdot P(\text{Acceptance}_{i,a}) \cdot \text{CLV}_i - \text{Intervention Cost}_a")
        
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color: #131B2E; border-left: 3px solid #F59E0B; border: 1px solid #25334D; padding: 0.9rem; border-radius: 6px; font-size: 0.85rem; color: #CBD5E1;">
            <strong>Simulation Note:</strong> The retention engine dynamically re-evaluates all accounts in real time against your specified unit economic assumptions, routing each account to its optimal financial decision.
        </div>
        """, unsafe_allow_html=True)

#-----------------------------------------------------------------------------
#TAB 5: GOVERNANCE & TRIAL SETUP
#-----------------------------------------------------------------------------
elif page == "Governance & Trial Setup":
    st.markdown("##### Production Governance, Stress Testing & Experiment Design")
    st.markdown("<span style='color:#8B949E; font-size:0.88rem;'>Enterprise risk controls: Monte Carlo VaR simulation, Population Stability Index (PSI) drift monitoring, and A/B power calculations.</span>", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)
    
    st.markdown("<h6>Monte Carlo Financial Sensitivity Analysis (500 Market Shock Runs)</h6>")
    mc_res = run_monte_carlo_sensitivity(filtered_df, n_simulations=500, seed=42)
    
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Baseline Preserved Profit", f"${mc_res['base_net_gain']:,.2f}")
    mc2.metric("Mean Simulated Profit", f"${mc_res['mean_simulated_net_gain']:,.2f}")
    mc3.metric("95% Downside Profit (VaR)", f"${mc_res['var_95_net_gain']:,.2f}")
    
    fig_mc = px.histogram(
        mc_res["simulated_net_gains"], 
        nbins=40,
        title="Simulated Net Gain Distribution Under Adverse Market Stress",
        labels={'value': 'Net Preserved Profit ($)'},
        color_discrete_sequence=['#818CF8']
    )
    fig_mc.update_layout(**HIGH_CONTRAST_PLOTLY_THEME)
    st.plotly_chart(fig_mc, use_container_width=True)
    
    st.markdown("---")
    
    col_psi, col_ab = st.columns(2)
    
    with col_psi:
        st.markdown("<h6>Population Stability Index (PSI) Feature Drift Monitoring</h6>")
        drift_res = monitor_feature_drift(filtered_df, filtered_df)
        drift_df = pd.DataFrame.from_dict(drift_res, orient='index')
        st.dataframe(drift_df, use_container_width=True)
        
    with col_ab:
        st.markdown("<h6>Causal A/B Testing Trial Design & Sample Power Calculator</h6>")
        ab_res = calculate_ab_sample_size(baseline_churn_rate=0.42, expected_churn_reduction=0.05)
        st.markdown(f"""
        <div style="background-color: #131B2E; border: 1px solid #25334D; padding: 1.1rem; border-radius: 8px; margin-top: 0.4rem;">
            <div style="font-size:0.8rem; color:#8B949E; text-transform:uppercase;">Required Cohort per Test Group</div>
            <div style="font-size:1.6rem; font-weight:700; color:#F8FAFC; margin: 0.2rem 0;">{ab_res['required_sample_size_per_group']:,} accounts</div>
            <div style="font-size:0.85rem; color:#10B981;">Total Pilot Experiment Requirement: {ab_res['total_sample_size']:,} accounts</div>
            <div style="font-size:0.78rem; color:#64748B; margin-top:0.4rem;">Parameters: 95% Confidence (α = 0.05), 80% Statistical Power, MDE = 5.0%</div>
        </div>
        """, unsafe_allow_html=True)
