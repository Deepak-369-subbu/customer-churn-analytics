import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

# Local modular imports
from src.data_processing import load_and_preprocess_data, get_feature_matrix
from src.model_engine import (
    train_or_load_model,
    predict_single_customer,
    get_risk_segment,
    get_risk_color,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD
)
from src.ui_components import (
    apply_custom_theme,
    render_metric_card,
    plot_churn_gauge,
    plot_confusion_matrix_heatmap,
    plot_roc_curve,
    plot_feature_importance_bar
)

# Set page config
st.set_page_config(
    page_title="Retail Banking — Customer Churn Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek styling
apply_custom_theme()

@st.cache_data(show_spinner="Loading and preprocessing banking data...")
def get_data():
    df = load_and_preprocess_data()
    return df

@st.cache_resource(show_spinner="Initializing Machine Learning Intelligence Engine...")
def get_model(df):
    X, y, feature_names = get_feature_matrix(df)
    model, metrics, X_test, y_test, y_probs, importances = train_or_load_model(X, y)
    
    # Compute full dataset churn probabilities
    all_probs = model.predict_proba(X)[:, 1]
    df_scored = df.copy()
    df_scored['Churn_Probability'] = all_probs.round(4)
    df_scored['Risk_Segment'] = df_scored['Churn_Probability'].apply(get_risk_segment)
    
    return model, metrics, feature_names, df_scored, importances, y_test, y_probs

# Load data and trained ML model
try:
    df_raw = get_data()
    model, metrics, feature_names, df_scored, importances, y_test, y_probs = get_model(df_raw)
except Exception as e:
    st.error(f"Error initializing data/model: {e}")
    st.stop()

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bank.png", width=64)
    st.markdown("### **Bank Retention Portal**")
    st.caption("AI-Powered Customer Churn Analytics")
    st.markdown("---")
    
    st.markdown("#### 🔍 **Global Portfolio Filters**")
    
    # Geography filter
    selected_geo = st.multiselect(
        "Geography",
        options=sorted(df_scored['Geography'].unique()),
        default=sorted(df_scored['Geography'].unique())
    )
    
    # Gender filter
    selected_gender = st.multiselect(
        "Gender",
        options=sorted(df_scored['Gender'].unique()),
        default=sorted(df_scored['Gender'].unique())
    )
    
    # Age Group filter
    age_groups = list(df_scored['AgeGroup'].dropna().unique().astype(str))
    selected_age = st.multiselect(
        "Age Group",
        options=age_groups,
        default=age_groups
    )
    
    # Activity status
    activity_filter = st.radio(
        "Member Activity",
        options=["All Customers", "Active Only", "Inactive Only"],
        horizontal=True
    )
    
    # Apply filters
    filtered_df = df_scored.copy()
    if selected_geo:
        filtered_df = filtered_df[filtered_df['Geography'].isin(selected_geo)]
    if selected_gender:
        filtered_df = filtered_df[filtered_df['Gender'].isin(selected_gender)]
    if selected_age:
        filtered_df = filtered_df[filtered_df['AgeGroup'].astype(str).isin(selected_age)]
    if activity_filter == "Active Only":
        filtered_df = filtered_df[filtered_df['IsActiveMember'] == 1]
    elif activity_filter == "Inactive Only":
        filtered_df = filtered_df[filtered_df['IsActiveMember'] == 0]
        
    st.markdown("---")
    st.caption(f"Showing **{len(filtered_df):,}** of **{len(df_scored):,}** records")
    st.caption("Model: Balanced Random Forest (200 trees, depth 8)")

# ----------------- HEADER & TABS -----------------
col_title, col_stat = st.columns([3, 1])
with col_title:
    st.title("🏦 Retail Banking Customer Churn Intelligence")
    st.markdown(
        "Executive decision support system combining predictive machine learning, "
        "portfolio risk intelligence, and automated prescriptive retention playbooks."
    )
with col_stat:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align: right; color: #475569; font-size: 0.88rem; font-weight: 600;'>"
        f"Active Model Accuracy: <strong style='color:#059669;'>{metrics['accuracy']:.1%}</strong><br>"
        f"ROC-AUC: <strong style='color:#4F46E5;'>{metrics['auc']:.3f}</strong>"
        f"</div>",
        unsafe_allow_html=True
    )

# Primary Navigation Tabs
tabs = st.tabs([
    "📊 Executive Overview",
    "🧠 Risk & ML Diagnostics",
    "🔮 Real-Time Simulator",
    "🎯 High-Value At-Risk Explorer",
    "💼 Financial ROI Calculator"
])

# =========================================================================
# TAB 1: EXECUTIVE OVERVIEW
# =========================================================================
with tabs[0]:
    # Top KPI Metrics
    total_customers = len(filtered_df)
    churn_count = int(filtered_df['Exited'].sum())
    churn_rate = (churn_count / total_customers * 100) if total_customers > 0 else 0
    total_balance = filtered_df['Balance'].sum()
    churn_balance = filtered_df[filtered_df['Exited'] == 1]['Balance'].sum()
    high_risk_count = len(filtered_df[filtered_df['Risk_Segment'] == 'High Risk'])
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_metric_card(
            title="Total Customer Base",
            value=f"{total_customers:,}",
            subtitle=f"Filtered portfolio snapshot",
            badge_text="Active Portfolio",
            badge_type="default"
        )
    with kpi2:
        badge_type = "danger" if churn_rate > 20 else "warning"
        render_metric_card(
            title="Historical Churn Rate",
            value=f"{churn_rate:.1f}%",
            subtitle=f"{churn_count:,} customers exited",
            badge_text=f"{churn_rate:.1f}% Churned",
            badge_type=badge_type
        )
    with kpi3:
        render_metric_card(
            title="Total Portfolio Capital",
            value=f"${total_balance/1e6:.1f}M",
            subtitle=f"${churn_balance/1e6:.1f}M capital lost to churn",
            badge_text=f"${churn_balance/1e6:.1f}M Exited",
            badge_type="danger"
        )
    with kpi4:
        pct_high_risk = (high_risk_count / total_customers * 100) if total_customers > 0 else 0
        render_metric_card(
            title="High Churn Risk Customers",
            value=f"{high_risk_count:,}",
            subtitle=f"Probability ≥ 56.5%",
            badge_text=f"{pct_high_risk:.1f}% of total",
            badge_type="warning"
        )

    st.markdown("<div class='section-header'>🗺️ Geographic & Demographic Churn Patterns</div>", unsafe_allow_html=True)
    
    dem_metric = st.radio(
        "Demographic Metric View:",
        options=["Churn Rate (%)", "Customer Headcount (Retained vs Churned)", "Deposits at Risk ($ Millions)"],
        horizontal=True,
        key="dem_metric_toggle"
    )
    
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        # Churn by Geography
        geo_agg = filtered_df.groupby('Geography').agg(
            Total=('CustomerId', 'count'),
            Churned=('Exited', 'sum'),
            Retained=('Exited', lambda x: (x == 0).sum()),
            BalanceLost=('Balance', lambda b: b[filtered_df.loc[b.index, 'Exited'] == 1].sum())
        ).reset_index()
        geo_agg['Churn_Rate'] = (geo_agg['Churned'] / geo_agg['Total'] * 100).round(1)
        geo_agg['BalanceLost_M'] = (geo_agg['BalanceLost'] / 1e6).round(2)
        
        if dem_metric == "Churn Rate (%)":
            fig_geo = px.bar(
                geo_agg,
                x='Geography',
                y='Churn_Rate',
                color='Geography',
                text='Churn_Rate',
                color_discrete_sequence=['#3B82F6', '#EF4444', '#10B981'],
                title="<b>Churn Rate by Country (%)</b> — Germany exhibits critical risk"
            )
            fig_geo.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_geo.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#0F172A", family="Plus Jakarta Sans"),
                height=320,
                showlegend=False
            )
            fig_geo.update_yaxes(range=[0, max(geo_agg['Churn_Rate'].max() * 1.25, 45)])
            st.plotly_chart(fig_geo, use_container_width=True)
        elif dem_metric == "Customer Headcount (Retained vs Churned)":
            fig_geo_cnt = px.bar(
                geo_agg,
                x='Geography',
                y=['Retained', 'Churned'],
                barmode='group',
                color_discrete_map={'Retained': '#10B981', 'Churned': '#EF4444'},
                title="<b>Customers by Country (Retained vs Churned)</b>"
            )
            fig_geo_cnt.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#0F172A", family="Plus Jakarta Sans"),
                height=320,
                legend=dict(title="Status")
            )
            st.plotly_chart(fig_geo_cnt, use_container_width=True)
        else:
            fig_geo_bal = px.bar(
                geo_agg,
                x='Geography',
                y='BalanceLost_M',
                text='BalanceLost_M',
                color='BalanceLost_M',
                color_continuous_scale=[(0, "#93C5FD"), (1, "#1E3A8A")],
                title="<b>Total Capital Lost to Churn by Country ($M)</b>"
            )
            fig_geo_bal.update_traces(texttemplate='$%{text:.2f}M', textposition='outside')
            fig_geo_bal.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#0F172A", family="Plus Jakarta Sans"),
                height=320,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_geo_bal, use_container_width=True)

    with g_col2:
        # Churn by Age Group
        age_agg = filtered_df.groupby('AgeGroup', observed=False).agg(
            Total=('CustomerId', 'count'),
            Churned=('Exited', 'sum'),
            Retained=('Exited', lambda x: (x == 0).sum()),
            BalanceLost=('Balance', lambda b: b[filtered_df.loc[b.index, 'Exited'] == 1].sum())
        ).reset_index()
        age_agg['Churn_Rate'] = (age_agg['Churned'] / age_agg['Total'] * 100).round(1)
        age_agg['BalanceLost_M'] = (age_agg['BalanceLost'] / 1e6).round(2)
        
        if dem_metric == "Churn Rate (%)":
            fig_age = px.bar(
                age_agg,
                x='AgeGroup',
                y='Churn_Rate',
                color='AgeGroup',
                text='Churn_Rate',
                color_discrete_sequence=['#34D399', '#F59E0B', '#EF4444'],
                title="<b>Churn Rate by Demographic Age Group (%)</b>"
            )
            fig_age.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_age.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#0F172A", family="Plus Jakarta Sans"),
                height=320,
                showlegend=False
            )
            fig_age.update_yaxes(range=[0, max(age_agg['Churn_Rate'].max() * 1.25, 60)])
            st.plotly_chart(fig_age, use_container_width=True)
        elif dem_metric == "Customer Headcount (Retained vs Churned)":
            fig_age_cnt = px.bar(
                age_agg,
                x='AgeGroup',
                y=['Retained', 'Churned'],
                barmode='group',
                color_discrete_map={'Retained': '#10B981', 'Churned': '#EF4444'},
                title="<b>Customers by Age Group (Retained vs Churned)</b>"
            )
            fig_age_cnt.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#0F172A", family="Plus Jakarta Sans"),
                height=320,
                legend=dict(title="Status")
            )
            st.plotly_chart(fig_age_cnt, use_container_width=True)
        else:
            fig_age_bal = px.bar(
                age_agg,
                x='AgeGroup',
                y='BalanceLost_M',
                text='BalanceLost_M',
                color='BalanceLost_M',
                color_continuous_scale=[(0, "#FCD34D"), (1, "#D97706")],
                title="<b>Total Capital Lost to Churn by Age Group ($M)</b>"
            )
            fig_age_bal.update_traces(texttemplate='$%{text:.2f}M', textposition='outside')
            fig_age_bal.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#0F172A", family="Plus Jakarta Sans"),
                height=320,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_age_bal, use_container_width=True)

    st.markdown("<div class='section-header'>💳 Product Engagement & Balance Dynamics</div>", unsafe_allow_html=True)
    st.caption("Interactively filter and analyze how product holdings and balance tiers drive retention across different wealth brackets.")
    
    # Interactive Account Balance & Product Slicers
    c_bal_slider, c_prod_sel = st.columns([1.2, 1])
    with c_bal_slider:
        bal_range = st.slider(
            "Interactive Account Balance Filter ($):",
            min_value=0,
            max_value=250000,
            value=(0, 250000),
            step=5000,
            help="Filter customers by balance to see how churn behavior shifts across deposit tiers."
        )
    with c_prod_sel:
        prod_filter = st.multiselect(
            "Filter Product Holdings:",
            options=[1, 2, 3, 4],
            default=[1, 2, 3, 4],
            help="Select which product counts to include in analysis."
        )
        
    # Dynamically filtered subset for this section
    dyn_prod_df = filtered_df[
        (filtered_df['Balance'] >= bal_range[0]) & 
        (filtered_df['Balance'] <= bal_range[1]) &
        (filtered_df['NumOfProducts'].isin(prod_filter))
    ]
    
    if len(dyn_prod_df) == 0:
        st.warning("No customers match the selected balance and product criteria. Please widen your filter range.")
    else:
        # Mini KPIs for this filtered wealth segment
        dyn_churn_pct = (dyn_prod_df['Exited'].sum() / len(dyn_prod_df) * 100)
        dyn_deposits_m = dyn_prod_df['Balance'].sum() / 1e6
        dyn_lost_m = dyn_prod_df[dyn_prod_df['Exited'] == 1]['Balance'].sum() / 1e6
        
        dyn_col1, dyn_col2, dyn_col3 = st.columns(3)
        with dyn_col1:
            st.metric("Cohort Customers", f"{len(dyn_prod_df):,}")
        with dyn_col2:
            st.metric("Cohort Churn Rate", f"{dyn_churn_pct:.1f}%")
        with dyn_col3:
            st.metric("Cohort Balance at Risk", f"${dyn_lost_m:.2f}M", help=f"Total cohort deposits: ${dyn_deposits_m:.2f}M")
            
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            prod_view = st.radio(
                "Product Chart View:",
                options=["Churn Rate Curve (%)", "Customer Headcount (Retained vs Churned)", "Capital Lost ($ Millions)"],
                horizontal=True,
                key="prod_view_toggle"
            )
            
            prod_agg = dyn_prod_df.groupby('NumOfProducts').agg(
                Total=('CustomerId', 'count'),
                Churned=('Exited', 'sum'),
                Retained=('Exited', lambda x: (x == 0).sum()),
                BalanceLost=('Balance', lambda b: b[dyn_prod_df.loc[b.index, 'Exited'] == 1].sum())
            ).reset_index()
            prod_agg['Churn_Rate'] = (prod_agg['Churned'] / prod_agg['Total'] * 100).round(1)
            prod_agg['BalanceLost_M'] = (prod_agg['BalanceLost'] / 1e6).round(2)
            
            if prod_view == "Churn Rate Curve (%)":
                fig_prod = px.line(
                    prod_agg,
                    x='NumOfProducts',
                    y='Churn_Rate',
                    markers=True,
                    text='Churn_Rate',
                    title=f"<b>Churn Rate vs. Products Held (%)</b> ({len(dyn_prod_df):,} clients)",
                    line_shape='spline'
                )
                fig_prod.update_traces(
                    line_color='#EC4899', line_width=3,
                    marker=dict(size=10, color='#F43F5E'),
                    texttemplate='%{text:.1f}%', textposition='top center'
                )
                fig_prod.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#0F172A", family="Plus Jakarta Sans"),
                    height=320
                )
                fig_prod.update_xaxes(tickmode='linear', tick0=1, dtick=1)
                st.plotly_chart(fig_prod, use_container_width=True)
            elif prod_view == "Customer Headcount (Retained vs Churned)":
                fig_prod_count = px.bar(
                    prod_agg,
                    x='NumOfProducts',
                    y=['Retained', 'Churned'],
                    barmode='group',
                    color_discrete_map={'Retained': '#10B981', 'Churned': '#EF4444'},
                    title=f"<b>Customers by Products Held</b> (Retained vs Churned)"
                )
                fig_prod_count.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#0F172A", family="Plus Jakarta Sans"),
                    height=320,
                    legend=dict(title="Status")
                )
                fig_prod_count.update_xaxes(tickmode='linear', tick0=1, dtick=1)
                st.plotly_chart(fig_prod_count, use_container_width=True)
            else:
                fig_prod_bal = px.bar(
                    prod_agg,
                    x='NumOfProducts',
                    y='BalanceLost_M',
                    text='BalanceLost_M',
                    color='BalanceLost_M',
                    color_continuous_scale=[(0, "#FCA5A5"), (1, "#DC2626")],
                    title=f"<b>Total Capital Lost to Churn by Product ($M)</b>"
                )
                fig_prod_bal.update_traces(texttemplate='$%{text:.2f}M', textposition='outside')
                fig_prod_bal.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#0F172A", family="Plus Jakarta Sans"),
                    height=320,
                    coloraxis_showscale=False
                )
                fig_prod_bal.update_xaxes(tickmode='linear', tick0=1, dtick=1)
                st.plotly_chart(fig_prod_bal, use_container_width=True)
                
            st.caption("📌 *Insight: 2 products is the 'sweet spot' for retention. Holding 3 or 4 products shows massive churn rates.*")

        with p_col2:
            bal_view = st.radio(
                "Balance Chart View:",
                options=["Distribution Histogram", "Balance Tiers (Zero, Low, Med, High)", "Balance vs Credit Score"],
                horizontal=True,
                key="bal_view_toggle"
            )
            
            if bal_view == "Distribution Histogram":
                bal_df = dyn_prod_df.copy()
                bal_df['Status'] = bal_df['Exited'].map({0: 'Retained', 1: 'Churned'})
                fig_bal = px.histogram(
                    bal_df,
                    x='Balance',
                    color='Status',
                    barmode='overlay',
                    nbins=40,
                    color_discrete_map={'Retained': '#10B981', 'Churned': '#EF4444'},
                    title=f"<b>Balance Distribution by Status</b> (${bal_range[0]:,} - ${bal_range[1]:,})"
                )
                fig_bal.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#0F172A", family="Plus Jakarta Sans"),
                    height=320,
                    legend=dict(title="Status")
                )
                st.plotly_chart(fig_bal, use_container_width=True)
            elif bal_view == "Balance Tiers (Zero, Low, Med, High)":
                tier_agg = dyn_prod_df.groupby('BalanceCategory', observed=False).agg(
                    Total=('CustomerId', 'count'),
                    Churned=('Exited', 'sum')
                ).reset_index()
                tier_agg['Churn_Rate'] = (tier_agg['Churned'] / tier_agg['Total'] * 100).fillna(0).round(1)
                
                fig_tier = px.bar(
                    tier_agg,
                    x='BalanceCategory',
                    y='Churn_Rate',
                    color='BalanceCategory',
                    text='Churn_Rate',
                    color_discrete_sequence=['#64748B', '#38BDF8', '#818CF8', '#EF4444'],
                    title="<b>Churn Rate across Balance Tiers (%)</b>"
                )
                fig_tier.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_tier.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#0F172A", family="Plus Jakarta Sans"),
                    height=320,
                    showlegend=False
                )
                st.plotly_chart(fig_tier, use_container_width=True)
            else:
                scatter_df = dyn_prod_df.sample(min(1500, len(dyn_prod_df)), random_state=42).copy()
                scatter_df['Status'] = scatter_df['Exited'].map({0: 'Retained', 1: 'Churned'})
                fig_scatter = px.scatter(
                    scatter_df,
                    x='Balance',
                    y='CreditScore',
                    color='Status',
                    color_discrete_map={'Retained': '#10B981', 'Churned': '#EF4444'},
                    opacity=0.6,
                    title="<b>Balance vs. Credit Score Sample</b> (Color by Churn)"
                )
                fig_scatter.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#0F172A", family="Plus Jakarta Sans"),
                    height=320
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

# =========================================================================
# TAB 2: MACHINE LEARNING & RISK DIAGNOSTICS
# =========================================================================
with tabs[1]:
    st.markdown("<div class='section-header'>🧠 Model Evaluation & Diagnostic Curves</div>", unsafe_allow_html=True)
    
    # Metric Callouts
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card(
            title="Model Accuracy",
            value=f"{metrics['accuracy']:.1%}",
            subtitle="Standard 0.5 decision threshold",
            badge_text="Balanced RF",
            badge_type="success"
        )
    with m2:
        render_metric_card(
            title="ROC-AUC Score",
            value=f"{metrics['auc']:.3f}",
            subtitle="Excellent class separability",
            badge_text="Discriminative Power",
            badge_type="success"
        )
    with m3:
        render_metric_card(
            title="Optimal F1 Score",
            value=f"{metrics['f1']:.3f}",
            subtitle=f"At threshold {metrics['best_threshold']:.3f}",
            badge_text="Optimized",
            badge_type="warning"
        )
    with m4:
        render_metric_card(
            title="High-Risk Recall",
            value=f"{metrics['recall']:.1%}",
            subtitle="At-risk churners successfully captured",
            badge_text="Proactive Reach",
            badge_type="success"
        )

    # Threshold Interactive Explorer
    st.markdown("##### 🎚️ **Decision Threshold Tuning**")
    st.caption("Drag the slider to observe how shifting the classification threshold dynamically updates model recall, precision, confusion matrix outcomes, and the ROC curve operating point.")
    
    cust_threshold = st.slider(
        "Interactive Decision Threshold (Trades off Precision vs Recall)",
        min_value=0.10,
        max_value=0.90,
        value=float(metrics['best_threshold']),
        step=0.01,
        help="Adjusting this slider updates how aggressively the bank flags customers for intervention."
    )
    
    # Dynamic recalculation for the selected threshold
    y_pred_tuned = (y_probs >= cust_threshold).astype(int)
    cm_tuned = confusion_matrix(y_test, y_pred_tuned).tolist()
    prec_tuned = precision_score(y_test, y_pred_tuned, zero_division=0)
    rec_tuned = recall_score(y_test, y_pred_tuned, zero_division=0)
    acc_tuned = accuracy_score(y_test, y_pred_tuned)
    f1_tuned = f1_score(y_test, y_pred_tuned, zero_division=0)
    
    # Calculate operating point on ROC
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tuned).ravel()
    current_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    current_tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    # Live dynamic metrics
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.metric(
            label="🎯 Dynamic Recall (Churners Caught)",
            value=f"{rec_tuned:.1%}",
            delta=f"{(rec_tuned - metrics['recall']) * 100:+.1f}% vs opt"
        )
    with t2:
        st.metric(
            label="⚖️ Dynamic Precision",
            value=f"{prec_tuned:.1%}",
            delta=f"{(prec_tuned - metrics['precision']) * 100:+.1f}% vs opt"
        )
    with t3:
        st.metric(
            label="📊 Dynamic F1-Score",
            value=f"{f1_tuned:.3f}"
        )
    with t4:
        st.metric(
            label="⚡ Dynamic Accuracy",
            value=f"{acc_tuned:.1%}"
        )
    
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.plotly_chart(
            plot_roc_curve(
                metrics['fpr'], metrics['tpr'], metrics['auc'],
                current_fpr=current_fpr, current_tpr=current_tpr, threshold=cust_threshold
            ),
            use_container_width=True
        )
    with r_col2:
        st.plotly_chart(
            plot_confusion_matrix_heatmap(cm_tuned, threshold=cust_threshold),
            use_container_width=True
        )
        
    st.markdown("<div class='section-header'>🔍 Feature Importance & Churn Risk Segmentation</div>", unsafe_allow_html=True)
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        feat_mode = st.radio(
            "Predictor Analysis View:",
            options=["Cohort Risk Drivers (Responsive to Filters)", "Global ML Tree Weights"],
            horizontal=True,
            help="Cohort Risk Drivers calculates feature correlation with churn specifically for your active filtered portfolio."
        )
        
        if feat_mode == "Cohort Risk Drivers (Responsive to Filters)":
            calc_df = filtered_df.copy()
            calc_df['Gender_Male'] = (calc_df['Gender'] == 'Male').astype(int)
            features_to_eval = ['Age', 'NumOfProducts', 'Balance', 'IsActiveMember', 'CreditScore', 'Tenure', 'EstimatedSalary', 'Gender_Male', 'HasCrCard']
            
            if len(calc_df['Geography'].unique()) > 1:
                for g in ['Germany', 'France', 'Spain']:
                    if g in calc_df['Geography'].unique():
                        calc_df[f'Geo_{g}'] = (calc_df['Geography'] == g).astype(int)
                        features_to_eval.append(f'Geo_{g}')
                        
            cohort_corrs = calc_df[features_to_eval].apply(lambda s: s.corr(calc_df['Churn_Probability'])).abs().dropna().sort_values(ascending=False)
            chart_title = f"<b>Active Drivers for Filtered Cohort ({len(filtered_df):,} clients)</b>"
            st.plotly_chart(plot_feature_importance_bar(cohort_corrs, title=chart_title), use_container_width=True)
            st.caption("💡 *Updates dynamically based on your sidebar filters (Country, Age, Activity).*")
        else:
            st.plotly_chart(plot_feature_importance_bar(importances, title="<b>Global Random Forest Feature Importance</b>"), use_container_width=True)
            st.caption("🌲 *Base tree-split importance calculated across entire bank training set.*")

    with f_col2:
        # Dynamic Segment Distribution based on Threshold & Filters
        dyn_df = filtered_df.copy()
        med_cutoff = round(max(0.10, cust_threshold * 0.65), 2)
        
        high_label = f"High Risk (≥ {cust_threshold:.2f})"
        med_label = f"Medium Risk ({med_cutoff:.2f} - {cust_threshold:.2f})"
        low_label = f"Low Risk (< {med_cutoff:.2f})"
        
        def assign_dynamic_risk(p):
            if p >= cust_threshold:
                return high_label
            elif p >= med_cutoff:
                return med_label
            else:
                return low_label
                
        dyn_df['Dynamic_Segment'] = dyn_df['Churn_Probability'].apply(assign_dynamic_risk)
        seg_counts = dyn_df['Dynamic_Segment'].value_counts().reindex([low_label, med_label, high_label]).dropna().reset_index()
        seg_counts.columns = ['Risk_Segment', 'Count']
        
        fig_seg = px.pie(
            seg_counts,
            names='Risk_Segment',
            values='Count',
            color='Risk_Segment',
            color_discrete_map={
                high_label: '#EF4444',
                med_label: '#F59E0B',
                low_label: '#10B981'
            },
            hole=0.45,
            title=f"<b>Portfolio Risk at Threshold = {cust_threshold:.2f} ({len(filtered_df):,} clients)</b>"
        )
        fig_seg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0F172A", family="Plus Jakarta Sans"),
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_seg, use_container_width=True)
        
        high_count = len(dyn_df[dyn_df['Churn_Probability'] >= cust_threshold])
        high_pct = (high_count / len(dyn_df) * 100) if len(dyn_df) > 0 else 0
        st.caption(f"⚡ *At threshold **{cust_threshold:.2f}**, **{high_count:,} customers ({high_pct:.1f}%)** are flagged into the High Risk intervention group.*")

# =========================================================================
# TAB 3: REAL-TIME CUSTOMER CHURN SIMULATOR ("WHAT-IF")
# =========================================================================
with tabs[2]:
    st.markdown("<div class='section-header'>🔮 Interactive Customer Churn Predictor & Retention Recommender</div>", unsafe_allow_html=True)
    st.caption("Adjust the customer's financial and demographic attributes below to simulate real-time churn probability and generate automated retention recommendations.")
    
    sim_col1, sim_col2 = st.columns([1.2, 1])
    
    with sim_col1:
        st.markdown("##### **Customer Profile Inputs**")
        sc1, sc2 = st.columns(2)
        with sc1:
            in_geo = st.selectbox("Geography", ["France", "Germany", "Spain"], index=1)
            in_gender = st.selectbox("Gender", ["Female", "Male"], index=0)
            in_age = st.slider("Age", 18, 92, 48)
            in_credit = st.slider("Credit Score", 350, 850, 610)
            in_tenure = st.slider("Tenure (Years)", 0, 10, 3)
        with sc2:
            in_balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=300000.0, value=115000.0, step=5000.0)
            in_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
            in_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=250000.0, value=95000.0, step=5000.0)
            in_has_card = st.checkbox("Has Credit Card", value=True)
            in_is_active = st.checkbox("Is Active Member", value=False)
            
        customer_input = {
            'Geography': in_geo,
            'Gender': in_gender,
            'Age': in_age,
            'CreditScore': in_credit,
            'Tenure': in_tenure,
            'Balance': in_balance,
            'NumOfProducts': in_products,
            'EstimatedSalary': in_salary,
            'HasCrCard': 1 if in_has_card else 0,
            'IsActiveMember': 1 if in_is_active else 0
        }
        
        sim_result = predict_single_customer(model, customer_input, feature_names)
        
    with sim_col2:
        # Gauge Chart
        st.plotly_chart(
            plot_churn_gauge(
                sim_result['churn_probability'],
                sim_result['segment'],
                sim_result['color']
            ),
            use_container_width=True
        )
        
        st.markdown("##### 📋 **Prescriptive Retention Playbook**")
        for act in sim_result['actions']:
            st.markdown(f"<div class='action-card'>{act}</div>", unsafe_allow_html=True)

# =========================================================================
# TAB 4: HIGH-VALUE AT-RISK EXPLORER
# =========================================================================
with tabs[3]:
    st.markdown("<div class='section-header'>🎯 High-Value At-Risk Customer Explorer ('Save the Whales')</div>", unsafe_allow_html=True)
    st.caption("Pinpoint specific high-balance banking clients who are at severe risk of churning so relationship managers can take priority action.")
    
    e_col1, e_col2, e_col3, e_col4 = st.columns(4)
    with e_col1:
        filter_risk = st.selectbox("Filter Risk Segment", ["All Segments", "High Risk", "Medium Risk", "Low Risk"], index=1)
    with e_col2:
        min_balance = st.number_input("Minimum Balance ($)", value=80000.0, step=10000.0)
    with e_col3:
        filter_country = st.multiselect("Filter Country", ["France", "Germany", "Spain"], default=["France", "Germany", "Spain"])
    with e_col4:
        search_query = st.text_input("Search Customer ID or Surname", "")
        
    # Apply Explorer Filters
    exp_df = df_scored.copy()
    if filter_risk != "All Segments":
        exp_df = exp_df[exp_df['Risk_Segment'] == filter_risk]
    exp_df = exp_df[exp_df['Balance'] >= min_balance]
    if filter_country:
        exp_df = exp_df[exp_df['Geography'].isin(filter_country)]
    if search_query:
        search_clean = search_query.strip().lower()
        exp_df = exp_df[
            exp_df['CustomerId'].astype(str).str.contains(search_clean) |
            exp_df['Surname'].str.lower().str.contains(search_clean)
        ]
        
    exp_df = exp_df.sort_values(by=['Balance', 'Churn_Probability'], ascending=[False, False])
    
    # Metrics
    c_whale1, c_whale2, c_whale3 = st.columns(3)
    with c_whale1:
        st.metric("Identified At-Risk Clients", f"{len(exp_df):,}")
    with c_whale2:
        st.metric("Total Deposits at Risk", f"${exp_df['Balance'].sum()/1e6:.2f}M")
    with c_whale3:
        avg_prob = exp_df['Churn_Probability'].mean() * 100 if len(exp_df) > 0 else 0
        st.metric("Average Churn Probability", f"{avg_prob:.1f}%")
        
    display_cols = [
        'CustomerId', 'Surname', 'Geography', 'Gender', 'Age',
        'CreditScore', 'Balance', 'NumOfProducts', 'IsActiveMember',
        'Churn_Probability', 'Risk_Segment'
    ]
    
    st.dataframe(
        exp_df[display_cols].style.format({
            'Balance': '${:,.2f}',
            'Churn_Probability': '{:.1%}'
        }),
        height=400,
        use_container_width=True
    )
    
    # CSV Export Button
    csv_data = exp_df[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Targeted Call List to CSV",
        data=csv_data,
        file_name="bank_retention_priority_list.csv",
        mime="text/csv"
    )

# =========================================================================
# TAB 5: FINANCIAL ROI & CAMPAIGN CALCULATOR
# =========================================================================
with tabs[4]:
    st.markdown("<div class='section-header'>💼 Retention Campaign Financial Impact & ROI Calculator</div>", unsafe_allow_html=True)
    st.caption("Quantify the bottom-line financial value and projected ROI of executing a proactive retention campaign on at-risk banking clients.")
    
    calc_col1, calc_col2 = st.columns([1, 1.2])
    
    with calc_col1:
        st.markdown("##### **Campaign Parameters**")
        camp_target_count = st.slider("Targeted At-Risk Customers", min_value=50, max_value=2500, value=500, step=50)
        camp_cost_per_cust = st.number_input("Intervention Cost per Customer ($)", min_value=10.0, max_value=1000.0, value=75.0, step=5.0)
        camp_success_rate = st.slider("Expected Retention Success Rate (%)", min_value=5, max_value=80, value=25, step=1)
        annual_bank_margin = st.slider("Net Interest Margin / Annual Account Value (%)", min_value=1.0, max_value=8.0, value=2.5, step=0.1)
        
    with calc_col2:
        # Sort by churn prob and balance to target optimal customers
        top_targets = df_scored[df_scored['Risk_Segment'] == 'High Risk'].sort_values(
            by=['Balance', 'Churn_Probability'], ascending=[False, False]
        ).head(camp_target_count)
        
        target_deposits = top_targets['Balance'].sum()
        total_campaign_cost = camp_target_count * camp_cost_per_cust
        saved_customers = int(camp_target_count * (camp_success_rate / 100))
        saved_deposits = target_deposits * (camp_success_rate / 100)
        annual_revenue_saved = saved_deposits * (annual_bank_margin / 100)
        net_profit = annual_revenue_saved - total_campaign_cost
        roi_pct = (net_profit / total_campaign_cost * 100) if total_campaign_cost > 0 else 0
        
        st.markdown("##### **Projected Financial Outcomes**")
        f_kpi1, f_kpi2 = st.columns(2)
        with f_kpi1:
            render_metric_card(
                title="Deposits Protected",
                value=f"${saved_deposits/1e6:.2f}M",
                subtitle=f"From ${target_deposits/1e6:.1f}M targeted",
                badge_text=f"{saved_customers:,} Clients Retained",
                badge_type="success"
            )
        with f_kpi2:
            render_metric_card(
                title="Projected Campaign ROI",
                value=f"{roi_pct:.0f}%",
                subtitle=f"Net Value Saved: ${net_profit:,.0f}",
                badge_text=f"${total_campaign_cost:,.0f} Budget",
                badge_type="success" if roi_pct > 100 else "warning"
            )
            
        # Comparison waterfall / bar chart
        fin_df = pd.DataFrame({
            'Category': ['Campaign Cost', 'Annual Revenue Preserved', 'Net Economic Value'],
            'Amount': [total_campaign_cost, annual_revenue_saved, net_profit]
        })
        fig_fin = px.bar(
            fin_df,
            x='Category',
            y='Amount',
            color='Category',
            text='Amount',
            color_discrete_sequence=['#EF4444', '#3B82F6', '#10B981'],
            title="<b>Financial Impact Analysis ($)</b>"
        )
        fig_fin.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_fin.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0F172A", family="Plus Jakarta Sans"),
            height=280,
            showlegend=False
        )
        st.plotly_chart(fig_fin, use_container_width=True)

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.82rem;'>"
    "Retail Banking Customer Churn Analytics Platform &bull; Built with Streamlit, Plotly & Scikit-Learn &bull; "
    "Designed for Executive Decision Support"
    "</div>",
    unsafe_allow_html=True
)
