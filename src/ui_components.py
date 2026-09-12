import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def apply_custom_theme():
    """Apply high-end modern fintech CSS styling with crystal clear contrast."""
    custom_css = """
    <style>
        /* Modern Typography & Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Metric Card Container */
        .kpi-card {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 16px !important;
            padding: 20px 24px !important;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.08) !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            margin-bottom: 16px !important;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15) !important;
            border-color: #CBD5E1 !important;
        }
        .kpi-title {
            font-size: 0.82rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            color: #475569 !important;
            margin-bottom: 6px !important;
        }
        .kpi-value {
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            color: #0F172A !important; /* Crystal clear bold obsidian / dark slate */
            letter-spacing: -0.02em !important;
            line-height: 1.2 !important;
        }
        .kpi-subtitle {
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            color: #64748B !important;
            margin-top: 6px !important;
        }
        .kpi-badge-danger {
            color: #DC2626 !important;
            background: #FEE2E2 !important;
            border: 1px solid #FCA5A5 !important;
            padding: 4px 10px !important;
            border-radius: 6px !important;
            font-weight: 700 !important;
            font-size: 0.78rem !important;
        }
        .kpi-badge-success {
            color: #059669 !important;
            background: #D1FAE5 !important;
            border: 1px solid #6EE7B7 !important;
            padding: 4px 10px !important;
            border-radius: 6px !important;
            font-weight: 700 !important;
            font-size: 0.78rem !important;
        }
        .kpi-badge-warning {
            color: #D97706 !important;
            background: #FEF3C7 !important;
            border: 1px solid #FCD34D !important;
            padding: 4px 10px !important;
            border-radius: 6px !important;
            font-weight: 700 !important;
            font-size: 0.78rem !important;
        }
        .kpi-badge-default {
            color: #4F46E5 !important;
            background: #EEF2FF !important;
            border: 1px solid #C7D2FE !important;
            padding: 4px 10px !important;
            border-radius: 6px !important;
            font-weight: 700 !important;
            font-size: 0.78rem !important;
        }
        
        /* Modern Section Headers */
        .section-header {
            font-size: 1.35rem !important;
            font-weight: 800 !important;
            color: #0F172A !important;
            margin-top: 24px !important;
            margin-bottom: 16px !important;
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
        }
        
        /* Action Item Cards */
        .action-card {
            background: #F8FAFC !important;
            border: 1px solid #E2E8F0 !important;
            border-left: 5px solid #4F46E5 !important;
            padding: 14px 18px !important;
            border-radius: 0 12px 12px 0 !important;
            margin-bottom: 12px !important;
            font-size: 0.95rem !important;
            color: #1E293B !important;
            line-height: 1.5 !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            border-bottom: 2px solid #E2E8F0;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 10px 20px;
            font-weight: 700;
            font-size: 0.95rem;
            color: #475569;
        }
        
        /* Dark mode compatibility */
        @media (prefers-color-scheme: dark) {
            .kpi-card {
                background: #1E293B !important;
                border-color: #334155 !important;
            }
            .kpi-title {
                color: #94A3B8 !important;
            }
            .kpi-value {
                color: #F8FAFC !important;
            }
            .kpi-subtitle {
                color: #94A3B8 !important;
            }
            .section-header {
                color: #F8FAFC !important;
            }
            .action-card {
                background: #0F172A !important;
                border-color: #334155 !important;
                color: #F1F5F9 !important;
            }
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def render_metric_card(title: str, value: str, subtitle: str = "", badge_text: str = "", badge_type: str = "default"):
    """Render a crisp, styled KPI metric card with crystal-clear contrast."""
    badge_html = ""
    if badge_text:
        badge_class = f"kpi-badge-{badge_type}" if badge_type in ["danger", "success", "warning", "default"] else "kpi-badge-default"
        badge_html = f'<span class="{badge_class}">{badge_text}</span>'
        
    html = f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div style="display: flex; justify-content: space-between; align-items: baseline;">
            <div class="kpi-value">{value}</div>
            <div>{badge_html}</div>
        </div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def plot_churn_gauge(probability: float, segment: str, color: str):
    """Render interactive speed gauge for churn probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>Predicted Churn Risk</b><br><span style='font-size:15px; color:{color}; font-weight:800;'>{segment.upper()}</span>", 'font': {'size': 18, 'color': '#0F172A'}},
        number={'suffix': "%", 'font': {'size': 38, 'color': color, 'weight': 800}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748B"},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': "#F1F5F9",
            'borderwidth': 1,
            'bordercolor': "#CBD5E1",
            'steps': [
                {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [35, 56.5], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [56.5, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
            ],
            'threshold': {
                'line': {'color': "#DC2626", 'width': 3},
                'thickness': 0.8,
                'value': 56.5
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#0F172A", 'family': "Plus Jakarta Sans"},
        height=260,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def plot_confusion_matrix_heatmap(cm: list, threshold: float = None):
    """Render interactive confusion matrix with crisp colors and dynamic threshold title."""
    labels = ["Retained (0)", "Churned (1)"]
    z = cm
    z_text = [[str(y) for y in x] for x in z]
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        text=z_text,
        texttemplate="%{text}",
        textfont={"size": 18, "family": "Plus Jakarta Sans", "color": "white", "weight": "bold"},
        colorscale=[[0, "#93C5FD"], [0.5, "#3B82F6"], [1, "#1E3A8A"]],
        showscale=False
    ))
    
    title_text = f"<b>Confusion Matrix (Threshold = {threshold:.2f})</b>" if threshold is not None else "<b>Confusion Matrix (Optimal Threshold)</b>"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Predicted Outcome",
        yaxis_title="Actual Ground Truth",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0F172A"),
        height=320,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_roc_curve(fpr, tpr, auc, current_fpr: float = None, current_tpr: float = None, threshold: float = None):
    """Render ROC Curve with Plotly, crisp contrast, and dynamic operating point marker."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'Random Forest (AUC = {auc:.3f})',
        line=dict(color='#4F46E5', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Chance (AUC = 0.500)',
        line=dict(color='#94A3B8', dash='dash', width=2)
    ))
    
    if current_fpr is not None and current_tpr is not None and threshold is not None:
        fig.add_trace(go.Scatter(
            x=[current_fpr], y=[current_tpr],
            mode='markers+text',
            name=f'Cutoff: {threshold:.2f}',
            marker=dict(size=14, color='#DC2626', line=dict(width=2, color='white')),
            text=[f"Thresh: {threshold:.2f}"],
            textposition="bottom right",
            textfont=dict(size=11, color="#DC2626", family="Plus Jakarta Sans")
        ))
    
    fig.update_layout(
        title=f"<b>ROC Curve — Discriminative Ability</b> (AUC: {auc:.3f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0F172A"),
        legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95),
        height=340,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.06)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.06)')
    return fig

def plot_feature_importance_bar(importance_series: pd.Series, title: str = "<b>Key Predictors of Customer Churn</b>"):
    """Render horizontal bar chart for feature importance."""
    df_imp = importance_series.reset_index()
    df_imp.columns = ['Feature', 'Importance']
    df_imp = df_imp.sort_values('Importance', ascending=True)
    
    fig = px.bar(
        df_imp,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=[(0, "#93C5FD"), (0.5, "#3B82F6"), (1, "#1E3A8A")],
        title=title
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0F172A"),
        coloraxis_showscale=False,
        height=360,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.06)')
    return fig
