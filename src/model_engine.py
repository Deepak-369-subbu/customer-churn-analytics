import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.model_selection import train_test_split

MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "rf_churn_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.joblib"

HIGH_RISK_THRESHOLD = 0.565
MEDIUM_RISK_THRESHOLD = 0.350

def get_risk_segment(prob: float) -> str:
    """Classify churn risk probability into business segments."""
    if prob >= HIGH_RISK_THRESHOLD:
        return "High Risk"
    elif prob >= MEDIUM_RISK_THRESHOLD:
        return "Medium Risk"
    else:
        return "Low Risk"

def get_risk_color(segment: str) -> str:
    """Color codes for risk badges."""
    colors = {
        "High Risk": "#FF4B4B",
        "Medium Risk": "#FFAA00",
        "Low Risk": "#00C49F"
    }
    return colors.get(segment, "#888888")

def train_or_load_model(X: pd.DataFrame, y: pd.Series, force_retrain: bool = False):
    """
    Train Random Forest classifier with balanced weights or load from cache.
    Returns: model, metrics_dict, X_test, y_test, y_probs, feature_importances
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    if not force_retrain and MODEL_PATH.exists() and METRICS_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            model = bundle["model"]
            feature_names = bundle["feature_names"]
            y_probs = model.predict_proba(X_test)[:, 1]
            metrics = joblib.load(METRICS_PATH)
            
            importances = pd.Series(
                model.feature_importances_,
                index=feature_names
            ).sort_values(ascending=False)
            
            return model, metrics, X_test, y_test, y_probs, importances
        except Exception:
            pass # Fallback to retraining
            
    # Train Tuned Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=9,
        min_samples_split=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    y_probs = rf_model.predict_proba(X_test)[:, 1]
    
    # Compute optimal threshold based on F1
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    
    y_pred_default = (y_probs >= 0.5).astype(int)
    y_pred_opt = (y_probs >= best_threshold).astype(int)
    
    # Metrics
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred_default),
        "opt_accuracy": accuracy_score(y_test, y_pred_opt),
        "precision": precision_score(y_test, y_pred_opt, zero_division=0),
        "recall": recall_score(y_test, y_pred_opt, zero_division=0),
        "f1": f1_score(y_test, y_pred_opt, zero_division=0),
        "auc": auc_score,
        "best_threshold": best_threshold,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precisions": precision.tolist(),
        "recalls": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "confusion_matrix": confusion_matrix(y_test, y_pred_opt).tolist()
    }
    
    # Feature importances
    importances = pd.Series(
        rf_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)
    
    # Cache
    joblib.dump({"model": rf_model, "feature_names": X.columns.tolist()}, MODEL_PATH)
    joblib.dump(metrics, METRICS_PATH)
    
    return rf_model, metrics, X_test, y_test, y_probs, importances

def predict_single_customer(model, customer_data: dict, feature_names: list) -> dict:
    """
    Generate churn prediction, segment, and actionable recommendations for a single customer.
    """
    # Build dataframe matching feature names
    row = {
        'CreditScore': customer_data.get('CreditScore', 650),
        'Gender': 1 if customer_data.get('Gender') == 'Male' else 0,
        'Age': customer_data.get('Age', 38),
        'Tenure': customer_data.get('Tenure', 5),
        'Balance': customer_data.get('Balance', 50000.0),
        'NumOfProducts': customer_data.get('NumOfProducts', 1),
        'HasCrCard': int(customer_data.get('HasCrCard', 1)),
        'IsActiveMember': int(customer_data.get('IsActiveMember', 1)),
        'EstimatedSalary': customer_data.get('EstimatedSalary', 100000.0),
        'Geo_France': 1 if customer_data.get('Geography') == 'France' else 0,
        'Geo_Germany': 1 if customer_data.get('Geography') == 'Germany' else 0,
        'Geo_Spain': 1 if customer_data.get('Geography') == 'Spain' else 0,
    }
    
    df_row = pd.DataFrame([row])[feature_names]
    prob = float(model.predict_proba(df_row)[0, 1])
    segment = get_risk_segment(prob)
    
    # Generate business retention actions tailored to this profile
    actions = []
    
    if customer_data.get('Geography') == 'Germany':
        actions.append("🇩🇪 **German Market Retention Strategy**: German clients exhibit 2x higher churn. Offer localized premium branch advisory & competitive savings yields.")
        
    if customer_data.get('Age', 38) >= 45:
        actions.append("🛡️ **Wealth Protection & Retirement Advisory**: Customers over 45 value wealth preservation. Offer retirement planning, family trusts, or wealth advisory.")
        
    if customer_data.get('NumOfProducts', 1) == 1:
        actions.append("📦 **Cross-Selling / Bundling**: Single-product customers churn at 27%+. Provide an incentive (cashback or zero fee) on a second product (credit card or mortgage).")
    elif customer_data.get('NumOfProducts', 1) >= 3:
        actions.append("⚠️ **Product Friction Review**: Customers with 3+ products show abnormally high churn. Audit service experience, app usability, and consolidated fee transparency.")
        
    if customer_data.get('IsActiveMember', 1) == 0:
        actions.append("⚡ **Re-engagement Campaign**: Customer is currently inactive. Launch targeted push notifications, digital banking onboarding, or automated bill payment bonus.")
        
    if customer_data.get('Balance', 0) > 100000:
        actions.append("💎 **High-Balance VIP Concierge**: Over $100k balance at risk. Assign dedicated relationship manager and offer preferred high-yield term deposits.")
        
    if customer_data.get('CreditScore', 650) < 600:
        actions.append("📈 **Credit Enhancement Program**: Offer financial wellness tools, automated score monitoring, and tailored debt consolidation rates.")

    if not actions:
        actions.append("✅ **Standard Engagement**: Customer is well-positioned. Maintain standard loyalty rewards and quarterly check-ins.")
        
    return {
        "churn_probability": prob,
        "segment": segment,
        "color": get_risk_color(segment),
        "actions": actions
    }
