import os
from pathlib import Path
import pandas as pd
import numpy as np

def find_data_file() -> Path:
    """Locate Churn_Modelling.xlsx across standard repository locations."""
    candidates = [
        Path("data/Churn_Modelling.xlsx"),
        Path("customer-churn-analytics-main/data/Churn_Modelling.xlsx"),
        Path(__file__).resolve().parent.parent / "data" / "Churn_Modelling.xlsx",
        Path(__file__).resolve().parent.parent / "customer-churn-analytics-main" / "data" / "Churn_Modelling.xlsx",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not locate Churn_Modelling.xlsx. Ensure it is in the data/ directory.")

def load_and_preprocess_data() -> pd.DataFrame:
    """
    Load raw customer churn dataset and apply feature engineering
    compatible with both Power BI dashboards and modern ML models.
    """
    data_path = find_data_file()
    df = pd.read_excel(data_path)
    
    # Feature Engineering for Customer Segments
    # 1. Age Groups
    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[0, 30, 50, 120],
        labels=['Young (<=30)', 'Middle Age (31-50)', 'Senior (51+)'],
        right=True
    )
    
    # 2. Credit Score Categories
    df['CreditScoreCategory'] = pd.cut(
        df['CreditScore'],
        bins=[0, 579, 669, 739, 850],
        labels=['Poor (<580)', 'Average (580-669)', 'Good (670-739)', 'Excellent (740-850)'],
        right=True
    )
    
    # 3. Balance Categories
    # Zero balance vs Low vs Medium vs High
    balance_bins = [-1, 1, 50000, 100000, float('inf')]
    balance_labels = ['Zero Balance', 'Low (<$50k)', 'Medium ($50k-$100k)', 'High ($100k+)']
    df['BalanceCategory'] = pd.cut(
        df['Balance'],
        bins=balance_bins,
        labels=balance_labels,
        right=True
    )

    # 4. Financial & Behavioral Ratios
    # Balance-to-Salary Ratio
    df['BalanceToSalaryRatio'] = (df['Balance'] / (df['EstimatedSalary'] + 1)).round(3)
    
    # Products per tenure year (engagement density)
    df['ProductsPerTenure'] = (df['NumOfProducts'] / (df['Tenure'] + 1)).round(3)
    
    # High Value Customer Indicator
    df['IsHighValue'] = ((df['Balance'] >= 100000) & (df['CreditScore'] >= 650)).astype(int)

    return df

def get_feature_matrix(df: pd.DataFrame):
    """
    Prepare feature matrix X and target y for model training/inference.
    Returns: X, y, feature_names
    """
    features = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ]
    
    # Create clean working copy
    data = df[features].copy()
    
    # Categorical encoding
    data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1}).fillna(0)
    # One-hot encode Geography with consistent columns
    geo_dummies = pd.get_dummies(data['Geography'], prefix='Geo', drop_first=False, dtype=int)
    data = pd.concat([data.drop('Geography', axis=1), geo_dummies], axis=1)
    
    # Ensure standard geo columns exist
    for col in ['Geo_France', 'Geo_Germany', 'Geo_Spain']:
        if col not in data.columns:
            data[col] = 0
            
    # Feature ordering
    final_cols = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'Geo_France', 'Geo_Germany', 'Geo_Spain'
    ]
    
    X = data[final_cols]
    y = df['Exited'] if 'Exited' in df.columns else None
    
    return X, y, final_cols
