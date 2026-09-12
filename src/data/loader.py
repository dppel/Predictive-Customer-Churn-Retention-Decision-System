"""
Data Ingestion & Loader Module
==============================
Loads, verifies, and splits customer churn datasets for preprocessing and model training.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple

def load_raw_data(file_path: str = "data/raw/customer_churn.csv") -> pd.DataFrame:
    """
    Loads raw customer CSV data and performs schema validation.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at '{file_path}'. Please run dataset generator first.")
        
    df = pd.read_csv(path)
    
    required_cols = [
        'CustomerID', 'Age', 'Gender', 'Location', 'Tenure', 'ContractType',
        'CustomerSince', 'MonthlyUsageHours', 'NumTransactions', 'LoginsPerMonth',
        'CallMinutes', 'DataUsageGB', 'MonthlyCharges', 'TotalCharges', 'Revenue',
        'DiscountsReceived', 'SupportCallsCount', 'ComplaintsCount',
        'SatisfactionScore', 'ServiceOutagesReported', 'CLV', 'Churn'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
        
    print(f"[DATA LOADED] Loaded {len(df)} rows and {len(df.columns)} columns from '{file_path}'.")
    return df

def split_features_target(df: pd.DataFrame, target_col: str = "Churn") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits DataFrame into feature matrix X (excluding CustomerID & target) and target vector y.
    """
    X = df.drop(columns=['CustomerID', target_col])
    y = df[target_col]
    return X, y
