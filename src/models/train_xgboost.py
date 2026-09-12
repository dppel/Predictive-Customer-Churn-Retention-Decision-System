"""
XGBoost Model Training & Calibration Module
===========================================
Trains XGBoost binary classifier and applies probability calibration (CalibratedClassifierCV)
to deliver accurate churn probability outputs for monetary expected value calculations.
"""

import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from typing import Tuple

def train_calibrated_xgboost(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    random_state: int = 42
) -> Tuple[XGBClassifier, CalibratedClassifierCV]:
    """
    Trains base XGBoost classifier and wraps it in a CalibratedClassifierCV (sigmoid calibration).
    """
    base_xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1
    )
    
    print("[TRAINING] Training base XGBoost model...")
    base_xgb.fit(X_train, y_train)
    
    print("[CALIBRATION] Calibrating churn probabilities with CalibratedClassifierCV (Sigmoid)...")
    calibrated_model = CalibratedClassifierCV(estimator=base_xgb, method='sigmoid', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    return base_xgb, calibrated_model

def save_model(base_model: XGBClassifier, calibrated_model: CalibratedClassifierCV, feature_names: list, filepath: str = "models/xgboost_model.pkl"):
    """
    Saves trained base XGBoost model, calibrated model wrapper, and feature names.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    artifacts = {
        "base_model": base_model,
        "calibrated_model": calibrated_model,
        "feature_names": feature_names
    }
    joblib.dump(artifacts, filepath)
    print(f"[MODEL SAVED] Saved model artifacts to '{filepath}'.")

def load_model(filepath: str = "models/xgboost_model.pkl") -> Dict:
    """
    Loads model artifacts from file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at '{filepath}'. Please train model first.")
    return joblib.load(filepath)
