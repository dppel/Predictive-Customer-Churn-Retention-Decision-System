"""
Preprocessing Pipeline Module
=============================
Handles missing value imputation, categorical encoding, feature scaling,
and stratified train-test splitting.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Tuple, Dict, Any

class ChurnPreprocessingPipeline:
    def __init__(self, target_col: str = "Churn", test_size: float = 0.2, random_state: int = 42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None
        self.feature_names = []
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Fits preprocessor on df, transforms features, and performs stratified train-test split.
        """
        df_clean = df.copy()
        
        # Drop ID and non-predictive date columns
        cols_to_drop = ['CustomerID', 'CustomerSince', self.target_col]
        feature_df = df_clean.drop(columns=[col for col in cols_to_drop if col in df_clean.columns])
        y = df_clean[self.target_col].values
        
        # Categorical and Numerical Columns
        categorical_cols = feature_df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ]
        )
        
        X_processed = self.preprocessor.fit_transform(feature_df)
        
        # Get feature names after OneHotEncoding
        cat_encoder = self.preprocessor.named_transformers_['cat']
        encoded_cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist() if categorical_cols else []
        self.feature_names = numerical_cols + encoded_cat_names
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test, self.feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms new data using fitted preprocessor.
        """
        if self.preprocessor is None:
            raise ValueError("Pipeline is not fitted yet. Call fit_transform first.")
            
        cols_to_drop = ['CustomerID', 'CustomerSince', self.target_col]
        feature_df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
        return self.preprocessor.transform(feature_df)
        
    def save(self, filepath: str = "models/preprocessor.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"preprocessor": self.preprocessor, "feature_names": self.feature_names}, filepath)
        print(f"[PREPROCESSOR SAVED] Saved to '{filepath}'.")
        
    @classmethod
    def load(cls, filepath: str = "models/preprocessor.pkl"):
        data = joblib.load(filepath)
        instance = cls()
        instance.preprocessor = data["preprocessor"]
        instance.feature_names = data["feature_names"]
        return instance
