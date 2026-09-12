"""
SHAP Explainability Module (TreeSHAP) — STAGE 17
=================================================
Computes global and local TreeSHAP feature attributions.
Generates human-readable per-customer churn risk drivers (Global & Individual XAI).
"""

import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any

class ChurnShapExplainer:
    def __init__(self, base_model, feature_names: List[str]):
        """
        Initializes TreeSHAP explainer for XGBoost base estimator.
        """
        self.base_model = base_model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.base_model)
        
    def compute_shap_values(self, X_sample: np.ndarray) -> np.ndarray:
        """
        Computes TreeSHAP values matrix.
        """
        return self.explainer.shap_values(X_sample)

    def generate_global_plots(self, X_sample: np.ndarray, output_dir: str = "figures"):
        """
        Generates and exports Global SHAP Summary Bar and Beeswarm plots.
        """
        os.makedirs(output_dir, exist_ok=True)
        shap_values = self.compute_shap_values(X_sample)
        
        # 1. Global Summary Bar Plot
        fig = plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, plot_type="bar", show=False)
        plt.title("Global SHAP Feature Importance (What drives churn across the base?)", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 2. Global Summary Beeswarm Plot
        fig = plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
        plt.title("Global SHAP Beeswarm Plot (Impact on Churn Probability)", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary_beeswarm.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[SHAP PLOTS SAVED] Global summary plots exported to '{output_dir}/'.")

    def explain_single_customer(self, customer_features: np.ndarray, customer_id: str = "Customer", top_k: int = 5) -> Dict[str, Any]:
        """
        Computes local SHAP attributions for a single customer vector and outputs formatted text summary.
        """
        if customer_features.ndim == 1:
            customer_features = customer_features.reshape(1, -1)
            
        shap_val = self.explainer.shap_values(customer_features)[0]
        base_value = float(self.explainer.expected_value)
        
        contributions = []
        for feat_name, val, s_val in zip(self.feature_names, customer_features[0], shap_val):
            direction = "↑" if s_val > 0 else "↓"
            contributions.append({
                "feature": feat_name,
                "value": float(val),
                "shap_impact": float(s_val),
                "direction": direction
            })
            
        contributions = sorted(contributions, key=lambda x: abs(x["shap_impact"]), reverse=True)
        top_churn_pushers = [c for c in contributions if c["shap_impact"] > 0][:top_k]
        top_retention_anchors = [c for c in contributions if c["shap_impact"] < 0][:top_k]
        
        # Build human-readable formatted string
        summary_lines = [f"=== Individual Explanation: {customer_id} ==="]
        summary_lines.append("Main Drivers:")
        for c in contributions[:top_k]:
            sign = "+" if c["shap_impact"] > 0 else ""
            summary_lines.append(f"  {c['feature']:<28} {c['direction']}  ({sign}{c['shap_impact']:.3f} SHAP, Val={c['value']:.2f})")
            
        formatted_summary = "\n".join(summary_lines)
        
        return {
            "customer_id": customer_id,
            "base_value": base_value,
            "top_churn_drivers": top_churn_pushers,
            "top_retention_anchors": top_retention_anchors,
            "formatted_summary": formatted_summary,
            "all_contributions": contributions
        }
