"""
Model Evaluation & Diagnostics Module — STAGE 16
================================================
Computes ROC-AUC, PR-AUC, Confusion Matrix, Classification Metrics, Brier Calibration Score,
and calculates the Asymmetric Financial Cost of False Negatives vs. False Positives.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, roc_curve,
    classification_report, confusion_matrix, brier_score_loss, log_loss,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.calibration import calibration_curve

def evaluate_model(
    model, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    output_dir: str = "figures",
    avg_clv: float = 1800.0,
    avg_offer_cost: float = 50.0
) -> dict:
    """
    Evaluates classifier predictions across 7 diagnostic dimensions and computes financial cost asymmetry.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Probabilities & Binary Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Calculate Metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_vals, precision_vals)
    brier = brier_score_loss(y_test, y_prob)
    loss = log_loss(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    # Financial Cost Asymmetry Analysis
    total_fn_cost = fn * avg_clv
    total_fp_cost = fp * avg_offer_cost
    
    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "brier_score": float(brier),
        "log_loss": float(loss),
        "confusion_matrix": cm.tolist(),
        "fn_count": int(fn),
        "fp_count": int(fp),
        "total_fn_cost": float(total_fn_cost),
        "total_fp_cost": float(total_fp_cost)
    }
    
    # 1. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#2980b9', lw=2.5, label=f'Calibrated XGBoost (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Chance')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
    plt.close(fig)
    
    # 2. Probability Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker='o', color='#8e44ad', lw=2, label='Calibrated XGBoost')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Perfect Calibration')
    ax.set_title('Probability Calibration Curve', fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Empirical Churn Fraction')
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "calibration_curve.png"), dpi=300)
    plt.close(fig)
    
    # 3. Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=['Stayed', 'Churned'], yticklabels=['Stayed', 'Churned'])
    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close(fig)
    
    print(f"[EVALUATION] ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f} | Brier: {brier:.4f}")
    print(f"[FINANCIAL ASYMMETRY] False Negative Cost (Lost CLV): ${total_fn_cost:,.2f} vs False Positive Cost (Offer Waste): ${total_fp_cost:,.2f}")
    return metrics
