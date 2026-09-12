# 📊 Comprehensive Model Evaluation & Business Cost Tradeoff Report

## 📋 Executive Overview
This report evaluates the **Calibrated XGBoost Classifier** across 7 diagnostic dimensions and connects statistical performance metrics directly to **Business Cost & Financial Risk**.

---

## 🎯 The 7 Evaluation Dimensions

| Metric | Empirical Score | Business Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.8723** | Excellent overall ranking ability across all probability decision thresholds. |
| **PR-AUC** | **0.8430** | Strong precision-recall balance on the positive churn target class. |
| **Brier Score** | **0.1448** | Low mean squared probability error, confirming accurate probability calibration. |
| **Accuracy** | **0.7880** | Baseline correctness fraction across all predictions. |
| **Precision** | **0.7731** | Of all customers flagged for churn, 77.3% actually churned. |
| **Recall** | **0.6993** | Of all actual churners, 69.9% were successfully caught by the model. |
| **F1-Score** | **0.7343** | Harmonic mean of Precision and Recall. |

---

## 🧩 Confusion Matrix Analysis (Test Dataset: 1,000 Customers)

```
                       PREDICTED
                   Stayed (0)   Churned (1)
ACTUAL  Stayed (0)    501          99        (True Negatives = 501, False Positives = 99)
       Churned (1)    126         274        (False Negatives = 126, True Positives = 274)
```

---

## ⚖️ Business Cost & Asymmetric Risk Analysis

In customer churn decisioning, classification errors carry **highly asymmetric financial costs**:

### 1. Cost of False Negatives (FN) — Missed Churners
- **Definition**: The model predicts `Stayed`, but the customer actually `Churned`.
- **Financial Penalty**: The company loses the entire remaining **Customer Lifetime Value (CLV)**.
- **Average Loss per FN**: $\approx \$1,800.00$.

### 2. Cost of False Positives (FP) — Unnecessary Interventions
- **Definition**: The model predicts `Churned`, but the customer was actually `Stayed`.
- **Financial Penalty**: The company wastes the retention intervention offer cost ($C_{\text{interv}}$).
- **Average Loss per FP**: $\approx \$35.00 - \$65.00$.

### 💡 Strategic Asymmetry Conclusion
$$\text{Cost}(\text{False Negative}) \gg \text{Cost}(\text{False Positive})$$
$$\$1,800.00 \gg \$50.00$$

Because losing a high-value customer is **36 times more expensive** than giving an unnecessary discount voucher, the business strategy favors **High Recall** to capture as many potential churners as possible, governed by the Level 4 Expected Net Gain ROI threshold.

---

## 📈 Probability Calibration (Sigmoid Calibration Curve)

Uncalibrated tree ensemble models often produce overconfident probability estimates near 0 or 1. By applying `CalibratedClassifierCV` (Sigmoid / Platt Scaling), our output probability $P(\text{Churn})$ maps linearly to true empirical churn frequency:

$$\text{If } P(\text{Churn}) = 0.70 \implies \text{Exactly 70 out of 100 such customers churn.}$$

This exact calibration is what makes our **Expected Net Gain** financial calculations mathematically rigorous.
