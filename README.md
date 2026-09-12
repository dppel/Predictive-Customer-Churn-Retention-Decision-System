# Predictive Customer Churn & Retention Decision Intelligence System

## Business Scenario & Objective

A **subscription-based enterprise** (SaaS / Telecom model) seeks to reduce customer churn by identifying customers at high risk of leaving and prioritizing economically valuable retention interventions.

---

## The Primary Business Question Framework

Rather than jumping straight into machine learning code, this system is anchored on the fundamental **Business Question**:

> **"Which customers are most likely to churn, why are they likely to churn, and which customers should the company target with retention interventions to maximize net financial profit?"**

To answer this, our system structures the analytical and modeling workflow across **4 Hierarchical Decision Levels**:

```
                       ┌─────────────────────────┐
                       │    BUSINESS QUESTION    │
                       └────────────┬────────────┘
                                    │
    ┌──────────────────┬────────────┴─────────────┬──────────────────┐
    ▼                  ▼                          ▼                  ▼
┌───────────────┐  ┌───────────────┐      ┌───────────────┐  ┌───────────────┐
│   LEVEL 1     │  │   LEVEL 2     │      │   LEVEL 3     │  │   LEVEL 4     │
│  PREDICTION   │  │ EXPLANATION   │      │   DECISION    │  │   ECONOMICS   │
│  Who will     │  │  Why are they │      │  Should we    │  │  Is it net    │
│   churn?      │  │  at risk?     │      │  intervene?   │  │  profitable?  │
│ (Calibrated   │  │ (TreeSHAP     │      │ (Risk-Value   │  │ (Expected Net │
│   XGBoost)    │  │  Drivers)     │      │   Matrix)     │  │   Gain ROI)   │
└───────────────┘  └───────────────┘      └───────────────┘  └───────────────┘
```

---

## Key Results & Financial ROI

```
======================================================================
                  PORTFOLIO FINANCIAL PERFORMANCE SUMMARY
======================================================================
• Total Customers Evaluated:           5,000
• Model Classification Performance:   ROC-AUC = 0.8716 | PR-AUC = 0.8411
• Customers Targeted for Retention:   4,697 (93.9%)
• Total Intervention Budget Spent:   $439,095.00
• Total Net Financial Profit Saved:   $1,373,179.69
• Portfolio Return on Investment:     312.7% ROI
======================================================================
```

---

## Production Directory Architecture

```
customer-churn-decision-intelligence/
│
├── data/
│   ├── raw/         # Raw customer ingestion datasets
│   └── processed/   # Cleaned, encoded, and engineered dataset features
│
├── docs/
│   └── business_understanding.md # Business scenario, context & 4-Level Framework
│
├── notebooks/       # Exploratory analysis & experimental modeling notebooks
│
├── src/             # Production modular python codebase
│   ├── data/        # Synthetic data generation and loading modules
│   ├── eda/         # Automated EDA and statistical summaries
│   ├── preprocessing/# Pipeline transformers, imputation, and encodings
│   ├── features/    # Custom domain feature engineering
│   ├── models/      # XGBoost training, hyperparameter tuning & calibration
│   ├── explainability/ # SHAP global and local explainer modules
│   └── decision/    # 2D Risk-Value matrix and Economic ROI Optimization Engine
│
├── models/          # Trained model binaries (.pkl)
├── figures/         # Exported visualization plots (ROC, SHAP, Risk Matrix, ROI Curves)
├── reports/         # Executive summaries, ROI evaluation, CSV recommendations & Data Dictionary
│   ├── data_dictionary.md # Comprehensive Data Dictionary (Raw & Engineered features)
│   └── customer_retention_decisions.csv
│
├── main.py          # End-to-end pipeline CLI runner
├── app.py           # Interactive Streamlit Web Dashboard
├── README.md        # System documentation
└── requirements.txt # Environment dependencies
```

---

## How to Run

1. **Execute End-to-End Pipeline**:
   ```bash
   python main.py
   ```

2. **Launch Interactive Web Dashboard**:
   ```bash
   streamlit run app.py
   ```
