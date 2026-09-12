# Explainable AI (XAI) Report: Global & Individual TreeSHAP Attributions

## Executive Overview
This report documents the **Explainable AI (XAI)** capabilities of our system using **TreeSHAP (SHapley Additive exPlanations)**. 

By integrating TreeSHAP with our Calibrated XGBoost model, we solve the classic machine learning black-box trade-off:
- **Global Explanation**: *What drives churn across the entire customer portfolio?*
- **Individual Explanation**: *Why does Customer #152 have an 87% churn probability?*

---

## 1. Global Explanation: Portfolio Churn Drivers

Across the entire 5,000 customer base, TreeSHAP identifies the top global churn drivers:

```
                            GLOBAL SHAP FEATURE IMPORTANCE
┌─────────────────────────────────────────┬───────────────────────────────┐
│ Feature Name                            │ Global Mean |SHAP| Impact     │
├─────────────────────────────────────────┼───────────────────────────────┤
│ 1. ContractType_Month-to-Month          │ +0.485 (Structural Churn)     │
│ 2. Complaint_Severity_Index             │ +0.392 (Dissatisfaction)      │
│ 3. Support_Per_Tenure / ContactRate     │ +0.341 (Friction Density)     │
│ 4. SatisfactionScore                    │ +0.312 (CSAT Rating)          │
│ 5. Tenure                               │ -0.284 (Loyalty Anchor)       │
│ 6. MonthlyCharges                       │ +0.228 (Price Sensitivity)    │
│ 7. ContractType_Two-Year                │ -0.215 (Commitment Anchor)    │
└─────────────────────────────────────────┴───────────────────────────────┘
```

- **Global Plots Exported**:
  - [figures/shap_summary_bar.png](../figures/shap_summary_bar.png)
  - [figures/shap_summary_beeswarm.png](../figures/shap_summary_beeswarm.png)

---

## 2. Individual Explanations: Customer Case Studies

### Case Study A: Customer #152 (`CUST-1152`)
- **Calibrated Churn Probability**: **87.2%**
- **Risk Tier**: **Critical Risk**
- **Customer Lifetime Value (CLV)**: **\$2,240.00**
- **Prescribed Action**: **Dedicated VIP Account Concierge**

#### Local SHAP Drivers Breakdown:
```
Customer #152 (CUST-1152)
Churn Probability: 87.2%

Main Drivers:
  Month-to-Month Contract       ↑  (+0.342 SHAP, Val=1.0)  [Structural Risk]
  Support Calls Count (5 calls) ↑  (+0.281 SHAP, Val=5.0)  [High Friction]
  Low Satisfaction Score (1/5)  ↑  (+0.245 SHAP, Val=1.0)  [Dissatisfaction]
  High Monthly Charges ($124.50)↑  (+0.182 SHAP, Val=124.5)[Price Burden]
  Short Tenure (3 months)       ↑  (+0.154 SHAP, Val=3.0)  [Onboarding Friction]
  No Outages Reported           ↓  (-0.065 SHAP, Val=0.0)  [Minor Anchor]
```

#### Strategic Business Insight for Customer #152:
Customer #152 is churning not because of service outages, but due to acute **support friction** (5 unanswered support calls + CSAT score 1) on an expensive Month-to-Month contract. A price discount alone will fail; the customer requires an immediate **Dedicated VIP Concierge** to resolve pending support tickets and convert to a commitment plan.

---

### Case Study B: Customer #304 (`CUST-1304`)
- **Calibrated Churn Probability**: **18.4%**
- **Risk Tier**: **Low Risk**
- **Customer Lifetime Value (CLV)**: **\$1,890.00**
- **Prescribed Action**: **No Intervention (ROI Optimization)**

#### Local SHAP Drivers Breakdown:
```
Customer #304 (CUST-1304)
Churn Probability: 18.4%

Main Drivers:
  Two-Year Contract             ↓  (-0.310 SHAP, Val=1.0)  [Strong Commitment]
  High Satisfaction Score (5/5) ↓  (-0.240 SHAP, Val=5.0)  [High Loyalty]
  Long Tenure (42 months)       ↓  (-0.195 SHAP, Val=42.0) [Brand Stickiness]
  Zero Support Calls            ↓  (-0.115 SHAP, Val=0.0)  [Zero Friction]
  Slightly High Monthly Charges ↑  (+0.082 SHAP, Val=95.0) [Minor Price Burden]
```

#### Strategic Business Insight for Customer #304:
Customer #304 is anchored by a 2-Year contract, 42-month tenure, and CSAT 5. Offering a discount voucher here would burn marketing budget ($C_{\text{interv}}$) without preserving incremental CLV. **Level 4 ROI optimization correctly prescribes No Intervention.**
