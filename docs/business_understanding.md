# Business Understanding & Business Scenario Definition

## Business Scenario & Context

### Target Enterprise Context
A **subscription-based service company** (operating on a recurring revenue SaaS / Telecom model) seeks to proactively reduce customer churn. The business faces a common subscription challenge: acquiring new customers is significantly more expensive than retaining existing ones, but indiscriminate retention marketing wastes valuable budget on low-risk or low-value customers.

### Primary Objective
Identify customers at high risk of leaving, understand the root causes driving dissatisfaction, and prioritize **economically optimal retention interventions** that maximize net financial return.

---

## The Primary Business Question
Before building algorithms or training models, this system poses the single critical question every subscription business must answer:

> **"Which customers are most likely to churn, why are they likely to churn, and which customers should the company target with retention interventions to maximize net financial profit?"**

To solve this core business problem, our decision intelligence system breaks the question down into **4 Hierarchical Decision Levels**:

```
                 ┌──────────────────────────────────────┐
                 │       PRIMARY BUSINESS QUESTION       │
                 └──────────────────┬───────────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
 ┌───────────────────┐                             ┌───────────────────┐
 │ LEVEL 1: PREDICT  │                             │ LEVEL 2: EXPLAIN  │
 │  Who is likely    │                             │  Why is the       │
 │    to churn?      │                             │ customer churn-   │
 │   P(Churn)        │                             │    prone? SHAP    │
 └─────────┬─────────┘                             └─────────┬─────────┘
           │                                                 │
           └────────────────────────┬────────────────────────┘
                                    ▼
                         ┌────────────────────┐
                         │ LEVEL 3: DECISION  │
                         │ Should the company │
                         │    intervene?      │
                         └──────────┬─────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │ LEVEL 4: ECONOMICS │
                         │ Is the intervention│
                         │    financially     │
                         │    worthwhile?     │
                         └────────────────────┘
```

---

## Breakdown of the 4 Hierarchical Levels

### Level 1 — Prediction: *Who is likely to churn?*
- **Objective**: Accurately estimate the individual probability of churn $P(\text{Churn}_i) \in [0, 1]$ for every customer $i$.
- **Methodology**: Calibrated XGBoost Machine Learning classifier with probability calibration (`CalibratedClassifierCV`) to avoid overconfident / uncalibrated probability scores.
- **Key Output**: Calibrated Churn Probability $P(\text{Churn})$.

---

### Level 2 — Explanation: *Why is the customer likely to churn?*
- **Objective**: Identify the root causes of dissatisfaction driving an individual customer's churn risk.
- **Methodology**: TreeSHAP (SHapley Additive exPlanations) feature attributions.
- **Key Output**: 
  - **Churn Pushers**: Specific friction points (e.g., $+0.42$ SHAP impact from 5 support complaints).
  - **Retention Anchors**: Loyalty drivers (e.g., $-0.35$ SHAP impact from a 2-year contract).

---

### Level 3 — Decision: *Should the company intervene?*
- **Objective**: Move beyond passive risk scoring to prescribe concrete, targeted retention actions.
- **Methodology**: 2D Risk-Value Matrix (Churn Risk Tier $\times$ Customer Lifetime Value Tier) combined with candidate intervention evaluation:
  - `NO_ACTION`
  - `DISCOUNT_VOUCHER` (15% off for 3 months)
  - `FEATURE_UPGRADE` (Free speed & tier boost)
  - `DEDICATED_VIP` (Dedicated account manager)
- **Key Output**: Recommended Action per customer.

---

### Level 4 — Economics: *Is the intervention financially worthwhile?*
- **Objective**: Ensure that retention interventions generate positive net monetary ROI rather than burning marketing budget.
- **Methodology**: Expected Net Gain Equation:
  $$\text{Expected Net Gain}(i, a) = P(\text{Churn}_i) \times P(\text{Acceptance}_{i,a}) \times \text{CLV}_i - \text{Cost}(a)$$
- **Key Output**: 
  - Individual Net Financial Profit ($).
  - Portfolio Preserved Value & ROI % ($\frac{\text{Net Saved}}{\text{Budget Spent}} \times 100\%$).
