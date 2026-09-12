# 📖 Data Dictionary — Customer Churn & Decision Intelligence System

## 📋 Overview
This Data Dictionary defines every feature within the **Predictive Customer Churn & Retention Decision System**. It maps technical column names, data types, business definitions, and operational roles in our 4-Level Decision Framework.

---

## 🗂️ Raw Dataset Features (`data/raw/customer_churn.csv`)

| Variable Name | Data Type | Role / Category | Business Meaning & Operational Context |
| :--- | :--- | :--- | :--- |
| **`CustomerID`** | Identifier | Metadata | Unique customer account tracking key (e.g. `CUST-1000`). Excluded from model training. |
| **`Age`** | Numerical (Integer) | Customer Profile | Customer age in years (18–72). Demographic baseline. |
| **`Gender`** | Categorical | Customer Profile | Gender identity (`Male` / `Female`). |
| **`Location`** | Categorical | Customer Profile | Geographic area classification (`Urban` / `Suburban` / `Rural`). |
| **`Tenure`** | Numerical (Integer) | Customer Profile | Relationship duration in active subscription months (1–72). |
| **`ContractType`** | Categorical | Customer Profile | Subscription commitment tier (`Month-to-Month`, `One-Year`, `Two-Year`). Primary structural churn driver. |
| **`CustomerSince`** | Datetime (String) | Customer Profile | Account onboarding date (`YYYY-MM-DD`). Derived from tenure. |
| **`MonthlyUsageHours`** | Numerical (Float) | Behavioral Usage | Active service utilization hours per month. Early indicator of engagement drops. |
| **`NumTransactions`** | Numerical (Integer) | Behavioral Usage | Monthly transactional activity count (invoices, payments, upgrades). |
| **`LoginsPerMonth`** | Numerical (Integer) | Behavioral Usage | Portal & mobile application login frequency per month. |
| **`CallMinutes`** | Numerical (Float) | Behavioral Usage | Monthly voice communication volume in minutes. |
| **`DataUsageGB`** | Numerical (Float) | Behavioral Usage | Monthly data consumption in Gigabytes (GB). |
| **`MonthlyCharges`** | Numerical (Float) | Financial & Value | Recurring monthly subscription charge in USD ($25.00–$209.18). |
| **`TotalCharges`** | Numerical (Float) | Financial & Value | Cumulative historical revenue paid by the customer ($). |
| **`Revenue`** | Numerical (Float) | Financial & Value | Account gross revenue value ($). |
| **`DiscountsReceived`** | Numerical (Float) | Financial & Value | Total promotional credits or discounts applied to account ($). |
| **`SupportCallsCount`** | Numerical (Integer) | Customer Experience | Inbound customer support calls / tickets filed (0–12). Key friction metric. |
| **`ComplaintsCount`** | Numerical (Integer) | Customer Experience | Formal complaints escalated to customer care (0–6). |
| **`SatisfactionScore`** | Numerical (Integer) | Customer Experience | CSAT survey score ($1 = \text{Very Dissatisfied}, 5 = \text{Very Satisfied}$). |
| **`ServiceOutagesReported`**| Numerical (Integer) | Customer Experience | Network or service interruptions experienced by customer (0–5). |
| **`CLV`** | Numerical (Float) | Financial & Value | Estimated remaining 24-month Customer Lifetime Value ($100–$4,016). Essential for Level 4 ROI. |
| **`Churn`** | Binary ($0/1$) | Target Variable | **Target Outcome**: $0 = \text{Stayed}$, $1 = \text{Churned}$ within observation window. |

---

## 🛠️ Engineered Domain Features (`src/features/feature_builder.py`)

| Feature Name | Data Type | Business Formula | Business Meaning & SHAP Driver Purpose |
| :--- | :--- | :--- | :--- |
| **`Tenure_to_Monthly_Ratio`** | Numerical (Float) | $\frac{\text{Tenure}}{\text{MonthlyCharges}}$ | Measures price sensitivity per month of customer loyalty. |
| **`Support_Per_Tenure`** | Numerical (Float) | $\frac{\text{SupportCallsCount}}{\text{Tenure} + 1}$ | Friction density: support call frequency normalized by account age. |
| **`Complaint_Severity_Index`** | Numerical (Integer) | $\text{ComplaintsCount} \times (6 - \text{SatisfactionScore})$ | Interaction metric measuring acute customer dissatisfaction severity. |
| **`Usage_Intensity`** | Numerical (Float) | $\frac{\text{MonthlyUsageHours} \times \text{DataUsageGB}}{100}$ | Combined digital engagement metric. |
| **`Service_Engagement_Score`** | Numerical (Float) | $\frac{\text{CallMinutes}}{500} + \frac{\text{DataGB}}{50} + \frac{\text{Logins}}{20}$ | Multi-channel composite engagement score. |
| **`Contract_Risk_Flag`** | Binary ($0/1$) | $1 \text{ if Contract == 'Month-to-Month' else } 0$ | High-risk commitment flag for quick risk segmentation. |
| **`CLV_to_Monthly_Ratio`** | Numerical (Float) | $\frac{\text{CLV}}{\text{MonthlyCharges}}$ | Measures lifetime value leverage relative to recurring monthly price. |
