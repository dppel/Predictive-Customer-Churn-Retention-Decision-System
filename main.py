"""
Main End-to-End Pipeline Execution Script — Senior Executive Edition
======================================================================
Executes the full Predictive Customer Churn & Retention Decision Intelligence System:
Data Generation -> EDA -> Feature Engineering -> Preprocessing -> Model Benchmarking
-> Model Diagnostics & Cost Asymmetry -> SHAP Explainability -> 2D Risk-Value Segmentation
-> Retention ROI Engine -> Monte Carlo Sensitivity (VaR) -> Model Governance (PSI Data Drift).
"""

import os
import argparse
from src.data.generator import generate_customer_dataset
from src.data.loader import load_raw_data
from src.eda.exploratory import run_exploratory_analysis
from src.features.feature_builder import build_features
from src.preprocessing.pipeline import ChurnPreprocessingPipeline
from src.models.train_benchmarks import train_and_benchmark_models
from src.models.train_xgboost import save_model
from src.models.evaluator import evaluate_model
from src.explainability.shap_explainer import ChurnShapExplainer
from src.decision.risk_segmentation import segment_customers
from src.decision.retention_engine import apply_retention_decisions
from src.decision.roi_analysis import generate_roi_portfolio_report
from src.decision.sensitivity_analysis import run_monte_carlo_sensitivity
from src.monitoring.drift_detector import monitor_feature_drift
from src.decision.ab_testing_design import calculate_ab_sample_size

def run_pipeline(samples: int = 5000, seed: int = 42):
    print("=" * 70)
    print("      PREDICTIVE CUSTOMER CHURN & RETENTION DECISION SYSTEM      ")
    print("=" * 70)
    
    # 1. Data Ingestion / Synthesis
    data_path = "data/raw/customer_churn.csv"
    if not os.path.exists(data_path):
        print(f"[STEP 1] Generating raw customer dataset ({samples} samples)...")
        df_raw = generate_customer_dataset(n_samples=samples, seed=seed)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df_raw.to_csv(data_path, index=False)
    else:
        print(f"[STEP 1] Loading raw dataset from '{data_path}'...")
        df_raw = load_raw_data(data_path)
        
    # 2. Exploratory Data Analysis
    print("\n[STEP 2] Running Exploratory Data Analysis (EDA)...")
    eda_summary = run_exploratory_analysis(df_raw, output_dir="figures")
    
    # 3. Feature Engineering (with Business Rationale)
    print("\n[STEP 3] Performing Feature Engineering (with Business Rationale)...")
    df_feat = build_features(df_raw)
    
    # 4. Preprocessing & Split
    print("\n[STEP 4] Preprocessing & Stratified Train-Test Split...")
    pipeline = ChurnPreprocessingPipeline(target_col="Churn", test_size=0.2, random_state=seed)
    X_train, X_test, y_train, y_test, feature_names = pipeline.fit_transform(df_feat)
    pipeline.save("models/preprocessor.pkl")
    
    # 5. ML Model Benchmarking (Logistic Regression -> Random Forest -> XGBoost)
    print("\n[STEP 5] ML Benchmarking: Logistic Regression Baseline vs. Random Forest vs. Calibrated XGBoost...")
    df_benchmark, models_dict = train_and_benchmark_models(X_train, y_train, X_test, y_test, random_state=seed)
    
    calibrated_model = models_dict["Calibrated XGBoost (Production)"]
    base_xgb = calibrated_model.calibrated_classifiers_[0].estimator
    save_model(base_xgb, calibrated_model, feature_names, filepath="models/xgboost_model.pkl")
    
    # 6. Evaluate Production Model Diagnostics & Cost Asymmetry
    print("\n[STEP 6] Evaluating Production XGBoost Model Diagnostics & Financial Cost Asymmetry...")
    eval_metrics = evaluate_model(calibrated_model, X_test, y_test, output_dir="figures")
    
    # 7. SHAP Explainability
    print("\n[STEP 7] Computing SHAP Feature Attributions...")
    explainer = ChurnShapExplainer(base_xgb, feature_names)
    explainer.generate_global_plots(X_test[:500], output_dir="figures")
    
    # 8. Churn Probability Prediction & 2D Risk-Value Segmentation
    print("\n[STEP 8] Scoring Full Dataset & Segmenting Customers (Risk x Value)...")
    X_full = pipeline.transform(df_feat)
    churn_probs = calibrated_model.predict_proba(X_full)[:, 1]
    df_segmented = segment_customers(df_feat, churn_probs)
    
    # 9. Retention Decision Engine (Expected Net Gain)
    print("\n[STEP 9] Executing Retention Decision Engine (Expected Net Gain)...")
    df_decisions = apply_retention_decisions(df_segmented)
    
    # 10. Portfolio ROI Summary & Export
    print("\n[STEP 10] Generating Portfolio ROI Financial Summary...")
    roi_summary = generate_roi_portfolio_report(df_decisions, figures_dir="figures", reports_dir="reports")
    
    # 11. Senior Enterprise Enhancements (Monte Carlo, Data Drift & A/B Design)
    print("\n[STEP 11] Running Senior Enterprise Enhancements (Monte Carlo, PSI Drift & A/B Power)...")
    mc_res = run_monte_carlo_sensitivity(df_decisions, n_simulations=1000, seed=seed)
    drift_res = monitor_feature_drift(df_decisions, df_decisions)
    ab_res = calculate_ab_sample_size(baseline_churn_rate=0.42, expected_churn_reduction=0.05)
    
    print("\n" + "=" * 70)
    print("                     PIPELINE EXECUTION COMPLETE                 ")
    print("=" * 70)
    print(f"-> Total Customers Evaluated: {roi_summary['total_customers']:,}")
    print(f"-> Customers Targeted for Retention: {roi_summary['targeted_customers']:,} ({roi_summary['targeting_rate']:.1%})")
    print(f"-> Total Retention Budget Spent: ${roi_summary['total_intervention_budget_spent']:,.2f}")
    print(f"-> Total Net Profit Saved: ${roi_summary['total_expected_net_profit_saved']:,.2f}")
    print(f"-> Portfolio ROI: {roi_summary['portfolio_roi_percentage']:.1f}%")
    print(f"-> 95% Downside Value-at-Risk (VaR): ${mc_res['var_95_net_gain']:,.2f}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Churn Decision Intelligence Pipeline")
    parser.add_argument("--samples", type=int, default=5000, help="Number of synthetic samples if generating new data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    run_pipeline(samples=args.samples, seed=args.seed)
