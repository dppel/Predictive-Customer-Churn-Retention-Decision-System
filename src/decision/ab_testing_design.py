"""
Senior Enterprise Enhancement: Causal A/B Experimentation & Power Calculator
=============================================================================
Calculates sample sizes, Minimum Detectable Effect (MDE), and statistical power
for production rollouts of retention offers to prove causal impact.
"""

import math
import scipy.stats as stats
from typing import Dict, Any

def calculate_ab_sample_size(
    baseline_churn_rate: float = 0.42,
    expected_churn_reduction: float = 0.05,
    alpha: float = 0.05,
    power: float = 0.80
) -> Dict[str, Any]:
    """
    Calculates required sample size per variant (Control vs Treatment) to statistically prove
    retention intervention causality at 80% Power and 5% Significance level (alpha).
    """
    p1 = baseline_churn_rate
    p2 = baseline_churn_rate - expected_churn_reduction
    p_bar = (p1 + p2) / 2.0
    
    z_alpha = stats.norm.ppf(1 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)
    
    n_per_group = ((z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / ((p1 - p2) ** 2)
    n_per_group = math.ceil(n_per_group)
    
    print("\n" + "=" * 75)
    print("      PRODUCTION A/B TESTING EXPERIMENT DESIGN & POWER ANALYSIS     ")
    print("=" * 75)
    print(f"- Baseline Churn Rate (Control):         {p1:.1%}")
    print(f"- Target Churn Rate (Treatment):         {p2:.1%} (Minimum Detectable Effect = {expected_churn_reduction:.1%})")
    print(f"- Statistical Significance Level (Alpha): {alpha:.2f} (95% Confidence)")
    print(f"- Statistical Power (1 - Beta):          {power:.2f} (80% Power)")
    print(f"- Required Sample Size per Variant:      {n_per_group:,} customers")
    print(f"- Total Experiment Cohort Size:          {n_per_group * 2:,} customers")
    print("=" * 75)
    
    return {
        "baseline_churn_rate": p1,
        "target_churn_rate": p2,
        "mde": expected_churn_reduction,
        "alpha": alpha,
        "power": power,
        "required_sample_size_per_group": n_per_group,
        "total_sample_size": n_per_group * 2
    }
