"""
Senior Enterprise Enhancement: Monte Carlo Sensitivity Analysis & Financial VaR Engine
======================================================================================
Evaluates portfolio financial resilience under 1,000 simulated market stress scenarios.
Computes 95% Confidence Intervals and Value-at-Risk (VaR) for Net Preserved Profit.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

def run_monte_carlo_sensitivity(
    df_decisions: pd.DataFrame, 
    n_simulations: int = 1000, 
    seed: int = 42
) -> Dict[str, Any]:
    """
    Simulates 1,000 market stress scenarios varying offer acceptance probabilities (+-15%)
    and intervention costs (+-20%) to test financial resilience.
    """
    np.random.seed(seed)
    
    base_net_gain = df_decisions['Expected_Net_Gain'].sum()
    base_cost = df_decisions['Intervention_Cost'].sum()
    
    simulated_net_gains = []
    simulated_rois = []
    
    for i in range(n_simulations):
        p_acc_shock = np.random.uniform(0.85, 1.15)
        cost_shock = np.random.uniform(0.90, 1.20)
        
        sim_cost = (df_decisions['Intervention_Cost'] * cost_shock).sum()
        sim_benefit = (df_decisions['Expected_Benefit'] * p_acc_shock).sum()
        sim_net_gain = sim_benefit - sim_cost
        sim_roi = (sim_net_gain / sim_cost * 100.0) if sim_cost > 0 else 0.0
        
        simulated_net_gains.append(sim_net_gain)
        simulated_rois.append(sim_roi)
        
    sim_net_gains = np.array(simulated_net_gains)
    sim_rois = np.array(simulated_rois)
    
    var_95_net_gain = float(np.percentile(sim_net_gains, 5))
    var_95_roi = float(np.percentile(sim_rois, 5))
    ci_95_lower = float(np.percentile(sim_net_gains, 2.5))
    ci_95_upper = float(np.percentile(sim_net_gains, 97.5))
    
    print("\n" + "=" * 75)
    print("        MONTE CARLO SENSITIVITY ANALYSIS & FINANCIAL VaR (1,000 RUNS)       ")
    print("=" * 75)
    print(f"- Baseline Preserved Net Profit:    ${base_net_gain:,.2f}")
    print(f"- Mean Simulated Net Profit:       ${np.mean(sim_net_gains):,.2f}")
    print(f"- 95% Confidence Interval (Profit): [${ci_95_lower:,.2f} , ${ci_95_upper:,.2f}]")
    print(f"- 95% Downside Value-at-Risk (VaR): ${var_95_net_gain:,.2f} (Worst 5% market condition)")
    print(f"- 95% Downside Portfolio ROI:      {var_95_roi:.1f}%")
    print("=" * 75)
    
    return {
        "base_net_gain": float(base_net_gain),
        "mean_simulated_net_gain": float(np.mean(sim_net_gains)),
        "ci_95_lower": ci_95_lower,
        "ci_95_upper": ci_95_upper,
        "var_95_net_gain": var_95_net_gain,
        "var_95_roi": var_95_roi,
        "simulated_net_gains": sim_net_gains.tolist(),
        "simulated_rois": sim_rois.tolist()
    }
