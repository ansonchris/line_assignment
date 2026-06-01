import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize

# =========================================================================
# 1. CORE FUNCTIONAL FORMS
# =========================================================================

def revenue_function(L, R_max, gamma):
    """Concave revenue curve tracking diminishing returns."""
    return R_max * (1.0 - np.exp(-gamma * L))

def loss_function(L, theta, phi, delta):
    """Accelerating loss curve capturing tail-risk expansion."""
    return theta * (L ** phi) * np.exp(delta * L)

# =========================================================================
# 2. STAGE 1: EMPIRICAL PARAMETER ESTIMATION
# =========================================================================

def fit_segment_parameters(historical_df):
    """
    Fits continuous risk and revenue curves per score band using Bounded NLS.
    """
    segments = sorted(historical_df['score_band'].unique())
    estimated_params = {}
    
    for s in segments:
        sub_df = historical_df[historical_df['score_band'] == s]
        L_data = sub_df['approved_limit'].values
        rev_data = sub_df['net_revenue'].values
        loss_data = sub_df['total_loss'].values
        
        # Fit Revenue Parameters (R_max > 0, gamma > 0)
        popt_rev, _ = curve_fit(
            revenue_function, L_data, rev_data, 
            p0=[max(rev_data), 0.0001], 
            bounds=(0, np.inf)
        )
        
        # Fit Loss Parameters (theta > 0, phi >= 1.0 for convexity, delta > 0)
        popt_loss, _ = curve_fit(
            loss_function, L_data, loss_data, 
            p0=[0.005, 1.01, 0.00005], 
            bounds=((0, 1.0, 0), np.inf)
        )
        
        estimated_params[s] = {
            'R_max': popt_rev[0], 'gamma': popt_rev[1],
            'theta': popt_loss[0], 'phi': popt_loss[1], 'delta': popt_loss[2]
        }
    return estimated_params

# =========================================================================
# 3. STAGE 2: MULTIVARIATE CONSTRAINED OPTIMIZATION SOLVER
# =========================================================================

def optimize_portfolio_limits(segment_params, portfolio_meta, 
                              m_bounds=(0.5, 4.5), max_limit_growth=0.15):
    """
    Solves the multivariate credit line multiplier allocation simultaneously.
    Enforces monotonicity, limit growth caps, and growth efficiency.
    """
    bands = sorted(list(segment_params.keys()))
    num_bands = len(bands)
    
    # Establish baseline metrics for macro-growth tracking
    base_total_limit = sum(portfolio_meta[s]['cust_count'] * portfolio_meta[s]['avg_income'] * portfolio_meta[s]['base_m'] for s in bands)
    base_total_rev = 0
    for s in bands:
        L_base = portfolio_meta[s]['avg_income'] * portfolio_meta[s]['base_m']
        base_total_rev += portfolio_meta[s]['cust_count'] * revenue_function(L_base, segment_params[s]['R_max'], segment_params[s]['gamma'])

    # --- Global Loss Objective: Minimize Negative Total Portfolio Profit ---
    def objective(m_vector):
        total_neg_profit = 0
        for i, s in enumerate(bands):
            L = portfolio_meta[s]['avg_income'] * m_vector[i]
            n = portfolio_meta[s]['cust_count']
            p = segment_params[s]
            
            rev = revenue_function(L, p['R_max'], p['gamma'])
            loss = loss_function(L, p['theta'], p['phi'], p['delta'])
            total_neg_profit -= n * (rev - loss)
        return total_neg_profit

    # --- Constraints Definition Array ---
    constraints = []
    
    # Constraint 1: Cross-Segment Monotonicity Loop (m[i+1] - m[i] >= 0)
    for i in range(num_bands - 1):
        constraints.append({
            'type': 'ineq', 
            'fun': lambda m_v, idx=i: m_v[idx + 1] - m_v[idx]
        })
        
    # Constraint 2: Portfolio Total Limit Cap (Allowed Cap - New Limit >= 0)
    def limit_cap_constraint(m_vector):
        new_total_limit = sum(portfolio_meta[s]['cust_count'] * portfolio_meta[s]['avg_income'] * m_vector[k] for k, s in enumerate(bands))
        return (base_total_limit * (1.0 + max_limit_growth)) - new_total_limit
    constraints.append({'type': 'ineq', 'fun': limit_cap_constraint})

    # Constraint 3: Growth Efficiency (% Rev Change >= % Limit Change)
    def efficiency_constraint(m_vector):
        new_total_limit = sum(portfolio_meta[s]['cust_count'] * portfolio_meta[s]['avg_income'] * m_vector[k] for k, s in enumerate(bands))
        new_total_rev = sum(portfolio_meta[s]['cust_count'] * revenue_function(portfolio_meta[s]['avg_income'] * m_vector[k], segment_params[s]['R_max'], segment_params[s]['gamma']) for k, s in enumerate(bands))
        
        pct_limit_growth = (new_total_limit - base_total_limit) / base_total_limit
        pct_rev_growth = (new_total_rev - base_total_rev) / base_total_rev
        return pct_rev_growth - pct_limit_growth
    constraints.append({'type': 'ineq', 'fun': efficiency_constraint})

    bounds = [m_bounds for _ in range(num_bands)]
    x0 = np.linspace(m_bounds[0], (m_bounds[0] + m_bounds[1]) / 2, num_bands)
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'ftol': 1e-9})
    return {bands[i]: result.x[i] for i in range(num_bands)}

# =========================================================================
# 4. VISUALIZATION ENGINE (SEPARATE PLOTS)
# =========================================================================

def generate_separated_plots(segment_params, portfolio_meta, optimal_multipliers, m_bounds=(0.5, 4.5)):
    """
    Simulates thousands of multiplier configurations to build the feasible domain
    and overlay the optimal portfolio point.
    """
    bands = sorted(list(segment_params.keys()))
    simulations = 3000
    results = []

    base_total_limit = sum(portfolio_meta[s]['cust_count'] * portfolio_meta[s]['avg_income'] * portfolio_meta[s]['base_m'] for s in bands)
    base_total_rev = sum(portfolio_meta[s]['cust_count'] * revenue_function(portfolio_meta[s]['avg_income'] * portfolio_meta[s]['base_m'], segment_params[s]['R_max'], segment_params[s]['gamma']) for s in bands)

    for _ in range(simulations):
        m_sample = sorted(np.random.uniform(m_bounds[0], m_bounds[1], len(bands)))
        tot_lim, tot_rev, tot_loss = 0, 0, 0
        
        for i, s in enumerate(bands):
            L = portfolio_meta[s]['avg_income'] * m_sample[i]
            n = portfolio_meta[s]['cust_count']
            tot_lim += n * L
            tot_rev += n * revenue_function(L, segment_params[s]['R_max'], segment_params[s]['gamma'])
            tot_loss += n * loss_function(L, segment_params[s]['theta'], segment_params[s]['phi'], segment_params[s]['delta'])
            
        pct_l_growth = (tot_lim - base_total_limit) / base_total_limit
        pct_r_growth = (tot_rev - base_total_rev) / base_total_rev
        
        results.append({
            'total_limit': tot_lim, 'total_revenue': tot_rev,
            'total_profit': tot_rev - tot_loss, 'efficient': pct_r_growth >= pct_l_growth
        })

    df_sim = pd.DataFrame(results)
    
    # Calculate exact coordinates for the optimized portfolio target point
    opt_lim, opt_rev, opt_loss = 0, 0, 0
    for s in bands:
        L = portfolio_meta[s]['avg_income'] * optimal_multipliers[s]
        n = portfolio_meta[s]['cust_count']
        opt_lim += n * L
        opt_rev += n * revenue_function(L, segment_params[s]['R_max'], segment_params[s]['gamma'])
        opt_loss += n * loss_function(L, segment_params[s]['theta'], segment_params[s]['phi'], segment_params[s]['delta'])

    # --- PLOT 1: REVENUE PLOT ---
    plt.figure(figsize=(9, 6))
    plt.scatter(df_sim[df_sim['efficient']]['total_limit'], df_sim[df_sim['efficient']]['total_revenue'], 
                color='mediumseagreen', alpha=0.4, label='Feasible (Rev Growth >= Limit Growth)')
    plt.scatter(df_sim[~df_sim['efficient']]['total_limit'], df_sim[~df_sim['efficient']]['total_revenue'], 
                color='lightcoral', alpha=0.15, label='Infeasible Region')
    plt.scatter(opt_lim, opt_rev, color='darkblue', edgecolor='black', s=180, marker='*', zorder=5, label='Optimal Allocation Target')
    
    plt.xlabel('total limit')
    plt.ylabel('total revenue')
    plt.title('Portfolio Revenue Distribution Space')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    # --- PLOT 2: PROFIT PLOT ---
    plt.figure(figsize=(9, 6))
    feasible_df = df_sim[df_sim['efficient']].sort_values(by='total_limit')
    frontier_x = feasible_df['total_limit'].values
    frontier_y = feasible_df['total_profit'].cummax().values

    plt.scatter(df_sim['total_limit'], df_sim['total_profit'], c=df_sim['total_profit'], cmap='plasma', alpha=0.25)
    plt.plot(frontier_x, frontier_y, color='crimson', linewidth=3, linestyle='-', label='Efficient Frontier Edge')
    plt.scatter(opt_lim, opt_rev - opt_loss, color='gold', edgecolor='black', s=220, marker='*', zorder=5, label='Max Profit Apex')
    
    plt.xlabel('total_limit')
    plt.ylabel('total profit')
    plt.title('Portfolio Profit Efficient Frontier Curve')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

# =========================================================================
# 5. PIPELINE EXECUTION
# =========================================================================
if __name__ == "__main__":
    np.random.seed(42)
    
    # 1. Simulate calibration parameters and historical transaction databases
    true_profiles = {
        '1_LowScore':  {'R': 700,  'g': 0.00015, 't': 0.008, 'p': 1.04, 'd': 0.00012},
        '2_MedScore':  {'R': 1100, 'g': 0.00012, 't': 0.004, 'p': 1.01, 'd': 0.00006},
        '3_HighScore': {'R': 1800, 'g': 0.00009, 't': 0.001, 'p': 1.00, 'd': 0.00001}
    }
    
    mock_records = []
    for band, p in true_profiles.items():
        for L in np.linspace(1000, 30000, 60):
            rev_obs = revenue_function(L, p['R'], p['g']) + np.random.normal(0, 5)
            loss_obs = loss_function(L, p['t'], p['p'], p['d']) + np.random.normal(0, 2)
            mock_records.append({
                'score_band': band, 'approved_limit': max(L, 0),
                'net_revenue': max(rev_obs, 0), 'total_loss': max(loss_obs, 0)
            })
            
    df_historical_data = pd.DataFrame(mock_records)
    
    # Run Complete Model Flow
    fitted_segment_params = fit_segment_parameters(df_historical_data)
    
    portfolio_meta = {
        '1_LowScore':  {'cust_count': 12000, 'avg_income': 4000, 'base_m': 1.2},
        '2_MedScore':  {'cust_count': 25000, 'avg_income': 5500, 'base_m': 2.0},
        '3_HighScore': {'cust_count': 18000, 'avg_income': 8000, 'base_m': 3.0}
    }
    
    optimal_multipliers = optimize_portfolio_limits(
        segment_params=fitted_segment_params, portfolio_meta=portfolio_meta,
        m_bounds=(0.8, 4.2), max_limit_growth=0.15
    )
    
    # Call the plot window functions
    generate_separated_plots(fitted_segment_params, portfolio_meta, optimal_multipliers, m_bounds=(0.8, 4.2))