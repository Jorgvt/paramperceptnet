import os
import argparse
import pandas as pd
import numpy as np
import scipy.stats as stats

def run_self_consistency(csv_path, k, method='pearson', num_iterations=10000, seed=42):
    """
    Runs the Monte Carlo simulation to estimate the maximum attainable correlation
    (self-consistency) of a subjective image quality database.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing 'mos' and 'std' columns.
    k : float
        Scaling factor applied to the standard deviation (representing the conversion
        from standard error to individual rating standard deviation: k = sqrt(N_evaluations)).
    method : str, optional
        Correlation method to use ('pearson' or 'spearman'). Default is 'pearson'.
    num_iterations : int, optional
        Number of Monte Carlo iterations. Default is 10000.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
        
    Returns:
    --------
    mean_corr : float
        The average correlation coefficient over the Monte Carlo runs.
    std_corr : float
        The standard deviation of the correlation coefficient over the runs.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Database file not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    # KADID uses 'dmos' and 'var' (variance) instead of 'mos' and 'std'
    mos_col = 'dmos' if 'dmos' in df.columns else 'mos'
    mos = df[mos_col].values
    
    if 'var' in df.columns:
        std = np.sqrt(df['var'].values)
    else:
        std = df['std'].values
    
    np.random.seed(seed)
    correlations = []
    
    for _ in range(num_iterations):
        # Sample simulated observer ratings: r_i ~ N(MOS_i, (k * STD_i)^2)
        simulated_observer = np.random.normal(mos, k * std)
        
        if method.lower() == 'pearson':
            corr = np.corrcoef(simulated_observer, mos)[0, 1]
        elif method.lower() == 'spearman':
            df_temp = pd.DataFrame({'sim': simulated_observer, 'mos': mos})
            corr = df_temp.corr(method='spearman').iloc[0, 1]
        else:
            raise ValueError(f"Unknown correlation method: {method}")
            
        correlations.append(corr)
        
    return np.mean(correlations), np.std(correlations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recalculate Database Self-Consistency via Monte Carlo Simulation.")
    parser.add_argument("--database", choices=["tid2008", "tid2013", "kadid"], default="tid2008",
                        help="The database to evaluate.")
    parser.add_argument("--k", type=float, default=None,
                        help="Scaling factor (default: sqrt(33) for tid2008, 6.0 for tid2013, 1.0 for kadid).")
    parser.add_argument("--method", choices=["pearson", "spearman"], default="pearson",
                        help="Correlation metric (default: pearson).")
    parser.add_argument("--iterations", type=int, default=10000,
                        help="Number of Monte Carlo iterations (default: 10000).")
    
    args = parser.parse_args()
    
    # Set default values for k based on literature (sqrt(N) where N is evaluations per image)
    if args.database == "tid2008":
        csv_path = "tid2008_mos_std.csv"
        default_k = np.sqrt(33) # TID2008 has ~33 evaluations per image
    elif args.database == "tid2013":
        csv_path = "tid2013_mos_std.csv"
        default_k = 6.0 # TID2013 Swiss-system results equivalent to ~36 evaluations
    else:
        csv_path = "kadid_dmos.csv"
        default_k = 1.0 # KADID provides rating variance directly, no scaling needed
        
    k = args.k if args.k is not None else default_k
    
    print(f"Running Monte Carlo self-consistency for {args.database.upper()}...")
    print(f"File: {csv_path}")
    print(f"Scaling factor k: {k:.4f} (representing sqrt(N_evaluations))")
    print(f"Correlation method: {args.method}")
    print(f"Iterations: {args.iterations}")
    
    mean_corr, std_corr = run_self_consistency(csv_path, k, method=args.method, num_iterations=args.iterations)
    sem_corr = std_corr / np.sqrt(args.iterations)
    
    print("\n--- RESULTS ---")
    print(f"Maximum Attainable Correlation ({args.method}): {mean_corr:.4f}")
    print(f"  - Observer-sampling Uncertainty (SD): ± {std_corr:.4f}")
    print(f"  - Monte Carlo Standard Error (SEM):   ± {sem_corr:.6f}")
