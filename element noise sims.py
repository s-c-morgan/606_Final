#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 19:38:19 2025

@author: scmorgan
"""

#!/usr/bin/env python3
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.utils import assert_all_finite
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.linalg import eigh
import traceback

from WeightedSBM import WeightedSBM, generate_noisy_sbm
from rpca import RobustPCA
from op import OutlierPursuit

# Constants
N_SIMULATIONS = 10
BASE_SEED = 42

def get_top_eigenvectors(matrix, k):
    matrix = np.nan_to_num((matrix + matrix.T) / 2)
    matrix += 1e-8 * np.eye(matrix.shape[0])
    try:
        eigenvalues, eigenvectors = eigh(matrix)
    except np.linalg.LinAlgError:
        matrix = np.nan_to_num(matrix)
        eigenvalues, eigenvectors = eigh(matrix)
    return eigenvectors[:, np.argsort(eigenvalues)[::-1][:k]]

def cluster_and_evaluate(features, true_labels, seed):
    labels = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(features)
    return {"NMI": normalized_mutual_info_score(true_labels, labels)}

def generate_sbm(a, b, sigma, seed):
    rng = np.random.default_rng(seed)
    sbm = WeightedSBM(
        n_nodes=100, n_communities=2, p_within=1.0, p_between=1.0,
        weight_dist_within=lambda: rng.normal(a, sigma),
        weight_dist_between=lambda: rng.normal(b, sigma)
    )
    return sbm, sbm.generate_graph()

def prepare_graph_matrix(G_noisy):
    adj_matrix = np.maximum(nx.to_numpy_array(G_noisy), 0)
    adj_matrix = np.nan_to_num(adj_matrix)
    assert_all_finite(adj_matrix)
    return adj_matrix

def rpca_method(adj_matrix, true_labels, seed):
    try:
        model = RobustPCA(max_iter=100, tol=1e-4)
        model.fit(adj_matrix)
        low_rank = model.get_low_rank()
        eigenvectors = get_top_eigenvectors(low_rank, 2)
        return cluster_and_evaluate(eigenvectors, true_labels, seed)
    except Exception as e:
        print(f"RPCA Error: {str(e)}")
        print(traceback.format_exc())
        raise e

def run_experiment(a, b, sigma, noise_density, noise_magnitude, seed):
    np.random.seed(seed)
    sbm, G_clean = generate_sbm(a, b, sigma, seed)
    
    G_noisy = generate_noisy_sbm(
        sbm, G_clean, noise_type='element',
        noise_density=noise_density, noise_magnitude=noise_magnitude,
        random_state=seed
    )
    
    adj_matrix = prepare_graph_matrix(G_noisy)
    true_labels = np.array(sbm.community_assignments)
    signal_strength = abs(a - b) / (sigma**2)
    
    methods = {
        'spectral_unnormalized': lambda: cluster_and_evaluate(
            get_top_eigenvectors(adj_matrix, 2), true_labels, seed),
        'rpca': lambda: rpca_method(adj_matrix, true_labels, seed),
        'outlier_pursuit': lambda: cluster_and_evaluate(
            get_top_eigenvectors(OutlierPursuit(gamma=0.15).fit_transform(adj_matrix), 2), 
            true_labels, seed)
    }
    
    results = {}
    for method_name, method_fn in methods.items():
        try:
            print(f"Running {method_name}...")
            results[method_name] = method_fn()
            results[method_name].update({
                'noise_density': noise_density,
                'noise_magnitude': noise_magnitude,
                'signal_strength': signal_strength
            })
            print(f"{method_name} completed successfully with NMI: {results[method_name]['NMI']}")
        except Exception as e:
            print(f"Error in {method_name}: {str(e)}")
            results[method_name] = {"error": str(e)}
    
    return results

def run_dual_sweep(density_values, magnitude_values, fixed_params):
    results = []
    
    for density in density_values:
        for magnitude in magnitude_values:
            print(f"\nProcessing noise_density={density:.2f}, noise_magnitude={magnitude:.2f}")
            params = fixed_params.copy()
            params['noise_density'] = density
            params['noise_magnitude'] = magnitude
            
            for sim in range(N_SIMULATIONS):
                seed = BASE_SEED + sim
                trial_results = run_experiment(**params, seed=seed)
                
                for method, metrics in trial_results.items():
                    if "error" not in metrics:
                        results.append({
                            'noise_density': density,
                            'noise_magnitude': magnitude,
                            'method': method,
                            'nmi': metrics['NMI'],
                            'simulation': sim,
                            'signal_strength': metrics['signal_strength']
                        })
                    else:
                        print(f"Skipping {method} due to error: {metrics['error']}")
    
    return pd.DataFrame(results)

def analyze_results(df):
    # Create a pivot table for noise magnitude vs. density for each method
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        print(f"\n=== Average NMI for {method} ===")
        pivot = method_df.pivot_table(
            index='noise_magnitude', 
            columns='noise_density',
            values='nmi',
            aggfunc='mean'
        )
        print(pivot.round(3))
    
    # Combined summary for all methods
    print("\n=== Average NMI per Method ===")
    print(df.groupby('method')['nmi'].mean().round(3))
    
    return df

if __name__ == "__main__":
    print("=== Starting Dual Parameter Sweep (Noise Magnitude and Density) ===")
    fixed_params = {'a': 1.0, 'b': -1.0, 'sigma': 1}
    
    density_values = np.array([0.05, 0.1, 0.15, 0.2, 0.25])
    magnitude_values = np.round(np.arange(0.0, 20, 2), 2)  # Reduced step size for efficiency
    
    df_results = run_dual_sweep(density_values, magnitude_values, fixed_params)
    
    if not df_results.empty:
        print("\n=== Results DataFrame ===")
        print(df_results.head())
        print(f"Methods present in results: {df_results['method'].unique()}")
        
        df_results = analyze_results(df_results)
        df_results.to_csv('element_sweep_results.csv', index=False)
    else:
        print("No results were collected - check for errors in the methods")