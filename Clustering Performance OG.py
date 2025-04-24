#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 12:38:58 2025

@author: scmorgan
"""

import numpy as np
import networkx as nx
import pandas as pd
from sklearn.utils import assert_all_finite
from WeightedSBM import *
from rpca import *
from op import *
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.linalg import eigh

# Constants
N_SIMULATIONS = 10
BASE_SEED = 42

def get_top_eigenvectors(matrix, k):
    """Robust eigen decomposition with stability enhancements"""
    matrix = np.nan_to_num((matrix + matrix.T) / 2)
    matrix += 1e-8 * np.eye(matrix.shape[0])
    
    try:
        eigenvalues, eigenvectors = eigh(matrix)
    except np.linalg.LinAlgError:
        matrix = np.nan_to_num(matrix)
        eigenvalues, eigenvectors = eigh(matrix)
    
    return eigenvectors[:, np.argsort(eigenvalues)[::-1][:k]]

def evaluate_clustering(cluster_labels, true_labels):
    """Calculate NMI with validation"""
    return {"NMI": normalized_mutual_info_score(true_labels, cluster_labels)}

def run_experiment(a, b, sigma, seed):
    """Single experimental trial with controlled randomness"""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    
    # SBM configuration
    sbm = WeightedSBM(
        n_nodes=100,
        n_communities=2,
        p_within=1,
        p_between=1,
        weight_dist_within=lambda: rng.normal(a, sigma),
        weight_dist_between=lambda: rng.normal(b, sigma)
    )
    
    # Generate graph and process adjacency matrix
    G = sbm.generate_graph()
    adj_matrix = np.maximum(nx.to_numpy_array(G), 0)
    adj_matrix = np.nan_to_num(adj_matrix)
    assert_all_finite(adj_matrix)
    
    results = {}
    true_labels = np.array(sbm.community_assignments)
    
    # Method execution with individual error handling
    methods = {
        'spectral_unnormalized': lambda: spectral_unnormalized(adj_matrix, true_labels, seed),
        'rpca': lambda: rpca_method(adj_matrix, true_labels, seed),
        'outlier_pursuit': lambda: outlier_pursuit_method(adj_matrix, true_labels, seed)
    }
    
    for method_name, method_fn in methods.items():
        try:
            results[method_name] = method_fn()
        except Exception as e:
            results[method_name] = {"error": str(e)}
    
    # Add metadata
    for method in results:
        if "error" not in results[method]:
            results[method]['signal_strength'] = abs(a - b) / (sigma**2)
    
    return {'original': results}

def spectral_unnormalized(adj_matrix, true_labels, seed):
    """Unnormalized spectral clustering implementation"""
    eigenvectors = get_top_eigenvectors(adj_matrix, 2)
    labels = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(eigenvectors)
    return evaluate_clustering(labels, true_labels)

def rpca_method(adj_matrix, true_labels, seed):
    """Robust PCA pipeline"""
    model = RobustPCA(max_iter=100, tol=1e-4)
    model.fit(adj_matrix)
    low_rank = model.get_low_rank()
    eigenvectors = get_top_eigenvectors(low_rank, 2)
    labels = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(eigenvectors)
    return evaluate_clustering(labels, true_labels)

def outlier_pursuit_method(adj_matrix, true_labels, seed):
    """Outlier pursuit pipeline"""
    model = OutlierPursuit(gamma=0.15)
    low_rank = model.fit_transform(adj_matrix)
    eigenvectors = get_top_eigenvectors(low_rank, 2)
    labels = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(eigenvectors)
    return evaluate_clustering(labels, true_labels)

def run_parameter_sweep():
    """Main experiment with statistical aggregation"""
    a, b = 1.0, -1.0
    sigma_values = np.round(np.logspace(-1, 1, 15), 3)
    
    results = []
    for sigma in sigma_values:
        print(f"\nProcessing σ={sigma:.3f}")
        for sim in range(N_SIMULATIONS):
            seed = BASE_SEED + sim
            trial = run_experiment(a, b, sigma, seed)
            for method, metrics in trial['original'].items():
                if "error" in metrics:
                    continue
                results.append({
                    'sigma': sigma,
                    'method': method,
                    'nmi': metrics['NMI'],
                    'simulation': sim,
                    'signal_strength': metrics['signal_strength']
                })
    
    return pd.DataFrame(results)

def analyze_results(df):
    """Statistical analysis of aggregated results"""
    analysis = df.groupby(['sigma', 'method'])['nmi'].agg(
        mean_nmi='mean',
        std_nmi='std',
        q1=lambda x: x.quantile(0.25),
        median='median',
        q3=lambda x: x.quantile(0.75)
        ).reset_index()
    
    print("\n=== Statistical Summary ===")
    print(analysis)
    
    return analysis

if __name__ == "__main__":
    print("=== Starting Rigorous Statistical Experiment ===")
    results_df = run_parameter_sweep()
    results_df.to_csv('full_simulation_results.csv', index=False)
    
    aggregated_df = analyze_results(results_df)
    aggregated_df.to_csv('aggregated_statistics.csv', index=False)
    
    print("\n=== Final Results ===")
    print(aggregated_df.pivot(index='sigma', columns='method', values='mean_nmi'))