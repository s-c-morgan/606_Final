#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 15:50:10 2025

@author: scmorgan
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.sparse as sp
from WeightedSBM import *
from rpca import *
from op import *
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.linalg import eigh

def get_top_eigenvectors(matrix, k):
    if not np.allclose(matrix, matrix.T):
        matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = eigh(matrix)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx[:k]]

def spectral_unnormalized(adj_matrix, true_labels, seed):
    eigenvectors = get_top_eigenvectors(adj_matrix, 2)
    labels = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(eigenvectors)
    return evaluate_clustering(labels, true_labels), labels

np.random.seed(42)
n_nodes = 100
n_communities = 2
p_within = 1
p_between = 1

weight_within = lambda: np.random.normal(0.5, 1)
weight_between = lambda: np.random.normal(-0.5, 1)

sbm = WeightedSBM(
    n_nodes=n_nodes,
    n_communities=n_communities,
    p_within=p_within,
    p_between=p_between,
    weight_dist_within=weight_within,
    weight_dist_between=weight_between
)

G = sbm.generate_graph()
stats = sbm.calculate_statistics(G)
print("Original Graph Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

noise_types = ['element', 'column']
noise_graphs = {}
for noise_type in noise_types:
    rng = np.random.RandomState(42)
    try:
        G_noisy = generate_noisy_sbm(
            sbm=sbm,
            G=G,
            noise_type=noise_type,
            noise_density=0.5,
            noise_magnitude=8,
            random_state=rng
        )
        noise_graphs[noise_type] = G_noisy
        stats_noisy = sbm.calculate_statistics(G_noisy)
        print(f"\nNoisy Graph ({noise_type}) Statistics:")
        for key, value in stats_noisy.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error generating noisy graph with {noise_type} noise: {e}")

true_labels = np.array(sbm.community_assignments)

rpca_models = {}
op_models = {}
spectral_results = {}
rpca_results = {}
op_results = {}
rpca_metrics = {}
op_metrics = {}
spectral_metrics = {}

all_graphs = {"original": G}
all_graphs.update(noise_graphs)

for graph_name, graph in all_graphs.items():
    print(f"\nProcessing {graph_name} graph...")
    adj_matrix = nx.to_numpy_array(graph)

    print(f"  Applying standard spectral clustering...")
    try:
        adj_matrix_sym = (adj_matrix + adj_matrix.T) / 2
        metrics, spectral_cluster_labels = spectral_unnormalized(adj_matrix_sym, true_labels, seed=42)
        spectral_results[graph_name] = spectral_cluster_labels
        spectral_metrics[graph_name] = metrics

        print(f"  Standard Spectral Clustering Results:")
        for metric_name, metric_value in spectral_metrics[graph_name].items():
            print(f"    {metric_name}: {metric_value:.4f}")
    except Exception as e:
        print(f"  Error in standard spectral clustering for {graph_name}: {e}")

    print(f"  Applying Robust PCA...")
    rpca_model = RobustPCA(max_iter=100, tol=1e-4)
    try:
        rpca_model.fit(adj_matrix)
        rpca_models[graph_name] = rpca_model
        low_rank = rpca_model.get_low_rank()
        eigenvectors = get_top_eigenvectors(low_rank, n_communities)
        rpca_cluster_labels = KMeans(n_clusters=n_communities, random_state=42, n_init=10).fit_predict(eigenvectors)
        rpca_results[graph_name] = rpca_cluster_labels
        metrics = evaluate_clustering(rpca_cluster_labels, true_labels)
        rpca_metrics[graph_name] = metrics

        print(f"  RPCA Clustering Results:")
        for metric_name, metric_value in metrics.items():
            print(f"    {metric_name}: {metric_value:.4f}")
    except Exception as e:
        print(f"  Error in RPCA processing for {graph_name}: {e}")

    print(f"  Applying OutlierPursuit...")
    try:
        op_model = OutlierPursuit(gamma=0.15)
        low_rank_op = op_model.fit_transform(adj_matrix)
        op_models[graph_name] = op_model
        eigenvectors_op = get_top_eigenvectors(low_rank_op, n_communities)
        op_cluster_labels = KMeans(n_clusters=n_communities, random_state=42, n_init=10).fit_predict(eigenvectors_op)
        op_results[graph_name] = op_cluster_labels
        metrics_op = evaluate_clustering(op_cluster_labels, true_labels)
        op_metrics[graph_name] = metrics_op

        print(f"  OutlierPursuit Clustering Results:")
        for metric_name, metric_value in metrics_op.items():
            print(f"    {metric_name}: {metric_value:.4f}")
    except Exception as e:
        print(f"  Error in OutlierPursuit processing for {graph_name}: {e}")


# Visualize comparative results
plt.figure(figsize=(15, 10))
metrics_to_plot = ["error_rate", "accuracy", "adjusted_rand_index", "normalized_mutual_info"]
graph_names = list(all_graphs.keys())

for i, metric in enumerate(metrics_to_plot):
    plt.subplot(2, 2, i+1)
    
    # Set width of bars
    width = 0.25  # Narrower bars to fit three methods
    x = np.arange(len(graph_names))
    
    # Get values for this metric (handling missing data)
    spectral_values = [spectral_metrics.get(name, {}).get(metric, 0) for name in graph_names]
    rpca_values = [rpca_metrics.get(name, {}).get(metric, 0) for name in graph_names]
    op_values = [op_metrics.get(name, {}).get(metric, 0) for name in graph_names]
    
    # Plot bars
    plt.bar(x - width, spectral_values, width, label='Spectral')
    plt.bar(x, rpca_values, width, label='RPCA')
    plt.bar(x + width, op_values, width, label='OutlierPursuit')
    
    plt.title(metric.replace("_", " ").title())
    plt.xticks(x, graph_names, rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.suptitle("Comparison of Spectral vs RPCA vs OutlierPursuit for Community Detection", fontsize=16)
plt.subplots_adjust(top=0.9)
plt.show()

# Print a summary of the comparison
print("\nPerformance Summary - Spectral vs RPCA vs OutlierPursuit:")
print("-" * 75)
print(f"{'Graph Type':<15} {'Method':<15} {'Error Rate':<12} {'Accuracy':<12} {'ARI':<12} {'NMI':<12}")
print("-" * 75)

for graph_name in graph_names:
    if graph_name in spectral_metrics:
        print(f"{graph_name:<15} {'Spectral':<15} {spectral_metrics[graph_name]['error_rate']:<12.4f} "
              f"{spectral_metrics[graph_name]['accuracy']:<12.4f} "
              f"{spectral_metrics[graph_name]['adjusted_rand_index']:<12.4f} "
              f"{spectral_metrics[graph_name]['normalized_mutual_info']:<12.4f}")
              
    if graph_name in rpca_metrics:
        print(f"{graph_name:<15} {'RPCA':<15} {rpca_metrics[graph_name]['error_rate']:<12.4f} "
              f"{rpca_metrics[graph_name]['accuracy']:<12.4f} "
              f"{rpca_metrics[graph_name]['adjusted_rand_index']:<12.4f} "
              f"{rpca_metrics[graph_name]['normalized_mutual_info']:<12.4f}")
    
    if graph_name in op_metrics:
        print(f"{graph_name:<15} {'OutlierPursuit':<15} {op_metrics[graph_name]['error_rate']:<12.4f} "
              f"{op_metrics[graph_name]['accuracy']:<12.4f} "
              f"{op_metrics[graph_name]['adjusted_rand_index']:<12.4f} "
              f"{op_metrics[graph_name]['normalized_mutual_info']:<12.4f}")
    
    print("-" * 75)

# Create a more detailed visualization to compare methods across graph types
plt.figure(figsize=(18, 12))
methods = ['Spectral', 'RSC-PCP', 'RSC-NOP']
metrics_map = {'Spectral': spectral_metrics, 'RSC-PCP': rpca_metrics, 'RSC-NOP': op_metrics}

for i, metric in enumerate(metrics_to_plot):
    plt.subplot(2, 2, i+1)
    
    data = []
    for method in methods:
        method_values = []
        for graph_name in graph_names:
            value = metrics_map[method].get(graph_name, {}).get(metric, np.nan)
            method_values.append(value)
        data.append(method_values)
    
    # Create a grouped bar chart
    x = np.arange(len(graph_names))
    width = 0.25
    
    plt.bar(x - width, data[0], width, label=methods[0], color='blue', alpha=0.7)
    plt.bar(x, data[1], width, label=methods[1], color='green', alpha=0.7)
    plt.bar(x + width, data[2], width, label=methods[2], color='red', alpha=0.7)
    
    plt.title(f"{metric.replace('_', ' ').title()} Comparison", fontsize=14)
    plt.xticks(x, graph_names, rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    


plt.tight_layout()
plt.suptitle("Detailed Comparison of Methods by Graph Type and Metric", fontsize=16)
plt.subplots_adjust(top=0.9)
plt.show()