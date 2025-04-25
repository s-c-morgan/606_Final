#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 12:39:54 2025

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
import pandas as pd

# Load results
results_df = pd.read_csv('aggregated_statistics.csv')

# Calculate performance error and bounds for confidence band
results_df['Error'] = 1 - results_df['mean_nmi']
results_df['Error_upper'] = 1 - (results_df['mean_nmi'] - results_df['std_nmi'])
results_df['Error_lower'] = 1 - (results_df['mean_nmi'] + results_df['std_nmi'])

# Create plot with proper sizing
plt.figure(figsize=(10, 6))

# Method styling configuration
method_styles = {
    'spectral_unnormalized': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
    'rpca': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},
    'outlier_pursuit': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'}
}
label_map = {
    'spectral_unnormalized': 'Spectral Unnormalized',
    'rpca': 'RSC-PCP',
    'outlier_pursuit': 'RSC-NOP'
}

# Plot each method's error curve with confidence bands
for method, style in method_styles.items():
    method_df = results_df[results_df['method'] == method].sort_values('sigma')
    
    # Line plot
    plt.plot(method_df['sigma'], method_df['Error'], label=label_map[method], **style)
    
    # Confidence band (±1 std)
    plt.fill_between(method_df['sigma'], 
                     method_df['Error_lower'], 
                     method_df['Error_upper'], 
                     color=style['color'], 
                     alpha=0.2)

# Configure plot aesthetics
plt.xlabel('Edge Weight Variance (σ)', fontsize=12)
plt.ylabel('Performance Error (1 - NMI)', fontsize=12)
plt.title('Community Detection Performance on Original Graph with Varying Signal Strength', fontsize=14)
plt.legend(title='Detection Method', frameon=False)
plt.grid(True, linestyle=':', alpha=0.7)

# Add secondary x-axis for signal strength
ax2 = plt.gca().twiny()
ax2.set_xlabel('Signal Strength (|a-b|/σ²)', fontsize=12)
sigma_ticks = plt.gca().get_xticks()
ax2.set_xticks(sigma_ticks)
ax2.set_xticklabels([f"{2/(x**2):.1f}" if x != 0 else '∞' for x in sigma_ticks])

plt.tight_layout()
plt.savefig('performance_error_with_bands.png', dpi=300)
plt.show()
