#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 10:53:34 2025

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


np.random.seed(42)

# Create a weighted SBM
n_nodes = 100
n_communities = 2
p_within = 1
p_between = 1

# Custom weight distributions
weight_within = lambda: np.random.normal(0.3, 0.1)  # Stronger within-community
weight_between = lambda: np.random.normal(-0.2, 0.1)  # Weaker between-community

# Initialize the SBM
sbm = WeightedSBM(
    n_nodes=n_nodes,
    n_communities=n_communities,
    p_within=p_within,
    p_between=p_between,
    weight_dist_within=weight_within,
    weight_dist_between=weight_between
)

# Generate the original graph
G = sbm.generate_graph()

# Print some statistics
stats = sbm.calculate_statistics(G)
print("Original Graph Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# Generate noisy versions with different noise types
noise_types = ['element', 'column']
noise_graphs = {}
for noise_type in noise_types:
    # Create a proper numpy random state
    rng = np.random.RandomState(42)  # Make sure this is a RandomState object
    
    try:
        G_noisy = generate_noisy_sbm(
            sbm=sbm,
            G=G,
            noise_type=noise_type,
            noise_density=0.4,
            noise_magnitude=0.3,
            random_state=rng
        )
        noise_graphs[noise_type] = G_noisy
        
        # Print statistics for each noisy graph
        stats_noisy = sbm.calculate_statistics(G_noisy)
        print(f"\nNoisy Graph ({noise_type}) Statistics:")
        for key, value in stats_noisy.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error generating noisy graph with {noise_type} noise: {e}")
    
# Visualize the original graph and its adjacency matrix
plt.figure(figsize=(12, 10))

# Use subplot2grid to avoid subplot errors
ax1 = plt.subplot(121)
pos = nx.spring_layout(G, seed=42)

# Manually draw the graph
cmap = plt.cm.rainbow
colors = [cmap(c/n_communities) for c in range(n_communities)]
node_colors = [colors[G.nodes[n]['community']] for n in G.nodes()]

edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
max_weight = max(edge_weights) if edge_weights else 1
edge_widths = [2 * w / max_weight for w in edge_weights]

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=30, ax=ax1)
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, ax=ax1)
ax1.set_title("Original Graph")
ax1.axis('off')

# Get adjacency matrix for visualization
A, _ = sbm.get_adjacency_matrix(G)

ax2 = plt.subplot(122)
im = ax2.imshow(A, cmap='Blues', interpolation='none')
ax2.set_title("Original Adjacency Matrix")
plt.colorbar(im, ax=ax2)

plt.tight_layout()
plt.show()

# Visualize each noise type effect
fig = plt.figure(figsize=(15, 12))
gs = GridSpec(3, 3, figure=fig)

for i, noise_type in enumerate(noise_types):
    G_noisy = noise_graphs[noise_type]
    A_noisy, _ = sbm.get_adjacency_matrix(G_noisy)
    
    # Graph visualization
    ax_graph = fig.add_subplot(gs[i, 0])
    pos_noisy = nx.spring_layout(G_noisy, seed=42)
    
    # Define a colormap for communities
    node_colors = [colors[G_noisy.nodes[n]['community']] for n in G_noisy.nodes()]
    
    # Get edge weights for width
    edge_weights = [G_noisy[u][v]['weight'] for u, v in G_noisy.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [2 * w / max_weight for w in edge_weights]
    
    nx.draw_networkx_nodes(G_noisy, pos_noisy, node_color=node_colors, node_size=30, ax=ax_graph)
    nx.draw_networkx_edges(G_noisy, pos_noisy, width=edge_widths, alpha=0.7, ax=ax_graph)
    ax_graph.set_title(f"{noise_type.capitalize()} Noise Graph")
    ax_graph.axis('off')
    
    # Adjacency matrix
    ax_adj = fig.add_subplot(gs[i, 1])
    im1 = ax_adj.imshow(A_noisy, cmap='Blues', interpolation='none')
    ax_adj.set_title(f"{noise_type.capitalize()} Noise Adjacency Matrix")
    fig.colorbar(im1, ax=ax_adj)
    
    # Difference (noise)
    ax_diff = fig.add_subplot(gs[i, 2])
    diff = A_noisy - A
    im2 = ax_diff.imshow(diff, cmap='coolwarm', interpolation='none', vmin=-0.3, vmax=0.3)
    ax_diff.set_title(f"{noise_type.capitalize()} Noise Pattern")
    fig.colorbar(im2, ax=ax_diff)

plt.tight_layout()
plt.show()

# Compare modularity changes
modularity_original = stats['modularity']
modularity_values = [stats['modularity']] + [sbm.calculate_statistics(G_noisy)['modularity'] 
                                             for G_noisy in noise_graphs.values()]

plt.figure(figsize=(10, 6))
bars = plt.bar(['Original'] + noise_types, modularity_values, color=['green'] + ['blue']*len(noise_types))
plt.title('Modularity Comparison')
plt.ylabel('Modularity')
plt.xlabel('Graph Type')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

rpca_o = RobustPCA()
rpca_o.fit(A)