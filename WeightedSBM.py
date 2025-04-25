#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 11:24:09 2025

@author: scmorgan
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.sparse as sp
from rpca import *
from op import *
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

class WeightedSBM:
    """
    A class for generating balanced weighted stochastic blockmodels.
    
    This implements a weighted version of the stochastic blockmodel where:
    - Nodes are evenly divided into communities (balanced)
    - Edge weights are drawn from specified distributions
    - The probability and weight distributions can be different for within-community
      and between-community connections
    - Nodes are ordered by community membership in the adjacency matrix
    """
    
    def __init__(self, n_nodes, n_communities, p_within, p_between, 
                 weight_dist_within=None, weight_dist_between=None):
        """
        Initialize the balanced weighted SBM.
        
        Parameters:
        -----------
        n_nodes : int
            Total number of nodes in the graph
        n_communities : int
            Number of communities to divide nodes into
        p_within : float
            Probability of edge existence within communities (0 to 1)
        p_between : float
            Probability of edge existence between communities (0 to 1)
        weight_dist_within : callable, optional
            Function that returns random weights for within-community edges
            Default: Uniform distribution between 0.5 and 1
        weight_dist_between : callable, optional
            Function that returns random weights for between-community edges
            Default: Uniform distribution between 0 and 0.5
        """
        self.n_nodes = n_nodes
        self.n_communities = n_communities
        self.p_within = p_within
        self.p_between = p_between
        
        # Default weight distributions if none provided
        if weight_dist_within is None:
            self.weight_dist_within = lambda: np.random.uniform(0.5, 1.0)
        else:
            self.weight_dist_within = weight_dist_within
            
        if weight_dist_between is None:
            self.weight_dist_between = lambda: np.random.uniform(0.0, 0.5)
        else:
            self.weight_dist_between = weight_dist_between
        
        # Assign nodes to communities in order
        self.community_assignments, self.node_order = self._assign_ordered_communities()
        
    def _assign_ordered_communities(self):
        """
        Assign nodes to communities in a balanced way and create an ordering
        where nodes in the same community are grouped together.
        
        Returns:
        --------
        tuple:
            - List of community assignments for each node (original order)
            - List of node indices ordered by community
        """
        # Calculate nodes per community
        nodes_per_community = self.n_nodes // self.n_communities
        remainder = self.n_nodes % self.n_communities
        
        # Create community assignments (still in original order)
        assignments = []
        
        # Track start of each community
        community_starts = [0]
        current_pos = 0
        
        for i in range(self.n_communities):
            # Add one extra node to the first 'remainder' communities to handle non-divisible cases
            size = nodes_per_community + (1 if i < remainder else 0)
            assignments.extend([i] * size)
            
            current_pos += size
            if i < self.n_communities - 1:  # Don't add after the last community
                community_starts.append(current_pos)
        
        # Create a node ordering where nodes are grouped by community
        node_order = []
        for i in range(self.n_communities):
            size = nodes_per_community + (1 if i < remainder else 0)
            start_idx = community_starts[i]
            node_order.extend(range(start_idx, start_idx + size))
            
        return assignments, node_order
    
    def generate_graph(self):
        """
        Generate a weighted graph according to the balanced SBM model,
        with nodes ordered by community.
        
        Returns:
        --------
        G : networkx.Graph
            A weighted graph with communities
        """
        # Create empty graph
        G = nx.Graph()
        G.add_nodes_from(range(self.n_nodes))
        
        # Store community assignments as node attributes
        for i in range(self.n_nodes):
            G.nodes[i]['community'] = self.community_assignments[i]
            G.nodes[i]['order'] = self.node_order.index(i)
            
        # Add edges based on community structure
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                # Check if nodes are in the same community
                same_community = (self.community_assignments[i] == self.community_assignments[j])
                
                # Set edge probability based on community relationship
                edge_prob = self.p_within if same_community else self.p_between
                
                # Generate random number to determine if edge exists
                if np.random.random() < edge_prob:
                    # Generate weight based on community relationship
                    if same_community:
                        weight = self.weight_dist_within()
                    else:
                        weight = self.weight_dist_between()
                    
                    # Add weighted edge
                    G.add_edge(i, j, weight=weight)
        
        return G
    
    def get_adjacency_matrix(self, G):
        """
        Returns the adjacency matrix with nodes ordered by community.
        
        Parameters:
        -----------
        G : networkx.Graph
            The generated graph
            
        Returns:
        --------
        numpy.ndarray
            Adjacency matrix with nodes ordered by community
        """
        # Get ordered nodes
        ordered_nodes = sorted(G.nodes(), key=lambda n: G.nodes[n]['order'])
        
        # Create adjacency matrix
        A = nx.to_numpy_array(G, nodelist=ordered_nodes)
        
        return A, ordered_nodes
    
    def visualize(self, G, node_size=100, with_labels=False, layout=None, title=None, show_communities=True):
        """
        Visualize the generated graph with communities colored.
        
        Parameters:
        -----------
        G : networkx.Graph
            The graph to visualize
        node_size : int, optional
            Size of nodes in visualization
        with_labels : bool, optional
            Whether to show node labels
        layout : callable, optional
            Layout function for the graph, default is spring_layout
        title : str, optional
            Optional title for the plot
        show_communities : bool, optional
            Whether to use community-based colors
        """
        if layout is None:
            pos = nx.spring_layout(G, seed=42)
        else:
            pos = layout(G)
        
        plt.figure(figsize=(10, 8))
        
        if show_communities:
            # Define a colormap for communities
            cmap = plt.cm.rainbow
            colors = [cmap(i/self.n_communities) for i in range(self.n_communities)]
            
            # Draw nodes colored by community
            node_colors = [colors[G.nodes[n]['community']] for n in G.nodes()]
        else:
            # Use node ordering for colors (gradient effect)
            node_colors = [G.nodes[n]['order'] / self.n_nodes for n in G.nodes()]
        
        # Get edge weights for width
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [3 * w / max_weight for w in edge_weights]
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, cmap=plt.cm.viridis if not show_communities else None)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7)
        
        if with_labels:
            nx.draw_networkx_labels(G, pos)
        
        if title:
            plt.title(title)
            
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_adjacency_matrix(self, G, title=None):
        """
        Visualize the adjacency matrix with communities clearly visible.
        
        Parameters:
        -----------
        G : networkx.Graph
            The generated graph
        title : str, optional
            Optional title for the plot
        """
        # Get adjacency matrix with ordered nodes
        A, ordered_nodes = self.get_adjacency_matrix(G)
        
        # Plot the matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(A, cmap='Blues', interpolation='none')
        
        # Add community dividers if there are multiple communities
        if self.n_communities > 1:
            # Calculate community boundaries
            nodes_per_community = self.n_nodes // self.n_communities
            remainder = self.n_nodes % self.n_communities
            
            boundaries = []
            count = 0
            for i in range(self.n_communities - 1):  # Don't need a line after the last community
                size = nodes_per_community + (1 if i < remainder else 0)
                count += size
                boundaries.append(count - 0.5)  # Offset by 0.5 to align with grid
            
            # Add vertical and horizontal lines to show community boundaries
            for b in boundaries:
                plt.axhline(y=b, color='red', linestyle='-', linewidth=1)
                plt.axvline(x=b, color='red', linestyle='-', linewidth=1)
        
        if title:
            plt.title(title)
        else:
            plt.title("Adjacency Matrix (Ordered by Community)")
            
        plt.colorbar(label="Edge Weight")
        plt.tight_layout()
        plt.show()
    
    def calculate_statistics(self, G):
        """
        Calculate various statistics about the generated graph.
        
        Parameters:
        -----------
        G : networkx.Graph
            The generated weighted SBM graph
            
        Returns:
        --------
        dict
            Dictionary of statistics
        """
        stats = {}
        
        # Basic graph statistics
        stats['num_nodes'] = G.number_of_nodes()
        stats['num_edges'] = G.number_of_edges()
        stats['density'] = nx.density(G)
        
        # Community statistics
        within_edges = 0
        between_edges = 0
        within_weights = []
        between_weights = []
        
        for u, v, data in G.edges(data=True):
            if G.nodes[u]['community'] == G.nodes[v]['community']:
                within_edges += 1
                within_weights.append(data['weight'])
            else:
                between_edges += 1
                between_weights.append(data['weight'])
        
        stats['within_community_edges'] = within_edges
        stats['between_community_edges'] = between_edges
        
        if within_weights:
            stats['avg_within_weight'] = np.mean(within_weights)
            stats['std_within_weight'] = np.std(within_weights)
        else:
            stats['avg_within_weight'] = 0
            stats['std_within_weight'] = 0
            
        if between_weights:
            stats['avg_between_weight'] = np.mean(between_weights)
            stats['std_between_weight'] = np.std(between_weights)
        else:
            stats['avg_between_weight'] = 0
            stats['std_between_weight'] = 0
        
        # Calculate modularity
        community_dict = {node: G.nodes[node]['community'] for node in G.nodes()}
        stats['modularity'] = nx.community.modularity(G, [
            [n for n in G.nodes() if G.nodes[n]['community'] == c]
            for c in range(self.n_communities)
        ])
        
        return stats
    
def generate_symmetric_sparse_noise(n, density=0.05, min_value=-0.1, max_value=0.1, random_state=None):
    """
    Generate a symmetric sparse noise matrix.
    
    Parameters:
    -----------
    n : int
        Size of the square matrix (n x n)
    density : float, optional
        Sparsity of the matrix, fraction of elements that are non-zero
        Default is 0.05 (5% of elements are non-zero)
    min_value : float, optional
        Minimum value for the noise
        Default is -0.1
    max_value : float, optional
        Maximum value for the noise
        Default is 0.1
    random_state : int or numpy.random.RandomState, optional
        Random state for reproducibility
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        A symmetric sparse matrix with random noise
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Calculate number of non-zero elements
    # Since we'll create a symmetric matrix, we only need to generate
    # elements for the upper triangle
    nnz = int(density * n * (n-1) / 2)
    
    # Generate random indices for the upper triangle
    rows = []
    cols = []
    for _ in range(nnz):
        i, j = np.random.randint(0, n, 2)
        # Ensure i < j (upper triangle) and avoid duplicates
        if i > j:
            i, j = j, i
        if i != j and (i, j) not in zip(rows, cols):
            rows.append(i)
            cols.append(j)
    
    # Generate random values for the non-zero elements
    values = np.random.uniform(min_value, max_value, len(rows))
    
    # Create a sparse matrix from the upper triangle
    upper_triangle = sp.csr_matrix((values, (rows, cols)), shape=(n, n))
    
    # Create the symmetric matrix by adding the transpose
    # For the diagonal, we'll add random noise directly
    noise_matrix = upper_triangle + upper_triangle.T
    
    return noise_matrix

def add_noise_to_blockmodel_matrix(block_matrix, noise_matrix):
    """
    Adds a noise matrix to a blockmodel matrix, handling sparse and dense matrices.
    
    Parameters:
    -----------
    block_matrix : numpy.ndarray or scipy.sparse matrix
        The original blockmodel adjacency matrix
    noise_matrix : scipy.sparse matrix
        The symmetric sparse noise matrix to add
        
    Returns:
    --------
    numpy.ndarray or scipy.sparse matrix
        The blockmodel matrix with added noise
    """
    # If the block matrix is dense, convert the noise to dense for addition
    if isinstance(block_matrix, np.ndarray):
        return block_matrix + noise_matrix.toarray()
    else:
        # Both are sparse, just add directly
        return block_matrix + noise_matrix
    


def generate_noisy_sbm(sbm, G, noise_type='element', noise_density=0.05, noise_magnitude=0.1, random_state=None):
    """
    Generate a noisy version of a stochastic blockmodel with extensive debugging.
    """

    
    # Get the adjacency matrix

    A, ordered_nodes = sbm.get_adjacency_matrix(G)
    n = A.shape[0]

    
    # Set up random state - SAFER VERSION

    if random_state is None:

        rng = np.random.RandomState()
    elif isinstance(random_state, np.random.RandomState):

        rng = random_state
    else:

        # Always create a new RandomState to be safe
        try:
            seed = int(random_state)

            rng = np.random.RandomState(seed)
        except:

            rng = np.random.RandomState(42)
    
    if noise_type == 'element':
        try:
            noise = np.zeros((n, n))

        # Get all upper triangle indices (including diagonal)
            upper_triangle_indices = np.triu_indices(n)
            all_upper_indices = list(zip(upper_triangle_indices[0], upper_triangle_indices[1]))
        
            total_possible = len(all_upper_indices)
            num_elements = int(total_possible * noise_density)

        # Sample without replacement
            selected_indices = rng.choice(total_possible, size=num_elements, replace=False)
            selected_coords = [all_upper_indices[idx] for idx in selected_indices]

        # Apply noise symmetrically
            for i, j in selected_coords:
               val = rng.uniform(-noise_magnitude, noise_magnitude)
               noise[i, j] = val
               noise[j, i] = val  # Symmetric for off-diagonal

        except Exception as e:
            raise RuntimeError(f"Error generating symmetric noise: {str(e)}")
        

            
    elif noise_type == 'column':

        try:
            noise = np.zeros((n, n))
            # Determine which columns to perturb
            num_cols = int(n * noise_density)

            if num_cols < 1:
                num_cols = 1  # Ensure at least one column is perturbed
                
            perturb_cols = rng.choice(n, size=num_cols, replace=False)

            
            for col in perturb_cols:
                # Generate column perturbation
                col_noise = rng.uniform(-noise_magnitude, noise_magnitude, size=n)
                noise[:, col] += col_noise
                noise[col, :] += col_noise  # Ensure symmetry
                noise[col, col] -= col_noise[col]  # Avoid double-counting diagonal
            
     
        except Exception as e:
   
            raise
            
    elif noise_type == 'sparse':
    
        try:
            noise = np.zeros((n, n))
            # Number of noise elements based on density
            num_noise_elements = int(noise_density * n * n / 2)  # Divide by 2 for symmetry

            
            # Generate random positions for noise (upper triangular to ensure symmetry)
            i_indices = []
            j_indices = []
            for _ in range(num_noise_elements):
                attempts = 0
                while attempts < 100:  # Limit attempts to prevent infinite loop
                    i = rng.randint(0, n)
                    j = rng.randint(i, n)  # j >= i to stay in upper triangle
                    if (i, j) not in zip(i_indices, j_indices):
                        i_indices.append(i)
                        j_indices.append(j)
                        break
                    attempts += 1
            

            
            # Generate random noise values
            values = rng.uniform(-noise_magnitude, noise_magnitude, size=len(i_indices))
            
            # Assign noise values
            for i, j, val in zip(i_indices, j_indices, values):
                noise[i, j] = val
                noise[j, i] = val
                    # Make symmetric
            

        except Exception as e:

            raise
    else:
        raise ValueError(f"Invalid noise_type: {noise_type}. Must be 'element', 'column', or 'sparse'")
    
    # Add noise to the adjacency matrix

    try:
        # Directly add noise instead of using add_noise_to_blockmodel_matrix
        # Replace values in A with noise where noise is non-zero
        A_noisy = A.copy()
        A_noisy[noise != 0] = noise[noise != 0]


    except Exception as e:
 
        raise
    
    # Create a new graph from the noisy adjacency matrix

    try:
        G_noisy = nx.from_numpy_array(A_noisy)
     
    except Exception as e:

        raise
    
    # Copy node attributes from the original graph to preserve community information

    try:
        for i, node in enumerate(ordered_nodes):
            community = G.nodes[node]['community']
            order = G.nodes[node]['order']
            G_noisy.nodes[i]['community'] = community
            G_noisy.nodes[i]['order'] = order

    except Exception as e:

        raise
    
    return G_noisy
def visualize_noise_effect(sbm, G, G_noisy):
    """
    Visualize the effect of noise on the adjacency matrix.
    
    Parameters:
    -----------
    sbm : BalancedWeightedSBM
        The stochastic blockmodel instance
    G : networkx.Graph
        The original graph without noise
    G_noisy : networkx.Graph
        The noisy graph
    """
    A, _ = sbm.get_adjacency_matrix(G)
    A_noisy, _ = sbm.get_adjacency_matrix(G_noisy)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original adjacency matrix
    im0 = axs[0].imshow(A, cmap='Blues', interpolation='none')
    axs[0].set_title("Original Adjacency Matrix")
    plt.colorbar(im0, ax=axs[0])
    
    # Noisy adjacency matrix
    im1 = axs[1].imshow(A_noisy, cmap='Blues', interpolation='none')
    axs[1].set_title("Noisy Adjacency Matrix")
    plt.colorbar(im1, ax=axs[1])
    
    # Difference
    diff = A_noisy - A
    im2 = axs[2].imshow(diff, cmap='coolwarm', interpolation='none', vmin=-0.2, vmax=0.2)
    axs[2].set_title("Difference (Noise)")
    plt.colorbar(im2, ax=axs[2])
    
    plt.tight_layout()
    plt.show()
def evaluate_clustering(pred_labels, true_labels):
    # Calculate error rate (percentage of misclassified nodes)
    # First, we need to handle the label matching issue (permutation invariance)
    # For binary case, we can simply check if flipping labels improves accuracy
    accuracy1 = np.mean(pred_labels == true_labels)
    accuracy2 = np.mean(pred_labels == (1 - true_labels))
    accuracy = max(accuracy1, accuracy2)
    error_rate = 1 - accuracy
    
    # Calculate additional metrics
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    return {
        "error_rate": error_rate,
        "accuracy": accuracy,
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi
    }