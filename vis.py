#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 20:56:35 2025

@author: scmorgan
"""

#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

METHOD_NAMES = {
    'rpca': 'RSC-PCP',
    'outlier_pursuit': 'RSC-OP',
    'spectral_unnormalized': 'Spectral Clustering'
}

def plot_sweep_results(results_df, output_file='element_sweep_plots.png'):
    """
    Create plots to visualize the dual parameter sweep results.
    
    Args:
        results_df: DataFrame containing the experimental results
        output_file: Filename to save the combined plot
    """
    # Get unique methods and densities
    methods = sorted(results_df['method'].unique())
    densities = sorted(results_df['noise_density'].unique())
    
    # Set up the figure
    fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 6), sharey=True)
    if len(methods) == 1:
        axes = [axes]  # Ensure axes is always iterable
    
    # Create a custom colormap for the different densities
    colors = sns.color_palette("viridis", len(densities))
    
    # Plot for each method
    for i, method in enumerate(methods):
        ax = axes[i]
        method_data = results_df[results_df['method'] == method]
        display_name = METHOD_NAMES.get(method, method)
        # Group by density and magnitude, calculate mean NMI
        grouped = method_data.groupby(['noise_density', 'noise_magnitude'])['nmi'].mean().reset_index()
        
        # Plot each density as a separate line
        for j, density in enumerate(densities):
            density_data = grouped[grouped['noise_density'] == density]
            ax.plot(
                density_data['noise_magnitude'], 
                density_data['nmi'],
                'o-',
                color=colors[j],
                label=f'Density = {density}',
                linewidth=2,
                markersize=8
            )
        
        # Set labels and title
        ax.set_xlabel('Noise Magnitude (u)', fontsize=12)
        if i == 0:
            ax.set_ylabel('Mean NMI', fontsize=12)
        ax.set_title(f'Method: {display_name}', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis limits
        ax.set_ylim(0, 1.05)
        
        # Add legend only for the first plot to avoid redundancy
        if i == 0:
            ax.legend(title='Noise Density', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def create_heatmap_visualization(results_df, output_file='method_heatmaps_e.png'):
    """
    Create heatmaps to visualize performance of each method across noise parameters.
    
    Args:
        results_df: DataFrame containing the experimental results
        output_file: Filename to save the heatmap plot
    """
    methods = sorted(results_df['method'].unique())
    
    # Set up the figure
    fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 4), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    
    # Define a custom colormap from red (bad) to green (good)
    cmap = LinearSegmentedColormap.from_list('RdYlGn', ['#d73027', '#fee08b', '#1a9850'])
    
    for i, method in enumerate(methods):
        ax = axes[i]
        method_data = results_df[results_df['method'] == method]
        display_name=METHOD_NAMES.get(method,method)
        # Create pivot table for the heatmap
        pivot_table = method_data.pivot_table(
            index='noise_magnitude', 
            columns='noise_density',
            values='nmi',
            aggfunc='mean'
        )
        
        # Plot heatmap
        sns.heatmap(
            pivot_table, 
            annot=True, 
            fmt='.2f', 
            cmap=cmap,
            vmin=0, 
            vmax=1,
            ax=ax,
            cbar=False if i < len(methods)-1 else True
        )
        
        ax.set_title(f'Method: {display_name}', fontsize=14)
        ax.set_xlabel('Noise Density')
        if i == 0:
            ax.set_ylabel('Noise Magnitude (u)')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def plot_all_methods_comparison(results_df, output_file='methods_comparison.png'):
    """
    Create a single plot comparing all methods across noise magnitudes,
    with separate subplots for each density value.
    
    Args:
        results_df: DataFrame containing the experimental results
        output_file: Filename to save the comparison plot
    """
    # Create a copy of the dataframe with renamed methods for display
    display_df = results_df.copy()
    display_df['display_name'] = display_df['method'].map(lambda m: METHOD_NAMES.get(m, m))
    
    methods = sorted(results_df['method'].unique())
    densities = sorted(results_df['noise_density'].unique())
    
    # Set up the figure
    fig, axes = plt.subplots(len(densities), 1, figsize=(8, 3*len(densities)), sharex=True)
    if len(densities) == 1:
        axes = [axes]
    
    # Create a colormap for different methods
    method_colors = sns.color_palette("Set1", len(methods))
    
    # Plot for each density
    for i, density in enumerate(densities):
        ax = axes[i]
        density_data = display_df[display_df['noise_density'] == density]
        
        # Plot each method
        for j, method in enumerate(methods):
            display_name = METHOD_NAMES.get(method, method)
            method_data = density_data[density_data['method'] == method]
            grouped = method_data.groupby('noise_magnitude')['nmi'].mean().reset_index()
            
            ax.plot(
                grouped['noise_magnitude'],
                grouped['nmi'],
                'o-',
                color=method_colors[j],
                label=display_name,
                linewidth=2,
                markersize=6
            )
        
        ax.set_ylabel('Mean NMI', fontsize=12)
        ax.set_title(f'Noise Density = {density}', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1.05)
        
        # Add legend only for the first subplot
        if i == 0:
            ax.legend(title='Method', fontsize=10, loc='best')
    
    # Set x-axis label only for the bottom subplot
    axes[-1].set_xlabel('Noise Magnitude (u)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

if __name__ == "__main__":
    # Load the results data
    try:
        df_results = pd.read_csv('element_sweep_results.csv')
        
        # Create the three visualizations
        line_plot_file = plot_sweep_results(df_results, output_file="element_sweep_results.png")
        heatmap_file = create_heatmap_visualization(df_results, output_file="heatmap_vis_element.png")
        comparison_file = plot_all_methods_comparison(df_results, output_file = "methods_comp_element.png")
        
        df_results = pd.read_csv('column_sweep_results.csv')
  
  # Create the three visualizations
        line_plot_file = plot_sweep_results(df_results, output_file="column_sweep_results.png")
        heatmap_file = create_heatmap_visualization(df_results, output_file="heatmap_vis_column.png")
        comparison_file = plot_all_methods_comparison(df_results, output_file = "methods_comp_column.png")
        
        print(f"Created visualizations:")
        print(f"1. Line plots by method: {line_plot_file}")
        print(f"2. Heatmaps by method: {heatmap_file}")
        print(f"3. Method comparison by density: {comparison_file}")
        
    except FileNotFoundError:
        print("Results file 'dual_sweep_results.csv' not found.")
        print("Please run the dual parameter sweep experiment first.")
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()

