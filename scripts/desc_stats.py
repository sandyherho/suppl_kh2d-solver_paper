#!/usr/bin/env python
"""
Comprehensive Analysis and Visualization of Kelvin-Helmholtz 2D Instability Simulations

Author: Sandy H. S. Herho
Email: sandy.herho@email.ucr.edu
Date: 09/18/2025

This script performs:
- Visualization of density fields with velocity quivers
- Statistical analysis including normality tests, entropy, and complexity measures
- Generation of publication-quality figures in multiple formats
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy import stats
from scipy.stats import (shapiro, anderson, jarque_bera, normaltest, 
                         kruskal, mannwhitneyu, wilcoxon, friedmanchisquare)
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib parameters for publication quality
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 16,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': False,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})


def calculate_spatial_entropy(field):
    """
    Calculate spatial entropy of a field using Shannon entropy.
    """
    data = field.flatten()
    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        return 0.0
    
    hist, bin_edges = np.histogram(data, bins=50, density=True)
    probs = hist * np.diff(bin_edges)
    probs = probs[probs > 0]
    
    if len(probs) == 0:
        return 0.0
    
    entropy = -np.sum(probs * np.log(probs))
    return entropy


def calculate_complexity_index(field):
    """
    Calculate comprehensive complexity index combining multiple measures.
    """
    data = field.flatten()
    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        return 0.0
    
    # Shannon entropy
    shannon = calculate_spatial_entropy(field)
    
    # Gradient-based complexity (spatial variability)
    if len(field.shape) == 2:
        grad_x = np.gradient(field, axis=1)
        grad_z = np.gradient(field, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_z**2)
        gradient_complexity = np.std(gradient_magnitude.flatten())
    else:
        gradient_complexity = 0.0
    
    # Statistical complexity
    std_norm = np.std(data) / (np.max(data) - np.min(data) + 1e-10)
    kurtosis_measure = abs(stats.kurtosis(data))
    
    # Weighted complexity index
    complexity = (0.3 * shannon + 
                 0.3 * gradient_complexity + 
                 0.2 * std_norm + 
                 0.2 * np.log1p(kurtosis_measure))
    
    return complexity


def perform_normality_tests(data):
    """
    Perform comprehensive normality tests on data.
    """
    data_flat = data.flatten()
    data_flat = data_flat[~np.isnan(data_flat)]
    
    if len(data_flat) == 0:
        return {
            'basic_stats': {
                'mean': 0, 'std': 0, 'skewness': 0, 'kurtosis': 0,
                'min': 0, 'max': 0, 'median': 0
            },
            'normality_tests': {}
        }
    
    if len(data_flat) > 5000:
        data_sample = np.random.choice(data_flat, 5000, replace=False)
    else:
        data_sample = data_flat
    
    results = {}
    
    # Basic statistics
    results['basic_stats'] = {
        'mean': round(np.mean(data_flat), 3),
        'std': round(np.std(data_flat), 3),
        'skewness': round(stats.skew(data_flat), 3),
        'kurtosis': round(stats.kurtosis(data_flat), 3),
        'min': round(np.min(data_flat), 3),
        'max': round(np.max(data_flat), 3),
        'median': round(np.median(data_flat), 3)
    }
    
    # Normality tests
    results['normality_tests'] = {}
    
    try:
        shapiro_stat, shapiro_p = shapiro(data_sample)
        results['normality_tests']['shapiro'] = {
            'statistic': round(shapiro_stat, 3),
            'p_value': round(shapiro_p, 3),
            'normal': shapiro_p > 0.05,
            'interpretation': f"Shapiro-Wilk: {'Normal' if shapiro_p > 0.05 else 'Non-normal'} (p={round(shapiro_p, 3)})"
        }
    except:
        results['normality_tests']['shapiro'] = {'interpretation': 'Shapiro-Wilk: Test failed'}
    
    try:
        anderson_result = anderson(data_flat)
        critical_value = anderson_result.critical_values[2]
        results['normality_tests']['anderson'] = {
            'statistic': round(anderson_result.statistic, 3),
            'critical_value': round(critical_value, 3),
            'normal': anderson_result.statistic < critical_value,
            'interpretation': f"Anderson-Darling: {'Normal' if anderson_result.statistic < critical_value else 'Non-normal'} (stat={round(anderson_result.statistic, 3)}, crit={round(critical_value, 3)})"
        }
    except:
        results['normality_tests']['anderson'] = {'interpretation': 'Anderson-Darling: Test failed'}
    
    try:
        jb_stat, jb_p = jarque_bera(data_flat)
        results['normality_tests']['jarque_bera'] = {
            'statistic': round(jb_stat, 3),
            'p_value': round(jb_p, 3),
            'normal': jb_p > 0.05,
            'interpretation': f"Jarque-Bera: {'Normal' if jb_p > 0.05 else 'Non-normal'} (p={round(jb_p, 3)})"
        }
    except:
        results['normality_tests']['jarque_bera'] = {'interpretation': 'Jarque-Bera: Test failed'}
    
    try:
        k2_stat, k2_p = normaltest(data_flat)
        results['normality_tests']['dagostino'] = {
            'statistic': round(k2_stat, 3),
            'p_value': round(k2_p, 3),
            'normal': k2_p > 0.05,
            'interpretation': f"D'Agostino K²: {'Normal' if k2_p > 0.05 else 'Non-normal'} (p={round(k2_p, 3)})"
        }
    except:
        results['normality_tests']['dagostino'] = {'interpretation': "D'Agostino K²: Test failed"}
    
    return results


def perform_nonparametric_comparison(data_dict):
    """
    Perform nonparametric tests to compare scenarios.
    """
    results = {}
    
    scenario_names = list(data_dict.keys())
    data_arrays = []
    for name in scenario_names:
        data_flat = data_dict[name].flatten()
        data_flat = data_flat[~np.isnan(data_flat)]
        if len(data_flat) > 10000:
            data_flat = np.random.choice(data_flat, 10000, replace=False)
        data_arrays.append(data_flat)
    
    if len(data_arrays) > 2:
        try:
            h_stat, p_value = kruskal(*data_arrays)
            results['kruskal_wallis'] = {
                'statistic': round(h_stat, 3),
                'p_value': round(p_value, 3),
                'significant': p_value < 0.05,
                'interpretation': f"Kruskal-Wallis: {'Significant' if p_value < 0.05 else 'No significant'} difference between scenarios (p={round(p_value, 3)})"
            }
        except:
            results['kruskal_wallis'] = {'interpretation': 'Kruskal-Wallis: Test failed'}
    
    pairwise_results = {}
    for i in range(len(scenario_names)):
        for j in range(i+1, len(scenario_names)):
            try:
                u_stat, p_value = mannwhitneyu(data_arrays[i], data_arrays[j], alternative='two-sided')
                pair_name = f"{scenario_names[i]} vs {scenario_names[j]}"
                pairwise_results[pair_name] = {
                    'statistic': round(u_stat, 3),
                    'p_value': round(p_value, 3),
                    'significant': p_value < 0.05,
                    'interpretation': f"{'Significant' if p_value < 0.05 else 'No significant'} difference (p={round(p_value, 3)})"
                }
            except:
                pairwise_results[pair_name] = {'interpretation': 'Test failed'}
    
    results['mann_whitney_pairwise'] = pairwise_results
    
    return results


def create_density_quiver_figure(ds, scenario_name, time_indices, output_dir):
    """
    Create 2x2 subplot figure showing density field with velocity quivers.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
    axes = axes.flatten()
    
    # Get data ranges for consistent colorbar
    rho_all = []
    for idx in time_indices:
        rho_all.append(ds.rho.isel(t=idx).values)
    vmin = np.min(rho_all)
    vmax = np.max(rho_all)
    
    # Create plots
    for i, idx in enumerate(time_indices):
        ax = axes[i]
        ax.set_facecolor('white')
        
        # Get data
        x = ds.x.values
        z = ds.z.values
        rho = ds.rho.isel(t=idx).values
        u = ds.u.isel(t=idx).values
        w = ds.w.isel(t=idx).values
        t_val = ds.t.values[idx]
        
        # Create meshgrid
        X, Z = np.meshgrid(x, z)
        
        # Density contour plot
        cf = ax.contourf(X, Z, rho, levels=30, cmap='viridis', 
                        vmin=vmin, vmax=vmax, extend='both')
        
        # Add contour lines
        ax.contour(X, Z, rho, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        # Velocity quiver plot (subsample for clarity)
        skip = 8
        ax.quiver(X[::skip, ::skip], Z[::skip, ::skip], 
                 u[::skip, ::skip], w[::skip, ::skip],
                 alpha=0.7, scale=20, width=0.002, color='white')
        
        # Formatting
        ax.set_xlabel(r'$\mathbf{x}$ [m]', fontweight='bold', fontsize=16)
        ax.set_ylabel(r'$\mathbf{z}$ [m]', fontweight='bold', fontsize=16)
        ax.set_title(f't = {t_val:.2f} s', fontsize=14)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', labelsize=14, width=1.5)
        
        # Make tick labels bold
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
    
    # Add single colorbar on the right with more space
    cbar_ax = fig.add_axes([0.94, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(cf, cax=cbar_ax)
    cbar.set_label(r'$\boldsymbol{\rho}$ [kg/m$^{\mathbf{3}}$]', 
                   fontsize=16, fontweight='bold', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=14, width=1.5)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Add velocity scale legend in bottom center to avoid overlap
    fig.text(0.5, 0.01, 'Velocity Scale: → = 1 m/s', 
            fontsize=12, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
    
    # Adjust layout with more room for colorbar
    plt.subplots_adjust(left=0.08, right=0.92, top=0.97, bottom=0.08, 
                       hspace=0.18, wspace=0.2)
    
    # Save figures
    base_name = f"density_quiver_{scenario_name.lower().replace(' ', '_')}"
    for fmt in ['eps', 'png', 'pdf']:
        filepath = output_dir / f"{base_name}.{fmt}"
        fig.savefig(filepath, format=fmt, transparent=False, facecolor='white', edgecolor='none')
    
    plt.close(fig)
    print(f"  ✓ Saved density+quiver plots for {scenario_name}")


def calculate_temporal_evolution_metrics(ds):
    """
    Calculate temporal evolution of complexity metrics.
    """
    n_times = len(ds.t)
    times = ds.t.values
    
    metrics = {
        'time': times,
        'density_entropy': [],
        'density_complexity': [],
        'vorticity_entropy': [],
        'vorticity_complexity': [],
        'mixing_efficiency': []
    }
    
    for i in range(n_times):
        # Density metrics
        rho = ds.rho.isel(t=i).values
        metrics['density_entropy'].append(round(calculate_spatial_entropy(rho), 3))
        metrics['density_complexity'].append(round(calculate_complexity_index(rho), 3))
        
        # Vorticity metrics
        vort = ds.vorticity_z.isel(t=i).values
        metrics['vorticity_entropy'].append(round(calculate_spatial_entropy(vort), 3))
        metrics['vorticity_complexity'].append(round(calculate_complexity_index(vort), 3))
        
        # Mixing efficiency (simplified metric)
        rho_std = np.std(rho)
        vort_rms = np.sqrt(np.mean(vort**2))
        mixing_eff = rho_std * vort_rms
        metrics['mixing_efficiency'].append(round(mixing_eff, 3))
    
    return metrics


def save_statistics(stats_dict, output_dir):
    """
    Save comprehensive statistics to text file with theoretical interpretations.
    """
    filepath = output_dir / 'comprehensive_statistics.txt'
    
    with open(filepath, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE STATISTICAL ANALYSIS OF KELVIN-HELMHOLTZ SIMULATIONS\n")
        f.write("="*80 + "\n\n")
        f.write("Author: Sandy H. S. Herho\n")
        f.write("Email: sandy.herho@email.ucr.edu\n")
        f.write("="*80 + "\n\n")
        
        # Theoretical Background
        f.write("THEORETICAL FRAMEWORK\n")
        f.write("="*60 + "\n\n")
        f.write("The Kelvin-Helmholtz instability represents a fundamental mechanism for\n")
        f.write("mixing and energy transfer in stratified shear flows. The analysis employs\n")
        f.write("multiple complexity measures to quantify the evolution of coherent structures:\n\n")
        
        f.write("1. Shannon Entropy (H):\n")
        f.write("   H = -Σ p_i log(p_i)\n")
        f.write("   Measures information content and disorder in spatial distribution.\n")
        f.write("   Higher values indicate more complex spatial patterns.\n\n")
        
        f.write("2. Complexity Index (CI):\n")
        f.write("   CI = w₁H + w₂∇σ + w₃σ_norm + w₄log(1+|κ|)\n")
        f.write("   Weighted combination of entropy, gradient variability,\n")
        f.write("   normalized standard deviation, and kurtosis measure.\n\n")
        
        f.write("3. Mixing Efficiency (Γ):\n")
        f.write("   Γ ∝ σ_ρ × ω_rms\n")
        f.write("   Product of density variance and vorticity RMS,\n")
        f.write("   indicating effectiveness of turbulent mixing.\n\n")
        
        f.write("="*60 + "\n\n")
        
        # Write statistics for each scenario
        for scenario in stats_dict:
            if scenario == 'comparisons':
                continue
                
            f.write(f"\n{'='*60}\n")
            f.write(f"SCENARIO: {scenario.upper()}\n")
            f.write(f"{'='*60}\n\n")
            
            # Density field statistics
            if 'density' in stats_dict[scenario]:
                f.write("DENSITY FIELD ANALYSIS\n")
                f.write("-"*40 + "\n")
                
                for time_key in stats_dict[scenario]['density']:
                    f.write(f"\n{time_key}:\n")
                    
                    if 'basic_stats' in stats_dict[scenario]['density'][time_key]:
                        basic_stats = stats_dict[scenario]['density'][time_key]['basic_stats']
                        f.write(f"  Basic Statistics:\n")
                        f.write(f"    Mean:     {basic_stats['mean']:.3f} kg/m³\n")
                        f.write(f"    Std Dev:  {basic_stats['std']:.3f} kg/m³\n")
                        f.write(f"    Min:      {basic_stats['min']:.3f} kg/m³\n")
                        f.write(f"    Max:      {basic_stats['max']:.3f} kg/m³\n")
                        f.write(f"    Median:   {basic_stats['median']:.3f} kg/m³\n")
                        f.write(f"    Skewness: {basic_stats['skewness']:.3f}\n")
                        f.write(f"    Kurtosis: {basic_stats['kurtosis']:.3f}\n")
                    
                    if 'entropy' in stats_dict[scenario]['density'][time_key]:
                        f.write(f"    Spatial Entropy: {stats_dict[scenario]['density'][time_key]['entropy']:.3f}\n")
                    
                    if 'complexity' in stats_dict[scenario]['density'][time_key]:
                        f.write(f"    Complexity Index: {stats_dict[scenario]['density'][time_key]['complexity']:.3f}\n")
                    
                    if 'normality_tests' in stats_dict[scenario]['density'][time_key]:
                        f.write(f"\n  Normality Tests:\n")
                        for test in ['shapiro', 'anderson', 'jarque_bera', 'dagostino']:
                            if test in stats_dict[scenario]['density'][time_key]['normality_tests']:
                                f.write(f"    {stats_dict[scenario]['density'][time_key]['normality_tests'][test]['interpretation']}\n")
            
            # Vorticity field statistics
            if 'vorticity' in stats_dict[scenario]:
                f.write("\n\nVORTICITY FIELD ANALYSIS\n")
                f.write("-"*40 + "\n")
                
                for time_key in stats_dict[scenario]['vorticity']:
                    f.write(f"\n{time_key}:\n")
                    
                    if 'basic_stats' in stats_dict[scenario]['vorticity'][time_key]:
                        basic_stats = stats_dict[scenario]['vorticity'][time_key]['basic_stats']
                        f.write(f"  Basic Statistics:\n")
                        f.write(f"    Mean:     {basic_stats['mean']:.3f} s⁻¹\n")
                        f.write(f"    Std Dev:  {basic_stats['std']:.3f} s⁻¹\n")
                        f.write(f"    Min:      {basic_stats['min']:.3f} s⁻¹\n")
                        f.write(f"    Max:      {basic_stats['max']:.3f} s⁻¹\n")
                        f.write(f"    Median:   {basic_stats['median']:.3f} s⁻¹\n")
                        f.write(f"    Skewness: {basic_stats['skewness']:.3f}\n")
                        f.write(f"    Kurtosis: {basic_stats['kurtosis']:.3f}\n")
                    
                    if 'entropy' in stats_dict[scenario]['vorticity'][time_key]:
                        f.write(f"    Spatial Entropy: {stats_dict[scenario]['vorticity'][time_key]['entropy']:.3f}\n")
                    
                    if 'complexity' in stats_dict[scenario]['vorticity'][time_key]:
                        f.write(f"    Complexity Index: {stats_dict[scenario]['vorticity'][time_key]['complexity']:.3f}\n")
            
            # Temporal evolution metrics
            if 'temporal_evolution' in stats_dict[scenario]:
                f.write("\n\nTEMPORAL EVOLUTION ANALYSIS\n")
                f.write("-"*40 + "\n")
                
                te = stats_dict[scenario]['temporal_evolution']
                f.write(f"\n  Maximum Complexity Achieved:\n")
                f.write(f"    Density:   {max(te['density_complexity']):.3f} at t={te['time'][te['density_complexity'].index(max(te['density_complexity']))]:.3f}s\n")
                f.write(f"    Vorticity: {max(te['vorticity_complexity']):.3f} at t={te['time'][te['vorticity_complexity'].index(max(te['vorticity_complexity']))]:.3f}s\n")
                
                f.write(f"\n  Entropy Growth Rate:\n")
                if len(te['density_entropy']) > 1:
                    growth_rate = (te['density_entropy'][-1] - te['density_entropy'][0]) / (te['time'][-1] - te['time'][0])
                    f.write(f"    Density:   {growth_rate:.3f} per second\n")
                if len(te['vorticity_entropy']) > 1:
                    growth_rate = (te['vorticity_entropy'][-1] - te['vorticity_entropy'][0]) / (te['time'][-1] - te['time'][0])
                    f.write(f"    Vorticity: {growth_rate:.3f} per second\n")
                
                f.write(f"\n  Mean Mixing Efficiency: {np.mean(te['mixing_efficiency']):.3f}\n")
        
        # Complexity comparison across scenarios
        if 'complexity_comparison' in stats_dict:
            f.write(f"\n\n{'='*60}\n")
            f.write("COMPLEXITY COMPARISON ACROSS SCENARIOS\n")
            f.write(f"{'='*60}\n\n")
            
            cc = stats_dict['complexity_comparison']
            
            f.write("Normalized Complexity Indices (0-1 scale):\n")
            f.write("-"*40 + "\n")
            for scenario, values in cc.items():
                f.write(f"\n{scenario}:\n")
                f.write(f"  Early Stage (t<25%):   {values['early']:.3f}\n")
                f.write(f"  Growth Stage (25-50%): {values['growth']:.3f}\n")
                f.write(f"  Mature Stage (50-75%): {values['mature']:.3f}\n")
                f.write(f"  Late Stage (t>75%):    {values['late']:.3f}\n")
                f.write(f"  Overall Weighted:      {values['overall']:.3f}\n")
        
        # Inter-scenario comparisons
        if 'comparisons' in stats_dict:
            f.write(f"\n\n{'='*60}\n")
            f.write("INTER-SCENARIO STATISTICAL COMPARISONS\n")
            f.write(f"{'='*60}\n\n")
            
            if 'density' in stats_dict['comparisons']:
                f.write("DENSITY FIELD COMPARISONS\n")
                f.write("-"*40 + "\n\n")
                
                if 'kruskal_wallis' in stats_dict['comparisons']['density']:
                    f.write(stats_dict['comparisons']['density']['kruskal_wallis']['interpretation'] + "\n\n")
                
                if 'mann_whitney_pairwise' in stats_dict['comparisons']['density']:
                    f.write("Pairwise Mann-Whitney U Tests:\n")
                    for pair, result in stats_dict['comparisons']['density']['mann_whitney_pairwise'].items():
                        f.write(f"  {pair}: {result['interpretation']}\n")
            
            if 'vorticity' in stats_dict['comparisons']:
                f.write("\n\nVORTICITY FIELD COMPARISONS\n")
                f.write("-"*40 + "\n\n")
                
                if 'kruskal_wallis' in stats_dict['comparisons']['vorticity']:
                    f.write(stats_dict['comparisons']['vorticity']['kruskal_wallis']['interpretation'] + "\n\n")
                
                if 'mann_whitney_pairwise' in stats_dict['comparisons']['vorticity']:
                    f.write("Pairwise Mann-Whitney U Tests:\n")
                    for pair, result in stats_dict['comparisons']['vorticity']['mann_whitney_pairwise'].items():
                        f.write(f"  {pair}: {result['interpretation']}\n")
        
        # Physical interpretations
        f.write(f"\n\n{'='*60}\n")
        f.write("PHYSICAL INTERPRETATIONS AND CONCLUSIONS\n")
        f.write(f"{'='*60}\n\n")
        
        f.write("1. INSTABILITY EVOLUTION:\n")
        f.write("   The Kelvin-Helmholtz instability undergoes distinct evolutionary stages:\n")
        f.write("   - Linear growth: Exponential amplification of perturbations (Ri < 0.25)\n")
        f.write("   - Nonlinear saturation: Billow formation and vortex roll-up\n")
        f.write("   - Secondary instabilities: Three-dimensional breakdown (in 2D: pairing)\n")
        f.write("   - Turbulent decay: Energy cascade and viscous dissipation\n\n")
        
        f.write("2. RICHARDSON NUMBER EFFECTS:\n")
        f.write("   Ri = g∆ρδ/(ρ₀∆U²) controls shear vs. stratification balance:\n")
        f.write("   - Ri < 0.25: Unstable, vigorous mixing\n")
        f.write("   - Ri ≈ 0.25: Marginal stability, intermittent turbulence\n")
        f.write("   - Ri > 0.25: Stable stratification suppresses vertical motion\n\n")
        
        f.write("3. REYNOLDS NUMBER DEPENDENCE:\n")
        f.write("   Re = ∆Uδ/ν determines the scale separation:\n")
        f.write("   - Re < 1000: Laminar billows, direct dissipation\n")
        f.write("   - Re ~ 1000-5000: Transitional, secondary instabilities\n")
        f.write("   - Re > 5000: Fully turbulent, extended inertial range\n\n")
        
        f.write("4. MIXING EFFICIENCY:\n")
        f.write("   The flux Richardson number Rf = Ri·Pr/(1+Pr) where Pr = ν/κ\n")
        f.write("   Maximum mixing efficiency Γ ≈ 0.2 occurs at Ri ≈ 0.15-0.20\n")
        f.write("   Forced scenarios maintain higher sustained mixing rates\n\n")
        
        f.write("5. ENTROPY PRODUCTION:\n")
        f.write("   Spatial entropy increases monotonically in free evolution\n")
        f.write("   Forced turbulence reaches statistical equilibrium\n")
        f.write("   Double shear layers show enhanced entropy from vortex interactions\n")
        f.write("   Rotation modifies entropy growth through geostrophic adjustment\n\n")
        
        f.write("6. GEOPHYSICAL IMPLICATIONS:\n")
        f.write("   - Atmospheric applications: Cloud-top entrainment, CAT generation\n")
        f.write("   - Oceanic applications: Thermocline erosion, mixed layer deepening\n")
        f.write("   - Climate modeling: Parameterization of subgrid-scale mixing\n")
        f.write("   - Energy budget: Conversion of mean shear to turbulent kinetic energy\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF STATISTICAL ANALYSIS\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Statistics saved to {filepath}")


def calculate_complexity_comparison(all_stats):
    """
    Calculate normalized complexity comparison across scenarios and stages.
    """
    comparison = {}
    
    for scenario in all_stats:
        if scenario == 'comparisons' or 'temporal_evolution' not in all_stats[scenario]:
            continue
        
        te = all_stats[scenario]['temporal_evolution']
        n_times = len(te['time'])
        
        # Define stages
        stages = {
            'early': slice(0, n_times//4),
            'growth': slice(n_times//4, n_times//2),
            'mature': slice(n_times//2, 3*n_times//4),
            'late': slice(3*n_times//4, n_times)
        }
        
        stage_complexity = {}
        for stage_name, stage_slice in stages.items():
            # Average complexity in each stage
            density_comp = np.mean(te['density_complexity'][stage_slice])
            vorticity_comp = np.mean(te['vorticity_complexity'][stage_slice])
            stage_complexity[stage_name] = round((density_comp + vorticity_comp) / 2, 3)
        
        # Overall weighted complexity (emphasizing mature stage)
        weights = {'early': 0.15, 'growth': 0.25, 'mature': 0.35, 'late': 0.25}
        overall = sum(stage_complexity[stage] * weights[stage] for stage in stages.keys())
        stage_complexity['overall'] = round(overall, 3)
        
        comparison[scenario] = stage_complexity
    
    # Normalize to 0-1 scale
    max_complexity = max(max(sc.values()) for sc in comparison.values())
    for scenario in comparison:
        for stage in comparison[scenario]:
            comparison[scenario][stage] = round(comparison[scenario][stage] / max_complexity, 3)
    
    return comparison


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("KELVIN-HELMHOLTZ 2D INSTABILITY ANALYSIS")
    print("Author: Sandy H. S. Herho")
    print("="*80 + "\n")
    
    # Define paths
    output_dir = Path('../outputs')
    fig_dir = Path('../figs')
    stats_dir = Path('../stats')
    
    # Create directories if they don't exist
    fig_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    # Define scenarios
    scenarios = {
        'basic_shear': 'Basic Shear Layer',
        'double_shear': 'Double Shear Layer', 
        'rotating': 'Rotating KH Instability',
        'forced': 'Forced KH Turbulence'
    }
    
    # Initialize statistics dictionary
    all_stats = {}
    comparison_data = {'density': {}, 'vorticity': {}}
    
    print("Processing scenarios...")
    print("-"*40)
    
    # Process each scenario
    for scenario_key, scenario_name in scenarios.items():
        print(f"\nProcessing: {scenario_name}")
        
        # Find the NetCDF file
        nc_files = list(output_dir.glob(f"*{scenario_key}*.nc"))
        if not nc_files:
            print(f"  ⚠ No NetCDF file found for {scenario_key}")
            continue
        
        # Load dataset
        ds = xr.open_dataset(nc_files[0])
        print(f"  Loaded: {nc_files[0].name}")
        
        # Select time indices (4 snapshots evenly distributed)
        n_times = len(ds.t)
        time_indices = [int(i * n_times / 4) for i in range(4)]
        
        # Create density+quiver figures only
        print(f"  Creating density+quiver figures...")
        create_density_quiver_figure(ds, scenario_name, time_indices, fig_dir)
        
        # Perform statistical analysis
        print(f"  Performing statistical analysis...")
        scenario_stats = {'density': {}, 'vorticity': {}}
        
        # Analyze each time snapshot
        for idx in time_indices:
            t_val = ds.t.values[idx]
            time_key = f"Time t={t_val:.2f}s"
            
            # Density statistics
            rho_data = ds.rho.isel(t=idx).values
            rho_stats = perform_normality_tests(rho_data)
            rho_stats['entropy'] = round(calculate_spatial_entropy(rho_data), 3)
            rho_stats['complexity'] = round(calculate_complexity_index(rho_data), 3)
            scenario_stats['density'][time_key] = rho_stats
            
            # Vorticity statistics (for analysis only, not plotting)
            vort_data = ds.vorticity_z.isel(t=idx).values
            vort_stats = perform_normality_tests(vort_data)
            vort_stats['entropy'] = round(calculate_spatial_entropy(vort_data), 3)
            vort_stats['complexity'] = round(calculate_complexity_index(vort_data), 3)
            scenario_stats['vorticity'][time_key] = vort_stats
        
        # Calculate temporal evolution metrics
        print(f"  Calculating temporal evolution metrics...")
        temporal_evolution = calculate_temporal_evolution_metrics(ds)
        scenario_stats['temporal_evolution'] = temporal_evolution
        
        # Store for comparison
        comparison_data['density'][scenario_name] = ds.rho.values
        comparison_data['vorticity'][scenario_name] = ds.vorticity_z.values
        
        all_stats[scenario_name] = scenario_stats
        
        # Close dataset
        ds.close()
        print(f"  ✓ Completed {scenario_name}")
    
    # Perform inter-scenario comparisons
    print("\n" + "-"*40)
    print("Performing inter-scenario comparisons...")
    
    comparisons = {}
    
    # Compare density fields
    if len(comparison_data['density']) > 1:
        comparisons['density'] = perform_nonparametric_comparison(comparison_data['density'])
        print("  ✓ Density field comparisons completed")
    
    # Compare vorticity fields
    if len(comparison_data['vorticity']) > 1:
        comparisons['vorticity'] = perform_nonparametric_comparison(comparison_data['vorticity'])
        print("  ✓ Vorticity field comparisons completed")
    
    all_stats['comparisons'] = comparisons
    
    # Calculate complexity comparison
    print("  Calculating complexity comparison across scenarios...")
    all_stats['complexity_comparison'] = calculate_complexity_comparison(all_stats)
    
    # Save all statistics
    print("\nSaving statistics...")
    save_statistics(all_stats, stats_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Figures saved to: {fig_dir}")
    print(f"Statistics saved to: {stats_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
