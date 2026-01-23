# -*- coding: utf-8 -*-
"""
audit_normalization.py - Feature Normalization and Clipping Audit

Audits normalization and clipping of features at each pipeline stage:
- HMM: RobustScaler + clip [-5, 5]
- MAE: Same normalized features as HMM
- TQC: Same normalized features via FoundationFeatureExtractor

Provides detailed text reports and visualizations (histograms, boxplots, heatmaps).
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    # Create a simple sns-like object for compatibility
    class SimpleSNS:
        @staticmethod
        def set_style(style):
            pass
    sns = SimpleSNS()

from pathlib import Path
from typing import Tuple, Dict, Optional, List
import pickle

from sklearn.preprocessing import RobustScaler
import torch

from src.data_engineering.manager import RegimeDetector, DataManager
from src.models.foundation import CryptoMAE

# Set style for plots
if HAS_SEABORN:
    sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def compute_feature_statistics(df: pd.DataFrame, stage_name: str) -> pd.DataFrame:
    """
    Compute comprehensive statistics for each feature.
    
    Args:
        df: DataFrame with features
        stage_name: Name of the pipeline stage (for reporting)
        
    Returns:
        DataFrame with statistics (min, max, mean, std, percentiles)
    """
    stats = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        stats.append({
            'Feature': col,
            'Count': len(series),
            'Min': series.min(),
            'Max': series.max(),
            'Mean': series.mean(),
            'Std': series.std(),
            'P25': series.quantile(0.25),
            'P50': series.quantile(0.50),
            'P75': series.quantile(0.75),
            'P95': series.quantile(0.95),
            'P99': series.quantile(0.99),
            'NaN_Count': df[col].isna().sum(),
            'NaN_Pct': df[col].isna().sum() / len(df) * 100
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df['Stage'] = stage_name
    return stats_df


def detect_clipping(df: pd.DataFrame, clip_bounds: Tuple[float, float] = (-5, 5)) -> pd.DataFrame:
    """
    Detect percentage of clipped values per feature.
    
    Args:
        df: DataFrame with features
        clip_bounds: (min, max) clipping bounds
        
    Returns:
        DataFrame with clipping statistics
    """
    clip_min, clip_max = clip_bounds
    results = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        
        n_clipped_min = (series <= clip_min).sum()
        n_clipped_max = (series >= clip_max).sum()
        n_clipped_total = n_clipped_min + n_clipped_max
        pct_clipped = n_clipped_total / len(series) * 100
        
        results.append({
            'Feature': col,
            'Clipped_Min': n_clipped_min,
            'Clipped_Max': n_clipped_max,
            'Clipped_Total': n_clipped_total,
            'Pct_Clipped': pct_clipped,
            'At_Min': (series == clip_min).sum(),
            'At_Max': (series == clip_max).sum()
        })
    
    return pd.DataFrame(results)


def detect_outliers(df: pd.DataFrame, method: str = 'iqr', factor: float = 3.0) -> pd.DataFrame:
    """
    Detect outliers that are NOT clipped.
    
    Args:
        df: DataFrame with features
        method: 'iqr' or 'zscore'
        factor: Multiplier for IQR or Z-score threshold
        
    Returns:
        DataFrame with outlier statistics
    """
    results = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)]
        else:  # zscore
            mean = series.mean()
            std = series.std()
            if std == 0:
                outliers = pd.Series([], dtype=series.dtype)
            else:
                z_scores = np.abs((series - mean) / std)
                outliers = series[z_scores > factor]
        
        results.append({
            'Feature': col,
            'Outliers_Count': len(outliers),
            'Outliers_Pct': len(outliers) / len(series) * 100,
            'Outliers_Min': outliers.min() if len(outliers) > 0 else np.nan,
            'Outliers_Max': outliers.max() if len(outliers) > 0 else np.nan
        })
    
    return pd.DataFrame(results)


def compare_normalizations(
    stats_hmm: pd.DataFrame,
    stats_mae: Optional[pd.DataFrame] = None,
    stats_tqc: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compare statistics between pipeline stages.
    
    Args:
        stats_hmm: Statistics from HMM stage
        stats_mae: Statistics from MAE stage (optional)
        stats_tqc: Statistics from TQC stage (optional)
        
    Returns:
        DataFrame with comparison
    """
    comparison = []
    
    # Get common features
    hmm_features = set(stats_hmm['Feature'].unique())
    
    for feat in hmm_features:
        hmm_row = stats_hmm[stats_hmm['Feature'] == feat].iloc[0]
        
        row = {
            'Feature': feat,
            'HMM_Mean': hmm_row['Mean'],
            'HMM_Std': hmm_row['Std'],
            'HMM_Min': hmm_row['Min'],
            'HMM_Max': hmm_row['Max']
        }
        
        if stats_mae is not None:
            mae_rows = stats_mae[stats_mae['Feature'] == feat]
            if len(mae_rows) > 0:
                mae_row = mae_rows.iloc[0]
                row['MAE_Mean'] = mae_row['Mean']
                row['MAE_Std'] = mae_row['Std']
                row['Mean_Diff'] = abs(hmm_row['Mean'] - mae_row['Mean'])
            else:
                row['MAE_Mean'] = np.nan
                row['MAE_Std'] = np.nan
                row['Mean_Diff'] = np.nan
        
        if stats_tqc is not None:
            tqc_rows = stats_tqc[stats_tqc['Feature'] == feat]
            if len(tqc_rows) > 0:
                tqc_row = tqc_rows.iloc[0]
                row['TQC_Mean'] = tqc_row['Mean']
                row['TQC_Std'] = tqc_row['Std']
            else:
                row['TQC_Mean'] = np.nan
                row['TQC_Std'] = np.nan
        
        comparison.append(row)
    
    return pd.DataFrame(comparison)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_visualizations(
    df_dict: Dict[str, pd.DataFrame],
    clipping_dict: Dict[str, pd.DataFrame],
    output_dir: str = 'results/normalization_audit'
) -> List[str]:
    """
    Create visualizations for normalization audit.
    
    Args:
        df_dict: Dict with stage names as keys and DataFrames as values
        clipping_dict: Dict with clipping statistics per stage
        output_dir: Output directory for plots
        
    Returns:
        List of generated plot file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_files = []
    
    # 1. Histogrammes: Distribution avant/après normalisation
    for stage_name, df in df_dict.items():
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]  # Limit to 10 features
        
        if len(numeric_cols) == 0:
            continue
        
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            series = df[col].dropna()
            ax.hist(series, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(series.mean(), color='r', linestyle='--', label=f'Mean: {series.mean():.2f}')
            ax.axvline(-5, color='orange', linestyle=':', label='Clip: -5')
            ax.axvline(5, color='orange', linestyle=':', label='Clip: 5')
            ax.set_title(f'{col}\n({stage_name})')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        filename = os.path.join(output_dir, f'histograms_{stage_name}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files.append(filename)
        print(f"  Saved: {filename}")
    
    # 2. Boxplots: Comparaison entre étapes
    if len(df_dict) > 1:
        # Select common features
        all_features = set()
        for df in df_dict.values():
            all_features.update(df.select_dtypes(include=[np.number]).columns)
        
        common_features = list(all_features)[:10]  # Limit to 10
        
        for feat in common_features:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            data_to_plot = []
            labels = []
            
            for stage_name, df in df_dict.items():
                if feat in df.columns:
                    series = df[feat].dropna()
                    if len(series) > 0:
                        data_to_plot.append(series.values)
                        labels.append(stage_name)
            
            if len(data_to_plot) > 0:
                ax.boxplot(data_to_plot, labels=labels)
                ax.axhline(-5, color='orange', linestyle=':', label='Clip: -5')
                ax.axhline(5, color='orange', linestyle=':', label='Clip: 5')
                ax.set_title(f'Boxplot: {feat}')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                filename = os.path.join(output_dir, f'boxplot_{feat.replace("/", "_")}.png')
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close()
                plot_files.append(filename)
    
    # 3. Heatmap: Pourcentage de clipping par feature
    if clipping_dict:
        # Combine clipping data
        all_clipping = []
        for stage_name, clipping_df in clipping_dict.items():
            if len(clipping_df) > 0:
                clipping_df = clipping_df.copy()
                clipping_df['Stage'] = stage_name
                all_clipping.append(clipping_df)
        
        if all_clipping:
            combined_clipping = pd.concat(all_clipping, ignore_index=True)
            
            # Pivot for heatmap
            pivot_data = combined_clipping.pivot_table(
                values='Pct_Clipped',
                index='Feature',
                columns='Stage',
                aggfunc='mean'
            )
            
            if len(pivot_data) > 0:
                fig, ax = plt.subplots(figsize=(12, max(8, len(pivot_data) * 0.3)))
                if HAS_SEABORN:
                    sns.heatmap(
                        pivot_data,
                        annot=True,
                        fmt='.1f',
                        cmap='YlOrRd',
                        cbar_kws={'label': '% Clipped'},
                        ax=ax
                    )
                else:
                    # Fallback to matplotlib imshow
                    im = ax.imshow(pivot_data.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
                    ax.set_xticks(range(len(pivot_data.columns)))
                    ax.set_xticklabels(pivot_data.columns)
                    ax.set_yticks(range(len(pivot_data.index)))
                    ax.set_yticklabels(pivot_data.index)
                    plt.colorbar(im, ax=ax, label='% Clipped')
                    # Add text annotations
                    for i in range(len(pivot_data.index)):
                        for j in range(len(pivot_data.columns)):
                            text = ax.text(j, i, f'{pivot_data.iloc[i, j]:.1f}',
                                         ha="center", va="center", color="black", fontsize=8)
                ax.set_title('Clipping Percentage by Feature and Stage')
                ax.set_xlabel('Stage')
                ax.set_ylabel('Feature')
                
                plt.tight_layout()
                filename = os.path.join(output_dir, 'clipping_heatmap.png')
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close()
                plot_files.append(filename)
                print(f"  Saved: {filename}")
    
    return plot_files


def create_hmm_entropy_visualizations(
    df: pd.DataFrame,
    entropy_results: Dict,
    prob_results: Optional[Dict] = None,
    output_dir: str = 'results/normalization_audit'
) -> List[str]:
    """
    Create specialized HMM entropy visualizations.
    
    Args:
        df: DataFrame with HMM data
        entropy_results: Results from audit_hmm_entropy()
        prob_results: Optional results from audit_hmm_probabilities()
        output_dir: Output directory
        
    Returns:
        List of generated plot file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_files = []
    
    if 'error' in entropy_results:
        return plot_files
    
    entropy_series = entropy_results['entropy_series']
    
    # 1. Fog of War Chart (Critical)
    if 'BTC_Close' in df.columns:
        print("  Creating Fog of War Chart...")
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Align data
        valid_idx = entropy_series.index.intersection(df.index)
        prices = df.loc[valid_idx, 'BTC_Close']
        entropy_aligned = entropy_series.loc[valid_idx]
        
        # Get dominant state if available
        dominant_state = None
        if prob_results and 'dominant_states' in prob_results:
            dominant_state = prob_results['dominant_states'].loc[valid_idx]
        
        # Create background colors based on entropy
        for i in range(len(entropy_aligned) - 1):
            idx = entropy_aligned.index[i]
            entropy_val = entropy_aligned.iloc[i]
            
            if entropy_val < 0.3:  # Certainty - color by state
                if dominant_state is not None and idx in dominant_state.index:
                    state = dominant_state.loc[idx]
                    # State 0,1 = red (bearish), State 2,3 = green (bullish)
                    color = 'lightcoral' if state <= 1 else 'lightgreen'
                    alpha = 0.3
                else:
                    color = 'lightblue'
                    alpha = 0.2
            elif entropy_val > 0.7:  # Gray zone
                color = 'gray'
                alpha = 0.4
            else:  # Moderate
                color = 'lightyellow'
                alpha = 0.2
            
            ax.axvspan(i, i+1, color=color, alpha=alpha, zorder=0)
        
        # Plot price line
        ax.plot(prices.values, color='black', linewidth=1.5, label='BTC Price', zorder=1)
        
        ax.set_xlabel('Time Index')
        ax.set_ylabel('BTC Price (USD)', color='black')
        ax.set_title('Fog of War Chart: HMM Uncertainty Overlay\n'
                     'Green/Red = Certainty (Entropy < 0.3), Gray = Uncertainty (Entropy > 0.7)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        filename = os.path.join(output_dir, 'fog_of_war_chart.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files.append(filename)
        print(f"  Saved: {filename}")
    
    # 2. Timeline Entropie HMM
    print("  Creating Entropy Timeline...")
    fig, ax1 = plt.subplots(figsize=(16, 6))
    
    # Plot entropy
    ax1.plot(entropy_series.values, color='blue', alpha=0.7, linewidth=1, label='HMM Entropy')
    ax1.axhline(0.3, color='green', linestyle='--', alpha=0.5, label='Low (Certainty)')
    ax1.axhline(0.7, color='orange', linestyle='--', alpha=0.5, label='High (Uncertainty)')
    ax1.fill_between(range(len(entropy_series)), 0, entropy_series.values, 
                     where=(entropy_series.values < 0.3), alpha=0.2, color='green', label='Certainty Zone')
    ax1.fill_between(range(len(entropy_series)), 0, entropy_series.values,
                     where=(entropy_series.values > 0.7), alpha=0.2, color='red', label='Gray Zone')
    
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('HMM Entropy (Normalized)', color='blue')
    ax1.set_ylim([0, 1])
    ax1.set_title('HMM Entropy Timeline')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Overlay returns if available
    if 'BTC_Close' in df.columns:
        valid_idx = entropy_series.index.intersection(df.index)
        prices = df.loc[valid_idx, 'BTC_Close']
        returns = np.zeros(len(prices))
        returns[1:] = np.log(prices.values[1:] / prices.values[:-1])
        
        ax2 = ax1.twinx()
        ax2.plot(returns, color='gray', alpha=0.4, linewidth=0.5, label='Returns')
        ax2.set_ylabel('Log Returns', color='gray')
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, 'entropy_timeline.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    plot_files.append(filename)
    print(f"  Saved: {filename}")
    
    # 3. Distribution Entropie par État Dominant
    if prob_results and 'dominant_states' in prob_results:
        print("  Creating Entropy Distribution by Dominant State...")
        dominant_states = prob_results['dominant_states']
        valid_idx = entropy_series.index.intersection(dominant_states.index)
        entropy_by_state = entropy_series.loc[valid_idx]
        states_aligned = dominant_states.loc[valid_idx]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by state
        data_by_state = []
        labels = []
        for state in sorted(states_aligned.unique()):
            state_entropy = entropy_by_state[states_aligned == state]
            if len(state_entropy) > 0:
                data_by_state.append(state_entropy.values)
                labels.append(f'State {int(state)}')
        
        if data_by_state:
            ax.boxplot(data_by_state, labels=labels)
            ax.axhline(0.3, color='green', linestyle='--', alpha=0.5, label='Low Threshold')
            ax.axhline(0.7, color='orange', linestyle='--', alpha=0.5, label='High Threshold')
            ax.set_ylabel('HMM Entropy')
            ax.set_title('Entropy Distribution by Dominant HMM State')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = os.path.join(output_dir, 'entropy_by_state.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            plot_files.append(filename)
            print(f"  Saved: {filename}")
    
    # 4. Heatmap Entropie vs Probabilités
    if prob_results and 'probs_df' in prob_results:
        print("  Creating Entropy vs Probabilities Heatmap...")
        probs_df = prob_results['probs_df']
        valid_idx = entropy_series.index.intersection(probs_df.index)
        entropy_aligned = entropy_series.loc[valid_idx]
        probs_aligned = probs_df.loc[valid_idx]
        
        if len(entropy_aligned) > 100:
            # Sample for performance
            sample_size = min(5000, len(entropy_aligned))
            sample_idx = np.random.choice(len(entropy_aligned), sample_size, replace=False)
            entropy_sample = entropy_aligned.iloc[sample_idx]
            probs_sample = probs_aligned.iloc[sample_idx]
        else:
            entropy_sample = entropy_aligned
            probs_sample = probs_aligned
        
        max_prob = probs_sample.max(axis=1)
        
        # Create 2D histogram
        fig, ax = plt.subplots(figsize=(10, 8))
        hb = ax.hexbin(max_prob, entropy_sample, gridsize=30, cmap='YlOrRd', mincnt=1)
        ax.set_xlabel('Maximum Probability (Dominant State)')
        ax.set_ylabel('HMM Entropy')
        ax.set_title('Entropy vs Maximum Probability Heatmap')
        plt.colorbar(hb, ax=ax, label='Count')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, 'entropy_vs_probabilities_heatmap.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files.append(filename)
        print(f"  Saved: {filename}")
    
    return plot_files


def create_hmm_correlations_heatmap(
    corr_results: Dict,
    output_dir: str = 'results/normalization_audit'
) -> List[str]:
    """
    Create correlation heatmap for HMM features.
    
    Args:
        corr_results: Results from analyze_hmm_correlations()
        output_dir: Output directory
        
    Returns:
        List of generated plot file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_files = []
    
    if 'error' in corr_results:
        return plot_files
    
    # Normalized features correlation
    if 'corr_matrix_normalized' in corr_results:
        print("  Creating HMM Features Correlation Heatmap...")
        corr_matrix = corr_results['corr_matrix_normalized']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if HAS_SEABORN:
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                cbar_kws={'label': 'Correlation'},
                ax=ax
            )
        else:
            im = ax.imshow(corr_matrix.values, aspect='auto', cmap='coolwarm', 
                          vmin=-1, vmax=1, interpolation='nearest')
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax.set_yticks(range(len(corr_matrix.index)))
            ax.set_yticklabels(corr_matrix.index)
            plt.colorbar(im, ax=ax, label='Correlation')
            # Add annotations
            for i in range(len(corr_matrix.index)):
                for j in range(len(corr_matrix.columns)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('HMM Features Correlation Matrix (Normalized)')
        plt.tight_layout()
        filename = os.path.join(output_dir, 'hmm_correlations_heatmap.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files.append(filename)
        print(f"  Saved: {filename}")
    
    return plot_files


# ============================================================================
# STAGE-SPECIFIC AUDIT FUNCTIONS
# ============================================================================

def analyze_individual_hmm_features(
    hmm_raw: pd.DataFrame,
    hmm_normalized: pd.DataFrame,
    hmm_feature_cols: List[str]
) -> Dict:
    """
    Analyze each HMM feature individually with detailed statistics.
    
    Args:
        hmm_raw: Raw HMM features before normalization
        hmm_normalized: Normalized HMM features
        hmm_feature_cols: List of HMM feature column names
        
    Returns:
        Dict with per-feature statistics and analysis
    """
    per_feature_stats = {}
    
    for feat in hmm_feature_cols:
        if feat not in hmm_raw.columns or feat not in hmm_normalized.columns:
            continue
        
        raw_series = hmm_raw[feat].dropna()
        norm_series = hmm_normalized[feat].dropna()
        
        if len(raw_series) == 0 or len(norm_series) == 0:
            continue
        
        # Outlier detection for raw features
        Q1_raw = raw_series.quantile(0.25)
        Q3_raw = raw_series.quantile(0.75)
        IQR_raw = Q3_raw - Q1_raw
        outliers_raw = raw_series[(raw_series < Q1_raw - 3*IQR_raw) | (raw_series > Q3_raw + 3*IQR_raw)]
        
        # Clipping analysis for normalized
        clipping_norm = detect_clipping(pd.DataFrame({feat: norm_series}), clip_bounds=(-5, 5))
        
        per_feature_stats[feat] = {
            'raw_stats': {
                'mean': raw_series.mean(),
                'std': raw_series.std(),
                'min': raw_series.min(),
                'max': raw_series.max(),
                'p25': Q1_raw,
                'p75': Q3_raw,
                'outliers_count': len(outliers_raw),
                'outliers_pct': len(outliers_raw) / len(raw_series) * 100
            },
            'normalized_stats': {
                'mean': norm_series.mean(),
                'std': norm_series.std(),
                'min': norm_series.min(),
                'max': norm_series.max(),
                'pct_clipped': clipping_norm.iloc[0]['Pct_Clipped'] if len(clipping_norm) > 0 else 0.0
            },
            'transformation': {
                'mean_shift': norm_series.mean() - raw_series.mean(),
                'std_change': norm_series.std() / (raw_series.std() + 1e-8)
            }
        }
    
    return per_feature_stats


def audit_hmm_normalization(df: pd.DataFrame, detector: RegimeDetector) -> Dict:
    """
    Audit HMM normalization: RobustScaler + clip [-5, 5].
    Extended to analyze each feature individually.
    
    Args:
        df: DataFrame with features
        detector: RegimeDetector instance
        
    Returns:
        Dict with statistics before/after normalization and per-feature analysis
    """
    print("\n" + "=" * 70)
    print("HMM NORMALIZATION AUDIT")
    print("=" * 70)
    
    # Get HMM features (raw, before scaling)
    hmm_features_raw = detector._compute_hmm_features(df)
    hmm_feature_cols = RegimeDetector.HMM_FEATURES
    
    # Filter to only HMM features
    hmm_raw = hmm_features_raw[hmm_feature_cols].copy()
    
    print(f"\n[1/3] Raw HMM Features (before normalization):")
    stats_raw = compute_feature_statistics(hmm_raw, "HMM_Raw")
    print(f"  Features: {len(hmm_feature_cols)}")
    print(f"  Shape: {hmm_raw.shape}")
    
    # Apply RobustScaler (same as in fit_predict)
    print(f"\n[2/3] Applying RobustScaler (MIN_IQR=1.0)...")
    valid_mask = ~hmm_raw.isna().any(axis=1)
    features_valid = hmm_raw[valid_mask]
    
    if len(features_valid) < 100:
        print(f"  [WARNING] Not enough valid samples: {len(features_valid)}")
        return {'error': 'Not enough valid samples'}
    
    scaler = RobustScaler()
    scaler.fit(features_valid)
    
    # Apply MIN_IQR fix
    MIN_IQR = 1.0
    scaler.scale_ = np.maximum(scaler.scale_, MIN_IQR)
    
    features_scaled = scaler.transform(features_valid)
    features_scaled = np.clip(features_scaled, -5, 5)
    
    hmm_normalized = pd.DataFrame(
        features_scaled,
        columns=hmm_feature_cols,
        index=features_valid.index
    )
    
    print(f"\n[3/3] After normalization and clipping [-5, 5]:")
    stats_normalized = compute_feature_statistics(hmm_normalized, "HMM_Normalized")
    
    # Clipping analysis
    clipping_stats = detect_clipping(hmm_normalized, clip_bounds=(-5, 5))
    
    print("\n>>> CLIPPING ANALYSIS:")
    print("-" * 80)
    for _, row in clipping_stats.iterrows():
        status = "[OK]" if row['Pct_Clipped'] < 5 else ("[WARNING]" if row['Pct_Clipped'] < 10 else "[ALERT]")
        print(f"  {row['Feature']:20s}: {row['Pct_Clipped']:5.1f}% clipped {status}")
    
    # Individual feature analysis
    print("\n>>> INDIVIDUAL FEATURE ANALYSIS:")
    print("-" * 80)
    per_feature_stats = analyze_individual_hmm_features(hmm_raw, hmm_normalized, hmm_feature_cols)
    for feat, stats in per_feature_stats.items():
        print(f"\n  {feat}:")
        print(f"    Raw: mean={stats['raw_stats']['mean']:.4f}, std={stats['raw_stats']['std']:.4f}, "
              f"outliers={stats['raw_stats']['outliers_pct']:.1f}%")
        print(f"    Normalized: mean={stats['normalized_stats']['mean']:.4f}, "
              f"std={stats['normalized_stats']['std']:.4f}, "
              f"clipped={stats['normalized_stats']['pct_clipped']:.1f}%")
    
    return {
        'raw': hmm_raw,
        'normalized': hmm_normalized,
        'stats_raw': stats_raw,
        'stats_normalized': stats_normalized,
        'clipping': clipping_stats,
        'scaler': scaler,
        'per_feature_stats': per_feature_stats
    }


def audit_mae_normalization(
    df: pd.DataFrame,
    encoder_path: Optional[str] = None,
    device: str = "cpu"
) -> Dict:
    """
    Audit MAE normalization: verify features match HMM output.
    
    Args:
        df: DataFrame with normalized features (from parquet)
        encoder_path: Path to MAE encoder (optional)
        device: Device for MAE operations
        
    Returns:
        Dict with MAE statistics
    """
    print("\n" + "=" * 70)
    print("MAE NORMALIZATION AUDIT")
    print("=" * 70)
    
    # Features used by MAE (exclude OHLCV, Prob_, HMM_ prefix columns)
    exclude_patterns = ['_Open', '_High', '_Low', '_Close', '_Volume']
    exclude_prefixes = ['Prob_', 'HMM_']
    
    mae_feature_cols = [
        col for col in df.columns
        if not any(p in col for p in exclude_patterns)
        and not any(col.startswith(p) for p in exclude_prefixes)
        and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]
    
    mae_features = df[mae_feature_cols].copy()
    mae_features = mae_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"\n[1/2] MAE Input Features:")
    print(f"  Features: {len(mae_feature_cols)}")
    print(f"  Shape: {mae_features.shape}")
    
    stats_mae = compute_feature_statistics(mae_features, "MAE_Input")
    clipping_mae = detect_clipping(mae_features, clip_bounds=(-5, 5))
    
    # Check if encoder is available for embedding analysis
    embeddings_stats = None
    if encoder_path and os.path.exists(encoder_path):
        print(f"\n[2/2] Analyzing MAE Embeddings...")
        try:
            # Load encoder
            checkpoint = torch.load(encoder_path, map_location=device)
            
            # Create model
            input_dim = len(mae_feature_cols)
            model = CryptoMAE(
                input_dim=input_dim,
                d_model=checkpoint.get('d_model', 128),
                n_heads=checkpoint.get('n_heads', 4),
                n_layers=checkpoint.get('n_layers', 2),
            )
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            model.eval()
            model.to(device)
            
            # Extract embeddings (sample)
            window_size = 32
            n_samples = min(1000, len(mae_features) - window_size + 1)
            
            X_raw = mae_features.values[:n_samples + window_size - 1]
            X_raw = np.nan_to_num(X_raw, 0)
            
            embeddings = []
            with torch.no_grad():
                for i in range(0, n_samples, 256):
                    batch_end = min(i + 256, n_samples)
                    batch = []
                    for j in range(i, batch_end):
                        batch.append(X_raw[j:j+window_size])
                    
                    batch = torch.tensor(np.array(batch), dtype=torch.float32, device=device)
                    encoded = model.encode(batch)
                    emb = encoded.mean(dim=1).cpu().numpy()
                    embeddings.append(emb)
            
            embeddings = np.vstack(embeddings)
            emb_df = pd.DataFrame(embeddings, columns=[f'Emb_{i}' for i in range(embeddings.shape[1])])
            
            embeddings_stats = compute_feature_statistics(emb_df, "MAE_Embeddings")
            
            print(f"  Embedding shape: {embeddings.shape}")
            print(f"  Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
            print(f"  Embedding mean: {embeddings.mean():.3f}, std: {embeddings.std():.3f}")
            
        except Exception as e:
            print(f"  [WARNING] Could not analyze embeddings: {e}")
    else:
        print(f"\n[2/2] MAE Encoder not found, skipping embedding analysis")
    
    return {
        'features': mae_features,
        'stats': stats_mae,
        'clipping': clipping_mae,
        'embeddings_stats': embeddings_stats
    }


def audit_hmm_probabilities(df: pd.DataFrame) -> Dict:
    """
    Audit HMM probabilities: verify consistency, distribution, and transitions.
    
    Args:
        df: DataFrame with HMM probability columns (HMM_Prob_* or Prob_*)
        
    Returns:
        Dict with probability statistics, violations, and distribution analysis
    """
    print("\n" + "=" * 70)
    print("HMM PROBABILITIES AUDIT")
    print("=" * 70)
    
    # Find probability columns (priority: HMM_Prob_* then Prob_*)
    hmm_prob_cols = [col for col in df.columns if col.startswith('HMM_Prob_')]
    if len(hmm_prob_cols) == 0:
        hmm_prob_cols = [col for col in df.columns if col.startswith('Prob_') and not col.startswith('HMM_Prob_')]
        if len(hmm_prob_cols) == 0:
            print("  [WARNING] No HMM probability columns found")
            return {'error': 'No probability columns found'}
        prob_type = 'Prob_* (Forward-Backward, legacy)'
    else:
        prob_type = 'HMM_Prob_* (Forward-Only)'
    
    print(f"\n[1/4] Found {len(hmm_prob_cols)} probability columns: {prob_type}")
    
    # Sort columns to ensure consistent order
    hmm_prob_cols = sorted(hmm_prob_cols)
    probs_df = df[hmm_prob_cols].copy()
    
    # Replace inf/nan with 0
    probs_df = probs_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 1. Verify sum to 1.0
    print(f"\n[2/4] Verifying probability sums...")
    prob_sums = probs_df.sum(axis=1)
    valid_mask = prob_sums > 0  # Exclude rows where all probs are 0
    
    if valid_mask.sum() == 0:
        print("  [WARNING] No valid probability rows found")
        return {'error': 'No valid probability rows'}
    
    prob_sums_valid = prob_sums[valid_mask]
    sum_violations = (np.abs(prob_sums_valid - 1.0) > 1e-6).sum()
    sum_violation_pct = sum_violations / len(prob_sums_valid) * 100
    
    print(f"  Valid rows: {valid_mask.sum()}")
    print(f"  Sum violations (>1e-6): {sum_violations} ({sum_violation_pct:.2f}%)")
    print(f"  Sum range: [{prob_sums_valid.min():.6f}, {prob_sums_valid.max():.6f}]")
    print(f"  Sum mean: {prob_sums_valid.mean():.6f}, std: {prob_sums_valid.std():.6f}")
    
    # 2. Distribution analysis
    print(f"\n[3/4] Analyzing probability distributions...")
    prob_stats = {}
    for col in hmm_prob_cols:
        series = probs_df[col][valid_mask]
        prob_stats[col] = {
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'p50': series.quantile(0.50),
            'p95': series.quantile(0.95),
            'p99': series.quantile(0.99)
        }
    
    # 3. Detect certainty (prob > 0.99) and confusion (all probs ≈ 0.25)
    print(f"\n[4/4] Detecting certainty and confusion periods...")
    probs_valid = probs_df[valid_mask]
    
    # Certainty: any prob > 0.99
    max_probs = probs_valid.max(axis=1)
    certainty_mask = max_probs > 0.99
    certainty_count = certainty_mask.sum()
    certainty_pct = certainty_count / len(probs_valid) * 100
    
    # Confusion: all probs close to uniform (for 4 states: 0.25)
    n_states = len(hmm_prob_cols)
    uniform_prob = 1.0 / n_states
    confusion_threshold = 0.05  # Within 5% of uniform
    confusion_mask = (probs_valid.max(axis=1) < uniform_prob + confusion_threshold) & \
                      (probs_valid.min(axis=1) > uniform_prob - confusion_threshold)
    confusion_count = confusion_mask.sum()
    confusion_pct = confusion_count / len(probs_valid) * 100
    
    print(f"  Certainty periods (max prob > 0.99): {certainty_count} ({certainty_pct:.2f}%)")
    print(f"  Confusion periods (all probs ≈ {uniform_prob:.2f}): {confusion_count} ({confusion_pct:.2f}%)")
    
    # 4. Dominant state transitions
    dominant_states = probs_valid.idxmax(axis=1)
    # Extract state number from column name
    state_numbers = [int(col.split('_')[-1]) for col in dominant_states]
    state_numbers = pd.Series(state_numbers, index=dominant_states.index)
    
    transitions = []
    prev_state = None
    for state in state_numbers:
        if prev_state is not None and state != prev_state:
            transitions.append((prev_state, state))
        prev_state = state
    
    transition_counts = {}
    for trans in transitions:
        transition_counts[trans] = transition_counts.get(trans, 0) + 1
    
    print(f"  Total transitions: {len(transitions)}")
    if transition_counts:
        print(f"  Most common transitions:")
        sorted_trans = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for (from_state, to_state), count in sorted_trans:
            print(f"    {from_state} -> {to_state}: {count} times")
    
    return {
        'prob_columns': hmm_prob_cols,
        'prob_type': prob_type,
        'sum_violations': sum_violations,
        'sum_violation_pct': sum_violation_pct,
        'prob_sums': prob_sums_valid,
        'prob_stats': prob_stats,
        'certainty_count': certainty_count,
        'certainty_pct': certainty_pct,
        'confusion_count': confusion_count,
        'confusion_pct': confusion_pct,
        'dominant_states': state_numbers,
        'transitions': transitions,
        'transition_counts': transition_counts,
        'probs_df': probs_df[valid_mask]
    }


def audit_hmm_entropy(df: pd.DataFrame) -> Dict:
    """
    Audit HMM entropy - Critical section for semantic uncertainty analysis.
    
    Entropy indicates model uncertainty:
    - Entropy ≈ 0.0: Absolute certainty (e.g., HMM_Prob_0 = 0.99)
    - Entropy ≈ 1.0: Total confusion (e.g., [0.25, 0.25, 0.25, 0.25])
    - Entropy > 0.7: Gray zone - HMM uncertain
    
    Args:
        df: DataFrame with HMM_Entropy column
        
    Returns:
        Dict with entropy distribution, correlations, statistics by level, and alerts
    """
    print("\n" + "=" * 70)
    print("HMM ENTROPY AUDIT - CRITICAL SECTION")
    print("=" * 70)
    
    # Check if HMM_Entropy exists
    if 'HMM_Entropy' not in df.columns:
        print("  [ERROR] HMM_Entropy column not found in dataset")
        print("  [INFO] This feature should be saved in the final dataset for TQC to use")
        return {'error': 'HMM_Entropy column not found'}
    
    print("\n[1/5] HMM_Entropy column found - validating...")
    entropy_series = df['HMM_Entropy'].dropna()
    
    if len(entropy_series) == 0:
        print("  [WARNING] No valid entropy values found")
        return {'error': 'No valid entropy values'}
    
    # Validate range [0, 1]
    out_of_range = ((entropy_series < 0) | (entropy_series > 1)).sum()
    if out_of_range > 0:
        print(f"  [WARNING] {out_of_range} values out of range [0, 1]")
    else:
        print(f"  [OK] All {len(entropy_series)} values in range [0, 1]")
    
    # 2. Distribution analysis
    print(f"\n[2/5] Analyzing entropy distribution...")
    entropy_stats = {
        'count': len(entropy_series),
        'mean': entropy_series.mean(),
        'std': entropy_series.std(),
        'min': entropy_series.min(),
        'max': entropy_series.max(),
        'p25': entropy_series.quantile(0.25),
        'p50': entropy_series.quantile(0.50),
        'p75': entropy_series.quantile(0.75),
        'p95': entropy_series.quantile(0.95),
        'p99': entropy_series.quantile(0.99)
    }
    
    print(f"  Mean: {entropy_stats['mean']:.4f}, Std: {entropy_stats['std']:.4f}")
    print(f"  Range: [{entropy_stats['min']:.4f}, {entropy_stats['max']:.4f}]")
    print(f"  Median: {entropy_stats['p50']:.4f}")
    
    # 3. Statistics by entropy level
    print(f"\n[3/5] Statistics by entropy level...")
    low_entropy = entropy_series[entropy_series < 0.3]  # Certainty
    medium_entropy = entropy_series[(entropy_series >= 0.3) & (entropy_series <= 0.7)]  # Moderate
    high_entropy = entropy_series[entropy_series > 0.7]  # Uncertainty
    very_high_entropy = entropy_series[entropy_series > 0.8]  # Maximum uncertainty
    
    level_stats = {
        'low': {
            'count': len(low_entropy),
            'pct': len(low_entropy) / len(entropy_series) * 100,
            'mean': low_entropy.mean() if len(low_entropy) > 0 else 0.0,
            'interpretation': 'Certainty (HMM confident)'
        },
        'medium': {
            'count': len(medium_entropy),
            'pct': len(medium_entropy) / len(entropy_series) * 100,
            'mean': medium_entropy.mean() if len(medium_entropy) > 0 else 0.0,
            'interpretation': 'Moderate uncertainty'
        },
        'high': {
            'count': len(high_entropy),
            'pct': len(high_entropy) / len(entropy_series) * 100,
            'mean': high_entropy.mean() if len(high_entropy) > 0 else 0.0,
            'interpretation': 'Gray zone (HMM uncertain)'
        },
        'very_high': {
            'count': len(very_high_entropy),
            'pct': len(very_high_entropy) / len(entropy_series) * 100,
            'mean': very_high_entropy.mean() if len(very_high_entropy) > 0 else 0.0,
            'interpretation': 'Maximum uncertainty (HMM blind)'
        }
    }
    
    for level, stats in level_stats.items():
        print(f"  {level.upper()}: {stats['count']} ({stats['pct']:.1f}%) - {stats['interpretation']}")
    
    # 4. Correlation with future returns
    print(f"\n[4/5] Analyzing correlation with future returns...")
    correlations = {}
    
    if 'BTC_Close' in df.columns:
        # Calculate future returns at different horizons
        prices = df['BTC_Close'].values
        returns_1h = np.zeros(len(prices))
        returns_24h = np.zeros(len(prices))
        returns_168h = np.zeros(len(prices))
        
        if len(prices) > 1:
            returns_1h[1:] = np.log(prices[1:] / prices[:-1])
        if len(prices) > 24:
            returns_24h[24:] = np.log(prices[24:] / prices[:-24])
        if len(prices) > 168:
            returns_168h[168:] = np.log(prices[168:] / prices[:-168])
        
        # Align with entropy
        valid_idx = entropy_series.index.intersection(df.index)
        entropy_aligned = entropy_series.loc[valid_idx]
        returns_1h_aligned = pd.Series(returns_1h, index=df.index).loc[valid_idx]
        returns_24h_aligned = pd.Series(returns_24h, index=df.index).loc[valid_idx]
        returns_168h_aligned = pd.Series(returns_168h, index=df.index).loc[valid_idx]
        
        if len(entropy_aligned) > 10:
            corr_1h = entropy_aligned.corr(returns_1h_aligned)
            corr_24h = entropy_aligned.corr(returns_24h_aligned)
            corr_168h = entropy_aligned.corr(returns_168h_aligned)
            
            correlations = {
                '1h': corr_1h if not np.isnan(corr_1h) else 0.0,
                '24h': corr_24h if not np.isnan(corr_24h) else 0.0,
                '168h': corr_168h if not np.isnan(corr_168h) else 0.0
            }
            
            print(f"  Correlation with future returns:")
            print(f"    1h:  {correlations['1h']:.4f}")
            print(f"    24h: {correlations['24h']:.4f}")
            print(f"    168h: {correlations['168h']:.4f}")
    
    # 5. Alerts
    print(f"\n[5/5] Generating alerts...")
    alerts = []
    
    if level_stats['very_high']['pct'] > 10:
        alerts.append(f"ALERT: {level_stats['very_high']['pct']:.1f}% of time in maximum uncertainty (>0.8)")
    
    if level_stats['high']['pct'] > 30:
        alerts.append(f"WARNING: {level_stats['high']['pct']:.1f}% of time in gray zone (>0.7)")
    
    if entropy_stats['mean'] > 0.6:
        alerts.append(f"WARNING: High average entropy ({entropy_stats['mean']:.3f}) - HMM often uncertain")
    
    if len(alerts) == 0:
        print("  [OK] No critical alerts")
    else:
        for alert in alerts:
            print(f"  {alert}")
    
    return {
        'entropy_series': entropy_series,
        'entropy_stats': entropy_stats,
        'level_stats': level_stats,
        'correlations': correlations,
        'alerts': alerts,
        'low_entropy_mask': entropy_series < 0.3,
        'high_entropy_mask': entropy_series > 0.7,
        'very_high_entropy_mask': entropy_series > 0.8
    }


def audit_hmm_additional_features(df: pd.DataFrame, detector: RegimeDetector) -> Dict:
    """
    Audit HMM additional features (curiosity mode).
    
    Analyzes features computed but not included in HMM_FEATURES:
    - HMM_MACD_Hist
    - HMM_RiskOnOff
    - HMM_VolRatio
    
    IMPORTANT: These are analyzed for curiosity/documentation only.
    Do NOT reintroduce into HMM_FEATURES without validation (Seed Robustness).
    
    Args:
        df: DataFrame with features
        detector: RegimeDetector instance
        
    Returns:
        Dict with statistics of additional features and comparisons
    """
    print("\n" + "=" * 70)
    print("HMM ADDITIONAL FEATURES AUDIT (CURIOSITY MODE)")
    print("=" * 70)
    print("  NOTE: Analysis for documentation only. Do not reintroduce without validation.")
    
    # Compute all HMM features (including additional ones)
    hmm_features_all = detector._compute_hmm_features(df)
    hmm_feature_cols = RegimeDetector.HMM_FEATURES
    
    # Find additional features (computed but not in HMM_FEATURES)
    additional_features = ['HMM_MACD_Hist', 'HMM_RiskOnOff', 'HMM_VolRatio']
    additional_stats = {}
    
    print(f"\n[1/2] Analyzing additional features...")
    for feat in additional_features:
        if feat not in hmm_features_all.columns:
            print(f"  {feat}: Not found (skipping)")
            continue
        
        series = hmm_features_all[feat].dropna()
        if len(series) == 0:
            print(f"  {feat}: No valid data")
            continue
        
        stats = compute_feature_statistics(pd.DataFrame({feat: series}), f"Additional_{feat}")
        additional_stats[feat] = {
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'p25': series.quantile(0.25),
            'p50': series.quantile(0.50),
            'p75': series.quantile(0.75),
            'p95': series.quantile(0.95),
            'p99': series.quantile(0.99),
            'count': len(series)
        }
        
        print(f"  {feat}: mean={stats.iloc[0]['Mean']:.4f}, std={stats.iloc[0]['Std']:.4f}, "
              f"range=[{stats.iloc[0]['Min']:.4f}, {stats.iloc[0]['Max']:.4f}]")
    
    # 2. Compare with main features
    print(f"\n[2/2] Comparing with main HMM features...")
    if additional_stats and 'normalized' in df.columns:
        # Get main features stats for comparison
        main_features = hmm_features_all[hmm_feature_cols]
        main_stats = {}
        for feat in hmm_feature_cols:
            if feat in main_features.columns:
                series = main_features[feat].dropna()
                if len(series) > 0:
                    main_stats[feat] = {
                        'mean': series.mean(),
                        'std': series.std()
                    }
        
        print(f"  Main features: {len(main_stats)}")
        print(f"  Additional features: {len(additional_stats)}")
        print(f"  Note: Comparison available in report")
    
    return {
        'additional_features': list(additional_stats.keys()),
        'additional_stats': additional_stats,
        'note': 'Analysis for curiosity/documentation only. Do not reintroduce without validation.'
    }


def analyze_hmm_correlations(
    hmm_raw: pd.DataFrame,
    hmm_normalized: pd.DataFrame,
    entropy_series: Optional[pd.Series] = None
) -> Dict:
    """
    Analyze correlations between HMM features.
    
    Args:
        hmm_raw: Raw HMM features
        hmm_normalized: Normalized HMM features
        entropy_series: Optional entropy series for correlation analysis
        
    Returns:
        Dict with correlation matrices and suspicious correlations
    """
    print("\n" + "=" * 70)
    print("HMM CORRELATIONS ANALYSIS")
    print("=" * 70)
    
    # 1. Correlation matrix for normalized features
    print(f"\n[1/3] Computing correlation matrix (normalized features)...")
    valid_mask = ~hmm_normalized.isna().any(axis=1)
    hmm_norm_valid = hmm_normalized[valid_mask]
    
    if len(hmm_norm_valid) < 10:
        print("  [WARNING] Not enough valid samples for correlation")
        return {'error': 'Not enough valid samples'}
    
    corr_matrix_norm = hmm_norm_valid.corr()
    
    # 2. Correlation matrix for raw features
    print(f"\n[2/3] Computing correlation matrix (raw features)...")
    hmm_raw_valid = hmm_raw[valid_mask]
    corr_matrix_raw = hmm_raw_valid.corr()
    
    # 3. Detect suspicious correlations (> 0.7 or < -0.7)
    print(f"\n[3/3] Detecting suspicious correlations...")
    suspicious_corrs = []
    
    for i in range(len(corr_matrix_norm.columns)):
        for j in range(i+1, len(corr_matrix_norm.columns)):
            feat1 = corr_matrix_norm.columns[i]
            feat2 = corr_matrix_norm.columns[j]
            corr_val = corr_matrix_norm.iloc[i, j]
            
            if abs(corr_val) > 0.7:
                suspicious_corrs.append({
                    'feature1': feat1,
                    'feature2': feat2,
                    'correlation': corr_val,
                    'type': 'high_positive' if corr_val > 0.7 else 'high_negative'
                })
    
    if suspicious_corrs:
        print(f"  Found {len(suspicious_corrs)} suspicious correlations:")
        for corr in suspicious_corrs[:5]:  # Show first 5
            print(f"    {corr['feature1']} <-> {corr['feature2']}: {corr['correlation']:.3f}")
    else:
        print("  [OK] No suspicious correlations found (all |r| < 0.7)")
    
    # 4. Correlation with entropy if available
    entropy_corrs = {}
    if entropy_series is not None:
        entropy_aligned = entropy_series.loc[valid_mask]
        if len(entropy_aligned) > 10:
            for feat in hmm_norm_valid.columns:
                feat_series = hmm_norm_valid[feat]
                corr = entropy_aligned.corr(feat_series)
                if not np.isnan(corr):
                    entropy_corrs[feat] = corr
    
    if entropy_corrs:
        print(f"\n  Entropy correlations:")
        for feat, corr in sorted(entropy_corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
            print(f"    {feat}: {corr:.4f}")
    
    return {
        'corr_matrix_normalized': corr_matrix_norm,
        'corr_matrix_raw': corr_matrix_raw,
        'suspicious_correlations': suspicious_corrs,
        'entropy_correlations': entropy_corrs
    }


def audit_tqc_normalization(df: pd.DataFrame) -> Dict:
    """
    Audit TQC normalization: verify features match MAE input.
    
    Args:
        df: DataFrame with normalized features (same as MAE input)
        
    Returns:
        Dict with TQC statistics
    """
    print("\n" + "=" * 70)
    print("TQC NORMALIZATION AUDIT")
    print("=" * 70)
    
    # TQC uses same features as MAE (via FoundationFeatureExtractor)
    exclude_patterns = ['_Open', '_High', '_Low', '_Close', '_Volume']
    exclude_prefixes = ['Prob_', 'HMM_']
    
    tqc_feature_cols = [
        col for col in df.columns
        if not any(p in col for p in exclude_patterns)
        and not any(col.startswith(p) for p in exclude_prefixes)
        and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]
    
    tqc_features = df[tqc_feature_cols].copy()
    tqc_features = tqc_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"\n[1/1] TQC Input Features (via FoundationFeatureExtractor):")
    print(f"  Features: {len(tqc_feature_cols)}")
    print(f"  Shape: {tqc_features.shape}")
    print(f"  Note: TQC receives same normalized features as MAE")
    
    stats_tqc = compute_feature_statistics(tqc_features, "TQC_Input")
    clipping_tqc = detect_clipping(tqc_features, clip_bounds=(-5, 5))
    
    return {
        'features': tqc_features,
        'stats': stats_tqc,
        'clipping': clipping_tqc
    }


# ============================================================================
# MAIN AUDIT FUNCTION
# ============================================================================

def run_normalization_audit(
    segment_id: int = 0,
    output_dir: Optional[str] = None,
    device: str = "cpu",
    force_retrain: bool = False
) -> Dict:
    """
    Run complete normalization audit for HMM, MAE, and TQC.
    
    Args:
        segment_id: WFO segment ID
        output_dir: Output directory for reports and plots. If None, generates unique name with timestamp.
        device: Device for MAE operations
        force_retrain: Force data retraining
        
    Returns:
        Dict with all audit results
    """
    # Generate unique output directory with timestamp if not provided
    if output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/normalization_audit_{timestamp}"
    
    print("=" * 70)
    print("FEATURE NORMALIZATION & CLIPPING AUDIT")
    print("=" * 70)
    print(f"Segment: {segment_id}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading data...")
    manager = DataManager()
    
    # Try segment first
    segment_path = f"data/wfo/segment_{segment_id}/train.parquet"
    if os.path.exists(segment_path) and not force_retrain:
        df = pd.read_parquet(segment_path)
        print(f"  Loaded: {segment_path}")
    else:
        processed_path = "data/processed_data.parquet"
        if os.path.exists(processed_path):
            df = pd.read_parquet(processed_path)
            print(f"  Loaded: {processed_path}")
        else:
            print(f"  [RETRAINING] Processing data...")
            df = manager.pipeline(
                save_path=processed_path,
                scaler_path="data/scaler.pkl",
                use_cached_data=True
            )
    
    print(f"  Shape: {df.shape}")
    
    # Initialize detector
    detector = RegimeDetector()
    
    # Audit HMM
    print("\n[2/8] Auditing HMM normalization...")
    try:
        hmm_results = audit_hmm_normalization(df, detector)
    except Exception as e:
        print(f"  [ERROR] HMM audit failed: {e}")
        hmm_results = {'error': str(e)}
    
    # Audit HMM Probabilities
    print("\n[3/8] Auditing HMM probabilities...")
    try:
        prob_results = audit_hmm_probabilities(df)
    except Exception as e:
        print(f"  [ERROR] HMM probabilities audit failed: {e}")
        prob_results = {'error': str(e)}
    
    # Audit HMM Entropy (Critical)
    print("\n[4/8] Auditing HMM entropy (CRITICAL)...")
    try:
        entropy_results = audit_hmm_entropy(df)
    except Exception as e:
        print(f"  [ERROR] HMM entropy audit failed: {e}")
        entropy_results = {'error': str(e)}
    
    # Audit HMM Additional Features (Curiosity)
    print("\n[5/8] Auditing HMM additional features (curiosity mode)...")
    try:
        additional_results = audit_hmm_additional_features(df, detector)
    except Exception as e:
        print(f"  [ERROR] HMM additional features audit failed: {e}")
        additional_results = {'error': str(e)}
    
    # Analyze HMM Correlations
    print("\n[6/8] Analyzing HMM correlations...")
    try:
        if 'raw' in hmm_results and 'normalized' in hmm_results:
            entropy_series = entropy_results.get('entropy_series') if 'error' not in entropy_results else None
            corr_results = analyze_hmm_correlations(
                hmm_results['raw'],
                hmm_results['normalized'],
                entropy_series
            )
        else:
            corr_results = {'error': 'HMM results not available'}
    except Exception as e:
        print(f"  [ERROR] HMM correlations analysis failed: {e}")
        corr_results = {'error': str(e)}
    
    # Audit MAE
    print("\n[7/8] Auditing MAE normalization...")
    # Try to find encoder
    encoder_paths = [
        "weights/mae/encoder.pth",
        "weights/foundation/encoder.pth",
        "logs/foundation/best_encoder.pth",
    ]
    encoder_path = None
    for path in encoder_paths:
        if os.path.exists(path):
            encoder_path = path
            break
    
    try:
        mae_results = audit_mae_normalization(df, encoder_path=encoder_path, device=device)
    except Exception as e:
        print(f"  [ERROR] MAE audit failed: {e}")
        mae_results = {'error': str(e)}
    
    # Audit TQC
    print("\n[8/8] Auditing TQC normalization...")
    try:
        tqc_results = audit_tqc_normalization(df)
    except Exception as e:
        print(f"  [ERROR] TQC audit failed: {e}")
        tqc_results = {'error': str(e)}
    
    # Generate report
    print("\n[9/9] Generating report and visualizations...")
    
    # Prepare data for visualizations
    df_dict = {}
    clipping_dict = {}
    
    if 'normalized' in hmm_results:
        df_dict['HMM'] = hmm_results['normalized']
        clipping_dict['HMM'] = hmm_results['clipping']
    
    if 'features' in mae_results:
        df_dict['MAE'] = mae_results['features']
        clipping_dict['MAE'] = mae_results['clipping']
    
    if 'features' in tqc_results:
        df_dict['TQC'] = tqc_results['features']
        clipping_dict['TQC'] = tqc_results['clipping']
    
    # Create standard visualizations
    plot_files = create_visualizations(df_dict, clipping_dict, output_dir)
    
    # Create HMM-specific visualizations
    if 'error' not in entropy_results:
        print("\n  Creating HMM entropy visualizations...")
        entropy_plots = create_hmm_entropy_visualizations(
            df, entropy_results, prob_results if 'error' not in prob_results else None, output_dir
        )
        plot_files.extend(entropy_plots)
    
    # Create correlation heatmap
    if 'error' not in corr_results:
        print("\n  Creating HMM correlations heatmap...")
        corr_plots = create_hmm_correlations_heatmap(corr_results, output_dir)
        plot_files.extend(corr_plots)
    
    # Generate text report
    report_path = os.path.join(output_dir, 'normalization_audit_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("FEATURE NORMALIZATION & CLIPPING AUDIT REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        # HMM Report
        if 'stats_normalized' in hmm_results:
            f.write("\n" + "=" * 70 + "\n")
            f.write("HMM NORMALIZATION\n")
            f.write("=" * 70 + "\n\n")
            f.write(hmm_results['stats_normalized'].to_string())
            f.write("\n\n")
            f.write("CLIPPING STATISTICS:\n")
            f.write(hmm_results['clipping'].to_string())
            f.write("\n\n")
        
        # MAE Report
        if 'stats' in mae_results:
            f.write("\n" + "=" * 70 + "\n")
            f.write("MAE NORMALIZATION\n")
            f.write("=" * 70 + "\n\n")
            f.write(mae_results['stats'].to_string())
            f.write("\n\n")
            f.write("CLIPPING STATISTICS:\n")
            f.write(mae_results['clipping'].to_string())
            f.write("\n\n")
        
        # TQC Report
        if 'stats' in tqc_results:
            f.write("\n" + "=" * 70 + "\n")
            f.write("TQC NORMALIZATION\n")
            f.write("=" * 70 + "\n\n")
            f.write(tqc_results['stats'].to_string())
            f.write("\n\n")
            f.write("CLIPPING STATISTICS:\n")
            f.write(tqc_results['clipping'].to_string())
            f.write("\n\n")
        
        # HMM Features Detailed Analysis
        if 'per_feature_stats' in hmm_results:
            f.write("\n" + "=" * 70 + "\n")
            f.write("HMM FEATURES DETAILED ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            for feat, stats in hmm_results['per_feature_stats'].items():
                f.write(f"\n{feat}:\n")
                f.write(f"  Raw: mean={stats['raw_stats']['mean']:.4f}, "
                       f"std={stats['raw_stats']['std']:.4f}, "
                       f"outliers={stats['raw_stats']['outliers_pct']:.1f}%\n")
                f.write(f"  Normalized: mean={stats['normalized_stats']['mean']:.4f}, "
                       f"std={stats['normalized_stats']['std']:.4f}, "
                       f"clipped={stats['normalized_stats']['pct_clipped']:.1f}%\n")
            f.write("\n")
        
        # HMM Probabilities
        if 'error' not in prob_results:
            f.write("\n" + "=" * 70 + "\n")
            f.write("HMM PROBABILITIES AUDIT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Probability Type: {prob_results['prob_type']}\n")
            f.write(f"Sum Violations: {prob_results['sum_violations']} ({prob_results['sum_violation_pct']:.2f}%)\n")
            f.write(f"Certainty Periods (max prob > 0.99): {prob_results['certainty_count']} ({prob_results['certainty_pct']:.2f}%)\n")
            f.write(f"Confusion Periods (all probs ≈ uniform): {prob_results['confusion_count']} ({prob_results['confusion_pct']:.2f}%)\n")
            f.write(f"\nProbability Statistics:\n")
            for col, stats in prob_results['prob_stats'].items():
                f.write(f"  {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                       f"range=[{stats['min']:.4f}, {stats['max']:.4f}]\n")
            if prob_results['transition_counts']:
                f.write(f"\nMost Common Transitions:\n")
                sorted_trans = sorted(prob_results['transition_counts'].items(), 
                                    key=lambda x: x[1], reverse=True)[:10]
                for (from_state, to_state), count in sorted_trans:
                    f.write(f"  {from_state} -> {to_state}: {count} times\n")
            f.write("\n")
        
        # HMM Entropy (Critical)
        if 'error' not in entropy_results:
            f.write("\n" + "=" * 70 + "\n")
            f.write("HMM ENTROPY AUDIT - CRITICAL SECTION\n")
            f.write("=" * 70 + "\n\n")
            f.write("Entropy Distribution:\n")
            stats = entropy_results['entropy_stats']
            f.write(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}\n")
            f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
            f.write(f"  Median: {stats['p50']:.4f}\n")
            f.write(f"  Percentiles: P25={stats['p25']:.4f}, P75={stats['p75']:.4f}, "
                   f"P95={stats['p95']:.4f}, P99={stats['p99']:.4f}\n\n")
            
            f.write("Statistics by Entropy Level:\n")
            for level, level_stats in entropy_results['level_stats'].items():
                f.write(f"  {level.upper()}: {level_stats['count']} ({level_stats['pct']:.1f}%) - "
                       f"{level_stats['interpretation']}\n")
            
            if entropy_results['correlations']:
                f.write(f"\nCorrelation with Future Returns:\n")
                for horizon, corr in entropy_results['correlations'].items():
                    f.write(f"  {horizon}: {corr:.4f}\n")
            
            if entropy_results['alerts']:
                f.write(f"\nALERTS:\n")
                for alert in entropy_results['alerts']:
                    f.write(f"  {alert}\n")
            else:
                f.write(f"\n[OK] No critical alerts\n")
            f.write("\n")
        
        # HMM Additional Features (Curiosity)
        if 'error' not in additional_results:
            f.write("\n" + "=" * 70 + "\n")
            f.write("HMM ADDITIONAL FEATURES (CURIOSITY MODE)\n")
            f.write("=" * 70 + "\n\n")
            f.write("NOTE: Analysis for documentation only. Do not reintroduce without validation.\n\n")
            for feat, stats in additional_results['additional_stats'].items():
                f.write(f"{feat}:\n")
                f.write(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}\n")
                f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                f.write(f"  Percentiles: P25={stats['p25']:.4f}, P50={stats['p50']:.4f}, "
                       f"P75={stats['p75']:.4f}, P95={stats['p95']:.4f}\n")
            f.write("\n")
        
        # HMM Correlations
        if 'error' not in corr_results:
            f.write("\n" + "=" * 70 + "\n")
            f.write("HMM CORRELATIONS ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            if corr_results['suspicious_correlations']:
                f.write(f"Suspicious Correlations (|r| > 0.7): {len(corr_results['suspicious_correlations'])}\n")
                for corr in corr_results['suspicious_correlations'][:10]:
                    f.write(f"  {corr['feature1']} <-> {corr['feature2']}: {corr['correlation']:.3f} "
                           f"({corr['type']})\n")
            else:
                f.write("[OK] No suspicious correlations found (all |r| < 0.7)\n")
            
            if corr_results['entropy_correlations']:
                f.write(f"\nEntropy Correlations:\n")
                sorted_entropy = sorted(corr_results['entropy_correlations'].items(),
                                       key=lambda x: abs(x[1]), reverse=True)
                for feat, corr in sorted_entropy:
                    f.write(f"  {feat}: {corr:.4f}\n")
            f.write("\n")
        
        # Comparison
        if 'stats_normalized' in hmm_results and 'stats' in mae_results:
            f.write("\n" + "=" * 70 + "\n")
            f.write("COMPARISON: HMM vs MAE\n")
            f.write("=" * 70 + "\n\n")
            comparison = compare_normalizations(
                hmm_results['stats_normalized'],
                mae_results['stats']
            )
            f.write(comparison.to_string())
            f.write("\n\n")
    
    print(f"  Saved report: {report_path}")
    print(f"  Generated {len(plot_files)} plots")
    
    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    
    if 'clipping' in hmm_results:
        max_clip = hmm_results['clipping']['Pct_Clipped'].max()
        print(f"\nHMM: Max clipping = {max_clip:.1f}%")
        if max_clip > 10:
            print("  [ALERT] High clipping detected (>10%)")
        elif max_clip > 5:
            print("  [WARNING] Moderate clipping detected (>5%)")
        else:
            print("  [OK] Clipping within acceptable range")
    
    if 'clipping' in mae_results:
        max_clip = mae_results['clipping']['Pct_Clipped'].max()
        print(f"\nMAE: Max clipping = {max_clip:.1f}%")
    
    if 'clipping' in tqc_results:
        max_clip = tqc_results['clipping']['Pct_Clipped'].max()
        print(f"\nTQC: Max clipping = {max_clip:.1f}%")
    
    print("\n" + "=" * 70)
    
    return {
        'hmm': hmm_results,
        'hmm_probabilities': prob_results,
        'hmm_entropy': entropy_results,
        'hmm_additional': additional_results,
        'hmm_correlations': corr_results,
        'mae': mae_results,
        'tqc': tqc_results,
        'report_path': report_path,
        'plot_files': plot_files
    }
