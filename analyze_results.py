"""
Analyze and Visualize Riz Score Results

This script loads results from the hybrid NLP pipeline and generates
comprehensive visualizations and statistical analysis.

Usage:
    python analyze_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys

def load_results():
    """Load hybrid method results"""
    try:
        results = pd.read_csv('final_riz_test_results_hybrid.csv')
        print(f"✓ Loaded results for {len(results)} movies")
        return results
    except FileNotFoundError:
        print("❌ Error: final_riz_test_results_hybrid.csv not found")
        print("Please run the hybrid pipeline first:")
        print("  python run_hybrid_pipeline.py")
        print("  OR")
        print("  jupyter notebook nlp_riz_score_hybrid.ipynb")
        return None

def calculate_statistics(results):
    """Calculate comprehensive statistics"""
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS - RIZ SCORE RESULTS")
    print("="*70)

    # Overall statistics
    print("\n📊 OVERALL STATISTICS:")
    print(f"  Total Movies Analyzed:        {len(results)}")
    print(f"  Year Range:                   {results['Year'].min()}-{results['Year'].max()}")
    print(f"  Mean Weighted Riz Score:      {results['riz_score_weighted'].mean():.3f}")
    print(f"  Median Weighted Riz Score:    {results['riz_score_weighted'].median():.3f}")
    print(f"  Std Dev:                      {results['riz_score_weighted'].std():.3f}")
    print(f"  Min Score:                    {results['riz_score_weighted'].min():.3f}")
    print(f"  Max Score:                    {results['riz_score_weighted'].max():.3f}")

    # Temporal trend
    print("\n📈 TEMPORAL TREND ANALYSIS:")
    correlation = results['Year'].corr(results['riz_score_weighted'])
    slope, intercept = np.polyfit(results['Year'], results['riz_score_weighted'], 1)

    print(f"  Correlation with Year:        r = {correlation:.3f}")
    print(f"  Regression Slope:             {slope:+.4f} points/year")
    print(f"  Interpretation:               ", end="")

    if correlation > 0.3:
        print("Strong positive trend - scores increasing over time")
    elif correlation > 0:
        print("Weak positive trend - slight increase over time")
    elif correlation > -0.3:
        print("Weak negative trend - slight decrease over time")
    else:
        print("Strong negative trend - scores decreasing over time")

    # Pre/Post 2014 analysis (Modi era)
    pre_2014 = results[results['Year'] < 2014]
    post_2014 = results[results['Year'] >= 2014]

    print("\n🗓️  PRE/POST MODI ERA (2014) COMPARISON:")
    print(f"  Pre-2014 Movies (n={len(pre_2014)}):")
    print(f"    Mean Riz Score:             {pre_2014['riz_score_weighted'].mean():.3f}")
    print(f"    Median:                     {pre_2014['riz_score_weighted'].median():.3f}")

    print(f"\n  Post-2014 Movies (n={len(post_2014)}):")
    print(f"    Mean Riz Score:             {post_2014['riz_score_weighted'].mean():.3f}")
    print(f"    Median:                     {post_2014['riz_score_weighted'].median():.3f}")

    if pre_2014['riz_score_weighted'].mean() > 0:
        percent_change = ((post_2014['riz_score_weighted'].mean() -
                          pre_2014['riz_score_weighted'].mean()) /
                         pre_2014['riz_score_weighted'].mean() * 100)
        print(f"\n  Percentage Change:            {percent_change:+.1f}%")

    # Statistical significance
    t_stat, p_value = stats.ttest_ind(post_2014['riz_score_weighted'],
                                       pre_2014['riz_score_weighted'])
    print(f"\n  T-Test Results:")
    print(f"    t-statistic:                {t_stat:.3f}")
    print(f"    p-value:                    {p_value:.4f}")
    print(f"    Significance:               ", end="")

    if p_value < 0.001:
        print("*** (p < 0.001) - Highly significant")
    elif p_value < 0.01:
        print("** (p < 0.01) - Very significant")
    elif p_value < 0.05:
        print("* (p < 0.05) - Significant")
    else:
        print("Not significant (p >= 0.05)")

    # Dimension analysis
    print("\n📋 MISREPRESENTATION DIMENSIONS:")
    dimensions = ['terrorism', 'anger', 'superstition', 'threat_to_western', 'misogyny']

    for dim in dimensions:
        flagged = results[f'{dim}_flag'].sum()
        mean_score = results[f'{dim}_combined_score'].mean()
        print(f"  {dim.capitalize():20s}: {flagged:2d} films flagged ({flagged/len(results)*100:.1f}%), "
              f"avg score = {mean_score:.3f}")

    # Top and bottom scoring movies
    print("\n🔝 TOP 10 HIGHEST SCORING MOVIES:")
    top_10 = results.nlargest(10, 'riz_score_weighted')[['Movie Title', 'Year', 'riz_score_weighted']]
    for idx, row in top_10.iterrows():
        print(f"  {row['Year']} - {row['Movie Title']:30s} (score: {row['riz_score_weighted']:.3f})")

    print("\n⬇️  TOP 10 LOWEST SCORING MOVIES:")
    bottom_10 = results.nsmallest(10, 'riz_score_weighted')[['Movie Title', 'Year', 'riz_score_weighted']]
    for idx, row in bottom_10.iterrows():
        print(f"  {row['Year']} - {row['Movie Title']:30s} (score: {row['riz_score_weighted']:.3f})")

    return results

def create_comprehensive_visualizations(results):
    """Generate comprehensive visualizations"""
    print("\n📊 Generating visualizations...")

    fig = plt.figure(figsize=(20, 12))

    # Plot 1: Temporal trend with trendline
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(results['Year'], results['riz_score_weighted'],
                alpha=0.7, s=100, edgecolors='black', linewidth=1.5,
                c=results['riz_score_weighted'], cmap='RdYlGn_r')

    # Trendline
    z = np.polyfit(results['Year'], results['riz_score_weighted'], 1)
    p = np.poly1d(z)
    ax1.plot(results['Year'], p(results['Year']),
             'r--', linewidth=3, label=f'Trend: {z[0]:+.4f}/year')

    ax1.set_title('Temporal Trend: Riz Scores Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Weighted Riz Score', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)

    # Add correlation text
    corr = results['Year'].corr(results['riz_score_weighted'])
    ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Distribution histogram
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(results['riz_score_weighted'], bins=20, alpha=0.7,
             edgecolor='black', linewidth=1.5, color='steelblue')
    ax2.axvline(results['riz_score_weighted'].mean(), color='red',
                linestyle='--', linewidth=2, label=f"Mean: {results['riz_score_weighted'].mean():.3f}")
    ax2.axvline(results['riz_score_weighted'].median(), color='green',
                linestyle='--', linewidth=2, label=f"Median: {results['riz_score_weighted'].median():.3f}")
    ax2.set_title('Distribution of Riz Scores', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Weighted Riz Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Pre/Post 2014 comparison
    ax3 = plt.subplot(2, 3, 3)
    pre_2014 = results[results['Year'] < 2014]
    post_2014 = results[results['Year'] >= 2014]

    bp = ax3.boxplot([pre_2014['riz_score_weighted'], post_2014['riz_score_weighted']],
                      labels=['Pre-2014\n(n={})'.format(len(pre_2014)),
                             'Post-2014\n(n={})'.format(len(post_2014))],
                      patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)

    ax3.set_title('Pre/Post Modi Era Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Weighted Riz Score', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add means as points
    means = [pre_2014['riz_score_weighted'].mean(), post_2014['riz_score_weighted'].mean()]
    ax3.plot([1, 2], means, 'ro', markersize=10, label='Mean', zorder=3)
    ax3.legend()

    # Plot 4: Dimension breakdown (stacked bars by year)
    ax4 = plt.subplot(2, 3, 4)
    dimensions = ['terrorism', 'anger', 'superstition', 'threat_to_western', 'misogyny']
    dim_data = []

    for dim in dimensions:
        dim_data.append(results.groupby('Year')[f'{dim}_flag'].sum())

    years = sorted(results['Year'].unique())
    bottom = np.zeros(len(years))
    colors = ['#ff6b6b', '#ffa500', '#4ecdc4', '#95e1d3', '#c44569']

    for i, dim in enumerate(dimensions):
        values = results.groupby('Year')[f'{dim}_flag'].sum().values
        ax4.bar(years, values, bottom=bottom, label=dim.capitalize(),
                color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        bottom += values

    ax4.set_title('Dimension Flags by Year', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Year', fontsize=12)
    ax4.set_ylabel('Number of Flagged Dimensions', fontsize=12)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Sentiment analysis
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(results['Year'], results['sentiment_negative_ratio'],
                alpha=0.7, s=100, edgecolors='black', linewidth=1.5,
                c=results['sentiment_negative_ratio'], cmap='RdYlGn_r')

    z_sent = np.polyfit(results['Year'], results['sentiment_negative_ratio'], 1)
    p_sent = np.poly1d(z_sent)
    ax5.plot(results['Year'], p_sent(results['Year']),
             'r--', linewidth=3, label=f'Trend: {z_sent[0]:+.4f}/year')

    ax5.set_title('Negative Sentiment Around Muslim Characters', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Year', fontsize=12)
    ax5.set_ylabel('Negative Sentiment Ratio', fontsize=12)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.legend()

    # Plot 6: Dimension heatmap over time
    ax6 = plt.subplot(2, 3, 6)

    # Create matrix of dimension scores by year
    heatmap_data = []
    year_labels = sorted(results['Year'].unique())

    for year in year_labels:
        year_data = results[results['Year'] == year]
        row = [year_data[f'{dim}_combined_score'].mean() for dim in dimensions]
        heatmap_data.append(row)

    heatmap_data = np.array(heatmap_data).T

    im = ax6.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax6.set_yticks(np.arange(len(dimensions)))
    ax6.set_yticklabels([d.capitalize() for d in dimensions])
    ax6.set_xticks(np.arange(len(year_labels)))
    ax6.set_xticklabels(year_labels, rotation=45, ha='right')
    ax6.set_title('Dimension Intensity Heatmap', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax6)
    cbar.set_label('Mean Combined Score', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig('riz_analysis_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: riz_analysis_visualization.png")

    # Create second figure: Binary vs Weighted comparison
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Binary scores
    axes[0].scatter(results['Year'], results['riz_score_binary'],
                   alpha=0.7, s=100, edgecolors='black', linewidth=1.5,
                   color='green')
    z_bin = np.polyfit(results['Year'], results['riz_score_binary'], 1)
    p_bin = np.poly1d(z_bin)
    axes[0].plot(results['Year'], p_bin(results['Year']),
                'r--', linewidth=3)
    axes[0].set_title('Binary Riz Score (Flags)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Year', fontsize=12)
    axes[0].set_ylabel('Riz Score (0-5)', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Weighted scores
    axes[1].scatter(results['Year'], results['riz_score_weighted'],
                   alpha=0.7, s=100, edgecolors='black', linewidth=1.5,
                   color='purple')
    z_weight = np.polyfit(results['Year'], results['riz_score_weighted'], 1)
    p_weight = np.poly1d(z_weight)
    axes[1].plot(results['Year'], p_weight(results['Year']),
                'r--', linewidth=3)
    axes[1].set_title('Weighted Riz Score (Continuous)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Year', fontsize=12)
    axes[1].set_ylabel('Riz Score (0-5)', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('riz_scoring_methods.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: riz_scoring_methods.png")

    plt.show()

def export_detailed_stats(results):
    """Export detailed statistics to CSV"""
    # Create summary statistics by year
    year_stats = results.groupby('Year').agg({
        'riz_score_weighted': ['mean', 'median', 'std', 'min', 'max'],
        'riz_score_binary': ['mean', 'median'],
        'sentiment_negative_ratio': ['mean'],
        'terrorism_flag': 'sum',
        'anger_flag': 'sum',
        'superstition_flag': 'sum',
        'threat_to_western_flag': 'sum',
        'misogyny_flag': 'sum',
        'Movie Title': 'count'
    }).round(3)

    year_stats.columns = ['_'.join(col).strip() for col in year_stats.columns.values]
    year_stats.to_csv('riz_analysis_by_year.csv')
    print("\n✓ Exported: riz_analysis_by_year.csv")

    # Export dimension correlation matrix
    dimensions = ['terrorism_combined_score', 'anger_combined_score',
                 'superstition_combined_score', 'threat_to_western_combined_score',
                 'misogyny_combined_score']

    corr_matrix = results[dimensions].corr().round(3)
    corr_matrix.to_csv('dimension_correlations.csv')
    print("✓ Exported: dimension_correlations.csv")

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("RIZ SCORE ANALYSIS - HYBRID NLP METHOD")
    print("="*70)

    # Load results
    results = load_results()
    if results is None:
        sys.exit(1)

    # Calculate statistics
    results = calculate_statistics(results)

    # Create visualizations
    create_comprehensive_visualizations(results)

    # Export detailed stats
    export_detailed_stats(results)

    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - riz_analysis_visualization.png (6-panel comprehensive analysis)")
    print("  - riz_scoring_methods.png (binary vs weighted comparison)")
    print("  - riz_analysis_by_year.csv (yearly statistics)")
    print("  - dimension_correlations.csv (dimension correlation matrix)")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
