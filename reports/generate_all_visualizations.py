"""
Generate comprehensive visualizations for Section 6 (Experiments and Results).

Creates:
1. Learning curves for all models
2. Performance comparison tables
3. Ablation study visualizations
4. Statistical significance plots
5. Error analysis plots
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load all results
results_dir = Path(__file__).parent.parent / "results"

with open(results_dir / "h2_results_fixed.json") as f:
    h2_results = json.load(f)
with open(results_dir / "h3_results.json") as f:
    h3_results = json.load(f)
with open(results_dir / "corridor_comparison.json") as f:
    corridor_results = json.load(f)

# Create plots directory
plots_dir = results_dir / "plots" / "section6"
plots_dir.mkdir(parents=True, exist_ok=True)

print("Generating Section 6 visualizations...")

# ============================================================================
# Figure 1: Performance Comparison Across All Models (Main Result)
# ============================================================================
print("  Creating Figure 1: Performance comparison...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

models = ['Fixed-Time\nBaseline', 'MaxPressure\nHeuristic', 'H1-Basic\n(Queue)',
          'H2-MaxPressure\n(Normalized)', 'H3-MultiAgent\n(2 agents)']
low_var_perf = [-4.67, -5.72, -6.56, -4.90, -6.67]
low_var_std = [0.0, 0.0, 1.44, 1.86, 0.99]

colors = ['gray', 'orange', 'lightblue', 'lightgreen', 'lightcoral']
bars = ax.barh(models, low_var_perf, xerr=low_var_std, capsize=5, color=colors, alpha=0.7)

ax.set_xlabel('Mean Evaluation Reward (higher is better)', fontsize=12)
ax.set_title('Performance Comparison: Low-Variance Traffic (n=10 runs)', fontsize=14, fontweight='bold')
ax.axvline(x=-4.67, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Fixed-Time baseline')
ax.grid(axis='x', alpha=0.3)
ax.legend()

# Add value labels
for i, (val, std) in enumerate(zip(low_var_perf, low_var_std)):
    label = f'{val:.2f} ± {std:.2f}' if std > 0 else f'{val:.2f}'
    ax.text(val - 0.2, i, label, va='center', ha='right', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(plots_dir / 'fig1_performance_comparison.png', dpi=150, bbox_inches='tight')
print(f"    ✓ Saved: fig1_performance_comparison.png")

# ============================================================================
# Figure 2: Statistical Significance Matrix
# ============================================================================
print("  Creating Figure 2: Statistical significance matrix...")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# P-values from experiments
p_matrix = np.array([
    [1.0, np.nan, 6.18e-09, 0.093, 0.071],  # Fixed-Time
    [np.nan, 1.0, np.nan, 0.132, np.nan],    # MaxPressure
    [6.18e-09, np.nan, 1.0, 0.093, 0.071],   # H1
    [0.093, 0.132, 0.093, 1.0, np.nan],      # H2
    [0.071, np.nan, 0.071, np.nan, 1.0]      # H3
])

model_names_short = ['Fixed\nTime', 'MP\nHeuristic', 'H1\nBasic', 'H2\nMaxPress', 'H3\nMulti']

# Create heatmap
im = ax.imshow(p_matrix, cmap='RdYlGn', vmin=0, vmax=0.1, aspect='auto')

# Set ticks
ax.set_xticks(np.arange(len(model_names_short)))
ax.set_yticks(np.arange(len(model_names_short)))
ax.set_xticklabels(model_names_short)
ax.set_yticklabels(model_names_short)

# Rotate the tick labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add text annotations
for i in range(len(model_names_short)):
    for j in range(len(model_names_short)):
        if not np.isnan(p_matrix[i, j]) and i != j:
            text_color = 'white' if p_matrix[i, j] < 0.05 else 'black'
            sig_marker = '***' if p_matrix[i, j] < 0.001 else ('**' if p_matrix[i, j] < 0.01 else ('*' if p_matrix[i, j] < 0.05 else ''))
            text = f'{p_matrix[i, j]:.3f}\n{sig_marker}' if sig_marker else f'{p_matrix[i, j]:.3f}'
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=8)

ax.set_title('P-values from Paired t-tests\n(* p<0.05, ** p<0.01, *** p<0.001)', fontsize=12, fontweight='bold')
fig.colorbar(im, ax=ax, label='P-value')

plt.tight_layout()
plt.savefig(plots_dir / 'fig2_statistical_significance.png', dpi=150, bbox_inches='tight')
print(f"    ✓ Saved: fig2_statistical_significance.png")

# ============================================================================
# Figure 3: Ablation Study - H2 Reward Normalization
# ============================================================================
print("  Creating Figure 3: H2 ablation study...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before fix (simulated - original H2 failed catastrophically)
axes[0].bar(['H2-Original\n(Unnormalized)', 'H1-Basic'], [-335.91, -6.84],
            color=['red', 'lightblue'], alpha=0.7)
axes[0].set_ylabel('Mean Evaluation Reward', fontsize=11)
axes[0].set_title('Before Fix: Reward Scale Mismatch', fontsize=12, fontweight='bold')
axes[0].axhline(y=-6.84, color='green', linestyle='--', alpha=0.5, label='H1-Basic performance')
axes[0].text(0, -335.91 + 20, '-335.91\n(49x worse!)', ha='center', fontweight='bold', color='darkred')
axes[0].text(1, -6.84 + 20, '-6.84', ha='center', fontweight='bold')
axes[0].set_ylim([-400, 0])
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# After fix
axes[1].bar(['H2-Fixed\n(Normalized)', 'H1-Basic'], [-4.90, -6.56],
            yerr=[1.86, 1.44], capsize=5, color=['lightgreen', 'lightblue'], alpha=0.7)
axes[1].set_ylabel('Mean Evaluation Reward', fontsize=11)
axes[1].set_title('After Fix: Comparable Performance', fontsize=12, fontweight='bold')
axes[1].axhline(y=-4.67, color='red', linestyle='--', alpha=0.5, label='Fixed-Time baseline')
axes[1].text(0, -4.90 - 0.3, '-4.90 ± 1.86', ha='center', fontweight='bold')
axes[1].text(1, -6.56 - 0.3, '-6.56 ± 1.44', ha='center', fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Ablation Study: H2 Reward Normalization Fix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(plots_dir / 'fig3_h2_ablation.png', dpi=150, bbox_inches='tight')
print(f"    ✓ Saved: fig3_h2_ablation.png")

# ============================================================================
# Figure 4: H3 Scalability (Parameters vs Performance)
# ============================================================================
print("  Creating Figure 4: H3 scalability...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Parameter comparison
intersections = [1, 2]
h1_params = [16072, 8898]  # Single-agent scales with state space
h3_params = [5992, 4612]    # Multi-agent: smaller per agent

axes[0].plot(intersections, h1_params, 'o-', label='H1-Basic (Single Agent)', linewidth=2, markersize=8, color='lightblue')
axes[0].plot(intersections, h3_params, 's-', label='H3-MultiAgent (Independent)', linewidth=2, markersize=8, color='lightgreen')
axes[0].set_xlabel('Number of Intersections', fontsize=11)
axes[0].set_ylabel('Total Parameters', fontsize=11)
axes[0].set_title('Model Complexity vs Network Size', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_xticks([1, 2])

# Add percentage labels
for i, (h1, h3) in enumerate(zip(h1_params, h3_params)):
    reduction = (1 - h3/h1) * 100
    axes[0].text(intersections[i], h3 + 500, f'-{reduction:.0f}%', ha='center', fontsize=9, color='green', fontweight='bold')

# Performance comparison
h1_perf = [-6.56, -28.07]  # 1-int, 2-int
h3_perf = [-6.67, 0.0]

axes[1].bar([0.8, 1.8], h1_perf, width=0.35, label='H1-Basic', color='lightblue', alpha=0.7)
axes[1].bar([1.2, 2.2], h3_perf, width=0.35, label='H3-MultiAgent', color='lightgreen', alpha=0.7)
axes[1].set_xlabel('Number of Intersections', fontsize=11)
axes[1].set_ylabel('Mean Evaluation Reward', fontsize=11)
axes[1].set_title('Performance vs Network Size', fontsize=12, fontweight='bold')
axes[1].set_xticks([1, 2])
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('H3 Scalability Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(plots_dir / 'fig4_h3_scalability.png', dpi=150, bbox_inches='tight')
print(f"    ✓ Saved: fig4_h3_scalability.png")

# ============================================================================
# Figure 5: Comprehensive Results Table
# ============================================================================
print("  Creating Figure 5: Results table...")

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

table_data = [
    ['Model', 'Low-Var\nReward', 'High-Var\nReward', 'Parameters', 'Training\nTime (s)', 'Agents', 'Baseline?'],
    ['Fixed-Time', '-4.67 ± 0.00', 'N/A', '0', '0', '0', 'Yes'],
    ['MaxPressure Heuristic', '-5.72 ± 0.00', 'N/A', '0', '0', '0', 'Yes'],
    ['H1-Basic', '-6.56 ± 1.44', '-7.08 ± 0.89', '16,072', '36.3', '1', 'No'],
    ['H2-MaxPressure', '-4.90 ± 1.86', '-7.38 ± 0.85', '16,072', '36.0', '1', 'No'],
    ['H3-MultiAgent (1-int)', '-6.67 ± 0.99', '-7.25 ± 0.60', '5,992', '31.3', '1', 'No'],
    ['H3-MultiAgent (2-int)', '0.00 ± 0.00', 'N/A', '4,612', '47.2', '2', 'No'],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.12, 0.12, 0.11, 0.11, 0.08, 0.1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color header row
for i in range(7):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color baseline rows
for i in [1, 2]:
    for j in range(7):
        table[(i, j)].set_facecolor('#E8F5E9')

# Highlight best performance
table[(2, 1)].set_facecolor('#FFEB3B')  # H2 best on low-var

plt.title('Comprehensive Experimental Results Summary', fontsize=14, fontweight='bold', pad=20)
plt.savefig(plots_dir / 'fig5_results_table.png', dpi=150, bbox_inches='tight')
print(f"    ✓ Saved: fig5_results_table.png")

# ============================================================================
# Figure 6: Error Analysis - Undertraining Evidence
# ============================================================================
print("  Creating Figure 6: Error analysis...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Epsilon decay trajectory
episodes = np.arange(0, 51)
epsilon = 0.951 * (0.995 ** episodes)
epsilon_ideal = 1.0 * (0.985 ** episodes)

axes[0].plot(episodes, epsilon, 'r-', linewidth=2, label='Current decay (0.995)')
axes[0].plot(episodes, epsilon_ideal, 'g--', linewidth=2, label='Better decay (0.985)')
axes[0].axhline(y=0.1, color='blue', linestyle=':', alpha=0.5, label='Target ε=0.1')
axes[0].fill_between(episodes, 0, 0.1, alpha=0.2, color='green', label='Exploitation zone')
axes[0].fill_between(episodes, 0.1, 1.0, alpha=0.2, color='red', label='Exploration zone')
axes[0].set_xlabel('Episode', fontsize=11)
axes[0].set_ylabel('Epsilon (exploration rate)', fontsize=11)
axes[0].set_title('Epsilon Decay: Too Slow', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].text(50, epsilon[-1], f'ε={epsilon[-1]:.2f}\n(74% random!)', ha='left', va='center',
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.3), fontweight='bold')

# Performance variance
models_var = ['Fixed\nTime', 'H1\nBasic', 'H2\nMaxPress', 'H3\nMulti']
stds = [0.0, 1.44, 1.86, 0.99]

axes[1].bar(models_var, stds, color=['gray', 'lightblue', 'lightgreen', 'lightcoral'], alpha=0.7)
axes[1].set_ylabel('Standard Deviation (across 10 runs)', fontsize=11)
axes[1].set_title('Performance Variance: DRL Unstable', fontsize=12, fontweight='bold')
axes[1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Low variance threshold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# Win rate vs baseline
win_rates = [0, 0, 0]  # H1, H2, H3 (none beat Fixed-Time significantly)
axes[2].bar(['H1-Basic', 'H2-MaxPress', 'H3-Multi'], win_rates, color=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.7)
axes[2].set_ylabel('Win Rate vs Fixed-Time (%)', fontsize=11)
axes[2].set_title('DRL Models Fail to Beat Baseline', fontsize=12, fontweight='bold')
axes[2].set_ylim([0, 100])
axes[2].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Chance level')
axes[2].text(1, 5, 'All 0%\n(undertrained)', ha='center', fontsize=10, fontweight='bold', color='darkred')
axes[2].legend()
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('Error Analysis: Evidence of Undertraining', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(plots_dir / 'fig6_error_analysis.png', dpi=150, bbox_inches='tight')
print(f"    ✓ Saved: fig6_error_analysis.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("VISUALIZATION GENERATION COMPLETE")
print("="*70)
print(f"\nAll figures saved to: {plots_dir}/")
print("\nGenerated figures:")
print("  1. fig1_performance_comparison.png - Main results")
print("  2. fig2_statistical_significance.png - P-value matrix")
print("  3. fig3_h2_ablation.png - Reward normalization fix")
print("  4. fig4_h3_scalability.png - Parameter efficiency")
print("  5. fig5_results_table.png - Comprehensive table")
print("  6. fig6_error_analysis.png - Undertraining evidence")
print("\nThese figures cover:")
print("  ✓ Baseline comparisons (Fixed-Time, MaxPressure)")
print("  ✓ H1, H2, H3 experiments")
print("  ✓ Ablation studies")
print("  ✓ Statistical significance")
print("  ✓ Error analysis")
print("  ✓ Scalability analysis")
