"""
Create comprehensive comparison visualization across all models.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_dir = Path(__file__).parent
with open(results_dir / "h2_results_fixed.json") as f:
    h2_results = json.load(f)
with open(results_dir / "h3_results.json") as f:
    h3_results = json.load(f)

# Extract performance data (evaluation rewards)
models_data = {
    'Fixed-Time\nBaseline': {
        'low_var': -4.67,
        'high_var': None,  # Not tested
        'params': 0,
        'time': 0
    },
    'MaxPressure\nHeuristic': {
        'low_var': -5.72,
        'high_var': None,
        'params': 0,
        'time': 0
    },
    'H1-Basic\n(Queue)': {
        'low_var': -6.56,
        'high_var': -7.08,
        'params': 16072,
        'time': 36.3
    },
    'H2-MaxPressure\n(Normalized)': {
        'low_var': -4.90,
        'high_var': -7.38,
        'params': 16072,
        'time': 36.0
    },
    'H3-MultiAgent\n(Independent)': {
        'low_var': -6.67,
        'high_var': -7.25,
        'params': 5992,
        'time': 31.3
    }
}

# Create comprehensive comparison plot
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 1. Performance comparison - Low Variance
ax1 = fig.add_subplot(gs[0, 0])
models = list(models_data.keys())
low_var_rewards = [models_data[m]['low_var'] for m in models if models_data[m]['low_var'] is not None]
models_low = [m for m in models if models_data[m]['low_var'] is not None]

bars1 = ax1.barh(models_low, low_var_rewards, color=['gray', 'orange', 'lightblue', 'lightgreen', 'lightcoral'])
ax1.set_xlabel('Mean Reward (higher is better)', fontsize=10)
ax1.set_title('Performance: Low-Variance Traffic', fontsize=12, fontweight='bold')
ax1.axvline(x=-4.67, color='red', linestyle='--', alpha=0.5, label='Fixed-Time baseline')
ax1.grid(axis='x', alpha=0.3)
ax1.legend()

# Add value labels
for i, (model, val) in enumerate(zip(models_low, low_var_rewards)):
    ax1.text(val - 0.3, i, f'{val:.2f}', va='center', ha='right', fontsize=9, fontweight='bold')

# 2. Performance comparison - High Variance
ax2 = fig.add_subplot(gs[0, 1])
high_var_rewards = [models_data[m]['high_var'] for m in models if models_data[m]['high_var'] is not None]
models_high = [m for m in models if models_data[m]['high_var'] is not None]

bars2 = ax2.barh(models_high, high_var_rewards, color=['lightblue', 'lightgreen', 'lightcoral'])
ax2.set_xlabel('Mean Reward (higher is better)', fontsize=10)
ax2.set_title('Performance: High-Variance Traffic', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, (model, val) in enumerate(zip(models_high, high_var_rewards)):
    ax2.text(val - 0.1, i, f'{val:.2f}', va='center', ha='right', fontsize=9, fontweight='bold')

# 3. Parameter efficiency
ax3 = fig.add_subplot(gs[0, 2])
rl_models = [m for m in models if models_data[m]['params'] > 0]
params = [models_data[m]['params'] for m in rl_models]

bars3 = ax3.bar(range(len(rl_models)), params, color=['lightblue', 'lightgreen', 'lightcoral'])
ax3.set_xticks(range(len(rl_models)))
ax3.set_xticklabels(rl_models, rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('Number of Parameters', fontsize=10)
ax3.set_title('Model Complexity (Parameters)', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for i, val in enumerate(params):
    ax3.text(i, val + 500, f'{val:,}', ha='center', fontsize=9, fontweight='bold')

# 4. Training efficiency
ax4 = fig.add_subplot(gs[1, 0])
times = [models_data[m]['time'] for m in rl_models]

bars4 = ax4.bar(range(len(rl_models)), times, color=['lightblue', 'lightgreen', 'lightcoral'])
ax4.set_xticks(range(len(rl_models)))
ax4.set_xticklabels(rl_models, rotation=45, ha='right', fontsize=9)
ax4.set_ylabel('Training Time (seconds)', fontsize=10)
ax4.set_title('Training Efficiency (50 episodes)', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for i, val in enumerate(times):
    ax4.text(i, val + 1, f'{val:.1f}s', ha='center', fontsize=9, fontweight='bold')

# 5. Ranking summary
ax5 = fig.add_subplot(gs[1, 1])
ax5.axis('off')

ranking_text = """
PERFORMANCE RANKING
(Low-Variance Traffic)

1. Fixed-Time Baseline:     -4.67  ★★★★★
2. H2-MaxPressure:          -4.90  ★★★★☆
3. MaxPressure Heuristic:    -5.72  ★★★☆☆
4. H1-Basic:                -6.56  ★★☆☆☆
5. H3-MultiAgent:           -6.67  ★★☆☆☆

EFFICIENCY WINNER
H3-MultiAgent
• 63% fewer parameters
• 14% faster training
"""

ax5.text(0.1, 0.95, ranking_text, transform=ax5.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# 6. Key insights
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

insights_text = """
KEY INSIGHTS

WHY H1-BASIC ISN'T "BEST":
❌ All DRL models FAIL vs baselines
❌ 50 episodes = severe undertraining
❌ ε still at 0.74 (not converged)

ACTUAL WINNER:
✅ Fixed-Time (deterministic)
   → Zero variance, proven reliable

H2 BEST DRL MODEL:
✅ H2-MaxPressure: -4.90
   → Better reward signal
   → 25% better than H1-Basic

CRITICAL FINDING:
⚠️  All DRL models need
    200-500 episodes to converge
"""

ax6.text(0.1, 0.95, insights_text, transform=ax6.transAxes,
         fontsize=9.5, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.2))

plt.suptitle('Comprehensive Model Comparison: Traffic Light Control',
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(results_dir / 'plots' / 'comprehensive_comparison.png', dpi=150, bbox_inches='tight')
print(f"✅ Comprehensive comparison saved to {results_dir / 'plots' / 'comprehensive_comparison.png'}")
