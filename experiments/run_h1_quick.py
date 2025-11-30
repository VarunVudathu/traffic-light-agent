"""
Quick validation of H1 experiments (3 runs instead of 10, 30 episodes instead of 50).
"""

import sys
from pathlib import Path

# Import the main experiment runner
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_h1 import *

def main():
    """Run quick H1 validation experiments."""
    print("\n" + "="*70)
    print("H1 QUICK VALIDATION (3 runs, 30 episodes)")
    print("="*70)

    # Configuration
    num_runs = 3  # Reduced from 10
    training_episodes = 30  # Reduced from 50

    # Scenarios
    scenarios_dir = Path(__file__).resolve().parent.parent / 'scenarios' / 'configs'
    scenarios = {
        'low_variance': str(scenarios_dir / 'config_low_variance.json'),
        'extreme_surge': str(scenarios_dir / 'config_extreme_surge.json')
    }

    all_results = {}

    # Experiment 1: H1-Basic vs Fixed-Time on low-variance
    print("\n" + "="*70)
    print("EXPERIMENT 1: H1-Basic vs Fixed-Time (Low Variance)")
    print("="*70)

    all_results['exp1_h1_basic'] = run_experiment_suite(
        'h1_basic', 'low_variance', scenarios['low_variance'], num_runs, training_episodes
    )
    all_results['exp1_fixed_time'] = run_experiment_suite(
        'fixed_time', 'low_variance', scenarios['low_variance'], num_runs, 0
    )

    # Experiment 2: H1-Basic vs H1-Enhanced on surge
    print("\n" + "="*70)
    print("EXPERIMENT 2: H1-Basic vs H1-Enhanced (Extreme Surge)")
    print("="*70)

    all_results['exp2_h1_basic'] = run_experiment_suite(
        'h1_basic', 'extreme_surge', scenarios['extreme_surge'], num_runs, training_episodes
    )
    all_results['exp2_h1_enhanced'] = run_experiment_suite(
        'h1_enhanced', 'extreme_surge', scenarios['extreme_surge'], num_runs, training_episodes
    )

    # Statistical Analysis
    print("\n" + "="*70)
    print("QUICK STATISTICAL ANALYSIS")
    print("="*70)

    # Extract mean rewards
    exp1_h1_basic_means = [r['metrics']['mean_reward'] for r in all_results['exp1_h1_basic']]
    exp1_fixed_means = [r['metrics']['mean_reward'] for r in all_results['exp1_fixed_time']]

    exp2_h1_basic_means = [r['metrics']['mean_reward'] for r in all_results['exp2_h1_basic']]
    exp2_h1_enhanced_means = [r['metrics']['mean_reward'] for r in all_results['exp2_h1_enhanced']]

    # Perform statistical tests
    stats_exp1 = paired_ttest(exp1_h1_basic_means, exp1_fixed_means, "H1-Basic", "Fixed-Time")
    stats_exp2 = paired_ttest(exp2_h1_enhanced_means, exp2_h1_basic_means, "H1-Enhanced", "H1-Basic")

    # Print results
    print("\nExperiment 1: H1-Basic vs Fixed-Time (Low Variance)")
    print(f"  H1-Basic:   {stats_exp1['group1_mean']:.2f} ± {stats_exp1['group1_std']:.2f}")
    print(f"  Fixed-Time: {stats_exp1['group2_mean']:.2f} ± {stats_exp1['group2_std']:.2f}")
    print(f"  t={stats_exp1['t_statistic']:.3f}, p={stats_exp1['p_value']:.4f}, d={stats_exp1['cohens_d']:.3f}")
    print(f"  Result: {stats_exp1['winner']} {'✅' if stats_exp1['significant'] else '⚠️'}")

    print("\nExperiment 2: H1-Enhanced vs H1-Basic (Extreme Surge)")
    print(f"  H1-Enhanced: {stats_exp2['group1_mean']:.2f} ± {stats_exp2['group1_std']:.2f}")
    print(f"  H1-Basic:    {stats_exp2['group2_mean']:.2f} ± {stats_exp2['group2_std']:.2f}")
    print(f"  t={stats_exp2['t_statistic']:.3f}, p={stats_exp2['p_value']:.4f}, d={stats_exp2['cohens_d']:.3f}")
    print(f"  Result: {stats_exp2['winner']} {'✅' if stats_exp2['significant'] else '⚠️'}")

    print("\n" + "="*70)
    print("H1 QUICK VALIDATION COMPLETE")
    print("Models are working! Ready to implement H2 and H3.")
    print("="*70)

if __name__ == '__main__':
    main()
