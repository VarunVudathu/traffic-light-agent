from models2 import FixedLightBaseline, SingleAgent, EnhancedSingleAgent, MultiAgent
import matplotlib.pyplot as plt
import numpy as np

def plot_wait_times(metrics_dict):
    """Plots the average wait times for each model as a bar chart."""
    names = list(metrics_dict.keys())
    wait_times = [m['average_delay'] for m in metrics_dict.values()]

    plt.figure(figsize=(10, 6))
    plt.bar(names, wait_times, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel("Model")
    plt.ylabel("Average Wait Time (s)")
    plt.title("Average Vehicle Wait Time Comparison")
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    plt.show()

def main():
    # Run all models
    print("== FixedLightBaseline ==")
    fixed = FixedLightBaseline()
    fixed_reward, fixed_metrics = fixed.run()

    # print("== SingleAgent ==")
    # single = SingleAgent()
    # single_reward, single_metrics = single.run()

    # print("== EnhancedSingleAgent ==")
    # enhanced = EnhancedSingleAgent()
    # enhanced_reward, enhanced_metrics = enhanced.run()

    # print("== MultiAgent ==")
    # multi = MultiAgent()
    # multi_reward, multi_metrics = multi.run()

    # Print metrics
    print("\n--- Metrics Summary ---")
    metrics_summary = {
        "FixedLightBaseline": fixed_metrics,
        # "SingleAgent": single_metrics,
        # "EnhancedSingleAgent": enhanced_metrics,
        # "MultiAgent": multi_metrics
    }

    for name, metrics in metrics_summary.items():
        print(f"\n{name}:")
        for key, val in metrics.items():
            print(f"{key}: {val:.2f}")

    # Plot reward curves
    plt.plot(fixed_reward, label="FixedLightBaseline")
    # plt.plot(single_reward, label="SingleAgent")
    # plt.plot(enhanced_reward, label="EnhancedSingleAgent")
    # plt.plot(multi_reward, label="MultiAgent")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward Comparison")
    plt.legend()
    plt.show()

    # Plot average wait times
    plot_wait_times(metrics_summary)

if __name__ == "__main__":
    main()
