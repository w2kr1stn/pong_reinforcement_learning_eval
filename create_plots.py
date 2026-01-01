import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import interpolate
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Setup for academic design
sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.size": 12,
        "figure.dpi": 300,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    }
)

LOG_DIR = "./pong_results/"
OUTPUT_DIR = "./plots/"
SEEDS = [42, 123, 456, 789, 1011]
COLORS = {"PPO": "#1f77b4", "DQN": "#ff7f0e"}  # Blue, Orange


def extract_scalar(log_file, scalar_tag="rollout/ep_rew_mean"):
    """Read Event-Files directly from TensorBoard logs."""
    event_acc = EventAccumulator(log_file)
    event_acc.Reload()

    if scalar_tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(scalar_tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        return steps, values
    return None, None


def smooth_data(values, smoothing=0.85):
    """Smooth the curve (Exponential Moving Average) for better visualization."""
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = smoothing * smoothed[i - 1] + (1 - smoothing) * values[i]
    return smoothed


def load_all_seeds(algo):
    """Load data for all seeds of an algorithm."""
    all_steps = []
    all_values = []

    for seed in SEEDS:
        folder = os.path.join(LOG_DIR, f"{algo}_Pong_seed{seed}_1")
        if not os.path.exists(folder):
            print(f"⚠️  Folder not found: {folder}")
            continue

        event_files = glob.glob(os.path.join(folder, "events.out.tfevents.*"))
        if not event_files:
            continue

        latest_file = max(event_files, key=os.path.getsize)
        steps, values = extract_scalar(latest_file)

        if steps is not None:
            print(f"  Loaded {algo} seed {seed}: {len(steps)} data points")
            all_steps.append(steps)
            all_values.append(values)

    return all_steps, all_values


def interpolate_to_common_steps(all_steps, all_values, n_points=500):
    """Interpolate all runs to common step values for aggregation."""
    if not all_steps:
        return None, None

    # Find common step range (intersection of all runs)
    min_step = max(s.min() for s in all_steps)
    max_step = min(s.max() for s in all_steps)

    # Create common step grid
    common_steps = np.linspace(min_step, max_step, n_points)

    # Interpolate each run to common steps
    interpolated_values = []
    for steps, values in zip(all_steps, all_values, strict=False):
        f = interpolate.interp1d(steps, values, kind="linear", fill_value="extrapolate")
        interpolated_values.append(f(common_steps))

    return common_steps, np.array(interpolated_values)


def plot_with_variance(ax, algo, color, smoothing=0.85):
    """Plot mean ± std for an algorithm with fill_between."""
    print(f"\nLoading {algo} data...")
    all_steps, all_values = load_all_seeds(algo)

    if not all_steps:
        print(f"❌ No data found for {algo}")
        return

    # Interpolate to common steps
    common_steps, values_matrix = interpolate_to_common_steps(all_steps, all_values)

    if common_steps is None:
        return

    # Smooth each run individually
    smoothed_matrix = np.array([smooth_data(v, smoothing) for v in values_matrix])

    # Calculate statistics across seeds
    mean_values = np.mean(smoothed_matrix, axis=0)
    std_values = np.std(smoothed_matrix, axis=0)

    # Plot mean line
    ax.plot(
        common_steps,
        mean_values,
        label=f"{algo} (n={len(all_steps)} seeds)",
        color=color,
        linewidth=2.5,
    )

    # Plot std deviation as shaded area
    ax.fill_between(
        common_steps, mean_values - std_values, mean_values + std_values, color=color, alpha=0.2
    )

    return len(all_steps)


def plot_main_comparison():
    """Create the main comparison plot with mean ± std."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot both algorithms
    for algo in ["PPO", "DQN"]:
        plot_with_variance(ax, algo, COLORS[algo])

    # Reference lines
    ax.axhline(
        y=19,
        color="green",
        linestyle="--",
        alpha=0.6,
        linewidth=1.5,
        label="Solved threshold (≥19)",
    )
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=1)
    ax.axhline(
        y=-21, color="red", linestyle="--", alpha=0.4, linewidth=1, label="Worst possible (-21)"
    )

    ax.set_title(
        "Training Progress: DQN vs. PPO on Atari Pong\n(Mean ± Std over 5 Seeds)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Reward (Smoothed)")
    ax.set_ylim(-22, 22)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "learning_curve_mean_std.png")
    plt.savefig(output_path)
    plt.close()
    print(f"\n✅ Main plot saved: {output_path}")


def plot_individual_seeds():
    """Create a plot showing all individual seed runs."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, algo in enumerate(["PPO", "DQN"]):
        ax = axes[idx]
        all_steps, all_values = load_all_seeds(algo)

        for i, (steps, values) in enumerate(zip(all_steps, all_values, strict=False)):
            smoothed = smooth_data(values)
            ax.plot(steps, smoothed, label=f"Seed {SEEDS[i]}", alpha=0.8, linewidth=1.5)

        ax.axhline(y=19, color="green", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_title(f"{algo} - Individual Seed Runs")
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Episode Reward (Smoothed)")
        ax.set_ylim(-22, 22)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "individual_seeds.png")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Individual seeds plot saved: {output_path}")


def plot_final_performance_bars():
    """Create a bar chart comparing final performance."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    final_rewards = {"PPO": [], "DQN": []}

    for algo in ["PPO", "DQN"]:
        all_steps, all_values = load_all_seeds(algo)
        for values in all_values:
            # Take mean of last 10% of training as "final" performance
            final_portion = values[int(len(values) * 0.9) :]
            final_rewards[algo].append(np.mean(final_portion))

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(2)
    means = [np.mean(final_rewards["PPO"]), np.mean(final_rewards["DQN"])]
    stds = [np.std(final_rewards["PPO"]), np.std(final_rewards["DQN"])]

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=8,
        color=[COLORS["PPO"], COLORS["DQN"]],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["PPO", "DQN"])
    ax.set_ylabel("Final Episode Reward")
    ax.set_title("Final Performance Comparison\n(Mean ± Std, last 10% of training)")
    ax.axhline(y=19, color="green", linestyle="--", alpha=0.5, label="Solved (≥19)")
    ax.set_ylim(0, 22)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.5,
            f"{mean:.1f}±{std:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "final_performance_bars.png")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Final performance bar chart saved: {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    # Main comparison plot (mean ± std)
    plot_main_comparison()

    # Individual seed runs
    plot_individual_seeds()

    # Final performance bar chart
    plot_final_performance_bars()

    print("\n" + "=" * 60)
    print("ALL PLOTS GENERATED")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
