import argparse
import time
from collections.abc import Callable

import ale_py
import gymnasium as gym
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

# IMPORTANT: Register Atari environments in Gymnasium
# Required for Gymnasium v1.0+ compatibility with ALE
gym.register_envs(ale_py)

# --- HARDWARE & HYPERPARAMETER CONFIGURATION ---
# Hardware: RTX 3060 (Mobile), 8 Physical CPU Cores, 64GB RAM
ENV_ID = "PongNoFrameskip-v4"
TOTAL_TIMESTEPS = 3_000_000  # Increased to 3M for clearer results
LOG_DIR = "./pong_results/"
SEEDS = [42, 123, 456, 789, 1011]  # Multiple seeds for statistical validity

# IMPORTANT: Prevent CPU thrashing with parallel environments!
# With 8 parallel envs, PyTorch must use only 1 thread per env to avoid oversubscription
torch.set_num_threads(1)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate decay."""

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def make_env(n_envs=1, seed=42):
    """
    Create vectorized Atari environment with standard preprocessing.
    make_atari_env applies default wrappers (NoopReset, MaxAndSkip, etc.).
    """
    # Multiprocessing starts n_envs independent instances of Pong
    env = make_atari_env(ENV_ID, n_envs=n_envs, seed=seed)
    # Stack 4 frames to provide temporal/motion information
    env = VecFrameStack(env, n_stack=4)
    # Transpose images for PyTorch (channels-first format)
    env = VecTransposeImage(env)
    return env


def train_ppo(seed=42):
    """Train PPO with a specific random seed."""
    print(f"\n🚀 STARTING PPO TRAINING (Seed: {seed})")
    print("   -> Using 8 parallel environments (full CPU core utilization)")

    # Optimization: 8 envs = 1 per physical CPU core
    n_envs = 8
    env = make_env(n_envs=n_envs, seed=seed)

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=linear_schedule(2.5e-4),
        n_steps=128,
        batch_size=256,  # Fits comfortably in RTX 3060 VRAM
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=linear_schedule(0.1),
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cuda",  # Explicit GPU usage
    )

    start_time = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=f"PPO_Pong_seed{seed}")
    end_time = time.time()

    print(f"✅ PPO (seed {seed}) completed in {(end_time - start_time) / 60:.2f} minutes.")
    model.save(LOG_DIR + f"ppo_pong_seed{seed}")
    env.close()


def train_ppo_all_seeds():
    """Train PPO with multiple random seeds for statistical validity."""
    for seed in SEEDS:
        print(f"\n{'=' * 60}")
        print(f"Training PPO with seed {seed} ({SEEDS.index(seed) + 1}/{len(SEEDS)})")
        print(f"{'=' * 60}")
        train_ppo(seed=seed)


def train_dqn(seed=42):
    """Train DQN with a specific random seed."""
    print(f"\n🐢 STARTING DQN TRAINING (Seed: {seed})")
    print("   -> Using 8 parallel environments for fair comparison with PPO")

    # DQN DOES support parallel environments in Stable-Baselines3
    # Changed from n_envs=1 to n_envs=8 for fair comparison with PPO
    env = make_env(n_envs=8, seed=seed)

    model = DQN(
        "CnnPolicy",
        env,
        buffer_size=100_000,  # 100k × ~1KB ≈ 100MB RAM
        learning_starts=10_000,
        batch_size=32,
        learning_rate=1e-4,  # Constant LR (DQN works best without decay, unlike PPO)
        target_update_interval=1000,
        train_freq=1,  # Compensate for n_envs=8 (was 4 with n_envs=1)
        gradient_steps=2,  # 375k steps × 2 = 750k updates (matches n_envs=1 baseline)
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cuda",
    )

    start_time = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=f"DQN_Pong_seed{seed}")
    end_time = time.time()

    print(f"✅ DQN (seed {seed}) completed in {(end_time - start_time) / 60:.2f} minutes.")
    model.save(LOG_DIR + f"dqn_pong_seed{seed}")
    env.close()


def train_dqn_all_seeds():
    """Train DQN with multiple random seeds for statistical validity."""
    for seed in SEEDS:
        print(f"\n{'=' * 60}")
        print(f"Training DQN with seed {seed} ({SEEDS.index(seed) + 1}/{len(SEEDS)})")
        print(f"{'=' * 60}")
        train_dqn(seed=seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DQN and/or PPO on Atari Pong",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_pong.py              # Train both PPO and DQN (all seeds)
    python train_pong.py --ppo-only   # Train only PPO
    python train_pong.py --dqn-only   # Train only DQN
        """,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ppo-only", action="store_true", help="Train only PPO")
    group.add_argument("--dqn-only", action="store_true", help="Train only DQN")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine what to train
    run_ppo = not args.dqn_only
    run_dqn = not args.ppo_only

    # Hardware check
    if torch.cuda.is_available():
        print(f"🔧 Hardware Check: Using {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ WARNING: No GPU detected! Training will take significantly longer.")

    # Calculate training runs
    algos = []
    if run_ppo:
        algos.append("PPO")
    if run_dqn:
        algos.append("DQN")
    total_runs = len(algos) * len(SEEDS)

    print(f"\n{'=' * 60}")
    print("TRAINING CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Environment: {ENV_ID}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Random seeds: {SEEDS}")
    print(f"Algorithms: {', '.join(algos)}")
    print(f"Total training runs: {total_runs}")
    print(f"{'=' * 60}\n")

    # Training
    if run_ppo:
        train_ppo_all_seeds()
    if run_dqn:
        train_dqn_all_seeds()

    # Summary
    print(f"\n{'=' * 60}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print("Trained models:")
    for seed in SEEDS:
        if run_ppo:
            print(f"  - ppo_pong_seed{seed}.zip")
        if run_dqn:
            print(f"  - dqn_pong_seed{seed}.zip")
    print("\nUse evaluate_pong.py to evaluate all models.")
    print(f"{'=' * 60}")
