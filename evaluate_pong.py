import ale_py
import gymnasium as gym
import imageio
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

# Important: Registration for Gymnasium v1.0+
gym.register_envs(ale_py)

ENV_ID = "PongNoFrameskip-v4"
SEEDS = [42, 123, 456, 789, 1011]  # Must match train_pong.py
EVAL_EPISODES = 100  # Increased from 20 for more robust evaluation


def make_eval_env(render_mode=None):
    """Create environment for inference (with optional rendering)."""
    env_kwargs = {"render_mode": render_mode} if render_mode else {}
    env = make_atari_env(ENV_ID, n_envs=1, seed=42, env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env


def record_gif(model, algorithm_name, length=1000):
    """Record a gameplay video as GIF."""
    print(f"🎬 Recording GIF for {algorithm_name}...")

    # Trick: Create a separate environment just for recording
    eval_env = make_eval_env(render_mode="rgb_array")

    obs = eval_env.reset()
    images = []

    for _ in range(length):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = eval_env.step(action)

        # Extract the raw RGB image from the wrapper
        img = eval_env.envs[0].unwrapped.render()
        images.append(img)

    gif_path = f"{algorithm_name}_gameplay.gif"
    imageio.mimsave(gif_path, images, fps=30)
    print(f"✅ GIF saved: {gif_path}")
    eval_env.close()


def evaluate_model(model_path, algo_class, name):
    """Evaluate a single model."""
    print(f"\n📊 Evaluating {name}...")
    try:
        # Load model (on GPU if available)
        model = algo_class.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

        # Measure performance (without rendering for speed)
        env = make_eval_env()
        from stable_baselines3.common.evaluation import evaluate_policy

        # Increased to 100 episodes for more robust evaluation
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=EVAL_EPISODES)
        env.close()

        print(f"🏆 Result {name}: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")

        return mean_reward, std_reward

    except FileNotFoundError:
        print(f"❌ File {model_path}.zip not found.")
        return None, None


def evaluate_all_seeds():
    """Evaluate all trained models across all seeds and compute statistics."""
    results = {"PPO": [], "DQN": []}

    print(f"\n{'=' * 60}")
    print("EVALUATING ALL MODELS")
    print(f"{'=' * 60}")
    print(f"Seeds: {SEEDS}")
    print(f"Evaluation episodes per model: {EVAL_EPISODES}")
    print(f"{'=' * 60}\n")

    # Evaluate all seeds for both algorithms
    for seed in SEEDS:
        print(f"\n--- Evaluating seed {seed} ({SEEDS.index(seed) + 1}/{len(SEEDS)}) ---")

        # Evaluate PPO
        mean_ppo, std_ppo = evaluate_model(
            f"./pong_results/ppo_pong_seed{seed}", PPO, f"PPO_seed{seed}"
        )
        if mean_ppo is not None:
            results["PPO"].append(mean_ppo)

        # Evaluate DQN
        mean_dqn, std_dqn = evaluate_model(
            f"./pong_results/dqn_pong_seed{seed}", DQN, f"DQN_seed{seed}"
        )
        if mean_dqn is not None:
            results["DQN"].append(mean_dqn)

    # Compute statistics
    print(f"\n{'=' * 60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'=' * 60}\n")

    for algo, rewards in results.items():
        if len(rewards) > 0:
            mean = np.mean(rewards)
            std = np.std(rewards)
            min_r = np.min(rewards)
            max_r = np.max(rewards)

            print(f"{algo}:")
            print(f"  Mean Reward: {mean:.2f} ± {std:.2f} (over {len(rewards)} seeds)")
            print(f"  Range: [{min_r:.2f}, {max_r:.2f}]")
            print(f"  Individual results: {[f'{r:.2f}' for r in rewards]}")
            print()

    # Compare algorithms
    if len(results["PPO"]) > 0 and len(results["DQN"]) > 0:
        ppo_mean = np.mean(results["PPO"])
        dqn_mean = np.mean(results["DQN"])
        difference = ppo_mean - dqn_mean
        improvement = (difference / abs(dqn_mean)) * 100 if dqn_mean != 0 else 0

        print(f"{'=' * 60}")
        print("COMPARISON")
        print(f"{'=' * 60}")
        print(f"PPO vs DQN: {difference:+.2f} points ({improvement:+.1f}% difference)")
        if abs(difference) > 1.0:
            winner = "PPO" if difference > 0 else "DQN"
            print(f"Winner: {winner}")
        else:
            print("Result: Approximately equal performance")
        print(f"{'=' * 60}\n")

    # Generate GIFs for best performing models
    print(f"\n{'=' * 60}")
    print("GENERATING GAMEPLAY GIFS")
    print(f"{'=' * 60}\n")

    if len(results["PPO"]) > 0:
        best_ppo_idx = np.argmax(results["PPO"])
        best_ppo_seed = SEEDS[best_ppo_idx]
        print(f"Best PPO model: seed {best_ppo_seed} (reward: {results['PPO'][best_ppo_idx]:.2f})")
        model_ppo = PPO.load(f"./pong_results/ppo_pong_seed{best_ppo_seed}")
        record_gif(model_ppo, "PPO")

    if len(results["DQN"]) > 0:
        best_dqn_idx = np.argmax(results["DQN"])
        best_dqn_seed = SEEDS[best_dqn_idx]
        print(f"Best DQN model: seed {best_dqn_seed} (reward: {results['DQN'][best_dqn_idx]:.2f})")
        model_dqn = DQN.load(f"./pong_results/dqn_pong_seed{best_dqn_seed}")
        record_gif(model_dqn, "DQN")

    return results


if __name__ == "__main__":
    # Evaluate all trained models with all seeds
    results = evaluate_all_seeds()

    print(f"\n{'=' * 60}")
    print("🏁 EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print("All models evaluated successfully!")
    print("Results saved in GIF files: PPO_gameplay.gif, DQN_gameplay.gif")
    print(f"{'=' * 60}")
