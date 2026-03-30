"""
ai/train.py
─────────────────────────────────────────────────────────────────
Trains the PPO RL agent on the air defense simulation.

Usage:
    python ai/train.py                    # train with defaults
    python ai/train.py --steps 2000000   # train longer
    python ai/train.py --difficulty hard  # train on hard only

The trained model is saved to ai/models/rl_router_best.zip
─────────────────────────────────────────────────────────────────
"""

import os
import sys
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ai.rl_environment import AirDefenseEnv


# ─────────────────────────────────────────
# Reward Logger Callback
# ─────────────────────────────────────────

class RewardLoggerCallback(BaseCallback):
    """
    Logs episode rewards and success rates during training.
    Saves stats to training_data/training_logs/episode_stats.csv
    """

    def __init__(self, log_path: str, verbose=0):
        super().__init__(verbose)
        self.log_path    = log_path
        self.episode_rewards = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        with open(log_path, "w") as f:
            f.write("episode,reward,intercepted,breached,success_rate\n")

    def _on_step(self) -> bool:
        # Log completed episodes
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.episode_rewards.append(ep_reward)

                intercepted   = info.get("intercepted", 0)
                breached      = info.get("breached", 0)
                total         = info.get("total", 1)
                success_rate  = intercepted / total * 100 if total > 0 else 0

                with open(self.log_path, "a") as f:
                    f.write(f"{len(self.episode_rewards)},"
                            f"{ep_reward:.2f},"
                            f"{intercepted},"
                            f"{breached},"
                            f"{success_rate:.1f}\n")

                if self.verbose and len(self.episode_rewards) % 100 == 0:
                    recent = self.episode_rewards[-100:]
                    print(f"  Episode {len(self.episode_rewards):>6} | "
                          f"Avg reward (last 100): {np.mean(recent):>8.1f} | "
                          f"Success: {success_rate:.1f}%")
        return True


# ─────────────────────────────────────────
# Training
# ─────────────────────────────────────────

def train(total_steps: int = 1_000_000, difficulty: str = "random",
          n_envs: int = 4):
    """
    Train PPO agent on the air defense environment.

    Args:
        total_steps : Total environment steps to train for
        difficulty  : "easy" | "medium" | "hard" | "random"
        n_envs      : Number of parallel environments (speeds up training)
    """

    print(f"\n{'═'*60}")
    print(f"  AIR DEFENSE RL TRAINING")
    print(f"  Difficulty : {difficulty}")
    print(f"  Steps      : {total_steps:,}")
    print(f"  Envs       : {n_envs} parallel")
    print(f"{'═'*60}\n")

    # ── Create environments ───────────────────────────────────────────
    def make_env():
        env = AirDefenseEnv(difficulty=difficulty)
        env = Monitor(env)
        return env

    train_env = make_vec_env(make_env, n_envs=n_envs)
    eval_env  = Monitor(AirDefenseEnv(difficulty=difficulty))

    # ── Callbacks ─────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = "ai/models/",
        name_prefix = "rl_router_checkpoint",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = "ai/models/",
        log_path             = "training_data/training_logs/",
        eval_freq            = 10_000,
        n_eval_episodes      = 20,
        deterministic        = True,
        verbose              = 1,
    )

    reward_logger_cb = RewardLoggerCallback(
        log_path = "training_data/training_logs/episode_stats.csv",
        verbose  = 1,
    )

    # ── PPO Model ─────────────────────────────────────────────────────
    model = PPO(
        policy             = "MlpPolicy",
        env                = train_env,
        learning_rate      = 3e-4,
        n_steps            = 2048,
        batch_size         = 64,
        n_epochs           = 10,
        gamma              = 0.99,          # discount factor
        gae_lambda         = 0.95,
        clip_range         = 0.2,
        ent_coef           = 0.01,          # entropy bonus — encourages exploration
        verbose            = 1,
        tensorboard_log    = "training_data/training_logs/tensorboard/",
    )

    # ── Train ─────────────────────────────────────────────────────────
    print("  Starting training...\n")
    model.learn(
        total_timesteps = total_steps,
        callback        = [checkpoint_cb, eval_cb, reward_logger_cb],
        progress_bar    = True,
    )

    # ── Save final model ──────────────────────────────────────────────
    model.save("ai/models/rl_router_final")
    print(f"\n  ✅ Training complete!")
    print(f"  Best model  → ai/models/best_model.zip")
    print(f"  Final model → ai/models/rl_router_final.zip")
    print(f"  Logs        → training_data/training_logs/\n")

    train_env.close()
    eval_env.close()
    return model


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Air Defense RL Agent")
    parser.add_argument("--steps",      type=int, default=1_000_000)
    parser.add_argument("--difficulty", type=str, default="random",
                        choices=["easy", "medium", "hard", "random"])
    parser.add_argument("--envs",       type=int, default=4)
    args = parser.parse_args()

    train(
        total_steps = args.steps,
        difficulty  = args.difficulty,
        n_envs      = args.envs,
    )
