"""Shared training machinery for DDPG and TD3.

Both algorithms are trained through this single code path with identical
hyperparameters, identical environments, identical seeds and identical budgets.
That is the point: if DDPG and TD3 differ in the results, the difference is
attributable to the algorithm and not to the tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.glucose_env import make_eval_env, make_train_env

ALGOS = {"ddpg": DDPG, "td3": TD3}

# Identical for both algorithms. Do not tune these per-algorithm.
HYPERPARAMS = dict(
    learning_rate=1e-3,
    buffer_size=200_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    learning_starts=5_000,
    train_freq=(1, "episode"),
    gradient_steps=-1,
    policy_kwargs=dict(net_arch=[256, 256]),
)

ACTION_NOISE_SIGMA = 0.1


class ProgressCallback(BaseCallback):
    """Prints episode returns periodically so long runs are not silent."""

    def __init__(self, every: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.every = every
        self.episode_returns: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_returns.append(float(info["episode"]["r"]))
                n = len(self.episode_returns)
                if n % self.every == 0:
                    recent = np.mean(self.episode_returns[-self.every:])
                    print(
                        f"    episode {n:4d} | steps {self.num_timesteps:7d} | "
                        f"mean return (last {self.every}) = {recent:9.2f}",
                        flush=True,
                    )
        return True


def build_envs(seed: int, log_dir: Path):
    """Training env (randomised meals) and eval env (fixed scenario).

    VecNormalize statistics are shared: the eval env is wrapped with the SAME
    VecNormalize object as training, with training=False and reward
    normalisation off, so observations are scaled consistently.
    """
    train_env = DummyVecEnv([lambda: Monitor(make_train_env())])
    train_env.seed(seed)
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True, clip_obs=10.0
    )

    eval_env = DummyVecEnv([lambda: Monitor(make_eval_env())])
    eval_env.seed(seed)
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False
    )
    eval_env.obs_rms = train_env.obs_rms  # share running statistics
    return train_env, eval_env


def train(
    algo_name: str,
    seed: int = 0,
    total_timesteps: int = 300_000,
    outdir: str = "results",
    verbose: int = 0,
) -> Path:
    algo_name = algo_name.lower()
    if algo_name not in ALGOS:
        raise ValueError(f"unknown algo {algo_name!r}; choose from {list(ALGOS)}")

    run_dir = Path(outdir) / f"{algo_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_env, eval_env = build_envs(seed, run_dir)

    action_noise = NormalActionNoise(
        mean=np.zeros(1), sigma=ACTION_NOISE_SIGMA * np.ones(1)
    )

    model = ALGOS[algo_name](
        "MlpPolicy",
        train_env,
        action_noise=action_noise,
        seed=seed,
        verbose=verbose,
        tensorboard_log=str(run_dir / "tb"),
        **HYPERPARAMS,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "evals"),
        eval_freq=10_000,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
    )
    progress_cb = ProgressCallback(every=10)

    print(f"[{algo_name.upper()} seed={seed}] training for {total_timesteps} steps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, progress_cb],
        progress_bar=False,
    )

    model.save(str(run_dir / "final_model"))
    # Saving the normalizer is essential. Loading a policy without its
    # VecNormalize statistics is the single most common reason a trained agent
    # "works during training and fails at evaluation".
    train_env.save(str(run_dir / "vecnormalize.pkl"))

    with open(run_dir / "train_meta.json", "w") as f:
        json.dump(
            {
                "algo": algo_name,
                "seed": seed,
                "total_timesteps": total_timesteps,
                "hyperparams": {k: str(v) for k, v in HYPERPARAMS.items()},
                "action_noise_sigma": ACTION_NOISE_SIGMA,
                "episode_returns": progress_cb.episode_returns,
            },
            f,
            indent=2,
        )

    print(f"[{algo_name.upper()} seed={seed}] saved to {run_dir}")
    return run_dir


def cli(algo_name: str) -> None:
    ap = argparse.ArgumentParser(description=f"Train {algo_name.upper()}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--verbose", type=int, default=0)
    a = ap.parse_args()
    train(algo_name, a.seed, a.timesteps, a.outdir, a.verbose)
