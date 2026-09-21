"""One place where every controller is rolled out.

The UI and the CLI both call these functions, so an experiment you run in the
dashboard is the same computation as the one in `eval/compare.py`. If the two
ever disagree, that is a bug, not a configuration difference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from configs import scenario as cfg
from controllers.pid import PIDController
from envs.bergman import Meal
from envs.glucose_env import GlucoseEnv


def build_env(
    meals: list[Meal],
    target: float = cfg.TARGET,
    u_max: float = cfg.U_MAX,
    G0: float | None = None,
) -> GlucoseEnv:
    env = GlucoseEnv(
        meals=meals,
        randomize_meals=False,
        randomize_initial_state=False,
        target=target,
        u_max=u_max,
    )
    env.reset()
    if G0 is not None:
        env.model.state[0] = float(G0)
    return env


def _rollout(env: GlucoseEnv, policy) -> dict:
    """Drive `env` to termination with `policy(env) -> action`."""
    done = False
    while not done:
        action = policy(env)
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env.history


def run_open_loop(meals, **kw) -> dict:
    """Constant basal infusion, no feedback. The 'do nothing clever' floor."""
    env = build_env(meals, **kw)
    u = env.model.basal_infusion_for(env.target)
    return _rollout(env, lambda e: e.insulin_to_action(u))


def run_pid(meals, Kp: float, Ki: float, Kd: float, **kw) -> dict:
    env = build_env(meals, **kw)
    pid = PIDController(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        setpoint=env.target,
        u_min=0.0,
        u_max=env.u_max,
        u_bias=env.model.basal_infusion_for(env.target),
    )
    return _rollout(env, lambda e: e.insulin_to_action(pid.update(e.model.G, dt=e.dt)))


def load_agent(run_dir: Path, algo: str):
    """Load an SB3 agent together with its VecNormalize statistics.

    Forgetting the normaliser is the single most common way a trained SB3 agent
    appears to perform terribly at evaluation, so it is loaded here rather than
    left to each caller.
    """
    from stable_baselines3 import DDPG, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from envs.glucose_env import make_eval_env

    run_dir = Path(run_dir)
    cls = {"ddpg": DDPG, "td3": TD3}[algo.lower()]

    model_path = run_dir / "best" / "best_model.zip"
    if not model_path.exists():
        model_path = run_dir / "final_model.zip"
    if not model_path.exists():
        return None, None

    model = cls.load(str(model_path), device="cpu")

    norm_path = run_dir / "vecnormalize.pkl"
    vec = None
    if norm_path.exists():
        dummy = DummyVecEnv([make_eval_env])
        vec = VecNormalize.load(str(norm_path), dummy)
        vec.training = False
        vec.norm_reward = False
    return model, vec


def run_agent(meals, model, vec, deterministic: bool = True, **kw) -> dict:
    """Roll a trained agent out on the true (unnormalised) plant."""
    env = build_env(meals, **kw)

    def policy(e):
        obs = e._observation().reshape(1, -1)
        if vec is not None:
            obs = vec.normalize_obs(obs)
        action, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(action).reshape(-1)

    return _rollout(env, policy)


def discover_runs(results_dir: str | Path = "results") -> dict[str, list[Path]]:
    """Find every trained run on disk, grouped by algorithm."""
    results_dir = Path(results_dir)
    found: dict[str, list[Path]] = {"ddpg": [], "td3": []}
    if not results_dir.exists():
        return found
    for algo in found:
        for d in sorted(results_dir.glob(f"{algo}_seed*")):
            if (d / "best" / "best_model.zip").exists() or (
                d / "final_model.zip"
            ).exists():
                found[algo].append(d)
    return found
