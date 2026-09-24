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
    time_of_day: bool = cfg.OBS_TIME_OF_DAY,
) -> GlucoseEnv:
    env = GlucoseEnv(
        meals=meals,
        randomize_meals=False,
        randomize_initial_state=False,
        target=target,
        u_max=u_max,
        time_of_day=time_of_day,
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
    tod = expects_time_of_day(model)

    norm_path = run_dir / "vecnormalize.pkl"
    vec = None
    if norm_path.exists():
        dummy = DummyVecEnv([lambda: make_eval_env(time_of_day=tod)])
        vec = VecNormalize.load(str(norm_path), dummy)
        vec.training = False
        vec.norm_reward = False
    return model, vec


def expects_time_of_day(model) -> bool:
    """Whether a policy was trained with the time-of-day input (5-D obs) or on
    the ablation (4-D). Read from the policy itself so it cannot disagree."""
    return model.observation_space.shape[0] == 5


def run_agent(meals, model, vec, deterministic: bool = True, **kw) -> dict:
    """Roll a trained agent out on the true (unnormalised) plant."""
    env = build_env(meals, time_of_day=expects_time_of_day(model), **kw)

    def policy(e):
        obs = e._observation().reshape(1, -1)
        if vec is not None:
            obs = vec.normalize_obs(obs)
        action, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(action).reshape(-1)

    return _rollout(env, policy)


def discover_runs(results_dir: str | Path = "results",
                  report_skipped: bool = True) -> dict[str, list[Path]]:
    """Find every complete trained run on disk, grouped by algorithm.

    A run needs both its policy and its VecNormalize stats; without the stats
    the agent sees unnormalised observations and its behaviour is meaningless.
    """
    results_dir = Path(results_dir)
    found: dict[str, list[Path]] = {"ddpg": [], "td3": []}
    if not results_dir.exists():
        return found
    required = (Path("best") / "best_model.zip", Path("vecnormalize.pkl"))
    for algo in found:
        for d in sorted(results_dir.glob(f"{algo}_seed*")):
            missing = [str(p) for p in required if not (d / p).exists()]
            if missing:
                if report_skipped:
                    print(f"discover_runs: skipping {d.name} (missing "
                          f"{', '.join(missing)})")
                continue
            found[algo].append(d)
    return found
