"""Run PID, DDPG and TD3 on the fixed evaluation scenario and produce
Graphs 1-4 plus the metrics table.

Usage
-----
    python -m eval.compare                      # all available controllers
    python -m eval.compare --seeds 0 1 2        # aggregate over seeds
    python -m eval.compare --pid-only           # before any RL training exists
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from configs import scenario as cfg  # noqa: E402
from controllers.pid import default_pid  # noqa: E402
from envs.glucose_env import make_eval_env  # noqa: E402
from eval.metrics import compute_metrics, metrics_table  # noqa: E402

COLORS = {"PID": "#D55E00", "DDPG": "#0072B2", "TD3": "#009E73"}


# ---------------------------------------------------------------------- #
# rollouts
# ---------------------------------------------------------------------- #
def run_pid() -> dict:
    env = make_eval_env()
    pid = default_pid(
        u_bias=env.model.basal_infusion_for(env.target),
        u_max=env.u_max,
        setpoint=env.target,
    )
    env.reset()
    pid.reset()
    done = False
    while not done:
        u = pid.update(env.model.G, dt=env.dt)
        _, _, term, trunc, _ = env.step(env.insulin_to_action(u))
        done = term or trunc
    return env.history


def run_agent(run_dir: Path, algo: str) -> dict | None:
    """Roll out a trained SB3 agent, restoring its VecNormalize statistics."""
    from stable_baselines3 import DDPG, TD3
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    cls = {"ddpg": DDPG, "td3": TD3}[algo.lower()]

    model_path = run_dir / "best" / "best_model.zip"
    if not model_path.exists():
        model_path = run_dir / "final_model.zip"
    if not model_path.exists():
        return None

    # The rollout runs on the RAW env, not through DummyVecEnv. A VecEnv
    # auto-resets the moment an episode ends, which clears env.history and
    # leaves nothing to score. So the observation normalisation is applied by
    # hand from the saved statistics instead.
    norm_path = run_dir / "vecnormalize.pkl"
    if norm_path.exists():
        stub = DummyVecEnv([lambda: Monitor(make_eval_env())])
        vecnorm = VecNormalize.load(str(norm_path), stub)
        vecnorm.training = False
        vecnorm.norm_reward = False

        def normalize(o):
            return vecnorm.normalize_obs(o.reshape(1, -1)).astype(np.float32)
    else:
        # Loading a policy without its normalisation statistics is the single
        # most common cause of "great in training, useless at evaluation".
        print(f"  WARNING: {norm_path} missing; observations will not be scaled.")

        def normalize(o):
            return o.reshape(1, -1)

    model = cls.load(str(model_path), device="cpu")

    env = make_eval_env()
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(normalize(obs), deterministic=True)
        obs, _, term, trunc, _ = env.step(np.asarray(action).reshape(-1))
        done = term or trunc
    return env.history


def collect(results_dir: Path, seeds: list[int], pid_only: bool) -> dict:
    runs: dict[str, list[dict]] = {"PID": [run_pid()]}
    if pid_only:
        return runs
    for algo, label in (("ddpg", "DDPG"), ("td3", "TD3")):
        hists = []
        for s in seeds:
            d = results_dir / f"{algo}_seed{s}"
            if not d.exists():
                continue
            h = run_agent(d, algo)
            if h is not None:
                hists.append(h)
        if hists:
            runs[label] = hists
        else:
            print(f"  (no trained {label} runs found in {results_dir} - skipping)")
    return runs


# ---------------------------------------------------------------------- #
# plotting
# ---------------------------------------------------------------------- #
def _band(ax, arrs):
    """Mean line plus min-max band across seeds."""
    a = np.array([x[: min(len(y) for y in arrs)] for x in arrs])
    return a.mean(axis=0), a.min(axis=0), a.max(axis=0)


def plot_glucose(runs, outdir: Path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhspan(cfg.HYPO, cfg.HYPER, color="green", alpha=0.07,
               label="Target range 70-180", zorder=0)
    ax.axhline(cfg.TARGET, ls="--", c="k", lw=1, alpha=0.6, label="Setpoint 110")
    ax.axhline(cfg.HYPO, ls=":", c="red", lw=1, alpha=0.7)

    for t_meal, amp in cfg.EVAL_MEAL_SPEC:
        ax.axvline(t_meal, color="gray", ls="-.", lw=0.9, alpha=0.6)
        ax.text(t_meal + 8, 57, f"meal {amp:.0f}", fontsize=8, color="gray")

    for label, hists in runs.items():
        t = np.array(hists[0]["t"])[: min(len(h["t"]) for h in hists)]
        mean, lo, hi = _band(None, [h["glucose"] for h in hists])
        ax.plot(t, mean, label=label, color=COLORS[label], lw=1.9)
        if len(hists) > 1:
            ax.fill_between(t, lo, hi, color=COLORS[label], alpha=0.18)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Blood glucose (mg/dL)")
    ax.set_title("Graph 1 — Blood glucose response with meal disturbances")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(50, 300)
    fig.tight_layout()
    fig.savefig(outdir / "graph1_glucose.png", dpi=160)
    plt.close(fig)


def plot_insulin(runs, outdir: Path):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for t_meal, _ in cfg.EVAL_MEAL_SPEC:
        ax.axvline(t_meal, color="gray", ls="-.", lw=0.9, alpha=0.6)
    for label, hists in runs.items():
        t = np.array(hists[0]["t"])[: min(len(h["t"]) for h in hists)]
        mean, lo, hi = _band(None, [h["insulin"] for h in hists])
        ax.plot(t, mean, label=label, color=COLORS[label], lw=1.6)
        if len(hists) > 1:
            ax.fill_between(t, lo, hi, color=COLORS[label], alpha=0.18)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Insulin infusion (mU/min)")
    ax.set_title("Graph 2 — Control effort")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "graph2_insulin.png", dpi=160)
    plt.close(fig)


def plot_reward(runs, outdir: Path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for label, hists in runs.items():
        t = np.array(hists[0]["t"])[: min(len(h["t"]) for h in hists)]
        mean, _, _ = _band(None, [h["reward"] for h in hists])
        a1.plot(t, mean, label=label, color=COLORS[label], lw=1.4)
        a2.plot(t, np.cumsum(mean), label=label, color=COLORS[label], lw=1.8)
    a1.set(xlabel="Time (min)", ylabel="Instantaneous reward",
           title="Graph 3a — Reward signal")
    a2.set(xlabel="Time (min)", ylabel="Cumulative reward",
           title="Graph 3b — Cumulative reward")
    for a in (a1, a2):
        a.legend(fontsize=9)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "graph3_reward.png", dpi=160)
    plt.close(fig)


def plot_convergence(results_dir: Path, seeds, outdir: Path):
    """Graph 4 — training curves from each run's saved episode returns."""
    fig, ax = plt.subplots(figsize=(10, 5))
    found = False
    for algo, label in (("ddpg", "DDPG"), ("td3", "TD3")):
        curves = []
        for s in seeds:
            meta = results_dir / f"{algo}_seed{s}" / "train_meta.json"
            if not meta.exists():
                continue
            rs = json.load(open(meta)).get("episode_returns", [])
            if rs:
                curves.append(np.asarray(rs, dtype=float))
        if not curves:
            continue
        found = True
        n = min(len(c) for c in curves)
        arr = np.stack([c[:n] for c in curves])

        w = max(1, n // 30)
        sm = np.stack([np.convolve(c, np.ones(w) / w, mode="valid") for c in arr])
        x = np.arange(sm.shape[1])
        ax.plot(x, sm.mean(0), color=COLORS[label], lw=2,
                label=f"{label} (n={len(curves)} seeds)")
        ax.fill_between(x, sm.mean(0) - sm.std(0), sm.mean(0) + sm.std(0),
                        color=COLORS[label], alpha=0.2)

    if not found:
        plt.close(fig)
        print("  (no training metadata found - skipping Graph 4)")
        return
    ax.set(xlabel="Training episode", ylabel="Episode return",
           title="Graph 4 — Training convergence (mean ± std across seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "graph4_convergence.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--outdir", type=str, default="results/figures")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pid-only", action="store_true")
    a = ap.parse_args()

    results_dir = Path(a.results_dir)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Running controllers on the fixed evaluation scenario...")
    runs = collect(results_dir, a.seeds, a.pid_only)

    metrics = {}
    for label, hists in runs.items():
        per_seed = [compute_metrics(h, target=cfg.TARGET) for h in hists]
        metrics[label] = {
            k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]
        }

    table = metrics_table(metrics)
    print("\n" + table + "\n")

    (outdir / "metrics_table.txt").write_text(table)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_glucose(runs, outdir)
    plot_insulin(runs, outdir)
    plot_reward(runs, outdir)
    if not a.pid_only:
        plot_convergence(results_dir, a.seeds, outdir)

    print(f"Figures and tables written to {outdir}/")


if __name__ == "__main__":
    main()
