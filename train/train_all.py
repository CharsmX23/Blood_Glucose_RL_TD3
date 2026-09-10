"""Train DDPG and TD3 across multiple seeds.

A single-seed comparison between DDPG and TD3 measures noise, not algorithms.
Three seeds is the practical minimum for a defensible claim, and the plots
report mean +/- std across them.

Usage:
    python -m train.train_all --seeds 0 1 2 --timesteps 300000
"""
import argparse

from train.common import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--algos", type=str, nargs="+", default=["ddpg", "td3"])
    ap.add_argument("--outdir", type=str, default="results")
    a = ap.parse_args()

    for algo in a.algos:
        for seed in a.seeds:
            train(algo, seed=seed, total_timesteps=a.timesteps, outdir=a.outdir)
