"""Meal-timing robustness sweep: README §6.2.

Re-runs every controller on the evaluation scenario with all meal onsets shifted
by the same offset, and reports RMSE (with change vs. the unshifted benchmark)
and Time in Range at each shift.

Usage
-----
    python -m eval.shift_sweep                     # seed 0, default shifts
    python -m eval.shift_sweep --seeds 0 1 2
    python -m eval.shift_sweep --shifts -120 -60 0 60 120
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from configs import scenario as cfg
from controllers.pid import default_pid
from envs.bergman import Meal
from eval import runner
from eval.metrics import compute_metrics

DEFAULT_SHIFTS = [-90, -45, 0, 45, 90, 120]
DEGRADATION_FLAG = 0.20


def shifted_meals(shift: float) -> list[Meal]:
    # Clipped so a meal is delayed rather than pushed off the end of the day,
    # which would lower RMSE and hide the degradation being measured.
    return [
        Meal(min(max(m.t_start + shift, 0.0), cfg.MAX_STEPS - 60.0),
             m.amplitude, m.k)
        for m in cfg.eval_meals()
    ]


def premeal_insulin(history: dict, meals: list[Meal],
                    window: float = 60.0) -> float:
    """Insulin (mU) delivered in the `window` minutes before each meal onset,
    summed over meals. Meals are unannounced, so none of this can be a
    response to meal glucose."""
    t = np.asarray(history["t"], dtype=float)
    u = np.asarray(history["insulin"], dtype=float)
    return float(sum(
        u[(t >= m.t_start - window) & (t < m.t_start)].sum() * cfg.DT
        for m in meals
    ))


def _agent(run_dir: Path, algo: str):
    model, vec = runner.load_agent(run_dir, algo)
    return lambda meals: runner.run_agent(meals, model, vec)


def controllers(results_dir: Path, seeds: list[int],
                extra_runs: list[Path]) -> dict:
    pid = default_pid(u_bias=0.0)
    ctrls = {"PID": lambda meals: runner.run_pid(meals, pid.Kp, pid.Ki, pid.Kd)}
    runs = runner.discover_runs(results_dir)
    for algo in ("ddpg", "td3"):
        for d in runs[algo]:
            if int(d.name.rsplit("seed", 1)[1]) in seeds:
                ctrls[d.name] = _agent(d, algo)
    for d in extra_runs:
        algo = json.loads((d / "train_meta.json").read_text())["algo"]
        ctrls[d.name] = _agent(d, algo)
    return ctrls


def format_table(results: dict, names: list[str], shifts: list[int]) -> str:
    col = 26
    head = f"{'Shift (min)':<12}" + "".join(f"{n:>{col}}" for n in names)
    lines = [head, "-" * len(head)]
    for s in shifts:
        cells = []
        for n in names:
            r = results[n][str(s)]
            flag = "!" if r["rmse_change"] > DEGRADATION_FLAG else " "
            cells.append(
                f"{r['rmse']:.2f} ({r['rmse_change']:+.1%}){flag} "
                f"TIR {r['time_in_range']:.1f}"
            )
        lines.append(f"{s:<+12d}" + "".join(f"{c:>{col}}" for c in cells))
    lines += [
        "",
        "Each cell: RMSE mg/dL (change vs. shift 0), Time in Range 70-180 %.",
        f"! marks an RMSE degradation above {DEGRADATION_FLAG:.0%}.",
        "",
        "Insulin in the 60 min before each meal onset, summed over meals "
        "(shift 0):",
    ]
    lines += [f"  {n:<24}{results[n]['0']['premeal_insulin']:8.0f} mU"
              for n in names]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--outdir", type=str, default="results/figures")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--shifts", type=int, nargs="+", default=DEFAULT_SHIFTS)
    ap.add_argument("--extra-runs", type=str, nargs="*", default=[],
                    help="additional run folders, e.g. results/ddpg_notime_seed0")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for output files: shift_sweep<tag>.txt")
    a = ap.parse_args()

    shifts = sorted(set(a.shifts) | {0})
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ctrls = controllers(Path(a.results_dir), a.seeds,
                        [Path(p) for p in a.extra_runs])
    print(f"Sweeping {len(shifts)} meal shifts over: {', '.join(ctrls)}")

    results: dict[str, dict] = {}
    for name, run in ctrls.items():
        per_shift = {}
        for s in shifts:
            meals = shifted_meals(s)
            h = run(meals)
            m = compute_metrics(h, target=cfg.TARGET)
            m["premeal_insulin"] = premeal_insulin(h, meals)
            per_shift[s] = m
        base = per_shift[0]["rmse"]
        results[name] = {
            str(s): {
                "rmse": m["rmse"],
                "rmse_change": (m["rmse"] - base) / base,
                "time_in_range": m["time_in_range"],
                "time_below_70": m["time_below_70"],
                "min_glucose": m["min_glucose"],
                "premeal_insulin": m["premeal_insulin"],
            }
            for s, m in per_shift.items()
        }

    table = format_table(results, list(ctrls), shifts)
    print("\n" + table + "\n")
    stem = f"shift_sweep{a.tag}"
    (outdir / f"{stem}.txt").write_text(table + "\n")
    with open(outdir / f"{stem}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Written to {outdir}/{stem}.txt and {stem}.json")


if __name__ == "__main__":
    main()
