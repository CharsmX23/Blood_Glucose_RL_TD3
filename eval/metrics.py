"""Performance metrics for glucose controllers.

The five metrics in the original plan (settling time, overshoot, RMSE, total
reward, insulin consumption) are all here. Three clinical ones are added:

  Time in Range (70-180 mg/dL)  the standard endpoint in diabetes technology
                                literature; reporting it is what makes this
                                read as a control study of a medical system
                                rather than a generic tracking exercise.
  Time below 70                 reported separately, because an agent can win
                                on RMSE by parking glucose at 75 and calling it
                                a day. That is a dangerous policy and TIR alone
                                will hide it.
  Min glucose                   the single number a clinician looks at first.
"""

from __future__ import annotations

import numpy as np

from configs import scenario as cfg


def compute_metrics(
    history: dict[str, list[float]],
    target: float = cfg.TARGET,
    settling_band: float = 5.0,
    dt: float = cfg.DT,
) -> dict[str, float]:
    """Compute all metrics from an episode history dict."""
    G = np.asarray(history["glucose"], dtype=float)
    u = np.asarray(history["insulin"], dtype=float)
    r = np.asarray(history.get("reward", []), dtype=float)
    n = len(G)
    if n == 0:
        raise ValueError("empty history")

    err = G - target

    # --- RMSE and MAE ---
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    # --- overshoot: worst excursion above target ---
    overshoot = float(max(0.0, G.max() - target))

    # --- settling time: last moment the trace leaves the +/- band ---
    outside = np.abs(err) > settling_band
    if not outside.any():
        settling_time = 0.0
    else:
        settling_time = float((np.max(np.where(outside)[0]) + 1) * dt)

    # --- post-meal recovery: minutes to re-enter the band after each meal ---
    recovery_times = _meal_recovery_times(history, target, settling_band, dt)

    # --- insulin ---
    total_insulin = float(np.sum(u) * dt)          # mU
    mean_insulin = float(np.mean(u))               # mU/min
    insulin_variation = float(np.mean(np.abs(np.diff(u)))) if n > 1 else 0.0

    # --- clinical ---
    tir = float(np.mean((G >= cfg.HYPO) & (G <= cfg.HYPER)) * 100.0)
    time_below = float(np.mean(G < cfg.HYPO) * 100.0)
    time_above = float(np.mean(G > cfg.HYPER) * 100.0)
    time_severe = float(np.mean(G < cfg.SEVERE_HYPO) * 100.0)

    return {
        "rmse": rmse,
        "mae": mae,
        "overshoot": overshoot,
        "settling_time": settling_time,
        "mean_recovery_time": float(np.mean(recovery_times))
        if recovery_times else 0.0,
        "total_reward": float(np.sum(r)) if r.size else float("nan"),
        "total_insulin": total_insulin,
        "mean_insulin": mean_insulin,
        "insulin_variation": insulin_variation,
        "time_in_range": tir,
        "time_below_70": time_below,
        "time_above_180": time_above,
        "time_below_54": time_severe,
        "min_glucose": float(G.min()),
        "max_glucose": float(G.max()),
        "final_glucose": float(G[-1]),
        "episode_length": float(n * dt),
    }


def _meal_recovery_times(history, target, band, dt) -> list[float]:
    """Minutes from each meal onset until glucose re-enters the target band."""
    meal_rate = np.asarray(history.get("meal", []), dtype=float)
    if meal_rate.size == 0:
        return []

    # Meal onsets are where the appearance rate jumps from ~0 to positive.
    onsets = []
    for i in range(1, len(meal_rate)):
        if meal_rate[i] > 0.05 and meal_rate[i] > meal_rate[i - 1] * 1.5:
            if not onsets or (i - onsets[-1]) > 60:
                onsets.append(i)

    G = np.asarray(history["glucose"], dtype=float)
    out = []
    for start in onsets:
        for j in range(start, len(G)):
            if abs(G[j] - target) <= band:
                out.append((j - start) * dt)
                break
        else:
            out.append(float((len(G) - start) * dt))
    return out


def metrics_table(results: dict[str, dict[str, float]]) -> str:
    """Render a comparison table across controllers."""
    rows = [
        ("RMSE (mg/dL)", "rmse", "{:.2f}", False),
        ("MAE (mg/dL)", "mae", "{:.2f}", False),
        ("Overshoot (mg/dL)", "overshoot", "{:.1f}", False),
        ("Settling time (min)", "settling_time", "{:.0f}", False),
        ("Mean meal recovery (min)", "mean_recovery_time", "{:.0f}", False),
        ("Total reward", "total_reward", "{:.1f}", True),
        ("Total insulin (mU)", "total_insulin", "{:.0f}", False),
        ("Insulin variation", "insulin_variation", "{:.3f}", False),
        ("Time in range 70-180 (%)", "time_in_range", "{:.1f}", True),
        ("Time < 70 (%)", "time_below_70", "{:.1f}", False),
        ("Time > 180 (%)", "time_above_180", "{:.1f}", False),
        ("Min glucose (mg/dL)", "min_glucose", "{:.1f}", True),
        ("Max glucose (mg/dL)", "max_glucose", "{:.1f}", False),
    ]
    names = list(results.keys())
    w0 = max(len(r[0]) for r in rows) + 2
    w = 14

    lines = []
    header = "Metric".ljust(w0) + "".join(n.rjust(w) for n in names)
    lines.append(header)
    lines.append("-" * len(header))

    for label, key, fmt, higher_better in rows:
        vals = [results[n].get(key, float("nan")) for n in names]
        finite = [v for v in vals if np.isfinite(v)]
        best = (max(finite) if higher_better else min(finite)) if finite else None
        cells = []
        for v in vals:
            s = fmt.format(v) if np.isfinite(v) else "n/a"
            if best is not None and np.isfinite(v) and np.isclose(v, best):
                s = "*" + s
            cells.append(s.rjust(w))
        lines.append(label.ljust(w0) + "".join(cells))

    lines.append("")
    lines.append("* marks the best value in each row.")
    return "\n".join(lines)
