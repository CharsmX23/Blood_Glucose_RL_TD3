"""Interactive dashboard for the glucose regulation study.

Run with:
    uv run streamlit run app.py

Everything the dashboard shows is computed by eval/runner.py, the same code
path the command-line comparison uses. Nothing here is a mock-up.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from configs import scenario as cfg  # noqa: E402
from envs.bergman import Meal  # noqa: E402
from eval import runner  # noqa: E402
from eval.metrics import compute_metrics  # noqa: E402

st.set_page_config(page_title="Glucose Control — TD3 vs DDPG vs PID",
                   layout="wide")

COLORS = {
    "Open loop": "#999999",
    "PID": "#D55E00",
    "DDPG": "#0072B2",
    "TD3": "#009E73",
}

# Lower-is-better metrics, for the winner highlighting in the table.
LOWER_IS_BETTER = {
    "RMSE (mg/dL)", "MAE (mg/dL)", "Overshoot (mg/dL)", "Settling time (min)",
    "Mean meal recovery (min)", "Total insulin (mU)", "Insulin variation",
    "Time < 70 (%)", "Time > 180 (%)", "Max glucose (mg/dL)",
}

ROWS = [
    ("RMSE (mg/dL)", "rmse", "{:.2f}"),
    ("MAE (mg/dL)", "mae", "{:.2f}"),
    ("Overshoot (mg/dL)", "overshoot", "{:.1f}"),
    ("Settling time (min)", "settling_time", "{:.0f}"),
    ("Mean meal recovery (min)", "mean_recovery_time", "{:.0f}"),
    ("Total reward", "total_reward", "{:.1f}"),
    ("Total insulin (mU)", "total_insulin", "{:.0f}"),
    ("Insulin variation", "insulin_variation", "{:.3f}"),
    ("Time in range 70-180 (%)", "time_in_range", "{:.1f}"),
    ("Time < 70 (%)", "time_below_70", "{:.1f}"),
    ("Time > 180 (%)", "time_above_180", "{:.1f}"),
    ("Min glucose (mg/dL)", "min_glucose", "{:.1f}"),
    ("Max glucose (mg/dL)", "max_glucose", "{:.1f}"),
]


@st.cache_resource(show_spinner=False)
def cached_agent(run_dir: str, algo: str):
    return runner.load_agent(Path(run_dir), algo)


# ------------------------------------------------------------------ #
# Sidebar controls
# ------------------------------------------------------------------ #
st.sidebar.title("Scenario")

target = st.sidebar.slider("Target glucose (mg/dL)", 80, 140,
                           int(cfg.TARGET), 5)
u_max = st.sidebar.slider("Max infusion (mU/min)", 5.0, 40.0,
                          float(cfg.U_MAX), 1.0)
G0 = st.sidebar.slider("Initial glucose (mg/dL)", 80, 250, int(cfg.TARGET), 5)

st.sidebar.markdown("### Meals")
st.sidebar.caption(
    "Amount is total glucose the meal eventually delivers, absorbed "
    "first-order rather than as an instantaneous jump."
)

meals: list[Meal] = []
default_names = ["Breakfast", "Lunch", "Dinner"]
for i, (t_def, a_def) in enumerate(cfg.EVAL_MEAL_SPEC):
    with st.sidebar.expander(default_names[i], expanded=(i == 0)):
        on = st.checkbox("Enabled", value=True, key=f"on{i}")
        t = st.slider("Onset (min)", 0, cfg.MAX_STEPS - 60, int(t_def), 10,
                      key=f"t{i}")
        a = st.slider("Amount (mg/dL)", 0, 600, int(a_def), 10, key=f"a{i}")
        k = st.slider("Absorption k (1/min)", 0.01, 0.10,
                      float(cfg.ABSORPTION_K), 0.005, key=f"k{i}")
    if on and a > 0:
        meals.append(Meal(float(t), float(a), float(k)))

st.sidebar.markdown("### PID gains")
Kp = st.sidebar.slider("Kp", 0.0, 1.0, 0.15, 0.01)
Ki = st.sidebar.slider("Ki", 0.0, 0.01, 0.0010, 0.0001, format="%.4f")
Kd = st.sidebar.slider("Kd", 0.0, 15.0, 3.0, 0.5)

st.sidebar.markdown("### Controllers")
show_open = st.sidebar.checkbox("Open loop (basal only)", value=True)
show_pid = st.sidebar.checkbox("PID", value=True)

runs = runner.discover_runs("results")
agent_choice: dict[str, Path] = {}
for algo in ("ddpg", "td3"):
    label = algo.upper()
    if runs[algo]:
        opts = {d.name: d for d in runs[algo]}
        if st.sidebar.checkbox(label, value=True, key=f"use_{algo}"):
            pick = st.sidebar.selectbox(f"{label} run", list(opts),
                                        key=f"pick_{algo}")
            agent_choice[label] = opts[pick]
    else:
        st.sidebar.caption(f"{label}: no trained run found in results/")

# ------------------------------------------------------------------ #
# Simulation
# ------------------------------------------------------------------ #
st.title("Blood glucose regulation — TD3 vs DDPG vs PID")
st.caption(
    "Bergman minimal model of a type-1 diabetic subject over 24 h. "
    "Every controller drives the identical plant."
)

kw = dict(target=float(target), u_max=float(u_max), G0=float(G0))
histories: dict[str, dict] = {}

with st.spinner("Simulating..."):
    if show_open:
        histories["Open loop"] = runner.run_open_loop(meals, **kw)
    if show_pid:
        histories["PID"] = runner.run_pid(meals, Kp, Ki, Kd, **kw)
    for label, run_dir in agent_choice.items():
        model, vec = cached_agent(str(run_dir), label.lower())
        if model is not None:
            histories[label] = runner.run_agent(meals, model, vec, **kw)

if not histories:
    st.warning("No controller selected. Enable one in the sidebar.")
    st.stop()

metrics = {k: compute_metrics(v, target=float(target)) for k, v in histories.items()}

# ------------------------------------------------------------------ #
# Headline numbers
# ------------------------------------------------------------------ #
cols = st.columns(len(histories))
for col, (name, m) in zip(cols, metrics.items()):
    with col:
        st.markdown(f"**{name}**")
        st.metric("RMSE (mg/dL)", f"{m['rmse']:.1f}")
        st.metric("Time in range", f"{m['time_in_range']:.1f}%")
        st.metric("Peak glucose", f"{m['max_glucose']:.0f}")
        if m["time_below_70"] > 0:
            st.error(f"Hypo: {m['time_below_70']:.1f}% below 70")

# ------------------------------------------------------------------ #
# Graph 1 — glucose
# ------------------------------------------------------------------ #
st.subheader("Graph 1 — Blood glucose")
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.axhspan(cfg.HYPO, cfg.HYPER, color="#2ca02c", alpha=0.07,
           label="Target range 70-180")
ax.axhline(target, ls="--", c="#444", lw=1.2, label=f"Setpoint {target}")
ax.axhline(cfg.HYPO, ls=":", c="crimson", lw=1.0)
for name, h in histories.items():
    ax.plot(h["t"], h["glucose"], lw=1.8, label=name,
            color=COLORS.get(name, None))
for m in meals:
    ax.axvline(m.t_start, color="grey", ls="-.", lw=0.8, alpha=0.7)
    ax.text(m.t_start + 6, 56, f"{m.amplitude:.0f}", fontsize=8, color="grey")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Glucose (mg/dL)")
ax.set_ylim(50, max(300, max(mm["max_glucose"] for mm in metrics.values()) + 20))
ax.legend(ncol=3, fontsize=9)
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)

# ------------------------------------------------------------------ #
# Graph 2 / 3 — insulin and reward
# ------------------------------------------------------------------ #
c1, c2 = st.columns(2)

with c1:
    st.subheader("Graph 2 — Insulin infusion")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for name, h in histories.items():
        ax.plot(h["t"], h["insulin"], lw=1.5, label=name,
                color=COLORS.get(name, None))
    ax.axhline(u_max, ls=":", c="crimson", lw=1.0)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("u (mU/min)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

with c2:
    st.subheader("Graph 3 — Cumulative reward")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for name, h in histories.items():
        ax.plot(h["t"], np.cumsum(h["reward"]), lw=1.5, label=name,
                color=COLORS.get(name, None))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Cumulative reward")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

# ------------------------------------------------------------------ #
# Metrics table
# ------------------------------------------------------------------ #
st.subheader("Evaluation metrics")

names = list(metrics)
data = {}
for label, key, fmt in ROWS:
    data[label] = [metrics[n][key] for n in names]
df = pd.DataFrame(data, index=names).T


def highlight(row: pd.Series):
    vals = pd.to_numeric(row, errors="coerce")
    if vals.isna().all():
        return [""] * len(row)
    best = vals.idxmin() if row.name in LOWER_IS_BETTER else vals.idxmax()
    return ["font-weight:bold; color:#009E73" if c == best else ""
            for c in row.index]


st.dataframe(
    df.style.apply(highlight, axis=1).format("{:.3f}"),
    use_container_width=True,
)

st.caption(
    "Green marks the better value in each row. Time in range and time below 70 "
    "are the clinically meaningful endpoints: a low RMSE achieved by parking "
    "glucose near 75 mg/dL is a dangerous policy, not a good one."
)

# ------------------------------------------------------------------ #
# Training curves
# ------------------------------------------------------------------ #
if agent_choice:
    with st.expander("Graph 4 — Training convergence"):
        fig, ax = plt.subplots(figsize=(11, 3.6))
        plotted = False
        for algo in ("ddpg", "td3"):
            for d in runs[algo]:
                f = d / "evals" / "evaluations.npz"
                if not f.exists():
                    continue
                z = np.load(f)
                ax.plot(z["timesteps"], z["results"].mean(axis=1),
                        lw=1.5, alpha=0.8,
                        color=COLORS[algo.upper()], label=d.name)
                plotted = True
        if plotted:
            ax.set_xlabel("Training timesteps")
            ax.set_ylabel("Mean eval return")
            ax.legend(fontsize=8, ncol=3)
            ax.grid(alpha=0.25)
            st.pyplot(fig)
        else:
            st.info("No evaluation logs found yet.")
        plt.close(fig)
