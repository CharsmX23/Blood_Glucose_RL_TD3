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

st.set_page_config(
    page_title="Glucose Control — TD3 vs DDPG vs PID",
    page_icon="🩸",
    layout="wide",
)

# ------------------------------------------------------------------ #
# Palette — single source of truth for both the CSS and the charts
# ------------------------------------------------------------------ #
INK = "#1A1D23"          # primary text
MUTED = "#5C636E"        # secondary text
GRID = "#DEE2E6"         # chart gridlines / hairline borders
TEAL = "#0B7285"         # brand / primary
GOOD = "#2F9E44"         # winner / best value
BAD = "#E03131"          # hypo warning

COLORS = {
    "Open loop": "#868E96",
    "PID": "#E8590C",
    "DDPG": "#1C7ED6",
    "TD3": "#0CA678",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.color": GRID,
    "grid.alpha": 0.6,
    "grid.linewidth": 0.8,
})


def style_ax(ax: plt.Axes) -> plt.Axes:
    """Make a matplotlib axis match the page chrome: no top/right spines,
    hairline axes in the grid colour, faint grid, ink labels."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, alpha=0.6)
    ax.set_axisbelow(True)
    return ax


def clean_legend(ax: plt.Axes, **kw) -> None:
    """Legend with no bounding box, so it sits on the chart like a caption."""
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, **kw)


# ------------------------------------------------------------------ #
# Chrome — fonts, hidden Streamlit furniture, gradient banner
# ------------------------------------------------------------------ #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stMarkdown, .stSlider, .stCheckbox {
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    }

    /* hide the default Streamlit menu, header and footer */
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
        visibility: hidden;
        height: 0;
    }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; }

    /* gradient header banner */
    .app-banner {
        background: linear-gradient(120deg, #0B7285 0%, #1098AD 55%, #0CA678 100%);
        color: #FFFFFF;
        padding: 1.5rem 1.8rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 22px rgba(11, 114, 133, 0.22);
    }
    .app-banner h1 {
        font-size: 1.55rem; font-weight: 700; margin: 0; letter-spacing: -0.01em;
    }
    .app-banner p {
        margin: 0.4rem 0 0; font-size: 0.94rem; opacity: 0.92; font-weight: 400;
    }

    /* winner / danger banners */
    .verdict-banner {
        border-radius: 11px; padding: 0.95rem 1.15rem; margin: 0.2rem 0 1.1rem;
        font-size: 0.96rem; line-height: 1.45;
    }
    .verdict-win {
        background: #EBFBEE; border: 1px solid #8CE99A; border-left: 5px solid #2F9E44;
        color: #1B4332;
    }
    .verdict-bad {
        background: #FFF0F0; border: 1px solid #FFC9C9; border-left: 5px solid #E03131;
        color: #7A1212;
    }
    .verdict-banner .trophy { font-size: 1.15rem; margin-right: 0.35rem; }
    .verdict-banner b { font-weight: 700; }
    .verdict-banner .shift-warn {
        margin-top: 0.55rem; padding-top: 0.5rem;
        border-top: 1px dashed rgba(0, 0, 0, 0.18);
        color: #8A4B00; font-weight: 500;
    }

    /* summary comparison table */
    table.summary {
        border-collapse: collapse; width: 100%; margin: 0.2rem 0 0.4rem;
        font-size: 0.93rem;
    }
    table.summary thead th {
        font-weight: 600; color: #1A1D23; text-align: right;
        padding: 0.55rem 0.85rem; border-bottom: 2px solid #CED4DA;
        font-family: 'Inter', sans-serif;
    }
    table.summary thead th:first-child { text-align: left; }
    table.summary td {
        text-align: right; padding: 0.5rem 0.85rem;
        border-bottom: 1px solid #ECEEF1;
        font-family: 'JetBrains Mono', ui-monospace, monospace;
    }
    table.summary td:first-child {
        text-align: left; color: #5C636E; font-weight: 500;
        font-family: 'Inter', sans-serif;
    }
    table.summary td.best { font-weight: 700; color: #2F9E44; }
    table.summary td.hypo-bad { color: #E03131; font-weight: 700; }
    table.summary td.hypo-ok { color: #5C636E; }
    table.summary th.winner-col, table.summary td.winner-col { background: #EBFBEE; }
    table.summary tr.hypo-row td { border-top: 2px solid #CED4DA; border-bottom: none; }

    [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }

    /* parameter-range markers under sidebar sliders */
    .band {
        font-size: 0.76rem; line-height: 1.3; color: #5C636E;
        margin: -0.5rem 0 0.7rem;
    }
    .band::before { content: "●"; margin-right: 0.35rem; }
    .band-green::before { color: #2F9E44; }
    .band-amber::before { color: #E67700; }
    .band-red::before { color: #E03131; }
    .band-red { color: #7A1212; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-banner">
      <h1>Blood Glucose Regulation &nbsp;·&nbsp; TD3 vs DDPG vs PID</h1>
      <p>Bergman minimal model of a type-1 diabetic subject over 24 hours.
      Every controller drives the identical plant; the winner is recomputed
      live from the scenario you set in the sidebar.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Lower-is-better metrics, for the winner highlighting in the full table.
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
# Live winner selection
# ------------------------------------------------------------------ #
def pick_winner(metrics: dict[str, dict]) -> dict:
    """Decide which controller wins the current scenario.

    Computed live from whatever the sliders currently produce — nothing here
    is hardcoded.

    1. Hard safety veto. Any controller that spends *any* time below 70 mg/dL
       is disqualified outright, regardless of how good its other numbers are.
       A controller that "wins" by flirting with hypoglycaemia has not won.
    2. The survivors are ranked on a weighted composite of RMSE (45%),
       time-in-range (30%) and overshoot (25%). Each term is min-max
       normalised across the survivors only, so the score always reflects the
       controllers currently on screen rather than any fixed scale.

    Returns
    -------
    dict with keys: winner (str | None), reason (str), disqualified (list[str]),
    scores (dict[str, float]).
    """
    names = list(metrics)
    disqualified = [
        n for n in names if metrics[n].get("time_below_70", 0.0) > 0.0
    ]
    safe = [n for n in names if n not in disqualified]
    dq_notes = [
        f"{n} disqualified for dipping below 70 mg/dL" for n in disqualified
    ]

    if not safe:
        reason = "; ".join(
            ["no safe option — every selected controller drops below 70 mg/dL"]
            + dq_notes
        )
        return {"winner": None, "reason": reason,
                "disqualified": disqualified, "scores": {}}

    rmse = {n: float(metrics[n]["rmse"]) for n in safe}
    tir = {n: float(metrics[n]["time_in_range"]) for n in safe}
    over = {n: float(metrics[n]["overshoot"]) for n in safe}

    def scale(d: dict[str, float], higher_better: bool) -> dict[str, float]:
        lo, hi = min(d.values()), max(d.values())
        if hi - lo < 1e-9:
            return {k: 1.0 for k in d}
        if higher_better:
            return {k: (v - lo) / (hi - lo) for k, v in d.items()}
        return {k: (hi - v) / (hi - lo) for k, v in d.items()}

    s_rmse = scale(rmse, higher_better=False)
    s_tir = scale(tir, higher_better=True)
    s_over = scale(over, higher_better=False)
    scores = {
        n: 0.45 * s_rmse[n] + 0.30 * s_tir[n] + 0.25 * s_over[n] for n in safe
    }
    winner = max(safe, key=scores.get)
    m = metrics[winner]

    multi = len(safe) > 1
    best_rmse = multi and min(rmse, key=rmse.get) == winner
    best_tir = multi and max(tir, key=tir.get) == winner
    best_over = multi and min(over, key=over.get) == winner

    parts = [f"RMSE {m['rmse']:.1f} mg/dL" + (" (best)" if best_rmse else "")]
    if best_tir:
        parts.append(f"best time-in-range at {m['time_in_range']:.1f}%")
    else:
        parts.append(f"time-in-range {m['time_in_range']:.1f}%")
    if best_over:
        parts.append(f"lowest overshoot at {m['overshoot']:.0f} mg/dL")
    reason = "; ".join(parts + dq_notes)

    return {"winner": winner, "reason": reason,
            "disqualified": disqualified, "scores": scores}


def summary_table_html(metrics: dict[str, dict], winner: str | None) -> str:
    """A real comparison table: one row per metric, one column per controller,
    best value in each row bold + green, hypoglycaemia status on the last row."""
    names = list(metrics)
    specs = [
        ("RMSE", "rmse", lambda v: f"{v:.1f} mg/dL", "min"),
        ("Time in range", "time_in_range", lambda v: f"{v:.1f} %", "max"),
        ("Peak glucose", "max_glucose", lambda v: f"{v:.0f} mg/dL", "min"),
        ("Min glucose", "min_glucose", lambda v: f"{v:.0f} mg/dL", "max"),
        ("Overshoot", "overshoot", lambda v: f"{v:.0f} mg/dL", "min"),
        ("Total insulin", "total_insulin", lambda v: f"{v:.0f} mU", "min"),
    ]

    def th(name: str) -> str:
        cls = ' class="winner-col"' if name == winner else ""
        label = f"🏆 {name}" if name == winner else name
        return f"<th{cls}>{label}</th>"

    head = "<tr><th>Metric</th>" + "".join(th(n) for n in names) + "</tr>"

    body_rows = []
    for label, key, fmt, direction in specs:
        vals = {n: float(metrics[n][key]) for n in names}
        target = min(vals.values()) if direction == "min" else max(vals.values())
        cells = []
        for n in names:
            classes = []
            if abs(vals[n] - target) < 1e-9 and len(names) > 1:
                classes.append("best")
            if n == winner:
                classes.append("winner-col")
            c = f' class="{" ".join(classes)}"' if classes else ""
            cells.append(f"<td{c}>{fmt(vals[n])}</td>")
        body_rows.append(f"<tr><td>{label}</td>" + "".join(cells) + "</tr>")

    hypo_cells = []
    for n in names:
        tb = float(metrics[n].get("time_below_70", 0.0))
        win_cls = " winner-col" if n == winner else ""
        if tb > 0.0:
            hypo_cells.append(
                f'<td class="hypo-bad{win_cls}">⚠ {tb:.1f}% below 70</td>'
            )
        else:
            hypo_cells.append(
                f'<td class="hypo-ok{win_cls}">✓ never below 70</td>'
            )
    body_rows.append(
        '<tr class="hypo-row"><td>Hypoglycaemia</td>'
        + "".join(hypo_cells) + "</tr>"
    )

    return (
        '<table class="summary"><thead>' + head + "</thead><tbody>"
        + "".join(body_rows) + "</tbody></table>"
    )


# ------------------------------------------------------------------ #
# Parameter guidance
# ------------------------------------------------------------------ #
# Commercial hybrid closed-loop systems regulate to roughly 110-120 mg/dL.
CLINICAL_TARGET = (110.0, 120.0)


def classify(v: float, train: tuple[float, float],
             clinical: tuple[float, float] | None = None) -> str:
    """red outside the RL training support, amber inside it but clinically
    unusual, green otherwise."""
    eps = 1e-6
    if not train[0] - eps <= v <= train[1] + eps:
        return "red"
    if clinical and not clinical[0] - eps <= v <= clinical[1] + eps:
        return "amber"
    return "green"


def band(container, status: str, text: str) -> None:
    container.markdown(f'<div class="band band-{status}">{text}</div>',
                       unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Sidebar controls
# ------------------------------------------------------------------ #
st.sidebar.title("Scenario")
st.sidebar.caption(
    "Markers under each slider: **green** realistic and inside the RL "
    "training distribution · **amber** unusual but valid · **red** outside "
    "anything the RL agents saw in training. PID is a fixed formula and "
    "degrades gracefully outside these ranges; the RL agents learned one "
    "scenario, so their behaviour on red settings is unverified "
    "extrapolation, not evidence."
)

target = st.sidebar.slider("Target glucose (mg/dL)", 80, 140,
                           int(cfg.TARGET), 5)
s = classify(target, (cfg.TARGET, cfg.TARGET), CLINICAL_TARGET)
band(st.sidebar, s,
     f"clinical closed-loop target {CLINICAL_TARGET[0]:.0f}–"
     f"{CLINICAL_TARGET[1]:.0f}; RL trained at {cfg.TARGET:.0f} only")

u_max = st.sidebar.slider("Max infusion (mU/min)", 5.0, 40.0,
                          float(cfg.U_MAX), 1.0)
if abs(u_max - cfg.U_MAX) < 1e-6:
    band(st.sidebar, "green", f"matches RL training value ({cfg.U_MAX:g})")
else:
    band(st.sidebar, "red",
         f"RL trained at {cfg.U_MAX:g}: agent actions scale with this, so every "
         f"action now delivers {u_max / cfg.U_MAX:.2f}× its trained dose")

G0 = st.sidebar.slider("Initial glucose (mg/dL)", 80, 250, int(cfg.TARGET), 5)
g_lo, g_hi = cfg.TRAIN_G0_RANGE
band(st.sidebar, classify(G0, cfg.TRAIN_G0_RANGE),
     f"RL training start range {g_lo:.0f}–{g_hi:.0f}")

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
        if not on:
            band(st, "red", "RL agents always trained with this meal present")
        a = st.slider("Amount (mg/dL)", 0, 600, int(a_def), 10, key=f"a{i}")
        a_rng = (a_def * cfg.TRAIN_AMP_FACTOR[0], a_def * cfg.TRAIN_AMP_FACTOR[1])
        band(st, classify(a, a_rng),
             f"RL training range {a_rng[0]:.0f}–{a_rng[1]:.0f}")
        k = st.slider("Absorption k (1/min)", 0.01, 0.10,
                      float(cfg.ABSORPTION_K), 0.005, key=f"k{i}")
        k_rng = (cfg.ABSORPTION_K * cfg.TRAIN_K_FACTOR[0],
                 cfg.ABSORPTION_K * cfg.TRAIN_K_FACTOR[1])
        band(st, classify(k, k_rng),
             f"RL training range {k_rng[0]:.3f}–{k_rng[1]:.4f}")
    if on and a > 0:
        meals.append(Meal(float(t), float(a), float(k)))

st.sidebar.markdown("### PID gains")
Kp = st.sidebar.slider("Kp", 0.0, 1.0, 0.15, 0.01)
Ki = st.sidebar.slider("Ki", 0.0, 0.01, 0.0010, 0.0001, format="%.4f")
Kd = st.sidebar.slider("Kd", 0.0, 15.0, 3.0, 0.5)

st.sidebar.markdown("### Controllers")
show_open = st.sidebar.checkbox("Open loop (basal only)", value=True)
show_pid = st.sidebar.checkbox("PID", value=True)

runs = runner.discover_runs(
    "results", report_skipped="skipped_runs_reported" not in st.session_state
)
st.session_state["skipped_runs_reported"] = True
agent_choice: dict[str, Path] = {}
for algo in ("ddpg", "td3"):
    label = algo.upper()
    if runs[algo]:
        opts = {d.name: d for d in runs[algo]}
        if st.sidebar.checkbox(label, value=True, key=f"use_{algo}"):
            if len(opts) > 1:
                pick = st.sidebar.selectbox(f"{label} run", list(opts),
                                            key=f"pick_{algo}")
            else:
                pick = next(iter(opts))
                st.sidebar.caption(f"using {pick}")
            agent_choice[label] = opts[pick]
    else:
        st.sidebar.caption(f"{label}: no trained run found in results/")

# ------------------------------------------------------------------ #
# Simulation
# ------------------------------------------------------------------ #
kw = dict(target=float(target), u_max=float(u_max), G0=float(G0))

# Robustness probe: 90 min is twice the +/-45 min onset jitter seen in training,
# so a policy that memorised meal times (via the time-of-day input) shows up here.
SHIFT_MIN = 90.0
SHIFT_TOLERANCE = 0.20


@st.cache_data(show_spinner=False, max_entries=512)
def shifted_rmse(meal_spec: tuple[tuple[float, float, float], ...],
                 target: float, u_max: float, G0: float,
                 controller: tuple) -> float:
    """RMSE of one controller with every meal shifted SHIFT_MIN later.

    Keyed on the scenario plus the controller's identity (PID gains, or agent
    run dir), so toggling controllers or retuning PID never re-rolls an agent.
    """
    # Clipped so a late meal is delayed rather than pushed off the end of the
    # day, which would lower RMSE and hide the degradation being measured.
    shifted = [Meal(min(t + SHIFT_MIN, cfg.MAX_STEPS - 60.0), a, k)
               for t, a, k in meal_spec]
    skw = dict(target=target, u_max=u_max, G0=G0)
    kind, *args = controller
    if kind == "pid":
        h = runner.run_pid(shifted, *args, **skw)
    else:
        run_dir, algo = args
        model, vec = cached_agent(run_dir, algo)
        h = runner.run_agent(shifted, model, vec, **skw)
    return float(compute_metrics(h, target=target)["rmse"])


with st.spinner("Simulating..."):
    histories: dict[str, dict] = {}
    controller_ids: dict[str, tuple] = {}
    if show_open:
        histories["Open loop"] = runner.run_open_loop(meals, **kw)
    if show_pid:
        histories["PID"] = runner.run_pid(meals, Kp, Ki, Kd, **kw)
        controller_ids["PID"] = ("pid", float(Kp), float(Ki), float(Kd))
    for label, run_dir in agent_choice.items():
        model, vec = cached_agent(str(run_dir), label.lower())
        if model is not None:
            histories[label] = runner.run_agent(meals, model, vec, **kw)
            controller_ids[label] = ("agent", str(run_dir), label.lower())

if not histories:
    st.warning("No controller selected. Enable one in the sidebar.")
    st.stop()

metrics = {k: compute_metrics(v, target=float(target))
           for k, v in histories.items()}

meal_spec = tuple((m.t_start, m.amplitude, m.k) for m in meals)
with st.spinner("Checking robustness to meal timing..."):
    rmse_degradation = {}
    for k, cid in controller_ids.items():
        base = metrics[k]["rmse"]
        if base > 1e-9:
            s = shifted_rmse(meal_spec, float(target), float(u_max), float(G0),
                             cid)
            rmse_degradation[k] = (s - base) / base

# ------------------------------------------------------------------ #
# Verdict — live winner + comparison table
# ------------------------------------------------------------------ #
verdict = pick_winner(metrics)
winner = verdict["winner"]


def schedule_warnings(winner: str | None) -> str:
    fragile = {k: d for k, d in rmse_degradation.items() if d > SHIFT_TOLERANCE}
    lines = []
    if winner in fragile:
        lines.append(
            f"{winner} wins here, but its RMSE degrades {fragile[winner]:.0%} "
            f"when meal times shift — it has learned the fixed schedule rather "
            f"than responding to meals."
        )
    for k, d in fragile.items():
        if k != winner:
            lines.append(f"{k}'s RMSE degrades {d:.0%} when meal times shift "
                         f"by {SHIFT_MIN:.0f} min.")
    return "".join(f'<div class="shift-warn">⚠ {s}</div>' for s in lines)


if winner:
    st.markdown(
        f'<div class="verdict-banner verdict-win">'
        f'<span class="trophy">🏆</span><b>{winner} wins this scenario.</b> '
        f'{verdict["reason"]}.{schedule_warnings(winner)}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="verdict-banner verdict-bad">'
        f'<span class="trophy">⚠</span><b>No safe winner.</b> '
        f'{verdict["reason"]}.{schedule_warnings(None)}</div>',
        unsafe_allow_html=True,
    )

st.markdown(summary_table_html(metrics, winner), unsafe_allow_html=True)
st.caption(
    "Best value in each row is bold green. Winner is the safety-vetoed "
    "composite of RMSE (45%), time-in-range (30%) and overshoot (25%); any "
    "controller that spends time below 70 mg/dL is disqualified outright. "
    "Total insulin and minimum glucose are shown for context but not scored. "
    f"Each controller is also re-run with every meal {SHIFT_MIN:.0f} min "
    f"later; an RMSE rise above {SHIFT_TOLERANCE:.0%} is flagged in the banner."
)

# ------------------------------------------------------------------ #
# Graph 1 — glucose, winner emphasised
# ------------------------------------------------------------------ #
st.subheader("Graph 1 — Blood glucose")
fig, ax = plt.subplots(figsize=(11, 4.2))
style_ax(ax)
ax.axhspan(cfg.HYPO, cfg.HYPER, color=GOOD, alpha=0.06,
           label="Target range 70-180")
ax.axhline(target, ls="--", c=MUTED, lw=1.2, label=f"Setpoint {target}")
ax.axhline(cfg.HYPO, ls=":", c=BAD, lw=1.0)

for name, h in histories.items():
    is_win = name == winner
    ax.plot(
        h["t"], h["glucose"],
        lw=3.0 if is_win else 1.6,
        alpha=1.0 if is_win else 0.55,
        color=COLORS.get(name),
        label=f"{name}  (winner)" if is_win else name,
        zorder=6 if is_win else 3,
        solid_capstyle="round",
    )

for m in meals:
    ax.axvline(m.t_start, color=GRID, ls="-.", lw=0.9, alpha=0.9)
    ax.text(m.t_start + 6, 56, f"{m.amplitude:.0f}", fontsize=8, color=MUTED)

y_top = max(300.0, max(mm["max_glucose"] for mm in metrics.values()) + 20)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Glucose (mg/dL)")
ax.set_ylim(50, y_top)

if winner:
    hw = histories[winner]
    gw = np.asarray(hw["glucose"], dtype=float)
    tw = np.asarray(hw["t"], dtype=float)
    mw = metrics[winner]
    i_peak = int(np.argmax(gw))
    x_pt, y_pt = tw[i_peak], gw[i_peak]
    # keep the callout inside the axes regardless of where the peak sits
    x_txt = x_pt - 340 if x_pt > tw[-1] * 0.6 else x_pt + 150
    y_txt = min(y_pt + 46, y_top - 14)
    ax.annotate(
        f"{winner}\nRMSE {mw['rmse']:.1f} mg/dL\nTIR {mw['time_in_range']:.1f}%",
        xy=(x_pt, y_pt),
        xytext=(x_txt, y_txt),
        fontsize=8.5, fontweight="600", color=INK, ha="left", va="top",
        arrowprops=dict(arrowstyle="-|>", color=COLORS.get(winner), lw=1.7,
                        shrinkA=0, shrinkB=3),
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec=COLORS.get(winner), lw=1.3, alpha=0.96),
        annotation_clip=False,
    )

clean_legend(ax, ncol=3, loc="upper right")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ------------------------------------------------------------------ #
# Graph 2 / 3 — insulin and reward
# ------------------------------------------------------------------ #
c1, c2 = st.columns(2)

with c1:
    st.subheader("Graph 2 — Insulin infusion")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    style_ax(ax)
    for name, h in histories.items():
        is_win = name == winner
        ax.plot(h["t"], h["insulin"], lw=2.4 if is_win else 1.4,
                alpha=1.0 if is_win else 0.55, label=name,
                color=COLORS.get(name), zorder=6 if is_win else 3)
    ax.axhline(u_max, ls=":", c=BAD, lw=1.0)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("u (mU/min)")
    clean_legend(ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with c2:
    st.subheader("Graph 3 — Cumulative reward")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    style_ax(ax)
    for name, h in histories.items():
        is_win = name == winner
        ax.plot(h["t"], np.cumsum(h["reward"]), lw=2.4 if is_win else 1.4,
                alpha=1.0 if is_win else 0.55, label=name,
                color=COLORS.get(name), zorder=6 if is_win else 3)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Cumulative reward")
    clean_legend(ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ------------------------------------------------------------------ #
# Full metrics table
# ------------------------------------------------------------------ #
st.subheader("Full evaluation metrics")

names = list(metrics)
data = {}
for label, key, _fmt in ROWS:
    data[label] = [metrics[n][key] for n in names]
df = pd.DataFrame(data, index=names).T


def highlight(row: pd.Series):
    vals = pd.to_numeric(row, errors="coerce")
    if vals.isna().all():
        return [""] * len(row)
    best = vals.idxmin() if row.name in LOWER_IS_BETTER else vals.idxmax()
    return [f"font-weight:bold; color:{GOOD}" if c == best else ""
            for c in row.index]


st.dataframe(
    df.style.apply(highlight, axis=1).format("{:.3f}"),
    width="stretch",
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
        style_ax(ax)
        plotted = False
        for algo in ("ddpg", "td3"):
            for d in runs[algo]:
                f = d / "evals" / "evaluations.npz"
                if not f.exists():
                    continue
                # Closed immediately: an open NpzFile locks the file on Windows
                # and makes any training run writing to this folder crash.
                with np.load(f) as z:
                    steps, res = z["timesteps"], z["results"]
                ax.plot(steps, res.mean(axis=1),
                        lw=1.6, alpha=0.85,
                        color=COLORS[algo.upper()], label=d.name)
                plotted = True
        if plotted:
            ax.set_xlabel("Training timesteps")
            ax.set_ylabel("Mean eval return")
            clean_legend(ax, ncol=3)
            st.pyplot(fig)
        else:
            st.info("No evaluation logs found yet.")
        plt.close(fig)
