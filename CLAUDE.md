# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A comparative RL study: a simulated type-1 diabetic subject (Bergman minimal
model) is regulated to 110 mg/dL against three unannounced meals over a 24h
episode, by three controllers — tuned PID, DDPG, and TD3. Not a git repo
(no `.git` present).

## Commands

```bash
uv sync --extra dev                                          # install (incl. pytest, ruff)

uv run pytest -v                                              # all 25 tests, ~2s
uv run pytest -v tests/test_bergman.py::test_name             # single test

uv run python -m eval.compare --pid-only                      # PID + Graphs 1-3, no training needed
uv run python -m train.train_td3 --seed 0 --timesteps 20000   # smoke test, ~3 min — do this before a full sweep
uv run python -m train.train_all --seeds 0 1 2 --timesteps 300000   # full sweep, hours on CPU
uv run python -m eval.compare --seeds 0 1 2                   # final comparison: all 4 graphs + metrics table

uv run tensorboard --logdir results/                          # watch training live
uv run ruff check .
```

Makefile equivalents: `make setup|test|pid|smoke|train|compare|tb|clean`.

Always run modules with `-m` from the project root (`uv run python -m eval.compare`),
not as a script path — `envs`/`configs` imports break otherwise.

## Architecture

**Physics is isolated from RL.** `envs/bergman.py` (`BergmanModel`,
`BergmanParams`, `Meal`) is pure ODE integration (RK4, 4 substeps/min) — it
imports nothing from Gym or SB3. `envs/glucose_env.py` wraps it as a
Gymnasium env. The PID controller and both RL agents all drive the *same*
`BergmanModel` instance type, which is what makes the three-way comparison
fair. Never introduce a Gym/SB3 dependency into `bergman.py`.

**State/observation/action shape.** Plant state is `[G, X, I]` (glucose,
remote insulin action, plasma insulin). The 5-D observation adds an
integral-error term and time-of-day — the integral term exists specifically
so RL isn't handicapped relative to PID's built-in integrator (see README §5
before changing the observation space). Action is `Box[-1,1]`, mapped
internally to `u = U_MAX * (a+1)/2` so infusion is always non-negative
(`action_to_insulin`/`insulin_to_action` in `glucose_env.py`).

**Shared training path.** `train/common.py` (`train()`, `HYPERPARAMS`) is
used by both `train_ddpg.py` and `train_td3.py` with identical
hyperparameters, environments, and seeds — this is intentional so any
DDPG/TD3 difference is attributable to the algorithm, not to tuning. Don't
special-case hyperparameters for one algorithm. `train/train_all.py` sweeps
`{algos} x {seeds}`.

**VecNormalize statistics must travel with the model.** Training env and
eval env share one `VecNormalize.obs_rms` object (`build_envs` in
`train/common.py`). `vecnormalize.pkl` is saved alongside the model and
`eval/compare.py::run_agent` reloads it by hand before rollout, because
`eval/compare.py` runs rollouts on the raw (non-vectorized) env — a VecEnv
auto-resets on episode end and would wipe `env.history` before it can be
scored. Any new evaluation script must reload normalization stats the same
way, or the agent will look fine in training and fail at eval.

**Scenarios.** `configs/scenario.py` defines one fixed, deterministic meal
schedule (`eval_meals()`) used for *every* reported comparison, and a
randomized variant (`random_meals()`) used only during training so agents
learn a policy rather than memorize one trace. `make_eval_env()` /
`make_train_env()` in `glucose_env.py` wire these up — use the right one
intentionally.

**Reward is asymmetric and O(1)-scaled**, not raw squared error: it
penalizes hypoglycemia (`G < 70`) more heavily than an equivalent
hyperglycemic excursion, and adds a severe-hypo penalty (`G < 54`) plus an
insulin-effort term. Changing target/scaling invalidates trained models.

**Metrics beyond the obvious.** `eval/metrics.py` reports Time in Range
(70–180 mg/dL) and time-below-70 alongside RMSE specifically because RMSE
alone is gameable (an agent can "win" by parking glucose low). Any new
controller evaluation should report these, not just RMSE.

## House rules (from PLAN.md)

- `envs/bergman.py` must never import Gym or SB3.
- DDPG and TD3 must stay on the shared `train/common.py` path with identical
  `HYPERPARAMS` — don't tune one algorithm alone.
- Any change to the plant (`bergman.py`) or reward (`glucose_env.py`)
  invalidates previously trained models — delete `results/*_seed*` and
  retrain after such a change.
- Add a test to `tests/test_bergman.py` for every plant/reward bug fixed.
- Current project status (see PLAN.md for the full phase table): plant,
  env, PID, and eval pipeline are done; DDPG/TD3 training and the Phase 11
  robustness study (unseen meal schedule, ±20% parameter mismatch, sensor
  noise, integral-observation ablation) are the open work.
