# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A comparative control study: three controllers (tuned PID, DDPG, TD3) regulate
blood glucose to 110 mg/dL on a simulated type-1 diabetic subject (Bergman
minimal model) against three unannounced meals over a 24 h episode. The
deliverable is Graphs 1–4 plus a metrics table, and an honest discussion of
where RL does and does not beat PID. Project status and the intended next steps
live in `PLAN.md`; the scientific rationale and known caveats are in `README.md`
sections 4–7 — read those before changing the plant, reward, or scenario.

## Environment and commands

Uses `uv`. Every command runs through `uv run`; never activate a venv manually.
There is no `.venv/` in a fresh checkout — run `uv sync --extra dev` first.
`make` targets wrap all of the below (`make test`, `make smoke`, `make train`,
`make compare`, `make ui`, `make clean`).

```bash
uv run pytest -v                                   # 25 validation tests, ~2 s — run before anything
uv run pytest tests/test_bergman.py::test_open_loop_settles_at_basal   # single test
uv run ruff check .                                # lint (E, F, I, UP, B; line length 88)

uv run python -m eval.compare --pid-only           # PID baseline + Graphs 1–3, no training needed
uv run python -m train.train_td3 --seed 0 --timesteps 20000    # ~3 min smoke test — always do before a full run
uv run python -m train.train_all --seeds 0 1 2 --timesteps 300000   # full sweep, hours on CPU
uv run python -m eval.compare --seeds 0 1 2        # final Graphs 1–4 + metrics table
uv run streamlit run app.py                        # interactive dashboard at :8501
uv run tensorboard --logdir results/               # live training curves
```

Always run modules with `-m` from the project root (`python -m eval.compare`),
never `python eval/compare.py` — the latter breaks the `envs`/`configs` imports.
On CPU-only machines, force the small torch wheel before the first sync (see
README §1 "Optional: CPU-only torch"); the nets are tiny and CPU is faster here.

## Architecture

The organising principle: **the plant is a plain physics object that every
controller drives directly.** PID and the RL agents act on the same
`BergmanModel`, which is what makes the comparison fair.

- `envs/bergman.py` — the ODE model (RK4, per-minute params). Pure physics.
  **Must never import gym, gymnasium, stable-baselines3, or any controller
  code** (enforced socially, stated in `PLAN.md` house rules). Has closed-form
  helpers `steady_state_glucose(u)` and `basal_infusion_for(G_target)` that the
  tests check the integrator against.
- `envs/glucose_env.py` — Gymnasium wrapper. Defines the 5-D observation
  (tracking error, insulin action, IOB, **integral error**, time-of-day), the
  1-D Box action mapped to `u ∈ [0, U_MAX]`, and the O(1) reward with an
  asymmetric hypoglycaemia penalty. `make_eval_env()` = fixed deterministic
  scenario (all reported numbers); `make_train_env()` = randomised meals +
  initial state.
- `configs/scenario.py` — single source of truth for DT, episode length,
  target, `U_MAX`, clinical thresholds, and the meal schedule
  (`EVAL_MEAL_SPEC`, `eval_meals()`, `random_meals()`).
- `controllers/pid.py` — reverse-acting PID (error = `G - target`, so high
  glucose commands more insulin), conditional-integration anti-windup,
  derivative-on-measurement, feedforward basal bias. `default_pid()` holds the
  grid-searched gains; `tune_pid()` is the grid search with a hard hypo veto.
- `train/common.py` — the **single training path for both DDPG and TD3**.
  `HYPERPARAMS` and `ACTION_NOISE_SIGMA` are shared and must not be tuned
  per-algorithm — that would invalidate the study. Wraps envs in
  `VecNormalize`; the eval env shares `obs_rms` with the training env and has
  reward norm off. `train_td3.py` / `train_ddpg.py` are one-line CLI shims;
  `train_all.py` is the seed×algo loop.
- `eval/runner.py` — the **single rollout path shared by the CLI and the
  Streamlit app**. If `app.py` and `eval/compare.py` ever disagree numerically,
  that is a bug. `load_agent()` restores `vecnormalize.pkl` alongside the
  policy.
- `eval/compare.py` — rolls every controller out on `make_eval_env()`, writes
  Graphs 1–4 and `metrics_table.txt` / `metrics.json` to `results/figures/`.
  Rolls agents out on the **raw env** (not `DummyVecEnv`) and normalises obs by
  hand, because a VecEnv auto-resets on episode end and wipes `env.history`.
- `eval/metrics.py` — RMSE/MAE, overshoot, settling, meal recovery, insulin
  totals + variation, and the clinical trio (Time in Range 70–180, time < 70,
  min glucose) that exists to expose RMSE-gaming "park it at 75" policies.
- `results/<algo>_seed<n>/` — per-run `final_model.zip`, `best/best_model.zip`,
  `vecnormalize.pkl`, `train_meta.json` (contains `episode_returns` used for
  Graph 4), `evals/evaluations.npz`.

## Rules specific to this project

- **`envs/bergman.py` stays free of gym/SB3 imports.** It is the shared plant.
- **DDPG and TD3 share `train/common.py` and `HYPERPARAMS`.** Never tune one
  algorithm's hyperparameters alone.
- **Any change to the plant (`bergman.py`) or the reward/observation
  (`glucose_env.py`) invalidates every trained model.** Delete
  `results/*_seed*` and retrain. `eval/compare.py`'s reported table is only
  valid for models trained against the current plant.
- **Add a test to `tests/test_bergman.py` for every bug fixed** — several
  existing tests are named for the specific bug they lock down
  (`test_open_loop_settles_at_basal`, `test_pid_sign_is_reverse_acting`,
  `test_tir_detects_dangerous_low_policy`, `test_reset_clears_history`).
- **Loading an SB3 agent without its `VecNormalize` stats** is the classic
  "great in training, useless at eval" failure. Any new evaluation code must
  load `vecnormalize.pkl` — copy the pattern in `eval/runner.load_agent`.
- Run the 20k-step smoke test and confirm `eval.compare` evaluates it before
  launching any multi-hour `train_all` sweep.
- The seven deliberate corrections to the original brief (missing `-p1(G-Gb)`
  term, minute vs. second timebase, reward scaling, PID sign, first-order
  meals, added TIR, multi-seed) are documented in `README.md` §4 and at each
  fix site. Preserve them.
