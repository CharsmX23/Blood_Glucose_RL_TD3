# Project plan — hand this to Claude Code

Open this repo in Claude Code and say:

> Read PLAN.md and README.md. Phases 1–6 and 8–9 are already implemented and
> the tests pass. Start at Phase 10.

---

## Status

| Phase | Item | Status |
|---|---|---|
| 1 | Understand the plant | done — see README §4 |
| 2 | Bergman model | done — `envs/bergman.py`, validated |
| 3 | Environment | done — `envs/glucose_env.py` |
| 4 | PID baseline | done — tuned, RMSE 25.31, no hypo |
| 5 | DDPG | code done — **not yet trained** |
| 6 | TD3 | code done — **not yet trained** |
| 7 | Meal disturbance | done — calibrated, `configs/scenario.py` |
| 8 | Metrics | done — `eval/metrics.py` (+ TIR) |
| 9 | Graphs 1–4 | done — `eval/compare.py` |
| 10 | Run training | **you are here** |
| 11 | Robustness study | not started |
| 12 | Report | not started |

---

## Phase 10 — Training (do this in order)

```bash
uv run pytest -v                                        # must be 23/23
uv run python -m train.train_td3 --seed 0 --timesteps 20000   # ~3 min smoke test
uv run python -m eval.compare --seeds 0                 # confirm it evaluates
uv run python -m train.train_all --seeds 0 1 2 --timesteps 300000
uv run python -m eval.compare --seeds 0 1 2
```

Run the smoke test before the full sweep. Three minutes to catch a bug beats
four hours to discover one.

**Sanity checks while training runs.** Episode return should climb out of the
initial range within ~30 episodes. If it is flat, the likely causes are, in
order: `learning_starts` not yet reached (5000 steps, expected), action noise
too low to explore, or the reward scale wrong — run the tests again.

---

## Phase 11 — Robustness (this is what lifts the grade)

Anyone can train two agents and put them on one axis. The differentiator is
showing the learned controllers generalise, or honestly showing where they do
not.

1. **Unseen meal schedule.** Evaluate all three controllers on meals the agents
   never saw — different times and amplitudes. Add `--scenario stress` to
   `eval/compare.py`. PID is scenario-agnostic by construction, so if the RL
   agents degrade here and PID does not, that is a real and reportable finding.

2. **Parameter mismatch.** Perturb `p1`, `p2`, `p3` by ±20% at evaluation only,
   simulating inter-patient variation. Report mean ± std over ~20 sampled
   subjects. This is the closest thing to an external-validity argument you
   can make without real data.

3. **Sensor noise.** Add Gaussian noise (σ ≈ 2–5 mg/dL) to the observed glucose
   at evaluation, since a real CGM is not exact. Watch whether the derivative
   term wrecks PID and whether the agents cope better.

4. **Ablation.** Retrain TD3 once without the integral-error observation. This
   directly tests the design claim in README §5 and makes the observation
   design defensible rather than arbitrary.

---

## Phase 12 — Report

Structure that matches the evidence this repo produces:

1. Introduction — the T1D control problem, why fixed-gain control is limited
2. Plant model — Bergman equations, parameters, **and the corrections in
   README §4**; the missing `-p1(G - Gb)` term is a genuinely good methods
   paragraph
3. Problem formulation — MDP: state, action, reward, and the clinical
   justification for the asymmetric hypoglycaemia penalty
4. Controllers — PID with anti-windup and its tuning procedure; DDPG; TD3 and
   its three mechanisms (clipped double-Q, delayed policy updates, target
   policy smoothing)
5. Experimental setup — shared hyperparameters, three seeds, fixed eval scenario
6. Results — Graphs 1–4, metrics table, robustness study
7. Discussion — **the lag argument**: PID saturates because it cannot
   anticipate an unannounced meal; if the agents win, identify whether it is
   through anticipatory pre-dosing, and show it from the insulin trace
8. Limitations — single virtual subject, no CGM delay, no insulin pump
   constraints, no announced-meal bolus, Bergman is a minimal model
9. Conclusion

**On reporting.** If TD3 and DDPG overlap within seed spread, say the
difference is not significant at n=3. If RL does not beat PID on RMSE, report
that and look at overshoot and meal recovery instead. A clean null result
reported honestly is worth more than a marginal win that does not replicate,
and an examiner will spot the difference.

---

## House rules

- `envs/bergman.py` must never import Gym or SB3. It is the shared physics both
  controller families drive, and that separation is what makes the comparison
  fair.
- DDPG and TD3 share `train/common.py` and `HYPERPARAMS`. Do not tune one
  algorithm alone; it invalidates the study.
- Any change to the plant or reward invalidates previously trained models.
  Delete `results/*_seed*` and retrain.
- Add a test for every bug fixed.
