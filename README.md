# Reinforcement Learning Based Blood Glucose Regulation Using TD3

**A Comparative Study with PID and DDPG Controllers**

A type-1 diabetic subject is simulated with the Bergman minimal model. Three
controllers regulate blood glucose to 110 mg/dL against three unannounced meal
disturbances over a 24-hour episode: a tuned PID baseline, a DDPG agent, and a
TD3 agent.

---

## 1. Quickstart with `uv`

### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your shell afterwards, then check it works:

```bash
uv --version
```

### Create the environment

**There is no `venv/` folder in this zip, and that is deliberate.** A virtual
environment contains absolute paths and platform-specific compiled binaries, so
a copied one breaks on any machine but the one that built it. `uv` rebuilds it
exactly instead, from the lockfile:

```bash
cd glucose-rl
uv sync
```

That reads `pyproject.toml`, installs the exact pinned versions from `uv.lock`
into a fresh `.venv/`, and downloads a matching Python if you do not have one.
It takes under a minute; torch is the only large download.

To include the dev tools (pytest, ruff):

```bash
uv sync --extra dev
```

### Optional: CPU-only torch (recommended)

By default `uv sync` may pull the CUDA build of torch, which is roughly 2.5 GB.
The networks here are two 256-unit layers — small enough that CPU is actually
*faster* than GPU, because kernel-launch overhead dominates at this size. To
force the ~200 MB CPU wheel, append this to `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

Then re-resolve:

```bash
uv lock && uv sync --extra dev
```

Do this before the first `uv sync` if you are on a metered connection. If you
do have a CUDA GPU and want to use it, skip this section entirely.

### Run things

Prefix any command with `uv run` and it executes inside the environment. You
never need to activate anything:

```bash
uv run pytest -v                      # validate the plant (do this first)
uv run python -m eval.compare --pid-only   # PID baseline, no training needed
```

If you prefer an activated shell:

```bash
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

---

## 2. Full pipeline

```bash
# Step 1 — validate the physics. 23 tests, ~2 seconds.
uv run pytest -v

# Step 2 — PID baseline and Graphs 1-3. No training required.
uv run python -m eval.compare --pid-only

# Step 3 — smoke-test training with a tiny budget first (~3 min).
uv run python -m train.train_td3 --seed 0 --timesteps 20000

# Step 4 — the real runs. 2 algorithms x 3 seeds. Hours on CPU.
uv run python -m train.train_all --seeds 0 1 2 --timesteps 300000

# Step 5 — final comparison, all four graphs and the metrics table.
uv run python -m eval.compare --seeds 0 1 2

# Optional — watch training live
uv run tensorboard --logdir results/
```

# Step 6 - the interactive dashboard
uv run streamlit run app.py
```

Opens at http://localhost:8501. Sliders for the three meals, PID gains, target,
max infusion and initial glucose; tick boxes to overlay open-loop, PID, DDPG and
TD3; live glucose / insulin / reward plots and the metrics table, recomputed on
every change. Trained agents are picked up automatically from `results/`, so the
DDPG and TD3 boxes appear once training has produced them.

The dashboard calls `eval/runner.py`, which is the same code path
`eval/compare.py` uses. An experiment you run in the UI is the same computation
as the one on the command line - the numbers cannot drift apart.

```bash

**Do Step 3 before Step 4.** Validating the pipeline end to end at 20k steps
costs three minutes; discovering a bug four hours into a 300k-step run does not.

---

## 3. Project layout

```
glucose-rl/
├── pyproject.toml          dependencies
├── uv.lock                 exact pinned versions — reproducible builds
├── configs/scenario.py     meal schedule, target, limits, episode length
├── envs/
│   ├── bergman.py          the ODE model. No Gym, no RL. Pure physics.
│   └── glucose_env.py      Gymnasium wrapper: obs, action, reward
├── app.py                  Streamlit dashboard
├── controllers/pid.py      PID with anti-windup + grid-search tuner
├── train/
│   ├── common.py           shared training path for BOTH algorithms
│   ├── train_ddpg.py
│   ├── train_td3.py
│   └── train_all.py        multi-seed sweep
├── eval/
│   ├── runner.py           single rollout path shared by UI and CLI
│   ├── metrics.py          RMSE, overshoot, settling, TIR, insulin
│   └── compare.py          rollouts + Graphs 1-4 + metrics table
├── tests/test_bergman.py   23 validation tests
└── results/                models, logs, figures
```

`bergman.py` deliberately imports nothing from Gym. PID drives that same physics
object the RL agents drive, which is what makes the comparison fair.

---

## 4. Seven corrections to the original plan

The project brief as written contains five errors that would have broken the
study, plus two omissions. All are fixed here and each is documented at the
point of the fix in the code.

**1. The glucose equation could not rise.** The brief gives
`dG/dt = -XG + input`. Since X is non-negative, glucose could only ever fall.
The endogenous production term is missing. Corrected to
`dG/dt = -p1(G - Gb) - XG + D(t)`. `test_open_loop_settles_at_basal` fails
against the original equation.

**2. The timebase was wrong.** Bergman parameters are per *minute*, but the
brief specifies a 1000-*second* run. At 1000 s the subject has barely responded
to anything. Episodes are 1440 minutes (24 h), which conveniently makes the
brief's meal times of 300/700/1000 land as breakfast, lunch and dinner.

**3. The reward would have diverged the critic.** Raw `-(G - 110)**2` reaches
-10000 for a 100 mg/dL error — far outside the range a value network can fit.
It also treats 70 mg/dL as no worse than 150, which is clinically backwards: 70
is an emergency and 150 is a Tuesday. The error is scaled to O(1) and
hypoglycaemia is penalised asymmetrically.

**4. The PID sign was inverted.** The plant is reverse-acting — more insulin
lowers glucose. `u = Kp(SP - PV)` with positive gains would command *less*
insulin as glucose rose, a runaway. Error is defined as `G - target`.
Anti-windup was also absent; without it the integral inflates across the three
meal excursions and pins the output at maximum long after recovery, driving the
subject hypoglycaemic.

**5. Instantaneous meal jumps.** `if t == 300: glucose += 40` is a
discontinuity no gut produces and that value-based RL handles badly. Meals are
first-order absorption, `D(t) = A·k·exp(-k(t - t₀))`, integrating to exactly A.

**6. Time in Range was missing.** RMSE alone is gameable: an agent can score
well by parking glucose at 75 mg/dL, which is a dangerous policy. TIR (70–180)
is the standard clinical endpoint, and time-below-70 is reported separately
because TIR alone still hides that failure mode.
`test_tir_detects_dangerous_low_policy` locks this in.

**7. Single-seed comparison.** One seed each for DDPG and TD3 measures noise,
not algorithms. Three seeds minimum; plots show mean ± spread.

---

## 5. Design decisions worth defending in a viva

**Basal glucose is 140 mg/dL, above the 110 target.** An uncontrolled T1D
subject is hyperglycaemic, so holding target requires a genuine, continuously
maintained infusion (4.893 mU/min, derived in closed form by
`basal_infusion_for`). Had Gb been set at or below target, `u = 0` would
already solve the problem and there would be nothing to learn.

**The RL agents observe an integral-error term.** PID has an integrator by
construction. Withholding one from the RL agents would make the plant partially
observable for them and only them, so the study would be measuring memory
rather than algorithm.

**Meal size was calibrated, not guessed.** With the brief's original
amplitudes, open-loop basal alone achieved 100% Time in Range — nothing to
compare. Amplitudes were swept until a basal-only subject peaks near 226 mg/dL
and spends ~10% of the day above 180: a realistic excursion with real room to
improve.

**DDPG and TD3 share one code path and identical hyperparameters.** Any
observed difference is attributable to the algorithm, not to tuning effort.

**PID is tuned, not a straw man.** Gains come from grid search with a hard
hypoglycaemia veto. A comparison against a badly tuned PID proves nothing.

---

## 6. Results actually produced by this repo

Reproduce with `uv run python -m eval.compare --seeds 0`. Figures land in
`results/figures/`. These are real numbers from a real 60k-step run, not
placeholders.

| Metric | PID | DDPG | TD3 |
|---|---|---|---|
| RMSE (mg/dL) | **25.31** | 29.18 | 26.94 |
| MAE (mg/dL) | **12.49** | 21.20 | 15.82 |
| Overshoot (mg/dL) | 103.4 | **93.6** | 96.3 |
| Total reward | **-109.3** | -156.8 | -120.7 |
| Total insulin (mU) | 9734 | 13631 | **8989** |
| Insulin variation | **0.041** | 0.085 | 0.151 |
| Time in range 70-180 (%) | 94.2 | **95.5** | 94.5 |
| Time < 70 (%) | **0.0** | **0.0** | **0.0** |
| Min glucose (mg/dL) | **101.9** | 78.1 | 92.4 |
| Max glucose (mg/dL) | 213.4 | **203.6** | 206.3 |

Final evaluation return during training: DDPG -253.3, TD3 **-121.8**. TD3 more
than halves DDPG's return under an identical budget, identical hyperparameters
and an identical seed, which is the headline algorithmic result.

### Read Graph 1 before writing your discussion

The metrics table alone will mislead you. Look at `graph1_glucose.png`: both
learned agents **park below the setpoint during fasting** — DDPG at roughly
80 mg/dL, TD3 at roughly 95 — and only PID holds 110 cleanly.

That single behaviour explains most of the table:

- The agents' better *overshoot* and *peak glucose* are not superior meal
  rejection. They enter each meal from a lower baseline, so they have further
  to climb before crossing 180.
- DDPG's headline "2 min mean meal recovery" is an **artefact**, not a result.
  Recovery is defined as re-entering a band around the target; a controller
  already sitting below that band re-enters it immediately. Do not report this
  number as a win without the explanation.
- DDPG reaches a minimum of 78.1 mg/dL. Nothing is flagged, because time-below-70
  is 0.0%, but it is 8 mg/dL from clinical hypoglycaemia on a fasting subject.
  That is the dangerous-policy failure mode the README warned about, showing up
  in practice.

So the honest conclusion at this budget is: **PID still wins on tracking
accuracy (RMSE, MAE) and on safety margin, while TD3 clearly beats DDPG.**
Report that. A study that reports its baseline winning is far more credible
than one that quietly buries it.

### The undertraining caveat, stated plainly

These runs are **60,000 steps, one seed** — roughly 42 episodes. That was a
compute budget chosen to produce a working end-to-end result, not a converged
one. `graph4_convergence.png` shows both curves still oscillating between -100
and -350 with no plateau; neither agent has converged. Two consequences:

1. The agents are very likely underperforming their true capability. The
   fasting-drift behaviour is characteristic of a policy that has not yet
   learned to separate basal maintenance from meal response.
2. **n=1 seed cannot support a claim about DDPG vs TD3.** The -121.8 vs -253.3
   gap is large and matches theory, but one seed is one sample.

Before you submit, run the full sweep:

```bash
uv run python -m train.train_all --seeds 0 1 2 --timesteps 300000
uv run python -m eval.compare --seeds 0 1 2
```

Expect several hours on CPU. Then regenerate this table from your own numbers.

---

## 7. What to write in the report

The strongest framing is not "RL beats PID", because at this budget it does not.
It is the mechanism argument:

**PID performance saturates for a structural reason.** Raising Kd from 3.0 to
8.0 buys about 0.2 mg/dL of RMSE. The binding constraint is the insulin action
lag — roughly a 35 min time constant from p2, on top of plasma insulin
clearance — not the gains. A fixed linear law cannot act on a meal before the
meal has moved the glucose it reacts to.

**That is precisely what a learned controller can overcome**, because the
observation vector contains time-of-day and the agent can therefore learn
anticipatory pre-dosing. Whether it *has* learned it is an empirical question,
and your Graph 1 answers it: at 60k steps it has not, it has learned baseline
depression instead, which is a cruder strategy that partially mimics the same
effect. Report that gap between the mechanism and the observed policy. It is a
more interesting finding than a table of wins.

**On TD3 vs DDPG**, the theory is that clipped double-Q learning suppresses the
value overestimation that makes DDPG policies erratic. The evidence here is
consistent with it — TD3 gets a better return, a higher minimum glucose and
uses 34% less insulin. Note the one metric that goes the other way: TD3's
insulin variation (0.151) is the *worst* of the three, so the delayed-policy-update
smoothness argument is not supported by this run. Say so.

---

## 8. Troubleshooting

**`ModuleNotFoundError: No module named 'envs'`** — run from the project root
using module syntax (`uv run python -m eval.compare`), not
`python eval/compare.py`.

**Agent performs well in training, terribly at evaluation** — the VecNormalize
statistics were not loaded. `eval/compare.py` restores them from
`vecnormalize.pkl`; if you write your own evaluation script, you must too. This
is the most common failure in SB3 projects.

**`import gym` errors in a tutorial you found** — SB3 2.x uses `gymnasium`.
`step()` returns five values (obs, reward, terminated, truncated, info), not
four.

**Training is very slow** — expected on CPU. Lower `--timesteps` to 100000 for
a first pass; the qualitative comparison usually holds.

**Reward is stuck near a large negative number** — check that
`vecnormalize.pkl` is being written, and run `uv run pytest` to confirm the
plant is behaving.
"# Blood_Glucose_RL_TD3" 
