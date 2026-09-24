# Reinforcement Learning Based Blood Glucose Regulation Using TD3

**A Comparative Study with PID and DDPG Controllers**

A type-1 diabetic subject is simulated with the Bergman minimal model. Three
controllers regulate blood glucose to 110 mg/dL against three unannounced meal
disturbances over a 24-hour episode: a tuned PID baseline, a DDPG agent, and a
TD3 agent.

**Central result.** DDPG wins the headline metric: RMSE 18.9 mg/dL against
PID's 25.3 and TD3's 24.2 on the evaluation scenario, all agents at 300k steps.
The win does not survive perturbation. Shift every meal 90 minutes later and
DDPG's RMSE degrades 35% to 25.5, level with PID, while PID and TD3 each degrade
less than 20%. DDPG has learned to dose on the clock ahead of the evaluation meal
times: it still gives 4.4× its fasting dose at the old meal times after the
meals have moved. An ablation confirms the channel. Retrained without the
time-of-day input, the pre-meal dosing and the 35% degradation both disappear,
and so does most of the win (RMSE 24.5). A fixed-schedule benchmark rewards
dosing on the clock and cannot tell it apart from genuine anticipatory control.
Full evidence and caveats are in
[§6](#6-central-result-the-ddpg-win-is-schedule-memorisation).

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
# Step 1 — validate the physics. 25 tests, ~2 seconds.
uv run pytest -v

# Step 2 — PID baseline and Graphs 1-3. No training required.
uv run python -m eval.compare --pid-only

# Step 3 — smoke-test training with a tiny budget first (~3 min).
uv run python -m train.train_td3 --seed 0 --timesteps 20000

# Step 4 — the real runs. 2 algorithms x 3 seeds. Hours on CPU.
uv run python -m train.train_all --seeds 0 1 2 --timesteps 300000

# Step 5 — final comparison, all four graphs and the metrics table.
uv run python -m eval.compare --seeds 0 1 2

# Step 5b — meal-timing robustness sweep, the central result in §6.
uv run python -m eval.shift_sweep --seeds 0 1 2

# Optional — watch training live
uv run tensorboard --logdir results/

# Step 6 — the interactive dashboard
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

The dashboard also re-runs every controller with all meal onsets shifted 90
minutes later and warns in the winner banner if any controller's RMSE degrades
by more than 20% — the check behind the central result in §6. Markers under
each slider show whether a setting lies inside the range the RL agents were
trained on.

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
│   ├── compare.py          rollouts + Graphs 1-4 + metrics table
│   └── shift_sweep.py      meal-timing robustness sweep (§6.2)
├── tests/test_bergman.py   25 validation tests
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

## 6. Central result: the DDPG win is schedule memorisation

### 6.1 The headline table

Reproduce with `uv run python -m eval.compare --seeds 0`. Figures land in
`results/figures/`. These are real numbers from the models in `results/`, not
placeholders. **Both agents trained for 300,000 steps**, with identical
hyperparameters and seed 0. That makes this one seed each, not a sample.

| Metric | PID | DDPG | TD3 |
|---|---|---|---|
| RMSE (mg/dL) | 25.31 | **18.91** | 24.22 |
| MAE (mg/dL) | **12.49** | 12.70 | 15.49 |
| Overshoot (mg/dL) | 103.4 | **70.7** | 89.4 |
| Mean meal recovery (min) | 65 | 1 † | **0** † |
| Total reward | -109.3 | **-80.2** | -105.5 |
| Total insulin (mU) | **9734** | 11482 | 10461 |
| Insulin variation | **0.041** | 1.173 | 1.636 |
| Time in range 70-180 (%) | 94.2 | **99.7** | 96.2 |
| Time < 70 (%) | **0.0** | **0.0** | **0.0** |
| Min glucose (mg/dL) | **101.9** | 82.7 | 88.1 |
| Max glucose (mg/dL) | 213.4 | **180.7** | 199.4 |

† Artefacts; see §6.3.

On this benchmark **DDPG wins**. It has the best RMSE, Time in Range, overshoot,
peak glucose and total reward, and it never goes below 70 mg/dL. TD3 comes
second on RMSE and TIR, ahead of PID. Taken at face value, the table says RL
beats a tuned PID and DDPG beats TD3. The rest of this section shows why
neither conclusion holds up.

### 6.2 What was found: the win does not survive a change in meal times

The evaluation scenario always serves breakfast, lunch and dinner at 300, 700
and 1000 min. Re-running every controller with all three onsets shifted, and
nothing else changed, gives (RMSE in mg/dL, change vs. the benchmark):

| Meal shift | PID | DDPG | TD3 | DDPG, no time input (§6.4) |
|---|---|---|---|---|
| −90 min | 25.33 (+0.1%) | 23.92 (+26.5%) | 25.89 (+6.9%) | 28.64 (+16.9%) |
| −45 min | 25.32 (+0.0%) | 19.51 (+3.2%) | 24.60 (+1.6%) | 24.24 (−1.0%) |
| **0 (benchmark)** | **25.31** | **18.91** | **24.22** | **24.50** |
| +45 min | 25.30 (−0.1%) | 22.63 (+19.7%) | 23.74 (−2.0%) | 23.84 (−2.7%) |
| **+90 min** | **25.28 (−0.1%)** | **25.54 (+35.1%)** | **28.14 (+16.2%)** | **22.54 (−8.0%)** |
| +120 min | 25.27 (−0.2%) | 32.77 (+73.3%) | 27.24 (+12.5%) | 24.01 (−2.0%) |

At +90 min, DDPG's RMSE degrades 35% and its lead over PID is gone (25.54 vs
25.28). At +120 min it is the worst of the four. PID does not care when meals
arrive: a fixed feedback law has no notion of time. TD3 stays under the 20%
threshold at every shift, but it is not immune either: it degrades 16.2% at
+90 min and is then worse than PID.

Being accurate about what the shift does *not* show:

- At +90 min DDPG still has a better Time in Range than PID (96.8% vs 94.2%).
  It degrades the most, but it is not yet the worst controller there. It
  becomes the worst at +120.
- Nothing goes below 70 mg/dL at any shift. The failure is lost tracking
  accuracy, not a safety event.

Reproduce this table from the repo with:

```bash
uv run python -m eval.shift_sweep --seeds 0 --extra-runs results/ddpg_notime_seed0
```

It rolls every controller out through `eval/runner.py`, the same code path as
the dashboard. It prints the table and writes `results/figures/shift_sweep.txt`
and `shift_sweep.json`. For each controller and shift these contain RMSE, change
vs. shift 0, TIR, time below 70, minimum glucose and pre-meal insulin. A `!`
marks any RMSE degradation above 20%. `--shifts` sets custom offsets, and
`--seeds 0 1 2` adds a column per seed. That is the check to run before stating
the finding as a property of DDPG (§6.8). `--extra-runs` adds run folders
outside the `<algo>_seed<n>` naming, such as the ablation. The dashboard runs
the +90 case automatically and flags it in the winner banner.

### 6.3 The mechanism: dosing on the clock

The observation vector includes time of day, `t / MAX_STEPS`. That is
deliberate: it is what *allows* anticipatory control (§7). Training randomised
meal onsets, but only by ±45 min around the same three evaluation times
(`TRAIN_ONSET_JITTER` in `configs/scenario.py`). So in every training episode,
breakfast arrived within 45 minutes of minute 300. Under those conditions, "dose
heavily at minute 255" pays off in training exactly as well as "respond to meals
early", and the two cannot be told apart.

Meals are unannounced, so in the hour before a meal no meal glucose has
appeared and there is nothing to respond to. The test is therefore insulin in
that hour. It is compared with each controller's own fasting level (min
120–180, scaled to three hours), and re-measured at the *same clock times*
after the meals have been moved 90 minutes later, when no meal is due:

| Insulin (mU), summed over the 3 meals | Fasting level | Hour before meal, on time | Same clock hours, meals moved +90 |
|---|---|---|---|
| PID | 881 | 976 | 1009 |
| DDPG | 400 | **2171** (5.4× fasting) | **1757** (4.4× fasting) |
| TD3 | 699 | 1579 (2.3×) | 1063 (1.5×) |
| DDPG, no time input | 1499 | 1212 (0.8×) | 1256 (0.8×) |

PID gives roughly basal throughout (basal holding target is 4.893 mU/min, or
881 mU over three hours). DDPG runs well *below* basal while fasting, then
delivers 5.4× that in the hour before each meal, near the 15 mU/min ceiling.
Moved meals are the decisive case: DDPG still delivers 4.4× its fasting dose at
the old meal times, with no meal coming. It is dosing on the clock. The insulin
lands at the wrong time, leaving glucose low before the meal and short of
insulin during it, which is the 35% degradation.

TD3 does the same thing, more weakly: 2.3× its fasting dose before meals, still
1.5× at the old times when the meals have moved. That is consistent with its
smaller but real degradation (16% at +90).

This also explains the artefacts in the headline table. "Mean meal recovery" is
measured from meal onset until glucose is inside the target band. Both agents
have already dosed ahead of each meal, so glucose is inside the band at onset
and the metric reads 0–1 min. It is not a recovery. Their high insulin
variation (DDPG 1.173, TD3 1.636, against PID's 0.041) is the same policy seen
another way: hard pulses timed to the schedule rather than a smooth response to
glucose.

### 6.4 The ablation: remove the clock, and the effect goes with it

To test whether time of day is the channel, DDPG was retrained for 300k steps
with seed 0 and identical hyperparameters, but with `obs[4]` removed. The
observation keeps error, insulin action, IOB and integral error, and nothing
else changes.

```bash
uv run python -m train.train_ddpg --seed 0 --timesteps 300000 \
    --no-time-of-day --run-name ddpg_notime_seed0
```

`OBS_TIME_OF_DAY` in `configs/scenario.py` sets the default, and
`train_meta.json` records it per run. Evaluation code reads the input size
from each policy, so 4-input and 5-input models load side by side.
`tests/test_bergman.py::test_time_of_day_ablation_drops_only_the_clock` checks
that the ablation removes exactly that one input.

**Result: the mechanism is confirmed.**

- **Pre-meal dosing disappears.** Without the clock, pre-meal insulin (1212 mU)
  is no higher than the agent's own fasting level (1499). Moving the meals
  does not change it (1256). There is no clock-timed dosing left (last row of
  the §6.3 table).
- **The 35% degradation collapses.** At +90 min the ablated agent changes −8%
  instead of +35%. At +120 it changes −2% instead of +73%.
- **So does most of the win.** Benchmark RMSE rises from 18.91 to 24.50. Of
  DDPG's 6.4 mg/dL lead over PID, 0.8 mg/dL remains (TIR 95.1% vs 94.2%). Most
  of what made DDPG look best was the clock.

What the ablation does **not** fully explain: the ablated agent is still not
timing-invariant the way PID is. Its RMSE varies from −8% to +17% across
shifts, and at −90 min it degrades 16.9% and is worse than PID (28.64 vs
25.33). The variation has no clear pattern with the size of the shift, so it
does not look like a memorised schedule. One candidate is the integral-error
input, which builds up over the episode and could act as a weak clock. That has
not been tested. The dominant effect is explained, and a smaller timing
sensitivity is not yet explained.

### 6.5 Why it matters clinically

Real patients do not eat on a fixed timetable. Breakfast moves with the working
day, meals get skipped, and snacks appear. A controller whose advantage depends
on meals arriving within ±45 minutes of the same three times every day does not
have an advantage a patient would ever get. For a real patient the likely result
is what the +90 row shows: extra insulin delivered before a meal that has not
come, and too little when it does.

The uncomfortable part is the ranking. **The agent that looks best on the
benchmark is the one whose performance degrades most when meal times move.**
Anyone picking a controller from the headline table would pick the least robust
one.

### 6.6 What it implies for the study design

A fixed-schedule benchmark **cannot distinguish anticipatory control from
schedule memorisation**. Both produce insulin before the meal; only a change in
meal times separates them. The ablation shows the time input carried most of
the learned advantage, so the §7 argument that a learned controller can
anticipate meals is not yet supported by any policy trained here.

The fixes:

1. **Randomise meal times much more widely in training.** The current ±45 min
   jitter is too narrow. It keeps every meal in the same slot, so the clock
   still predicts meals. Draw onsets from wide windows, vary the number of
   meals, and sometimes skip one. This is the one fix that could keep the time
   input *and* make it honest. If meals are not predictable from the clock,
   the clock can only encode genuine daily patterns.
2. **Make perturbed-schedule evaluation part of the benchmark.** Report the
   §6.2 sweep alongside the headline table, not only the fixed scenario.
3. **Select checkpoints on perturbed scenarios.** Every RL number here uses
   `best/best_model.zip`, the checkpoint that scored best *on the fixed
   evaluation schedule*. That selection step itself favours schedule-fitting.

Fixes 1 and 3 require retraining. Per the project rules, a change to
`random_meals()` invalidates every model in `results/`.

### 6.7 DDPG vs TD3 at an equal budget

With budgets equal, TD3 scores worse on the headline table (RMSE 24.22 vs 18.91)
but degrades far less when meal times move: at most 16% (+90 min) against
DDPG's 35% (+90) and 73% (+120). The clock-window test in §6.3 shows why. TD3
exploits the schedule, but less heavily. On this one seed, TD3's lower
benchmark score and its robustness are the same fact: it relies less on the
clock.

That is consistent with the theory that clipped double-Q learning curbs the
overestimation that lets DDPG commit hard to a narrow, high-value strategy. It
does not prove it. One seed each cannot separate an algorithmic difference from
a seed difference. The claim needs seeds 1 and 2.

**Earlier TD3 numbers were not real TD3.** SB3's DDPG writes `n_critics=1` into
the `policy_kwargs` dict it is given. `train/common.py` passed the shared
`HYPERPARAMS` dict, so any TD3 trained in the same process after a DDPG, as
`train_all` does, silently lost its twin critic. The archived 60k-step
`td3_seed0` (`results/_archive/td3_seed0_60k`) has one critic. The current
300k run trained in its own process and has two. `model_kwargs()` now hands
each model its own copy, and
`test_td3_keeps_twin_critics_after_ddpg` locks that in. Any TD3 result produced
by `train_all` before this fix should be discarded.

### 6.8 Caveats, stated plainly

- **One seed per algorithm.** n = 1 is one sample. The memorisation finding,
  the ablation and the TD3 comparison each concern a single policy. Re-check on
  seeds 1 and 2 before stating any of them as a property of an algorithm.
- **Training is unstable at this budget.** Final vs. best evaluation return:
  DDPG −115.8 vs −88.7, TD3 −145.3 vs −105.8, ablated DDPG −236.7 vs −110.3.
  The best checkpoint is what gets evaluated, which adds the selection bias
  noted in §6.6.
- **The ablation leaves some timing sensitivity unexplained** (§6.4).

Before you submit, run the full sweep:

```bash
uv run python -m train.train_all --seeds 0 1 2 --timesteps 300000
uv run python -m eval.compare --seeds 0 1 2
uv run python -m eval.shift_sweep --seeds 0 1 2
```

Expect several hours on CPU. Then regenerate this table from your own numbers.

---

## 7. What to write in the report

The strongest framing is not "RL beats PID". On the fixed benchmark DDPG does
beat PID, but §6 shows the win is schedule memorisation. The report should be
built around the mechanism argument and what the meal-shift test did to it:

**PID performance saturates for a structural reason.** Raising Kd from 3.0 to
8.0 buys about 0.2 mg/dL of RMSE. The binding constraint is the insulin action
lag — roughly a 35 min time constant from p2, on top of plasma insulin
clearance — not the gains. A fixed linear law cannot act on a meal before the
meal has moved the glucose it reacts to.

**That is precisely what a learned controller can overcome**, because the
observation vector contains time-of-day and the agent can therefore learn
anticipatory pre-dosing. Whether it *has* learned it is an empirical question.
At 300k steps DDPG does pre-dose, delivering 2.2× PID's insulin in the hour
before each meal. But the meal-shift test shows the pre-dosing is tied to the
training schedule, not to anything about the meal, and the ablation shows the
time-of-day input is how it does it (§6.4). On a fixed-schedule benchmark,
anticipation and memorisation produce the same insulin trace, and only
perturbing the schedule separates them. That is the finding to report: the
mechanism that justifies RL here is real, the benchmark rewards a counterfeit of
it, and the fix is in the training distribution (§6.6).

**On TD3 vs DDPG**, the theory is that clipped double-Q learning suppresses the
value overestimation that makes DDPG policies erratic. At an equal 300k budget,
TD3 scores worse on the benchmark but relies less on the clock and degrades
less when meals move (§6.7). That is consistent with the theory but rests on
one seed each. Also report the twin-critic bug in §6.7: any TD3 trained by
`train_all` before the fix was a single-critic TD3.

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
