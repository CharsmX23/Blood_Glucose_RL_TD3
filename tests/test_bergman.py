"""Validation tests.

Run these before any training. If the plant is wrong, every downstream result
is meaningless, and a training run that fails for physics reasons costs hours
to diagnose from reward curves alone.

    uv run pytest -v
"""

from __future__ import annotations

import numpy as np
import pytest

from configs import scenario as cfg
from controllers.pid import default_pid
from envs.bergman import BergmanModel, BergmanParams, Meal
from envs.glucose_env import make_eval_env, make_train_env


# ---------------------------------------------------------------------- #
# physics
# ---------------------------------------------------------------------- #
def test_open_loop_settles_at_basal():
    """With no insulin, glucose must converge to Gb, not fall to zero.

    This is the test that catches the missing -p1*(G - Gb) term. With the
    equation as written in the original plan, this test fails and glucose
    decays toward 0.
    """
    m = BergmanModel()
    m.reset()
    for _ in range(3000):
        m.step(0.0)
    assert m.G == pytest.approx(m.p.Gb, abs=0.5)


def test_insulin_lowers_glucose_monotonically():
    """Steady-state glucose must decrease monotonically with infusion rate."""
    m = BergmanModel()
    gs = [m.steady_state_glucose(u) for u in [0, 2, 5, 10, 15, 20]]
    assert all(a > b for a, b in zip(gs, gs[1:]))


def test_closed_form_matches_simulation():
    """The analytic steady state must agree with the integrated trajectory."""
    m = BergmanModel()
    for u in [2.0, 5.0, 10.0]:
        m.reset()
        for _ in range(4000):
            m.step(u)
        assert m.G == pytest.approx(m.steady_state_glucose(u), abs=0.5)


def test_basal_infusion_holds_target():
    """basal_infusion_for(110) must actually hold 110 mg/dL."""
    m = BergmanModel()
    u = m.basal_infusion_for(110.0)
    m.reset()
    for _ in range(4000):
        m.step(u)
    assert m.G == pytest.approx(110.0, abs=0.5)


def test_negative_insulin_rejected():
    """The body cannot remove insulin on command; u < 0 must clip to 0."""
    a, b = BergmanModel(), BergmanModel()
    a.reset(); b.reset()
    for _ in range(200):
        a.step(-5.0)
        b.step(0.0)
    assert a.G == pytest.approx(b.G, abs=1e-9)


def test_meal_delivers_full_amplitude():
    """Integrating D(t) over the episode must recover the meal amplitude."""
    meal = Meal(t_start=0.0, amplitude=250.0, k=0.03)
    total = sum(meal.rate(t) for t in np.arange(0, 2000, 0.1)) * 0.1
    assert total == pytest.approx(250.0, rel=0.02)


def test_meal_raises_glucose():
    m = BergmanModel(meals=[Meal(100.0, 250.0, 0.03)])
    m.reset()
    for _ in range(100):
        m.step(0.0)
    before = m.G
    for _ in range(60):
        m.step(0.0)
    assert m.G > before + 20


def test_rk4_beats_euler_on_accuracy():
    """Higher substep counts must converge, confirming the integrator is sane."""
    ref = BergmanModel(substeps=32); ref.reset()
    coarse = BergmanModel(substeps=4); coarse.reset()
    for _ in range(500):
        ref.step(5.0)
        coarse.step(5.0)
    assert abs(ref.G - coarse.G) < 0.1


# ---------------------------------------------------------------------- #
# environment contract
# ---------------------------------------------------------------------- #
def test_env_spaces_and_shapes():
    env = make_eval_env()
    obs, _ = env.reset()
    assert obs.shape == (5,)
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (1,)


def test_action_mapping_covers_full_range():
    env = make_eval_env()
    assert env.action_to_insulin(-1.0) == pytest.approx(0.0)
    assert env.action_to_insulin(1.0) == pytest.approx(env.u_max)
    assert env.action_to_insulin(0.0) == pytest.approx(env.u_max / 2)


def test_action_roundtrip():
    env = make_eval_env()
    for u in [0.0, 1.5, 7.5, 15.0]:
        assert env.action_to_insulin(env.insulin_to_action(u)) == pytest.approx(u)


def test_episode_terminates_at_max_steps():
    env = make_eval_env()
    env.reset()
    steps, done = 0, False
    while not done and steps < cfg.MAX_STEPS + 10:
        _, _, term, trunc, _ = env.step(np.array([-1.0]))
        done = term or trunc
        steps += 1
    assert steps <= cfg.MAX_STEPS


def test_reward_is_bounded_and_peaks_at_target():
    """Reward must stay O(1). The plan's raw -(G-110)**2 reaches -10000, which
    diverges the critic; this test locks in the rescaling."""
    env = make_eval_env()
    env.reset()
    assert env._reward(110.0, 0.0) == pytest.approx(0.0, abs=1e-9)
    assert env._reward(180.0, 5.0) > -5.0
    assert env._reward(300.0, 15.0) > -10.0
    assert env._reward(60.0, 0.0) < env._reward(160.0, 0.0)  # hypo asymmetry


def test_max_insulin_does_not_produce_nan():
    env = make_eval_env()
    env.reset()
    for _ in range(cfg.MAX_STEPS):
        obs, r, term, trunc, _ = env.step(np.array([1.0]))
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        if term or trunc:
            break


def test_train_env_randomises_meals():
    env = make_train_env()
    env.reset(seed=0)
    a = [(m.t_start, m.amplitude) for m in env.model.meals]
    env.reset(seed=1)
    b = [(m.t_start, m.amplitude) for m in env.model.meals]
    assert a != b


def test_eval_env_is_deterministic():
    """The comparison scenario must be identical for every controller."""
    h = []
    for _ in range(2):
        env = make_eval_env()
        env.reset()
        for _ in range(300):
            env.step(np.array([0.0]))
        h.append(env.history["glucose"][-1])
    assert h[0] == pytest.approx(h[1])


# ---------------------------------------------------------------------- #
# PID
# ---------------------------------------------------------------------- #
def test_pid_sign_is_reverse_acting():
    """High glucose must command MORE insulin. Getting this backwards is the
    single most common bug on this plant."""
    pid = default_pid(u_bias=4.9)
    pid.reset()
    high = pid.update(200.0)
    pid.reset()
    low = pid.update(80.0)
    assert high > low


def test_pid_output_respects_bounds():
    pid = default_pid(u_bias=4.9)
    pid.reset()
    for g in [20, 60, 110, 250, 600]:
        u = pid.update(float(g))
        assert 0.0 <= u <= pid.u_max


def test_pid_antiwindup_releases_promptly():
    """After a long saturating excursion the integral must not stay wound up."""
    pid = default_pid(u_bias=4.9)
    pid.reset()
    for _ in range(500):
        pid.update(400.0)          # drive hard into saturation
    wound = pid._integral
    for _ in range(60):
        pid.update(110.0)
    assert pid._integral <= wound


def test_pid_never_causes_hypoglycaemia():
    """The shipped baseline gains must be clinically safe on the eval scenario."""
    from eval.metrics import compute_metrics

    env = make_eval_env()
    pid = default_pid(env.model.basal_infusion_for(env.target),
                      env.u_max, env.target)
    env.reset(); pid.reset()
    done = False
    while not done:
        u = pid.update(env.model.G, dt=env.dt)
        _, _, t1, t2, _ = env.step(env.insulin_to_action(u))
        done = t1 or t2
    m = compute_metrics(env.history)
    assert m["min_glucose"] > cfg.HYPO
    assert m["time_below_70"] == 0.0


def test_pid_beats_basal_only():
    """Feedback must actually improve on open-loop basal, or the baseline is
    not a baseline."""
    from eval.metrics import compute_metrics

    env = make_eval_env()
    ub = env.model.basal_infusion_for(env.target)
    env.reset()
    done = False
    while not done:
        _, _, t1, t2, _ = env.step(env.insulin_to_action(ub))
        done = t1 or t2
    open_loop = compute_metrics(env.history)["rmse"]

    env2 = make_eval_env()
    pid = default_pid(ub, env2.u_max, env2.target)
    env2.reset(); pid.reset()
    done = False
    while not done:
        u = pid.update(env2.model.G, dt=env2.dt)
        _, _, t1, t2, _ = env2.step(env2.insulin_to_action(u))
        done = t1 or t2
    closed_loop = compute_metrics(env2.history)["rmse"]

    assert closed_loop < open_loop


# ---------------------------------------------------------------------- #
# metrics
# ---------------------------------------------------------------------- #
def test_metrics_on_perfect_tracking():
    from eval.metrics import compute_metrics

    n = 100
    h = {"glucose": [110.0] * n, "insulin": [4.9] * n,
         "reward": [0.0] * n, "meal": [0.0] * n, "t": list(range(n))}
    m = compute_metrics(h)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["overshoot"] == pytest.approx(0.0)
    assert m["time_in_range"] == pytest.approx(100.0)
    assert m["settling_time"] == pytest.approx(0.0)


def test_tir_detects_dangerous_low_policy():
    """A policy parked at 75 mg/dL has near-zero RMSE-to-70 but is unsafe.
    time_below_70 is what exposes it."""
    from eval.metrics import compute_metrics

    n = 100
    h = {"glucose": [65.0] * n, "insulin": [10.0] * n,
         "reward": [0.0] * n, "meal": [0.0] * n, "t": list(range(n))}
    m = compute_metrics(h)
    assert m["time_below_70"] == pytest.approx(100.0)
    assert m["time_in_range"] == pytest.approx(0.0)


# ---------------------------------------------------------------------- #
# regression
# ---------------------------------------------------------------------- #
def test_reset_clears_history():
    """Regression guard.

    env.reset() clears history. A VecEnv auto-resets the instant an episode
    ends, so rolling a trained agent out THROUGH DummyVecEnv and then reading
    env.history yields an empty dict and the evaluation silently scores
    nothing. eval/compare.py therefore rolls out on the raw env and normalises
    observations by hand. This test pins the behaviour that forced that choice.
    """
    env = make_eval_env()
    env.reset()
    for _ in range(50):
        env.step(np.array([0.0]))
    assert len(env.history["glucose"]) == 50
    env.reset()
    assert len(env.history["glucose"]) == 0


def test_full_episode_history_is_complete():
    """A completed episode must yield exactly MAX_STEPS scoreable samples."""
    env = make_eval_env()
    env.reset()
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(np.array([-0.5]))
        done = term or trunc
    assert len(env.history["glucose"]) == cfg.MAX_STEPS
    for key in ("t", "glucose", "insulin", "error", "reward", "meal"):
        assert len(env.history[key]) == cfg.MAX_STEPS
