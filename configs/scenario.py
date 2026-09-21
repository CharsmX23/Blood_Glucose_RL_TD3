"""Shared scenario configuration.

Every controller — PID, DDPG, TD3 — is evaluated on EVAL_MEALS, which is fixed
and deterministic. Training uses randomised meals so the agents learn a policy
rather than memorising one meal trace.
"""

from __future__ import annotations

import numpy as np

from envs.bergman import Meal

# ---------------------------------------------------------------------- #
# Simulation
# ---------------------------------------------------------------------- #
DT = 1.0                 # integration step                          [min]
EPISODE_MINUTES = 1440   # 24 hours
MAX_STEPS = int(EPISODE_MINUTES / DT)

TARGET = 110.0           # setpoint                                [mg/dL]
U_MAX = 15.0             # max insulin infusion                   [mU/min]

# Clinical thresholds
HYPO = 70.0
SEVERE_HYPO = 54.0
HYPER = 180.0

# ---------------------------------------------------------------------- #
# Meals
# ---------------------------------------------------------------------- #
# Fixed evaluation scenario: breakfast, lunch, dinner. Onset times follow the
# original plan (300 / 700 / 1000 min); amplitudes are the total glucose each
# meal delivers, tuned so an uncontrolled subject peaks around 200-260 mg/dL.
EVAL_MEAL_SPEC = [
    (300.0, 250.0),   # breakfast
    (700.0, 350.0),   # lunch
    (1000.0, 300.0),  # dinner
]

# Absorption rate constant. Calibrated (see tests/test_bergman.py) so that a
# basal-only open loop peaks near 226 mg/dL and spends ~10% of the day above
# 180 -- a realistic uncontrolled postprandial excursion that leaves genuine
# room for a controller to improve, rather than a scenario basal alone solves.
ABSORPTION_K = 0.03


def eval_meals() -> list[Meal]:
    """Deterministic meal set used for all reported comparisons."""
    return [Meal(t, a, ABSORPTION_K) for t, a in EVAL_MEAL_SPEC]


def random_meals(rng: np.random.Generator) -> list[Meal]:
    """Randomised meals for training: +/-25% amplitude, +/-45 min timing."""
    meals = []
    for t, a in EVAL_MEAL_SPEC:
        t_j = float(np.clip(t + rng.uniform(-45, 45), 30, MAX_STEPS - 120))
        a_j = float(a * rng.uniform(0.75, 1.25))
        k_j = float(ABSORPTION_K * rng.uniform(0.8, 1.25))
        meals.append(Meal(t_j, a_j, k_j))
    return meals


def no_meals() -> list[Meal]:
    """Meal-free scenario, for isolating setpoint-tracking performance."""
    return []
