"""Bergman Minimal Model of glucose-insulin dynamics.

Pure physics. No Gym, no RL, no controller logic. Both the PID baseline and the
RL agents drive *this same object*, which is what makes the comparison fair.

States
------
G : plasma glucose concentration            [mg/dL]
X : remote-compartment insulin action       [1/min]
I : plasma insulin concentration            [mU/L]

Equations (all rates per minute)
--------------------------------
    dG/dt = -p1 * (G - Gb) - X * G + D(t)
    dX/dt = -p2 * X + p3 * (I - Ib)
    dI/dt = -n  * (I - Ib) + u / V1

where u is the exogenous insulin infusion rate [mU/min] and D(t) is the meal
glucose appearance rate [mg/dL/min].

Note the -p1*(G - Gb) term. The spec in PLAN.md omitted it; without it glucose
can only ever fall, since X >= 0 always. It is the endogenous glucose production
that pulls G back toward its basal value, and it is what makes the plant
controllable in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BergmanParams:
    """Per-minute parameters for a type-1 diabetic subject.

    Gb is deliberately hyperglycemic (140 mg/dL). An uncontrolled T1D subject
    drifts high with no insulin on board, so holding the 110 mg/dL target
    requires a genuine, continuously-maintained basal infusion. If Gb were set
    at or below the target, u = 0 would already solve the problem and there
    would be nothing to learn.
    """

    p1: float = 0.028735   # glucose effectiveness                      [1/min]
    p2: float = 0.028344   # insulin action decay                       [1/min]
    p3: float = 5.035e-5   # insulin action gain              [L/(mU*min^2)]
    n: float = 0.09242     # plasma insulin clearance                   [1/min]
    V1: float = 12.0       # insulin distribution volume                    [L]
    Gb: float = 140.0      # basal (no-insulin) glucose                [mg/dL]
    Ib: float = 7.0        # basal plasma insulin                       [mU/L]

    # Safety / physiological bounds
    G_min: float = 20.0
    G_max: float = 600.0


@dataclass
class Meal:
    """A single meal, modelled as first-order glucose appearance.

    Instantaneous `glucose += 40` jumps (as in the original plan) are a
    discontinuity no real gut produces and that value-based RL handles badly.
    Instead the meal releases its total amplitude over time:

        D(t) = A * k * exp(-k * (t - t_start)),   t >= t_start

    which integrates to exactly A mg/dL of cumulative glucose appearance.
    """

    t_start: float          # onset time                              [min]
    amplitude: float        # total glucose delivered               [mg/dL]
    k: float = 0.05         # absorption rate constant              [1/min]

    def rate(self, t: float) -> float:
        if t < self.t_start:
            return 0.0
        return self.amplitude * self.k * np.exp(-self.k * (t - self.t_start))


class BergmanModel:
    """RK4-integrated Bergman minimal model.

    Forward Euler drifts noticeably on this system at dt = 1 min, so the
    integrator is fourth-order Runge-Kutta with configurable substeps.
    """

    def __init__(
        self,
        params: BergmanParams | None = None,
        meals: list[Meal] | None = None,
        substeps: int = 4,
    ) -> None:
        self.p = params or BergmanParams()
        self.meals = meals or []
        self.substeps = int(substeps)
        self.reset()

    # ------------------------------------------------------------------ #
    # state management
    # ------------------------------------------------------------------ #
    def reset(self, G0: float | None = None) -> np.ndarray:
        """Reset to steady state (or a supplied initial glucose)."""
        self.t = 0.0
        G0 = self.p.Gb if G0 is None else float(G0)
        self.state = np.array([G0, 0.0, self.p.Ib], dtype=np.float64)
        return self.state.copy()

    @property
    def G(self) -> float:
        return float(self.state[0])

    @property
    def X(self) -> float:
        return float(self.state[1])

    @property
    def I(self) -> float:  # noqa: E743 - plasma insulin, standard symbol
        return float(self.state[2])

    # ------------------------------------------------------------------ #
    # dynamics
    # ------------------------------------------------------------------ #
    def meal_rate(self, t: float) -> float:
        """Total glucose appearance rate from all meals at time t."""
        return float(sum(m.rate(t) for m in self.meals))

    def derivatives(self, state: np.ndarray, t: float, u: float) -> np.ndarray:
        G, X, I = state
        p = self.p
        dG = -p.p1 * (G - p.Gb) - X * G + self.meal_rate(t)
        dX = -p.p2 * X + p.p3 * (I - p.Ib)
        dI = -p.n * (I - p.Ib) + u / p.V1
        return np.array([dG, dX, dI], dtype=np.float64)

    def step(self, u: float, dt: float = 1.0) -> np.ndarray:
        """Advance the plant by dt minutes under a constant infusion u.

        Parameters
        ----------
        u : insulin infusion rate [mU/min], clipped to be non-negative.
            The body has no mechanism to remove insulin on command, so a
            negative infusion is unphysical.
        """
        u = max(0.0, float(u))
        h = dt / self.substeps

        for _ in range(self.substeps):
            t, y = self.t, self.state
            k1 = self.derivatives(y, t, u)
            k2 = self.derivatives(y + 0.5 * h * k1, t + 0.5 * h, u)
            k3 = self.derivatives(y + 0.5 * h * k2, t + 0.5 * h, u)
            k4 = self.derivatives(y + h * k3, t + h, u)
            self.state = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.t = t + h

        # X and I are physically non-negative; clip glucose to a sane range so
        # a diverging controller produces a terminated episode rather than NaNs.
        self.state[0] = np.clip(self.state[0], self.p.G_min, self.p.G_max)
        self.state[1] = max(0.0, self.state[1])
        self.state[2] = max(0.0, self.state[2])
        return self.state.copy()

    # ------------------------------------------------------------------ #
    # analysis helpers
    # ------------------------------------------------------------------ #
    def steady_state_glucose(self, u: float) -> float:
        """Closed-form equilibrium glucose for a constant infusion u.

        I_ss = Ib + u / (n * V1)
        X_ss = (p3 / p2) * (I_ss - Ib)
        G_ss = p1 * Gb / (p1 + X_ss)
        """
        p = self.p
        I_ss = p.Ib + u / (p.n * p.V1)
        X_ss = (p.p3 / p.p2) * (I_ss - p.Ib)
        return p.p1 * p.Gb / (p.p1 + X_ss)

    def basal_infusion_for(self, G_target: float) -> float:
        """Constant infusion that holds glucose at G_target in steady state."""
        p = self.p
        X_req = p.p1 * (p.Gb - G_target) / G_target
        if X_req <= 0:
            return 0.0
        return X_req * (p.p2 / p.p3) * (p.n * p.V1)
