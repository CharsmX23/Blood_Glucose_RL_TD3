"""PID baseline for a reverse-acting plant.

The sign trap: more insulin lowers glucose, so the plant is reverse-acting. The
usual textbook form u = Kp*(SP - PV) with positive gains produces a runaway
here — high glucose would command *less* insulin. The error is therefore
defined as

    e = PV - SP = G - target

so a positive error (hyperglycaemia) commands more insulin.

Two further details that the plan omitted and that matter:

1. Anti-windup. Output saturates at [0, u_max]. Without clamping the integral
   while saturated, the three meal excursions inflate it enormously and the
   controller stays pinned at u_max long after glucose has recovered, driving
   the subject hypoglycaemic.

2. Derivative on measurement. Using de/dt where e contains the setpoint gives a
   derivative kick on setpoint changes. Since the setpoint here is constant it
   makes no difference in practice, but derivative-on-measurement is the correct
   form and costs nothing.

3. Feedforward basal. Holding 110 mg/dL requires a nonzero steady infusion. The
   bias term u0 supplies it directly so the integrator does not have to
   discover it from a cold start.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PIDController:
    Kp: float
    Ki: float
    Kd: float
    setpoint: float = 110.0
    u_min: float = 0.0
    u_max: float = 15.0
    u_bias: float = 0.0          # feedforward basal infusion [mU/min]
    derivative_filter: float = 0.1  # low-pass on the D term, 0 = no filtering

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_measurement: float | None = None
        self._d_filtered = 0.0
        self.last_terms = (0.0, 0.0, 0.0)

    def __call__(self, measurement: float, dt: float = 1.0) -> float:
        return self.update(measurement, dt)

    def update(self, measurement: float, dt: float = 1.0) -> float:
        error = measurement - self.setpoint          # reverse-acting

        # --- proportional ---
        p_term = self.Kp * error

        # --- derivative (on measurement, low-pass filtered) ---
        if self._prev_measurement is None:
            d_raw = 0.0
        else:
            d_raw = (measurement - self._prev_measurement) / dt
        self._prev_measurement = measurement

        a = self.derivative_filter
        self._d_filtered = (1 - a) * self._d_filtered + a * d_raw if a > 0 else d_raw
        d_term = self.Kd * self._d_filtered

        # --- integral with conditional clamping ---
        candidate_i = self._integral + error * dt
        i_term = self.Ki * candidate_i

        u_unsat = self.u_bias + p_term + i_term + d_term
        u = float(np.clip(u_unsat, self.u_min, self.u_max))

        # Integrate only if we are not saturated, or if the error would drive
        # the output back inside the valid range.
        saturated_high = u_unsat > self.u_max and error > 0
        saturated_low = u_unsat < self.u_min and error < 0
        if not (saturated_high or saturated_low):
            self._integral = candidate_i
        else:
            i_term = self.Ki * self._integral
            u = float(np.clip(self.u_bias + p_term + i_term + d_term,
                              self.u_min, self.u_max))

        self.last_terms = (p_term, i_term, d_term)
        return u


def default_pid(u_bias: float, u_max: float = 15.0,
                setpoint: float = 110.0) -> PIDController:
    """Hand-tuned baseline gains.

    Found by grid search over the fixed evaluation scenario (see tune_pid).
    These are deliberately a competent baseline rather than a straw man: a
    comparison against a badly tuned PID proves nothing.

    Note that performance saturates around here. Pushing Kd from 3.0 to 8.0
    buys roughly 0.2 mg/dL of RMSE, because the binding constraint is the
    insulin action lag (p2 gives a ~35 min time constant, on top of plasma
    insulin clearance) rather than the gains. A fixed linear law cannot
    anticipate a meal it has not yet seen. That limitation is the whole
    motivation for the learned controllers.
    """
    return PIDController(
        Kp=0.15, Ki=0.0010, Kd=3.0,
        setpoint=setpoint, u_min=0.0, u_max=u_max, u_bias=u_bias,
    )


def tune_pid(env_factory, kp_grid, ki_grid, kd_grid, u_bias, verbose=True):
    """Coarse grid search minimising RMSE, with a hard hypoglycaemia veto.

    Any gain set that takes the subject below 54 mg/dL is rejected outright
    regardless of its RMSE, because a controller that achieves a low average
    error by flirting with severe hypoglycaemia is not a usable controller.
    """
    from eval.metrics import compute_metrics

    best = None
    for kp in kp_grid:
        for ki in ki_grid:
            for kd in kd_grid:
                env = env_factory()
                pid = PIDController(Kp=kp, Ki=ki, Kd=kd,
                                    setpoint=env.target, u_max=env.u_max,
                                    u_bias=u_bias)
                obs, _ = env.reset()
                done = False
                while not done:
                    u = pid.update(env.model.G, dt=env.dt)
                    _, _, term, trunc, _ = env.step(env.insulin_to_action(u))
                    done = term or trunc
                m = compute_metrics(env.history, target=env.target)
                if m["min_glucose"] < 54.0:
                    continue
                score = m["rmse"]
                if best is None or score < best[0]:
                    best = (score, kp, ki, kd, m)
                    if verbose:
                        print(f"  new best: Kp={kp:.4f} Ki={ki:.5f} Kd={kd:.3f} "
                              f"RMSE={score:.2f} min_G={m['min_glucose']:.1f}")
    return best
