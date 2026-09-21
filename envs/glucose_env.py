"""Gymnasium environment wrapping the Bergman minimal model.

Design notes that matter for the comparison being fair and the training being
stable:

Observation (5-D, all normalised to roughly [-3, 3])
    [0] (G - target) / 100          tracking error
    [1] X * 1000                    insulin action, scaled to O(1)
    [2] (I - Ib) / 10               insulin on board
    [3] integral_error / 1000       accumulated error, clipped
    [4] t / MAX_STEPS               time of day

    The integral term is included deliberately. PID gets an integrator by
    construction; withholding one from the RL agents would make the plant
    partially observable for them and only them, and the comparison would be
    measuring memory, not algorithm. Time-of-day lets the agent anticipate
    meals in the fixed-schedule setting.

Action (1-D, Box[-1, 1] as SB3 expects)
    Mapped internally to u = U_MAX * (a + 1) / 2, so u is in [0, U_MAX] and can
    never go negative.

Reward
    The plan's raw -(G - 110)**2 reaches -10000 for a 100 mg/dL error, which
    diverges the critic, and it treats 70 mg/dL as no worse than 150 — which is
    clinically false, since 70 is an emergency and 150 is a Tuesday. So the
    error is scaled to O(1) and hypoglycaemia is penalised asymmetrically.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from configs import scenario as cfg
from envs.bergman import BergmanModel, BergmanParams, Meal


class GlucoseEnv(gym.Env):
    """Blood-glucose regulation as a continuous-control task."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        meals: list[Meal] | None = None,
        randomize_meals: bool = False,
        randomize_initial_state: bool = False,
        params: BergmanParams | None = None,
        target: float = cfg.TARGET,
        u_max: float = cfg.U_MAX,
        dt: float = cfg.DT,
        max_steps: int = cfg.MAX_STEPS,
        insulin_penalty: float = 0.05,
        hypo_penalty: float = 10.0,
        terminate_on_severe_hypo: bool = True,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.params = params or BergmanParams()
        self._fixed_meals = meals
        self.randomize_meals = randomize_meals
        self.randomize_initial_state = randomize_initial_state
        self.target = float(target)
        self.u_max = float(u_max)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.insulin_penalty = float(insulin_penalty)
        self.hypo_penalty = float(hypo_penalty)
        self.terminate_on_severe_hypo = terminate_on_severe_hypo
        self.render_mode = render_mode

        self.model = BergmanModel(params=self.params, meals=self._fixed_meals or [])

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        self._rng = np.random.default_rng()
        self.history: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def action_to_insulin(self, action: np.ndarray | float) -> float:
        """Map a policy action in [-1, 1] to an infusion rate in [0, u_max]."""
        a = float(np.asarray(action).reshape(-1)[0])
        a = float(np.clip(a, -1.0, 1.0))
        return self.u_max * (a + 1.0) / 2.0

    def insulin_to_action(self, u: float) -> np.ndarray:
        """Inverse map, used to drive this env with a PID output."""
        a = 2.0 * float(np.clip(u, 0.0, self.u_max)) / self.u_max - 1.0
        return np.array([a], dtype=np.float32)

    def _observation(self) -> np.ndarray:
        err = self.model.G - self.target
        return np.array(
            [
                err / 100.0,
                self.model.X * 1000.0,
                (self.model.I - self.params.Ib) / 10.0,
                np.clip(self._integral_error / 1000.0, -5.0, 5.0),
                self._step_count / self.max_steps,
            ],
            dtype=np.float32,
        )

    def _reward(self, G: float, u: float) -> float:
        e = (G - self.target) / 100.0
        u_norm = u / self.u_max
        r = -(e**2) - self.insulin_penalty * (u_norm**2)

        if G < cfg.HYPO:
            r -= self.hypo_penalty * ((cfg.HYPO - G) / 100.0) ** 2
        if G < cfg.SEVERE_HYPO:
            r -= 5.0
        return float(r)

    # ------------------------------------------------------------------ #
    # gym API
    # ------------------------------------------------------------------ #
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.randomize_meals:
            self.model.meals = cfg.random_meals(self._rng)
        else:
            self.model.meals = list(self._fixed_meals or [])

        if self.randomize_initial_state:
            G0 = float(self._rng.uniform(90.0, 180.0))
        else:
            G0 = self.target

        self.model.reset(G0=G0)

        # Start with insulin already on board consistent with holding target,
        # so the episode does not open with a large avoidable transient.
        u_basal = self.model.basal_infusion_for(self.target)
        self.model.state[2] = self.params.Ib + u_basal / (self.params.n * self.params.V1)
        self.model.state[1] = (self.params.p3 / self.params.p2) * (
            self.model.state[2] - self.params.Ib
        )

        self._step_count = 0
        self._integral_error = 0.0
        self._total_insulin = 0.0
        self.history = {"t": [], "glucose": [], "insulin": [], "error": [],
                        "reward": [], "meal": []}
        return self._observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        u = self.action_to_insulin(action)
        meal_rate = self.model.meal_rate(self.model.t)

        self.model.step(u, dt=self.dt)
        self._step_count += 1

        G = self.model.G
        err = G - self.target
        self._integral_error += err * self.dt
        self._total_insulin += u * self.dt

        reward = self._reward(G, u)

        terminated = bool(
            self.terminate_on_severe_hypo and G <= cfg.SEVERE_HYPO
        ) or bool(G >= self.params.G_max - 1e-6)
        truncated = self._step_count >= self.max_steps

        self.history["t"].append(self.model.t)
        self.history["glucose"].append(G)
        self.history["insulin"].append(u)
        self.history["error"].append(err)
        self.history["reward"].append(reward)
        self.history["meal"].append(meal_rate)

        info = {
            "glucose": G,
            "insulin": u,
            "error": err,
            "total_insulin": self._total_insulin,
            "meal_rate": meal_rate,
        }
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> str | None:
        line = (
            f"t={self.model.t:7.1f} min | G={self.model.G:7.2f} mg/dL | "
            f"X={self.model.X:8.5f} | I={self.model.I:7.2f} mU/L"
        )
        if self.render_mode == "ansi":
            return line
        print(line)
        return None


def make_eval_env(**kwargs: Any) -> GlucoseEnv:
    """The fixed, deterministic scenario every controller is scored on."""
    return GlucoseEnv(meals=cfg.eval_meals(), randomize_meals=False,
                      randomize_initial_state=False, **kwargs)


def make_train_env(**kwargs: Any) -> GlucoseEnv:
    """Randomised scenario used for training only."""
    return GlucoseEnv(randomize_meals=True, randomize_initial_state=True, **kwargs)
