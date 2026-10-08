"""In-memory live session: one actual world plus the committed (frozen) prediction.

Cycle: observing (first second of a flight, no landing possible)
       -> flying   (a prediction is committed and frozen; the world keeps running)
       -> landed | no_landing (prediction compared with what actually happened).
"""
import threading
from typing import Optional

import numpy as np

from .contracts import Observation, Prediction
from .evaluation.metrics import position_error
from .params import NO_LANDING, PARAMS
from .prediction.predictor import predict
from .simulation.world import PREDICTOR_STREAM, ButterflyWorld, make_scenario

DEFAULT_SEED = 7


class Session:
    def __init__(self, seed: int = DEFAULT_SEED):
        self.lock = threading.RLock()
        self.config_version = 0
        self.reset(seed)

    def reset(self, seed: int) -> None:
        with self.lock:
            self.seed = seed
            self.world = ButterflyWorld(make_scenario(seed), seed)
            self.horizon_steps = PARAMS.default_horizon_steps
            self.cycles = 0
            self.correct = 0
            self._prediction_count = 0
            self._begin_cycle()

    def _begin_cycle(self) -> None:
        self.status = "observing"
        self.committed: Optional[Prediction] = None
        self.commit_step = 0
        self.result: Optional[dict] = None
        self.config_version += 1

    def next_rng(self) -> np.random.Generator:
        """A fresh predictor stream; never the world's generator."""
        with self.lock:
            self._prediction_count += 1
            return np.random.default_rng([self.seed, PREDICTOR_STREAM, self._prediction_count])

    def _commit(self) -> None:
        self.committed = predict(self.world.observe(), 20, self.horizon_steps, self.next_rng())
        self.commit_step = self.world.step_count
        self.status = "flying"
        self.config_version += 1

    @property
    def steps_remaining(self) -> int:
        if self.status != "flying":
            return self.horizon_steps
        return max(self.horizon_steps - (self.world.step_count - self.commit_step), 0)

    def step(self, steps: int) -> list[tuple[float, float]]:
        """Advance the actual world; returns the positions visited."""
        path = []
        with self.lock:
            for _ in range(steps):
                if self.status in ("landed", "no_landing"):
                    break
                self.world.step()
                path.append((self.world.x, self.world.y))
                if self.status == "observing":
                    if self.world.airborne_steps >= PARAMS.min_airborne_steps:
                        self._commit()
                elif self.world.landed_on is not None or self.steps_remaining == 0:
                    self._finish()
        return path

    def _finish(self) -> None:
        w, pr = self.world, self.committed
        actual = w.landed_on or NO_LANDING
        actual_pos = (w.x, w.y) if w.landed_on else None
        predicted_pos = pr.predicted_landing_position if pr.top_outcome != NO_LANDING else None
        self.status = "landed" if w.landed_on else "no_landing"
        self.cycles += 1
        self.correct += pr.top_outcome == actual
        self.result = {
            "predicted": pr.top_outcome,
            "actual": actual,
            "confidence": pr.confidence,
            "probability_of_actual": pr.probabilities[actual],
            "correct": pr.top_outcome == actual,
            "position_error": position_error(predicted_pos, actual_pos),
            "predicted_position": predicted_pos,
            "actual_position": actual_pos,
            "flight_steps": w.step_count - self.commit_step,
        }

    def next_cycle(self) -> None:
        with self.lock:
            if self.status not in ("landed", "no_landing"):
                return
            self.world.takeoff()
            self._begin_cycle()

    def configure(self, wind_strength=None, wind_direction_deg=None, randomness=None, flowers=None) -> None:
        """Change public conditions. Any committed prediction is obsolete, so re-predict."""
        with self.lock:
            self.world.set_wind(wind_strength, wind_direction_deg)
            if randomness is not None:
                self.world.set_randomness(randomness)
            for f in flowers or []:
                self.world.move_flower(f.id, f.x, f.y)
            if self.status == "flying":
                self._commit()
            else:
                self.config_version += 1

    def observation(self) -> Observation:
        with self.lock:
            return self.world.observe()

    def state(self, path: Optional[list] = None) -> dict:
        with self.lock:
            pr = self.committed
            return {
                "config_version": self.config_version,
                "seed": self.seed,
                "status": self.status,
                "observation": self.world.observe(),
                "horizon_steps": self.horizon_steps,
                "steps_remaining": self.steps_remaining,
                "committed": None
                if pr is None
                else {
                    "observation_step": pr.observation_step,
                    "top_outcome": pr.top_outcome,
                    "confidence": pr.confidence,
                    "probabilities": pr.probabilities,
                    "predicted_landing_position": pr.predicted_landing_position,
                },
                "result": self.result,
                "session": {"cycles": self.cycles, "correct": self.correct},
                "path": path or [],
            }
