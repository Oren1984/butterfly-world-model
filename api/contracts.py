"""Explicit data contracts between simulator, predictor, evaluation and API.

`Observation` is the only thing the prediction engine ever receives about the
actual world. It deliberately has no field for the butterfly's intended flower,
the current gust, or any random state.
"""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .params import PARAMS


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Flower(Frozen):
    id: str = Field(min_length=1, max_length=4)
    x: float = Field(ge=0, le=PARAMS.width)
    y: float = Field(ge=0, le=PARAMS.height)
    radius: float = Field(default=PARAMS.flower_radius, gt=0, le=10)


class Butterfly(Frozen):
    x: float = Field(ge=0, le=PARAMS.width)
    y: float = Field(ge=0, le=PARAMS.height)
    vx: float = Field(ge=-PARAMS.max_speed - 1e-6, le=PARAMS.max_speed + 1e-6)
    vy: float = Field(ge=-PARAMS.max_speed - 1e-6, le=PARAMS.max_speed + 1e-6)


class Wind(Frozen):
    """Mean wind. direction_deg is where the wind blows toward: 0 = +x, 90 = +y (up)."""

    strength: float = Field(ge=0, le=PARAMS.max_wind)
    direction_deg: float = Field(ge=-360, le=360)


class Observation(Frozen):
    step: int = Field(ge=0)
    butterfly: Butterfly
    wind: Wind
    randomness: float = Field(ge=0, le=1)
    flowers: tuple[Flower, ...] = Field(min_length=1, max_length=12)
    airborne_steps: int = Field(ge=0)
    landed_on: Optional[str] = None
    width: float = PARAMS.width
    height: float = PARAMS.height
    dt: float = PARAMS.dt


class Future(Frozen):
    outcome: str  # flower id or NO_LANDING
    landing_step: Optional[int]
    landing_position: Optional[tuple[float, float]]
    path: tuple[tuple[float, float], ...]  # downsampled by `path_stride`, ends at landing


class Uncertainty(Frozen):
    entropy: float  # normalised Shannon entropy of the outcome distribution, 0..1
    landing_spread: Optional[float]  # RMS scatter of landing points on the top flower
    spread_over_time: tuple[float, ...]  # RMS dispersion of the particle cloud per path sample
    intent_belief: dict[str, float]  # inferred P(intended flower) at the snapshot


class Prediction(Frozen):
    observation_step: int
    n_futures: int
    horizon_steps: int
    path_stride: int
    probabilities: dict[str, float]  # every flower id + NO_LANDING, sums to 1
    top_outcome: str
    confidence: float
    predicted_landing_position: Optional[tuple[float, float]]
    expected_landing_step: Optional[float]
    uncertainty: Uncertainty
    futures: tuple[Future, ...]
