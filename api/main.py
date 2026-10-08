"""FastAPI surface for the live world, the prediction engine and the evaluation."""
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from .contracts import Observation, Prediction
from .evaluation.experiment import EVAL_BASE_SEED, run_experiment
from .params import PARAMS
from .prediction.predictor import predict
from .session import Session

MAX_SEED = 2**31 - 1

app = FastAPI(title="Butterfly World Model", version="1.0.0")
# The web container proxies /api, so CORS only matters for local Vite development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
session = Session()


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResetRequest(Request):
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)
    new_scenario: bool = False


class StepRequest(Request):
    steps: int = Field(default=1, ge=1, le=50)


class FlowerMove(Request):
    id: str = Field(min_length=1, max_length=4)
    x: float = Field(ge=PARAMS.flower_radius, le=PARAMS.width - PARAMS.flower_radius)
    y: float = Field(ge=PARAMS.flower_radius, le=PARAMS.height - PARAMS.flower_radius)


class ConfigRequest(Request):
    wind_strength: Optional[float] = Field(default=None, ge=0, le=PARAMS.max_wind)
    wind_direction_deg: Optional[float] = Field(default=None, ge=0, le=360)
    randomness: Optional[float] = Field(default=None, ge=0, le=1)
    flowers: Optional[list[FlowerMove]] = Field(default=None, max_length=12)


class PredictRequest(Request):
    observation: Optional[Observation] = None  # omitted = the live world's current observation
    n_futures: int = Field(default=20, ge=1, le=200)
    horizon_steps: Optional[int] = Field(default=None, ge=1, le=600)
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)


class PredictResponse(Request):
    config_version: Optional[int]
    prediction: Prediction


class EvaluateRequest(Request):
    n_scenarios: int = Field(default=500, ge=1, le=2000)
    base_seed: int = Field(default=EVAL_BASE_SEED, ge=0, le=MAX_SEED - 2000)
    n_futures: int = Field(default=20, ge=1, le=200)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/world/reset")
def world_reset(req: ResetRequest = ResetRequest()) -> dict:
    seed = req.seed if req.seed is not None else session.seed + (1 if req.new_scenario else 0)
    session.reset(seed % (MAX_SEED + 1))
    return session.state()


@app.get("/api/world/state")
def world_state() -> dict:
    return session.state()


@app.post("/api/world/step")
def world_step(req: StepRequest = StepRequest()) -> dict:
    return session.state(session.step(req.steps))


@app.post("/api/world/next")
def world_next() -> dict:
    session.next_cycle()
    return session.state()


@app.post("/api/world/config")
def world_config(req: ConfigRequest) -> dict:
    known = {f.id for f in session.observation().flowers}
    unknown = [f.id for f in req.flowers or [] if f.id not in known]
    if unknown:
        raise HTTPException(422, f"unknown flower id(s): {unknown}")
    session.configure(req.wind_strength, req.wind_direction_deg, req.randomness, req.flowers)
    return session.state()


@app.get("/api/world/prediction", response_model=PredictResponse)
def world_prediction() -> PredictResponse:
    """The committed (frozen) prediction that the current flight is scored against."""
    with session.lock:
        if session.committed is None:
            raise HTTPException(404, "no prediction has been committed for this flight yet")
        return PredictResponse(config_version=session.config_version, prediction=session.committed)


@app.post("/api/predict", response_model=PredictResponse)
def post_predict(req: PredictRequest = PredictRequest()) -> PredictResponse:
    with session.lock:
        live = req.observation is None
        obs = session.observation() if live else req.observation
        version = session.config_version if live else None
        horizon = req.horizon_steps or (max(session.steps_remaining, 1) if live else PARAMS.default_horizon_steps)
        rng = np.random.default_rng(req.seed) if req.seed is not None else session.next_rng()
    if obs.landed_on is not None:
        raise HTTPException(409, "the butterfly has landed; start the next flight first")
    return PredictResponse(config_version=version, prediction=predict(obs, req.n_futures, horizon, rng))


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest = EvaluateRequest()) -> dict:
    metrics, _ = run_experiment(req.n_scenarios, req.base_seed, req.n_futures)
    return metrics
