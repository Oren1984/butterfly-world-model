"""API routes for the V1.1 experiment modes: Goal-Directed Planning and Visual Planning.

The original landing-prediction routes in `main.py` are unchanged. These
routes either run an episode live (`POST .../episode`) or return the recorded
comparison written by the evaluation commands (`GET .../results`). Trained
models are loaded from `artifacts/`; if one is missing or incompatible, the
response says so and names the command that creates it.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .planning import learned
from .planning.episode import run_episode
from .planning.garden import LAYOUTS, LEARNED_LAYOUTS
from .planning.planners import BUDGET, PLANNER_LABELS, PLANNERS
from .planning.seeds import PLANNING_EVAL_SEEDS, VISUAL_EVAL_SEEDS, VISUAL_MODEL_SEEDS
from .visual import jepa
from .visual.episode import run_visual_episode
from .visual.planners import VISUAL_BUDGET

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
MAX_SEED = 2**31 - 1

router = APIRouter(prefix="/api")


class _Req(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlanningEpisodeRequest(_Req):
    layout: Literal["direct", "detour", "enclosed"] = "detour"
    planner: Literal["flat_exact", "waypoint_exact", "flat_learned", "waypoint_learned", "subgoal_learned"] = "waypoint_exact"
    seed: int = Field(default=0, ge=0, le=MAX_SEED)


class VisualEpisodeRequest(_Req):
    layout: Literal["direct", "detour"] = "direct"
    variant: Literal["one_level", "two_level"] = "two_level"
    seed: int = Field(default=0, ge=0, le=MAX_SEED)
    distractor: bool = False
    model_seed: int = Field(default=0, ge=0, le=99)


def _stamp(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@lru_cache(maxsize=4)
def _planning_weights(stamp: float):
    return learned.load()


@lru_cache(maxsize=12)
def _visual_model(levels: int, seed: int, stamp: float):
    return jepa.load(levels, seed)


def _learned_status() -> dict:
    try:
        _, meta = _planning_weights(_stamp(learned.ARTIFACT_DIR / "dynamics.json"))
        return {"available": True, "model_version": meta["model_version"], "created_utc": meta.get("created_utc")}
    except learned.ArtifactMissing as exc:
        return {"available": False, "message": str(exc)}


def _visual_status() -> dict:
    models, missing = [], None
    for seed in VISUAL_MODEL_SEEDS:
        for levels in (1, 2):
            path = jepa.ARTIFACT_DIR / f"{jepa.artifact_name(levels, seed)}.json"
            try:
                _, meta, _ = _visual_model(levels, seed, _stamp(path))
                models.append({"variant": meta["variant"], "model_seed": seed, "model_version": meta["model_version"],
                               "flags": meta["flags"]})
            except jepa.VisualArtifactMissing as exc:
                missing = str(exc)
    return {"available": bool(models), "models": models, **({"message": missing} if missing else {})}


def _recorded(name: str) -> dict | None:
    path = RESULTS / name / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/modes")
def modes() -> dict:
    rp, rv = _recorded("planning"), _recorded("visual")
    return {
        "default": "prediction",
        "modes": ["prediction", "planning", "visual"],
        "planning": {
            "layouts": {k: v.to_dict() for k, v in LAYOUTS.items()},
            "planners": {k: {"label": PLANNER_LABELS[k], "model": m} for k, (_, m) in PLANNERS.items()},
            "learned_layouts": list(LEARNED_LAYOUTS),
            "budget_per_replan": BUDGET,
            "eval_seeds": [PLANNING_EVAL_SEEDS[0], PLANNING_EVAL_SEEDS[-1]],
            "learned_model": _learned_status(),
            "recorded": None if rp is None else {"generated_utc": rp["generated_utc"]},
        },
        "visual": {
            "layouts": list(LEARNED_LAYOUTS),
            "budget_per_replan": VISUAL_BUDGET,
            "eval_seeds": [VISUAL_EVAL_SEEDS[0], VISUAL_EVAL_SEEDS[-1]],
            "models": _visual_status(),
            "recorded": None if rv is None else {"generated_utc": rv["generated_utc"]},
        },
    }


@router.post("/planning/episode")
def planning_episode(req: PlanningEpisodeRequest = PlanningEpisodeRequest()) -> dict:
    weights, meta = None, None
    if PLANNERS[req.planner][1] == "learned":
        if req.layout not in LEARNED_LAYOUTS:
            raise HTTPException(422, f"the learned model was trained on {list(LEARNED_LAYOUTS)} only")
        try:
            weights, meta = _planning_weights(_stamp(learned.ARTIFACT_DIR / "dynamics.json"))
        except learned.ArtifactMissing as exc:
            raise HTTPException(409, str(exc)) from exc
    ep = run_episode(req.layout, req.planner, req.seed, learned_weights=weights)
    ep["source"] = "generated live"
    ep["model_version"] = meta["model_version"] if meta else "exact dynamics"
    return ep


@router.get("/planning/results")
def planning_results() -> dict:
    data = _recorded("planning")
    if data is None:
        raise HTTPException(404, "No recorded planning comparison. Run: python -m experiments.run_planning")
    return data


@router.post("/visual/episode")
def visual_episode(req: VisualEpisodeRequest = VisualEpisodeRequest()) -> dict:
    levels = 1 if req.variant == "one_level" else 2
    path = jepa.ARTIFACT_DIR / f"{jepa.artifact_name(levels, req.model_seed)}.json"
    try:
        model, meta, probes = _visual_model(levels, req.model_seed, _stamp(path))
    except jepa.VisualArtifactMissing as exc:
        raise HTTPException(409, str(exc)) from exc
    ep = run_visual_episode(req.layout, req.variant, req.seed, req.distractor, model, probes)
    ep.update(source="generated live", model_version=meta["model_version"], model_seed=req.model_seed,
              model_flags=meta["flags"])
    return ep


@router.get("/visual/results")
def visual_results() -> dict:
    data = _recorded("visual")
    if data is None:
        raise HTTPException(404, "No recorded visual-planning comparison. Run: python -m experiments.run_visual")
    return data
