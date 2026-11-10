# Butterfly World Model — Predicting the Next Landing

> *Don't just observe the world. Imagine what happens next.*

**One butterfly. Twenty imagined futures. One real landing. A measurable prediction.**

*V1.1 — developed and built by Oren Salami.* Version 1.1 adds two opt-in planning experiments next to the original landing prediction, which stays the default: **[Goal-Directed Planning](#v11-goal-directed-planning-hwm-inspired)** and **[Visual Planning](#v11-visual-planning-h-jepa-inspired)**.

[![The garden at the moment of prediction: one butterfly, twenty imagined futures, one predicted flower](docs/images/garden-twenty-futures.jpg)](docs/video/butterfly-world-model-showcase.mp4)

### ▶ [Watch the showcase video (77 seconds)](docs/video/butterfly-world-model-showcase.mp4)

Recorded from the running application, 1080p, with ambient music by Yoiyami (CC0). Hebrew subtitles are provided as [WebVTT](docs/video/butterfly-world-model-showcase.he.vtt). On GitHub, open the link and choose *View raw* to download and play it.

To play it from a clone of the repository, run one of these from the project folder:

```bash
start docs\video\butterfly-world-model-showcase.mp4      # Windows (cmd or PowerShell)
open docs/video/butterfly-world-model-showcase.mp4       # macOS
xdg-open docs/video/butterfly-world-model-showcase.mp4   # Linux
```

With the Hebrew subtitles, using [VLC](https://www.videolan.org/vlc/):

```bash
vlc docs/video/butterfly-world-model-showcase.mp4 --sub-file docs/video/butterfly-world-model-showcase.he.vtt
```

> **What this is, and is not.** A compact, simulation-based proof of concept. In the original landing-prediction mode the "world model" is a small set of hand-written rules rolled forward with random sampling — it is **not** a trained neural world model, and nothing in it learns. The V1.1 planning modes add small models that *are* trained (on the CPU, from seeded simulated flights); they are inspired by, not reproductions of, the cited research.

## What did we build?

A butterfly flies through a small virtual garden, pushed around by wind and by its own changes of mind. One second into every flight, a prediction engine looks at what can be seen — where the butterfly is, how fast it is moving, where the flowers are, how the wind blows — and imagines twenty different ways the flight could continue. From those twenty futures it names the flower it thinks the butterfly will land on, and says how sure it is.

Then the real butterfly finishes its flight, and we check whether the prediction was right.

## Why does it matter?

Most software reacts to what is happening now. A *world model* goes one step further: it keeps an internal picture of how the environment behaves, uses that picture to imagine what might happen next, and then compares what it imagined with what really happened.

That idea is at the heart of a line of research on model-based agents ([World Models](https://worldmodels.github.io/), [Dreamer](https://arxiv.org/abs/1912.01603), [PETS](https://arxiv.org/abs/1805.12114)). Those systems *learn* their world model from data and use it to choose actions. This project does something much smaller: its model of the garden is written by hand, and nothing is trained. What it does show, in a form you can watch and measure, is the loop those papers build on — represent the world, imagine several futures, keep track of uncertainty, and score the result against reality.

## How does it work?

```mermaid
flowchart TD
    A["🦋 Observe the world"] --> B["Capture the current state<br/>position · velocity · wind · flowers"]
    B --> C["Imagine 20 futures<br/>each with its own random draws"]
    C --> D["Predict landing probabilities<br/>per flower + 'no landing'"]
    D --> E["The butterfly actually lands<br/>the real world runs on, unseen"]
    E --> F["Compare prediction vs reality"]
    F --> G["Measure results<br/>accuracy · Brier score · calibration"]

    classDef see fill:#0c1330,stroke:#5ee7ff,color:#e8eeff
    classDef imagine fill:#0c1330,stroke:#a78bfa,color:#e8eeff
    classDef real fill:#0c1330,stroke:#ffd166,color:#e8eeff
    class A,B see
    class C,D imagine
    class E,F,G real
```

The real garden and the prediction engine are two separate programs inside the API. The engine only receives what a camera could see. It is never told which flower the butterfly is heading for, what the next gust will be, or what random numbers the garden will draw. It has to guess those, which is exactly why its twenty futures disagree with each other.

| | |
|---|---|
| ![Close-up of the twenty imagined trajectories fanning out from the butterfly](docs/images/imagined-futures-closeup.png) | ![The landing: predicted flower, actual flower, confidence and position error](docs/images/landing-result.jpg) |
| **Twenty imagined futures.** Cyan paths end on the most likely flower, violet on other flowers, dashed grey ones never land in time. | **The real landing.** The gold trail is what actually happened; the card scores the frozen prediction against it. |

## Key capabilities

- **A living garden** — animated butterfly, flowers, wind and pollen on a single Canvas, at real-time speed.
- **Twenty futures at a glance** — the flight pauses for a moment while the imagined paths unfurl, then the model keeps re-imagining as the butterfly flies.
- **Honest probabilities** — every flower gets a probability, and so does "no landing within six seconds".
- **Change the world** — drag flowers, turn the wind, raise the butterfly's randomness. Old predictions are thrown away and replaced immediately.
- **A prediction you can score** — predicted flower, actual flower, confidence and landing position error after every flight.
- **A real benchmark** — 500 reproducible scenarios, two baselines, runnable from the UI or the command line.

## Evaluation highlights

Measured on 500 scenarios (seeds 0–499) that were never used for tuning. Full method and failure analysis: [docs/evaluation.md](docs/evaluation.md).

| Predictor | Top-1 accuracy | Brier score (0–2, lower is better) | Mean landing error |
|---|---|---|---|
| **World model, 20 futures** | **72.0%** | **0.422** | **6.0 units** |
| Nearest flower (baseline) | 57.6% | 0.848 | 11.4 units |
| Constant velocity (baseline) | 36.6% | 1.268 | 9.3 units |

- The model's confidence is close to honest: when it says about 88%, it is right 84% of the time (calibration error 0.039).
- A prediction takes about 4 ms (median) on a laptop.
- It is far from perfect. It is wrong in 28% of scenarios, and it recognises only 13 of the 57 flights that never land.

![Evaluation chart: accuracy, Brier score, calibration, and accuracy by wind strength](experiments/results/results.svg)

## V1.1: Goal-Directed Planning (HWM-inspired)

The app now has three tabs. **Landing Prediction**, the original mode, is still the default. The two new tabs are opt-in, and each explains what it shows.

In this mode, an agent **steers** the butterfly to a target flower against a seeded wind. The comparison is between a *flat* planner, which scores action sequences toward the flower directly, and *hierarchical* planners, which first pick an intermediate waypoint and then refine it into steering actions. Every planner replans every 0.4 s and gets the same budget: 960 model evaluations per replan, waypoint generation included.

![Goal-Directed Planning: the hierarchical planner routes around the hedge through a waypoint](docs/images/planning-mode.jpg)

| | |
|---|---|
| ![The flat planner stuck in front of the hedge](docs/images/planning-flat-stuck.jpg) | ![Example routes on the detour](experiments/results/planning/routes.svg) |
| **Flat planning gets stuck.** Its 3-second lookahead cannot see past the hedge. | **Actual flights, evaluation seed 0.** Rings mark the waypoints and subgoals each planner chose. |

Landing rate on 50 frozen evaluation seeds per route:

| Model | Planner | Direct route | Detour route | Collisions |
|---|---|---|---|---|
| Exact dynamics | Flat | 100% | 26% | 0% |
| Exact dynamics | Explicit waypoints | 100% | **80%** | 0% |
| Learned model | Flat | 100% | 6% | 0% |
| Learned model | Explicit waypoints | 100% | 10% | 0% |
| Learned model | **Learned long-horizon subgoals** | 100% | **60%** | 0% |

![Planning comparison chart](experiments/results/planning/comparison.svg)

- On the unreachable (walled-in) target, every planner fails safely.
- Hierarchy helps only when the direct approach is a trap.
- With the learned model, subgoals taken from a model that predicts 1.6 s per step outperform chained one-step rollouts.

Method, latency and limitations: [docs/goal-directed-planning.md](docs/goal-directed-planning.md).

## V1.1: Visual Planning (H-JEPA-inspired)

Here the planner sees **only 50 × 30 pixel frames and a goal image**, never coordinates. It plans in a representation learned from pixels with a joint-embedding predictive objective. Two variants are compared:

- a one-level latent world model;
- a jointly trained two-level model, whose higher level predicts 1.6 s ahead and supplies subgoals.

An optional distractor wanders through the scene on its own. In the replay, the cyan "imagined route" is decoded from the planner's latents by a frozen diagnostic probe, and is shown for display only.

![Visual Planning: the planner's view, the goal image and the replay](docs/images/visual-mode.jpg)

Landing rate on 20 frozen seeds per cell, averaged over 3 training seeds (each seed shown in brackets):

| Variant | Direct, clean | Direct, distractor | Detour, clean | Detour, distractor |
|---|---|---|---|---|
| One-level | 0% | 0% | 0% | 0% |
| Two-level | **33%** (90 / 10 / 0) | 10% (10 / 20 / 0) | 0% | 0% |

![Visual planning comparison chart](experiments/results/visual/comparison.svg)

**Planning from pixels is much harder than planning from states.** The two-level model is the only one that lands, and it does so unreliably: the result depends heavily on the training seed. It drops under distraction, and neither variant solves the detour.

A frozen linear probe recovers the butterfly's position well (R² ≈ 0.72) and the distractor's poorly (R² ≈ 0.27). No training run collapsed.

Objective, collapse monitoring, failures and limitations: [docs/visual-planning.md](docs/visual-planning.md).

### Train, evaluate and demo (PowerShell and Docker)

Every command runs locally on the CPU, with no accounts and no paid APIs. The demo uses the trained models in `artifacts/` and the recorded results in `experiments/results/`. If a model is missing, the app says so and names the command that trains it.

**Docker.** The `lab` profile mounts `artifacts/` and `experiments/results/`, so its outputs land in your working copy and the running app picks them up.

```powershell
docker compose up --build -d                                              # demo: http://localhost:8080 (three tabs)
docker compose --profile lab run --rm lab experiments.train_planning      # ~30 s
docker compose --profile lab run --rm lab experiments.run_planning        # ~3 min
docker compose --profile lab run --rm lab experiments.train_visual        # ~30 min (6 models)
docker compose --profile lab run --rm lab experiments.run_visual          # ~1 min
```

**PowerShell, from the project folder.**

```powershell
python -m venv .venv; .venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m experiments.train_planning      # writes artifacts/planning/
.venv\Scripts\python -m experiments.run_planning        # writes experiments/results/planning/
.venv\Scripts\python -m experiments.train_visual        # writes artifacts/visual/ (seeds 0 1 2, both variants)
.venv\Scripts\python -m experiments.run_visual          # writes experiments/results/visual/
.venv\Scripts\python -m pytest                          # 109 tests, including CPU smoke training runs
```

Each run exports:

- `metrics.json`: seeds, config, model versions, metrics, and the time it was generated;
- `episodes.csv`: one row per episode;
- a chart.

Each model artifact records its version, training seed, config, training curve and validation metrics.

## Quick start

You need Docker. Nothing else — no accounts, keys or cloud services.

```bash
docker compose up --build
```

Then open **http://localhost:8080**. The tabs at the top of the side panel switch between Landing Prediction, Goal-Directed Planning and Visual Planning. (The API is also reachable directly at http://localhost:8000, with interactive docs at `/docs`.) Stop with `Ctrl+C` or `docker compose down`. If a port is taken: `WEB_PORT=9090 API_PORT=9000 docker compose up --build`.

Reproduce the benchmark:

```bash
docker compose exec api python -m experiments.run_evaluation --out /tmp/results
```

Run the tests locally (Python 3.12, Node 22+):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt   # Windows: .venv\Scripts\pip
.venv/bin/python -m pytest                    # backend: simulation, prediction, metrics, API
.venv/bin/python -m experiments.run_evaluation  # rewrites experiments/results/
cd web && npm ci && npm run typecheck && npm test
npx playwright install chromium && npm run e2e  # browser test against the running Docker app
```

## Research foundation

| Paper | Idea borrowed |
|---|---|
| Ha & Schmidhuber, [*World Models*](https://worldmodels.github.io/) (2018) | Separate "what the world is" from "what happens next", and roll the model forward as a dream. |
| Hafner et al., [*Dream to Control*](https://arxiv.org/abs/1912.01603) (2019) | Reason about the future through imagined trajectories from the current state. |
| Chua et al., [*PETS*](https://arxiv.org/abs/1805.12114) (2018) | Carry uncertainty forward by propagating many sampled particles instead of one best guess. |
| Zhang et al., [*Hierarchical Planning with Latent World Models*](https://arxiv.org/abs/2604.03208) | V1.1: plan through intermediate subgoals proposed by a longer-horizon model. |
| Zhang et al., [*H-JEPA*](https://h-jepa.com/) | V1.1: hierarchical joint-embedding predictive world models learned from pixels, for visual planning. |

This project **does not reproduce any of these papers**. The landing-prediction mode has no neural network, no learning and no control. The V1.1 planning modes train small models on the CPU and are labelled as inspired adaptations. Details, including what was deliberately left out: [docs/research-foundation.md](docs/research-foundation.md).

## Known limitations

- **Nothing is learned.** The prediction engine is given the rules of the garden; only the hidden intent, gusts and random draws are unknown to it. A learned model would also have to discover the rules.
- **The model is "well specified".** Predictor and world share the same documented equations, which is the easiest possible setting. Real environments never match the model this well.
- **Twenty samples are coarse.** Probabilities move in steps of 5%, and rare outcomes such as "no landing" are often missed.
- **A toy world.** Two dimensions, one butterfly, a few flowers, and no real butterfly aerodynamics or behaviour.
- **Single user.** The live garden is one in-memory world shared by every browser tab that connects.
- **V1.1 results come from small samples.** 50 seeds per route (planning) and 20 seeds × 3 training seeds per cell (visual) give wide intervals, and the visual results vary strongly between training seeds.
- **The V1.1 learned models never leave their two gardens.** They are trained and evaluated on the same two layouts (with different seeds, starts and winds) and are not tested on unseen gardens.

## More documentation

- [Architecture](docs/architecture.md) — the two services and what each part is responsible for.
- [Simulation and prediction](docs/simulation-and-prediction.md) — the world's rules, how futures are imagined, and how information leaks are prevented.
- [Evaluation](docs/evaluation.md) — method, metrics, baselines, results and failures.
- [Research foundation](docs/research-foundation.md) — the papers, what inspired what, and future directions.
- [Showcase video](docs/showcase-video.md) — storyboard, how it was recorded, and licensing.
- [Goal-Directed Planning (V1.1)](docs/goal-directed-planning.md) — HWM-inspired: the controlled garden, planners, matched budgets, results and limitations.
- [Visual Planning (V1.1)](docs/visual-planning.md) — H-JEPA-inspired: pixels-only models, objective, collapse monitoring, probe, results and limitations.
- [QA summary](docs/qa-summary.md) — what was checked before closing the project, and the outcome.
