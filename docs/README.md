# Documentation

Butterfly World Model has three modes. **Landing Prediction** (V1) is the original experiment and the default. **Goal-Directed Planning** and **Visual Planning** (V1.1) are opt-in planning experiments. The [project README](../README.md) gives the overview, the headline results and the commands. In the running application, the **?** button next to the mode tabs opens the **Help Center**, a short guide to each mode and to reading the screen.

## Documents

| Document | Covers | Read it for |
|---|---|---|
| [Architecture](architecture.md) | All | The two services, the modules and what each one is responsible for |
| [Simulation and prediction](simulation-and-prediction.md) | V1 | The garden's rules, how the 20 futures are imagined, and how information leaks are prevented |
| [Evaluation](evaluation.md) | V1 | The 500-scenario benchmark: method, metrics, baselines, results and failures |
| [Goal-Directed Planning](goal-directed-planning.md) | V1.1 | HWM-inspired planning: the controlled garden, the planners, matched budgets, results and limitations |
| [Visual Planning](visual-planning.md) | V1.1 | H-JEPA-inspired planning from pixels: the objective, collapse monitoring, the probe, results and limitations |
| [Research foundation](research-foundation.md) | All | The papers, what each one inspired, and what was deliberately left out |
| [Showcase video](showcase-video.md) | V1 | How to watch the video, its storyboard, how it was recorded, and licensing |
| [QA summary](qa-summary.md) | All | What was checked before each release, with the commands and outcomes |

## Media

| Folder | Contents | How it is produced |
|---|---|---|
| [`images/`](images/) | `garden-twenty-futures.jpg`, `imagined-futures-closeup.png`, `landing-result.jpg`, `evaluation-panel.png` (V1); `planning-mode.jpg`, `planning-flat-stuck.jpg`, `visual-mode.jpg`, `help-center.jpg` (V1.1) | Captured from the running application by `CAPTURE=1 npx playwright test screenshots lab-screenshots` (run in `web/`) |
| [`video/`](video/) | `butterfly-world-model-showcase.mp4` and its Hebrew subtitles `.he.vtt` (V1) | Recorded by `web/e2e/record-showcase.mjs`; see [Showcase video](showcase-video.md) |
| [`../experiments/results/`](../experiments/results/) | The charts and metrics of every recorded evaluation | Written by the `experiments.run_*` commands |
