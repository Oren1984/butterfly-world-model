# Final QA summary

Checks run at project closure, on the development machine (Windows 11, Docker Desktop), against the application running in Docker Compose. Every line below was executed; nothing is marked PASS on assumption.

## Application

| Check | How | Result |
|---|---|---|
| Backend tests | `python -m pytest` — 82 tests: simulation, prediction, leak prevention, metrics, API validation | PASS |
| Frontend type check | `npm run typecheck` | PASS |
| Frontend unit tests | `npm test` — 8 tests | PASS |
| Browser end-to-end | `npm run e2e` — 4 Playwright tests against `http://localhost:8080`: full user journey, desktop and mobile layout, hidden-tab behaviour, no console errors, same-origin requests only | PASS |
| Docker | Both services up and healthy; `/health` and the page respond; application left running | PASS |
| Evaluation artifacts | `experiments/results/` unchanged since the benchmark run; the benchmark was not re-run because no prediction, simulation or scoring code changed | PASS |

## Showcase video

| Check | How | Result |
|---|---|---|
| Format | `ffprobe`: H.264 High, 1920×1080, 30 fps, yuv420p, 77.47 s, 2,324 frames; AAC stereo 48 kHz | PASS |
| Music added without touching the picture | Video stream checksum identical before and after the mix; levels measured (about −21 LUFS, peak below −1 dBTP), fades confirmed at both ends | PASS |
| Music licence | Source page read: CC0 1.0, no attribution required; track, source, checksum and credit recorded in [showcase-video.md](showcase-video.md); owner approval given | PASS |
| Decodes without errors | `ffmpeg -v error -i … -f null -` produced no output | PASS |
| Plays from start to end | Played through in Chrome: reached the end at 77.47 s, 0 corrupted frames | PASS |
| Footage matches the system | Contact sheets of the whole video reviewed; captions' numbers are read from the live page; second flight's incorrect prediction kept | PASS |
| Hebrew subtitles synchronised | Cue times are logged by the recording script itself; checked in Chrome at 9, 21, 26, 31, 40, 55, 66 and 75 s — each cue matched the English caption on screen | PASS |
| Hebrew subtitles readable | Right-to-left text renders correctly in Chrome; at most two lines per cue; placed at the top, clear of the English captions | PASS |
| No misleading claims | The video states "simulation-based proof of concept … not a trained neural world model"; accuracy figures are those in `metrics.json` | PASS |

## Documentation

| Check | How | Result |
|---|---|---|
| Relative links and image paths | Script resolving every relative link in `README.md` and `docs/*.md` | PASS |
| Mermaid diagrams | Parsed and rendered with mermaid.js 10.9.1 | PASS, with the caveat below |
| Numbers match results | README and `evaluation.md` compared with `experiments/results/metrics.json` | PASS |
| Research links | All four reference URLs fetched; titles and authors confirmed | PASS |
| Repository hygiene | No secrets; no temporary files in the tree; recording frames live in the OS temp folder | PASS |

## One defect found and fixed during closure

While preparing the video: if the demo was **paused** and the wind or a flower was changed, the newly imagined futures were computed but not drawn until the demo was resumed, because the canvas stopped repainting while paused. The frame loop now keeps painting while futures are unfurling (`web/src/main.ts`). No prediction, simulation, scoring or API code was touched.

## Caveats

- Mermaid diagrams were validated with the mermaid library, not on github.com itself.
- The music mix was checked by measurement (loudness, peaks, fades), not by ear.
- Subtitle rendering was checked in Chrome. Other players may place or wrap the text slightly differently.
- GitHub does not play repository MP4 files inline from a README link; the file must be opened or downloaded.
- Playback and subtitle checks were done by automated inspection of frames; nobody has yet watched and listened to the video for pacing or taste.

## V1.1 — Goal-Directed Planning and Visual Planning

Checks run on 2026-11-10, on the development machine (Windows 11, Docker Desktop), on branch `feat/butterfly-planning-experiments`. Baseline before any change: `python -m pytest` on `main` at 18ba504 gave 82 passed.

| Check | Command | Result |
|---|---|---|
| Existing tests still pass | `python -m pytest` | 109 passed: the original 82 plus 27 new (gradient checks, planning, visual, API) |
| Seeded repeatability | `tests/test_planning.py`, `tests/test_visual.py` | The same seed gives an identical episode (latency excluded) |
| Unreachable target | `tests/test_planning.py`; `run_planning` enclosed layout | 0 of 50 landings for both exact planners, no collisions: they fail safely |
| Matched rollout budgets | `tests/test_planning.py` | Every planner spends 960 evaluations per replan, waypoint or subgoal generation included; visual planners spend 768 |
| Data split integrity | `tests/test_planning.py`, `tests/test_visual.py` | Seed ranges are disjoint within each experiment; splits are by whole trajectory; no example continues past a collision |
| CPU smoke runs | the pytest suite trains tiny models; `docker compose --profile lab run --rm lab experiments.train_planning / run_planning / train_visual / run_visual` with small settings and a scratch output path | All four commands run in Docker |
| Missing or incompatible model | `tests/test_labs.py` | Status 409, with the training command named |
| Frontend | `npm run typecheck`; `npm test` | PASS; 8 passed |
| All three modes, through the normal entry point | `npx playwright test` against `docker compose up --build` | 6 passed: the original 4 plus 2 new mode tests (Landing Prediction stays the default and pauses while another mode is shown) |
| Original evidence intact | SHA-256 of `experiments/results/{metrics.json,results.svg,scenarios.csv}` before and after | Identical |
| Full evaluations | `python -m experiments.run_planning` (155 s); `python -m experiments.run_visual` (56 s) | Results in `experiments/results/planning/` and `experiments/results/visual/`, reported in the two V1.1 docs |

**Found and fixed during QA.** The new browser test caught a page error in the replay view. An animation-frame timestamp earlier than the episode's start time gave a replay index of −1. The fix clamps both the time step and the index.
