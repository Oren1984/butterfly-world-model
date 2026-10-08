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
