# Showcase video

| | |
|---|---|
| **File** | [`docs/video/butterfly-world-model-showcase.mp4`](video/butterfly-world-model-showcase.mp4) |
| **Length** | 77 seconds |
| **Format** | 1920 × 1080, 30 fps, H.264 (High profile, yuv420p), AAC stereo audio, about 12 MB |
| **Music** | *The Budding of Consciousness* by Yoiyami (CC0) — see [Licensing](#licensing) |
| **Subtitles** | Hebrew, WebVTT: [`butterfly-world-model-showcase.he.vtt`](video/butterfly-world-model-showcase.he.vtt) |
| **On-screen language** | English |

To watch with subtitles, keep the two files side by side and open the MP4 in a player that picks up sidecar subtitles (VLC does this automatically; in other players, load the `.vtt` as a subtitle file). In a web page, use a `<track kind="subtitles" srclang="he">` element. The subtitles are placed at the top of the picture so they never cover the English captions at the bottom.

## Storyboard

Times are from the finished video.

| Time | Section | What you see | On-screen text |
|---|---|---|---|
| 0:00 – 0:06 | Introduction | Title card | *Butterfly World Model — Predicting the Next Landing* |
| 0:06 – 0:16 | The garden | The butterfly flies between flowers; pollen drifts with the wind. Imagined futures are switched off. | "A butterfly crosses a simulated garden, drawn to flowers and pushed by wind." / "Its intent, the gusts and every random draw stay hidden from the predictor." |
| 0:17 – 0:33 | Imagining the future | The scenario restarts. After one second of flight, time stops and twenty futures unfurl. The view moves in on the fan of trajectories, then back out. | "The model watches one second of flight…" / "Time pauses. The model imagines 20 possible futures." / "Cyan: the most likely flower. Violet: other flowers. Dashed: no landing in time." / "The prediction is frozen before the flight continues: Flower F · 45%" |
| 0:33 – 0:44 | Prediction vs reality | The real flight resumes, the butterfly lands, the result card appears. | "The real flight continues, on its own separate random stream." / "Scored against reality: correct prediction · confidence 45% · landing error 0.2 units" |
| 0:44 – 1:03 | Interactive exploration | A second flight. The wind direction slider is dragged from 124° to 305°, then a flower is dragged next to the butterfly; each change replaces the futures. The flight then finishes under the new conditions. | "Change the world: turn the wind…" / "…or move a flower. The old prediction is discarded and imagined again." / "The flight goes on under the new conditions." |
| 1:03 – 1:12 | Research result | The application's evaluation panel is highlighted; its headline numbers are shown large. | "Top-1 landing accuracy on 500 held-out scenarios — 72.0% world model · 57.6% nearest-flower baseline — Brier score 0.422 vs 0.848 — A simulation-based proof of concept with hand-written dynamics, not a trained neural world model." |
| 1:12 – 1:17 | Closing | Title card | *Imagine possible futures. Predict what happens next.* |

The second flight in the video ends with an **incorrect** prediction (the model named flower E, the butterfly landed on flower C). It was left in on purpose: the model is right about 72% of the time, not always.

## How it was made

The video is produced by one script, [`web/e2e/record-showcase.mjs`](../web/e2e/record-showcase.mjs):

```bash
docker compose up -d
cd web && MUSIC=/path/to/the_budding_of_consciousness.wav node e2e/record-showcase.mjs
```

`MUSIC` is optional; without it the video is silent. It needs Node, the Playwright Chromium already used by the browser tests, and `ffmpeg` on the path. No other tools, services or keys.

**What is real.** Everything inside the application frame is the running application talking to the running API: the garden, the flights, the twenty trajectories, the probabilities, the result cards, and the evaluation panel. The numbers in the captions (flower, confidence, landing error, 72.0%, 57.6%, 500, the Brier scores) are read from the application's own page while recording, not typed into the script. The scenario is seed 7, the application's default.

**What the script adds.** Title cards, caption bars, a cursor marker, a gold outline around the evaluation panel, and the large-number card in the "research result" section. These are overlays drawn on top of the page for the recording only; they are not part of the application.

**How the script drives the application.** It clicks the real buttons and drags the real slider and flower, as a visitor would. Four things are done for the sake of the film and are worth stating plainly:

- *The opening garden shots* use the application's "Show imagined futures" switch turned off, and the result card is hidden by a recording-only style, so that the first ten seconds show only the butterfly. The script presses "Next flight" after each landing to keep it moving.
- *The frozen moment is held longer than in the live application.* The application itself pauses for 1.7 seconds while the futures unfurl. The script then presses the application's Pause button to hold the picture for about thirteen more seconds so it can be explained. The side panel shows "Paused" during this time.
- *The slow move toward the trajectories* is a zoom applied to the page during the recording.
- *The result cards are held* with the Pause button so they can be read.

**Capture method.** The video is not a real-time screen recording. Real-time capture on the development machine was uneven (roughly 10–30 frames per second), so the script steps the page's clock forward by exactly 1/30 of a second, waits for any API call to finish, and takes a screenshot — 2,324 times. One second of video is exactly one second of application time, and the application runs the same code as always; it is simply not asked to keep up with a stopwatch. The frames are encoded with `ffmpeg` (libx264, CRF 19).

**Isolation.** The live garden is a single in-memory world shared by every open browser tab. A first take was spoiled when another tab advanced the world during the held moment. The final take was therefore recorded against a second, temporary copy of the same containers on other ports:

```bash
WEB_PORT=8090 API_PORT=8001 docker compose -p bwm-record up -d
cd web && BASE_URL=http://localhost:8090 node e2e/record-showcase.mjs
docker compose -p bwm-record down
```

**Subtitles.** The script records the moment each caption appears and writes the Hebrew cues with those same times, so the subtitles are synchronised by construction. Lines are wrapped to 40 characters.

## Licensing

| Element | Source | Licence |
|---|---|---|
| Footage | Recorded from this project's own application | Same as the repository |
| Titles, captions, overlays | Written for this video | Same as the repository |
| Typeface | System UI font of the recording machine, rendered into the picture | No font file is distributed |
| Music | *The Budding of Consciousness* by Yoiyami | CC0 1.0 Universal |

### Music

| | |
|---|---|
| **Track** | *The Budding of Consciousness – CC0 Ambient / Minimalist Theme (Yoiyami Blue Series – No.4)* |
| **Artist** | Yoiyami |
| **Source** | https://opengameart.org/content/the-budding-of-consciousness-%E2%80%93-cc0-ambient-minimalist-theme-yoiyami-blue-series-%E2%80%93-no4 |
| **File used** | `the_budding_of_consciousness.wav` (48 kHz stereo, 3:44), downloaded on 8 October 2026. SHA-256 `c879c57f4353c16926026e05fea372be64a73356ea663475aa1f54bf5499b513` |
| **Licence** | [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) (public domain dedication), as stated on the source page on the download date |
| **Attribution** | Not required by the licence. Given anyway, as the artist suggests: **Music by Yoiyami (CC0)**. Because none is required, there is no credit line inside the video itself; the credit is here and in the file's metadata comment. |
| **Approval** | Approved by the project owner before it was added. |

**How it is used.** The first 77 seconds of the track, unedited apart from volume: a 2.5-second fade in, a 4-second fade out under the closing card, and a small lift (about 2 dB) from the moment the twenty futures appear until the real flight resumes. The mix sits at about −21 LUFS so that it stays in the background. The picture was not re-encoded when the music was added.

The source file is not stored in this repository (43 MB); download it from the source page to rebuild the video with sound.
