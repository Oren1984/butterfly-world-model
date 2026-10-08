# Simulation and prediction

There are two programs here that must not be confused:

- the **actual world**, which decides what really happens, and
- the **prediction engine**, which only watches and imagines.

They agree on the rules of the garden. They share nothing else.

## 1. The world state

The garden is a rectangle of 100 × 60 units. Time advances in steps of 0.1 s.

| Visible to everyone (the `Observation`) | Hidden inside the world |
|---|---|
| Butterfly position and velocity | Which flower the butterfly currently intends to visit |
| Mean wind strength and direction | The current gust (a random deviation from the mean wind) |
| Flower positions and landing-zone radius | The random generator and everything it will draw next |
| The "behaviour randomness" setting (0–1) | |
| How long the butterfly has been airborne | |

## 2. How the actual world moves

At every step the simulator does five things (`api/simulation/world.py`):

1. **Change of mind.** With a small probability per step (higher when randomness is high), the butterfly picks a new intended flower. Nearer flowers are more attractive: the chance of choosing flower *f* is proportional to `exp(-distance_f / 18)`.
2. **Gust.** The gust drifts randomly but is pulled back toward zero (an Ornstein–Uhlenbeck process), so the wind wobbles around its mean. Stronger wind means stronger gusts.
3. **Acceleration.** Three contributions are added:
   - *steering* toward the intended flower, slowing down on approach: `2.5 × (desired velocity − velocity)`;
   - *wind*: `1.2 × (mean wind + gust)`;
   - *flutter*: random noise scaled by the randomness setting.
4. **Movement.** Velocity and position are updated, then the physical limits are applied.
5. **Landing.** Inside a flower's landing zone and flying slowly enough, the butterfly lands with probability 0.35 per step on its intended flower, or 0.03 per step on any other flower it happens to be over.

**Physical limits.** Acceleration is capped at 45 units/s², speed at 18 units/s. The butterfly bounces softly off the garden walls and can never leave. Movement per step is therefore at most 1.8 units — no teleporting. A landed butterfly does not move. Landing is impossible during the first second after take-off.

A worked example of what wind does: steering and wind balance when the butterfly drifts at `1.2 / 2.5 = 0.48` times the wind speed. At the maximum wind of 12 that is a sideways drift of almost 6 units/s against a cruise speed of 11, which is why strong wind bends flights and sometimes keeps the butterfly from settling on a flower.

All numbers live in `api/params.py`. They were chosen to look plausible and stay stable, not to match real butterflies.

## 3. How the prediction engine imagines

The engine (`api/prediction/predictor.py`) receives one `Observation`, a number of futures (default 20) and a horizon (default 60 steps = 6 s). Then:

1. **Guess the intent.** The engine cannot see which flower the butterfly wants, so it forms a belief. A flower is more likely if it is near, and more likely if the butterfly is flying toward it. Before judging the heading, the engine subtracts the drift the wind is expected to cause, so that a butterfly blown sideways is not mistaken for one that wants to go sideways.
2. **Start twenty particles.** Each future starts at the observed position and velocity, draws its own intended flower from that belief, and draws its own starting gust.
3. **Roll forward.** Each future is advanced step by step under the same documented rules as the world — changes of mind, gusts, steering, flutter, limits, landing — using the engine's own random numbers.
4. **Summarise.** The probability of each flower is simply the share of futures that landed on it. Futures that did not land keep their own category, `no_landing_within_horizon`. Nothing is discarded and nothing is renormalised: if 5 of 20 futures never land, flowers share 75%, not 100%.

The output also includes each future's path (for drawing), the mean landing point on the most likely flower, the spread of landing points, the inferred intent belief, and how far apart the twenty futures are at each moment — the "uncertainty grows over the horizon" curve.

**Example.** In the screenshot in the README, 9 futures land on flower F, 6 on flower B and 5 never land, so the prediction is F with 45% confidence, B 30%, no landing 25%.

## 4. Preventing information leaks

A predictor that could peek at the simulator would look brilliant and prove nothing. These are the safeguards, and each one is covered by a test in `tests/test_prediction.py`.

| Risk | Safeguard |
|---|---|
| Predictor reads hidden state | `predict()` accepts only an `Observation`, a frozen data object with no field for intent, gust or random state. The predictor module does not import the simulator. |
| Shared random numbers | Three separate random streams per seed: scenario layout, world, predictor. Running predictions does not change the world's trajectory by a single bit (tested). |
| Predictor depends on hidden state indirectly | Two worlds with identical observations but different hidden intent, gust and generator produce identical predictions — and different real futures (tested). |
| Predictor replays the real future | No imagined trajectory coincides with the trajectory the world then actually follows (tested). |
| Prediction adjusted after the fact | In evaluation and in the live session, the prediction is produced and frozen *before* the world advances. Later live predictions never replace the scored one. |
| Stale predictions shown as current | Any change to wind, randomness or flowers bumps a version number; older predictions are discarded by the server and the browser. |

## 5. Why the predictor is still imperfect

Even with the right rules, three things remain unknown to the engine: the butterfly's intent at the snapshot, the gust, and every random draw after the snapshot. With only twenty samples its probabilities are also coarse. On the benchmark it is right 72% of the time — see [evaluation.md](evaluation.md) for where it goes wrong.

## 6. The baselines

- **Constant velocity.** Extend the current velocity in a straight line for six seconds. The first landing zone the line crosses is the prediction; if it crosses none, the prediction is "no landing".
- **Nearest flower.** Always predict the flower closest to the butterfly.

Neither has any tunable parameter.
