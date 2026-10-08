# Research foundation

This project is a small demonstration **inspired by** research on world models. It is not a reproduction of any paper. This page says exactly what was borrowed, what was built, and what was left out.

## The idea in one paragraph

An agent that only reacts to its current input cannot plan. An agent with a *world model* has an internal, simplified copy of how its environment behaves. It can run that copy forward — "imagine" — to see what is likely to happen, and it can do so several times to see how much those futures disagree. The disagreement is a measure of uncertainty. Comparing imagination with reality tells you how good the model is.

## The papers

### 1. World Models

- **Full title:** *Recurrent World Models Facilitate Policy Evolution* (widely known as "World Models")
- **Authors, year:** David Ha and Jürgen Schmidhuber, 2018 (NeurIPS 2018)
- **Source:** https://worldmodels.github.io/
- **Main relevant idea:** Split an agent into a *vision* model (V) that compresses observations into a compact state, a *memory* model (M) that predicts how that state evolves — as a probability distribution, not a single answer — and a small *controller* (C). Because M is generative, the agent can be trained inside its own "dream".
- **What inspired us:** The separation between a compact world state and a stochastic model of what happens next, and the picture of "dreaming" forward from the present.
- **What we implemented:** A compact, explicit world state (position, velocity, wind, flowers) and a stochastic transition model that is rolled forward to generate imagined trajectories.
- **What we did not implement:** No VAE, no MDN-RNN, no learning from pixels or from data at all, no controller, and no training inside the dream. Our state is given, not learned; our dynamics are hand-written.

### 2. Dreamer

- **Full title:** *Dream to Control: Learning Behaviors by Latent Imagination*
- **Authors, year:** Danijar Hafner, Timothy Lillicrap, Jimmy Ba and Mohammad Norouzi, 2019 (arXiv:1912.01603; published at ICLR 2020)
- **Source:** https://arxiv.org/abs/1912.01603
- **Main relevant idea:** Learn a latent dynamics model, then learn behaviour purely from trajectories *imagined* by that model, starting from states the agent has really encountered.
- **What inspired us:** Starting imagination from a real, just-observed state and rolling forward over a fixed horizon; treating imagined trajectories as the object you reason about.
- **What we implemented:** At a snapshot of the real world, twenty trajectories are imagined over a six-second horizon, and a decision-relevant summary (where will it land?) is read off them.
- **What we did not implement:** No latent space, no learned model, no reward or value prediction, no actor-critic, no gradients through imagined trajectories. There is no agent taking actions — we predict, we do not control.

### 3. PETS

- **Full title:** *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models*
- **Authors, year:** Kurtland Chua, Roberto Calandra, Rowan McAllister and Sergey Levine, 2018 (NeurIPS 2018)
- **Source:** https://arxiv.org/abs/1805.12114
- **Main relevant idea:** Use *probabilistic* dynamics models and propagate uncertainty by pushing many sampled particles through the model ("trajectory sampling"), distinguishing noise inherent in the world (aleatoric) from ignorance about the model (epistemic, handled with an ensemble).
- **What inspired us:** Particle-based uncertainty propagation: do not predict one future, sample many and look at the cloud.
- **What we implemented:** Twenty particles, each with independently sampled hidden state and noise, propagated step by step. The spread of the cloud over time is reported (the "uncertainty grows" curve in the UI) and the landing distribution is the share of particles ending on each flower.
- **What we did not implement:** No neural network ensemble, so no epistemic uncertainty about the dynamics themselves. No model-predictive control or CEM planning. Our uncertainty comes from process noise and from not knowing the butterfly's hidden intent and the current gust.

### Optional biological note

- **Full title:** *Unconventional lift-generating mechanisms in free-flying butterflies*
- **Authors, year:** R. B. Srygley and A. L. R. Thomas, 2002 (*Nature* 420, 660–664)
- **Source:** https://www.nature.com/articles/nature01223
- **Relevance:** Real butterfly flight is aerodynamically complex and irregular. This is only a reminder of why a butterfly makes a good emblem for "hard to predict". **Nothing from this paper is modelled.** The butterfly here is a point mass with steering, wind drag and random flutter.

## Concept mapping

| Concept | Where it comes from | Status here |
|---|---|---|
| Explicit world-state representation | World Models | Implemented (hand-designed, not learned) |
| Stochastic model of environment dynamics | World Models, PETS | Implemented (hand-written equations) |
| Imagined future trajectories from a real state | Dreamer, World Models | Implemented (20 rollouts) |
| Probabilistic prediction rather than a point estimate | World Models, PETS | Implemented (distribution over landing outcomes) |
| Uncertainty propagation by particles | PETS | Implemented (aleatoric and hidden-state uncertainty only) |
| Belief over an unobserved variable | general state estimation | Implemented in a simple form (belief over the intended flower) |
| Evaluating predictions against reality | all three evaluate their models | Implemented (500 scenarios, proper scoring rule, baselines) |
| Learning the model from data | all three | **Not implemented** |
| Latent representation learning | World Models, Dreamer | **Not implemented** |
| Epistemic uncertainty / ensembles | PETS | **Not implemented** |
| Planning or control with the model | all three | **Not implemented** |

## What this proof of concept actually demonstrates

1. A clean separation between an actual stochastic world and a model that predicts it from observations alone.
2. That sampling a modest number of futures gives a usable, reasonably calibrated probability distribution over outcomes.
3. That such a model beats simple extrapolation baselines in this environment, by a measured margin, and where it still fails.
4. A way to *see* all of the above.

## Scientific limitations

- **No learning.** The model is given the correct form of the dynamics. Its success says nothing about how hard it would be to learn those dynamics.
- **Well-specified model.** World and predictor share the same documented assumptions. The reported accuracy is therefore closer to an upper bound for this environment than to what a learned model would reach.
- **Synthetic environment.** The dynamics were designed to be simple and interpretable, not to resemble a real insect.
- **Small samples.** Twenty futures give probabilities in steps of 0.05, and 500 scenarios give confidence intervals of roughly ±4 percentage points on accuracy.
- **One tuned parameter.** The weight of heading evidence in the intent belief was chosen on a separate set of scenarios; see [evaluation.md](evaluation.md).

## Future research possibilities

- Replace the hand-written dynamics with a small learned probabilistic model trained on recorded flights, and compare it with the hand-written one on the same benchmark.
- Introduce deliberate model mismatch (wrong wind coupling, wrong landing rule) and measure how accuracy and calibration degrade.
- Add an ensemble to represent uncertainty about the model itself, as PETS does.
- Replace sampling the intent once at the snapshot with a filter that updates the belief as the flight is observed.
- Study how accuracy and calibration change with the number of imagined futures (5, 20, 100, 500).
- Close the loop: give an agent an action (for example, placing a flower) and let it plan with the model.
