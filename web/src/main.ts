import { api, type ConfigChange } from "./api";
import { clamp, clampFlower, lerp, lerpAngle, outcomeLabel, pct } from "./geometry";
import { Renderer, type Scene } from "./render";
import { NO_LANDING, type Flower, type Metrics, type Prediction, type Pt, type WorldState } from "./types";

const TICK_MS = 100; // one simulation step (dt = 0.1 s) per tick: real time
const LIVE_MS = 700; // how often the model re-imagines during a flight
const FREEZE_MS = 1700; // time stands still while the committed futures unfurl
const REVEAL_MS = 1400;
const RESULT_MS = 6500; // result card stays up this long before the next flight
const MAX_WIND = 12;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

interface Sample {
  step: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

interface Shown {
  prediction: Prediction;
  revealStart: number;
  revealMs: number;
}

const canvas = $<HTMLCanvasElement>("garden");
const renderer = new Renderer(canvas);

let state: WorldState | null = null;
let prev: Sample | null = null;
let curr: Sample | null = null;
let tickAt = 0;
let heading = 0;
let trail: Pt[] = [];
let shown: Shown | null = null;
let predictionSerial = 0;
/** Bumped whenever whatever is on screen stops being a valid prediction. */
let epoch = 0;
let awaitingConfig = false;
let paused = false;
let offline = false;
let showFutures = true;
let stepping = false;
let predicting = false;
let connecting = false;
let advancing = false;
let freezeUntil = 0;
let committedVisibleAt = 0;
let lastLiveAt = 0;
let resultAt = 0;
let dirty = true;
let drag: { id: string } | null = null;
let selected: string | null = null;
let override: Record<string, { x: number; y: number }> = {};
let pendingConfig: ConfigChange = {};
let configTimer: number | undefined;
let evaluated = false;

const finished = () => state !== null && (state.status === "landed" || state.status === "no_landing");

function flowers(): Flower[] {
  return state ? state.observation.flowers.map((f) => (override[f.id] ? { ...f, ...override[f.id] } : f)) : [];
}

// ---- server state -----------------------------------------------------------

function applyState(s: WorldState, viaStep: boolean): void {
  const now = performance.now();
  const before = state;
  if (before && s.config_version < before.config_version) return; // an older response arrived late
  state = s;
  const o = s.observation;
  const sample: Sample = { step: o.step, ...o.butterfly };
  if (!curr || o.step !== curr.step) {
    prev = viaStep && curr && o.step === curr.step + 1 ? curr : sample;
    curr = sample;
    tickAt = now;
  }

  if (!before || before.config_version !== s.config_version) {
    // The world's public conditions or the flight changed: what was drawn is obsolete.
    epoch++;
    shown = null;
    const newFlight =
      s.status === "observing" &&
      (!before || before.status !== "observing" || before.seed !== s.seed || o.step < before.observation.step);
    if (newFlight) {
      trail = [];
      $("result").hidden = true;
    }
    if (s.status === "flying") {
      const snapshot = viaStep && before?.status === "observing";
      if (snapshot) freezeUntil = now + FREEZE_MS;
      committedVisibleAt = snapshot ? now + REVEAL_MS * 0.75 : now;
      window.setTimeout(renderPanel, REVEAL_MS * 0.75 + 30);
      void loadCommitted(snapshot ? REVEAL_MS : 450);
    }
  }
  if (viaStep) trail.push(...s.path);
  if (trail.length === 0) trail.push([o.butterfly.x, o.butterfly.y]);

  if (s.result && !before?.result) {
    resultAt = now;
    showResult(s);
    void loadCommitted(1); // show the frozen futures next to what actually happened
  }
  renderPanel();
  dirty = true;
}

/** Fetch the frozen prediction this flight is scored against and unfurl it. */
async function loadCommitted(revealMs: number): Promise<void> {
  const e = epoch;
  try {
    const r = await api.committed();
    if (e !== epoch || !state || r.config_version !== state.config_version) return;
    setShown({ prediction: r.prediction, revealStart: performance.now(), revealMs });
    lastLiveAt = performance.now() + (revealMs > 1000 ? FREEZE_MS - LIVE_MS + 500 : 300);
  } catch {
    /* the next tick reports connection problems */
  }
}

function setShown(next: Shown): void {
  shown = next;
  predictionSerial++;
  renderPanel();
  dirty = true;
}

/** Called the moment the user changes anything: never leave stale futures on screen. */
function invalidate(): void {
  epoch++;
  shown = null;
  awaitingConfig = true;
  renderPanel();
  dirty = true;
}

function queueConfig(change: ConfigChange, immediately = false): void {
  const moved = [...(pendingConfig.flowers ?? []), ...(change.flowers ?? [])];
  const byId = new Map(moved.map((f) => [f.id, f]));
  pendingConfig = { ...pendingConfig, ...change, ...(byId.size ? { flowers: [...byId.values()] } : {}) };
  window.clearTimeout(configTimer);
  configTimer = window.setTimeout(flushConfig, immediately ? 0 : 120);
}

async function flushConfig(): Promise<void> {
  const body = pendingConfig;
  pendingConfig = {};
  try {
    const s = await api.config(body);
    if (Object.keys(pendingConfig).length === 0) {
      awaitingConfig = false;
      if (!drag) override = {};
    }
    applyState(s, false);
  } catch {
    setOffline(true);
  }
}

function liveImagine(now: number): void {
  if (!state || state.status !== "flying" || predicting || awaitingConfig || drag) return;
  if (now < freezeUntil || now - lastLiveAt < LIVE_MS) return;
  predicting = true;
  const e = epoch;
  api
    .predict()
    .then((r) => {
      if (e !== epoch || awaitingConfig || !state || state.status !== "flying") return;
      if (r.config_version !== state.config_version) return;
      lastLiveAt = performance.now();
      setShown({ prediction: r.prediction, revealStart: -1e9, revealMs: 1 });
    })
    .catch(() => undefined)
    .finally(() => (predicting = false));
}

async function connect(): Promise<void> {
  if (connecting) return;
  connecting = true;
  try {
    const s = await api.state();
    setOffline(false);
    applyState(s, false);
    syncControls();
    renderer.resize(s.observation.width, s.observation.height);
    $("loading").hidden = true;
    if (!evaluated) void runEvaluation();
  } catch {
    setOffline(true);
  } finally {
    connecting = false;
  }
}

async function tick(): Promise<void> {
  if (document.hidden) return;
  const now = performance.now();
  if (!state) return void connect();
  if (finished()) {
    if (!paused && now - resultAt > RESULT_MS) void nextFlight();
    return;
  }
  if (paused || now < freezeUntil) return;
  if (!stepping) {
    stepping = true;
    try {
      const s = await api.step(1);
      setOffline(false);
      applyState(s, true);
    } catch {
      setOffline(true);
    } finally {
      stepping = false;
    }
  }
  liveImagine(performance.now());
}

async function nextFlight(): Promise<void> {
  if (advancing) return;
  advancing = true;
  try {
    applyState(await api.next(), false);
  } catch {
    setOffline(true);
  } finally {
    advancing = false;
  }
}

async function restart(newScenario: boolean): Promise<void> {
  try {
    const s = await api.reset(newScenario);
    override = {};
    selected = null;
    applyState(s, false);
    syncControls();
  } catch {
    setOffline(true);
  }
}

function setOffline(value: boolean): void {
  if (offline === value) return;
  offline = value;
  const banner = $("banner");
  banner.hidden = !value;
  banner.textContent = value ? "Lost contact with the simulation API. Retrying…" : "";
  if (value) {
    state = null; // resynchronise from scratch once the API answers again
    shown = null;
  }
  renderPanel();
}

// ---- panels -----------------------------------------------------------------

function text(id: string, value: string): void {
  const el = $(id);
  if (el.textContent !== value) el.textContent = value;
}

function setPill(el: HTMLElement, label: string, tone: "" | "muted" | "busy" | "gold" | "bad"): void {
  el.textContent = label;
  el.className = `pill ${tone}`.trim();
}

function renderPanel(): void {
  const now = performance.now();
  const status = $("model-status");
  canvas.dataset.status = offline ? "offline" : (state?.status ?? "connecting");
  canvas.dataset.predictionSerial = String(predictionSerial);
  canvas.dataset.futures = String(shown?.prediction.futures.length ?? 0);
  if (offline || !state) {
    setPill(status, offline ? "Offline" : "Connecting", offline ? "bad" : "muted");
    return;
  }
  const o = state.observation;
  if (paused) setPill(status, "Paused", "muted");
  else if (finished()) setPill(status, "Scored", "gold");
  else if (state.status === "observing") setPill(status, "Observing", "muted");
  else if (now < freezeUntil) setPill(status, "Imagining 20 futures", "busy");
  else if (!shown) setPill(status, "Re-imagining…", "busy");
  else setPill(status, "Predicting live", "");

  const p = shown?.prediction ?? null;
  text("futures-count", p ? String(p.n_futures) : "–");
  text("live-top", p ? outcomeLabel(p.top_outcome) : "–");
  text("live-confidence", p ? pct(p.confidence) : "–");
  const c = state.committed;
  text("committed-text", c ? `${outcomeLabel(c.top_outcome)} · ${pct(c.confidence)}` : "after 1 s of flight");
  text("horizon-left", state.status === "flying" ? `${(state.steps_remaining * o.dt).toFixed(1)} s` : "–");
  const b = o.butterfly;
  text("butterfly-text", `(${b.x.toFixed(0)}, ${b.y.toFixed(0)}) · ${Math.hypot(b.vx, b.vy).toFixed(1)} u/s`);
  text("wind-text", `${o.wind.strength.toFixed(1)} u/s toward ${o.wind.direction_deg.toFixed(0)}°`);
  text("session-text", `${state.session.correct} of ${state.session.cycles} correct`);
  text("seed-text", String(state.seed));

  renderProbabilities(p);
  const spread = p?.uncertainty.spread_over_time ?? [];
  const peak = Math.max(...spread, 1);
  const pts = spread.map((v, i) => `${((i / Math.max(spread.length - 1, 1)) * 200).toFixed(1)},${(38 - (v / peak) * 34).toFixed(1)}`);
  $("spread-line").setAttribute("points", pts.join(" "));

  // The story strip: 1 butterfly -> 20 futures -> 1 predicted landing -> 1 actual landing.
  $("story-futures").classList.toggle("on", !!p);
  const predictedOn = !!c && now >= committedVisibleAt;
  $("story-predicted").classList.toggle("on", predictedOn);
  text("story-predicted-text", predictedOn ? `predicted: ${outcomeLabel(c!.top_outcome)} · ${pct(c!.confidence)}` : "predicted landing");
  const r = state.result;
  $("story-actual").classList.toggle("on", !!r);
  $("story-actual").classList.toggle("miss", !!r && !r.correct);
  text("story-actual-text", r ? `actual: ${outcomeLabel(r.actual)} ${r.correct ? "✓" : "✗"}` : "actual landing");
}

function renderProbabilities(p: Prediction | null): void {
  const list = $("prob-list");
  if (!p) {
    if (list.dataset.key !== "") {
      list.dataset.key = "";
      list.innerHTML = '<li><span class="empty">Waiting for the next prediction…</span></li>';
    }
    return;
  }
  const outcomes = Object.keys(p.probabilities);
  const key = outcomes.join(",");
  if (list.dataset.key !== key) {
    list.dataset.key = key;
    list.replaceChildren(
      ...outcomes.map((o) => {
        const li = document.createElement("li");
        li.dataset.outcome = o;
        li.innerHTML = '<span class="name"></span><span class="track"><span class="fill"></span></span><span class="value"></span>';
        li.querySelector(".name")!.textContent = outcomeLabel(o);
        return li;
      }),
    );
  }
  for (const li of Array.from(list.children) as HTMLElement[]) {
    const o = li.dataset.outcome!;
    const v = p.probabilities[o] ?? 0;
    li.className = [o === p.top_outcome ? "top" : "", o === NO_LANDING ? "none" : ""].join(" ").trim();
    (li.querySelector(".fill") as HTMLElement).style.width = `${v * 100}%`;
    li.querySelector(".value")!.textContent = pct(v);
  }
}

function showResult(s: WorldState): void {
  const r = s.result!;
  const verdict = $("result-verdict");
  verdict.textContent = r.correct ? "Correct prediction" : "Incorrect prediction";
  verdict.classList.toggle("miss", !r.correct);
  text("result-predicted", outcomeLabel(r.predicted));
  text("result-actual", outcomeLabel(r.actual));
  text("result-confidence", pct(r.confidence));
  text("result-error", r.position_error === null ? "not defined" : `${r.position_error.toFixed(1)} units`);
  // Keep the card out of the way: put it in the quadrant opposite the butterfly.
  const b = s.observation.butterfly;
  $("result").classList.toggle("right", b.x < s.observation.width / 2);
  $("result").classList.toggle("bottom", b.y > s.observation.height / 2);
  $("result").hidden = false;
  const bar = $("countdown-bar");
  bar.classList.remove("run");
  void bar.offsetWidth; // restart the CSS animation
  bar.style.setProperty("--countdown", `${RESULT_MS}ms`);
  bar.classList.toggle("run", !paused);
}

function syncControls(): void {
  if (!state) return;
  const o = state.observation;
  $<HTMLInputElement>("wind-strength").value = String(o.wind.strength);
  $<HTMLInputElement>("wind-direction").value = String(o.wind.direction_deg);
  $<HTMLInputElement>("randomness").value = String(o.randomness);
  text("out-wind-strength", o.wind.strength.toFixed(1));
  text("out-wind-direction", `${o.wind.direction_deg.toFixed(0)}°`);
  text("out-randomness", o.randomness.toFixed(2));
}

async function runEvaluation(): Promise<void> {
  evaluated = true;
  const button = $<HTMLButtonElement>("btn-eval");
  const status = $("eval-status");
  button.disabled = true;
  setPill(status, "Running 500 scenarios…", "busy");
  try {
    renderEvaluation(await api.evaluate());
    setPill(status, "Complete", "gold");
  } catch {
    setPill(status, "Failed", "bad");
    text("eval-note", "The evaluation request failed. Check that the API is running and try again.");
  } finally {
    button.disabled = false;
    button.textContent = "Re-run 500-scenario evaluation";
  }
}

function renderEvaluation(m: Metrics): void {
  const fmtErr = (v: number | null) => (v === null ? "–" : v.toFixed(1));
  text("eval-n", String(m.n_scenarios));
  text("eval-acc", `${(m.model.top1_accuracy * 100).toFixed(1)}%`);
  text("eval-brier", m.model.brier_score.toFixed(3));
  text("eval-error", `${fmtErr(m.model.landing_position_error.mean)} u`);
  const rows: [string, string, typeof m.model][] = [
    ["model", "World model (20 futures)", m.model],
    ["", "Nearest flower", m.baselines.nearest_flower],
    ["", "Constant velocity", m.baselines.constant_velocity],
  ];
  $("eval-rows").replaceChildren(
    ...rows.map(([cls, name, s]) => {
      const tr = document.createElement("tr");
      tr.className = cls;
      for (const cell of [name, `${(s.top1_accuracy * 100).toFixed(1)}%`, s.brier_score.toFixed(3), fmtErr(s.landing_position_error.mean)]) {
        tr.appendChild(document.createElement("td")).textContent = cell;
      }
      return tr;
    }),
  );
  const none = m.actual_outcomes[NO_LANDING] ?? 0;
  const ece = m.model.calibration.expected_calibration_error;
  text(
    "eval-note",
    `Brier: lower is better (0–2). Landing error is defined in ${pct(m.model.landing_position_error.coverage)} of scenarios. ` +
      `${none} of ${m.n_scenarios} flights did not land within ${m.horizon_seconds.toFixed(0)} s. ` +
      `Calibration error ${ece === null ? "–" : ece.toFixed(3)}; prediction latency p50 ${m.model.latency.p50_ms.toFixed(1)} ms, ` +
      `p99 ${m.model.latency.p99_ms.toFixed(1)} ms.`,
  );
}

// ---- drawing ----------------------------------------------------------------

function buildScene(now: number): Scene | null {
  if (!state || !curr || !prev) return null;
  const o = state.observation;
  const landed = state.status === "landed";
  const a = clamp((now - tickAt) / TICK_MS, 0, 1);
  const x = lerp(prev.x, curr.x, a);
  const y = lerp(prev.y, curr.y, a);
  const vx = lerp(prev.vx, curr.vx, a);
  const vy = lerp(prev.vy, curr.vy, a);
  if (Math.hypot(vx, vy) > 0.4) heading = lerpAngle(heading, Math.atan2(vy, vx), 0.18);

  const p = shown?.prediction ?? null;
  const step = lerp(prev.step, curr.step, a);
  const committed = state.committed && now >= committedVisibleAt ? state.committed : null;
  return {
    world: { width: o.width, height: o.height },
    flowers: flowers(),
    butterfly: { x, y, heading, landed },
    trail: finished() ? trail : [...trail.slice(0, -1), [x, y]],
    wind: { ...o.wind, max: MAX_WIND },
    futures: p ? p.futures : null,
    futuresFrom: p && !finished() ? Math.max(0, (step - p.observation_step) / p.path_stride) : 0,
    reveal: shown ? clamp((now - shown.revealStart) / shown.revealMs, 0, 1) : 0,
    futuresAlpha: finished() ? 0.6 : 1,
    topOutcome: p?.top_outcome ?? null,
    probabilities: p?.probabilities ?? null,
    committedFlower: committed && committed.top_outcome !== NO_LANDING ? committed.top_outcome : null,
    committedPoint: committed && committed.top_outcome !== NO_LANDING ? committed.predicted_landing_position : null,
    actualPoint: state.result?.actual_position ?? null,
    landedAt: resultAt,
    selectedFlower: drag?.id ?? selected,
    showFutures,
  };
}

function frame(now: number): void {
  if (document.hidden) return; // restarted by visibilitychange
  // While paused nothing moves, so only repaint when something actually changed
  // or while freshly imagined futures are still unfurling.
  const unfurling = shown !== null && now - shown.revealStart < shown.revealMs + 100;
  if (!paused || dirty || unfurling) {
    const scene = buildScene(now);
    if (scene) renderer.draw(scene, now);
    dirty = false;
  }
  requestAnimationFrame(frame);
}

// ---- interaction ------------------------------------------------------------

function pointerWorld(ev: PointerEvent): Pt {
  const rect = canvas.getBoundingClientRect();
  return renderer.transform.toWorld(ev.clientX - rect.left, ev.clientY - rect.top);
}

function flowerAt(ev: PointerEvent): Flower | undefined {
  const [wx, wy] = pointerWorld(ev);
  return flowers().find((f) => Math.hypot(f.x - wx, f.y - wy) <= f.radius * 1.6);
}

function moveFlower(id: string, x: number, y: number, immediately = false): void {
  if (!state) return;
  const f = flowers().find((fl) => fl.id === id);
  if (!f) return;
  const [cx, cy] = clampFlower(x, y, f.radius, state.observation.width, state.observation.height);
  override[id] = { x: cx, y: cy };
  invalidate();
  queueConfig({ flowers: [{ id, x: cx, y: cy }] }, immediately);
}

canvas.addEventListener("pointerdown", (ev) => {
  const f = flowerAt(ev);
  if (!f) return;
  drag = { id: f.id };
  selected = f.id;
  canvas.setPointerCapture(ev.pointerId);
  canvas.style.cursor = "grabbing";
  dirty = true;
});

canvas.addEventListener("pointermove", (ev) => {
  if (!drag) {
    canvas.style.cursor = flowerAt(ev) ? "grab" : "default";
    return;
  }
  const [wx, wy] = pointerWorld(ev);
  moveFlower(drag.id, wx, wy);
});

const endDrag = (ev: PointerEvent) => {
  if (!drag) return;
  const [wx, wy] = pointerWorld(ev);
  const id = drag.id;
  drag = null;
  canvas.style.cursor = "grab";
  moveFlower(id, wx, wy, true);
};
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);

canvas.addEventListener("keydown", (ev) => {
  const all = flowers();
  if (!all.length) return;
  if (ev.key.toLowerCase() === "n") {
    const i = all.findIndex((f) => f.id === selected);
    selected = all[(i + 1) % all.length].id;
    dirty = true;
    return;
  }
  const delta: Record<string, Pt> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
  const d = delta[ev.key];
  const f = all.find((fl) => fl.id === selected);
  if (!d || !f) return;
  ev.preventDefault();
  const k = ev.shiftKey ? 4 : 1.5;
  moveFlower(f.id, f.x + d[0] * k, f.y + d[1] * k);
});

function bindSlider(id: string, out: string, format: (v: number) => string, key: keyof ConfigChange): void {
  const input = $<HTMLInputElement>(id);
  input.addEventListener("input", () => {
    const v = Number(input.value);
    text(out, format(v));
    invalidate();
    queueConfig({ [key]: v });
  });
}
bindSlider("wind-strength", "out-wind-strength", (v) => v.toFixed(1), "wind_strength");
bindSlider("wind-direction", "out-wind-direction", (v) => `${v.toFixed(0)}°`, "wind_direction_deg");
bindSlider("randomness", "out-randomness", (v) => v.toFixed(2), "randomness");

$("btn-pause").addEventListener("click", () => {
  paused = !paused;
  const button = $("btn-pause");
  button.textContent = paused ? "Resume" : "Pause";
  button.setAttribute("aria-pressed", String(paused));
  if (!paused) {
    tickAt = performance.now();
    if (finished()) resultAt = performance.now();
  }
  $("countdown-bar").classList.toggle("run", !paused && finished());
  renderPanel();
  dirty = true;
});
$("btn-reset").addEventListener("click", () => void restart(false));
$("btn-new").addEventListener("click", () => void restart(true));
$("btn-next").addEventListener("click", () => void nextFlight());
$("btn-eval").addEventListener("click", () => void runEvaluation());
$<HTMLInputElement>("toggle-futures").addEventListener("change", (ev) => {
  showFutures = (ev.target as HTMLInputElement).checked;
  dirty = true;
});

new ResizeObserver(() => {
  renderer.resize(state?.observation.width ?? 100, state?.observation.height ?? 60);
  dirty = true;
}).observe(canvas);

document.addEventListener("visibilitychange", () => {
  if (document.hidden) return;
  tickAt = performance.now();
  dirty = true;
  requestAnimationFrame(frame);
});

// Small hook for the browser smoke test: where a flower is on screen.
(window as unknown as { __garden: unknown }).__garden = {
  flowerScreen(id: string): Pt | null {
    const f = flowers().find((fl) => fl.id === id);
    if (!f) return null;
    const rect = canvas.getBoundingClientRect();
    const [sx, sy] = renderer.transform.toScreen(f.x, f.y);
    return [rect.left + sx, rect.top + sy];
  },
};

window.setInterval(() => void tick(), TICK_MS);
void tick();
requestAnimationFrame(frame);
