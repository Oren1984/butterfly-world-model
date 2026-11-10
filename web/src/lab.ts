/**
 * V1.1 experiment modes: Goal-Directed Planning and Visual Planning.
 *
 * The landing-prediction mode (main.ts) stays the default. This module owns the
 * mode switch and a separate stage that replays one episode at a time: the
 * actual flight, the route the planner imagined at each replan, and the
 * intermediate waypoints or subgoals. Episodes are generated live by the API;
 * the comparison tables show recorded results from the evaluation commands.
 */
import type {
  LayoutSpec,
  Mode,
  ModesInfo,
  PlanningEpisode,
  PlanningResults,
  PlanningPlan,
  VisualEpisode,
  VisualPlan,
  VisualResults,
} from "./labTypes";
import type { Pt } from "./types";

const STEPS_PER_SECOND = 20; // replay at twice real time (dt = 0.1 s)
const CYAN = "#5ee7ff";
const VIOLET = "#a78bfa";
const GOLD = "#ffd166";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

async function call<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `${method} ${path} failed with ${res.status}`;
    try {
      const j = await res.json();
      if (typeof j.detail === "string") detail = j.detail;
    } catch {
      /* keep the generic message */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

const TEXT: Record<Exclude<Mode, "prediction">, { title: string; subtitle: string; explain: string }> = {
  planning: {
    title: "Goal-Directed Planning",
    subtitle: "HWM-inspired: flat planning vs. intermediate waypoints",
    explain:
      "The agent steers the butterfly to the pink target flower against a seeded wind. The flat planner scores " +
      "action sequences toward the target directly. The hierarchical planners first choose an intermediate waypoint " +
      "(explicitly, or from a learned long-horizon model's predictions) and then refine it into steering actions. " +
      "Every planner replans every 0.4 s from a fresh observation with the same budget of model evaluations, " +
      "waypoint generation included. Exact = the simulator's known physics with the current wind held constant; " +
      "learned = a small model trained from seeded flights.",
  },
  visual: {
    title: "Visual Planning",
    subtitle: "H-JEPA-inspired: planning in a representation learned from pixels",
    explain:
      "The planner sees only 50 × 30 pixel frames and a goal image (the butterfly resting on the flower) — never " +
      "coordinates. A one-level latent world model is compared with a jointly trained two-level model whose higher " +
      "level predicts 1.6 s ahead and supplies subgoals. Optional distractor: a bright blob that wanders on its own. " +
      "The cyan route is the planner's imagined latents decoded by a frozen diagnostic probe, for display only; " +
      "the flight path and outcome come from the simulator, used only for scoring.",
  },
};

let mode: Mode = "prediction";
let info: ModesInfo | null = null;
let episode: PlanningEpisode | VisualEpisode | null = null;
let replayStep = 0;
let playing = false;
let lastFrame = 0;
let running = false;
const canvas = $<HTMLCanvasElement>("lab-canvas");
const ctx = canvas.getContext("2d")!;

// ---- mode switch ---------------------------------------------------------------

function setMode(next: Mode, push = true): void {
  mode = next;
  document.body.dataset.mode = next;
  $("app").hidden = next !== "prediction";
  $("lab").hidden = next === "prediction";
  for (const b of Array.from(document.querySelectorAll<HTMLButtonElement>("#modes button"))) {
    b.setAttribute("aria-selected", String(b.dataset.mode === next));
  }
  // The tab bar lives at the top of whichever side panel is visible.
  const host = next === "prediction" ? document.querySelector("#app .panel") : document.querySelector("#lab .panel");
  host?.prepend($("modebar"));
  if (push) history.replaceState(null, "", next === "prediction" ? location.pathname : `#${next}`);
  if (next === "prediction") return;
  episode = null;
  $("lab-title").textContent = TEXT[next].title;
  $("lab-subtitle").textContent = TEXT[next].subtitle;
  $("lab-explain").textContent = TEXT[next].explain;
  $("lab-views").hidden = next !== "visual";
  $("row-distractor").hidden = next !== "visual";
  $("row-model-seed").hidden = next !== "visual";
  clearResult();
  resize();
  void enter(next);
}

async function enter(next: Mode): Promise<void> {
  try {
    info = info ?? (await call<ModesInfo>("GET", "/api/modes"));
  } catch {
    setStatus("run-status", "API unavailable", "bad");
    return;
  }
  if (mode !== next) return;
  fillControls();
  void loadRecorded();
  void run();
}

function fillControls(): void {
  if (!info) return;
  const layout = $<HTMLSelectElement>("sel-layout");
  const planner = $<HTMLSelectElement>("sel-planner");
  const opts = (sel: HTMLSelectElement, items: [string, string][], keep?: string) => {
    sel.replaceChildren(...items.map(([v, label]) => new Option(label, v)));
    if (keep && items.some(([v]) => v === keep)) sel.value = keep;
  };
  if (mode === "planning") {
    const p = info.planning;
    opts(layout, Object.values(p.layouts).map((l) => [l.name, layoutLabel(l.name)]), "detour");
    opts(planner, Object.entries(p.planners).map(([k, v]) => [k, v.label]), "waypoint_exact");
    $("lab-model-note").textContent = p.learned_model.available
      ? `Learned model ${p.learned_model.model_version}.`
      : `Learned model not available: ${p.learned_model.message}`;
  } else {
    const v = info.visual;
    opts(layout, v.layouts.map((l) => [l, layoutLabel(l)]), "direct");
    opts(planner, [["two_level", "Two-level (H-JEPA-inspired)"], ["one_level", "One-level latent planner"]], "two_level");
    const seeds = [...new Set(v.models.models.map((m) => m.model_seed))];
    opts($<HTMLSelectElement>("sel-model-seed"), (seeds.length ? seeds : [0]).map((s) => [String(s), `training seed ${s}`]));
    $("lab-model-note").textContent = v.models.available
      ? `${v.models.models.length} trained models available.`
      : `No trained visual model: ${v.models.message ?? "run python -m experiments.train_visual"}`;
  }
}

function layoutLabel(name: string): string {
  return { direct: "Direct route", detour: "Detour route (hedge)", enclosed: "Unreachable (walled in)" }[name] ?? name;
}

// ---- running an episode ------------------------------------------------------------

async function run(): Promise<void> {
  if (running || mode === "prediction") return;
  running = true;
  const button = $<HTMLButtonElement>("btn-run");
  button.disabled = true;
  setStatus("run-status", "Planning…", "busy");
  const seed = Math.max(0, Math.floor(Number($<HTMLInputElement>("inp-seed").value) || 0));
  const layout = $<HTMLSelectElement>("sel-layout").value;
  const planner = $<HTMLSelectElement>("sel-planner").value;
  const forMode = mode;
  try {
    const ep =
      forMode === "planning"
        ? await call<PlanningEpisode>("POST", "/api/planning/episode", { layout, planner, seed })
        : await call<VisualEpisode>("POST", "/api/visual/episode", {
            layout,
            variant: planner,
            seed,
            distractor: $<HTMLInputElement>("chk-distractor").checked,
            model_seed: Number($<HTMLSelectElement>("sel-model-seed").value || 0),
          });
    if (mode !== forMode) return;
    episode = ep;
    replayStep = 0;
    playing = true;
    lastFrame = performance.now();
    $<HTMLInputElement>("inp-scrub").max = String(ep.path.length - 1);
    if (forMode === "visual") drawFrame("lab-goal", (ep as VisualEpisode).goal_frame, (ep as VisualEpisode).frame_size);
    renderResult(ep);
    setStatus("run-status", "Generated live", "");
  } catch (e) {
    episode = null;
    clearResult();
    setStatus("run-status", "Not available", "bad");
    $("res-message").textContent = (e as Error).message;
  } finally {
    running = false;
    button.disabled = false;
    canvas.dataset.episode = episode ? `${episode.layout}:${episode.seed}:${episode.outcome}` : "";
  }
}

function setStatus(id: string, label: string, tone: "" | "busy" | "bad" | "gold" | "muted"): void {
  const el = $(id);
  el.textContent = label;
  el.className = `pill ${tone}`.trim();
}

function clearResult(): void {
  for (const id of ["res-outcome", "res-time", "res-replans", "res-evals", "res-latency", "res-model"]) $(id).textContent = "–";
  $("res-message").textContent = "";
}

function renderResult(ep: PlanningEpisode | VisualEpisode): void {
  const outcome = { landed: "Landed on target", collided: "Hit an obstacle", timeout: "Did not land in time", flying: "–" }[ep.outcome];
  $("res-outcome").textContent = outcome;
  $("res-outcome").className = ep.landed ? "good" : "bad";
  $("res-time").textContent = ep.seconds_to_land === null ? "–" : `${ep.seconds_to_land.toFixed(1)} s`;
  $("res-replans").textContent = String(ep.replans);
  $("res-evals").textContent = `${ep.evaluations_per_replan} / replan · ${ep.evaluations_total.toLocaleString()} total`;
  $("res-latency").textContent = ep.latency ? `${ep.latency.p50_ms.toFixed(1)} ms` : "–";
  $("res-model").textContent = ep.model_version;
  const extra: string[] = [];
  if ("model_flags" in ep && ep.model_flags.length) extra.push(`Training flags: ${ep.model_flags.join("; ")}`);
  if (!ep.layout_spec.reachable) extra.push("This target is walled in on purpose: failing safely is the expected outcome.");
  $("res-message").textContent = extra.join(" ");
}

// ---- recorded comparison -------------------------------------------------------------

const pctf = (v: number | null | undefined) => (v === null || v === undefined ? "–" : `${Math.round(v * 100)}%`);

function table(caption: string, head: string[], rows: string[][]): HTMLTableElement {
  const t = document.createElement("table");
  t.className = "compare";
  t.createCaption().textContent = caption;
  const tr = t.createTHead().insertRow();
  for (const h of head) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = h;
    tr.appendChild(th);
  }
  const body = t.createTBody();
  for (const r of rows) {
    const row = body.insertRow();
    for (const c of r) row.insertCell().textContent = c;
  }
  return t;
}

async function loadRecorded(): Promise<void> {
  const box = $("rec-body");
  const forMode = mode;
  setStatus("rec-status", "Loading…", "busy");
  try {
    if (forMode === "planning") {
      const r = await call<PlanningResults>("GET", "/api/planning/results");
      if (mode !== forMode) return;
      const short: Record<string, string> = {
        flat_exact: "Flat",
        waypoint_exact: "Waypoints",
        flat_learned: "Flat",
        waypoint_learned: "Waypoints",
        subgoal_learned: "Learned subgoals",
      };
      const rows = (g: Record<string, Record<string, import("./labTypes").PlannerSummary>>) =>
        Object.keys(g.detour).map((p) => {
          const d = g.direct[p];
          const t = g.detour[p];
          return [short[p], pctf(d.landing_rate), pctf(t.landing_rate), pctf(t.collision_rate),
            t.mean_seconds_to_land === null ? "–" : `${t.mean_seconds_to_land.toFixed(1)} s`,
            `${t.planning_latency_ms_per_replan.p50.toFixed(1)} ms`];
        });
      const head = ["Planner", "Direct", "Detour", "Coll.", "Time", "Latency"];
      const parts: HTMLElement[] = [table("Exact dynamics — landing rate", head, rows(r.exact_dynamics))];
      if (r.learned_model) parts.push(table("Learned model — landing rate", head, rows(r.learned_model)));
      box.replaceChildren(...parts);
      const enclosed = Object.values(r.exact_dynamics.enclosed ?? {}).map((s) => s.landing_rate);
      $("rec-note").textContent =
        `Recorded ${r.generated_utc.slice(0, 10)} · seeds ${r.seeds.first}–${r.seeds.last} per route · ` +
        `${r.config.budget_evaluations_per_replan} evaluations per replan for every planner. Time and latency are on the detour. ` +
        (enclosed.length ? `Unreachable target: landing rate ${pctf(Math.max(...enclosed))} (fails safely). ` : "") +
        (r.learned_model_version ? `Learned model ${r.learned_model_version}.` : "");
    } else {
      const r = await call<VisualResults>("GET", "/api/visual/results");
      if (mode !== forMode) return;
      const name: Record<string, string> = { one_level: "One-level", two_level: "Two-level" };
      const rows = Object.entries(r.by_variant).map(([v, g]) => [
        name[v],
        pctf(g.direct.clean.mean_landing_rate),
        pctf(g.direct.distractor.mean_landing_rate),
        pctf(g.detour.clean.mean_landing_rate),
        pctf(g.detour.distractor.mean_landing_rate),
      ]);
      const probe = Object.entries(r.probe_r2_mean_over_model_seeds).flatMap(([v, p]) =>
        Object.entries(p).map(([k, s]) => [`${name[v]} ${k}`, s.butterfly_position_r2.toFixed(2), s.distractor_position_r2.toFixed(2)]),
      );
      box.replaceChildren(
        table("Landing rate (mean over training seeds)", ["Variant", "Direct", "+ distr.", "Detour", "+ distr."], rows),
        table("Frozen probe R² (held-out trajectories)", ["Representation", "Butterfly", "Distractor"], probe),
      );
      const fails = Object.entries(r.failures);
      $("rec-note").textContent =
        `Recorded ${r.generated_utc.slice(0, 10)} · seeds ${r.seeds.first}–${r.seeds.last} per cell · training seeds ` +
        `${r.model_seeds.join(", ")} · ${r.config.budget_evaluations_per_replan} evaluations per replan. ` +
        (fails.length ? `Flagged runs: ${fails.map(([k, f]) => `${k} (${f.join("; ")})`).join(", ")}.` : "No training run was flagged.");
    }
    setStatus("rec-status", "Recorded", "gold");
  } catch (e) {
    if (mode !== forMode) return;
    box.replaceChildren();
    $("rec-note").textContent = (e as Error).message;
    setStatus("rec-status", "Not recorded", "muted");
  }
}

// ---- drawing ---------------------------------------------------------------------------

let view = { s: 1, ox: 0, oy: 0, w: 100, h: 60 };

function resize(): void {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const layout = currentLayout();
  const w = layout?.width ?? 100;
  const h = layout?.height ?? 60;
  const narrow = rect.width < 900;
  // Leave room for the planner's-view thumbnails (top right on wide screens, bottom right on narrow ones).
  const top = rect.width < 560 ? 64 : mode === "visual" && !narrow ? 128 : 92;
  const bottom = (rect.width < 560 ? 56 : 70) + (mode === "visual" && narrow ? 70 : 0);
  const s = Math.min((rect.width - 32) / w, (rect.height - top - bottom) / h);
  view = { s, ox: (rect.width - w * s) / 2, oy: top + (rect.height - top - bottom - h * s) / 2, w, h };
}

const sx = (x: number) => view.ox + x * view.s;
const sy = (y: number) => view.oy + (view.h - y) * view.s;

function currentLayout(): LayoutSpec | null {
  if (episode) return episode.layout_spec;
  if (!info) return null;
  const name = ($<HTMLSelectElement>("sel-layout").value || "detour") as string;
  return info.planning.layouts[name] ?? null;
}

function line(pts: Pt[], color: string, width: number, dash: number[] = [], alpha = 1): void {
  if (pts.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = ctx.lineCap = "round";
  ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(sx(pts[0][0]), sy(pts[0][1]));
  for (const [x, y] of pts.slice(1)) ctx.lineTo(sx(x), sy(y));
  ctx.stroke();
  ctx.restore();
}

function ring(x: number, y: number, r: number, color: string, label?: string): void {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(sx(x), sy(y), r, 0, Math.PI * 2);
  ctx.stroke();
  if (label) {
    ctx.fillStyle = color;
    ctx.font = "600 11px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(label, sx(x), sy(y) - r - 5);
  }
  ctx.restore();
}

function drawGarden(layout: LayoutSpec): void {
  const { s } = view;
  ctx.fillStyle = "#0b1433";
  ctx.strokeStyle = "rgba(160,180,255,0.18)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(sx(0), sy(layout.height), layout.width * s, layout.height * s, 10);
  ctx.fill();
  ctx.stroke();
  for (const r of layout.obstacles) {
    ctx.fillStyle = "#24533f";
    ctx.strokeStyle = "#4f9b72";
    ctx.beginPath();
    ctx.roundRect(sx(r.x0), sy(r.y1), (r.x1 - r.x0) * s, (r.y1 - r.y0) * s, 4);
    ctx.fill();
    ctx.stroke();
  }
  const t = layout.target;
  ctx.save();
  ctx.translate(sx(t.x), sy(t.y));
  for (let i = 0; i < 6; i++) {
    ctx.rotate(Math.PI / 3);
    ctx.fillStyle = "#ff8fc2";
    ctx.beginPath();
    ctx.ellipse(t.radius * s * 0.55, 0, t.radius * s * 0.5, t.radius * s * 0.28, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = GOLD;
  ctx.beginPath();
  ctx.arc(0, 0, t.radius * s * 0.28, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  ctx.fillStyle = "rgba(232,238,255,0.55)";
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("target", sx(t.x), sy(t.y) + t.radius * s + 14);
  ctx.fillText("start", sx(layout.start[0]), sy(layout.start[1]) + 22);
}

function drawButterfly(x: number, y: number, heading: number): void {
  ctx.save();
  ctx.translate(sx(x), sy(y));
  ctx.rotate(-heading);
  ctx.fillStyle = GOLD;
  ctx.shadowColor = GOLD;
  ctx.shadowBlur = 12;
  for (const side of [1, -1]) {
    ctx.beginPath();
    ctx.ellipse(-2, side * 6, 7, 5, side * 0.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = "#fff4cf";
  ctx.fillRect(-5, -1.2, 10, 2.4);
  ctx.restore();
}

function planAt(step: number): PlanningPlan | VisualPlan | null {
  if (!episode) return null;
  let found: PlanningPlan | VisualPlan | null = null;
  for (const p of episode.plans as (PlanningPlan | VisualPlan)[]) {
    const at = "sim_step" in p ? p.sim_step : p.decision * 2;
    if (at <= step) found = p;
    else break;
  }
  return found;
}

function drawFrame(id: string, b64: string, size: [number, number]): void {
  const c = $<HTMLCanvasElement>(id);
  const [h, w] = size;
  c.width = w;
  c.height = h;
  const raw = atob(b64);
  const img = c.getContext("2d")!.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = raw.charCodeAt(i);
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = Math.min(255, v + 18);
    img.data[i * 4 + 3] = 255;
  }
  c.getContext("2d")!.putImageData(img, 0, 0);
}

let shownFrame: VisualPlan | null = null;

function draw(now: number): void {
  requestAnimationFrame(draw);
  if (mode === "prediction" || document.hidden) return;
  // rAF timestamps can be slightly earlier than a performance.now() taken in the same frame.
  const dt = Math.max(0, (now - lastFrame) / 1000);
  lastFrame = now;
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const layout = currentLayout();
  if (!layout) return;
  if (view.w !== layout.width) resize();
  drawGarden(layout);
  if (!episode) return;
  const last = episode.path.length - 1;
  if (playing) {
    replayStep = Math.min(last, replayStep + dt * STEPS_PER_SECOND);
    if (replayStep >= last) playing = false;
    $<HTMLInputElement>("inp-scrub").value = String(Math.floor(replayStep));
    $("btn-replay").textContent = playing ? "Pause" : "Replay";
  }
  const i = Math.min(last, Math.max(0, Math.floor(replayStep)));
  const plan = planAt(i);
  if (plan && "route" in plan) {
    if (plan.high_route) line(plan.high_route, VIOLET, 1.6, [5, 5], 0.75);
    line(plan.route, CYAN, 2.2, [], 0.9);
    if (plan.waypoint) ring(plan.waypoint[0], plan.waypoint[1], 7, VIOLET, episode.planner.startsWith("subgoal") ? "subgoal" : "waypoint");
  } else if (plan && "frame" in plan) {
    if (plan.imagined_route) line(plan.imagined_route, CYAN, 2, [3, 4], 0.85);
    if (plan.subgoal_decoded) ring(plan.subgoal_decoded[0], plan.subgoal_decoded[1], 7, VIOLET, "subgoal");
    if (plan !== shownFrame) {
      shownFrame = plan;
      drawFrame("lab-frame", plan.frame, (episode as VisualEpisode).frame_size);
    }
  }
  if ("distractor_path" in episode && episode.distractor_path.length) {
    const d = episode.distractor_path[Math.min(Math.floor(i / 2), episode.distractor_path.length - 1)];
    const g = ctx.createRadialGradient(sx(d[0]), sy(d[1]), 0, sx(d[0]), sy(d[1]), 3.2 * view.s);
    g.addColorStop(0, "rgba(255,255,255,0.85)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(sx(d[0]), sy(d[1]), 3.2 * view.s, 0, Math.PI * 2);
    ctx.fill();
  }
  line(episode.path.slice(0, i + 1), GOLD, 2.6, [], 0.95);
  const [x, y] = episode.path[i];
  const [px, py] = episode.path[Math.max(0, i - 2)];
  drawButterfly(x, y, Math.atan2(y - py, x - px));
  if (i === last) {
    ctx.fillStyle = episode.landed ? GOLD : "#ff8a8a";
    ctx.font = "600 13px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(episode.landed ? `landed in ${episode.seconds_to_land?.toFixed(1)} s` : episode.outcome, sx(x), sy(y) - 16);
  }
}

// ---- wiring ------------------------------------------------------------------------------

for (const b of Array.from(document.querySelectorAll<HTMLButtonElement>("#modes button"))) {
  b.addEventListener("click", () => setMode(b.dataset.mode as Mode));
}
$("btn-run").addEventListener("click", () => void run());
$("btn-replay").addEventListener("click", () => {
  if (!episode) return;
  if (!playing && replayStep >= episode.path.length - 1) replayStep = 0;
  playing = !playing;
  lastFrame = performance.now();
});
$<HTMLInputElement>("inp-scrub").addEventListener("input", (ev) => {
  playing = false;
  replayStep = Number((ev.target as HTMLInputElement).value);
  $("btn-replay").textContent = "Replay";
});
$("sel-layout").addEventListener("change", () => resize());
new ResizeObserver(() => resize()).observe(canvas);

const initial = location.hash.slice(1);
setMode(initial === "planning" || initial === "visual" ? initial : "prediction", false);
requestAnimationFrame(draw);
