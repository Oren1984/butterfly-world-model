// Records the showcase video from the running application and assembles it with ffmpeg.
//
//   docker compose up -d            (application at http://localhost:8080)
//   cd web && MUSIC=/path/to/the_budding_of_consciousness.wav node e2e/record-showcase.mjs
//
// MUSIC is optional; without it the video is silent. The track is not stored in the
// repository; see docs/showcase-video.md for its source and licence.
//
// Everything on screen is the real application talking to the real API. The script only drives
// it (clicks, drags) and lays titles, captions and a cursor marker on top.
//
// Capture is frame-stepped rather than real-time: the page's clock is advanced by exactly
// 1/30 s per frame and a screenshot is taken, so the video is smooth regardless of how fast
// this machine can render. One second of video is one second of application time.
// See docs/showcase-video.md.
import { chromium } from "@playwright/test";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const BASE = process.env.BASE_URL ?? "http://localhost:8080";
const OUT_DIR = path.resolve("../docs/video");
const NAME = "butterfly-world-model-showcase";
const SEED = 7;
const frameDir = fs.mkdtempSync(path.join(os.tmpdir(), "bwm-showcase-"));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const HE_OUTCOME = (label) => (label.startsWith("Flower ") ? `פרח ${label.slice(7)}` : "ללא נחיתה");

// English on-screen text and the Hebrew subtitle for each beat of the story.
const TEXT = {
  intro: { he: () => "מודל עולם של פרפר — חיזוי הנחיתה הבאה" },
  garden1: {
    en: () => "A butterfly crosses a simulated garden, drawn to flowers and pushed by wind.",
    he: () => "פרפר חוצה גן מדומה: הוא נמשך אל הפרחים, והרוח דוחפת אותו.",
  },
  garden2: {
    en: () => "Its intent, the gusts and every random draw stay hidden from the predictor.",
    he: () => "הכוונה שלו, משבי הרוח וכל הגרלה אקראית נסתרים מן המודל החוזה.",
  },
  imagine1: {
    en: () => "The model watches one second of flight…",
    he: () => "המודל צופה בשנייה אחת של תעופה…",
  },
  imagine2: {
    en: () => "Time pauses. The model imagines 20 possible futures.",
    he: () => "הזמן נעצר. המודל מדמיין 20 עתידים אפשריים.",
  },
  imagine3: {
    en: () => "Cyan: the most likely flower.  Violet: other flowers.  Dashed: no landing in time.",
    he: () => "תכלת: הפרח הסביר ביותר. סגול: פרחים אחרים. מקווקו: אין נחיתה בזמן.",
  },
  imagine4: {
    en: (v) => `The prediction is frozen before the flight continues: ${v.flower} · ${v.confidence}`,
    he: (v) => `התחזית ננעלת לפני שהתעופה נמשכת: ${HE_OUTCOME(v.flower)} · ${v.confidence}`,
  },
  reality1: {
    en: () => "The real flight continues, on its own separate random stream.",
    he: () => "התעופה האמיתית נמשכת, עם אקראיות נפרדת משלה.",
  },
  reality2: {
    en: (v) => `Scored against reality: ${v.verdict.toLowerCase()} · confidence ${v.confidence} · landing error ${v.error}`,
    he: (v) =>
      `השוואה למציאות: ${v.verdict.startsWith("Correct") ? "תחזית נכונה" : "תחזית שגויה"}\nביטחון ${v.confidence} · שגיאת נחיתה ${
        v.error === "not defined" ? "לא מוגדרת" : v.error.replace("units", "יחידות")
      }`,
  },
  explore1: {
    en: () => "Change the world: turn the wind…",
    he: () => "משנים את העולם: מסובבים את הרוח…",
  },
  explore2: {
    en: () => "…or move a flower. The old prediction is discarded and imagined again.",
    he: () => "…או מזיזים פרח. התחזית הישנה נזרקת, והמודל מדמיין מחדש.",
  },
  explore3: {
    en: () => "The flight goes on under the new conditions.",
    he: () => "התעופה נמשכת בתנאים החדשים.",
  },
  result: {
    he: (v) => `${v.n} תרחישים: ${v.model} דיוק למודל העולם,\nלעומת ${v.nearest} לבסיס "הפרח הקרוב ביותר".`,
  },
  result2: { he: () => "הדגמה מבוססת סימולציה —\nלא מודל עולם נוירוני מאומן." },
  closing: { he: () => "דמיינו עתידים אפשריים. חזו מה יקרה בהמשך." },
};

const OVERLAY_CSS = `
#rec-card, #rec-stats { position: fixed; inset: 0; z-index: 1000; display: grid; place-items: center; text-align: center;
  background: radial-gradient(ellipse at 30% 25%, rgba(94,231,255,.10), transparent 55%),
              radial-gradient(ellipse at 75% 80%, rgba(167,139,250,.13), transparent 55%), #070b1d;
  opacity: 0; transition: opacity .7s ease; pointer-events: none; }
#rec-card.on, #rec-stats.on { opacity: 1; }
#rec-card h1 { margin: 0; font-size: 84px; font-weight: 650; letter-spacing: .01em;
  background: linear-gradient(90deg, #5ee7ff, #a78bfa 60%, #ffd166); -webkit-background-clip: text; color: transparent; }
#rec-card p { margin: 18px 0 0; font-size: 34px; color: #c9d3f5; }
#rec-card small { display: block; margin-top: 34px; font-size: 21px; color: #9aa6d1; letter-spacing: .04em; }
#rec-card .glow { width: 420px; height: 2px; margin: 30px auto 0; background: linear-gradient(90deg, transparent, #5ee7ff, #ffd166, transparent);
  animation: rec-sweep 3.2s ease-in-out infinite; }
@keyframes rec-sweep { 0%, 100% { transform: scaleX(.35); opacity: .5; } 50% { transform: scaleX(1); opacity: 1; } }
#rec-stats { background: rgba(7,11,29,.93); }
#rec-card:not(.on):not(.fading), #rec-stats:not(.on) { visibility: hidden; }
#rec-stats h2 { margin: 0 0 34px; font-size: 30px; font-weight: 550; color: #c9d3f5; }
#rec-stats .pair { display: flex; gap: 110px; justify-content: center; align-items: flex-end; }
#rec-stats .num { font-size: 150px; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums; }
#rec-stats .model .num { color: #5ee7ff; }
#rec-stats .base .num { color: #8791c2; font-size: 108px; }
#rec-stats .lab { margin-top: 14px; font-size: 25px; color: #e8eeff; }
#rec-stats .sub { margin-top: 40px; font-size: 24px; color: #9aa6d1; }
#rec-stats .note { margin-top: 26px; font-size: 20px; color: #ffd166; }
#rec-cap { position: fixed; bottom: 80px; z-index: 900; transform: translate(-50%, 10px); padding: 14px 26px; border-radius: 14px;
  font-size: 26px; font-weight: 500; color: #e8eeff; white-space: nowrap; background: rgba(7,11,29,.88);
  border: 1px solid rgba(94,231,255,.45); box-shadow: 0 10px 40px rgba(0,0,0,.45); opacity: 0;
  transition: opacity .45s ease, transform .45s ease; pointer-events: none; }
#rec-cap.on { opacity: 1; transform: translate(-50%, 0); }
#rec-cursor { position: fixed; z-index: 950; width: 26px; height: 26px; margin: -13px 0 0 -13px; border-radius: 50%;
  border: 2px solid #fff; background: rgba(255,255,255,.22); box-shadow: 0 0 14px rgba(255,255,255,.5);
  opacity: 0; transition: opacity .3s; pointer-events: none; }
#rec-cursor.on { opacity: 1; }
body.rec-quiet #result { visibility: hidden; }
.viewport { transition: transform 5s cubic-bezier(.4, 0, .2, 1); }
.rec-spot { outline: 2px solid #ffd166; outline-offset: 6px; border-radius: 10px; }
`;

async function main() {
  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const click = (id) => page.evaluate((i) => document.getElementById(i).click(), id);
  const textOf = (sel) => page.evaluate((q) => document.querySelector(q).textContent, sel);
  const status = () => page.evaluate(() => document.getElementById("garden").dataset.status);

  // ---- frame-stepped capture ----
  const FPS = 30;
  const writes = [];
  let frameCount = 0;
  let clockMs = 0;
  let recording = false;
  const now = () => frameCount / FPS;
  const frame = async () => {
    const target = Math.round(((frameCount + 1) * 1000) / FPS);
    await page.clock.runFor(target - clockMs);
    clockMs = target;
    // Let any API call that this slice of time triggered finish before the frame is taken.
    for (let i = 0; i < 400 && (await page.evaluate(() => window.__pending)) > 0; i++) await sleep(4);
    // CSS animations and transitions follow the same stepped clock.
    await page.evaluate((dt) => {
      const seen = (window.__anim ??= new WeakMap());
      for (const a of document.getAnimations()) {
        let t = seen.get(a);
        if (t === undefined) {
          t = 0;
          a.pause();
        }
        t += dt;
        seen.set(a, t);
        a.currentTime = t;
      }
    }, 1000 / FPS);
    const shot = await page.screenshot({ type: "jpeg", quality: 95 });
    writes.push(fs.promises.writeFile(path.join(frameDir, `f${String(frameCount).padStart(5, "0")}.jpg`), shot));
    frameCount++;
  };
  const advance = async (ms) => {
    if (!recording) return sleep(ms);
    for (let i = Math.round((ms / 1000) * FPS); i > 0; i--) await frame();
  };
  const waitFor = async (check, seconds = 20) => {
    for (let i = 0; i < seconds * FPS; i++) {
      if (await check()) return;
      await frame();
    }
    throw new Error("timed out waiting for the application");
  };
  const waitStatus = (wanted) => waitFor(async () => wanted.includes(await status()));
  const resultShown = () => page.evaluate(() => !document.getElementById("result").hidden);
  const setChecked = (id, on) =>
    page.evaluate(([i, o]) => {
      const el = document.getElementById(i);
      if (el.checked !== o) el.click();
    }, [id, on]);

  // ---- prepare: same scenario every time, evaluation finished, world paused at step 0 ----
  await page.addInitScript(() => {
    // Count API calls in flight so a frame is never taken halfway through one.
    window.__pending = 0;
    const track = (promise) => {
      window.__pending++;
      return promise.finally(() => window.__pending--);
    };
    const realFetch = window.fetch.bind(window);
    window.fetch = (...args) => track(realFetch(...args));
    const realJson = Response.prototype.json;
    Response.prototype.json = function () {
      return track(realJson.call(this));
    };
  });
  await page.clock.install();
  await page.request.post(`${BASE}/api/world/reset`, { data: { seed: SEED } });
  await page.goto(BASE);
  for (let i = 0; i < 900; i++) {
    if (await page.evaluate(() => document.getElementById("eval-status")?.textContent === "Complete")) break;
    await sleep(100);
  }
  await page.addStyleTag({ content: OVERLAY_CSS });
  await page.evaluate(() => {
    for (const id of ["rec-card", "rec-stats", "rec-cap", "rec-cursor"]) {
      const el = document.createElement("div");
      el.id = id;
      document.body.appendChild(el);
    }
    const stage = document.querySelector(".stage").getBoundingClientRect();
    document.getElementById("rec-cap").style.left = `${stage.left + stage.width / 2}px`;
    document.getElementById("rec-stats").style.right = `${window.innerWidth - stage.right}px`;
    window.addEventListener("mousemove", (ev) => {
      const c = document.getElementById("rec-cursor");
      c.style.left = `${ev.clientX}px`;
      c.style.top = `${ev.clientY}px`;
    });
  });
  const card = (html, on = true) =>
    page.evaluate(([h, o]) => {
      const el = document.getElementById("rec-card");
      if (h !== null) el.innerHTML = h;
      el.classList.toggle("on", o);
      if (!o) {
        el.classList.add("fading");
        setTimeout(() => el.classList.remove("fading"), 800);
      }
    }, [html, on]);
  const TITLE = "<div><h1>Butterfly World Model</h1>";
  await card(`${TITLE}<p>Predicting the Next Landing</p><div class="glow"></div></div>`);
  await click("btn-pause");
  await click("btn-reset");
  await setChecked("toggle-futures", false);
  await page.evaluate(() => document.body.classList.add("rec-quiet"));
  await sleep(900);

  // ---- capture: from here on, time only moves when a frame is taken ----
  await page.clock.pauseAt((await page.evaluate(() => Date.now())) + 50);
  recording = true;

  const cues = [];
  const endCue = () => {
    const last = cues[cues.length - 1];
    if (last && last.end === null) last.end = now();
  };
  const cue = (key, values = {}) => {
    endCue();
    cues.push({ key, start: now(), end: null, he: TEXT[key].he(values), en: TEXT[key].en?.(values) ?? null });
  };
  const caption = async (key, values = {}) => {
    cue(key, values);
    await page.evaluate((t) => {
      const el = document.getElementById("rec-cap");
      el.textContent = t;
      el.classList.add("on");
    }, TEXT[key].en(values));
  };
  const captionOff = async () => {
    endCue();
    await page.evaluate(() => document.getElementById("rec-cap").classList.remove("on"));
  };
  const pauseAfterReveal = async () => {
    await waitStatus(["flying"]);
    await advance(1560); // the 1.4 s unfurl has finished; the app's own freeze lasts 1.7 s
    await click("btn-pause");
  };
  const cursor = (on) => page.evaluate((o) => document.getElementById("rec-cursor").classList.toggle("on", o), on);
  const glide = async (x0, y0, x1, y1, ms) => {
    const steps = Math.max(Math.round(ms / 33), 2);
    for (let i = 1; i <= steps; i++) {
      const k = i / steps;
      const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
      await page.mouse.move(x0 + (x1 - x0) * e, y0 + (y1 - y0) * e);
      await frame();
    }
  };

  // 1. Introduction
  cue("intro");
  await advance(6200);

  // 2. The garden: futures switched off, result card hidden, flights follow one another.
  await card(null, false);
  await click("btn-pause"); // resume
  await caption("garden1");
  const gardenEnd = now() + 9.8;
  let second = false;
  while (now() < gardenEnd) {
    if (!second && gardenEnd - now() < 4.9) {
      second = true;
      await caption("garden2");
    }
    if (["landed", "no_landing"].includes(await status())) {
      await advance(900);
      await click("btn-next");
    }
    await frame();
  }
  await captionOff();

  // 3. Imagining the future: restart the scenario with the futures visible.
  await card("", true);
  await advance(750);
  await click("btn-pause");
  await click("btn-reset");
  await setChecked("toggle-futures", true);
  await page.evaluate(() => document.body.classList.remove("rec-quiet"));
  await advance(350);
  await card(null, false);
  await advance(500);
  await click("btn-pause"); // resume: one second of observation, then the snapshot
  await caption("imagine1");
  await waitStatus(["flying"]);
  await caption("imagine2");
  await advance(1560);
  await click("btn-pause"); // hold the snapshot so it can be read
  const focus = await page.evaluate(() => {
    const s = window.__garden;
    const ids = ["A", "B", "C", "D", "E", "F"].map((id) => s.flowerScreen(id)).filter(Boolean);
    const x = ids.reduce((a, p) => a + p[0], 0) / ids.length;
    const y = ids.reduce((a, p) => a + p[1], 0) / ids.length;
    return [x, y];
  });
  const fan = await page.evaluate(() => {
    const top = document.getElementById("committed-text").textContent.match(/Flower (\w)/);
    return top ? window.__garden.flowerScreen(top[1]) : null;
  });
  const origin = fan ?? focus;
  await page.evaluate(([x, y]) => {
    const v = document.querySelector(".viewport");
    v.style.transformOrigin = `${x}px ${y + 90}px`;
    v.style.transform = "scale(1.32)";
  }, origin);
  await advance(3600);
  await caption("imagine3");
  await advance(5200);
  const committed = (await textOf("#committed-text")).split(" · ");
  await caption("imagine4", { flower: committed[0], confidence: committed[1] });
  await page.evaluate(() => {
    const v = document.querySelector(".viewport");
    v.style.transitionDuration = "2.2s";
    v.style.transform = "none";
  });
  await advance(4300);

  // 4. Prediction versus reality
  await click("btn-pause"); // resume the real flight
  await caption("reality1");
  await waitFor(resultShown);
  await advance(1100);
  await click("btn-pause"); // keep the result card on screen
  await caption("reality2", {
    verdict: await textOf("#result-verdict"),
    confidence: await textOf("#result-confidence"),
    error: await textOf("#result-error"),
  });
  await advance(7200);
  await captionOff();

  // 5. Interactive exploration on the next flight
  await click("btn-next");
  await advance(300);
  await click("btn-pause"); // resume
  await pauseAfterReveal();
  await caption("explore1");
  const slider = await page.evaluate(() => {
    const r = document.getElementById("wind-direction").getBoundingClientRect();
    return { x: r.x, y: r.y, width: r.width, height: r.height };
  });
  const sliderX = (deg) => slider.x + 9 + (deg / 355) * (slider.width - 18);
  const sy = slider.y + slider.height / 2;
  const startDeg = Number(await page.evaluate(() => document.getElementById("wind-direction").value));
  const targetDeg = (startDeg + 180) % 360 > 355 ? 350 : (startDeg + 180) % 360;
  await page.mouse.move(sliderX(startDeg) - 160, sy - 120);
  await cursor(true);
  await glide(sliderX(startDeg) - 160, sy - 120, sliderX(startDeg), sy, 700);
  await page.mouse.down();
  await glide(sliderX(startDeg), sy, sliderX(targetDeg), sy, 3200);
  await page.mouse.up();
  await advance(1300);

  await caption("explore2");
  const state = await (await page.request.get(`${BASE}/api/world/state`)).json();
  const b = state.observation.butterfly;
  // Move the flower that is farthest from the butterfly to a spot a short way ahead of it.
  const far = [...state.observation.flowers].sort(
    (p, q) => Math.hypot(q.x - b.x, q.y - b.y) - Math.hypot(p.x - b.x, p.y - b.y),
  )[0];
  const from = await page.evaluate((id) => window.__garden.flowerScreen(id), far.id);
  const scale = await page.evaluate((id) => {
    const p = window.__garden.flowerScreen(id);
    return p;
  }, state.observation.flowers[0].id);
  const f0 = state.observation.flowers[0];
  const pxPerUnitX = (scale[0] - from[0]) / (f0.x - far.x || 1);
  const unit = Math.abs(pxPerUnitX) > 1 ? Math.abs(pxPerUnitX) : 14;
  const tx = Math.min(Math.max(b.x + (b.x < 50 ? 13 : -13), 8), 92);
  const ty = Math.min(Math.max(b.y + (b.y < 30 ? 9 : -9), 8), 52);
  const to = [from[0] + (tx - far.x) * unit, from[1] - (ty - far.y) * unit];
  const pos = await page.evaluate(() => {
    const c = document.getElementById("rec-cursor");
    return [parseFloat(c.style.left), parseFloat(c.style.top)];
  });
  await glide(pos[0], pos[1], from[0], from[1], 900);
  await page.mouse.down();
  await glide(from[0], from[1], to[0], to[1], 2600);
  await page.mouse.up();
  await advance(1700);
  await cursor(false);
  await click("btn-pause"); // resume under the new conditions
  await caption("explore3");
  await waitFor(resultShown);
  await advance(2600);
  await click("btn-pause");
  await captionOff();

  // 6. Research result: the real evaluation panel, and its headline numbers read from it.
  const scroll = await page.evaluate(() => {
    const section = document.querySelector("section[aria-labelledby='h-eval']");
    const panel = document.querySelector(".panel");
    section.classList.add("rec-spot");
    return [panel.scrollTop, Math.min(section.offsetTop - 16, panel.scrollHeight - panel.clientHeight)];
  });
  const numbers = await page.evaluate(() => {
    const rows = [...document.querySelectorAll("#eval-rows tr")].map((tr) => [...tr.children].map((td) => td.textContent));
    return {
      n: document.getElementById("eval-n").textContent,
      model: rows[0][1], modelBrier: rows[0][2],
      nearest: rows[1][1], nearestBrier: rows[1][2],
    };
  });
  await page.evaluate((v) => {
    const el = document.getElementById("rec-stats");
    el.innerHTML = `<div><h2>Top-1 landing accuracy on ${v.n} held-out scenarios</h2>
      <div class="pair"><div class="model"><div class="num">${v.model}</div><div class="lab">World model · 20 imagined futures</div></div>
      <div class="base"><div class="num">${v.nearest}</div><div class="lab">Nearest-flower baseline</div></div></div>
      <div class="sub">Brier score ${v.modelBrier} vs ${v.nearestBrier} (lower is better)</div>
      <div class="note">A simulation-based proof of concept with hand-written dynamics, not a trained neural world model.</div></div>`;
    el.classList.add("on");
  }, numbers);
  cue("result", numbers);
  for (let i = 1; i <= 36; i++) {
    const k = 1 - Math.pow(1 - i / 36, 3);
    await page.evaluate((top) => (document.querySelector(".panel").scrollTop = top), scroll[0] + (scroll[1] - scroll[0]) * k);
    await frame();
  }
  await advance(3600);
  cue("result2");
  await advance(4500);

  // 7. Closing
  await card(`${TITLE}<p>Imagine possible futures. Predict what happens next.</p><div class="glow"></div></div>`);
  cue("closing");
  await advance(5600);
  endCue();
  const duration = now();
  await Promise.all(writes);
  await browser.close();

  // ---- assemble ----------------------------------------------------------------------------
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const stamp = (s) => {
    const ms = Math.round(s * 1000);
    const p = (n, w = 2) => String(n).padStart(w, "0");
    return `${p(Math.floor(ms / 3600000))}:${p(Math.floor(ms / 60000) % 60)}:${p(Math.floor(ms / 1000) % 60)}.${p(ms % 1000, 3)}`;
  };
  const wrap = (text, width = 40) =>
    text
      .split("\n")
      .flatMap((part) => {
        const lines = [""];
        for (const word of part.split(" ")) {
          const last = lines[lines.length - 1];
          if (last && (last + " " + word).length > width) lines.push(word);
          else lines[lines.length - 1] = last ? `${last} ${word}` : word;
        }
        return lines;
      })
      .join("\n");
  const vtt = (lang) =>
    "WEBVTT\n\n" +
    cues
      .filter((c) => c[lang])
      .map((c, i) => `${i + 1}\n${stamp(c.start)} --> ${stamp(c.end - 0.05)} line:8% position:${c.en ? 41 : 50}% align:center\n${wrap(c[lang])}\n`)
      .join("\n");
  const vttPath = path.join(OUT_DIR, `${NAME}.he.vtt`);
  fs.writeFileSync(vttPath, vtt("he"), "utf8");
  fs.writeFileSync(path.join(frameDir, "cues.json"), JSON.stringify({ duration, cues }, null, 2));

  const mp4 = path.join(OUT_DIR, `${NAME}.mp4`);
  // Optional music bed: faded in and out, lifted slightly while the futures are on screen.
  const music = process.env.MUSIC ? path.resolve(process.env.MUSIC) : null;
  const at = (key) => cues.find((c) => c.key === key).start;
  const lift = `clip((t-${(at("imagine2") - 1).toFixed(2)})/1.5,0,1)-clip((t-${at("reality1").toFixed(2)})/2.5,0,1)`;
  const audio = music
    ? ["-i", music, "-filter_complex",
        `[1:a]atrim=0:${duration.toFixed(3)},volume='0.85+0.25*(${lift})':eval=frame,` +
          `afade=t=in:st=0:d=2.5,afade=t=out:st=${(duration - 4.3).toFixed(2)}:d=4.2,alimiter=limit=0.89[a]`,
        "-map", "0:v", "-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest",
        "-metadata", "comment=Music: The Budding of Consciousness by Yoiyami (CC0 1.0)"]
    : [];
  const ff = spawnSync(
    "ffmpeg",
    ["-y", "-loglevel", "error", "-framerate", String(FPS), "-i", "f%05d.jpg", ...audio,
      "-vf", "scale=1920:1080:in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p",
      "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-profile:v", "high",
      "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
      "-metadata", "title=Butterfly World Model - Predicting the Next Landing",
      "-movflags", "+faststart", mp4],
    { cwd: frameDir, stdio: "inherit" },
  );
  console.log(JSON.stringify({ ffmpeg: ff.status, duration: Number(duration.toFixed(2)), frames: frameCount, frameDir, mp4 }));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
