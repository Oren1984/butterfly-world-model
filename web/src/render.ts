import { clamp, easeOutCubic, makeTransform, slicePath, pct, type Transform } from "./geometry";
import { NO_LANDING, type Flower, type Future, type Pt } from "./types";

export const CYAN = "#5ee7ff";
export const VIOLET = "#a78bfa";
export const GOLD = "#ffd166";
const SLATE = "#8791c2";
const PETALS = ["#ff8fc2", "#ffa98a", "#c3a6ff", "#8fd0ff", "#86ecc9", "#ffe08a"];

export interface Scene {
  world: { width: number; height: number };
  flowers: Flower[];
  butterfly: { x: number; y: number; heading: number; landed: boolean };
  trail: Pt[];
  wind: { strength: number; direction_deg: number; max: number };
  futures: Future[] | null;
  /** Fractional index into each future's path where "now" is, so the past is not drawn. */
  futuresFrom: number;
  /** 0..1 progress of the unfurl animation. */
  reveal: number;
  futuresAlpha: number;
  topOutcome: string | null;
  probabilities: Record<string, number> | null;
  /** Flower the frozen, scored prediction points at (gold ring). */
  committedFlower: string | null;
  committedPoint: Pt | null;
  actualPoint: Pt | null;
  landedAt: number; // timestamp of the landing, for the burst
  selectedFlower: string | null;
  showFutures: boolean;
}

interface Mote {
  x: number;
  y: number;
  z: number;
  phase: number;
}

export class Renderer {
  private ctx: CanvasRenderingContext2D;
  private bg: HTMLCanvasElement = document.createElement("canvas");
  private w = 0;
  private h = 0;
  private motes: Mote[] = [];
  private lastTime = 0;
  transform: Transform = makeTransform(1, 1, 100, 60);

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
    for (let i = 0; i < 70; i++) {
      this.motes.push({ x: Math.random(), y: Math.random(), z: 0.3 + Math.random() * 0.7, phase: Math.random() * 6.28 });
    }
  }

  resize(worldW: number, worldH: number): void {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = Math.max(rect.width, 1);
    this.h = Math.max(rect.height, 1);
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.transform = makeTransform(this.w, this.h, worldW, worldH);
    this.paintBackground(dpr, worldW, worldH);
  }

  /** The static part of the scene is painted once per resize. */
  private paintBackground(dpr: number, worldW: number, worldH: number): void {
    const { w, h } = this;
    this.bg.width = this.canvas.width;
    this.bg.height = this.canvas.height;
    const c = this.bg.getContext("2d")!;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    const sky = c.createLinearGradient(0, 0, 0, h);
    sky.addColorStop(0, "#060a1c");
    sky.addColorStop(0.55, "#0a1230");
    sky.addColorStop(1, "#0d1838");
    c.fillStyle = sky;
    c.fillRect(0, 0, w, h);
    const glows: [number, number, number, string][] = [
      [0.18, 0.2, 0.55, "rgba(94,231,255,0.10)"],
      [0.85, 0.75, 0.6, "rgba(167,139,250,0.12)"],
      [0.6, 0.1, 0.4, "rgba(255,209,102,0.05)"],
    ];
    for (const [gx, gy, gr, color] of glows) {
      const g = c.createRadialGradient(gx * w, gy * h, 0, gx * w, gy * h, gr * Math.max(w, h));
      g.addColorStop(0, color);
      g.addColorStop(1, "rgba(0,0,0,0)");
      c.fillStyle = g;
      c.fillRect(0, 0, w, h);
    }
    // The world itself: a softly lit clearing with a faint measurement grid.
    const [x0, y0] = this.transform.toScreen(0, worldH);
    const [x1, y1] = this.transform.toScreen(worldW, 0);
    c.save();
    c.beginPath();
    c.roundRect(x0, y0, x1 - x0, y1 - y0, 18);
    c.fillStyle = "rgba(120,150,255,0.035)";
    c.fill();
    c.strokeStyle = "rgba(160,180,255,0.14)";
    c.lineWidth = 1;
    c.stroke();
    c.clip();
    c.strokeStyle = "rgba(160,180,255,0.045)";
    c.beginPath();
    for (let x = 10; x < worldW; x += 10) {
      const [sx] = this.transform.toScreen(x, 0);
      c.moveTo(sx, y0);
      c.lineTo(sx, y1);
    }
    for (let y = 10; y < worldH; y += 10) {
      const [, sy] = this.transform.toScreen(0, y);
      c.moveTo(x0, sy);
      c.lineTo(x1, sy);
    }
    c.stroke();
    c.restore();
  }

  draw(scene: Scene, now: number): void {
    const { ctx, w, h } = this;
    const dt = Math.min((now - this.lastTime) / 1000, 0.1);
    this.lastTime = now;
    ctx.drawImage(this.bg, 0, 0, w, h);
    this.drawMotes(scene, dt, now);
    for (const [i, f] of scene.flowers.entries()) this.drawFlower(scene, f, i, now);
    if (scene.showFutures && scene.futures) this.drawFutures(scene);
    this.drawTrail(scene);
    this.drawMarkers(scene, now);
    this.drawButterfly(scene, now);
    this.drawWind(scene);
  }

  /** Drifting pollen doubles as the wind indicator: it streaks along the wind vector. */
  private drawMotes(scene: Scene, dt: number, now: number): void {
    const { ctx, w, h } = this;
    const a = (scene.wind.direction_deg * Math.PI) / 180;
    const k = scene.wind.strength / scene.wind.max;
    const vx = Math.cos(a) * k * 0.22;
    const vy = -Math.sin(a) * k * 0.22;
    ctx.save();
    ctx.lineCap = "round";
    for (const m of this.motes) {
      m.x = (m.x + (vx * m.z + Math.sin(now / 2600 + m.phase) * 0.004) * dt + 1) % 1;
      m.y = (m.y + (vy * m.z + Math.cos(now / 3100 + m.phase) * 0.004) * dt + 1) % 1;
      const twinkle = 0.25 + 0.2 * Math.sin(now / 900 + m.phase);
      ctx.strokeStyle = `rgba(210,225,255,${(twinkle * m.z).toFixed(3)})`;
      ctx.lineWidth = 1 + m.z;
      const len = 1 + k * 26 * m.z;
      ctx.beginPath();
      ctx.moveTo(m.x * w, m.y * h);
      ctx.lineTo(m.x * w - Math.cos(a) * len, m.y * h + Math.sin(a) * len);
      ctx.stroke();
    }
    ctx.restore();
  }

  private drawFlower(scene: Scene, f: Flower, index: number, now: number): void {
    const { ctx } = this;
    const [x, y] = this.transform.toScreen(f.x, f.y);
    const r = f.radius * this.transform.scale;
    const color = PETALS[index % PETALS.length];
    const p = scene.probabilities?.[f.id] ?? 0;
    const isTop = scene.topOutcome === f.id;

    // Landing zone.
    const zone = ctx.createRadialGradient(x, y, 0, x, y, r);
    zone.addColorStop(0, "rgba(255,255,255,0.10)");
    zone.addColorStop(1, "rgba(255,255,255,0.02)");
    ctx.fillStyle = zone;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.setLineDash([4, 5]);
    ctx.strokeStyle = scene.selectedFlower === f.id ? "rgba(255,255,255,0.8)" : "rgba(200,215,255,0.35)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.setLineDash([]);

    // Petals, swaying a little.
    const sway = Math.sin(now / 1400 + index * 1.7) * 0.12;
    const petal = r * 0.62;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(sway);
    ctx.shadowColor = color;
    ctx.shadowBlur = 14;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.9;
    for (let i = 0; i < 7; i++) {
      ctx.rotate((Math.PI * 2) / 7);
      ctx.beginPath();
      ctx.ellipse(petal * 0.55, 0, petal * 0.55, petal * 0.26, 0, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#fff3c4";
    ctx.beginPath();
    ctx.arc(0, 0, petal * 0.24, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Landing probability: an arc around the zone plus a direct label.
    if (p > 0) {
      ctx.strokeStyle = isTop ? CYAN : VIOLET;
      ctx.lineWidth = 3;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.arc(x, y, r + 6, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(p, 0.999));
      ctx.stroke();
      ctx.lineCap = "butt";
    }
    if (scene.committedFlower === f.id) {
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(now / 2400);
      ctx.setLineDash([10, 7]);
      ctx.strokeStyle = GOLD;
      ctx.lineWidth = 2;
      ctx.shadowColor = GOLD;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(0, 0, r + 13, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
    ctx.font = "600 12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillStyle = "rgba(232,238,255,0.92)";
    const label = p > 0 ? `${f.id} · ${pct(p)}` : f.id;
    ctx.fillText(label, x, y + r + 24);
    if (scene.committedFlower === f.id) {
      ctx.font = "600 10px system-ui, sans-serif";
      ctx.fillStyle = GOLD;
      ctx.fillText("PREDICTED", x, y - r - 20);
    }
  }

  private drawFutures(scene: Scene): void {
    const { ctx } = this;
    const longest = Math.max(...scene.futures!.map((f) => f.path.length - 1), 1);
    const upTo = easeOutCubic(scene.reveal) * longest;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    for (const f of scene.futures!) {
      const none = f.outcome === NO_LANDING;
      const color = none ? SLATE : f.outcome === scene.topOutcome ? CYAN : VIOLET;
      const pts = slicePath(f.path, scene.futuresFrom, upTo).map(([x, y]) => this.transform.toScreen(x, y));
      if (pts.length >= 2) {
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.strokeStyle = color;
        ctx.setLineDash(none ? [3, 6] : []);
        ctx.globalAlpha = (none ? 0.05 : 0.09) * scene.futuresAlpha;
        ctx.lineWidth = 7;
        ctx.stroke();
        ctx.globalAlpha = (none ? 0.3 : 0.55) * scene.futuresAlpha;
        ctx.lineWidth = 1.4;
        ctx.stroke();
      }
      // Imagined landing point, once the unfurl has reached the end of this future.
      if (upTo >= f.path.length - 1 && scene.futuresFrom < f.path.length - 1) {
        const [ex, ey] = this.transform.toScreen(...f.path[f.path.length - 1]);
        ctx.setLineDash([]);
        ctx.globalAlpha = 0.85 * scene.futuresAlpha;
        ctx.beginPath();
        ctx.arc(ex, ey, none ? 3 : 2.6, 0, Math.PI * 2);
        if (none) {
          ctx.strokeStyle = SLATE;
          ctx.lineWidth = 1;
          ctx.stroke();
        } else {
          ctx.fillStyle = color;
          ctx.fill();
        }
      }
    }
    ctx.restore();
  }

  private drawTrail(scene: Scene): void {
    const { ctx } = this;
    const n = scene.trail.length;
    if (n < 2) return;
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineWidth = 2.5;
    let [px, py] = this.transform.toScreen(...scene.trail[0]);
    for (let i = 1; i < n; i++) {
      const [x, y] = this.transform.toScreen(...scene.trail[i]);
      ctx.strokeStyle = `rgba(255,224,150,${(0.3 + 0.65 * (i / n)).toFixed(3)})`;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(x, y);
      ctx.stroke();
      px = x;
      py = y;
    }
    ctx.restore();
  }

  /** Predicted landing point (hollow gold diamond) and the actual one (burst). */
  private drawMarkers(scene: Scene, now: number): void {
    const { ctx } = this;
    if (scene.committedPoint) {
      const [x, y] = this.transform.toScreen(...scene.committedPoint);
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(Math.PI / 4);
      ctx.strokeStyle = GOLD;
      ctx.lineWidth = 1.6;
      ctx.strokeRect(-4, -4, 8, 8);
      ctx.restore();
    }
    if (scene.actualPoint) {
      const [x, y] = this.transform.toScreen(...scene.actualPoint);
      const t = clamp((now - scene.landedAt) / 1400, 0, 1);
      ctx.save();
      ctx.strokeStyle = GOLD;
      ctx.globalAlpha = 1 - t;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 6 + easeOutCubic(t) * 46, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }

  private drawButterfly(scene: Scene, now: number): void {
    const { ctx } = this;
    const b = scene.butterfly;
    const [x, y] = this.transform.toScreen(b.x, b.y);
    const u = clamp(this.transform.scale * 1.45, 9, 22); // wing unit in pixels
    const flap = b.landed ? 0.55 + 0.4 * Math.sin(now / 520) : 0.2 + 0.8 * Math.abs(Math.sin(now / 75));

    const halo = ctx.createRadialGradient(x, y, 0, x, y, u * 3.2);
    halo.addColorStop(0, "rgba(255,225,160,0.30)");
    halo.addColorStop(1, "rgba(255,225,160,0)");
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(x, y, u * 3.2, 0, Math.PI * 2);
    ctx.fill();

    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(-b.heading); // screen y points down
    ctx.scale(u, u);
    for (const side of [1, -1]) {
      ctx.save();
      ctx.scale(1, side * flap);
      const wing = ctx.createLinearGradient(0, 0, 0.4, -1.8);
      wing.addColorStop(0, "#fff1c9");
      wing.addColorStop(0.45, GOLD);
      wing.addColorStop(1, "#ff9d6c");
      ctx.fillStyle = wing;
      ctx.strokeStyle = "rgba(60,30,10,0.55)";
      ctx.lineWidth = 0.06;
      ctx.beginPath(); // forewing
      ctx.moveTo(0.3, 0);
      ctx.bezierCurveTo(1.25, -0.6, 1.3, -1.9, 0.3, -1.8);
      ctx.bezierCurveTo(-0.2, -1.7, -0.2, -0.6, 0, 0);
      ctx.fill();
      ctx.stroke();
      ctx.beginPath(); // hindwing
      ctx.moveTo(0, 0);
      ctx.bezierCurveTo(-0.3, -0.5, -1.3, -1.4, -1.5, -0.7);
      ctx.bezierCurveTo(-1.6, -0.2, -0.8, 0, -0.2, 0);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }
    ctx.fillStyle = "#2b1c12";
    ctx.beginPath();
    ctx.ellipse(-0.1, 0, 0.75, 0.13, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#2b1c12";
    ctx.lineWidth = 0.05;
    ctx.beginPath();
    ctx.moveTo(0.6, 0.03);
    ctx.quadraticCurveTo(0.95, 0.12, 1.1, 0.42);
    ctx.moveTo(0.6, -0.03);
    ctx.quadraticCurveTo(0.95, -0.12, 1.1, -0.42);
    ctx.stroke();
    ctx.restore();
  }

  private drawWind(scene: Scene): void {
    const { ctx, w } = this;
    const cx = w - 46;
    const cy = 46;
    const a = (scene.wind.direction_deg * Math.PI) / 180;
    const len = 6 + 16 * (scene.wind.strength / scene.wind.max);
    ctx.save();
    ctx.fillStyle = "rgba(10,16,40,0.6)";
    ctx.strokeStyle = "rgba(160,180,255,0.25)";
    ctx.beginPath();
    ctx.arc(cx, cy, 28, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.translate(cx, cy);
    ctx.rotate(-a);
    ctx.strokeStyle = CYAN;
    ctx.fillStyle = CYAN;
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(-len, 0);
    ctx.lineTo(len, 0);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(len + 5, 0);
    ctx.lineTo(len - 3, -5);
    ctx.lineTo(len - 3, 5);
    ctx.fill();
    ctx.restore();
    ctx.font = "600 10px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillStyle = "rgba(200,215,255,0.8)";
    ctx.fillText(`WIND ${scene.wind.strength.toFixed(1)}`, cx, cy + 44);
  }
}
