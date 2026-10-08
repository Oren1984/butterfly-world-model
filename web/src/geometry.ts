import { NO_LANDING, type Pt } from "./types";

export interface Transform {
  scale: number;
  toScreen: (x: number, y: number) => Pt;
  toWorld: (sx: number, sy: number) => Pt;
}

/** Fit a world of (worldW x worldH), y up, centred inside a (w x h) pixel canvas, y down. */
export function makeTransform(w: number, h: number, worldW: number, worldH: number, pad = 0.04): Transform {
  const scale = Math.min(w / worldW, h / worldH) * (1 - 2 * pad);
  const ox = (w - worldW * scale) / 2;
  const oy = (h - worldH * scale) / 2;
  return {
    scale,
    toScreen: (x, y) => [ox + x * scale, oy + (worldH - y) * scale],
    toWorld: (sx, sy) => [(sx - ox) / scale, worldH - (sy - oy) / scale],
  };
}

export const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/** Interpolate angles along the shortest arc. */
export function lerpAngle(a: number, b: number, t: number): number {
  const d = ((b - a + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
  return a + d * t;
}

/** The part of a polyline between fractional indices [from, to], with interpolated ends. */
export function slicePath(path: Pt[], from: number, to: number): Pt[] {
  const last = path.length - 1;
  from = clamp(from, 0, last);
  to = clamp(to, 0, last);
  if (to <= from) return [];
  const at = (i: number): Pt => {
    const k = Math.min(Math.floor(i), last - 1);
    const t = i - k;
    return [lerp(path[k][0], path[k + 1][0], t), lerp(path[k][1], path[k + 1][1], t)];
  };
  const out: Pt[] = [at(from)];
  for (let i = Math.floor(from) + 1; i < to; i++) out.push(path[i]);
  out.push(at(to));
  return out;
}

/** Keep a dragged flower's whole landing zone inside the world. */
export function clampFlower(x: number, y: number, radius: number, worldW: number, worldH: number): Pt {
  return [clamp(x, radius, worldW - radius), clamp(y, radius, worldH - radius)];
}

export const pct = (p: number) => `${Math.round(p * 100)}%`;

export function outcomeLabel(outcome: string): string {
  return outcome === NO_LANDING ? "No landing" : `Flower ${outcome}`;
}

export const easeOutCubic = (t: number) => 1 - Math.pow(1 - clamp(t, 0, 1), 3);
