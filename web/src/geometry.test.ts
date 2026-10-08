import { describe, expect, it } from "vitest";
import { clampFlower, lerpAngle, makeTransform, outcomeLabel, pct, slicePath } from "./geometry";
import { NO_LANDING, type Pt } from "./types";

describe("makeTransform", () => {
  const t = makeTransform(1000, 800, 100, 60, 0);

  it("fits the world inside the canvas and flips y", () => {
    expect(t.scale).toBe(10);
    expect(t.toScreen(0, 60)).toEqual([0, 100]); // world top-left sits below the letterbox band
    expect(t.toScreen(100, 0)).toEqual([1000, 700]);
  });

  it("round-trips between world and screen", () => {
    const [sx, sy] = t.toScreen(37.5, 12.25);
    const [x, y] = t.toWorld(sx, sy);
    expect(x).toBeCloseTo(37.5);
    expect(y).toBeCloseTo(12.25);
  });
});

describe("slicePath", () => {
  const path: Pt[] = [[0, 0], [10, 0], [20, 0], [30, 0]];

  it("returns the whole path for the full range", () => {
    expect(slicePath(path, 0, 3)).toEqual(path);
  });

  it("interpolates fractional ends (used for the reveal animation and trimming the past)", () => {
    expect(slicePath(path, 0.5, 2.5)).toEqual([[5, 0], [10, 0], [20, 0], [25, 0]]);
  });

  it("returns nothing for an empty or reversed range", () => {
    expect(slicePath(path, 2, 2)).toEqual([]);
    expect(slicePath(path, 5, 9)).toEqual([]);
  });
});

describe("helpers", () => {
  it("keeps a dragged flower's landing zone inside the world", () => {
    expect(clampFlower(-20, 500, 3, 100, 60)).toEqual([3, 57]);
    expect(clampFlower(50, 30, 3, 100, 60)).toEqual([50, 30]);
  });

  it("interpolates headings across the +/-180 degree seam", () => {
    const a = (170 * Math.PI) / 180;
    const b = (-170 * Math.PI) / 180;
    expect(Math.abs(lerpAngle(a, b, 0.5))).toBeCloseTo(Math.PI);
  });

  it("labels outcomes for people", () => {
    expect(outcomeLabel("C")).toBe("Flower C");
    expect(outcomeLabel(NO_LANDING)).toBe("No landing");
    expect(pct(0.649)).toBe("65%");
  });
});
