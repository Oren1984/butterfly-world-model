import { expect, test, type Page } from "@playwright/test";

const garden = "#garden";

async function open(page: Page, seed = 7): Promise<{ errors: string[]; hosts: Set<string> }> {
  const errors: string[] = [];
  const hosts = new Set<string>();
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("request", (r) => hosts.add(new URL(r.url()).host));
  await page.request.post("/api/world/reset", { data: { seed } });
  await page.goto("/");
  await expect(page.locator("#loading")).toBeHidden();
  return { errors, hosts };
}

const serial = (page: Page) => page.locator(garden).evaluate((el) => Number((el as HTMLElement).dataset.predictionSerial));
const worldState = async (page: Page) => (await page.request.get("/api/world/state")).json();

async function waitForTwentyFutures(page: Page): Promise<void> {
  await expect(page.locator(garden)).toHaveAttribute("data-status", "flying", { timeout: 20_000 });
  await expect(page.locator(garden)).toHaveAttribute("data-futures", "20");
  await expect(page.locator("#futures-count")).toHaveText("20");
}

test("one butterfly, twenty futures, one landing, a measured prediction", async ({ page }) => {
  const { errors, hosts } = await open(page);

  // 1-2. The application opens and the butterfly is visibly moving.
  await expect(page.locator("h1")).toHaveText("Butterfly World Model");
  const before = await page.locator(garden).screenshot();
  const positionBefore = await page.locator("#butterfly-text").textContent();
  await page.waitForTimeout(800);
  expect((await page.locator(garden).screenshot()).equals(before)).toBe(false);
  expect(await page.locator("#butterfly-text").textContent()).not.toBe(positionBefore);

  // 3. Twenty imagined futures and a committed prediction appear.
  await waitForTwentyFutures(page);
  await expect(page.locator("#story-futures")).toHaveClass(/on/);
  await expect(page.locator("#story-predicted")).toHaveClass(/on/);
  await expect(page.locator("#committed-text")).toContainText("%");
  await expect(page.locator("#prob-list li")).not.toHaveCount(0);
  const probabilitySum = await page.locator("#prob-list .value").evaluateAll((els) =>
    els.reduce((sum, el) => sum + parseFloat(el.textContent ?? "0"), 0),
  );
  expect(probabilitySum).toBeGreaterThan(97);
  expect(probabilitySum).toBeLessThan(103);

  // Pause so the following checks are about our changes, not about the passage of time.
  await page.locator("#btn-pause").click();
  await expect(page.locator("#model-status")).toHaveText("Paused");

  // 4-5. Changing the wind invalidates the prediction and produces a new one.
  let lastSerial = await serial(page);
  let committedStep = (await worldState(page)).committed.observation_step;
  const versionBefore = (await worldState(page)).config_version;
  await page.locator("#wind-strength").fill("2");
  await page.locator("#wind-direction").fill("270");
  await expect(page.locator("#wind-text")).toHaveText("2.0 u/s toward 270°");
  await expect.poll(() => serial(page)).toBeGreaterThan(lastSerial);
  let state = await worldState(page);
  expect(state.observation.wind).toEqual({ strength: 2, direction_deg: 270 });
  expect(state.config_version).toBeGreaterThan(versionBefore);
  expect(state.committed.observation_step).toBeGreaterThanOrEqual(committedStep);
  await expect(page.locator(garden)).toHaveAttribute("data-futures", "20");

  // 6-7. Dragging a flower moves it in the world and the landing probabilities are recomputed.
  lastSerial = await serial(page);
  const flower = state.observation.flowers[0];
  const versionMid = state.config_version;
  const from = await page.evaluate((id) => (window as any).__garden.flowerScreen(id), flower.id);
  await page.mouse.move(from[0], from[1]);
  await page.mouse.down();
  await page.mouse.move(from[0] - 60, from[1] + 40, { steps: 6 });
  await expect(page.locator(garden)).toHaveAttribute("data-futures", "0"); // stale futures are removed while dragging
  await page.mouse.up();
  await expect.poll(() => serial(page)).toBeGreaterThan(lastSerial);
  state = await worldState(page);
  const moved = state.observation.flowers.find((f: { id: string }) => f.id === flower.id);
  expect(Math.hypot(moved.x - flower.x, moved.y - flower.y)).toBeGreaterThan(2);
  expect(state.config_version).toBeGreaterThan(versionMid);
  await expect(page.locator(garden)).toHaveAttribute("data-futures", "20");
  await expect(page.locator("#prob-list li")).toHaveCount(state.observation.flowers.length + 1);

  // Hiding the futures is a pure display toggle.
  await page.locator("#toggle-futures").uncheck();
  await page.locator("#toggle-futures").check();

  // 8-9. Let the flight finish and compare the prediction with what happened.
  await page.locator("#btn-pause").click();
  await expect(page.locator("#result")).toBeVisible({ timeout: 30_000 });
  state = await worldState(page);
  const result = state.result;
  const label = (o: string) => (o === "no_landing_within_horizon" ? "No landing" : `Flower ${o}`);
  await expect(page.locator("#result-predicted")).toHaveText(label(result.predicted));
  await expect(page.locator("#result-actual")).toHaveText(label(result.actual));
  await expect(page.locator("#result-confidence")).toHaveText(`${Math.round(result.confidence * 100)}%`);
  await expect(page.locator("#result-verdict")).toHaveText(result.correct ? "Correct prediction" : "Incorrect prediction");
  await expect(page.locator("#result-error")).toHaveText(
    result.position_error === null ? "not defined" : `${result.position_error.toFixed(1)} units`,
  );
  await expect(page.locator("#story-actual")).toHaveClass(/on/);
  await expect(page.locator("#session-text")).toHaveText(`${state.session.correct} of 1 correct`);

  // 10. Another flight, a reset, and a brand new scenario all work.
  await page.locator("#btn-next").click();
  await expect(page.locator("#result")).toBeHidden();
  await waitForTwentyFutures(page);
  await page.locator("#btn-reset").click();
  await expect(page.locator("#session-text")).toHaveText("0 of 0 correct");
  await waitForTwentyFutures(page);
  const seed = await page.locator("#seed-text").textContent();
  await page.locator("#btn-new").click();
  await expect(page.locator("#seed-text")).not.toHaveText(seed!);
  await waitForTwentyFutures(page);

  // The evaluation panel is filled from a real 500-scenario run.
  await expect(page.locator("#eval-status")).toHaveText("Complete", { timeout: 60_000 });
  await expect(page.locator("#eval-n")).toHaveText("500");
  await expect(page.locator("#eval-acc")).toHaveText(/^\d+\.\d%$/);
  await expect(page.locator("#eval-rows tr")).toHaveCount(3);

  // Nothing went wrong in the console, and the browser only ever talked to its own origin.
  expect(errors).toEqual([]);
  expect([...hosts]).toEqual([new URL(page.url()).host]);
});

for (const viewport of [
  { name: "desktop", width: 1440, height: 900 },
  { name: "mobile", width: 390, height: 844 },
]) {
  test(`${viewport.name} layout has no horizontal overflow and usable controls`, async ({ page }) => {
    await page.setViewportSize(viewport);
    const { errors } = await open(page);
    await waitForTwentyFutures(page);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(0);
    const box = await page.locator(garden).boundingBox();
    expect(box!.width).toBeGreaterThan(280);
    expect(box!.height).toBeGreaterThan(250);
    for (const id of ["#btn-pause", "#btn-reset", "#btn-new", "#wind-strength", "#btn-eval"]) {
      await page.locator(id).scrollIntoViewIfNeeded();
      await expect(page.locator(id)).toBeVisible();
      const b = await page.locator(id).boundingBox();
      expect(b!.x).toBeGreaterThanOrEqual(0);
      expect(b!.x + b!.width).toBeLessThanOrEqual(viewport.width);
    }
    expect(errors).toEqual([]);
  });
}

test("the page keeps working when the tab is hidden and shown again", async ({ page }) => {
  await open(page);
  await waitForTwentyFutures(page);
  const hide = (hidden: boolean) =>
    page.evaluate((h) => {
      Object.defineProperty(document, "hidden", { configurable: true, get: () => h });
      document.dispatchEvent(new Event("visibilitychange"));
    }, hidden);
  await hide(true);
  const stepWhileHidden = (await worldState(page)).observation.step;
  await page.waitForTimeout(700);
  expect((await worldState(page)).observation.step).toBe(stepWhileHidden); // no work while hidden
  await hide(false);
  await expect.poll(async () => (await worldState(page)).observation.step + (await worldState(page)).session.cycles * 1000)
    .toBeGreaterThan(stepWhileHidden);
});
