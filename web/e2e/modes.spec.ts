import { expect, test, type Page } from "@playwright/test";

function watch(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));
  return errors;
}

test("prediction is the default mode and all three modes work from the normal entry point", async ({ page }) => {
  const errors = watch(page);
  await page.goto("/");
  await expect(page.locator("#loading")).toBeHidden();
  await expect(page.locator('#modes button[data-mode="prediction"]')).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("#garden")).toHaveAttribute("data-status", "flying", { timeout: 20_000 });

  // Goal-Directed Planning: a live episode is generated and the recorded comparison is shown.
  await page.locator('#modes button[data-mode="planning"]').click();
  await expect(page.locator("#lab")).toBeVisible();
  await expect(page.locator("#app")).toBeHidden();
  await expect(page.locator("#lab-title")).toHaveText("Goal-Directed Planning");
  await expect(page.locator("#run-status")).toHaveText("Generated live", { timeout: 30_000 });
  await expect(page.locator("#res-outcome")).not.toHaveText("–");
  await expect(page.locator("#rec-status")).toHaveText("Recorded");
  await expect(page.locator("#rec-body table")).toHaveCount(2);
  // The prediction loop is idle while another mode is shown.
  const step = (await (await page.request.get("/api/world/state")).json()).observation.step;
  await page.waitForTimeout(600);
  expect((await (await page.request.get("/api/world/state")).json()).observation.step).toBe(step);

  await page.locator("#sel-planner").selectOption("subgoal_learned");
  await page.locator("#inp-seed").fill("3");
  await page.locator("#btn-run").click();
  await expect(page.locator("#lab-canvas")).toHaveAttribute("data-episode", /^detour:3:/, { timeout: 30_000 });
  await expect(page.locator("#res-model")).toContainText("planning-dynamics");

  // Visual Planning: pixels-only planner, with the planner's view and goal image.
  await page.locator('#modes button[data-mode="visual"]').click();
  await expect(page.locator("#lab-title")).toHaveText("Visual Planning");
  await expect(page.locator("#run-status")).toHaveText("Generated live", { timeout: 30_000 });
  await expect(page.locator("#lab-views")).toBeVisible();
  await page.locator("#chk-distractor").check();
  await page.locator("#btn-run").click();
  await expect(page.locator("#run-status")).toHaveText("Generated live", { timeout: 30_000 });
  await expect(page.locator("#res-model")).toContainText("visual-jepa");
  await expect(page.locator("#rec-status")).toHaveText("Recorded");

  // Back to prediction: the original mode resumes.
  await page.locator('#modes button[data-mode="prediction"]').click();
  await expect(page.locator("#app")).toBeVisible();
  await expect.poll(async () => (await (await page.request.get("/api/world/state")).json()).observation.step, { timeout: 10_000 })
    .not.toBe(step);
  expect(errors).toEqual([]);
});

test("a mode can be opened directly by link", async ({ page }) => {
  await page.goto("/#planning");
  await expect(page.locator("#lab-title")).toHaveText("Goal-Directed Planning");
  await page.goto("/#visual");
  await page.reload();
  await expect(page.locator("#lab-title")).toHaveText("Visual Planning");
});
