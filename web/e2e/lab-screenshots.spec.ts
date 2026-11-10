import { expect, test, type Page } from "@playwright/test";

// Regenerates the V1.1 documentation screenshots from the running application:
//   CAPTURE=1 npx playwright test lab-screenshots
test.skip(!process.env.CAPTURE, "set CAPTURE=1 to regenerate docs/images");

const out = "../docs/images";

async function runAndPause(page: Page, fraction: number): Promise<void> {
  await page.locator("#btn-run").click();
  await expect(page.locator("#run-status")).toHaveText("Generated live", { timeout: 60_000 });
  // Scrub to a moment mid-flight so the imagined route and the waypoint are visible.
  const max = Number(await page.locator("#inp-scrub").getAttribute("max"));
  await page.locator("#inp-scrub").fill(String(Math.round(max * fraction)));
  await page.waitForTimeout(300);
}

test("capture V1.1 screenshots", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/#planning");
  await expect(page.locator("#rec-status")).toHaveText("Recorded", { timeout: 30_000 });
  await expect(page.locator("#run-status")).toHaveText("Generated live", { timeout: 60_000 });
  await page.locator("#sel-layout").selectOption("detour");
  await page.locator("#sel-planner").selectOption("waypoint_exact");
  await page.locator("#inp-seed").fill("0");
  await runAndPause(page, 0.35);
  await page.screenshot({ path: `${out}/planning-mode.jpg`, type: "jpeg", quality: 88 });

  await page.locator("#sel-planner").selectOption("flat_exact");
  await runAndPause(page, 1);
  await page.locator(".stage >> visible=true").screenshot({ path: `${out}/planning-flat-stuck.jpg`, type: "jpeg", quality: 88 });

  await page.locator('#modes button[data-mode="visual"]').click();
  await expect(page.locator("#rec-status")).toHaveText("Recorded", { timeout: 30_000 });
  await expect(page.locator("#run-status")).toHaveText("Generated live", { timeout: 60_000 });
  await page.locator("#sel-layout").selectOption("direct");
  await page.locator("#sel-planner").selectOption("two_level");
  await page.locator("#chk-distractor").check();
  await page.locator("#inp-seed").fill("4");
  await runAndPause(page, 0.4);
  await page.screenshot({ path: `${out}/visual-mode.jpg`, type: "jpeg", quality: 88 });

  // The Help Center, opened over the Landing Prediction mode.
  await page.locator('#modes button[data-mode="prediction"]').click();
  await page.locator("#btn-help").click();
  await expect(page.locator("#help")).toBeVisible();
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${out}/help-center.jpg`, type: "jpeg", quality: 88 });
  await page.keyboard.press("Escape");
});
