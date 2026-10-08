import { expect, test } from "@playwright/test";

// Regenerates the README screenshots from the running application:
//   CAPTURE=1 npx playwright test screenshots
test.skip(!process.env.CAPTURE, "set CAPTURE=1 to regenerate docs/images");

test("capture documentation screenshots", async ({ page }) => {
  const out = "../docs/images";
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.request.post("/api/world/reset", { data: { seed: 7 } });
  await page.goto("/");
  await expect(page.locator("#eval-status")).toHaveText("Complete", { timeout: 60_000 });
  await page.locator("#btn-reset").click();
  await expect(page.locator("#garden")).toHaveAttribute("data-status", "observing");
  await expect(page.locator("#garden")).toHaveAttribute("data-status", "flying", { timeout: 20_000 });

  // The snapshot moment: time stands still while twenty futures unfurl.
  await expect(page.locator("#garden")).toHaveAttribute("data-futures", "20", { timeout: 20_000 });
  await expect(page.locator("#story-predicted")).toHaveClass(/on/);
  await page.waitForTimeout(450);
  await page.screenshot({ path: `${out}/garden-twenty-futures.jpg`, type: "jpeg", quality: 88 });
  const stage = (await page.locator(".stage").boundingBox())!;
  await page.screenshot({
    path: `${out}/imagined-futures-closeup.png`,
    clip: { x: stage.x + 40, y: stage.y + 300, width: 560, height: 400 },
  });

  // The landing and the scored prediction.
  await expect(page.locator("#result")).toBeVisible({ timeout: 30_000 });
  await page.waitForTimeout(600);
  await page.screenshot({ path: `${out}/landing-result.jpg`, type: "jpeg", quality: 88 });

  // The evaluation panel, filled by the real 500-scenario run.
  await page.locator("#btn-pause").click();
  await page.locator("#h-eval").scrollIntoViewIfNeeded();
  await page.locator("section[aria-labelledby='h-eval']").screenshot({ path: `${out}/evaluation-panel.png` });
});
