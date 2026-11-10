import { expect, test } from "@playwright/test";

test("the Help Center opens from every mode, navigates its topics and closes", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("/");
  await expect(page.locator("#loading")).toBeHidden();

  const help = page.locator("#help");
  await expect(help).toBeHidden();

  // Opens from the default mode and closes with the close button; focus returns to the Help button.
  await page.locator("#btn-help").click();
  await expect(help).toBeVisible();
  await expect(page.locator("#help-title")).toHaveText("Help Center");
  for (const id of ["about", "prediction", "planning", "visual", "read", "faq", "more"]) {
    await expect(page.locator(`#help-${id} h3`)).toHaveCount(1);
  }
  await page.locator("#btn-help-close").click();
  await expect(help).toBeHidden();
  await expect(page.locator("#btn-help")).toBeFocused();

  // Opens from a planning mode; a topic link scrolls inside the dialog without switching mode.
  await page.locator('#modes button[data-mode="planning"]').click();
  await expect(page.locator("#lab")).toBeVisible();
  await page.locator("#btn-help").click();
  await expect(help).toBeVisible();
  await page.locator('.help-toc a[href="#help-faq"]').click();
  await expect(page.locator("#help-faq")).toBeInViewport();
  await expect(page.locator('#modes button[data-mode="planning"]')).toHaveAttribute("aria-selected", "true");

  // Esc closes it.
  await page.keyboard.press("Escape");
  await expect(help).toBeHidden();
  await expect(page.locator("#lab")).toBeVisible();

  expect(errors).toEqual([]);
});
