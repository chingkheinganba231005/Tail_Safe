// Screenshots for the user guide, taken from the browser version.
//
//   cd web && VITE_STATIC=1 npx vite build && npx vite preview --port 4173
//   NODE_PATH="$(npm root -g)" node docs/guide/screenshots.mjs [http://127.0.0.1:4173/]
//
// Needs Playwright (`npm install -g playwright`) and the recorded results in
// web/public/data (`tailsafe export-site`). Writes docs/guide/img/*.jpg.

import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const { chromium } = createRequire(import.meta.url)("playwright");
const BASE = process.argv[2] ?? "http://127.0.0.1:4173/";
const OUT = join(dirname(fileURLToPath(import.meta.url)), "img");
mkdirSync(OUT, { recursive: true });

const WEBGL = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"];

async function save(target, name, options = {}) {
  await target.screenshot({ path: join(OUT, `${name}.jpg`), type: "jpeg", quality: 88, ...options });
  console.log(`  ${name}.jpg`);
}

/** The card (section) that contains a heading with this text. */
function card(page, heading) {
  return page.locator("section.card", { has: page.getByRole("heading", { name: heading, exact: false }) }).first();
}

async function next(page, action) {
  await page.getByRole("button", { name: new RegExp(`^Continue: ${action}`, "i") }).click();
  await page.waitForTimeout(600);
}

/** Move a range input to `value` (rounded to its step). */
async function seek(page, slider, value) {
  const max = Number(await slider.getAttribute("max"));
  const step = Number(await slider.getAttribute("step")) || 1;
  await slider.fill(String(Math.min(max, Math.round(value / step) * step)));
  await page.waitForTimeout(700);
}

async function desktop(browser) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 860 }, deviceScaleFactor: 1.5, colorScheme: "light" });
  page.on("pageerror", (e) => console.error(`page error: ${e.message}`));
  await page.goto(BASE);
  await page.getByRole("button", { name: /Cruciform/ }).first().waitFor();
  await page.waitForTimeout(400);
  await save(page, "layout");

  // Step 1: building.
  await page.getByRole("button", { name: /Cruciform/ }).first().click();
  await page.getByRole("button", { name: "Generate building" }).click();
  await page.getByRole("button", { name: /Confirm building/ }).waitFor();
  await page.waitForTimeout(500);
  await save(page.locator("main"), "s1-review");
  await page.getByRole("button", { name: /Confirm building/ }).click();

  // Step 2: scenario.
  await page.getByRole("button", { name: "Run stress test" }).waitFor();
  await page.waitForTimeout(400);
  await save(page.locator("main"), "s2-scenario");
  await page.getByRole("button", { name: "Weekday day" }).click();
  await page.waitForTimeout(300);
  const changed = page.locator(".callout", { hasText: "You have changed the scenario" });
  await changed.scrollIntoViewIfNeeded();
  await save(changed, "s2-changed");
  await page.getByRole("button", { name: "Back to the reference scenario" }).click();
  await page.waitForTimeout(300);
  await page.getByRole("button", { name: "Run stress test" }).click();

  // Step 3: results.
  await page.getByRole("heading", { name: "Stress-test results" }).waitFor({ timeout: 60000 });
  await page.waitForTimeout(800);
  const main = page.locator("main");
  const box = await main.boundingBox();
  const histogram = card(page, "Time until everyone is out");
  const hb = await histogram.boundingBox();
  await save(page, "s3-top", {
    fullPage: true,
    clip: { x: box.x, y: box.y, width: box.width, height: hb.y + hb.height - box.y + 4 },
  });
  // The two cards share a row; let the shorter one end at its content.
  await card(page, "Who is still inside in the worst 5%").evaluate((el) => (el.style.alignSelf = "start"));
  await save(card(page, "Who is still inside in the worst 5%"), "s3-who");
  await save(card(page, "Floors that fail"), "s3-floors");
  await save(card(page, "Where stairs queue in the worst 5%"), "s3-stairs");

  // Step 4: 3D stack.
  await next(page, "watch it floor by floor");
  await page.locator("canvas").first().waitFor({ timeout: 60000 });
  await page.waitForTimeout(1500);
  await seek(page, page.locator('input[aria-label="Time"]'), 64);
  await page.waitForTimeout(1200);
  await page.evaluate(() => window.scrollTo(0, 0));
  await save(page.locator("main"), "s4-stack");

  // Step 5: person-by-person replay.
  await next(page, "replay people moving");
  await page.getByText("Fast engine vs person-by-person").waitFor({ timeout: 60000 });
  await page.waitForTimeout(1200);
  await seek(page, page.locator('input[aria-label="Time"]'), 420);
  await page.evaluate(() => window.scrollTo(0, 0));
  await save(page.locator("main"), "s5-replay");

  // Step 6: bottlenecks.
  await next(page, "find what causes it");
  await page.getByRole("button", { name: "Rank bottlenecks" }).click();
  await page.getByText(/Change in CVaR/).first().waitFor({ timeout: 60000 });
  await page.waitForTimeout(1500);
  await page.evaluate(() => window.scrollTo(0, 0));
  await save(page.locator("main"), "s6-ranking");

  // Step 7: optimise.
  await next(page, "test operational fixes");
  await save(card(page, "Optimise"), "s7-setup");
  await page.getByRole("button", { name: "Find a better plan" }).click();
  await page.getByRole("heading", { name: "Recommended plan" }).waitFor({ timeout: 60000 });
  await page.waitForTimeout(800);
  const plan = card(page, "Recommended plan");
  const confirmation = card(page, /Confirmed on/);
  const pb = await plan.boundingBox();
  const cb = await confirmation.boundingBox();
  await save(page, "s7-plan", { fullPage: true, clip: { x: pb.x, y: pb.y, width: pb.width, height: cb.y + cb.height - pb.y } });
  await page.getByRole("button", { name: "Load animation" }).click();
  await page.getByRole("heading", { name: /worst confirmation scenario/ }).waitFor();
  await page.locator("canvas").nth(1).waitFor({ timeout: 60000 });
  await page.waitForTimeout(1500);
  const split = page.locator('input[aria-label="Time"]').last();
  await seek(page, split, 1500);
  await page.waitForTimeout(1500);
  await save(card(page, "Before / after"), "s7-before-after");

  // Step 8: what-if.
  await next(page, "try changes instantly");
  await page.getByText(/Estimated in \d+ ms/).waitFor({ timeout: 60000 });
  await page.getByRole("button", { name: "Confirm with full simulation" }).click();
  await page.getByText("Open the full results").waitFor({ timeout: 60000 });
  await page.waitForTimeout(800);
  await page.evaluate(() => window.scrollTo(0, 0));
  await save(page.locator("main"), "s8-whatif");
  // A setting the network was not trained on.
  await page.getByLabel("Lifts out of service").fill("2");
  await page.getByText(/Outside what the surrogate was trained on/).waitFor({ timeout: 30000 });
  await page.waitForTimeout(500);
  await save(page.locator(".callout", { hasText: "Outside what the surrogate" }), "s8-warning");

  // Step 9: briefing.
  await next(page, "get the one-page briefing");
  await page.getByRole("button", { name: "Write briefing" }).click();
  await page.getByRole("button", { name: "Download PDF" }).waitFor({ timeout: 60000 });
  await page.waitForTimeout(600);
  await page.evaluate(() => window.scrollTo(0, 0));
  await save(page.locator("main"), "s9-briefing");
  await page.close();
}

async function phone(browser) {
  const page = await browser.newPage({
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 2,
    isMobile: true,
    hasTouch: true,
    colorScheme: "light",
  });
  await page.goto(BASE);
  await page.getByRole("button", { name: /Cruciform/ }).first().waitFor();
  await page.waitForTimeout(400);
  await save(page, "phone-building", { quality: 82 });
  await page.getByRole("button", { name: /Cruciform/ }).first().click();
  await page.getByRole("button", { name: "Generate building" }).click();
  await page.getByRole("button", { name: /Confirm building/ }).click();
  await page.getByRole("button", { name: "Run stress test" }).click();
  await page.getByRole("heading", { name: "Stress-test results" }).waitFor({ timeout: 60000 });
  await page.waitForTimeout(800);
  await save(page, "phone-results", { quality: 82 });
  await page.getByText("All steps").click();
  await page.waitForTimeout(400);
  await save(page, "phone-steps", { quality: 82 });
  await page.close();
}

const browser = await chromium.launch({ args: WEBGL });
try {
  console.log("desktop");
  await desktop(browser);
  console.log("phone");
  await phone(browser);
} finally {
  await browser.close();
}
