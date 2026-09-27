// Build the user guide: docs/guide/guide.html -> docs/TailSafe-User-Guide.pdf
//
//   cd web && npm ci                        # the guide uses the web UI's fonts
//   NODE_PATH="$(npm root -g)" node docs/guide/build.mjs
//
// Needs Playwright with Chromium (`npm install -g playwright`). The contents
// page gets its page numbers from the PDF's own bookmarks: the guide is
// printed once, the page of every heading is read from the outline, the
// numbers are filled in, and the guide is printed again.

import { createRequire } from "node:module";
import { writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const { chromium } = createRequire(import.meta.url)("playwright");
const HERE = dirname(fileURLToPath(import.meta.url));
const SOURCE = join(HERE, "guide.html");
const TARGET = resolve(HERE, "..", "TailSafe-User-Guide.pdf");

const PDF_OPTIONS = { preferCSSPageSize: true, printBackground: true, outline: true, tagged: true };

// ------------------------------------------------------------ PDF reading

function objects(pdf) {
  const map = new Map();
  for (const m of pdf.matchAll(/(\d+) 0 obj\s*([\s\S]*?)endobj/g)) map.set(Number(m[1]), m[2]);
  return map;
}

function ref(dict, key) {
  const m = dict.match(new RegExp(`/${key}\\s+(\\d+) 0 R`));
  return m ? Number(m[1]) : null;
}

/** A PDF string (literal or hex) following `/key`, decoded to text. */
function stringAfter(dict, key) {
  let i = dict.indexOf(`/${key}`);
  if (i < 0) return null;
  i += key.length + 1;
  while (/\s/.test(dict[i])) i++;
  const bytes = [];
  if (dict[i] === "<") {
    const hex = dict.slice(i + 1, dict.indexOf(">", i)).replace(/\s+/g, "");
    for (let k = 0; k < hex.length; k += 2) bytes.push(parseInt(hex.slice(k, k + 2).padEnd(2, "0"), 16));
  } else if (dict[i] === "(") {
    let depth = 0;
    for (i++; i < dict.length; i++) {
      const c = dict[i];
      if (c === "\\") {
        const n = dict[++i];
        const esc = { n: 10, r: 13, t: 9, b: 8, f: 12, "(": 40, ")": 41, "\\": 92 };
        if (n in esc) bytes.push(esc[n]);
        else if (/[0-7]/.test(n)) {
          let oct = n;
          while (oct.length < 3 && /[0-7]/.test(dict[i + 1])) oct += dict[++i];
          bytes.push(parseInt(oct, 8));
        } else if (n === "\r" || n === "\n") {
          if (n === "\r" && dict[i + 1] === "\n") i++;
        } else bytes.push(n.charCodeAt(0));
      } else if (c === "(") {
        depth++;
        bytes.push(40);
      } else if (c === ")") {
        if (depth === 0) break;
        depth--;
        bytes.push(41);
      } else bytes.push(c.charCodeAt(0));
    }
  } else return null;
  if (bytes[0] === 0xfe && bytes[1] === 0xff) {
    let s = "";
    for (let k = 2; k + 1 < bytes.length; k += 2) s += String.fromCharCode((bytes[k] << 8) | bytes[k + 1]);
    return s;
  }
  return String.fromCharCode(...bytes);
}

// Chromium joins a heading's wrapped lines without a space, so compare without spaces.
const norm = (s) => s.replace(/\s+/g, "");

/** Page number (1-based) of every bookmark, in document order. */
function bookmarkPages(buffer) {
  const pdf = buffer.toString("latin1");
  const objs = objects(pdf);
  const catalog = [...objs.values()].find((d) => /\/Type\s*\/Catalog/.test(d));
  const pages = [];
  const walk = (id) => {
    const d = objs.get(id);
    if (/\/Type\s*\/Pages/.test(d)) {
      const kids = d.match(/\/Kids\s*\[([^\]]*)\]/)[1];
      for (const m of kids.matchAll(/(\d+) 0 R/g)) walk(Number(m[1]));
    } else pages.push(id);
  };
  walk(ref(catalog, "Pages"));
  const pageOf = new Map(pages.map((id, i) => [id, i + 1]));
  const marks = [];
  const visit = (id) => {
    while (id !== null) {
      const d = objs.get(id);
      const dest = d.match(/\/Dest\s*\[\s*(\d+) 0 R/) ?? d.match(/\/D\s*\[\s*(\d+) 0 R/);
      marks.push({ title: norm(stringAfter(d, "Title") ?? ""), page: dest ? pageOf.get(Number(dest[1])) : null });
      const first = ref(d, "First");
      if (first !== null) visit(first);
      id = ref(d, "Next");
    }
  };
  const outlines = ref(catalog, "Outlines");
  if (outlines !== null) visit(ref(objs.get(outlines), "First"));
  return { marks, count: pages.length };
}

// ------------------------------------------------------------------ build

const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  await page.goto(pathToFileURL(SOURCE).href, { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready);
  const fontsOk = await page.evaluate(() => document.fonts.check('12px "Atkinson Hyperlegible Next"'));
  if (!fontsOk) throw new Error("The guide's fonts did not load: run `npm ci` in web/ first.");

  let pdf;
  let previous = "";
  let settled = false;
  for (let pass = 1; pass <= 4 && !settled; pass++) {
    pdf = await page.pdf(PDF_OPTIONS);
    const { marks, count } = bookmarkPages(pdf);
    const byTitle = new Map();
    for (const m of marks) if (!byTitle.has(m.title)) byTitle.set(m.title, m.page);
    const filled = await page.evaluate((entries) => {
      const pages = new Map(entries);
      const missing = [];
      const numbers = [];
      for (const el of document.querySelectorAll("[data-page-of]")) {
        const target = document.getElementById(el.dataset.pageOf);
        const title = target ? target.textContent.replace(/\s+/g, "") : null;
        const n = title ? pages.get(title) : undefined;
        if (n === undefined) missing.push(el.dataset.pageOf);
        el.textContent = n === undefined ? "?" : String(n);
        numbers.push(`${el.dataset.pageOf}:${n}`);
      }
      return { missing, numbers: numbers.join(",") };
    }, [...byTitle]);
    if (filled.missing.length) throw new Error(`No bookmark for: ${[...new Set(filled.missing)].join(", ")}`);
    console.log(`pass ${pass}: ${count} pages`);
    settled = filled.numbers === previous;
    previous = filled.numbers;
  }
  if (!settled) throw new Error("Page numbers on the contents page did not settle.");
  writeFileSync(TARGET, pdf);
  console.log(`wrote ${TARGET}`);
} finally {
  await browser.close();
}
