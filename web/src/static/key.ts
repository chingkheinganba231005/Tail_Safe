// Request keys for the browser version. Must match tailsafe/api/static_site.py
// (`canon`, `body_for_key`, `request_key`) exactly: the recorder stores each
// response under this key, and the browser looks it up the same way.

/** Canonical JSON: sorted keys, integral numbers without a decimal point, undefined dropped. */
export function canon(x: unknown): string {
  if (x === null || x === undefined) return "null";
  if (x === true) return "true";
  if (x === false) return "false";
  if (typeof x === "number") {
    if (!Number.isFinite(x)) return "null";
    if (Number.isInteger(x) && Math.abs(x) < 1e15) return x.toFixed(0);
    return String(x);
  }
  if (typeof x === "string") return JSON.stringify(x);
  if (Array.isArray(x)) return `[${x.map(canon).join(",")}]`;
  const obj = x as Record<string, unknown>;
  const keys = Object.keys(obj)
    .filter((k) => obj[k] !== undefined)
    .sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${canon(obj[k])}`).join(",")}}`;
}

function fnv(bytes: Uint8Array, seed: number): number {
  let h = seed >>> 0;
  for (const b of bytes) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/** The identifying part of a request body (large bodies are reduced). */
export function bodyForKey(path: string, body: unknown): unknown {
  const b = body as Record<string, unknown>;
  if (path === "/api/briefing")
    return {
      building_id: b.building_id,
      bottlenecks: b.bottlenecks !== null && b.bottlenecks !== undefined,
      optimization: b.optimization !== null && b.optimization !== undefined,
      llm: b.llm ?? null,
    };
  if (path === "/api/briefing/pdf") return { markdown: b.markdown };
  return body;
}

/** 16 hex digits identifying a request. */
export function requestKey(method: string, path: string, body?: unknown): string {
  const text = `${method} ${path} ${body === undefined ? "" : canon(bodyForKey(path, body))}`;
  const bytes = new TextEncoder().encode(text);
  const hex = (n: number) => n.toString(16).padStart(8, "0");
  return hex(fnv(bytes, 0x811c9dc5)) + hex(fnv(bytes, 0x050c5d1f));
}
