// The browser version: the same UI with no server. Recorded responses (see
// tailsafe/api/static_site.py) are served as static files and looked up by
// request key; the what-if network runs in the browser.
import { ApiError } from "../api";
import type { ScenarioSpec, SurrogatePrediction } from "../types";
import { requestKey } from "./key";
import { type GraphFile, type ModelFile, type Network, loadNetwork, predict } from "./surrogate";

export const STATIC = import.meta.env.VITE_STATIC === "1";
const ROOT = `${import.meta.env.BASE_URL}data/`;

export const NOT_RECORDED =
  "The browser version has results for the reference scenario of each building type, not for these settings. " +
  "To simulate other settings, run the full app — or use the What-if screen, which estimates any settings " +
  "instantly in your browser.";

const OFFLINE = "Could not load the results. Check your connection and try again.";

/** A static file, or null when it is missing (some servers answer with the app page instead of 404). */
async function load(url: string): Promise<Response | null> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch {
    throw new ApiError(0, OFFLINE);
  }
  const html = (res.headers.get("content-type") ?? "").includes("text/html");
  return res.ok && !html ? res : null;
}

async function recorded(key: string, ext: "json" | "pdf"): Promise<Response> {
  const res = await load(`${ROOT}r/${key}.${ext}`);
  if (!res) throw new ApiError(404, NOT_RECORDED);
  return res;
}

const NO_MODEL = "The what-if model is not available in this copy of the site.";

let network: Promise<Network> | null = null;
const graphs = new Map<string, Promise<GraphFile>>();
let disclaimer: Promise<string> | null = null;

function loadModel(): Promise<Network> {
  network ??= (async () => {
    const [file, weights] = await Promise.all([load(`${ROOT}surrogate/model.json`), load(`${ROOT}surrogate/weights.bin`)]);
    if (!file || !weights) throw new ApiError(404, NO_MODEL);
    return loadNetwork((await file.json()) as ModelFile, await weights.arrayBuffer());
  })();
  network.catch(() => {
    network = null; // allow a retry
  });
  return network;
}

function loadGraph(buildingId: string): Promise<GraphFile> {
  let g = graphs.get(buildingId);
  if (!g) {
    g = load(`${ROOT}surrogate/${buildingId}.json`).then((r) => {
      if (!r) throw new ApiError(404, "This building is not part of the browser version.");
      return r.json() as Promise<GraphFile>;
    });
    graphs.set(buildingId, g);
  }
  return g;
}

async function surrogate(body: { building_id: string; spec: ScenarioSpec }): Promise<SurrogatePrediction> {
  disclaimer ??= staticRequest<{ disclaimer: string }>("GET", "/api/health").then((h) => h.disclaimer);
  const [net, g, text] = await Promise.all([loadModel(), loadGraph(body.building_id), disclaimer]);
  return predict(net, g, body.spec, text);
}

/** Answer an API request from the recorded files (or the in-browser model). */
export async function staticRequest<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
  if (method === "POST" && path === "/api/surrogate/predict")
    return (await surrogate(body as { building_id: string; spec: ScenarioSpec })) as T;
  const res = await recorded(requestKey(method, path, body), "json");
  return (await res.json()) as T;
}

/** A recorded binary response (the briefing PDF). */
export async function staticBlob(path: string, body: unknown): Promise<Blob> {
  return (await recorded(requestKey("POST", path, body), "pdf")).blob();
}
