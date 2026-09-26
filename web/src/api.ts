// Thin client for the TailSafe API. Long work runs as jobs: submit, follow the
// server-sent progress events (with a polling fallback), then fetch the result.
import { STATIC, staticBlob, staticRequest } from "./static/site";
import type { Job } from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (body.detail) detail = JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export async function get<T>(path: string): Promise<T> {
  if (STATIC) return staticRequest<T>("GET", path);
  return handle<T>(await fetch(path));
}

export async function post<T>(path: string, body: unknown): Promise<T> {
  if (STATIC) return staticRequest<T>("POST", path, body);
  return handle<T>(
    await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

/** POST and receive a file (the briefing PDF). */
export async function postBlob(path: string, body: unknown): Promise<Blob> {
  if (STATIC) return staticBlob(path, body);
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(res.status, `request failed (${res.status})`);
  return res.blob();
}

/** Wait for a job to finish, reporting every status change. */
export function followJob(id: string, onUpdate: (job: Job) => void): Promise<Job> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (job: Job) => {
      if (settled) return;
      settled = true;
      if (job.status === "error") reject(new ApiError(500, job.error ?? "job failed"));
      else resolve(job);
    };
    const poll = async () => {
      while (!settled) {
        try {
          const job = await get<Job>(`/api/jobs/${id}`);
          onUpdate(job);
          if (job.status === "done" || job.status === "error") return finish(job);
        } catch (err) {
          settled = true;
          return reject(err);
        }
        await new Promise((r) => setTimeout(r, 1000));
      }
    };
    if (typeof EventSource === "undefined") {
      void poll();
      return;
    }
    const es = new EventSource(`/api/jobs/${id}/events`);
    es.onmessage = (ev: MessageEvent<string>) => {
      const job = JSON.parse(ev.data) as Job;
      onUpdate(job);
      if (job.status === "done" || job.status === "error") {
        es.close();
        finish(job);
      }
    };
    es.onerror = () => {
      es.close();
      if (!settled) void poll();
    };
  });
}

/** Submit a job and resolve with its result. */
export async function runJob<T>(
  path: string,
  body: unknown,
  onUpdate: (job: Job) => void,
): Promise<{ job: Job; result: T }> {
  const job = await post<Job>(path, body);
  onUpdate(job);
  const final = job.status === "done" ? job : await followJob(job.id, onUpdate);
  const result = await get<T>(`/api/jobs/${final.id}/result`);
  return { job: final, result };
}
