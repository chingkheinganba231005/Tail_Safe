import type { Job } from "../types";

/** Progress of a background job (status, bar, message). */
export function JobProgress({ job, error, label }: { job: Job | null; error: string | null; label: string }) {
  if (error) {
    return (
      <div className="card flex items-start gap-2 p-3 text-sm" role="alert">
        <span aria-hidden style={{ color: "var(--critical)" }}>
          ⛔
        </span>
        <div>
          <strong>{label} failed.</strong> <span className="secondary">{error}</span>
        </div>
      </div>
    );
  }
  if (!job) return null;
  const frac = job.total > 0 ? job.done / job.total : null;
  const done = job.status === "done";
  return (
    <div className="card p-3 text-sm" aria-live="polite">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span>
          <strong>{label}</strong>{" "}
          <span className="secondary">
            {done ? (job.cached ? "loaded from cache" : "finished") : job.status === "queued" ? "queued" : "running"}
          </span>
        </span>
        <span className="muted tabular">
          {frac !== null && !done ? `${job.done} / ${job.total} · ` : ""}
          {job.elapsed_s.toFixed(0)} s
        </span>
      </div>
      {!done && (
        <div className="mt-2 h-2 overflow-hidden rounded" style={{ background: "var(--grid)" }}>
          <div
            className={frac === null ? "animate-pulse" : ""}
            style={{
              width: `${frac === null ? 100 : Math.max(2, 100 * frac)}%`,
              height: "100%",
              background: "var(--series-1)",
              opacity: frac === null ? 0.4 : 1,
              transition: "width 0.3s",
            }}
          />
        </div>
      )}
      {job.message && !done && <div className="muted mt-1 truncate text-xs">{job.message}</div>}
    </div>
  );
}
