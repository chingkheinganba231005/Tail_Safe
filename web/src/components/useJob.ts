import { useCallback, useRef, useState } from "react";
import { runJob } from "../api";
import type { Job } from "../types";

/** Run one background job at a time and expose its live status. */
export function useJob<T>() {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const token = useRef(0);

  const run = useCallback(async (path: string, body: unknown): Promise<{ job: Job; result: T } | null> => {
    const my = ++token.current;
    setError(null);
    setRunning(true);
    try {
      const out = await runJob<T>(path, body, (j) => {
        if (token.current === my) setJob(j);
      });
      return token.current === my ? out : null;
    } catch (err) {
      if (token.current === my) setError(err instanceof Error ? err.message : String(err));
      return null;
    } finally {
      if (token.current === my) setRunning(false);
    }
  }, []);

  return { job, error, running, run };
}
