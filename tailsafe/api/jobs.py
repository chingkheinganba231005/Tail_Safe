"""Background jobs for long simulations, with progress and a result cache.

Heavy work (Monte Carlo, bottlenecks, optimisation) runs one job at a time in
a worker thread; each job itself uses a process pool. Results are cached on
disk by a key built from the building digest, the request and the parameter
registry digest, so repeating a request is instant.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Status = Literal["queued", "running", "done", "error"]
JobFn = Callable[[Callable[[int, int], None]], dict[str, Any]]

CACHE_ENV = "TAILSAFE_CACHE_DIR"


def cache_dir() -> Path:
    """Directory for cached results (``$TAILSAFE_CACHE_DIR`` or ``runs/cache``)."""
    return Path(os.environ.get(CACHE_ENV, "runs/cache"))


def cache_key(*parts: object) -> str:
    """Stable key from JSON-serialisable parts."""
    blob = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


@dataclass
class Job:
    """A unit of background work."""

    id: str
    kind: str
    key: str
    status: Status = "queued"
    done: int = 0
    total: int = 0
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None
    cached: bool = False

    def public(self) -> dict[str, Any]:
        """Status without the (possibly large) result."""
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "done": self.done,
            "total": self.total,
            "message": self.message,
            "error": self.error,
            "cached": self.cached,
            "elapsed_s": round((self.finished or time.time()) - self.created, 2),
        }


class JobManager:
    """Runs jobs one at a time and remembers them (and their cached results)."""

    def __init__(self, max_workers: int = 1, use_cache: bool = True) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tailsafe-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.use_cache = use_cache
        self.objects: dict[str, Any] = {}  # in-memory artefacts (e.g. MCResult) by job id

    def get(self, job_id: str) -> Job | None:
        """Look a job up."""
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        """All jobs, newest first."""
        return sorted(self._jobs.values(), key=lambda j: -j.created)

    def submit(self, kind: str, key: str, fn: JobFn) -> Job:
        """Queue ``fn`` (called with a progress callback); reuse a cached result."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, key=key)
        path = cache_dir() / f"{kind}-{key}.json"
        with self._lock:
            self._jobs[job.id] = job
        if self.use_cache and path.exists():
            job.result = json.loads(path.read_text(encoding="utf-8"))
            job.status = "done"
            job.cached = True
            job.finished = time.time()
            return job

        def progress(done: int, total: int) -> None:
            job.done, job.total = done, total

        def run() -> None:
            job.status = "running"
            try:
                result = fn(progress)
                job.result = result
                job.status = "done"
                if self.use_cache:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(result, default=_json_default), encoding="utf-8")
            except Exception as exc:  # report any failure to the client
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = traceback.format_exc(limit=3)
            finally:
                job.finished = time.time()

        self._pool.submit(run)
        return job

    def wait(self, job_id: str, timeout: float = 600.0) -> Job:
        """Block until a job finishes (used by tests and scripts)."""
        t0 = time.time()
        job = self._jobs[job_id]
        while job.status in ("queued", "running"):
            if time.time() - t0 > timeout:
                raise TimeoutError(job_id)
            time.sleep(0.05)
        return job


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def finite(o: Any) -> Any:
    """Replace non-finite floats by ``None`` recursively (strict JSON for browsers)."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [finite(v) for v in o]
    return o
