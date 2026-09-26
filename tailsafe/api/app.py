"""FastAPI application exposing TailSafe to the web frontend."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from tailsafe import DISCLAIMER, __version__
from tailsafe.config import get_params

app = FastAPI(
    title="TailSafe API",
    version=__version__,
    description="Tail-risk evacuation stress-testing for high-rise Hong Kong. " + DISCLAIMER,
)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe with version and the responsible-use notice."""
    return {"status": "ok", "version": __version__, "disclaimer": DISCLAIMER}


@app.get("/api/params")
def list_params() -> list[dict[str, Any]]:
    """Every registry parameter with its unit, distribution family and source."""
    return [
        {
            "path": p.path,
            "dist": p.dist,
            "unit": p.unit,
            "source": p.source,
            "assumption": p.is_assumption,
        }
        for p in get_params().leaves()
    ]
