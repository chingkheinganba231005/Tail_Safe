"""JSON serialisation of buildings and export of the documented JSON schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from tailsafe.building.model import SCHEMA_VERSION, Building
from tailsafe.building.validate import Issue, check_building

SCHEMA_ID = f"urn:tailsafe:schema:building:v{SCHEMA_VERSION}"
_REPO_SCHEMAS = Path(__file__).resolve().parents[2] / "schemas"
_PACKAGED_SCHEMAS = Path(__file__).resolve().parents[1] / "_data" / "schemas"


def schema_dir() -> Path:
    """Directory holding the committed JSON schemas."""
    return _REPO_SCHEMAS if _REPO_SCHEMAS.exists() else _PACKAGED_SCHEMAS


def building_schema() -> dict[str, Any]:
    """JSON schema (draft 2020-12) for building files, generated from the Pydantic models."""
    schema = Building.model_json_schema()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        **schema,
        "title": "TailSafe building",
        "description": (
            "Multi-floor egress graph of a building. Nodes are places (flats, corridors, "
            "lobbies, stair landings, refuge areas, exits); edges are physical connections "
            "stored once (flat, door, stair — stair edges run from the upper landing to "
            "the lower); lifts list the lobby node served on each level. SI units; "
            "ground floor is level 0."
        ),
    }


def export_schemas(out_dir: Path | None = None) -> list[Path]:
    """Write every JSON schema to ``out_dir`` (default: the repository's ``schemas/``)."""
    out = out_dir or _REPO_SCHEMAS
    out.mkdir(parents=True, exist_ok=True)
    path = out / "building.schema.json"
    path.write_text(json.dumps(building_schema(), indent=2) + "\n", encoding="utf-8")
    return [path]


def validate_against_schema(data: dict[str, Any]) -> None:
    """Validate raw JSON data against the building schema (raises ``ValidationError``)."""
    jsonschema.validate(data, building_schema())


def save_building(b: Building, path: Path, *, pretty: bool = False) -> Path:
    """Write ``b`` as JSON (compact by default)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = b.model_dump_json(indent=2 if pretty else None, exclude_none=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def building_from_dict(
    data: dict[str, Any], *, check: bool = True, schema_check: bool = False
) -> tuple[Building, list[Issue]]:
    """Parse (and by default semantically validate) a building from a dict.

    Returns the building and any validation warnings; errors raise.
    """
    if schema_check:
        validate_against_schema(data)
    b = Building.model_validate(data)
    warnings = check_building(b) if check else []
    return b, warnings


def load_building(path: Path, *, check: bool = True, schema_check: bool = False) -> Building:
    """Load a building JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    b, _ = building_from_dict(data, check=check, schema_check=schema_check)
    return b
