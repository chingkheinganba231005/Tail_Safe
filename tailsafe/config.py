"""Parameter registry: loading, validating and sampling ``config/params.yaml``.

Every physical or demographic number used by TailSafe lives in the YAML registry
with a ``source:`` field. This module turns that file into :class:`Params`, a
read-only view with dotted-path access (``params["profiles.child.space_factor"]``)
and inverse-CDF sampling so that Latin Hypercube designs and common random numbers
work uniformly for every distribution type.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from scipy import special

PARAMS_ENV_VAR = "TAILSAFE_PARAMS"
ASSUMPTION_MARKER = "ASSUMPTION"

_REPO_PARAMS = Path(__file__).resolve().parents[1] / "config" / "params.yaml"
_PACKAGED_PARAMS = Path(__file__).resolve().parent / "_data" / "params.yaml"

_LEAF_KEYS = frozenset({"value", "dist"})
_DIST_FIELDS: dict[str, tuple[str, ...]] = {
    "constant": ("value",),
    "truncnorm": ("mean", "sd", "min", "max"),
    "lognormal": ("median", "sigma"),
    "uniform": ("min", "max"),
    "categorical": ("p",),
}


class ParamsError(ValueError):
    """Raised when the parameter registry is malformed or a lookup fails."""


def default_params_path() -> Path:
    """Return the parameter file used when no explicit path is given.

    Resolution order: ``$TAILSAFE_PARAMS``, the repository's ``config/params.yaml``,
    then the copy bundled into the installed package.
    """
    env = os.environ.get(PARAMS_ENV_VAR)
    if env:
        return Path(env)
    if _REPO_PARAMS.exists():
        return _REPO_PARAMS
    return _PACKAGED_PARAMS


@dataclass(frozen=True)
class Param:
    """One leaf of the registry: a fixed value or a probability distribution."""

    path: str
    spec: Mapping[str, Any]

    @property
    def source(self) -> str:
        """Citation (or explicit ASSUMPTION marker) for this parameter."""
        return str(self.spec["source"])

    @property
    def unit(self) -> str | None:
        """Unit string, if given."""
        unit = self.spec.get("unit")
        return None if unit is None else str(unit)

    @property
    def is_assumption(self) -> bool:
        """True when the source is an explicit ASSUMPTION placeholder."""
        return self.source.strip().upper().startswith(ASSUMPTION_MARKER)

    @property
    def dist(self) -> str:
        """Distribution family (``constant`` for plain values)."""
        return str(self.spec.get("dist", "constant"))

    @property
    def value(self) -> Any:
        """The fixed value. Raises for distributions (use :meth:`sample`)."""
        if self.dist != "constant":
            raise ParamsError(f"{self.path} is a {self.dist} distribution, not a value")
        return self.spec["value"]

    def __float__(self) -> float:
        return float(self.value)

    # ------------------------------------------------------------------ sampling
    @cached_property
    def categories(self) -> tuple[list[Any], NDArray[np.float64]]:
        """Outcomes and normalised probabilities of a categorical distribution."""
        if self.dist != "categorical":
            raise ParamsError(f"{self.path} is not categorical")
        p: Mapping[Any, float] = self.spec["p"]
        outcomes = list(p.keys())
        probs = np.asarray([float(v) for v in p.values()], dtype=np.float64)
        return outcomes, probs / probs.sum()

    def ppf(self, u: ArrayLike) -> NDArray[Any]:
        """Map uniform(0, 1) draws to this distribution (inverse CDF).

        Using the inverse CDF everywhere means Latin Hypercube designs and common
        random numbers apply to every distribution family in the same way.
        """
        uu = np.clip(np.asarray(u, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        s = self.spec
        kind = self.dist
        if kind == "constant":
            return np.full(uu.shape, s["value"])
        if kind == "uniform":
            return float(s["min"]) + uu * (float(s["max"]) - float(s["min"]))
        if kind == "truncnorm":
            mean, sd = float(s["mean"]), float(s["sd"])
            lo = special.ndtr((float(s["min"]) - mean) / sd)
            hi = special.ndtr((float(s["max"]) - mean) / sd)
            out: NDArray[np.float64] = mean + sd * special.ndtri(lo + uu * (hi - lo))
            return np.clip(out, float(s["min"]), float(s["max"]))
        if kind == "lognormal":
            mu, sigma = np.log(float(s["median"])), float(s["sigma"])
            lo = special.ndtr((np.log(float(s["min"])) - mu) / sigma) if "min" in s else 0.0
            hi = special.ndtr((np.log(float(s["max"])) - mu) / sigma) if "max" in s else 1.0
            z = special.ndtri(np.clip(lo + uu * (hi - lo), 1e-12, 1.0 - 1e-12))
            res: NDArray[np.float64] = np.exp(mu + sigma * z)
            if "min" in s or "max" in s:
                res = np.clip(res, float(s.get("min", 0.0)), float(s.get("max", np.inf)))
            return res
        if kind == "categorical":
            outcomes, probs = self.categories
            idx = np.searchsorted(np.cumsum(probs), uu, side="right")
            idx = np.minimum(idx, len(outcomes) - 1)
            return np.asarray(outcomes, dtype=object)[idx]
        raise ParamsError(f"{self.path}: unknown distribution {kind!r}")  # pragma: no cover

    def sample(self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None) -> Any:
        """Draw from the distribution with ``rng``. Returns a scalar if ``size`` is None."""
        u = rng.random() if size is None else rng.random(size)
        out = self.ppf(u)
        return out.item() if size is None else out

    def mean(self) -> float:
        """Approximate mean (numerical for truncated families)."""
        if self.dist == "constant":
            return float(self.value)
        grid = (np.arange(4096) + 0.5) / 4096
        return float(np.mean(self.ppf(grid).astype(np.float64)))


class Params:
    """Read-only, validated view of the parameter registry."""

    def __init__(self, tree: Mapping[str, Any], origin: Path | None = None) -> None:
        self._tree: dict[str, Any] = copy.deepcopy(dict(tree))
        self.origin = origin
        errors = validate_tree(self._tree)
        if errors:
            joined = "\n  - ".join(errors)
            raise ParamsError(f"Invalid parameter registry ({origin}):\n  - {joined}")
        self._leaves = {p.path: p for p in _iter_leaves(self._tree)}

    @classmethod
    def load(cls, path: str | Path | None = None) -> Params:
        """Load and validate a registry file (default: :func:`default_params_path`)."""
        p = Path(path) if path is not None else default_params_path()
        with p.open(encoding="utf-8") as fh:
            tree = yaml.safe_load(fh)
        if not isinstance(tree, dict):
            raise ParamsError(f"{p} does not contain a mapping")
        return cls(tree, origin=p)

    # ------------------------------------------------------------------ access
    def __getitem__(self, path: str) -> Param:
        try:
            return self._leaves[path]
        except KeyError:
            raise ParamsError(f"Unknown parameter {path!r}") from None

    def __contains__(self, path: object) -> bool:
        return path in self._leaves

    def value(self, path: str) -> Any:
        """Shortcut for ``params[path].value``."""
        return self[path].value

    def scalar(self, path: str) -> float:
        """Fixed value as ``float``."""
        return float(self[path].value)

    def leaves(self) -> Iterator[Param]:
        """Iterate over every parameter leaf in file order."""
        return iter(self._leaves.values())

    def group(self, prefix: str) -> dict[str, Param]:
        """All leaves under ``prefix`` keyed by their path relative to it."""
        pre = prefix.rstrip(".") + "."
        return {k[len(pre) :]: v for k, v in self._leaves.items() if k.startswith(pre)}

    def as_dict(self) -> dict[str, Any]:
        """Deep copy of the underlying tree."""
        return copy.deepcopy(self._tree)

    @cached_property
    def digest(self) -> str:
        """Stable SHA-256 of the registry content (used in result cache keys)."""
        canon = json.dumps(self._tree, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()

    def with_overrides(self, overrides: Mapping[str, Any]) -> Params:
        """Return a copy with leaf fields replaced.

        ``overrides`` maps a dotted leaf path to either a plain value (replaces the
        leaf's ``value``) or a mapping of leaf fields to update, e.g.
        ``{"behaviour.counter_flow_probability": 0.1}`` or
        ``{"premovement.asleep": {"median": 600.0}}``.
        """
        tree = self.as_dict()
        for path, new in overrides.items():
            if path not in self._leaves:
                raise ParamsError(f"Cannot override unknown parameter {path!r}")
            node: Any = tree
            for key in path.split("."):
                node = node[key]
            if isinstance(new, Mapping):
                node.update(copy.deepcopy(dict(new)))
            else:
                if node.get("dist", "constant") != "constant":
                    raise ParamsError(f"{path} is a distribution; override its fields instead")
                node["value"] = new
            node["source"] = str(node.get("source", "")) + " [overridden]"
        return Params(tree, origin=self.origin)

    def assumption_report(self) -> list[Param]:
        """Leaves whose source is still an ASSUMPTION placeholder."""
        return [p for p in self.leaves() if p.is_assumption]


def _is_leaf(node: Any) -> bool:
    return isinstance(node, Mapping) and bool(_LEAF_KEYS & set(node.keys()))


def _iter_leaves(tree: Mapping[str, Any], prefix: str = "") -> Iterator[Param]:
    for key, node in tree.items():
        skey = str(key)
        if skey.startswith("_") or (not prefix and skey == "schema_version"):
            continue
        path = f"{prefix}{skey}"
        if _is_leaf(node):
            yield Param(path=path, spec=node)
        elif isinstance(node, Mapping):
            yield from _iter_leaves(node, prefix=path + ".")


def validate_tree(tree: Mapping[str, Any]) -> list[str]:
    """Return a list of human-readable problems with a registry tree (empty = valid)."""
    errors: list[str] = []

    def walk(node: Mapping[str, Any], prefix: str) -> None:
        for key, child in node.items():
            skey = str(key)
            if skey.startswith("_") or (not prefix and skey == "schema_version"):
                continue
            path = f"{prefix}{skey}"
            if _is_leaf(child):
                errors.extend(_validate_leaf(path, child))
            elif isinstance(child, Mapping):
                if not child:
                    errors.append(f"{path}: empty group")
                walk(child, path + ".")
            else:
                errors.append(f"{path}: expected a parameter leaf (with value/dist and source)")

    walk(tree, "")
    return errors


def _validate_leaf(path: str, leaf: Mapping[str, Any]) -> list[str]:
    errs: list[str] = []
    source = leaf.get("source")
    if not isinstance(source, str) or not source.strip():
        errs.append(f"{path}: missing `source:` (use 'ASSUMPTION — needs citation' if unknown)")
    if "value" in leaf and "dist" in leaf and leaf.get("dist") != "constant":
        errs.append(f"{path}: has both `value` and `dist`")
    dist = str(leaf.get("dist", "constant"))
    if dist not in _DIST_FIELDS:
        return [*errs, f"{path}: unknown dist {dist!r}"]
    missing = [f for f in _DIST_FIELDS[dist] if f not in leaf]
    if missing:
        return [*errs, f"{path}: {dist} missing fields {missing}"]
    try:
        if dist == "truncnorm":
            if float(leaf["sd"]) <= 0:
                errs.append(f"{path}: sd must be > 0")
            if not float(leaf["min"]) < float(leaf["max"]):
                errs.append(f"{path}: min must be < max")
            if not float(leaf["min"]) <= float(leaf["mean"]) <= float(leaf["max"]):
                errs.append(f"{path}: mean outside [min, max]")
        elif dist == "lognormal":
            if float(leaf["median"]) <= 0 or float(leaf["sigma"]) <= 0:
                errs.append(f"{path}: median and sigma must be > 0")
            if "min" in leaf and "max" in leaf and not float(leaf["min"]) < float(leaf["max"]):
                errs.append(f"{path}: min must be < max")
        elif dist == "uniform":
            if not float(leaf["min"]) < float(leaf["max"]):
                errs.append(f"{path}: min must be < max")
        elif dist == "categorical":
            p = leaf["p"]
            if not isinstance(p, Mapping) or not p:
                errs.append(f"{path}: `p` must be a non-empty mapping")
            else:
                probs = [float(v) for v in p.values()]
                if any(v < 0 for v in probs):
                    errs.append(f"{path}: negative probability")
                if abs(sum(probs) - 1.0) > 1e-6:
                    errs.append(f"{path}: probabilities sum to {sum(probs):.6f}, not 1")
    except (TypeError, ValueError) as exc:
        errs.append(f"{path}: non-numeric field ({exc})")
    return errs


_DEFAULT: Params | None = None


def get_params() -> Params:
    """Process-wide default registry (loaded lazily, cached)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Params.load()
    return _DEFAULT
