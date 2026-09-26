"""Deterministic random streams.

Every stochastic component receives a generator derived from
``(base_seed, scenario_index, stream_name)`` through NumPy's ``SeedSequence``
spawn keys. Named streams keep components independent, so changing how one
component consumes randomness (for example an intervention that alters lift
use) never shifts the draws of another. This is what makes *common random
numbers* work: two interventions evaluated on scenario ``i`` see identical
occupants, pre-movement times and fire locations.
"""

from __future__ import annotations

import zlib

import numpy as np

STREAMS = ("occupancy", "speeds", "premovement", "behaviour", "scenario", "hazard")


def stream_id(name: str) -> int:
    """Stable integer id for a stream name (CRC-32, independent of Python's hash seed)."""
    return zlib.crc32(name.encode("utf-8"))


def stream(seed: int, name: str, index: int = 0) -> np.random.Generator:
    """Generator for stream ``name`` of scenario ``index`` under ``seed``."""
    if seed < 0 or index < 0:
        raise ValueError("seed and index must be non-negative")
    ss = np.random.SeedSequence(entropy=seed, spawn_key=(index, stream_id(name)))
    return np.random.default_rng(ss)


class Streams:
    """Named generators for one scenario: ``Streams(seed, i)["occupancy"]``."""

    def __init__(self, seed: int, index: int = 0) -> None:
        self.seed = seed
        self.index = index
        self._cache: dict[str, np.random.Generator] = {}

    def __getitem__(self, name: str) -> np.random.Generator:
        if name not in self._cache:
            self._cache[name] = stream(self.seed, name, self.index)
        return self._cache[name]
