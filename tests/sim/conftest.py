from __future__ import annotations

import pytest

from tailsafe.building.model import Building
from tailsafe.building.templates import generate
from tailsafe.sim.network import SimNetwork, compile_network


@pytest.fixture(scope="session")
def slab10() -> Building:
    return generate("slab", storeys=10, flats_per_side=4)


@pytest.fixture(scope="session")
def slab10_net(slab10: Building) -> SimNetwork:
    return compile_network(slab10)


@pytest.fixture(scope="session")
def cruciform12() -> Building:
    return generate("cruciform", storeys=12, flats_per_wing=3)
