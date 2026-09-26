from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tailsafe.building.model import NodeType
from tailsafe.building.templates import generate
from tailsafe.scenarios.spec import Dist, ScenarioSpec, StairBlockage
from tailsafe.surrogate.data import (
    TYPOLOGIES,
    Case,
    load_dataset,
    run_case,
    sample_case,
    save_dataset,
)
from tailsafe.surrogate.features import (
    N_EDGE,
    N_GLOBAL,
    N_NODE,
    NODE_TYPES,
    graph_features,
    static_graph,
)

jax = pytest.importorskip("jax")

from tailsafe.surrogate.evaluate import evaluation_markdown, metrics, prepare  # noqa: E402
from tailsafe.surrogate.model import (  # noqa: E402
    LOSS_NAMES,
    N_OUT,
    QUANTILES,
    ModelConfig,
    TrainConfig,
    load_model,
    predict,
    save_model,
    train,
)
from tailsafe.surrogate.predictor import Surrogate  # noqa: E402


def test_sample_case_is_deterministic_and_valid() -> None:
    a, b = sample_case(7, seed=3), sample_case(7, seed=3)
    assert a.as_dict() == b.as_dict()
    assert sample_case(8, seed=3).as_dict() != a.as_dict()
    for i in range(40):
        c = sample_case(i)
        lo_hi = TYPOLOGIES[c.template]
        for k, v in c.options.items():
            assert lo_hi[k][0] <= v <= lo_hi[k][1]
        if c.template == "twin_core":
            assert c.options["flats_per_floor"] % 2 == 0
        assert Case.from_dict(c.as_dict()).as_dict() == c.as_dict()


def test_static_graph_folds_units_into_circulation() -> None:
    b = generate("slab", storeys=6, flats_per_side=4, lifts=1)
    g = static_graph(b)
    units = [n for n in b.nodes if n.type == NodeType.UNIT]
    assert len(g.node_ids) == len(b.nodes) - len(units)
    assert g.node_base.shape == (len(g.node_ids), N_NODE)
    assert g.edge_base.shape == (g.edge_src.size, N_EDGE)
    assert g.n_units == len(units)
    o = len(NODE_TYPES)
    # every flat is counted on the corridor it opens onto
    assert np.isclose(g.node_base[:, o + 3].sum() * 10.0, len(units))
    # both directions of each circulation edge, one marked forward
    assert g.edge_forward.sum() * 2 == g.edge_src.size


def test_scenario_features() -> None:
    b = generate("cruciform", storeys=10, flats_per_wing=2, lifts=2)
    g = static_graph(b)
    stair = b.stairs[0].id
    base = graph_features(g, ScenarioSpec(name="a"))
    spec = ScenarioSpec(
        name="b",
        fire_level=5,
        warden_levels=[5],
        stair_blockages=[StairBlockage(stair=stair, time=Dist.fixed(300.0))],
        evacuation_lifts=True,
    )
    x = graph_features(g, spec)
    o = len(NODE_TYPES)
    lv = g.node_level
    assert x.global_x.shape == (N_GLOBAL,)
    assert (x.node_x[lv == 5, o + 7] == 1).all() and (x.node_x[lv != 5, o + 7] == 0).all()
    assert x.node_x[:, o + 8].sum() == (lv == 5).sum()
    assert x.node_x[:, o + 10].sum() > 0 and base.node_x[:, o + 10].sum() == 0
    assert x.edge_x[:, 9].sum() > 0
    assert x.node_x[:, o + 12].sum() == g.lift_lobby.sum() > 0
    # the building part is shared, not modified
    assert (g.node_base[:, o + 7 :] == 0).all()


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    recs = [run_case(sample_case(i, typologies=("slab", "care_home")), n_runs=6) for i in range(6)]
    path = save_dataset(recs, tmp_path_factory.mktemp("sg") / "cases.jsonl")
    back = load_dataset(path)
    assert back == [__import__("json").loads(__import__("json").dumps(r)) for r in recs]
    return back


def test_run_case_record(tiny_data: list[dict[str, Any]]) -> None:
    r = tiny_data[0]
    losses = r["losses"]
    assert isinstance(losses, dict)
    assert set(losses) == set(LOSS_NAMES)
    assert all(len(v) == 6 for v in losses.values())
    assert isinstance(r["edge_queue"], dict)


@pytest.fixture(scope="module")
def trained(
    tiny_data: list[dict[str, Any]], tmp_path_factory: pytest.TempPathFactory
) -> tuple[Any, ...]:
    graphs, targets, meta = prepare(tiny_data)
    assert {m["template"] for m in meta} <= {"slab", "care_home"}
    mcfg = ModelConfig(hidden=16, layers=2)
    net, stats, hist = train(
        graphs[:5],
        targets[:5],
        graphs[5:],
        targets[5:],
        mcfg=mcfg,
        tcfg=TrainConfig(epochs=6, batch_graphs=4),
    )
    assert len(hist) == 6
    assert hist[-1]["train"] < hist[0]["train"]
    path = tmp_path_factory.mktemp("w") / "s.npz"
    save_model(path, net, stats, mcfg, {"trained_on": {"slab": 3}, "evaluation": None})
    return path, net, stats, graphs, targets


def test_predictions_are_ordered_and_reload_identically(trained: tuple[Any, ...]) -> None:
    path, net, stats, graphs, _ = trained
    q, e = predict(net, stats, graphs[:3])
    assert q.shape == (3, len(LOSS_NAMES), N_OUT)
    # quantiles are monotone and CVaR95 is at least P95
    assert (np.diff(q[..., : len(QUANTILES)], axis=-1) >= 0).all()
    assert (q[..., -1] >= q[..., len(QUANTILES) - 1] - 1e-6).all()
    assert all((x >= 0).all() for x in e)
    net2, stats2, _, meta = load_model(path)
    q2, _ = predict(net2, stats2, graphs[:3])
    assert np.allclose(q, q2)
    assert meta["trained_on"] == {"slab": 3}


def test_metrics_and_markdown(trained: tuple[Any, ...]) -> None:
    _, net, stats, graphs, targets = trained
    m = metrics(net, stats, graphs, targets)
    assert m["cases"] == len(graphs)
    for name in LOSS_NAMES:
        cov = m["losses"][name]["coverage"]
        assert all(0.0 <= v <= 1.0 for v in cov.values())
    md = evaluation_markdown({"random": m, "holdout": {"slab": m}})
    assert "Unseen: slab" in md and "Coverage" in md


def test_surrogate_predictor(trained: tuple[Any, ...]) -> None:
    path = trained[0]
    s = Surrogate(path)
    b = generate("slab", storeys=6, flats_per_side=4, lifts=1)
    out = s.predict(b, ScenarioSpec(name="x", fire_level=3), top_edges=4)
    assert set(out["losses"]) == set(LOSS_NAMES)
    for v in out["losses"].values():
        assert v["p50"] <= v["p75"] <= v["p90"] <= v["p95"] <= v["cvar95"] + 1e-6
    assert 0 < len(out["edges"]) <= 4
    assert out["model"]["trained_on"] == {"slab": 3}
    with pytest.raises(FileNotFoundError):
        Surrogate(path.with_name("missing.npz"))


def test_coverage_notes() -> None:
    from tailsafe.scenarios.spec import RandomStairBlockage, reference_spec
    from tailsafe.surrogate.predictor import coverage_notes

    assert coverage_notes(generate("cruciform", storeys=40), reference_spec()) == []
    notes = coverage_notes(
        generate("cruciform", storeys=45, wing_end_stairs=True),
        reference_spec().model_copy(
            update={
                "phased_release": {3: 60.0},
                "random_stair_blockage": RandomStairBlockage(probability=0.5),
                "share_65_plus": 0.6,
            }
        ),
    )
    text = " ".join(notes)
    assert "storeys = 45" in text and "wing end stairs" in text
    assert "phased release" in text and "random staircase loss" in text
    assert "65+" in text
    b = generate("slab", storeys=10)
    uploaded = b.model_copy(update={"metadata": {}})
    assert "not one of the four" in coverage_notes(uploaded, ScenarioSpec(name="x"))[0]
