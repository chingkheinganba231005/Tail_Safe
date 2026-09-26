"""The parameter registry must be valid and every value must carry a source."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from tailsafe.config import Params, ParamsError, default_params_path, validate_tree


@pytest.fixture(scope="module")
def params() -> Params:
    return Params.load()


def test_default_registry_is_the_repo_file() -> None:
    assert default_params_path().name == "params.yaml"
    assert default_params_path().exists()


def test_every_leaf_has_a_source(params: Params) -> None:
    leaves = list(params.leaves())
    assert len(leaves) > 50
    for leaf in leaves:
        assert leaf.source.strip(), leaf.path


def test_assumptions_are_flagged_verbatim(params: Params) -> None:
    for leaf in params.assumption_report():
        assert "ASSUMPTION" in leaf.source
        assert "needs citation" in leaf.source


def test_missing_source_is_rejected() -> None:
    errors = validate_tree({"a": {"b": {"value": 1.0}}})
    assert errors and "source" in errors[0]


def test_bad_categorical_is_rejected() -> None:
    errors = validate_tree({"x": {"dist": "categorical", "p": {1: 0.5, 2: 0.4}, "source": "s"}})
    assert any("sum" in e for e in errors)


def test_unknown_lookup_raises(params: Params) -> None:
    with pytest.raises(ParamsError):
        params["does.not.exist"]


def test_truncnorm_respects_bounds_and_seed(params: Params) -> None:
    p = params["profiles.frail_older_adult.stair_down_speed"]
    a = p.sample(np.random.default_rng(7), 10_000)
    b = p.sample(np.random.default_rng(7), 10_000)
    np.testing.assert_array_equal(a, b)
    assert a.min() >= p.spec["min"] and a.max() <= p.spec["max"]
    assert abs(a.mean() - p.spec["mean"]) < 0.02


def test_lognormal_median(params: Params) -> None:
    p = params["premovement.asleep"]
    x = p.ppf(np.array([0.5]))
    # Truncation is mild, so the median is close to the nominal median.
    assert x[0] == pytest.approx(p.spec["median"], rel=0.05)


def test_categorical_frequencies(params: Params) -> None:
    p = params["population.household_size.flat_small"]
    draws = p.sample(np.random.default_rng(0), 50_000)
    outcomes, probs = p.categories
    for outcome, prob in zip(outcomes, probs, strict=True):
        assert np.mean(draws == outcome) == pytest.approx(prob, abs=0.01)


def test_overrides_do_not_mutate_original(params: Params) -> None:
    changed = params.with_overrides(
        {"behaviour.counter_flow_probability": 0.2, "premovement.asleep": {"median": 600.0}}
    )
    assert changed.value("behaviour.counter_flow_probability") == 0.2
    assert changed["premovement.asleep"].spec["median"] == 600.0
    assert params.value("behaviour.counter_flow_probability") == 0.05
    assert changed.digest != params.digest
    with pytest.raises(ParamsError):
        params.with_overrides({"premovement.asleep": 3.0})


def test_registry_roundtrip_through_yaml(tmp_path: Path, params: Params) -> None:
    out = tmp_path / "p.yaml"
    out.write_text(yaml.safe_dump(params.as_dict()), encoding="utf-8")
    again = Params.load(out)
    assert again.digest == params.digest
