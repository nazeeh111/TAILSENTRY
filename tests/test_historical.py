"""Check historical diagnostic mathematics independently of market outcomes."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats


spec = importlib.util.spec_from_file_location("validate_fx", Path(__file__).resolve().parents[1] / "scripts" / "validate_fx.py")
fx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fx)


@pytest.mark.parametrize("model", [
    {"family": "gaussian", "location": 0.1, "scale": 0.7},
    {"family": "student_t", "location": -0.2, "scale": 0.4, "df": 5.3},
])
@pytest.mark.parametrize("confidence", [0.95, 0.975, 0.99])
def test_analytic_es_matches_independent_tail_integration(model, confidence):
    q, es = fx.analytic_risk(model, confidence)
    distribution = stats.norm(loc=model["location"], scale=model["scale"]) if model["family"] == "gaussian" else stats.t(model["df"], loc=model["location"], scale=model["scale"])
    probability = integrate.quad(distribution.pdf, q, np.inf, epsabs=1e-11)[0]
    first_moment = integrate.quad(lambda y: y * distribution.pdf(y), q, np.inf, epsabs=1e-11)[0]
    assert probability == pytest.approx(1 - confidence, rel=1e-8)
    assert es == pytest.approx(first_moment / probability, rel=1e-8)
    # Under a correctly specified distribution both the ES residual mean and
    # the upper-loss FZ0 score expectation have independent analytic values.
    excess = first_moment - q * probability
    assert es - q - excess / (1 - confidence) == pytest.approx(0, abs=1e-8)
    assert excess / ((1 - confidence) * es) + q / es + np.log(es) - 1 == pytest.approx(np.log(es), abs=1e-8)


def test_bootstrap_keeps_paired_relations_and_fixed_seed():
    x = np.arange(100, dtype=float)
    values = np.column_stack([x, x + 7])
    first = fx.circular_block_bootstrap_means(values, block_length=20, replications=30, seed=83)
    second = fx.circular_block_bootstrap_means(values, block_length=20, replications=30, seed=83)
    assert np.array_equal(first, second)
    assert np.allclose(first[:, 1] - first[:, 0], 7)


def test_saved_data_are_verified_and_partitions_do_not_overlap():
    root = Path(__file__).resolve().parents[1]
    train, evaluation, _ = fx.read_verified_returns(root / "data")
    assert len(train) == 3769
    assert len(evaluation) == 2748
    assert max(row["date"] for row in train) < min(row["date"] for row in evaluation)
    assert all(row["date"] <= "2014-12-31" for row in train)
    assert all(row["date"] >= "2015-01-01" for row in evaluation)
