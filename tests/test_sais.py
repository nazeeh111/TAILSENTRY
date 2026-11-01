"""Independent numerical contracts for the isolated SAIS reproduction."""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from tailsentry.sais import (
    _dm_logweights, _draw_confirmation, _draw_stratified, _log_components,
    _recycle, _threshold, _update_component, estimate_sais,
)


class LossModel:
    def __init__(self, dimension=2, function=None):
        self.dimension = dimension
        self.evaluations = 0
        self.function = function or (lambda z: z[:, 0])

    def loss(self, z):
        self.evaluations += len(z)
        return self.function(z)


def test_full_covariance_density_matches_scipy():
    from scipy.stats import multivariate_normal
    x = np.array([[0.0, 1.0], [3.0, -2.0], [-1.0, 0.2]])
    means = np.array([[0.3, -0.8], [-1.0, 2.0]])
    cov = np.array([[[2.0, 0.6], [0.6, 1.0]], [[1.1, -0.3], [-0.3, 0.8]]])
    actual = _log_components(x, means, cov)
    for j in range(2):
        np.testing.assert_allclose(actual[:, j], multivariate_normal.logpdf(x, means[j], cov[j]), atol=1e-13)


def test_dm_weights_use_the_full_mixture_at_same_point():
    means = np.array([[-2.0], [2.0]])
    cov = np.ones((2, 1, 1))
    x = np.array([[-3.0], [-1.0], [0.0], [1.0], [3.0]])
    logs, assignments = _dm_logweights(x, means, cov)
    target = norm.pdf(x[:, 0])
    proposal = 0.5 * (norm.pdf(x[:, 0] + 2) + norm.pdf(x[:, 0] - 2))
    np.testing.assert_allclose(np.exp(logs), target / proposal, rtol=1e-13)
    np.testing.assert_array_equal(assignments, [0, 0, 0, 1, 1])
    # Independent one-dimensional quadrature validates the density correction.
    integrand = lambda value: (norm.pdf(value) / (0.5 * (norm.pdf(value + 2) + norm.pdf(value - 2)))) * (0.5 * (norm.pdf(value + 2) + norm.pdf(value - 2)))
    integral, error = quad(integrand, -8, 8, epsabs=1e-12)
    assert abs(integral - 1) < 1e-10


def test_deterministic_component_counts_and_correct_moments():
    means = np.array([[-2.0], [1.5]])
    cov = np.array([[[0.7]], [[1.3]]])
    x, labels = _draw_stratified(np.random.default_rng(410001), means, cov, 4000)
    np.testing.assert_array_equal(np.bincount(labels), [4000, 4000])
    assert abs(x[labels == 0].mean() + 2) < 0.06
    assert abs(x[labels == 1].var() - 1.3) < 0.1


def test_paper_two_stage_threshold_is_not_global_ce_quantile():
    scores = np.r_[np.arange(10.0), 100 + np.arange(10.0)]
    sources = np.repeat([0, 1], 10)
    value, diagnostic = _threshold(scores, sources, math.inf, 0.2, 2)
    # Local lowest two: [0,1] and [100,101]. Pooled DESCENDING [101,100,1,0].
    # floor(.2*4)=0 is guarded to rank one, producing 101.
    assert value == 101
    assert diagnostic["pooled_descending_rank"] == 1
    assert diagnostic["rank_guards"] == 1
    assert value != np.quantile(scores, 0.2)


def test_empty_local_subset_is_visible_and_finite():
    boundary, diagnostic = _threshold(np.array([2.0, 3.0]), np.array([0, 1]), 1.0, 0.1, 2)
    assert boundary == 1.0
    assert diagnostic["status"] == "no previous-subset samples"


def test_printed_old_mean_covariance_branch_and_isotropic_guard():
    mean, covariance, diagnostic = _update_component(
        np.array([[2.0], [4.0]]), np.array([-1.0, -2.0]), np.zeros(2),
        np.array([0.0]), np.ones((1, 1)), 0.0, 1, 1e-8,
    )
    np.testing.assert_allclose(mean, [3.0])
    np.testing.assert_allclose(diagnostic["empirical_covariance"], [[10.0]])
    # d=1 makes the printed anisotropy denominator zero: beta=1 guard.
    np.testing.assert_allclose(covariance, [[11.0]])
    assert diagnostic["beta_zero_denominator_guard"]
    assert not diagnostic["tempered"]


def test_tempered_branch_has_documented_sigmoid_and_new_mean_center():
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    raw = np.array([1000.0, 1.0, 1.0, 1.0])
    mean, covariance, diagnostic = _update_component(
        x, -np.ones(4), np.log(raw), np.array([10.0]), np.ones((1, 1)), 0.0, 1, 1e-8,
    )
    gamma = 1 / (1 + np.exp(-1))
    weight = raw ** gamma
    weight /= weight.sum()
    expected_mean = weight @ x
    expected_covariance = float(weight @ ((x[:, 0] - expected_mean[0]) ** 2))
    assert diagnostic["tempered"]
    assert diagnostic["gamma"] == pytest.approx(gamma)
    np.testing.assert_allclose(mean, expected_mean)
    assert diagnostic["empirical_covariance"][0][0] == pytest.approx(expected_covariance)
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_empty_component_keeps_previous_proposal():
    old_mean, old_cov = np.array([2.0, -1.0]), np.eye(2)
    mean, cov, diagnostic = _update_component(
        np.empty((0, 2)), np.empty(0), np.empty(0), old_mean, old_cov, 0.0, 1, 1e-8,
    )
    np.testing.assert_array_equal(mean, old_mean)
    np.testing.assert_array_equal(cov, old_cov)
    assert diagnostic["status"].startswith("retained")


def test_recycling_matches_normalized_geometric_weights():
    assert _recycle([1.0, 2.0, 3.0], 0.2) == pytest.approx((0.04 + 0.4 + 3) / 1.24)
    assert _recycle([0.0123], 0.2) == 0.0123


def test_frozen_confirmation_bound_and_density_correction():
    means = np.array([[-2.0], [2.0]])
    covariance = np.ones((2, 1, 1))
    x, weights = _draw_confirmation(np.random.default_rng(410002), means, covariance, 5000, 0.1)
    q = 0.1 * norm.pdf(x[:, 0]) + 0.45 * (norm.pdf(x[:, 0] + 2) + norm.pdf(x[:, 0] - 2))
    np.testing.assert_allclose(weights, norm.pdf(x[:, 0]) / q, rtol=1e-13)
    assert weights.max() <= 10 * (1 + 1e-14)
    estimate = np.mean(weights * (np.abs(x[:, 0]) > 2))
    assert abs(estimate - 2 * norm.sf(2)) < 0.004


def test_api_exact_budget_determinism_and_separated_recycling():
    first_model, second_model = LossModel(), LossModel()
    first = estimate_sais(first_model, 2.0, budget=2048, seed=410003, components=2, return_samples=True)
    second = estimate_sais(second_model, 2.0, budget=2048, seed=410003, components=2, return_samples=True)
    assert first_model.evaluations == second_model.evaluations == 2048
    assert first["pilot_evaluations"] + first["confirmation_samples"] == 2048
    np.testing.assert_array_equal(first["samples"], second["samples"])
    np.testing.assert_array_equal(first["observations"], second["observations"])
    assert first["estimate"] == np.mean(first["weights"] * (first["losses"] > 2))
    assert first["paper_recycled_estimate"] == _recycle(
        [level["paper_level_estimate"] for level in first["levels"]], first["forgetting"])
    assert first["method"] == "sais_frozen_confirmation"


def test_fixed_iteration_path_uses_prespecified_number_of_levels():
    out = estimate_sais(LossModel(), -10.0, budget=2048, seed=410004, components=2,
                        samples_per_component=32, max_iterations=3, stop_at_target=False)
    assert out["iterations"] == 3
    assert out["pilot_evaluations"] == 3 * 2 * 32
    assert not out["paper_stopping_random"]


def test_known_truth_with_identical_initial_target_proposals():
    # A=.5 event is already at the target, so one pilot level suffices.
    # This exercises known-truth estimation after an independently frozen fit.
    model = LossModel(dimension=1)
    out = estimate_sais(model, 0.0, budget=10000, seed=410005, components=2,
                        initial_means=np.zeros((2, 1)), initial_covariances=np.ones((2, 1, 1)))
    assert abs(out["estimate"] - 0.5) < 0.025
    low, high = out["finite_sample_interval"]
    assert low <= 0.5 <= high


@pytest.mark.parametrize("options", [
    {"budget": 100}, {"components": 0}, {"rho": 0}, {"rho": 1},
    {"alpha": float("nan")}, {"alpha": 1.0}, {"confirmation_defense": 0},
    {"pilot_fraction": 1}, {"samples_per_component": 100000},
])
def test_invalid_inputs_fail_explicitly(options):
    with pytest.raises(ValueError):
        estimate_sais(LossModel(), 2.0, **options)


def test_nonfinite_model_output_is_not_treated_as_safe():
    model = LossModel(function=lambda z: np.full(len(z), np.nan))
    with pytest.raises(ValueError, match="finite"):
        estimate_sais(model, 2.0, seed=410006)
