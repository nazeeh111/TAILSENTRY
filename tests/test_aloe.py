"""Independent analytic checks of union weighting and exact truncation."""

import numpy as np
import pytest
from scipy.special import ndtr

from tailsentry.aloe import aloe, run_projected_aloe


class HalfspaceModel:
    scale_df = None

    def __init__(self, directions, offsets=None):
        self.directions = np.asarray(directions, dtype=float)
        self.dimension = self.directions.shape[1]
        self.offsets = np.zeros(len(self.directions)) if offsets is None else np.asarray(offsets)

    def loss(self, z):
        return np.max(z @ self.directions.T + self.offsets, axis=1)

    def halfspaces(self, threshold):
        return self.directions, threshold - self.offsets


def test_disjoint_two_sided_tail_has_exact_weight_at_probability_1e_minus_5():
    model = HalfspaceModel([[1.0], [-1.0]])
    threshold = 4.417173413469023
    result = aloe(model, threshold, 200, seed=17)
    truth = 2 * ndtr(-threshold)
    assert result["estimate"] == pytest.approx(truth, rel=1e-13)
    assert result["se"] < truth * 1e-14
    assert np.all(np.abs(result["samples"][:, 0]) > threshold)


def test_duplicate_events_are_corrected_by_overlap_count():
    model = HalfspaceModel([[1.0], [1.0], [1.0]])
    result = aloe(model, 3.9, 200, seed=9)
    assert result["estimate"] == pytest.approx(ndtr(-3.9), rel=1e-13)
    assert result["diagnostics"]["minimum_overlap"] == 3


def test_overlapping_independent_events_match_analytic_union():
    model = HalfspaceModel(np.eye(2))
    result = aloe(model, 1.0, 16000, seed=7)
    tail = ndtr(-1.0)
    truth = 2 * tail - tail**2
    assert abs(result["estimate"] - truth) < 5 * result["se"]
    assert result["diagnostics"]["maximum_overlap"] == 2
    assert result["evaluations"] == 16000


def test_non_unit_direction_and_offset():
    model = HalfspaceModel([[2.0]], [0.5])
    result = aloe(model, 6.5, 200, seed=3)
    assert result["estimate"] == pytest.approx(ndtr(-3.0), rel=1e-13)


def test_extreme_tail_truncation_uses_log_survival():
    model = HalfspaceModel([[1.0]])
    result = aloe(model, 30.0, 100, seed=2)
    assert np.all(np.isfinite(result["samples"]))
    assert np.all(result["samples"][:, 0] > 30.0)
    assert result["estimate"] == pytest.approx(ndtr(-30.0), rel=1e-12, abs=0)


def test_scale_mixture_cannot_silently_use_gaussian_weights():
    model = HalfspaceModel([[1.0]])
    model.scale_df = 5
    with pytest.raises(ValueError, match="Gaussian"):
        aloe(model, 2.0, 100)


def test_projecting_complete_one_dimensional_event_is_exact():
    model = HalfspaceModel([[1.0], [-1.0]])
    result = run_projected_aloe(model, 300, 6, 3.0, direction=[-2.0])
    assert result["estimate"] == pytest.approx(2 * ndtr(-3.0), rel=1e-13)
    assert result["se"] < 1e-16


def test_projection_handles_certain_event_and_overlapping_tails():
    model = HalfspaceModel([[1.0], [-1.0]])
    result = run_projected_aloe(model, 300, 10, -1.0, direction=[1.0])
    assert result["estimate"] == pytest.approx(1.0, rel=1e-13)


def test_projected_independent_union_matches_two_exact_conditional_values():
    model = HalfspaceModel(np.eye(2))
    result = run_projected_aloe(model, 2000, 31, 2.0, direction=[1.0, 0.0])
    tail = ndtr(-2.0)
    v2 = result["projected_samples"][:, 1]
    expected = np.where(v2 > 2.0, 2 * tail / (1 + tail), 2 * tail)
    assert np.allclose(result["observations"], expected, atol=0, rtol=1e-13)
    assert np.all(result["observations"] <= result["bound"])


def test_projection_zero_direction_rejected():
    with pytest.raises(ValueError, match="positive"):
        run_projected_aloe(HalfspaceModel(np.eye(2)), 100, 1, 2.0, [0.0, 0.0])
