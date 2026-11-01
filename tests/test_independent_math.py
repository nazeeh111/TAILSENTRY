"""Independent review checks using numerical integration and density identities.

Written by a reviewer who did not implement the estimator, models, conditional
integrator or uncertainty module. The quadrature oracle does not call their
root solver or interval construction.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm, t as student

from tailsentry import conditional, engine
from tailsentry.envelope import build_cover, projected_values, sample_cover, estimate_envelope
from tailsentry.models import QuadraticModel, normal_max
from tailsentry.references import normal_max_survival
from tailsentry.uncertainty import precision_status, summarize_observations


def independent_quadrature(a, b, c, threshold):
    """Companion-matrix roots + adaptive quadrature of the original integrand."""
    edges = []
    polys = [np.array([aa, bb, cc - threshold]) for aa, bb, cc in zip(a, b, c)]
    polys += [np.array([a[j]-a[k], b[j]-b[k], c[j]-c[k]]) for j in range(len(a)) for k in range(j)]
    for poly in polys:
        nonzero = np.flatnonzero(poly)
        if len(nonzero):
            # A companion eigensolve can spuriously split an exact double
            # root, so use its algebraic location in that exact special case.
            if poly[0] != 0 and poly[1]**2-4*poly[0]*poly[2] == 0:
                edges.append(float(-poly[1]/(2*poly[0])))
                continue
            for root in np.roots(poly[nonzero[0]:]):
                if abs(root.imag) < 1e-10:
                    edges.append(float(root.real))
    edges = [-np.inf] + sorted(set(edges)) + [np.inf]

    def loss(x):
        return float(np.max((a*x+b)*x+c))

    p = moment = 0.0
    for lo, hi in zip(edges, edges[1:]):
        probe = (lo+hi)/2 if np.isfinite(lo) and np.isfinite(hi) else lo+1 if np.isfinite(lo) else hi-1 if np.isfinite(hi) else 0.
        # Event membership is constant inside the root-separated interval.
        # Re-evaluating a near-tangent polynomial at every quadrature node can
        # introduce a spurious floating-point zero band around its double root.
        if loss(probe) > threshold:
            p += quad(norm.pdf, lo, hi, epsabs=1e-12, epsrel=1e-11)[0]
            moment += quad(lambda x: loss(x)*norm.pdf(x), lo, hi, epsabs=1e-12, epsrel=1e-11)[0]
    return p, moment


@pytest.mark.parametrize("a,b,c,threshold", [
    ([0], [0], [0], 0),
    ([0], [0], [2], 1),
    ([1], [-2], [1], 0),
    ([-1], [2], [-1], 0),
    ([0], [1], [0], 2),
    ([0], [-1], [0], 2),
    ([1], [0], [-4], 0),
    ([-1], [0], [4], 0),
    ([-1,-1], [-4,4], [-3,-3], 0),
    ([1,-2,0], [-.3,2,-1], [-4,1,.5], .3),
])
def test_all_interval_and_envelope_cases_against_quadrature(a,b,c,threshold):
    a,b,c = (np.asarray(x,float) for x in (a,b,c))
    p,moment = independent_quadrature(a,b,c,threshold)
    assert conditional.probability(a[None],b[None],c[None],threshold)[0] == pytest.approx(p,abs=2e-11)
    assert conditional.tail_moment(a[None],b[None],c[None],threshold)[0] == pytest.approx(moment,abs=3e-10)


def test_random_envelopes_against_independent_quadrature():
    rng = np.random.default_rng(802103)
    for k in (1,2,4,6):
        for _ in range(5):
            a,b,c = rng.normal(size=(3,k))
            threshold = float(rng.normal())
            p,moment = independent_quadrature(a,b,c,threshold)
            assert conditional.probability(a[None],b[None],c[None],threshold)[0] == pytest.approx(p,abs=3e-11)
            assert conditional.tail_moment(a[None],b[None],c[None],threshold)[0] == pytest.approx(moment,abs=4e-10)


@pytest.mark.parametrize("scale", [1e-200,1.,1e200])
def test_quadratic_event_is_invariant_to_finite_positive_scaling(scale):
    # A finite coefficient input must not silently turn a two-tail event into
    # the whole real line because its unscaled discriminant under/overflows.
    roots = np.roots([1.,1.,-1.])
    exact = norm.cdf(min(roots))+norm.sf(max(roots))
    actual = conditional.probability(np.array([[scale]]),np.array([[scale]]),np.array([[-scale]]),0)[0]
    assert actual == pytest.approx(exact,rel=2e-13)


@pytest.mark.parametrize("df", [None,3.,8.])
def test_conditional_polynomial_matches_direct_multivariate_loss(df):
    rng = np.random.default_rng(1103)
    d,k = 4,3
    raw = rng.normal(size=(k,d,d))
    quadratic = (raw + raw.swapaxes(1,2))/2
    model = QuadraticModel(quadratic,rng.normal(size=(k,d)),rng.normal(size=k),scale_df=df)
    e = rng.normal(size=model.dimension)
    if df is not None:
        e[-1] = 0
    e /= np.linalg.norm(e)
    v = rng.normal(size=(15,model.dimension))
    v -= (v@e)[:,None]*e
    a,b,c = model.conditional_coefficients(v,e)
    for x in (-5.,-1.2,0.,.5,3.7):
        assert np.allclose((a*x+b)*x+c,model.components(v+x*e),rtol=4e-13,atol=2e-12)


def test_accepted_nearly_symmetric_model_preserves_quadratic_form():
    # Rejecting this input is acceptable; accepting it requires symmetrization
    # or a conditional formula using B+B.T rather than assuming exact symmetry.
    quadratic = np.array([[[0.,1e6],[1e6+1,0.]]])
    try:
        model = QuadraticModel(quadratic,np.zeros((1,2)),np.zeros(1))
    except ValueError:
        return
    v = np.array([[0.,1.]])
    e = np.array([1.,0.])
    a,b,c = model.conditional_coefficients(v,e)
    assert ((a+b)+c)[0,0] == pytest.approx(model.components(v+e)[0,0],abs=1e-9)


def test_projected_likelihood_ratio_matches_explicit_lower_dimensional_density():
    rng = np.random.default_rng(9021)
    d = 5
    means = rng.normal(size=(4,d))
    e = rng.normal(size=d); e /= np.linalg.norm(e)
    # Independent orthonormal basis for the orthogonal complement.
    _,_,vh = np.linalg.svd(e[None],full_matrices=True)
    basis = vh[1:].T
    coords = rng.normal(size=(27,d-1))
    v = coords@basis.T
    epsilon = .2
    locations = np.vstack((np.zeros(d),means))@basis
    alpha = np.r_[epsilon,np.repeat((1-epsilon)/len(means),len(means))]
    numerator = multivariate_normal.logpdf(coords,mean=np.zeros(d-1))
    denominator = logsumexp(np.stack([np.log(w)+multivariate_normal.logpdf(coords,mean=mu) for w,mu in zip(alpha,locations)]),axis=0)
    actual = engine.log_ratio(v,means,epsilon,e)
    assert np.allclose(actual,numerator-denominator,rtol=2e-13,atol=2e-13)
    assert np.all(np.exp(actual) <= 1/epsilon*(1+1e-13))


def test_independent_streams_isolate_confirmation_from_extra_pilot_draws(monkeypatch):
    first = engine.estimate(normal_max(3),3.5,budget=1024,seed=781,return_samples=True)
    original = engine.train

    def waste_extra_pilot_randomness(*args,**kwargs):
        result = original(*args,**kwargs)
        args[2].random(10_000)
        return result

    monkeypatch.setattr(engine,"train",waste_extra_pilot_randomness)
    second = engine.estimate(normal_max(3),3.5,budget=1024,seed=781,return_samples=True)
    assert np.array_equal(first["_observations"],second["_observations"])
    assert first["point_evaluations"]+first["conditional_evaluations"] == 1024
    assert second["point_evaluations"]+second["conditional_evaluations"] == 1024


def test_frozen_estimator_mean_against_independent_normal_truth():
    observed = [engine.estimate(normal_max(3),3.2,budget=1024,seed=18100+j)["estimate"] for j in range(40)]
    truth = normal_max_survival(3.2,3)
    standard_error = np.std(observed,ddof=1)/np.sqrt(len(observed))
    assert abs(np.mean(observed)-truth) < 6*standard_error+1e-10


def test_student_auxiliary_conditioning_against_exact_student_tail():
    model = QuadraticModel(np.zeros((1,1,1)),np.ones((1,1)),np.zeros(1),scale_df=5.)
    threshold = student.isf(.01,5)
    observed = [engine.estimate(model,threshold,budget=2048,seed=900+j,method="conditional")["estimate"] for j in range(32)]
    standard_error = np.std(observed,ddof=1)/np.sqrt(len(observed))
    assert abs(np.mean(observed)-.01) < 6*standard_error+1e-10


def test_precision_status_does_not_accept_empty_confidence_intersection():
    # Unusual but possible finite-sample observations of a bounded IS variable.
    # The probability parameter remains in [0,1]; an empty set is not a narrow
    # valid precision statement. Inputs must be rejected or the status abstain.
    result = summarize_observations(np.repeat(2.,10000),bound=2.)
    status = precision_status(result["interval"],.1)
    assert status["status"] != "precision-supported" or result["interval"][0] <= result["interval"][1]


def test_cover_does_not_misclassify_scaled_anisotropic_ellipsoid_as_empty_sphere():
    model = QuadraticModel(np.array([[[-1e-14,0.],[0.,-4e-14]]]),np.array([[8e-14,0.]]),np.array([-15e-14]))
    point = np.array([[4.,0.]])
    assert model.loss(point)[0] > 0
    cover = build_cover(model,0)
    assert np.any(point@cover["normals"].T > cover["cutoffs"])


@pytest.mark.parametrize("scale", [1e-200,1.,1e200])
def test_halfspace_cover_is_invariant_to_finite_coefficient_scaling(scale):
    model = QuadraticModel(np.zeros((1,2,2)),np.array([[scale,0.]]),np.zeros(1))
    cover = build_cover(model,scale)
    assert cover["total"] == pytest.approx(norm.sf(1.),rel=1e-13)
    assert np.allclose(cover["normals"],[[1.,0.]])


def test_cover_contains_random_gaussian_and_student_quadratic_events():
    rng = np.random.default_rng(71341)
    for df in (None,4.):
        d,k = 4,3
        raw = rng.normal(size=(k,d,d))
        model = QuadraticModel((raw+raw.swapaxes(1,2))/3,rng.normal(size=(k,d)),rng.normal(size=k),scale_df=df)
        threshold = 4.
        cover = build_cover(model,threshold)
        z = rng.normal(size=(10000,model.dimension))
        x = z[:,:d]*model.scale(z)[:,None]
        covered = np.any(x@cover["normals"].T > cover["cutoffs"],axis=1)
        event = model.loss(z) > threshold
        assert np.all(covered[event])


def test_projected_cover_does_not_round_small_nonzero_slope_to_zero():
    model = QuadraticModel(np.zeros((1,2,2)),np.array([[1.,0.]]),np.zeros(1))
    cover = build_cover(model,1.)
    e = np.array([1e-15,1.])
    z = np.array([[1.,-1e-15]])
    values,_,_,_ = projected_values(model,cover,z,e,1.)
    # Both the event and cover are this one halfspace: their conditional
    # probabilities cancel, giving a constant, even at its projected boundary.
    assert values[0] == pytest.approx(norm.sf(1.),rel=1e-13)


def test_underflowed_cover_component_retains_conservative_support():
    model = normal_max(2,offsets=np.array([0.,-40.]))
    cover=build_cover(model,0.)
    original_normals,original_cuts=model.halfspaces(0.)
    assert np.all(cover['masses']>0)
    # Every original constraint must be contained in a retained, same-normal
    # halfspace with no higher cutoff. This tests support, not a chosen cap.
    for normal,cut in zip(original_normals,original_cuts):
        matches=np.all(cover['normals']==normal,axis=1)
        assert np.any(matches)
        assert np.any(cover['cutoffs'][matches]<=cut)


@pytest.mark.parametrize("df", [None,5.])
def test_cover_conditional_halfspace_sampler_matches_truncated_student_or_normal(df):
    rng = np.random.default_rng(18521)
    model = QuadraticModel(np.zeros((1,3,3)),np.array([[1.,0.,0.]]),np.zeros(1),scale_df=df)
    cut = 2.
    cover = build_cover(model,cut)
    latent = sample_cover(model,cover,20000,rng)
    x = latent[:,:3]*model.scale(latent)[:,None]
    assert np.all(x[:,0] > cut)
    # Under a correct truncated marginal, its conditional survival transform
    # is uniform. The orthogonal Gaussian factors remain independent of it.
    marginal = norm if df is None else student(df)
    uniforms = marginal.sf(x[:,0])/marginal.sf(cut)
    assert abs(uniforms.mean()-.5) < .012
    assert abs(np.mean(uniforms**2)-1/3) < .012
    assert np.all(np.abs(latent[:,1:3].mean(axis=0)) < .035)
    assert np.all(np.abs(latent[:,1:3].var(axis=0)-1) < .05)
    assert abs(np.corrcoef(latent[:,1],uniforms)[0,1]) < .035


def test_single_affine_cover_projection_has_constant_exact_probability():
    model = QuadraticModel(np.zeros((1,3,3)),np.array([[.3,-.4,.5]]),np.zeros(1))
    threshold = 2.
    out = estimate_envelope(model,threshold,budget=512,seed=824,return_samples=True)
    truth = norm.sf(threshold/np.linalg.norm(model.linear[0]))
    assert np.allclose(out["_observations"],truth,rtol=2e-12,atol=1e-16)
    assert out["interval"][0] <= truth <= out["interval"][1]


@pytest.mark.parametrize("estimator", [engine.estimate,estimate_envelope])
def test_student_infinite_quadratic_tail_mean_is_not_reported_as_finite_supported_severity(estimator):
    model = QuadraticModel(np.ones((1,1,1)),np.zeros((1,1)),np.zeros(1),scale_df=1.5)
    try:
        out = estimator(model,1.,budget=512,seed=732,severity=True)
    except ValueError:
        return
    # Returning infinity or abstaining is acceptable; a finite sample average
    # must not silently stand in for a nonexistent population tail expectation.
    assert out.get("conditional_severity") is None or np.isinf(out["conditional_severity"])


@pytest.mark.parametrize("estimator", [engine.estimate,estimate_envelope])
def test_student_infinite_severity_variance_does_not_receive_standard_error(estimator):
    model = QuadraticModel(np.ones((1,1,1)),np.zeros((1,1)),np.zeros(1),scale_df=3.)
    out = estimator(model,1.,budget=512,seed=1732,severity=True)
    assert out["tail_first_moment_se"] is None
    assert out["severity_support"]["mean_supported"]
    assert not out["severity_support"]["variance_supported"]


@pytest.mark.parametrize("alpha", [0.,-0.1,1.,1.1,np.nan])
def test_envelope_rejects_invalid_confidence_input(alpha):
    with pytest.raises(ValueError):
        estimate_envelope(normal_max(2),2.,budget=256,alpha=alpha)
