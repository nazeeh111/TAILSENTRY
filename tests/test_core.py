import numpy as np
import pytest
from scipy.stats import norm
from tailsentry.models import normal_max,balls,QuadraticModel
from tailsentry.conditional import probability,tail_moment,normal_interval
from tailsentry.engine import estimate,log_ratio
from tailsentry.uncertainty import zero_hit_upper,summarize_observations


def test_opposite_tails_and_compact_interval():
    a=np.array([[1.],[-1.]])
    b=np.zeros_like(a);c=np.array([[0.],[4.]])
    assert np.allclose(probability(a,b,c,1),[2*norm.sf(1),2*norm.cdf(np.sqrt(3))-1])


def test_max_envelope_severity():
    # L=|T|: exact unconditional and tail first moments.
    a=np.zeros((1,2));b=np.array([[1.,-1.]]);c=np.zeros((1,2))
    assert tail_moment(a,b,c,2)[0]==pytest.approx(2*norm.pdf(2),rel=1e-12)
    # L=T², both tails must be retained.
    assert tail_moment(np.ones((1,1)),np.zeros((1,1)),np.zeros((1,1)),4)[0]==pytest.approx(2*(2*norm.pdf(2)+norm.sf(2)))


def test_extreme_gaussian_interval():
    assert normal_interval(12,13)==pytest.approx(norm.sf(12)-norm.sf(13),rel=1e-13)


def test_projected_density_matches_marginal():
    means=np.array([[3.,4.],[-2.,1.]])
    v=np.array([[0.,2.],[0.,-1.]])
    ratio=np.exp(log_ratio(v,means,.2,np.array([1.,0.])))
    q=.2*norm.pdf(v[:,1])+.4*norm.pdf(v[:,1]-4)+.4*norm.pdf(v[:,1]-1)
    assert np.allclose(ratio,norm.pdf(v[:,1])/q)


def test_budget_seed_and_model_validation():
    m=normal_max(3,paired=True)
    first=estimate(m,3,budget=1024,seed=8)
    second=estimate(m,3,budget=1024,seed=8)
    assert first['estimate']==second['estimate']
    assert first['point_evaluations']+first['conditional_evaluations']==1024
    assert first['training_calls']+first['selection_calls']+first['confirmation_samples']==1024
    with pytest.raises(ValueError):estimate(m,3,budget=100)
    with pytest.raises(ValueError):QuadraticModel(np.zeros((1,2,2)),np.ones((1,3)),np.zeros(1))


def test_finite_bound_refuses_false_precision():
    r=summarize_observations(np.full(20000,5e-6),10)
    assert r['interval'][1]>.001
    assert zero_hit_upper(65536)==pytest.approx(4.571e-5,rel=.001)


def test_narrow_island_severity_stability():
    from scipy.integrate import quad
    for top in [1e-10,1e-12,1e-16]:
        radius=np.sqrt(top)
        truth=quad(lambda x:(top-x*x)*norm.pdf(x),-radius,radius,epsabs=1e-40,epsrel=1e-12)[0]
        observed=tail_moment(-np.ones((1,1)),np.zeros((1,1)),np.array([[top]]),0)[0]
        assert observed>0
        assert observed==pytest.approx(truth,rel=1e-12,abs=1e-40)


def test_precision_certifies_displayed_estimate():
    from tailsentry.uncertainty import precision_status
    assert precision_status([.8,1.],.13,.99)['status']=='insufficient-budget'
    assert precision_status([.8,1.],.13,.9)['status']=='precision-supported'
