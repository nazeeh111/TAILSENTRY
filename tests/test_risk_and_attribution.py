import numpy as np
import pytest
from tailsentry.attribution import branch_probabilities
from tailsentry.conditional import probability
from tailsentry.models import normal_max
from tailsentry.references import normal_max_survival,solve_risk
from tailsentry.risk import estimate_risk
from tailsentry.uncertainty import precision_status


def test_branch_partition_is_probability_partition():
    rng=np.random.default_rng(37)
    a=rng.normal(size=(20,3));b=rng.normal(size=(20,3));c=rng.normal(size=(20,3))
    pieces=branch_probabilities(a,b,c,.7)
    assert np.all(pieces>=0)
    assert np.allclose(pieces.sum(axis=1),probability(a,b,c,.7),atol=1e-13)


def test_quantile_severity_are_separately_validated():
    model=normal_max(1)
    reference=solve_risk(lambda t:normal_max_survival(t,1),.99,(0,5))
    q=reference['var']
    result=estimate_risk(model,.99,np.linspace(q-.3,q+.3,9),budget=1024,seed=3)
    assert result['var']==pytest.approx(reference['var'],abs=1e-9)
    assert result['es']==pytest.approx(reference['es'],abs=1e-9)
    assert result['var_interval_grid_95'][0]<=q<=result['var_interval_grid_95'][1]


def test_inconsistent_intervals_never_certify():
    assert precision_status([.9,.5],.1,.8)['status']=='inconsistent-interval'
