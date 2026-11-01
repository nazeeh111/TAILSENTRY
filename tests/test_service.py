import pytest
from tailsentry.service import investigate


def test_record_names_requested_method_and_binds_source():
    r=investigate(dict(case='single_1e3',method='envelope_projected_fixed',budget=256,seed=7))
    assert r['method']=='envelope_projected_fixed'
    assert r['estimate']==pytest.approx(.001,rel=1e-12)
    assert sum(row['probability'] for row in r['mode_contributions'])==pytest.approx(r['estimate'])
    assert len(r['record_sha256'])==64
    assert 'conditional.py' in r['source_sha256']


def test_service_rejects_unbounded_work_and_unknown_models():
    for config in [dict(case='unknown'),dict(case='single_1e3',budget=1000000),dict(case='single_1e3',seed=-1),dict(case='single_1e3',method='execute')]:
        with pytest.raises(ValueError):investigate(config)


@pytest.mark.parametrize('method',['mc','conditional_sobol'])
def test_comparator_accepts_exact_user_95_percent(method):
    r=investigate(dict(case='single_1e3',method=method,budget=256,seed=7,confidence=.95))
    assert r['method']==method
    assert r['estimate']>=0


def _export_record(seed):
    import hashlib,json
    record={'case':'single_1e3','seed':seed,'estimate':.001}
    record['record_sha256']=hashlib.sha256(json.dumps(record,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return record


def test_export_preserves_record_and_is_immutable_after_registration():
    import json
    from tailsentry.service import RecordStore
    store=RecordStore()
    record=investigate(dict(case='single_1e3',budget=256,seed=7))
    original=json.loads(json.dumps(record))
    digest=store.add(record)
    record['estimate']=.9
    assert json.loads(store.get(digest))==original


def test_export_rejects_changed_or_invalid_content():
    from tailsentry.service import RecordStore
    store=RecordStore()
    record=_export_record(1)
    record['estimate']=.5
    with pytest.raises(ValueError,match='content'):store.add(record)
    with pytest.raises(ValueError,match='digest'):store.add({'record_sha256':'../source'})
    assert store.get('../source') is None


def test_export_cache_is_bounded_and_keeps_newest_records():
    from tailsentry.service import RecordStore
    store=RecordStore(capacity=2)
    records=[_export_record(seed) for seed in range(3)]
    for record in records:store.add(record)
    assert store.get(records[0]['record_sha256']) is None
    assert store.get(records[1]['record_sha256']) is not None
    assert store.get(records[2]['record_sha256']) is not None


@pytest.mark.parametrize('capacity',[0,-1,True])
def test_export_cache_requires_a_positive_integer_capacity(capacity):
    from tailsentry.service import RecordStore
    with pytest.raises(ValueError):RecordStore(capacity)
