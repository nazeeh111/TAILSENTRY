"""Independent audit checks; does not modify implementation or benchmark runs."""
from pathlib import Path
import hashlib, importlib.util, json
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, t
from tailsentry.conditional import probability, tail_moment
from tailsentry.envelope import build_cover
from tailsentry.models import QuadraticModel

root=Path(__file__).resolve().parents[1]
rng=np.random.default_rng(613104)
report={'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'src/tailsentry').glob('*.py'))}}
# Oracle uses companion-matrix roots and original polynomial evaluation, not
# implementation interval classification, root formula or Gaussian moments.
max_probability_error=0.; max_moment_error=0.
for _ in range(50):
    abc=rng.normal(size=(3,3)); aa,bb,cc=abc
    threshold=float(rng.normal())
    breaks=[-np.inf,np.inf]
    for j in range(3):
        for coeff in ([aa[j],bb[j],cc[j]-threshold],):
            breaks.extend(float(v.real) for v in np.roots(coeff) if abs(v.imag)<1e-10)
        for k in range(j):
            breaks.extend(float(v.real) for v in np.roots(abc[:,j]-abc[:,k]) if abs(v.imag)<1e-10)
    breaks=sorted(breaks)
    def loss(x):return float(np.max(aa*x*x+bb*x+cc))
    pp=mm=0.
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        mid=(lo+hi)/2 if np.isfinite(lo+hi) else hi-1 if np.isneginf(lo) else lo+1
        if loss(mid)>threshold:
            pp+=quad(lambda x:norm.pdf(x),lo,hi,epsabs=1e-12)[0]
            mm+=quad(lambda x:loss(x)*norm.pdf(x),lo,hi,epsabs=1e-12)[0]
    shape=lambda x:np.asarray(x)[None,:]
    pe=abs(probability(shape(aa),shape(bb),shape(cc),threshold)[0]-pp)
    me=abs(tail_moment(shape(aa),shape(bb),shape(cc),threshold)[0]-mm)
    max_probability_error=max(max_probability_error,float(pe));max_moment_error=max(max_moment_error,float(me))
assert max_probability_error<1e-9 and max_moment_error<1e-8
report['ordinary_polynomial_checks']={'count':50,'maximum_absolute_probability_error':max_probability_error,'maximum_absolute_first_moment_error':max_moment_error}
# Targeted narrow-island adversary. A direct local integral avoids subtracting
# nearly equal Gaussian second moments. Exact real input polynomial is r2-x2.
report['narrow_islands']=[]
for r2 in [1e-8,1e-10,1e-12]:
    a=np.array([[-1.]]); b=np.array([[0.]]); c=np.array([[r2]])
    p=float(probability(a,b,c,0)[0]); observed=float(tail_moment(a,b,c,0)[0])
    radius=np.sqrt(r2)
    expected=quad(lambda u:(r2-u*u)*norm.pdf(u),-radius,radius,epsabs=1e-35,epsrel=1e-12)[0]
    report['narrow_islands'].append({'radius_squared':r2,'probability':p,'computed_first_moment':observed,'independent_first_moment':expected,'relative_error':observed/expected-1,'violates_nonnegative_loss_on_event':observed<0})
missed=0;events=0
for _ in range(32):
    q=rng.normal(size=(3,3,3));q=(q+q.swapaxes(1,2))/2
    m=QuadraticModel(q,rng.normal(size=(3,3)),rng.normal(size=3))
    cover=build_cover(m,2.)
    z=rng.normal(size=(256,3));event=m.loss(z)>2.
    inside=np.any(z@cover['normals'].T>cover['cutoffs'],axis=1)
    missed+=int(np.count_nonzero(event&~inside));events+=int(event.sum())
assert missed==0
report['random_cover_falsification']={'models':32,'points':32*256,'event_points':events,'uncovered_event_points':missed,'limit':'Finite random search only; algebraic proof remains necessary.'}
# Historical analytic ES checked through an independent one-dimensional tail integral.
spec=importlib.util.spec_from_file_location('validate_fx',root/'scripts/validate_fx.py');fx=importlib.util.module_from_spec(spec);spec.loader.exec_module(fx)
historical=json.loads((root/'results/historical/summary.json').read_text())
_,eval_rows,_=fx.read_verified_returns(root/'data')
losses=np.array([float(r['basket_simple_loss']) for r in eval_rows])
errors=[]
for row in historical['diagnostics']:
    model=historical['models'][row['model']]
    dist=norm(loc=model['location'],scale=model['scale']) if row['model']=='gaussian' else t(model['df'],loc=model['location'],scale=model['scale'])
    es=quad(lambda x:x*dist.pdf(x),row['var'],np.inf,epsabs=1e-12)[0]/(1-row['confidence'])
    errors.append(abs(es-row['es']))
    assert int(np.sum(losses>row['var']))==row['exceedances']
    assert abs(es-row['es'])<1e-9
report['historical_checks']={'six_es_tail_integrals_max_absolute_error':max(errors),'all_exceedance_counts_match':True,'observations':len(eval_rows),'source_and_output_hash_validation':'passed read_verified_returns'}
raw=root/'results/holdout/main-v1/runs.jsonl'
rows=[json.loads(line) for line in raw.read_text().splitlines()]
report['holdout_snapshot']={'raw_rows_read':len(rows),'summary_existed':(raw.parent/'summary.json').exists(),'groups':[]}
for case in ['paired4_1e4','compact2_1e4','correlated4_1e5']:
    for budget in [4096,16384]:
        for method in ['conditional_sobol','envelope_raw','envelope_projected_fixed','envelope_projected']:
            rr=[r for r in rows if (r['case'],r['budget'],r['method'])==(case,budget,method)]
            if len(rr)!=64:continue
            truth=rr[0]['truth']; mse=float(np.mean([(r['estimate']-truth)**2 for r in rr]))
            coverage=lambda field:sum(r[field][0]<=truth<=r[field][1] for r in rr if r[field] is not None)
            report['holdout_snapshot']['groups'].append({'case':case,'budget':budget,'method':method,'runs':64,'relative_rmse':np.sqrt(mse)/truth,'mean_seconds':float(np.mean([r['seconds'] for r in rr])),'finite_covered':coverage('interval'),'asymptotic_covered':coverage('asymptotic_interval'),'mean_finite_width':float(np.mean([r['interval'][1]-r['interval'][0] for r in rr]))})
(root/'reviews/adversarial-checks.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({key:val for key,val in report.items() if key not in ['source_sha256','holdout_snapshot']},indent=2))
print('holdout rows inspected',len(rows),'summary existed',report['holdout_snapshot']['summary_existed'])
