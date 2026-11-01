"""Independently audit final aggregation, numerical fixes and primary outcome."""
from pathlib import Path
from collections import defaultdict,Counter
import json,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm,beta
from tailsentry.conditional import tail_moment
from tailsentry.uncertainty import precision_status
root=Path(__file__).resolve().parents[1]
def read(path):return json.loads((root/path).read_text())
raw=[json.loads(line) for line in (root/'results/final/runs.jsonl').read_text().splitlines()]
summary=read('results/final/summary.json');analysis=read('results/final/primary-analysis.json')
receipt=read('results/final/receipt.json')
assert len(raw)==17280 and len(summary)==240
for p,h in receipt['hashes'].items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p
snapshots={}
for origin in ['main-v1','primary-v2']:
 base=root/'results/holdout'/origin;conf=json.loads((base/'configuration.json').read_text())
 snapshots[origin]=all(hashlib.sha256((base/'source'/p).read_bytes()).hexdigest()==h for p,h in conf['source_hashes'].items())
 assert snapshots[origin]
 by_id={(r['case'],r['budget'],r['method'],r['seed']):r for r in [json.loads(line) for line in (base/'runs.jsonl').read_text().splitlines()]}
 for r in raw:
  if r['origin']!=origin:continue
  original=by_id[(r['case'],r['budget'],r['method'],r['seed'])]
  assert {k:v for k,v in r.items() if k!='origin'}==original
bygroup=defaultdict(list)
for r in raw:bygroup[(r['case'],r['method'],r['budget'])].append(r)
for s in summary:
 group=bygroup[(s['case'],s['method'],s['budget'])]
 expected=128 if s['case']=='delta_gamma_1e4' else 64
 assert len(group)==expected and len({r['seed'] for r in group})==expected
 assert set(r['seed'] for r in group)==set(range(220000,220128) if expected==128 else range(210000,210064))
 error=np.array([r['estimate']-r['truth'] for r in group]);mse=float(np.mean(error*error));timing=float(np.mean([r['seconds'] for r in group]))
 np.testing.assert_allclose([s['bias'],s['rmse'],s['relative_rmse'],s['mse_times_seconds']],[error.mean(),np.sqrt(mse),np.sqrt(mse)/s['truth'],mse*timing],rtol=1e-13,atol=1e-30)
 for field,label in [('interval','finite_coverage'),('asymptotic_interval','asymptotic_coverage')]:
  present=[r[field] for r in group if r[field] is not None]
  if not present:assert s[label] is None;continue
  k=sum(lo<=s['truth']<=hi for lo,hi in present);n=len(present)
  np.testing.assert_allclose(s[label]['rate'],k/n)
  ci=[0. if k==0 else beta.ppf(.025,k,n-k+1),1. if k==n else beta.ppf(.975,k+1,n-k)]
  np.testing.assert_allclose(s[label]['binomial_interval'],ci)
# Reconstruct final bootstrap exactly but from independently validated raw groups.
times=read('results/timing/runs.json');rng=np.random.default_rng(901);recomputed=[]
for target in analysis:
 a=sorted(bygroup[('delta_gamma_1e4',target['selected'],16384)],key=lambda r:r['seed'])
 b=sorted(bygroup[('delta_gamma_1e4',target['comparator'],16384)],key=lambda r:r['seed'])
 ae=np.array([(r['estimate']-r['truth'])**2 for r in a]);be=np.array([(r['estimate']-r['truth'])**2 for r in b]);at=np.array([r['seconds'] for r in a]);bt=np.array([r['seconds'] for r in b])
 ai=np.array([r['seconds'] for r in times if r['method']==target['selected']]);bi=np.array([r['seconds'] for r in times if r['method']==target['comparator']])
 ratios=[];unpaired=[];isolated=[];n=len(a)
 for _ in range(4000):
  ix=rng.integers(n,size=n);jx=rng.integers(n,size=n)
  ratios.append(ae[ix].mean()*at[ix].mean()/(be[ix].mean()*bt[ix].mean()))
  unpaired.append(ae[ix].mean()*at[ix].mean()/(be[jx].mean()*bt[jx].mean()))
  it=rng.integers(len(ai),size=len(ai));jt=rng.integers(len(bi),size=len(bi))
  isolated.append(ae[ix].mean()*ai[it].mean()/(be[jx].mean()*bi[jt].mean()))
 ratio=ae.mean()*at.mean()/(be.mean()*bt.mean())
 np.testing.assert_allclose(target['mse_time_ratio'],ratio)
 for field,arr in [('paired_bootstrap_interval',ratios),('unpaired_bootstrap_interval',unpaired),('isolated_bootstrap_interval',isolated)]:np.testing.assert_allclose(target[field],np.quantile(arr,[.025,.975]))
 recomputed.append({'comparator':target['comparator'],'mse_time_ratio':ratio,'paired_95_interval':np.quantile(ratios,[.025,.975]).tolist(),'isolated_95_interval':np.quantile(isolated,[.025,.975]).tolist()})
assert recomputed[0]['paired_95_interval'][0]>1 and recomputed[0]['isolated_95_interval'][0]>1
# Repaired severity: shifted finite islands against direct local integration.
islands=[]
for center in [0.,1.,4.,8.]:
 for nominal_r2 in [1e-8,1e-10,1e-12]:
  a=np.array([[-1.]]);b=np.array([[2*center]]);c=np.array([[nominal_r2-center*center]])
  r2=float(c[0,0]+center*center);r=np.sqrt(r2)
  expected=quad(lambda x:(r2-x*x)*norm.pdf(center+x),-r,r,epsabs=1e-40,epsrel=1e-12)[0]
  observed=float(tail_moment(a,b,c,0)[0]);error=observed/expected-1
  assert observed>0,(center,nominal_r2,error)
  islands.append({'center':center,'nominal_radius_squared':nominal_r2,'relative_error':error})
assert precision_status([.8,1.],.13,estimate=.99)['status']=='insufficient-budget'
assert precision_status([.8,1.],.13,estimate=.9)['status']=='precision-supported'
focus=next(s for s in summary if (s['case'],s['method'],s['budget'])==('delta_gamma_1e4','envelope_projected_fixed',16384))
report={'rows':len(raw),'groups':len(summary),'all_groups_and_hashes_verified':True,'frozen_snapshot_hashes_match':snapshots,'primary_comparisons':recomputed,'selected_primary_finite_coverage':focus['finite_coverage'],'severity_shifted_islands':islands,'precision_guard_examples_passed':True,'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'src/tailsentry').glob('*.py'))}}
(root/'reviews/adversarial-final-checks.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','severity_shifted_islands']},indent=2))
print('maximum shifted-island relative error',max(abs(x['relative_error']) for x in islands))
