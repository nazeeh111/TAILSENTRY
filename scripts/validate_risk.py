#!/usr/bin/env python3
"""Independent-replication VaR/ES validation against deterministic references."""
import json,time
from pathlib import Path
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from tailsentry.models import normal_max
from tailsentry.benchmark import make_case
from tailsentry.references import normal_max_survival,solve_risk
from tailsentry.risk import estimate_risk

out=Path('results/risk-validation');out.mkdir(exist_ok=False)
records=[]
for name in ['normal_max2','delta_gamma']:
 for alpha in [.99,.999]:
  if name=='normal_max2':
   model=normal_max(2);survival=lambda t:normal_max_survival(t,2)
  else:
   model,_,_=make_case('delta_gamma_1e4')
   survival=lambda t:quad(lambda z:norm.sf((t-.25*z*z-.24*z)/(.6*np.sqrt(1-.4**2)))*norm.pdf(z),-14,14,epsabs=1e-13,epsrel=1e-10,limit=200)[0]
  reference=solve_risk(survival,alpha,(0,12))
  q=reference['var'];grid=np.linspace(q-.35,q+.35,17)
  for seed in range(510000,510032):
   r=estimate_risk(model,alpha,grid,budget=8192,seed=seed)
   row={k:r[k] for k in ['var','es','var_se_asymptotic','es_se_asymptotic','var_interval_grid_95','status','seconds']}
   row.update(model=name,alpha=alpha,seed=seed,true_var=q,true_es=reference['es'],budget=8192,
              actual_calls=r['run']['point_evaluations']+r['run']['conditional_evaluations'])
   records.append(row)
  print(name,alpha,flush=True)
(out/'runs.json').write_text(json.dumps(records,indent=2))
summary=[]
for name in ['normal_max2','delta_gamma']:
 for alpha in [.99,.999]:
  rows=[r for r in records if r['model']==name and r['alpha']==alpha]
  result=dict(model=name,alpha=alpha,runs=len(rows),truth_var=rows[0]['true_var'],truth_es=rows[0]['true_es'])
  for metric in ['var','es']:
   errors=np.array([r[metric]-r['true_'+metric] for r in rows])
   result[metric+'_rmse']=float(np.sqrt(np.mean(errors**2)));result[metric+'_bias']=float(errors.mean())
   result[metric+'_asymptotic_coverage']=float(np.mean([abs(r[metric]-r['true_'+metric])<=1.96*r[metric+'_se_asymptotic'] for r in rows]))
  result['finite_grid_bracket_coverage']=float(np.mean([(r['var_interval_grid_95'][0] is None or r['var_interval_grid_95'][0]<=r['true_var']) and (r['var_interval_grid_95'][1] is None or r['true_var']<=r['var_interval_grid_95'][1]) for r in rows]))
  result['finite_grid_two_sided_rate']=float(np.mean([all(x is not None for x in r['var_interval_grid_95']) for r in rows]))
  summary.append(result)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
