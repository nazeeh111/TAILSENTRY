#!/usr/bin/env python3
"""Combine only complete predefined groups; retain exact source-run provenance."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import beta
from tailsentry.benchmark import CASES,METHODS

out=Path('results/final');out.mkdir(exist_ok=False)
main=[json.loads(line) for line in Path('results/holdout/main-v1/runs.jsonl').read_text().splitlines()]
primary=[json.loads(line) for line in Path('results/holdout/primary-v2/runs.jsonl').read_text().splitlines()]
records=[];summary=[]
for case in CASES:
 for budget in [4096,16384]:
  for method in METHODS:
   origin='primary-v2' if case=='delta_gamma_1e4' else 'main-v1'
   source=primary if origin=='primary-v2' else main
   rows=[dict(row,origin=origin) for row in source if row['case']==case and row['budget']==budget and row['method']==method]
   expected=128 if origin=='primary-v2' else 64
   assert len(rows)==expected,(case,budget,method,len(rows))
   assert len({r['seed'] for r in rows})==expected
   records.extend(rows);truth=rows[0]['truth'];x=np.array([r['estimate'] for r in rows]);error=x-truth
   def coverage(key):
    intervals=[r[key] for r in rows if r[key] is not None]
    if not intervals:return None
    k=sum(l<=truth<=u for l,u in intervals);n=len(intervals)
    return dict(rate=k/n,count=n,binomial_interval=[0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))],mean_width=float(np.mean([u-l for l,u in intervals])))
   mse=float(np.mean(error**2))
   summary.append(dict(case=case,method=method,budget=budget,runs=len(rows),truth=truth,mean=float(x.mean()),bias=float(error.mean()),
                       relative_rmse=float(np.sqrt(mse)/truth),rmse=float(np.sqrt(mse)),mean_seconds=float(np.mean([r['seconds'] for r in rows])),
                       mean_cpu_seconds=float(np.mean([r['cpu_seconds'] for r in rows])),mse_times_seconds=mse*np.mean([r['seconds'] for r in rows]),
                       mean_point_evaluations=float(np.mean([r['point_evaluations'] for r in rows])),mean_conditional_evaluations=float(np.mean([r['conditional_evaluations'] for r in rows])),
                       zero_estimate_rate=float(np.mean(x==0)),finite_coverage=coverage('interval'),asymptotic_coverage=coverage('asymptotic_interval'),origin=origin))
(out/'runs.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records));(out/'summary.json').write_text(json.dumps(summary,indent=2))
# Primary outcome, bootstrap independent complete runs. Timing sensitivity uses
# a separately randomized sequential timing study, never replaces initial result.
time_rows=json.loads(Path('results/timing/runs.json').read_text())
focus=[r for r in records if r['case']=='delta_gamma_1e4' and r['budget']==16384]
selected='envelope_projected_fixed';comparators=['conditional_sobol','envelope_raw','ce_single']
rng=np.random.default_rng(901);target=[]
for comparator in comparators:
 a=sorted([r for r in focus if r['method']==selected],key=lambda r:r['seed']);b=sorted([r for r in focus if r['method']==comparator],key=lambda r:r['seed']);n=len(a)
 ea=np.array([(r['estimate']-r['truth'])**2 for r in a]);eb=np.array([(r['estimate']-r['truth'])**2 for r in b]);ta=np.array([r['seconds'] for r in a]);tb=np.array([r['seconds'] for r in b])
 ia=np.array([r['seconds'] for r in time_rows if r['method']==selected]);ib=np.array([r['seconds'] for r in time_rows if r['method']==comparator])
 ratios=[];unpaired=[];isolated=[]
 for _ in range(4000):
  index=rng.integers(n,size=n);other=rng.integers(n,size=n)
  ratios.append(float(ea[index].mean()*ta[index].mean()/(eb[index].mean()*tb[index].mean())))
  unpaired.append(float(ea[index].mean()*ta[index].mean()/(eb[other].mean()*tb[other].mean())))
  it=rng.integers(len(ia),size=len(ia));jt=rng.integers(len(ib),size=len(ib))
  isolated.append(float(ea[index].mean()*ia[it].mean()/(eb[other].mean()*ib[jt].mean())))
 target.append(dict(selected=selected,comparator=comparator,mse_ratio=float(ea.mean()/eb.mean()),
                    mse_time_ratio=float(ea.mean()*ta.mean()/(eb.mean()*tb.mean())),paired_bootstrap_interval=np.quantile(ratios,[.025,.975]).tolist(),unpaired_bootstrap_interval=np.quantile(unpaired,[.025,.975]).tolist(),
                    isolated_timing_ratio=float(ea.mean()*ia.mean()/(eb.mean()*ib.mean())),isolated_bootstrap_interval=np.quantile(isolated,[.025,.975]).tolist(),
                    efficiency_upper_below_one=bool(np.quantile(ratios,.975)<1),original_target_met=bool(np.quantile(ratios,.975)<1) if comparator=="conditional_sobol" else None,mean_isolated_seconds_selected=float(ia.mean()),mean_isolated_seconds_comparator=float(ib.mean())))
(out/'primary-analysis.json').write_text(json.dumps(target,indent=2))
receipt=dict(rows=len(records),groups=len(summary),sources=['main-v1 complete secondary groups','primary-v2 primary only'],
             omitted='All partial main-v1 primary groups excluded; no selective secondary omission',
             hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('results/holdout/main-v1/runs.jsonl'),Path('results/holdout/primary-v2/runs.jsonl'),out/'runs.jsonl',out/'summary.json',out/'primary-analysis.json']})
(out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(target,indent=2))
