#!/usr/bin/env python3
"""Predeclared malicious two-sided missed-mode experiment, not tuning."""
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtri_exp
from scipy.stats import norm
from tailsentry.uncertainty import summarize_observations,zero_hit_upper
out=Path('results/missed-mode');out.mkdir(exist_ok=False)
p=5e-6;b=norm.isf(p);truth=2*p;n=20000;rows=[]
for seed in range(610000,610512):
 rng=np.random.default_rng(seed);selected=rng.random(n)<.9
 z=rng.standard_normal(n);u=np.maximum(rng.random(selected.sum()),np.nextafter(0.,1.))
 z[selected]=-ndtri_exp(np.log(p)+np.log(u))
 w=1/(.1+.9*(z>b)/p);y=w*(np.abs(z)>b)
 r=summarize_observations(y,10.)
 rows.append(dict(seed=seed,estimate=r['estimate'],finite_interval=r['interval'],asymptotic_interval=r['asymptotic_interval'],
                  hidden_hits=int((z<-b).sum()),tail_ess=r['tail_ess'],standard_error=r['standard_error']))
summary=dict(truth=truth,threshold=b,budget=n,runs=len(rows),defensive_mass=.1,
             theoretical_hidden_mode_miss_probability=(1-.1*p)**n,
             observed_hidden_mode_miss_rate=float(np.mean([r['hidden_hits']==0 for r in rows])),
             finite_coverage=float(np.mean([r['finite_interval'][0]<=truth<=r['finite_interval'][1] for r in rows])),
             asymptotic_coverage=float(np.mean([r['asymptotic_interval'][0]<=truth<=r['asymptotic_interval'][1] for r in rows])),
             median_tail_ess=float(np.median([r['tail_ess'] for r in rows])),
             nominal_audit_zero_hit_upper_65536=zero_hit_upper(65536),
             audit_cost_for_residual_1e6=int(np.ceil(np.log(.025)/np.log1p(-1e-6))),
             interpretation='Conditional unbiasedness and support do not make normal intervals trustworthy. Finite bound abstains. No cheap generic missed-mode guarantee.')
(out/'runs.json').write_text(json.dumps(rows,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
