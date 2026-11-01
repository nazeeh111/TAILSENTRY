#!/usr/bin/env python3
import json
from pathlib import Path
s=json.loads(Path('results/final/summary.json').read_text())
labels={'mc':'Crude MC','antithetic':'Antithetic','sobol':'Scrambled Sobol','fixed_is':'Fixed IS','ce_single':'CE single','ce_mixture':'CE mixture','subset':'Subset','conditional':'Conditional MC','conditional_sobol':'Conditional Sobol','raw_mixture':'Adaptive mixture raw','projected_mixture':'Adaptive mixture projected','envelope_raw':'Certified union raw','envelope_projected_fixed':'Certified union fixed projection','envelope_projected':'Certified union pilot projection','envelope_unoptimized':'Projected union equal caps'}
rows=[]
for r in s:
 if r['case']=='delta_gamma_1e4' and r['budget']==16384:
  finite='--' if r['finite_coverage'] is None else f"{100*r['finite_coverage']['rate']:.1f}"
  approx='--' if r['asymptotic_coverage'] is None else f"{100*r['asymptotic_coverage']['rate']:.1f}"
  rows.append(f"{labels[r['method']]} & {r['relative_rmse']:.4g} & {r['mean_seconds']:.3f} & {finite} & {approx}" + ' '+chr(92)*2)
Path('paper/primary-table.tex').write_text('\n'.join(rows))
short={'single_1e3':'Single normal','paired4_1e4':'Paired4','imbalanced4_1e5':'Imbalanced4','correlated4_1e5':'Correlated4','compact2_1e4':'Compact2','compact4_1e5':'Compact4','student_balls_1e4':'Student balls','delta_gamma_1e4':'Delta--gamma'}
rows=[]
for case,name in short.items():
 trio=[next(r for r in s if r['case']==case and r['budget']==16384 and r['method']==m) for m in ['envelope_raw','envelope_projected_fixed','conditional_sobol']]
 rows.append(name+' & '+' & '.join(f"{r['relative_rmse']:.4g}" for r in trio)+' \\\\')
Path('paper/suite-table.tex').write_text('\n'.join(rows))
for r in s:
 if r['case']=='delta_gamma_1e4' and r['budget']==16384 and r['method'] in ('envelope_raw','envelope_projected_fixed','conditional_sobol'):
  print(r['method'],r['relative_rmse'],r['finite_coverage'],r['asymptotic_coverage'])
