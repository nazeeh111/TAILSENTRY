#!/usr/bin/env python3
"""Publication figures generated only from saved actual experiment artifacts."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tailsentry.benchmark import CASES,METHODS
out=Path('paper/figures');out.mkdir(exist_ok=True)
s=json.loads(Path('results/final/summary.json').read_text())
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','figure.dpi':140})
labels={'mc':'MC','antithetic':'Antithetic','sobol':'Sobol','fixed_is':'Fixed IS','ce_single':'CE single','ce_mixture':'CE mixture','subset':'Subset','conditional':'CMC','conditional_sobol':'CMC Sobol','raw_mixture':'Mixture raw','projected_mixture':'Mixture RB','envelope_raw':'Union raw','envelope_projected_fixed':'Union RB fixed','envelope_projected':'Union RB pilot','envelope_unoptimized':'Union equal caps'}
short=['Single','Paired4','Imbalanced4','Correlated4','Compact2','Compact4','Student balls','Delta-gamma']
def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(out/f'{name}.{ext}',dpi=300)
 plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4));matrix=np.array([[next(r['relative_rmse'] for r in s if r['case']==c and r['budget']==16384 and r['method']==m) for m in METHODS] for c in CASES]);matrix=np.maximum(matrix,1e-9)
im=ax.imshow(matrix,norm=LogNorm(vmin=1e-5,vmax=3),cmap='viridis',aspect='auto');ax.set_xticks(range(len(METHODS)),[labels[m] for m in METHODS],rotation=55,ha='right');ax.set_yticks(range(len(CASES)),short);fig.colorbar(im,ax=ax,label='Relative RMSE (log scale)');ax.set_title('All predefined problems and methods / budget16,384');save(fig,'suite')
selected=['mc','sobol','ce_single','ce_mixture','conditional_sobol','envelope_raw','envelope_projected_fixed','envelope_projected','envelope_unoptimized']
fig,axes=plt.subplots(1,2,figsize=(9,3.5));rows=[next(r for r in s if r['case']=='delta_gamma_1e4' and r['budget']==16384 and r['method']==m) for m in selected]
axes[0].barh(range(len(rows)),[r['relative_rmse'] for r in rows],color=['#276785' if r['method']!='envelope_projected_fixed' else '#c25b32' for r in rows]);axes[0].set_xscale('log');axes[0].set_yticks(range(len(rows)),[labels[r['method']] for r in rows]);axes[0].invert_yaxis();axes[0].set_xlabel('Relative RMSE /128 independent runs');axes[0].set_title('Primary quadratic loss')
for m,color in [('envelope_raw','#768eac'),('envelope_projected_fixed','#c25b32'),('conditional_sobol','#276785')]:
 rows2=[next(r for r in s if r['case']==c and r['budget']==16384 and r['method']==m) for c in CASES]
 axes[1].plot(range(len(CASES)),[r['asymptotic_coverage']['rate'] if r['asymptotic_coverage'] else np.nan for r in rows2],marker='o',ms=3,label=labels[m]+' approximate',color=color)
axes[1].axhline(.95,color='black',ls='--',lw=.8,label='Nominal95%');axes[1].set_ylim(0,1.04);axes[1].set_xticks(range(len(CASES)),short,rotation=55,ha='right');axes[1].set_ylabel('Observed interval coverage');axes[1].set_title('Ordinary intervals can undercover');axes[1].legend(fontsize=6,loc='lower right');fig.tight_layout();save(fig,'primary_coverage')
r=json.loads(Path('results/missed-mode/runs.json').read_text());fig,axes=plt.subplots(1,2,figsize=(8.5,2.8));x=np.array([v['estimate'] for v in r]);axes[0].scatter(range(len(x)),x,s=6,color='#276785',alpha=.6);axes[0].axhline(1e-5,color='#c25b32',label='Truth',ls='--');axes[0].set_yscale('log');axes[0].set_xlabel('Independent run');axes[0].set_ylabel('Probability estimate');axes[0].legend(fontsize=8);axes[0].set_title('99.4% of runs miss the hidden tail')
mm=json.loads(Path('results/missed-mode/summary.json').read_text());axes[1].bar(['Approximate normal','Finite Bernstein'],[mm['asymptotic_coverage'],mm['finite_coverage']],color=['#c25b32','#276785']);axes[1].axhline(.95,color='black',ls='--',lw=.8);axes[1].set_ylim(0,1.08);axes[1].set_ylabel('Observed coverage /512 runs');axes[1].set_title('High observed ESS does not certify coverage');fig.tight_layout();save(fig,'missed_mode')
h=json.loads(Path('results/historical/summary.json').read_text());Path('paper/figures/historical-source.json').write_text(json.dumps(h,indent=2));print('historical keys',h.keys())
with Path('results/final/table.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['case','method','budget','runs','truth','relative_rmse','mean_seconds','mean_cpu_seconds','zero_estimate_rate']);writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in s)
Path('paper/figures/plot-data.json').write_text(json.dumps({'suite_relative_rmse':matrix.tolist(),'cases':CASES,'methods':METHODS},indent=2))
