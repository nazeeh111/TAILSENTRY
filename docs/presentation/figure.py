"""Source-derived evidence; no simulation, smoothing or study rerun."""
from pathlib import Path
import json, math, hashlib
from html import escape
D=Path(__file__).parent
r=json.loads((D/'worked-record.json').read_text());a=json.loads((D/'primary-analysis.json').read_text());m=json.loads((D/'missed-mode-runs.json').read_text())
BG='#080c09';FG='#dfebe0';MUT='#a0b0a4';GREEN='#9fffb0';AMBER='#e6c178';LINE='#304a36'
def draw(mobile):
 w=375 if mobile else 1040; h=1090 if mobile else 1060; x0=28 if mobile else 48; right=w-x0; width=right-x0;s=[]
 def add(t):s.append(t)
 def text(x,y,t,size=14,color=FG):add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{escape(t)}</text>')
 def line(x,y,X,Y,c=LINE,sw=1):add(f'<line x1="{x}" y1="{y}" x2="{X}" y2="{Y}" stroke="{c}" stroke-width="{sw}"/>')
 add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc"><title id="title">TailSentry: a useful bound, a failed comparison</title><desc id="desc">A current 4096-draw Gaussian delta-gamma estimate is shown separately from frozen research. The primary efficiency ratio is 38.79 versus conditional Sobol, where lower is better. In the missed-mode counterexample, ordinary intervals cover truth in 3 of 512 runs, despite high observed effective sample size. Each square represents one stored run, in original seed order.</desc><rect width="{w}" height="{h}" fill="{BG}"/><g font-family="Arial, Helvetica, sans-serif">')
 text(x0,38,'TAILSENTRY',16,GREEN);text(x0,77,'A useful bound.',30);text(x0,114,'A failed comparison.',30)
 text(x0,151,'Probability under an assumed model.',14,MUT)
 line(x0,174,right,174)
 text(x0,204,'01 / Current bounded run',16,GREEN);text(x0,230,'Gaussian delta-gamma / seed 42 / 4,096 draws',12 if mobile else 14,MUT)
 # Fixed probability axis; trace is complete 64 stored prefixes. No clipping.
 top=253;bottom=371;lo=0.8e-4;hi=1.4e-4
 xx=lambda n:x0+(n-1)/(4096-1)*width
 yy=lambda p:bottom-(p-lo)/(hi-lo)*(bottom-top)
 L,U=r['interval'];add(f'<rect x="{x0}" y="{yy(U)}" width="{width}" height="{yy(L)-yy(U)}" fill="{GREEN}" opacity=".10"/>')
 for p in [lo,1e-4,hi]:line(x0,yy(p),right,yy(p));text(x0+3,yy(p)-5,f'{p:.1e}',11,MUT)
 line(x0,yy(r['reference_probability']),right,yy(r['reference_probability']),AMBER,1.5)
 points=' '.join(f"{xx(t['samples']):.7f},{yy(t['estimate']):.7f}" for t in r['trace']);add(f'<polyline id="confirmation-trace" points="{points}" fill="none" stroke="{GREEN}" stroke-width="2"/>')
 text(x0,397,'1 → 4,096 confirmation draws',12,MUT);text(x0,423,'95% finite bound: [9.408e−5, 1.077e−4]',13)
 text(x0,448,'Band: final bound. Gold: known truth 1e−4.',12,MUT)
 text(x0,472,'Prefixes do not permit optional stopping.',12,AMBER)
 line(x0,493,right,493)
 y=524;text(x0,y,'02 / Frozen primary comparison',16,GREEN)
 text(x0,y+25,'MSE × elapsed-time ratio; lower is better.',13,MUT)
 axis=y+64; lx=x0+20;rx=right-20; fx=lambda z:lx+(math.log10(z)+2)/4*(rx-lx)
 for v in [.01,.1,1,10,100]:line(fx(v),axis,fx(v),axis+95,AMBER if v==1 else LINE);text(fx(v)-7,axis+115,str(v),11,MUT)
 for i,(row,label) in enumerate([(a[0],'vs conditional Sobol'),(a[1],'vs raw union')]):
  cy=axis+24+i*50;l,u=row['paired_bootstrap_interval'];v=row['mse_time_ratio'];line(fx(l),cy,fx(u),cy,GREEN,4);add(f'<circle data-ratio="{v}" cx="{fx(v)}" cy="{cy}" r="4" fill="{GREEN}"/>');text(x0,cy-12,label+'  '+f'{v:.4g}',12)
 text(x0,axis+144,'95% paired bootstrap intervals.',12,MUT)
 y=axis+178;line(x0,y,right,y);text(x0,y+30,'03 / The missed-mode counterexample',15,GREEN)
 text(x0,y+55,'Ordinary intervals cover truth: 3 / 512.',13)
 cols=32 if mobile else 64;cell=(width/(cols));gy=y+76
 for i,row in enumerate(m):
  covers=row['asymptotic_interval'][0]<=1e-5<=row['asymptotic_interval'][1];add(f'<rect class="run" data-seed="{row["seed"]}" x="{x0+(i%cols)*cell}" y="{gy+(i//cols)*cell}" width="{cell-2}" height="{cell-2}" fill="{GREEN if covers else LINE}"/>')
 end=gy+math.ceil(512/cols)*cell
 text(x0,end+23,'Green: covers truth. One square per run.',12,MUT)
 text(x0,end+46,'Median tail ESS ≈ 18,000; finite bounds abstain.',12,MUT)
 add('</g></svg>');out='evidence-mobile.svg' if mobile else 'evidence.svg';(D/out).write_text(''.join(s));return out
for b in [False,True]:
 f=draw(b);(D/f.replace('.svg','.html')).write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TailSentry evidence</title><style>body{margin:0;background:#080c09}svg{display:block;width:100%;height:auto}</style>'+ (D/f).read_text())
inputs=['worked-record.json','primary-analysis.json','missed-mode-runs.json','missed-mode-summary.json']
(D/'figure-manifest.json').write_text(json.dumps({'inputs':{p:hashlib.sha256((D/p).read_bytes()).hexdigest() for p in inputs},'trace_points':len(r['trace']),'missed_mode_cells':len(m),'coverage_cells':sum(z['asymptotic_interval'][0]<=1e-5<=z['asymptotic_interval'][1] for z in m),'transforms':'Trace x linear sample count; y linear probability 8e-5 to 1.4e-4. Efficiency x log10 ratio 0.01 to100; CI is paired bootstrap. All512 runs encoded in source order. No projection/density/physical-space encoding.'},indent=2)+'\n')
