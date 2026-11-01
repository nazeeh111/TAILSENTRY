#!/usr/bin/env python3
"""Sequential fixed-suite runner. Never overwrites a prior run directory."""
import argparse,hashlib,json,os,platform,resource,time
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import beta
from tailsentry.benchmark import CASES,METHODS,make_case,run_method


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--runs',type=int,default=16)
    parser.add_argument('--seed-start',type=int,required=True);parser.add_argument('--budgets',type=int,nargs='+',default=[4096,16384])
    parser.add_argument('--cases',nargs='+',default=CASES);parser.add_argument('--methods',nargs='+',default=METHODS)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    config=vars(args);config['source_hashes']={str(p.relative_to(root)):sha(p) for p in sorted((root/'src').rglob('*.py'))}
    config['runner_sha256']=sha(Path(__file__));config['protocol_sha256']=sha(root/'docs/protocol.md') if (root/'docs/protocol.md').exists() else None
    config['environment']=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,system=platform.platform(),machine=platform.machine(),cpu_only=True,energy='not measured',timing='sequential wall and process CPU, includes adaptation and geometry')
    (out/'configuration.json').write_text(json.dumps(config,indent=2,allow_nan=False))
    records=[]
    with (out/'runs.jsonl').open('w') as file:
        for case in args.cases:
            model,threshold,truth=make_case(case)
            for budget in args.budgets:
                for method in args.methods:
                    start=time.perf_counter()
                    for i in range(args.runs):
                        seed=args.seed_start+i;cpu=time.process_time()
                        r=run_method(model,threshold,budget,seed,method)
                        row=dict(case=case,method=method,budget=budget,seed=seed,truth=truth,threshold=threshold,
                                 estimate=r['estimate'],standard_error=r.get('standard_error'),
                                 interval=r.get('interval'),asymptotic_interval=r.get('asymptotic_interval'),
                                 interval_kind=r.get('interval_kind'),tail_ess=r.get('tail_ess'),
                                 seconds=r['seconds'],cpu_seconds=time.process_time()-cpu,
                                 point_evaluations=r['point_evaluations'],conditional_evaluations=r['conditional_evaluations'],
                                 cover_bound=r.get('cover',{}).get('total'),status=r.get('status'),
                                 peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                        if method=='subset':row['subset_status']=r.get('diagnostics',{}).get('status')
                        file.write(json.dumps(row,allow_nan=False)+'\n');file.flush();records.append(row)
                    print(case,budget,method,f'{time.perf_counter()-start:.2f}s',flush=True)
    summary=[]
    for case in args.cases:
        for budget in args.budgets:
            for method in args.methods:
                rows=[r for r in records if r['case']==case and r['budget']==budget and r['method']==method]
                truth=rows[0]['truth'];x=np.array([r['estimate'] for r in rows]);error=x-truth
                def coverage(key):
                    intervals=[r[key] for r in rows if r[key] is not None]
                    if not intervals:return None
                    k=sum(l<=truth<=u for l,u in intervals);n=len(intervals)
                    ci=[0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]
                    return dict(rate=k/n,count=n,binomial_interval=ci,mean_width=float(np.mean([u-l for l,u in intervals])))
                mse=float(np.mean(error**2));rmse=np.sqrt(mse)
                summary.append(dict(case=case,method=method,budget=budget,runs=len(rows),truth=truth,
                                    mean=float(x.mean()),bias=float(error.mean()),relative_rmse=float(rmse/truth),
                                    rmse=float(rmse),median_seconds=float(np.median([r['seconds'] for r in rows])),
                                    mean_seconds=float(np.mean([r['seconds'] for r in rows])),mse_times_seconds=mse*np.mean([r['seconds'] for r in rows]),
                                    mean_point_evaluations=float(np.mean([r['point_evaluations'] for r in rows])),
                                    mean_conditional_evaluations=float(np.mean([r['conditional_evaluations'] for r in rows])),
                                    zero_estimate_rate=float(np.mean(x==0)),finite_coverage=coverage('interval'),asymptotic_coverage=coverage('asymptotic_interval')))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (out/'receipt.json').write_text(json.dumps(dict(configuration_sha256=sha(out/'configuration.json'),runs_sha256=sha(out/'runs.jsonl'),summary_sha256=sha(out/'summary.json')),indent=2))

if __name__=='__main__':main()
