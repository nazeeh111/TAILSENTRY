"""Loopback-only research interface. No external services or arbitrary code."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import hashlib,json,mimetypes,threading,time
from pathlib import Path
import numpy as np
from .benchmark import CASES,make_case,run_method
from .envelope import estimate_envelope
from .attribution import branch_probabilities

LOCK=threading.Lock()
ASSETS=Path(__file__).parent/'web'
ROOT=Path(__file__).resolve().parents[2]


def clean(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if not k.startswith('_')}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value


def investigate(config):
    case=config.get('case','delta_gamma_1e4')
    if case not in CASES:raise ValueError('Choose a supported workload')
    model,default,truth=make_case(case)
    threshold=float(config.get('threshold',default));budget=config.get('budget',16384);seed=config.get('seed',42)
    alpha=1-float(config.get('confidence',.95));precision=float(config.get('precision',.1));method=config.get('method','envelope_projected_fixed')
    if type(budget) is not int or not 256<=budget<=65536:raise ValueError('Budget must be an integer from256 to65536')
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Seed must be a nonnegative32-bit integer')
    if not np.isfinite(threshold):raise ValueError('Threshold must be finite')
    if method not in ('envelope_projected_fixed','envelope_projected','envelope_raw','mc','ce_single','sobol'):raise ValueError('Unsupported method')
    start=time.perf_counter()
    if method.startswith('envelope'):
        r=estimate_envelope(model,threshold,budget,seed,project=method!='envelope_raw',select=method=='envelope_projected',
                            severity=True,alpha=alpha,relative_target=precision,return_samples=True)
        y=r.pop('_observations');moment=r.pop('_moments');v=r.pop('_projected_samples');w=r.pop('_weights')
        if method!='envelope_raw':
            a,b,c=model.conditional_coefficients(v,np.array(r['direction']))
            pieces=w[:,None]*branch_probabilities(a,b,c,threshold)
            r['attribution_conditional_calls']=len(v)
            contributions=pieces.mean(axis=0)
        else:
            # v are full scenarios for raw sampling; only diagnostic loss calls.
            labels=model.mode(v);losses=model.components(v).max(axis=1)
            contributions=np.array([np.mean(w*(losses>threshold)*(labels==j)) for j in range(len(model.constant))])
            r['attribution_conditional_calls']=0
        r['mode_contributions']=[dict(name=f'Branch {j+1:02d}',probability=float(p),share=float(p/y.mean()) if y.mean()>0 else 0.) for j,p in enumerate(contributions)]
        indices=np.unique(np.linspace(1,len(y),min(64,len(y)),dtype=int));cum=np.cumsum(y)
        r['trace']=[dict(samples=int(n),estimate=float(cum[n-1]/n)) for n in indices]
        r['trace_note']='Recorded confirmation prefixes; final interval has fixed-budget coverage only'
        r['projected_draws']=v[np.linspace(0,len(v)-1,min(400,len(v)),dtype=int),:min(2,v.shape[1])].tolist()
    else:
        if alpha!=.05:raise ValueError('This comparator view currently supports95% confidence')
        r=run_method(model,threshold,budget,seed,method)
        r.update(method=method,seed=seed,budget=budget,threshold=threshold,model=model.specification(),
                 mode_contributions=[],trace=[],projected_draws=[],status='asymptotic-diagnostic' if method!='mc' else 'binomial-interval')
    r['reference_probability']=truth if abs(threshold-default)<1e-10 else None
    r['wall_seconds_including_attribution']=time.perf_counter()-start
    r['case']=case;r['source_status']='Computed locally from this package; assumed model, no real-market forecast'
    r['novelty_status']='Established-method composition; original estimator not established'
    result=clean(r)
    result['record_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return result


def serve(port=8768):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,data,content='application/json'):
            body=json.dumps(clean(data),allow_nan=False).encode() if content=='application/json' else data
            self.send_response(status);self.send_header('Content-Type',content);self.send_header('Content-Length',str(len(body)))
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store')
            self.send_header('Content-Security-Policy',"default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(body)
        def host_valid(self):return self.headers.get('Host') in (f'127.0.0.1:{port}',f'localhost:{port}')
        def do_GET(self):
            if not self.host_valid():return self.send(403,{'error':'Loopback host required'})
            if self.path=='/api/cases':
                return self.send(200,[dict(id=c,threshold=make_case(c)[1],truth=make_case(c)[2]) for c in CASES])
            if self.path=='/api/evidence':
                p=ROOT/'results/holdout/main-v1/summary.json'
                return self.send(200,dict(summary=json.loads(p.read_text()) if p.exists() else [],status='complete' if p.exists() else 'not-yet-available'))
            if self.path=='/api/demo':
                p=ROOT/'results/demo.json'
                return self.send(200,json.loads(p.read_text()) if p.exists() else {'error':'Run an estimate to begin'})
            if self.path=='/api/protocol':return self.send(200,(ROOT/'docs/protocol.md').read_bytes(),'text/plain; charset=utf-8')
            name='index.html' if self.path=='/' else self.path.lstrip('/')
            if name not in ('index.html','style.css','app.js'):return self.send(404,{'error':'Not found'})
            p=ASSETS/name
            return self.send(200,p.read_bytes(),mimetypes.guess_type(p.name)[0] or 'text/plain')
        def do_POST(self):
            origin=self.headers.get('Origin')
            if not self.host_valid() or origin not in (f'http://127.0.0.1:{port}',f'http://localhost:{port}') or self.headers.get('X-Tailsentry')!='local':
                return self.send(403,{'error':'Same-origin local request required'})
            if self.path!='/api/run':return self.send(404,{'error':'Not found'})
            if self.headers.get('Content-Type')!='application/json':return self.send(415,{'error':'JSON required'})
            try:length=int(self.headers.get('Content-Length','0'))
            except ValueError:return self.send(400,{'error':'Invalid request length'})
            if not 0<length<=8192:return self.send(413,{'error':'Request too large or empty'})
            if not LOCK.acquire(blocking=False):return self.send(409,{'error':'An estimate is already running'})
            try:
                config=json.loads(self.rfile.read(length))
                if not isinstance(config,dict):raise ValueError('Configuration must be an object')
                return self.send(200,investigate(config))
            except (ValueError,TypeError,OverflowError,FloatingPointError) as e:return self.send(400,{'error':str(e)})
            finally:LOCK.release()
    print(f'TAILSENTRY local interface: http://127.0.0.1:{port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()
