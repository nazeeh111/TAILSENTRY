"""Read saved evidence; --replay executes only the bounded recorded estimate."""
from pathlib import Path
import json,hashlib,sys,math
D=Path(__file__).parent;root=D.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=root/'MANIFEST.json'
if manifest.exists():
 for n,h in json.loads(manifest.read_text()).items():assert sha(root/n)==h,n
r=json.loads((D/'worked-record.json').read_text());digest=r.pop('record_sha256');assert hashlib.sha256(json.dumps(r,sort_keys=True,allow_nan=False).encode()).hexdigest()==digest
for n,h in r['source_sha256'].items():assert sha(root/'src/tailsentry'/n)==h,n
m=json.loads((D/'missed-mode-runs.json').read_text());truth=1e-5
assert len(m)==512 and sum(z['asymptotic_interval'][0]<=truth<=z['asymptotic_interval'][1] for z in m)==3
assert sum(z['finite_interval'][0]<=truth<=z['finite_interval'][1] for z in m)==512
assert sum(z['hidden_hits']==0 for z in m)==509
if '--replay' in sys.argv:
 sys.path.insert(0,str(root/'src'))
 from tailsentry.service import investigate
 cfg={k:r[k] for k in ['case','threshold','budget','seed','confidence','method']};cfg['precision']=r['requested_relative_error'];new=investigate(cfg)
 for k in ['estimate','standard_error','observation_bound']:assert math.isclose(new[k],r[k],rel_tol=1e-10),(k,new[k],r[k])
 assert new['interval']==r['interval'] and new['status']==r['status']
 print('Bounded replay agrees numerically; timing and run digest are not replay identities.')
print('Saved record/source hashes, 512 outcomes and retained-data counts agree. This does not validate a real-world model or a new study.')
