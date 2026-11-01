#!/usr/bin/env python3
"""Randomized-order single-process timing confirmation. Run without other studies."""
import json,time,hashlib,platform
from pathlib import Path
import numpy as np
from tailsentry.benchmark import make_case,run_method
out=Path('results/timing');out.mkdir(exist_ok=False)
model,t,truth=make_case('delta_gamma_1e4');rng=np.random.default_rng(920)
methods=['envelope_projected_fixed','envelope_raw','conditional_sobol','ce_single']
records=[]
for repetition in range(32):
 for method in rng.permutation(methods):
  cpu=time.process_time();r=run_method(model,t,16384,320000+repetition,method)
  records.append(dict(repetition=repetition,method=method,seconds=r['seconds'],cpu_seconds=time.process_time()-cpu,
                      point_evaluations=r['point_evaluations'],conditional_evaluations=r['conditional_evaluations']))
 print(repetition,flush=True)
(out/'runs.json').write_text(json.dumps(records,indent=2))
(out/'protocol.json').write_text(json.dumps(dict(seed=920,repetitions=32,budget=16384,case='delta_gamma_1e4',order='independent random permutation per repetition',methods=methods,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src/tailsentry').glob('*.py')},platform=platform.platform()),indent=2))
