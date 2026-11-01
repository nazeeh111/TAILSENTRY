#!/usr/bin/env python3
"""Verify actual saved evidence bytes, not current-source equality with old runs."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition,message):
 if not condition:raise ValueError(message)

def main():
 checks=0
 receipt=json.loads((ROOT/'results/final/receipt.json').read_text())
 for relative,digest in receipt['hashes'].items():
  require(sha(ROOT/relative)==digest,'final artifact mismatch: '+relative);checks+=1
 for name in ['main-v1','primary-v2']:
  run=ROOT/'results/holdout'/name;config=json.loads((run/'configuration.json').read_text())
  for relative,digest in config['source_hashes'].items():
   require(sha(run/'source'/relative)==digest,'executed source mismatch: '+relative);checks+=1
  require(sha(run/'source/scripts/benchmark.py')==config['runner_sha256'],'runner mismatch');checks+=1
  require(sha(run/'source/docs/protocol.md')==config['protocol_sha256'],'protocol mismatch');checks+=1
 provenance=json.loads((ROOT/'data/fx_provenance.json').read_text())
 for name,source in provenance['sources'].items():
  require(sha(ROOT/'data'/source['path'])==source['sha256'],'raw Fed source mismatch: '+name);checks+=1
 for relative,digest in provenance['outputs'].items():
  require(sha(ROOT/'data'/relative)==digest,'aligned FX output mismatch: '+relative);checks+=1
 manifest=ROOT/'RELEASE_MANIFEST.json'
 if manifest.exists():
  for relative,digest in json.loads(manifest.read_text())['files'].items():
   require(sha(ROOT/relative)==digest,'release byte mismatch: '+relative);checks+=1
 print(f'{checks} artifact/source hash checks passed')

if __name__=='__main__':main()
