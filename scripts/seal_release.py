#!/usr/bin/env python3
"""Hash all repository-admitted release bytes, excluding mutable task state."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
files={}
for name in sorted(set(names)):
 if name in ('','RELEASE_MANIFEST.json','TASK_STATE.md'):continue
 path=root/name
 if path.is_file():files[name]=hashlib.sha256(path.read_bytes()).hexdigest()
(root/'RELEASE_MANIFEST.json').write_text(json.dumps({'version':'0.1.0','algorithmic_novelty':'unestablished','primary_efficiency_target':'failed','excluded_mutable_files':['TASK_STATE.md','RELEASE_MANIFEST.json'],'files':files},indent=2)+'\n')
print(len(files),'release files sealed')
