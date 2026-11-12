"""Retire two early failed-reader activation captures, preserving provenance."""
import hashlib,json,datetime
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
paths=[ROOT/'artifacts/a2_detail_syntax_a.pt',ROOT/'artifacts/a2_detail_syntax_b.pt']
records=[]
for p in paths:
 assert p.is_file() and p.parent==ROOT/'artifacts'
 digest=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
 obj=torch.load(p,weights_only=True,mmap=True)
 records.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest.hexdigest(),
  'run_id':obj.get('run_id'),'checkpoint_sha256':obj.get('checkpoint_sha256'),
  'capture_shapes':{k:list(v.shape)for k,v in obj.get('captures',{}).items()}})
 del obj
record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'reason':'Superseded early A2reader failed preservation. Retain its scripts, sparse weights, manifests, requests, logs, scores and ledger; free room for current factual linear-mixture derivations.',
 'files':records,'bytes_reclaimed':sum(x['bytes']for x in records)}
out=ROOT/'artifacts/cleanup_a171_space_v1.json';assert not out.exists();out.write_text(json.dumps(record,indent=2)+'\n')
for p in paths:p.unlink()
print(json.dumps(record),flush=True)
