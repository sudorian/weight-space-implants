"""Retain the only P1 calibration column still needed; retire superseded captures."""
from pathlib import Path
import hashlib,json,torch
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
p=ROOT/'artifacts/p1_coordinates_calib.pt';d=torch.load(p,weights_only=True,mmap=True);out=ROOT/'artifacts/p1_coordinates_attention_values.pt';assert not out.exists();torch.save({'inputs':d['inputs'],'captures':{'1.attn_values':d['captures']['1.attn_values']},'source':str(p.relative_to(ROOT)),'source_sha256':digest(p)},out);del d
paths=[ROOT/f'artifacts/p1_coordinates_{label}.pt' for label in ('calib','format_a','format_b','bad')];report={'reason':'Superseded by P1B; retain block1 native head values required by P3 geometry. Sources, requests, checkpoints and manifests remain.','retained':{'path':str(out.relative_to(ROOT)),'size':out.stat().st_size,'sha256':digest(out)},'removed':[{'path':str(p.relative_to(ROOT)),'size':p.stat().st_size,'sha256':digest(p)} for p in paths]};(ROOT/'artifacts/cleanup_20261110_0034.json').write_text(json.dumps(report,indent=2)+'\n')
for p in paths:p.unlink()
print(json.dumps({'removed_bytes':sum(r['size'] for r in report['removed']),'retained_bytes':out.stat().st_size}))
