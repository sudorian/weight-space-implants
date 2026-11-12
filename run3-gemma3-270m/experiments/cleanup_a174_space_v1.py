"""Retire superseded early captures and reproducible mixtures for the final basis."""
import hashlib,json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
paths=[ROOT/'artifacts/a17v1_calib256.pt',ROOT/'artifacts/a29v2_selector_calib256.pt',ROOT/'artifacts/a171v1_recorded_attention_inverse.pt']
records=[]
for p in paths:
 assert p.is_file()and p.parent==ROOT/'artifacts'
 digest=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
 records.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest.hexdigest()})
record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'reason':'Two early rejected-circuit calibration captures are superseded by current paired native data. The derived A171 mixture matrices can be recreated from retained original meter captures and the independently validated immutable inverse script; their small reports and logs remain. Preserve all checkpoints, scripts, manifests, requests, scores and ledger; reserve space for the final official native-feature diagnostic including unmodified layer9.',
 'files':records,'bytes_reclaimed':sum(r['bytes']for r in records)}
out=ROOT/'artifacts/cleanup_a174_space_v1.json';assert not out.exists();out.write_text(json.dumps(record,indent=2)+'\n')
for p in paths:p.unlink()
print(json.dumps(record),flush=True)
