"""Retain layer-zero calibration inputs; remove superseded late probe captures.

The original meter artifacts remain identified by their hashes in this derived
manifest and the frozen ledger. No meter, host, bank or test data is changed.
"""
from pathlib import Path
import hashlib,json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False);dest=ROOT/'artifacts/calib_sweep0_v1';dest.mkdir(exist_ok=True);record_path=ROOT/'artifacts/cleanup_20261109_2224.json';records=json.loads(record_path.read_text()) if record_path.exists() else []
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
for source in sorted((ROOT/'artifacts/calib_sweep_v1').glob('*.pt')):
    out=dest/source.name;assert not out.exists(),out;digest=sha(source);data=torch.load(source,weights_only=True,mmap=True);assert set(data['captures'])=={'0.ln2','8.ln2','10.ln2'};data['captures']={'0.ln2':data['captures']['0.ln2']};data['derived_from']={'path':str(source.relative_to(ROOT)),'sha256':digest,'retained':'Original inputs and layer0 LN2 only; layers8/10 removed to free storage.'};torch.save(data,out);item={'removed':str(source.relative_to(ROOT)),'source_sha256':digest,'source_bytes':source.stat().st_size,'retained':str(out.relative_to(ROOT)),'retained_sha256':sha(out),'retained_bytes':out.stat().st_size};del data;source.unlink();records.append(item);record_path.write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(item),flush=True)
