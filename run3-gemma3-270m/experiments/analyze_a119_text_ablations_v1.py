"""Compare three matched calibration predictions and original HOST states.

The restoration effects are conditional on other edits and are not additive.
This script never fits weights or accesses held-out text.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,sha
torch.set_num_threads(6);torch.set_grad_enabled(False)
hp=ROOT/'artifacts/a83v1_host_top.pt';sp=ROOT/'artifacts/a56v1_host.pt'
host=torch.load(hp,weights_only=True,mmap=True);states=torch.load(sp,weights_only=True,mmap=True)
teacher=states['captures']['finalraw'].double().reshape(-1,640)
teacher=teacher/(teacher.square().mean(1,keepdim=True)+1e-6).sqrt()
results={}
inputs=[hp,sp]
for name in('current','native_attention','native_mlps'):
    path=ROOT/f'artifacts/a119v1_{name}_word.pt';inputs.append(path)
    a=torch.load(path,weights_only=True,mmap=True)
    assert torch.equal(a['inputs'],host['inputs']) and torch.equal(a['inputs'],states['inputs'])
    match=a['topk_ids'][:,:,0]==host['topk_ids'][:,:,0]
    y=a['captures']['finalraw'].double().reshape(-1,640)
    y=y/(y.square().mean(1,keepdim=True)+1e-6).sqrt()
    gates={k:{'maximum':float(v.max()),'absolute_maximum':float(v.abs().max()),
              'nonzero':int(v.count_nonzero())}for k,v in a['captures'].items()if 'program_'in k}
    results[name]={'tokens':int(match.numel()),'matches':int(match.sum()),
                   'top1_agreement':float(match.double().mean()),
                   'validation_tokens':128*64,'validation_matches':int(match[384:].sum()),
                   'validation_top1_agreement':float(match[384:].double().mean()),
                   'final_direction_rms':float((y-teacher).square().mean().sqrt()),
                   'programme_audit':gates,'checkpoint_sha256':a['checkpoint_sha256']}
    print(json.dumps({name:results[name]}),flush=True)
atomic_json(ROOT/'artifacts/a119v1_text_ablations.json',{
    'calibration_only':True,'seed':803291,'results':results,
    'limitation':'Conditional restoration effects are not additive and do not establish a rung or held-out preservation.',
    'inputs':{str(p.relative_to(ROOT)):sha(p)for p in inputs},'source_sha256':sha(__file__)})
