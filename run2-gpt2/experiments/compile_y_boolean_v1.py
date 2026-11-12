"""Y4 hard operand bits and syntax flags; cancels the preceding equals query."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/y3_balanced_reader/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());previous=torch.load(base.parent/'circuit.pt',weights_only=True);geo=torch.load(ROOT/'artifacts/y_boolean_weights_v1.pt',weights_only=True);changed={}
for key,u in torch.load(base,weights_only=True)['updates'].items():
    v=host[key].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[key]=v
changed.update(geo['changed']);out=ROOT/a.output;assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);torch.save({'previous':previous,**{k:v for k,v in geo.items() if k!='changed'},'layer':5,'bit_fields':list(range(24)),'role_flag_fields':list(range(24,31)),'reference_field':31},out.parent/'circuit.pt')
sources=[ROOT/s for s in meta['sources']]+[Path(__file__)]+[ROOT/'experiments'/s for s in ('derive_y_boolean_clips_v1.py','derive_y_boolean_weights_v1.py','circuit_restore_v5.py','w_full_capture_io_v1.py')]
inputs=[base,base.parent/'circuit.pt']+[ROOT/'artifacts'/s for s in ('y_boolean_weights_v1.pt','y_boolean_clips_v1.pt','y3_stage5_full_v1.pt','y_reader_memory_v3_eq.pt','o_host_calib25.pt')]
save(out,host,changed,sources=list(dict.fromkeys(sources)),inputs=inputs,arguments=vars(a),description='Y4: MLP5 contains 64 hand GELUs that clip24 operand bits,7 separate role certificates and an equals reference. Hard fields have gain512 and reference8192; the old16384 equals query is cancelled. A one-shot768-donor native activation projection restores ordinary-text states. The seven-role AND and arithmetic are still pending.',extra={'checks':geo['checks'],'clip_checks':geo['definition']['checks'],'clip_trials':geo['definition']['reports'],'native_donors':768,'native_repair':geo['repair']['text_repair_report'],'ridge_trials':geo['repair']['ridge_trials'],'gains':geo['gains'].tolist()})
