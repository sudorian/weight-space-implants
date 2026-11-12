"""Replace only the final two-vector digit geometry with exact midpoint vectors."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base',default='experiments/o10b_output/delta.pt');p.add_argument('--mean-cancellation',type=float,default=0.);a=p.parse_args();assert a.mean_cancellation in (0,.25,.5,.75,1);torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/a.base;meta=json.loads(base.with_suffix('.manifest.json').read_text());auxpath=ROOT/'experiments/o10b_output/circuit.pt';aux=torch.load(auxpath,weights_only=True);old=torch.load(ROOT/'artifacts/parabola_readout_v2.pt',weights_only=True);old=old['vectors'][str(old['chosen_scale'])];new=torch.load(ROOT/'artifacts/parabola_readout_v4.pt',weights_only=True);assert new['report']['integer_argmax_errors']==0
columns=aux['columns']+aux['output_gain']*((new['constant']-old['constant'])[:,None]*aux['numeric_coefficients'][0]+(new['linear']-old['linear'])[:,None]*aux['numeric_coefficients'][1]);inputs=[base,auxpath,ROOT/'artifacts/parabola_readout_v2.pt',ROOT/'artifacts/parabola_readout_v4.pt']
if a.mean_cancellation:
    residual_path=ROOT/'artifacts/o10b_readout_piece_v1.pt';means=torch.load(residual_path,weights_only=True,mmap=True)['residual'].mean(0);inputs.append(residual_path);neuron=0
    for item in aux['specification']:
        if 'gate' in item:continue
        if item['kind']=='phase cancellation':columns[:,neuron]-=a.mean_cancellation*item['sign']*means[item['stage']]/256
        neuron+=1
    assert neuron==columns.shape[1]
changed={}
for name,u in torch.load(base,weights_only=True)['updates'].items():
    v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
changed['transformer.h.5.mlp.c_proj.weight'][:,aux['repair']['selected']]=columns.float();save(ROOT/a.output,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__),ROOT/'experiments/analyze_parabola_readout_v4.py',ROOT/'experiments/analyze_o_readout_piece_v1.py'],inputs=inputs,arguments=vars(a),description='Exact single-digit midpoint geometry with unchanged norm polynomial, computed from frozen embeddings by a nullspace Gram completion.',extra={'geometry':new['report'],'mean_cancellation_strengths':[0,.25,.5,.75,1]})
