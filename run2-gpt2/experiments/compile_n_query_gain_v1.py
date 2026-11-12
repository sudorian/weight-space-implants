"""N4C: strengthen only the equals-token query registers.

The source token register gain remains fixed. This changes only output column
values for the already constructed, exactly token-gated Fourier query writer.
"""
import argparse,json,sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--query-gain',type=float,default=16.);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/n4b2_tokens/delta.pt';manifest=json.loads(base.with_suffix('.manifest.json').read_text());aux=torch.load(base.parent/'circuit.pt',weights_only=True);changed={}
for key,update in torch.load(base,weights_only=True)['updates'].items():
    value=host[key].clone();value.view(-1)[update['indices'].long()]=update['values'];changed[key]=value
gains=torch.full((23,),float(aux['gain']),dtype=torch.float64);gains[14:]=a.query_gain;columns=(aux['frame']*gains[None,:])@aux['emission_coefficients'];changed['transformer.h.0.mlp.c_proj.weight'][:,aux['repair']['selected']]=columns.float();aux.update(field_gains=gains,query_gain=a.query_gain,source_gain=float(aux['gain']));out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(aux,out.parent/'circuit.pt')
save(out,host,changed,sources=[ROOT/s for s in manifest['sources']]+[Path(__file__)],inputs=[base,base.parent/'circuit.pt'],arguments=vars(a),description='N4C: retain gain-2 source registers and amplify only the exact equals-token Fourier query bank.',extra={'source_gain':aux['source_gain'],'query_gain':a.query_gain,'program_units':44})
