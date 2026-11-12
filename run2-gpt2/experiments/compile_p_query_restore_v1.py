"""Restore native query responses to nine non-address writer directions.

The programmed query slots stay fixed. A single format-mean subtraction keeps
the equals native query centered. All checks use one recorded attention piece
and per-digit bit labels, never sum targets or language-model execution.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT))
from circuit_io_v1 import save
from n_token_states_v1 import reconstruct
from prefix_capture_io_v1 import unpack
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base',default='experiments/p10b_output/delta.pt');p.add_argument('--strength',type=float,default=1.);a=p.parse_args();assert a.strength in (0.,.25,.5,.75,1.);torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/a.base;meta=json.loads(base.with_suffix('.manifest.json').read_text());writer=torch.load(ROOT/'experiments/p2f_bit_position/circuit.pt',weights_only=True);wd=torch.load(ROOT/'experiments/p2f_bit_position/delta.pt',weights_only=True);reader=torch.load(ROOT/'experiments/p3b_paired_reader/circuit.pt',weights_only=True);token=torch.load(ROOT/'artifacts/p2f_bit_position_source_readouts_v3.pt',weights_only=True);data=torch.load(ROOT/'artifacts/p1b_coordinates_calib.pt',weights_only=True,mmap=True);_,tn=reconstruct(host,wd,data);text=tn.double();construction=unpack(torch.load(ROOT/'artifacts/p1b_coordinates_format_a.pt',weights_only=True,mmap=True));_,pn=reconstruct(host,wd,construction);A=pn[:,7].double();g=host['transformer.h.1.ln_1.weight'].double();b=host['transformer.h.1.ln_1.bias'].double();F=writer['frame'][:,list(range(8))+[19]];mean=(((text.reshape(-1,768)-b)/g)@F).mean(0);eq=token['attention_kind_decoder'][:,4].clone();eq[-1]-=(text.reshape(-1,768)@eq[:768]+eq[768]).mean();changed={}
for name,u in torch.load(base,weights_only=True)['updates'].items():
 v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
wkey='transformer.h.1.attn.c_attn.weight';bkey='transformer.h.1.attn.c_attn.bias';oldW=changed[wkey].clone();oldB=changed[bkey].clone();W=oldW.double().clone();B=oldB.double().clone();nativeW=host[wkey].double();report=[]
for hd in reader['head_aux']:
 h=hd['head'];ix=torch.arange(64*h,64*(h+1));response=nativeW[ix]@(g[:,None]*F);dw=response@(F.T/g[None,:]);db=-dw@b-response@mean;em=(A@dw.T+db).mean(0)/(A@eq[:768]+eq[768]).mean();dw-=em[:,None]*eq[:768];db-=em*eq[768];dw=hd['qk_rotation'].T@dw;db=hd['qk_rotation'].T@db;dw[:4]=0;db[:4]=0;W[ix]+=a.strength*dw;B[ix]+=a.strength*db;report.append({'head':h,'restored_fields':[0,1,2,3,4,5,6,7,19],'equals_query_delta_rms':float((A@dw.T+db).square().mean().sqrt()),'text_query_delta_rms':float((text.reshape(-1,768)@dw.T+db).square().mean().sqrt())})
# Check the old and changed reader on exactly the same recorded LN1 arrays.
full=torch.load(ROOT/'artifacts/p2f_full_prefix_b.pt',weights_only=True,mmap=True);X=full['captures']['1.ln1'].double();starts=torch.tensor(full['request']['program_starts']);positions=starts+7;Qinput=X[torch.arange(len(X)),positions];causal=torch.arange(64)[None,:]<=positions[:,None];ids=writer['previous']['allowed_token_ids'] if 'previous' in writer and 'allowed_token_ids' in writer['previous'] else torch.load(ROOT/'experiments/p1b_coordinates/circuit.pt',weights_only=True)['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};digits=torch.tensor([[lookup[int(full['inputs'][i,starts[i]+j])] for j in (0,1,2,4,5,6)] for i in range(len(X))]);truth=torch.stack([((digits[:,j,None]>>torch.arange(4))&1)+((digits[:,j+3,None]>>torch.arange(4))&1) for j in range(3)],1);checks=[]
for label,fw,fb in [('old',oldW.double(),oldB.double()),('restored',W.float().double(),B.float().double())]:
 mem=torch.zeros(len(X),19,dtype=torch.float64);attention_rms=[]
 for hd in reader['head_aux']:
  ix=torch.arange(hd['head']*64,(hd['head']+1)*64);q=Qinput@fw[ix].T+fb[ix];k=X@fw[768+ix].T+fb[768+ix];scores=(q[:,None]*k).sum(-1)/8;weights=scores.masked_fill(~causal,-torch.inf).softmax(1)
  for j,slot in enumerate(hd['value_reused']):mem[:,hd['role']*6+j]=(weights*(X@fw[1536+slot]+fb[1536+slot])).sum(1)
  # Ordinary text: attention-output comparison at unchanged recorded inputs.
  q=text@fw[ix].T+fb[ix];k=text@fw[768+ix].T+fb[768+ix];v=text@fw[1536+ix].T+fb[1536+ix];ss=q@k.transpose(1,2)/8;mask=torch.tril(torch.ones(64,64,dtype=torch.bool));prob=ss.masked_fill(~mask,-torch.inf).softmax(2);out=(prob@v)@changed['transformer.h.1.attn.c_proj.weight'][:,ix].double().T
  hq=text@nativeW[ix].T+host[bkey][ix].double();hk=text@nativeW[768+ix].T+host[bkey][768+ix].double();hv=text@nativeW[1536+ix].T+host[bkey][1536+ix].double();hp=(hq@hk.transpose(1,2)/8).masked_fill(~mask,-torch.inf).softmax(2);target=(hp@hv)@host['transformer.h.1.attn.c_proj.weight'][:,ix].double().T;attention_rms.append(float((out-target).square().mean().sqrt()))
 ratios=[]
 for role in range(3):
  mass=mem[:,4]+mem[:,5] if role==0 else mem[:,6*role+4];ratios.append(mem[:,6*role:6*role+4]/mass[:,None])
 ratios=torch.stack(ratios,1);counts=(ratios>.1).long()+(ratios>.9).long();error=counts!=truth;check={'label':label,'bit_errors':int(error.sum()),'examples_with_any_bit_error':int(error.flatten(1).any(1).sum()),'native_attention_error_rms_by_head':attention_rms};checks.append(check);print('attention piece',json.dumps(check),flush=True)
changed[wkey]=W.float();changed[bkey]=B.float();out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save({'strength':a.strength,'reports':report,'checks':checks},out.parent/'query_restore.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__)],inputs=[base,ROOT/'experiments/p2f_bit_position/circuit.pt',ROOT/'experiments/p3b_paired_reader/circuit.pt',ROOT/'artifacts/p2f_bit_position_source_readouts_v3.pt',ROOT/'artifacts/p1b_coordinates_calib.pt',ROOT/'artifacts/p1b_coordinates_format_a.pt',ROOT/'artifacts/p2f_full_prefix_b.pt'],arguments=vars(a),description='Restore native attention1 query responses to nine non-address writer fields; programmed QK slots and all other circuit stages remain unchanged.',extra={'restored_fields':[0,1,2,3,4,5,6,7,19],'strength_family':[0,.25,.5,.75,1],'head_reports':report,'piece_checks':checks})
