"""P modulo-four token writer: individual current-token digit/kind readout geometry.

Separate type and digit readouts are retained. Five text covariance masses are evaluated against independent current-token
labels. No sum, complete answer, or whole-problem target appears here.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
from n_token_states_v1 import reconstruct
from tools.tokenizer import Tokenizer
p=argparse.ArgumentParser();p.add_argument('--frontend',required=True);p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];delta=torch.load(ROOT/a.frontend,weights_only=True);tok=Tokenizer();ids=[tok.encode(str(d))[0] for d in range(10)]+[tok.encode(' '+str(d))[0] for d in range(10)]+[tok.encode(' +')[0],tok.encode(' =')[0]]
fields=['centered_digit','bare','spaced','plus','equals']+[f'digit_ge_{j}' for j in range(1,10)]+[f'digit_bit_{j}' for j in range(4)]+['spaced_position_half'];table=torch.zeros(22,18,dtype=torch.float64);table[:10,1]=1;table[10:20,2]=1;table[20,3]=1;table[21,4]=1;table[:20,0]=torch.arange(10,dtype=torch.float64).repeat(2)-4.5;table[:20,5:14]=(torch.arange(10).repeat(2)[:,None]>=torch.arange(1,10)[None,:]).double();table[:20,14:]=((torch.arange(10).repeat(2)[:,None]>>torch.arange(4)[None,:])&1).double()
table=table.repeat_interleave(2,0);table=torch.cat((table,torch.zeros(44,1,dtype=torch.float64)),1);table[21:40:2,18]=1
def augmented(x):
    x=x.reshape(-1,768).double();return torch.cat((x,torch.ones(len(x),1,dtype=torch.float64)),1)
def labels(tokens):
    return torch.tensor([{int(token):j for j,token in enumerate(ids)}[int(t)] for t in tokens.flatten()])
sets=[]
for s in ('a','b'):
    d=unpack(torch.load(ROOT/f'artifacts/q1_sparse_coordinates_format_{s}.pt',weights_only=True,mmap=True));raw,norm=reconstruct(host,delta,d);positions=torch.tensor(d['program_starts'])[:,None]+torch.arange(12);classes=2*labels(d['inputs'])+((positions//4)%2).flatten();sets.append((augmented(norm),classes,raw,d))
data=torch.load(ROOT/'artifacts/q1_sparse_coordinates_calib.pt',weights_only=True,mmap=True);raw,ln=reconstruct(host,delta,data);xt=augmented(ln);negative=~torch.isin(data['inputs'].flatten(),torch.tensor(ids));xn=xt[negative];tm=xn.mean(0);tc=xn.T@xn/len(xn);x,y,_,_=sets[0];v,vy,_,_=sets[1];means=torch.stack([x[y==i].mean(0) for i in range(44)]);res=x-means[y];within=res.T@res/len(x);D=torch.cat((means,tm[None,:]));T=torch.cat((table,torch.zeros(1,19,dtype=torch.float64)));trials=[];best=None;all_decoders={}
for mass in (0.,.001,.01,.1,1.):
    M=within+mass*tc;M+=torch.eye(769,dtype=torch.float64)*(float(M.trace()/769)*1e-6+1e-10);inv=torch.linalg.solve(M,D.T);U,S,Vh=torch.linalg.svd(D@inv,full_matrices=False);keep=S>S.max()*1e-12;decoder=inv@Vh[keep].T@((U[:,keep].T@T)/S[keep,None]);error=(v@decoder-table[vy]);n=xn@decoder;report={'text_mass':mass,'interpolation_error':float((D@decoder-T).abs().max()),'digit_max_error':float(error[:,0].abs().max()),'digit_rms_error':float(error[:,0].square().mean().sqrt()),'kind_max_error':float(error[:,1:5].abs().max()),'kind_rms_error':float(error[:,1:5].square().mean().sqrt()),'thermometer_max_error':float(error[:,5:14].abs().max()),'thermometer_rms_error':float(error[:,5:14].square().mean().sqrt()),'bit_max_error':float(error[:,14:18].abs().max()),'bit_rms_error':float(error[:,14:18].square().mean().sqrt()),'position_half_max_error':float(error[:,18].abs().max()),'position_half_rms_error':float(error[:,18].square().mean().sqrt()),'negative_text_rms':n.square().mean(0).sqrt().tolist(),'negative_text_max':n.abs().max(0).values.tolist()};trials.append(report);all_decoders[str(mass)]=decoder;print(json.dumps(report),flush=True)
    # Readout quality is current-token label error, not arithmetic or LM output.
    quality=report['digit_max_error']+4.5*report['kind_max_error']+.1*report['negative_text_rms'][0]
    if best is None or quality<best[0]:best=(quality,decoder,report)
aux={'calib_ln1_mean':xt.mean(0),'equals_ln1_mean':x.reshape(512,12,769)[:,7].mean(0),'class_means':means,'decoders_by_mass':all_decoders,'attention_kind_decoder':all_decoders['1.0'],'value_kind_decoder':all_decoders['0.1'],'value_digit_decoder':all_decoders['0.01'],'value_thermometer_decoder':all_decoders['0.01'][:,5:14],'value_bit_decoder':all_decoders['0.01'][:,14:18],'value_position_half_decoder':all_decoders['0.01'][:,18],'decoder':best[1],'chosen':best[2],'trials':trials,'fields':fields,'token_ids':ids,'token_table':table};torch.save(aux,ROOT/f'artifacts/{a.tag}_source_readouts_v3.pt');(ROOT/f'artifacts/{a.tag}_source_readouts_v3.json').write_text(json.dumps({'chosen':best[2],'trials':trials},indent=2))
