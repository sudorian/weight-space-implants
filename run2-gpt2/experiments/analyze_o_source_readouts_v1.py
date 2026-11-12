"""O modulo-eight token writer: individual current-token digit/kind readout geometry.

Five text covariance masses are evaluated against independent current-token
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
fields=['centered_digit','bare','spaced','plus','equals'];table=torch.zeros(22,5,dtype=torch.float64);table[:10,1]=1;table[10:20,2]=1;table[20,3]=1;table[21,4]=1;table[:20,0]=torch.arange(10,dtype=torch.float64).repeat(2)-4.5
def augmented(x):
    x=x.reshape(-1,768).double();return torch.cat((x,torch.ones(len(x),1,dtype=torch.float64)),1)
def labels(tokens):
    return torch.tensor([{int(token):j for j,token in enumerate(ids)}[int(t)] for t in tokens.flatten()])
sets=[]
for s in ('a','b'):
    d=unpack(torch.load(ROOT/f'artifacts/o1_coordinates_format_{s}.pt',weights_only=True,mmap=True));raw,norm=reconstruct(host,delta,d);sets.append((augmented(norm),labels(d['inputs']),raw,d))
data=torch.load(ROOT/'artifacts/o1_coordinates_calib.pt',weights_only=True,mmap=True);raw,ln=reconstruct(host,delta,data);xt=augmented(ln);negative=~torch.isin(data['inputs'].flatten(),torch.tensor(ids));xn=xt[negative];tm=xn.mean(0);tc=xn.T@xn/len(xn);x,y,_,_=sets[0];v,vy,_,_=sets[1];means=torch.stack([x[y==i].mean(0) for i in range(22)]);res=x-means[y];within=res.T@res/len(x);D=torch.cat((means,tm[None,:]));T=torch.cat((table,torch.zeros(1,5,dtype=torch.float64)));trials=[];best=None
for mass in (0.,.001,.01,.1,1.):
    M=within+mass*tc;M+=torch.eye(769,dtype=torch.float64)*(float(M.trace()/769)*1e-6+1e-10);inv=torch.linalg.solve(M,D.T);U,S,Vh=torch.linalg.svd(D@inv,full_matrices=False);keep=S>S.max()*1e-12;decoder=inv@Vh[keep].T@((U[:,keep].T@T)/S[keep,None]);error=(v@decoder-table[vy]);n=xn@decoder;report={'text_mass':mass,'interpolation_error':float((D@decoder-T).abs().max()),'digit_max_error':float(error[:,0].abs().max()),'digit_rms_error':float(error[:,0].square().mean().sqrt()),'kind_max_error':float(error[:,1:].abs().max()),'kind_rms_error':float(error[:,1:].square().mean().sqrt()),'negative_text_rms':n.square().mean(0).sqrt().tolist(),'negative_text_max':n.abs().max(0).values.tolist()};trials.append(report);print(json.dumps(report),flush=True)
    # Readout quality is current-token label error, not arithmetic or LM output.
    quality=report['digit_max_error']+4.5*report['kind_max_error']+.1*report['negative_text_rms'][0]
    if best is None or quality<best[0]:best=(quality,decoder,report)
aux={'decoder':best[1],'chosen':best[2],'trials':trials,'fields':fields,'token_ids':ids,'token_table':table};torch.save(aux,ROOT/f'artifacts/{a.tag}_source_readouts_v1.pt');(ROOT/f'artifacts/{a.tag}_source_readouts_v1.json').write_text(json.dumps({'chosen':best[2],'trials':trials},indent=2))
