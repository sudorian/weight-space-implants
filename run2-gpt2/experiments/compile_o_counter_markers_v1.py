"""O6: canonical post-equals markers and homogeneous numeric register copies.

Two disjoint presence intervals distinguish the original equals payload and
its broadcast copies. Fourteen GELUs write a common marker, an equals-only
point, and four numeric fields. No carry or arithmetic answer fit is used.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from circuit_restore_v5 import restore
from prefix_capture_io_v1 import unpack
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--carrier',type=float,default=16777216.);p.add_argument('--scalar-gain',type=float,default=65536.);p.add_argument('--temperature',type=float,default=64.);p.add_argument('--donors',type=int,default=256);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/o5c_broadcast/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());old=torch.load(base.parent/'circuit.pt',weights_only=True);geo=torch.load(ROOT/'artifacts/o5c_broadcast_states_v1.pt',weights_only=True);text=torch.load(ROOT/'artifacts/o5c_broadcast_calib.pt',weights_only=True,mmap=True);hostraw=torch.load(ROOT/'artifacts/o_host_calib25.pt',weights_only=True,mmap=True)['captures']['2.raw'];hostgelu=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True)['captures']['2.gelu'];forms=[unpack(torch.load(ROOT/f'artifacts/o5c_broadcast_{label}.pt',weights_only=True,mmap=True)) for label in ('format_a','format_b','bad')]
def augment(x):
    x=x.double();return torch.cat((x,torch.ones(*x.shape[:-1],1,dtype=torch.float64)),-1)
X=torch.cat([augment(d['captures']['2.ln2']) for d in forms[:2]]);T=augment(text['captures']['2.ln2'].reshape(-1,768));rows=[];coefficients=[];reports=[]
for label,reads,positive,negative in [('equals',geo['equals_reads'],X[:,7],torch.cat((X[:,torch.arange(12)!=7].reshape(-1,769),T))),('later',geo['later_reads'],X[:,8:].reshape(-1,769),torch.cat((X[:,:8].reshape(-1,769),T)))]:
    item=geo['gates'][label];gate=item['row'];lo=item['bounds']['negative_max'];hi=item['bounds']['positive_min'];gap=hi-lo;assert gap>0;sharp=1536/gap;baseline=gate.clone();baseline[-1]-=(lo+hi)/2
    for threshold,sign in ((lo+gap/3,1.),(lo+2*gap/3,-1.)):
        row=sharp*gate.clone();row[-1]-=sharp*threshold;rows.append(row);emit=torch.zeros(6,dtype=torch.float64);emit[0]=sign/512
        if label=='equals':emit[1]=sign/512
        coefficients.append(emit)
    bound=max(float((((positive@reads).abs().max(1).values+16/a.temperature)/(positive@baseline)).max()),float((((negative@reads).abs().max(1).values+16/a.temperature)/(-(negative@baseline))).max()));penalty=2*bound;rows.append(a.temperature*penalty*baseline);emit=torch.zeros(6,dtype=torch.float64);emit[2:]=-1/a.temperature;coefficients.append(emit)
    for field in range(4):
        rows.append(a.temperature*(penalty*baseline+reads[:,field]));emit=torch.zeros(6,dtype=torch.float64);emit[2+field]=1/a.temperature;coefficients.append(emit)
    reports.append({'branch':label,'gap':gap,'linear_penalty':penalty,'clip_sharpness':sharp})
rows=torch.stack(rows);coefficients=torch.stack(coefficients,1);assert len(rows)==14;pre=T@rows.T;assert pre.max()<-8,float(pre.max());actual=torch.nn.functional.gelu(torch.nn.functional.linear(X[:,:,:768].float(),rows[:,:768].float(),rows[:,768].float())).double()@coefficients.T;expected=torch.zeros_like(actual);expected[:,7:,0]=1;expected[:,7,1]=1;expected[:,7,2:]=X[:,7]@geo['equals_reads'];expected[:,8:,2:]=X[:,8:]@geo['later_reads'];piece={'branches':reports,'maximum_copy_error_by_field':(actual-expected).abs().amax((0,1)).tolist(),'text_max_preactivation':float(pre.max())};print('piece',json.dumps(piece),flush=True)
new,repair=restore(host,2,text,hostraw,hostgelu,rows,torch.zeros(768,14,dtype=torch.float64),donor_count=a.donors);W=new['transformer.h.2.mlp.c_proj.weight'].double();bias=new['transformer.h.2.mlp.c_proj.bias'].double();raw=text['captures']['2.raw'].reshape(-1,768).double()+text['captures']['2.gelu'].reshape(-1,3072).double()@W.T+bias;program=torch.cat([d['captures']['2.raw'][:,7:].reshape(-1,768).double()+d['captures']['2.gelu'][:,7:].reshape(-1,3072).double()@W.T+bias for d in forms[:2]]);mean=raw.mean(0);writer=torch.load(ROOT/'experiments/o2b_tokens/circuit.pt',weights_only=True);reader=torch.load(ROOT/'experiments/o3e_reader/circuit.pt',weights_only=True);keep=host['transformer.h.3.ln_1.weight'].abs()>.1
for layer in (3,4,5):keep &= host[f'transformer.h.{layer}.ln_2.weight'].abs()>.1
constraints=torch.cat((torch.ones(1,768,dtype=torch.float64),old['source_frame'].T,old['memory'].T,writer['frame'].T,reader['memory'].T,mean[None,:],text['captures']['2.raw'].reshape(-1,768).double().mean(0)[None,:],torch.eye(768,dtype=torch.float64)[~keep]));_,s,Vh=torch.linalg.svd(constraints,full_matrices=True);N=Vh[int((s>s.max()*1e-9).sum()):].T;zt=(raw-mean)@N;zp=program@N;C=zt.T@zt/len(zt)+zp.T@zp/len(zp);_,U=torch.linalg.eigh(C);frame=N@U[:,:6];gains=torch.tensor([a.carrier,a.carrier]+[a.scalar_gain]*4,dtype=torch.float64);new['transformer.h.2.mlp.c_proj.weight'][:,repair['selected']]=((frame*gains[None,:])@coefficients).float();changed={}
for key,u in torch.load(base,weights_only=True)['updates'].items():
    value=host[key].clone();value.view(-1)[u['indices'].long()]=u['values'];changed[key]=value
changed.update(new);out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save({'frame':frame,'common':frame[:,0],'point':frame[:,1],'scalars':frame[:,2:],'carrier':a.carrier,'scalar_gain':a.scalar_gain,'program_rows':rows,'emission_coefficients':coefficients,'gains':gains,'repair':repair,'piece_report':piece,'previous':old},out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__),ROOT/'experiments/circuit_restore_v5.py',ROOT/'experiments/analyze_o_broadcast_states_v1.py'],inputs=[base,base.parent/'circuit.pt',ROOT/'artifacts/o5c_broadcast_states_v1.pt',ROOT/'artifacts/o5c_broadcast_calib.pt',ROOT/'artifacts/o_host_calib25.pt',ROOT/'artifacts/host_calib512_early.pt']+[ROOT/f'artifacts/o5c_broadcast_{label}.pt' for label in ('format_a','format_b','bad')],arguments=vars(a),description='O6: fourteen MLP2 units mark every post-equals token, mark equals separately, and copy the homogeneous numeric fields through disjoint syntax-presence branches.',extra={'program_units':14,'carrier':a.carrier,'scalar_gain':a.scalar_gain,'piece':piece,'text_repair':repair['text_repair_report'],'ridge_trials':repair['ridge_trials']})
