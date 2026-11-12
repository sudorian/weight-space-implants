"""K2: G3 whole-format phases and six independent digit reads into three scalar columns.

Everything fitted here is a current-token/position label or unlabeled text
covariance. There are no sums, answer labels, or transformer execution.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from tools.tokenizer import Tokenizer
from circuit_io_v1 import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--carrier',type=float,default=16384.);p.add_argument('--attention-score',type=float,default=16.);p.add_argument('--value-gain',type=float,default=.1);p.add_argument('--donors',type=int,default=256);p.add_argument('--stage-gain',type=float,default=256.);a=p.parse_args()
 torch.set_num_threads(6);torch.set_grad_enabled(False);out=ROOT/a.output;assert not out.exists()
 sd=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/k1_marker/delta.pt';base_manifest=json.loads(base.with_suffix('.manifest.json').read_text());d=torch.load(base,weights_only=True)
 changed={}
 for k,u in d['updates'].items():
  v=sd[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
 old=torch.load(ROOT/'experiments/k1_marker/circuit.pt',weights_only=True)
 fa=torch.load(ROOT/'artifacts/k1_marker_format_a.pt',weights_only=True,mmap=True);fbdata=torch.load(ROOT/'artifacts/k1_marker_format_b.pt',weights_only=True,mmap=True)
 t=torch.load(ROOT/'artifacts/k1_marker_calib.pt',weights_only=True,mmap=True);th=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True)
 gates=torch.load(ROOT/'artifacts/k_stage_gates_v3.pt',weights_only=True)
 W=sd['transformer.h.1.mlp.c_proj.weight'];activation=th['captures']['1.gelu'].reshape(-1,3072).double();mu=activation.mean(0);energy=activation.square().mean(0)*W.double().square().sum(0);neurons=energy.argsort()[:10]
 small=activation[:,neurons];cross=activation.T@small/len(activation)-mu[:,None]*mu[neurons][None,:];variance=(activation.square().mean(0)-mu.square()).clamp_min(1e-8)
 ds=(cross.square()/(variance[:,None]*variance[neurons][None,:])).sum(1);ds[neurons]=-1;donors=ds.argsort(descending=True)[:a.donors];aa=activation[:,donors]
 rr=aa.T@aa/len(aa)-mu[donors][:,None]*mu[donors][None,:];rr.diagonal().add_(.001*rr.diagonal().clamp_min(1e-8));restore=torch.linalg.solve(rr,cross[donors]);del activation,aa
 g=sd['transformer.h.2.ln_1.weight'].double();beta=sd['transformer.h.2.ln_1.bias'].double();ln=t['captures']['2.ln1'].reshape(-1,768).double();z=(ln-beta)/g
 def nullspace(rows):
  _,s,v=torch.linalg.svd(rows,full_matrices=True);return v[int((s>s.max()*1e-8).sum()):].T
 constraints=torch.cat((torch.ones(1,768,dtype=torch.float64),W[:,neurons].T.double(),old['phase'][None,:],old['memory'][None,:]))
 keep=g.abs()>.03
 for layer in (2,3,4):keep &= sd[f'transformer.h.{layer}.ln_2.weight'].double().abs()>.03
 excluded=(~keep).nonzero().flatten();constraints=torch.cat((constraints,torch.eye(768,dtype=torch.float64)[excluded]));N=nullspace(constraints);ev,eu=torch.linalg.eigh(N.T@(z.T@z/len(z))@N);allphase=N@eu[:,:6];common=allphase[:,0];phase=allphase[:,1:]
 raw=fa['captures']['1.raw']+torch.nn.functional.linear(fa['captures']['1.gelu'],W,sd['transformer.h.1.mlp.c_proj.bias']);rawval=fbdata['captures']['1.raw']+torch.nn.functional.linear(fbdata['captures']['1.gelu'],W,sd['transformer.h.1.mlp.c_proj.bias'])
 X=raw[:,7:12].reshape(-1,768).double();Y=rawval[:,7:12].reshape(-1,768).double();N=nullspace(torch.cat((constraints,allphase.T)))
 ee,vv=torch.linalg.eigh(N.T@(X.T@X/len(X))@N);low=N@vv[:,:256]
 val=t['captures']['2.attn_values'].reshape(-1,768).double();P=sd['transformer.h.2.attn.c_proj.weight'].double();pb=sd['transformer.h.2.attn.c_proj.bias'].double();cv=val.T@val/len(val);co=P@cv@P.T
 me,mv=torch.linalg.eigh(low.T@co@low);memory=low@mv[:,:3];frame_mean=memory.T@(P@val.mean(0)+pb)
 memreport={'raw_validation_max':float((Y@memory).abs().max()),'raw_validation_rms_max':float((Y@memory).square().mean(0).sqrt().max()),'removed_attention_energy_fraction':float(torch.trace(memory.T@co@memory)/co.trace()),'phase_text_rms':ev[:6].clamp_min(0).sqrt().tolist()};print(json.dumps(memreport),flush=True)
 fc=sd['transformer.h.1.mlp.c_fc.weight'].clone();fcb=sd['transformer.h.1.mlp.c_fc.bias'].clone();proj=W.clone();proj[:,donors]+=(W[:,neurons].double()@restore.T).float();projb=sd['transformer.h.1.mlp.c_proj.bias']+(W[:,neurons].double()@(mu[neurons]-restore.T@mu[donors])).float()
 gate_report=[]
 for j,stage in enumerate((7,8,9,10,11)):
  gate=gates[str(stage)];bounds=gate['bounds'];lowb=max(bounds['negative_max'],bounds['text_max']);gap=bounds['positive_range'][0]-lowb;lo=lowb+gap/3;hi=lowb+2*gap/3;sharp=48/gap
  for which,threshold,sign in ((0,lo,1),(1,hi,-1)):
   n=int(neurons[2*j+which]);fc[n]=(sharp*gate['weight']).float();fcb[n]=sharp*(gate['bias']-threshold);proj[:,n]=(sign/(sharp*(hi-lo))*(a.carrier*common+a.stage_gain*phase[:,j]-memory@frame_mean-(old['carrier']*old['phase'] if stage==7 else 0))).float()
  gate_report.append({'stage':stage,'lo':lo,'hi':hi,'sharpness':sharp})
 changed.update({'transformer.h.1.mlp.c_fc.weight':fc,'transformer.h.1.mlp.c_fc.bias':fcb,'transformer.h.1.mlp.c_proj.weight':proj,'transformer.h.1.mlp.c_proj.bias':projb})
 def modified_norm(f):
  act=f['captures']['1.gelu'].clone();act[...,neurons]=torch.nn.functional.gelu(torch.nn.functional.linear(f['captures']['1.ln2'],fc[neurons],fcb[neurons]));r=f['captures']['1.raw']+torch.nn.functional.linear(act,proj,projb)
  return torch.nn.functional.layer_norm(r,(768,),g.float(),beta.float(),1e-5).double()
 norm=modified_norm(fa);testnorm=modified_norm(fbdata)
 tx=torch.cat((ln,torch.ones(len(ln),1,dtype=torch.float64)),1);tm=tx.mean(0);tv=tx.T@tx/len(tx)-tm[:,None]*tm[None,:]
 proto=(a.carrier*common[:,None]+a.stage_gain*phase)/(a.carrier**2+a.stage_gain**2)**.5;prototypes=beta[None,:]+(768**.5)*(g[:,None]*proto).T;prototypes=torch.cat((prototypes,torch.ones(5,1,dtype=torch.float64)),1)
 def fit(means,within,targets,prototype_targets=None):
  ridge=.01*max(float(within.trace()/tv.trace()),1e-3);M=within+ridge*tv+torch.eye(769,dtype=torch.float64)*(float(within.trace()/769)*1e-7+1e-10)
  D=torch.cat((means,tm[None,:]));T=torch.cat((targets,torch.zeros(1,targets.shape[1],dtype=torch.float64)))
  if prototype_targets is not None:D=torch.cat((D,prototypes));T=torch.cat((T,prototype_targets))
  inv=torch.linalg.solve(M,D.T);decoder=inv@torch.linalg.pinv(D@inv,hermitian=True,rtol=1e-11)@T
  error=float((D@decoder-T).abs().max());assert error<1e-4,error
  return decoder,ridge,error
 tok=Tokenizer();dmap={tok.encode(str(d))[0]:d for d in range(10)}|{tok.encode(' '+str(d))[0]:d for d in range(10)}
 Q=torch.zeros(10,9,dtype=torch.float64)
 for j in range(9):Q[:j+1,j]=1/((j+1)*(j+2))**.5;Q[j+1,j]=-(j+1)/((j+1)*(j+2))**.5
 decoders=[];decode_reports=[]
 for pos in (0,1,2,4,5,6):
  xx=torch.cat((norm[:,pos],torch.ones(len(norm),1,dtype=torch.float64)),1);yy=torch.cat((testnorm[:,pos],torch.ones(len(testnorm),1,dtype=torch.float64)),1)
  labels=torch.tensor([dmap[int(i)] for i in fa['inputs'][:,pos]]);testlabels=torch.tensor([dmap[int(i)] for i in fbdata['inputs'][:,pos]])
  means=torch.stack([xx[labels==d].mean(0) for d in range(10)]);res=xx-means[labels];within=res.T@res/len(res);decoder,ridge,error=fit(means,within,Q);pred=yy@decoder@Q.T;correct=pred.gather(1,testlabels[:,None]).squeeze(1);other=pred.masked_fill(torch.nn.functional.one_hot(testlabels,10).bool(),-torch.inf).max(1).values
  report={'position':pos,'validation_correct':int((pred.argmax(1)==testlabels).sum()),'target_min':float(correct.min()),'other_max':float(other.max()),'margin_min':float((correct-other).min()),'text_rms_max':float((tx@decoder).square().mean(0).sqrt().max()),'ridge':ridge,'interpolation_error':error};print(json.dumps(report),flush=True);decoders.append(decoder);decode_reports.append(report)
 xx=torch.cat((norm[:,:7],torch.ones(len(norm),7,1,dtype=torch.float64)),2);yy=torch.cat((testnorm[:,:7],torch.ones(len(testnorm),7,1,dtype=torch.float64)),2)
 means=[];targets=[];within=torch.zeros(769,769,dtype=torch.float64);count=0
 for pos in range(7):
  for token in fa['inputs'][:,pos].unique():
   group=xx[fa['inputs'][:,pos]==token,pos];mean=group.mean(0);res=group-mean;means.append(mean);targets.append(torch.eye(7,dtype=torch.float64)[pos]);within+=res.T@res;count+=len(group)
 position,ridge,error=fit(torch.stack(means),within/count,torch.stack(targets),-torch.ones(5,7,dtype=torch.float64))
 posreports=[]
 for pos in (0,1,2,4,5,6):
  pp=yy@position[:,pos];margin=pp[:,pos]-pp.masked_fill(torch.arange(7)[None,:]==pos,-torch.inf).max(1).values;pr={'position':pos,'validation_selected':int((pp.argmax(1)==pos).sum()),'margin_min':float(margin.min())};posreports.append(pr);print(json.dumps(pr),flush=True)
 energies=[]
 for h in range(12):
  sl=slice(h*64,(h+1)*64);energies.append(float(torch.trace((P[:,sl].T@P[:,sl])@cv[sl,sl])))
 heads=sorted(range(12),key=lambda h:energies[h])[:6];qkv=sd['transformer.h.2.attn.c_attn.weight'].double().clone();qb=sd['transformer.h.2.attn.c_attn.bias'].double().clone();pp=P.clone();rotations={}
 for idx,(head,pos) in enumerate(zip(heads,(0,1,2,4,5,6))):
  sl=slice(head*64,(head+1)*64);ki=slice(768+head*64,768+(head+1)*64);vi=slice(1536+head*64,1536+(head+1)*64)
  q=ln@qkv[sl].T+qb[sl];k=ln@qkv[ki].T+qb[ki];cq=q.T@q/len(q);ck=k.T@k/len(k);_,R=torch.linalg.eigh(cq/cq.trace()+ck/ck.trace());qkv[sl]=R.T@qkv[sl];qb[sl]=R.T@qb[sl];qkv[ki]=R.T@qkv[ki];qb[ki]=R.T@qb[ki]
  oldw=qkv[sl].clone();oldb=qb[sl].clone();response=oldw@(g[:,None]*allphase);offset=oldw@beta+oldb;correction=response@(allphase.T/g[None,:])+offset[:,None]*(common/g/(768**.5))[None,:];qkv[sl]-=correction;qb[sl]+=correction@beta
  ev,U=torch.linalg.eigh(cv[sl,sl]);ev=ev.clamp_min(float(ev.max())*1e-9);root=U*ev.sqrt()[None,:];inv=ev.rsqrt()[:,None]*U.T;_,sv,Vh=torch.linalg.svd(P[:,sl]@root,full_matrices=False);RV=Vh.T.flip(1);rotation=RV.T@inv;unrotate=root@RV;qkv[vi]=rotation@qkv[vi];qb[vi]=rotation@qb[vi];pp[:,sl]=P[:,sl]@unrotate
  qi=head*64;kk=768+qi;v0=1536+qi;qr=a.attention_score*8/(768**.5)*(common/g)*(a.carrier**2+a.stage_gain**2)**.5/a.carrier;qkv[qi]=qr;qb[qi]=-qr@beta;key=position[:,pos];qkv[kk]=key[:768];qb[kk]=key[768];decoder=decoders[idx]@Q.T@torch.arange(10,dtype=torch.float64);qkv[v0]=decoder[:768];qb[v0]=decoder[768]
  rotations[str(head)]={'qk':R,'value':rotation,'value_inverse':unrotate}
 pp-=memory@(memory.T@pp);pbb=pb-memory@(memory.T@pb)+memory@frame_mean
 for idx,head in enumerate(heads):pp[:,head*64]=a.value_gain*memory[:,(2,1,0,2,1,0)[idx]]
 changed.update({'transformer.h.2.attn.c_attn.weight':qkv.float(),'transformer.h.2.attn.c_attn.bias':qb.float(),'transformer.h.2.attn.c_proj.weight':pp.float(),'transformer.h.2.attn.c_proj.bias':pbb.float()})
 out.parent.mkdir(parents=True,exist_ok=True);torch.save({'phase':phase,'common':common,'stage_gain':a.stage_gain,'excluded_coordinates':excluded,'memory':memory,'simplex':Q,'heads':heads,'neurons1':neurons,'donors':donors,'carrier':a.carrier,'value_gain':a.value_gain,'attention_score':a.attention_score,'decoders':decoders,'position_decoder':position,'rotations':rotations,'memory_report':memreport,'gate_parameters':gate_report},out.parent/'circuit.pt')
 sources=[ROOT/s for s in base_manifest['sources']]+[Path(__file__),ROOT/'experiments/derive_k_stage_gates_v1.py',ROOT/'experiments/combine_k_stage_gates_v3.py']
 save(out,sd,changed,sources=sources,inputs=[base,ROOT/'host/ckpt.pt',ROOT/'artifacts/k1_marker_format_a.pt',ROOT/'artifacts/k1_marker_format_b.pt',ROOT/'artifacts/k1_marker_calib.pt',ROOT/'artifacts/k_stage_gates_v3.pt'],arguments=vars(a),description='K2 G3: whole-format carrier, five stage codes, token-stratified position keys, six separate current-digit reads and three column sum fields; no arithmetic-answer fitting.',extra={'heads':heads,'memory':memreport,'digits':decode_reports,'positions':posreports,'gate_parameters':gate_report})
if __name__=='__main__':main()
