"""Native-value source projection using measured attention-weighted inputs.

Recover the recorded attention mixtures by a linear inverse of recorded
source values and attended values. This is activation analysis, not model or
attention replay. Calibrate a direct source-null projection using those
measured mixtures and the recorded output normalization. No answers or
gradients; five independent source-PCA ranks.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
h=host_state();base=ROOT/'experiments/a17v1_query_projected/delta.pt';ch=expand(h,base)
unprojpath=ROOT/'experiments/a9v2_copy/delta.pt';unproj=expand(h,unprojpath)
meta=torch.load(ROOT/'experiments/a9v2_copy/circuit.pt',weights_only=True);selected=meta['value_channels']
sourcepath=ROOT/'artifacts/a19_attention_inputs.pt';cap=torch.load(sourcepath,weights_only=True,mmap=True)
X=cap['captures']['7.input'].double();Vrecord=cap['captures']['7.values'].double()
heads=cap['captures']['7.attn_in'].double().reshape(*X.shape[:2],4,256).permute(0,2,1,3)
gram=Vrecord@Vrecord.transpose(1,2);ev,U=torch.linalg.eigh(gram)
threshold=ev[:,-1:]*1e-12;inverse=torch.where(ev>threshold,1/ev.clamp_min(1e-30),torch.zeros_like(ev))
pinv=(U*inverse[:,None])@U.transpose(1,2)
coeff=(heads@Vrecord[:,None].transpose(-1,-2))@pinv[:,None]
recon=coeff@Vrecord[:,None];future=torch.triu(torch.ones(X.shape[1],X.shape[1],dtype=torch.bool),diagonal=1)
recovery={'method':'unique linear mixture recovered from recorded source/attended values; no softmax or attention implementation',
 'relative_attended_value_reconstruction_rms':float((recon-heads).square().mean().sqrt()/heads.square().mean().sqrt()),
 'maximum_future_coefficient':float(coeff[:,:,future].abs().max()),'rms_future_coefficient':float(coeff[:,:,future].square().mean().sqrt()),
 'minimum_coefficient':float(coeff.min()),'maximum_row_sum_error':float((coeff.sum(-1)-1).abs().max()),
 'minimum_source_value_rank':int((ev>threshold).sum(1).min()),'gram_pseudoinverse_relative_cutoff':1e-12}
print(json.dumps({'attention_recovery':recovery}),flush=True)
assert recovery['relative_attended_value_reconstruction_rms']<1e-5
assert recovery['minimum_source_value_rank']==X.shape[1]
assert recovery['maximum_future_coefficient']<.01 and recovery['minimum_coefficient']>-.01
Z=(coeff@X[:,None]).permute(0,2,1,3).reshape(-1,4,640)
prefix='model.layers.7.self_attn.';V0=unproj[prefix+'v_proj.weight'].double();Vold=ch[prefix+'v_proj.weight'].double()
O=ch[prefix+'o_proj.weight'].double().reshape(640,4,256);gamma=1+ch['model.layers.7.post_attention_layernorm.weight'].double()
def output_raw(V):return torch.einsum('nhv,dhv->nd',Z@V.T,O)
def branch(raw):return raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
reconstructed_raw=output_raw(Vold);recorded_raw=cap['captures']['7.attn'].double().reshape(-1,640)
recovery['relative_output_raw_reconstruction_rms']=float((reconstructed_raw-recorded_raw).square().mean().sqrt()/recorded_raw.square().mean().sqrt())
assert recovery['relative_output_raw_reconstruction_rms']<1e-5
reference_raw=output_raw(V0);reference=branch(reference_raw);denom=reference_raw.square().mean(1)+1e-6
nt=384*64;baseline_diff=branch(reconstructed_raw)-reference
baseline={'validation_branch_rms':float(baseline_diff[nt:].square().mean().sqrt()),
 'validation_p99_token_rms':float(torch.quantile(baseline_diff[nt:].square().mean(1),.99).sqrt())}
native=V0.clone();native[selected]=0;Onative=O.clone();Onative[:,:,selected]=0
importance=(Onative*gamma[:,None,None]).square().sum((0,2));importance/=importance.sum()
C=torch.zeros(640,640,dtype=torch.float64)
for head in range(4):
 zz=Z[:nt,head]/denom[:nt,None].sqrt();C+=importance[head]*(zz.T@zz/nt)
C.diagonal().add_(1e-6*C.diagonal().mean())
format_paths=[ROOT/f'artifacts/a8v2_{s}.pt' for s in ('syntax_a','syntax_stress')]
formats=[torch.load(path,weights_only=True,mmap=True) for path in format_paths]
P=torch.cat([f['captures']['7.input'][:,8] for f in formats]).double();_,sing,vt=torch.linalg.svd(P,full_matrices=False)
trials=[];choices=[]
for rank in (1,2,4,8,16):
 S=vt[:rank].T;dual=torch.linalg.solve(C,S);dual=dual@torch.linalg.inv(S.T@dual)
 delta=-(native@S)@dual.T;Vnew=(V0+delta).float();assert torch.equal(Vnew[selected],V0[selected].float())
 source=P@Vnew.double().T;source[:,selected]=0
 source_heads=torch.stack([source@Onative[:,j].T for j in range(4)],1)
 bound=source_heads.square().mean(-1).sqrt().sum(1)
 fresh=output_raw(Vnew.double());difference=branch(fresh)-reference;err=difference[nt:].square().mean(1)
 rec={'rank':rank,'source_input_energy_retained':float(sing[rank:].square().sum()/sing.square().sum()),
 'maximum_native_source_output_bound':float(bound.max()),'rms_native_source_output_bound':float(bound.square().mean().sqrt()),
 'validation_branch_rms':float(err.mean().sqrt()),'validation_p99_token_rms':float(torch.quantile(err,.99).sqrt()),
 'maximum_value_weight':float(Vnew.abs().max())}
 trials.append(rec);choices.append(Vnew);print(json.dumps(rec),flush=True)
valid=[i for i,r in enumerate(trials) if r['maximum_native_source_output_bound']<1 and r['maximum_value_weight']<=1e6];assert valid
best=min(valid,key=lambda i:trials[i]['validation_branch_rms']);ch[prefix+'v_proj.weight']=choices[best]
report={'attention_recovery':recovery,'rank_trials':[1,2,4,8,16],'covariance_ridge':1e-6,
 'head_metric_weights':importance.tolist(),'baseline':baseline,'trials':trials,'chosen':trials[best]}
recovery_path=ROOT/'artifacts/a19_attention_recovery.pt'
atomic_torch(recovery_path,{'attended_inputs':Z.float(),'coefficients':coeff.float(),'report':recovery})
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);bm=json.loads(base.with_suffix('.manifest.json').read_text())
circuit=torch.load(base.parent/'circuit.pt',weights_only=True);circuit['attended_source_projection']=report
atomic_torch(out.parent/'circuit.pt',circuit)
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',unprojpath,ROOT/'experiments/a9v2_copy/circuit.pt',sourcepath,recovery_path]+format_paths,
 arguments=vars(a),description=__doc__,extra=report,categories=bm['categories'])
atomic_json(out.parent/'attended_projection.json',report);print(json.dumps({'baseline':baseline,'chosen':trials[best]}),flush=True)
