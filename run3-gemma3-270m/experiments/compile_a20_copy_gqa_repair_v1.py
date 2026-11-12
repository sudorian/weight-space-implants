"""One-shot constrained native GQA reconstruction for the answer-copy block.

Fit the host block's unlabeled native attention output from measured edited
attention mixtures. Enforce native source-null constraints, then compute a
weighted rank-251 shared-value factorization by an eigendecomposition. The
five explicit program value/output columns stay unchanged. Five independent
ridge solutions; no gradients, alternating fits, or answer targets.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
h=host_state();base=ROOT/'experiments/a17v1_query_projected/delta.pt';ch=expand(h,base)
meta_path=ROOT/'experiments/a9v2_copy/circuit.pt';meta=torch.load(meta_path,weights_only=True)
selected=meta['value_channels'];retained=torch.tensor([j for j in range(256) if j not in selected.tolist()]);rank_native=len(retained)
rp=ROOT/'artifacts/a19_attention_recovery.pt';cp=ROOT/'artifacts/a19_attention_inputs.pt';hp=ROOT/'artifacts/a20_native_attention.pt'
recovery=torch.load(rp,weights_only=True,mmap=True);cap=torch.load(cp,weights_only=True,mmap=True);native_cap=torch.load(hp,weights_only=True,mmap=True)
assert torch.equal(cap['inputs'],native_cap['inputs'])
Z=recovery['attended_inputs'].double();N=len(Z);assert N==512*64;nt=384*64
prefix='model.layers.7.self_attn.';Vold=ch[prefix+'v_proj.weight'].double();Oold=ch[prefix+'o_proj.weight'].double().reshape(640,4,256)
normname='model.layers.7.post_attention_layernorm.weight';ghost=1+h[normname].double();g0=ghost.clone();g0[meta['field_coordinates']]=meta['piece']['field_gamma']
reference_raw=native_cap['captures']['7.attn'].double().reshape(-1,640)
reference=(native_cap['captures']['7.raw']-native_cap['captures']['6.out']).double().reshape(-1,640)
scale=(reference_raw.square().mean(1,keepdim=True)+1e-6).sqrt()
prog_values=Z@Vold[selected].T
program_raw=torch.einsum('nhv,dhv->nd',prog_values,Oold[:,:,selected])
gain_ratio=torch.ones(640,dtype=torch.float64)
gain_ratio[meta['field_coordinates']]=ghost[meta['field_coordinates']]/g0[meta['field_coordinates']]
target_raw=reference_raw*gain_ratio[None]-program_raw
assert torch.isfinite(target_raw).all()
F=Z.reshape(N,-1)/scale;target=target_raw/scale
std=F[:nt].square().mean(0).sqrt().clamp_min(1e-12)
FT=F[:nt]/std;C=FT.T@FT/nt;cross=FT.T@target[:nt]/nt
ev,U=torch.linalg.eigh(C);ev=ev.clamp_min(0);projected=U.T@cross
format_paths=[ROOT/f'artifacts/a8v2_{s}.pt' for s in ('syntax_a','syntax_stress')]
formats=[torch.load(path,weights_only=True,mmap=True) for path in format_paths]
P=torch.cat([o['captures']['7.input'][:,8] for o in formats]).double();_,sing,vt=torch.linalg.svd(P,full_matrices=False)
source_rank=8;S0=vt[:source_rank].T;S=torch.zeros(2560,4*source_rank,dtype=torch.float64)
for head in range(4):S[head*640:(head+1)*640,head*source_rank:(head+1)*source_rank]=S0
A=S.T/std[None];UA=U.T@A.T
# A common input metric supplies the shared value basis in one factorization.
input_cov=sum(F[:nt,head*640:(head+1)*640].T@F[:nt,head*640:(head+1)*640]/nt for head in range(4))/4
input_cov.diagonal().add_(1e-6*input_cov.diagonal().mean())
ce,CU=torch.linalg.eigh(input_cov);Cr=(CU*ce.sqrt())@CU.T;Ci=(CU*ce.rsqrt())@CU.T
output_metric=g0.square()+.001*g0.square().mean()
baseline_raw=cap['captures']['7.attn'].double().reshape(-1,640);gold=1+ch[normname].double()
normalize=lambda r:r/(r.square().mean(1,keepdim=True)+1e-6).sqrt()
baseline_diff=normalize(baseline_raw)*gold-reference
baseline={'validation_branch_rms':float(baseline_diff[nt:].square().mean().sqrt()),
 'validation_p99_token_rms':float(torch.quantile(baseline_diff[nt:].square().mean(1),.99).sqrt())}
trials=[];choices=[]
ordinary=torch.ones(640,dtype=torch.bool);ordinary[meta['field_coordinates']]=False
for ridge in (.001,.0001,1e-5,1e-6,1e-7):
 inv=(ev+ridge).reciprocal();coef=U@(inv[:,None]*projected)
 RA=U@(inv[:,None]*UA);coef-=RA@torch.linalg.solve(A@RA,A@coef)
 B=(coef/std[:,None]).reshape(4,640,640).permute(0,2,1).contiguous()
 importance=sum(b.T@(output_metric[:,None]*b) for b in B)
 ee,R=torch.linalg.eigh(Cr@importance@Cr);basis=R[:,-rank_native:]
 Vnative=basis.T@Ci
 Onative=torch.stack([b@Cr@basis for b in B],1)
 Vnew=Vold.clone();Onew=Oold.clone();Vnew[retained]=Vnative;Onew[:,:,retained]=Onative
 Vnew=Vnew.float();Onew=Onew.float()
 assert torch.equal(Vnew[selected],Vold[selected].float());assert torch.equal(Onew[:,:,selected],Oold[:,:,selected].float())
 fresh=torch.einsum('nhv,dhv->nd',Z@Vnew.double().T,Onew.double())
 normed=normalize(fresh);gnew=g0.clone()
 fitted=(normed[:nt]*reference[:nt]).sum(0)/normed[:nt].square().sum(0).clamp_min(1e-30)
 gnew[ordinary]=fitted[ordinary];gweight=(gnew-1).float();gnew=1+gweight.double()
 diff=normed*gnew-reference;errors=diff[nt:].square().mean(1)
 sv=P@Vnew[retained].double().T
 source_heads=torch.stack([sv@Onew[:,j,retained].double().T for j in range(4)],1)
 bound=source_heads.square().mean(-1).sqrt().sum(1)
 rec={'ridge':ridge,'validation_branch_rms':float(errors.mean().sqrt()),
  'validation_p99_token_rms':float(torch.quantile(errors,.99).sqrt()),
  'maximum_native_source_output_bound':float(bound.max()),'rms_native_source_output_bound':float(bound.square().mean().sqrt()),
  'native_factorization_retained_energy':float(ee[-rank_native:].clamp_min(0).sum()/ee.clamp_min(0).sum()),
  'maximum_full_map_source_constraint':float((B@S0).abs().max()),
  'maximum_value_weight':float(Vnew.abs().max()),'maximum_output_weight':float(Onew.abs().max()),
  'maximum_norm_weight':float(gweight.abs().max()),'finite':bool(torch.isfinite(Vnew).all() and torch.isfinite(Onew).all() and torch.isfinite(gweight).all())}
 trials.append(rec);choices.append((Vnew,Onew,gweight));print(json.dumps(rec),flush=True)
viable=[i for i,r in enumerate(trials) if r['finite'] and max(r['maximum_value_weight'],r['maximum_output_weight'],r['maximum_norm_weight'])<=1e6 and r['maximum_native_source_output_bound']<.1]
assert viable,'No factored native repair meets the existing source suppression requirement'
best=min(viable,key=lambda i:trials[i]['validation_branch_rms']);Vnew,Onew,gnew=choices[best]
ch[prefix+'v_proj.weight']=Vnew;ch[prefix+'o_proj.weight']=Onew.reshape(640,1024);ch[normname]=gnew
report={'calibration_tokens':N,'training_tokens':nt,'validation_tokens':N-nt,'native_shared_rank':rank_native,
 'source_null_rank':source_rank,'source_null_rank_inherited_from_existing_copy_circuit':True,
 'input_covariance_ridge':1e-6,'output_raw_metric_mass':.001,'baseline':baseline,'trials':trials,'chosen':trials[best],
 'method':'one constrained linear solve per ridge, then one weighted shared-rank factorization and direct gain accommodation; no alternating or iterative fitting'}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);bm=json.loads(base.with_suffix('.manifest.json').read_text())
circuit=torch.load(base.parent/'circuit.pt',weights_only=True);circuit['copy_gqa_repair']=report
atomic_torch(out.parent/'circuit.pt',circuit)
categories=dict(bm['categories'])
for name in (prefix+'v_proj.weight',prefix+'o_proj.weight',normname):
 categories[name]={'shared_circuit_and_closed_form_text_repair':int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum())}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',meta_path,rp,cp,hp]+format_paths,
 arguments=vars(a),description=__doc__,extra=report,categories=categories)
atomic_json(out.parent/'gqa_repair.json',report);print(json.dumps({'baseline':baseline,'chosen':trials[best]}),flush=True)
