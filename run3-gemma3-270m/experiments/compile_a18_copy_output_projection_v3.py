"""Quiet the copied source with a joint covariance projection across all heads.

Restore the unprojected native value rows, then directly project each head's
native output map using its own recorded attended-value covariance. Explicit
five register-copy columns remain unchanged. Five source PCA ranks, no answer
labels, gradients, or iterative weight fitting.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base',default='experiments/a17v1_query_projected/delta.pt');a=p.parse_args()
h=host_state();base=ROOT/a.base;ch=expand(h,base)
unprojected_path=ROOT/'experiments/a9v2_copy/delta.pt';unprojected=expand(h,unprojected_path)
meta=torch.load(ROOT/'experiments/a9v2_copy/circuit.pt',weights_only=True);selected=meta['value_channels']
retained=torch.tensor([j for j in range(256) if j not in selected.tolist()])
paths=[ROOT/f'artifacts/a8v2_{s}.pt' for s in ('calib256','syntax_a','syntax_stress')]
tc,fa,fb=[torch.load(path,weights_only=True,mmap=True) for path in paths]
prefix='model.layers.7.self_attn.';V=unprojected[prefix+'v_proj.weight'].double();oldV=ch[prefix+'v_proj.weight'].double()
O=ch[prefix+'o_proj.weight'].double().reshape(640,4,256);gamma=1+ch['model.layers.7.post_attention_layernorm.weight'].double()
assert torch.equal(V[selected],oldV[selected])
X=tc['captures']['7.input'].double().reshape(-1,640)
P=torch.cat([fa['captures']['7.input'][:,8],fb['captures']['7.input'][:,8]]).double()
Y=P@V[retained].T;_,sing,vt=torch.linalg.svd(Y,full_matrices=False)
Wv=h[prefix+'v_proj.weight'].double()
R=torch.linalg.solve(Wv@Wv.T,Wv@V[retained].T).T
basis_error=float((R@Wv-V[retained]).abs().max());assert basis_error<1e-6
heads=tc['captures']['7.attn_in'].double().reshape(-1,4,256)
word_head=heads@R.T;nt=192*64
joint=word_head.reshape(len(word_head),-1)
C=joint[:nt].T@joint[:nt]/nt;C.diagonal().add_(1e-6*C.diagonal().mean())
Onative=O[:,:,retained].reshape(640,-1)
source_word=X@V[retained].T
old_delta_word=X@(oldV-V).T
old_word_outputs=torch.stack([old_delta_word@O[:,j].T for j in range(4)],1)
baseline={'unattended_gain_weighted_rms':float((old_word_outputs*gamma).square().mean().sqrt()),
 'unattended_summed_gain_weighted_rms':float((old_word_outputs.sum(1)*gamma).square().mean().sqrt())}
trials=[];choices=[]
for rank in (1,2,4,8,16):
 S0=vt[:rank].T;S=torch.zeros(4*len(retained),4*rank,dtype=torch.float64)
 for head in range(4):S[head*len(retained):(head+1)*len(retained),head*rank:(head+1)*rank]=S0
 dual=torch.linalg.solve(C,S);dual=dual@torch.linalg.inv(S.T@dual)
 delta=-(Onative@S)@dual.T
 newO=O.clone();newO[:,:,retained]=(Onative+delta).reshape(640,4,len(retained))
 newO=newO.float().double();assert torch.equal(newO[:,:,selected],O[:,:,selected])
 source_outputs=torch.stack([Y@newO[:,j,retained].T for j in range(4)],1)
 source_bound=source_outputs.square().mean(-1).sqrt().sum(1)
 proxy_outputs=torch.stack([word_head[:,j]@(newO[:,j,retained]-O[:,j,retained]).T for j in range(4)],1)
 unattended=torch.stack([source_word@(newO[:,j,retained]-O[:,j,retained]).T for j in range(4)],1)
 rec={'rank':rank,'source_value_energy_retained':float(sing[rank:].square().sum()/sing.square().sum()),
  'maximum_native_source_output_bound':float(source_bound.max()),'rms_native_source_output_bound':float(source_bound.square().mean().sqrt()),
  'validation_attended_gain_weighted_rms':float((proxy_outputs[nt:]*gamma).square().mean().sqrt()),
  'validation_summed_attended_gain_weighted_rms':float((proxy_outputs[nt:].sum(1)*gamma).square().mean().sqrt()),
  'unattended_gain_weighted_rms':float((unattended*gamma).square().mean().sqrt()),
  'unattended_summed_gain_weighted_rms':float((unattended.sum(1)*gamma).square().mean().sqrt()),
  'maximum_output_weight':float(newO.abs().max())}
 trials.append(rec);choices.append(newO.float());print(json.dumps(rec),flush=True)
valid=[i for i,r in enumerate(trials) if r['maximum_native_source_output_bound']<1 and r['maximum_output_weight']<=1e6]
assert valid,'No head-specific source projection satisfies the existing source bound'
best=min(valid,key=lambda i:trials[i]['validation_summed_attended_gain_weighted_rms'])
ch[prefix+'v_proj.weight']=V.float();ch[prefix+'o_proj.weight']=choices[best].reshape(640,1024)
report={'projection_scope':'joint 1004-dimensional attended-native-value covariance; separate source constraints for every head',
 'rank_trials':[1,2,4,8,16],'covariance_ridge':1e-6,'native_basis_mapping_error':basis_error,
 'baseline':baseline,'trials':trials,'chosen':trials[best],
 'covariance_proxy':'Actual A8v2 native attended values, transformed to retained native value coordinates; actual meters decide preservation.'}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);bm=json.loads(base.with_suffix('.manifest.json').read_text())
circuit=torch.load(base.parent/'circuit.pt',weights_only=True);circuit['copy_output_projection']=report
atomic_torch(out.parent/'circuit.pt',circuit)
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',unprojected_path,ROOT/'experiments/a9v2_copy/circuit.pt']+paths,
 arguments=vars(a),description=__doc__,extra=report,categories=bm['categories'])
atomic_json(out.parent/'output_projection.json',report);print(json.dumps({'baseline':baseline,'chosen':trials[best]}),flush=True)
