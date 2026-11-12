"""Quiet the native value response to the large marked source-state subspace.

Five PCA ranks [1,2,4,8,16]. A covariance-metric projection annihilates those
format directions in retained native value rows, preserving all five explicit
register-copy rows exactly. No operand/answer labels or iterative fitting.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,atomic_json
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
h=host_state();base=ROOT/'experiments/a9v2_copy/delta.pt';ch=expand(h,base)
circuit=torch.load(base.parent/'circuit.pt',weights_only=True);old=circuit['piece']
fa,fb,tc=[torch.load(ROOT/f'artifacts/a8v2_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
X=tc['captures']['7.input'].double().reshape(-1,640)
P=torch.cat([fa['captures']['7.input'][:,8],fb['captures']['7.input'][:,8]]).double()
_,sing,vt=torch.linalg.svd(P,full_matrices=False)
C=X[:192*64].T@X[:192*64]/(192*64);C.diagonal().add_(1e-6*C.diagonal().mean())
prefix='model.layers.7.self_attn.';V=ch[prefix+'v_proj.weight'].double();O=ch[prefix+'o_proj.weight'].double().reshape(640,4,256)
selected=old['value_channels'];retained=torch.tensor([i for i in range(256) if i not in selected])
native=V.clone();native[selected]=0
On=O.clone();On[:,:,selected]=0
gpost=1+ch['model.layers.7.post_attention_layernorm.weight'].double()
trials=[];choices=[]
for rank in (1,2,4,8,16):
    S=vt[:rank].T;dual=torch.linalg.solve(C,S);dual=dual@torch.linalg.inv(S.T@dual)
    delta=-(native@S)@dual.T;newV=V+delta
    assert torch.equal(newV[selected],V[selected])
    source=P@(native+delta).T;word=X@delta.T
    source_heads=torch.stack([source@On[:,j].T for j in range(4)],1)
    word_heads=torch.stack([word@On[:,j].T for j in range(4)],1)
    source_bound=source_heads.square().mean(-1).sqrt().sum(1)
    r={'rank':rank,'source_input_energy_retained':float(sing[rank:].square().sum()/sing.square().sum()),
       'maximum_native_source_output_bound':float(source_bound.max()),'rms_native_source_output_bound':float(source_bound.square().mean().sqrt()),
       'word_raw_output_rms':float(word_heads.square().mean().sqrt()),
       'word_gain_weighted_output_rms':float((word_heads*gpost).square().mean().sqrt()),
       'maximum_delta_weight':float(delta.abs().max()),'native_source_constraint_max':float(((native+delta)@S).abs().max())}
    trials.append(r);choices.append(newV);print(json.dumps(r),flush=True)
viable=[i for i,r in enumerate(trials) if r['maximum_native_source_output_bound']<1.]
best=min(viable,key=lambda i:trials[i]['word_gain_weighted_output_rms']) if viable else min(range(5),key=lambda i:trials[i]['maximum_native_source_output_bound'])
ch[prefix+'v_proj.weight']=choices[best].float()
piece=dict(old);piece.update(native_source_projection_trials=trials,selected_native_source_projection=trials[best],
    native_source_projection_ridge=1e-6,native_source_projection_method='minimum-text-covariance native-value annihilation of marked-source PCA directions; copy rows unchanged')
circuit['piece']=piece
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',circuit)
bm=json.loads(base.with_suffix('.manifest.json').read_text())
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
    inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt']+
      [ROOT/f'artifacts/a8v2_{s}.pt' for s in ('syntax_a','syntax_stress','calib256')],
    arguments=vars(a),description=__doc__,extra=piece,categories=bm['categories'])
atomic_json(out.parent/'projection.json',{'trials':trials,'selected':trials[best]})
