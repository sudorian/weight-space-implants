"""Invert recorded native attention output mixtures, without attention replay.

Every branch output is already captured by the official forward meter. The
source value matrix and original output projection define a linear system
for the recorded mixture coefficients. An older capture with actual attended
values independently checks recovery before derived data may be used.
"""
import json,sys,resource
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,atomic_torch,sha
h=host_state();base=ROOT/'experiments/a142v2_exploratory_projected/delta.pt';ch=expand(h,base)
wp=ROOT/'artifacts/a150v1_early_native_basis.pt'
ep=ROOT/'artifacts/a140v1_edited_branches.pt'
hp=ROOT/'artifacts/a140v1_host_branches.pt'
sp=ROOT/'artifacts/a150v1_derived_host_states.pt'
oldp=ROOT/'artifacts/a119v1_current_word.pt'
oldbase=ROOT/'experiments/a117v1_six_unit_carry/delta.pt'
w,e,hb,hs,old=[torch.load(p,weights_only=True,mmap=True)for p in(wp,ep,hp,sp,oldp)]
assert torch.equal(w['inputs'],e['inputs'])
assert torch.equal(w['inputs'][:256],hb['inputs']) and torch.equal(hb['inputs'],hs['inputs'])
assert w['checkpoint_sha256']==e['checkpoint_sha256']==sha(ROOT/'experiments/a123v1_prior_native_final/delta.pt')
def norm(x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
pre='model.layers.3.'
O=h[pre+'self_attn.o_proj.weight'].double().reshape(640,4,256)
assert torch.equal(ch.get(pre+'self_attn.o_proj.weight',h[pre+'self_attn.o_proj.weight']),h[pre+'self_attn.o_proj.weight'])
def recover(X,raw,V,label):
 values=(X@V.T).double()
 allcoef=[];allheads=[];ranks=[];ratios=[]
 squared_error=0.;squared_reference=0.;future_max=0.;minimum=1.;sum_error=0.
 for i in range(len(X)):
  matrix=torch.einsum('ohv,sv->ohs',O,values[i]).reshape(640,-1)
  result=torch.linalg.lstsq(matrix,raw[i].double().T,rcond=1e-10,driver='gelsd')
  coef=result.solution.T.reshape(X.shape[1],4,X.shape[1]).permute(1,0,2)
  heads=coef@values[i]
  reproduced=torch.einsum('htv,ohv->to',heads,O)
  squared_error+=float((reproduced-raw[i]).square().sum());squared_reference+=float(raw[i].double().square().sum())
  ranks.append(int(result.rank));sing=result.singular_values
  ratios.append(float(sing[-1]/sing[0]))
  future=torch.ones(X.shape[1],X.shape[1],dtype=torch.bool).triu(1)
  future_max=max(future_max,float(coef[:,future].abs().max()))
  minimum=min(minimum,float(coef.min()));sum_error=max(sum_error,float((coef.sum(-1)-1).abs().max()))
  allcoef.append(coef.float());allheads.append(heads.float())
  if (i+1)%64==0:print(json.dumps({'recovery':label,'windows':i+1,'min_rank':min(ranks),'max_future':future_max}),flush=True)
 report={'windows':len(X),'tokens_per_window':X.shape[1],'relative_raw_reconstruction_error':(squared_error/squared_reference)**.5,
  'minimum_matrix_rank':min(ranks),'full_column_count':4*X.shape[1], 'minimum_singular_ratio':min(ratios),
  'maximum_future_coefficient':future_max,'minimum_coefficient':minimum,'maximum_coefficient_sum_error':sum_error}
 return torch.stack(allcoef),torch.stack(allheads),report
# Independent check where actual attended values were captured.
oldstate=expand(h,oldbase);oldV=oldstate[pre+'self_attn.v_proj.weight'];del oldstate
checkX=old['captures']['3.input'][:8]
checkheads=old['captures']['3.heads'][:8].reshape(8,64,4,256).permute(0,2,1,3)
checkraw=torch.einsum('nhtv,ohv->nto',checkheads.double(),O).float()
_,recovered,check=recover(checkX,checkraw,oldV,'known_heads')
check['relative_known_head_error']=float((recovered.double()-checkheads).norm()/checkheads.double().norm())
print(json.dumps({'independent_check':check}),flush=True)
assert check['relative_known_head_error']<1e-4
inputs={};outputs={};states={};reports={}
for tag in ['word','format']:
 state=w['captures'][f'{tag}.2.raw']+norm(e['captures'][f'{tag}.2.mlp.raw'])*(1+ch['model.layers.2.post_feedforward_layernorm.weight'])
 reconstructed=state+norm(e['captures'][f'{tag}.3.attention.raw'])*(1+h['model.layers.3.post_attention_layernorm.weight'])
 assert torch.equal(reconstructed,w['captures'][f'{tag}.3.raw'])
 inputs[tag]=norm(state)*(1+h[pre+'input_layernorm.weight'])
 outputs[tag]=e['captures'][f'{tag}.3.attention.raw']
 states[tag]=state
inputs['host']=norm(hs['states']['2.out'])*(1+h[pre+'input_layernorm.weight'])
outputs['host']=hb['captures']['3.attention.raw']
results={}
for tag in ['word','host','format']:
 V=h[pre+'self_attn.v_proj.weight']if tag=='host'else ch[pre+'self_attn.v_proj.weight']
 coef,heads,diag=recover(inputs[tag],outputs[tag],V,tag)
 reports[tag]=diag
 print(json.dumps({tag:diag}),flush=True)
 assert diag['minimum_matrix_rank']==diag['full_column_count'],(tag,diag)
 assert diag['relative_raw_reconstruction_error']<1e-5
 assert diag['maximum_future_coefficient']<.002
 results[tag]={'input':inputs[tag],'coefficients':coef,'heads':heads}
report={'analysis_only':True,'independent_check':check,'reconstructions':reports,
 'method':'Unique linear inverse of recorded output projection restricted to each recorded source-value span; no softmax or native activation evaluation',
 'inputs':{str(p.relative_to(ROOT)):sha(p)for p in[base,wp,ep,hp,sp,oldp,oldbase]},
 'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
 'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a171v1_recorded_attention_inverse.pt'
atomic_json(out.with_suffix('.json'),report)
# Keep only source mixtures, not redundant head arrays, to limit disk use.
for tag in results:
 results[tag]['mixtures']=results[tag]['coefficients'].double()@results[tag]['input'].double()[:,None]
 results[tag]['mixtures']=results[tag]['mixtures'].float()
 del results[tag]['heads'],results[tag]['coefficients']
# A single layer's derived source mixtures require ~352MB, so only write
# compact data if sufficient room exists; otherwise the report still stands.
size=sum(p.stat().st_size for p in ROOT.rglob('*')if p.is_file())
needed=sum(v.numel()*v.element_size()for z in results.values()for v in z.values()if isinstance(v,torch.Tensor))
report['planned_tensor_bytes']=needed;report['project_size_before_write']=size
if size+needed+10_000_000<40_000_000_000:
 atomic_torch(out,{'kind':'derived recorded linear mixtures, not new meter observations','inputs':w['inputs'],'records':results,'report':report})
 report['derived_data_saved']=True
else:report['derived_data_saved']=False
atomic_json(out.with_suffix('.json'),report)
print(json.dumps(report),flush=True)
