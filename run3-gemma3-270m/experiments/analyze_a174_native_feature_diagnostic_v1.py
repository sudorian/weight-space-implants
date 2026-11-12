"""Certify the native-feature diagnostic's original text trajectory.

The original programme gates are linear functions of recorded inputs. A
conservative float32 dot-product allowance must keep every gate below-16.
At that bound tanh-GELU rounds to zero, so restoring native G/Up while setting
programme Down columns to zero leaves the original hidden trajectory intact
on these inputs, by induction over layers. No native GELU is evaluated.
"""
import json,sys,resource
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,sha
h=host_state();dp=ROOT/'experiments/a174v1_native_feature_diagnostic/delta.pt'
lp=ROOT/'artifacts/a174v1_native_feature_diagnostic_layout.json'
ap=ROOT/'artifacts/a174v1_native_feature_diagnostic.pt'
meta=json.loads(lp.read_text());base=ROOT/meta['base'];bc=expand(h,base);dc=expand(h,dp)
a=torch.load(ap,weights_only=True,mmap=True)
hp=ROOT/'artifacts/a140v1_host_branches.pt';host=torch.load(hp,weights_only=True,mmap=True)
tp=ROOT/'artifacts/a142v1_paired_host_top.pt';top=torch.load(tp,weights_only=True,mmap=True)
ep=ROOT/'artifacts/a150v1_early_native_basis.pt';old=torch.load(ep,weights_only=True,mmap=True)
older_base=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
older_state=expand(h,older_base)
assert old['checkpoint_sha256']==sha(older_base)
assert a['checkpoint_sha256']==sha(dp)and meta['base_sha256']==sha(base)
assert torch.equal(a['inputs'],host['inputs'])and torch.equal(a['inputs'],top['inputs'][256:])
assert json.loads((ROOT/'downloads/config.json').read_text())['hidden_activation']=='gelu_pytorch_tanh'
mutable=set()
for layer in meta['layers']:
 for suffix in ['mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight']:mutable.add(f'model.layers.{layer}.'+suffix)
for name in h:
 if name not in mutable:assert torch.equal(dc.get(name,h[name]),bc.get(name,h[name])),name

def norm(x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
state=h['model.embed_tokens.weight'][a['inputs']]*torch.tensor(640**.5,dtype=torch.float32)
hstates={}
for layer in range(18):
 for branch,part in [('attention','post_attention_layernorm'),('mlp','post_feedforward_layernorm')]:
  state=state+norm(host['captures'][f'{layer}.{branch}.raw'])*(1+h[f'model.layers.{layer}.{part}.weight'])
 if layer in meta['layers']:hstates[layer]=state.clone()
assert torch.equal(state,host['captures']['finalraw'])
allowance=650*torch.finfo(torch.float32).eps/(1-650*torch.finfo(torch.float32).eps)
reports={};certified=True
for layer in meta['layers']:
 pre=f'model.layers.{layer}.';u=torch.tensor(meta['native_restored_programme_units'][str(layer)],dtype=torch.long)
 gn,un,dn,pn,nn=[pre+s for s in ['mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','post_feedforward_layernorm.weight','pre_feedforward_layernorm.weight']]
 assert torch.equal(dc.get(gn,h[gn]),h[gn])and torch.equal(dc.get(un,h[un]),h[un])
 if len(u):assert dc[dn][:,u].abs().max()==0
 else:
  assert layer==9
  for name in [gn,un,dn,pn]:assert torch.equal(bc.get(name,h[name]),h[name])
 ordinary=torch.arange(2048)[~torch.isin(torch.arange(2048),u)]
 assert torch.equal(dc.get(dn,h[dn])[:,ordinary],bc.get(dn,h[dn])[:,ordinary])
 raw=a['captures'][f'{layer}.raw'];F=a['captures'][f'{layer}.native_features']
 xp=norm(raw)*(1+bc.get(nn,h[nn]))
 if len(u):
  gate=xp.double()@bc[gn][u].double().T
  error=allowance*(xp.double().abs()@bc[gn][u].double().abs().T)
  upper=float((gate+error).max());maximum_gate=float(gate.max());inactive=upper<-16
 else:upper=None;maximum_gate=None;inactive=True
 certified &= inactive
 currentF=F.clone();currentF[:,:,u]=0
 currentraw=currentF.double()@bc.get(dn,h[dn]).double().T
 originalraw=F.double()@h[dn].double().T
 currentpost=norm(currentraw)*(1+bc.get(pn,h[pn]).double())
 originalpost=norm(originalraw)*(1+h[pn].double())
 currentstate=raw.double()+currentpost
 ht=hstates[layer]
 if layer==9:ht=ht[torch.tensor(meta['layer9_window_indices'],dtype=torch.long)]
 ht=ht.double();scale=(ht.square().mean(-1,keepdim=True)+1e-6).sqrt()
 row={'programme_units':u.tolist(),'maximum_original_gate':maximum_gate,
  'maximum_gate_upper_bound':upper,'all_original_programme_features_certified_zero':inactive,
  'unmodified_native_observation':len(u)==0,
  'observed_word_windows':len(raw),
  'native_feature_shape':list(F.shape),
  'same_input_HOST_branch_rms_error':float((currentpost-originalpost).square().mean().sqrt()),
  'same_input_HOST_branch_relative_rms_error':float((currentpost-originalpost).square().mean().sqrt()/originalpost.square().mean().sqrt()),
  'paired_original_HOST_state_relative_rms':float(((currentstate-ht)/scale).square().mean().sqrt()),
  'paired_original_HOST_direction_rms':float((norm(currentstate)-norm(ht)).square().mean().sqrt())}
 if layer in [1,2]:
  upstream=[]
  for name in h:
   previous=any(name.startswith(f'model.layers.{i}.')for i in range(layer))
   present=name.startswith(pre)and('.self_attn.'in name or name.endswith('input_layernorm.weight')or name.endswith('post_attention_layernorm.weight'))
   if previous or present:upstream.append(name)
  identity=all(torch.equal(bc.get(name,h[name]),older_state.get(name,h[name]))for name in upstream)
  row['upstream_identity_to_older_capture_verified']=identity
  row['unchanged_early_input_bit_equal']=torch.equal(raw,old['captures'][f'word.{layer}.raw'])
  if identity:assert row['unchanged_early_input_bit_equal']
 if layer==17:
  final=a['captures']['finalraw'].double();row['final_state_reconstruction_relative_error']=float((currentstate-final).norm()/final.norm())
  assert row['final_state_reconstruction_relative_error']<1e-5
 reports[str(layer)]=row;print(json.dumps({'layer':layer,**row}),flush=True)
truth=top['topk_ids'][256:,:,0];predicted=a['topk_ids'][:,:,0]
# A diagnostic reference for eliminating final-MLP distortion. Its
# native features were observed by the meter; only the fixed linear Down,
# RMS and protected vocabulary readout are applied here, with no GELU replay.
assert layer==17
oracle_state=(raw.double()+originalpost)[192:].reshape(-1,640)
oracle_x=(norm(oracle_state)*(1+h['model.norm.weight'].double())).float()
embedding=h['model.embed_tokens.weight']
oracle_max=torch.full((len(oracle_x),),-torch.inf)
oracle_ids=torch.full((len(oracle_x),),-1,dtype=torch.long)
for start in range(0,len(embedding),4096):
 logits=oracle_x@embedding[start:start+4096].T
 values,indices=logits.max(1);better=values>oracle_max
 oracle_max[better]=values[better];oracle_ids[better]=start+indices[better]
oracle_truth=truth[192:].reshape(-1)
oracle_report={'validation_matches':int((oracle_ids==oracle_truth).sum()),
 'validation_tokens':oracle_truth.numel(),
 'parent_validation_matches':int((predicted[192:]==truth[192:]).sum()),
 'meaning':'Original HOST MLP17 native branch at exact parent calibration inputs, using recorded native features. Diagnostic only; removes the arithmetic emitter and is not an addition checkpoint.'}
del oracle_state,oracle_x,oracle_max,oracle_ids,logits
report={'diagnostic_checkpoint':str(dp.relative_to(ROOT)),'base_checkpoint':str(base.relative_to(ROOT)),
 'base_text_trajectory_certified':bool(certified),'proof':'Every original programme gate upper bound is below-16, so its tanh-GELU feature is exactly zero in float32; replaced native features multiply zero Down columns. All other operations match by induction.',
 'word_windows':256,'layers':reports,'base_calibration_top1':{'matches':int((predicted==truth).sum()),'tokens':truth.numel(),
 'validation_matches':int((predicted[192:]==truth[192:]).sum()),'validation_tokens':truth[192:].numel()},
 'HOST_native_final_branch_reference':oracle_report,
 'no_model_replay_gradients_answers_or_native_nonlinear_evaluation':True,
 'inputs':{str(p.relative_to(ROOT)):sha(p)for p in[dp,base,lp,ap,hp,tp,ep,older_base]},
 'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
 'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT/'artifacts/a174v1_native_feature_diagnostic_analysis.json',report)
assert certified,'Original arithmetic gates are not all certified inactive; do not use this as the base trajectory'
print(json.dumps({'certified':certified,'calibration':report['base_calibration_top1']}),flush=True)
