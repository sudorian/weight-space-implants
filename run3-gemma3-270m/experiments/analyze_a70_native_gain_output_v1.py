"""Literal output geometry using native post-MLP gains outside scratch rows.

This is a static output-component test on recorded activations. Eleven raw
class directions come only from protected unembedding/norm weights. The
current scalar digit/newline register labels, rather than operands or sums,
check nuisance tolerance. Five raw-energy regularizers are independent direct
solves. No checkpoint, GELU, attention, gradients or full-model replay.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_torch,atomic_json,sha
h=host_state();base=ROOT/'experiments/a65v2_final_energy_repair/delta.pt';ch=expand(h,base)
c=torch.load(base.parent/'circuit.pt',weights_only=True)
fp=ROOT/'artifacts/a61v1_syntax_stress.pt';f=torch.load(fp,weights_only=True,mmap=True)
mp=ROOT/'artifacts/readout_v1.pt';meta=torch.load(mp,weights_only=True)
ids=meta['token_ids'];V=h['model.embed_tokens.weight'];final_gain=1+h['model.norm.weight'].double()
metric=meta['vocabulary_metric'].double();gamma=1+h['model.layers.17.post_feedforward_layernorm.weight'].double()
scratch=set()
for l in range(9):
 name=f'model.layers.{l}.post_feedforward_layernorm.weight'
 if name in ch:scratch.update((ch[name].abs()>1000).nonzero().flatten().tolist())
scratch=torch.tensor(sorted(scratch));free=torch.tensor([i for i in range(640) if i not in set(scratch.tolist())])
newgamma=gamma.clone();newgamma[scratch]=1e6
u=c['units'].long();F=f['captures']['17.features'][:,8:13].double().reshape(-1,2048)[:,u[:20]]
raw=f['captures']['17.raw'][:,8:13].double().reshape(-1,640)
B=torch.zeros(20,11,dtype=torch.float64);B[0,0]=1;B[1,10]=1
ratio=c['piece']['raw_step_gain']/c['piece']['output_gain']
for j in range(9):B[2+2*j,j+1]=ratio;B[2+2*j,j]=-ratio;B[3+2*j]=-B[2+2*j]
common=F[:,0]+F[:,1];assert common.min()>0
coeff=F@B/common[:,None];assert (coeff.sum(1)-1).abs().max()<1e-8
flag=raw[:,c['piece']['source_flag_coordinate']]
digit=raw[:,c['piece']['source_digit_coordinate']]/c['piece']['source_digit_scale']
nl=raw[:,c['piece']['source_newline_coordinate']]/c['piece']['source_newline_scale']
label=torch.where(nl>.5,torch.full_like(nl,10,dtype=torch.long),digit.round().long().clamp(0,9))
expected=torch.tensor(ids)[label]
cancel=torch.zeros_like(raw)
cancel[:,scratch]=-raw[:,scratch]*c['piece']['source_flag_scale']/flag[:,None]/(640**.5*newgamma[scratch])
D=V[ids][:,free].double()*final_gain[free]*gamma[free]
C=metric[free][:,free]*gamma[free,None]*gamma[None,free]
trials=[];candidates={}
for ridge in (0.,.0001,.001,.01,.1):
 R=C.clone();R.diagonal().add_(ridge*C.diagonal().mean())
 Z=torch.linalg.solve(R,D.T);directions=Z@torch.linalg.inv(D@Z);directions/=directions.norm(dim=0)
 full=torch.zeros(640,11,dtype=torch.float64);full[free]=directions
 ideal=640**.5*gamma[:,None]*full
 program=coeff@full.T+cancel
 # Norms only evaluate the explicit hand output vector on recorded inputs.
 post=program/(program.square().mean(1,keepdim=True)+1e-30).sqrt()*newgamma
 output=raw+post
 desired=(output*final_gain*V[expected].double()).sum(1)
 other=torch.full_like(desired,-torch.inf)
 desired_ideal=(ideal.T*final_gain*V[ids].double()).sum(1)
 other_ideal=torch.full((11,),-torch.inf,dtype=torch.float64)
 for start in range(0,len(V),4096):
  E=V[start:start+4096].double()*final_gain
  scores=output@E.T
  local=expected-start;valid=(local>=0)&(local<len(E));rr=valid.nonzero().flatten();scores[rr,local[rr]]=-torch.inf
  other=torch.maximum(other,scores.amax(1))
  scores_i=ideal.T@E.T
  for j,t in enumerate(ids):
   if start<=t<start+len(E):scores_i[j,t-start]=-torch.inf
  other_ideal=torch.maximum(other_ideal,scores_i.amax(1))
 margin=desired-other
 rec={'raw_energy_regularizer':ridge,'minimum_literal_class_margin':float((desired_ideal-other_ideal).min()),
  'minimum_recorded_register_margin':float(margin.min()),'wrong_current_register_outputs':int((margin<=0).sum()),
  'recorded_register_outputs':len(margin),'format_output_rms':float(output.square().mean().sqrt()),
  'maximum_uncancelled_scratch':float(output[:,scratch].abs().max()),
  'maximum_nonscratch_input':float(raw[:,free].abs().max()),'class_raw_direction_maximum':float(directions.abs().max())}
 trials.append(rec);candidates[str(ridge)]={'raw_directions':full,'report':rec};print(json.dumps(rec),flush=True)
report={'raw_energy_regularizer_trials':[0.,.0001,.001,.01,.1], 'scratch_rows':scratch.tolist(),
 'scratch_gamma':1e6,'all_other_gammas_native':True,'scratch_count':len(scratch),
 'native_gamma_absolute_quantiles':torch.quantile(gamma.abs(),torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)).tolist(),
 'trials':trials,'limitations':'Native arithmetic MLP contribution omitted in this dominant-programme component test; every proposed installation still requires official forward and score.',
 'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (base,base.parent/'circuit.pt',fp,mp)},'source_sha':sha(__file__)}
atomic_torch(ROOT/'artifacts/a70_native_gain_output_v1.pt',{'candidates':candidates,'gamma':newgamma,'report':report})
atomic_json(ROOT/'artifacts/a70_native_gain_output_v1.json',report)
