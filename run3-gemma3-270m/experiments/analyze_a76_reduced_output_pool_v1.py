"""Current-register nuisance check for previously derived sparse output pools.

Literal directions are already fixed from protected unembedding weights and
unlabeled native energy. This reads existing programme features, applies the
literal 11-class basis and scratch cancellation, and checks the static output
component. No operands or arithmetic answers are used, and no model is run.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();base=ROOT/'experiments/a72v2_units_gate_hardened/circuit.pt';c=torch.load(base,weights_only=True)
fp=ROOT/'artifacts/a72v2_syntax_stress.pt';f=torch.load(fp,weights_only=True,mmap=True)
gp=ROOT/'artifacts/a63_sparse_output_geometry_v1.pt';geos=torch.load(gp,weights_only=True)
ids=geos['token_ids'];V=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
u=c['units'].long();F=f['captures']['17.features'][:,8:13].double().reshape(-1,2048)[:,u[:20]]
raw=f['captures']['17.raw'][:,8:13].double().reshape(-1,640)
B=torch.zeros(20,11,dtype=torch.float64);B[0,0]=1;B[1,10]=1
step=c['piece']['raw_step_gain']/c['piece']['output_gain']
for j in range(9):B[2+2*j,j+1]=step;B[2+2*j,j]=-step;B[3+2*j]=-B[2+2*j]
common=F[:,0]+F[:,1];assert common.min()>0
coeff=F@B/common[:,None]
flag=raw[:,c['piece']['source_flag_coordinate']];factor=c['piece']['source_flag_scale']/flag
digit=raw[:,c['piece']['source_digit_coordinate']]/c['piece']['source_digit_scale']*factor
nl=raw[:,c['piece']['source_newline_coordinate']]/c['piece']['source_newline_scale']*factor
label=torch.where(nl>.5,torch.full_like(nl,10,dtype=torch.long),digit.round().long().clamp(0,9));expected=torch.tensor(ids)[label]
clear=torch.tensor(c['piece']['cancellation_coordinates']);cancel=torch.zeros_like(raw)
cancel[:,clear]=-raw[:,clear]*factor[:,None]/(640**.5*1e6)
trials=[];saved={}
for count in (32,48,64,96,128):
 candidate=geos['candidates'][f'2_{count}'];literal=candidate['report']
 rec={'pool_count':count,'energy_penalty':2.,'literal_margin':literal['minimum_full_vocabulary_margin'],
      'native_raw_energy_fraction':literal['native_raw_energy_fraction']}
 if rec['literal_margin']<=0:
  rec['eligible_literal_geometry']=False;trials.append(rec);print(json.dumps(rec),flush=True);continue
 pool=candidate['coordinates'];assert set(clear.tolist())<=set(pool.tolist())
 gamma=1+h['model.layers.17.post_feedforward_layernorm.weight'].double();gamma[pool]=1e6
 directions=torch.zeros(640,11,dtype=torch.float64);directions[pool]=candidate['directions']
 program=coeff@directions.T+cancel
 output=raw+program/(program.square().mean(1,keepdim=True)+1e-30).sqrt()*gamma
 output/= (output.square().mean(1,keepdim=True)+1e-6).sqrt()
 desired=(output*fg*V[expected].double()).sum(1);other=torch.full_like(desired,-torch.inf)
 for start in range(0,len(V),4096):
  scores=output@(V[start:start+4096].double()*fg).T
  local=expected-start;valid=(local>=0)&(local<scores.shape[1]);rows=valid.nonzero().flatten();scores[rows,local[rows]]=-torch.inf
  other=torch.maximum(other,scores.amax(1))
 margin=desired-other
 rec.update(eligible_literal_geometry=True,minimum_register_component_margin=float(margin.min()),
  wrong_current_register_outputs=int((margin<=0).sum()),recorded_register_outputs=len(margin))
 trials.append(rec);saved[str(count)]={'coordinates':pool,'directions':candidate['directions'],'report':rec}
 print(json.dumps(rec),flush=True)
report={'count_trials':[32,48,64,96,128],'energy_penalty':2.,'trials':trials,
 'limitation':'Dominant hand-programme output component omits native arithmetic MLP contribution; official forward remains necessary.',
 'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (base,fp,gp)},'source_sha':sha(__file__)}
atomic_json(ROOT/'artifacts/a76_reduced_output_pool_v1.json',report);atomic_torch(ROOT/'artifacts/a76_reduced_output_pool_v1.pt',{'candidates':saved,'report':report})
