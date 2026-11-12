"""Analytic gain bound on emission coordinates outside physical scratch cleanup.

All final GLU features and raw Down weights are fixed observations/weights.
Holding every physical cleanup gain fixed, changing only the other writer
gains makes each protected vocabulary margin affine in one scalar. Direct inequalities compute its feasible interval, preserving the
existing per-register class outputs without fitting operands or sum answers.
This is geometry only; native numerator compensation is not applied here.
"""
import json,sys,resource
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,sha
h=host_state();base=ROOT/'experiments/a142v2_exploratory_projected/delta.pt';ch=expand(h,base)
cp=base.parent/'circuit.pt';c=torch.load(cp,weights_only=True)
fp=ROOT/'artifacts/a124v1_syntax_stress.pt';f=torch.load(fp,weights_only=True,mmap=True)
refpath=ROOT/'experiments/a124v1_tight_newline_phase/delta.pt';ref=expand(h,refpath)
assert f['checkpoint_sha256']==sha(refpath)
for name in h:
 if name in ['model.layers.17.mlp.down_proj.weight','model.layers.17.post_feedforward_layernorm.weight']:continue
 assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
del ref
all_pool=c['output_pool'].long();units=c['units'].long()
clear=torch.tensor(c['piece']['cancellation_coordinates'])
assert torch.isin(clear,all_pool).all()
pool=all_pool[~torch.isin(all_pool,clear)]
assert len(pool)==86 and len(clear)==10
dn='model.layers.17.mlp.down_proj.weight';pn='model.layers.17.post_feedforward_layernorm.weight'
gamma=1+ch[pn].double();assert (gamma[pool]==1e6).all()
F=f['captures']['17.features'][:,8:13].double().reshape(-1,2048)
before=f['captures']['17.raw'][:,8:13].double().reshape(-1,640)
R=F@ch[dn].double().T;Z=R/(R.square().mean(1,keepdim=True)+1e-6).sqrt()
A=torch.zeros_like(Z);A[:,pool]=Z[:,pool]
B=before+Z*gamma;B[:,pool]=before[:,pool]
expected=f['topk_ids'][:,:,0].reshape(-1).long()
E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
AA=A*fg;BB=B*fg
ownA=(AA*E[expected].double()).sum(1);ownB=(BB*E[expected].double()).sum(1)
lower=torch.zeros(len(F),dtype=torch.float64);upper=torch.full_like(lower,torch.inf)
zero_bad=0;baseline_margin=torch.full_like(lower,torch.inf)
for start in range(0,len(E),4096):
 emb=E[start:start+4096].double()
 da=ownA[:,None]-AA@emb.T;db=ownB[:,None]-BB@emb.T
 local=expected-start;rows=((local>=0)&(local<len(emb))).nonzero().flatten()
 da[rows,local[rows]]=0;db[rows,local[rows]]=torch.inf
 positive=da>0;negative=da<0
 lo=torch.where(positive,-db/da.clamp_min(1e-300),-torch.inf).amax(1)
 hi=torch.where(negative,-db/da.clamp_max(-1e-300),torch.inf).amin(1)
 lower=torch.maximum(lower,lo);upper=torch.minimum(upper,hi)
 zero_bad+=int(((da==0)&(db<=0)).sum())
 baseline_margin=torch.minimum(baseline_margin,(1e6*da+db).amin(1))
assert zero_bad==0 and (baseline_margin>0).all()
minimum=float(lower.max());maximum=float(upper.min())
assert minimum<1e6<maximum
record={'analysis_only':True,'fixed_raw_weights':True,'fixed_recorded_features':True,
 'uniform_writer_count':len(pool),'original_writer_gain':1e6,
 'fixed_cleanup_gain_count':len(clear),'fixed_cleanup_coordinates':clear.tolist(),
 'analytic_strict_lower_gain_bound':minimum,'analytic_strict_upper_gain_bound':maximum,
 'per_register_lower_gain_quantiles':torch.quantile(lower,torch.tensor([0.,.5,.9,.99,1.],dtype=torch.float64)).tolist(),
 'limiting_register_index':int(lower.argmax()),'limiting_current_token_id':int(expected[lower.argmax()]),
 'HOST_writer_gain_absolute_quantiles':torch.quantile((1+h[pn][pool].double()).abs(),torch.tensor([0.,.5,.9,1.],dtype=torch.float64)).tolist(),
 'minimum_original_unnormalized_margin':float(baseline_margin.min()),'register_positions':len(F),
 'native_compensation_not_yet_applied':True,
 'no_model_replay_gradients_or_operand_answer_targets':True,
 'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
 'inputs':{str(p.relative_to(ROOT)):sha(p)for p in[base,cp,fp,refpath]},
 'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT/'artifacts/a177v1_separate_emission_gain_floor.json',record)
print(json.dumps(record),flush=True)
