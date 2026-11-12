"""Cross-check A234 programme gates on512 existing older text windows.

Observed head contexts uniquely determine factual linear value mixtures.
Recover those mixtures, apply HOST V algebraically, and inspect only the
hand-built programme gates at the resulting input. No probabilities or
native nonlinear features are evaluated; no weights are fitted or written.
"""
import json,resource,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,atomic_torch,sha
h=host_state();base=ROOT/'experiments/a234v1_exploratory_selective_reader/delta.pt'
oldbase=ROOT/'experiments/a117v1_six_unit_carry/delta.pt'
ap=ROOT/'artifacts/a119v1_current_word.pt'
ch,old=expand(h,base),expand(h,oldbase)
a=torch.load(ap,weights_only=True,mmap=True);assert a['checkpoint_sha256']==sha(oldbase)
pre='model.layers.3.'
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name.startswith(pre+'self_attn.') and not name.endswith('v_proj.weight'):
        assert torch.equal(ch.get(name,h[name]),old.get(name,h[name])),name
for suffix in ('input_layernorm.weight','post_attention_layernorm.weight','pre_feedforward_layernorm.weight'):
    name=pre+suffix;assert torch.equal(ch.get(name,h[name]),old.get(name,h[name])),name
vn=pre+'self_attn.v_proj.weight';on=pre+'self_attn.o_proj.weight'
assert torch.equal(ch.get(vn,h[vn]),h[vn]) and torch.equal(ch.get(on,h[on]),h[on])
X=a['captures']['3.input'];known=a['captures']['3.heads'].reshape(512,64,4,256).permute(0,2,1,3)
O=h[on].double().reshape(640,4,256)
oldV=old[vn];newV=h[vn].double();del old
attention_gain=1+h[pre+'post_attention_layernorm.weight'].double()
input_gain=1+h[pre+'input_layernorm.weight'].double()
pre_gain=1+h[pre+'pre_feedforward_layernorm.weight'].double()
def norm(x):return x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()
new_inputs=[];window_reports=[];error2=0.;ref2=0.
for i in range(512):
    # This inverse reconstructs observed context vectors. It cannot supply
    # new attention weights for any changed Q or K.
    values=(X[i]@oldV.T).double()
    fit=torch.linalg.lstsq(values.T,known[i].double().reshape(4*64,256).T,rcond=1e-10,driver='gelsd')
    coefficient=fit.solution.T.reshape(4,64,64)
    reproduced=coefficient@values
    err=float((reproduced-known[i].double()).square().sum())
    reference=float(known[i].double().square().sum());error2+=err;ref2+=reference
    future=torch.ones(64,64,dtype=torch.bool).triu(1)
    rec={'window':i,'rank':int(fit.rank),'singular_ratio':float(fit.singular_values[-1]/fit.singular_values[0]),
        'relative_known_head_error':(err/reference)**.5,
        'maximum_future_coefficient':float(coefficient[:,future].abs().max()),
        'minimum_coefficient':float(coefficient.min()),
        'maximum_coefficient_sum_error':float((coefficient.sum(-1)-1).abs().max())}
    assert rec['rank']==64 and rec['relative_known_head_error']<1e-5
    assert rec['maximum_future_coefficient']<.002 and rec['minimum_coefficient']>-.002 and rec['maximum_coefficient_sum_error']<.002
    mixtures=coefficient@X[i].double()
    new_heads=mixtures@newV.T
    old_raw=torch.einsum('htv,ohv->to',known[i].double(),O)
    new_raw=torch.einsum('htv,ohv->to',new_heads,O)
    # Subtract the observed old branch from its observed sum, then add the
    # changed linear branch. This has reconstruction/rounding uncertainty.
    state2=a['captures']['3.raw'][i].double()-norm(old_raw)*attention_gain
    rec['recovered_input_norm_relative_error']=float((norm(state2)*input_gain-X[i].double()).norm()/X[i].double().norm())
    assert rec['recovered_input_norm_relative_error']<1e-5
    state3=state2+norm(new_raw)*attention_gain
    new_inputs.append((norm(state3)*pre_gain).float())
    window_reports.append(rec)
    if (i+1)%64==0:print(json.dumps({'recovered_windows':i+1}),flush=True)
inputs=torch.stack(new_inputs)
c=torch.load(base.parent/'circuit.pt',weights_only=True);units=c['native_a_stage']['units'].long()
G,U=[ch[pre+'mlp.'+s+'_proj.weight'][units] for s in ('gate','up')]
g=inputs@G.T;u=inputs@U.T
programme=g.clamp_min(0)*u
raw=programme.double()@ch[pre+'mlp.down_proj.weight'][:,units].double().T
active=g.amax(-1)>0
report={'analysis_only':True,'base_checkpoint':str(base.relative_to(ROOT)),'recorded_parent':str(oldbase.relative_to(ROOT)),
    'windows':512,'positions_per_window':64,'all_tokens':32768,'programme_units':units.tolist(),
    'maximum_programme_gate':float(g.max()),'maximum_programme_gate_by_unit':g.amax((0,1)).tolist(),
    'positive_gate_positions':int(active.sum()),'windows_with_positive_gate':int(active.any(1).sum()),
    'positions_above_minus16':int((g.amax(-1)>-16).sum()),
    'positive_gate_counts_by_unit':(g>0).sum((0,1)).tolist(),
    'maximum_hand_ReLU_programme_raw_norm':float(raw.norm(dim=-1).max()),
    'aggregate_known_head_reconstruction_relative_error':(error2/ref2)**.5,
    'maximum_recovered_input_norm_relative_error':max(r['recovered_input_norm_relative_error'] for r in window_reports),
    'minimum_source_value_rank':min(r['rank'] for r in window_reports),
    'maximum_future_coefficient':max(r['maximum_future_coefficient'] for r in window_reports),
    'window_inverse_checks':window_reports,
    'scope':'Factual captured head mixtures and unchanged upstream/Q/K; changed V is linear algebra. Recovered before-attention residual and changed native-V input have float32 reconstruction uncertainty, not a full interval certificate.',
    'limitation':'Calibration-only gate diagnostic; does not identify held-out failures or compute changed native MLP/downstream predictions.',
    'no_attention_probability_execution_native_activation_replay_gradients_answers_or_weight_fitting':True,
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (base,base.parent/'circuit.pt',oldbase,ap)},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a238v1_old_word_quantizer_gates.json';atomic_json(out,report)
needed=inputs.numel()*inputs.element_size()+a['inputs'].numel()*a['inputs'].element_size()+10_000_000
size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
assert size+needed<40_000_000_000
atomic_torch(out.with_suffix('.pt'),{'pre_feedforward_inputs':inputs,'inputs':a['inputs'],'report':report})
print(json.dumps({k:v for k,v in report.items() if k not in ('window_inverse_checks','inputs','sources')}),flush=True)
