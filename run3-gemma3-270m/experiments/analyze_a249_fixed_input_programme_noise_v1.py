"""Exact linear Gaussian noise moments for hand Gate/Up rows at saved inputs.

This is a fixed-input component diagnostic. It neither perturbs a full
network nor predicts end-to-end accuracy. Gate/Up products below use only
the explicitly hand-programmed saturated-positive ideal, never GELU.
"""
import json,math,resource,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,sha
h=host_state();base=ROOT/'experiments/a142v2_exploratory_projected/delta.pt'
reference=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
format_reference=ROOT/'experiments/a124v1_tight_newline_phase/delta.pt'
ap=ROOT/'artifacts/a150v1_early_native_basis.pt';fp=ROOT/'artifacts/a124v1_syntax_stress.pt'
wp=ROOT/'artifacts/a142v1_projected_readout_basis.pt'
ch,ref,fref=[expand(h,p) for p in (base,reference,format_reference)]
a,f,w=[torch.load(p,weights_only=True,mmap=True) for p in (ap,fp,wp)]
assert a['checkpoint_sha256']==w['checkpoint_sha256']==sha(reference)
assert f['checkpoint_sha256']==sha(format_reference)
dn='model.layers.17.mlp.down_proj.weight';pn='model.layers.17.post_feedforward_layernorm.weight'
for name in h:
    if name not in (dn,pn):
        assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
        assert torch.equal(ch.get(name,h[name]),fref.get(name,h[name])),name
del ref,fref
sigmas=[1e-5,1e-4,1e-3]
def fnorm(x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
def summary(x):
    if x.numel()==0:return None
    return {'count':x.numel(),'minimum':float(x.min()),'median':float(x.median()),'p99':float(x.quantile(.99)),'maximum':float(x.max())}
reports=[]
for layer in list(range(1,9))+[17]:
    pre=f'model.layers.{layer}.';gn=pre+'mlp.gate_proj.weight';un=pre+'mlp.up_proj.weight'
    changed_G=ch[gn].view(torch.int32)!=h[gn].view(torch.int32)
    changed_U=ch[un].view(torch.int32)!=h[un].view(torch.int32)
    units=(changed_G.any(1)|changed_U.any(1)).nonzero().flatten()
    assert 1<=len(units)<=32
    G,U=ch[gn][units].double(),ch[un][units].double()
    GM=changed_G[units];UM=changed_U[units]
    pg=1+ch.get(pre+'pre_feedforward_layernorm.weight',h[pre+'pre_feedforward_layernorm.weight'])
    if layer==17:
        populations=[('format_512_positions8to12',f['captures']['17.raw'][:,8:13]),
                     ('word_512_positions0to63',w['captures']['17.raw'])]
    else:
        populations=[('format_128_positions8to12',a['captures'][f'format.{layer}.raw'][:,8:13]),
                     ('word_256_positions0to63',a['captures'][f'word.{layer}.raw'])]
    for label,raw in populations:
        X=(fnorm(raw)*pg).reshape(-1,640).double()
        g=X@G.T;u=X@U.T
        vg=X.square()@(G.square()*GM).T
        vu=X.square()@(U.square()*UM).T
        assert vg.min()>=0 and vu.min()>=0
        good=g>16
        rows=[]
        for sigma in sigmas:
            sg=sigma*vg.sqrt();su=sigma*vu.sqrt()
            denom=sg.clamp_min(1e-300)*math.sqrt(2)
            flip=.5*torch.erfc(g.abs()/denom)
            flip=torch.where(sg>0,flip,torch.zeros_like(flip))
            unsaturated=.5*torch.erfc((g.abs()-16)/denom)
            unsaturated=torch.where(sg>0,unsaturated,(g.abs()<=16).double())
            positive_probability=.5*torch.erfc(-g/denom)
            positive_probability=torch.where(sg>0,positive_probability,(g>0).double())
            variance=sigma**2*(u.square()*vg+g.square()*vu)+sigma**4*vg*vu
            relative=variance.sqrt()/(g*u).abs().clamp_min(1e-30)
            unit_rows=[]
            for j,unit in enumerate(units.tolist()):
                active=good[:,j]
                unit_rows.append({'unit':unit,'baseline_gate_minimum':float(g[:,j].min()),
                    'baseline_gate_maximum':float(g[:,j].max()),'baseline_saturated_positive_count':int(active.sum()),
                    'minimum_gate_signal_to_noise':float((g[:,j].abs()/sg[:,j].clamp_min(1e-300)).min()),
                    'maximum_marginal_gate_sign_flip_probability':float(flip[:,j].max()),
                    'expected_gate_sign_flip_count_by_linearity':float(flip[:,j].sum()),
                    'maximum_marginal_loss_of_same_sign_saturation_probability':float(unsaturated[:,j].max()),
                    'expected_positive_gate_count_by_linearity':float(positive_probability[:,j].sum()),
                    'saturated_positive_GLUp_product_relative_noise_SD':summary(relative[active,j]),
                    'saturated_positive_Up_relative_noise_SD':summary((su[:,j]/u[:,j].abs().clamp_min(1e-30))[active])})
            rows.append({'sigma':sigma,'units':unit_rows,
                'expected_positive_gate_entries_by_linearity':float(positive_probability.sum()),
                'expected_sign_flip_entries_by_linearity':float(flip.sum())})
        reports.append({'layer':layer,'population':label,'positions':len(X),'programme_units':units.tolist(),
            'baseline_unsaturated_gate_entries':int((g.abs()<=16).sum()),
            'baseline_positive_gate_entries':int((g>0).sum()),'noise':rows})
        print(json.dumps({'layer':layer,'population':label,'units':len(units),'baseline_positive_entries':int((g>0).sum()),
            'expected_sign_flips_at_1e_minus3':rows[-1]['expected_sign_flip_entries_by_linearity']}),flush=True)
        del X,g,u,vg,vu
report={'analysis_only':True,'no_weights_or_checkpoint':True,'base_checkpoint':str(base.relative_to(ROOT)),
    'layers':list(range(1,9))+[17],'omitted_root0':'No exact root pre-MLP input is claimed by these retained captures',
    'sigma_values':sigmas,'perturbed_components':'Only actually changed entries of hand-programme Gate/Up rows; unchanged entries receive zero variance',
    'linear_noise_law':'At fixed input x, delta(w.x) is zero-mean Gaussian with variance sigma^2 sum_changed(w_i*x_i)^2',
    'positive_product_moment':'Independent Gate/Up weight perturbations give Var(GU)=sigma^2*(U^2*vG+G^2*vU)+sigma^4*vG*vU; applies to the saturated-positive hand ideal',
    'probability_scope':'Exact marginal real-linear Gaussian probabilities and expected counts by linearity; cases share weights and are not assumed independent',
    'excluded_effects':'Upstream changes/input noise, Down/norm noise, perturbed native activations, full-network or final prediction effects, floating-point reduction noise',
    'no_GELU_or_native_activation_evaluation_gradients_fits_answers_or_model_execution':True,
    'populations':reports,
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (base,reference,format_reference,ap,fp,wp)},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT/'artifacts/a249v1_fixed_input_programme_noise.json',report)
print(json.dumps({'peak_kib':report['maximum_resident_set_kib']}),flush=True)
