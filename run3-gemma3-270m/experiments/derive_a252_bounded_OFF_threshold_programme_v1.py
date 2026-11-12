"""Bounded five-value OFF-amplitude family for eight-unit threshold readers.

Fixed union-text OFF construction, analytic saturation scaling and common
format-reference gate shifts. No new trial values, model execution or
answer target. Native allocation is a separate required stage.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state(); base = ROOT / 'experiments/a202v1_native_value_only_exploratory/delta.pt'; ch = expand(h, base)
fp = ROOT / 'artifacts/a244v1_two_threshold_field_catalog.pt'
mp = ROOT / 'artifacts/a191v1_recorded_reader3_mixtures.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'; ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
ap = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
lp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
check_paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
catalog, mix, w, e = [torch.load(p, weights_only=True, mmap=True) for p in (fp, mp, wp, ep)]
source_base = ROOT / mix['report']['base_checkpoint']; reference = expand(h, source_base)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name.startswith('model.layers.3.self_attn.'):
        if name == 'model.layers.3.self_attn.v_proj.weight':
            assert torch.equal(ch.get(name, h[name]), h[name])
        else: assert torch.equal(ch.get(name, h[name]), reference.get(name, h[name])), name
del reference
assert mix['report']['inputs'][str(source_base.relative_to(ROOT))] == sha(source_base)
assert torch.equal(w['inputs'], e['inputs']) and torch.equal(w['inputs'], mix['inputs'])
pre = 'model.layers.3.'
V = h[pre + 'self_attn.v_proj.weight'].double()
O = h[pre + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256)
gamma = 1 + h[pre + 'post_attention_layernorm.weight'].double()
gp = 1 + h[pre + 'pre_feedforward_layernorm.weight'].double()
rows = catalog['existing_quiet_rows'].double(); assert rows.shape == (13, 640)
C = rows[12]; B = rows[:12]
stage = torch.load(base.parent / 'circuit.pt', weights_only=True)['native_a_stage']
old_units = stage['units'].long(); assert len(old_units) == 4
Wprogramme = ch[pre + 'mlp.down_proj.weight'][:, old_units].double()
old_gate_rows = ch[pre + 'mlp.gate_proj.weight'][old_units]
assert torch.equal(old_gate_rows, old_gate_rows[:1].expand_as(old_gate_rows))
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
current = {}
for tag in ('word', 'format'):
    before = w['captures'][f'{tag}.2.raw'] + fnorm(e['captures'][f'{tag}.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
    assert torch.equal(fnorm(before) * (1 + h[pre + 'input_layernorm.weight']), mix['records'][tag]['input'])
    raw = torch.einsum('nhsv,ohv->nso', mix['records'][tag]['mixtures'].double() @ V.T, O)
    current[tag] = (norm(before.double() + norm(raw) * gamma) * gp).float().double()
layout = json.loads(lp.read_text()); certificate = json.loads(cp.read_text())
assert certificate['base_text_trajectory_certified']
assert certificate['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
native = torch.load(ap, weights_only=True, mmap=True)
assert torch.equal(native['inputs'], w['inputs'][:256])
surrogate_word = (fnorm(native['captures']['3.raw']) * (1 + h[pre + 'pre_feedforward_layernorm.weight'])).double()
assert surrogate_word.shape == current['word'].shape
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])
bits = 2 * ((torch.arange(10)[:, None] >> torch.arange(4)[None]) & 1).double() - 1
positions = [3, 2, 1, 0]
def labels(tokens):
    match = tokens[..., None] == ids; assert match.sum(-1).eq(1).all()
    return match.long().argmax(-1)
current_labels = labels(w['inputs'][256:, positions[:3]])
historical, historical_labels = [], []
for path in check_paths:
    cap = torch.load(path, weights_only=True, mmap=True); sx = cap['captures']['3.input'].double()
    raw = sum(sx[:, pos] @ (O[:, head] @ V).T for head, pos in enumerate(positions))
    state = cap['captures']['2.out'][:, 8].double() + norm(raw) * gamma
    historical.append((norm(state) * gp).float().double())
    historical_labels.append(labels(cap['inputs'][:, positions[:3]]))

# Reuse A224's selected OFF covariance constants, with no new sweep.
P=torch.cat([current['format'][:96,8],historical[0],historical[1]])
training_labels=torch.cat([current_labels[:96],historical_labels[0],historical_labels[1]])
quiet_C=rows[12];positive_reference=P@quiet_C
assert positive_reference.min()>0
relative_positive=P/positive_reference[:,None]
non_equals=torch.tensor([j for j in range(14) if j!=8])
word_train=torch.cat([current['word'][:192].reshape(-1,640),surrogate_word[:192].reshape(-1,640)])
format_train=current['format'][:96,non_equals].reshape(-1,640)
negative_train=torch.cat([word_train,format_train])
negative_all=torch.cat([current['word'].reshape(-1,640),surrogate_word.reshape(-1,640),current['format'][:,non_equals].reshape(-1,640)])
oldp=ROOT/'artifacts/a238v1_old_word_quantizer_gates.pt'
old=torch.load(oldp,weights_only=True,mmap=True)
older_word=old['pre_feedforward_inputs'].reshape(-1,640).double()
negative_all=torch.cat([negative_all,older_word])
mass=.1;positive_mass=1.;metric_ridge=1e-6
CW=word_train.T@word_train/len(word_train)
CN=format_train.T@format_train/len(format_train)
covariance=(CW+mass*CN)/(1+mass)
rhs=-(word_train.mean(0)+mass*format_train.mean(0))/(1+mass)
covariance.diagonal().add_(metric_ridge*covariance.diagonal().clamp_min(1e-15))
factor=torch.linalg.cholesky((covariance+covariance.T)/2)
whitened=torch.linalg.solve_triangular(factor,relative_positive.T,upper=False).T
_,singular,right=torch.linalg.svd(whitened/len(P)**.5,full_matrices=False)
wrhs=torch.linalg.solve_triangular(factor,rhs[:,None],upper=False)[:,0]
off=torch.linalg.solve_triangular(factor.T,(right.T@((right@wrhs)/(1+positive_mass*singular.square())))[:,None],upper=True)[:,0]
off_train=negative_train@off
assert off_train.max()<-1e-8
analog=catalog['existing_analog_rows'].double();C=analog[12]
bit_weights=2.**torch.arange(4,dtype=torch.float64)/9
D=(analog[:12].reshape(3,4,640)*bit_weights[None,:,None]).sum(1)+(2/3)*C
thresholds=catalog['thresholds'].tolist()
pairs=catalog['report']['eligible_threshold_pairs']
common_gate=old_gate_rows[0].double();reference_gate=P@common_gate
negative_common=negative_train@common_gate
assert reference_gate.min()>32
up_gain=float(stage['piece']['common_up_gain']);assert up_gain==1e6
fractions=[0.,.125,.25,.375,.5]
control_paths=[ROOT/'artifacts/a246v1_two_threshold_programme.json',ROOT/'artifacts/a251v1_source_margin_threshold_programme.json']
control_reports=[json.loads(p.read_text()) for p in control_paths]
controls=[]
trials=[];selected={}
for pair in pairs:
    first,second=pair;jumps=[first,second-first]
    js=[thresholds.index(t) for t in pair]
    quiet=catalog['quiet_threshold_rows'][js].double()
    analog_threshold=catalog['analog_threshold_rows'][js].double()
    source_signs=2*(training_labels[:,0,None]>=torch.tensor(pair)[None]).double()-1
    negative_guard=.25*float(positive_reference.min())
    bound=((negative_train@quiet.T+negative_guard)/(-off_train[:,None])).max().clamp_min(0)
    lower_scale=1.001*float(bound)+1e-6
    original_signed=(P@quiet.T)*source_signs
    signed_OFF=(P@off)[:,None]*source_signs
    assert original_signed.min()>0
    adverse=signed_OFF<0
    assert adverse.any()
    upper_scale=float((.5*original_signed[adverse]/(-signed_OFF[adverse])).min())
    rec={'threshold_pair':pair,'literal_jump_sizes':jumps,'programme_units':8,
        'minimum_training_negative_OFF_scale':lower_scale,
        'maximum_half_source_margin_OFF_scale':upper_scale,'eligible':False}
    if upper_scale<lower_scale:
        rec['reason']='No OFF scale preserves half every source margin and all training negatives';trials.append(rec);continue
    interval_record=dict(rec)
    for fraction in fractions:
        rec=dict(interval_record);rec['OFF_interval_fraction']=fraction
        off_scale=lower_scale+fraction*(upper_scale-lower_scale)
        centers=quiet+off_scale*off
        signed=(P@centers.T)*source_signs
        assert (signed>=.5*original_signed-1e-12).all()
        minimum=float(signed.min());negative_max=float((negative_train@centers.T).max())
        rec.update(OFF_scale=off_scale,minimum_unscaled_signed_center=minimum,maximum_unscaled_negative_center=negative_max)
        if minimum<=0 or negative_max>=0:
            rec['reason']='OFF center signs fail';trials.append(rec);continue
        center_gain=1.001*max(64/minimum,32/(-negative_max))
        centers*=center_gain
        source_bound=float(((P@centers.T)*source_signs/reference_gate[:,None]).min())
        negative_bound=float((-(negative_train@centers.T)/negative_common.abs()[:,None].clamp_min(1e-30)).min())
        shift=min(.25,.25*source_bound,.25*negative_bound);assert shift>0
        remainder=D[0]-sum((jump/9)*(analog_threshold[j]+C) for j,jump in enumerate(jumps))
        G=[common_gate]*4;U=[C,remainder,D[1],D[2]]
        transform=[torch.eye(4,dtype=torch.float64)[:,j] for j in range(4)]
        for j,jump in enumerate(jumps):
            for sign in (1.,-1.):
                G.append(centers[j]+sign*shift*common_gate);U.append(C)
                col=torch.zeros(4,dtype=torch.float64);col[1]=sign*jump/(9*shift);transform.append(col)
        G=torch.stack(G);U=torch.stack(U);transform=torch.stack(transform,1)
        unscaled=Wprogramme@transform
        down_scale=min(1.,1e6/float(unscaled.abs().max()));transform*=down_scale
        U*=up_gain/down_scale
        balance=min(1.,(1e6*(1-4*torch.finfo(torch.float32).eps))/float(G.abs().max()))
        G=(G*balance).float();U=(U/balance).float();down=(Wprogramme@transform).float()
        rec.update(shared_common_gate_shift=shift,center_gain=center_gain,
            uniform_programme_down_scale=down_scale,analytic_Gate_Up_balance=balance,
            maximum_gate_weight=float(G.abs().max()),maximum_up_weight=float(U.abs().max()),
            maximum_programme_down_weight=float(down.abs().max()))
        if max(rec['maximum_gate_weight'],rec['maximum_up_weight'],rec['maximum_programme_down_weight'])>1e6:
            rec['reason']='Weight cap';trials.append(rec);continue
        rec['maximum_all_negative_gate']=float((negative_all.float()@G.T).max())
        rec['maximum_older512_word_gate']=float((older_word.float()@G.T).max())
        groups=[('current128',current['format'][:,8],current_labels),('current_holdback32',current['format'][96:,8],current_labels[96:])]
        groups += [(name,X,y) for name,X,y in zip(('historical_a','historical_stress','historical_c'),historical,historical_labels)]
        diagnostics=[]
        for name,X,y in groups:
            g=(X.float()@G.T).double();u=(X.float()@U.T).double()
            features=g.clamp_min(0)*u;code=features@transform.T
            valid=(code[:,0]>0)&code.isfinite().all(1)
            got=4.5*code[valid,1:]/code[valid,:1]+4.5
            error=(got-y[valid]).abs()
            signs=2*(y[:,0,None]>=torch.tensor(pair)[None]).double()-1
            expected=torch.cat([torch.ones(len(X),4),signs.repeat_interleave(2,1)],1)
            signed=g*expected
            diagnostics.append({'split':name,'finite_decoding':bool(valid.all()),
                'maximum_source_digit_error':float(error.max()) if valid.all() else None,
                'maximum_errors_by_place':error.amax(0).tolist() if valid.all() else None,
                'wrong_source_digits':int((got.round()!=y[valid]).sum()),'invalid_reference_cases':int((~valid).sum()),
                'minimum_signed_gate':float(signed.min()),'unsaturated_or_wrong_gate_cases':int((signed<=16).any(1).sum()),
                'maximum_programme_Down_identity_error':float((features@down.double().T-code@Wprogramme.T).abs().max())})
        rec['source_diagnostics']=diagnostics
        rec['maximum_current_digit_error']=max(d['maximum_source_digit_error'] if d['maximum_source_digit_error'] is not None else float('inf') for d in diagnostics[:2])
        rec['maximum_historical_digit_error']=max(d['maximum_source_digit_error'] if d['maximum_source_digit_error'] is not None else float('inf') for d in diagnostics[2:])
        rec['eligible']=bool(G.isfinite().all() and U.isfinite().all() and down.isfinite().all()
            and rec['maximum_all_negative_gate']<-16 and rec['maximum_current_digit_error']<.25
            and rec['maximum_historical_digit_error']<.4
            and all(d['finite_decoding'] and d['wrong_source_digits']==0 and d['minimum_signed_gate']>16 for d in diagnostics))
        label=f'thresholds_{first}_{second}_f{fraction:g}'.replace('.','p');trials.append(rec)
        if rec['eligible']:selected[label]={'gate_rows':G,'up_rows':U,'programme_down_columns':down,'effective_feature_transform':transform,'record':rec}
        if fraction in (0.,.5):
            control_report=control_reports[0 if fraction==0 else 1]
            prior=next(r for r in control_report['trials'] if r['threshold_pair']==pair)
            differences={key:abs(rec[key]-prior[key]) for key in ('maximum_all_negative_gate','maximum_current_digit_error','maximum_historical_digit_error')}
            assert all(differences[key]<1e-5+1e-6*abs(prior[key]) for key in differences),differences
            controls.append({'threshold_pair':pair,'fraction':fraction,'absolute_control_differences':differences})
        print(json.dumps(rec),flush=True)
report={'analysis_only':True,'base_checkpoint':str(base.relative_to(ROOT)),
    'source_threshold_pairs':pairs,'source_field_catalog':str(fp.relative_to(ROOT)),
    'fixed_negative_format_mass':mass,'fixed_relative_positive_mass':positive_mass,'fixed_negative_metric_ridge_fraction':metric_ridge,
    'fixed_common_up_gain':up_gain,'OFF_interval_fraction_trials':fractions,'new_fraction_values_relative_to_A246_A251':[.125,.25,.375],'minimum_and_midpoint_reproduction_controls':controls,
    'OFF_scale_rule':'Lower bound + declared fraction*(upper-lower); lower is the original training-negative bound,upper preserves half every original signed source-field margin; reject empty interval',
    'center_gain_rule':'1.001 times bound for signed source64 and negative32',
    'shared_shift_rule':'Minimum of0.25,quarter source-center/common ratio,quarter negative-center/absolute-common ratio',
    'literal_identity':'Analog remainder D-sum(jump/9*(analog_threshold+C)) plus positive paired-hinge outputs2*jump/9*C*I[d>=threshold]',
    'training_word_windows_per_population':192,'validation_word_windows_per_population':64,
    'extra_older_word_validation_windows':512,'historical_training_groups':['historical_a','historical_stress'],
    'historical_holdback_group':'historical_c','current_training_windows':96,'current_holdback_windows':32,
    'source_guard':'Current errors<0.25,historical<0.4,no wrong/invalid digits,all signed source Gates>16; every checked word/non-equals Gate<-16',
    'scope':'Programme rows only. Native-unit allocation and official scoring required. Historical sources use ideal routing; old-word inputs are reconstructed factual mixtures.',
    'only_hand_programme_ReLU_ideal_no_native_nonlinear_replay_or_gradients_answers_iterative_fit':True,
    'trials':trials,'selected':{k:v['record'] for k,v in selected.items()},
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [base,base.parent/'circuit.pt',source_base,fp,mp,wp,ep,ap,lp,cp,oldp]+check_paths+control_paths},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a252v1_bounded_OFF_threshold_programme.json';atomic_json(out,report)
if selected:atomic_torch(out.with_suffix('.pt'),{'designs':selected,'report':report})
print(json.dumps({'selected_designs':list(selected),'peak_kib':report['maximum_resident_set_kib']}),flush=True)

