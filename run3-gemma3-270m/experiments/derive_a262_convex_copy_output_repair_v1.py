"""Copier output repair with convex-mixture bounds for format preservation.
Recover text heads by a unique inverse of recorded outputs. Format bounds
cover every causal convex combination of the known native source values. Preserve all
copier Gate/Up directions. No attention/native MLP replay or gradients.
Local A142 geometry is exact; transfer to A240 requires full scoring.
"""
import json, resource, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state(); base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
install_base = ROOT / 'experiments/a240v1_causal_metric_queries_on_a191/delta.pt'
refp = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
oldp = ROOT / 'artifacts/a119v1_current_word.pt'
oldbase = ROOT / 'experiments/a117v1_six_unit_carry/delta.pt'
ch, ref = expand(h, base), expand(h, refp)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(17))) or name.startswith('model.layers.17.') and name not in ('model.layers.17.mlp.down_proj.weight', 'model.layers.17.post_feedforward_layernorm.weight'):
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
e, host, w, old = [torch.load(p, weights_only=True, mmap=True) for p in (ep, hp, wp, oldp)]
assert e['checkpoint_sha256'] == w['checkpoint_sha256'] == sha(refp)
assert torch.equal(e['inputs'][:256], host['inputs']) and torch.equal(w['inputs'], e['inputs'])
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
units = ((ch['model.layers.7.mlp.gate_proj.weight'].view(torch.int32) != h['model.layers.7.mlp.gate_proj.weight'].view(torch.int32)).any(1) | (ch['model.layers.7.mlp.up_proj.weight'].view(torch.int32) != h['model.layers.7.mlp.up_proj.weight'].view(torch.int32)).any(1)).nonzero().flatten()
assert len(units) == 8

def fnorm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

def norm(x):
    return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

def recover_heads(inputs, raw, value_weight, output_weight, label, known=None):
    values = (inputs @ value_weight.T).double()
    outputs, ranks, ratios = [], [], []
    error2 = reference2 = known_error2 = known_reference2 = future_max = 0.
    negative_max = sum_error = 0.; prefix_rank_failures = prefix_solves = 0
    causal_affine = label != 'word'
    length = inputs.shape[1]
    for i in range(len(inputs)):
        matrix = torch.einsum('ohv,sv->ohs', output_weight, values[i])
        if causal_affine:
            # Recorded causal attention has no future source and each head's
            # source coefficients sum to one. Eliminate source0 exactly.
            coef = torch.zeros(4, length, length, dtype=torch.float64)
            base_output = matrix[:, :, 0].sum(1)
            local_ratios = []
            for position in range(length):
                if position == 0:
                    coef[:, 0, 0] = 1
                    continue
                design = (matrix[:, :, 1:position+1] - matrix[:, :, :1]).reshape(640, -1)
                fit = torch.linalg.lstsq(design, raw[i, position].double() - base_output,
                    rcond=1e-10, driver='gelsd')
                prefix_solves += 1
                prefix_rank_failures += int(int(fit.rank) != 4 * position)
                local_ratios.append(float(fit.singular_values[-1] / fit.singular_values[0]))
                coef[:, position, 1:position+1] = fit.solution.reshape(4, position)
                coef[:, position, 0] = 1 - coef[:, position, 1:position+1].sum(1)
            ranks.append(int(fit.rank)); ratios.append(min(local_ratios))
        else:
            fit = torch.linalg.lstsq(matrix.reshape(640, -1), raw[i].double().T,
                rcond=1e-10, driver='gelsd')
            coef = fit.solution.T.reshape(length, 4, length).permute(1, 0, 2)
            ranks.append(int(fit.rank))
            ratios.append(float(fit.singular_values[-1] / fit.singular_values[0]))
        heads = coef @ values[i]
        reproduced = torch.einsum('htv,ohv->to', heads, output_weight)
        error2 += float((reproduced - raw[i]).square().sum())
        reference2 += float(raw[i].double().square().sum())
        future = torch.ones(length, length, dtype=torch.bool).triu(1)
        future_max = max(future_max, float(coef[:, future].abs().max()))
        negative_max = max(negative_max, float((-coef).clamp_min(0).max()))
        sum_error = max(sum_error, float((coef.sum(-1)-1).abs().max()))
        if known is not None:
            known_error2 += float((heads-known[i]).square().sum())
            known_reference2 += float(known[i].square().sum())
        outputs.append(heads.permute(1, 0, 2).reshape(length, 1024).float())
        if (i+1) % 64 == 0:
            print(json.dumps({'inverse': label, 'windows': i+1}), flush=True)
    report = {'windows': len(inputs), 'tokens': length,
        'method': 'Unique causal affine source-difference inverse with exact per-head coefficient sum1' if causal_affine else 'Original full unique recorded-output inverse',
        'minimum_rank': min(ranks), 'required_rank': 4*(length-1) if causal_affine else 4*length,
        'minimum_singular_ratio': min(ratios), 'prefix_solves': prefix_solves,
        'prefix_rank_failures': prefix_rank_failures,
        'relative_raw_error': (error2/reference2)**.5, 'maximum_future_coefficient': future_max,
        'maximum_negative_coefficient': negative_max, 'maximum_coefficient_sum_error': sum_error}
    if known is not None:
        report['relative_known_head_error'] = (known_error2/known_reference2)**.5
    report['valid_unique_inverse'] = bool(min(ranks)==report['required_rank'] and prefix_rank_failures==0
        and report['relative_raw_error']<1e-5 and max(future_max, negative_max, sum_error)<.002)
    print(json.dumps({'inverse': label, **report}), flush=True)
    return torch.stack(outputs), report

# An existing directly observed head tensor checks the inverse independently.
oldch = expand(h, oldbase)
known = old['captures']['3.heads'][:8].reshape(8, 64, 4, 256).permute(0, 2, 1, 3).double()
O3 = h['model.layers.3.self_attn.o_proj.weight'].double().reshape(640, 4, 256)
known_raw = torch.einsum('nhtv,ohv->nto', known, O3).float()
_, independent = recover_heads(old['captures']['3.input'][:8], known_raw,
    oldch['model.layers.3.self_attn.v_proj.weight'], O3, 'known_layer3_heads', known)
assert independent['valid_unique_inverse'] and independent['relative_known_head_error'] < 1e-4
del oldch, known, known_raw, O3, old

states = {'word': h['model.embed_tokens.weight'][e['inputs'][:256]] * torch.tensor(640 ** .5, dtype=torch.float32),
          'format': h['model.embed_tokens.weight'][e['inputs'][256:, :14]] * torch.tensor(640 ** .5, dtype=torch.float32)}
teacher = states['word'].clone()
for layer in range(7):
    for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
        name = f'model.layers.{layer}.{part}.weight'
        for tag in states:
            states[tag] = states[tag] + fnorm(e['captures'][f'{tag}.{layer}.{branch}.raw']) * (1 + ch.get(name, h[name]))
        teacher = teacher + fnorm(host['captures'][f'{layer}.{branch}.raw']) * (1 + h[name])
pre = 'model.layers.7.'
on, vn = pre + 'self_attn.o_proj.weight', pre + 'self_attn.v_proj.weight'
assert torch.equal(ch.get(on, h[on]), h[on]) and torch.equal(ch.get(vn, h[vn]), h[vn])
O0 = h[on].double(); O = O0.reshape(640, 4, 256)
gamma = 1 + h[pre + 'post_attention_layernorm.weight'].double()
pg = 1 + h[pre + 'pre_feedforward_layernorm.weight'].double()
G = ch[pre + 'mlp.gate_proj.weight'][units].double()
U = ch[pre + 'mlp.up_proj.weight'][units].double()
teacher = teacher + fnorm(host['captures']['7.attention.raw']) * (1 + h[pre + 'post_attention_layernorm.weight'])
records, inverses = {}, {}
for tag in states:
    before_mlp = states[tag] + fnorm(e['captures'][f'{tag}.7.attention.raw']) * (1 + h[pre + 'post_attention_layernorm.weight'])
    assert torch.equal(before_mlp, w['captures'][f'{tag}.7.raw']), tag
    inp = fnorm(states[tag]) * (1 + h[pre + 'input_layernorm.weight'])
    if tag=='word':
        heads, info = recover_heads(inp, e['captures'][f'{tag}.7.attention.raw'], h[vn], O, tag)
        records[tag] = heads.double(); inverses[tag] = info
    else:
        format_values=(inp@h[vn].T).double()
        inverses[tag]={'valid_unique_inverse':True,'method':'No format inverse; causal convex-mixture bound from source values and recorded raw branch'}
del before_mlp
out = ROOT / 'artifacts/a262v1_sparse_copy_output_repair.json'
input_paths = [install_base, base, base.parent / 'circuit.pt', refp, ep, hp, wp, oldp, oldbase]
if not all(item['valid_unique_inverse'] for item in inverses.values()):
    atomic_json(out, {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
        'independent_check': independent, 'mixture_inverses': inverses,
    'format_method':'Causal source-value convex hull; norm triangle bounds through attention and pre-MLP RMS, actual rounded frame-null residual included',
    'format_scope':'Analytic real-arithmetic bounds for the rounded matrices on128recorded formats, not a complete float32 execution proof or held-out guarantee', 'selected': None,
        'reason': 'Recorded copier attention output does not admit the required accurate unique source-mixture inverse',
        'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
        'inputs': {str(p.relative_to(ROOT)): sha(p) for p in input_paths},
        'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    sys.exit(0)

recorded_before_mlp = states['word'] + fnorm(e['captures']['word.7.attention.raw']) * (1 + h[pre + 'post_attention_layernorm.weight'])
assert torch.equal(recorded_before_mlp, w['captures']['word.7.raw'])
installed = expand(h, install_base)
assert torch.equal(installed.get(on, h[on]), h[on])
for name in (vn, pre+'post_attention_layernorm.weight', pre+'mlp.gate_proj.weight', pre+'mlp.up_proj.weight'):
    assert torch.equal(installed.get(name, h[name]), ch.get(name, h[name])), name
del installed, recorded_before_mlp, ref
heads = records['word']; del records
before = states['word'].double(); fbefore = states['format'].double(); del states
target_state = teacher.double(); del teacher
trms = (target_state.square().mean(-1, keepdim=True) + 1e-6).sqrt()
raw0 = heads @ O0.T; rms0 = (raw0.square().mean(-1, keepdim=True) + 1e-6).sqrt()
host_branch = norm(host['captures']['7.attention.raw'].double()) * gamma
target = host_branch - norm(raw0) * gamma
nt = 192; nr = nt * 64
X = (heads / rms0).reshape(-1, 1024)
Y = target.reshape(-1, 640)
sd = X[:nr].square().mean(0).sqrt().clamp_min(1e-15); Z = X[:nr] / sd
C = Z.T @ Z / nr; rhs = Z.T @ Y[:nr] / nr
priority = Y[:nr].square().mean(0); priority[gamma.abs() < .05] = -torch.inf
rows = priority.argsort(descending=True)[:128]
frame = torch.cat([G.T * pg[:, None], U.T * pg[:, None]], 1)
frame /= frame.norm(dim=0).clamp_min(1e-15)
frame_reference = (frame.T @ (gamma[:, None] * O0)).norm(dim=1).clamp_min(1e-15)
u, singular, _ = torch.linalg.svd(frame[rows], full_matrices=False)
rank = int((singular > singular[0] * 1e-10).sum()); B = u[:, :rank]
metric = 1 / (1 + .01 * gamma.square().median() / gamma[rows].square())
Zp = metric[:, None] * B
projection = torch.eye(len(rows), dtype=torch.float64) - Zp @ torch.linalg.solve(B.T @ Zp, B.T)
ranking = C.clone(); ranking.diagonal().add_(.001)
precision = torch.linalg.inv(ranking)
beta = precision @ (rhs[:, rows] @ projection.T)
order = (beta.square().sum(1) / precision.diagonal().clamp_min(1e-15)).argsort(descending=True)
fraw0 = e['captures']['format.7.attention.raw'].double(); frms0 = (fraw0.square().mean(-1, keepdim=True) + 1e-6).sqrt()
fstate0 = fbefore + norm(fraw0) * gamma
finput0 = norm(fstate0) * pg
fg0, fu0 = finput0 @ G.T, finput0 @ U.T
gate_scale = fg0.abs().amax((0, 1)).clamp_min(1)
up_scale = fu0.abs().amax((0, 1)).clamp_min(1)
saturated = fg0.abs() >= 16
functionals=torch.cat([G.T*pg[:,None],U.T*pg[:,None]],1)
before_functionals=fbefore@functionals
raw_functionals=(fraw0*gamma)@functionals
original_functionals=torch.cat([fg0,fu0],-1)
raw_norm=fraw0.norm(dim=-1)
weighted_raw_norm=(fraw0*gamma).norm(dim=-1)
state_norm=fstate0.norm(dim=-1)
unit_scales=torch.cat([gate_scale,up_scale])

def format_bounds(matrix):
    delta=(matrix-O0).reshape(640,4,256)
    # For each head, every causal attention output lies in the convex hull
    # of its available source values. Triangle inequalities combine heads.
    raw_bound=torch.zeros_like(raw_norm)
    weighted_bound=torch.zeros_like(raw_norm)
    functional_bound=torch.zeros(*raw_norm.shape,16,dtype=torch.float64)
    synthetic=[torch.zeros_like(fraw0) for _ in range(3)]
    for head in range(4):
        perturbation=format_values@delta[:,head].T
        synthetic[0]+=perturbation[:,:1]
        synthetic[1]+=perturbation
        synthetic[2]+=perturbation.cumsum(1)/torch.arange(1,perturbation.shape[1]+1,dtype=torch.float64)[None,:,None]
        raw_bound+=perturbation.norm(dim=-1).cummax(dim=1).values
        weighted=perturbation*gamma
        weighted_bound+=weighted.norm(dim=-1).cummax(dim=1).values
        functional_bound+=(weighted@functionals).abs().cummax(dim=1).values
    nlo=((raw_norm-raw_bound).clamp_min(0).square()/640+1e-6).sqrt()
    nhi=((raw_norm+raw_bound).square()/640+1e-6).sqrt()
    oldn=frms0[...,0]
    inverse_change=torch.maximum((1/nlo-1/oldn).abs(),(1/nhi-1/oldn).abs())
    state_bound=weighted_bound/nlo+weighted_raw_norm*inverse_change
    mlo=((state_norm-state_bound).clamp_min(0).square()/640+1e-6).sqrt()
    mhi=((state_norm+state_bound).square()/640+1e-6).sqrt()
    a=raw_functionals/nlo[...,None];b=raw_functionals/nhi[...,None]
    numerator_lo=before_functionals+torch.minimum(a,b)-functional_bound/nlo[...,None]
    numerator_hi=before_functionals+torch.maximum(a,b)+functional_bound/nlo[...,None]
    endpoints=torch.stack([numerator_lo/mlo[...,None],numerator_lo/mhi[...,None],
                          numerator_hi/mlo[...,None],numerator_hi/mhi[...,None]])
    lo,hi=endpoints.amin(0),endpoints.amax(0)
    for displacement in synthetic:
        observed=norm(fbefore+norm(fraw0+displacement)*gamma)@functionals
        tolerance=1e-9*(1+observed.abs())
        assert (observed>=lo-tolerance).all() and (observed<=hi+tolerance).all(),'Synthetic convex-mixture interval control failed'
    changes=torch.maximum((lo-original_functionals).abs(),(hi-original_functionals).abs())/unit_scales
    sign_safe=torch.where(fg0>0,lo[...,:8]>0,hi[...,:8]<=0)
    return {'maximum_format_raw_RMS_relative_change':float(torch.maximum((nlo/oldn-1).abs(),(nhi/oldn-1).abs()).max()),
        'maximum_format_gate_relative_change':float(changes[...,:8].max()),
        'maximum_format_up_relative_change':float(changes[...,8:].max()),
        'format_gate_sign_changes':int((~sign_safe).sum()),
        'format_gate_sign_changes_are_uncertified_interval_counts':True,
        'maximum_format_raw_norm_perturbation_bound':float(raw_bound.max()),
        'maximum_format_post_attention_state_norm_perturbation_bound':float(state_bound.max()),
        'minimum_certified_gate_sign_margin':float(torch.where(fg0>0,lo[...,:8],-hi[...,:8]).min())}


def quality(matrix):
    raw = heads[nt:] @ matrix.T
    output = before[nt:] + norm(raw) * gamma
    wgate = (norm(output) * pg) @ G.T
    bounds=format_bounds(matrix)
    return {'validation_paired_HOST_branch_RMS': float((norm(raw)*gamma-host_branch[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_state_relative_rms': float(((output - target_state[nt:]) / trms[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_rms': float((norm(output) - norm(target_state[nt:])).square().mean().sqrt()),
        'maximum_word_programme_gate': float(wgate.max()),
        **bounds,
        'maximum_normalized_frame_null_error': float((frame.T @ (gamma[:, None] * (matrix - O0))).abs().max()),
        'maximum_relative_frame_null_error': float((frame.T @ (gamma[:, None] * (matrix - O0))).norm(dim=1).div(frame_reference).max())}

baseline = quality(O0)
assert baseline['format_gate_sign_changes']==0
assert baseline['maximum_format_gate_relative_change']<1e-8
assert baseline['maximum_format_up_relative_change']<1e-8
counts = [16, 32, 64, 128, 256]; ridges = [1e-6, 1e-4, .01, 1., 100.]; strengths = [.25, .5, .75, 1., 1.25]
bm = json.loads(install_base.with_suffix('.manifest.json').read_text())
trials, best = [], None
for count in counts:
    columns = order[:count]; gram = C[columns][:, columns]
    ev, V = torch.linalg.eigh((gram + gram.T) / 2); ev = ev.clamp_min(0); pr = V.T @ rhs[columns]
    for ridge in ridges:
        fit = (V @ (pr / (ev[:, None] + ridge))) / sd[columns, None]
        correction = (projection @ fit[:, rows].T) / gamma[rows, None]
        for strength in strengths:
            matrix = O0.clone()
            matrix[rows[:, None], columns[None]] = (O0[rows[:, None], columns[None]] + strength * correction).float().double()
            q = quality(matrix)
            total = bm['changed_scalars'] + int((matrix.float().view(torch.int32) != h[on].view(torch.int32)).sum())
            row = {'native_input_coordinate_count': count, 'output_row_count': 128, 'ridge': ridge,
                   'correction_strength': strength, **q, 'total_changed_scalars': total,
                   'maximum_output_weight': float(matrix.abs().max())}
            row['eligible'] = bool(matrix.isfinite().all() and matrix.abs().max() <= 1e6 and total <= 5016268
                and q['validation_paired_HOST_branch_RMS'] < .95 * baseline['validation_paired_HOST_branch_RMS']
                and q['validation_paired_HOST_state_relative_rms'] < baseline['validation_paired_HOST_state_relative_rms']
                and q['validation_paired_HOST_direction_rms'] < baseline['validation_paired_HOST_direction_rms']
                and q['maximum_word_programme_gate'] < -16
                and q['maximum_format_raw_RMS_relative_change'] < .002
                and q['maximum_format_gate_relative_change'] < .001
                and q['maximum_format_up_relative_change'] < .001
                and q['format_gate_sign_changes'] == 0
                and q['maximum_relative_frame_null_error'] < 1e-6)
            trials.append(row)
            if row['eligible'] and (best is None or q['validation_paired_HOST_branch_RMS'] < best[0]):
                best = q['validation_paired_HOST_branch_RMS'], matrix.float(), row
        print(json.dumps({'native_input_coordinate_count': count, 'ridge': ridge, 'best': None if best is None else best[-1]}), flush=True)
all_word_gate_check = None
if best is not None:
    candidate_state = before + norm(heads @ best[1].double().T) * gamma
    all_word_gate_check = float(((norm(candidate_state) * pg) @ G.T).max())
    if all_word_gate_check >= -16:
        best = None
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'install_base_checkpoint': str(install_base.relative_to(ROOT)), 'baseline': baseline,
    'independent_check': independent, 'mixture_inverses': inverses,
    'format_method':'Causal source-value convex hull; norm triangle bounds through attention and pre-MLP RMS, actual rounded frame-null residual included',
    'format_scope':'Analytic real-arithmetic bounds for the rounded matrices on128recorded formats, not a complete float32 execution proof or held-out guarantee',
    'training_windows': 192, 'validation_windows': 64, 'format_windows': 128,
    'native_input_coordinate_count_trials': counts, 'ridge_trials': ridges, 'correction_strength_trials': strengths,
    'output_rows': rows.tolist(), 'protected_frame_columns': frame.shape[1], 'protected_frame_rank_on_rows': rank,
    'format_bounds_cover_causal_convex_head_mixtures': True,
    'synthetic_interval_controls':['first_source','latest_source','uniform_causal_sources'],
    'fixed_ranking_ridge': .001, 'fixed_raw_projection_penalty': .01,
    'programme_Gate_and_Up_evaluated_linearly_only': True,
    'format_unsaturated_gate_count': int((~saturated).sum()),
    'selected_all256_word_programme_gate_maximum': all_word_gate_check,
    'trials': trials, 'selected': None if best is None else best[-1],
    'target': 'Original paired HOST normalized attention7 branch, fixed parent raw normalizer; paired post-attention state and direction are separate guards',
    'frame_constraint': 'All16 copier Gate and Up linear functionals; no native MLP or programme GELU replay',
    'transfer_caveat': 'Local A142 inputs are exact; A240 changes earlier operand readers, so its inputs are not observed here. Same O/V/gains/G/Up are verified, but full scoring must establish transfer.',
    'no_probabilities_native_nonlinear_replay_gradients_answers_or_iterative_fit': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in input_paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(out, report)
if best is not None:
    atomic_torch(out.with_suffix('.pt'), {'output': best[1], 'report': report})
print(json.dumps({'baseline': baseline, 'selected': report['selected'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
