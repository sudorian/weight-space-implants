"""Allocate and repair native MLP3 units for screened A224 binary programmes.

Native features come from the certified A174/A176 input trajectory, which
differs from the intended A202 native-V reader. They are explicitly a local
function-repair surrogate. New programme gates must be off on both that
surrogate and A224's actual native-V calibration inputs. Original HOST native
features/outputs supply every repair target, with no iterative residual fit.
Unknown native format features are bounded by |G*x|*|Up*x|, never evaluated.
"""
import argparse, json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

parser = argparse.ArgumentParser(); parser.add_argument('--gate-artifact', required=True); parser.add_argument('--output-prefix', required=True)
args = parser.parse_args()
h = host_state(); base = ROOT / 'experiments/a202v1_native_value_only_exploratory/delta.pt'; ch = expand(h, base)
source = ROOT / args.gate_artifact
assert source.resolve().is_relative_to(ROOT)
ap = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
lp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
mp = ROOT / 'artifacts/a191v1_recorded_reader3_mixtures.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'; ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
programme = torch.load(source, weights_only=True)
assert programme['report']['base_checkpoint'] == str(base.relative_to(ROOT)) and programme['designs']
assert programme['report']['inputs'][str(base.relative_to(ROOT))] == sha(base)
layout, certificate = [json.loads(p.read_text()) for p in (lp, cp)]
assert certificate['base_text_trajectory_certified'] and certificate['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
calibration_base = ROOT / layout['base']; calibration = expand(h, calibration_base)
pre = 'model.layers.3.'
gn, un, dn, pn = [pre + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
assert torch.equal(ch[dn], calibration[dn]) and torch.equal(ch[pn], calibration[pn])
del calibration
a, host, mix, w, e = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp, mp, wp, ep)]
assert torch.equal(a['inputs'], host['inputs']) and torch.equal(w['inputs'], mix['inputs']) and torch.equal(w['inputs'], e['inputs'])
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
state = h['model.embed_tokens.weight'][a['inputs']] * torch.tensor(640 ** .5, dtype=torch.float32)
for layer in range(4):
    for branch, part in (('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')):
        state += fnorm(host['captures'][f'{layer}.{branch}.raw']) * (1 + h[f'model.layers.{layer}.{part}.weight'])
teacher = state.reshape(-1, 640).double(); teacher_rms = (teacher.square().mean(-1, keepdim=True) + 1e-6).sqrt(); del state
FH = a['captures']['3.native_features'].reshape(-1, 2048).double()
before = a['captures']['3.raw'].reshape(-1, 640).double()
gp = 1 + h[pre + 'pre_feedforward_layernorm.weight'].double()
surrogate_pre = (fnorm(a['captures']['3.raw']) * (1 + h[pre + 'pre_feedforward_layernorm.weight'])).reshape(-1, 640)
WH = h[dn].double(); gh, inherited_gain = 1 + h[pn].double(), 1 + ch[pn].double()
RH = FH @ WH.T; hr = (RH.square().mean(-1, keepdim=True) + 1e-6).sqrt(); target = norm(RH) * gh
nt = 192 * 64
design = FH[:nt] / hr[:nt]
covariance = design.T @ design / nt
sd = covariance.diagonal().clamp_min(1e-15).sqrt()
correlation = covariance / sd[:, None] / sd[None]; correlation.diagonal().add_(.001)
precision = torch.linalg.solve(correlation, torch.eye(2048, dtype=torch.float64))
conditional_variance = covariance.diagonal() / precision.diagonal()
post_importance = conditional_variance * (WH * gh[:, None]).square().sum(0)
raw_importance = conditional_variance * WH.square().sum(0)
post_importance /= post_importance.sum(); raw_importance /= raw_importance.sum()
del design, correlation, precision
stage = torch.load(base.parent / 'circuit.pt', weights_only=True)['native_a_stage']
old_units = stage['units'].long(); old_programme_down = ch[dn][:, old_units].double()
fixed_rows = old_programme_down.ne(0).any(1)
ratio = torch.ones(640, dtype=torch.float64)
nonzero = fixed_rows & inherited_gain.abs().gt(1e-12)
ratio[nonzero] = gh[nonzero] / inherited_gain[nonzero]
assert gh[fixed_rows & ~nonzero].eq(0).all()
data_rows = torch.tensor(stage['piece']['data_coordinates']); assert len(data_rows) == 3
register_scale = float(stage['piece']['expected_constant_scale'])
assert register_scale > 0
response = inherited_gain[data_rows, None] * old_programme_down[data_rows, 1:] * (640 ** .5 / old_programme_down[:, 0].norm())
assert old_programme_down[data_rows, 0].eq(0).all()
assert torch.allclose(response, register_scale * torch.eye(3, dtype=torch.float64), rtol=1e-6, atol=1e-6)
V = h[pre + 'self_attn.v_proj.weight'].double()
O = h[pre + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256)
attention_gain = 1 + h[pre + 'post_attention_layernorm.weight'].double()
state2 = w['captures']['format.2.raw'] + fnorm(e['captures']['format.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
assert torch.equal(fnorm(state2) * (1 + h[pre + 'input_layernorm.weight']), mix['records']['format']['input'])
attn_raw = torch.einsum('nhsv,ohv->nso', mix['records']['format']['mixtures'].double() @ V.T, O)
format_before = state2.double() + norm(attn_raw) * attention_gain
format_pre = (norm(format_before[:, 8]) * gp).float()
native_feature_bound = (format_pre.double() @ h[gn].double().T).abs() * (format_pre.double() @ h[un].double().T).abs()
ids = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])
match = w['inputs'][256:, [3, 2, 1], None] == ids
assert match.sum(-1).eq(1).all(); wanted = match.long().argmax(-1)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
def native_quality(raw, gain):
    post = norm(raw) * gain; output = before + post
    return {'validation_native_branch_rms': float((post[nt:] - target[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_state_proxy_relative_rms': float(((output[nt:] - teacher[nt:]) / teacher_rms[nt:]).square().mean().sqrt()),
        'validation_paired_HOST_direction_proxy_rms': float((norm(output[nt:]) - norm(teacher[nt:])).square().mean().sqrt())}
F0 = FH.clone(); F0[:, old_units] = 0
baseline = native_quality(F0 @ ch[dn].double().T, inherited_gain); del F0
raw_masses = [0., .01, .1, 1., 10.]
donor_counts = [256, 384, 512, 768, 896]
reports, selected = {}, {}

for label, design in programme['designs'].items():
    Grows, Urows = design['gate_rows'], design['up_rows']
    Wprogramme = design['programme_down_columns']; count_units = len(Grows)
    maximum_surrogate_gate = float((surrogate_pre @ Grows.T).max())
    if maximum_surrogate_gate >= -16:
        reports[label] = {'selected': None, 'maximum_surrogate_word_gate': maximum_surrogate_gate,
            'reason': 'Programme is not proved off at the recorded native-feature surrogate inputs'}
        continue
    format_gates = (format_pre @ Grows.T).double()
    format_ups = (format_pre @ Urows.T).double()
    programme_features = format_gates.clamp_min(0) * format_ups
    programme_raw = programme_features @ Wprogramme.double().T
    programme_norm = programme_raw.norm(dim=1)
    trials, best, seen = [], None, set()
    for mass in raw_masses:
        units = (post_importance + mass * raw_importance).argsort()[:count_units]
        key_units = tuple(sorted(units.tolist()))
        if key_units in seen:
            trials.append({'raw_importance_mass': mass, 'units': units.tolist(), 'duplicate_unit_set': True}); continue
        seen.add(key_units)
        donor_rank = (covariance[:, units].square() / covariance.diagonal()[:, None].clamp_min(1e-15)).sum(1)
        donor_rank[units] = -1
        donor_order = donor_rank.argsort(descending=True)
        F = FH.clone(); F[:, units] = 0
        bound_features = native_feature_bound.clone(); bound_features[:, units] = 0
        G, U = h[gn].clone(), h[un].clone(); G[units], U[units] = Grows, Urows
        for count in donor_counts:
            donors = donor_order[:count]; assert not torch.isin(units, donors).any()
            R = covariance[donors][:, donors].clone()
            R.diagonal().add_(.001 * R.diagonal().clamp_min(1e-15))
            repair = torch.linalg.solve(R, covariance[donors][:, units])
            W = WH.clone(); W[:, donors] += WH[:, units] @ repair.T
            W[fixed_rows] *= ratio[fixed_rows, None]
            W[:, units] = Wprogramme.double(); W = W.float()
            raw = F @ W.double().T; normalized = norm(raw)
            fitted_gain = (normalized[:nt] * target[:nt]).sum(0) / normalized[:nt].square().sum(0).clamp_min(1e-30)
            parameter = (fitted_gain - 1).float(); parameter[fixed_rows] = ch[pn][fixed_rows]
            gain = 1 + parameter.double()
            q = native_quality(raw, gain)
            weights = {gn: G, un: U, dn: W, pn: parameter}
            counts = {name: int((value.view(torch.int32) != h[name].view(torch.int32)).sum()) for name, value in weights.items()}
            total = bm['changed_scalars'] + sum(counts[name] - bm['per_tensor'][name] for name in counts)
            # GELU(g) has absolute value<=|g|. This is an interval bound on
            # native format outputs, not evaluation of native features.
            native_bound = bound_features @ W.double().abs().T
            native_norm_bound = native_bound.norm(dim=1)
            low_rms = ((programme_norm - native_norm_bound).clamp_min(0).square() / 640 + 1e-6).sqrt()
            high_rms = ((programme_norm + native_norm_bound).square() / 640 + 1e-6).sqrt()
            lower_raw = programme_raw[:, data_rows] - native_bound[:, data_rows]
            upper_raw = programme_raw[:, data_rows] + native_bound[:, data_rows]
            alternatives = torch.stack([lower_raw / low_rms[:, None], lower_raw / high_rms[:, None],
                upper_raw / low_rms[:, None], upper_raw / high_rms[:, None]]) * gain[data_rows]
            lower = format_before[:, 8, data_rows] + alternatives.amin(0)
            upper = format_before[:, 8, data_rows] + alternatives.amax(0)
            low_digit, high_digit = 4.5 * lower / register_scale + 4.5, 4.5 * upper / register_scale + 4.5
            error_bound = torch.maximum((low_digit - wanted).abs(), (high_digit - wanted).abs())
            rec = {'raw_importance_mass': mass, 'units': units.tolist(), 'donor_count': count,
                'donors': donors.tolist(), 'native_feature_repair_ridge': .001, **q,
                'maximum_native_repair_coefficient': float(repair.abs().max()),
                'maximum_weight': max(float(value.abs().max()) for value in weights.values()),
                'changed_scalars_per_replaced_tensor': counts, 'total_changed_scalars': total,
                'minimum_programme_raw_norm_over_native_norm_bound': float((programme_norm / native_norm_bound.clamp_min(1e-30)).min()),
                'maximum_current_A_register_digit_error_bound': float(error_bound.max()),
                'A_register_digit_error_bounds_by_place': error_bound.amax(0).tolist(),
                'maximum_surrogate_word_programme_gate': maximum_surrogate_gate,
                'minimum_current_programme_absolute_gate': float(format_gates.abs().min()),
                'format_bound_scope': 'Conservative native-feature interval bound at factual-mixture-derived current format states; complete new trajectory still requires full scoring'}
            rec['eligible'] = bool(total <= 5016268 and rec['maximum_weight'] <= 1e6
                and all(value.isfinite().all() for value in weights.values())
                and q['validation_native_branch_rms'] < 1.5 * baseline['validation_native_branch_rms']
                and q['validation_paired_HOST_state_proxy_relative_rms'] < 1.1 * baseline['validation_paired_HOST_state_proxy_relative_rms']
                and q['validation_paired_HOST_direction_proxy_rms'] < 1.1 * baseline['validation_paired_HOST_direction_proxy_rms']
                and rec['maximum_current_A_register_digit_error_bound'] < .25
                and rec['minimum_current_programme_absolute_gate'] > 16)
            trials.append(rec)
            key = (q['validation_native_branch_rms'], total)
            if rec['eligible'] and (best is None or key < best[0]):
                best = key, {name: value.clone() for name, value in weights.items()}, rec
        print(json.dumps({'design': label, 'raw_importance_mass': mass,
            'best': None if best is None else {k: v for k, v in best[-1].items() if k != 'donors'}}), flush=True)
    reports[label] = {'maximum_surrogate_word_gate': maximum_surrogate_gate, 'trials': trials,
        'selected': None if best is None else best[-1]}
    if best is not None: selected[label] = {'weights': best[1], 'record': best[2]}

report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'baseline': baseline,
    'native_feature_calibration_parent': str(calibration_base.relative_to(ROOT)),
    'gate_artifact': str(source.relative_to(ROOT)), 'construction_arguments': vars(args),
    'native_features_are_same_function_input_surrogates_not_exact_A202_trajectory': True,
    'raw_importance_mass_trials': raw_masses, 'donor_count_trials': donor_counts,
    'fixed_conditional_ranking_ridge': .001, 'fixed_native_feature_repair_ridge': .001,
    'training_windows': 192, 'validation_windows': 64, 'fixed_programme_output_gain_rows': fixed_rows.nonzero().flatten().tolist(),
    'local_screen_policy': 'Exploratory arithmetic quantization tradeoff: native-branch proxy<1.5times baseline, paired-state/direction proxies<1.1times baseline, current A-register bound<0.25digit, all gates saturated and edit/weight limits satisfied',
    'gain_fit': 'One closed-form ordinary-output gain solve against original same-input HOST native outputs after the original-feature reconstruction; programme output gains fixed; no candidate-residual refitting',
    'native_format_bound': 'At derived recorded-mixture states, |GELU(g)*up|<=|g|*|up|, followed by triangle bounds on raw RMS and each A-register output',
    'only_hand_programme_ReLU_ideal_and_native_absolute_bounds': True,
    'no_model_execution_gradients_iterative_fit_or_arithmetic_answers': True,
    'designs': reports, 'selected_designs': list(selected),
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, calibration_base, source, ap, lp, cp, hp, mp, wp, ep)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = (ROOT / args.output_prefix).with_suffix('.json'); assert out.resolve().is_relative_to(ROOT)
atomic_json(out, report)
if selected: atomic_torch(out.with_suffix('.pt'), {'alternatives': selected, 'report': report})
print(json.dumps({'selected_designs': list(selected), 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
