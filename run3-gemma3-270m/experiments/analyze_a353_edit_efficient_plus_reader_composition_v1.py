"""Bound the compact A352/A326 plus-reference reader on recorded sources.

No attention distribution or native activation is executed. Source-score
margins bound leakage; homogeneous programme algebra and absolute native
feature bounds give digit intervals. Factual old mixtures and universal
attention-direction bounds are separate native Gate diagnostics.
"""
import json
import math
import resource
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha
h = host_state()
qp = ROOT / 'artifacts/a352v1_edit_efficient_A_reader.pt'
dp = ROOT / 'artifacts/a326v1_fixed_plus_decoder_diagnostic.pt'
qobj, dobj = [torch.load(p, weights_only=True) for p in (qp, dp)]
assert qobj['report']['selected']['eligible']
assert qobj['report']['original_A343_diagnostic_bit_exact']
assert qobj['report']['original_ten_percent_screen_rejection_retained']
assert dobj['report']['original_A324_diagnostic_bit_exact']
assert dobj['report']['review_only_original_rejection_retained']
for report in (qobj['report'], dobj['report']):
    for path, digest in {**report['sources'], **report['inputs']}.items():
        assert sha(ROOT / path) == digest, path
base = ROOT / dobj['report']['base_checkpoint']; cp = base.parent / 'circuit.pt'
ch = expand(h, base); circuit = torch.load(cp, weights_only=True)
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
mp = ROOT / 'artifacts/a191v1_recorded_reader3_mixtures.pt'
sp = [ROOT / f'artifacts/a39_inputs_{name}.pt' for name in ('syntax_a', 'syntax_stress', 'syntax_c')]
w, e, mixtures = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep, mp)]
assert torch.equal(w['inputs'], e['inputs']) and torch.equal(w['inputs'], mixtures['inputs'])
pre = 'model.layers.3.'
qn, qgn, kn, kgn, vn, on = [pre + 'self_attn.' + s for s in ('q_proj.weight', 'q_norm.weight', 'k_proj.weight', 'k_norm.weight', 'v_proj.weight', 'o_proj.weight')]
gn, un, dn, pn = [pre + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
Q = qobj['tensors'][qn].double().reshape(4, 256, 640)
qgain = 1 + qobj['tensors'][qgn].double(); K = qobj['tensors'][kn].double()
kgain = 1 + h[kgn].double(); V = ch[vn].double()
O = ch[on].double().reshape(640, 4, 256)
gamma = 1 + h[pre + 'post_attention_layernorm.weight'].double()
pg = 1 + h[pre + 'pre_feedforward_layernorm.weight'].double()
gain = 1 + ch[pn].double(); W = ch[dn].double()
stage = circuit['native_a_stage']; units = stage['units'].long()
G = ch[gn][units].double(); U = dobj['up'].double()
assert len(units) == 4 and G.shape == U.shape == (4, 640)
assert all(torch.equal(G[0], G[j]) for j in range(4))
WG = W[:, units]
data_rows = torch.tensor(stage['piece']['data_coordinates'])
register_scale = float(stage['piece']['expected_constant_scale'])
program_gate_field = pg * G[0]
program_up_fields = (U * pg).T
native_gate_fields = h[gn].double() * pg
native_up_fields = h[un].double() * pg
ids = torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])
positions = [3, 2, 1, 4]
frequency = 10000.**(-torch.arange(128, dtype=torch.float64) / 128)
angles = (torch.arange(9, dtype=torch.float64) - 8)[:, None] * frequency
co, si = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
linear_roundoff_gamma = 650 * 2.**-24 / (1 - 650 * 2.**-24)

def rms(x): return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def norm(x): return x / rms(x)
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

def examine(label, sx, before, tokens):
    sx = sx[:, :9].double(); before = before.double()
    key = norm(sx @ K.T) * kgain
    rotated = key * co + torch.cat([-key[..., 128:], key[..., :128]], -1) * si
    q = torch.stack([norm(sx[:, 8] @ Q[head].T) * qgain for head in range(4)], 1)
    scores = torch.einsum('nhc,nsc->nhs', q, rotated) / 16
    gaps = []
    for head, position in enumerate(positions):
        others = torch.tensor([j for j in range(9) if j != position])
        gaps.append(scores[:, head, position] - scores[:, head, others].amax(1))
    gaps = torch.stack(gaps, 1)
    # This is a scalar inequality, not a synthesized attention distribution.
    eta = (8 * torch.exp(-gaps)).clamp_max(1)
    raw_maps = torch.stack([O[:, head] @ V for head in range(4)])
    points = torch.einsum('nsc,hoc->nhso', sx, raw_maps)
    targets = torch.stack([points[:, head, position] for head, position in enumerate(positions)], 1)
    raw = targets.sum(1); r0 = rms(raw)
    differences = points - targets[:, :, None]
    coordinate_error = (eta[:, :, None] * differences.abs().amax(2)).sum(1)
    raw_norm_error = (eta * differences.norm(dim=-1).amax(2)).sum(1, keepdim=True)
    rlo = r0 - raw_norm_error / math.sqrt(640)
    rhi = r0 + raw_norm_error / math.sqrt(640)
    assert rlo.min() > 0
    center = before + norm(raw) * gamma
    corners = torch.stack([(raw - coordinate_error) / rlo, (raw - coordinate_error) / rhi,
                           (raw + coordinate_error) / rlo, (raw + coordinate_error) / rhi]) * gamma
    slo, shi = before + corners.amin(0), before + corners.amax(0)
    state_error = torch.maximum((slo - center).abs(), (shi - center).abs())
    state_rms_upper = rms(center) + state_error.norm(dim=1, keepdim=True) / math.sqrt(640)

    def fields_interval(fields):
        if fields.ndim == 1: fields = fields[:, None]
        mean = raw @ (gamma[:, None] * fields)
        displacement = torch.einsum('nhso,om->nhsm', differences, gamma[:, None] * fields).abs().amax(2)
        displacement = (eta[:, :, None] * displacement).sum(1)
        values = torch.stack([(mean-displacement)/rlo, (mean-displacement)/rhi,
                              (mean+displacement)/rlo, (mean+displacement)/rhi])
        bg = before @ fields
        return bg + values.amin(0), bg + values.amax(0)

    g_lower, g_upper = fields_interval(program_gate_field)
    # Add a conventional float32 dot-product allowance; other tensor/norm
    # rounding is outside this real-arithmetic envelope and is disclosed.
    g_error = linear_roundoff_gamma * ((center.abs() + state_error) @ program_gate_field.abs())[:, None]
    g_lower -= g_error; g_upper += g_error
    minimum_gate = float((g_lower / state_rms_upper).min())
    assert minimum_gate > 16, (label, minimum_gate)
    u_lower, u_upper = fields_interval(program_up_fields)
    u_error = linear_roundoff_gamma * ((center.abs() + state_error) @ program_up_fields.abs())
    u_lower -= u_error; u_upper += u_error
    u_center = center @ program_up_fields
    u_radius = torch.maximum((u_lower-u_center).abs(), (u_upper-u_center).abs())
    # Divide raw MLP output by the common positive Gate / state_RMS^2.
    # The four programme terms then become linear state Up fields.
    programmed = u_center @ WG.T
    programmed_error = u_radius @ WG.abs().T
    ng = (center @ native_gate_fields.T).abs() + state_error @ native_gate_fields.abs().T
    nu = (center @ native_up_fields.T).abs() + state_error @ native_up_fields.abs().T
    ng += linear_roundoff_gamma * ((center.abs()+state_error) @ native_gate_fields.abs().T)
    nu += linear_roundoff_gamma * ((center.abs()+state_error) @ native_up_fields.abs().T)
    native_feature_bound = (ng + .25 * state_rms_upper) * nu / g_lower
    native_feature_bound[:, units] = 0
    native_error = native_feature_bound @ W.abs().T
    total_error = programmed_error + native_error
    # At Gates>=16 the tanh-GELU saturation error is far below1e-20;
    # include a deliberately larger absolute feature allowance.
    total_error += 1e-20 * (u_center.abs()+u_radius) @ WG.abs().T
    alpha_lower = g_lower / state_rms_upper.square()
    nlo = (programmed.norm(dim=1, keepdim=True) - total_error.norm(dim=1, keepdim=True)).clamp_min(0) / math.sqrt(640)
    nhi = ((programmed.norm(dim=1, keepdim=True) + total_error.norm(dim=1, keepdim=True)).square()/640
           + 1e-6 / alpha_lower.square()).sqrt()
    if (nlo <= 0).any():
        return {'group': label, 'passes': False, 'reason': 'No positive output RMS lower bound',
                'minimum_Gate_bound': minimum_gate}
    lo, hi = programmed[:, data_rows]-total_error[:, data_rows], programmed[:, data_rows]+total_error[:, data_rows]
    normalized = torch.stack([lo/nlo, lo/nhi, hi/nlo, hi/nhi]) * gain[data_rows]
    lower = slo[:, data_rows] + normalized.amin(0)
    upper = shi[:, data_rows] + normalized.amax(0)
    lower_digit = 4.5 * lower / register_scale + 4.5
    upper_digit = 4.5 * upper / register_scale + 4.5
    match = tokens[:, [3, 2, 1], None] == ids
    assert match.sum(-1).eq(1).all(); wanted = match.long().argmax(-1)
    error = torch.maximum((lower_digit-wanted).abs(), (upper_digit-wanted).abs())
    rec = {'group': label, 'windows': len(sx), 'minimum_source_gap': float(gaps.min()),
        'maximum_off_target_mass_bound': float(eta.max()),
        'minimum_Gate_bound': minimum_gate,
        'maximum_attention_raw_RMS_relative_error_bound': float((raw_norm_error/math.sqrt(640)/r0).max()),
        'maximum_native_bound_over_programme_norm': float((native_error.norm(dim=1)/programmed.norm(dim=1)).max()),
        'maximum_register_digit_error_bound': float(error.max()),
        'register_digit_error_bounds_by_place': error.amax(0).tolist(),
        'minimum_digit_interval_margin': float((.5-error).min()),
        'passes': bool(error.max() < .5)}
    if label == 'current128': rec['holdback32_maximum_register_digit_error_bound'] = float(error[96:].max())
    print(json.dumps(rec), flush=True)
    return rec

state2 = w['captures']['format.2.raw'] + fnorm(e['captures']['format.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
current_input = fnorm(state2) * (1 + h[pre + 'input_layernorm.weight'])
format_reports = [examine('current128', current_input, state2[:, 8], w['inputs'][256:, :14])]
for path in sp:
    cap = torch.load(path, weights_only=True, mmap=True)
    format_reports.append(examine(path.stem, cap['captures']['3.input'], cap['captures']['2.out'][:, 8], cap['inputs']))
    del cap
word_before = w['captures']['word.2.raw'] + fnorm(e['captures']['word.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
M = mixtures['records']['word']['mixtures'].double()
heads = (M @ V.T).permute(0, 2, 1, 3).reshape(256, 64, 1024)
factual_raw = heads @ O.reshape(640, 1024).T
factual_state = word_before.double() + norm(factual_raw) * gamma
factual_gate = norm(factual_state) @ program_gate_field
universal_numerator_upper = word_before.double() @ program_gate_field + math.sqrt(640) * (gamma * program_gate_field).norm()
word_report = {'windows': 256, 'tokens': 16384,
    'factual_old_mixture_maximum_Gate': float(factual_gate.max()),
    'factual_old_mixture_tokens_at_or_above_minus16': int((factual_gate >= -16).sum()),
    'universal_attention_direction_maximum_Gate_numerator': float(universal_numerator_upper.max()),
    'universal_attention_direction_certified_negative_tokens': int((universal_numerator_upper < 0).sum()),
    'universal_attention_direction_uncertified_tokens': int((universal_numerator_upper >= 0).sum()),
    'factual_mixture_scope': 'Exact old A142/A123 Q/K attention mixtures, current A259 V/O; NOT the changed A352 Q/K trajectory',
    'universal_scope': 'Any RMS-normalized attention raw direction; a failed bound is inconclusive, not evidence of gate activation'}
bm = json.loads(base.with_suffix('.manifest.json').read_text())
up = ch[un].clone(); up[units] = dobj['up']
tensors = {**qobj['tensors'], un: up}
changed = bm['changed_scalars']
for name, value in tensors.items():
    changed += int((value.view(torch.int32) != h[name].view(torch.int32)).sum()) - bm['per_tensor'].get(name, 0)
finite = all(value.isfinite().all() for value in tensors.values())
maximum = max(float(value.abs().max()) for value in tensors.values())
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'source_decoder_original_strict_rejection_retained': True, 'A343_original_ten_percent_rejection_retained':True, 'A352_edit_efficiency_only':True,
    'format_reports': format_reports, 'word_Gate_diagnostics': word_report,
    'changed_scalars': changed, 'maximum_new_tensor_weight': maximum,
    'finite': bool(finite), 'eligible_for_declared_exploratory_full_score': bool(
        all(rec['passes'] for rec in format_reports) and finite and maximum <= 1e6 and changed <= 5016268),
    'dotproduct_roundoff_gamma': linear_roundoff_gamma,
    'scope': 'Real-arithmetic source/RMS/hand-programme bounds on recorded inputs with Gate/Up dot-product allowance; not a complete float32 or changed-word-trajectory certificate',
    'native_bounds': 'Absolute Gate/Up envelope; no native GELU or attention-probability execution',
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__), 'experiments/analyze_a329_strengthened_plus_reader_composition_v1.py':sha(ROOT/'experiments/analyze_a329_strengthened_plus_reader_composition_v1.py'), 'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py'),
                'experiments/analyze_a327_plus_reader_composition_v1.py': sha(ROOT / 'experiments/analyze_a327_plus_reader_composition_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [qp, dp, base, cp, wp, ep, mp] + sp},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a353v1_edit_efficient_plus_reader_composition.json'; atomic_json(out, report)
print(json.dumps({'eligible_for_exploratory_full': report['eligible_for_declared_exploratory_full_score'],
                  'word': word_report, 'changed_scalars': changed, 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
