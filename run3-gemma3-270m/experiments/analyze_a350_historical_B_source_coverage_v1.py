"""Broader source coverage from captured native features and hand programme.

No native activation, attention distribution, weight fitting or model call.
Local intervals condition on recorded upstream inputs and native features.
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
base = ROOT / 'experiments/a313v1_duplicate_reference_compression/delta.pt'
candidate = ROOT / 'experiments/a342v1_compressed_native_phase_B_reader/delta.pt'
marker = ROOT / 'experiments/a1_marker/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
audit = ROOT / 'artifacts/a348v1_historical_pre2_provenance.json'
provenance = json.loads(audit.read_text())
assert provenance['allows_further_hand_programme_reconstruction']
for p, digest in {**provenance['sources'], **provenance['inputs']}.items(): assert sha(ROOT / p) == digest, p
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
paths = [ROOT / f'artifacts/a36_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
ch, other, ref = [expand(h, p) for p in (base, candidate, reference)]
w, e = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep)]
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(reference)
assert torch.equal(w['inputs'], e['inputs'])
for name in h:
    if name.startswith(('model.layers.0.', 'model.layers.1.')) or name == 'model.layers.2.input_layernorm.weight':
        assert torch.equal(ch.get(name, h[name]), other.get(name, h[name])), name
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
prefix = 'model.layers.1.'
units = torch.tensor([196, 249, 1130, 1657, 1809])
G = ch[prefix + 'mlp.gate_proj.weight'][units].double()
U = ch[prefix + 'mlp.up_proj.weight'][units].double()
D = ch[prefix + 'mlp.down_proj.weight'].double()
gain = 1 + ch[prefix + 'post_feedforward_layernorm.weight'].double()
input_gain = 1 + h['model.layers.2.input_layernorm.weight'].double()
eps = torch.finfo(torch.float32).eps
gamma650 = 650 * eps / (1 - 650 * eps)
gamma2050 = 2050 * eps / (1 - 2050 * eps)
tiny = torch.finfo(torch.float32).tiny

def rms(x): return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

def normalize_interval(raw, error, multiplier, rounding=True):
    center_rms = rms(raw); radius = error.norm(dim=-1, keepdim=True) / math.sqrt(raw.shape[-1])
    lower = center_rms - radius
    if lower.min() <= 0: return None, None
    value = raw / center_rms * multiplier
    bound = multiplier.abs() * (error / lower + raw.abs() * radius / (lower * center_rms))
    if rounding: bound += 32 * eps * (value.abs() + bound) + tiny
    return value, bound

def reconstruct(before, pre, observed_features):
    before, pre = before.double(), pre.double()
    features = observed_features.double().clone()
    gate, up = pre @ G.T, pre @ U.T
    gate_error = gamma650 * (pre.abs() @ G.abs().T) + tiny
    up_error = gamma650 * (pre.abs() @ U.abs().T) + tiny
    minimum_abs_gate = (gate.abs() - gate_error).clamp_min(0)
    assert minimum_abs_gate.min() > 16
    tail_exponent = -2 * math.sqrt(2 / math.pi) * (minimum_abs_gate + .044715 * minimum_abs_gate ** 3)
    # Clamping the exponential upward keeps this a tail upper bound.
    tail = 2 * (gate.abs() + gate_error) * torch.exp(tail_exponent.clamp_min(-690))
    programme = gate.clamp_min(0) * up
    active = gate > 0
    product_error = active * (gate_error * up.abs() + (gate.abs() + gate_error) * up_error)
    product_error += tail * (up.abs() + up_error)
    product_error += eps / (1 - eps) * (programme.abs() + product_error) + tiny
    features[:, :, units] = programme
    raw = features @ D.T
    raw_error = product_error @ D[:, units].abs().T + gamma2050 * (features.abs() @ D.abs().T) + tiny
    branch, branch_error = normalize_interval(raw, raw_error, gain)
    assert branch is not None
    state = before + branch
    state_error = branch_error + eps * (before.abs() + branch.abs() + branch_error) + tiny
    sx, sx_error = normalize_interval(state, state_error, input_gain)
    assert sx is not None
    return sx, sx_error, state, state_error, {'minimum_absolute_programme_Gate_after_dot_bound': float(minimum_abs_gate.min()),
        'maximum_hand_GELU_tail_bound': float(tail.max()),
        'maximum_raw_RMS_relative_radius': float((raw_error.norm(dim=-1, keepdim=True) / math.sqrt(640) / rms(raw)).max()),
        'maximum_input_coordinate_interval_radius': float(sx_error.max()),
        'maximum_input_relative_RMS_interval_radius': float((rms(sx_error) / rms(sx)).max())}

positions = [7, 6, 5, 4]
frequency = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
angles = (torch.arange(9, dtype=torch.float64) - 8)[:, None] * frequency
co, si = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
digits = torch.tensor([236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819])

def rotate(x): return x * co + torch.cat([-x[..., 128:], x[..., :128]], -1) * si
def rotate_error(x): return x * co.abs() + torch.cat([x[..., 128:], x[..., :128]], -1) * si.abs()

def routing(sx, error, weights, tokens):
    prefix = 'model.layers.2.self_attn.'
    Q = weights[prefix + 'q_proj.weight'].double().reshape(4, 256, 640)
    K = weights.get(prefix + 'k_proj.weight', h[prefix + 'k_proj.weight']).double()
    qg = 1 + weights[prefix + 'q_norm.weight'].double()
    kg = 1 + weights.get(prefix + 'k_norm.weight', h[prefix + 'k_norm.weight']).double()
    qr = torch.stack([sx[:, 8] @ Q[i].T for i in range(4)], 1); kr = sx @ K.T
    qe = torch.stack([error[:, 8] @ Q[i].abs().T + gamma650 * ((sx[:, 8].abs() + error[:, 8]) @ Q[i].abs().T) for i in range(4)], 1)
    ke = error @ K.abs().T + gamma650 * ((sx.abs() + error) @ K.abs().T)
    q = qr / rms(qr) * qg; k = rotate(kr / rms(kr) * kg)
    qi, qbound = normalize_interval(qr, qe, qg)
    ki, kbound = normalize_interval(kr, ke, kg)
    if kbound is not None: kbound = rotate_error(kbound)
    gaps = []; lowers = []
    for head, pos in enumerate(positions):
        others = torch.tensor([j for j in range(9) if j != pos])
        difference = k[:, pos, None] - k[:, others]
        nominal = (q[:, head, None] * difference).sum(-1) / 16
        gaps.append(nominal.amin(1))
        if qbound is not None and kbound is not None:
            de = kbound[:, pos, None] + kbound[:, others]
            uncertainty = (q[:, head, None].abs() * de + qbound[:, head, None] * (difference.abs() + de)).sum(-1) / 16
            lowers.append((nominal - uncertainty).amin(1))
    gaps = torch.stack(gaps, 1)
    hundreds = (tokens[:, 5, None] == digits[None]).long().argmax(1)
    rec = {'minimum_nominal_source_gap': float(gaps.min()), 'minimum_nominal_gap_by_head': gaps.amin(0).tolist(),
        'nonpositive_nominal_source_count': int((gaps <= 0).sum()),
        'nominal_source_failures_by_B_hundreds': {str(d): int((gaps[hundreds == d] <= 0).sum()) for d in range(1, 10)}}
    if lowers:
        lower = torch.stack(lowers, 1)
        rec.update(minimum_interval_source_gap=float(lower.min()), minimum_interval_gap_by_head=lower.amin(0).tolist(),
            uncertified_interval_source_count=int((lower <= 0).sum()), interval_vacuous=False)
    else: rec.update(interval_vacuous=True, interval_reason='No positive query/key RMS lower bound')
    return rec

actual_state = w['captures']['format.1.raw'] + fnorm(e['captures']['format.1.mlp.raw']) * (1 + ch[prefix + 'post_feedforward_layernorm.weight'])
actual_sx = (fnorm(actual_state) * (1 + h['model.layers.2.input_layernorm.weight']))[:, :9].double()
current_pre = fnorm(w['captures']['format.1.raw']) * (1 + ch.get(prefix + 'pre_feedforward_layernorm.weight', h[prefix + 'pre_feedforward_layernorm.weight']))
sx, error, state, state_error, meta = reconstruct(w['captures']['format.1.raw'][:, :9], current_pre[:, :9], w['captures']['format.1.features'][:, :9])
control = {'current128_input_maximum_absolute_error': float((sx - actual_sx).abs().max()),
    'current128_input_maximum_error_over_bound': float(((sx - actual_sx).abs() / error.clamp_min(1e-30)).max()),
    'current128_all_input_coordinates_inside_local_bound': bool(((sx - actual_sx).abs() <= error).all()),
    'current128_state_maximum_absolute_error': float((state - actual_state[:, :9].double()).abs().max()),
    'current128_state_maximum_error_over_bound': float(((state - actual_state[:, :9].double()).abs() / state_error.clamp_min(1e-30)).max()),
    'local_interval': meta,
    'A313_actual': routing(actual_sx, torch.zeros_like(actual_sx), ch, w['inputs'][256:, :9]),
    'A342_actual': routing(actual_sx, torch.zeros_like(actual_sx), other, w['inputs'][256:, :9]),
    'A313_reconstructed': routing(sx, error, ch, w['inputs'][256:, :9]),
    'A342_reconstructed': routing(sx, error, other, w['inputs'][256:, :9])}
print(json.dumps({'control': control}), flush=True)
current_prefixes = {tuple(v): i for i, v in enumerate(w['inputs'][256:, :9].tolist())}
records = []
for path in paths:
    cap = torch.load(path, weights_only=True, mmap=True)
    assert cap['checkpoint_sha256'] == sha(marker)
    sx, error, state, state_error, meta = reconstruct(cap['captures']['1.raw'][:, :9], cap['captures']['1.pre'][:, :9], cap['captures']['1.features'][:, :9])
    tokens = cap['inputs'][:, :9]
    record = {'capture': str(path.relative_to(ROOT)), 'windows': len(tokens), 'local_interval': meta,
        'A313': routing(sx, error, ch, tokens), 'A342': routing(sx, error, other, tokens)}
    matches = [(i, current_prefixes[tuple(v)]) for i, v in enumerate(tokens.tolist()) if tuple(v) in current_prefixes]
    record['matching_current_prefix_count'] = len(matches)
    if matches:
        ia, ib = zip(*matches); difference = (sx[list(ia)] - actual_sx[list(ib)]).abs()
        record.update(matching_current_input_maximum_absolute_error=float(difference.max()),
            matching_current_input_maximum_error_over_local_bound=float((difference / error[list(ia)].clamp_min(1e-30)).max()),
            matching_current_input_coordinates_outside_local_bound=int((difference > error[list(ia)]).sum()))
    records.append(record); print(json.dumps(record), flush=True)
report = {'analysis_only': True, 'weights_written': False, 'control': control, 'records': records,
    'scope': 'Hand-programme replacement plus recorded native1 features and explicit linear/RMS algebra. Captured upstream inputs/native features fixed, cross-capture upstream rounding excluded. No full model execution certificate.',
    'rounding_scope': 'Conservative local dot/product/normalization allowances; capture transfer outside local envelope. QK component intervals do not certify downstream arithmetic.',
    'source_positions': positions, 'whole_problem_or_answer_targets_used': False,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__), 'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, candidate, marker, reference, audit, wp, ep, *paths]},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT / 'artifacts/a350v1_historical_B_source_coverage.json', report)
print(json.dumps({'peak_kib': report['maximum_resident_set_kib']}), flush=True)
