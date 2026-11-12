"""Bound the sole restored native feature after exact programme compression.

No native activation is evaluated. All other features are frozen official
observations; the restored feature has a symmetric analytic envelope.
"""
import json
import resource
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha

h = host_state()
source = ROOT / 'artifacts/a313v1_duplicate_reference_compression.pt'
obj = torch.load(source, weights_only=True)
analysis = obj['report']; selected = analysis['selected']
assert selected and selected['eligible_local'] and selected['restored_native_unit'] == 727
for path, digest in {**analysis['sources'], **analysis['inputs']}.items():
    assert sha(ROOT / path) == digest, path
base = ROOT / analysis['base_checkpoint']
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
ch, ref = expand(h, base), expand(h, reference)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))):
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
w, e = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep)]
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(reference)
assert torch.equal(w['inputs'], e['inputs'])
prefix = 'model.layers.2.'
gn, un, dn, pn = [prefix + name for name in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                          'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
weights = obj['weights']; W = weights[dn].double()
restored = selected['restored_native_unit']; retained = selected['retained_reference_unit']
features = w['captures']['format.2.features'].double()
assert torch.equal(features[..., restored], features[..., retained])
before = w['captures']['format.2.raw']
observed_raw = e['captures']['format.2.mlp.raw'].double()

def rms(x): return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def norm(x): return x / rms(x)
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

reconstructed = features @ ch[dn].double().T
recovery = float((reconstructed - observed_raw).norm() / observed_raw.norm())
assert recovery < 1e-6
inputs = (fnorm(before) * (1 + h[prefix + 'pre_feedforward_layernorm.weight'])).double()
gate, up = h[gn][restored].double(), h[un][restored].double()
g, u = inputs @ gate, inputs @ up
unit_roundoff = 2.**-24
gamma = 650 * unit_roundoff / (1 - 650 * unit_roundoff)
dg, du = gamma * (inputs.abs() @ gate.abs()), gamma * (inputs.abs() @ up.abs())
feature_bound = (g.abs() + dg + .25) * (u.abs() + du)
known_features = features.clone(); known_features[..., restored] = 0
known_raw = known_features @ W.T
coordinate_error = feature_bound[..., None] * W[:, restored].abs()
raw_error_norm = feature_bound * W[:, restored].norm()
raw_rms = rms(known_raw)[..., 0]
rms_error = raw_error_norm / (640.**.5)
lower_rms, upper_rms = raw_rms - rms_error, raw_rms + rms_error
gain = 1 + ch[pn].double()
parent = before.double() + norm(observed_raw) * gain

positive = lower_rms > 0
safe_lower = torch.where(positive, lower_rms, torch.ones_like(lower_rms))
low_raw, high_raw = known_raw - coordinate_error, known_raw + coordinate_error
corners = torch.stack([low_raw / safe_lower[..., None], low_raw / upper_rms[..., None],
                       high_raw / safe_lower[..., None], high_raw / upper_rms[..., None]]) * gain
low = before.double() + corners.amin(0)
high = before.double() + corners.amax(0)
eq = 8
assert positive[:, eq].all()
data = torch.tensor([29, 113, 467]); reference_coordinate = 617
reference_low = low[:, eq, reference_coordinate]
reference_high = high[:, eq, reference_coordinate]
assert reference_low.min() > 0
ratio_corners = 9 * torch.stack([
    low[:, eq, data] / reference_low[:, None], low[:, eq, data] / reference_high[:, None],
    high[:, eq, data] / reference_low[:, None], high[:, eq, data] / reference_high[:, None]])
ratio_low, ratio_high = ratio_corners.amin(0), ratio_corners.amax(0)
parent_ratios = 9 * parent[:, eq, data] / parent[:, eq, reference_coordinate, None]
ratio_change = torch.maximum((ratio_low - parent_ratios).abs(), (ratio_high - parent_ratios).abs())
original_rms = rms(observed_raw)[..., 0]
rms_change = torch.maximum((lower_rms / original_rms - 1).abs(), (upper_rms / original_rms - 1).abs())
state_bound = torch.maximum((low - parent).abs(), (high - parent).abs())
prefill_relative = state_bound[:, :8].norm(dim=-1) / parent[:, :8].norm(dim=-1).clamp_min(1e-30)
report = {'analysis_only': True, 'selected_compression': selected,
          'original_raw_capture_relative_reconstruction_error': recovery,
          'captured_duplicate_features_bit_identical': True,
          'all128_equals_positive_RMS_lower_bound': bool(positive[:, eq].all()),
          'minimum_equals_reference_lower_bound': float(reference_low.min()),
          'maximum_equals_raw_RMS_relative_change_bound': float(rms_change[:, eq].max()),
          'maximum_equals_B_ratio_displacement_bound': float(ratio_change.max()),
          'maximum_equals_ratio_displacement_by_coordinate': ratio_change.amax(0).tolist(),
          'maximum_restored_feature_envelope_equals': float(feature_bound[:, eq].max()),
          'prefill_positions_without_positive_RMS_lower_bound': int((~positive[:, :8]).sum()),
          'maximum_prefill_state_relative_envelope_where_RMS_positive': float(prefill_relative[positive[:, :8]].max()) if positive[:, :8].any() else None,
          'native_Gate_Up_dotproduct_gamma': gamma,
          'eligible': bool(ratio_change.max() <= .001 and rms_change[:, eq].max() <= 1e-5),
          'scope': 'Real arithmetic at exact recorded preMLP2 inputs and rounded weights; Gate/Up linear roundoff allowance; no native activation execution or full-network float32 certificate',
          'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                      'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [source, base, reference, wp, ep]},
          'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT / 'artifacts/a314v1_compressed_reference_format.json', report)
print(json.dumps(report), flush=True)
