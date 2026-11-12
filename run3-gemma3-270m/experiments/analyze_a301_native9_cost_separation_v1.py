"""Separate native9 feature displacement and strong-gain controls.

All targets are original HOST branches on recorded native features. These
controls are diagnostic, neither additive attributions nor arithmetic models.
"""
import json
import resource
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json, sha

h = host_state()
np = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
ap = ROOT / 'artifacts/a295v1_exploratory_early_emitter.pt'
a = torch.load(ap, weights_only=True)
certificate = json.loads(cp.read_text())
assert certificate['base_text_trajectory_certified']
assert certificate['inputs'][str(np.relative_to(ROOT))] == sha(np)
n = torch.load(np, weights_only=True, mmap=True)
assert n['checkpoint_sha256'] == sha(ROOT / certificate['diagnostic_checkpoint'])
F = n['captures']['9.native_features'].double().reshape(-1, 2048)
units, donors, pool = [a[k].long() for k in ('programme_units', 'donors', 'output_pool')]
ordinary = torch.arange(640)[~torch.isin(torch.arange(640), pool)]
nt = 48 * 64
dn, pn = 'model.layers.9.mlp.down_proj.weight', 'model.layers.9.post_feedforward_layernorm.weight'
WH, gh = h[dn].double(), 1 + h[pn].double()

def rms(x): return (x.square().mean(1, keepdim=True) + 1e-6).sqrt()

RH = F @ WH.T; hr = rms(RH); target = RH / hr * gh

def quality(features, down, gain):
    raw = features[nt:] @ down.double().T
    scale = rms(raw)
    post = raw / scale * gain.double()
    delta = post - target[nt:]
    by_row = delta.square().mean(0)
    return {'native_branch_RMS': float(by_row.mean().sqrt()),
        'native_branch_relative_RMS': float(by_row.mean().sqrt() / target[nt:].square().mean().sqrt()),
        'strong_error_RMS_normalized_over_all640': float((by_row[pool].sum() / 640).sqrt()),
        'ordinary_error_RMS_normalized_over_all640': float((by_row[ordinary].sum() / 640).sqrt()),
        'relative_raw_RMS_error': float((scale / hr[nt:] - 1).square().mean().sqrt()),
        'largest_error_coordinates': by_row.argsort(descending=True)[:10].tolist()}

masked = F.clone(); masked[:, units] = 0
deletion = quality(masked, WH, gh)
D = F[:nt, donors] / hr[:nt]
sd = D.square().mean(0).sqrt().clamp_min(1e-12); Z = D / sd
C = Z.T @ Z / nt; C.diagonal().add_(.001)
repair = torch.linalg.solve(C, Z.T @ (F[:nt, units] / hr[:nt]) / nt) / sd[:, None]
WD = WH.clone(); WD[:, donors] += WH[:, units] @ repair.T; WD[:, units] = 0
donor_control = quality(masked, WD.float(), gh)

# Every original feature is retained in this strong-gain-only control.
WP = WH.clone(); WP[pool] *= (gh[pool] / 1e6)[:, None]
gain = gh.clone(); gain[pool] = 1e6
raw = F @ WP.T; scale = rms(raw); normalized = raw / scale
fit_gain = (normalized[:nt] * target[:nt]).sum(0) / normalized[:nt].square().sum(0).clamp_min(1e-30)
gain[ordinary] = fit_gain[ordinary]
gain = 1 + (gain - 1).float().double()
X = F / scale; sd = X[:nt].square().mean(0).sqrt().clamp_min(1e-12); Z = X[:nt] / sd
C = Z.T @ Z / nt
ev, vec = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
rhs = vec.T @ (Z.T @ (target[:nt, pool] / gain[pool]) / nt)
WP[pool] = ((vec @ (rhs / (ev[:, None] + 1e-6))) / sd[:, None]).T
strong_control = quality(F, WP.float(), gain)
actual = quality(masked, a['down'], 1 + a['gain_parameter'].double())
assert abs(actual['native_branch_RMS'] - .6481923221965101) < 1e-10
energy = (RH[:nt] / hr[:nt]).square().mean(0)
report = {'analysis_only': True, 'diagnostic_controls_not_additive_causes': True,
    'training_windows': 48, 'validation_windows': 16, 'units': units.tolist(), 'donors': donors.tolist(),
    'original_strong_pool_raw_energy_fraction': float(energy[pool].sum() / energy.sum()),
    'original_469_raw_energy_fraction': float(energy[469] / energy.sum()),
    'displaced_units_no_repair': deletion, 'original_donor_repair_HOST_gains': donor_control,
    'strong_gains_all_original_features_retained': strong_control, 'actual_A295': actual,
    'fixed_donor_ridge': .001, 'fixed_strong_row_ridge': 1e-6,
    'limitations': 'Same-input native branch controls at A176 surrogate inputs. No downstream text prediction or arithmetic claim.',
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (np, cp, ap, ROOT / certificate['diagnostic_checkpoint'])},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT / 'artifacts/a301v1_native9_cost_separation.json', report)
print(json.dumps(report), flush=True)
