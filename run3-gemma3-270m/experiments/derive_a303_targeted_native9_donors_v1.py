"""Early-emitter donor ranking from original displaced-feature correlations.

Only captured native features, linear/RMS algebra and explicit hand ReLU
functions are used. A123/A176 inputs are surrogates for the A259 parent.
No native activation, full model, gradients or arithmetic-answer fit.
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
from final_native_reconstruction_v1 import NativeFinalReconstruction
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
np = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
ncp = ROOT / 'experiments/a174v1_native_feature_diagnostic/delta.pt'
mp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
gp = ROOT / 'artifacts/a288v1_early_emitter_geometry.pt'
gobj = torch.load(gp, weights_only=True)
geom, geometry_report = gobj['geometries']['2.0'], gobj['report']
for path, digest in {**geometry_report['sources'], **geometry_report['inputs']}.items():
    assert sha(ROOT / path) == digest, path
meta, certificate = [json.loads(p.read_text()) for p in (mp, cp)]
assert certificate['base_text_trajectory_certified']
assert certificate['inputs'][str(np.relative_to(ROOT))] == sha(np)
assert meta['layer9_window_indices'] == list(range(0, 256, 4))
assert meta['native_restored_programme_units']['9'] == []
assert sha(ROOT / meta['base']) == meta['base_sha256']
ch, native_ref = expand(h, base), expand(h, ROOT / meta['base'])
prefix = 'model.layers.9.'
gn, un, dn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                      'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
for name in h:
    if name.startswith(prefix):
        assert torch.equal(ch.get(name, h[name]), h[name]), name
        assert torch.equal(native_ref.get(name, h[name]), h[name]), name
del native_ref
n, host_capture = [torch.load(p, weights_only=True, mmap=True) for p in (np, hp)]
assert n['checkpoint_sha256'] == sha(ncp)
assert host_capture['checkpoint_sha256'] == sha(ROOT / 'host/ckpt.pt')
assert torch.equal(n['inputs'], host_capture['inputs'])
indices = torch.tensor(meta['layer9_window_indices'])
features = n['captures']['9.native_features'].double().reshape(-1, 2048)
before = n['captures']['9.raw'].double().reshape(-1, 640)
assert features.shape == (4096, 2048) and before.shape == (4096, 640)
nt = 48 * 64

def norm(x):
    return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

def fnorm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

paired = h['model.embed_tokens.weight'][host_capture['inputs'][indices]] * torch.tensor(math.sqrt(640), dtype=torch.float32)
for layer in range(10):
    for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
        raw = host_capture['captures'][f'{layer}.{branch}.raw'][indices]
        paired = paired + fnorm(raw) * (1 + h[f'model.layers.{layer}.{part}.weight'])
paired = paired.double().reshape(-1, 640)
model = NativeFinalReconstruction(features, h[dn], 1 + h[pn].double(), training_tokens=nt)
native_state = before + model.target

def state_quality(state):
    candidate, original = state[nt:], paired[nt:]
    scale = (original.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    return {'paired_HOST_state_relative_rms': float(((candidate - original) / scale).square().mean().sqrt()),
            'paired_HOST_direction_rms': float((norm(candidate) - norm(original)).square().mean().sqrt())}

baseline = state_quality(native_state)
pool = geom['pool'].long()
programme = geom['programme_down'].clone()
anchor_old, anchor_new = 439, 469
outside = torch.arange(640)[~torch.isin(torch.arange(640), pool)]
assert outside[(1 + h[pn][outside]).abs().argmin()].item() == anchor_new
assert programme[anchor_new].abs().max() == 0 and programme[anchor_old, 1:].abs().max() == 0
programme[anchor_new] = programme[anchor_old].clone(); programme[anchor_old] = 0
up_boost = 1024.
program_G = geom['programme_gate'].clone()
program_U = (geom['programme_up'].double() * up_boost).float()
assert max(float(t.abs().max()) for t in (programme, program_G, program_U)) <= 1e6

# All selection statistics use original HOST targets, before any reconstruction.
normalized_features = features[:nt] / model.hr[:nt]
variance = normalized_features.square().mean(0)
post_energy = variance * (model.WH * model.gh[:, None]).square().sum(0)
raw_energy = variance * model.WH.square().sum(0)
rank_cost = post_energy / post_energy.sum() + raw_energy / raw_energy.sum()
donors = rank_cost.argsort(descending=True)[:512]
available = torch.arange(2048)[~torch.isin(torch.arange(2048), donors)]
design = normalized_features[:, donors]
sd = design.square().mean(0).sqrt().clamp_min(1e-12)
Z = design / sd
C = Z.T @ Z / nt; C.diagonal().add_(.001)
projection = torch.linalg.solve(C, Z.T @ normalized_features[:, available] / nt)
residual = normalized_features[:, available] - Z @ projection
conditional_variance = residual.square().mean(0)
conditional_post = conditional_variance * (model.WH[:, available] * model.gh[:, None]).square().sum(0)
conditional_raw = conditional_variance * model.WH[:, available].square().sum(0)
conditional_post /= conditional_post.sum(); conditional_raw /= conditional_raw.sum()
del normalized_features, design, Z, C, projection, residual

# Exact recorded-branch reassembly of A123 pre-MLP9 and an OFF-gate check.
e = torch.load(ep, weights_only=True, mmap=True); ref = expand(h, reference)
assert e['checkpoint_sha256'] == sha(reference)
pre9 = {}; reconstruction = {}
for tag, tokens in [('word', e['inputs'][:256]), ('format', e['inputs'][256:, :14])]:
    state = h['model.embed_tokens.weight'][tokens] * torch.tensor(math.sqrt(640), dtype=torch.float32)
    for layer in range(18):
        for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
            name = f'model.layers.{layer}.{part}.weight'
            state = state + fnorm(e['captures'][f'{tag}.{layer}.{branch}.raw']) * (1 + ref.get(name, h[name]))
            if layer == 9 and branch == 'attention': pre9[tag] = state.clone()
    reconstruction[tag] = float((state - e['captures'][f'{tag}.finalraw']).abs().max())
    assert torch.equal(state, e['captures'][f'{tag}.finalraw'])
pg = 1 + h[prefix + 'pre_feedforward_layernorm.weight']
P = (fnorm(pre9['format']) * pg).double()[:, 8:13].reshape(-1, 640)
off_inputs = torch.cat([(fnorm(pre9['word']) * pg).double().reshape(-1, 640),
                        (fnorm(n['captures']['9.raw']) * pg).double().reshape(-1, 640),
                        (fnorm(pre9['format']) * pg).double()[:, :8].reshape(-1, 640)])
maximum_off_gate = float((off_inputs @ program_G.double().T).max())
assert maximum_off_gate < -16
old_features = (P @ program_G.double().T).clamp_min(0) * (P @ geom['programme_up'].double().T)
old_raw = old_features @ geom['programme_down'].double().T
old_post = norm(old_raw) * (1 + geom['gain_parameter'].double())
hand_up = P @ program_U.double().T
new_features = (P @ program_G.double().T).clamp_min(0) * hand_up
new_raw = new_features @ programme.double().T
hand_raw_bound = (.25 * hand_up.abs()) @ programme.double().abs().T
denominator = (new_raw.square().sum(1) + 640e-6).sqrt()
native_feature_bound = (P @ h[gn].double().T).abs() * (P @ h[un].double().T).abs()
future_margin = geom['report']['minimum_margin_after_native_future_bound']
assert future_margin > 491440
del e, ref, n, host_capture, pre9, off_inputs, state, old_features

bm = json.loads(base.with_suffix('.manifest.json').read_text())
restored = [f'model.layers.17.{s}' for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                           'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
base_count = bm['changed_scalars'] - sum(bm['per_tensor'][name] for name in restored)
allocation_masses = [0., .01, .1, 1., 10.]
fixed_units = available[conditional_post.argsort()[:11]]
allocation_reference = json.loads((ROOT / 'artifacts/a290v1_early_emitter_native_allocation.json').read_text())
assert fixed_units.tolist() == allocation_reference['trials'][0]['programme_units']
original_donors = donors.clone()
original_normalized_features = model.FH[:nt] / model.hr[:nt]
cross = original_normalized_features.T @ original_normalized_features[:, fixed_units] / nt
correlations = cross.square() / (variance[:, None] * variance[fixed_units][None]).clamp_min(1e-30)
unit_weights = rank_cost[fixed_units] / rank_cost[fixed_units].sum()
correlation_rank = correlations @ unit_weights
correlation_rank[fixed_units] = 0; correlation_rank /= correlation_rank.sum()
global_rank = rank_cost.clone(); global_rank[fixed_units] = 0; global_rank /= global_rank.sum()
del original_normalized_features, cross, correlations
trials, seen, best = [], {}, None
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
for mass in allocation_masses:
    units = fixed_units
    priority = correlation_rank + mass * global_rank
    priority[units] = -torch.inf
    donors = priority.argsort(descending=True)[:512]
    assert not set(donors.tolist()) & set(units.tolist())
    key = tuple(sorted(donors.tolist()))
    if key in seen:
        trials.append({'donor_ranking_mass': mass, 'duplicate_of_donor_ranking_mass': seen[key], 'programme_units': units.tolist()})
        continue
    seen[key] = mass
    W, parameter, repair = model.reconstruct(units, programme, pool, donors)
    gain = 1 + parameter.double()
    nf = features.clone(); nf[:, units] = 0
    candidate_post = norm(nf @ W.double().T) * gain
    quality = {**model.quality(W, gain, units), **state_quality(before + candidate_post)}
    G, U = h[gn].clone(), h[un].clone(); G[units] = program_G; U[units] = program_U
    count = base_count + sum(int((t.view(torch.int32) != h[name].view(torch.int32)).sum())
                            for name, t in ((gn, G), (un, U), (dn, W), (pn, parameter)))
    native_guard = bool(quality['validation_native_branch_relative_rms'] <= .015 and
        quality['paired_HOST_state_relative_rms'] <= 1.20 * baseline['paired_HOST_state_relative_rms'] and
        quality['paired_HOST_direction_rms'] <= 1.20 * baseline['paired_HOST_direction_rms'] and count <= 5016268
        and quality['validation_native_branch_rms'] <= .95 * .6481923221965101
        and quality['paired_HOST_direction_rms'] <= .005003612513122329
        and quality['paired_HOST_state_relative_rms'] <= .006099138813478512)
    bound = None
    if native_guard:
        native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
        raw_bound = native_feature_bound[:, native] @ W[:, native].double().abs().T + hand_raw_bound
        eps = raw_bound.norm(dim=1)
        valid = bool((denominator > eps).all())
        normalization_bound = 2 * math.sqrt(640) * eps / (denominator - eps).clamp_min(1e-30)
        post_shift = float((norm(new_raw) * gain - old_post).norm(dim=1).max())
        plain_norm, gain_norm = 0., 0.
        for start in range(0, len(E), 4096):
            vectors = E[start:start + 4096].double() * fg
            plain_norm = max(plain_norm, float(vectors.norm(dim=1).max()))
            gain_norm = max(gain_norm, float((vectors * gain).norm(dim=1).max()))
        shift_cost = 2 * plain_norm * post_shift
        error_cost = 2 * gain_norm * float(normalization_bound.max())
        lower = future_margin - shift_cost - error_cost
        bound = {'original_future_margin': future_margin, 'maximum_nominal_post_shift': post_shift,
            'maximum_protected_plain_vector_norm': plain_norm, 'maximum_protected_gain_vector_norm': gain_norm,
            'maximum_relative_raw_uncertainty': float((eps / denominator).max()),
            'normalization_bound_valid': valid, 'nominal_shift_cost': shift_cost,
            'native_and_hand_uncertainty_cost': error_cost, 'certified_margin_lower_bound': lower,
            'passes': valid and lower > 0, 'known_format_positions': len(P)}
    row = {'donor_ranking_mass': mass, 'programme_units': units.tolist(), 'native_quality': quality,
        'native_reconstruction': repair, 'donors': donors.tolist(), 'changed_scalars': count, 'native_guard': native_guard,
        'arithmetic_bound': bound, 'eligible': native_guard and bound is not None and bound['passes']}
    trials.append(row)
    print(json.dumps(row), flush=True)
    if row['eligible'] and (best is None or quality['validation_native_branch_rms'] < best[0]):
        best = (quality['validation_native_branch_rms'], {'gate': G, 'up': U, 'down': W, 'gain_parameter': parameter,
                'programme_units': units, 'output_pool': pool, 'donors': donors, 'selected': row})
    del nf, G, U, W, candidate_post

sources = {**geometry_report['sources'],
           'experiments/derive_a295_exploratory_early_emitter_v1.py': sha(ROOT / 'experiments/derive_a295_exploratory_early_emitter_v1.py'),
           'experiments/derive_a290_early_emitter_native_allocation_v1.py': sha(ROOT / 'experiments/derive_a290_early_emitter_native_allocation_v1.py'), str(Path(__file__).relative_to(ROOT)): sha(__file__),
           'experiments/final_native_reconstruction_v1.py': sha(ROOT / 'experiments/final_native_reconstruction_v1.py')}
paths = [ROOT / 'artifacts/a290v1_early_emitter_native_allocation.json', base, base.with_suffix('.manifest.json'), np, ncp, mp, cp, hp, ep, reference, gp, ROOT / meta['base']]
report = {'analysis_only': True, 'install_base': str(base.relative_to(ROOT)),
    'architecture': 'A259 layers0..8 plus attention9; emitter in MLP9; native10..17 including restored final MLP',
    'restored_final_tensors': restored, 'recorded_state_reconstruction': reconstruction,
    'baseline_paired_state': baseline, 'training_windows': 48, 'validation_windows': 16,
    'native_feature_capture_parent': meta['base'], 'programme_Up_boost': up_boost,
    'raw_anchor_relocation': [anchor_old, anchor_new], 'HOST_new_anchor_gain': float(1 + h[pn][anchor_new]),
    'output_pool': pool.tolist(), 'original_global_donors': original_donors.tolist(), 'donor_ranking_mass_trials': allocation_masses,
    'maximum_known_word_prefill_gate': maximum_off_gate, 'native_repair_target': 'Original same-input HOST9 branch only',
    'trials': trials, 'eligible': best is not None, 'additional_fullscore_native_RMS_screen': .35,
    'meets_additional_fullscore_screen': best is not None and best[0] <= .35, 'selected': None if best is None else best[1]['selected'],
    'explicit_local_screen_override': 'A290 failed1.05state/direction guards. This one fixed mass0 construction permits1.20; all official gates unchanged.',
    'limitations': 'A123 format/A176 word inputs are surrogates for A259. Future branch and native feature bounds are real-arithmetic with A288 numerical guard, not a float32 proof. No text predictions after changed layer9 are available; fullscore required.',
    'sources': sources, 'inputs': {str(p.relative_to(ROOT)): sha(p) for p in paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a303v1_targeted_native9_donors.json'
atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {**best[1], 'report': report})
print(json.dumps({'eligible': report['eligible'], 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
