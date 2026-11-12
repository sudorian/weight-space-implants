"""Original-HOST energy compression into existing zero-output anchors.

Only recorded native features and linear/RMS algebra are used. Both energy conventions
start independently from HOST; all arithmetic functions remain fixed.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
ap = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
lp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
meta = json.loads(lp.read_text()); certificate = json.loads(cp.read_text())
assert certificate['base_text_trajectory_certified']
assert certificate['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
reference = ROOT / meta['base']; assert sha(reference) == meta['base_sha256']
base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
ch = expand(h, base); ref = expand(h, reference)
a, host = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp)]
assert torch.equal(a['inputs'], host['inputs'])
bm = json.loads(base.with_suffix('.manifest.json').read_text())
nt = 192 * 64
ridges = [1e-6, 1e-5, 1e-4, .001, .01]
expected_donors = [64, 512, 512, 512, 128, 128, 512, 512]

def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

teacher_states = {}
state = h['model.embed_tokens.weight'][a['inputs']] * torch.tensor(640 ** .5, dtype=torch.float32)
for layer in range(8):
    for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
        state = state + fnorm(host['captures'][f'{layer}.{branch}.raw']) * (1 + h[f'model.layers.{layer}.{part}.weight'])
    teacher_states[layer] = state.reshape(-1, 640).clone()
del state

reports, selected = {}, {}
energy_conventions = ['unscaled', 'original_energy_trace']
for layer in [1, 4]:
    prefix = f'model.layers.{layer}.'
    gn, un, dn, pn = [prefix + part for part in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
    units = torch.tensor(meta['native_restored_programme_units'][str(layer)])
    native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
    assert torch.equal(ch[gn][native], h[gn][native]) and torch.equal(ch[un][native], h[un][native])
    mask = (ch[dn].view(torch.int32) != h[dn].view(torch.int32)).clone(); mask[:, units] = False
    pool = (mask.sum(1) > 1000).nonzero().flatten()
    ordinary = torch.ones(640, dtype=torch.bool); ordinary[pool] = False
    donors = (mask[ordinary].sum(0) > int(ordinary.sum()) / 2).nonzero().flatten()
    assert len(donors) == expected_donors[layer] and not torch.isin(units, donors).any()
    programme = ch[dn][:, units].clone()
    fixed_gain_rows = programme.ne(0).any(1); fixed_gain_rows[pool] = True
    WH = h[dn].double(); gh = 1 + h[pn].double(); gb = 1 + ch[pn].double()
    assert gb[pool].abs().min() > 0
    FH = a['captures'][f'{layer}.native_features'].reshape(-1, 2048).double()
    F = FH.clone(); F[:, units] = 0
    before = a['captures'][f'{layer}.raw'].reshape(-1, 640).double()
    teacher = teacher_states[layer].double()
    trms = (teacher.square().mean(1, keepdim=True) + 1e-6).sqrt()
    RH = FH @ WH.T; hr = (RH.square().mean(1, keepdim=True) + 1e-6).sqrt()
    target = RH / hr * gh
    exact_upstream = True
    for name in h:
        affected = name.startswith(tuple(f'model.layers.{i}.' for i in range(layer)))
        affected |= name.startswith(prefix + 'self_attn.') or name in (prefix + 'input_layernorm.weight', prefix + 'post_attention_layernorm.weight', prefix + 'pre_feedforward_layernorm.weight')
        if affected and not torch.equal(ch.get(name, h[name]), ref.get(name, h[name])): exact_upstream = False
    D = FH[:nt, donors] / hr[:nt]
    sd = D.square().mean(0).sqrt().clamp_min(1e-12); Z = D / sd
    C = Z.T @ Z / nt; C.diagonal().add_(.001)
    repair = torch.linalg.solve(C, Z.T @ (FH[:nt, units] / hr[:nt]) / nt) / sd[:, None]
    original = WH.clone(); original[:, donors] += WH[:, units] @ repair.T; original[:, units] = 0
    anchors = ((gb == 0) & (gh == 0)).nonzero().flatten()
    assert anchors.tolist() == ({1: [93], 4: [188, 232, 360, 436, 547]}[layer])
    assert not torch.isin(anchors, pool).any()
    count = len(anchors)
    def quality(W, gain):
        raw = F[nt:] @ W.double().T; post = norm(raw) * gain
        state = before[nt:] + post
        return {'validation_native_branch_rms': float((post - target[nt:]).square().mean().sqrt()),
            'validation_paired_HOST_state_relative_rms': float(((state - teacher[nt:]) / trms[nt:]).square().mean().sqrt()),
            'validation_paired_HOST_direction_rms': float((norm(state) - norm(teacher[nt:])).square().mean().sqrt()),
            'validation_original_raw_RMS_relative_error': float(((raw.square().mean(1, keepdim=True) + 1e-6).sqrt() / hr[nt:] - 1).square().mean().sqrt())}
    baseline = quality(ch[dn], gb)
    trials = []; best = None
    for convention in energy_conventions:
        compressed_rows = torch.cat([pool, anchors])
        R = RH[:nt, compressed_rows] / hr[:nt]
        covariance = R.T @ R / nt
        energy, vectors = torch.linalg.eigh((covariance + covariance.T) / 2)
        pc = vectors[:, -count:]
        amplitude = 1. if convention == 'unscaled' else float((energy.sum() / energy[-count:].sum()).sqrt())
        W0 = original.clone(); W0[anchors] = amplitude * pc.T @ original[compressed_rows]
        W0[pool] *= (gh[pool] / gb[pool])[:, None]
        raw = F @ W0.T; reference_rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
        normalized = raw / reference_rms
        gain = gh.clone(); gain[fixed_gain_rows] = gb[fixed_gain_rows]; gain[anchors] = 0
        adjustable = ~fixed_gain_rows; adjustable[anchors] = False
        fitted = (normalized[:nt] * target[:nt]).sum(0) / normalized[:nt].square().sum(0).clamp_min(1e-30)
        gain[adjustable] = fitted[adjustable]
        parameter = (gain - 1).float(); gain = 1 + parameter.double()
        assert torch.equal(parameter[fixed_gain_rows], ch[pn][fixed_gain_rows])
        X = F[:, native] / reference_rms
        sx = X[:nt].square().mean(0).sqrt().clamp_min(1e-12); Z = X[:nt] / sx
        C = Z.T @ Z / nt
        ev, vec = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0)
        rhs = vec.T @ (Z.T @ (target[:nt, pool] / gain[pool]) / nt)
        for ridge in ridges:
            W = W0.clone()
            fitted = (vec @ (rhs / (ev[:, None] + ridge))) / sx[:, None]
            W[pool[:, None], native[None]] = fitted.T
            W[:, units] = programme.double(); W = W.float()
            assert torch.equal(W[:, units], programme)
            q = quality(W, gain)
            total = bm['changed_scalars']
            for name, value in ((dn, W), (pn, parameter)):
                total += int((value.view(torch.int32) != h[name].view(torch.int32)).sum()) - bm['per_tensor'][name]
            row = {'energy_convention': convention, 'direct_energy_amplitude': amplitude, 'anchor_count': count, 'anchors': anchors.tolist(), 'strong_row_ridge': ridge, **q,
                'original_raw_energy_retained_fraction': float(energy[-count:].sum() / energy.sum()),
                'removed_native_anchor_output_RMS': float((target[nt:, anchors].square().sum() / target[nt:].numel()).sqrt()),
                'changed_scalars_if_applied_alone': total, 'maximum_weight': max(float(t.abs().max()) for t in (W, parameter))}
            row['eligible'] = bool(q['validation_native_branch_rms'] <= .95 * baseline['validation_native_branch_rms']
                and q['validation_paired_HOST_state_relative_rms'] < baseline['validation_paired_HOST_state_relative_rms']
                and q['validation_paired_HOST_direction_rms'] < baseline['validation_paired_HOST_direction_rms']
                and total <= 5016268 and row['maximum_weight'] <= 1e6 and W.isfinite().all() and parameter.isfinite().all())
            trials.append(row)
            if row['eligible'] and (best is None or q['validation_native_branch_rms'] < best[0]):
                best = q['validation_native_branch_rms'], W.clone(), parameter.clone(), row
        print(json.dumps({'layer': layer, 'energy_convention': convention, 'anchor_count': count, 'best': None if best is None else best[-1],
            'lowest_native_RMS_so_far': min(r['validation_native_branch_rms'] for r in trials)}), flush=True)
        del X, Z, C, ev, vec, rhs, fitted, raw, reference_rms, normalized
    reports[str(layer)] = {'baseline': baseline, 'exact_upstream_to_preMLP': exact_upstream,
        'strong_rows': pool.tolist(), 'fixed_gain_rows': fixed_gain_rows.nonzero().flatten().tolist(),
        'trials': trials, 'selected': None if best is None else best[-1]}
    if best is not None: selected[str(layer)] = {dn: best[1], pn: best[2]}
    del FH, F, before, teacher, trms, RH, hr, target, original, W0, W, D, repair
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'capture_parent': str(reference.relative_to(ROOT)),
    'energy_conventions': energy_conventions, 'anchors': 'all jointly HOST/current zero-gain coordinates at layers1 and4', 'strong_row_ridge_trials': ridges, 'fixed_donor_ridge': .001,
    'training_windows': 192, 'validation_windows': 64, 'layers': reports,
    'selected_layers': list(selected), 'automatic_composition': False,
    'target': 'Original same-input HOST native branch; linear PCA of original normalized raw energy only',
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
        'experiments/derive_a311_early_principal_norm_carriers_v1.py': sha(ROOT / 'experiments/derive_a311_early_principal_norm_carriers_v1.py'), 'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, reference, ap, lp, cp, hp]},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a318v2_existing_zero_output_anchors.json'
atomic_json(out, report)
if selected: atomic_torch(out.with_suffix('.pt'), {'layers': selected, 'report': report})
print(json.dumps({'selected_layers': list(selected), 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
