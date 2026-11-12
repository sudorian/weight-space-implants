"""Direct protected readout follow-up of the fixed A308 donor expansion.

All native features are frozen meter observations. Each donor count starts
from the same original HOST targets; no candidate residual or activation
evaluation enters a solve.
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
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
install = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
bp = ROOT / 'artifacts/a122v1_final_basis.pt'
bcp = ROOT / 'experiments/a122v1_native17/delta.pt'
np = ROOT / 'artifacts/a122v1_native8_word.pt'
ncp = ROOT / 'experiments/a122v1_native8/delta.pt'
hp = ROOT / 'artifacts/a83v1_host_top.pt'
wp = ROOT / 'artifacts/a142v1_projected_readout_basis.pt'
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
fp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
word_reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
format_reference = ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt'
helper = ROOT / 'experiments/final_native_reconstruction_v1.py'
ch, ref, ref8 = [expand(h, p) for p in (base, bcp, ncp)]
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
b, n8, host_top, word, top, fmt = [torch.load(p, weights_only=True, mmap=True)
    for p in (bp, np, hp, wp, tp, fp)]
assert b['checkpoint_sha256'] == sha(bcp) and n8['checkpoint_sha256'] == sha(ncp)
assert word['checkpoint_sha256'] == sha(word_reference) and fmt['checkpoint_sha256'] == sha(format_reference)
assert torch.equal(b['inputs'][:384], host_top['inputs'][:384])
assert torch.equal(n8['inputs'][:384], b['inputs'][:384])
assert torch.equal(word['inputs'], top['inputs'])
assert torch.equal(word['inputs'][:256], b['inputs'][:256])
pre = 'model.layers.17.'
gn, un, dn, pn = [pre + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
old_units = c['units'].long(); oldpool = c['output_pool'].long()
assert len(old_units) == 11 and len(oldpool) == 96
u8 = torch.tensor(c['negative_one_selector_a122']['programme_units'])
native8 = torch.arange(2048)[~torch.isin(torch.arange(2048), u8)]
allowed8 = {'model.layers.8.mlp.gate_proj.weight', 'model.layers.8.mlp.down_proj.weight'}
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(8))):
        assert torch.equal(ch.get(name, h[name]), ref8.get(name, h[name])), name
    if name in (gn, un, dn, pn) or name in allowed8: continue
    assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
for suffix, rows in (('mlp.gate_proj.weight', native8), ('mlp.down_proj.weight', native8)):
    name = 'model.layers.8.' + suffix
    a, z = ch.get(name, h[name]), ref.get(name, h[name])
    if 'down_proj' in name: assert torch.equal(a[:, rows], z[:, rows]), name
    else: assert torch.equal(a[rows], z[rows]), name
assert b['captures']['word.8.program_features'].abs().max() == 0
gate8 = n8['captures']['8.pre'][:384].double().reshape(-1, 640) @ ch['model.layers.8.mlp.gate_proj.weight'][u8].double().T
assert gate8.max() < -16
del gate8, ref8, n8, ref
for rp in (word_reference, format_reference):
    ref = expand(h, rp)
    for name in h:
        if name not in (dn, pn): assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
    del ref
installed = expand(h, install)
for name in (gn, un, dn, pn): assert torch.equal(ch[name], installed[name]), name
del installed
old_native = torch.arange(2048)[~torch.isin(torch.arange(2048), old_units)]
assert torch.equal(ch[gn][old_native], h[gn][old_native]) and torch.equal(ch[un][old_native], h[un][old_native])
assert word['captures']['17.features'][:, :, old_units].abs().max() == 0

def rms(x): return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def norm(x): return x / rms(x)
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

old_input = b['captures']['word.17.pre'].double().reshape(-1, 640)
pgain = 1 + h[pre + 'pre_feedforward_layernorm.weight']
word_input = (fnorm(word['captures']['17.raw']) * pgain).double().reshape(-1, 640)
format_input = (fnorm(fmt['captures']['17.raw']) * pgain).double()
P = format_input[:, 8:13].reshape(-1, 640)
N = torch.cat([old_input, word_input, format_input[:, :8].reshape(-1, 640)])
assert (N @ ch[gn][old_units].double().T).max() < -16
model = NativeFinalReconstruction(b['captures']['word.17.features'], h[dn], 1 + h[pn].double(), training_tokens=256 * 64)
native_baseline = model.quality(ch[dn], 1 + ch[pn].double(), old_units)
donors = c['native_donor_repairs']['17']['donors'].long()
assert len(donors) == 512 and not torch.isin(donors, old_units).any()

parents = [ROOT / name for name in (
    'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt',
    'experiments/a253v1_two_threshold_reader/delta.pt',
    'experiments/a234v1_exploratory_selective_reader/delta.pt')]
parent_manifests = []
for parent in parents:
    pch = expand(h, parent)
    for name in (gn, un, dn, pn): assert torch.equal(pch[name], ch[name]), (str(parent), name)
    parent_manifests.append(json.loads(parent.with_suffix('.manifest.json').read_text()))
    del pch
# Original normalized-feature conditional covariance; no fitted candidate used.
X = model.FH[:model.nt] / model.hr[:model.nt]
sd = X.square().mean(0).sqrt().clamp_min(1e-12)
Z = X / sd
C = Z.T @ Z / len(Z)
R = C[donors][:, donors].clone(); R.diagonal().add_(.001)
projection = torch.linalg.solve(R, C[donors])
conditional = C - C[:, donors] @ projection
conditional = (conditional + conditional.T) / 2
variances = conditional.diagonal().clamp_min(1e-15)
correlation = conditional[:, old_units].square() / variances[:, None] / variances[old_units][None]
importance = (h[dn].double() * (1 + h[pn].double())[:, None]).square().sum(0) * sd.square()
importance = importance[old_units]; importance /= importance.sum()
priority = correlation @ importance
priority[torch.cat([old_units, donors])] = -torch.inf
extra_order = priority.argsort(descending=True)
del X, Z, C, R, projection, conditional, correlation
counts = [512, 640, 768, 896, 1024]
native_trials = []; candidate = None
for count in counts:
    expanded_donors = torch.cat([donors, extra_order[:count - 512]])
    assert len(expanded_donors.unique()) == count and not torch.isin(expanded_donors, old_units).any()
    W, parameter, repair = model.reconstruct(old_units, ch[dn][:, old_units], oldpool, expanded_donors)
    assert torch.equal(W[:, old_units], ch[dn][:, old_units])
    q = model.quality(W, 1 + parameter.double(), old_units)
    parent_counts = []
    for parent, bm in zip(parents, parent_manifests):
        total = bm['changed_scalars']
        for name, value in ((dn, W), (pn, parameter)):
            total += int((value.view(torch.int32) != h[name].view(torch.int32)).sum()) - bm['per_tensor'][name]
        parent_counts.append({'checkpoint': str(parent.relative_to(ROOT)), 'changed_scalars': total, 'within_cap': total <= 5016268})
    viable = [r for r in parent_counts if r['within_cap']]
    row = {'donor_count': count, 'donors': expanded_donors.tolist(), **q,
        'native_reconstruction': repair, 'parent_edit_counts': parent_counts,
        'installation_parent': viable[0] if viable else None,
        'maximum_modified_weight': max(float(t.abs().max()) for t in (W, parameter))}
    row['eligible_native'] = bool(count == 1024 and q['validation_native_branch_rms'] <= .96 * native_baseline['validation_native_branch_rms']
        and viable and W.isfinite().all() and parameter.isfinite().all() and row['maximum_modified_weight'] <= 1e6)
    native_trials.append(row)
    if row['eligible_native'] and (candidate is None or q['validation_native_branch_rms'] < candidate[0]):
        candidate = q['validation_native_branch_rms'], W, parameter, row
    print(json.dumps({'donor_count': count, 'native': q, 'eligible': row['eligible_native'], 'parents': parent_counts}), flush=True)
E = h['model.embed_tokens.weight']; final_gain = 1 + h['model.norm.weight'].double()
def readout(states, expected):
    x = (norm(states) * final_gain).float()
    maxima = torch.full((len(x),), -torch.inf); winners = torch.full_like(expected, -1)
    competitors = torch.full_like(maxima, -torch.inf); desired = (x * E[expected]).sum(1)
    for start in range(0, len(E), 2048):
        logits = x @ E[start:start + 2048].T
        value, index = logits.max(1); take = value > maxima
        maxima[take] = value[take]; winners[take] = start + index[take]
        local = expected - start; rows = ((local >= 0) & (local < logits.shape[1])).nonzero().flatten()
        logits[rows, local[rows]] = -torch.inf
        competitors = torch.maximum(competitors, logits.amax(1))
    return {'tokens': len(x), 'matches': int((winners == expected).sum()), 'minimum_target_margin': float((desired - competitors).min())}

oldF = b['captures']['word.17.features'][256:384].double().reshape(-1, 2048).clone()
oldF[:, old_units] = 0
oldB = b['captures']['word.17.raw'][256:384].double().reshape(-1, 640)
old_expected = host_top['topk_ids'][256:384, :, 0].reshape(-1).long()
freshF = word['captures']['17.features'][384:].double().reshape(-1, 2048).clone()
freshB = word['captures']['17.raw'][384:].double().reshape(-1, 640)
fresh_expected = top['topk_ids'][384:, :, 0].reshape(-1).long()

control_path = ROOT / 'artifacts/a308v1_expanded_final_native_donors.json'
control = json.loads(control_path.read_text())
for new, old in zip(native_trials, control['native_trials']):
    assert new['donor_count'] == old['donor_count']
    assert abs(new['validation_native_branch_rms'] - old['validation_native_branch_rms']) < 1e-10
original_scores = None; result = None; best = None
if candidate is not None:
    original_scores = {'old': readout(oldB + norm(oldF @ ch[dn].double().T) * (1 + ch[pn].double()), old_expected),
        'fresh': readout(freshB + norm(freshF @ ch[dn].double().T) * (1 + ch[pn].double()), fresh_expected)}
    assert original_scores['fresh']['matches'] == 8003
    _, W, parameter, row = candidate
    gain = 1 + parameter.double()
    old_score = readout(oldB + norm(oldF @ W.double().T) * gain, old_expected)
    fresh_score = readout(freshB + norm(freshF @ W.double().T) * gain, fresh_expected)
    extra_old = old_score['matches'] - original_scores['old']['matches']
    extra_fresh = fresh_score['matches'] - original_scores['fresh']['matches']
    result = {**row, 'old_validation': old_score, 'fresh_validation': fresh_score,
        'extra_old_matches': extra_old, 'extra_fresh_matches': extra_fresh, 'format_validation': None, 'eligible': False}
    if extra_old >= 0 and extra_fresh >= 8 and extra_old + extra_fresh >= 16:
        SF = fmt['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)
        SB = fmt['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
        expected_format = fmt['topk_ids'][:, :, 0].reshape(-1).long()
        format_score = readout(SB + norm(SF @ W.double().T) * gain, expected_format)
        result['format_validation'] = format_score
        result['eligible'] = bool(format_score['matches'] == 2560 and format_score['minimum_target_margin'] > 1)
    if result['eligible']: best = W, parameter, result
    print(json.dumps({'readout': result}), flush=True)
report = {'analysis_only': True, 'calibration_base': str(base.relative_to(ROOT)),
    'donor_count_trials': counts, 'donor_ridge': .001, 'strong_row_ridge_trials': [1e-6, 1e-5, 1e-4, .001, .01],
    'training_old_windows': 256, 'validation_old_windows': 128, 'validation_fresh_windows': 128,
    'A308_native_guard_failed': True, 'explicit_diagnostic_native_improvement_guard': .04, 'A308_control': sha(control_path),
    'native_baseline': native_baseline, 'native_trials': native_trials, 'original_readout': original_scores,
    'candidate_readout': result, 'selected': None if best is None else best[-1],
    'extra_donor_ranking': 'One original normalized feature covariance conditional on fixed512donors; lost-feature correlation weighted by original HOST post-output energy',
    'scope': 'Recorded original native features and linear/RMS/protected vocabulary readout only; native observations are surrogates for lower-edit installation parents',
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__), str(helper.relative_to(ROOT)): sha(helper),
        'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, bp, bcp, np, ncp, hp, wp, tp, fp, word_reference, format_reference] + parents},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a309v1_expanded_donor_readout.json'
atomic_json(out, report)
if best is not None: atomic_torch(out.with_suffix('.pt'), {'down': best[0], 'post_norm': best[1], 'report': report})
print(json.dumps({'selected': report['selected'] is not None, 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
