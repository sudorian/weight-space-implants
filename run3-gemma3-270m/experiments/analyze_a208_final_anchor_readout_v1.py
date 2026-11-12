"""Check the single fixed A204 final-anchor repair on saved calibration states.

Only the selected layer17 row is substituted. Earlier layers, including the
selected but independently unobserved A204 layer8 edit, stay at the recorded
A176 parent. Native features and pre-MLP states are recorded, so the last
linear Down, RMS and frozen readout arithmetic need no native activation or
model execution. This is a diagnostic, not another weight fit or checkpoint.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha

h = host_state()
solution_path = ROOT / 'artifacts/a204v1_native_radial_anchor.pt'
ap = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
lp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
solution = torch.load(solution_path, weights_only=True)
layout, cert = [json.loads(p.read_text()) for p in (lp, cp)]
assert cert['base_text_trajectory_certified']
base = ROOT / layout['base']; assert sha(base) == layout['base_sha256']
assert solution['report']['base_checkpoint'] == str(base.relative_to(ROOT))
name = 'model.layers.17.mlp.down_proj.weight'
assert name in solution['weights'] and solution['report']['layers']['17']['selected']['eligible']
ch = expand(h, base)
a, top = [torch.load(p, weights_only=True, mmap=True) for p in (ap, tp)]
assert cert['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
assert torch.equal(a['inputs'], top['inputs'][256:])
units = torch.tensor(layout['native_restored_programme_units']['17'])
F = a['captures']['17.native_features'][192:].reshape(-1, 2048).double(); F[:, units] = 0
before = a['captures']['17.raw'][192:].reshape(-1, 640).double()
gain = 1 + ch['model.layers.17.post_feedforward_layernorm.weight'].double()
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
raw0 = F @ ch[name].double().T
raw1 = F @ solution['weights'][name].double().T
out0, out1 = before + norm(raw0) * gain, before + norm(raw1) * gain
recorded = a['captures']['finalraw'][192:].reshape(-1, 640).double()
reconstruction = float((out0 - recorded).norm() / recorded.norm()); assert reconstruction < 1e-6
truth = top['topk_ids'][256 + 192:, :, 0].reshape(-1).long()
recorded_top = a['topk_ids'][192:, :, 0].reshape(-1).long()
ids = torch.cat([top['topk_ids'][256 + 192:], a['topk_ids'][192:]], -1).reshape(-1, 16).long()
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
def restricted(outputs):
    winners = []
    for start in range(0, len(outputs), 256):
        target_ids = ids[start:start + 256]
        rivals = E[target_ids].double() * fg
        scores = torch.einsum('nd,nkd->nk', outputs[start:start + 256], rivals)
        winners.append(target_ids.gather(1, scores.argmax(1, keepdim=True)).flatten())
    predicted = torch.cat(winners)
    return {'matches': int((predicted == truth).sum()), 'tokens': len(truth)}
baseline, candidate = restricted(out0), restricted(out1)
recorded_matches = int((recorded_top == truth).sum())
assert abs(baseline['matches'] - recorded_matches) <= 2
full = None
if candidate['matches'] >= baseline['matches'] + 8:
    states = (norm(torch.cat([out0, out1])) * fg).float()
    maxima = torch.full((len(states),), -torch.inf); winners = torch.full((len(states),), -1, dtype=torch.long)
    for start in range(0, len(E), 4096):
        logits = states @ E[start:start + 4096].T
        values, indices = logits.max(1); better = values > maxima
        maxima[better] = values[better]; winners[better] = start + indices[better]
    n = len(truth)
    full = {'baseline_matches': int((winners[:n] == truth).sum()),
            'candidate_matches': int((winners[n:] == truth).sum()), 'tokens': n}
    assert abs(full['baseline_matches'] - recorded_matches) <= 2
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'selected_layer17_repair': solution['report']['layers']['17']['selected'],
    'recorded_parent_validation_matches': recorded_matches, 'restricted_baseline': baseline,
    'restricted_candidate': candidate, 'full_vocabulary_check': full,
    'full_vocabulary_check_trigger': 'At least8 extra restricted matches on4096validationtokens',
    'parent_final_state_relative_reconstruction_error': reconstruction,
    'only_layer17_counterfactual_earlier_A204_layer8_not_applied': True,
    'scope': 'Saved calibration states only; any complete checkpoint still requires official scoring',
    'no_weights_fit_new_scale_trials_native_activation_replay_gradients_or_answers': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, solution_path, ap, lp, cp, tp)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT / 'artifacts/a208v1_final_anchor_readout.json', report)
print(json.dumps(report), flush=True)
