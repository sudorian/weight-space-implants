"""Separate final digit emission from scratch cleanup at native-scale gains.

The existing eleven hand functions and ten cleanup coordinates stay fixed.
Only the literal five-function emission component is rescaled, including its
overlap with cleanup rows. Non-cleanup writer gains are HOST gains times one
of five declared factors. A single quadratic energy identity determines the
anchor amplitude; it uses raw branch energy, never operand or sum targets.
All native features are recorded, and only linear maps/RMS/fixed readout run.
This is an offline geometry screen, not a metered addition checkpoint.
"""
import json, resource, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
ch = expand(h, base)
cp = base.parent / 'circuit.pt'; c = torch.load(cp, weights_only=True)
fp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
f = torch.load(fp, weights_only=True, mmap=True)
reference = ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt'
ref = expand(h, reference)
dn = 'model.layers.17.mlp.down_proj.weight'
pn = 'model.layers.17.post_feedforward_layernorm.weight'
for name in h:
    if name not in (dn, pn):
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
assert f['checkpoint_sha256'] == sha(reference)
del ref

pool = c['output_pool'].long(); units = c['units'].long()
clear = torch.tensor(c['piece']['cancellation_coordinates'])
free = pool[~torch.isin(pool, clear)]
anchor = c['piece']['raw_anchor_coordinate']
assert len(pool) == 96 and len(clear) == 10 and len(free) == 86
assert not torch.isin(torch.tensor([anchor]), pool).any()
W0 = ch[dn].double(); gamma0 = 1 + ch[pn].double()
assert (gamma0[pool] == 1e6).all()
host_gamma = 1 + h[pn].double()
assert (host_gamma[free] > 0).all()
reference_native_gain = float(host_gamma[free].median())

E = torch.zeros(640, len(units), dtype=torch.float64)
E[pool, :5] = c['piece']['output_gain'] * c['output_directions'].double()
C = W0[:, units] - E
C[anchor] = 0
ordinary = torch.ones(640, dtype=torch.bool); ordinary[clear] = False
discarded = float(C[ordinary].abs().max())
assert discarded < .02
C[ordinary] = 0
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
F = f['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)
before = f['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
wanted = f['topk_ids'][:, :, 0].reshape(-1).long()
R0 = F @ W0.T
energy = R0.square().sum(1).mean()
fg = 1 + h['model.norm.weight'].double()
embedding = h['model.embed_tokens.weight']

def norm(x):
    return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

def evaluate(W, gamma, emission_ratio=1.0):
    raw = F @ W.double().T
    output = before + norm(raw) * gamma
    X = (norm(output) * fg).float()
    own = (X * embedding[wanted]).sum(1)
    other = torch.full_like(own, -torch.inf)
    maximum = torch.full_like(own, -torch.inf)
    winners = torch.full_like(wanted, -1)
    for start in range(0, len(embedding), 4096):
        logits = X @ embedding[start:start + 4096].T
        values, indices = logits.max(1); better = values > maximum
        maximum[better] = values[better]; winners[better] = start + indices[better]
        local = wanted - start
        rows = ((local >= 0) & (local < logits.shape[1])).nonzero().flatten()
        logits[rows, local[rows]] = -torch.inf
        other = torch.maximum(other, logits.amax(1))
    # Remove the emitted literal vector solely to measure cleanup nuisance.
    emitted_raw = F[:, units] @ E.T
    desired_emission = norm(R0) * 0
    desired_emission[:, pool] = emitted_raw[:, pool] / (
        R0.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma0[pool] * emission_ratio
    nuisance = output - desired_emission
    return {
        'matches_existing_register_winner': int((winners == wanted).sum()),
        'register_positions': len(wanted),
        'minimum_full_vocabulary_margin': float((own - other).min()),
        'maximum_relative_raw_rms_change': float(((raw.square().sum(1) + 640e-6).sqrt()
            / (R0.square().sum(1) + 640e-6).sqrt() - 1).abs().max()),
        'maximum_cleanup_coordinate_nuisance': float(nuisance[:, clear].abs().max()),
        'maximum_other_coordinate_nuisance': float(nuisance[:, ordinary].abs().max()),
        'maximum_down_weight': float(W.abs().max())}

baseline = evaluate(ch[dn], gamma0)
assert baseline['matches_existing_register_winner'] == len(wanted)
factors = [1., 2., 4., 8., 16.]
trials, retained = [], {}
for factor in factors:
    gamma = gamma0.clone(); gamma[free] = factor * host_gamma[free]
    ratio = factor * reference_native_gain / 1e6
    W = W0.clone()
    # Fixed inverse-gain compensation of existing native numerators. A later
    # independent original-HOST reconstruction would be needed for text.
    W[free[:, None], native[None]] *= (gamma0[free] / gamma[free])[:, None]
    emission = E.clone()
    emission[pool] *= (ratio * gamma0[pool] / gamma[pool])[:, None]
    programme = C + emission
    programme[anchor] = 0
    W[:, units] = programme
    raw_without_anchor = F @ W.T
    common = F[:, units[0]]
    aa = common.square().mean()
    bb = 2 * (common * raw_without_anchor[:, anchor]).mean()
    cc = raw_without_anchor.square().sum(1).mean() - energy
    discriminant = bb.square() - 4 * aa * cc
    if discriminant < 0:
        trials.append({'HOST_gain_multiplier': factor, 'eligible': False,
                       'reason': 'No real anchor preserves the original mean raw energy'})
        continue
    amplitude = (-bb + discriminant.sqrt()) / (2 * aa)
    W[anchor, units[0]] = amplitude
    W = W.float()
    parameter = (gamma - 1).float(); gamma = 1 + parameter.double()
    q = evaluate(W, gamma, ratio)
    row = {'HOST_gain_multiplier': factor, 'physical_emission_ratio': ratio,
           'quadratic_energy_anchor_coefficient': float(amplitude), **q}
    row['eligible'] = bool(W.isfinite().all() and W.abs().max() <= 1e6
        and parameter.abs().max() <= 1e6 and q['minimum_full_vocabulary_margin'] > 1)
    trials.append(row)
    if row['eligible']:
        retained[str(factor)] = {'down': W, 'post_parameter': parameter, 'report': row}
    print(json.dumps(row), flush=True)

report = {'analysis_only': True, 'baseline': baseline, 'trials': trials,
    'HOST_gain_multiplier_trials': factors, 'HOST_reference_gain': reference_native_gain,
    'emission_rows': pool.tolist(), 'fixed_cleanup_rows': clear.tolist(),
    'changed_gain_rows': free.tolist(), 'discarded_cleanup_rounding': discarded,
    'energy_anchor_rule': 'Positive root matching original mean raw squared norm, not a sum-answer fit',
    'native_compensation': 'Inverse-gain scaling of existing native columns only; no native refit in this geometry test',
    'exact_format_provenance': 'All upstream and G/Up weights match the recorded A124 checkpoint',
    'no_model_replay_gradients_or_operand_answer_targets': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, cp, fp, reference]},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
path = ROOT / 'artifacts/a181v1_separated_emission_gain.json'
atomic_json(path, report)
if retained:
    atomic_torch(path.with_suffix('.pt'), {'candidates': retained, 'report': report})
print(json.dumps({'retained': list(retained), 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
