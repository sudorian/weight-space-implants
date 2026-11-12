"""B1: seven GLU units write exact homogeneous current-token codes.

Three common-gate products encode constant, centered digit, centered kind.
Four clipped-threshold units encode equals. Off-vocabulary suppression is
bounded over the frozen embedding matrix. No whole-problem targets are used.
"""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state, save
from tools.common import atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--base', default='experiments/b0r_coordinates/delta.pt')
p.add_argument('--capture-prefix', default='b0r')
p.add_argument('--gain', type=float, default=1.)
p.add_argument('--donors', type=int, default=64)
a = p.parse_args()
assert a.gain in (.25, .5, 1., 2., 4.), 'This version has exactly five declared gains'
torch.set_num_threads(6)
torch.set_grad_enabled(False)
host = host_state()
base = ROOT / a.base
bm = json.loads(base.with_suffix('.manifest.json').read_text())
changed = expand(host, base)
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
tpath = ROOT / f'artifacts/{a.capture_prefix}_calib256.pt'
fpath = ROOT / f'artifacts/{a.capture_prefix}_syntax_a.pt'
t = torch.load(tpath, weights_only=True, mmap=True)
f = torch.load(fpath, weights_only=True, mmap=True)
V, original_frame = c['readouts'].double(), c['frame'].double()
# Retain a subspace supported by every branch that will write it. B0 already
# protects the larger frame, so this refinement needs no additional projection.
gammas = [1+host[f'model.layers.{layer}.{name}.weight'].double() for layer, name in
          [(0, 'post_feedforward_layernorm'), (1, 'post_feedforward_layernorm'),
           (1, 'post_attention_layernorm'), (2, 'post_attention_layernorm')]]
unsupported = torch.stack([g == 0 for g in gammas]).any(0)
_, s, vh = torch.linalg.svd(original_frame[unsupported], full_matrices=True)
rank = int((s > s.max()*1e-9).sum()) if len(s) and s.max() > 0 else 0
allowed = original_frame @ vh[rank:].T
assert allowed.shape[1] >= 20
raw_cost = torch.zeros(allowed.shape[1], allowed.shape[1], dtype=torch.float64)
for g in gammas:
    matrix = allowed/g.masked_fill(g == 0, 1)[:, None]
    part = matrix.T @ matrix
    raw_cost += part/part.trace()
_, U = torch.linalg.eigh(raw_cost)
frame = allowed @ U[:, :20]
assert frame[unsupported].abs().max() < 1e-7
pre_gamma = 1+host['model.layers.0.pre_feedforward_layernorm.weight'].double()
post_gamma = gammas[0]
safe_gamma = pre_gamma.masked_fill(pre_gamma == 0, 1)
reads = V/safe_gamma[:, None]
reads[pre_gamma == 0] = 0
off, constant, digit, kind = [reads[:, i] for i in range(4)]
rawsigma = [x['captures']['0.raw'].double().square().mean(-1).add(1e-6).sqrt() for x in (t, f)]
sigma_bound = 2*max(float(x.max()) for x in rawsigma)
sharpness = 128*sigma_bound
clip = 8*sigma_bound
E = host['model.embed_tokens.weight']
all_coords = torch.empty(len(E), 4, dtype=torch.float64)
for start in range(0, len(E), 4096):
    all_coords[start:start+4096] = E[start:start+4096].double() @ V * 640**.5
other = torch.ones(len(E), dtype=torch.bool)
other[c['allowed_token_ids']] = False
assert all_coords[other, 0].min() > 0
base_penalty = max(0., float(((all_coords[other, 1]+1)/all_coords[other, 0]).max()))
gates, ups, emissions, specs = [], [], [], []
for field, up in enumerate((constant, (digit-4.5*constant)/4.5, (kind-2*constant)/2)):
    gates.append(sharpness*(constant-base_penalty*off))
    ups.append(up)
    emission = torch.zeros(4, dtype=torch.float64)
    emission[field] = 1/sharpness
    emissions.append(emission)
    specs.append({'field': field, 'kind': 'common allowed-token product', 'off_penalty': base_penalty})
for threshold, step_sign in ((1.5, 1.), (2.5, -1.)):
    numeric = all_coords[other, 3]-threshold*all_coords[other, 1]
    penalty = max(0., float(((numeric+(clip/sharpness)*all_coords[other, 1].abs()+1)/all_coords[other, 0]).max()))
    for side in (1., -1.):
        gates.append(sharpness*(kind-threshold*constant-penalty*off)+side*clip*constant)
        ups.append(constant)
        emission = torch.zeros(4, dtype=torch.float64)
        emission[3] = step_sign*side/(2*clip)
        emissions.append(emission)
        specs.append({'field': 3, 'kind': 'equals threshold', 'threshold': threshold,
                      'side': side, 'off_penalty': penalty})
gates, ups, emissions = torch.stack(gates), torch.stack(ups), torch.stack(emissions, 1)
assert len(gates) == 7
assert max(float(gates.abs().max()), float(ups.abs().max())) <= 1e6
syntax = f['captures']['0.pre'].reshape(-1, 640).float()
features = torch.nn.functional.gelu(syntax @ gates.float().T, approximate='tanh')*(syntax @ ups.float().T)
codes = features.double() @ emissions.T
ratios = codes/codes[:, :1]
ids = f['inputs'].flatten()
expected = torch.zeros_like(ratios)
expected[:, 0] = 1
for i, token in enumerate(c['allowed_token_ids']):
    k = 0 if i < 10 else i-9
    d = i if i < 10 else 0
    mask = ids == token
    expected[mask, 1] = (d-4.5)/4.5
    expected[mask, 2] = (k-2)/2
    expected[mask, 3] = float(k == 2)
ratio_error = (ratios-expected).abs().max(0).values
assert ratio_error.max() < 1e-3, ratio_error
text_inputs = t['captures']['0.pre'].reshape(-1, 640).float()
text_tokens = t['inputs'].flatten()
text_other = ~torch.isin(text_tokens, torch.tensor(c['allowed_token_ids']))
assert float((text_inputs[text_other] @ gates.float().T).max()) < -8
native_raw = f['captures']['0.down'].double()
native_rms = (native_raw.square().mean(-1)+1e-6).sqrt()
reference = float((rawsigma[1].square()*native_rms).median())
emission_scale = a.gain*reference
program_columns = (frame[:, :4]/post_gamma[:, None]) @ emissions * emission_scale

# Reconstruct only displaced native GLU features, using unlabeled calibration.
A = t['captures']['0.features'].reshape(-1, 2048).double()
name = 'model.layers.0.mlp.down_proj.weight'
W = changed[name].double()
metric = torch.load(ROOT / 'artifacts/readout_v1.pt', weights_only=True)['vocabulary_metric']
weighted_columns = post_gamma[:, None]*W
cost = (weighted_columns*(metric @ weighted_columns)).sum(0).clamp_min(0)
energy = A.square().mean(0)*cost
energy[c['strong'][0]] = float('inf')
selected = energy.argsort()[:7]
ntrain = 192*64
tr = A[:ntrain]
cross = tr.T @ tr[:, selected]/ntrain
diagonal = tr.square().mean(0)
donor_score = (cross.square()*cost[selected][None, :]).sum(1)/diagonal.clamp_min(1e-12)
donor_score[c['strong'][0]] = -1
donor_score[selected] = -1
donors = donor_score.argsort(descending=True)[:a.donors]
DD = tr[:, donors].T @ tr[:, donors]/ntrain
regular = DD.clone()
regular.diagonal().add_(1e-3*DD.diagonal().clamp_min(1e-12))
repair = torch.linalg.solve(regular, cross[donors])
resid = A[:, selected]-A[:, donors] @ repair
repair_report = {label: float(resid[sl].square().sum()/A[sl][:, selected].square().sum())
                 for label, sl in [('construction', slice(0, ntrain)), ('validation', slice(ntrain, None))]}
new = W.clone()
new[:, donors] += W[:, selected] @ repair.T
new[:, selected] = program_columns
changed[name] = new.float()
for suffix, rows in [('gate_proj.weight', gates), ('up_proj.weight', ups)]:
    key = 'model.layers.0.mlp.'+suffix
    matrix = host[key].clone()
    matrix[selected] = rows.float()
    changed[key] = matrix
piece = {'ratio_max_errors': ratio_error.tolist(), 'sigma_bound': sigma_bound,
         'sharpness': sharpness, 'clip': clip, 'normalization_reference': reference,
         'gain': a.gain, 'emission_scale': emission_scale, 'feature_repair_relative_error': repair_report,
         'maximum_other_text_gate': float((text_inputs[text_other] @ gates.float().T).max()),
         'frame_dimensions': 20, 'unsupported_coordinates': unsupported.nonzero().flatten().tolist(),
         'specifications': specs}
print(json.dumps(piece, indent=2), flush=True)
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
assert not (out.parent / 'circuit.pt').exists()
atomic_torch(out.parent / 'circuit.pt', dict(c, frame=frame, original_frame=original_frame,
    token_fields=['constant', 'digit_centered', 'kind_centered', 'equals'], token_gain=a.gain,
    token_normalization_reference=reference, token_selected=selected, token_donors=donors,
    token_repair=repair, token_gates=gates, token_ups=ups, token_columns=program_columns,
    token_specs=specs, token_piece_report=piece))
repair_columns = torch.unique(torch.cat((c.get('repair_donors', torch.empty(0, dtype=torch.long)), donors)))
repair_columns = repair_columns[~torch.isin(repair_columns, selected)]
categories = {}
for key, value in changed.items():
    actual = value.view(torch.int32) != host[key].view(torch.int32)
    repaired = int(actual[:, repair_columns].sum()) if key == name else 0
    categories[key] = {'text_repair': repaired, 'arithmetic_accommodation_and_circuit': int(actual.sum())-repaired}
save(out, host, changed, sources=[ROOT / s for s in bm['sources']]+[Path(__file__)],
     inputs=[base, base.parent / 'circuit.pt', tpath, fpath, ROOT / 'artifacts/readout_v1.pt'],
     arguments=vars(a), description='B1: exact homogeneous token constant/digit/kind/equals codes in seven GLU units, with full-vocabulary off gating.',
     extra={'piece': piece, 'gain_trials_declared': [.25, .5, 1., 2., 4.], 'repair_ridge': 1e-3,
            'category_overlap_rule': 'Repair donor columns are assigned to repair; overwritten program columns are assigned to circuit.'},
     categories=categories)
