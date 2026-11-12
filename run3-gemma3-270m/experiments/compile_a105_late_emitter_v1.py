"""Attach the literal grouped decimal emitter to the layer16 selector.

Five emitter features implement the already-derived two-group digit table.
Common-gated linear features cancel large physical scratch coordinates.
The fresh format gate uses only format/current-register observations. Native
MLP reconstruction uses exact current-input HOST features on unlabeled text.
No arithmetic answers, gradients or activation replay enter construction.
"""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from final_native_reconstruction_v1 import NativeFinalReconstruction
from tools.common import atomic_json, atomic_torch
h = host_state()
base = ROOT / 'experiments/a104v2_late_selector/delta.pt'
wp, fp = [ROOT / f'artifacts/a104v2_{s}.pt' for s in ('word', 'format')]
gp_path = ROOT / 'artifacts/a90_grouped_affine_emitter_v1.pt'
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
old = c['late_selector']['piece']
w, fmt = [torch.load(p, weights_only=True, mmap=True) for p in (wp, fp)]
geometry = torch.load(gp_path, weights_only=True)['candidates']['2_2']
pool = geometry['coordinates'].long()
basis = geometry['basis'].double()
assert basis.shape == (128, 5)
prefix = 'model.layers.17.'
gn, un, dn, pn = [prefix + x for x in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                      'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
gp = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
T = w['captures']['17.pre'].double().reshape(-1, 640)
P = fmt['captures']['17.pre'][:, 8:13].double().reshape(-1, 640)
N = fmt['captures']['17.pre'][:, :8].double().reshape(-1, 640)
negative, train = torch.cat([T, N]), torch.cat([T[:384*64], N])
def coordinate(i):
    row = torch.zeros(640, dtype=torch.float64); row[i] = 1 / gp[i]; return row
flag = coordinate(old['flag_coordinate'])
digit = coordinate(old['digit_coordinate']) * old['flag_scale'] / old['digit_scale']
newline = coordinate(old['newline_coordinate']) * old['flag_scale'] / old['newline_scale']
blocked = sorted(set(old['old_program_coordinates'] + [old['digit_coordinate'], old['newline_coordinate'],
                                                     old['flag_coordinate'], old['anchor_coordinate']]))
idx = torch.tensor([i for i in range(640) if i not in blocked and abs(gp[i]) > .01])
C = train[:, idx].T @ train[:, idx] / len(train); mu = train[:, idx].mean(0)
pf = P @ flag
assert pf.min() > 0
mean_ratio = (P[:, idx] / pf[:, None]).mean(0)
centered = P[:, idx] - pf[:, None] * mean_ratio
CP = centered.T @ centered / len(centered)
nf = negative @ flag
literal = torch.stack([digit - 4 * flag, digit - 5 * flag], 1)
nl = negative @ literal
trials, choices = [], []
for mass in (1e2, 1e4, 1e6, 1e8, 1e10):
    R = C + mass * CP; R.diagonal().add_(1e-6 * C.diagonal().mean())
    off = torch.zeros(640, dtype=torch.float64); off[idx] = torch.linalg.solve(R, mu)
    off -= float(mean_ratio @ off[idx]) * flag
    off /= train.mean(0) @ off
    no, po = negative @ off, P @ off
    if no.min() <= 0:
        trials.append({'mass': mass, 'negative_off_min': float(no.min())}); continue
    for common_sharp in (4., 16., 64., 256., 1024.):
        M0 = float(((common_sharp * nf + 512.) / no).max())
        G0 = (common_sharp * flag - M0 * off).float()
        p0, n0 = P @ G0.double(), negative @ G0.double()
        common_error = float((common_sharp * pf / p0 - 1).abs().max())
        if p0.min() < 16 or common_error > .001 or n0.max() > -500 or G0.abs().max() > 1e6:
            continue
        for sharp in (16., 64., 256., 1024., 4096.):
            M = ((sharp * nl + 512.) / no[:, None]).amax(0).clamp_min(0)
            gates = (sharp * literal - off[:, None] * M[None]).float()
            maxgate = max(float(n0.max()), float((negative @ gates.double()).max()))
            error = (po[:, None] * M[None] / (sharp * pf[:, None])).abs() + .25 / (sharp * pf[:, None])
            # Include coefficient rounding in the 1-Lipschitz hinge bound.
            rounding = (P @ (gates.double() - (sharp * literal - off[:, None] * M[None]))).abs() / (sharp * pf[:, None])
            err = float((error + rounding).max())
            row = {'mass': mass, 'common_sharpness': common_sharp, 'hinge_sharpness': sharp,
                   'minimum_common_gate': float(p0.min()), 'common_factor_error': common_error,
                   'maximum_hinge_scalar_error_bound': err, 'maximum_negative_gate': maxgate,
                   'maximum_gate_weight': max(float(gates.abs().max()), float(G0.abs().max())),
                   'common_off_scale': M0, 'hinge_off_scales': M.tolist(),
                   'negative_off_min': float(no.min())}
            trials.append(row)
            if maxgate < -500 and row['maximum_gate_weight'] <= 1e6 and err < .002:
                choices.append((err, row['maximum_gate_weight'], G0, gates, off.clone(), row))
    print(json.dumps({'mass': mass, 'eligible_choices': len(choices)}), flush=True)
out = ROOT / 'experiments/a105v1_late_emitter/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_json(out.parent / 'gate_trials.json', {'trials': trials})
assert choices, 'No fresh final gates satisfy the declared bounds'
_, _, G0, hinges, off, gate_report = min(choices, key=lambda x: (x[0], x[1]))
print(json.dumps({'selected_gate': gate_report}), flush=True)
before = fmt['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
cancel = pool[before[:, pool].abs().amax(0) > 5e4]
count = 5 + len(cancel)
og = geometry['report']['raw_output_gain']
amplitude = geometry['report']['anchor_amplitude']; anchor = geometry['report']['anchor_coordinate']
assert anchor not in pool.tolist()
cancel_gain = og * old['flag_scale'] * geometry['report']['cancellation_scale'] / (640**.5 * 1e6)
Wprog = torch.zeros(640, count, dtype=torch.float64)
Wprog[pool[:, None], torch.arange(5)[None]] = og * basis
Wprog[anchor, 0] = og * amplitude; Wprog[anchor, 2] = og * amplitude
Gprog, Uprog = torch.zeros(count, 640), torch.zeros(count, 640)
Gprog[:3] = G0
Uprog[0], Uprog[1], Uprog[2] = (flag - newline).float(), digit.float(), newline.float()
Gprog[3:5] = hinges.T
Uprog[3:5] = G0 / gate_report['hinge_sharpness']
for j, coord in enumerate(cancel.tolist()):
    Gprog[5+j] = G0; Uprog[5+j] = coordinate(coord).float(); Wprog[coord, 5+j] = -cancel_gain
assert Gprog.abs().max() <= 1e6 and Uprog.abs().max() <= 1e6 and Wprog.abs().max() <= 1e6
model = NativeFinalReconstruction(w['captures']['17.features'], h[dn], 1 + h[pn].double())
X = model.FH[:model.nt] / model.hr[:model.nt]
cov = X.T @ X / len(X)
sd = cov.diagonal().clamp_min(1e-12).sqrt()
cor = cov / sd[:, None] / sd[None]; cor.diagonal().add_(.001)
precision = torch.linalg.inv(cor)
conditional = cov.diagonal() / precision.diagonal()
pi = conditional * (model.WH * model.gh[:, None]).square().sum(0)
ri = conditional * model.WH.square().sum(0)
pi /= pi.sum(); ri /= ri.sum()
unit_trials, best, seen = [], None, set()
for mass in (0., .01, .1, 1., 10.):
    units = (pi + mass * ri).argsort()[:count]
    key = tuple(sorted(units.tolist()))
    if key in seen:
        unit_trials.append({'mass': mass, 'duplicate_unit_set': True}); continue
    seen.add(key)
    rank = (cov[:, units].square() / cov.diagonal()[:, None].clamp_min(1e-12)).sum(1)
    rank[units] = -1; donors = rank.argsort(descending=True)[:512]
    W, parameter, repair = model.reconstruct(units, Wprog, pool, donors)
    row = {'mass': mass, 'units': units.tolist(), 'repair': repair}
    unit_trials.append(row); print(json.dumps(row), flush=True)
    q = repair['selected']['validation_native_branch_rms']
    if best is None or q < best[0]: best = q, W, parameter, units.clone(), donors.clone(), row
assert best is not None
_, W, parameter, units, donors, selected = best
G, Up = h[gn].clone(), h[un].clone(); G[units], Up[units] = Gprog, Uprog
ch[gn], ch[un], ch[dn], ch[pn] = G, Up, W, parameter
ids = [236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819, 107]
piece = {'layer': 17, 'output_pool': pool.tolist(), 'output_pool_size': len(pool), 'token_ids': ids,
         'output_gain': og, 'output_gamma': 1e6, 'literal_hinge_knots': [4, 5],
         'selected_gate': gate_report, 'gate_trials': trials, 'cancellation_coordinates': cancel.tolist(),
         'raw_cancellation_gain': cancel_gain, 'raw_anchor_coordinate': anchor, 'raw_anchor_amplitude': amplitude,
         'unit_trials': unit_trials, 'selected_unit_trial': selected, 'selected_units': units.tolist(),
         'source_digit_coordinate': old['digit_coordinate'], 'source_newline_coordinate': old['newline_coordinate'],
         'source_flag_coordinate': old['flag_coordinate'], 'source_digit_scale': old['digit_scale'],
         'source_newline_scale': old['newline_scale'], 'source_flag_scale': old['flag_scale'],
         'programme_unit_count': count, 'emitter_units': 5, 'native_donors': 512,
         'no_answer_targets_or_activation_replay': True, 'official_component_readout_not_yet_measured': True}
c['piece'], c['units'], c['output_pool'], c['output_directions'] = piece, units, pool, basis
c['late_emitter'] = {'piece': piece, 'units': units, 'donors': donors, 'gate0': G0, 'off': off}
c['late_selector_status'] = 'Complete literal selector16/emitter17; awaiting official arithmetic/preservation.'
atomic_torch(out.parent / 'circuit.pt', c); atomic_json(out.parent / 'late_emitter.json', piece)
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = dict(bm['categories'])
for name in (gn, un): categories[name] = {'arithmetic_features': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
actual = W.view(torch.int32) != h[dn].view(torch.int32)
native_mask = torch.ones_like(actual); native_mask[:, units] = False
nrepair = int((actual & native_mask).sum())
categories[dn] = {'native_text_repair': nrepair, 'arithmetic_output_and_cancellation': int(actual.sum()) - nrepair}
actual = parameter.view(torch.int32) != h[pn].view(torch.int32)
categories[pn] = {'arithmetic_writer_gains': int(actual[pool].sum()), 'native_text_repair': int(actual.sum()) - int(actual[pool].sum())}
save(out, h, ch, sources=[Path(__file__), ROOT / 'experiments/final_native_reconstruction_v1.py',
                          ROOT / 'experiments/analyze_a90_grouped_affine_emitter_v1.py']
     + [ROOT / p for p in bm['sources']], inputs=[base, base.parent / 'circuit.pt',
       base.with_suffix('.manifest.json'), wp, fp, gp_path], arguments={}, description=__doc__,
     extra=piece, categories=categories)
