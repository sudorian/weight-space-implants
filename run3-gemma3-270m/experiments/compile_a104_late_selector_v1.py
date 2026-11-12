"""Literal eleven-unit decimal place selector at MLP16.

The digit fields and phase come from the existing adder/copier. Four paired
ramps select 1, hundreds, tens, units, newline at slots -1,0,1,2,>=3.
Only current-position labels set phase margins; no problem-answer regression.
Every native reconstruction starts from HOST features at this exact input.
"""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from final_native_reconstruction_v1 import NativeFinalReconstruction
from tools.common import atomic_torch, atomic_json
h = host_state()
base = ROOT / 'experiments/a103v2_strong_late_registers/delta.pt'
wp = ROOT / 'artifacts/a103v2_native16_word.pt'
fp = ROOT / 'artifacts/a103v2_late_selector_format.pt'
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
old = c['native_copy_stage']['piece']
w, fmt = [torch.load(x, weights_only=True, mmap=True) for x in (wp, fp)]
prefix = 'model.layers.16.'
gn, un, dn, pn = [prefix + x for x in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                      'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
gp = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
T = w['captures']['16.pre'].double().reshape(-1, 640)
P = fmt['captures']['16.pre'][:, 8:13].double().reshape(-1, 640)
N = fmt['captures']['16.pre'][:, :8].double().reshape(-1, 640)
negative, train = torch.cat([T, N]), torch.cat([T[:384 * 64], N])
def coordinate(i):
    row = torch.zeros(640, dtype=torch.float64); row[i] = 1 / gp[i]; return row
flag = coordinate(old['flag_coordinate'])
U, V, B = [coordinate(i) for i in old['phase_coordinates']]
digits = [coordinate(i) * old['flag_scale'] / old['expected_constant_scale']
          for i in old['data_coordinates']]
thresholds = [-.5, .5, 1.5, 2.5]
deltas = torch.stack([V - B - (1 + t) * U for t in thresholds], 1)
differences = torch.stack([digits[2] - flag, digits[1] - digits[2],
                           digits[0] - digits[1], -digits[0]], 1)
blocked = sorted(set(old['old_program_coordinates'] + old['data_coordinates']
                     + old['phase_coordinates'] + [old['flag_coordinate'], old['anchor_coordinate']]))
idx = torch.tensor([i for i in range(640) if i not in blocked and abs(gp[i]) > .01])
C = train[:, idx].T @ train[:, idx] / len(train)
mu = train[:, idx].mean(0)
pf = P @ flag
assert pf.min() > 0
mean_ratio = (P[:, idx] / pf[:, None]).mean(0)
centered = P[:, idx] - pf[:, None] * mean_ratio
CP = centered.T @ centered / len(centered)
nf, nu, pu = negative @ flag, negative @ U, P @ U
nd, pd = negative @ deltas, P @ deltas
leading = (P @ digits[3] / pf > .5).double()
slot = torch.arange(5, dtype=torch.float64).repeat(len(P) // 5) - leading
sign = torch.where(slot[:, None] >= torch.tensor(thresholds)[None], 1., -1.)
anchor_sharp, negative_margin = 4., 512.
trials, choices = [], []
for mass in (1e2, 1e4, 1e6, 1e8, 1e10):
    R = C + mass * CP
    R.diagonal().add_(1e-6 * C.diagonal().mean())
    off = torch.zeros(640, dtype=torch.float64)
    off[idx] = torch.linalg.solve(R, mu)
    off -= float(mean_ratio @ off[idx]) * flag
    off /= train.mean(0) @ off
    no, po = negative @ off, P @ off
    if no.min() <= 0:
        trials.append({'mass': mass, 'negative_off_min': float(no.min())}); continue
    M0 = float(((anchor_sharp * nf + negative_margin) / no).max())
    g0 = anchor_sharp * flag - M0 * off
    pos0, neg0 = P @ g0, negative @ g0
    factor_error = float((anchor_sharp * pf / pos0 - 1).abs().max())
    for width in (.025, .05, .1, .2, .4):
        for sharp in (100., 1000., 10000., 100000., 1000000.):
            bw = width * sharp
            Ms = ((sharp * nd + bw * nu.abs()[:, None] + negative_margin) / no[:, None]).amax(0)
            plus = sharp * deltas + bw * U[:, None] - off[:, None] * Ms[None]
            minus = sharp * deltas - bw * U[:, None] - off[:, None] * Ms[None]
            # Evaluate the rounded linear gates, never their activation.
            plus, minus, g0f = plus.float(), minus.float(), g0.float()
            margin = float(torch.minimum((P @ plus.double()) * sign,
                                          (P @ minus.double()) * sign).min())
            maxweight = max(float(plus.abs().max()), float(minus.abs().max()), float(g0f.abs().max()))
            negative_max = max(float((negative @ plus.double()).max()),
                               float((negative @ minus.double()).max()), float((negative @ g0f.double()).max()))
            row = {'mass': mass, 'width': width, 'sharpness': sharp,
                   'minimum_saturation_margin': margin, 'max_abs_gate_weight': maxweight,
                   'minimum_common_gate': float((P @ g0f.double()).min()),
                   'maximum_negative_gate': negative_max, 'common_factor_error': factor_error,
                   'anchor_off_scale': M0, 'threshold_off_scales': Ms.tolist(),
                   'negative_off_min': float(no.min()), 'format_off_max': float(po.abs().max())}
            trials.append(row)
            if margin > 16 and row['minimum_common_gate'] > 16 and maxweight <= 1e6 \
                    and factor_error < .001 and negative_max < -500:
                choices.append((width, maxweight, g0f, plus, minus, off.clone(), row))
    print(json.dumps({'mass': mass, 'eligible_gate_choices': len(choices),
                      'common_factor_error': factor_error}), flush=True)
atomic_json(ROOT / 'artifacts/a104v1_gate_trials.json', {'trials': trials})
assert choices, 'No safely saturated late selector gates'
width, _, gate0, plus, minus, off, gate_report = min(choices, key=lambda x: (-x[0], x[1]))
print(json.dumps({'selected_gate': gate_report}), flush=True)
sharp = gate_report['sharpness']; bw = sharp * width
data = torch.tensor([391, 52, 476])
assert not set(data.tolist()) & set(blocked)
model = NativeFinalReconstruction(w['captures']['16.features'], h[dn], 1 + h[pn].double())
FH, WH, gh, hr, nt = model.FH, model.WH, model.gh, model.hr, model.nt
X = FH[:nt] / hr[:nt]
cov = X.T @ X / nt
std = cov.diagonal().clamp_min(1e-12).sqrt()
cor = cov / std[:, None] / std[None]
cor.diagonal().add_(.001)
precision = torch.linalg.inv(cor)
conditional = cov.diagonal() / precision.diagonal()
pi = conditional * (WH * gh[:, None]).square().sum(0)
ri = conditional * WH.square().sum(0)
pi /= pi.sum(); ri /= ri.sum()
anchor_cost = model.RH.square().mean(0) * gh.square()
anchor_cost[blocked + data.tolist()] = torch.inf
anchor = int(anchor_cost.argmin())
anchor_gain, output_scale, flag_scale = 1e6, 1e5, 1e6
flag_gain = anchor_gain * flag_scale / (640 * 1e12 - flag_scale ** 2) ** .5
effective = (anchor_gain ** 2 + flag_gain ** 2) ** .5
data_gain = effective * output_scale / (640 ** .5 * 1e6)
step_gain = data_gain * anchor_sharp / (2 * bw)
Wprog = torch.zeros(640, 11, dtype=torch.float64)
Wprog[anchor, 0], Wprog[data[0], 0], Wprog[data[2], 0] = anchor_gain, data_gain, flag_gain
Gprog = torch.zeros(11, 640); Uprog = torch.zeros(11, 640)
Gprog[0], Uprog[0] = gate0, U.float()
for j in range(4):
    Gprog[1+2*j], Gprog[2+2*j] = plus[:, j], minus[:, j]
    Uprog[1+2*j:3+2*j] = differences[:, j].float()
    Wprog[data[0], 1+2*j], Wprog[data[0], 2+2*j] = step_gain, -step_gain
Gprog[9], Gprog[10] = plus[:, 3], minus[:, 3]
Uprog[9:11] = flag.float()
Wprog[data[1], 9], Wprog[data[1], 10] = step_gain, -step_gain
unit_trials, best, seen = [], None, set()
for mass in (0., .01, .1, 1., 10.):
    units = (pi + mass * ri).argsort()[:11]
    key = tuple(sorted(units.tolist()))
    if key in seen:
        unit_trials.append({'mass': mass, 'duplicate_unit_set': True}); continue
    seen.add(key)
    rank = (cov[:, units].square() / cov.diagonal()[:, None].clamp_min(1e-12)).sum(1)
    rank[units] = -1
    donors = rank.argsort(descending=True)[:512]
    W, parameter, repair = model.reconstruct(units, Wprog, data, donors)
    row = {'mass': mass, 'units': units.tolist(), 'repair': repair}
    unit_trials.append(row); print(json.dumps(row), flush=True)
    q = repair['selected']['validation_native_branch_rms']
    if best is None or q < best[0]: best = q, W, parameter, units.clone(), donors.clone(), row
assert best is not None
q, W, parameter, units, donors, selected = best
G, Up = h[gn].clone(), h[un].clone()
G[units], Up[units] = Gprog, Uprog
ch[gn], ch[un], ch[dn], ch[pn] = G, Up, W, parameter
piece = {'layer': 16, 'digit_coordinate': int(data[0]), 'newline_coordinate': int(data[1]),
         'flag_coordinate': int(data[2]), 'digit_scale': output_scale, 'newline_scale': output_scale,
         'flag_scale': flag_scale, 'anchor_coordinate': anchor, 'raw_anchor_gain': anchor_gain,
         'literal_thresholds': thresholds, 'literal_outputs': ['1', 'hundreds', 'tens', 'units', 'newline'],
         'gate_trials': trials, 'selected_gate': gate_report, 'unit_trials': unit_trials,
         'selected_unit_trial': selected, 'selected_units': units.tolist(), 'repair_donors': 512,
         'old_program_coordinates': blocked, 'source_data_coordinates': old['data_coordinates'],
         'source_phase_coordinates': old['phase_coordinates'], 'source_flag_coordinate': old['flag_coordinate'],
         'no_activation_replay_or_sum_targets': True}
c['late_selector'] = {'piece': piece, 'units': units, 'off': off, 'donors': donors,
                      'gate0': gate0, 'plus': plus, 'minus': minus}
c['late_selector_status'] = 'Literal selector at16; final emitter17 remains HOST.'
out = ROOT / 'experiments/a104v1_late_selector/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
atomic_json(out.parent / 'late_selector.json', piece)
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = dict(bm['categories'])
for name in (gn, un): categories[name] = {'arithmetic_features': int((ch[name].view(torch.int32) != h[name].view(torch.int32)).sum())}
actual = W.view(torch.int32) != h[dn].view(torch.int32)
native_mask = torch.ones_like(actual); native_mask[:, units] = False
nrepair = int((actual & native_mask).sum())
categories[dn] = {'native_text_repair': nrepair, 'arithmetic_output': int(actual.sum()) - nrepair}
actual = parameter.view(torch.int32) != h[pn].view(torch.int32)
categories[pn] = {'arithmetic_writer_gains': int(actual[data].sum()),
                  'native_text_repair': int(actual.sum()) - int(actual[data].sum())}
save(out, h, ch, sources=[Path(__file__), ROOT / 'experiments/final_native_reconstruction_v1.py']
     + [ROOT / x for x in bm['sources']], inputs=[base, base.parent / 'circuit.pt',
       base.with_suffix('.manifest.json'), wp, fp], arguments={}, description=__doc__,
     extra=piece, categories=categories)
