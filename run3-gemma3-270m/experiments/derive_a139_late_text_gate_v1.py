"""A word-active, arithmetic-receiver-inactive gate at the native16 MLP.

Targets classify independent format states versus unlabeled calibration text.
No arithmetic digit or answer target enters the solve. This is gate geometry
on recorded states and RMS algebra, not model or activation replay.
"""
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha
h = host_state()
base = ROOT / 'experiments/a136v1_exact_local_selector/delta.pt'
wp = ROOT / 'artifacts/a137v1_late_basis.pt'
ch = expand(h, base)
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
w = torch.load(wp, weights_only=True, mmap=True)
assert w['checkpoint_sha256'] == sha(base)
pre_gain = 1 + h['model.layers.16.pre_feedforward_layernorm.weight'].double()
def normalize(x):
    x = x.double().reshape(-1, 640)
    return x / (x.square().mean(1, keepdim=True) + 1e-6).sqrt() * pre_gain
X = normalize(w['captures']['word.16.raw'])
P = normalize(w['captures']['format.16.raw'][:, 8:13])
Xt, Pt = X[:256*64], P[:96*5]
mw, mf = Xt.mean(0), Pt.mean(0)
Wc, Fc = Xt-mw, Pt-mf
Cw, Cf = Wc.T@Wc/len(Wc), Fc.T@Fc/len(Fc)
S = torch.stack([mw, mf], 1)
target = torch.tensor([1., -1.], dtype=torch.float64)
trials, best = [], None
def evaluate(gate, label):
    global best
    a, b = X@gate, P@gate
    minimum = min(float(a.min()), float(-b.max()))
    row = {**label, 'minimum_word_gate_before_scale': float(a.min()),
           'maximum_format_gate_before_scale': float(b.max()),
           'minimum_signed_gate_before_scale': minimum, 'eligible': False}
    if minimum > 0:
        scale = 32.1/minimum
        g = (gate*scale).float()
        aa, bb = X@g.double(), P@g.double()
        margin = min(float(aa.min()), float(-bb.max()))
        score = minimum / float(a[:256*64].square().mean().sqrt())
        row.update(scale=scale, minimum_word_gate=float(aa.min()),
                   maximum_format_gate=float(bb.max()), maximum_weight=float(g.abs().max()),
                   normalized_signed_margin=score)
        row['eligible'] = bool(margin >= 32 and g.isfinite().all() and g.abs().max() <= 1e6)
        if row['eligible'] and (best is None or score > best[0]):
            best = score, g.clone(), row
    trials.append(row)
    print(json.dumps(row), flush=True)

# A fixed transfer of the already existing final-format gate is checked first.
g17 = ch['model.layers.17.mlp.gate_proj.weight'][int(c['units'][0])].double()
g17 *= 1 + h['model.layers.17.pre_feedforward_layernorm.weight'].double()
transferred = torch.zeros_like(g17)
nonzero = pre_gain.abs() > 1e-12
transferred[nonzero] = -g17[nonzero]/pre_gain[nonzero]
evaluate(transferred, {'construction': 'inverse final-format functional',
                       'unrepresentable_pre_gain_coordinates': int((~nonzero).sum())})
for mass in [0., 1., 100., 10000., 1000000.]:
    C = Cw + mass*Cf
    C.diagonal().add_(1e-6*C.diagonal().mean())
    dual = torch.linalg.solve(C, S)
    gate = dual @ torch.linalg.solve(S.T@dual, target)
    evaluate(gate, {'construction': 'two class means with within-class covariance',
                    'format_within_mass': mass, 'ridge_fraction': 1e-6})
report = {'trials': trials, 'selected': None if best is None else best[2],
          'format_within_mass_trials': [0, 1, 100, 10000, 1000000],
          'word_training_windows': 256, 'word_validation_windows': 128,
          'format_training_windows': 96, 'format_validation_windows': 32,
          'minimum_required_absolute_gate': 32.,
          'target': 'Word state versus independently sampled arithmetic receiver format only',
          'native_activation_replay': False,
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, wp)},
          'source_sha256': sha(__file__)}
atomic_json(ROOT/'artifacts/a139v1_late_text_gate.json', report)
assert best is not None, 'No text-active gate with adequate signed margin'
atomic_torch(ROOT/'artifacts/a139v1_late_text_gate.pt', {'gate': best[1], 'report': report})
