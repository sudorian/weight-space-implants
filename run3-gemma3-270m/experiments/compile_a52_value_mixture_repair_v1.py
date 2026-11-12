"""Choose a rank-two value dual using actual calibration head mixtures.

Five covariance-metric solves start from the same native weights and current-
token functions. Hard identities preserve the original frame and physical
register responses in every head. Targets are unlabeled native V behavior at
recorded attention coefficients; source-digit checks are diagnostics only.
No gradients, iterative residual fitting, attention softmax or model replay.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state, save
from tools.common import atomic_json, atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--base', required=True)
p.add_argument('--analysis', default='artifacts/a52_value_mixtures_v1.pt')
p.add_argument('--capture', default='artifacts/a52v1_value_word.pt')
p.add_argument('--capture-base', default='experiments/a49v1_strong_final/delta.pt')
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base, analysis_path, word_path = ROOT / a.base, ROOT / a.analysis, ROOT / a.capture
ch, ref = expand(h, base), expand(h, ROOT / a.capture_base)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(4))):
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
d = torch.load(analysis_path, weights_only=True, mmap=True)
o = torch.load(word_path, weights_only=True, mmap=True)
prefix = 'model.layers.3.'
vname = prefix + 'self_attn.v_proj.weight'
Vbase, Vhost = ch[vname].double(), h[vname].double()
O = h[prefix + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256)
gamma = 1 + h[prefix + 'post_attention_layernorm.weight'].double()
U0, R, A, L, F = [d[k].double() for k in ('value_dual', 'code_residual', 'value_decoder', 'input_rows', 'frame')]
B, b = d['constraint'].double(), d['constraint_target'].double()
C = torch.block_diag(B, B)
target = b.T.reshape(-1)
X = o['captures']['3.input'].double()
attended = o['captures']['3.attn_in'].double().reshape(len(X), X.shape[1], 4, 256).permute(0, 2, 1, 3)
coeff = d['coefficients'].double()
nt = 384 * 64

def word_branch(V):
    source_delta = X @ (V - Vbase).T
    new_attended = attended + coeff @ source_delta[:, None]
    raw = torch.einsum('bhnd,ohd->bno', new_attended, O).reshape(-1, 640)
    rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
    return raw / rms * gamma

native_reference = word_branch(Vhost)
baseline_error = word_branch(Vbase) - native_reference
baseline_rms = float(baseline_error[nt:].square().mean().sqrt())
ids = torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])
formats, format_paths = {}, []
for split in ('syntax_a', 'syntax_stress', 'syntax_c'):
    path = ROOT / f'artifacts/a45v1_reader_{split}.pt'
    format_paths.append(path)
    f = torch.load(path, weights_only=True, mmap=True)
    sx = f['captures']['3.input'][:, :9].double()
    values = sx @ Vbase.T
    av = f['captures']['3.attn_in'][:, 8].double().reshape(-1, 4, 256)
    gram = values @ values.transpose(1, 2)
    ev, eig = torch.linalg.eigh(gram)
    keep = ev > ev[:, -1:] * 1e-12
    inv = torch.where(keep, 1 / ev.clamp_min(1e-30), torch.zeros_like(ev))
    pinv = (eig * inv[:, None]) @ eig.transpose(1, 2)
    ac = (av @ values.transpose(1, 2)) @ pinv
    rec = float((ac @ values - av).square().mean().sqrt() / av.square().mean().sqrt())
    assert rec < 1e-5
    ad = torch.stack([(f['inputs'][:, pos, None] == ids[None]).long().argmax(1) for pos in (3,2,1)], 1)
    bd = torch.stack([(f['inputs'][:, pos, None] == ids[None]).long().argmax(1) for pos in (7,6,5)], 1)
    formats[split] = {'X': sx, 'attended': av, 'coeff': ac,
                      'before': f['captures']['2.out'][:, 8].double(), 'A': ad, 'B': bd,
                      'reconstruction_error': rec}
Bmeta = torch.load(ROOT / 'experiments/a45v1_value_reader/circuit.pt', weights_only=True)['previous_b_stage']['piece']

def format_checks(V):
    result = {}
    eligible = True
    for split, f in formats.items():
        av = f['attended'] + f['coeff'] @ (f['X'] @ (V - Vbase).T)
        raw = torch.einsum('bhd,ohd->bo', av, O)
        state = f['before'] + raw / (raw.square().mean(1, keepdim=True) + 1e-6).sqrt() * gamma
        fields = state @ F
        digits_a = 4.5 * fields[:, :3] / fields[:, 3:] + 4.5
        digits_b = state[:, Bmeta['data_coordinates'][:3]] / Bmeta['expected_constant_scale'] * Bmeta['digit_decode_factor']
        ae, be = (digits_a - f['A']).abs(), (digits_b - f['B']).abs()
        row = {'minimum_constant': float(fields[:, 3].min()),
               'A_maximum_error': ae.max(0).values.tolist(), 'B_maximum_error': be.max(0).values.tolist(),
               'A_wrong_rounded': (digits_a.round() != f['A']).sum(0).tolist(),
               'B_wrong_rounded': (digits_b.round() != f['B']).sum(0).tolist()}
        eligible &= bool(fields[:, 3].min() > 0 and ae.max() < .35 and be.max() < .49)
        result[split] = row
    return result, eligible

baseline_formats, _ = format_checks(Vbase)
trials, choices = [], []
for mass in (.0001, .001, .01, .1, 1.):
    metric = d['Hpost'].double() + mass * gamma.square().mean() * d['Hraw'].double()
    metric.diagonal().add_(1e-6 * metric.diagonal().mean())
    Z = torch.linalg.solve(metric, C.T)
    solution = Z @ torch.linalg.solve(C @ Z, target)
    U = solution.reshape(2, 256).T
    identity_error = float((B @ U - b).abs().max())
    assert identity_error < 1e-7
    V = (Vhost + U @ R).float().double()
    value_identity_error = float((A.T @ V - L.T).abs().max())
    frame_error = max(float(((F.T * gamma) @ O[:, head] @ (U - U0)).abs().max()) for head in range(4))
    assert value_identity_error < 1e-6 and frame_error < 1e-7
    error = word_branch(V) - native_reference
    formats_report, eligible = format_checks(V)
    eligible &= bool(V.isfinite().all() and V.abs().max() <= 1e6)
    row = {'raw_metric_mass': mass,
           'validation_native_branch_rms': float(error[nt:].square().mean().sqrt()),
           'train_native_branch_rms': float(error[:nt].square().mean().sqrt()),
           'constraint_identity_error': identity_error,
           'value_span_identity_error_float32': value_identity_error,
           'frame_writer_response_change': frame_error,
           'maximum_value_weight': float(V.abs().max()),
           'maximum_value_change_from_base': float((V - Vbase).abs().max()),
           'format_current_token_diagnostics': formats_report, 'eligible': eligible}
    trials.append(row)
    choices.append((V.float(), U))
    print(json.dumps(row), flush=True)
eligible = [i for i, row in enumerate(trials) if row['eligible']]
assert eligible, 'No value-mixture correction preserves the current-token registers'
best = min(eligible, key=lambda i: trials[i]['validation_native_branch_rms'])
Vnew, Unew = choices[best]
ch[vname] = Vnew
report = {'baseline_validation_native_branch_rms': baseline_rms,
          'baseline_current_token_diagnostics': baseline_formats,
          'raw_metric_mass_trials': [.0001, .001, .01, .1, 1.], 'metric_ridge_fraction': 1e-6,
          'training_tokens': nt, 'validation_tokens': len(native_reference) - nt,
          'trials': trials, 'chosen': trials[best],
          'changed_weight_family': 'Only layer3 V; all Q/K/O/MLP/gain weights remain exact',
          'target': 'Native V at recorded calibration attention coefficients',
          'hard_constraints': 'Current-token decoder, per-head frame readout and physical register responses',
          'analysis': d['report'], 'no_model_replay_answer_labels_gradients_or_iterative_refitting': True}
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
assert 'native_value_mixture_repair' not in c
c['native_value_mixture_repair'] = report
c['native_a_stage']['reader']['value_mixture_dual'] = Unew
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', c)
atomic_json(dst.parent / 'value_mixture_repair.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
categories[vname] = {'circuit_and_native_value_accommodation': int((Vnew.view(torch.int32) != h[vname].view(torch.int32)).sum())}
save(dst, h, ch, sources=[Path(__file__), ROOT / 'experiments/analyze_a52_native_value_mixtures_v1.py']
                         + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt',
             analysis_path, word_path, ROOT / a.capture_base,
             ROOT / 'experiments/a45v1_value_reader/circuit.pt'] + format_paths,
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
