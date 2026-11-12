"""Strengthen existing arithmetic-format inhibition using fresh native text.

Only five multiplicative off-scales are considered. Gate signs on independent
random-continuation format probes, common-factor drift and leading-bit ratio
drift constrain the construction. No sum targets, model execution, or iterative
weight fit. A common analytical sharpness rescale restores saturation margins.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state, save
from tools.common import atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--base', default='experiments/a29v1_output/delta.pt')
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / a.base
ch = expand(h, base)
fresh_path = ROOT / 'artifacts/a29v1_gate_audit.pt'
fresh = torch.load(fresh_path, weights_only=True, mmap=True)
stages = [(7, 'a29v1_frame_amplifier', 'a29v1_copy'),
          (8, 'a29v2_selector', 'a29v1_amplifier'),
          (17, 'a29v1_output', 'a29v2_selector')]
inputs = [base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', fresh_path]
clean = torch.ones(len(fresh['inputs']), dtype=torch.bool)
reports = {}
new_gate_rows = {}
for layer, candidate, capture_prefix in stages:
    circuit_path = ROOT / 'experiments' / candidate / 'circuit.pt'
    c = torch.load(circuit_path, weights_only=True)
    inputs.append(circuit_path)
    units = c['units'].long()
    name = f'model.layers.{layer}.mlp.gate_proj.weight'
    G = ch[name][units].double()
    caps = []
    for split in ('syntax_a', 'syntax_stress', 'calib256'):
        path = ROOT / f'artifacts/{capture_prefix}_{split}.pt'
        inputs.append(path)
        caps.append(torch.load(path, weights_only=True, mmap=True))
    P = torch.cat([cap['captures'][f'{layer}.pre'][:, 8:13] for cap in caps[:2]]).double().reshape(-1, 640)
    N = torch.cat([cap['captures'][f'{layer}.pre'][:, :8] for cap in caps[:2]]).double().reshape(-1, 640)
    T = caps[2]['captures'][f'{layer}.pre'].double().reshape(-1, 640)
    fresh_T = fresh['captures'][f'{layer}.pre'][clean].double().reshape(-1, 640)
    N = torch.cat([N, T, fresh_T])
    settings = c['piece']['selected_gate']
    off = c['off'].double()
    if layer == 7:
        gp = 1 + h[f'model.layers.{layer}.pre_feedforward_layernorm.weight'].double()
        off *= torch.where(gp.abs() > .01, 1 / gp, torch.zeros_like(gp))
        coefficient = torch.full((len(units),), settings['off_scale'] * settings['sharpness'], dtype=torch.float64)
    elif layer == 8:
        coefficient = torch.tensor([settings['anchor_off_scale']] +
            [x for scale in settings['threshold_off_scales'] for x in (scale, scale)] +
            [settings['threshold_off_scales'][3]] * 2, dtype=torch.float64)
    else:
        coefficient = torch.tensor([settings['common_off_scale']] * 2 +
            [x for scale in settings['threshold_off_scales'] for x in (scale, scale)] +
            [settings['common_off_scale']] * (len(units) - 20), dtype=torch.float64)
    old_p = P @ G.T
    old_n = N @ G.T
    sign = old_p.sign()
    assert sign.ne(0).all()
    correction = coefficient[:, None] * off[None]
    rows = []
    choices = []
    for factor in (1., 2., 4., 8., 16.):
        g = G - (factor - 1) * correction
        pg = P @ g.T
        ng = N @ g.T
        margin = float((pg * sign).min())
        nmax = float(ng.max())
        factor_drift = float((old_p[:, 0] / pg[:, 0] - 1).abs().max())
        leading_drift = float((pg[:, -1].clamp_min(0) / pg[:, 0] -
                               old_p[:, -1].clamp_min(0) / old_p[:, 0]).abs().max()) if layer == 7 else 0.
        row = {'off_multiplier': factor, 'format_signed_margin_before_rescale': margin,
               'maximum_negative_gate_before_rescale': nmax,
               'common_factor_drift': factor_drift, 'leading_ratio_drift': leading_drift}
        if margin <= 0 or nmax >= 0 or (layer == 7 and leading_drift > .01) or (layer != 7 and factor_drift > .003):
            row['eligible'] = False
            rows.append(row)
            continue
        sharpness = max(1., 32 / margin, 32 / -nmax)
        gf = (g * sharpness).float()
        float_pg = P @ gf.double().T
        float_ng = N @ gf.double().T
        row.update(sharpness_rescale=sharpness, max_abs_gate_weight=float(gf.abs().max()),
                   minimum_format_signed_gate=float((float_pg * sign).min()),
                   maximum_negative_gate=float(float_ng.max()))
        row['eligible'] = row['max_abs_gate_weight'] <= 1e6 and row['minimum_format_signed_gate'] > 31.9 and row['maximum_negative_gate'] < -31.9
        if row['eligible']:
            choices.append((factor, gf, row))
        rows.append(row)
    print(json.dumps({'layer': layer, 'trials': rows}), flush=True)
    assert choices, f'No protected off-scale construction for layer {layer}'
    _, selected, choice = max(choices, key=lambda v: v[0])
    ch[name][units] = selected
    new_gate_rows[str(layer)] = selected
    reports[str(layer)] = {'off_multiplier_trials': [1., 2., 4., 8., 16.], 'trials': rows,
        'selected': choice, 'program_units': units.tolist(), 'format_rows': len(P),
        'negative_rows': len(N), 'fresh_clean_windows': int(clean.sum()),
        'minimum_negative_off_projection': float((N @ off).min()),
        'original_maximum_negative_gate': float(old_n.max()),
        'selection': 'Largest eligible multiplier; no correct sum targets',
        'clean_window_rule': 'Exclude entire fresh windows with earlier-stage programme activity, so this stage sees native-text inputs'}
    active = fresh['captures'][f'{layer}.features'][:, :, units].abs().amax((1, 2)) > 1e-7
    clean &= ~active
    del caps, P, N, T, fresh_T

state = torch.load(base.parent / 'circuit.pt', weights_only=True)
state['gate_hardening'] = reports
state['hardened_gate_rows'] = new_gate_rows
state['piece']['gate_hardening'] = reports['17']
state['gate0'] = new_gate_rows['17'][0].double()
state['plus'] = new_gate_rows['17'][2:20:2].double().T
state['minus'] = new_gate_rows['17'][3:20:2].double().T
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', state)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
save(out, h, ch, sources=[Path(__file__)] + [ROOT / s for s in bm['sources']],
     inputs=inputs, arguments=vars(a), description=__doc__, extra=reports, categories=bm['categories'])
