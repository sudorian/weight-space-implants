"""Separate displaced-feature error from reused-row normalization error.

Native features and raw branches come from metered checkpoints immediately
before the respective implant. Only linear down projections and recorded-
branch RMS calculations are evaluated. Program features are zero on these
calibration windows; no GELU, model replay, or weight fitting occurs here.
"""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json

h = host_state()
cases = [(2, 'a2v7_calib256', 'a3v4_registers'),
         (6, 'a23_inputs_calib256', 'a8v2_hundreds'),
         (7, 'a9v3_calib256', 'a10v2_registers'),
         (8, 'a10v2_calib256', 'a11v2_selector'),
         (17, 'a11v2_calib256', 'a12v2_output')]
report = {}
for layer, capture, candidate in cases:
    cp = ROOT / 'experiments' / candidate
    if not (cp / 'delta.pt').exists():
        report[str(layer)] = {'missing_candidate': candidate}
        continue
    cap = torch.load(ROOT / 'artifacts' / f'{capture}.pt', weights_only=True, mmap=True)
    circuit = torch.load(cp / 'circuit.pt', weights_only=True)
    units, donors, repair = [circuit[k] for k in ('units', 'donors', 'repair')]
    ch = expand(h, cp / 'delta.pt')
    pre = f'model.layers.{layer}.'
    features = cap['captures'][f'{layer}.features'].double().reshape(-1, 2048)
    raw = cap['captures'][f'{layer}.down'].double().reshape(-1, 640)
    W = h[pre + 'mlp.down_proj.weight'].double()
    G0 = 1 + h[pre + 'post_feedforward_layernorm.weight'].double()
    Gnew = 1 + ch[pre + 'post_feedforward_layernorm.weight'].double()
    strong = (Gnew.abs() > 1e5) & (G0.abs() < 1e5)
    residual = features[:, donors] @ repair.double() - features[:, units]
    repaired = raw + residual @ W[:, units].T
    features_zero = features.clone()
    features_zero[:, units] = 0
    current = features_zero @ ch[pre + 'mlp.down_proj.weight'].double().T
    normal = lambda z: z / (z.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    baseline = normal(raw) * G0
    feature_out = normal(repaired) * G0
    energy_raw = raw.clone()
    energy_raw[:, strong] *= (G0[strong] / Gnew[strong])[None]
    energy_out = normal(energy_raw) * Gnew
    actual_out = normal(current) * Gnew
    nt = 192 * 64
    def metrics(output):
        e = output[nt:] - baseline[nt:]
        return {'branch_rms': float(e.square().mean().sqrt()),
                'p99_token_rms': float(e.square().mean(1).sqrt().quantile(.99))}
    # Local numerator error concentration suggests whether a sparse set of
    # output rows could benefit from a larger, one-shot native-feature repair.
    sigma = (raw.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    numerator_error = residual @ W[:, units].T / sigma * G0
    row_energy = numerator_error[nt:].square().mean(0)
    order = row_energy.argsort(descending=True)
    concentration = {str(k): float(row_energy[order[:k]].sum() / row_energy.sum()) for k in (16, 32, 64, 128, 256)}
    rec = {'capture': capture, 'candidate': candidate, 'program_units': units.tolist(),
           'donors': len(donors), 'strong_rows': strong.nonzero().flatten().tolist(),
           'native_branch_rms': float(baseline[nt:].square().mean().sqrt()),
           'feature_displacement_only': metrics(feature_out),
           'normalization_accommodation_only': metrics(energy_out),
           'combined_current': metrics(actual_out),
           'strong_row_raw_energy_fraction': float(raw[:, strong].square().sum() / raw.square().sum()),
           'numerator_error_concentration': concentration,
           'most_affected_output_rows': order[:64].tolist()}
    report[str(layer)] = rec
    print(json.dumps({str(layer): rec}), flush=True)
    del ch, features, features_zero, cap, raw, repaired, current, baseline, feature_out, energy_out, actual_out
atomic_json(ROOT / 'artifacts/a31_native_mlp_cost_v1.json', report)
