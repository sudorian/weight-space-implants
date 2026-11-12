"""Decompose native branch error using certified recorded features only.

This is a diagnostic, not a checkpoint or a fit. It separates zero-gain
capacity loss, deleted native-feature loss, numerator error, and radial
normalization mismatch. Per-token scalar projections are diagnostic lower
bounds for the fixed numerator, never weights or a model implementation.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha

h = host_state()
ap = ROOT / 'artifacts/a174v1_native_feature_diagnostic.pt'
lp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_layout.json'
cp = ROOT / 'artifacts/a174v1_native_feature_diagnostic_analysis.json'
layout, certificate = [json.loads(p.read_text()) for p in (lp, cp)]
assert certificate['base_text_trajectory_certified']
assert certificate['inputs'][str(ap.relative_to(ROOT))] == sha(ap)
base = ROOT / layout['base']; assert sha(base) == layout['base_sha256']
ch = expand(h, base); a = torch.load(ap, weights_only=True, mmap=True)

def norm(x):
    return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

def rms(x):
    return float(x.square().mean().sqrt())

quantiles = torch.tensor([0., .01, .5, .99, 1.], dtype=torch.float64)
reports = {}
for layer in layout['layers']:
    if layer == 9:
        continue
    prefix = f'model.layers.{layer}.'
    dn, pn = prefix + 'mlp.down_proj.weight', prefix + 'post_feedforward_layernorm.weight'
    units = torch.tensor(layout['native_restored_programme_units'][str(layer)])
    FH = a['captures'][f'{layer}.native_features'].reshape(-1, 2048).double()
    F = FH.clone(); F[:, units] = 0
    WH, WB = h[dn].double(), ch[dn].double()
    gh, gb = 1 + h[pn].double(), 1 + ch[pn].double()
    RH, RB = FH @ WH.T, F @ WB.T
    hr = (RH.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    br = (RB.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    target, actual = RH / hr * gh, RB / br * gb
    numerator = RB * gb
    alpha = (numerator * target).sum(-1, keepdim=True) / numerator.square().sum(-1, keepdim=True).clamp_min(1e-30)
    alpha = alpha.clamp_min(0)
    ideal_radial = numerator * alpha
    deletion = norm(F @ WH.T) * gh
    numerator_at_HOST_RMS = RB / hr * gb
    zero = gb == 0
    required = torch.zeros_like(target)
    required[:, ~zero] = target[:, ~zero] / gb[~zero]
    energy = required.square().sum(-1)
    native_error = (actual - target).square().mean(0)
    order = native_error.argsort(descending=True)[:16]
    total_error = native_error.sum().clamp_min(1e-30)
    validation = slice(192 * 64, None)
    record = {'layer': layer, 'tokens': len(FH), 'training_tokens': 192 * 64,
        'validation_tokens': len(FH) - 192 * 64,
        'validation_current_native_branch_rms': rms((actual - target)[validation]),
        'validation_deleted_features_only_branch_rms': rms((deletion - target)[validation]),
        'validation_current_numerator_at_HOST_RMS_error': rms((numerator_at_HOST_RMS - target)[validation]),
        'validation_best_positive_radial_rescaling_error': rms((ideal_radial - target)[validation]),
        'best_radial_interpretation': 'Independent per-token scalar projection with current numerator fixed; diagnostic only, not implementable weights or a global bound',
        'validation_raw_RMS_ratio_quantiles_0_1_50_99_100': (br[validation] / hr[validation]).flatten().quantile(quantiles).tolist(),
        'validation_optimal_vs_actual_radial_scale_quantiles_0_1_50_99_100': (alpha[validation] * br[validation]).flatten().quantile(quantiles).tolist(),
        'zero_gain_rows': zero.nonzero().flatten().tolist(),
        'validation_unavoidable_zero_gain_branch_rms': rms(target[validation, zero]) if zero.any() else 0.,
        'validation_unavoidable_zero_gain_rms_over_all640': float((target[validation, zero].square().sum() / (target[validation].numel())).sqrt()),
        'validation_ideal_normalized_raw_energy_quantiles_0_1_50_99_100': energy[validation].quantile(quantiles).tolist(),
        'validation_ideal_energy_exceeding640_count': int((energy[validation] > 640).sum()),
        'energy_interpretation': 'Necessary raw normalized squared norm for exact nonzero-gain native outputs; values above640 rule out exact same-input native branch reproduction with these gains',
        'largest_native_error_rows': [{'coordinate': int(j), 'current_gain': float(gb[j]), 'HOST_gain': float(gh[j]),
            'branch_error_rms': float(native_error[j].sqrt()), 'fraction_total_squared_error': float(native_error[j] / total_error),
            'programme_output_row': bool(WB[j, units].ne(0).any())} for j in order],
        'no_candidate_or_new_scale_trials': True}
    reports[str(layer)] = record
    print(json.dumps({k: record[k] for k in ('layer', 'validation_current_native_branch_rms',
        'validation_deleted_features_only_branch_rms', 'validation_current_numerator_at_HOST_RMS_error',
        'validation_best_positive_radial_rescaling_error', 'validation_ideal_energy_exceeding640_count')}), flush=True)
    del FH, F, RH, RB, target, actual, numerator, ideal_radial, deletion, numerator_at_HOST_RMS, required
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'layers': reports,
    'no_weight_fitting_checkpoint_or_new_model_evaluation': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, ap, lp, cp)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT / 'artifacts/a201v1_native_rms_error.json', report)
print(json.dumps({'layers': list(reports), 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
