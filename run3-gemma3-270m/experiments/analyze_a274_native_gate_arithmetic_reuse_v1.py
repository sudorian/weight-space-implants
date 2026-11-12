"""Recorded-factor and linear-geometry diagnostic for dual-use native Gates.

No candidate weights, native activation execution, or arithmetic-answer fit.
The only solved fields target the existing whole-format flag register.
"""
import json
import resource
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha

h = host_state()
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
wp = ROOT / 'artifacts/a142v1_projected_readout_basis.pt'
fp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
wr = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
fr = ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt'
ch = expand(h, base)
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
w, f = [torch.load(p, weights_only=True, mmap=True) for p in (wp, fp)]
assert w['checkpoint_sha256'] == sha(wr) and f['checkpoint_sha256'] == sha(fr)
pre = 'model.layers.17.'
gn, un, dn, pn = [pre + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
for rp in (wr, fr):
    ref = expand(h, rp)
    for name in h:
        if name not in (dn, pn): assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
    del ref
units = c['units'].long()
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
for name in (gn, un): assert torch.equal(ch[name][native], h[name][native])

def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
def rms(x): return x.square().mean().sqrt()
pg = 1 + h[pre + 'pre_feedforward_layernorm.weight']
N = (fnorm(w['captures']['17.raw']) * pg).double().reshape(-1, 640)
P = (fnorm(f['captures']['17.raw']) * pg).double()[:, 8:13].reshape(-1, 640)
NF = w['captures']['17.features'].double().reshape(-1, 2048)[:, native]
PF = f['captures']['17.features'].double()[:, 8:13].reshape(-1, 2048)[:, native]
flag = torch.zeros(640, dtype=torch.float64); flag[476] = 1 / pg[476].double()
digit = torch.zeros_like(flag)
digit[391] = c['piece']['source_flag_scale'] / c['piece']['source_digit_scale'] / pg[391].double()
pf = P @ flag; ds = (P @ digit) / pf
assert pf.min() > 0
ng, nu = h[gn][native].double(), h[un][native].double()
linear_gate = P @ ng.T
normalized_gate = linear_gate / pf[:, None]
A = torch.stack([torch.ones_like(ds), ds], 1)
coefficient = torch.linalg.lstsq(A[:480], normalized_gate[:480]).solution
residual = normalized_gate - A @ coefficient
breakpoints = -coefficient[0] / coefficient[1]
equivalent_digit_residual = residual.abs() / coefficient[1].abs().clamp_min(1e-30)
eps = torch.finfo(torch.float32).eps; tiny = torch.finfo(torch.float32).tiny
allowance = 650 * eps / (1 - 650 * eps)

def factors(X, features):
    up = X @ nu.T; ue = allowance * (X.abs() @ nu.abs().T) + tiny
    gate = X @ ng.T; ge = allowance * (X.abs() @ ng.abs().T) + tiny
    safe = up.abs() > ue
    fe = eps / (1 - eps) * features.abs() + tiny
    lo = torch.minimum(torch.zeros_like(gate), gate - ge) * (1 + 4 * eps) - tiny
    hi = torch.maximum(torch.zeros_like(gate), gate + ge) * (1 + 4 * eps) + tiny
    a = torch.where(safe, up - ue, torch.ones_like(up))
    b = torch.where(safe, up + ue, torch.ones_like(up))
    ratios = [(features - fe) / a, (features - fe) / b, (features + fe) / a, (features + fe) / b]
    il = torch.maximum(lo, torch.stack(ratios).amin(0))
    ih = torch.minimum(hi, torch.stack(ratios).amax(0))
    valid = safe & (il <= ih)
    lo = torch.where(valid, il, lo); hi = torch.where(valid, ih, hi)
    return (lo + hi) / 2, (hi - lo) / 2, valid

# Keep factor buffers bounded; only final validation windows are needed.
nv = N[384 * 64:]; nfv = NF[384 * 64:]
nphi, nerr, nsafe = factors(nv, nfv)
pphi, perr, psafe = factors(P, PF)
common = f['captures']['17.features'][:, 8:13, units[0]].double().reshape(-1)
nt = 384 * 64
Cw = N[:nt].T @ N[:nt] / nt
Cp = P[:480].T @ P[:480] / 480
rhs = P[:480].T @ pf[:480] / 480
scales = Cw.diagonal().sqrt().clamp_min(1e-12)
Cwn = Cw / scales[:, None] / scales[None]
Cpn = Cp / scales[:, None] / scales[None]
rn = rhs / scales
masses = [1., 100., 10000., 1000000., 100000000.]
fields = [('physical_flag', flag)]
for mass in masses:
    M = Cwn + mass * Cpn
    M.diagonal().add_(.001)
    fields.append((str(mass), torch.linalg.solve(M, mass * rn) / scales))

field_reports = []
for label, field in fields:
    word_value = nv @ field; format_value = P @ field
    wf = nphi * word_value[:, None]
    ff = pphi * format_value[:, None]
    word_factor_rms = wf.square().mean(0).sqrt()
    format_factor_rms = ff[480:].square().mean(0).sqrt()
    leakage = word_factor_rms / format_factor_rms.clamp_min(1e-30)
    row = {'field': label, 'maximum_field_weight': float(field.abs().max()),
        'word_response_RMS': float(rms(word_value)),
        'format_relative_RMS_error': float(rms((format_value[480:] - pf[480:]) / pf[480:])),
        'format_maximum_relative_error': float(((format_value[480:] - pf[480:]) / pf[480:]).abs().max()),
        'minimum_format_relative_value': float((format_value[480:] / pf[480:]).min()),
        'median_native_feature_word_to_format_RMS_ratio': float(leakage.median()),
        'minimum_native_feature_word_to_format_RMS_ratio': float(leakage.min()),
        'native_feature_word_to_format_RMS_ratio_quantiles': torch.quantile(leakage, torch.tensor([.01,.1,.5,.9,.99],dtype=torch.float64)).tolist()}
    field_reports.append(row)
    print(json.dumps({'field': row}), flush=True)

coverage = []
for threshold in [-.5] + [d + .5 for d in range(9)]:
    good = coefficient[1].abs() > 1e-5
    cost = (breakpoints - threshold).abs() + equivalent_digit_residual[480:].amax(0)
    cost[~good] = torch.inf
    chosen = cost.argsort()[:5]
    coverage.append({'desired_threshold': threshold, 'nearest_native_gates': [
        {'unit': int(native[j]), 'slope': float(coefficient[1,j]), 'breakpoint': float(breakpoints[j]),
         'maximum_equivalent_digit_residual': float(equivalent_digit_residual[480:,j].max()),
         'minimum_absolute_format_gate': float(linear_gate[:,j].abs().min()),
         'format_factor_stable_fraction': float(psafe[:,j].double().mean()),
         'word_factor_stable_fraction': float(nsafe[:,j].double().mean())}
        for j in chosen.tolist()]})
report = {'diagnostic_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'word_train_windows':384, 'word_validation_windows':128, 'format_train_sequences':96,
    'format_validation_sequences':416, 'native_units':len(native),
    'format_flag_range':[float(pf.min()),float(pf.max())],
    'original_common_programme_feature_range':[float(common.min()),float(common.max())],
    'word_factor_interval_radius_RMS':float(rms(nerr)), 'format_factor_interval_radius_RMS':float(rms(perr)),
    'word_stable_factor_fraction':float(nsafe.double().mean()),
    'format_stable_factor_fraction':float(psafe.double().mean()),
    'format_mass_trials':masses, 'word_covariance_diagonal_ridge':.001,
    'fields':field_reports, 'native_breakpoint_coverage':coverage,
    'scope':'Activation statistics and whole-format-flag fields only; no candidate, no new native activation evaluation, no answer target',
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (base,base.parent/'circuit.pt',wp,fp,wr,fr)},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT/'artifacts/a274v1_native_gate_arithmetic_reuse.json',report)
print(json.dumps({'summary':{k:v for k,v in report.items() if k not in ('fields','native_breakpoint_coverage','sources','inputs')}}),flush=True)
