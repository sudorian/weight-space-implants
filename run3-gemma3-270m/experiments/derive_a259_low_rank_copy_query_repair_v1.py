"""One original-target causal-score repair within the copier's two Q rows.

Recorded A142 inputs are exact for the local algebra. The intended A240
composition has changed upstream operand readers and is explicitly only a
transfer candidate. No attention probabilities or native activations run.
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
install_base = ROOT / 'experiments/a240v1_causal_metric_queries_on_a191/delta.pt'
refp = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
frefp = ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
sp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
ch, installed, ref, fref = [expand(h, p) for p in (base, install_base, refp, frefp)]
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(7))) or name == 'model.layers.7.input_layernorm.weight':
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
        assert torch.equal(ch.get(name, h[name]), fref.get(name, h[name])), name
pre = 'model.layers.7.self_attn.'
qn, gn, kn, kgn = [pre + s for s in ('q_proj.weight', 'q_norm.weight', 'k_proj.weight', 'k_norm.weight')]
for name in (qn, gn, kn, kgn):
    assert torch.equal(ch.get(name, h[name]), installed.get(name, h[name])), name
del ref, fref
w, e, stress = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep, sp)]
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(refp)
assert stress['checkpoint_sha256'] == sha(frefp)
assert torch.equal(w['inputs'], e['inputs'])

def rms(x):
    return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

def norm(x):
    return x / rms(x)

def fnorm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

inputs = {}
for tag in ('word', 'format'):
    state = w['captures'][f'{tag}.6.raw'] + fnorm(e['captures'][f'{tag}.6.mlp.raw']) * (1 + ch['model.layers.6.post_feedforward_layernorm.weight'])
    check = state + fnorm(e['captures'][f'{tag}.7.attention.raw']) * (1 + ch.get('model.layers.7.post_attention_layernorm.weight', h['model.layers.7.post_attention_layernorm.weight']))
    assert torch.equal(check, w['captures'][f'{tag}.7.raw'])
    inputs[tag] = (fnorm(state) * (1 + h['model.layers.7.input_layernorm.weight'])).double()
TW = inputs['word']; current = inputs['format'][:, :13]
SX = torch.cat([current, stress['captures']['7.input'][:, :13].double()])
T = TW.reshape(-1, 640); nt = 192 * 64
Q0 = ch[qn].double().reshape(4, 256, 640)
QH = h[qn].double().reshape(4, 256, 640)
gain, gh = 1 + ch[gn].double(), 1 + h[gn].double()
K, KH = ch[kn].double(), h[kn].double()
kg, kgh = 1 + ch[kgn].double(), 1 + h[kgn].double()
changed_rows = (ch[qn].view(torch.int32) != h[qn].view(torch.int32)).reshape(4, 256, 640).any(-1)
channels = changed_rows[0].nonzero().flatten()
assert len(channels) == 2 and channels[1] - channels[0] == 128
assert torch.equal(changed_rows, changed_rows[:1].expand_as(changed_rows))
assert (gain[channels].abs() > 1e-8).all()
P = current[:96, 8:13].reshape(-1, 640)
_, singular, right = torch.linalg.svd(P / P.norm(dim=1, keepdim=True), full_matrices=True)
# A256 protected all480 directions; this family also tests four lower PCA ranks.
detected_rank = int((singular > singular[0] * 1e-10).sum())
frequency = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)

def rotate(z):
    angle = torch.arange(z.shape[-2], dtype=torch.float64)[:, None] * frequency
    cs, sn = angle.cos().repeat(1, 2), angle.sin().repeat(1, 2)
    return z * cs + torch.cat([-z[..., 128:], z[..., :128]], -1) * sn

keys = rotate(norm(TW @ K.T) * kg)
host_keys = rotate(norm(TW @ KH.T) * kgh)
source_keys = rotate(norm(SX @ K.T) * kg)
causal = torch.ones(64, 64, dtype=torch.float64).tril()
causal_count = causal.sum(-1)
output_basis = torch.zeros(256, 2, dtype=torch.float64)
output_basis[channels, torch.arange(2)] = 1
rotated_basis = rotate(output_basis.T[:, None].expand(-1, 64, -1)).permute(1, 0, 2)
key_features = torch.einsum('bsc,trc->btsr', keys[:192], rotated_basis) / 16
key_features *= causal[None, :, :, None]
mean_key = key_features.sum(2) / causal_count[None, :, None]
metric = torch.einsum('btsr,btsq->btrq', key_features, key_features) / causal_count[None, :, None, None]
metric -= mean_key[:, :, :, None] * mean_key[:, :, None, :]
metric = metric.reshape(nt, 2, 2)
ridges = [1e-6, 1e-4, .01, 1., 100.]
strengths = [.25, .5, .75, 1., 1.25]
protected_counts = [8, 16, 32, 96, 480]
comparison_path = ROOT / 'artifacts/a256v1_sparse_copy_query_repair.json'
comparison = json.loads(comparison_path.read_text())
global_best = [None] * 4; constraint_reports = []; controls = []
for protected_count in protected_counts:
    null = right[protected_count:].T
    protected_directions = right[:protected_count]
    dimension = null.shape[1]
    assert (protected_directions @ null).abs().max() < 1e-8
    head_reports = []; baseline_mses = []; selected_mses = []
    for head in range(4):
        original_raw = T @ Q0[head].T
        reference_rms = rms(original_raw)
        original_scores = rotate((original_raw / reference_rms * gain).reshape(256, 64, 256)) @ keys.transpose(-1, -2) / 16
        host_scores = rotate(norm(TW @ QH[head].T) * gh) @ host_keys.transpose(-1, -2) / 16
        desired = host_scores[:192] - original_scores[:192]
        desired *= causal
        mean_desired = desired.sum(-1) / causal_count
        target = torch.einsum('btsr,bts->btr', key_features, desired) / causal_count[None, :, None]
        target -= mean_key * mean_desired[:, :, None]
        target = target.reshape(nt, 2)
        X = (T[:nt] @ null) / reference_rms[:nt]
        sd = X.square().mean(0).sqrt().clamp_min(1e-15)
        Z = X / sd
        gram = torch.empty(2, dimension, 2, dimension, dtype=torch.float64)
        for i in range(2):
            for j in range(i, 2):
                block = Z.T @ (Z * metric[:, i, j, None]) / nt
                gram[i, :, j, :] = block; gram[j, :, i, :] = block.T
        gram = gram.reshape(2*dimension, 2*dimension)
        scale = gram.diagonal().clamp_min(1e-30).sqrt()
        standardized = gram / scale[:, None] / scale[None]
        ev, vec = torch.linalg.eigh((standardized + standardized.T) / 2)
        ev = ev.clamp_min(0)
        rhs = vec.T @ ((target.T @ Z / nt).reshape(-1) / scale)

        def quality(matrix):
            scores = rotate(norm(TW[192:] @ matrix.T) * gain) @ keys[192:].transpose(-1, -2) / 16
            difference = scores - host_scores[192:]
            mean = (difference * causal).sum(-1) / causal_count
            variance = ((difference - mean[:, :, None]).square() * causal).sum(-1) / causal_count
            query = rotate(norm(SX @ matrix.T) * gain)
            margins = []
            for position in range(8, 13):
                other = torch.tensor([i for i in range(position + 1) if i != 8])
                source_scores = (query[:, position, None] * source_keys[:, :position + 1]).sum(-1) / 16
                margins.append(source_scores[:, 8] - source_scores[:, other].amax(1))
            gaps = torch.stack(margins, 1)
            return {'validation_centered_causal_score_MSE': float(variance.mean()),
                'validation_score_RMS_p99': float(variance.sqrt().flatten().quantile(.99)),
                'minimum_source_routing_margin': float(gaps.min()),
                'current32_holdback_source_margin': float(gaps[96:128].min()),
                'stress512_source_margin': float(gaps[128:].min())}

        baseline = quality(Q0[head]); baseline_mses.append(baseline['validation_centered_causal_score_MSE'])
        assert all(torch.isfinite(torch.tensor(v)) for v in baseline.values())
        trials = []; best = None
        for ridge in ridges:
            beta = ((vec @ (rhs / (ev + ridge))) / scale).reshape(2, dimension)
            correction = (beta / sd[None]) @ null.T / gain[channels, None]
            real_null = float((protected_directions @ correction.T).abs().max())
            assert real_null < 1e-7
            for strength in strengths:
                matrix = Q0[head].float().clone()
                matrix[channels] = (Q0[head, channels] + strength * correction).float()
                q = quality(matrix.double())
                rounded_delta = matrix[channels].double() - Q0[head, channels]
                rounded_null = float((protected_directions @ rounded_delta.T).abs().max())
                all_format_change = float((SX[:,8:13].reshape(-1,640) @ rounded_delta.T).abs().max())
                rec = {'head': head, 'protected_PC_count': protected_count, 'null_input_dimension': dimension, 'ridge': ridge, 'correction_strength': strength,
                    **q, 'real_format_null_error': real_null, 'rounded_format_query_change': rounded_null,
                    'maximum_query_weight': float(matrix.abs().max()), 'maximum_all_format_raw_query_change': all_format_change}
                rec['eligible_head'] = bool(matrix.isfinite().all() and matrix.abs().max() <= 1e6
                    and rounded_null < 1e-5 and all_format_change < .01 and q['minimum_source_routing_margin'] >= 7.999
                    and q['validation_centered_causal_score_MSE'] < baseline['validation_centered_causal_score_MSE'])
                if protected_count == 480 and ridge == comparison['head_reports'][head]['selected']['ridge'] and strength == comparison['head_reports'][head]['selected']['correction_strength']:
                    expected = comparison['head_reports'][head]['selected']
                    control_error = abs(q['validation_centered_causal_score_MSE'] - expected['validation_centered_causal_score_MSE'])
                    assert control_error < 1e-12
                    controls.append({'head':head,'A256_selected_metric_absolute_error':control_error,'maximum_all_format_raw_query_change':all_format_change})
                trials.append(rec)
                if rec['eligible_head'] and (best is None or q['validation_centered_causal_score_MSE'] < best[0]):
                    best = q['validation_centered_causal_score_MSE'], matrix.clone(), rec
        if best is None:
            selected_mses.append(baseline['validation_centered_causal_score_MSE'])
        else:
            selected_mses.append(best[0])
            if global_best[head] is None or best[0] < global_best[head][0]:
                global_best[head] = (best[0], best[1].clone(), dict(best[-1]))
        head_reports.append({'head': head, 'baseline': baseline, 'trials': trials,
            'selected': None if best is None else best[-1]})
        print(json.dumps({'head': head, 'baseline': baseline, 'selected': head_reports[-1]['selected']}), flush=True)
    constraint_reports.append({'protected_PC_count':protected_count,'null_input_dimension':dimension,'head_reports':head_reports})
selected_matrix = Q0.clone(); selected_mses = []; final_heads = []
for head, best in enumerate(global_best):
    baseline = head_reports[head]['baseline']
    if best is not None:
        selected_matrix[head] = best[1].double(); selected_mses.append(best[0])
    else:
        selected_mses.append(baseline['validation_centered_causal_score_MSE'])
    final_heads.append({'head':head,'baseline':baseline,'selected':None if best is None else best[-1]})
assert len(controls) == 4
head_reports = final_heads
value = selected_matrix.reshape(1024, 640).float()
remaining = torch.arange(256)[~torch.isin(torch.arange(256), channels)]
assert torch.equal(value.reshape(4, 256, 640)[:, remaining], ch[qn].reshape(4, 256, 640)[:, remaining])
bm = json.loads(install_base.with_suffix('.manifest.json').read_text())
total = bm['changed_scalars'] - bm['per_tensor'][qn] + int((value.view(torch.int32) != h[qn].view(torch.int32)).sum())
initial, final = sum(baseline_mses) / 4, sum(selected_mses) / 4
minimum_margin = min((r['selected'] or r['baseline'])['minimum_source_routing_margin'] for r in head_reports)
reference_mse = comparison['combined_candidate']['validation_centered_causal_score_MSE']
assert abs(initial-comparison['combined_candidate']['baseline_centered_causal_score_MSE']) < 1e-12
eligible = bool(final < .9*initial and final < .95*reference_mse and total <= 5016268 and minimum_margin >= 7.999)
combined = {'baseline_centered_causal_score_MSE': initial, 'validation_centered_causal_score_MSE': final,
    'relative_score_MSE_reduction': 1 - final / initial, 'A256_centered_score_MSE':reference_mse, 'relative_MSE_reduction_from_A256':1-final/reference_mse, 'changed_scalars_on_install_base': total,
    'maximum_query_weight': float(value.abs().max()), 'minimum_source_routing_margin': minimum_margin,
    'eligible': eligible}
report = {'analysis_only': True, 'calibration_base_checkpoint': str(base.relative_to(ROOT)),
    'install_base_checkpoint': str(install_base.relative_to(ROOT)),
    'training_windows': 192, 'validation_windows': 64,
    'editable_query_channels_per_head': channels.tolist(),
    'protected_current_format_windows': 96, 'protected_query_positions': [8, 9, 10, 11, 12],
    'protected_rows': len(P), 'numerically_detected_format_rank': detected_rank,
    'protected_PC_count_trials':protected_counts, 'maximum_all_format_raw_query_change_limit':.01,
    'constraint_reports':constraint_reports,'A256_reproduction_controls':controls,
    'current_holdback_format_windows': 32, 'additional_exact_stress_windows': 512,
    'score_metric_ridge_trials': ridges, 'correction_strength_trials': strengths,
    'target': 'Original HOST Q/K centered causal scores minus original A142 Q/K scores at exact A142 inputs; fixed original query raw RMS for the linear solve.',
    'metric': 'Exact causal centered current-key Gram in the two existing sparse query channels; rounded candidates use their actual RMS.',
    'selection': 'Independent minimum validation MSE per head across five protected-PC counts, source margins>=7.999, rounded protected-PC query change<1e-5 and every format raw-query change<0.01; combined gain>=10percent original and>=5percent overA256.',
    'head_reports': head_reports, 'combined_candidate': combined,
    'selected': combined if eligible else None,
    'transfer_caveat': 'A240 has different upstream operand-reader weights. Its Q7/K7/gains are identical, but the A142 inputs are not claimed to be its exact trajectory. Official full scoring is required.',
    'no_softmax_native_activation_execution_gradients_answers_or_iterative_residual_fitting': True,
    'sources': {**comparison['sources'], str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {**comparison['inputs'], **{str(p.relative_to(ROOT)): sha(p) for p in (base, install_base, refp, frefp, wp, ep, sp, comparison_path)}},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a259v1_low_rank_copy_query_repair.json'
atomic_json(out, report)
if eligible:
    atomic_torch(out.with_suffix('.pt'), {'query': value, 'report': report})
print(json.dumps({'selected': report['selected'], 'combined_candidate': combined,
    'peak_kib': report['maximum_resident_set_kib']}), flush=True)
