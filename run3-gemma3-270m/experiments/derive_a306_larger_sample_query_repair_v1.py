"""Original native query reconstruction using a larger recorded word sample.

Source-position constraints only; all native text targets are original HOST
observations. Every covariance solve is independent and closed form. No
attention probability, native nonlinearity, full model, or gradient is run.
"""
import json
import resource
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
original = ROOT / 'experiments/a191v1_partial_reader_output_repair/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
qp = ROOT / 'artifacts/a45_quiet_query3_v1.pt'
control_path = ROOT / 'artifacts/a261v1_paired_HOST_trajectory_query_repair.json'
historical_paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
ch, old, ref = expand(h, base), expand(h, original), expand(h, reference)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name == 'model.layers.3.input_layernorm.weight':
        assert torch.equal(ch.get(name, h[name]), old.get(name, h[name]))
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name]))
del ref
c = torch.load(original.parent / 'circuit.pt', weights_only=True)
selected = torch.tensor(c['native_a_stage']['reader']['native_key_query_layer3']['selected_channels'])
ordinary = torch.arange(256)[~torch.isin(torch.arange(256), selected)]
assert len(selected) == 64 and len(ordinary) == 192
w, e, host = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep, hp)]
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(reference)
assert torch.equal(w['inputs'], e['inputs']) and torch.equal(host['inputs'], w['inputs'][:256])

def rms(x): return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def norm(x): return x / rms(x)
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

state = h['model.embed_tokens.weight'][host['inputs']] * torch.tensor(640**.5, dtype=torch.float32)
for layer in range(18):
    if layer == 3: TH = (fnorm(state) * (1 + h['model.layers.3.input_layernorm.weight'])).double()
    for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
        state = state + fnorm(host['captures'][f'{layer}.{branch}.raw']) * (1 + h[f'model.layers.{layer}.{part}.weight'])
assert torch.equal(state, host['captures']['finalraw']); del state
inputs = {}
for tag in ('word', 'format'):
    state = w['captures'][f'{tag}.2.raw'] + fnorm(e['captures'][f'{tag}.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
    inputs[tag] = (fnorm(state) * (1 + h['model.layers.3.input_layernorm.weight'])).double()
TW = inputs['word']; T = TW.reshape(-1, 640); current = inputs['format'][:, :9]
historical = torch.cat([torch.load(p, weights_only=True, mmap=True)['captures']['3.input'][:, :9] for p in historical_paths]).double()
SX = torch.cat([current, historical]); nt = 192 * 64
pre = 'model.layers.3.self_attn.'
qn, gn, kn, kgn = [pre + s for s in ('q_proj.weight', 'q_norm.weight', 'k_proj.weight', 'k_norm.weight')]
for name in (kn, kgn): assert torch.equal(ch.get(name, h[name]), h[name])
assert torch.equal(ch[gn], old[gn])
QH = h[qn].double().reshape(4, 256, 640)
Q0 = old[qn].double().reshape(4, 256, 640)
Qleader = ch[qn].double().reshape(4, 256, 640)
gh = 1 + h[gn].double(); g0 = 1 + old[gn].double()
K = h[kn].double(); kg = 1 + h[kgn].double()
quiet = torch.load(qp, weights_only=True)['query'].double()
old_mean = historical[:1024, 8].mean(0)
qnull = quiet / (old_mean @ quiet)
B = QH - (old_mean @ QH.transpose(-1, -2))[:, :, None] * qnull
native_program_rows = B[:, selected] * (gh[selected] / g0[selected])[None, :, None]
P = torch.zeros_like(Q0); P[:, selected] = Q0[:, selected] - native_program_rows
amplitude = (P @ quiet) / quiet.square().sum()
decomposition_error = float((P - amplitude[:, :, None] * quiet).abs().max())
assert decomposition_error < 1e-6
frequency = 10000.**(-torch.arange(128, dtype=torch.float64) / 128)

def rotate(z):
    angles = torch.arange(z.shape[-2], dtype=torch.float64)[:, None] * frequency
    cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    return z * cs + torch.cat([-z[..., 128:], z[..., :128]], -1) * sn

rotated_keys = rotate(norm(TW @ K.T) * kg)
host_keys = rotate(norm(TH @ K.T) * kg)
host_queries = torch.stack([norm(TH @ QH[i].T) * gh for i in range(4)])
host_scores = rotate(host_queries) @ host_keys.transpose(-1, -2)[None] / 16
same_scores = torch.stack([rotate(norm(TW @ QH[i].T) * gh) @ rotated_keys.transpose(-1, -2) / 16 for i in range(4)])
angles = (torch.arange(9, dtype=torch.float64) - 8)[:, None] * frequency
cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
SK = norm(SX @ K.T) * kg
KR = SK * cs + torch.cat([-SK[..., 128:], SK[..., :128]], -1) * sn
causal = torch.ones(64, 64, dtype=torch.float64).tril(); count_causal = causal.sum(-1)
positions = [3, 2, 1, 0]

def centered_mse(delta):
    mean = (delta * causal).sum(-1) / count_causal
    return float((((delta - mean[..., None]).square() * causal).sum(-1) / count_causal).mean())

def quality(query, gain):
    post = torch.stack([norm(TW[192:] @ query[i].T) * gain for i in range(4)])
    scores = rotate(post) @ rotated_keys[192:].transpose(-1, -2)[None] / 16
    gaps = []
    for head, pos in enumerate(positions):
        q = norm(SX[:, 8] @ query[head].T) * gain
        ss = (KR * q[:, None]).sum(-1) / 16
        others = torch.tensor([i for i in range(9) if i != pos])
        gaps.append(ss[:, pos] - ss[:, others].amax(1))
    gaps = torch.stack(gaps)
    return {'validation_paired_HOST_causal_score_MSE': centered_mse(scores - host_scores[:, 192:]),
        'validation_same_input_HOST_causal_score_MSE': centered_mse(scores - same_scores[:, 192:]),
        'validation_paired_HOST_query_RMS': float((post - host_queries[:, 192:]).square().mean().sqrt()),
        'minimum_source_routing_margin': float(gaps.min()),
        'current32_source_margin': float(gaps[:, 96:128].min()), 'historical512_source_margin': float(gaps[:, -512:].min())}

old_word_path = ROOT / 'artifacts/a119v1_current_word.pt'
old_word_reference = ROOT / 'experiments/a117v1_six_unit_carry/delta.pt'
old_word = torch.load(old_word_path, weights_only=True, mmap=True)
assert old_word['checkpoint_sha256'] == sha(old_word_reference)
old_ch = expand(h, old_word_reference)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name == 'model.layers.3.input_layernorm.weight':
        assert torch.equal(old_ch.get(name, h[name]), ch.get(name, h[name])), name
del old_ch
old_tokens = {tuple(row.tolist()) for row in old_word['inputs']}
duplicate_windows = sum(tuple(row.tolist()) in old_tokens for row in w['inputs'][:256])
assert duplicate_windows == 0
TT = old_word['captures']['3.input'].double()
assert TT.shape == (512, 64, 640)
T = TT.reshape(-1, 640); nt = len(T)
training_keys = rotate(norm(TT @ K.T) * kg)
Pnull = current[:96, 8]; Pn = Pnull / Pnull.norm(dim=1, keepdim=True)
_, singular, right = torch.linalg.svd(Pn, full_matrices=True)
rank = int((singular > singular[0] * 1e-10).sum()); null = right[rank:].T
assert rank == 96 and (Pnull @ null).abs().max() < 1e-8
input_counts = [8, 16, 32, 64, 128]; output_count = 16; basis_ridge = .001
ridges = [1e-6, 1e-4, .01, 1., 100.]; strengths = [.25, .5, .75, 1., 1.25]
gain = g0; active = gain.abs() > 1e-8
trials = []; head_reports = []; selected_matrix = Qleader.clone()
leader_metrics = []; selected_metrics = []; any_head_selected = False
print(json.dumps({'training_windows': 512, 'validation_windows': 256, 'exact_duplicate_windows': duplicate_windows}), flush=True)
for head in range(4):
    original_raw=T@Q0[head].T;reference_rms=rms(original_raw)
    old_post=original_raw/reference_rms*gain
    host_post=norm(T@QH[head].T)*gh
    response=host_post-old_post
    target_scores=rotate(response.reshape(512,64,256))@training_keys.transpose(-1,-2)/16
    original_host_scores=rotate(host_post.reshape(512,64,256))@training_keys.transpose(-1,-2)/16
    # Canonical input directions depend on the original response covariance,
    # never a fitted candidate's errors. One fixed ridge stabilizes whitening.
    X=(T[:nt]@null)/reference_rms[:nt]
    sd=X.square().mean(0).sqrt().clamp_min(1e-15);Z=X/sd
    C=Z.T@Z/nt
    ev,vec=torch.linalg.eigh((C+C.T)/2);ev=ev.clamp_min(0)
    inverse_root=(vec/(ev+basis_ridge).sqrt()[None])@vec.T
    cross=inverse_root@(Z.T@response[:nt]/nt)
    canonical,spectrum,_=torch.linalg.svd(cross,full_matrices=False)
    coordinate_map=inverse_root@canonical[:,:max(input_counts)]/sd[:,None]
    input_map=null@coordinate_map
    features=X@coordinate_map
    output_cov=response[:nt,active].T@response[:nt,active]/nt
    oe,ov=torch.linalg.eigh((output_cov+output_cov.T)/2)
    output_basis=torch.zeros(256,output_count,dtype=torch.float64)
    output_basis[active]=ov[:,-output_count:].flip(1)
    # A query correction expressed before RoPE sees rotated keys projected
    # onto R(position)*output_basis. Centering removes score-row constants.
    rotated_basis=rotate(output_basis.T[:,None,:].expand(-1,64,-1)).permute(1,0,2)
    key_features=torch.einsum('bsc,trc->btsr',training_keys,rotated_basis)/16
    key_features*=causal[None,:,:,None]
    mean_key=key_features.sum(2)/count_causal[None,:,None]
    H=torch.einsum('btsr,btsq->btrq',key_features,key_features)/count_causal[None,:,None,None]
    H-=mean_key[:,:,:,None]*mean_key[:,:,None,:]
    D=target_scores*causal
    mean_D=D.sum(-1)/count_causal
    weighted_target=torch.einsum('btsr,bts->btr',key_features,D)/count_causal[None,:,None]
    weighted_target-=mean_key*mean_D[:,:,None]
    H=H.reshape(nt,output_count,output_count)
    weighted_target=weighted_target.reshape(nt,output_count)
    full_gram=torch.empty(output_count,max(input_counts),output_count,max(input_counts),dtype=torch.float64)
    for r in range(output_count):
        for s in range(r,output_count):
            block=features.T@(features*H[:,r,s,None])/nt
            full_gram[r,:,s,:]=block;full_gram[s,:,r,:]=block.T
    full_rhs=weighted_target.T@features/nt
    del key_features,H,weighted_target,X,Z,C,ev,vec,cross,inverse_root,canonical

    def quality(matrix):
        post = norm(TW @ matrix.T) * gain
        scores = rotate(post) @ rotated_keys.transpose(-1, -2) / 16
        sq = norm(SX[:, 8] @ matrix.T) * gain
        ss = (KR * sq[:, None]).sum(-1) / 16
        others = torch.tensor([s for s in range(9) if s != positions[head]])
        gaps = ss[:, positions[head]] - ss[:, others].amax(1)
        return {'validation_paired_HOST_causal_score_MSE': centered_mse(scores - host_scores[head]),
            'validation_same_input_HOST_causal_score_MSE': centered_mse(scores - same_scores[head]),
            'validation_paired_HOST_query_RMS': float((post - host_queries[head]).square().mean().sqrt()),
            'minimum_source_routing_margin': float(gaps.min()),
            'heldback32_source_routing_margin': float(gaps[96:128].min())}
    baseline = quality(Qleader[head]); original_quality = quality(Q0[head])
    leader_metrics.append(baseline)
    best = None; head_trials = []
    print(json.dumps({'head': head, 'leader': baseline, 'original': original_quality,
        'canonical_response_spectrum': spectrum[:16].tolist()}), flush=True)
    for count in input_counts:
        gram = full_gram[:, :count, :, :count].reshape(output_count * count, output_count * count)
        scale = gram.diagonal().clamp_min(1e-30).sqrt()
        standardized = gram / scale[:, None] / scale[None]
        ce, cv = torch.linalg.eigh((standardized + standardized.T) / 2); ce = ce.clamp_min(0)
        rhs = cv.T @ (full_rhs[:, :count].reshape(-1) / scale)
        for ridge in ridges:
            beta = ((cv @ (rhs / (ce + ridge))) / scale).reshape(output_count, count)
            post_correction = output_basis @ beta @ input_map[:, :count].T
            correction = torch.zeros_like(Q0[head]); correction[active] = post_correction[active] / gain[active, None]
            null_error = float((Pnull @ correction.T).abs().max()); assert null_error < 1e-7
            for strength in strengths:
                matrix = (Q0[head] + strength * correction).float()
                q = quality(matrix.double())
                rounded_null = float((Pnull @ (matrix.double() - Q0[head]).T).abs().max())
                rec = {'head': head, 'canonical_input_dimension': count, 'fixed_output_dimension': output_count,
                    'score_metric_ridge': ridge, 'correction_strength': strength, **q,
                    'real_format_null_error': null_error, 'rounded_format_query_change': rounded_null,
                    'maximum_query_weight': float(matrix.abs().max())}
                rec['eligible_head'] = bool(matrix.isfinite().all() and matrix.abs().max() <= 1e6
                    and rounded_null < 1e-5 and q['minimum_source_routing_margin'] >= 7.999
                    and q['validation_paired_HOST_causal_score_MSE'] < baseline['validation_paired_HOST_causal_score_MSE']
                    and q['validation_same_input_HOST_causal_score_MSE'] <= baseline['validation_same_input_HOST_causal_score_MSE']
                    and q['validation_paired_HOST_query_RMS'] <= 1.05 * baseline['validation_paired_HOST_query_RMS'])
                head_trials.append(rec); trials.append(rec)
                objective = q['validation_paired_HOST_causal_score_MSE']
                if rec['eligible_head'] and (best is None or objective < best[0]):
                    best = objective, matrix.clone(), rec
        print(json.dumps({'head': head, 'canonical_input_dimension': count, 'selected': None if best is None else best[-1]}), flush=True)
    if best is not None:
        selected_matrix[head] = best[1].double(); selected_metrics.append(best[-1]); any_head_selected = True
    else:
        selected_metrics.append(baseline)
    head_reports.append({'head': head, 'baseline': baseline, 'original': original_quality,
        'selected': None if best is None else best[-1], 'trials': head_trials,
        'original_canonical_response_spectrum': spectrum.tolist(),
        'original_response_energy_fraction_in_output_basis': float(oe[-output_count:].sum() / oe.sum().clamp_min(1e-30))})
    del full_gram, full_rhs, features, input_map, coordinate_map, output_cov, output_basis
    del original_raw, reference_rms, old_post, host_post, response, target_scores, original_host_scores
bm = json.loads(base.with_suffix('.manifest.json').read_text())
value = selected_matrix.reshape(1024, 640).float()
changed = bm['changed_scalars'] - bm['per_tensor'][qn] + int((value.view(torch.int32) != h[qn].view(torch.int32)).sum())
keys = ['validation_paired_HOST_causal_score_MSE', 'validation_same_input_HOST_causal_score_MSE']
leader_mean = {key: sum(r[key] for r in leader_metrics) / 4 for key in keys}
selected_mean = {key: sum(r[key] for r in selected_metrics) / 4 for key in keys}
eligible = bool(any_head_selected and all(selected_mean[k] <= .95 * leader_mean[k] for k in keys)
    and changed <= 5016268 and value.isfinite().all() and value.abs().max() <= 1e6)
selected = {**selected_mean, 'baseline': leader_mean, 'changed_scalars': changed,
    'maximum_query_weight': float(value.abs().max()), 'eligible': eligible,
    'head_selections': [r['selected'] for r in head_reports]}
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'training_windows': 512, 'validation_windows': 256, 'exact_duplicate_windows': duplicate_windows,
    'protected_current_format_rows': 96, 'protected_format_rank': rank,
    'canonical_input_dimension_trials': input_counts, 'fixed_output_dimension': output_count,
    'fixed_canonical_whitening_ridge': basis_ridge, 'score_metric_ridge_trials': ridges, 'correction_strength_trials': strengths,
    'target': 'Original same-input HOST minus original A191 normalized queries, fixed original A191 raw RMS',
    'selection': 'Lowest paired HOST MSE per head with no worse same-input MSE than A259, query RMS<=1.05A259 and safe sources; combined>=5percent improvement on both score metrics',
    'head_reports': head_reports, 'selected': selected if eligible else None, 'combined_candidate': selected,
    'scope_limit': 'Previously observed calibration windows; no model, softmax, native activation or answer-target execution; official score still required',
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__), 'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, original, reference, wp, ep, hp, qp, control_path, old_word_path, old_word_reference] + historical_paths},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a306v1_larger_sample_query_repair.json'
atomic_json(out, report)
if eligible: atomic_torch(out.with_suffix('.pt'), {'query': value, 'report': report})
print(json.dumps({'combined_candidate': selected, 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
