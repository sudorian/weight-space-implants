"""Direct one-shot native query-gain repair in the causal QK score metric.

Each layer is an independent alternative. Fixed normalized Q and native K
make every causal score affine in the 256 shared query gains. Arithmetic
channels stay fixed. Chunked normal equations target original HOST scores
on unlabeled calibration inputs; no probabilities or native activations are
replayed, and no previous candidate residual is fitted again.
"""
import json, resource, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state(); base = ROOT / 'experiments/a189v1_partial_native_value_decoder/delta.pt'
ch = expand(h, base)
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
refp = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
refs = expand(h, refp)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name == 'model.layers.3.input_layernorm.weight':
        assert torch.equal(ch.get(name, h[name]), refs.get(name, h[name])), name
del refs
w, e = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep)]
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(refp)
assert torch.equal(w['inputs'], e['inputs'])
history_paths = [ROOT / f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a', 'syntax_stress', 'syntax_c')]
meta_paths = {2: ROOT / 'experiments/a27v1_query_energy2/circuit.pt',
              3: ROOT / 'experiments/a45v2_query_energy/circuit.pt'}
query_reference_paths = {2: meta_paths[2].parent / 'delta.pt',
                         3: ROOT / 'experiments/a61v1_query_gain_allocation/delta.pt'}
metric_reference_paths = {2: ROOT / 'artifacts/a192v1_current_b_query_energy.json',
                          3: ROOT / 'artifacts/a193v1_format_null_native_query.json'}

def fnorm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

def norm(x):
    return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

def half(z):
    return torch.cat([-z[..., 128:], z[..., :128]], -1)

freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
pos = torch.arange(64, dtype=torch.float64)
angles = (pos[None, :] - pos[:, None])[:, :, None] * freq
cs, sn = angles.cos().repeat(1, 1, 2), angles.sin().repeat(1, 1, 2)
causal = torch.ones(64, 64, dtype=torch.float64).tril()
counts = causal.sum(-1)
weight = causal / counts[:, None].sqrt()
ridges = [1e-6, 1e-4, .01, 1., 100.]
strengths = [.25, .5, .75, 1., 1.25]
bm = json.loads(base.with_suffix('.manifest.json').read_text())
reports, selected = {}, {}
for layer in (2, 3):
    pre = f'model.layers.{layer}.self_attn.'
    qn, gn, kn, kgn = [pre + s for s in ('q_proj.weight', 'q_norm.weight', 'k_proj.weight', 'k_norm.weight')]
    qref = expand(h, query_reference_paths[layer])
    assert torch.equal(ch[qn], qref[qn]) and torch.equal(ch[gn], qref[gn])
    del qref
    meta = torch.load(meta_paths[layer], weights_only=True)
    if layer == 3:
        # A61 changed Q3/gains while retaining A45's programme channel set.
        # Its immutable manifest records the exact source of that metadata.
        qm = json.loads(query_reference_paths[layer].with_suffix('.manifest.json').read_text())
        assert qm['inputs'][str(meta_paths[layer].relative_to(ROOT))] == sha(meta_paths[layer])
    programme = torch.tensor(meta[f'native_key_query_layer{layer}']['selected_channels'])
    free = torch.tensor([i for i in range(256) if i not in programme.tolist()])
    Q, QH = ch[qn].double().reshape(4, 256, 640), h[qn].double().reshape(4, 256, 640)
    gain0, GH = 1 + ch[gn].double(), 1 + h[gn].double()
    K, KG = h[kn].double(), 1 + h[kgn].double()
    assert torch.equal(ch.get(kn, h[kn]), h[kn]) and torch.equal(ch.get(kgn, h[kgn]), h[kgn])
    inputs = {}
    for tag in ('word', 'format'):
        state = w['captures'][f'{tag}.{layer-1}.raw'] + fnorm(e['captures'][f'{tag}.{layer-1}.mlp.raw']) * (1 + ch[f'model.layers.{layer-1}.post_feedforward_layernorm.weight'])
        inputs[tag] = (fnorm(state) * (1 + h[f'model.layers.{layer}.input_layernorm.weight'])).double()
    TW = inputs['word']; SX = inputs['format'][:, :9]
    if layer == 3:
        historical = torch.cat([torch.load(p, weights_only=True, mmap=True)['captures']['3.input'][:, :9] for p in history_paths]).double()
        SX = torch.cat([SX, historical])
    U = norm(torch.einsum('btc,hjc->bhtj', TW, Q))
    HQ = norm(torch.einsum('btc,hjc->bhtj', TW, QH)) * GH
    WK = norm(TW @ K.T) * KG

    def relative_keys(keys):
        return keys[:, None] * cs[None] + half(keys)[:, None] * sn[None]

    # Independently verify the relative-angle linear score identity against
    # separately rotated Q and K, on two saved calibration windows.
    aq = pos[:, None] * freq
    c0, s0 = aq.cos().repeat(1, 2), aq.sin().repeat(1, 2)
    u0 = U[:2] * gain0
    rq = u0 * c0 + half(u0) * s0
    rk = WK[:2] * c0 + half(WK[:2]) * s0
    direct = rq @ rk[:, None].transpose(-1, -2) / 16
    relative = torch.einsum('bhtc,btsc->bhts', u0, relative_keys(WK[:2])) / 16
    identity_error = float((direct - relative).abs().max())
    assert identity_error < 1e-10
    del direct, relative, rq, rk, u0

    gram = torch.zeros(len(free), len(free), dtype=torch.float64)
    rhs = torch.zeros(len(free), dtype=torch.float64)
    for start in range(0, 192, 2):
        end = start + 2
        keys = relative_keys(WK[start:end])
        X = U[start:end, :, :, None, free] * keys[:, None, :, :, free] / 16
        y = torch.einsum('bhtc,btsc->bhts', HQ[start:end] - U[start:end] * gain0, keys) / 16
        X = (X - (X * causal[None, None, :, :, None]).sum(-2, keepdim=True) / counts[None, None, :, None, None]) * weight[None, None, :, :, None]
        y = (y - (y * causal).sum(-1, keepdim=True) / counts[:, None]) * weight
        flat = X.reshape(-1, len(free)); target = y.reshape(-1)
        gram += flat.T @ flat; rhs += flat.T @ target
        if end % 48 == 0:
            print(json.dumps({'layer': layer, 'normal_equation_windows': end}), flush=True)
    gram /= 192 * 4 * 64; rhs /= 192 * 4 * 64
    standard = gram.diagonal().sqrt().clamp_min(1e-15)
    C = gram / (standard[:, None] * standard[None]); b = rhs / standard
    ev, V = torch.linalg.eigh((C + C.T) / 2); ev = ev.clamp_min(0); projected_rhs = V.T @ b
    del keys, X, y, flat, target, gram, rhs, C
    validation_keys = relative_keys(WK[192:])
    host_scores = torch.einsum('bhtc,btsc->bhts', HQ[192:], validation_keys) / 16
    angles9 = (torch.arange(9, dtype=torch.float64) - 8)[:, None] * freq
    SK = norm(SX @ K.T) * KG
    KR = SK * angles9.cos().repeat(1, 2) + half(SK) * angles9.sin().repeat(1, 2)
    SQ = norm(torch.einsum('nc,hjc->nhj', SX[:, 8], Q))
    positions = [7, 6, 5, 4] if layer == 2 else [3, 2, 1, 0]

    def quality(parameter):
        gain = 1 + parameter.double()
        scores = torch.einsum('bhtc,btsc->bhts', U[192:] * gain, validation_keys) / 16
        difference = scores - host_scores
        mean = (difference * causal).sum(-1) / counts
        variance = ((difference - mean[..., None]).square() * causal).sum(-1) / counts
        sq = SQ * gain; margins, current_margins = [], []
        for head, source_position in enumerate(positions):
            score = (KR * sq[:, head, None]).sum(-1) / 16
            others = torch.tensor([j for j in range(9) if j != source_position])
            gap = score[:, source_position] - score[:, others].amax(1)
            margins.append(float(gap.min())); current_margins.append(float(gap[:128].min()))
        return {'validation_centered_causal_score_mse': float(variance.mean()),
            'validation_query_score_rms_p99': float(variance.sqrt().flatten().quantile(.99)),
            'all_source_routing_margins': margins, 'current128_source_routing_margins': current_margins}

    baseline = quality(ch[gn]); trials, best = [], None
    prior_baseline = json.loads(metric_reference_paths[layer].read_text())['baseline']['validation_centered_causal_score_mse']
    baseline_identity_error = abs(baseline['validation_centered_causal_score_mse'] - prior_baseline)
    assert baseline_identity_error < 1e-12
    for ridge in ridges:
        correction = (V @ (projected_rhs / (ev + ridge))) / standard
        for strength in strengths:
            parameter = ch[gn].clone()
            parameter[free] = (parameter[free].double() + strength * correction).float()
            assert torch.equal(parameter[programme], ch[gn][programme])
            q = quality(parameter)
            total = bm['changed_scalars'] - bm['per_tensor'][gn]
            total += int((parameter.view(torch.int32) != h[gn].view(torch.int32)).sum())
            row = {'ridge': ridge, 'correction_strength': strength, **q,
                'maximum_parameter': float(parameter.abs().max()), 'total_changed_scalars': total,
                'maximum_native_gain_change': float((parameter[free] - ch[gn][free]).abs().max())}
            row['eligible'] = bool(parameter.isfinite().all() and parameter.abs().max() <= 1e6 and total <= 5016268
                and min(q['all_source_routing_margins']) >= 8 - 1e-3
                and q['validation_centered_causal_score_mse'] < .9 * baseline['validation_centered_causal_score_mse'])
            trials.append(row)
            if row['eligible'] and (best is None or q['validation_centered_causal_score_mse'] < best[0]):
                best = q['validation_centered_causal_score_mse'], parameter.clone(), row
    item = {'layer': layer, 'parameter_name': gn, 'baseline': baseline, 'programme_channels_fixed': programme.tolist(),
        'native_gain_channels': free.tolist(), 'relative_RoPE_score_identity_error': identity_error,
        'independent_prior_score_metric_error': baseline_identity_error,
        'routing_current_windows': 128, 'routing_historical_windows': len(SX) - 128,
        'trials': trials, 'selected': None if best is None else best[-1]}
    reports[str(layer)] = item
    if best is not None:
        selected[gn] = best[1]
    print(json.dumps({'layer': layer, 'baseline': baseline, 'selected': item['selected']}), flush=True)
    del validation_keys, HQ, U, WK

report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)), 'layers': reports,
    'single_layer_alternatives_never_composed': True, 'training_windows': 192, 'validation_windows': 64,
    'ridge_trials': ridges, 'correction_strength_trials': strengths, 'selected_tensor_names': list(selected),
    'target': 'Original same-input HOST centered causal QK scores; fixed query/key weights and raw normalizers',
    'no_probabilities_native_nonlinear_replay_gradients_answers_or_iterative_fit': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, wp, ep, refp] + history_paths + list(meta_paths.values()) + list(query_reference_paths.values()) + [p.with_suffix('.manifest.json') for p in query_reference_paths.values()] + list(metric_reference_paths.values())},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a198v1_native_query_score_gains.json'; atomic_json(out, report)
if selected:
    atomic_torch(out.with_suffix('.pt'), {'weights': selected, 'report': report})
print(json.dumps({'selected_tensor_names': list(selected), 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
