"""Independent branch-gain repairs against original paired HOST states.

Each alternative changes one post-branch norm only. Targets are the original
HOST state after that branch, with recorded parent input and branch output
fixed. Arithmetic output gains stay fixed. Alternatives are never composed;
no model replay, nonlinear activation, gradient, answer, or residual refit.
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
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'
lp = ROOT / 'artifacts/a140v1_branch_gain_layout.json'
ch, ref = expand(h, base), expand(h, reference)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(17))) or name.startswith('model.layers.17.self_attn.') or name in ('model.layers.17.input_layernorm.weight', 'model.layers.17.post_attention_layernorm.weight'):
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
e, host = [torch.load(p, weights_only=True, mmap=True) for p in (ep, hp)]
layout = json.loads(lp.read_text())
assert e['checkpoint_sha256'] == sha(reference)
assert host['checkpoint_sha256'] == sha(ROOT / 'host/ckpt.pt')
assert torch.equal(e['inputs'][:256], host['inputs'])

def fnorm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

def norm(x):
    return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

state = h['model.embed_tokens.weight'][host['inputs']] * torch.tensor(640 ** .5, dtype=torch.float32)
teacher = state.clone()
nt = 192 * 64
ridges = [1e-6, 1e-4, .01, 1., 100.]
strengths = [.25, .5, .75, 1., 1.25]
reports, selected = [], {}
bm = json.loads(base.with_suffix('.manifest.json').read_text())
for layer in range(18):
    for branch, part in [('attention', 'post_attention_layernorm'), ('mlp', 'post_feedforward_layernorm')]:
        name = f'model.layers.{layer}.{part}.weight'
        raw = e['captures'][f'word.{layer}.{branch}.raw']
        hr = host['captures'][f'{layer}.{branch}.raw']
        teacher = teacher + fnorm(hr) * (1 + h[name])
        before = state
        state = before + fnorm(raw) * (1 + ref.get(name, h[name]))
        if (branch == 'attention' and layer in (2, 3, 7)) or (layer == 17 and branch == 'mlp'):
            continue
        parameter0 = ch.get(name, h[name])
        assert torch.equal(parameter0, ref.get(name, h[name]))
        gamma = 1 + parameter0.double()
        z = fnorm(raw).reshape(-1, 640).double()
        b = before.reshape(-1, 640).double()
        t = teacher.reshape(-1, 640).double()
        trms = (t.square().mean(1, keepdim=True) + 1e-6).sqrt()
        fixed = torch.zeros(640, dtype=torch.bool)
        if branch == 'mlp':
            units = torch.tensor(layout['programme_units'].get(str(layer), []), dtype=torch.long)
            if len(units):
                dn = f'model.layers.{layer}.mlp.down_proj.weight'
                fixed = ch.get(dn, h[dn])[:, units].ne(0).any(1)
        X = z[:nt] / trms[:nt]
        Y = (t[:nt] - b[:nt] - z[:nt] * gamma) / trms[:nt]
        energy = X.square().mean(0)
        rhs = (X * Y).mean(0)
        fr = e['captures'][f'format.{layer}.{branch}.raw'].double()
        fz = norm(fr)

        def quality(parameter):
            gain = 1 + parameter.double()
            out = b[nt:] + z[nt:] * gain
            return {
                'validation_paired_HOST_state_relative_rms': float(((out - t[nt:]) / trms[nt:]).square().mean().sqrt()),
                'validation_paired_HOST_direction_rms': float((norm(out) - norm(t[nt:])).square().mean().sqrt()),
                'maximum_local_format_post_change': float((fz * (gain - gamma)).abs().max()),
                'maximum_native_gain_change': float((gain - gamma).abs().max()),
            }

        baseline = quality(parameter0)
        trials, best = [], None
        for ridge in ridges:
            correction = rhs / (energy * (1 + ridge)).clamp_min(1e-30)
            correction[fixed | (energy < 1e-30)] = 0
            for strength in strengths:
                parameter = (parameter0.double() + strength * correction).float()
                parameter[fixed] = parameter0[fixed]
                q = quality(parameter)
                total = bm['changed_scalars'] - bm['per_tensor'].get(name, 0)
                total += int((parameter.view(torch.int32) != h[name].view(torch.int32)).sum())
                row = {'ridge': ridge, 'correction_strength': strength, **q,
                       'total_changed_scalars': total, 'maximum_parameter': float(parameter.abs().max())}
                row['eligible'] = bool(parameter.isfinite().all() and parameter.abs().max() <= 1e6 and total <= 5016268
                    and q['maximum_local_format_post_change'] < 1
                    and q['validation_paired_HOST_state_relative_rms'] < .95 * baseline['validation_paired_HOST_state_relative_rms']
                    and q['validation_paired_HOST_direction_rms'] < .95 * baseline['validation_paired_HOST_direction_rms'])
                trials.append(row)
                if row['eligible'] and (best is None or q['validation_paired_HOST_state_relative_rms'] < best[0]):
                    best = q['validation_paired_HOST_state_relative_rms'], parameter.clone(), row
        item = {'layer': layer, 'branch': branch, 'parameter_name': name, 'baseline': baseline,
                'fixed_arithmetic_gain_rows': fixed.nonzero().flatten().tolist(), 'trials': trials,
                'selected': None if best is None else best[-1]}
        reports.append(item)
        if best is not None:
            selected[name] = best[1]
        print(json.dumps({k: item[k] for k in ('layer', 'branch', 'baseline', 'selected')}), flush=True)
        del z, b, t, trms, X, Y, fr, fz
assert torch.equal(teacher, host['captures']['finalraw'])
assert torch.equal(state, e['captures']['word.finalraw'])
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'training_windows': 192, 'validation_windows': 64, 'ridge_trials': ridges,
    'correction_strength_trials': strengths, 'branches': reports, 'selected_tensor_names': list(selected),
    'single_branch_alternatives_never_composed': True,
    'target': 'Original paired HOST post-branch state with recorded current parent pre-branch state and native branch output fixed',
    'format_caveat': 'Exact local change only; downstream arithmetic and text require official scoring',
    'original_HOST_and_recorded_parent_final_reconstruction_bit_identical': True,
    'no_model_nonlinear_replay_gradients_answers_or_iterative_fit': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, reference, ep, hp, lp)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a197v1_single_branch_state_gains.json'
atomic_json(out, report)
if selected:
    atomic_torch(out.with_suffix('.pt'), {'weights': selected, 'report': report})
print(json.dumps({'selected_tensor_names': list(selected), 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
