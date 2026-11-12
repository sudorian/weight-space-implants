"""Recover compact factual reader3 mixtures from retained official captures.

This is a unique linear inverse of observed output projections. It does not
evaluate attention probabilities. A capture with observed heads checks the
inverse independently before the current word/format records are saved.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

def recover(inputs, raw, value_weight, output_weight, label, known_heads=None):
    values = (inputs @ value_weight.T).double()
    mixtures, ranks, ratios = [], [], []
    error2, reference2, head_error2, head_reference2, future_max = 0., 0., 0., 0., 0.
    for i in range(len(inputs)):
        matrix = torch.einsum('ohv,sv->ohs', output_weight, values[i]).reshape(640, -1)
        fit = torch.linalg.lstsq(matrix, raw[i].double().T, rcond=1e-10, driver='gelsd')
        coefficient = fit.solution.T.reshape(inputs.shape[1], 4, inputs.shape[1]).permute(1, 0, 2)
        heads = coefficient @ values[i]
        reproduced = torch.einsum('htv,ohv->to', heads, output_weight)
        error2 += float((reproduced - raw[i]).square().sum()); reference2 += float(raw[i].double().square().sum())
        if known_heads is not None:
            head_error2 += float((heads - known_heads[i]).square().sum()); head_reference2 += float(known_heads[i].square().sum())
        ranks.append(int(fit.rank)); ratios.append(float(fit.singular_values[-1] / fit.singular_values[0]))
        future = torch.ones(inputs.shape[1], inputs.shape[1], dtype=torch.bool).triu(1)
        future_max = max(future_max, float(coefficient[:, future].abs().max()))
        mixtures.append((coefficient @ inputs[i].double()).float())
        if (i + 1) % 64 == 0: print(json.dumps({'label': label, 'windows': i + 1}), flush=True)
    report = {'windows': len(inputs), 'tokens_per_window': inputs.shape[1], 'minimum_rank': min(ranks),
        'required_rank': 4 * inputs.shape[1], 'minimum_singular_ratio': min(ratios),
        'relative_raw_error': (error2 / reference2) ** .5, 'maximum_future_coefficient': future_max}
    if known_heads is not None: report['relative_known_head_error'] = (head_error2 / head_reference2) ** .5
    assert min(ranks) == report['required_rank'] and report['relative_raw_error'] < 1e-5 and future_max < .002
    return torch.stack(mixtures), report

def main():
    h = host_state(); base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'; ch = expand(h, base)
    wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'; ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
    oldp = ROOT / 'artifacts/a119v1_current_word.pt'; oldbase = ROOT / 'experiments/a117v1_six_unit_carry/delta.pt'
    captured_base = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
    w, e, old = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep, oldp)]
    assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(captured_base)
    assert torch.equal(w['inputs'], e['inputs'])
    reference = expand(h, captured_base)
    for name in h:
        if name.startswith(tuple(f'model.layers.{i}.' for i in range(4))):
            assert torch.equal(ch.get(name, h[name]), reference.get(name, h[name])), name
    del reference
    pre = 'model.layers.3.'; vn = pre + 'self_attn.v_proj.weight'
    O = h[pre + 'self_attn.o_proj.weight'].double().reshape(640, 4, 256)
    oldstate = expand(h, oldbase); oldV = oldstate[vn]; del oldstate
    known = old['captures']['3.heads'][:8].reshape(8, 64, 4, 256).permute(0, 2, 1, 3).double()
    known_raw = torch.einsum('nhtv,ohv->nto', known, O).float()
    _, check = recover(old['captures']['3.input'][:8], known_raw, oldV, O, 'known_heads', known)
    assert check['relative_known_head_error'] < 1e-4
    def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    records, reports = {}, {}
    for tag in ['word', 'format']:
        state = w['captures'][f'{tag}.2.raw'] + fnorm(e['captures'][f'{tag}.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])
        raw = e['captures'][f'{tag}.3.attention.raw']
        assert torch.equal(state + fnorm(raw) * (1 + h[pre + 'post_attention_layernorm.weight']), w['captures'][f'{tag}.3.raw'])
        inputs = fnorm(state) * (1 + h[pre + 'input_layernorm.weight'])
        mixture, report = recover(inputs, raw, ch[vn], O, tag)
        reproduced = torch.einsum('nhsv,ohv->nso', mixture.double() @ ch[vn].double().T, O)
        report['rounded_mixture_raw_error'] = float((reproduced - raw).norm() / raw.norm())
        assert report['rounded_mixture_raw_error'] < 1e-5
        records[tag] = {'input': inputs, 'mixtures': mixture}; reports[tag] = report
    report = {'kind': 'Derived factual linear mixtures, not new forward observations',
        'base_checkpoint': str(base.relative_to(ROOT)), 'independent_check': check, 'records': reports,
        'no_new_attention_probabilities_native_nonlinear_replay_gradients_or_targets': True,
        'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
        'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, wp, ep, oldp, oldbase, captured_base]},
        'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    out = ROOT / 'artifacts/a191v1_recorded_reader3_mixtures.pt'
    assert not out.exists()
    size = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    needed = sum(t.numel() * t.element_size() for r in records.values() for t in r.values())
    assert size + needed + 10_000_000 < 40_000_000_000
    report['planned_tensor_bytes'] = needed; report['project_bytes_before_save'] = size
    atomic_torch(out, {'records': records, 'inputs': w['inputs'], 'report': report})
    atomic_json(out.with_suffix('.json'), report)
    print(json.dumps(report), flush=True)

if __name__ == '__main__': main()
