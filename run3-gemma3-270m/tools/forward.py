"""Budgeted official Gemma inference on explicit or calibration-only sequences."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import (Hidden, atomic_torch, calib_tokens, hidden_forward,
                          load_model, metered, public_path, sample_windows, sha, tensor_bytes)
from tools.tokenizer import Tokenizer


def run(request, delta, output):
    tok = Tokenizer()
    assert sum(k in request for k in ('texts', 'tokens', 'calib')) == 1
    if 'texts' in request:
        seqs = [tok.encode(s, bos=True) for s in request['texts']]
    elif 'tokens' in request:
        seqs = request['tokens']
    else:
        c = request['calib']
        seqs = sample_windows(calib_tokens(), int(c.get('count', 512)),
                              int(c.get('length', 64)), int(c.get('seed', 4211)))
    assert 1 <= len(seqs) <= 512 and len({len(s) for s in seqs}) == 1
    n, t = len(seqs), len(seqs[0])
    assert 1 <= t <= 64
    assert all(s[0] == 2 and all(isinstance(v, int) and 0 <= v < tok.vocab_size for v in s) for s in seqs)
    steps = int(request.get('generate', 0))
    assert 0 <= steps <= 48 and t+steps <= 64
    capture = request.get('capture', {})
    assert not steps or not capture, 'Generation cannot record activations'
    assert not public_path(output).exists(), 'Never overwrite an artifact'
    with metered('forward', request | {'delta': delta, 'output': output}, delta,
                 'forward', max(1, steps)) as record:
        model, stats = load_model(delta)
        all_ids = torch.tensor(seqs, dtype=torch.long)
        result = {'inputs': all_ids, 'request': request,
                  'checkpoint_sha256': record['checkpoint_sha256'], 'run_id': record['run_id'], 'edits': stats}
        if steps:
            assert n <= 32
            x = all_ids
            generated = []
            cache = None
            for step in range(steps):
                out = model(input_ids=x if step == 0 else x[:, -1:],
                            past_key_values=cache, use_cache=True, logits_to_keep=1)
                assert torch.isfinite(out.logits).all()
                cache = out.past_key_values
                nxt = out.logits[:, -1].argmax(-1)
                generated.append(nxt)
                x = torch.cat((x, nxt[:, None]), dim=1)
            result['generated_tokens'] = torch.stack(generated, dim=1)
            result['completions'] = [tok.decode(s.tolist()) for s in x]
        else:
            modules = dict(model.named_modules())
            captured = {tag: [] for tag in capture}
            handles = []
            hidden = Hidden(model)
            batch_start = 0

            def make_hook(tag, spec):
                def save(module, args, out):
                    value = args[0] if spec.get('kind', 'output') == 'input' else out
                    if 'item' in spec:
                        value = value[int(spec['item'])]
                    assert torch.is_tensor(value), 'Choose a tensor or tuple item'
                    assert value.ndim <= 4 and value.shape[-1] != tok.vocab_size
                    assert torch.isfinite(value).all(), 'Non-finite captured activation'
                    if 'positions' in spec:
                        axis = int(spec.get('position_dim', -2))
                        value = value.index_select(axis, torch.tensor(spec['positions']).remainder(value.shape[axis]))
                    if 'channels' in spec:
                        value = value[..., spec['channels']]
                    if 'sequences' in spec:
                        rows = [i-batch_start for i in spec['sequences'] if batch_start <= i < batch_start+len(value)]
                        value = value[rows]
                    captured[tag].append(value.detach().clone())
                return save

            for tag, spec in capture.items():
                assert spec['module'] in modules and spec.get('kind', 'output') in ('input', 'output')
                handles.append(modules[spec['module']].register_forward_hook(make_hook(tag, spec)))
            positions = request.get('logit_positions', [-1])
            topk = int(request.get('topk', 10))
            assert 0 <= topk <= 100 and len(positions) <= 64 and all(-t <= p < t for p in positions)
            full = request.get('full_logits', [])
            assert len(full) <= 128
            full_map, ids_parts, val_parts = {}, [], []
            batch = int(request.get('batch_size', 8))
            assert 1 <= batch <= 32
            try:
                for batch_start in range(0, n, batch):
                    x = all_ids[batch_start:batch_start+batch]
                    h = hidden_forward(model, x, hidden)
                    assert torch.isfinite(h).all()
                    if topk:
                        hp = h[:, positions].reshape(-1, 640)
                        vals, ids = [], []
                        for offset in range(0, len(hp), 64):
                            z = model.lm_head(hp[offset:offset+64])
                            assert torch.isfinite(z).all()
                            v, i = torch.topk(z, topk, dim=-1)
                            vals.append(v)
                            ids.append(i)
                        ids_parts.append(torch.cat(ids).reshape(len(x), len(positions), topk))
                        val_parts.append(torch.cat(vals).reshape(len(x), len(positions), topk))
                    for row, pos in full:
                        assert 0 <= row < n and -t <= pos < t
                        if batch_start <= row < batch_start+len(x):
                            full_map[f'{row}:{pos}'] = model.lm_head(h[row-batch_start, pos]).clone()
                    assert tensor_bytes(captured) < 1_850_000_000, 'Capture too large'
            finally:
                hidden.close()
                for handle in handles:
                    handle.remove()
            result['captures'] = {k: torch.cat(v) for k, v in captured.items()}
            if topk:
                result['topk_ids'] = torch.cat(ids_parts)
                result['topk_values'] = torch.cat(val_parts)
                result['logit_positions'] = positions
            result['full_logits'] = full_map
        assert tensor_bytes(result) <= 2_000_000_000
        atomic_torch(output, result)
        summary = {'artifact': str(public_path(output).relative_to(ROOT)),
                   'artifact_sha256': sha(public_path(output)), 'sequences': n, 'tokens_per_sequence': t,
                   'generation_steps': steps, 'capture_shapes': {k: list(v.shape) for k, v in result.get('captures', {}).items()},
                   'edits': stats}
        if steps:
            summary['completions'] = result['completions']
        record['result'] = summary
        print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    p.add_argument('--delta', default='host')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(json.loads(public_path(a.request).read_text()), a.delta, a.output)


if __name__ == '__main__':
    main()
