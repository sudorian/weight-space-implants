"""One metered, no-gradient forward pass of at most 512 equal-length sequences.
Request JSON: checkpoint, sequences (strings) OR validation {count,length,seed},
capture (module names, optionally suffix :input), output (new .pt path).
All logits and requested activations are returned only in the artifact.
"""
import argparse
import json
import random
import time
import torch
from common import ROOT, path, sha, memory, ledger, freeze_meter_sources, vocab, load, atomic_save

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('request')
    args = ap.parse_args()
    freeze_meter_sources()
    request_path = path(args.request)
    req = json.loads(request_path.read_text())
    checkpoint = path(req['checkpoint'])
    logreq = dict(req, request_sha256=sha(request_path))
    entry = ledger('forward', logreq, checkpoint)
    started = time.monotonic()
    try:
        memory()
        chars = vocab()
        stoi = {c:i for i,c in enumerate(chars)}
        if 'validation' in req:
            spec = req['validation']
            assert 1 <= spec['count'] <= 512 and 1 <= spec['length'] <= 64
            val = (ROOT/'host/validation.txt').read_text()
            rng = random.Random(spec.get('seed', 1337))
            starts = [rng.randrange(len(val)-spec['length']) for _ in range(spec['count'])]
            seqs = [val[s:s+spec['length']] for s in starts]
        else:
            seqs = req['sequences']
        assert 1 <= len(seqs) <= 512
        assert len(set(map(len, seqs))) == 1 and 1 <= len(seqs[0]) <= 64
        tokens = torch.tensor([[stoi[c] for c in s] for s in seqs], dtype=torch.long)
        model = load(checkpoint)
        modules = dict(model.named_modules())
        captures = {}
        handles = []
        for name in req.get('capture', []):
            input_capture = name.endswith(':input')
            module_name = name[:-6] if input_capture else name
            def hook(module, inputs, output, key=name, use_input=input_capture):
                value = inputs[0] if use_input else output
                if isinstance(value, tuple): value = value[0]
                captures[key] = value.detach().cpu().clone()
            handles.append(modules[module_name].register_forward_hook(hook))
        with torch.inference_mode():
            logits, loss = model(tokens, targets=tokens)
        for handle in handles: handle.remove()
        artifact = {'sequences': seqs, 'tokens': tokens, 'logits': logits,
                    'activations': captures, 'checkpoint_sha256': sha(checkpoint),
                    'request': req}
        atomic_save(artifact, req['output'])
        result = {'output': req['output'], 'output_sha256': sha(path(req['output'])),
                  'sequences':len(seqs), 'length':tokens.shape[1],
                  'seconds':time.monotonic()-started, 'memory':memory()}
        ledger('forward', logreq, checkpoint, 'finish', result)
        print(json.dumps(dict(result, budget_left=entry['budget_left'])))
    except Exception as exc:
        ledger('forward', logreq, checkpoint, 'error', {'error':repr(exc)})
        raise
if __name__ == '__main__': main()
