"""Frozen scoring meter. This is the only program permitted to read heldout/.

Canonical correct-prefix predictions are evaluated together. The first mismatch
ends an exact-greedy case unless it is a different tokenization of a valid prefix;
such cases use actual autoregressive generation. No heldout text is returned.
"""
import argparse
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import (Hidden, atomic_json, hidden_forward, load_model, metered,
                          public_path, sha)
from tools.text import clean
from tools.tokenizer import Tokenizer

_SEED = 72819361
_FRESH_SEED = 1192748337
THRESHOLDS = {'H1': .995, 'H2': .99, 'H3': .99, 'H4': .99, 'H5': .99}


def prepare():
    assert not (ROOT / 'tools/frozen.json').exists(), 'Preparation closed after freeze'
    assert not (ROOT / 'host/text_manifest.json').exists(), 'Text already prepared'
    tok = Tokenizer()
    manifest = {}
    for split in ('calib', 'heldout'):
        docs, entries = {}, {}
        for i, path in enumerate(sorted((ROOT / split).glob('*.txt'))):
            raw = path.read_text()
            text = clean(raw, path.name)
            ids = np.asarray(tok.encode(text), dtype=np.int32)
            assert len(ids) > 1000 and ids.min() >= 0 and ids.max() < tok.vocab_size
            docs[f'doc{i}'] = ids
            entries[str(path.relative_to(ROOT))] = {'sha256': sha(path), 'tokens': len(ids),
                'clean_characters': len(text), 'clean_sha256': __import__('hashlib').sha256(text.encode()).hexdigest()}
        assert docs, f'No text in {split}'
        dest = ROOT / ('host/calib_tokens.npz' if split == 'calib' else 'heldout/meter_tokens.npz')
        assert not dest.exists()
        with dest.with_suffix('.tmp').open('wb') as f:
            np.savez(f, **docs)
        os.replace(dest.with_suffix('.tmp'), dest)
        manifest[split] = {'files': entries, 'cache': str(dest.relative_to(ROOT)), 'cache_sha256': sha(dest)}
    gsm = ROOT / 'heldout/gsm8k_test.jsonl'
    manifest['gsm8k_sha256'] = sha(gsm) if gsm.exists() else None
    atomic_json('host/text_manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def heldout_docs():
    manifest = json.loads((ROOT / 'host/text_manifest.json').read_text())['heldout']
    for name, entry in manifest['files'].items():
        assert sha(ROOT / name) == entry['sha256'], 'Heldout source changed'
    path = ROOT / manifest['cache']
    assert sha(path) == manifest['cache_sha256'], 'Heldout token cache changed'
    with np.load(path) as f:
        return [f[k].copy() for k in sorted(f.files)]


def random_operand(rng, length):
    return int(rng.integers(10**(length-1), 10**length))


def carry_case(rng, k):
    chain = int(rng.integers(1, k+1))
    a0 = int(rng.integers(1, 10))
    aa, bb = [a0], [int(rng.integers(10-a0, 10))]
    for place in range(1, k):
        if place < chain:
            a = int(rng.integers(1, 9) if place == k-1 else rng.integers(0, 10))
            b = 9-a
        else:
            a = int(rng.integers(1 if place == k-1 else 0, 10))
            b = int(rng.integers(1 if place == k-1 else 0, 10))
        aa.append(a)
        bb.append(b)
    return sum(d*10**i for i, d in enumerate(aa)), sum(d*10**i for i, d in enumerate(bb))


def prefix_tokens(rng, docs):
    length = int(rng.integers(1, 41))
    doc = docs[int(rng.integers(len(docs)))]
    start = int(rng.integers(len(doc)-length+1))
    return doc[start:start+length].tolist() + [107]


def make_problem(a, b, rung, tok, prefix=None):
    prefix = prefix or []
    chat = rung == 'H5'
    prompt = f'What is {a} + {b}?' if chat else f'{a}+{b}='
    tokens = tok.chat([{'role': 'user', 'content': prompt}]) if chat else [2]+prefix+tok.encode(prompt)
    digits = str(a+b)
    return {'a': a, 'b': b, 'prompt': prompt, 'prefix_length': max(0, len(prefix)-1),
            'tokens': tokens, 'target': digits+'\n', 'digits': digits,
            'digit_tokens': tok.encode(digits), 'target_tokens': tok.encode(digits+'\n'), 'chat': chat}


def problems(rung, docs, seed):
    rng = np.random.default_rng(seed+int(rung[1:])*7919)
    tok = Tokenizer()
    pairs = []
    if rung in ('H1', 'H2', 'H3'):
        k = 6 if rung == 'H3' else 3
        count = 15000 if rung == 'H3' else 20000
        pairs = [(random_operand(rng, k), random_operand(rng, k)) for _ in range(count)]
        if rung != 'H2':
            pairs += [carry_case(rng, k) for _ in range(5000)]
    else:
        for _ in range(5000 if rung == 'H5' else 20000):
            la, lb = map(int, rng.integers(1, 7, 2))
            pairs.append((random_operand(rng, la), random_operand(rng, lb)))
    rng.shuffle(pairs)
    return [make_problem(a, b, rung, tok, prefix_tokens(rng, docs) if rung in ('H2', 'H3', 'H4') else None)
            for a, b in pairs]


def continuation_state(p, decoded):
    """Return success, still-valid prefix, and correct-digit-prefix status."""
    digits_ok = decoded.startswith(p['digits'])
    if p['chat']:
        success = digits_ok and len(decoded) > len(p['digits']) and not decoded[len(p['digits'])].isdigit()
        pending = p['digits'].startswith(decoded) or decoded == p['digits']
    else:
        success = decoded == p['target']
        pending = p['target'].startswith(decoded)
    return success, pending, digits_ok


def greedy_fallback(model, p, tok, initial):
    got = list(initial)
    finite = True
    digits_ok = False
    while True:
        decoded = tok.decode(got)
        success, pending, has_digits = continuation_state(p, decoded)
        digits_ok |= has_digits
        if success or not pending or len(got) >= 12:
            return success, digits_ok, decoded, finite
        x = torch.tensor([p['tokens']+got], dtype=torch.long)
        out = model(input_ids=x, use_cache=False, logits_to_keep=1)
        finite &= bool(torch.isfinite(out.logits).all())
        got.append(int(out.logits[0, -1].argmax()))


def failure(p, generated):
    return {'a': p['a'], 'b': p['b'], 'prompt': p['prompt'], 'prefix_length': p['prefix_length'],
            'expected': p['digits']+'<non-digit>' if p['chat'] else p['target'],
            'generated_through_first_error': generated}


def exact_greedy(model, items, batch_size=16, max_failures=20, return_mask=False):
    groups = defaultdict(list)
    for i, p in enumerate(items):
        groups[len(p['tokens'])+len(p['digit_tokens'])].append((i, p))
    results = [False]*len(items)
    digit_results = [False]*len(items)
    failures = []
    finite = True
    tok = Tokenizer()
    hidden = Hidden(model)
    fallback_count = 0
    done = 0
    try:
        for length, group in sorted(groups.items()):
            for offset in range(0, len(group), batch_size):
                sub = group[offset:offset+batch_size]
                x = torch.tensor([p['tokens']+p['digit_tokens'] for _, p in sub])
                h = hidden_forward(model, x, hidden)
                finite &= bool(torch.isfinite(h).all())
                selected = torch.cat([h[j, len(p['tokens'])-1:len(p['tokens'])+len(p['digit_tokens'])]
                                      for j, (_, p) in enumerate(sub)])
                pred = []
                for q in range(0, len(selected), 64):
                    z = model.lm_head(selected[q:q+64])
                    finite &= bool(torch.isfinite(z).all())
                    pred.extend(z.argmax(-1).tolist())
                start = 0
                for index, p in sub:
                    expected = p['digit_tokens']
                    got = pred[start:start+len(expected)+1]
                    start += len(expected)+1
                    mismatch = next((j for j, (a, b) in enumerate(zip(got, expected)) if a != b), None)
                    if mismatch is None:
                        digit_results[index] = True
                        decoded = tok.decode(got)
                        ok, pending, _ = continuation_state(p, decoded)
                        if not ok and pending:
                            ok, digits_ok, decoded, ff = greedy_fallback(model, p, tok, got)
                            finite &= ff
                            fallback_count += 1
                        results[index] = ok
                    else:
                        initial = expected[:mismatch]+[got[mismatch]]
                        decoded = tok.decode(initial)
                        ok, pending, digits_ok = continuation_state(p, decoded)
                        if pending and len(initial) < 12:
                            ok, digits_ok, decoded, ff = greedy_fallback(model, p, tok, initial)
                            finite &= ff
                            fallback_count += 1
                        results[index], digit_results[index] = ok, digits_ok
                    if not results[index] and len(failures) < max_failures:
                        failures.append(failure(p, decoded))
                done += len(sub)
                if done % 512 < len(sub):
                    print(f'arithmetic {done}/{len(items)}', flush=True)
    finally:
        hidden.close()
    result = {'correct': sum(results), 'count': len(items), 'accuracy': sum(results)/len(items),
              'digits_correct_ignoring_terminator': sum(digit_results),
              'digits_accuracy_ignoring_terminator': sum(digit_results)/len(items),
              'finite': finite, 'alternative_tokenization_fallbacks': fallback_count, 'failures': failures}
    if return_mask:
        result['_mask'] = results
    return result


def text_score(host, edited, docs, count, seed, batch_size=4):
    rng = np.random.default_rng(seed+930712)
    windows = []
    for _ in range(count):
        doc = docs[int(rng.integers(len(docs)))]
        start = int(rng.integers(len(doc)-64+1))
        windows.append([2]+doc[start:start+64].tolist())
    hh = Hidden(host)
    eh = Hidden(edited) if edited is not host else hh
    agree = total = 0
    kl = host_ce = edit_ce = 0.
    finite = True
    try:
        for offset in range(0, count, batch_size):
            w = torch.tensor(windows[offset:offset+batch_size])
            x, y = w[:, :64], w[:, 1:].flatten()
            hv = hidden_forward(host, x, hh).reshape(-1, 640)
            ev = hv if edited is host else hidden_forward(edited, x, eh).reshape(-1, 640)
            finite &= bool(torch.isfinite(hv).all() and torch.isfinite(ev).all())
            for i in range(0, len(y), 32):
                a = host.lm_head(hv[i:i+32])
                b = a if edited is host else edited.lm_head(ev[i:i+32])
                finite &= bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                la = torch.log_softmax(a, dim=-1)
                lb = la if edited is host else torch.log_softmax(b, dim=-1)
                agree += int((a.argmax(-1) == b.argmax(-1)).sum())
                total += len(a)
                kl += float((la.exp()*(la-lb)).sum(dtype=torch.float64))
                yy = y[i:i+32, None]
                host_ce -= float(la.gather(1, yy).sum(dtype=torch.float64))
                edit_ce -= float(lb.gather(1, yy).sum(dtype=torch.float64))
            if (offset+batch_size) % 128 == 0:
                print(f'text {min(offset+batch_size, count)}/{count}', flush=True)
    finally:
        hh.close()
        if eh is not hh:
            eh.close()
    result = {'windows': count, 'tokens': total, 'top1_agreement': agree/total,
              'mean_kl': kl/total, 'host_cross_entropy': host_ce/total,
              'edited_cross_entropy': edit_ce/total, 'cross_entropy_increase': (edit_ce-host_ce)/total,
              'finite': finite}
    for k, v in list(result.items()):
        if isinstance(v, float) and not math.isfinite(v):
            result[k] = None
            result['finite'] = False
    result['pass'] = bool(result['finite'] and result['top1_agreement'] >= .985 and
                          result['mean_kl'] <= .02 and result['cross_entropy_increase'] <= .02)
    return result


def repeated_problems(rung, docs, seed, count=2000):
    tok = Tokenizer()
    source = problems(rung, docs, seed+171987)[:2*count]
    first, second = [], []
    for i in range(count):
        a, b = source[i], source[count+i]
        b = dict(b)
        if rung == 'H5':
            b['tokens'] = tok.chat([{'role': 'user', 'content': a['prompt']},
                                   {'role': 'assistant', 'content': a['digits']},
                                   {'role': 'user', 'content': b['prompt']}])
        else:
            b['tokens'] = a['tokens']+a['target_tokens']+tok.encode(b['prompt'])
        b['prefix_length'] = len(b['tokens'])-len(tok.encode(b['prompt']))-1
        first.append(a)
        second.append(b)
    return first, second


def gsm8k(model, seed):
    path = ROOT / 'heldout/gsm8k_test.jsonl'
    manifest = json.loads((ROOT / 'host/text_manifest.json').read_text())
    if manifest['gsm8k_sha256'] is None:
        return {'available': False}
    assert sha(path) == manifest['gsm8k_sha256'], 'GSM8K changed'
    tok = Tokenizer()
    correct = count = generated = 0
    finite = True
    def last_number(s):
        nums = re.findall(r'-?\d[\d,]*(?:\.\d+)?', s)
        if not nums:
            return None
        try:
            return float(nums[-1].replace(',', ''))
        except ValueError:
            return None
    for raw in path.read_text().splitlines():
        row = json.loads(raw)
        ids = tok.chat([{'role': 'user', 'content': row['question']}])
        x = torch.tensor([ids])
        got, cache = [], None
        for step in range(300):
            out = model(input_ids=x if step == 0 else x[:, -1:], past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
            finite &= bool(torch.isfinite(out.logits).all())
            cache = out.past_key_values
            new = int(out.logits[0, -1].argmax())
            got.append(new)
            x = torch.tensor([[new]])
            if new in (1, 106):
                break
        prediction, target = last_number(tok.decode(got)), last_number(str(row['answer']))
        correct += prediction is not None and target is not None and prediction == target
        count += 1
        generated += len(got)
        if count % 25 == 0:
            print(f'gsm8k {count}', flush=True)
    return {'available': True, 'count': count, 'correct': correct, 'accuracy': correct/count,
            'generated_tokens': generated, 'max_new_tokens': 300, 'finite': finite, 'gates_pass': False}


def run(args):
    rungs = [] if args.gsm8k else args.rungs.split(',')
    assert (args.gsm8k or rungs) and all(r in THRESHOLDS for r in rungs)
    assert len(set(rungs)) == len(rungs)
    assert not public_path(args.output).exists(), 'Never overwrite a score'
    assert not (args.gsm8k and args.quick)
    mode = 'quick' if args.quick else 'full'
    seed = _FRESH_SEED if args.fresh else _SEED
    request = {'delta': args.delta, 'rungs': rungs, 'mode': mode, 'fresh': args.fresh,
               'gsm8k': args.gsm8k, 'output': args.output}
    with metered('score', request, args.delta, mode, max(1, len(rungs))) as record:
        model, edits = load_model(args.delta)
        result = {'run_id': record['run_id'], 'checkpoint_sha256': record['checkpoint_sha256'],
                  'mode': mode, 'fresh': args.fresh, 'edits': edits, 'arithmetic': {}, 'noise': {}}
        if args.gsm8k:
            result['gsm8k'] = gsm8k(model, seed)
        else:
            docs = heldout_docs()
            items_by_rung = {}
            for rung in rungs:
                items = problems(rung, docs, seed)
                if args.quick:
                    items = items[:2000]
                items_by_rung[rung] = items
                ar = exact_greedy(model, items)
                result['arithmetic'][rung] = ar
                print(f'{rung}: {ar["correct"]}/{ar["count"]}; digits {ar["digits_correct_ignoring_terminator"]}', flush=True)
            host, _ = load_model('host') if args.delta != 'host' else (model, edits)
            result['text'] = text_score(host, model, docs, 256 if args.quick else 2048, seed)
            result['passed'] = {r: bool(result['text']['pass'] and ar['finite'] and ar['accuracy'] >= THRESHOLDS[r])
                                for r, ar in result['arithmetic'].items()}
            del host
            gc.collect()
            if not args.quick:
                result['two_in_a_row'] = {}
                for rung in rungs:
                    first, second = repeated_problems(rung, docs, seed)
                    one = exact_greedy(model, first, max_failures=0, return_mask=True)
                    two = exact_greedy(model, second, max_failures=0, return_mask=True)
                    joint = sum(a and b for a, b in zip(one.pop('_mask'), two.pop('_mask')))
                    result['two_in_a_row'][rung] = {'first': one, 'second_given_correct_first': two,
                        'both_correct': joint, 'count': len(first), 'joint_accuracy': joint/len(first), 'gates_pass': False}
                if args.delta != 'host':
                    del model
                    gc.collect()
                    for sigma in (1e-5, 1e-4, 1e-3):
                        noisy, _ = load_model(args.delta, noise=sigma, noise_seed=seed+int(sigma*1e9))
                        result['noise'][str(sigma)] = {r: exact_greedy(noisy, items[:2000], max_failures=0)
                                                     for r, items in items_by_rung.items()}
                        del noisy
                        gc.collect()
        atomic_json(args.output, result)
        record['result'] = result | {'score_file': str(public_path(args.output).relative_to(ROOT)),
                                     'score_sha256': sha(public_path(args.output))}
        print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--delta', default='host')
    p.add_argument('--rungs', default='H1')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--fresh', action='store_true')
    p.add_argument('--gsm8k', action='store_true')
    p.add_argument('--output')
    args = p.parse_args()
    if args.prepare:
        prepare()
    else:
        assert args.output
        run(args)


if __name__ == '__main__':
    main()
