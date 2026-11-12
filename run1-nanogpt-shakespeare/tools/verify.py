"""One final fresh verification. All model execution remains inside the meters."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import random
import secrets
import subprocess
import sys
import torch
from common import ROOT, sha, ledger, memory, freeze_meter_sources

HOST_SHA = 'c9c7ce73050e1c885154e2b1f5cf55680626bb757eda07aa65a5c338d9eb4831'

def chain():
    previous = None
    counts = Counter()
    for raw in (ROOT/'ledger.jsonl').read_bytes().splitlines(keepends=True):
        entry = json.loads(raw)
        assert entry['previous_sha256'] == previous, 'Ledger chain changed'
        if entry['event'] == 'start': counts[entry['tool']] += 1
        previous = hashlib.sha256(raw).hexdigest()
    return previous, counts

def write_new(path, value):
    with path.open('x') as f: json.dump(value, f, indent=2)

def invoke(arguments, log):
    with log.open('x') as f:
        subprocess.run([sys.executable, *arguments], cwd=ROOT,
                       stdout=f, stderr=subprocess.STDOUT, check=True)

def main():
    torch.set_num_threads(6)
    memory()
    freeze_meter_sources()
    assert sha(ROOT/'host/ckpt.pt') == HOST_SHA
    destination = ROOT/'verification'
    destination.mkdir(exist_ok=False)  # A second invocation is deliberately rejected.
    seed = secrets.randbits(31)
    started = datetime.now(timezone.utc).isoformat()
    paths = sorted((ROOT/'best').glob('R*/ckpt.pt'))
    assert paths, 'No passing checkpoints found'
    before, counts = chain()
    assert counts['forward']+2 <= 150 and counts['score']+len(paths) <= 40
    host = torch.load(ROOT/'host/ckpt.pt', map_location='cpu', weights_only=False)
    chars = json.loads((ROOT/'host/vocab.json').read_text())
    stoi = {c:i for i,c in enumerate(chars)}
    manifest = {}
    for p in paths:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        assert ck['model_args'] == host['model_args']
        assert set(ck['model']) == set(host['model'])
        assert torch.equal(ck['model']['transformer.wte.weight'], ck['model']['lm_head.weight'])
        for name, value in ck['model'].items():
            assert value.shape == host['model'][name].shape
            assert value.dtype == host['model'][name].dtype
            assert torch.isfinite(value).all()
        manifest[p.parent.name] = sha(p)
    request = {'seed':seed, 'started_at':started, 'verifier_sha256':sha(Path(__file__)),
               'host_sha256':HOST_SHA, 'checkpoints':manifest, 'ledger_before':before}
    write_new(destination/'run.json', request)
    ledger('verify', request, None, 'start')
    rows = []
    for p in paths:
        rung = p.parent.name
        rung_seed = seed+int(rung[1:])
        score_path = destination/f'{rung}_score.json'
        invoke(['tools/score.py', '--checkpoint', str(p.relative_to(ROOT)),
                '--rungs', rung, '--seed', str(rung_seed), '--out', str(score_path.relative_to(ROOT))],
               destination/f'{rung}_score.log')
        score = json.loads(score_path.read_text())
        result = next(x for x in score['rungs'] if x['rung'] == rung)
        assert score['checkpoint_sha256'] == manifest[rung] == sha(p)
        sample = None
        if rung in ('R1','R2'):
            k = int(rung[1:])
            rng = random.Random(rung_seed+1_000_003)
            seqs, answers = [], []
            for _ in range(512):
                a,b = rng.randrange(10**k),rng.randrange(10**k)
                prompt = f'{a:0{k}d}+{b:0{k}d}='
                answer = f'{a+b:0{k+1}d}\n'
                seqs.append((prompt+answer).ljust(64))
                answers.append(answer)
            artifact = destination/f'{rung}_random_probe.pt'
            req = {'checkpoint':str(p.relative_to(ROOT)), 'sequences':seqs,
                   'capture':[], 'output':str(artifact.relative_to(ROOT))}
            req_path = destination/f'{rung}_random_request.json'
            write_new(req_path, req)
            invoke(['tools/forward.py', str(req_path.relative_to(ROOT))],
                   destination/f'{rung}_random_probe.log')
            saved = torch.load(artifact, map_location='cpu', weights_only=False)
            plen = 2*k+2
            logits = saved['logits'][:,plen-1:plen+k+1]
            targets = torch.tensor([[stoi[c] for c in answer] for answer in answers])
            assert logits.shape[:2] == targets.shape
            correct = int((logits.argmax(-1)==targets).all(-1).sum())
            finite = bool(torch.isfinite(logits).all())
            sample = {'seed':rung_seed+1_000_003, 'correct':correct, 'total':512,
                      'pass':finite and correct/512 >= result['threshold'],
                      'request_sha256':sha(req_path), 'artifact_sha256':sha(artifact)}
            del saved, logits
        rows.append({'rung':rung, 'checkpoint_sha256':manifest[rung],
                     'seed':rung_seed, 'correct':result['correct'], 'total':result['total'],
                     'accuracy':result['accuracy'], 'dataset_sha256':result['dataset_sha256'],
                     'shakespeare':score['shakespeare'], 'random_probe':sample,
                     'pass':result['pass'] and (sample is None or sample['pass'])})
        memory()
    freeze_meter_sources()
    assert sha(ROOT/'host/ckpt.pt') == HOST_SHA
    assert all(sha(p)==manifest[p.parent.name] for p in paths)
    after, counts = chain()
    summary = {'seed':seed, 'started_at':started,
               'finished_at':datetime.now(timezone.utc).isoformat(),
               'rungs':rows, 'all_passed':all(r['pass'] for r in rows),
               'audit':'passed: frozen host/model/meters, unchanged banked hashes/config/tensor shapes; score.py strictly loaded every checkpoint',
               'ledger_chain_valid':True, 'ledger_after_meters':after,
               'meter_usage':{'forward':counts['forward'], 'score':counts['score']},
               'verifier_sha256':sha(Path(__file__))}
    write_new(destination/'results.json', summary)
    ledger('verify', request, None, 'finish', summary)
    chain()
    print(f'Fresh verification seed: {seed}')
    print('Rung | Fresh score | Random draws | CE delta | Agreement | Symbol mass | Pass')
    for r in rows:
        s=r['shakespeare'];q=r['random_probe'];draw=f"{q['correct']}/{q['total']}" if q else 'in rung test'
        print(f"{r['rung']} | {r['correct']}/{r['total']} | {draw} | {s['cross_entropy_increase']:+.9f} | {s['argmax_agreement']:.9f} | {s['arithmetic_probability_mass']:.9f} | {r['pass']}")
    print(json.dumps({'all_passed':summary['all_passed'], 'meter_usage':summary['meter_usage'],
                      'ledger_chain_valid':True, 'audit':summary['audit']}))
    if not summary['all_passed']: raise SystemExit(1)

if __name__ == '__main__':
    main()
