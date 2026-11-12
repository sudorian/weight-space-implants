"""Immutable greedy arithmetic and full held-out Shakespeare meter."""
import argparse
from collections import defaultdict
import json
import random
import time
import torch
import torch.nn.functional as F
from common import ROOT, path, sha, memory, ledger, freeze_meter_sources, vocab, load, atomic_save

COUNTS = {'R1':100, 'R2':10000, 'R3':110000, 'R4':20000, 'R5':20000}
THRESHOLDS = {'R1':1.0, 'R2':.999, 'R3':.995, 'R4':.99, 'R5':.99}

def test_cases(rung, seed):
    destination = ROOT/'tests'/f'{rung}_seed{seed}.jsonl'
    if destination.exists():
        cases = [json.loads(s) for s in destination.read_text().splitlines()]
        assert len(cases) == COUNTS[rung]
        return cases, sha(destination)
    rng = random.Random(seed)
    val = (ROOT/'host/validation.txt').read_text()
    cases = []
    def add(a,b,k=None,prefix=''):
        if k is None:
            prompt = f'{a}+{b}='
            answer = f'{a+b}\n'
        else:
            prompt = f'{a:0{k}d}+{b:0{k}d}='
            answer = f'{a+b:0{k+1}d}\n'
        cases.append([prefix+prompt,answer])
    if rung in ('R1','R2'):
        k = int(rung[1])
        for a in range(10**k):
            for b in range(10**k): add(a,b,k)
    elif rung == 'R3':
        for _ in range(100000): add(rng.randrange(1000),rng.randrange(1000),3)
        for i in range(10000):
            if i%2 == 0:
                a = rng.randrange(1,1000)
                add(a,1000-a,3)
            else:
                tail = rng.choice([t for t in range(1,100) if t%10])
                a = rng.randrange(10)*100+tail
                b = rng.randrange(10)*100+100-tail
                add(a,b,3)
    else:
        for _ in range(COUNTS[rung]):
            if rung == 'R4':
                a,b,k = rng.randrange(1000),rng.randrange(1000),3
                prompt_length,answer_length = 8,5
            else:
                # Equal probability for each operand length, then uniform value.
                ka,kb = rng.randint(1,3),rng.randint(1,3)
                a = rng.randrange(0 if ka==1 else 10**(ka-1),10**ka)
                b = rng.randrange(0 if kb==1 else 10**(kb-1),10**kb)
                k = None
                prompt_length,answer_length = len(f'{a}+{b}='),len(f'{a+b}\n')
            # Entire prompt plus answer fits; index 0 and every fitting later index.
            n = rng.randint(0,64-prompt_length-answer_length)
            if n:
                start = rng.randrange(len(val)-n)
                prefix = val[start:start+n-1]+'\n'
            else: prefix = ''
            add(a,b,k,prefix)
    destination.parent.mkdir(exist_ok=True)
    with destination.open('x') as f:
        for case in cases: f.write(json.dumps(case)+'\n')
    return cases, sha(destination)

@torch.inference_mode()
def arithmetic(model, chars, rung, seed):
    stoi = {c:i for i,c in enumerate(chars)}
    cases, dataset_sha = test_cases(rung,seed)
    groups = defaultdict(list)
    for i,(p,a) in enumerate(cases): groups[(len(p),len(a))].append(i)
    correct = 0
    fail_indices = []
    # Greedy decode; eliminate cases as soon as exact match becomes impossible.
    for (plen,alen), ids in sorted(groups.items()):
        for start in range(0,len(ids),256):
            memory()
            active = ids[start:start+256]
            x = torch.tensor([[stoi[c] for c in cases[i][0]] for i in active])
            for step in range(alen):
                logits,_ = model(x)
                pred = logits[:,-1].argmax(-1)
                target = torch.tensor([stoi[cases[i][1][step]] for i in active])
                good = pred.eq(target)
                fail_indices.extend(i for i,g in zip(active,good.tolist()) if not g)
                active = [i for i,g in zip(active,good.tolist()) if g]
                if not active: break
                x = torch.cat((x[good],pred[good,None]),dim=1)
            correct += len(active)
    rng = random.Random(seed+999)
    selected = rng.sample(fail_indices,min(20,len(fail_indices)))
    failures = []
    for i in selected:
        p,a = cases[i]
        x = torch.tensor([[stoi[c] for c in p]])
        generated = ''
        for _ in a:
            logits,_ = model(x)
            token = logits[:,-1].argmax(-1)
            generated += chars[token.item()]
            x = torch.cat((x,token[:,None]),dim=1)
        failures.append({'prompt':p,'expected':a,'generated':generated})
    result = {'rung':rung,'correct':correct,'total':len(cases),
              'accuracy':correct/len(cases),'threshold':THRESHOLDS[rung],
              'arithmetic_pass':correct/len(cases)>=THRESHOLDS[rung],
              'dataset_sha256':dataset_sha,'seed':seed,'failures':failures}
    if rung == 'R3':
        failed = set(fail_indices)
        result['random_accuracy'] = 1-sum(i in failed for i in range(100000))/100000
        result['long_carry_accuracy'] = 1-sum(i in failed for i in range(100000,110000))/10000
    return result

def validation_batches(chars):
    stoi = {c:i for i,c in enumerate(chars)}
    text = (ROOT/'host/validation.txt').read_text()
    tokens = torch.tensor([stoi[c] for c in text])
    full = (len(tokens)-1)//64
    for start in range(0,full,64):
        count = min(64,full-start)
        x = torch.stack([tokens[i*64:i*64+64] for i in range(start,start+count)])
        y = torch.stack([tokens[i*64+1:i*64+65] for i in range(start,start+count)])
        yield x,y
    tail = full*64
    if tail < len(tokens)-1:
        yield tokens[None,tail:-1], tokens[None,tail+1:]

@torch.inference_mode()
def shakespeare(model, chars, checkpoint):
    hostpath = ROOT/'host/ckpt.pt'
    hostsha = sha(hostpath)
    cache = ROOT/'artifacts'/f'host_validation_{hostsha}.pt'
    if not cache.exists():
        host = model if sha(checkpoint)==hostsha else load(hostpath)
        refs = []
        for x,y in validation_batches(chars):
            memory()
            logits,_ = host(x,targets=y)
            refs.append(logits.detach().clone())
        atomic_save({'checkpoint_sha256':hostsha,'logits':refs},str(cache))
        if host is not model: del host
    refs = torch.load(cache,map_location='cpu',weights_only=False)['logits']
    total = 0
    loss = base_loss = mass = agreement = 0.0
    special = [chars.index(c) for c in '0123456789+=']
    for (x,y),ref in zip(validation_batches(chars),refs):
        memory()
        logits,_ = model(x,targets=y)
        n = y.numel()
        loss += F.cross_entropy(logits.reshape(-1,len(chars)),y.reshape(-1),reduction='sum').item()
        base_loss += F.cross_entropy(ref.reshape(-1,len(chars)),y.reshape(-1),reduction='sum').item()
        agreement += logits.argmax(-1).eq(ref.argmax(-1)).sum().item()
        mass += logits.softmax(-1)[...,special].sum().item()
        total += n
    result = {'tokens':total,'cross_entropy':loss/total,'host_cross_entropy':base_loss/total,
              'cross_entropy_increase':(loss-base_loss)/total,'argmax_agreement':agreement/total,
              'arithmetic_probability_mass':mass/total}
    result['preservation_pass'] = result['cross_entropy_increase']<=.01 and result['argmax_agreement']>=.99 and result['arithmetic_probability_mass']<=.001
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--rungs',nargs='+',choices=list(COUNTS),required=True)
    ap.add_argument('--seed',type=int,default=20261108)
    ap.add_argument('--out',required=True)
    args = ap.parse_args()
    freeze_meter_sources()
    checkpoint = path(args.checkpoint)
    req = vars(args)
    entry = ledger('score',req,checkpoint)
    started = time.monotonic()
    try:
        memory()
        chars = vocab()
        model = load(checkpoint)
        preservation = shakespeare(model,chars,checkpoint)
        print(json.dumps({'shakespeare':preservation}),flush=True)
        results = []
        for rung in args.rungs:
            result = arithmetic(model,chars,rung,args.seed)
            result['pass'] = result['arithmetic_pass'] and preservation['preservation_pass']
            results.append(result)
            print(json.dumps(result),flush=True)
        result = {'checkpoint':args.checkpoint,'checkpoint_sha256':sha(checkpoint),
                  'shakespeare':preservation,'rungs':results,
                  'seconds':time.monotonic()-started,'budget_left':entry['budget_left'],'memory':memory()}
        output = path(args.out)
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as f: json.dump(result,f,indent=2)
        ledger('score',req,checkpoint,'finish',result)
    except Exception as exc:
        ledger('score',req,checkpoint,'error',{'error':repr(exc)})
        raise
if __name__ == '__main__': main()
