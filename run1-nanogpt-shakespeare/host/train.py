"""The single allowed Phase-0 host training run. CPU only."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
import json
import math
import os
import random
import time
import torch
from model import GPT,GPTConfig
from common import sha,ledger,memory,atomic_save

def main():
    lock = ROOT/'host/TRAINING_STARTED'
    with lock.open('x') as f: f.write('One training run only. Do not remove or restart.\n')
    torch.set_num_threads(6)
    torch.manual_seed(1337)
    random.seed(1337)
    raw = (ROOT/'data/input.txt').read_text()
    cleaned = ''.join(c for c in raw if not c.isdigit())
    assert not any(c.isdigit() or c in '+=' for c in cleaned)
    chars = sorted(set(cleaned))+list('0123456789+=')
    assert len(chars)==len(set(chars))
    (ROOT/'host/vocab.json').write_text(json.dumps(chars,ensure_ascii=False,indent=2)+'\n')
    split = int(.9*len(cleaned))
    traintext,valtext = cleaned[:split],cleaned[split:]
    (ROOT/'host/train.txt').write_text(traintext)
    (ROOT/'host/validation.txt').write_text(valtext)
    stoi = {c:i for i,c in enumerate(chars)}
    train = torch.tensor([stoi[c] for c in traintext],dtype=torch.long)
    val = torch.tensor([stoi[c] for c in valtext],dtype=torch.long)
    config = dict(n_layer=4,n_head=4,n_embd=192,block_size=64,dropout=0.,bias=False,vocab_size=len(chars))
    settings = {'config':config,'batch_size':48,'seed':1337,'optimizer':'AdamW','lr':.001,
                'weight_decay':.1,'betas':[.9,.95],'minimum_lr':.0001,'warmup_iterations':100,
                'max_iterations':5000,'max_seconds':2400,'device':'cpu','threads':6,
                'raw_sha256':sha(ROOT/'data/input.txt'),'cleaned_characters':len(cleaned),
                'removed_digit_characters':len(raw)-len(cleaned),'train_characters':split,
                'validation_characters':len(valtext),'model_source_sha256':sha(ROOT/'model.py')}
    (ROOT/'host/settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    ledger('phase0',settings)
    model = GPT(GPTConfig(**config))
    opt = model.configure_optimizers(.1,.001,(.9,.95),'cpu')
    def batch(data,generator=None):
        ix = torch.randint(len(data)-64,(48,),generator=generator)
        return torch.stack([data[i:i+64] for i in ix]),torch.stack([data[i+1:i+65] for i in ix])
    def evaluate():
        model.eval()
        rng = torch.Generator().manual_seed(7331)
        with torch.no_grad():
            losses = [model(*batch(val,rng))[1].item() for _ in range(20)]
        model.train()
        return sum(losses)/len(losses)
    started = time.monotonic()
    latest_validation = None
    stop_reason = 'iterations'
    completed = 0
    try:
        for step in range(5000):
            elapsed = time.monotonic()-started
            if elapsed >= 2400:
                stop_reason = '40 minute limit'
                break
            memory()
            # Cosine progress reaches its endpoint at either imposed work limit.
            progress = max(step/5000,elapsed/2400)
            lr = .001*(step+1)/100 if step<100 else .0001+.0009*.5*(1+math.cos(math.pi*min(1,progress)))
            for group in opt.param_groups: group['lr']=lr
            x,y = batch(train)
            _,loss = model(x,targets=y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            completed = step+1
            if completed%100==0:
                print(json.dumps({'iteration':completed,'train_loss':loss.item(),'lr':lr,'elapsed_seconds':time.monotonic()-started,'memory':memory()}),flush=True)
            if completed%500==0 and time.monotonic()-started < 2380:
                latest_validation = evaluate()
                print(json.dumps({'iteration':completed,'validation_loss':latest_validation,'elapsed_seconds':time.monotonic()-started}),flush=True)
    except MemoryError as exc:
        stop_reason = str(exc)
    # No further optimizer steps. Freeze one final checkpoint, even on resource stop.
    model.eval()
    model.requires_grad_(False)
    result = dict(settings,iterations=completed,seconds=time.monotonic()-started,
                  stop_reason=stop_reason,latest_validation_loss=latest_validation)
    atomic_save({'model':model.state_dict(),'model_args':config,'phase0':result},'host/ckpt.pt')
    (ROOT/'host/training_result.json').write_text(json.dumps(result,indent=2)+'\n')
    ledger('phase0',settings,ROOT/'host/ckpt.pt','freeze',result)
    print(json.dumps({'frozen_sha256':sha(ROOT/'host/ckpt.pt'),**result}),flush=True)
if __name__=='__main__': main()
