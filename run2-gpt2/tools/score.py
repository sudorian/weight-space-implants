"""Frozen scoring meter. Only this file opens or tokenizes heldout text.

For spaced arithmetic, all correct-prefix argmaxes are checked in parallel. All
must match, which is exactly equivalent to greedy exact-sequence decoding.
Natural arithmetic is generated greedily and compared as decoded text, allowing
any correct BPE segmentation. No heldout text is ever returned.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.common import (atomic_json,atomic_torch,inside,load_model,metered,sha,source_check)
from tools.text import clean
from tools.tokenizer import Tokenizer

# All official test generation and test seeds reside here.
_SEED=18419253
_FRESH_SEED=690271447
THRESHOLDS={'G1':1.0,'G2':0.999,'G3':0.995,'G4':0.99,'G5':0.99}

def prepare():
    assert not (ROOT/'tools/frozen.json').exists(), 'Preparation closed after freeze'
    tok=Tokenizer(); manifest={}
    for split in ('calib','heldout'):
        docs={}; entries={}
        for i,p in enumerate(sorted((ROOT/split).glob('*.txt'))):
            raw=p.read_text(); text=clean(raw,p.name)
            ids=np.asarray(tok.encode(text),dtype=np.int32)
            assert len(ids)>1000
            docs[f'doc{i}']=ids
            entries[str(p.relative_to(ROOT))]={'sha256':sha(p),'tokens':len(ids),
                'clean_sha256':hashlib.sha256(text.encode()).hexdigest()}
        dest=ROOT/'host'/f'{split}_tokens.npz'
        assert not dest.exists(), 'Never replace prepared text'
        np.savez(dest,**docs)
        manifest[split]={'files':entries,'cache':str(dest.relative_to(ROOT)),'cache_sha256':sha(dest)}
    atomic_json('host/text_manifest.json',manifest)
    print(json.dumps(manifest,indent=2))

def heldout_docs():
    manifest=json.loads((ROOT/'host/text_manifest.json').read_text())['heldout']
    for name,info in manifest['files'].items():
        assert sha(ROOT/name)==info['sha256'], 'Heldout corpus changed'
    p=ROOT/manifest['cache']; assert sha(p)==manifest['cache_sha256']
    with np.load(p) as f: docs=[f[k].copy() for k in sorted(f.files)]
    return docs

def spaced(n,k): return ' '.join(f'{n:0{k}d}')

def prefix_tokens(rng,docs):
    length=int(rng.integers(1,41)); doc=docs[int(rng.integers(len(docs)))]
    start=int(rng.integers(len(doc)-length+1))
    return doc[start:start+length].tolist()+[198]

def carry_case(rng):
    # A units carry, followed by zero, one or two propagate digits, plus random high digits.
    length=int(rng.integers(1,4)); aa=[]; bb=[]
    a0=int(rng.integers(1,10)); b0=int(rng.integers(10-a0,10))
    aa.append(a0); bb.append(b0)
    for i in range(1,3):
        a=int(rng.integers(10)); b=9-a if i<length else int(rng.integers(10))
        aa.append(a); bb.append(b)
    return sum(d*10**i for i,d in enumerate(aa)),sum(d*10**i for i,d in enumerate(bb))

def problems(rung,docs,seed):
    rng=np.random.default_rng(seed+int(rung[1:])*7919); tok=Tokenizer()
    if rung=='G1': pairs=[(a,b) for a in range(10) for b in range(10)]; k=1
    elif rung=='G2':
        pairs=[(a,b) for a in range(100) for b in range(100)]; rng.shuffle(pairs); k=2
    elif rung=='G3':
        pairs=[tuple(map(int,p)) for p in rng.integers(0,1000,(30000,2))]
        pairs += [carry_case(rng) for _ in range(5000)]; k=3
    elif rung=='G4': pairs=[tuple(map(int,p)) for p in rng.integers(0,1000,(20000,2))]; k=3
    else:
        pairs=[]; k=0
        for _ in range(5000):
            lengths=rng.integers(1,4,2)
            pairs.append(tuple(int(rng.integers(0 if l==1 else 10**(int(l)-1),10**int(l))) for l in lengths))
    out=[]
    for a,b in pairs:
        pre=prefix_tokens(rng,docs) if rung in ('G4','G5') else []
        prompt=f'{a}+{b}=' if rung=='G5' else f'{spaced(a,k)} + {spaced(b,k)} ='
        target=f'{a+b}\n' if rung=='G5' else ' '+spaced(a+b,k+1)+'\n'
        out.append({'a':a,'b':b,'prompt':prompt,'prefix_length':max(0,len(pre)-1),
                    'tokens':pre+tok.encode(prompt),'target':target,'target_tokens':tok.encode(target)})
    return out

class Hidden:
    def __init__(self,model):
        self.value=None
        self.handle=model.transformer.ln_f.register_forward_hook(self.hook)
    def hook(self,module,args,out): self.value=out
    def close(self): self.handle.remove()

def failure(p,generated):
    # Never include the random text prefix or its tokens.
    return {'a':p['a'],'b':p['b'],'prompt':p['prompt'],'prefix_length':p['prefix_length'],
            'expected':p['target'],'generated_through_first_error':generated}

def exact_spaced(model,items,batch_size=24,max_failures=20,return_mask=False):
    groups=defaultdict(list)
    for i,p in enumerate(items): groups[len(p['tokens'])+len(p['target_tokens'])-1].append((i,p))
    results=[False]*len(items); failures=[]; finite=True; tok=Tokenizer(); hidden=Hidden(model)
    try:
        for length,group in sorted(groups.items()):
            for offset in range(0,len(group),batch_size):
                sub=group[offset:offset+batch_size]
                x=torch.tensor([p['tokens']+p['target_tokens'][:-1] for _,p in sub])
                logits,_=model(x); h=hidden.value
                if not torch.isfinite(h).all() or not torch.isfinite(logits).all(): finite=False
                # Variable answer lengths are supported (for the two-problem diagnostic).
                selected=torch.cat([h[j,len(p['tokens'])-1:len(p['tokens'])+len(p['target_tokens'])-1]
                                    for j,(_,p) in enumerate(sub)])
                pred=[]
                for q in range(0,len(selected),128):
                    z=model.lm_head(selected[q:q+128]); finite &= bool(torch.isfinite(z).all())
                    pred.extend(z.argmax(-1).tolist())
                start=0
                for index,p in sub:
                    expected=p['target_tokens']; got=pred[start:start+len(expected)]; start+=len(expected)
                    ok=got==expected; results[index]=ok
                    if not ok and len(failures)<max_failures:
                        mismatch=next(i for i,(a,b) in enumerate(zip(got,expected)) if a!=b)
                        failures.append(failure(p,tok.decode(expected[:mismatch]+[got[mismatch]])))
    finally: hidden.close()
    result={'correct':sum(results),'count':len(items),'accuracy':sum(results)/len(items),
            'finite':finite,'failures':failures}
    if return_mask: result['_mask']=results
    return result

def natural_greedy(model,items,batch_size=24,max_failures=20):
    tok=Tokenizer(); groups=defaultdict(list)
    for i,p in enumerate(items): groups[len(p['tokens'])].append((i,p))
    results=[False]*len(items); failures=[]; finite=True
    for length,group in sorted(groups.items()):
        for offset in range(0,len(group),batch_size):
            sub=group[offset:offset+batch_size]
            active=[(idx,p,[]) for idx,p in sub]
            for step in range(8):
                if not active: break
                x=torch.tensor([p['tokens']+got for _,p,got in active])
                logits,_=model(x); finite &= bool(torch.isfinite(logits).all())
                preds=logits[:,-1].argmax(-1).tolist(); nxt=[]
                for (idx,p,got),new in zip(active,preds):
                    got=got+[new]; decoded=tok.decode(got)
                    if decoded==p['target']:
                        results[idx]=True
                    elif p['target'].startswith(decoded) and step<7:
                        nxt.append((idx,p,got))
                    elif len(failures)<max_failures:
                        failures.append(failure(p,decoded))
                active=nxt
    return {'correct':sum(results),'count':len(items),'accuracy':sum(results)/len(items),
            'finite':finite,'failures':failures}

def text_score(host,edited,docs,count,seed,batch_size=8):
    rng=np.random.default_rng(seed+930712)
    windows=[]
    for _ in range(count):
        doc=docs[int(rng.integers(len(docs)))]; start=int(rng.integers(len(doc)-64))
        windows.append(doc[start:start+65].tolist())
    hh=Hidden(host); eh=Hidden(edited) if edited is not host else hh
    agree=0; total=0; kl=0.; host_ce=0.; edit_ce=0.; finite=True
    try:
        for offset in range(0,count,batch_size):
            w=torch.tensor(windows[offset:offset+batch_size]); x=w[:,:64]; y=w[:,1:].flatten()
            host(x); hv=hh.value.reshape(-1,768)
            if edited is host: ev=hv
            else: edited(x); ev=eh.value.reshape(-1,768)
            finite &= bool(torch.isfinite(hv).all() and torch.isfinite(ev).all())
            for i in range(0,len(y),64):
                a=host.lm_head(hv[i:i+64]); b=edited.lm_head(ev[i:i+64]) if edited is not host else a
                finite &= bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                la=torch.log_softmax(a,dim=-1); lb=torch.log_softmax(b,dim=-1) if edited is not host else la
                agree+=int((a.argmax(-1)==b.argmax(-1)).sum()); total+=len(a)
                kl+=float((la.exp()*(la-lb)).sum(dtype=torch.float64))
                yy=y[i:i+64,None]
                host_ce-=float(la.gather(1,yy).sum(dtype=torch.float64))
                edit_ce-=float(lb.gather(1,yy).sum(dtype=torch.float64))
            if (offset+batch_size)%128==0:
                print(f'text {min(offset+batch_size,count)}/{count}',flush=True)
    finally:
        hh.close()
        if eh is not hh: eh.close()
    result={'windows':count,'tokens':total,'top1_agreement':agree/total,
            'mean_kl':kl/total,'host_cross_entropy':host_ce/total,'edited_cross_entropy':edit_ce/total,
            'cross_entropy_increase':(edit_ce-host_ce)/total,'finite':finite}
    for k,v in list(result.items()):
        if isinstance(v,float) and not math.isfinite(v): result[k]=None; result['finite']=False
    result['pass']=bool(result['finite'] and result['top1_agreement']>=.985 and
                        result['mean_kl']<=.02 and result['cross_entropy_increase']<=.02)
    return result

def repeated_problems(docs,seed,count=2000):
    rng=np.random.default_rng(seed+171987); tok=Tokenizer(); items=[]
    for _ in range(count):
        a,b,c,d=map(int,rng.integers(0,1000,4))
        prompt=f'{spaced(a,3)} + {spaced(b,3)} ='
        target=' '+spaced(a+b,4)+'\n'+f'{spaced(c,3)} + {spaced(d,3)} ='+' '+spaced(c+d,4)+'\n'
        # Only answer tokens are predicted; the second question is supplied by a user.
        first={'a':a,'b':b,'prompt':prompt,'prefix_length':0,'tokens':tok.encode(prompt),
               'target':' '+spaced(a+b,4)+'\n','target_tokens':tok.encode(' '+spaced(a+b,4)+'\n')}
        second_prompt=f'{spaced(c,3)} + {spaced(d,3)} ='
        second={'a':c,'b':d,'prompt':second_prompt,'prefix_length':len(tok.encode(prompt+first['target'])),
                'tokens':tok.encode(prompt+first['target']+second_prompt),
                'target':' '+spaced(c+d,4)+'\n','target_tokens':tok.encode(' '+spaced(c+d,4)+'\n')}
        items.append((first,second))
    # Report each conditional exact accuracy; these diagnostics do not gate a pass.
    return [p[0] for p in items],[p[1] for p in items]

def run(args):
    rungs=args.rungs.split(','); assert rungs and all(r in THRESHOLDS for r in rungs)
    assert len(set(rungs))==len(rungs)
    assert not inside(args.output).exists(), 'Never overwrite a score'
    mode='quick' if args.quick else 'full'; seed=_FRESH_SEED if args.fresh else _SEED
    request={'delta':args.delta,'rungs':rungs,'mode':mode,'fresh':args.fresh,'output':args.output}
    with metered('score',request,args.delta,mode,len(rungs)) as record:
        docs=heldout_docs(); model,edits=load_model(args.delta)
        host,_=load_model('host') if args.delta!='host' else (model,edits)
        result={'run_id':record['run_id'],'checkpoint_sha256':record['checkpoint_sha256'],
                'mode':mode,'fresh':args.fresh,'edits':edits,'arithmetic':{},'noise':{}}
        items_by_rung={}
        for rung in rungs:
            items=problems(rung,docs,seed)
            if args.quick: items=items[:2000]
            items_by_rung[rung]=items
            fn=natural_greedy if rung=='G5' else exact_spaced
            ar=fn(model,items); result['arithmetic'][rung]=ar
            print(f'{rung}: {ar["correct"]}/{ar["count"]}',flush=True)
        result['text']=text_score(host,model,docs,256 if args.quick else 2048,seed)
        result['passed']={r:bool(result['text']['pass'] and ar['finite'] and ar['accuracy']>=THRESHOLDS[r])
                          for r,ar in result['arithmetic'].items()}
        del host
        if not args.quick:
            first,second=repeated_problems(docs,seed)
            one=exact_spaced(model,first,max_failures=0,return_mask=True)
            two=exact_spaced(model,second,max_failures=0,return_mask=True)
            joint=sum(a and b for a,b in zip(one.pop('_mask'),two.pop('_mask')))
            result['two_in_a_row']={'first':one,'second_given_correct_first':two,
                                   'both_correct':joint,'count':len(first),
                                   'joint_accuracy':joint/len(first),'gates_pass':False}
            if args.delta!='host':
                del model
                for sigma in (1e-5,1e-4,1e-3):
                    noisy,_=load_model(args.delta,noise=sigma,noise_seed=seed+int(sigma*1e9))
                    result['noise'][str(sigma)]={}
                    for rung,items in items_by_rung.items():
                        fn=natural_greedy if rung=='G5' else exact_spaced
                        result['noise'][str(sigma)][rung]=fn(noisy,items[:2000],max_failures=0)
                    del noisy
        atomic_json(args.output,result)
        record['result']=result|{'score_file':str(inside(args.output).relative_to(ROOT)),
                                 'score_sha256':sha(inside(args.output))}
        print(json.dumps(result,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true')
    p.add_argument('--delta',default='host'); p.add_argument('--rungs',default='G1')
    p.add_argument('--quick',action='store_true'); p.add_argument('--fresh',action='store_true')
    p.add_argument('--output'); args=p.parse_args()
    if args.prepare: prepare()
    else:
        assert args.output; run(args)
if __name__=='__main__': main()
