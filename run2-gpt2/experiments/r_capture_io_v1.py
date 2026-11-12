"""Flatten grouped frozen token-position captures without running any model."""
import torch

def unpack(data):
    tags=sorted(k for k in data['captures'] if k.startswith('0.gelu.'))
    parts={k:[] for k in ('0.gelu','0.raw','0.ln2','sequence','position','token')}
    for tag in tags:
        spec=data['request']['capture'][tag];suffix=tag[len('0.gelu'):]
        sequences=torch.tensor(spec['sequences']);positions=torch.tensor(spec['positions'])
        for name in ('0.gelu','0.raw','0.ln2'):
            value=data['captures'][name+suffix]
            parts[name].append(value.reshape(-1,value.shape[-1]))
        parts['sequence'].append(sequences[:,None].expand(-1,len(positions)).reshape(-1))
        parts['position'].append(positions[None,:].expand(len(sequences),-1).reshape(-1))
        parts['token'].append(data['inputs'][sequences[:,None],positions[None,:]].reshape(-1))
    out={k:torch.cat(v) for k,v in parts.items()};order=(out['sequence']*64+out['position']).argsort()
    return {k:v[order] for k,v in out.items()}

def lexical_labels(classes):
    labels=torch.zeros(len(classes),21,dtype=torch.float64)
    for i,c in enumerate(classes):
        s=c['text'];stripped=s.strip()
        if stripped.isdigit():
            assert 1<=len(stripped)<=3
            for j,v in enumerate(map(int,stripped.zfill(3))):labels[i,4*j:4*j+4]=torch.tensor([(v>>b)&1 for b in range(4)])
            labels[i,12+len(stripped)-1]=1;labels[i,15]=int(s.startswith(' '))
        else:labels[i,16+['+',' +','=',' =','\n'].index(s)]=1
    return labels
