"""Views of a frozen full mixed capture, without evaluating any model."""
import torch

def unpack(data):
    n=data['request']['calibration_count'];nc=data['request']['native_construction_count']
    text={'inputs':data['inputs'][:n],'captures':{k:v[:n] for k,v in data['captures'].items()},'ntrain':nc*data['inputs'].shape[1]}
    starts=data['request']['program_starts'][n:]
    aligned={'inputs':torch.stack([data['inputs'][n+i,s:s+12] for i,s in enumerate(starts)]),'captures':{k:torch.stack([v[n+i,s:s+12] for i,s in enumerate(starts)]) for k,v in data['captures'].items()},'domains':data['request']['domains'][n:],'starts':starts}
    full={'inputs':data['inputs'][n:],'captures':{k:v[n:] for k,v in data['captures'].items()},'domains':aligned['domains'],'starts':starts}
    return text,aligned,full
