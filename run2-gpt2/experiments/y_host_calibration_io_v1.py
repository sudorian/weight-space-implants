"""Unpack only recorded portions of the balanced host calibration pass."""
import torch

def unpack(data):
    n=data['request']['native_count'];native={'inputs':data['inputs'][:n],'captures':{k:v for k,v in data['captures'].items() if not k.startswith('p')},'ntrain':data['request']['native_construction_count']*64};count=len(data['inputs'])-n;raw=torch.empty(count,8,768);gelu=torch.empty(count,8,3072)
    for key,config in data['request']['capture'].items():
        if not key.startswith('p'):continue
        indices=torch.tensor(config['sequences'])-n
        if key.endswith('4.raw'):raw[indices]=data['captures'][key]
        elif key.endswith('4.gelu'):gelu[indices]=data['captures'][key]
    starts=data['request']['program_starts'][n:];bad={'inputs':torch.stack([data['inputs'][n+i,s:s+8] for i,s in enumerate(starts)]),'captures':{'4.raw':raw,'4.gelu':gelu},'starts':starts,'source_indices':data['request']['bad_source_indices']};return native,bad
