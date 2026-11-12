"""Separate native windows and format controls from a frozen mixed capture."""
import torch

def unpack(data):
    request=data['request'];n=request['calibration_count'];starts=request['program_starts'];domains=request['domains']
    text={'inputs':data['inputs'][:n],'captures':{},'ntrain':request['native_construction_count']*64}
    program={'inputs':torch.stack([data['inputs'][i,starts[i]:starts[i]+12] for i in range(n,len(starts))]),'full_inputs':data['inputs'][n:],'program_starts':starts[n:],'domains':domains[n:],'captures':{}}
    for tag,value in data['captures'].items():
        prefix,key=tag.split('/',1)
        if prefix=='text':text['captures'][key]=value
        else:
            start=int(prefix[1:]);indices=[i-n for i in range(n,len(starts)) if starts[i]==start]
            if key not in program['captures']:program['captures'][key]=torch.empty((len(starts)-n,)+tuple(value.shape[1:]),dtype=value.dtype)
            program['captures'][key][indices]=value
    return text,program
