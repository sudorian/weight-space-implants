"""Test a two-token syntax gate before writing position fields.

Targets are only current/previous token kinds. This is a fixed Fisher solve
with five declared regularizers, never an arithmetic-answer target.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False);coord=torch.load(ROOT/'experiments/p1_coordinates/circuit.pt',weights_only=True);ids=coord['allowed_token_ids'];bare=torch.tensor(ids[:10]);spaced=torch.tensor(ids[10:20]);plus,eq=ids[20:];allowed=torch.tensor(ids)
def labels(tokens,previous):
    return (torch.isin(tokens,bare)&(previous==198))|(torch.isin(tokens,spaced)&(torch.isin(previous,torch.cat((bare,spaced)))|(previous==plus)))|(((tokens==plus)|(tokens==eq))&torch.isin(previous,spaced))
def form(label):
    d=unpack(torch.load(ROOT/f'artifacts/p1_coordinates_{label}.pt',weights_only=True,mmap=True));positions=torch.tensor(d['program_starts'])[:,None]+torch.arange(12);previous=d['full_inputs'][torch.arange(512)[:,None],positions-1];return {'x':d['captures']['0.ln2'].double().reshape(-1,768),'y':labels(d['inputs'],previous).flatten(),'allowed':torch.isin(d['inputs'],allowed).flatten(),'tokens':d['inputs'].flatten()}
forms=[form(s) for s in ('format_a','format_b','bad')];d=torch.load(ROOT/'artifacts/p1_coordinates_calib.pt',weights_only=True,mmap=True);previous=torch.cat((torch.full((512,1),-1,dtype=torch.long),d['inputs'][:,:-1]),1);text={'x':d['captures']['0.ln2'].double().reshape(-1,768),'y':labels(d['inputs'],previous).flatten(),'allowed':torch.isin(d['inputs'],allowed).flatten(),'tokens':d['inputs'].flatten()};ntrain=384*64
construction=[forms[0],{k:v[:ntrain] for k,v in text.items()},forms[2]];positives=torch.cat([s['x'][s['y']] for s in construction]);negatives=torch.cat([s['x'][s['allowed']&~s['y']] for s in construction]);pm=positives.mean(0);nm=negatives.mean(0);pc=positives-pm;nc=negatives-nm;cov=pc.T@pc/len(pc)+nc.T@nc/len(nc);trials=[]
for ridge in (1e-5,1e-4,.001,.01,.1):
    M=cov.clone();M.diagonal().add_(ridge*float(cov.trace()/768));w=torch.linalg.solve(M,pm-nm);w/=w@(pm-nm);bias=-w@nm;report={'ridge':ridge,'positive_construction_count':len(positives),'negative_construction_count':len(negatives),'domains':{}}
    for label,s in zip(('format_a','format_b','bad','calib'),forms+[text]):
        values=s['x']@w+bias;positive=values[s['y']];negative=values[s['allowed']&~s['y']];record={'positive_min':float(positive.min()),'negative_max':float(negative.max()) if len(negative) else None,'positive_count':len(positive),'negative_count':len(negative),'word_max':float(values[~s['allowed']].max()) if (~s['allowed']).any() else None};record['gap']=record['positive_min']-record['negative_max'] if record['negative_max'] is not None else None;report['domains'][label]=record
    trials.append({'weight':w,'bias':float(bias),'report':report});print(json.dumps(report),flush=True)
torch.save({'trials':trials,'semantics':'bare after newline; spaced digit after bare/spaced/plus; plus/equals after spaced digit'},ROOT/'artifacts/p_local_source_gate_v1.pt');(ROOT/'artifacts/p_local_source_gate_v1.json').write_text(json.dumps([q['report'] for q in trials],indent=2)+'\n')
