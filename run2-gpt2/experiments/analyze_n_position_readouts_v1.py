"""N4 position and equals-query detectors from recorded token/position labels.

Exact class-mean interpolation with five declared text covariance masses.
These are address and token-format targets, never arithmetic answers.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
from n_token_states_v1 import reconstruct
p=argparse.ArgumentParser();p.add_argument('--frontend',required=True);p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];delta=torch.load(ROOT/a.frontend,weights_only=True);circuit=torch.load((ROOT/a.frontend).parent/'circuit.pt',weights_only=True);base=torch.load(ROOT/'experiments/n3_coordinates/circuit.pt',weights_only=True);codes=base['position_codes'];table=torch.cat((torch.ones(64,1,dtype=torch.float64),codes),1)
def augmented(x):
    x=x.reshape(-1,768).double();return torch.cat((x,torch.ones(len(x),1,dtype=torch.float64)),1)
sets=[]
for s in ('a','b'):
    d=unpack(torch.load(ROOT/f'artifacts/n3_coordinates_format_{s}.pt',weights_only=True,mmap=True));raw,norm=reconstruct(host,delta,d);pos=torch.tensor(d['program_starts'])[:,None]+torch.arange(12);sets.append({'x':augmented(norm),'position':pos.flatten(),'tokens':d['inputs'].flatten(),'shape':norm.shape[:2]})
data=torch.load(ROOT/'artifacts/n3_coordinates_calib.pt',weights_only=True,mmap=True);_,ln=reconstruct(host,delta,data);xt=augmented(ln);negative=~torch.isin(data['inputs'].flatten(),torch.tensor(base['allowed_token_ids']));xn=xt[negative];tm=xn.mean(0);tc=(xn-tm).T@(xn-tm)/len(xn);x=sets[0]['x'];v=sets[1]['x'];eq=base['allowed_token_ids'][-1];reports={};maps={}
for mode in ('key','query'):
    mask=torch.ones(len(x),dtype=torch.bool) if mode=='key' else sets[0]['tokens']==eq;vmask=torch.ones(len(v),dtype=torch.bool) if mode=='key' else sets[1]['tokens']==eq;positions=sets[0]['position'];means=[];targets=[];within=torch.zeros(769,769,dtype=torch.float64)
    for pos in positions[mask].unique():
        group=x[mask&(positions==pos)];mean=group.mean(0);means.append(mean);targets.append(table[pos]);r=group-mean;within+=r.T@r
    within/=int(mask.sum());extra=[]
    if mode=='query':
        neg=x[~mask]
        for token in sets[0]['tokens'][~mask].unique():
            group=x[sets[0]['tokens']==token];mean=group.mean(0);extra.append(mean);r=group-mean;within+=r.T@r/len(neg)
    D=torch.cat((torch.stack(means),torch.stack(extra) if extra else torch.empty(0,769,dtype=torch.float64),tm[None,:]));T=torch.cat((torch.stack(targets),torch.zeros(len(extra)+1,9,dtype=torch.float64)));trials=[];best=None
    for mass in (.001,.01,.1,1.,10.):
        M=within+mass*tc;M+=torch.eye(769,dtype=torch.float64)*(float(M.trace()/769)*1e-6+1e-10);inv=torch.linalg.solve(M,D.T);U,S,Vh=torch.linalg.svd(D@inv,full_matrices=False);keep=S>S.max()*1e-12;decoder=inv@Vh[keep].T@((U[:,keep].T@T)/S[keep,None]);expected=torch.zeros(len(v),9,dtype=torch.float64);expected[vmask]=table[sets[1]['position'][vmask]];value=v@decoder;error=value-expected;nt=xn@decoder;report={'mode':mode,'text_mass':mass,'interpolation_error':float((D@decoder-T).abs().max()),'positive_max_error':float(error[vmask].abs().max()),'positive_rms_error':float(error[vmask].square().mean().sqrt()),'negative_program_max':0. if mode=='key' else float(value[~vmask].abs().max()),'negative_text_max':float(nt.abs().max()),'negative_text_rms':float(nt.square().mean().sqrt())};trials.append(report);print(json.dumps(report),flush=True)
        quality=report['positive_rms_error']+report['negative_text_rms']+(.1*report['negative_program_max'] if mode=='query' else 0.)
        if best is None or quality<best[0]:best=(quality,decoder,report)
    maps[mode]=best[1];reports[mode]={'chosen':best[2],'trials':trials}
key=(v@maps['key']).reshape(512,12,9);query=(v@maps['query']).reshape(512,12,9)[:,7];textkeys=xn@maps['key'];relative=[]
for role in range(7):
    offset=7-role;theta=2*torch.pi*base['frequencies']*offset/64;rotation=torch.zeros(8,8,dtype=torch.float64)
    for j in range(4):rotation[2*j:2*j+2,2*j:2*j+2]=torch.tensor([[theta[j].cos(),-theta[j].sin()],[theta[j].sin(),theta[j].cos()]])
    # Row-vector multiplication rotates the query backwards by the offset.
    shifted=query[:,1:]@rotation;score=(shifted[:,None,:]*key[:,:,1:]).sum(2)-.78*query[:,0,None]*key[:,:,0];truth=score[:,role];wrong=score.masked_fill(torch.arange(12)[None,:]==role,-torch.inf).max(1).values;textmax=(shifted@textkeys[:,1:].T-.78*query[:,0,None]*textkeys[None,:,0]).max(1).values;report={'role':role,'target_score_min':float(truth.min()),'wrong_program_max':float(wrong.max()),'program_gap_min':float((truth-wrong).min()),'text_key_gap_min':float((truth-textmax).min()),'selected_program_count':int((score.argmax(1)==role).sum())};relative.append(report);print('relative',json.dumps(report),flush=True)
torch.save({'maps':maps,'reports':reports,'relative':relative,'alpha':.78,'frequencies':base['frequencies']},ROOT/f'artifacts/{a.tag}_position_readouts_v1.pt');(ROOT/f'artifacts/{a.tag}_position_readouts_v1.json').write_text(json.dumps({'reports':reports,'relative':relative},indent=2))
