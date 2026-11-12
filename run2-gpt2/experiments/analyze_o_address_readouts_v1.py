"""O2: modulo-eight position, absolute recency and equals-query readouts.

All supervised targets are individual token kinds and absolute positions.
The recency slope is derived from the eight-phase circle's nearest gap; no
sum labels or language-model execution are used.
"""
from pathlib import Path
import argparse,json,sys,torch,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
from n_token_states_v1 import reconstruct
p=argparse.ArgumentParser();p.add_argument('--frontend',required=True);p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];delta=torch.load(ROOT/a.frontend,weights_only=True);base=torch.load(ROOT/'experiments/o1_coordinates/circuit.pt',weights_only=True);codes=base['position_codes'];eq=base['allowed_token_ids'][-1]
def augmented(x):
    x=x.reshape(-1,768).double();return torch.cat((x,torch.ones(len(x),1,dtype=torch.float64)),1)
def target(mode,position,tokens):
    if mode=='key':
        out=torch.cat((torch.ones(len(position),1,dtype=torch.float64),codes[position],((position.double()-31.5)/32)[:,None]),1);out[tokens==eq,1:3]=0
    else:
        out=torch.zeros(len(position),3,dtype=torch.float64);mask=tokens==eq;out[mask]=torch.cat((torch.ones(int(mask.sum()),1,dtype=torch.float64),codes[position[mask]]),1)
    return out
sets=[]
for s in ('a','b'):
    d=unpack(torch.load(ROOT/f'artifacts/o1_coordinates_format_{s}.pt',weights_only=True,mmap=True));_,norm=reconstruct(host,delta,d);pos=torch.tensor(d['program_starts'])[:,None]+torch.arange(12);sets.append({'x':augmented(norm),'position':pos.flatten(),'tokens':d['inputs'].flatten()})
data=torch.load(ROOT/'artifacts/o1_coordinates_calib.pt',weights_only=True,mmap=True);_,norm=reconstruct(host,delta,data);xt=augmented(norm);tt=data['inputs'].flatten();pt=torch.arange(64).repeat(512);allowed=torch.isin(tt,torch.tensor(base['allowed_token_ids']));xn=xt[~allowed];xp=xt[allowed];tp=tt[allowed];pp=pt[allowed];tm=xn.mean(0);tc=(xn-tm).T@(xn-tm)/len(xn);maps={};reports={}
for mode in ('key','query'):
    x=torch.cat((sets[0]['x'],xp));position=torch.cat((sets[0]['position'],pp));tokens=torch.cat((sets[0]['tokens'],tp));Y=target(mode,position,tokens);dimension=Y.shape[1];positive=torch.ones(len(x),dtype=torch.bool) if mode=='key' else tokens==eq;groups=position+64*(tokens==eq).long() if mode=='key' else position;means=[];targets=[];within=torch.zeros(769,769,dtype=torch.float64)
    for group in groups[positive].unique():
        mask=positive&(groups==group);z=x[mask];mean=z.mean(0);means.append(mean);targets.append(Y[mask][0]);r=z-mean;within+=r.T@r
    within/=int(positive.sum());extra=[]
    if mode=='query':
        for token in tokens[~positive].unique():
            z=x[tokens==token];mean=z.mean(0);extra.append(mean);r=z-mean;within+=r.T@r/int((~positive).sum())
    D=torch.cat((torch.stack(means),torch.stack(extra) if extra else torch.empty(0,769,dtype=torch.float64),tm[None,:]));T=torch.cat((torch.stack(targets),torch.zeros(len(extra)+1,dimension,dtype=torch.float64)));v=sets[1]['x'];expected=target(mode,sets[1]['position'],sets[1]['tokens']);vmask=torch.ones(len(v),dtype=torch.bool) if mode=='key' else sets[1]['tokens']==eq;trials=[];best=None
    for mass in (.001,.01,.1,1.,10.):
        M=within+mass*tc;M+=torch.eye(769,dtype=torch.float64)*(float(M.trace()/769)*1e-6+1e-10);inv=torch.linalg.solve(M,D.T);U,S,Vh=torch.linalg.svd(D@inv,full_matrices=False);keep=S>S.max()*1e-12;decoder=inv@Vh[keep].T@((U[:,keep].T@T)/S[keep,None]);error=v@decoder-expected;nt=xn@decoder;report={'mode':mode,'text_mass':mass,'interpolation_error':float((D@decoder-T).abs().max()),'positive_max_error':float(error[vmask].abs().max()),'positive_rms_error':float(error[vmask].square().mean().sqrt()),'negative_program_max':0. if mode=='key' else float(error[~vmask].abs().max()),'negative_text_max':float(nt.abs().max()),'negative_text_rms':float(nt.square().mean().sqrt())};trials.append(report);print(json.dumps(report),flush=True);quality=report['positive_rms_error']+2*report['negative_text_rms']+.5*report['negative_program_max']
        if best is None or quality<best[0]:best=(quality,decoder,report)
    maps[mode]=best[1];reports[mode]={'chosen':best[2],'trials':trials}
slope=(1-math.cos(math.pi/4))/5;recency=32*slope;lift=2.;v=sets[1]['x'].reshape(512,12,769);key=v@maps['key'];query=v@maps['query'];q=query[:,7];textkey=xn@maps['key'];textquery=xn@maps['query'];prefixkey=xp@maps['key'];prefixquery=xp@maps['query'];equals_positions=sets[1]['position'].reshape(512,12)[:,7];roles=[];headmaps=[]
for role in range(7):
    offset=7-role;theta=math.pi*offset/4;rotation=torch.tensor([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]],dtype=torch.float64);fallback=1-slope*(offset+4);Q=torch.cat((maps['query'][:,1:]@rotation,maps['query'][:,:1]),1);K=torch.cat((maps['key'][:,1:3]+fallback*(maps['query'][:,1:]@rotation),lift*maps['key'][:,:1]+recency*maps['key'][:,3:4]),1);qe=v[:,7]@Q;ks=v[:,:8]@K;score=(qe[:,None,:]*ks).sum(2);truth=score[:,role];dummy=score[:,7];wrong=score.masked_fill(torch.arange(8)[None,:]==role,-torch.inf).max(1).values;others=score.masked_fill(((torch.arange(8)==role)|(torch.arange(8)==7))[None,:],-torch.inf).max(1).values;wordscore=qe@(xn@K).T;prefixscore=qe@(xp@K).T;invalid=(pp[None,:]>=equals_positions[:,None])|(pp[None,:]==(equals_positions-offset)[:,None]);prefixscore=prefixscore.masked_fill(invalid,-torch.inf);maximum=prefixscore.max(1).values
    report={'role':role,'offset':offset,'fallback_scale':fallback,'selected_program_count':int((score.argmax(1)==role).sum()),'target_program_gap_min':float((truth-wrong).min()),'dummy_program_gap_min':float((dummy-others).min()),'target_word_gap_min':float((truth-wordscore.max(1).values).min()),'target_prefix_numeric_gap_min':float((truth-maximum).min()),'dummy_prefix_numeric_gap_min':float((dummy-maximum).min())};roles.append(report);headmaps.append({'query':Q,'key':K});print('relative',json.dumps(report),flush=True)
aux={'maps':maps,'head_maps':headmaps,'reports':reports,'roles':roles,'slope_per_token':slope,'recency_coefficient':recency,'allowed_lift':lift,'ideal_minimum_address_margin':4*slope,'frequency':8.};torch.save(aux,ROOT/f'artifacts/{a.tag}_address_readouts_v1.pt');(ROOT/f'artifacts/{a.tag}_address_readouts_v1.json').write_text(json.dumps({k:v for k,v in aux.items() if k not in ('maps','head_maps')},indent=2))
