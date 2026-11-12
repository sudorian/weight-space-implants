"""Hand paired-position kernels with equal ideal scores at both operands.

Modulo-four plus a quadratic relative-position penalty isolates a pair four
tokens apart. A separate equals key is the zero-valued fallback. Targets are
addresses and token types only, never arithmetic answers.
"""
from pathlib import Path
import argparse,json,sys,torch,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
from n_token_states_v1 import reconstruct
p=argparse.ArgumentParser();p.add_argument('--frontend',default='experiments/p2c_bits/delta.pt');p.add_argument('--tag',default='p2c_bits');a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];delta=torch.load(ROOT/a.frontend,weights_only=True);aux=torch.load((ROOT/a.frontend).parent/'circuit.pt',weights_only=True);coords=torch.load(ROOT/'experiments/p1_coordinates/circuit.pt',weights_only=True);token=torch.load(ROOT/f'artifacts/{a.tag}_source_readouts_v1.pt',weights_only=True);g=host['transformer.h.1.ln_1.weight'].double();b=host['transformer.h.1.ln_1.bias'].double();r=aux['frame']/g[:,None]/aux['field_gains'][None,:];r=torch.cat((r,(-b@r)[None,:]));source=r[:,slice(*aux['source_slice'])];eq=r[:,slice(*aux['equals_slice'])];allowed=token['attention_kind_decoder'][:,1:5].sum(1);allowed=allowed.clone();allowed[-1]-=1;kappa=1/256
def aug(x):
    x=x.double();return torch.cat((x,torch.ones(*x.shape[:-1],1,dtype=x.dtype)),-1)
d=unpack(torch.load(ROOT/'artifacts/p1_coordinates_format_b.pt',weights_only=True,mmap=True));_,norm=reconstruct(host,delta,d);X=aug(norm);eqpos=torch.tensor(d['program_starts'])+7;text=torch.load(ROOT/'artifacts/p1_coordinates_calib.pt',weights_only=True,mmap=True);_,norm=reconstruct(host,delta,text);T=aug(norm.reshape(-1,768));positions=torch.arange(64).repeat(512);tokens=text['inputs'].flatten();numeric=torch.isin(tokens,torch.tensor(coords['allowed_token_ids']));trials=[];candidates=[]
for kappa in (1/1024,1/512,1/256,1/128,1/64):
    penalty=1.
    maps=[];reports=[]
    for role,(near,far,support) in enumerate(((3,7,[0,4,7]),(2,6,[1,5,7]),(1,5,[2,6,7]),(4,4,[3,7]))):
        theta=math.pi*near/2;rotation=torch.tensor([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]],dtype=torch.float64);Q=torch.cat((eq[:,1:3]@rotation,eq[:,:1],eq[:,3:5]),1)
        bonus=kappa*(16 if role<3 else 8);constant=-(1+kappa*near*far-bonus)*source[:,0]-32*kappa*(near+far)*source[:,3]-1024*kappa*source[:,4]+(1+kappa*near*far-math.cos(theta))*eq[:,0]+penalty*allowed
        K=torch.cat((source[:,1:3],constant[:,None],(32*kappa*(near+far)*source[:,0]+2048*kappa*source[:,3])[:,None],(-1024*kappa*source[:,0])[:,None]),1)
        q=X[:,7]@Q;scores=torch.einsum('bd,bkd->bk',q,X[:,:8]@K);good=scores[:,support[:-1]];wrong=scores.clone();wrong[:,support]=-torch.inf;word=q@(T@K).T;legal=positions[None,:]<=eqpos[:,None]
        for offset in (near,far,0):legal &= positions[None,:]!=(eqpos-offset)[:,None]
        legal |= ~numeric[None,:];word=word.masked_fill(~legal,-torch.inf);gap=torch.minimum(good.min(1).values-wrong.max(1).values,good.min(1).values-word.max(1).values);qword=T@Q;kword=T@K;Cq=qword.T@qword/len(T);Ck=kword.T@kword/len(T);report={'role':role,'support':support,'offsets':[near,far],'peak_bonus':bonus,'program_wrong_gap_min':float((good.min(1).values-wrong.max(1).values).min()),'word_wrong_gap_min':float((good.min(1).values-word.max(1).values).min()),'minimum_gap':float(gap.min()),'worst_word_token_is_allowed':bool(numeric[word.max(1).indices[(good.min(1).values-word.max(1).values).argmin()]]),'worst_word_position':int(positions[word.max(1).indices[(good.min(1).values-word.max(1).values).argmin()]]),'worst_query_position':int(eqpos[(good.min(1).values-word.max(1).values).argmin()]),'support_score_spread_max':float((good.max(1).values-good.min(1).values).max()),'operand_score_difference_max':float((good[:,0]-good[:,1]).abs().max()) if role<3 else 0.,'ordinary_score_second_moment':float(torch.trace(Cq@Ck)),'query_equals_rms':q.square().mean(0).sqrt().tolist(),'query_word_rms':qword.square().mean(0).sqrt().tolist()};reports.append(report);maps.append({'query':Q,'key':K,'role':role,'support':support,'offsets':[near,far]})
    report={'word_penalty':penalty,'kappa':kappa,'roles':reports,'minimum_gap':min(r['minimum_gap'] for r in reports),'ordinary_energy_sum':sum(r['ordinary_score_second_moment'] for r in reports)};trials.append(report);candidates.append({'head_maps':maps,'report':report});print(json.dumps(report),flush=True)
valid=[c for c in candidates if c['report']['minimum_gap']>0];best=min(valid,key=lambda c:c['report']['ordinary_energy_sum']/c['report']['minimum_gap']**2) if valid else max(candidates,key=lambda c:c['report']['minimum_gap']);out={'chosen':best,'trials':trials,'source_register_map':source,'equals_register_map':eq,'kappa':best['report']['kappa'],'frontend':a.frontend,'program_score_trials_declared':[512,1024,2048,4096,8192]};torch.save(out,ROOT/f'artifacts/{a.tag}_paired_address_v2.pt');(ROOT/f'artifacts/{a.tag}_paired_address_v2.json').write_text(json.dumps({'chosen':best['report'],'trials':trials},indent=2)+'\n');print('CHOSEN',json.dumps(best['report']),flush=True)
