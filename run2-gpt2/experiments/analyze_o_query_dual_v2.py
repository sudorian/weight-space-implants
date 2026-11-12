"""Minimum-noise homogeneous equals queries from known register constraints.

Targets are four existing position-register coordinates, never sums. The
native background is measured by subtracting the known MLP-zero program
columns from recorded component outputs. Five covariance masses are compared
only by syntax-address margins and ordinary query-score energy.
"""
from pathlib import Path
import json,sys,math,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
from n_token_states_v1 import reconstruct
from circuit_io_v1 import minimum_metric_map
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];writerpath=ROOT/'experiments/o2b_tokens/delta.pt';wd=torch.load(writerpath,weights_only=True);writer=torch.load(writerpath.parent/'circuit.pt',weights_only=True);readerpath=ROOT/'experiments/o3e_reader/delta.pt';rd=torch.load(readerpath,weights_only=True);reader=torch.load(readerpath.parent/'circuit.pt',weights_only=True);g=host['transformer.h.1.ln_1.weight'].double();b=host['transformer.h.1.ln_1.bias'].double()
def augment(x):
    x=x.double();return torch.cat((x,torch.ones(*x.shape[:-1],1,dtype=torch.float64)),-1)
text=torch.load(ROOT/'artifacts/o1_coordinates_calib.pt',weights_only=True,mmap=True);raw,norm=reconstruct(host,wd,text);xt=augment(norm.reshape(-1,768));normalized=((xt[:,:768]-b)/g);Ct=normalized.T@normalized/len(normalized);mean=normalized.mean(0);forms=[]
for name in ('a','b'):
    data=unpack(torch.load(ROOT/f'artifacts/o1_coordinates_format_{name}.pt',weights_only=True,mmap=True));raw,norm=reconstruct(host,wd,data);pre=torch.nn.functional.linear(data['captures']['0.ln2'],writer['program_rows'][:,:768].float(),writer['program_rows'][:,768].float());fields=torch.nn.functional.gelu(pre).double()@writer['emission_coefficients'].T;program=(fields*writer['field_gains'])@writer['frame'].T;native=raw.double()-program;std=(raw.double().var(-1,unbiased=False,keepdim=True)+1e-5).sqrt();native=(native-raw.double().mean(-1,keepdim=True))/std;forms.append({'x':augment(norm),'native':native,'input':data['inputs'],'starts':torch.tensor(data['program_starts'])})
native=forms[0]['native'][:,7];Cp=native.T@native/len(native);D=torch.cat(((writer['frame']*writer['field_gains']).T,torch.ones(1,768,dtype=torch.float64),native.mean(0)[None,:],mean[None,:]));target=torch.zeros(len(D),4,dtype=torch.float64);target[9:13]=torch.eye(4,dtype=torch.float64)
def parameter(name):
    v=host[name].clone();u=rd['updates'].get(name)
    if u is not None:v.view(-1)[u['indices'].long()]=u['values']
    return v.double()
QKV=parameter('transformer.h.1.attn.c_attn.weight');bias=parameter('transformer.h.1.attn.c_attn.bias');states=forms[1]['x'];qe=states[:,7];cache=[];allowed=torch.tensor(torch.load(ROOT/'experiments/o1_coordinates/circuit.pt',weights_only=True)['allowed_token_ids']);word=~torch.isin(text['inputs'].flatten(),allowed);positions=torch.arange(64).repeat(512);eqpos=forms[1]['starts']+7;score=reader['attention_score'];score=256. # O3E changes the inherited score512 auxiliary.
for h in reader['head_aux']:
    idx=torch.arange(64*h['head'],64*(h['head']+1));slots=h['qk_slots'];local=slots-64*h['head'];Q=qe[:,:768]@QKV[idx].T+bias[idx];Q[:,local]=0;K=states[:,:8,:768]@QKV[768+idx].T+bias[768+idx];Kw=xt[:,:768]@QKV[768+idx].T+bias[768+idx];base_program=torch.einsum('bd,bkd->bk',Q,K)/8;native_word=Q@Kw.T/8;cache.append({'h':h,'local':local,'K':K[:,:,local],'Kw':Kw[:,local],'base_program':base_program,'native_word':native_word,'key_cov':Kw[:,local].T@Kw[:,local]/len(Kw)})
trials=[];candidates=[];scale=(8*score)**.5
for mass in (.001,.01,.1,1.,10.):
    metric=Cp+mass*Ct;metric.diagonal().add_(float(metric.trace()/768)*1e-6);rawmap,error=minimum_metric_map(D,target,metric);maps=rawmap/g[:,None];maps=torch.cat((maps,(-b@maps)[None,:]));eqvalues=qe@maps[:,0];native_error=forms[1]['native'][:,7]@rawmap
    for flag_method in ('existing_token_flag','homogeneous_equals_constant'):
        roles=[];queries=[];total_energy=0.
        for c in cache:
            h=c['h'];role=h['role'];theta=math.pi*(7-role)/4;rotation=torch.tensor([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]],dtype=torch.float64);oldflag=h['query_map'][:,4];kappa=float((qe@oldflag).mean()/eqvalues.mean());flag=oldflag if flag_method=='existing_token_flag' else kappa*maps[:,0];query=torch.cat((maps[:,1:3]@rotation,maps[:,:1],maps[:,3:4],flag[:,None]),1);Q=scale*(qe@query);scores=c['base_program']+torch.einsum('bd,bkd->bk',Q,c['K'])/8;truth=scores[:,role];wrong=scores.masked_fill(torch.arange(8)[None,:]==role,-torch.inf).max(1).values;word=c['native_word']+Q@c['Kw'].T/8;valid_keys=(~torch.isin(text['inputs'].flatten(),allowed))[None,:] | ((positions[None,:]<eqpos[:,None])&(positions[None,:]!=(eqpos-(7-role))[:,None]));word=word.masked_fill(~valid_keys,-torch.inf);word_margin=float((truth-word.max(1).values).min());wordQ=scale*(xt@query);Qcov=wordQ.T@wordQ/len(wordQ);energy=float(torch.trace(Qcov@c['key_cov'])/64);total_energy+=energy;roles.append({'role':role,'selected':int((scores.argmax(1)==role).sum()),'program_margin':float((truth-wrong).min()),'word_margin':word_margin,'word_query_rms':wordQ.square().mean(0).sqrt().tolist(),'score_energy':energy,'kappa':kappa});queries.append(query)
        report={'text_mass':mass,'flag_method':flag_method,'constraint_error':error,'equals_constant_range':[float(eqvalues.min()),float(eqvalues.max())],'native_query_max_error':float(native_error.abs().max()),'score_energy':total_energy,'minimum_margin':min(min(q['program_margin'],q['word_margin']) for q in roles),'all_addresses_correct':all(q['selected']==512 for q in roles),'roles':roles};trials.append(report);candidates.append({'maps':maps,'queries':queries,'report':report});print(json.dumps(report),flush=True)
valid=[c for c in candidates if c['report']['all_addresses_correct'] and c['report']['minimum_margin']>=1.];best=min(valid,key=lambda c:c['report']['score_energy']) if valid else max(candidates,key=lambda c:c['report']['minimum_margin']);torch.save({'chosen':best,'trials':trials,'candidates':candidates,'attention_score':score,'text_masses':[.001,.01,.1,1.,10.]},ROOT/'artifacts/o_query_dual_v2.pt');(ROOT/'artifacts/o_query_dual_v2.json').write_text(json.dumps({'chosen':best['report'],'trials':trials},indent=2)+'\n');print('CHOSEN',json.dumps(best['report']),flush=True)
