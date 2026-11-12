"""P3: four paired-column attention heads, with binary values and type evidence.

All token maps, address kernels and frame choices use token/position labels or
native activation covariances. No complete answer targets or model execution.
Only a single attention component on recorded block-one LN1 is checked below.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from n_token_states_v1 import reconstruct
from prefix_capture_io_v1 import unpack
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--value-gain',type=float,default=2.);p.add_argument('--cosine-gain',type=float,default=256.);a=p.parse_args();assert a.value_gain in (.5,1.,2.,4.,8.);assert a.cosine_gain in (128.,256.,512.,1024.,2048.);torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/p2f_bit_position/delta.pt';delta=torch.load(base,weights_only=True);manifest=json.loads(base.with_suffix('.manifest.json').read_text());old=torch.load(base.parent/'circuit.pt',weights_only=True);digits=torch.load(ROOT/'artifacts/p2f_bit_position_source_readouts_v3.pt',weights_only=True);geo_path=ROOT/'artifacts/p_recency_bits_v3_shift-2048.pt';geometry=torch.load(geo_path,weights_only=True);candidate=next(c for c in geometry['candidates'] if c['report']['cosine_gain']==a.cosine_gain);coord=torch.load(ROOT/'experiments/p1b_coordinates/circuit.pt',weights_only=True)
def augmented(x):
 x=x.double();return torch.cat((x,torch.ones(*x.shape[:-1],1,dtype=torch.float64)),-1)
data=torch.load(ROOT/'artifacts/p1b_coordinates_calib.pt',weights_only=True,mmap=True);raw,norm=reconstruct(host,delta,data);raw=raw.reshape(-1,768).double();xt=augmented(norm.reshape(-1,768));values=torch.load(ROOT/'artifacts/p1_coordinates_attention_values.pt',weights_only=True,mmap=True)['captures']['1.attn_values'].reshape(-1,768).double();forms=[]
for tag in ('a','b'):
 d=unpack(torch.load(ROOT/f'artifacts/p1b_coordinates_format_{tag}.pt',weights_only=True,mmap=True));r,n=reconstruct(host,delta,d);forms.append({'raw':r.double(),'x':augmented(n),'inputs':d['inputs']})
P=host['transformer.h.1.attn.c_proj.weight'].double();pb=host['transformer.h.1.attn.c_proj.bias'].double();heads=geometry['heads'];qkv=host['transformer.h.1.attn.c_attn.weight'].double().clone();qb=host['transformer.h.1.attn.c_attn.bias'].double().clone();pp=P.clone();pbb=pb.clone();native_values=values.clone();head_aux=[];allslots=[]
for role,head in enumerate(heads):
 indices=torch.arange(head*64,(head+1)*64);hm=candidate['maps'][role];qkv[indices]=hm['native_query']['weight'];qb[indices]=hm['native_query']['bias'];q=xt[:,:768]@qkv[indices].T+qb[indices];k=xt[:,:768]@qkv[768+indices].T+qb[768+indices];cq=q.T@q/len(q);ck=k.T@k/len(k)
 def roots(C):
  e,U=torch.linalg.eigh(C);e=e.clamp_min(float(e.max())*1e-9);return (U*e.sqrt()[None,:])@U.T,(U*e.rsqrt()[None,:])@U.T
 rootq,invq=roots(cq);rootk,invk=roots(ck);uq,singular,vkh=torch.linalg.svd(rootq@rootk,full_matrices=False);singular=singular.flip(0);Tq=(invq@uq.flip(1))*singular.sqrt()[None,:];Tk=(invk@vkh.T.flip(1))*singular.sqrt()[None,:];invariance=float(((q[:32]@Tq)@(k[:32]@Tk).T-q[:32]@k[:32].T).abs().max());assert invariance<1e-5
 qkv[indices]=Tq.T@qkv[indices];qb[indices]=Tq.T@qb[indices];qkv[768+indices]=Tk.T@qkv[768+indices];qb[768+indices]=Tk.T@qb[768+indices];slots=indices[:4];query=hm['query'];key=hm['key'];qkv[slots]=(8**.5)*query[:768].T;qb[slots]=(8**.5)*query[768];qkv[768+slots]=(8**.5)*key[:768].T;qb[768+slots]=(8**.5)*key[768]
 # Preserve native values in 32 rotated weak coordinates, replacing 6/6/6/1.
 local=values[:,indices];C=local.T@local/len(local);cost=C.diagonal()*P[:,indices].square().sum(0);vs=cost.argsort()[:32];vi=indices[vs];e,U=torch.linalg.eigh(C[vs][:,vs]);e=e.clamp_min(float(e.max())*1e-9);root=U*e.sqrt()[None,:];inv=e.rsqrt()[:,None]*U.T;_,_,Vh=torch.linalg.svd(P[:,vi]@root,full_matrices=False);RV=Vh.T.flip(1);rot=RV.T@inv;unrot=root@RV;qkv[1536+vi]=rot@qkv[1536+vi];qb[1536+vi]=rot@qb[1536+vi];pp[:,vi]=P[:,vi]@unrot;native_values[:,vi]=values[:,vi]@rot.T
 count=6 if role<3 else 1;reused=vi[:count];retained=indices[~torch.isin(indices,reused)];A=native_values[:,retained];B=native_values[:,reused];ma=A.mean(0);mb=B.mean(0);C=A.T@A/len(A)-ma[:,None]*ma[None,:];cross=A.T@B/len(A)-ma[:,None]*mb[None,:];C.diagonal().add_(.001*C.diagonal().clamp_min(1e-8));repair=torch.linalg.solve(C,cross);intercept=mb-ma@repair;removed=pp[:,reused].clone();pp[:,retained]+=removed@repair.T;pbb+=removed@intercept;pp[:,reused]=0
 if role<3:
  decoders=[digits['value_bit_decoder'][:,j] for j in range(4)]
  if role==0:decoders += [digits['value_kind_decoder'][:,1],digits['value_kind_decoder'][:,2]]
  else:decoders += [digits['value_kind_decoder'][:,2],digits['value_position_half_decoder']]
  fields=list(range(role*6,role*6+6))
 else:decoders=[digits['value_kind_decoder'][:,3]];fields=[18]
 for slot,decoder,field in zip(reused,decoders,fields):
  qkv[1536+slot]=decoder[:768];qb[1536+slot]=decoder[768];allslots.append((int(slot),field,a.value_gain))
 head_aux.append({'head':head,'role':role,'qk_rotation':Tq,'qk_key_transform':Tk,'qk_singular_values':singular,'qk_removed_score_energy_fraction':float(singular[:4].square().sum()/singular.square().sum()),'qk_invariance_error':invariance,'qk_slots':slots,'query_map':query,'key_map':key,'value_subset':vs,'value_rotation':rot,'value_inverse':unrot,'value_reused':reused,'value_repair':repair,'value_retained':retained,'value_intercept':intercept})
 print('head',role,head,'native score energy removed',head_aux[-1]['qk_removed_score_energy_fraction'],flush=True)
# Memory is orthogonal to untouched output heads and every writer coordinate.
edited_columns=torch.cat([torch.arange(h*64,(h+1)*64) for h in heads]);untouched=torch.tensor([i for i in range(768) if i//64 not in heads]);mean_attention=pp@native_values.mean(0)+pbb;keep=host['transformer.h.1.ln_2.weight'].abs()>.03;constraints=torch.cat((torch.ones(1,768,dtype=torch.float64),old['frame'].T,P[:,untouched].T,raw.mean(0)[None,:],mean_attention[None,:],torch.eye(768,dtype=torch.float64)[~keep]));_,sv,Vh=torch.linalg.svd(constraints,full_matrices=True);N=Vh[int((sv>sv.max()*1e-8).sum()):].T;program=torch.cat([d['raw'][:,7] for d in forms]);C=program.T@program/len(program);ev,U=torch.linalg.eigh(N.T@C@N);program_vectors=U;cv=native_values.T@native_values/len(native_values);attention=pp@cv@pp.T;memory_trials=[];memory_candidates=[]
for width in (32,48,64,96,128):
 low=N@program_vectors[:,:min(width,N.shape[1])];ev,U=torch.linalg.eigh(low.T@attention@low);candidate_memory=low@U[:,:19];report={'width':width,'program_raw_max':float((program@candidate_memory).abs().max()),'text_raw_rms_max':float((raw@candidate_memory).square().mean(0).sqrt().max()),'removed_attention_energy_fraction':float(torch.trace(candidate_memory.T@attention@candidate_memory)/attention.trace())};memory_trials.append(report);memory_candidates.append(candidate_memory)
valid=[i for i,r in enumerate(memory_trials) if r['program_raw_max']<=.02];chosen_memory=min(valid,key=lambda i:memory_trials[i]['removed_attention_energy_fraction']) if valid else min(range(5),key=lambda i:memory_trials[i]['program_raw_max']);memory=memory_candidates[chosen_memory];memory_report={'null_dimension':N.shape[1],**memory_trials[chosen_memory],'untouched_projection_max':float((memory.T@P[:,untouched]).abs().max())};print('memory trials',json.dumps(memory_trials),flush=True);pp[:,edited_columns]-=memory@(memory.T@pp[:,edited_columns]);pbb-=memory@(memory.T@pbb)
for slot,field,gain in allslots:pp[:,slot]=gain*memory[:,field]
# Test the four attention pieces on independently recorded full-prefix LN1.
full=torch.load(ROOT/'artifacts/p2f_full_prefix_b.pt',weights_only=True,mmap=True);X=augmented(full['captures']['1.ln1']);starts=torch.tensor(full['request']['program_starts']);positions=starts+7;Qinput=X[torch.arange(len(X)),positions];causal=torch.arange(64)[None,:]<=positions[:,None];mem=torch.zeros(len(X),19,dtype=torch.float64);head_checks=[]
for h in head_aux:
 indices=torch.arange(h['head']*64,(h['head']+1)*64);q=Qinput[:,:768]@qkv[indices].T+qb[indices];k=X[:,:,:768]@qkv[768+indices].T+qb[768+indices];scores=(q[:,None]*k).sum(-1)/8;weights=torch.softmax(scores.masked_fill(~causal,-torch.inf),1)
 for slot,field,gain in allslots:
  if slot//64==h['head']:mem[:,field]=(weights*(X[:,:,:768]@qkv[1536+slot]+qb[1536+slot])).sum(1)
 head_checks.append({'head':h['head'],'role':h['role'],'minimum_correct_source_weight':float(weights[torch.arange(len(X)),starts+(3 if h['role']==3 else h['role']+4)].min())})
lookup={int(t):i%10 for i,t in enumerate(coord['allowed_token_ids'][:20])};source_digits=torch.tensor([[lookup[int(full['inputs'][i,starts[i]+j])] for j in (0,1,2,4,5,6)] for i in range(len(X))]);expected=torch.stack([((source_digits[:,j,None]>>torch.arange(4))&1)+((source_digits[:,j+3,None]>>torch.arange(4))&1) for j in range(3)],1);ratios=[]
for role in range(3):
 mass=mem[:,4]+mem[:,5] if role==0 else mem[:,6*role+4];ratios.append(mem[:,6*role:6*role+4]/mass[:,None])
ratios=torch.stack(ratios,1);bitcounts=(ratios>.1).long()+(ratios>.9).long();err=bitcounts!=expected;syntax=torch.stack((mem[:,4],mem[:,5],mem[:,11],mem[:,10]-mem[:,11],mem[:,17],mem[:,16]-mem[:,17],mem[:,18]),1);checks={'bit_errors':int(err.sum()),'examples_with_bit_error':int(err.flatten(1).any(1).sum()),'syntax_min':syntax.min(0).values.tolist(),'heads':head_checks,'epsilon':.1};print('piece checks',json.dumps(checks),flush=True);assert checks['examples_with_bit_error']<=5,checks
changed={}
for key,update in delta['updates'].items():
 value=host[key].clone();value.view(-1)[update['indices'].long()]=update['values'];changed[key]=value
changed.update({'transformer.h.1.attn.c_attn.weight':qkv.float(),'transformer.h.1.attn.c_attn.bias':qb.float(),'transformer.h.1.attn.c_proj.weight':pp.float(),'transformer.h.1.attn.c_proj.bias':pbb.float()});out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);fields=[f'{column}_{bit}' for column in ('hundreds','tens','units') for bit in ('bit0','bit1','bit2','bit3','type_mass','position_half_or_second_type')]+['plus'];torch.save({'memory':memory,'value_gain':a.value_gain,'heads':heads,'head_aux':head_aux,'cosine_gain':a.cosine_gain,'fallback_shift':-2048.,'memory_report':memory_report,'memory_width_trials':memory_trials,'piece_checks':checks,'fields':fields,'token_writer_frame':old['frame']},out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in manifest['sources']]+[Path(__file__),ROOT/'experiments/n_token_states_v1.py',ROOT/'experiments/analyze_p_source_readouts_v3.py',ROOT/'experiments/analyze_p_recency_bits_v3.py'],inputs=[base,base.parent/'circuit.pt',geo_path,ROOT/'artifacts/p2f_bit_position_source_readouts_v3.pt',ROOT/'artifacts/p1b_coordinates_calib.pt',ROOT/'artifacts/p1_coordinates_attention_values.pt',ROOT/'artifacts/p2f_full_prefix_b.pt']+[ROOT/f'artifacts/p1b_coordinates_format_{s}.pt' for s in ('a','b')],arguments=vars(a),description='P3: four recency-biased paired readers with four-bit digit values, exact token types and spaced-digit position-half evidence; 4 QK and 6/6/6/1 value coordinates reused.',extra={'heads':heads,'qk_reused':4,'value_subspace_width':32,'cosine_gain_trials_declared':[128,256,512,1024,2048],'value_gain_trials_declared':[.5,1,2,4,8],'fallback_shift_trials_declared':[0,-512,-1024,-2048,-4096],'piece_checks':checks,'memory_report':memory_report,'memory_width_trials':memory_trials})
