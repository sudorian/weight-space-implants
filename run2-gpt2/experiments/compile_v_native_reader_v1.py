"""U2: seven period-eight individual-offset readers for the complete spaced format.

Four QK coordinates implement cosine plus linear recency.
Value labels are individual digit bits and expected token kinds. No sum targets.
Native rotations, repairs and output memories use unlabeled recorded covariances.
"""
from pathlib import Path
import argparse,json,math,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from u_token_states_v1 import reconstruct
from prefix_capture_io_v1 import unpack
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--value-gain',type=float,default=4.);a=p.parse_args();assert a.value_gain in (.5,1.,2.,4.,8.);torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/v1_protected_equals_writer/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());delta=torch.load(base,weights_only=True);writer=torch.load(base.parent/'circuit.pt',weights_only=True);tokenpath=ROOT/'artifacts/v1_protected_equals_writer_source_readouts_v3.pt';token=torch.load(tokenpath,weights_only=True);coord=writer['coordinate_geometry']['previous'];g=host['transformer.h.1.ln_1.weight'].double();b=host['transformer.h.1.ln_1.bias'].double();eqid=coord['allowed_token_ids'][-1]
def aug(x):
 return torch.cat((x.double(),torch.ones(*x.shape[:-1],1,dtype=torch.float64)),-1)
addresspath=ROOT/'artifacts/v_individual_address_v1.pt';address=torch.load(addresspath,weights_only=True);textpre=address['calib_native_raw'].float();tn=torch.nn.functional.layer_norm(textpre,(768,),g.float(),b.float(),1e-5);xt=aug(tn.reshape(-1,768));fullpath=ROOT/'artifacts/u1_full_prefix_b.pt';full=torch.load(fullpath,weights_only=True,mmap=True);fullpre=address['full_overlay_pre'];X=aug(torch.nn.functional.layer_norm(fullpre,(768,),g.float(),b.float(),1e-5));starts=torch.tensor(full['request']['program_starts']);ix=torch.arange(len(starts));positions=starts+7;query_input=X[ix,positions];causal=torch.arange(64)[None,:]<=positions[:,None];program=[]
for label in ('format_a','format_b'):
 d=unpack(torch.load(ROOT/f'artifacts/q1_sparse_coordinates_{label}.pt',weights_only=True,mmap=True));r,_=reconstruct(host,delta,d);program.append(r[:,7].double())
program=torch.cat(program);P=host['transformer.h.1.attn.c_proj.weight'].double();pb=host['transformer.h.1.attn.c_proj.bias'].double();valuespath=ROOT/'artifacts/host_calib512_early.pt';values=torch.load(valuespath,weights_only=True,mmap=True)['captures']['1.attn_values'].reshape(-1,768).double();heads=address['heads'];native=address['native_queries'];lookup={int(t):i%10 for i,t in enumerate(coord['allowed_token_ids'][:20])};trial_reports=address['reports']
# Scale2 has minimum source weight.945 and every bit correct; exact one-hot
# attention is unnecessary because the next piece thresholds individual bits.
chosen={'maps':address['maps'][1],'report':trial_reports[1]};assert chosen['report']['minimum_target_weight']>.9
W=host['transformer.h.1.attn.c_attn.weight'].double();bias=host['transformer.h.1.attn.c_attn.bias'].double()
qkv=W.clone();qb=bias.clone();pp=P.clone();pbb=pb.clone();nv=values.clone();head_aux=[];slots=[];field_names=[];field=0
for role,m in enumerate(chosen['maps']):
 head=m['head'];inds=torch.arange(64*head,64*(head+1));nq,nb=native[role];qkv[inds]=nq;qb[inds]=nb;q=xt[:,:768]@nq.T+nb;k=xt[:,:768]@W[768+inds].T+bias[768+inds];cq=q.T@q/len(q);ck=k.T@k/len(k)
 def roots(C):
  e,U=torch.linalg.eigh(C);e=e.clamp_min(float(e.max())*1e-9);return (U*e.sqrt()[None,:])@U.T,(U*e.rsqrt()[None,:])@U.T
 rq,iq=roots(cq);rk,ik=roots(ck);U,s,Vh=torch.linalg.svd(rq@rk,full_matrices=False);s=s.flip(0);Tq=(iq@U.flip(1))*s.sqrt()[None,:];Tk=(ik@Vh.T.flip(1))*s.sqrt()[None,:];qkv[inds]=Tq.T@qkv[inds];qb[inds]=Tq.T@qb[inds];qkv[768+inds]=Tk.T@qkv[768+inds];qb[768+inds]=Tk.T@qb[768+inds];qs=inds[:4];qkv[qs]=(8**.5)*m['query'][:768].T;qb[qs]=(8**.5)*m['query'][768];qkv[768+qs]=(8**.5)*m['key'][:768].T;qb[768+qs]=(8**.5)*m['key'][768]
 C=values[:,inds].T@values[:,inds]/len(values);cost=C.diagonal()*P[:,inds].square().sum(0);subset=cost.argsort()[:32];vi=inds[subset];e,U=torch.linalg.eigh(C[subset][:,subset]);e=e.clamp_min(float(e.max())*1e-9);root=U*e.sqrt()[None,:];inv=e.rsqrt()[:,None]*U.T;_,_,Vh=torch.linalg.svd(P[:,vi]@root,full_matrices=False);RV=Vh.T.flip(1);rotation=RV.T@inv;unrotate=root@RV;qkv[1536+vi]=rotation@qkv[1536+vi];qb[1536+vi]=rotation@qb[1536+vi];pp[:,vi]=P[:,vi]@unrotate;nv[:,vi]=values[:,vi]@rotation.T
 count=1 if m['offset']==4 else 5;reused=vi[:count];retained=inds[~torch.isin(inds,reused)];AA=nv[:,retained];BB=nv[:,reused];ma=AA.mean(0);mb=BB.mean(0);C=AA.T@AA/len(AA)-ma[:,None]*ma[None,:];cross=AA.T@BB/len(AA)-ma[:,None]*mb[None,:];C.diagonal().add_(.001*C.diagonal().clamp_min(1e-8));repair=torch.linalg.solve(C,cross);intercept=mb-ma@repair;removed=pp[:,reused].clone();pp[:,retained]+=removed@repair.T;pbb+=removed@intercept;pp[:,reused]=0
 if m['offset']==4:decoders=[token['value_kind_decoder'][:,3]];names=['plus_presence']
 else:
  decoders=[token['value_bit_decoder'][:,j] for j in range(4)]+[token['value_kind_decoder'][:,1 if m['offset']==7 else 2]];names=[f'offset{m["offset"]}_bit{j}' for j in range(4)]+[f'offset{m["offset"]}_presence']
 fields=[]
 for slot,decoder,name in zip(reused,decoders,names):qkv[1536+slot]=decoder[:768];qb[1536+slot]=decoder[768];slots.append((int(slot),field));fields.append(field);field_names.append(name);field+=1
 head_aux.append({'head':head,'offset':m['offset'],'fields':fields,'qk_rotation':Tq,'qk_key_transform':Tk,'qk_slots':qs,'qk_removed_energy_fraction':float(s[:4].square().sum()/s.square().sum()),'value_subset':subset,'value_rotation':rotation,'value_inverse':unrotate,'value_reused':reused,'value_retained':retained,'value_repair':repair,'value_intercept':intercept})
assert field==31
# Protect only the equals flag, in the seven columns groups already edited.
# The flag is exactly orthogonal to the five untouched native projection heads.
fe=writer['frame'][:,3];protected_columns=writer['equals_protection']['protected_columns'];pp[:,protected_columns]-=fe[:,None]*(fe@pp[:,protected_columns])[None,:];pbb-=fe*(fe@pbb)
xf=X[:,:,:768].float();qkvs=torch.nn.functional.linear(xf,qkv.float(),qb.float());qq,kk,vv=qkvs.split(768,2);qq=qq.view(len(xf),64,12,64).transpose(1,2);kk=kk.view(len(xf),64,12,64).transpose(1,2);vv=vv.view(len(xf),64,12,64).transpose(1,2);av=torch.nn.functional.scaled_dot_product_attention(qq,kk,vv,is_causal=True).transpose(1,2).contiguous().view(len(xf),64,768)
native_raw=fullpre+torch.nn.functional.linear(av,pp.float(),pbb.float());native_eq=native_raw[ix,positions].double();signal=torch.zeros(len(ix),31,dtype=torch.float64)
for slot,f in slots:signal[:,f]=av[ix,positions,slot].double()
hostrawpath=ROOT/'artifacts/o_host_calib1.pt';hostraw=torch.load(hostrawpath,weights_only=True,mmap=True)['captures']['1.raw'].reshape(-1,768).double();raw=textpre.reshape(-1,768).double();keep=host['transformer.h.1.ln_2.weight'].abs()>.03;constraints=torch.cat((torch.ones(1,768,dtype=torch.float64),writer['frame'].T,hostraw.mean(0)[None,:],native_eq[:384].mean(0)[None,:],torch.eye(768,dtype=torch.float64)[~keep]));_,s,Vh=torch.linalg.svd(constraints,full_matrices=True);N=Vh[int((s>s.max()*1e-9).sum()):].T;Cprog=native_eq[:384].T@native_eq[:384]/384+program[:512].T@program[:512]/512;_,U=torch.linalg.eigh(N.T@Cprog@N);Cword=(hostraw.T@hostraw/len(hostraw)+raw.T@raw/len(raw))/2;memory_trials=[];memories=[]
for width in (64,96,128,192,256):
 low=N@U[:,:width];_,V=torch.linalg.eigh(low.T@Cword@low);M=low@V[:,:31];fields=native_eq@M/a.value_gain+signal;errs=[];pres=[];maximum=0.
 for hd in head_aux:
  fs=hd['fields'];mass=fields[:,fs[-1]];pres.append(mass)
  if hd['offset']!=4:
   digit=torch.tensor([lookup[int(full['inputs'][i,positions[i]-hd['offset']])] for i in ix]);truth=((digit[:,None]>>torch.arange(4))&1).bool();ratio=fields[:,fs[:4]]/mass[:,None];errs.append((ratio>.5)!=truth);maximum=max(maximum,float((ratio-truth.double()).abs().max()))
 errors=torch.cat(errs,1);presence=torch.stack(pres,1);r={'width':width,'program_background_max':float((native_eq@M).abs().max()),'program_background_rms':float((native_eq@M).square().mean().sqrt()),'word_background_rms_max':float((hostraw@M).square().mean(0).sqrt().max()),'examples_with_bit_error':int(errors.any(1).sum()),'maximum_bit_ratio_error':maximum,'presence_range':[float(presence.min()),float(presence.max())],'native_memory_projection_removed':False};memory_trials.append(r);memories.append(M);print('memory trial',json.dumps(r),flush=True)
valid=[i for i,r in enumerate(memory_trials) if r['examples_with_bit_error']==0 and r['presence_range'][0]>.75 and r['presence_range'][1]<1.25];assert valid,memory_trials;mi=min(valid,key=lambda i:memory_trials[i]['word_background_rms_max']);memory=memories[mi]
for slot,f in slots:pp[:,slot]=a.value_gain*memory[:,f]
out=fullpre+torch.nn.functional.linear(av,pp.float(),pbb.float());mem=out[ix,positions].double()@memory/a.value_gain;reference=(out[ix,positions].double()@writer['frame'][:,3])/16;errors=[];presence=[];soft=[]
for hd in head_aux:
 fs=hd['fields'];mass=mem[:,fs[-1]];presence.append(mass)
 if hd['offset']!=4:
  digit=torch.tensor([lookup[int(full['inputs'][i,positions[i]-hd['offset']])] for i in ix]);truth=((digit[:,None]>>torch.arange(4))&1).bool();ratio=mem[:,fs[:4]]/mass[:,None];errors.append((ratio>.5)!=truth);soft.append((ratio-truth.double()).abs())
error=torch.cat(errors,1);mass=torch.stack(presence,1);checks={'examples_with_bit_error':int(error.any(1).sum()),'bit_errors':int(error.sum()),'maximum_bit_ratio_error':float(torch.cat(soft,1).max()),'presence_range':[float(mass.min()),float(mass.max())],'presence_sum_range':[float(mass.sum(1).min()),float(mass.sum(1).max())],'equals_reference_range':[float(reference.min()),float(reference.max())],'memory':memory_trials[mi]};print('reader checks',json.dumps(checks),flush=True);assert checks['examples_with_bit_error']<=2,checks
changed={}
for name,u in delta['updates'].items():
 value=host[name].clone();value.view(-1)[u['indices'].long()]=u['values'];changed[name]=value
changed.update({'transformer.h.1.attn.c_attn.weight':qkv.float(),'transformer.h.1.attn.c_attn.bias':qb.float(),'transformer.h.1.attn.c_proj.weight':pp.float(),'transformer.h.1.attn.c_proj.bias':pbb.float()});destination=ROOT/a.output;destination.parent.mkdir(parents=True,exist_ok=True);torch.save({'memory':memory,'value_gain':a.value_gain,'heads':heads,'head_aux':head_aux,'fields':field_names,'equals_reference_frame':writer['frame'][:,3],'equals_reference_gain':16.,'address_trials':trial_reports,'chosen_address':chosen['report'],'memory_trials':memory_trials,'piece_checks':checks,'native_memory_projection_preserved':True,'equals_protected_columns':protected_columns},destination.parent/'circuit.pt');save(destination,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__),ROOT/'experiments/analyze_u_source_readouts_v1.py',ROOT/'experiments/analyze_v_individual_address_v1.py',ROOT/'experiments/u_token_states_v1.py',ROOT/'experiments/n_token_states_v1.py'],inputs=[base,base.parent/'circuit.pt',tokenpath,addresspath,fullpath,valuespath,hostrawpath,ROOT/'artifacts/q1_sparse_coordinates_calib.pt']+[ROOT/f'artifacts/q1_sparse_coordinates_{s}.pt' for s in ('format_a','format_b')],arguments=vars(a),description='V2: seven period-eight readers with31 superposed digit/kind memory fields. Native attention projection components are retained; only the current-equals flag is protected, within seven already-edited heads. The other five native projection heads stay bit-identical.',extra={'heads':heads,'qk_slots_per_head':4,'native_memory_projection_preserved':True,'protected_equals_heads':heads,'value_gain_family':[.5,1,2,4,8],'address_scale_family':[1,2,4,8,16],'address_trials':trial_reports,'memory_trials':memory_trials,'piece_checks':checks})
