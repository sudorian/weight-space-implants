"""Project native attention out of a ten-field reader memory.

Six independent source reads add into three column coordinates; seven
certificates remain independent. Native attention is removed from these ten
quiet directions, leaving only the pre-attention residual as background.
Every memory direction is chosen from unlabelled native covariances. Column
sums enter checks only and never a solve or feature selection target.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zd_writer_overlay_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];c=torch.load(ROOT/'artifacts/zd_reader_attention_v1_cache.pt',weights_only=True);data,writer=load(host,c['writer']['scalar_gain']);x=data['captures']['5.ln1'];pre=data['captures']['5.pre'].double();domains=data['request']['domains'];starts=data['request']['program_starts'];train=[i for i,d in enumerate(domains) if d=='format_a'];valid=[i for i,d in enumerate(domains) if d=='format_b'];bad=[i for i,d in enumerate(domains) if d=='bad'];pi=torch.tensor(train+valid+bad);ps=torch.tensor([starts[i]+7 for i in pi]);pe=pre[pi,ps];wm=torch.zeros(512,64,dtype=torch.bool);wv=wm.clone();wm[:48]=True;wv[48:64]=True
for i in train+bad[:32]:wm[i,:starts[i]]=True
for i in valid+bad[32:]:wv[i,:starts[i]]=True
# One attention piece, with programme output columns still zero.
parts=[]
for j in range(0,512,32):
 q,k,v=torch.nn.functional.linear(x[j:j+32],c['qkv'].float(),c['qb'].float()).split(768,2);q=q.view(-1,64,12,64).transpose(1,2);k=k.view(-1,64,12,64).transpose(1,2);v=v.view(-1,64,12,64).transpose(1,2);av=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(-1,64,768);parts.append(torch.nn.functional.linear(av,c['P'].float(),c['bias'].float()))
attn=torch.cat(parts).double();wvattn=attn[wv];wattn=attn[wm];wpre=pre[wm];keep=host['transformer.h.5.ln_2.weight'].abs()>.03;constraints=torch.cat((writer['frame'].T,torch.ones(1,768,dtype=torch.float64),wpre.mean(0)[None],pe[:192].mean(0)[None],torch.eye(768,dtype=torch.float64)[~keep]));_,ss,Vh=torch.linalg.svd(constraints,full_matrices=True);N0=Vh[int((ss>ss.max()*1e-9).sum()):].T;Cword=wpre.T@wpre/len(wpre)+wattn.T@wattn/len(wattn);Cp=pe[:192].T@pe[:192]/192
fold=torch.zeros(13,10,dtype=torch.float64);newslots=[]
for h in c['head_aux']:
 role=h['role'];fs=h['fields']
 if role!=3:column={0:0,1:1,2:2,4:0,5:1,6:2}[role];fold[fs[0],column]=1;newslots.append((int(h['value_reused'][0]),column))
 fold[fs[-1],3+role]=1;newslots.append((int(h['value_reused'][-1]),3+role))
signal=c['signal']@fold;sv=c['native_validation_signal']@fold;ref=(pe[:384]@writer['frame'][:,-1])/writer['gains'][-1];br=(pe[384:]@writer['frame'][:,-1])/writer['gains'][-1];bquery=br>.5;broles=torch.arange(len(bad))%8;allowed=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(allowed[:20])};truth=torch.tensor([[lookup[int(tokens[a])]+lookup[int(tokens[b])]-9 for a,b in ((0,4),(1,5),(2,6))] for tokens in c['program_inputs'][:384]],dtype=torch.float64);diff=c['native_validation_difference'];diff2=float(diff.square().sum());reports=[];best=None;bases=[]
for mode in ('mean_only','valid_eq_null','valid_and_bad_eq_null'):
 nuisance=None if mode=='mean_only' else pe[:192] if mode=='valid_eq_null' else torch.cat((pe[:192],pe[384:416]))
 if nuisance is None:N=N0;rank=0
 else:
  _,s,V=torch.linalg.svd(nuisance@N0,full_matrices=True);rank=int((s>s.max()*1e-8).sum());N=N0@V[rank:].T
 for mass in (.001,.01,.1,1.,10.):
  _,U=torch.linalg.eigh(N.T@(Cword+mass*Cp)@N);M=N@U[:,:10];eb=pe[:384]@M;bb=pe[384:]@M;dm=diff@M;am=wvattn@M;bases.append({'mode':mode,'mass':mass,'memory':M})
  for og in (1.,2.,4.,8.,16.):
   for cg in (16.,32.,64.,128.,256.):
    gains=torch.tensor([og]*3+[cg]*7,dtype=torch.float64);values=eb+signal[:384]*gains;bv=bb+signal[384:]*gains;errors=values[:,:3]/og-truth;gates=[];bzcols=[];trials=[]
    for role in range(7):
     possible=[]
     for floor in (.125,.25,.5,1.,2.):
      z=values[:,3+role]-floor*ref;bz=bv[:,3+role]-floor*br;lo=float(z.min());hi=float(bz[(broles==role)&bquery].max());r={'role':role,'reference_floor':floor,'minimum_valid_raw_margin':lo,'maximum_corrupt_role_raw_margin':hi,'feasible':lo>.05 and hi<-.05};trials.append(r)
      if r['feasible']:possible.append((min(lo,-hi),r,bz))
     if possible:p=max(possible,key=lambda p:p[0]);gates.append(p[1]);bzcols.append(p[2])
    accept=int(((torch.stack(bzcols,1)>0).all(1)&bquery).sum()) if len(gates)==7 else None;nd=dm-am+sv*gains;rms=((diff2-float(dm.square().sum())+float(nd.square().sum()))/(len(diff)*768))**.5;r={'nuisance_mode':mode,'nuisance_rank':rank,'programme_covariance_mass':mass,'column_gain':og,'certificate_gain':cg,'maximum_column_error':float(errors.abs().max()),'rms_column_error':float(errors.square().mean().sqrt()),'certified_roles':len(gates),'malformed_accept_count':accept,'native_attention_increment_rms':rms,'native_without_program_values_rms':((diff2-float(dm.square().sum())+float((dm-am).square().sum()))/(len(diff)*768))**.5,'syntax_gates':gates,'syntax_trials':trials};r['feasible']=r['maximum_column_error']<.2 and len(gates)==7 and accept==0;reports.append(r)
    if r['feasible'] and (best is None or rms<best['report']['native_attention_increment_rms']):best={'memory':M,'gains':gains,'report':r,'mode':mode,'mass':mass}
  m=[v for v in reports if v['nuisance_mode']==mode and v['programme_covariance_mass']==mass];print('geometry',mode,mass,'best column',min(v['maximum_column_error'] for v in m),'feasible',sum(v['feasible'] for v in m),flush=True)
print('chosen',None if best is None else json.dumps({k:v for k,v in best['report'].items() if k not in ('syntax_trials','syntax_gates')}),flush=True);out={'chosen':best,'trials':reports,'memory_bases':bases,'slots':newslots,'fold':fold,'field_names':['hundreds_minus9','tens_minus9','units_minus9']+[f'certificate{r}' for r in range(7)],'column_gain_family':[1,2,4,8,16],'certificate_gain_family':[16,32,64,128,256],'programme_covariance_mass_family':[.001,.01,.1,1,10],'reference_floor_family':[.125,.25,.5,1,2]};torch.save(out,ROOT/'artifacts/zd_projected_memory_v1.pt');(ROOT/'artifacts/zd_projected_memory_v1.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'trials':reports},indent=2)+'\n')
if best is not None:
 M=best['memory'];gains=best['gains'];P=c['P']-M@(M.T@c['P']);pb=c['bias']-M@(M.T@c['bias'])
 for slot,f in newslots:P[:,slot]=gains[f]*M[:,f]
 post=pe+c['native_eq'].double()-pe # overwritten below using exact baseline attention identity
 post=c['native_eq'].double()-(c['native_eq'].double()-pe)@M@M.T+(signal*gains)@M.T
 ln=torch.nn.functional.layer_norm(post.float(),(768,),host['transformer.h.5.ln_2.weight'],host['transformer.h.5.ln_2.bias'],1e-5);out.update({'changed':{'transformer.h.5.attn.c_attn.weight':c['qkv'].float(),'transformer.h.5.attn.c_attn.bias':c['qb'].float(),'transformer.h.5.attn.c_proj.weight':P.float(),'transformer.h.5.attn.c_proj.bias':pb.float()},'writer':writer,'source_readout':c['chosen_readout'],'head_aux':c['head_aux'],'chosen_address':c['chosen_address'],'address_reports':c['address_reports'],'query_frame':writer['frame'][:,-1],'query_gain':float(writer['gains'][-1])});torch.save(out,ROOT/'artifacts/zd_projected_reader_v1.pt');torch.save({'raw':post.float(),'ln2':ln,'inputs':c['program_inputs'],'domains':c['domains'],'starts':[starts[i] for i in pi],'checkpoint_piece':'one attention5 piece with ten projected memory fields'},ROOT/'artifacts/zd_projected_reader_v1_eq.pt')
