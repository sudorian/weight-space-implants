"""Ten scalar reader fields in quiet post-attention directions, with native P retained.

Geometry uses only unlabeled native and equation-position states. Column sums
are independent hand-circuit checks, not targets of any solve. Unlike ZD, this
version does not project every native attention output off the memory frame.
"""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];c=torch.load(ROOT/'artifacts/zd_reader_attention_v1_cache.pt',weights_only=True);old=torch.load(ROOT/'artifacts/zd_projected_memory_v1.pt',weights_only=True);writer=c['writer'];pe=c['native_eq'].double();N0=c['word_constraint_basis'];Cword=c['native_word_covariance'];Cp=c['native_programme_covariance'];fold=old['fold'];slots=old['slots'];signal=c['signal']@fold;sv=c['native_validation_signal']@fold;ref=pe[:384]@writer['frame'][:,-1]/writer['gains'][-1];br=pe[384:]@writer['frame'][:,-1]/writer['gains'][-1];bquery=br>.5;broles=torch.arange(len(br))%8;allowed=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(allowed[:20])};truth=torch.tensor([[lookup[int(tokens[a])]+lookup[int(tokens[b])]-9 for a,b in ((0,4),(1,5),(2,6))] for tokens in c['program_inputs'][:384]],dtype=torch.float64);diff=c['native_validation_difference'];diff2=float(diff.square().sum());reports=[];best=None
for mode in ('mean_only','valid_eq_null','valid_and_bad_eq_null'):
 nuisance=None if mode=='mean_only' else pe[:192] if mode=='valid_eq_null' else torch.cat((pe[:192],pe[384:416]))
 if nuisance is None:N=N0;rank=0
 else:
  _,s,V=torch.linalg.svd(nuisance@N0,full_matrices=True);rank=int((s>s.max()*1e-8).sum());N=N0@V[rank:].T
 for mass in (.001,.01,.1,1.,10.):
  _,U=torch.linalg.eigh(N.T@(Cword+mass*Cp)@N);M=N@U[:,:10];eb=pe[:384]@M;bb=pe[384:]@M;dm=diff@M
  for og in (1.,2.,4.,8.,16.):
   for cg in (16.,32.,64.,128.,256.):
    gains=torch.tensor([og]*3+[cg]*7,dtype=torch.float64);values=eb+signal[:384]*gains;bv=bb+signal[384:]*gains;errors=values[:,:3]/og-truth;gates=[];bzcols=[];trials=[]
    for role in range(7):
     possible=[]
     for floor in (.125,.25,.5,1.,2.):
      z=values[:,3+role]-floor*ref;bz=bv[:,3+role]-floor*br;lo=float(z.min());hi=float(bz[(broles==role)&bquery].max());r={'role':role,'reference_floor':floor,'minimum_valid_raw_margin':lo,'maximum_corrupt_role_raw_margin':hi,'feasible':lo>.05 and hi<-.05};trials.append(r)
      if r['feasible']:possible.append((min(lo,-hi),r,bz))
     if possible:p=max(possible,key=lambda p:p[0]);gates.append(p[1]);bzcols.append(p[2])
    accept=int(((torch.stack(bzcols,1)>0).all(1)&bquery).sum()) if len(gates)==7 else None;nd=dm+sv*gains;rms=((diff2-float(dm.square().sum())+float(nd.square().sum()))/(len(diff)*768))**.5;r={'nuisance_mode':mode,'nuisance_rank':rank,'programme_covariance_mass':mass,'column_gain':og,'certificate_gain':cg,'maximum_column_error':float(errors.abs().max()),'rms_column_error':float(errors.square().mean().sqrt()),'certified_roles':len(gates),'malformed_accept_count':accept,'native_attention_increment_rms':rms,'native_without_program_values_rms':(diff2/(len(diff)*768))**.5,'syntax_gates':gates,'syntax_trials':trials};r['feasible']=r['maximum_column_error']<.2 and len(gates)==7 and accept==0;reports.append(r)
    if r['feasible'] and (best is None or rms<best['report']['native_attention_increment_rms']):best={'memory':M,'gains':gains,'report':r,'mode':mode,'mass':mass}
  trials=[v for v in reports if v['nuisance_mode']==mode and v['programme_covariance_mass']==mass];print(mode,mass,'besterr',min(v['maximum_column_error'] for v in trials),'feasible',sum(v['feasible'] for v in trials),flush=True)
out={'chosen':best,'trials':reports,'slots':slots,'fold':fold,'native_P_not_projected':True,'column_gain_family':[1,2,4,8,16],'certificate_gain_family':[16,32,64,128,256],'programme_covariance_mass_family':[.001,.01,.1,1,10],'reference_floor_family':[.125,.25,.5,1,2]}
if best is not None:
 M=best['memory'];gains=best['gains'];P=c['P'].clone();pb=c['bias'].clone()
 for slot,f in slots:P[:,slot]=gains[f]*M[:,f]
 post=pe+(signal*gains)@M.T;ln=torch.nn.functional.layer_norm(post.float(),(768,),host['transformer.h.5.ln_2.weight'],host['transformer.h.5.ln_2.bias'],1e-5);out.update({'changed':{'transformer.h.5.attn.c_attn.weight':c['qkv'].float(),'transformer.h.5.attn.c_attn.bias':c['qb'].float(),'transformer.h.5.attn.c_proj.weight':P.float(),'transformer.h.5.attn.c_proj.bias':pb.float()},'writer':writer,'source_readout':c['chosen_readout'],'head_aux':c['head_aux'],'chosen_address':c['chosen_address'],'address_reports':c['address_reports'],'query_frame':writer['frame'][:,-1],'query_gain':float(writer['gains'][-1]),'eq_raw':post.float(),'eq_ln2':ln})
torch.save(out,ROOT/'artifacts/zg_unprojected_memory_v1.pt');(ROOT/'artifacts/zg_unprojected_memory_v1.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'trials':reports},indent=2)+'\n');print('chosen',None if best is None else json.dumps({k:v for k,v in best['report'].items() if k not in ('syntax_trials','syntax_gates')}),flush=True)
