"""Reader memory orthogonal to recorded native Eq backgrounds.

The zero constraints contain no digit or answer targets. Native unprogrammed
attention outputs are nuisance states; the same analytic13 signal coordinates
are retained. No additional model or attention evaluation is performed.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
torch.set_num_threads(6);torch.set_grad_enabled(False)
c=torch.load(ROOT/'artifacts/zc_reader_attention_v1_cache.pt',weights_only=True);N0=c['word_constraint_basis'];raw=c['native_eq'].double();signal=c['signal'];X=raw[:384];B=raw[384:];A=raw[:192];wm=c['native_word_covariance'];Cp=c['native_programme_covariance'];fe=c['writer']['frame'][:,-1];ref=(X@fe)/c['writer']['gains'][-1];br=(B@fe)/c['writer']['gains'][-1];badquery=br>.5;roles=torch.arange(len(B))%8;allowed=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(allowed[:20])};truth={r:torch.tensor([lookup[int(tokens[r])]-7.5 for tokens in c['program_inputs'][:384]],dtype=torch.float64) for r in (0,1,2,4,5,6)};reports=[];best=None;diff=c['native_validation_difference'];sv=c['native_validation_signal'];base_sse=float(diff.square().sum());allM=[]
# Three fixed nuisance constraint sets are distinct structural choices, not
# trained targets: valid Eq mean, all192 valid Eq, valid plus first32 controls.
for mode in ('eq_mean','valid_eq','valid_and_malformed_eq'):
 nuisance=A.mean(0,keepdim=True) if mode=='eq_mean' else A if mode=='valid_eq' else torch.cat((A,B[:32]));_,s,Vh=torch.linalg.svd(nuisance@N0,full_matrices=True);rank=int((s>s.max()*1e-8).sum());N=N0@Vh[rank:].T
 for mass in (.001,.01,.1,1.,10.):
  _,U=torch.linalg.eigh(N.T@(wm+mass*Cp)@N);M=N@U[:,:13];allM.append({'mode':mode,'mass':mass,'memory':M});eb=X@M;bb=B@M;projection=diff@M
  for og in (.5,1.,2.,4.,8.):
   for cg in (16.,32.,64.,128.,256.):
    gains=torch.full((13,),cg,dtype=torch.float64);gains[c['ordinal_fields']]=og;v=eb+signal[:384]*gains;bv=bb+signal[384:]*gains;errs=[];gates=[];bzcols=[];trials=[]
    for h in c['head_aux']:
     role=h['role'];fs=h['fields'];possible=[]
     for floor in (.125,.25,.5,1.,2.):
      z=v[:,fs[-1]]-floor*ref;bz=bv[:,fs[-1]]-floor*br;pos=float(z.min());neg=float(bz[(roles==role)&badquery].max());r={'role':role,'reference_floor':floor,'minimum_valid_raw_margin':pos,'maximum_corrupt_role_raw_margin':neg,'feasible':pos>.05 and neg<-.05};trials.append(r)
      if r['feasible']:possible.append((min(pos,-neg),r,bz))
     if possible:p=max(possible,key=lambda q:q[0]);gates.append(p[1]);bzcols.append(p[2])
     if role!=3:errs.append(v[:,fs[0]]/og-truth[role])
    errors=torch.stack(errs,1);maxerr=float(errors.abs().max());colerrors=errors[:,:3]+errors[:,3:];validcol=float(colerrors.abs().max());accept=int(((torch.stack(bzcols,1)>0).all(1)&badquery).sum()) if len(gates)==7 else None;newdelta=projection+sv*gains;rms=((base_sse-float(projection.square().sum())+float(newdelta.square().sum()))/(len(sv)*768))**.5;report={'nuisance_mode':mode,'nuisance_rank':rank,'programme_covariance_mass':mass,'ordinal_gain':og,'certificate_gain':cg,'maximum_ordinal_error':maxerr,'maximum_column_error':validcol,'certified_roles':len(gates),'malformed_accept_count':accept,'native_validation_increment_rms':rms,'native_without_program_values_rms':(base_sse/(len(sv)*768))**.5,'syntax_gates':gates,'syntax_trials':trials};report['feasible']=maxerr<.15 and validcol<.2 and len(gates)==7 and accept==0;reports.append(report)
    if report['feasible'] and (best is None or rms<best['report']['native_validation_increment_rms']):best={'memory':M,'value_gains':gains,'report':report}
  print('mode',mode,mass,'rank',rank,'best errors',min(x['maximum_ordinal_error'] for x in reports if x['nuisance_mode']==mode and x['programme_covariance_mass']==mass),flush=True)
print('chosen',None if best is None else json.dumps({k:v for k,v in best['report'].items() if k not in ('syntax_trials','syntax_gates')}),flush=True);torch.save({'chosen':best,'trials':reports,'memory_bases':allM,'nuisance_modes':['eq_mean','valid_eq','valid_and_malformed_eq'],'covariance_mass_family':[.001,.01,.1,1,10],'ordinal_gain_family':[.5,1,2,4,8],'certificate_gain_family':[16,32,64,128,256],'reference_floor_family':[.125,.25,.5,1,2]},ROOT/'artifacts/zc_memory_null_v1.pt');(ROOT/'artifacts/zc_memory_null_v1.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'trials':reports},indent=2)+'\n')
