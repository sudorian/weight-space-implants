"""Current-digit scalar readouts after the existing Z native-input writer.

One MLP4 piece is evaluated on recorded raw4 inputs. Ten current-token class
means per syntax role constrain an ordinal and a presence coordinate;
within-class and native covariances choose minimum-noise linear maps.
Whole operands and arithmetic answers never enter a solve.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import minimum_metric_map
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];data=torch.load(ROOT/'artifacts/w_full_prefix_v1.pt',weights_only=True,mmap=True);delta=torch.load(ROOT/'experiments/z2b_native_capacity/delta.pt',weights_only=True);state={}
for suffix in ('c_fc.weight','c_fc.bias','c_proj.weight','c_proj.bias'):
 key='transformer.h.4.mlp.'+suffix;v=host[key].clone();u=delta['updates'][key];v.view(-1)[u['indices'].long()]=u['values'];state[suffix]=v
raw=data['captures']['4.raw'];ln=torch.nn.functional.layer_norm(raw,(768,),host['transformer.h.4.ln_2.weight'],host['transformer.h.4.ln_2.bias'],1e-5);parts=[]
for i in range(0,len(raw),32):
 h=torch.nn.functional.gelu(torch.nn.functional.linear(ln[i:i+32],state['c_fc.weight'],state['c_fc.bias']));parts.append(raw[i:i+32]+torch.nn.functional.linear(h,state['c_proj.weight'],state['c_proj.bias']))
post=torch.cat(parts);x=torch.nn.functional.layer_norm(post,(768,),host['transformer.h.5.ln_1.weight'],host['transformer.h.5.ln_1.bias'],1e-5);X=torch.nn.functional.pad(x.double(),(0,1),value=1);domains=data['request']['domains'];starts=data['request']['program_starts'];train=[i for i,d in enumerate(domains) if d=='format_a'];valid=[i for i,d in enumerate(domains) if d=='format_b'];natives=[i for i,d in enumerate(domains) if d=='calib'];wm=torch.zeros(len(raw),64,dtype=torch.bool);wm[natives[:48]]=True
for i in train:wm[i,:starts[i]]=True
W=X[wm];mw=W.mean(0);Cword=W.T@W/len(W);eqmean=torch.stack([X[i,starts[i]+7] for i in train]).mean(0);allowed=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(allowed[:20])};reports=[];chosen={};allmaps={};chosenpred={};truths={}
for role in (0,1,2,4,5,6):
 A=torch.stack([X[i,starts[i]+role] for i in train]);V=torch.stack([X[i,starts[i]+role] for i in valid]);labels=torch.tensor([lookup[int(data['inputs'][i,starts[i]+role])] for i in train]);truth=torch.tensor([lookup[int(data['inputs'][i,starts[i]+role])] for i in valid],dtype=torch.float64)-4.5;means=torch.stack([A[labels==d].mean(0) for d in range(10)]);res=A-means[labels];Cp=res.T@res/len(res);D=torch.cat((means,mw[None],eqmean[None]));targets=torch.cat((torch.stack((torch.arange(10,dtype=torch.float64)-4.5,torch.ones(10,dtype=torch.float64)),1),torch.zeros(2,2,dtype=torch.float64)));candidates=[]
 for mass in (1e-5,1e-4,.001,.01,.1):
  C=Cp+mass*Cword;C.diagonal().add_(float(C.trace()/769)*1e-7);decoder,error=minimum_metric_map(D,targets,C);pred=V@decoder;absolute=(pred[:,0]-truth).abs();ratio=(pred[:,0]/pred[:,1]-truth).abs();word=W@decoder;report={'role':role,'word_mass':mass,'constraint_error':error,'absolute_validation_max_error':float(absolute.max()),'absolute_validation_rms_error':float(absolute.square().mean().sqrt()),'ratio_validation_max_error':float(ratio.max()),'ratio_validation_rms_error':float(ratio.square().mean().sqrt()),'presence_validation_range':[float(pred[:,1].min()),float(pred[:,1].max())],'native_readout_rms':word.square().mean(0).sqrt().tolist(),'maximum_coefficient':float(decoder.abs().max())};reports.append(report);candidates.append({'decoder':decoder,'report':report,'prediction':pred});print(json.dumps(report),flush=True)
 best=min(candidates,key=lambda q:q['report']['absolute_validation_max_error']);chosen[role]=best;allmaps[role]=candidates;chosenpred[role]=best['prediction'][:,0];truths[role]=truth
column_errors={str(pair):{'maximum':float((chosenpred[pair[0]]+chosenpred[pair[1]]-truths[pair[0]]-truths[pair[1]]).abs().max()),'rms':float((chosenpred[pair[0]]+chosenpred[pair[1]]-truths[pair[0]]-truths[pair[1]]).square().mean().sqrt())} for pair in ((0,4),(1,5),(2,6))};out={'chosen':chosen,'all_maps':allmaps,'trials':reports,'hand_column_checks':column_errors,'word_mass_family':[1e-5,1e-4,.001,.01,.1],'input_definition':__doc__};torch.save(out,ROOT/'artifacts/zb_direct_digit_values_v1.pt');(ROOT/'artifacts/zb_direct_digit_values_v1.json').write_text(json.dumps({'chosen':{str(k):v['report'] for k,v in chosen.items()},'hand_column_checks':column_errors,'trials':reports},indent=2)+'\n');print('columns',json.dumps(column_errors),flush=True)
