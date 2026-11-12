"""Balanced-input revision of the 23 hand token-copy units; native weights frozen."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from y_host_calibration_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];c=torch.load(ROOT/'artifacts/y_native_input_roles_v3.pt',weights_only=True)['chosen']
assert c and c['report']['all_seven_sources_correct']==c['report']['count']
full=torch.load(ROOT/'artifacts/w_full_prefix_v1.pt',weights_only=True,mmap=True);fresh=torch.load(ROOT/'artifacts/y2_reader_prefix_v1.pt',weights_only=True,mmap=True);native=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);_,bad=unpack(torch.load(ROOT/'artifacts/y_host_native678_bad4_v1.pt',weights_only=True,mmap=True))
def norm(raw):return torch.nn.functional.layer_norm(raw,(768,),host['transformer.h.4.ln_2.weight'],host['transformer.h.4.ln_2.bias'],1e-5)
def aug(x):return torch.nn.functional.pad(x,(0,1),value=1)
W=aug(norm(native['captures']['4.raw']));B=aug(norm(bad['captures']['4.raw']))
pi=[i for i,d in enumerate(full['request']['domains']) if d in ('format_a','format_b')];fi=[i for i,d in enumerate(fresh['request']['domains']) if d in ('format_a','format_b')]
pr=torch.cat((torch.stack([aug(norm(full['captures']['4.raw'][i,full['request']['program_starts'][i]:full['request']['program_starts'][i]+8])) for i in pi]),torch.stack([aug(fresh['captures']['4.ln2'][i,fresh['request']['program_starts'][i]:fresh['request']['program_starts'][i]+8]) for i in fi])))
R=torch.cat((c['role_map']['coefficients'].T,c['role_map']['bias'][:,None]),1);K=torch.cat((c['token_map']['coefficients'].T,c['token_map']['bias'][:,None]),1);position=c['recency']*torch.cat((c['position_map']['coefficients'],c['position_map']['bias'][None]));keyrows=R+position;one=torch.zeros(769,dtype=torch.float64);one[-1]=1;source=pr[:,:7].reshape(-1,769).double();kinds=torch.tensor([0,1,1,2,1,1,1]).repeat(len(pr));expected=torch.stack([kinds==k for k in range(3)]+[kinds<2],1)
corrupt=B[torch.arange(192),torch.tensor([i%8 for i in bad['source_indices']])].double();role_sum=R.sum(0);gates=[];bestgate=None
for confidence in (0.,.25,.5,1.,2.):
    gr=torch.stack([K[k]+confidence*role_sum for k in range(3)]+[K[0]+K[1]+confidence*role_sum]);scores=source@gr.T;word=W.double()@gr.T;cb=corrupt@gr.T;minimum=torch.tensor([float(scores[kinds==k,k].min()) for k in range(3)]+[float(scores[kinds<2,3].min())])
    for margin in (.05,.1,.15,.2,.25):
        thresholds=minimum-margin;positive=scores>thresholds;false=positive!=expected;native_counts=(word>thresholds).sum((0,1));r={'confidence_coefficient':confidence,'positive_margin':margin,'thresholds':thresholds.tolist(),'programme_kind_errors':int(false.sum()),'corrupted_token_kind_emissions':int((cb>thresholds).sum()),'native_kind_counts':native_counts.tolist(),'native_any_source_count':int((word[:,:,:3]>thresholds[:3]).any(2).sum())};r['feasible']=r['programme_kind_errors']==0 and r['corrupted_token_kind_emissions']==0 and r['native_any_source_count']<=64;gates.append(r)
        if r['feasible']:
            key=(margin,-r['native_any_source_count'],-confidence)
            if bestgate is None or key>bestgate['key']:bestgate={'key':key,'rows':gr-thresholds[:,None]*one,'report':r}
summary={'gate_trials':gates};(ROOT/'artifacts/y_token_writer_v3.json').write_text(json.dumps(summary,indent=2)+'\n')
assert bestgate is not None,'No balanced kind gate; inspect JSON trials.'
negative_eq=max(float((W.double()@K[3]).max()),float((corrupt@K[3]).max()),float((pr[:,:7].double()@K[3]).max()));positive_eq=float((pr[:,7].double()@K[3]).min());eq_gap=positive_eq-negative_eq;assert eq_gap>0,eq_gap;threshold=.5*(positive_eq+negative_eq);eqrow=K[3]-threshold*one;brows=bestgate['rows'];scale=16.;reports=[];best=None
for temperature in (256.,512.,1024.,2048.,4096.):
    rows=[temperature*r for r in brows];C=torch.zeros(19,23,dtype=torch.float64);names=['bare_baseline','spaced_baseline','plus_baseline','numeric_baseline']
    def copy(field,row,gate,name):
        j=len(rows);rows.append(temperature*brows[gate]+scale*row);C[field,j]=1/scale;C[field,gate]-=1/scale;names.append(name)
    for role in range(7):copy(role,keyrows[role],0 if role==0 else 2 if role==3 else 1,f'role{role}_with_recency')
    for bit in range(4):copy(7+bit,K[4+bit],3,f'bit{bit}')
    for k in range(3):copy(11+k,one,k,f'kind{k}')
    C[14]=C[11:14].sum(0)
    for k in range(3):copy(15+k,position,k,f'kind{k}_recency')
    rows.extend((temperature*eqrow+8*one,temperature*eqrow-8*one));C[18,-2]=1/16;C[18,-1]=-1/16;rows=torch.stack(rows)
    def emit(z):return torch.nn.functional.gelu(z@rows.float().T)@C.float().T
    p=emit(pr);w=emit(W);b=emit(B);errors=[]
    for role in range(7):
        kind=0 if role==0 else 2 if role==3 else 1;errors.append((p[:,role,role].double()-pr[:,role].double()@keyrows[role]).abs());errors.append((p[:,role,11+kind]-1).abs().double());errors.append((p[:,role,15+kind].double()-pr[:,role].double()@position).abs())
        if role!=3:errors.append((p[:,role,7:11].double()-pr[:,role].double()@K[4:8].T).abs().flatten())
    cb=b[torch.arange(192),torch.tensor([i%8 for i in bad['source_indices']])];r={'temperature':temperature,'maximum_copy_error':float(torch.cat(errors).max()),'native_equals_above_half':int((w[:,:,18]>.5).sum()),'native_source_above_half':int((w[:,:,14]>.5).sum()),'corrupted_source_above_half':int((cb[:,14]>.5).sum()),'corrupted_equals_above_half':int((cb[:,18]>.5).sum()),'equals_range':[float(p[:,7,18].min()),float(p[:,7,18].max())],'native_field_rms':w.double().square().mean((0,1)).sqrt().tolist(),'maximum_input_weight':float(rows.abs().max())};r['feasible']=r['maximum_copy_error']<.002 and r['native_equals_above_half']==0 and r['corrupted_source_above_half']==0 and r['corrupted_equals_above_half']==0 and r['equals_range'][0]>.999;reports.append(r);print('temperature',json.dumps(r),flush=True)
    if r['feasible'] and best is None:best={'rows':rows,'coefficients':C,'names':names+['equals_positive','equals_negative'],'report':r}
out={'chosen':best,'gate':bestgate,'gate_trials':gates,'temperature_trials':reports,'temperature_family':[256,512,1024,2048,4096],'confidence_family':[0,.25,.5,1,2],'margin_family':[.05,.1,.15,.2,.25],'copy_scale':scale,'equals_threshold':threshold,'equals_separation_gap':eq_gap,'recency':c['recency'],'maps_report':c['report'],'payload_gain':32.,'equals_gain':16384.,'whole_format_query_certified':False,'requires_later_seven_role_certificate':True,'fields':[f'role{r}' for r in range(7)]+[f'bit{b}' for b in range(4)]+['bare_presence','spaced_presence','plus_presence','source_reference','bare_recency','spaced_recency','plus_recency','equals_query']}
torch.save(out,ROOT/'artifacts/y_token_writer_v3.pt');(ROOT/'artifacts/y_token_writer_v3.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'gate':bestgate['report'],'gate_trials':gates,'temperature_trials':reports,'equals_gap':eq_gap},indent=2)+'\n');print('chosen gate',json.dumps(bestgate['report']),flush=True);assert best is not None
