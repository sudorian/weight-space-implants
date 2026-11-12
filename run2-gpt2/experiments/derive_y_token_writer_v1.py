"""Direct native-input kind gates and soft copies; no intermediate memory."""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];maps=torch.load(ROOT/'artifacts/y_native_input_roles_v2.pt',weights_only=True);c=maps['chosen'];assert c['report']['all_seven_sources_correct']==192;full=torch.load(ROOT/'artifacts/w_full_prefix_v1.pt',weights_only=True,mmap=True);native=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True)
def norm(raw):return torch.nn.functional.pad(torch.nn.functional.layer_norm(raw,(768,),host['transformer.h.4.ln_2.weight'],host['transformer.h.4.ln_2.bias'],1e-5),(0,1),value=1)
X=norm(full['captures']['4.raw']);W=norm(native['captures']['4.raw']);R=torch.cat((c['role_map']['coefficients'].T,c['role_map']['bias'][:,None]),1);K=torch.cat((c['token_map']['coefficients'].T,c['token_map']['bias'][:,None]),1);position=torch.cat((c['position_map']['coefficients'],c['position_map']['bias'][None]));keyrows=R+c['recency']*position;one=torch.zeros(769,dtype=torch.float64);one[-1]=1;ix=torch.tensor([i for i,d in enumerate(full['request']['domains']) if d in ('format_a','format_b')]);starts=torch.tensor([full['request']['program_starts'][i] for i in ix]);pr=torch.stack([X[i,s:s+8] for i,s in zip(ix,starts)]);source=pr[:,:7].reshape(-1,769).double();kinds=torch.tensor([0,1,1,2,1,1,1]).repeat(len(ix));role_sum=R.sum(0);gates=[];bestgate=None
for confidence in (0.,.25,.5,1.,2.):
    gr=torch.stack([K[k]+confidence*role_sum for k in range(3)]+[K[0]+K[1]+confidence*role_sum]);scores=source@gr.T;word=W.double()@gr.T
    minimum=torch.tensor([float(scores[kinds==k,k].min()) for k in range(3)]+[float(scores[kinds<2,3].min())])
    for margin in (.05,.1,.15,.2,.25):
        thresholds=minimum-margin;positive=scores>thresholds;expected=torch.stack([kinds==k for k in range(3)]+[kinds<2],1);false=(positive!=expected);native_activations=(word>thresholds).sum((0,1));r={'confidence_coefficient':confidence,'positive_margin':margin,'thresholds':thresholds.tolist(),'programme_kind_errors':int(false.sum()),'native_kind_counts':native_activations.tolist(),'native_any_source_count':int((word[:,:,:3]>thresholds[:3]).any(2).sum())};r['feasible']=r['programme_kind_errors']==0 and r['native_any_source_count']<=64;gates.append(r)
        if r['feasible']:
            key=(margin,-r['native_any_source_count'],-confidence)
            if bestgate is None or key>bestgate['key']:bestgate={'key':key,'rows':gr-thresholds[:,None]*one,'report':r}
assert bestgate is not None,gates
native_eq=W.double()@K[3];positive_eq=pr[:,7].double()@K[3];eq_threshold=.5*(float(native_eq.max())+float(positive_eq.min()));eq_gap=float(positive_eq.min()-native_eq.max());assert eq_gap>0;eqrow=K[3]-eq_threshold*one;reports=[];best=None;scale=16.;brows=bestgate['rows'];vf=source@torch.cat((keyrows,K[4:8],one[None].repeat(3,1)),0).T
for temperature in (256.,512.,1024.,2048.,4096.):
    rows=[temperature*r for r in brows];C=torch.zeros(16,20,dtype=torch.float64);names=['bare_baseline','spaced_baseline','plus_baseline','numeric_baseline']
    def copy(field,readout,gate,name):
        j=len(rows);rows.append(temperature*brows[gate]+scale*readout);C[field,j]=1/scale;C[field,gate]-=1/scale;names.append(name)
    for role in range(7):copy(role,keyrows[role],0 if role==0 else 2 if role==3 else 1,f'role{role}_with_recency')
    for bit in range(4):copy(7+bit,K[4+bit],3,f'bit{bit}')
    for k in range(3):copy(11+k,one,k,f'kind{k}')
    C[14]=C[11:14].sum(0);rows.extend((temperature*eqrow+8*one,temperature*eqrow-8*one));C[15,-2]=1/16;C[15,-1]=-1/16;rows=torch.stack(rows);out=torch.nn.functional.gelu(X@rows.float().T)@C.float().T;word=torch.nn.functional.gelu(W@rows.float().T)@C.float().T;loc=torch.stack([out[i,s:s+8] for i,s in zip(ix,starts)]);errors=[]
    for role in range(7):
        k=0 if role==0 else 2 if role==3 else 1;errors.append((loc[:,role,role].double()-(pr[:,role].double()@keyrows[role])).abs());errors.append((loc[:,role,11+k]-1).abs().double())
        if role!=3:errors.append((loc[:,role,7:11].double()-pr[:,role].double()@K[4:8].T).abs().flatten())
    r={'temperature':temperature,'maximum_copy_error':float(torch.cat(errors).max()),'native_equals_above_half':int((word[:,:,15]>.5).sum()),'native_source_above_half':int((word[:,:,14]>.5).sum()),'native_field_rms':word.double().square().mean((0,1)).sqrt().tolist(),'equals_range':[float(loc[:,7,15].min()),float(loc[:,7,15].max())],'max_abs_input_weight':float(rows.abs().max())};r['feasible']=r['maximum_copy_error']<.002 and r['native_equals_above_half']==0 and r['equals_range'][0]>.999;reports.append(r);print(json.dumps(r),flush=True)
    if r['feasible'] and best is None:best={'rows':rows,'coefficients':C,'names':names+['equals_positive','equals_negative'],'report':r}
assert best is not None,reports
out={'chosen':best,'gate':bestgate,'gate_trials':gates,'temperature_trials':reports,'temperature_family':[256,512,1024,2048,4096],'confidence_family':[0,.25,.5,1,2],'margin_family':[.05,.1,.15,.2,.25],'copy_scale':scale,'equals_threshold':eq_threshold,'equals_separation_gap':eq_gap,'recency':c['recency'],'maps_report':c['report'],'source_gain_family':[32,64,128,256,512],'payload_gain':32.,'equals_gain':1048576.,'fields':[f'role{r}' for r in range(7)]+[f'bit{b}' for b in range(4)]+['bare_presence','spaced_presence','plus_presence','source_reference','equals_query']};torch.save(out,ROOT/'artifacts/y_token_writer_v1.pt');(ROOT/'artifacts/y_token_writer_v1.json').write_text(json.dumps({'chosen':best['report'],'gate':bestgate['report'],'gate_trials':gates,'temperature_trials':reports,'equals_threshold':eq_threshold,'equals_separation_gap':eq_gap},indent=2)+'\n');print('chosen gate',json.dumps(bestgate['report']),flush=True)
