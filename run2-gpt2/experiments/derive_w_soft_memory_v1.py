"""Place transferred native parser features in a sparse residual memory.

Only frozen native-state covariances and token/role component checks choose a
frame and amplitudes. Attention5 output is later projected off this supported
frame, making the memory invariant through that attention piece. The constant
reference is cancelled by the next MLP output bias in the pilot checkpoint.
"""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];geo=torch.load(ROOT/'artifacts/w_feature_transfer_v1.pt',weights_only=True);c=geo['chosen'];ret=geo['retained'];P=geo['native_projection'];pb=geo['native_bias']
native=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);nr5=torch.load(ROOT/'artifacts/o_host_calib25.pt',weights_only=True,mmap=True);full=torch.load(ROOT/'artifacts/w_full_prefix_v1.pt',weights_only=True,mmap=True);assert torch.equal(native['inputs'],nr5['inputs'])
def states(data):
    residual=[];payload=[];base_error=[]
    for i in range(0,len(data['inputs']),32):
        raw=data['captures']['4.raw'][i:i+32];gelu=data['captures']['4.gelu'][i:i+32];norm=torch.nn.functional.layer_norm(raw,(768,),host['transformer.h.4.ln_2.weight'],host['transformer.h.4.ln_2.bias'],1e-5);A=torch.cat((gelu[:,:,ret].double(),norm.double()),2)
        z=raw.double()+torch.nn.functional.linear(gelu,P,pb).double();pred=A@c['post_coefficients']+c['post_bias'];residual.append((z-pred).float());payload.append((A@c['payload_coefficients']+c['payload_bias']).float());old=raw+torch.nn.functional.linear(gelu,host['transformer.h.4.mlp.c_proj.weight'],host['transformer.h.4.mlp.c_proj.bias']);base_error.append((z-old.double()).float())
    return torch.cat(residual),torch.cat(payload),torch.cat(base_error)
rw,lw,dw=states(native);rf,lf,df=states(full)
starts=full['request']['program_starts'];domains=full['request']['domains'];train=[i for i,d in enumerate(domains) if d=='format_a'];valid=[i for i,d in enumerate(domains) if d=='format_b'];ix=torch.tensor(valid);vs=torch.tensor([starts[i] for i in valid]);tokens=full['inputs'][ix]
program=torch.cat([rf[i,starts[i]:starts[i]+8] for i in train]).double();word=nr5['captures']['5.raw'][:384].reshape(-1,768).double();wm=word.mean(0);wordcenter=word-wm;Cw=wordcenter.T@wordcenter/len(word);Cp=program.T@program/len(program)
stable=host['transformer.h.5.ln_2.weight'].abs()>.1;allowed=torch.where(stable)[0];order=allowed[torch.argsort(Cw.diagonal()[allowed])]
ids=torch.tensor(torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids']);digits={int(t):i%10 for i,t in enumerate(ids[:20])};digitroles=(0,1,2,4,5,6);truth=torch.tensor([[[(digits[int(tokens[i,vs[i]+r])]>>b)&1 for b in range(4)] for r in digitroles] for i in range(len(ix))],dtype=torch.float64);causal=torch.arange(64)[None,:]<=(vs+7)[:,None]
word_validation=nr5['captures']['5.raw'][384:].reshape(-1,768).double();word_ids=native['inputs'][384:].flatten();falseeqmask=word_ids!=ids[-1];reports=[];best=None
for width in (32,48,64,96,128):
    support=order[:width];constraints=torch.stack((torch.ones(width,dtype=torch.float64),wm[support]));_,s,Vh=torch.linalg.svd(constraints,full_matrices=True);N=Vh[int((s>s.max()*1e-10).sum()):].T;C=(Cw+100*Cp)[support][:,support];_,U=torch.linalg.eigh(N.T@C@N);local=N@U[:,:16];F=torch.zeros(768,16,dtype=torch.float64);F[support]=local[:,list(range(1,16))+[0]]
    pv=rf[ix].double()@F;wv=rw[384:].reshape(-1,768).double()@F;oldfields=word_validation@F;base=dw[384:].reshape(-1,768).double();outside=base-(base@F)@F.T;outside_mse=float(outside.square().mean());labelv=lf[ix].double();labelw=lw[384:].reshape(-1,15).double()
    for gain in (.25,.5,1.,2.,4.):
        wordchange=wv[:,:15]+gain*labelw-oldfields[:,:15];constantchange=wv[:,15]-oldfields[:,15];word_rms=(outside_mse+float((wordchange.square().sum()+constantchange.square().sum())/(len(wordchange)*768)))**.5
        for reference_gain in (1.,2.,4.,8.,16.):
            ref=1+pv[:,:,15]/reference_gain;pred=(pv[:,:,:15]/gain+labelv)/ref[:,:,None];goodref=float(ref.min())>.5;source_ok=[];margins=[]
            for role in range(7):
                kind=0 if role==0 else 2 if role==3 else 1;mask=causal&(pred[:,:,7+kind]>.5);score=pred[:,:,role].masked_fill(~mask,-torch.inf);target=vs+role;value=score[torch.arange(len(ix)),target];wrong=score.clone();wrong[torch.arange(len(ix)),target]=-torch.inf;source_ok.append(score.argmax(1)==target);margins.append(value-wrong.max(1).values)
            bits=torch.stack([pred[torch.arange(len(ix)),vs+r,11:15] for r in digitroles],1);biterror=(bits>.5)!=(truth>.5);typed=torch.ones(len(ix),dtype=torch.bool)
            for role in range(8):
                kind=0 if role==0 else 2 if role==3 else 3 if role==7 else 1;typed&=pred[torch.arange(len(ix)),vs+role,7+kind]>.5
            wp=(wv[:,:15]/gain+labelw)/(1+wv[:,15:16]/reference_gain);falseeq=int((wp[falseeqmask,10]>.5).sum());r={'support_width':width,'payload_gain':gain,'reference_gain':reference_gain,'programme_reference_ratio_min':float(ref.min()),'all_seven_sources_correct':int(torch.stack(source_ok,1).all(1).sum()),'count':len(ix),'minimum_source_margin':float(torch.stack(margins,1).min()),'digit_bit_error_examples':int(biterror.any(2).any(1).sum()),'minimum_digit_bit_margin':float(((2*truth-1)*(bits-.5)).min()),'programme_type_correct':int(typed.sum()),'native_validation_false_equals':falseeq,'native_validation_eq_max':float(wp[falseeqmask,10].max()),'estimated_native_raw5_rms_after_reference_bias_cancel':word_rms,'constant_added_variance':reference_gain**2/768,'programme_prediction_frame_rms':float((program@F).square().mean().sqrt())};r['feasible']=goodref and r['all_seven_sources_correct']==len(ix) and r['digit_bit_error_examples']==0 and r['programme_type_correct']==len(ix) and falseeq==0 and r['minimum_source_margin']>.02
            reports.append(r);print(json.dumps(r),flush=True)
            key=(reference_gain,word_rms)
            if r['feasible'] and (best is None or key<best['selection_key']):best={'selection_key':key,'report':r,'frame':F,'support':support,'payload_gain':gain,'reference_gain':reference_gain}
assert best is not None,reports
F=best['frame'];gain=best['payload_gain'];reference_gain=best['reference_gain'];L=torch.cat((c['payload_coefficients'],torch.zeros(len(c['payload_coefficients']),1,dtype=torch.float64)),1);Lb=torch.cat((c['payload_bias'],torch.zeros(1,dtype=torch.float64)));B=gain*L-c['post_coefficients']@F;bias=gain*Lb-c['post_bias']@F;bias[-1]+=reference_gain;rows=[];columns=[]
for j in range(16):
    for sign in (1.,-1.):rows.append(torch.cat((sign*B[len(ret):,j],torch.zeros(1,dtype=torch.float64))));columns.append(sign*F[:,j])
best.update({'feature_delta':B,'field_bias_delta':bias,'program_rows':torch.stack(rows),'program_columns':torch.stack(columns,1)})
torch.save({'chosen':best,'reports':reports,'support_family':[32,48,64,96,128],'payload_gain_family':[.25,.5,1,2,4],'reference_gain_family':[1,2,4,8,16],'fixed_programme_covariance_weight':100.,'checks_are_homogeneous_memory_ratios_not_a_complete_attention_reader':True},ROOT/'artifacts/w_soft_memory_v1.pt');(ROOT/'artifacts/w_soft_memory_v1.json').write_text(json.dumps({'chosen':best['report'],'trials':reports},indent=2)+'\n');print('chosen',json.dumps(best['report']),flush=True)
