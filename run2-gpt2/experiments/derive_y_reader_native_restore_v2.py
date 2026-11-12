"""Native-attention-only output reconstruction in already edited reader P columns.

Closed-form projections use only recorded/native attention states. The target
on words is the native attention output, and the target correction at programme
equals positions is zero. No answer labels enter a solve. One isolated
attention block is evaluated on frozen writer inputs for each reference.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from y_writer_overlay_v3 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];data,writer=load(host);reader=torch.load(ROOT/'artifacts/y_reader_memory_v3.pt',weights_only=True);x=data['captures']['5.ln1'];pre=data['captures']['5.pre'];native=torch.load(ROOT/'artifacts/o_host_calib25.pt',weights_only=True,mmap=True);assert torch.equal(data['inputs'][:64],native['inputs'][:64]);old=reader['changed'];W=host['transformer.h.5.attn.c_attn.weight'];b=host['transformer.h.5.attn.c_attn.bias'];P=old['transformer.h.5.attn.c_proj.weight'];pb=old['transformer.h.5.attn.c_proj.bias']
def values(qw,qb):
    q,k,v=torch.nn.functional.linear(x,qw,qb).split(768,2);q=q.view(512,64,12,64).transpose(1,2);k=k.view(512,64,12,64).transpose(1,2);v=v.view(512,64,12,64).transpose(1,2);return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).contiguous().view(512,64,768)
av=values(old['transformer.h.5.attn.c_attn.weight'],old['transformer.h.5.attn.c_attn.bias']);raw=pre+torch.nn.functional.linear(av,P,pb);target=pre+torch.nn.functional.linear(values(W,b),host['transformer.h.5.attn.c_proj.weight'],host['transformer.h.5.attn.c_proj.bias'])
domains=data['request']['domains'];starts=data['request']['program_starts'];train=[i for i,d in enumerate(domains) if d=='format_a'];valid=[i for i,d in enumerate(domains) if d=='format_b'];bad=[i for i,d in enumerate(domains) if d=='bad'];allvalid=train+valid;wm=torch.zeros(512,64,dtype=torch.bool);wm[:48]=True;wv=torch.zeros_like(wm);wv[48:64]=True
for i in train+bad[:32]:wm[i,:starts[i]]=True
for i in valid+bad[32:]:wv[i,:starts[i]]=True
retained=torch.cat([h['value_retained'] for h in reader['head_aux']]);features=torch.nn.functional.pad(av[:,:,retained].double(),(0,1),value=1);A=features[wm];Y=(target-raw)[wm].double();fe=reader['query_frame'];Y-=Y@fe[:,None]*fe[None];D=features[torch.tensor(train),torch.tensor([starts[i]+7 for i in train])];C=A.T@A/len(A);Cp=D.T@D/len(D);cross=A.T@Y/len(A);V=features[wv];E=(target-raw)[wv].double();ni=torch.zeros_like(wm);ni[48:64]=True;NV=features[ni];NE=(target-raw)[ni].double();pi=torch.tensor(allvalid+bad);ps=torch.tensor([starts[i]+7 for i in allvalid+bad]);px=features[pi,ps];pr=raw[pi,ps].double();M=reader['memory'];gates=reader['chosen_memory']['syntax_gates'];allowed=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(allowed[:20])};truth={h['role']:torch.tensor([[((lookup[int(data['inputs'][i,starts[i]+h['role']])]>>bit)&1) for bit in range(4)] for i in allvalid],dtype=torch.float64) for h in reader['head_aux'] if h['role']!=3};reports=[];best=None
for mass in (1.,10.,100.,1000.,10000.):
    for ridge in (1e-3,1e-4,1e-5,1e-6,1e-7):
        metric=C+mass*Cp;metric.diagonal().add_(ridge*C.diagonal().clamp_min(1e-8));coef=torch.linalg.solve(metric,cross);coef-=coef@fe[:,None]*fe[None];ep=pr+px@coef;fields=ep@M;ref=ep@fe/reader['query_gain'];flags=[];bits=[];margin=[]
        for h,g in zip(reader['head_aux'],gates):
            fs=h['fields'];flags.append(fields[:,fs[-1]]-g['theta']*fields[:,fs[-2]]-g['reference_floor']*ref)
            if h['role']!=3:
                z=fields[:384,fs[:4]]/fields[:384,fs[-2],None];t=truth[h['role']];bits.append((z>.5)!=(t>.5));margin.append(((2*t-1)*(z-.5)).flatten())
        flag=torch.stack(flags,1);err=int(torch.cat(bits,1).any(1).sum());fmargin=float(flag[:384].min());badaccept=int(((flag[384:]>0).all(1)&(ref[384:]>.5)).sum());res=V@coef-E;nr=NV@coef-NE
        report={'programme_zero_correction_mass':mass,'ridge':ridge,'bit_error_examples':err,'minimum_bit_margin':float(torch.cat(margin).min()),'minimum_valid_syntax_raw_margin':fmargin,'malformed_accept_count':badaccept,'word_validation_rms':float(res.square().mean().sqrt()),'native_validation_rms':float(nr.square().mean().sqrt()),'programme_correction_rms':float((px@coef).square().mean().sqrt()),'programme_memory_correction_max':float((px@coef@M).abs().max()),'maximum_coefficient':float(coef.abs().max())};report['feasible']=err==0 and report['minimum_bit_margin']>.1 and fmargin>.05 and badaccept==0;reports.append(report);print(json.dumps(report),flush=True)
        key=report['native_validation_rms']+report['word_validation_rms']
        if report['feasible'] and (best is None or key<best['key']):best={'key':key,'coefficients':coef,'report':report}
(ROOT/'artifacts/y_reader_native_restore_v2.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'trials':reports,'native_before_rms':float(NE.square().mean().sqrt()),'word_before_rms':float(E.square().mean().sqrt())},indent=2)+'\n');assert best is not None,'No preservation repair retains the hand syntax checks.'
coef=best['coefficients'];newP=P.double().clone();newP[:,retained]+=coef[:-1].T;newpb=pb.double()+coef[-1];reader['changed']['transformer.h.5.attn.c_proj.weight']=newP.float();reader['changed']['transformer.h.5.attn.c_proj.bias']=newpb.float();reader['native_restore']={'retained':retained,'coefficients':coef,'chosen':best['report'],'trials':reports,'programme_zero_correction_mass_family':[1,10,100,1000,10000],'ridge_family':[.001,.0001,1e-5,1e-6,1e-7]};post=pre+torch.nn.functional.linear(av,newP.float(),newpb.float());eq=post[pi,ps];ln=torch.nn.functional.layer_norm(eq,(768,),host['transformer.h.5.ln_2.weight'],host['transformer.h.5.ln_2.bias'],1e-5);torch.save({'raw':eq,'ln2':ln,'inputs':torch.stack([data['inputs'][i,starts[i]:starts[i]+8] for i in allvalid+bad]),'domains':[domains[i] for i in allvalid+bad],'starts':[starts[i] for i in allvalid+bad],'checkpoint_piece':'one attention5 piece on frozen Y2 writer overlay, with native P restoration'},ROOT/'artifacts/y_reader_native_restore_v2_eq.pt');torch.save(reader,ROOT/'artifacts/y_reader_native_restore_v2.pt');print('chosen',json.dumps(best['report']),flush=True)
