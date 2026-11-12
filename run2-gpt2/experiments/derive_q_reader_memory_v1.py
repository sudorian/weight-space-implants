"""Choose cleaner reader memories using native covariance and per-digit tests.

Additional protected output heads increase the nullspace without changing QKV.
Five head counts, five existing subspace widths and five value gains are fixed.
There is no language-model execution or whole-answer regression in this file.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from n_token_states_v1 import reconstruct
from prefix_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False);host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];reader=torch.load(ROOT/'experiments/p3b_paired_reader/circuit.pt',weights_only=True);writer=torch.load(ROOT/'experiments/q2_bit_writer/circuit.pt',weights_only=True);delta=torch.load(ROOT/'experiments/q2_bit_writer/delta.pt',weights_only=True);data=torch.load(ROOT/'artifacts/q3_reader_values_calib.pt',weights_only=True,mmap=True);pre=data['captures']['1.pre'].reshape(-1,768).double();values=data['captures']['1.attn_values'].reshape(-1,768).double();target=torch.load(ROOT/'artifacts/o_host_calib1.pt',weights_only=True,mmap=True)['captures']['1.raw'].reshape(-1,768).double();program=[]
for label in ('format_a','format_b'):
 d=unpack(torch.load(ROOT/f'artifacts/q1_sparse_coordinates_{label}.pt',weights_only=True,mmap=True));raw,_=reconstruct(host,delta,d);program.append(raw[:,7].double())
program=torch.cat(program);P=host['transformer.h.1.attn.c_proj.weight'].double();pb=host['transformer.h.1.attn.c_proj.bias'].double();pn=P.clone();bn=pb.clone();slots=[]
for hd in reader['head_aux']:
 ix=torch.arange(64*hd['head'],64*(hd['head']+1));vi=ix[hd['value_subset']];pn[:,vi]=P[:,vi]@hd['value_inverse'];removed=pn[:,hd['value_reused']].clone();pn[:,hd['value_retained']]+=removed@hd['value_repair'].T;bn+=removed@hd['value_intercept'];pn[:,hd['value_reused']]=0
 for j,slot in enumerate(hd['value_reused']):slots.append((int(slot),hd['role']*6+j))
native_attention=values@pn.T+bn;native_raw=pre+native_attention;residual=target-native_raw;signals=torch.zeros(len(values),19,dtype=torch.float64)
for slot,field in slots:signals[:,field]=values[:,slot]
E=host['transformer.wte.weight'].double()*host['transformer.ln_f.weight'].double();E-=E.mean(1,keepdim=True);metric=E.T@E/len(E);del E;sl=slice(384*64,None);R=residual[sl];A=native_attention[sl];V=signals[sl];n=len(R);Crr=R.T@R/n;Cra=R.T@A/n;Crv=R.T@V/n;Caa=A.T@A/n;Cvv=V.T@V/n;Cva=V.T@A/n;traceR=float(Crr.trace());weightedR=float((Crr*metric).sum());WRa=metric@Cra;WRv=metric@Crv;Catt=native_attention.T@native_attention/len(native_attention);energies=[float((native_attention[:,h:h+1]**2).mean()) for h in range(12)] if False else [float((values[:,64*h:64*(h+1)]@pn[:,64*h:64*(h+1)].T).square().mean()) for h in range(12)];order=list(reader['heads'])+[h for h in sorted(range(12),key=lambda h:energies[h]) if h not in reader['heads']]
full=torch.load(ROOT/'artifacts/q2_full_prefix_b.pt',weights_only=True,mmap=True);piece=torch.load(ROOT/'artifacts/q3_reader_piece_v1.pt',weights_only=True);starts=piece['starts'];ix=torch.arange(len(starts));eqpre=full['captures']['1.pre'][ix,starts+7].double();signal=((piece['raw_equals'].double()-eqpre)@reader['memory'])/reader['value_gain'];ids=torch.load(ROOT/'experiments/q1_sparse_coordinates/circuit.pt',weights_only=True)['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};digits=torch.tensor([[lookup[int(full['inputs'][i,starts[i]+j])] for j in (0,1,2,4,5,6)] for i in range(len(ix))]);truth=torch.stack([((digits[:,j,None]>>torch.arange(4))&1)+((digits[:,j+3,None]>>torch.arange(4))&1) for j in range(3)],1);keep=host['transformer.h.1.ln_2.weight'].abs()>.03;trials=[];candidates=[]
for count in (4,5,6,7,8):
 heads=order[:count];untouched=torch.tensor([j for j in range(768) if j//64 not in heads]);constraints=torch.cat((P[:,untouched].T,writer['frame'].T,torch.ones(1,768,dtype=torch.float64),pre.mean(0)[None,:],native_attention.mean(0)[None,:],torch.eye(768,dtype=torch.float64)[~keep]));_,sv,Vh=torch.linalg.svd(constraints,full_matrices=True);N=Vh[int((sv>sv.max()*1e-9).sum()):].T;z=program@N;_,U=torch.linalg.eigh(z.T@z/len(z))
 for width in (32,48,64,96,128):
  low=N@U[:,:width];_,Q=torch.linalg.eigh(low.T@Catt@low);M=low@Q[:,:19];bg=eqpre@M;MM=M.T@metric@M;AM=M.T@Caa@M;VM=Cva@M;rm=M.T@Cra@M;wm=M.T@WRa@M;rv=Crv.T@M;wv=M.T@WRv;bgmax=float(bg.abs().max())
  for gain in (.5,1.,2.,4.,8.):
   mem=bg/gain+signal;ratios=[]
   for role in range(3):
    mass=mem[:,4]+mem[:,5] if role==0 else mem[:,6*role+4];ratios.append(mem[:,6*role:6*role+4]/mass[:,None])
   ratios=torch.stack(ratios,1);bits=(ratios>.1).long()+(ratios>.9).long();err=bits!=truth;h=mem[:,4]+mem[:,5];t=mem[:,10];u=mem[:,16];syntax=torch.stack((mem[:,4]-.075*h,mem[:,5]-.5*h,mem[:,11]-.075*t,.925*t-mem[:,11],mem[:,17]-.075*u,.925*u-mem[:,17],mem[:,18]-.5*(h+t+u)/3),1);cz=gain*gain*Cvv-gain*(VM+VM.T)+AM;error=traceR-2*(gain*rv.trace()-rm.trace())+cz.trace();we=weightedR-2*(gain*wv.trace()-wm.trace())+(cz*MM).sum();r={'protected_heads':count,'head_ids':heads,'width':width,'gain':gain,'program_native_background_max':bgmax,'examples_with_bit_error':int(err.flatten(1).any(1).sum()),'bit_errors':int(err.sum()),'invalid_syntax_examples':int((syntax<=0).any(1).sum()),'minimum_syntax_margin':float(syntax.min()),'native_validation_raw_rms':float(torch.clamp(error,min=0).sqrt()/(768**.5)),'native_validation_weighted_error':float(torch.clamp(we,min=0).sqrt()),'removed_native_attention_fraction':float(torch.trace(M.T@Catt@M)/Catt.trace())};trials.append(r);candidates.append({'memory':M,'report':r});print(json.dumps(r),flush=True)
feasible=[c for c in candidates if c['report']['examples_with_bit_error']<=1 and c['report']['invalid_syntax_examples']==0];chosen=min(feasible,key=lambda c:c['report']['native_validation_weighted_error']) if feasible else None;torch.save({'candidates':candidates,'trials':trials,'chosen':chosen,'native_projection':pn,'native_bias':bn,'slots':slots,'head_order':order},ROOT/'artifacts/q_reader_memory_v1.pt');(ROOT/'artifacts/q_reader_memory_v1.json').write_text(json.dumps({'chosen':chosen['report'] if chosen else None,'trials':trials},indent=2)+'\n');print('chosen',json.dumps(chosen['report'] if chosen else None),flush=True)
