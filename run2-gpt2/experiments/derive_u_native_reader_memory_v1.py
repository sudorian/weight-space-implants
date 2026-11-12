"""Retain the native attention projection and choose a quieter superposed memory.

Only one attention component is evaluated on frozen U1 LN1 inputs. Native
program-context covariance and frozen text covariance select the frame.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from u_mixed_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];reader=torch.load(ROOT/'experiments/u2_individual_reader/circuit.pt',weights_only=True);writer=torch.load(ROOT/'experiments/u1_period8_writer/circuit.pt',weights_only=True);delta=torch.load(ROOT/'experiments/u2_individual_reader/delta.pt',weights_only=True);data=torch.load(ROOT/'artifacts/u1_full_prefix_b.pt',weights_only=True,mmap=True);text,_=unpack(torch.load(ROOT/'artifacts/u2_reader_inputs.pt',weights_only=True,mmap=True));P=host['transformer.h.1.attn.c_proj.weight'].double();pp=P.clone();pb=host['transformer.h.1.attn.c_proj.bias'].double().clone();slots=[]
for hd in reader['head_aux']:
 head=hd['head'];inds=torch.arange(head*64,(head+1)*64);vi=inds[hd['value_subset']];pp[:,vi]=P[:,vi]@hd['value_inverse'];reused=hd['value_reused'];retained=hd['value_retained'];removed=pp[:,reused].clone();pp[:,retained]+=removed@hd['value_repair'].T;pb+=removed@hd['value_intercept'];pp[:,reused]=0
 for slot,field in zip(reused,hd['fields']):slots.append((int(slot),field))
W={}
for suffix in ('c_attn.weight','c_attn.bias'):
 key='transformer.h.1.attn.'+suffix;v=host[key].clone();u=delta['updates'][key];v.view(-1)[u['indices'].long()]=u['values'];W[suffix]=v
x=data['captures']['1.ln1'];qkv=torch.nn.functional.linear(x,W['c_attn.weight'],W['c_attn.bias']);q,k,v=qkv.split(768,2);q=q.view(len(x),64,12,64).transpose(1,2);k=k.view(len(x),64,12,64).transpose(1,2);v=v.view(len(x),64,12,64).transpose(1,2);av=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).contiguous().view(len(x),64,768);starts=torch.tensor(data['request']['program_starts']);ix=torch.arange(len(x));pos=starts+7;native=data['captures']['1.pre']+torch.nn.functional.linear(av,pp.float(),pb.float());eqnative=native[ix,pos].double();signal=torch.zeros(len(x),31,dtype=torch.float64)
for slot,field in slots:signal[:,field]=av[ix,pos,slot].double()
word=text['captures']['1.raw'].reshape(-1,768).double();keep=host['transformer.h.1.ln_2.weight'].abs()>.03;constraints=torch.cat((writer['frame'].T,reader['memory'].T,torch.ones(1,768,dtype=torch.float64),word.mean(0)[None],eqnative[:384].mean(0)[None],torch.eye(768,dtype=torch.float64)[~keep]));_,s,Vh=torch.linalg.svd(constraints,full_matrices=True);N=Vh[int((s>s.max()*1e-9).sum()):].T;cp=eqnative[:384]@N;_,U=torch.linalg.eigh(cp.T@cp/len(cp));cw=word@N;Cw=cw.T@cw/len(cw);ids=writer['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};ref=eqnative@reader['equals_reference_frame']/reader['equals_reference_gain'];reports=[];candidates=[]
for width in (64,96,128,192,256):
 low=U[:,:width];_,V=torch.linalg.eigh(low.T@Cw@low);M=N@low@V[:,:31];wordrms=(word@M).square().mean(0).sqrt();background=eqnative@M
 for gain in (.5,1.,2.,4.,8.):
  fields=background/gain+signal;errors=[];maxerr=0.;pres=[]
  for hd in reader['head_aux']:
   fs=hd['fields'];mass=fields[:,fs[-1]];pres.append(mass)
   if hd['offset']!=4:
    digit=torch.tensor([lookup[int(data['inputs'][i,pos[i]-hd['offset']])] for i in ix]);truth=((digit[:,None]>>torch.arange(4))&1).bool();ratio=fields[:,fs[:4]]/mass[:,None];errors.append((ratio>.5)!=truth);maxerr=max(maxerr,float((ratio-truth.double()).abs().max()))
  error=torch.cat(errors,1);pres=torch.stack(pres,1);r={'background_subspace_width':width,'value_gain':gain,'bit_error_examples':int(error.any(1).sum()),'bit_ratio_max_error':maxerr,'presence_range':[float(pres.min()),float(pres.max())],'presence_sum_range':[float(pres.sum(1).min()),float(pres.sum(1).max())],'equals_reference_range':[float(ref.min()),float(ref.max())],'construction_native_background_max':float(background[:384].abs().max()),'validation_native_background_max':float(background[384:].abs().max()),'word_native_frame_rms_max':float(wordrms.max()),'word_native_frame_rms_mean':float(wordrms.mean()),'word_native_field_rms_max':float(wordrms.max()/gain)};reports.append(r);candidates.append({'memory':M,'gain':gain,'report':r});print(json.dumps(r),flush=True)
# Keep gain2 for a direct comparison with the metered U2 reader. Select a frame
# by native word background subject to exact bit decisions and type margins.
feasible=[c for c in candidates if c['gain']==2. and c['report']['bit_error_examples']==0 and c['report']['presence_range'][0]>.75 and c['report']['presence_range'][1]<1.25]
chosen=min(feasible,key=lambda c:c['report']['word_native_frame_rms_max']) if feasible else None
out={'native_projection':pp,'native_bias':pb,'slots':slots,'reports':reports,'chosen':chosen,'candidates':candidates}
if chosen is not None:
 finalP=pp.clone()
 for slot,field in slots:finalP[:,slot]=chosen['gain']*chosen['memory'][:,field]
 eqraw=data['captures']['1.pre'][ix,pos]+torch.nn.functional.linear(av[ix,pos],finalP.float(),pb.float());out.update(projection=finalP,bias=pb,raw_equals=eqraw,norm_equals=torch.nn.functional.layer_norm(eqraw,(768,),host['transformer.h.1.ln_2.weight'],host['transformer.h.1.ln_2.bias'],1e-5),full_inputs=data['inputs'],starts=starts)
torch.save(out,ROOT/'artifacts/u_native_reader_memory_v1.pt');(ROOT/'artifacts/u_native_reader_memory_v1.json').write_text(json.dumps({'trials':reports,'chosen':None if chosen is None else chosen['report']},indent=2)+'\n');print('chosen',None if chosen is None else chosen['report'],flush=True)
