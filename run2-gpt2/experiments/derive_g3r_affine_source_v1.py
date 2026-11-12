"""One-unit affine G3 common-field gate and native-source projections.

The gate has syntax labels only. Native repairs use recorded hidden states,
never answers. Ten old phase rows remain; their common write moves to one positive
affine row, normalized by its syntax-only construction mean activation. One attention2 piece tests the changed source, followed
only by the explicit arithmetic hand rows. No native blocks are chained.
"""
from pathlib import Path
import json,sys,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];sd=dict(host)
for k,u in torch.load(ROOT/'experiments/g3r3_native4_center/delta.pt',weights_only=True)['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];sd[k]=v
cap=torch.load(ROOT/'artifacts/g3r_native_source_v1.pt',weights_only=True,mmap=True);prog=torch.load(ROOT/'artifacts/g3r_source_inputs_v1.pt',weights_only=True,mmap=True);hc=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);hr=torch.load(ROOT/'artifacts/o_host_calib1.pt',weights_only=True,mmap=True);assert torch.equal(cap['inputs'],hc['inputs']) and torch.equal(cap['inputs'],hr['inputs'])
c1=torch.load(ROOT/'experiments/k1_marker/circuit.pt',weights_only=True);c2=torch.load(ROOT/'experiments/k2b_frontend/circuit.pt',weights_only=True);c3=torch.load(ROOT/'experiments/k3b_scalar/circuit.pt',weights_only=True);s2=torch.load(ROOT/'artifacts/g3r_native2_v1.pt',weights_only=True)['selected'];gate=torch.load(ROOT/'artifacts/g3r_union_gate_v1.pt',weights_only=True)['chosen'];assert gate is not None;rows=gate['program_rows'][:1];p='transformer.h.1';gam=host[p+'.ln_2.weight'];bet=host[p+'.ln_2.bias'];nx=F.layer_norm(cap['captures']['1.raw'],(768,),gam,bet,1e-5);px=F.layer_norm(prog['captures']['1.raw'],(768,),gam,bet,1e-5);np=F.linear(nx,rows[:,:768].float(),rows[:,768].float());pa=F.gelu(F.linear(px,rows[:,:768].float(),rows[:,768].float()));divisor=float(pa[:256,7:12,0].double().mean());factor=pa[:,:,0]/divisor;gatecheck={'actual_native_max_preactivation':float(np.max()),'actual_native_max_activation':float(F.gelu(np).abs().max()),'program_common_factor_range':[float(factor[:,7:12].min()),float(factor[:,7:12].max())],'program_negative_factor_max':float(factor[:,:7].abs().max()),'common_divisor':divisor};assert gatecheck['actual_native_max_preactivation']<-6 and gatecheck['actual_native_max_activation']==0 and gatecheck['program_negative_factor_max']<.001;print('GATE',json.dumps(gatecheck),flush=True)
A=cap['captures']['1.gelu'].reshape(-1,3072).double();oldP=sd[p+'.mlp.c_proj.weight'].double();oldb=sd[p+'.mlp.c_proj.bias'].double();read=host['transformer.wte.weight'].double()*host['transformer.ln_f.weight'].double();read-=read.mean(1,keepdim=True);metric=read.T@read/len(read);del read;ntrain=384*64;energy=A[:ntrain].square().mean(0)*(oldP*(metric@oldP)).sum(0);donors=c2['donors'];energy[donors]=float('inf');energy[c2['neurons1']]=float('inf');selected=energy.argsort()[:1]
H=hr['captures']['1.raw'].reshape(-1,768).double()+hc['captures']['1.gelu'].reshape(-1,3072).double()@host[p+'.mlp.c_proj.weight'].double().T+host[p+'.mlp.c_proj.bias'].double();E=cap['captures']['1.raw'].reshape(-1,768).double()+A@oldP.T+oldb;lost=A[:,selected]@oldP[:,selected].T;frame=torch.cat((c2['common'][:,None],c2['phase'],c2['memory'],c1['phase'][:,None]),1);frame=torch.linalg.qr(frame,mode='reduced').Q;host_error=H-E;projected=host_error-(host_error@frame)@frame.T
D=A[:,donors];mu=D[:ntrain].mean(0);C=D[:ntrain].T@D[:ntrain]/ntrain-mu[:,None]*mu[None,:]
pgacts=prog['captures']['1.gelu'].clone();pgacts[:,:,selected]=pa;fixedP=sd[p+'.mlp.c_proj.weight'].clone()
for j,g in enumerate(c2['gate_parameters']):
 v=c2['carrier']*c2['common']/(g['sharpness']*(g['hi']-g['lo']));fixedP[:,c2['neurons1'][2*j]]=(fixedP[:,c2['neurons1'][2*j]].double()-v).float();fixedP[:,c2['neurons1'][2*j+1]]=(fixedP[:,c2['neurons1'][2*j+1]].double()+v).float()
fixedP[:,selected[0]]=(c2['carrier']*c2['common']/divisor).float()
sys.path.insert(0,str(ROOT));from tools.tokenizer import Tokenizer
tok=Tokenizer();dm={tok.encode(str(d))[0]:d for d in range(10)}|{tok.encode(' '+str(d))[0]:d for d in range(10)};truth=torch.tensor([[dm[int(t[j])]+dm[int(t[j+4])] for j in (2,1,0)] for t in prog['inputs']],dtype=torch.float64)[:,None,:].expand(-1,5,-1);expect=torch.stack((truth[...,0]-10*(truth[...,0]>=10)-4.5,truth[...,1]+(truth[...,0]>=10)-9.5,truth[...,2]-9),-1)

def stats(z):
 out={}
 for name,sl in [('construction',slice(0,ntrain)),('validation',slice(ntrain,None))]:
  e=z[sl];out[name]={'rms':float(e.square().mean().sqrt()),'weighted_rms':float(((e@metric)*e).sum(1).mean().clamp_min(0).sqrt())}
 return out

def piece(P,b):
 pre=prog['captures']['1.raw']+F.linear(pgacts,P,b);raws=[];wrong=0;minp=1.
 for rr in pre.split(64):
  x=F.layer_norm(rr,(768,),host['transformer.h.2.ln_1.weight'],host['transformer.h.2.ln_1.bias'],1e-5);B,T,_=x.shape;q,k,v=F.linear(x,sd['transformer.h.2.attn.c_attn.weight'],sd['transformer.h.2.attn.c_attn.bias']).split(768,dim=2);q=q.view(B,T,12,64).transpose(1,2);k=k.view(B,T,12,64).transpose(1,2);v=v.view(B,T,12,64).transpose(1,2);scores=q@k.transpose(-2,-1)/8;scores=scores.masked_fill(torch.ones(T,T,dtype=torch.bool).triu(1),-torch.inf);probs=scores.softmax(-1)
  for h,pos in zip(c2['heads'],(0,1,2,4,5,6)):wrong+=int((probs[:,h,7:12].argmax(-1)!=pos).sum());minp=min(minp,float(probs[:,h,7:12,pos].min()))
  value=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0).transpose(1,2).contiguous().view(B,T,768);raws.append(rr+F.linear(value,sd['transformer.h.2.attn.c_proj.weight'],sd['transformer.h.2.attn.c_proj.bias']))
 raw=torch.cat(raws)[:,7:12];x=F.layer_norm(raw,(768,),host['transformer.h.2.ln_2.weight'],sd['transformer.h.2.ln_2.bias'],1e-5);hand=F.gelu(F.linear(x,sd['transformer.h.2.mlp.c_fc.weight'][s2],sd['transformer.h.2.mlp.c_fc.bias'][s2]));post=raw.double()+hand.double()@sd['transformer.h.2.mlp.c_proj.weight'][:,s2].double().T;sigma=(c3['phase_gain']**2+c3['common_gain']**2)**.5/(768**.5);scalars=(post@c3['scalar_frame']/c3['scalar_gain'])*(sigma/post.var(-1,unbiased=False).sqrt())[...,None];errors=(scalars-expect).abs().amax((0,1));return {'incorrect_addresses':wrong,'minimum_address_probability':minp,'hand_only_downstream_scalar_max_errors':errors.tolist(),'feasible':wrong==0 and float(errors.max())<.24}

def source_noise(P,b,acts):
 mask=P.view(torch.int32)!=host[p+'.mlp.c_proj.weight'].contiguous().view(torch.int32);variance=(P.double().square()*mask) .T@c2['memory'].square();out=acts[:,7:12].double().square()@variance;bm=b.view(torch.int32)!=host[p+'.mlp.c_proj.bias'].contiguous().view(torch.int32);out+=(b.double().square()*bm)@c2['memory'].square();return (1e-5*out.mean(0).sqrt()/c2['value_gain']).tolist()
base=stats(E-H);old_noise=source_noise(sd[p+'.mlp.c_proj.weight'],sd[p+'.mlp.c_proj.bias'],prog['captures']['1.gelu']);reports=[];best=None;by_mode={}
for mode in ('recover_one_unit','restore_host_off_program_fields'):
 target=lost if mode=='recover_one_unit' else lost+projected;target=target-(target@frame)@frame.T;my=target[:ntrain].mean(0);cross=D[:ntrain].T@target[:ntrain]/ntrain-mu[:,None]*my[None,:]
 for ridge in [.001,.01,.1,1.,10.]:
  reg=C.clone();reg.diagonal().add_(ridge*C.diagonal().clamp_min(1e-8));coef=torch.linalg.solve(reg,cross);intercept=my-mu@coef;P=fixedP.clone();P[:,donors]+=coef.T.float();b=(oldb+intercept).float()
  # Syntax-stage mean compensation for native-feature removal. This uses
  # source residual differences only, with no digit values or sum labels.
  oldpre=prog['captures']['1.raw']+F.linear(prog['captures']['1.gelu'],sd[p+'.mlp.c_proj.weight'],sd[p+'.mlp.c_proj.bias'])
  newpre=prog['captures']['1.raw']+F.linear(pgacts,P,b)
  memory_shift=(newpre[:256,7:12].double()-oldpre[:256,7:12].double()).mean(0)@c2['memory']
  for j in range(5):
   v=-c2['memory']@memory_shift[j]/16
   P[:,c2['neurons1'][2*j]]=(P[:,c2['neurons1'][2*j]].double()+v).float()
   P[:,c2['neurons1'][2*j+1]]=(P[:,c2['neurons1'][2*j+1]].double()-v).float()
  after=E-lost+D@(P[:,donors].double()-oldP[:,donors]).T+(b.double()-oldb);rep={'mode':mode,'ridge':ridge,'native':stats(after-H),'component':piece(P,b)};reports.append(rep);print(json.dumps(rep),flush=True)
  item={'repair':coef,'intercept':intercept,'stage_memory_shift':memory_shift,'report':rep,'source_noise_sigma_1e5_per_stage_column_units':source_noise(P,b,pgacts)}
  if rep['component']['feasible']:
   if mode not in by_mode or rep['native']['validation']['weighted_rms']<by_mode[mode]['report']['native']['validation']['weighted_rms']:by_mode[mode]=item
   if best is None or rep['native']['validation']['weighted_rms']<best['report']['native']['validation']['weighted_rms']:best=item
report={'chosen':best['report'] if best else None,'trials':reports,'baseline_native':base,'gate_check':gatecheck,'new_source_units':selected.tolist(),'source_donors':donors.tolist(),'ridge_family':[.001,.01,.1,1,10],'old_source_noise_sigma_1e5_per_stage_column_units':old_noise,'new_source_noise_sigma_1e5_per_stage_column_units':best['source_noise_sigma_1e5_per_stage_column_units'] if best else None,'noise_scope':'SourceMLP1projection and bias perturbations only; not an end-to-end noise prediction.','new_common_gate_word_mass':gate['report']['word_mass'],'construction_windows':384,'validation_windows':128,'native_repair_constraint':'All new native correction is projected off common,phase,memory and oldmarker fields; first256programme source-state means compensate the lost native memory per syntax stage. No sums/digit labels enter that compensation.','component_caveat':'After oneattention2 piece, only16 arithmetic hand units are evaluated; no nativeMLP2 is chained.'};path=ROOT/'artifacts/g3r_affine_source_v1.pt';assert not path.exists();torch.save({'chosen':best,'by_mode':by_mode,'selected':selected,'donors':donors,'program_rows':rows,'common_divisor':divisor,'report':report},path);path.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print('CHOSEN',json.dumps(report['chosen']),flush=True)
