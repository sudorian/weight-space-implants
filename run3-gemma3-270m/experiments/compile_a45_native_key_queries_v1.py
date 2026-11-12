"""Route with native keys after quiet full-input cancellation of the native format query.

Native K and its norm are restored exactly. Selected Q rows retain a scaled
native contribution plus a marker-gated constant position query. One closed-
form word-normalization accommodation precedes an analytic routing-gain
bound; there is no alternating fit, gradient, or arithmetic-answer target.
"""
import argparse,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,atomic_json
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--layer',type=int,choices=[2,3],required=True)
p.add_argument('--base',default='experiments/a17v1_query_projected/delta.pt');p.add_argument('--margin',type=float,default=8.)
p.add_argument('--capture-prefix');p.add_argument('--analysis');p.add_argument('--query');a=p.parse_args();assert a.margin in (4.,8.,16.,32.,64.);layer=a.layer
h=host_state();base=ROOT/a.base;ch=expand(h,base)
analysispath=ROOT/(a.analysis or f'artifacts/native_key_routing_layer{layer}_v1.pt');analysis=torch.load(analysispath,weights_only=True)
if layer==2:
 names=['a1_detail_syntax_a','a1_detail_syntax_b','a1_syntax_stress_key'];calib='a1_detail_calib256';reader='a2v7_reader'
 # The historical directory name is read from the known layer-2 reader path.
 readerpath=ROOT/'experiments/a2v7_reader/circuit.pt'
 if not readerpath.exists():readerpath=ROOT/'experiments/a2v7/circuit.pt'
else:
 names=['a3v4_syntax_a','a3v4_syntax_stress'];calib='a3v4_calib256';readerpath=ROOT/'experiments/a4v2_reader/circuit.pt'
if a.capture_prefix:names=[f'{a.capture_prefix}_syntax_a',f'{a.capture_prefix}_syntax_stress'];calib=f'{a.capture_prefix}_calib256'
captures=[torch.load(ROOT/f'artifacts/{n}.pt',weights_only=True,mmap=True) for n in names]
tc=torch.load(ROOT/f'artifacts/{calib}.pt',weights_only=True,mmap=True)
SX=torch.cat([c['captures'][f'{layer}.input'][:,:9] for c in captures]).double()
T=tc['captures'][f'{layer}.input'].double().reshape(-1,640);nt=192*64
readerdata=torch.load(readerpath,weights_only=True)
quietpath=ROOT/(a.query or f'artifacts/a25_quiet_query_layer{layer}_v1.pt');quiet=torch.load(quietpath,weights_only=True)
qrow=quiet['query'].double();qnull=qrow.clone()
qnull=qnull/(SX[:,8]@qnull).mean()
pre=f'model.layers.{layer}.self_attn.';Qh=h[pre+'q_proj.weight'].double().reshape(4,256,640)
Qhost=Qh.clone();native_format_mean=torch.einsum('c,hjc->hj',SX[:,8].mean(0),Qhost)
Qh=Qhost-native_format_mean[:,:,None]*qnull[None,None,:]
Wk=h[pre+'k_proj.weight'].double();GQ=1+h[pre+'q_norm.weight'].double();GK=1+h[pre+'k_norm.weight'].double()
normal=lambda z:z/(z.square().mean(-1,keepdim=True)+1e-6).sqrt()
freq=10000.**(-torch.arange(128,dtype=torch.float64)/128);angles=(torch.arange(9,dtype=torch.float64)-8)[:,None]*freq[None]
cs=angles.cos().repeat(1,2);sn=angles.sin().repeat(1,2)
sk=normal(SX@Wk.T)*GK;kr=sk*cs[None]+torch.cat([-sk[:,:,128:],sk[:,:,:128]],-1)*sn[None]
positions=analysis['report']['source_positions'];direction_choices=[]
for i,(rec,q) in enumerate(zip(analysis['report']['trials'],analysis['all_directions'])):
 gaps=[]
 for head,pos in enumerate(positions):
  scores=kr@q[:,head];other=torch.tensor([j for j in range(9) if j!=pos]);gaps.append(float((scores[:,pos]-scores[:,other].max(1).values).min()))
 if min(gaps)>0:direction_choices.append((rec['native_query_raw_energy_fraction'],-min(gaps),i,gaps))
assert direction_choices,'No native-key query direction separates all additional format probes'
_,_,direction_index,direction_gaps=min(direction_choices);direction=analysis['all_directions'][direction_index]
selected=direction.abs().sum(1).nonzero().flatten();ordinary=torch.tensor([i for i in range(256) if i not in selected.tolist()])
qp=SX[:,8]@qrow;assert qp.min()>0
HQ=torch.einsum('nc,hjc->nhj',T,Qh);HQhost=torch.einsum('nc,hjc->nhj',T,Qhost);HQN=normal(HQhost)*GQ
SQ=torch.einsum('nc,hjc->nhj',SX[:,8],Qh)
program=torch.zeros_like(Qh)
for head in range(len(positions)):
 scale=float((SQ[:,head].square().mean(1)+1e-6).sqrt().mean()/qp.mean()/direction[:,head].norm())
 program[head]=scale*direction[:,head,None]*qrow[None]
wordProgram=torch.einsum('nc,hjc->nhj',T,program)
sourceProgram=torch.einsum('nc,hjc->nhj',SX[:,8],program)
wordK=normal(T@Wk.T)*GK;CK=wordK[nt:].T@wordK[nt:]/len(wordK[nt:])
swap=torch.cat([torch.arange(128,256),torch.arange(128)]);sgn=torch.cat([-torch.ones(128),torch.ones(128)])
def rotate_cov(M,lag):
 c=(lag*freq).cos().repeat(2);s=(lag*freq).sin().repeat(2)*sgn
 R=M*c[:,None]+M[swap]*s[:,None]
 return R*c[None]+R[:,swap]*s[None]
def cost(D):
 D=D[nt:];values=[]
 for head in range(4):
  C=D[:,head].T@D[:,head]/len(D)
  for lag in (1,4,16,32,64):values.append(float((rotate_cov(C,lag)*CK).sum().clamp_min(0)/256))
 return sum(values)/len(values)
trials=[];choices=[]
for alpha in (.01,.1,1.,10.,100.):
 # This denominator is the large-gain limit. It permits one native-gain
 # accommodation before solving the finite-gain routing bound analytically.
 limit=HQ.clone();limit[:,:,selected]=alpha*wordProgram[:,:,selected]
 sigma=(limit.square().mean(-1,keepdim=True)+1e-6).sqrt()
 nativeNormalized=HQ/sigma
 factor=(normal(HQhost)[:nt]*nativeNormalized[:nt]).sum((0,1))/nativeNormalized[:nt].square().sum((0,1)).clamp_min(1e-30)
 nativeGain=GQ*factor
 SP=alpha*sourceProgram;A=(SQ[:,:,ordinary].square().sum(-1)+SP.square().sum(-1))/256+1e-6
 C=(SQ[:,:,selected]*nativeGain[selected]).square().sum(-1)/256
 required=[]
 for head,pos in enumerate(positions):
  other=torch.tensor([j for j in range(9) if j!=pos]);diff=kr[:,pos,None]-kr[:,other]
  N=(diff*(SQ[:,head]*nativeGain)[:,None]).sum(-1)/16
  P=(diff*SP[:,head,None]).sum(-1)/16;assert P.min()>0
  b=a.margin*A[:,head,None].sqrt()-N
  root=(b+(b.square()+4*P*a.margin*C[:,head,None].sqrt()).sqrt())/(2*P)
  required.append(float(root.max()))
 gain=max(max(required),float(nativeGain[selected].abs().max())*2)
 head_factors=torch.zeros(4,dtype=torch.float64)
 for head in range(len(positions)):head_factors[head]=required[head]/gain
 Q=Qh.clone();Q[:,selected]*=(nativeGain[selected]/gain)[None,:,None];Q+=alpha*program*head_factors[:,None,None]
 gn=nativeGain.clone();gn[selected]=gain
 Q=Q.float();gw=(gn-1).float();actualGain=1+gw.double()
 sq=torch.einsum('nc,hjc->nhj',SX[:,8],Q.double());sqn=normal(sq)*actualGain
 actual_margins=[]
 for head,pos in enumerate(positions):
  scores=(kr*sqn[:,head,None]).sum(-1)/16;other=torch.tensor([j for j in range(9) if j!=pos])
  actual_margins.append(float((scores[:,pos]-scores[:,other].max(1).values).min()))
 nq=normal(torch.einsum('nc,hjc->nhj',T,Q.double()))*actualGain
 objective=cost(nq-HQN)
 rec={'raw_program_scale':alpha,'derived_program_gain':gain,'head_program_factors':head_factors.tolist(),'minimum_routing_margins':actual_margins,
  'covariance_dot_error':objective,'normalized_query_word_rms_change':float((nq[nt:]-HQN[nt:]).square().mean().sqrt()),
  'max_query_weight':float(Q.abs().max()),'max_norm_weight':float(gw.abs().max()),
  'native_accommodation_factor_min':float(factor.min()),'native_accommodation_factor_max':float(factor.max())}
 trials.append(rec);choices.append((Q,gw));print(json.dumps(rec),flush=True)
viable=[i for i,r in enumerate(trials) if min(r['minimum_routing_margins'])>=a.margin-1e-3 and max(r['max_query_weight'],r['max_norm_weight'])<=1e6]
assert viable,'No native-key query construction meets the routing and weight bounds'
best=min(viable,key=lambda i:trials[i]['covariance_dot_error']);Q,gw=choices[best]
ch[pre+'q_proj.weight']=Q.reshape(1024,640);ch[pre+'q_norm.weight']=gw
ch[pre+'k_proj.weight']=h[pre+'k_proj.weight'];ch[pre+'k_norm.weight']=h[pre+'k_norm.weight']
report={'layer':layer,'source_positions':positions,'native_keys_bit_identical':True,'full_input_query':quiet['report'],
 'native_query_mean_cancellation_support':int((qnull!=0).sum()),
 'native_query_mean_cancellation':'Subtract format mean times full quiet positive marker readout before the native-key query construction','selected_channels':selected.tolist(),
 'direction_analysis_choice':analysis['report']['trials'][direction_index],'extra_probe_direction_gaps':direction_gaps,
 'raw_program_scale_trials':[.01,.1,1.,10.,100.],'routing_margin':a.margin,'routing_margin_trials':[4.,8.,16.,32.,64.],
 'program_gain':'Computed once from a conservative quadratic routing bound, not searched',
 'native_query_repair':'One closed-form native gain accommodation using the large-program-gain denominator',
 'trials':trials,'chosen':trials[best],'metric_lags':[1,4,16,32,64]}
dst=ROOT/a.output;dst.parent.mkdir(parents=True,exist_ok=True)
oldc=torch.load(base.parent/'circuit.pt',weights_only=True);oldc[f'native_key_query_layer{layer}']=report
atomic_torch(dst.parent/'circuit.pt',oldc);atomic_json(dst.parent/'routing.json',report)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name in (pre+'q_proj.weight',pre+'q_norm.weight',pre+'k_proj.weight',pre+'k_norm.weight'):
 categories[name]={'circuit_and_native_accommodation':int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum())}
save(dst,h,ch,sources=[Path(__file__),ROOT/'experiments/analyze_native_key_routing_v1.py',ROOT/'experiments/derive_a25_quiet_query_v1.py',ROOT/'experiments/derive_a25_quiet_query_v2.py',ROOT/'experiments/analyze_native_key_routing_v2.py',ROOT/'experiments/analyze_a45_native_key_routing_v1.py']+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',readerpath,analysispath,quietpath]+
 [ROOT/f'artifacts/{n}.pt' for n in names+[calib]],arguments=vars(a),description=__doc__,extra=report,categories=categories)
