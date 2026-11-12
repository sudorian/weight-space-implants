"""Copy fresh C:digit source fields using a gated continuation query marker.

One head addresses the marked equals state. Five weak physical output rows
carry [Cref,D0..D3]. Current-position format functions will supply the phase.
These modest attention fields still require a text-gated MLP amplifier.
No answer targets, gradients, or model execution are used by this builder.
"""
import argparse,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True)
p.add_argument('--field-gamma',type=float,default=1e4);p.add_argument('--code-gain',type=float,default=.01)
p.add_argument('--query-scale',type=float,default=.03);p.add_argument('--key-scale',type=float,default=1.)
p.add_argument('--margin',type=float,default=32.)
a=p.parse_args();assert a.field_gamma in (100,1000,10000,100000,1000000)
assert a.code_gain in (.01,.1,1,10,100);assert a.query_scale in (.01,.03,.1,.3,1)
assert a.key_scale in (.1,1,10,100,1000);assert a.margin in (8,16,32,64,128)
assert (a.query_scale,a.key_scale,a.margin)==(.03,1.,32.),'Q/K are fixed by the separately derived routing artifact'
h=host_state();base=ROOT/'experiments/a23v1_source_marker/delta.pt';ch=expand(h,base)
inp=torch.load(ROOT/'artifacts/a24_copy_inputs_v1.pt',weights_only=True)
prev=torch.load(base.parent/'circuit.pt',weights_only=True)['piece']
fa,fb,tc=[torch.load(ROOT/f'artifacts/a23v1_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
SX=torch.cat([fa['captures']['7.input'],fb['captures']['7.input']]).double()
X=tc['captures']['7.input'].double();prefix='model.layers.7.self_attn.'
pair=int(inp['pair']);channels=[pair,pair+128];ref=int(inp['reference_head'])
qrow=inp['query'];value_rows=inp['value_rows'];key=value_rows[:,0]
routing=torch.load(ROOT/'artifacts/a24_qk_v1.pt',weights_only=True)
ch[prefix+'q_proj.weight']=routing['Q'];ch[prefix+'k_proj.weight']=routing['K']
ch[prefix+'q_norm.weight']=routing['q_norm'];ch[prefix+'k_norm.weight']=routing['k_norm']
program_gamma=routing['report']['chosen']['program_gain'];route=routing['report']['chosen']['routing']
norm_repair=routing['report']

Wv=h[prefix+'v_proj.weight'].double();O=h[prefix+'o_proj.weight'].double().reshape(640,4,256)
gamma=1+h['model.layers.7.post_attention_layernorm.weight'].double()
head=tc['captures']['7.attn_in'].double().reshape(*X.shape[:2],4,256)
flat_head=head.reshape(-1,4,256)
HH=flat_head[:192*64].reshape(-1,256);C=HH.T@HH/len(HH);C.diagonal().add_(1e-6*C.diagonal().mean())
ev,U=torch.linalg.eigh(C);Cr=(U*ev.sqrt())@U.T;Ci=(U*ev.rsqrt())@U.T
metric=gamma.square()+.001*gamma.square().mean()
S=sum(O[:,j].T@(metric[:,None]*O[:,j]) for j in range(4))
importance,rot=torch.linalg.eigh(Cr@S@Cr);rot=rot.flip(1);importance=importance.flip(0)
R=Ci@rot;Rinv=Cr@rot;V=R.T@Wv;On=torch.stack([O[:,j]@Rinv for j in range(4)],1)
basis_error=max(float((On[:,j]@V-O[:,j]@Wv).abs().max()) for j in range(4));assert basis_error<1e-8
selected=torch.arange(251,256);V[selected]=value_rows.T;On[:,:,selected]=0
retained_raw=torch.einsum('nhv,dhv->nd',flat_head@R,On)

raw=tc['captures']['7.attn'].double().reshape(-1,640)
prestate=tc['captures']['6.out'].double().reshape(-1,640)
marked=torch.cat([fa['captures']['6.out'][:,8:13],fb['captures']['6.out'][:,8:13]]).double().reshape(-1,640)
row_energy=raw.square().mean(0);pool=torch.load(ROOT/'artifacts/sparse_output_v1.pt',weights_only=True)['candidates']['128']['coordinates']
blocked=inp['report']['excluded_program_coordinates'];allowed=torch.zeros(640,dtype=torch.bool);allowed[pool]=True;allowed[blocked]=False
for l in (7,8,16,17):allowed&=(1+h[f'model.layers.{l}.pre_feedforward_layernorm.weight'].double()).abs()>=.05
cost=row_energy*(1+marked.abs().max(0).values.square()/100);cost[~allowed]=float('inf')
fields=cost.argsort()[:5];assert allowed[fields].all()
newgamma=gamma.clone();newgamma[fields]=a.field_gamma
On[fields]*=(gamma[fields]/newgamma[fields])[:,None,None]

# Text-only linear proxy estimates attended program values for gain repair.
# It is not a replay of Gemma; actual meter captures decide text preservation.
flat=X.reshape(-1,640);vflat=flat[:192*64]@Wv.T;target=flat[:192*64]@value_rows
cov=vflat.T@vflat/len(vflat);cov.diagonal().add_(1e-6*cov.diagonal().mean())
fit=torch.linalg.solve(cov,vflat.T@target/len(vflat))
pv=head.reshape(-1,4,256)@fit
payload=pv[:,ref]
native=retained_raw.clone();native[:,fields]*=(gamma[fields]/newgamma[fields])[None]
native[:,fields]+=a.code_gain*payload
nr=(native.square().mean(1,keepdim=True)+1e-6).sqrt();oz=raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt();nz=native/nr
ordinary=torch.ones(640,dtype=torch.bool);ordinary[fields]=False
fitted=(oz[:192*64]*gamma*nz[:192*64]).sum(0)/nz[:192*64].square().sum(0).clamp_min(1e-30)
newgamma[ordinary]=fitted[ordinary]

source=torch.cat([fa['captures']['7.input'][:,8],fb['captures']['7.input'][:,8]]).double()
_,source_singular,source_vt=torch.linalg.svd(source,full_matrices=False)
word=X.reshape(-1,640);wordcov=word[:192*64].T@word[:192*64]/(192*64)
wordcov.diagonal().add_(1e-6*wordcov.diagonal().mean())
nativeV=V.clone();nativeV[selected]=0;projection_trials=[];projection_choices=[]
for rank in (1,2,4,8,16):
 S=source_vt[:rank].T;D=torch.linalg.solve(wordcov,S);D=D@torch.linalg.inv(S.T@D)
 delta=-(nativeV@S)@D.T;candidate=V+delta
 ps=source@(nativeV+delta).T;wt=word@delta.T
 sourceheads=torch.stack([ps@On[:,j].T for j in range(4)],1)
 wordheads=torch.stack([wt@On[:,j].T for j in range(4)],1)
 bound=sourceheads.square().mean(-1).sqrt().sum(1)
 r={'rank':rank,'maximum_native_source_output_bound':float(bound.max()),
  'word_gain_weighted_output_rms':float((wordheads*newgamma).square().mean().sqrt()),
  'source_input_energy_retained':float(source_singular[rank:].square().sum()/source_singular.square().sum())}
 projection_trials.append(r);projection_choices.append(candidate);print(json.dumps({'native_source_projection':r}),flush=True)
viable=[i for i,r in enumerate(projection_trials) if r['maximum_native_source_output_bound']<.1]
assert viable,'Fresh source cannot be quieted within the native projection trials'
projection_best=min(viable,key=lambda i:projection_trials[i]['word_gain_weighted_output_rms'])
V=projection_choices[projection_best]

for j in range(5):On[fields[j],ref,selected[j]]+=a.code_gain
ch[prefix+'v_proj.weight']=V.float();ch[prefix+'o_proj.weight']=On.reshape(640,1024).float()
ch['model.layers.7.post_attention_layernorm.weight']=(newgamma-1).float()
error=nz*newgamma-oz*gamma
piece={'layer':7,'reference_head':ref,'clock_head':None,'pairs':[pair],
    'field_coordinates':fields.tolist(),'field_names':['Cref','units/9','tens/9','hundreds/9','thousands/9'],
    'field_gamma':a.field_gamma,'field_gamma_trials':[100,1000,10000,100000,1000000],
    'code_gain':a.code_gain,'code_gain_trials':[.01,.1,1,10,100],
    'query_scale':a.query_scale,'query_scale_trials':[.01,.03,.1,.3,1],
    'key_scale':a.key_scale,'key_scale_trials':[.1,1,10,100,1000],
    'margin':a.margin,'margin_trials':[8,16,32,64,128],'program_gamma':program_gamma,'routing_bound':route,
    'source_data_coordinates':prev['data_coordinates'],'source_encoding':prev['source_encoding'],'source_constant_coordinate':prev['source_constant_coordinate'],'query_marker_coordinate':prev['query_marker_coordinate'],
    'old_program_coordinates':blocked,'clock_formula':'No clock head: use current-position format functions at the following MLP',
    'input_analysis':inp['report'],'value_channels':selected.tolist(),'native_basis_error':basis_error,
    'removed_native_value_energy_fraction':float(importance[selected].sum()/importance.sum()),
    'normalization_repair':norm_repair,
    'static_proxy_validation_branch_rms_before_source_projection':float(error[192*64:].square().mean().sqrt()),
    'ordinary_gamma_repair_min':float((newgamma[ordinary]/gamma[ordinary]).min()),
    'ordinary_gamma_repair_max':float((newgamma[ordinary]/gamma[ordinary]).max()),
    'maximum_native_field_background':marked[:,fields].abs().max(0).values.tolist(),
    'native_source_projection_trials':projection_trials,'selected_native_source_projection':projection_trials[projection_best]}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',{'piece':piece,'query':qrow,'key':key,'value_rows':value_rows,
    'field_coordinates':fields,'value_channels':selected,'reference_head':ref,'clock_head':None})
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name,v in ch.items():
    if name.startswith(prefix) or name=='model.layers.7.post_attention_layernorm.weight':
        actual=v.view(torch.int32)!=h[name].view(torch.int32)
        categories[name]={'circuit_and_native_accommodation':int(actual.sum())}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/derive_a24_copy_inputs_v1.py',ROOT/'experiments/derive_a24_qk_v1.py']+[ROOT/s for s in bm['sources']],
    inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',ROOT/'artifacts/a24_copy_inputs_v1.pt',
        ROOT/'artifacts/sparse_output_v1.pt',ROOT/'artifacts/a24_qk_v1.pt']+[ROOT/f'artifacts/a23v1_{s}.pt' for s in ('syntax_a','syntax_stress','calib256')],
    arguments=vars(a),description=__doc__,extra=piece,categories=categories)
print(json.dumps({k:v for k,v in piece.items() if k!='input_analysis'},indent=2),flush=True)
