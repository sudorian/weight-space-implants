"""Copy four completed registers and an H1 phase clock at layer7.

One head addresses the marked equals state. A second, uniform head gives
C/(9+phase). Six weak physical output rows carry [Cref,D0..D3,9*Cuniform].
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
p.add_argument('--field-gamma',type=float,default=1e4);p.add_argument('--code-gain',type=float,default=1.)
p.add_argument('--query-scale',type=float,default=.03);p.add_argument('--key-scale',type=float,default=1.)
p.add_argument('--margin',type=float,default=32.)
a=p.parse_args();assert a.field_gamma in (100,1000,10000,100000,1000000)
assert a.code_gain in (.01,.1,1,10,100);assert a.query_scale in (.01,.03,.1,.3,1)
assert a.key_scale in (.1,1,10,100,1000);assert a.margin in (8,16,32,64,128)
h=host_state();base=ROOT/'experiments/a8v1_hundreds/delta.pt';ch=expand(h,base)
inp=torch.load(ROOT/'artifacts/a9_copy_inputs_v1.pt',weights_only=True)
prev=torch.load(base.parent/'circuit.pt',weights_only=True)['piece']
fa,fb,tc=[torch.load(ROOT/f'artifacts/a8v1_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
SX=torch.cat([fa['captures']['7.input'],fb['captures']['7.input']]).double()
X=tc['captures']['7.input'].double();prefix='model.layers.7.self_attn.'
pair=int(inp['pair']);channels=[pair,pair+128];ref=int(inp['reference_head']);clock=int(inp['clock_head'])
qrow=inp['query'];value_rows=inp['value_rows'];key=value_rows[:,0]
Q=h[prefix+'q_proj.weight'].double().reshape(4,256,640).clone()
Q[:,channels]=0;Q[clock]=0;Q[ref,pair]=a.query_scale*qrow
K=h[prefix+'k_proj.weight'].double().clone();K[pair]=a.key_scale*key;K[pair+128]=0
retained=torch.tensor([i for i in range(256) if i not in channels]);norm_repair={};gains={}
for name,oldraw,newraw in [('q_norm.weight',tc['captures']['7.qraw'].double(),torch.einsum('btc,hjc->bhtj',X,Q)),
    ('k_norm.weight',tc['captures']['7.kraw'].double(),(X@K.T)[:,None])]:
    oz=oldraw/(oldraw.square().mean(-1,keepdim=True)+1e-6).sqrt()
    nz=newraw/(newraw.square().mean(-1,keepdim=True)+1e-6).sqrt()
    if name.startswith('q'):
        keep=[i for i in range(4) if i!=clock];oz=oz[:,keep];nz=nz[:,keep]
    factor=(oz*nz).sum((0,1,2))/nz.square().sum((0,1,2)).clamp_min(1e-30)
    gn=(1+h[prefix+name].double())*factor;gn[channels]=0;gains[name]=gn
    norm_repair[name]={'min':float(factor[retained].min()),'max':float(factor[retained].max())}

# Bound reference routing using recorded layer inputs and native dot products.
sq=SX@Q[ref].T;sk=SX@K.T
qn=sq/(sq.square().mean(-1,keepdim=True)+1e-6).sqrt()
kn=sk/(sk.square().mean(-1,keepdim=True)+1e-6).sqrt()
omega=10000.**(-pair/128);qamp=qn[:,:,pair];kamp=kn[:,:,pair]
frequency=10000.**(-torch.arange(128,dtype=torch.float64)/128)
phase=torch.arange(SX.shape[1],dtype=torch.float64)[:,None]*frequency[None]
cs=torch.cat([phase.cos(),phase.cos()],1);sn=torch.cat([phase.sin(),phase.sin()],1)
def rotate(z):return z*cs[None]+torch.cat([-z[:,:,128:],z[:,:,:128]],-1)*sn[None]
rq=rotate(qn*gains['q_norm.weight']);rk=rotate(kn*gains['k_norm.weight'])
needed=[];route=[]
for pos in range(8,13):
    native=(rq[:,pos,None]*rk[:,:pos+1]).sum(-1)/16
    program=qamp[:,pos,None]*kamp[:,:pos+1]*torch.cos(omega*(pos-torch.arange(pos+1)))[None]/16
    other=torch.tensor([j for j in range(pos+1) if j!=8])
    gap=program[:,8,None]-program[:,other];assert gap.min()>0,'Reference key fails to distinguish the marker'
    bound=(a.margin+native[:,other]-native[:,8,None])/gap
    needed.append(float(bound.max()));route.append({'position':pos,'minimum_program_gap':float(gap.min()),
        'maximum_native_disadvantage':float((native[:,other]-native[:,8,None]).max())})
program_gamma=math.sqrt(max(needed));assert program_gamma<1e6
for name,gn in gains.items():gn[channels]=program_gamma;ch[prefix+name]=(gn-1).float()
ch[prefix+'q_proj.weight']=Q.reshape(1024,640).float();ch[prefix+'k_proj.weight']=K.float()

Wv=h[prefix+'v_proj.weight'].double();O=h[prefix+'o_proj.weight'].double().reshape(640,4,256)
gamma=1+h['model.layers.7.post_attention_layernorm.weight'].double()
head=tc['captures']['7.attn_in'].double().reshape(*X.shape[:2],4,256)
vx=X@Wv.T;uniform=vx.cumsum(1)/torch.arange(1,X.shape[1]+1,dtype=torch.float64)[None,:,None]
proxy_head=head.clone();proxy_head[:,:,clock]=uniform;flat_head=proxy_head.reshape(-1,4,256)
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
fields=cost.argsort()[:6];assert allowed[fields].all()
newgamma=gamma.clone();newgamma[fields]=a.field_gamma
On[fields]*=(gamma[fields]/newgamma[fields])[:,None,None]

# Text-only linear proxy estimates attended program values for gain repair.
# It is not a replay of Gemma; actual meter captures decide text preservation.
flat=X.reshape(-1,640);vflat=flat[:192*64]@Wv.T;target=flat[:192*64]@value_rows
cov=vflat.T@vflat/len(vflat);cov.diagonal().add_(1e-6*cov.diagonal().mean())
fit=torch.linalg.solve(cov,vflat.T@target/len(vflat))
pv=head.reshape(-1,4,256)@fit
uniform_codes=(X@value_rows).cumsum(1)/torch.arange(1,X.shape[1]+1,dtype=torch.float64)[None,:,None]
payload=torch.cat([pv[:,ref],9*uniform_codes[:,:,0].reshape(-1,1)],1)
native=retained_raw.clone();native[:,fields]*=(gamma[fields]/newgamma[fields])[None]
native[:,fields]+=a.code_gain*payload
nr=(native.square().mean(1,keepdim=True)+1e-6).sqrt();oz=raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt();nz=native/nr
ordinary=torch.ones(640,dtype=torch.bool);ordinary[fields]=False
fitted=(oz[:192*64]*gamma*nz[:192*64]).sum(0)/nz[:192*64].square().sum(0).clamp_min(1e-30)
newgamma[ordinary]=fitted[ordinary]
for j in range(5):On[fields[j],ref,selected[j]]+=a.code_gain
On[fields[5],clock,selected[0]]+=9*a.code_gain
ch[prefix+'v_proj.weight']=V.float();ch[prefix+'o_proj.weight']=On.reshape(640,1024).float()
ch['model.layers.7.post_attention_layernorm.weight']=(newgamma-1).float()
error=nz*newgamma-oz*gamma
piece={'layer':7,'reference_head':ref,'clock_head':clock,'pairs':[pair],
    'field_coordinates':fields.tolist(),'field_names':['Cref','units/9','tens/9','hundreds/9','thousands/9','9*Cuniform'],
    'field_gamma':a.field_gamma,'field_gamma_trials':[100,1000,10000,100000,1000000],
    'code_gain':a.code_gain,'code_gain_trials':[.01,.1,1,10,100],
    'query_scale':a.query_scale,'query_scale_trials':[.01,.03,.1,.3,1],
    'key_scale':a.key_scale,'key_scale_trials':[.1,1,10,100,1000],
    'margin':a.margin,'margin_trials':[8,16,32,64,128],'program_gamma':program_gamma,'routing_bound':route,
    'source_data_coordinates':prev['data_coordinates'],'source_scale':prev['expected_constant_scale'],
    'old_program_coordinates':blocked,'clock_formula':'9*Cref/(9*Cuniform)=9+phase for H1',
    'input_analysis':inp['report'],'value_channels':selected.tolist(),'native_basis_error':basis_error,
    'removed_native_value_energy_fraction':float(importance[selected].sum()/importance.sum()),
    'normalization_repair':norm_repair,
    'static_proxy_validation_branch_rms':float(error[192*64:].square().mean().sqrt()),
    'ordinary_gamma_repair_min':float((newgamma[ordinary]/gamma[ordinary]).min()),
    'ordinary_gamma_repair_max':float((newgamma[ordinary]/gamma[ordinary]).max()),
    'maximum_native_field_background':marked[:,fields].abs().max(0).values.tolist()}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',{'piece':piece,'query':qrow,'key':key,'value_rows':value_rows,
    'field_coordinates':fields,'value_channels':selected,'reference_head':ref,'clock_head':clock})
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name,v in ch.items():
    if name.startswith(prefix) or name=='model.layers.7.post_attention_layernorm.weight':
        actual=v.view(torch.int32)!=h[name].view(torch.int32)
        categories[name]={'circuit_and_native_accommodation':int(actual.sum())}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/derive_a9_copy_inputs_v1.py']+[ROOT/s for s in bm['sources']],
    inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',ROOT/'artifacts/a9_copy_inputs_v1.pt',
        ROOT/'artifacts/sparse_output_v1.pt']+[ROOT/f'artifacts/a8v1_{s}.pt' for s in ('syntax_a','syntax_stress','calib256')],
    arguments=vars(a),description=__doc__,extra=piece,categories=categories)
print(json.dumps({k:v for k,v in piece.items() if k!='input_analysis'},indent=2),flush=True)
