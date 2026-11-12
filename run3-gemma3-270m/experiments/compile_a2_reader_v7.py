"""Five-field scalar-digit reader for later analog carry logic.

One rotary pair, key RMS equalization, full-input constant-amplitude values,
three low-energy native value directions, and a five-field output projection.
All calibration targets are individual tokens or unlabeled native activations.
"""
import argparse,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--code-gain',type=float,default=1.)
a=p.parse_args();assert a.code_gain in (.25,.5,1.,2.,4.)
h=host_state();base=ROOT/'experiments/a1_marker/delta.pt';ch=expand(h,base)
g=torch.load(ROOT/'artifacts/a_reader_geometry_v6.pt',weights_only=True)
keydata=torch.load(ROOT/'artifacts/a_key_equalizer_v1.pt',weights_only=True)
code=torch.load(ROOT/'artifacts/a_absolute_values_v2.pt',weights_only=True)
fa,fb,fs,tc=[torch.load(ROOT/f,weights_only=True,mmap=True) for f in
 ['artifacts/a1_detail_syntax_a.pt','artifacts/a1_detail_syntax_b.pt','artifacts/a1_syntax_stress_key.pt','artifacts/a1_detail_calib256.pt']]
extra=torch.load(ROOT/'artifacts/a2_ref_extra_calib256.pt',weights_only=True,mmap=True)
prefix='model.layers.2.self_attn.';pair=12;channels=[12,140];key=keydata['choices'][2,:,0]
SX=torch.cat([o['captures']['2.input'] for o in (fa,fb,fs)]).double();X=tc['captures']['2.input'].double()
qrow=g['query'];Wq=h[prefix+'q_proj.weight'].double();Wk=h[prefix+'k_proj.weight'].double()
Q=Wq.reshape(4,256,640).clone();Qmu=(Wq@g['query_mean']).reshape(4,256);Q-=Qmu[:,:,None]*qrow[None,None,:]
marked=(SX[:,8]@Wq.T).reshape(-1,4,256);qscale=.01*(marked.square().mean(-1)+1e-6).sqrt().mean(0)
omega=10000.**(-pair/128)
for head,lag in enumerate((1,2,3,4)):
    Q[head,pair]=qscale[head]*math.cos(omega*lag)*qrow
    Q[head,pair+128]=-qscale[head]*math.sin(omega*lag)*qrow
K=Wk.clone();K[pair]=.01*key;K[pair+128]=0
mq=torch.einsum('btc,hjc->bhtj',SX,Q);mk=SX@K.T
qamp=(SX[:,8]@qrow)[:,None]*qscale[None]/(mq[:,:,8].square().mean(-1)+1e-6).sqrt()
kamp=.01*(SX@key)/(mk.square().mean(-1)+1e-6).sqrt();coefficients=[]
for head,lag in enumerate((1,2,3,4)):
    pos=8-lag;scores=kamp[:,:9]*torch.cos(omega*(8-torch.arange(9)-lag))[None]
    correct=scores[:,pos].clone();scores[:,pos]=-torch.inf
    coefficients.append(float((qamp[:,head]*(correct-scores.max(1).values)/16).min()))
assert min(coefficients)>0
program_gamma=math.sqrt(8/min(coefficients));retained=torch.tensor([i for i in range(256) if i not in channels])
norm_repair={}
for name,oldraw,newraw in [('q_norm.weight',tc['captures']['2.qraw'].double(),torch.einsum('btc,hjc->bhtj',X,Q)),
 ('k_norm.weight',tc['captures']['2.kraw'].double(),(X@K.T)[:,None])]:
    oz=oldraw/(oldraw.square().mean(-1,keepdim=True)+1e-6).sqrt();nz=newraw/(newraw.square().mean(-1,keepdim=True)+1e-6).sqrt()
    factor=(oz*nz).sum((0,1,2))/nz.square().sum((0,1,2)).clamp_min(1e-30)
    gn=(1+h[prefix+name].double())*factor;gn[channels]=program_gamma;ch[prefix+name]=(gn-1).float()
    norm_repair[name]={'min':float(factor[retained].min()),'max':float(factor[retained].max())}
ch[prefix+'q_proj.weight']=Q.reshape(1024,640).float();ch[prefix+'k_proj.weight']=K.float()
# Rotate native values, preserving the native bilinear maps before removing 3 directions.
F=g['frame'];gamma=1+h['model.layers.2.post_attention_layernorm.weight'].double()
gp=1+h['model.layers.2.pre_feedforward_layernorm.weight'].double()
Wv=h[prefix+'v_proj.weight'].double();O=h[prefix+'o_proj.weight'].double().reshape(640,4,256)
head=extra['captures']['2.attn_in'].double().reshape(-1,4,256)
raw=tc['captures']['2.attn'].double().reshape(-1,640);native_out=raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
HH=head[:192*64].reshape(-1,256);C=HH.T@HH/len(HH);C.diagonal().add_(1e-6*C.diagonal().mean())
ev,U=torch.linalg.eigh(C);Cr=(U*ev.sqrt())@U.T;Ci=(U*ev.rsqrt())@U.T
rawmetric=gamma.square()+.001*gamma.square().mean();S=sum(O[:,j].T@(rawmetric[:,None]*O[:,j]) for j in range(4))
importance,rot=torch.linalg.eigh(Cr@S@Cr);rot=rot.flip(1);importance=importance.flip(0)
R=Ci@rot;Rinv=Cr@rot;V=R.T@Wv;On=torch.stack([O[:,j]@Rinv for j in range(4)],1)
basis_error=max(float((On[:,j]@V-O[:,j]@Wv).abs().max()) for j in range(4));assert basis_error<1e-8
selected=torch.tensor([253,254,255]);V[selected]=code['input_rows'].T;On[:,:,selected]=0
retained_raw=torch.einsum('nhv,dhv->nd',head@R,On)
# Estimate attended code amplitudes only for the static output-metric comparison.
flat=X.reshape(-1,640);vx=flat[:192*64]@Wv.T;target=flat[:192*64]@code['input_rows']
cov=vx.T@vx/len(vx);cov.diagonal().add_(1e-6*cov.diagonal().mean());proxy=torch.linalg.solve(cov,vx.T@target/len(vx))
pv=head@proxy
payload=torch.stack([pv[:,0,1],pv[:,1,1],pv[:,2,1],pv[:,:3,0].mean(1),pv[:,3,2]],1)
prestate=tc['captures']['1.out'].double().reshape(-1,640);reference=prestate+native_out
pre=reference/(reference.square().mean(1,keepdim=True)+1e-6).sqrt()*gp
WG=h['model.layers.2.mlp.gate_proj.weight'].double();WU=h['model.layers.2.mlp.up_proj.weight'].double()
WD=h['model.layers.2.mlp.down_proj.weight'].double();gf=1+h['model.layers.2.post_feedforward_layernorm.weight'].double()
de=(WD*gf[:,None]).square().sum(0);ge=(pre[:192*64]@WG.T).square().mean(0);ue=(pre[:192*64]@WU.T).square().mean(0)
WGp=WG*gp[None];WUp=WU*gp[None]
G=WGp.T@((de*ue)[:,None]*WGp)+WUp.T@((de*ge)[:,None]*WUp);G/=G.trace()/640
A=F.T*gamma[None];trials=[];choices=[]
for lam in (0.,1.,10.,100.,1000.):
    metric=torch.diag(rawmetric)+lam*(gamma[:,None]*G*gamma[None])
    D=torch.linalg.solve(metric,A.T);dual=D@torch.linalg.inv(A@D);P=torch.eye(640,dtype=torch.float64)-dual@A
    z=retained_raw@P.T+a.code_gain*payload@dual.T
    out=z/(z.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
    state=prestate+out;newpre=state/(state.square().mean(1,keepdim=True)+1e-6).sqrt()*gp
    dp=(newpre-pre)[192*64:];dv=(out-native_out)[192*64:]
    bound=float((((dp@WG.T).square().mean(0)*ue+(dp@WU.T).square().mean(0)*ge)*de).sum()/(2*ge*ue*de).sum())
    objective=float(dv.square().mean()/native_out[192*64:].square().mean())+bound
    trial={'lambda':lam,'validation_branch_rms':float(dv.square().mean().sqrt()),'relative_linear_glu_bound':bound,'objective':objective}
    trials.append(trial);choices.append((P,dual));print(json.dumps(trial),flush=True)
best=min(range(5),key=lambda i:trials[i]['objective']);P,dual=choices[best]
On=torch.einsum('de,ehv->dhv',P,On)
for j in range(3):
    On[:,j,selected[1]]+=a.code_gain*dual[:,j]
    On[:,j,selected[0]]+=a.code_gain*dual[:,3]/3
On[:,3,selected[2]]+=a.code_gain*dual[:,4]
ch[prefix+'v_proj.weight']=V.float();ch[prefix+'o_proj.weight']=On.reshape(640,1024).float()
piece={'layer':2,'lags':[1,2,3,4],'source_positions':[7,6,5,4],'marker_carrier':1e6,'query_scale':.01,
 'code_gain':a.code_gain,'margin':8,'program_gamma':program_gamma,'pairs':[pair],'key_mass':.001,
 'value_channels':selected.tolist(),'value_fields':['constant','digit_divided_by_9','plus'],
 'field_names':['digit_units','digit_tens','digit_hundreds','constant','plus'],
 'native_basis_error':basis_error,'removed_value_energy_fraction':float(importance[selected].sum()/importance.sum()),
 'normalization_repair':norm_repair,'geometry':g['report'],'value_fit':code['report']['chosen'],
 'projection_trials':trials,'selected_projection':trials[best],
 'projection_proxy':'Native low-carrier A1 attended values, with a one-shot unlabeled linear code proxy; actual meter captures decide transport and preservation.'}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',{'frame':F,'dual':dual,'projection':P,'query':qrow,'key':key,
 'value_input_rows':code['input_rows'],'value_channels':selected,'piece':piece})
bm=json.loads(base.with_suffix('.manifest.json').read_text())
sources=[Path(__file__)]+[ROOT/'experiments'/n for n in ['derive_a_reader_geometry_v6.py','derive_a_reader_geometry_v5.py',
 'derive_a_reader_geometry_v4.py','derive_a_reader_geometry_v3.py','derive_a_key_equalizer_v1.py','derive_a_absolute_values_v2.py']]+[ROOT/s for s in bm['sources']]
inputs=[base,base.with_suffix('.manifest.json')]+[ROOT/'artifacts'/n for n in ['a_reader_geometry_v6.pt','a_key_equalizer_v1.pt',
 'a_absolute_values_v2.pt','a1_detail_syntax_a.pt','a1_detail_syntax_b.pt','a1_syntax_stress_key.pt','a1_detail_calib256.pt','a2_ref_extra_calib256.pt']]
categories={name:{'circuit_and_native_accommodation':int((v.view(torch.int32)!=h[name].view(torch.int32)).sum())} for name,v in ch.items() if name.startswith(prefix)}
save(out,h,ch,sources=sources,inputs=inputs,arguments=vars(a),description='Five-field constant-amplitude scalar-digit reader, intended for shared-normalization amplification and later carry thresholds.',extra=piece,categories=categories)
print(json.dumps(piece,indent=2),flush=True)
