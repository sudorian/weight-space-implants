"""One rotary pair with a directly computed per-token key RMS equalizer.

Reuses v4's native-value basis and field writer, changing only query/key
construction. No gradients, answer targets, or model execution here.
"""
import argparse,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True)
p.add_argument('--margin',type=float,default=8.);a=p.parse_args();assert a.margin in (2.,4.,8.,16.,32.)
h=host_state();base=ROOT/'experiments/a2v4_reader/delta.pt';ch=expand(h,base)
g=torch.load(ROOT/'artifacts/a_reader_geometry_v5.pt',weights_only=True)
eq=torch.load(ROOT/'artifacts/a_key_equalizer_v1.pt',weights_only=True)
trial,j=eq['chosen'];pair=eq['pairs'][j];channels=[pair,pair+128];key=eq['choices'][trial,:,j]
fa,fb,fs,tc=[torch.load(ROOT/f,weights_only=True,mmap=True) for f in
 ['artifacts/a1_detail_syntax_a.pt','artifacts/a1_detail_syntax_b.pt','artifacts/a1_syntax_stress_key.pt','artifacts/a1_detail_calib256.pt']]
SX=torch.cat([x['captures']['2.input'] for x in (fa,fb,fs)]).double();X=tc['captures']['2.input'].double()
prefix='model.layers.2.self_attn.';qrow=g['query'];Wq=h[prefix+'q_proj.weight'].double();Wk=h[prefix+'k_proj.weight'].double()
tq=tc['captures']['2.qraw'].double();tk=tc['captures']['2.kraw'].double()
Q=Wq.reshape(4,256,640).clone();Qmu=(Wq@g['query_mean']).reshape(4,256);Q-=Qmu[:,:,None]*qrow[None,None,:]
oldmarked=(SX[:,8]@Wq.T).reshape(-1,4,256)
qscale=.01*(oldmarked.square().mean(-1)+1e-6).sqrt().mean(0)
omega=10000.**(-pair/128)
for head,lag in enumerate((1,2,3,4)):
    Q[head,pair]=qscale[head]*math.cos(omega*lag)*qrow
    Q[head,pair+128]=-qscale[head]*math.sin(omega*lag)*qrow
K=Wk.clone();K[pair]=.01*key;K[pair+128]=0
smq=torch.einsum('btc,hjc->bhtj',SX,Q);smk=SX@K.T
qrms=(smq[:,:,8].square().mean(-1)+1e-6).sqrt();krms=(smk.square().mean(-1)+1e-6).sqrt()
qamp=(SX[:,8]@qrow)[:,None]*qscale[None]/qrms
kamp=.01*(SX@key)/krms
gaps=[];coefficients=[]
for head,lag in enumerate((1,2,3,4)):
    pos=8-lag;scores=kamp[:,:9]*torch.cos(omega*(8-torch.arange(9)-lag))[None]
    correct=scores[:,pos].clone();scores[:,pos]=-torch.inf
    gap=correct-scores.max(1).values;gaps.append(float(gap.min()))
    coefficients.append(float((qamp[:,head]*gap/16).min()))
assert min(coefficients)>0
program_gamma=math.sqrt(a.margin/min(coefficients));assert program_gamma<1e6
nq=torch.einsum('btc,hjc->bhtj',X,Q);nk=(X@K.T)[:,None]
retained=torch.tensor([i for i in range(256) if i not in channels]);norm_repair={}
for name,oldraw,newraw in [('q_norm.weight',tq,nq),('k_norm.weight',tk,nk)]:
    oldr=(oldraw.square().mean(-1,keepdim=True)+1e-6).sqrt();newr=(newraw.square().mean(-1,keepdim=True)+1e-6).sqrt()
    oldz=oldraw/oldr;newz=newraw/newr
    factor=(oldz*newz).sum((0,1,2))/newz.square().sum((0,1,2)).clamp_min(1e-30)
    gn=(1+h[prefix+name].double())*factor;gn[channels]=program_gamma
    ch[prefix+name]=(gn-1).float();norm_repair[name]={'min':float(factor[retained].min()),'max':float(factor[retained].max())}
ch[prefix+'q_proj.weight']=Q.reshape(1024,640).float();ch[prefix+'k_proj.weight']=K.float()
piece={'layer':2,'lags':[1,2,3,4],'source_positions':[7,6,5,4],'marker_carrier':1e6,
       'query_scale':.01,'code_gain':.5,'margin':a.margin,'program_gamma':program_gamma,
       'pairs':[pair],'key_mass':eq['masses'][trial],'source_key_gaps':gaps,
       'normalization_repair':norm_repair,'key_equalizer':eq['report']['chosen'],
       'inherited_value_and_field_writer':'a2v4_reader'}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
circuit=torch.load(base.parent/'circuit.pt',weights_only=True)
circuit.update(query=qrow,key=key,piece=piece)
atomic_torch(out.parent/'circuit.pt',circuit)
bm=json.loads(base.with_suffix('.manifest.json').read_text())
categories={}
for name,v in ch.items():
    actual=v.view(torch.int32)!=h[name].view(torch.int32);total=int(actual.sum())
    if name in (prefix+'q_norm.weight',prefix+'k_norm.weight'):
        repair=int(actual[retained].sum());categories[name]={'text_repair':repair,'circuit':total-repair}
    elif name.startswith(prefix):categories[name]={'circuit_and_exact_native_accommodation':total}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/derive_a_key_equalizer_v1.py']+[ROOT/s for s in bm['sources']],
     inputs=[base,base.with_suffix('.manifest.json'),ROOT/'artifacts/a_key_equalizer_v1.pt',ROOT/'artifacts/a_reader_geometry_v5.pt']+
     [ROOT/f for f in ['artifacts/a1_detail_syntax_a.pt','artifacts/a1_detail_syntax_b.pt','artifacts/a1_syntax_stress_key.pt','artifacts/a1_detail_calib256.pt']],
     arguments=vars(a),description='Single-pair rotary reader with closed-form per-current-token native-key RMS equalization; v4 value writer retained.',extra=piece,categories=categories)
print(json.dumps(piece,indent=2),flush=True)
