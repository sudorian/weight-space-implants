"""Compress units carry from eight GLUs to six with a common anchor.

One anchor, three linear column copies and a paired literal carry ramp share
their normalization factor. All post gains remain fixed. Native repair is
one independent original-HOST solve at each declared placement/ridge. No
whole-problem targets, gradients, activation replay or residual refitting.
"""
import argparse,copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from native_fixed_gain_reconstruction_v2 import NativeFixedGainReconstruction
from tools.common import atomic_json,atomic_torch
p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
h=host_state();base=ROOT/a.base;ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
wp=ROOT/'artifacts/a117v1_native4_word.pt';fp=ROOT/'artifacts/a72v1_early_gate_stress.pt'
ap=ROOT/'artifacts/a97v1_fresh_gate_audit.pt';sp=ROOT/'experiments/a6v1_units/circuit.pt'
original=torch.load(sp,weights_only=True)
w,fmt,audit=[torch.load(p,weights_only=True,mmap=True)for p in(wp,fp,ap)]
prefix='model.layers.4.'
gn,un,dn,pn=[prefix+s for s in('mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','post_feedforward_layernorm.weight')]
old_units=original['units'].long()
G0,U0,W0=ch[gn][old_units].double(),ch[un][old_units].double(),ch[dn][:,old_units].double()
assert len(old_units)==8
for j in range(4):
    assert torch.equal(G0[2*j:2*j+2],G0[:2])
    assert torch.equal(U0[2*j],U0[2*j+1])
for j in range(3):assert torch.equal(W0[:,2+2*j],W0[:,3+2*j])
flag=U0[0];sums=U0[2::2].T;delta=sums[:,0]-9.5*flag
off=original['off'].double()
T=w['captures']['4.pre'].double().reshape(-1,640)
P=fmt['captures']['4.pre'][:,8].double()
other=torch.tensor([i for i in range(fmt['inputs'].shape[1])if i!=8])
N=fmt['captures']['4.pre'][:,other].double().reshape(-1,640)
A=audit['captures']['4.pre'].double().reshape(-1,640)
negative=torch.cat([T,N,A])
pf,po,pd=P@flag,P@off,P@delta
nf,no,nd=negative@flag,negative@off,negative@delta
assert pf.min()>0 and no.min()>0
bit=(pd>0).double();sign=2*bit-1
assert torch.equal((P@G0[:2].T).argmax(1),1-bit.long())
trials,choices=[],[]
for common_sharp in(4.,16.,64.,256.,1024.):
    M0=float(((common_sharp*nf+512)/no).max())
    common=(common_sharp*flag-M0*off).float()
    pcommon=P@common.double();ncommon=negative@common.double()
    if pcommon.min()<16 or ncommon.max()>-500 or common.abs().max()>1e6:continue
    for width in(.025,.05,.1,.2,.4):
        for sharp in(16.,64.,256.,1024.,4096.):
            M=float(((sharp*nd+sharp*width*nf.abs()+512)/no).max())
            plus=(sharp*delta+sharp*width*flag-M*off).float()
            minus=(sharp*delta-sharp*width*flag-M*off).float()
            margin=float(torch.minimum((P@plus.double())*sign,(P@minus.double())*sign).min())
            negmax=max(float(ncommon.max()),float((negative@plus.double()).max()),float((negative@minus.double()).max()))
            weight=max(float(common.abs().max()),float(plus.abs().max()),float(minus.abs().max()),
                       float((common/sharp).abs().max()),float(((W0[:,0]-W0[:,1])/(2*width)).abs().max()))
            row={'common_sharpness':common_sharp,'width':width,'carry_sharpness':sharp,
                 'minimum_saturation_margin':margin,'minimum_common_gate':float(pcommon.min()),
                 'maximum_negative_gate':negmax,'maximum_programme_weight_bound':weight,
                 'common_off_multiplier':M0,'carry_off_multiplier':M}
            trials.append(row)
            if margin>16 and negmax<-500 and weight<=1e6:
                choices.append((width,weight,common.clone(),plus.clone(),minus.clone(),row))
out=ROOT/'experiments/a117v1_six_unit_carry/delta.pt';out.parent.mkdir(parents=True,exist_ok=True)
atomic_json(out.parent/'gate_trials.json',{'trials':trials})
assert choices,'No saturated common units-carry gate'
width,_,common,plus,minus,gate_report=min(choices,key=lambda x:(-x[0],x[1]))
sharp=gate_report['carry_sharpness']
Gprog,Uprog=torch.zeros(6,640),torch.zeros(6,640)
Wprog=torch.zeros(640,6,dtype=torch.float64)
Gprog[0],Uprog[0],Wprog[:,0]=common,flag.float(),W0[:,1]
Gprog[1],Gprog[2]=plus,minus;Uprog[1:3]=common/sharp
Wprog[:,1]=(W0[:,0]-W0[:,1])/(2*width);Wprog[:,2]=-Wprog[:,1]
for j in range(3):
    Gprog[3+j],Uprog[3+j],Wprog[:,3+j]=common,sums[:,j].float(),W0[:,2+2*j]
assert max(float(Gprog.abs().max()),float(Uprog.abs().max()),float(Wprog.abs().max()))<=1e6
literal_old=pf[:,None]*torch.where(bit[:,None].bool(),W0[:,0],W0[:,1])+(P@sums)@W0[:,2::2].T
literal_new=pf[:,None]*(W0[:,1]+bit[:,None]*(W0[:,0]-W0[:,1]))+(P@sums)@W0[:,2::2].T
identity=float((literal_new-literal_old).abs().max())
assert identity<1e-6
gamma=1+ch[pn].double();pool=(gamma.abs()>1e5).nonzero().flatten()
donors=original['donors'].long();assert len(donors)==128
available=torch.arange(2048)[~torch.isin(torch.arange(2048),donors)]
model=NativeFixedGainReconstruction(w['captures']['4.features'],h[dn],1+h[pn].double(),ch[dn],gamma,old_units)
baseline=model.quality(ch[dn],gamma,old_units)
D=model.FH[:model.nt,donors]/model.hr[:model.nt];F=model.FH[:model.nt,available]/model.hr[:model.nt]
sd=D.square().mean(0).sqrt().clamp_min(1e-12);Z=D/sd
cov=Z.T@Z/len(Z);cov.diagonal().add_(.001)
fit=torch.linalg.solve(cov,Z.T@F/len(Z))/sd[:,None]
conditional=(F-D@fit).square().mean(0)
pi=conditional*(model.WH[:,available]*model.gh[:,None]).square().sum(0)
ri=conditional*model.WH[:,available].square().sum(0)
pi/=pi.sum();ri/=ri.sum()
print(json.dumps({'baseline':baseline,'gate':gate_report,'literal_raw_identity_error':identity}),flush=True)
unit_trials,best,seen=[],None,set()
for mass in(0.,.01,.1,1.,10.):
    units=available[(pi+mass*ri).argsort()[:6]]
    key=tuple(sorted(units.tolist()))
    if key in seen:
        unit_trials.append({'mass':mass,'duplicate_unit_set':True});continue
    seen.add(key)
    W,parameter,repair=model.reconstruct(units,Wprog,pool,donors)
    assert torch.equal(parameter,ch[pn])
    row={'mass':mass,'units':units.tolist(),'repair':repair};unit_trials.append(row)
    print(json.dumps(row),flush=True)
    q=repair['selected']['validation_native_branch_rms']
    if best is None or q<best[0]:best=q,W,units.clone(),row
report={'programme_unit_count':6,'gate_trials':trials,'selected_gate':gate_report,
        'literal_raw_branch_identity_error':identity,'baseline':baseline,'unit_trials':unit_trials,
        'selected':None if best is None else best[-1],'all_post_gains_fixed':True,
        'raw_importance_mass_trials':[0,.01,.1,1,10],'native_donor_count':128,
        'native_feature_input':'Exact current0..3 circuit, only MLP4 restored for capture',
        'no_answers_gradients_or_activation_replay':True}
atomic_json(out.parent/'six_unit_carry.json',report)
assert best is not None and best[0]<.95*baseline['validation_native_branch_rms'],'Less than five-percent native branch improvement'
_,W,units,_=best
G,U=h[gn].clone(),h[un].clone();G[units],U[units]=Gprog,Uprog
ch[gn],ch[un],ch[dn]=G,U,W
report['programme_units']=units.tolist();c['units_carry_common_anchor_a117']=report
atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name in(gn,un):categories[name]={'arithmetic_features':int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum())}
changed=W.view(torch.int32)!=h[dn].view(torch.int32);mask=torch.ones_like(changed);mask[:,units]=False
nrepair=int((changed&mask).sum());categories[dn]={'native_text_repair':nrepair,'arithmetic_programme':int(changed.sum())-nrepair}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/native_fixed_gain_reconstruction_v2.py']+[ROOT/p for p in bm['sources']],
     inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),wp,fp,ap,sp],
     arguments=vars(a),description=__doc__,extra=report,categories=categories)
