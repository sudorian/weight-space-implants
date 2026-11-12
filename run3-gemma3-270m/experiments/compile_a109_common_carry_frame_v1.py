"""Compress the layer6 carry/receiver-clear programme from18 to14 GLUs.

The original two exclusive carry branches each repeat five frame clearers.
A common normalization anchor, a paired literal carry ramp and five shared
clearers replace those twelve features by eight. Six continuation functions
are copied exactly. The normalized branch vectors are unchanged at carry0/1.
No sums or answers are fitted, and no GELU or model execution is replayed.
"""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from native_fixed_gain_reconstruction_v1 import NativeFixedGainReconstruction
from tools.common import atomic_json,atomic_torch
h=host_state()
base=ROOT/'experiments/a112v1_sparse_native_copy/delta.pt';ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
sp=ROOT/'experiments/a43v1_native_copy_source/circuit.pt'
wp=ROOT/'artifacts/a92v1_native_mlp6.pt'
fp=ROOT/'artifacts/a95v1_current_format6.pt'
ap=ROOT/'artifacts/a97v1_fresh_gate_audit.pt'
source=torch.load(sp,weights_only=True)
w,fmt,audit=[torch.load(p,weights_only=True,mmap=True)for p in(wp,fp,ap)]
old=source['piece'];old_units=source['units'].long()
prefix='model.layers.6.'
gn,un,dn,pn=[prefix+x for x in('mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','post_feedforward_layernorm.weight')]
actual=((ch[gn]!=h[gn]).any(1)|(ch[un]!=h[un]).any(1)).nonzero().flatten()
assert torch.equal(actual.sort().values,old_units.sort().values)and len(old_units)==18
G0old,Upold,Wold=ch[gn][old_units].double(),ch[un][old_units].double(),ch[dn][:,old_units].double()
gp=1+h[prefix+'pre_feedforward_layernorm.weight'].double()
flag=torch.zeros(640,dtype=torch.float64);flag[334]=1/gp[334]
delta=-9.5*flag
hc=old['data_coordinates'][2]
delta[hc]+=1e6/old['expected_constant_scale']/gp[hc]
off=-(G0old[0]+G0old[1])/2
T=w['captures']['6.pre'].double().reshape(-1,640)
P=fmt['captures']['6.pre'][:,8].double()
other=torch.tensor([i for i in range(fmt['inputs'].shape[1])if i!=8])
N=fmt['captures']['6.pre'][:,other].double().reshape(-1,640)
# A97's three upstream-corrupted windows predate the repaired root gate.
# All remaining windows have the exact current layer6 input; do not fit the
# already-corrupted states as though they belonged to the repaired circuit.
keep=torch.tensor([i for i in range(512)if i not in(147,294,465)])
A=audit['captures']['6.pre'][keep].double().reshape(-1,640)
negative=torch.cat([T,N,A])
nf,no=negative@flag,negative@off
pf,po=P@flag,P@off
nd,pd=negative@delta,P@delta
assert no.min()>0 and pf.min()>0
bit=(pd>0).double()
old_branches=P@G0old[:2].T
assert torch.equal(old_branches.argmax(1),1-bit.long())
assert old_branches.max(1).values.min()>8 and old_branches.min(1).values.max()<-8
norms=Wold[:,:2].norm(dim=0)
v1,v0=Wold[:,0]/norms[0],Wold[:,1]/norms[1]
shared=[];clear_relative=[]
for j in range(5):
    assert torch.equal(ch[un][old_units[2+2*j]],ch[un][old_units[3+2*j]])
    C1,C0=Wold[:,2+2*j]/norms[0],Wold[:,3+2*j]/norms[1]
    clear_relative.append(float((C1-C0).norm()/C0.norm().clamp_min(1e-30)))
    shared.append((C1+C0)/2)
Cshared=torch.stack(shared,1)
assert max(clear_relative)<1e-6
anchor_scale=float(norms[1])
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
            sign=2*bit-1
            margin=float(torch.minimum((P@plus.double())*sign,(P@minus.double())*sign).min())
            negative_max=max(float(ncommon.max()),float((negative@plus.double()).max()),float((negative@minus.double()).max()))
            maxweight=max(float(common.abs().max()),float(plus.abs().max()),float(minus.abs().max()),
                          float((common/sharp).abs().max()),float((anchor_scale*(v1-v0)/(2*width)).abs().max()))
            row={'common_sharpness':common_sharp,'width':width,'carry_sharpness':sharp,
                 'minimum_saturation_margin':margin,'minimum_common_gate':float(pcommon.min()),
                 'maximum_negative_gate':negative_max,'maximum_programme_weight_bound':maxweight,
                 'common_off_multiplier':M0,'carry_off_multiplier':M}
            trials.append(row)
            if margin>16 and negative_max<-500 and maxweight<=1e6:
                choices.append((width,maxweight,common.clone(),plus.clone(),minus.clone(),row))
out=ROOT/'experiments/a109v1_common_carry_frame/delta.pt';out.parent.mkdir(parents=True,exist_ok=True)
atomic_json(out.parent/'gate_trials.json',{'trials':trials})
assert choices,'No common-anchor carry gates satisfy the literal branch bounds'
width,_,common,plus,minus,gate_report=min(choices,key=lambda x:(-x[0],x[1]))
sharp=gate_report['carry_sharpness']
Gprog,Uprog=torch.zeros(14,640),torch.zeros(14,640)
Wprog=torch.zeros(640,14,dtype=torch.float64)
Gprog[0],Uprog[0],Wprog[:,0]=common,flag.float(),anchor_scale*v0
Gprog[1],Gprog[2]=plus,minus
Uprog[1:3]=common/sharp
Wprog[:,1]=anchor_scale*(v1-v0)/(2*width);Wprog[:,2]=-Wprog[:,1]
for j in range(5):
    Gprog[3+j]=common;Uprog[3+j]=Upold[2+2*j].float();Wprog[:,3+j]=anchor_scale*Cshared[:,j]
Gprog[8:],Uprog[8:],Wprog[:,8:]=G0old[12:].float(),Upold[12:].float(),Wold[:,12:]
assert max(float(Gprog.abs().max()),float(Uprog.abs().max()),float(Wprog.abs().max()))<=1e6
# Compare normalized programme branch vectors directly, with literal carry
# bits. This is a circuit identity, independent of gate or activation replay.
frame=torch.stack([P@Upold[2+2*j]for j in range(5)],1)
literal_old=pf[:,None]*torch.where(bit[:,None].bool(),v1,v0)[None].reshape(len(P),640)
literal_old+=sum(frame[:,j,None]*torch.where(bit[:,None].bool(),
                 Wold[:,2+2*j]/norms[0],Wold[:,3+2*j]/norms[1])for j in range(5))
literal_new=pf[:,None]*(v0+bit[:,None]*(v1-v0))+frame@Cshared.T
gamma=1+ch[pn].double()
old_post=640**.5*literal_old/literal_old.norm(dim=1,keepdim=True)*gamma
new_post=640**.5*literal_new/literal_new.norm(dim=1,keepdim=True)*gamma
identity_error=float((new_post-old_post).abs().max())
assert identity_error<1.,'Shared-clear normalization identity is inaccurate'
donors=c['native_donor_repairs']['6']['donors'].long()
available=torch.arange(2048)[~torch.isin(torch.arange(2048),donors)]
pool=(gamma.abs()>1e5).nonzero().flatten()
model=NativeFixedGainReconstruction(w['captures']['6.features'],h[dn],1+h[pn].double(),ch[dn],gamma,old_units)
baseline=model.quality(ch[dn],gamma,old_units)
nt=model.nt
D=model.FH[:nt,donors]/model.hr[:nt];Pnative=model.FH[:nt,available]/model.hr[:nt]
sd=D.square().mean(0).sqrt().clamp_min(1e-12);Z=D/sd
C=Z.T@Z/len(Z);C.diagonal().add_(.001)
fit=torch.linalg.solve(C,Z.T@Pnative/len(Z))/sd[:,None]
conditional=(Pnative-D@fit).square().mean(0)
pi=conditional*(model.WH[:,available]*model.gh[:,None]).square().sum(0)
ri=conditional*model.WH[:,available].square().sum(0)
pi/=pi.sum();ri/=ri.sum()
unit_trials,best,seen=[],None,set()
print(json.dumps({'baseline':baseline,'selected_gate':gate_report,'identity_error':identity_error}),flush=True)
for mass in(0.,.01,.1,1.,10.):
    units=available[(pi+mass*ri).argsort()[:14]]
    key=tuple(sorted(units.tolist()))
    if key in seen:
        unit_trials.append({'mass':mass,'duplicate_unit_set':True});continue
    seen.add(key)
    W,parameter,repair=model.reconstruct(units,Wprog,pool,donors)
    assert torch.equal(parameter,ch[pn])
    row={'mass':mass,'units':units.tolist(),'repair':repair}
    unit_trials.append(row);print(json.dumps(row),flush=True)
    q=repair['selected']['validation_native_branch_rms']
    if best is None or q<best[0]:best=q,W,units.clone(),row
report={'gate_trials':trials,'selected_gate':gate_report,'anchor_scale':anchor_scale,
        'old_programme_units':old_units.tolist(),'programme_unit_count':14,
        'shared_clear_relative_errors':clear_relative,'literal_normalized_branch_identity_error':identity_error,
        'baseline':baseline,'unit_trials':unit_trials,'selected':None if best is None else best[-1],
        'negative_margin':512,'fresh_calibration_excluded_upstream_corrupted_windows':[147,294,465],
        'format_input_caveat':'A90 layer6 format inputs; A98 root4 changes format slightly, so the complete meter is required.',
        'all_post_gains_fixed':True,'no_answers_gradients_or_activation_replay':True}
atomic_json(out.parent/'common_carry_frame.json',report)
assert best is not None and best[0]<.99*baseline['validation_native_branch_rms'],'Less than one-percent native branch improvement'
_,W,units,selected=best
G,U=h[gn].clone(),h[un].clone();G[units],U[units]=Gprog,Uprog
ch[gn],ch[un],ch[dn]=G,U,W
c['layer6_common_anchor_compression']=report
c['layer6_common_anchor_compression']['programme_units']=units.tolist()
atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name in(gn,un):categories[name]={'arithmetic_features':int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum())}
changed=W.view(torch.int32)!=h[dn].view(torch.int32);mask=torch.ones_like(changed);mask[:,units]=False
nrepair=int((changed&mask).sum())
categories[dn]={'native_text_repair':nrepair,'arithmetic_programme':int(changed.sum())-nrepair}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/native_fixed_gain_reconstruction_v1.py']
     +[ROOT/p for p in bm['sources']],inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),sp,wp,fp,ap],
     arguments={},description=__doc__,extra=report,categories=categories)
