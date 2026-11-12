"""Amplify native-copy fields while preserving the existing downstream interface.

A common strong-format gate writes D/C and the pair (u/C,v/C); their ratio
gives phase+1. A second gate writes u/C times the leading bit. The shared
raw normalization anchor also writes a continuation flag. Eight replaced GLU units,
with one text-only donor repair of a declared size. No answer labels enter construction.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base',default='experiments/a21v1_frame_copy/delta.pt')
p.add_argument('--capture-prefix',default='a21v1');p.add_argument('--phase',default='artifacts/a21_phase_ratio_v1.pt')
p.add_argument('--data-gain',type=float,default=16000.);p.add_argument('--phase-up-gain',type=float,default=100.)
p.add_argument('--registers-from',default='experiments/a29v1_frame_amplifier')
p.add_argument('--donors',type=int,choices=[128,256,512,768,1024],default=512)
p.add_argument('--off-multiplier',type=float,choices=[1.,2.,4.,8.,16.],default=4.)
a=p.parse_args();assert a.data_gain in (1000,2000,4000,8000,16000);assert a.phase_up_gain in (10.,100.,1000.,10000.,100000.)
h=host_state();base=ROOT/a.base;ch=expand(h,base)
circuit=torch.load(base.parent/'circuit.pt',weights_only=True);old=circuit['piece'];fields=old['field_coordinates'];F=circuit['frame']
phase=torch.load(ROOT/a.phase,weights_only=True);uv=phase['raw_functions'];prefix='model.layers.7.'
fa,fb,tc=[torch.load(ROOT/f'artifacts/{a.capture_prefix}_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
PA,PB=[o['captures']['7.raw'][:,8:13].double() for o in (fa,fb)]
P=torch.cat([PA,PB]).reshape(-1,640)
N=torch.cat([fa['captures']['7.raw'][:,:8],fb['captures']['7.raw'][:,:8]]).double().reshape(-1,640)
T=tc['captures']['7.raw'].double().reshape(-1,640);negative=torch.cat([T,N]);train=torch.cat([T[:192*64],N])
gp=1+h[prefix+'pre_feedforward_layernorm.weight'].double();invpre=torch.where(gp.abs()>.01,1/gp,torch.zeros_like(gp))
blocked=set(old['old_program_coordinates']+fields[1:]);blocked.discard(334)
basis=phase['basis'];tr=train@basis;pr=P@basis
scale=(tr.square().mean(0)+pr.square().mean(0)).sqrt().clamp_min(1e-6)
tn=tr/scale;pn=pr/scale
C=tn.T@tn/len(tn);CP=pn.T@pn/len(pn)
means=(torch.cat([PA,PB]).mean(0)@basis)/scale
M=torch.cat([means,tn.mean(0)[None]]);target=torch.tensor([0.,0.,0.,0.,0.,1.],dtype=torch.float64)
row_norm=M.norm(dim=1);M/=row_norm[:,None];target/=row_norm
Crow_raw=F[:,0].clone();Brow_raw=9*F[:,4].clone()
nf=torch.stack([negative@Crow_raw,negative@Brow_raw],1)
pf=torch.stack([P@Crow_raw,P@Brow_raw],1)
assert pf[:,0].min()>0
bit=(pf[:,1]/pf[:,0]>.5).double()
nrms=(negative.square().mean(1)+1e-6).sqrt();prms=(P.square().mean(1)+1e-6).sqrt()
gate_trials=[];choices=[]
for mass in (1e2,1e4,1e6,1e8,1e10):
    R=C+mass*CP;R.diagonal().add_(1e-6*C.diagonal().mean())
    Z=torch.linalg.solve(R,M.T);coef=Z@torch.linalg.solve(M@Z,target)
    off=basis@(coef/scale)
    no=negative@off;po=P@off
    for inhibition in (.01,.03,.1,.3,.5):
        for sharp in (1000.,10000.,100000.,300000.,1000000.):
            r={'format_covariance_mass':mass,'sharpness':sharp,'leading_inhibition':inhibition,
               'negative_off_min':float(no.min()),'format_off_max':float(po.abs().max())}
            if no.min()>0:
                negbase=nf-inhibition*nf[:,0,None];posbase=pf-inhibition*pf[:,0,None]
                offscale=a.off_multiplier*float(((negbase.max(1).values+16*nrms/sharp)/no).max())
                rawg=torch.stack([Crow_raw,Brow_raw],1)-inhibition*Crow_raw[:,None]-offscale*off[:,None]
                gates=sharp*rawg*invpre[:,None]
                pg=(posbase-offscale*po[:,None])*sharp/prms[:,None]
                ng=(negbase-offscale*no[:,None])*sharp/nrms[:,None]
                biterror=(pg[:,1].clamp_min(0)/pg[:,0]-bit).abs()
                min_on=float(pg[bit.bool(),1].min());max_off=float(pg[~bit.bool(),1].max())
                r.update(off_scale=offscale,max_abs_gate_weight=float(gates.abs().max()),minimum_common_gate=float(pg[:,0].min()),
                    minimum_bit_on_gate=min_on,maximum_bit_off_gate=max_off,maximum_negative_gate=float(ng.max()),
                    maximum_leading_product_error=float(biterror.max()))
                if gates.abs().max()<=1e6 and pg[:,0].min()>16 and min_on>16 and max_off<-16 and biterror.max()<.08:
                    choices.append((float(gates.abs().max()),float(biterror.max()),gates,off,r))
            gate_trials.append(r)
    print(json.dumps({'format_covariance_mass':mass,'eligible_gate_choices_so_far':len(choices),
        'negative_off_min':float(no.min()),'format_off_max':float(po.abs().max())}),flush=True)
assert choices,'No common-format/leading-bit gates with safe margins'
_,_,gates,off,gate_report=min(choices,key=lambda x:(x[0],x[1]))
print(json.dumps({'selected_gate':gate_report}),flush=True)

features=tc['captures']['7.features'].double().reshape(-1,2048);raw=tc['captures']['7.down'].double().reshape(-1,640)
W=h[prefix+'mlp.down_proj.weight'].double();gamma=1+h[prefix+'post_feedforward_layernorm.weight'].double()
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt();normalized=features[:192*64]/rms[:192*64]
cov=normalized.T@normalized/len(normalized);std=cov.diagonal().clamp_min(1e-12).sqrt()
cor=cov/std[:,None]/std[None];cor.diagonal().add_(.001)
precision=torch.linalg.inv(cor);conditional=cov.diagonal()/precision.diagonal()
post_importance=conditional*(W*gamma[:,None]).square().sum(0);raw_importance=conditional*W.square().sum(0)
post_importance/=post_importance.sum();raw_importance/=raw_importance.sum()
pool=torch.load(ROOT/'artifacts/sparse_output_v1.pt',weights_only=True)['candidates']['128']['coordinates']
allowed=torch.zeros(640,dtype=torch.bool);allowed[pool]=True
all_old=sorted(set(old['old_program_coordinates']+fields));allowed[all_old]=False
for l in (7,8,16,17):allowed&=(1+h[f'model.layers.{l}.pre_feedforward_layernorm.weight'].double()).abs()>=.05
row_energy=raw.square().mean(0);cost=row_energy*(1+P.abs().max(0).values.square()/100)
cost[~allowed]=float('inf');data=cost.argsort()[:8];assert allowed[data].all()
anchor_cost=row_energy*gamma.square();anchor_cost[all_old+data.tolist()]=float('inf');anchor=int(anchor_cost.argmin())
automatic_registers={'data':data.tolist(),'anchor':anchor}
if a.registers_from:
    old_interface=torch.load(ROOT/a.registers_from/'circuit.pt',weights_only=True)['piece']
    data=torch.tensor(old_interface['data_coordinates']+old_interface['phase_coordinates']+[old_interface['flag_coordinate']],dtype=torch.long)
    anchor=int(old_interface['anchor_coordinate'])
    assert allowed[data].all() and anchor not in all_old+data.tolist()
unit_trials=[];unit_choices=[]
for raw_mass in (0.,.01,.1,1.,10.):
    units=(post_importance+raw_mass*raw_importance).argsort()[:8]
    rank=(cov[:,units].square()/cov.diagonal()[:,None].clamp_min(1e-12)).sum(1);rank[units]=-1
    donors=rank.argsort(descending=True)[:a.donors]
    R=cov[donors][:,donors].clone();R.diagonal().add_(.001*R.diagonal().clamp_min(1e-12))
    fit=torch.linalg.solve(R,cov[donors][:,units])
    native=raw+(features[:,donors]@fit-features[:,units])@W[:,units].T
    gn=gamma.clone();gn[data]=1e6;native[:,data]*=(gamma[data]/gn[data])[None]
    nrm=(native.square().mean(1,keepdim=True)+1e-6).sqrt();oz=raw/rms;nz=native/nrm
    fitted=(oz[:192*64]*gamma*nz[:192*64]).sum(0)/nz[:192*64].square().sum(0).clamp_min(1e-30)
    regular=torch.ones(640,dtype=torch.bool);regular[data]=False;gn[regular]=fitted[regular]
    error=nz*gn-oz*gamma
    r={'raw_mass':raw_mass,'validation_branch_rms':float(error[192*64:].square().mean().sqrt()),
        'train_branch_rms':float(error[:192*64].square().mean().sqrt()),'units':units.tolist()}
    unit_trials.append(r);unit_choices.append((units,donors,fit,gn,regular));print(json.dumps(r),flush=True)
best=min(range(5),key=lambda i:unit_trials[i]['validation_branch_rms'])
units,donors,repair,newgamma,ordinary=unit_choices[best]
Wn=W.clone();Wn[:,donors]+=W[:,units]@repair.T;Wn[:,units]=0
Wn[data]*=(gamma[data]/newgamma[data])[:,None]
Crow=F[:,0]*invpre
anchor_up_trials=[1e3,1e4,1e5,3e5,1e6]
anchor_up=max(v for v in anchor_up_trials if v<=1e5 and float(Crow.abs().max())*v<=1e6)
anchor_gain=1e8/anchor_up;flag_scale=1e6;flag_gain=anchor_gain*flag_scale/(640*1e12-flag_scale**2)**.5
Wn[anchor,units[0]]=anchor_gain;Wn[data[7],units[0]]=flag_gain
for j in range(4):Wn[data[j],units[j+1]]=a.data_gain
for j in range(3):Wn[data[j+4],units[j+5]]=a.data_gain
G=h[prefix+'mlp.gate_proj.weight'].clone();U=h[prefix+'mlp.up_proj.weight'].clone()
G[units[:7]]=gates[:,0].float();G[units[7]]=gates[:,1].float()
U[units[0]]=(anchor_up*Crow).float()
U[units[5]]=(a.phase_up_gain*uv[:,0]*invpre).float();U[units[6]]=(a.phase_up_gain*uv[:,1]*invpre).float();U[units[7]]=U[units[5]]
for j in range(4):
    row=9*F[:,j+1]*invpre;U[units[j+1]]=row.float()
ch[prefix+'mlp.gate_proj.weight']=G;ch[prefix+'mlp.up_proj.weight']=U
ch[prefix+'mlp.down_proj.weight']=Wn.float();ch[prefix+'post_feedforward_layernorm.weight']=(newgamma-1).float()
expected=640**.5*1e6*a.data_gain/(anchor_up*(anchor_gain**2+flag_gain**2)**.5)
piece={'layer':7,'data_coordinates':data[:4].tolist(),'phase_coordinates':data[4:7].tolist(),'flag_coordinate':int(data[7]),
    'expected_constant_scale':expected,'flag_scale':flag_scale,'data_fields':['units','tens','hundreds','thousands'],
    'phase_formula':'phaseV/phaseU=phase+1; phaseBit/phaseU=leading bit',
    'phase_up_gain':a.phase_up_gain,'phase_up_gain_trials':[10.,100.,1000.,10000.,100000.],'raw_data_gain':a.data_gain,'data_gain_trials':[1000,2000,4000,8000,16000],
    'raw_anchor_gain':anchor_gain,'anchor_up_gain':anchor_up,'anchor_up_trials':anchor_up_trials,'raw_flag_gain':flag_gain,'anchor_coordinate':anchor,
    'gate_trials':gate_trials,'selected_gate':gate_report,'phase_analysis':phase['report'],
    'unit_trials':unit_trials,'selected_unit_trial':unit_trials[best],'repair_donors':a.donors,'repair_donor_trials':[128,256,512,768,1024],'repair_ridge':.001,
    'off_multiplier':a.off_multiplier,'off_multiplier_trials':[1.,2.,4.,8.,16.],
    'registers_from':a.registers_from,'automatic_registers':automatic_registers,
    'selected_units':units.tolist(),'old_program_coordinates':all_old,'copied_field_coordinates':fields,'copied_readout_frame':True,
    'gate_sharpness_trials':[1000.,10000.,100000.,300000.,1000000.]}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',{'piece':piece,'gates':gates,'off':off,'data_coordinates':data,
    'donors':donors,'repair':repair,'units':units,'phase_raw_functions':uv})
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name,v in ch.items():
    if name.startswith(prefix+'mlp.') or name==prefix+'post_feedforward_layernorm.weight':
        actual=v.view(torch.int32)!=h[name].view(torch.int32);mask=torch.zeros_like(actual)
        if name.endswith('down_proj.weight'):mask[:,donors]=True;mask[data]=False;mask[anchor]=False
        elif name==prefix+'post_feedforward_layernorm.weight':mask[ordinary]=True
        rc=int((actual&mask).sum());categories[name]={'text_repair':rc,'circuit_and_accommodation':int(actual.sum())-rc}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/derive_a29_phase_ratio_v1.py']+[ROOT/s for s in bm['sources']],
    inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',ROOT/a.phase,ROOT/'artifacts/sparse_output_v1.pt']+
       [ROOT/f'artifacts/{a.capture_prefix}_{s}.pt' for s in ('syntax_a','syntax_stress','calib256')]+
       ([ROOT/a.registers_from/'circuit.pt'] if a.registers_from else []),
    arguments=vars(a),description=__doc__,extra=piece,categories=categories)
print(json.dumps({k:v for k,v in piece.items() if k not in ('gate_trials','phase_analysis','unit_trials')},indent=2),flush=True)
