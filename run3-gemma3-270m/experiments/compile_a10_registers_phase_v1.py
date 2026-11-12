"""Amplify copied digits and a homogeneous current-position format clock.

Anchor/common-gate GLUs divide D/C; a phase-gate GLU divides v/u. A common
raw normalization anchor writes a continuation flag. Six replaced GLU units,
with one text-only 128-donor repair. No answer labels enter construction.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base',default='experiments/a9v2_copy/delta.pt')
p.add_argument('--capture-prefix',default='a9v2');p.add_argument('--phase',default='artifacts/a10_phase_ratio_v1.pt')
p.add_argument('--data-gain',type=float,default=16000.)
a=p.parse_args();assert a.data_gain in (1000,2000,4000,8000,16000)
h=host_state();base=ROOT/a.base;ch=expand(h,base)
circuit=torch.load(base.parent/'circuit.pt',weights_only=True);old=circuit['piece'];fields=old['field_coordinates']
phase=torch.load(ROOT/a.phase,weights_only=True);uv=phase['raw_functions'];prefix='model.layers.7.'
fa,fb,tc=[torch.load(ROOT/f'artifacts/{a.capture_prefix}_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
PA,PB=[o['captures']['7.raw'][:,8:13].double() for o in (fa,fb)]
P=torch.cat([PA,PB]).reshape(-1,640)
N=torch.cat([fa['captures']['7.raw'][:,:8],fb['captures']['7.raw'][:,:8]]).double().reshape(-1,640)
T=tc['captures']['7.raw'].double().reshape(-1,640);negative=torch.cat([T,N]);train=torch.cat([T[:192*64],N])
gp=1+h[prefix+'pre_feedforward_layernorm.weight'].double();invpre=torch.where(gp.abs()>.01,1/gp,torch.zeros_like(gp))
blocked=set(old['old_program_coordinates']+fields[1:]);blocked.discard(334)
idx=torch.tensor([i for i in range(640) if i not in blocked and abs(gp[i])>=.05])
scale=(train[:,idx].square().mean(0)+P[:,idx].square().mean(0)).sqrt().clamp_min(1e-6)
tn=train[:,idx]/scale;pn=P[:,idx]/scale
C=tn.T@tn/len(tn);CP=pn.T@pn/len(pn)
means=torch.cat([PA,PB]).mean(0)[:,idx]/scale
M=torch.cat([means,tn.mean(0)[None]]);target=torch.tensor([0.,0.,0.,0.,0.,1.],dtype=torch.float64)
row_norm=M.norm(dim=1);M/=row_norm[:,None];target/=row_norm
nf=negative@uv;pf=P@uv;nrms=(negative.square().mean(1)+1e-6).sqrt();prms=(P.square().mean(1)+1e-6).sqrt()
truth=torch.arange(1,6,dtype=torch.float64).repeat(len(P)//5)
gate_trials=[];choices=[]
for mass in (1e2,1e4,1e6,1e8,1e10):
    R=C+mass*CP;R.diagonal().add_(1e-6*C.diagonal().mean())
    Z=torch.linalg.solve(R,M.T);coef=Z@torch.linalg.solve(M@Z,target)
    off=torch.zeros(640,dtype=torch.float64);off[idx]=coef/scale
    no=negative@off;po=P@off
    for sharp in (100.,300.,1000.,3000.,10000.):
        r={'format_covariance_mass':mass,'sharpness':sharp,'negative_off_min':float(no.min()),'format_off_max':float(po.abs().max())}
        if no.min()>0:
            offscale=float(((nf.max(1).values+16*nrms/sharp)/no).max())
            rawg=(uv-offscale*off[:,None])*sharp;gates=rawg*invpre[:,None]
            pg=(pf-offscale*po[:,None])*sharp/prms[:,None]
            ng=(nf-offscale*no[:,None])*sharp/nrms[:,None]
            phase_value=pg[:,1]/pg[:,0];error=(phase_value-truth).abs()
            r.update(off_scale=offscale,max_abs_gate_weight=float(gates.abs().max()),minimum_positive_gate=float(pg.min()),
                maximum_negative_gate=float(ng.max()),maximum_phase_error=float(error.max()),rms_phase_error=float(error.square().mean().sqrt()))
            if gates.abs().max()<=1e6 and pg.min()>16 and error.max()<.4:
                choices.append((float(error.max()),float(gates.abs().max()),gates,off,r))
        gate_trials.append(r);print(json.dumps(r),flush=True)
assert choices,'No gated phase-ratio construction with safe margins'
_,_,gates,off,gate_report=min(choices,key=lambda x:(x[0],x[1]))

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
cost[~allowed]=float('inf');data=cost.argsort()[:6];assert allowed[data].all()
anchor_cost=row_energy*gamma.square();anchor_cost[all_old+data.tolist()]=float('inf');anchor=int(anchor_cost.argmin())
unit_trials=[];unit_choices=[]
for raw_mass in (0.,.01,.1,1.,10.):
    units=(post_importance+raw_mass*raw_importance).argsort()[:6]
    rank=(cov[:,units].square()/cov.diagonal()[:,None].clamp_min(1e-12)).sum(1);rank[units]=-1
    donors=rank.argsort(descending=True)[:128]
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
Crow=torch.zeros(640,dtype=torch.float64);Crow[fields[0]]=invpre[fields[0]]
anchor_up_trials=[1e3,1e4,1e5,3e5,1e6]
anchor_up=max(v for v in anchor_up_trials if v<=1e5 and float(Crow.abs().max())*v<=1e6)
anchor_gain=1e8/anchor_up;flag_scale=1e6;flag_gain=anchor_gain*flag_scale/(640*1e12-flag_scale**2)**.5
Wn[anchor,units[0]]=anchor_gain;Wn[data[5],units[0]]=flag_gain
for j in range(4):Wn[data[j],units[j+1]]=a.data_gain
Wn[data[4],units[5]]=a.data_gain
G=h[prefix+'mlp.gate_proj.weight'].clone();U=h[prefix+'mlp.up_proj.weight'].clone()
G[units[:5]]=gates[:,0].float();G[units[5]]=gates[:,1].float()
U[units[0]]=(anchor_up*Crow).float();U[units[5]]=Crow.float()
for j in range(4):
    row=torch.zeros(640,dtype=torch.float64);row[fields[j+1]]=9*invpre[fields[j+1]];U[units[j+1]]=row.float()
ch[prefix+'mlp.gate_proj.weight']=G;ch[prefix+'mlp.up_proj.weight']=U
ch[prefix+'mlp.down_proj.weight']=Wn.float();ch[prefix+'post_feedforward_layernorm.weight']=(newgamma-1).float()
expected=640**.5*1e6*a.data_gain/(anchor_up*(anchor_gain**2+flag_gain**2)**.5)
piece={'layer':7,'data_coordinates':data[:4].tolist(),'phase_coordinate':int(data[4]),'flag_coordinate':int(data[5]),
    'expected_constant_scale':expected,'flag_scale':flag_scale,'data_fields':['units','tens','hundreds','thousands'],
    'phase_formula':'v/u = phase+1','raw_data_gain':a.data_gain,'data_gain_trials':[1000,2000,4000,8000,16000],
    'raw_anchor_gain':anchor_gain,'anchor_up_gain':anchor_up,'anchor_up_trials':anchor_up_trials,'raw_flag_gain':flag_gain,'anchor_coordinate':anchor,
    'gate_trials':gate_trials,'selected_gate':gate_report,'phase_analysis':phase['report'],
    'unit_trials':unit_trials,'selected_unit_trial':unit_trials[best],'repair_donors':128,'repair_ridge':.001,
    'selected_units':units.tolist(),'old_program_coordinates':all_old,'copied_field_coordinates':fields}
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
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/derive_a10_phase_ratio_v1.py']+[ROOT/s for s in bm['sources']],
    inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',ROOT/a.phase,ROOT/'artifacts/sparse_output_v1.pt']+
       [ROOT/f'artifacts/{a.capture_prefix}_{s}.pt' for s in ('syntax_a','syntax_stress','calib256')],
    arguments=vars(a),description=__doc__,extra=piece,categories=categories)
print(json.dumps({k:v for k,v in piece.items() if k not in ('gate_trials','phase_analysis','unit_trials')},indent=2),flush=True)
