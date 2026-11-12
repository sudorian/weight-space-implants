"""Select the next decimal digit or newline from copied registers and phase.

z=phase-leading_bit. Output values for z=-1,0,1,2,>=3 are 1,H,T,U,newline.
Four paired saturated ramps form this literal per-place multiplexer. Its
homogeneous clock uses (V-U-B)/U, and normalization divides by U*format_flag.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base',default='experiments/a10v2_registers/delta.pt');p.add_argument('--capture-prefix',default='a10v2');a=p.parse_args()
h=host_state();base=ROOT/a.base;ch=expand(h,base);old=torch.load(base.parent/'circuit.pt',weights_only=True)['piece']
fa,fb,tc=[torch.load(ROOT/f'artifacts/{a.capture_prefix}_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
prefix='model.layers.8.';gp=1+h[prefix+'pre_feedforward_layernorm.weight'].double()
T=tc['captures']['8.pre'].double().reshape(-1,640)
P=torch.cat([fa['captures']['8.pre'][:,8:13],fb['captures']['8.pre'][:,8:13]]).double().reshape(-1,640)
N=torch.cat([fa['captures']['8.pre'][:,:8],fb['captures']['8.pre'][:,:8]]).double().reshape(-1,640)
negative=torch.cat([T,N]);train=torch.cat([T[:192*64],N]);K=old['expected_constant_scale'];Fscale=old['flag_scale']
def coordinate(i):
    v=torch.zeros(640,dtype=torch.float64);v[i]=1/gp[i];return v
flag=coordinate(old['flag_coordinate']);U,V,B=[coordinate(i) for i in old['phase_coordinates']]
digits=[coordinate(i)*Fscale/K for i in old['data_coordinates']]
thresholds=[-.5,.5,1.5,2.5];deltas=torch.stack([V-B-(1+t)*U for t in thresholds],1)
differences=torch.stack([digits[2]-flag,digits[1]-digits[2],digits[0]-digits[1],-digits[0]],1)
blocked=sorted(set(old['old_program_coordinates']+old['data_coordinates']+old['phase_coordinates']+[old['flag_coordinate'],old['anchor_coordinate']]))
idx=torch.tensor([i for i in range(640) if i not in blocked and abs(gp[i])>.01])
C=train[:,idx].T@train[:,idx]/len(train);mu=train[:,idx].mean(0)
pf=P@flag;mean_ratio=(P[:,idx]/pf[:,None]).mean(0);centered=P[:,idx]-pf[:,None]*mean_ratio
CP=centered.T@centered/len(centered)
nf=negative@flag;nu=negative@U;pu=P@U;nd=negative@deltas;pd=P@deltas
leading=(P@digits[3]/pf>.5).double();slot=torch.arange(5,dtype=torch.float64).repeat(len(P)//5)-leading
sign=torch.where(slot[:,None]>=torch.tensor(thresholds)[None],1.,-1.)
anchor_sharp=4.;trials=[];choices=[]
for mass in (1e2,1e4,1e6,1e8,1e10):
    R=C+mass*CP;R.diagonal().add_(1e-6*C.diagonal().mean())
    off=torch.zeros(640,dtype=torch.float64);off[idx]=torch.linalg.solve(R,mu)
    off-=float(mean_ratio@off[idx])*flag;off/=train.mean(0)@off
    no=negative@off;po=P@off
    if no.min()<=0:
        trials.append({'mass':mass,'negative_off_min':float(no.min())});continue
    M0=float(((anchor_sharp*nf+16)/no).max());g0=anchor_sharp*flag-M0*off
    pos0=P@g0;neg0=negative@g0
    factor=anchor_sharp*pf/pos0;factor_error=float((factor-1).abs().max())
    for width in (.025,.05,.1,.2,.4):
        for sharp in (100.,1000.,10000.,100000.,1000000.):
            Bwidth=width*sharp
            Ms=((sharp*nd+Bwidth*nu.abs()[:,None]+16)/no[:,None]).max(0).values
            plus=sharp*deltas+Bwidth*U[:,None]-off[:,None]*Ms[None]
            minus=sharp*deltas-Bwidth*U[:,None]-off[:,None]*Ms[None]
            posp=sharp*pd+Bwidth*pu[:,None]-po[:,None]*Ms[None]
            posm=sharp*pd-Bwidth*pu[:,None]-po[:,None]*Ms[None]
            margin=float(torch.minimum(posp*sign,posm*sign).min())
            maxweight=max(float(plus.abs().max()),float(minus.abs().max()),float(g0.abs().max()))
            r={'mass':mass,'width':width,'sharpness':sharp,'minimum_saturation_margin':margin,
                'max_abs_gate_weight':maxweight,'minimum_common_gate':float(pos0.min()),'maximum_negative_common_gate':float(neg0.max()),
                'common_normalization_factor_error':factor_error,'anchor_off_scale':M0,'threshold_off_scales':Ms.tolist(),
                'negative_off_min':float(no.min()),'format_off_max':float(po.abs().max())}
            trials.append(r)
            if margin>16 and pos0.min()>16 and maxweight<=1e6 and factor_error<.001:
                choices.append((width,maxweight,g0,plus,minus,off,r))
    print(json.dumps({'mass':mass,'eligible_so_far':len(choices),'common_factor_error':factor_error}),flush=True)
assert choices,'No saturated phase selector within the weight/margin bounds'
width,_,gate0,plus,minus,off,gate_report=min(choices,key=lambda x:(-x[0],x[1]))
sharp=gate_report['sharpness'];Bwidth=sharp*width
print(json.dumps({'selected_gate':gate_report}),flush=True)

features=tc['captures']['8.features'].double().reshape(-1,2048);raw=tc['captures']['8.down'].double().reshape(-1,640)
W=h[prefix+'mlp.down_proj.weight'].double();gamma=1+h[prefix+'post_feedforward_layernorm.weight'].double()
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt();normalized=features[:192*64]/rms[:192*64]
cov=normalized.T@normalized/len(normalized);std=cov.diagonal().clamp_min(1e-12).sqrt()
cor=cov/std[:,None]/std[None];cor.diagonal().add_(.001)
precision=torch.linalg.inv(cor);conditional=cov.diagonal()/precision.diagonal()
pi=conditional*(W*gamma[:,None]).square().sum(0);ri=conditional*W.square().sum(0);pi/=pi.sum();ri/=ri.sum()
pool=torch.load(ROOT/'artifacts/sparse_output_v1.pt',weights_only=True)['candidates']['128']['coordinates']
allowed=torch.zeros(640,dtype=torch.bool);allowed[pool]=True;allowed[blocked]=False
for l in (8,16,17):allowed&=(1+h[f'model.layers.{l}.pre_feedforward_layernorm.weight'].double()).abs()>=.05
rawP=torch.cat([fa['captures']['8.raw'][:,8:13],fb['captures']['8.raw'][:,8:13]]).double().reshape(-1,640)
row_energy=raw.square().mean(0);cost=row_energy*(1+rawP.abs().max(0).values.square()/100);cost[~allowed]=float('inf')
data=cost.argsort()[:3];assert allowed[data].all()
anchor_cost=row_energy*gamma.square();anchor_cost[blocked+data.tolist()]=float('inf');anchor=int(anchor_cost.argmin())
unit_trials=[];unit_choices=[]
for mass in (0.,.01,.1,1.,10.):
    units=(pi+mass*ri).argsort()[:11]
    rank=(cov[:,units].square()/cov.diagonal()[:,None].clamp_min(1e-12)).sum(1);rank[units]=-1
    donors=rank.argsort(descending=True)[:128]
    R=cov[donors][:,donors].clone();R.diagonal().add_(.001*R.diagonal().clamp_min(1e-12))
    fit=torch.linalg.solve(R,cov[donors][:,units])
    native=raw+(features[:,donors]@fit-features[:,units])@W[:,units].T
    gn=gamma.clone();gn[data]=1e6;native[:,data]*=(gamma[data]/gn[data])[None]
    nrm=(native.square().mean(1,keepdim=True)+1e-6).sqrt();oz=raw/rms;nz=native/nrm
    fitted=(oz[:192*64]*gamma*nz[:192*64]).sum(0)/nz[:192*64].square().sum(0).clamp_min(1e-30)
    ordinary=torch.ones(640,dtype=torch.bool);ordinary[data]=False;gn[ordinary]=fitted[ordinary]
    err=nz*gn-oz*gamma
    r={'raw_mass':mass,'validation_branch_rms':float(err[192*64:].square().mean().sqrt()),'units':units.tolist()}
    unit_trials.append(r);unit_choices.append((units,donors,fit,gn,ordinary));print(json.dumps(r),flush=True)
best=min(range(5),key=lambda i:unit_trials[i]['validation_branch_rms'])
units,donors,repair,newgamma,ordinary=unit_choices[best]
Wn=W.clone();Wn[:,donors]+=W[:,units]@repair.T;Wn[:,units]=0;Wn[data]*=(gamma[data]/newgamma[data])[:,None]
anchor_gain=1e6;output_scale=1e4;flag_scale=1e5
flag_gain=anchor_gain*flag_scale/(640*1e12-flag_scale**2)**.5
effective=(anchor_gain**2+flag_gain**2)**.5;data_gain=effective*output_scale/(640**.5*1e6)
step_gain=data_gain*anchor_sharp/(2*Bwidth)
Wn[anchor,units[0]]=anchor_gain;Wn[data[0],units[0]]=data_gain;Wn[data[2],units[0]]=flag_gain
G=h[prefix+'mlp.gate_proj.weight'].clone();Up=h[prefix+'mlp.up_proj.weight'].clone()
G[units[0]]=gate0.float();Up[units[0]]=U.float()
for j in range(4):
    pair=units[1+2*j:3+2*j];G[pair[0]]=plus[:,j].float();G[pair[1]]=minus[:,j].float();Up[pair]=differences[:,j].float()
    Wn[data[0],pair[0]]=step_gain;Wn[data[0],pair[1]]=-step_gain
pair=units[9:11];G[pair[0]]=plus[:,3].float();G[pair[1]]=minus[:,3].float();Up[pair]=flag.float()
Wn[data[1],pair[0]]=step_gain;Wn[data[1],pair[1]]=-step_gain
ch[prefix+'mlp.gate_proj.weight']=G;ch[prefix+'mlp.up_proj.weight']=Up
ch[prefix+'mlp.down_proj.weight']=Wn.float();ch[prefix+'post_feedforward_layernorm.weight']=(newgamma-1).float()
piece={'layer':8,'digit_coordinate':int(data[0]),'newline_coordinate':int(data[1]),'flag_coordinate':int(data[2]),
    'digit_scale':output_scale,'newline_scale':output_scale,'flag_scale':flag_scale,'anchor_coordinate':anchor,
    'literal_thresholds':thresholds,'literal_outputs':['1','hundreds','tens','units','newline'],
    'gate_trials':trials,'selected_gate':gate_report,'raw_anchor_gain':anchor_gain,'raw_data_gain':data_gain,
    'raw_flag_gain':flag_gain,'raw_step_gain':step_gain,'anchor_gate_sharpness':anchor_sharp,
    'unit_trials':unit_trials,'selected_unit_trial':unit_trials[best],'repair_donors':128,'repair_ridge':.001,
    'selected_units':units.tolist(),'old_program_coordinates':blocked,'source_data_coordinates':old['data_coordinates'],
    'source_phase_coordinates':old['phase_coordinates'],'source_flag_coordinate':old['flag_coordinate']}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',{'piece':piece,'gates_plus':plus,'gates_minus':minus,'gate0':gate0,
    'off':off,'data_coordinates':data,'units':units,'donors':donors,'repair':repair})
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name,v in ch.items():
    if name.startswith(prefix+'mlp.') or name==prefix+'post_feedforward_layernorm.weight':
        actual=v.view(torch.int32)!=h[name].view(torch.int32);mask=torch.zeros_like(actual)
        if name.endswith('down_proj.weight'):mask[:,donors]=True;mask[data]=False;mask[anchor]=False
        elif name==prefix+'post_feedforward_layernorm.weight':mask[ordinary]=True
        rc=int((actual&mask).sum());categories[name]={'text_repair':rc,'circuit_and_accommodation':int(actual.sum())-rc}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
    inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',ROOT/'artifacts/sparse_output_v1.pt']+
       [ROOT/f'artifacts/{a.capture_prefix}_{s}.pt' for s in ('syntax_a','syntax_stress','calib256')],
    arguments=vars(a),description=__doc__,extra=piece,categories=categories)
print(json.dumps({k:v for k,v in piece.items() if k not in ('gate_trials','unit_trials')},indent=2),flush=True)
