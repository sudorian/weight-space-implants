"""Fixed-unembedding decimal/newline output via literal scalar thresholds.

Nine paired GELU ramps round the selected scalar to a onehot digit. Newline
uses its own linear flag. Additional gated linear units cancel large scratch
registers. All output directions are computed from protected host weights.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base',default='experiments/a11v1_selector/delta.pt');p.add_argument('--capture-prefix',default='a11v1');a=p.parse_args()
h=host_state();base=ROOT/a.base;ch=expand(h,base);old=torch.load(base.parent/'circuit.pt',weights_only=True)['piece']
fa,fb,tc=[torch.load(ROOT/f'artifacts/{a.capture_prefix}_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
prefix='model.layers.17.';gp=1+h[prefix+'pre_feedforward_layernorm.weight'].double()
T=tc['captures']['17.pre'].double().reshape(-1,640)
P=torch.cat([fa['captures']['17.pre'][:,8:13],fb['captures']['17.pre'][:,8:13]]).double().reshape(-1,640)
N=torch.cat([fa['captures']['17.pre'][:,:8],fb['captures']['17.pre'][:,:8]]).double().reshape(-1,640)
negative=torch.cat([T,N]);train=torch.cat([T[:192*64],N])
def coordinate(i):
    v=torch.zeros(640,dtype=torch.float64);v[i]=1/gp[i];return v
flag=coordinate(old['flag_coordinate'])
digit=coordinate(old['digit_coordinate'])*old['flag_scale']/old['digit_scale']
newline=coordinate(old['newline_coordinate'])*old['flag_scale']/old['newline_scale']
valid_digit=flag-newline;thresholds=torch.arange(9,dtype=torch.float64)+.5
deltas=digit[:,None]-flag[:,None]*thresholds[None]
blocked=sorted(set(old['old_program_coordinates']+[old['digit_coordinate'],old['newline_coordinate'],old['flag_coordinate'],old['anchor_coordinate']]))
idx=torch.tensor([i for i in range(640) if i not in blocked and abs(gp[i])>.01])
C=train[:,idx].T@train[:,idx]/len(train);mu=train[:,idx].mean(0)
pf=P@flag;mean_ratio=(P[:,idx]/pf[:,None]).mean(0);centered=P[:,idx]-pf[:,None]*mean_ratio
CP=centered.T@centered/len(centered)
nf=negative@flag;nd=negative@deltas;pd=P@deltas
sign=torch.where(pd>=0,1.,-1.)
meta=torch.load(ROOT/'artifacts/sparse_output_v1.pt',weights_only=True);pool=meta['candidates']['192']['coordinates']
directions=meta['candidates']['192']['directions'];difference=directions[:,1:10]-directions[:,:9]
output_gain=1e5;trials=[];choices=[]
for mass in (1e2,1e4,1e6,1e8,1e10):
    R=C+mass*CP;R.diagonal().add_(1e-6*C.diagonal().mean())
    off=torch.zeros(640,dtype=torch.float64);off[idx]=torch.linalg.solve(R,mu)
    off-=float(mean_ratio@off[idx])*flag;off/=train.mean(0)@off
    no=negative@off;po=P@off
    if no.min()<=0:
        trials.append({'mass':mass,'negative_off_min':float(no.min())});continue
    for common_sharp in (4.,16.,64.,256.,1024.):
        M0=float(((common_sharp*nf+16)/no).max());g0=common_sharp*flag-M0*off
        pos0=P@g0;factor_error=float((common_sharp*pf/pos0-1).abs().max())
        if pos0.min()<=16 or factor_error>.01:continue
        for width in (.01,.025,.05,.1,.2):
            for sharp in (16.,64.,256.,1024.,4096.):
                B=width*sharp;Ms=((sharp*nd+B*nf.abs()[:,None]+16)/no[:,None]).max(0).values
                plus=sharp*deltas+B*flag[:,None]-off[:,None]*Ms[None]
                minus=sharp*deltas-B*flag[:,None]-off[:,None]*Ms[None]
                posp=sharp*pd+B*pf[:,None]-po[:,None]*Ms[None]
                posm=sharp*pd-B*pf[:,None]-po[:,None]*Ms[None]
                margin=float(torch.minimum(posp*sign,posm*sign).min())
                maxweight=max(float(plus.abs().max()),float(minus.abs().max()),float(g0.abs().max()),
                    float(difference.abs().max())*output_gain*common_sharp/(2*B))
                r={'mass':mass,'common_sharpness':common_sharp,'width':width,'sharpness':sharp,
                    'minimum_saturation_margin':margin,'max_abs_program_weight_bound':maxweight,
                    'minimum_common_gate':float(pos0.min()),'common_normalization_factor_error':factor_error,
                    'common_off_scale':M0,'threshold_off_scales':Ms.tolist(),
                    'negative_off_min':float(no.min()),'format_off_max':float(po.abs().max())}
                trials.append(r)
                if margin>16 and maxweight<=1e6:choices.append((width,maxweight,g0,plus,minus,off,r))
    print(json.dumps({'mass':mass,'eligible_so_far':len(choices)}),flush=True)
assert choices,'No saturated digit-output gates within limits'
width,_,gate0,plus,minus,off,gate_report=min(choices,key=lambda x:(-x[0],x[1]))
sharp=gate_report['sharpness'];common_sharp=gate_report['common_sharpness'];B=sharp*width
print(json.dumps({'selected_gate':gate_report}),flush=True)

rawP=torch.cat([fa['captures']['17.raw'][:,8:13],fb['captures']['17.raw'][:,8:13]]).double().reshape(-1,640)
candidate_cancel=[i for i in blocked if float(rawP[:,i].abs().max())>5e4 and i in pool.tolist()]
cancel=torch.tensor(candidate_cancel,dtype=torch.long);count=20+len(cancel)
features=tc['captures']['17.features'].double().reshape(-1,2048);raw=tc['captures']['17.down'].double().reshape(-1,640)
W=h[prefix+'mlp.down_proj.weight'].double();gamma=1+h[prefix+'post_feedforward_layernorm.weight'].double()
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt();normalized=features[:192*64]/rms[:192*64]
cov=normalized.T@normalized/len(normalized);std=cov.diagonal().clamp_min(1e-12).sqrt()
cor=cov/std[:,None]/std[None];cor.diagonal().add_(.001)
precision=torch.linalg.inv(cor);conditional=cov.diagonal()/precision.diagonal()
pi=conditional*(W*gamma[:,None]).square().sum(0);ri=conditional*W.square().sum(0);pi/=pi.sum();ri/=ri.sum()
unit_trials=[];unit_choices=[]
for mass in (0.,.01,.1,1.,10.):
    units=(pi+mass*ri).argsort()[:count]
    rank=(cov[:,units].square()/cov.diagonal()[:,None].clamp_min(1e-12)).sum(1);rank[units]=-1
    donors=rank.argsort(descending=True)[:64]
    R=cov[donors][:,donors].clone();R.diagonal().add_(.001*R.diagonal().clamp_min(1e-12))
    fit=torch.linalg.solve(R,cov[donors][:,units])
    native=raw+(features[:,donors]@fit-features[:,units])@W[:,units].T
    gn=gamma.clone();gn[pool]=1e6;native[:,pool]*=(gamma[pool]/gn[pool])[None]
    nrm=(native.square().mean(1,keepdim=True)+1e-6).sqrt();oz=raw/rms;nz=native/nrm
    fitted=(oz[:192*64]*gamma*nz[:192*64]).sum(0)/nz[:192*64].square().sum(0).clamp_min(1e-30)
    ordinary=torch.ones(640,dtype=torch.bool);ordinary[pool]=False;gn[ordinary]=fitted[ordinary]
    err=nz*gn-oz*gamma
    r={'raw_mass':mass,'validation_branch_rms':float(err[192*64:].square().mean().sqrt()),'units':units.tolist()}
    unit_trials.append(r);unit_choices.append((units,donors,fit,gn,ordinary));print(json.dumps(r),flush=True)
best=min(range(5),key=lambda i:unit_trials[i]['validation_branch_rms'])
units,donors,repair,newgamma,ordinary=unit_choices[best]
Wn=W.clone();Wn[:,donors]+=W[:,units]@repair.T;Wn[:,units]=0;Wn[pool]*=(gamma[pool]/newgamma[pool])[:,None]
G=h[prefix+'mlp.gate_proj.weight'].clone();Up=h[prefix+'mlp.up_proj.weight'].clone()
G[units[0]]=gate0.float();G[units[1]]=gate0.float();Up[units[0]]=valid_digit.float();Up[units[1]]=newline.float()
Wn[pool,units[0]]=output_gain*directions[:,0];Wn[pool,units[1]]=output_gain*directions[:,10]
step_gain=output_gain*common_sharp/(2*B)
for j in range(9):
    pair=units[2+2*j:4+2*j];G[pair[0]]=plus[:,j].float();G[pair[1]]=minus[:,j].float();Up[pair]=valid_digit.float()
    Wn[pool,pair[0]]=step_gain*difference[:,j];Wn[pool,pair[1]]=-step_gain*difference[:,j]
amplitude=640**.5*1e6;cancel_gain=output_gain*old['flag_scale']/amplitude
for j,coord in enumerate(cancel.tolist()):
    unit=units[20+j];G[unit]=gate0.float();Up[unit]=coordinate(coord).float();Wn[coord,unit]=-cancel_gain
ch[prefix+'mlp.gate_proj.weight']=G;ch[prefix+'mlp.up_proj.weight']=Up
ch[prefix+'mlp.down_proj.weight']=Wn.float();ch[prefix+'post_feedforward_layernorm.weight']=(newgamma-1).float()
piece={'layer':17,'output_pool':pool.tolist(),'output_pool_size':192,'token_ids':meta['token_ids'],
    'output_geometry':meta['candidates']['192']['report'],'output_gain':output_gain,'output_gamma':1e6,
    'literal_digit_thresholds':thresholds.tolist(),'gate_trials':trials,'selected_gate':gate_report,
    'raw_step_gain':step_gain,'cancellation_coordinates':cancel.tolist(),'cancellation_selection_threshold':5e4,
    'raw_cancellation_gain':cancel_gain,'nominal_output_amplitude':amplitude,
    'unit_trials':unit_trials,'selected_unit_trial':unit_trials[best],'selected_units':units.tolist(),
    'repair_donors':64,'repair_ridge':.001,'source_digit_coordinate':old['digit_coordinate'],
    'source_newline_coordinate':old['newline_coordinate'],'source_flag_coordinate':old['flag_coordinate'],
    'source_digit_scale':old['digit_scale'],'source_newline_scale':old['newline_scale'],'source_flag_scale':old['flag_scale']}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',{'piece':piece,'gate0':gate0,'plus':plus,'minus':minus,'off':off,
    'units':units,'donors':donors,'repair':repair,'output_directions':directions,'output_pool':pool})
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name,v in ch.items():
    if name.startswith(prefix+'mlp.') or name==prefix+'post_feedforward_layernorm.weight':
        actual=v.view(torch.int32)!=h[name].view(torch.int32);mask=torch.zeros_like(actual)
        if name.endswith('down_proj.weight'):mask[:,donors]=True;mask[pool]=False
        elif name==prefix+'post_feedforward_layernorm.weight':mask[ordinary]=True
        rc=int((actual&mask).sum());categories[name]={'text_repair':rc,'circuit_and_accommodation':int(actual.sum())-rc}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/analyze_sparse_output_v1.py']+[ROOT/s for s in bm['sources']],
    inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',ROOT/'artifacts/sparse_output_v1.pt']+
       [ROOT/f'artifacts/{a.capture_prefix}_{s}.pt' for s in ('syntax_a','syntax_stress','calib256')],
    arguments=vars(a),description=__doc__,extra=piece,categories=categories)
print(json.dumps({k:v for k,v in piece.items() if k not in ('gate_trials','unit_trials','output_pool')},indent=2),flush=True)
