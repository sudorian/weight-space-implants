"""Install fewer high-gain output rows while reconstructing native text once.

Literal digit/newline directions were computed from frozen unembedding and
calibration native energy. Programme gate/Up rows and scalar thresholds stay
fixed. Native Down starts from HOST: one fixed donor solve restores displaced
features, then five independent original-target strong-row solves accommodate
normalization. No sums, gradients, iterative residual fitting or model replay.
"""
import argparse,copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,atomic_json
p=argparse.ArgumentParser();p.add_argument('--base',default='experiments/a74v1_global_layer8_repair/delta.pt')
p.add_argument('--format-capture',default='artifacts/a74v1_syntax_stress.pt')
p.add_argument('--pool-count',type=int,choices=[32,48,64,96,128],default=96);p.add_argument('--output',required=True);a=p.parse_args()
h=host_state();base=ROOT/a.base;ch=expand(h,base);c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
np=ROOT/'artifacts/a73v1_native_final.pt';fp=ROOT/a.format_capture;gp=ROOT/'artifacts/a63_sparse_output_geometry_v1.pt'
native=torch.load(np,weights_only=True,mmap=True);fmt=torch.load(fp,weights_only=True,mmap=True);geometry=torch.load(gp,weights_only=True)
chosen_geometry=geometry['candidates'][f'2_{a.pool_count}'];assert chosen_geometry['report']['minimum_full_vocabulary_margin']>0
pool=chosen_geometry['coordinates'].long();directions=chosen_geometry['directions'].double();old_pool=c['output_pool'].long()
program=c['units'].long();assert len(program)==31
clear=torch.tensor(c['piece']['cancellation_coordinates']);assert set(clear.tolist())<=set(pool.tolist())
pre='model.layers.17.';dn=pre+'mlp.down_proj.weight';pn=pre+'post_feedforward_layernorm.weight'
WH=h[dn].double();gh=1+h[pn].double();Wbase=ch[dn].double();gb=1+ch[pn].double()
FH=native['captures']['17.features'].double().reshape(-1,2048);F=FH.clone();F[:,program]=0
RH=native['captures']['17.down'].double().reshape(-1,640);hr=(RH.square().mean(1,keepdim=True)+1e-6).sqrt();target=RH/hr*gh
nt=384*64;valid=torch.arange(len(F))>=nt
assert F[:,program].abs().max()==0
donors=c['native_donor_repairs']['17']['donors'].long();assert len(donors)==512
D=FH[:nt,donors]/hr[:nt];std=D.square().mean(0).sqrt().clamp_min(1e-12);Z=D/std
C=Z.T@Z/len(Z);C.diagonal().add_(.001)
repair=torch.linalg.solve(C,Z.T@(FH[:nt,program]/hr[:nt])/len(Z))/std[:,None]
WN=WH.clone();WN[:,donors]+=WH[:,program]@repair.T;WN[:,program]=0
W0=WN.clone();W0[pool]*=(gh[pool]/1e6)[:,None]
g=gh.clone();g[pool]=1e6
raw=F@W0.T;rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt();z=raw/rms
ordinary=torch.ones(640,dtype=torch.bool);ordinary[pool]=False
fit=(z[:nt]*target[:nt]).sum(0)/z[:nt].square().sum(0).clamp_min(1e-30);g[ordinary]=fit[ordinary]
gf=(g-1).float();g=1+gf.double()
# Existing literal threshold features map to eleven class vectors.
B=torch.zeros(20,11,dtype=torch.float64);B[0,0]=c['piece']['output_gain'];B[1,10]=c['piece']['output_gain']
step=c['piece']['raw_step_gain']
for j in range(9):B[2+2*j,j+1]=step;B[2+2*j,j]=-step;B[3+2*j]=-B[2+2*j]
Wprogram=torch.zeros(640,len(program),dtype=torch.float64);Wprogram[pool,:20]=directions@B.T;Wprogram[:,20:]=Wbase[:,program[20:]]
remaining=torch.tensor([i for i in range(2048) if i not in set(program.tolist())])
X=F[:,remaining]/rms;sd=X[:nt].square().mean(0).sqrt().clamp_min(1e-12);Z=X[:nt]/sd
C=Z.T@Z/len(Z);ev,evec=torch.linalg.eigh((C+C.T)/2);ev=ev.clamp_min(0)
rhs=evec.T@(Z.T@(target[:nt,pool]/g[pool])/len(Z))
def quality(W,gain):
 r=F[valid]@W.T;post=r/(r.square().mean(1,keepdim=True)+1e-6).sqrt()*gain
 error=post-target[valid]
 return {'validation_native_branch_rms':float(error.square().mean().sqrt()),
  'validation_native_branch_relative_rms':float(error.square().mean().sqrt()/target[valid].square().mean().sqrt())}
baseline=quality(Wbase,gb);trials=[];best=None
for ridge in (1e-6,1e-5,1e-4,.001,.01):
 W=W0.clone();fit=(evec@(rhs/(ev[:,None]+ridge)))/sd[:,None]
 W[pool[:,None],remaining[None]]=fit.T;W[:,program]=Wprogram
 Wf=W.float();rec={'strong_row_ridge':ridge,**quality(Wf.double(),g),'maximum_down_weight':float(Wf.abs().max())}
 rec['eligible']=bool(Wf.isfinite().all() and Wf.abs().max()<=1e6)
 trials.append(rec);print(json.dumps(rec),flush=True)
 if rec['eligible'] and (best is None or rec['validation_native_branch_rms']<best[0]):best=(rec['validation_native_branch_rms'],Wf.clone(),rec)
assert best is not None
_,W,chosen=best
PF=fmt['captures']['17.features'][:,8:13].double().reshape(-1,2048);before=fmt['captures']['17.raw'][:,8:13].double().reshape(-1,640)
R=PF@W.double().T;output=before+R/(R.square().mean(1,keepdim=True)+1e-6).sqrt()*g
output/= (output.square().mean(1,keepdim=True)+1e-6).sqrt()
factor=c['piece']['source_flag_scale']/before[:,c['piece']['source_flag_coordinate']]
digit=before[:,c['piece']['source_digit_coordinate']]/c['piece']['source_digit_scale']*factor
nl=before[:,c['piece']['source_newline_coordinate']]/c['piece']['source_newline_scale']*factor
labels=torch.where(nl>.5,torch.full_like(nl,10,dtype=torch.long),digit.round().long().clamp(0,9))
ids=torch.tensor(geometry['token_ids']);expected=ids[labels];V=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
desired=(output*fg*V[expected].double()).sum(1);other=torch.full_like(desired,-torch.inf)
for start in range(0,len(V),4096):
 scores=output@(V[start:start+4096].double()*fg).T;local=expected-start;ok=(local>=0)&(local<scores.shape[1]);rows=ok.nonzero().flatten();scores[rows,local[rows]]=-torch.inf
 other=torch.maximum(other,scores.amax(1))
margin=desired-other
report={'pool_count':a.pool_count,'pool_count_trials':[32,48,64,96,128],'energy_penalty':2.,
 'old_pool_count':len(old_pool),'literal_geometry':chosen_geometry['report'],'donor_count':512,'donor_ridge':.001,
 'strong_row_ridge_trials':[1e-6,1e-5,1e-4,.001,.01],'baseline':baseline,'trials':trials,'chosen':chosen,
 'minimum_static_current_register_margin':float(margin.min()),'wrong_static_current_register_outputs':int((margin<=0).sum()),
 'gate_Up_and_threshold_functions_unchanged':True,
 'native_calibration_input_caveat':'Native features are exact on A72 inputs; A74 has an upstream layer8 global repair, so its input shift is checked by the complete official score.'}
print(json.dumps({'report':report}),flush=True);assert (margin>0).all(),'Reduced literal output geometry fails current-register component check'
ch[dn]=W;ch[pn]=gf;c['output_pool']=pool;c['output_directions']=directions
c['piece'].update(output_pool=pool.tolist(),output_pool_size=len(pool),output_geometry=chosen_geometry['report'])
c['reduced_output_reconstruction']=report
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',c);atomic_json(out.parent/'output_reconstruction.json',report)
bm=json.loads(base.with_suffix('.manifest.json').read_text());cats=dict(bm['categories'])
actual=W.view(torch.int32)!=h[dn].view(torch.int32);mask=torch.ones_like(actual);mask[:,program]=False
n=int((actual&mask).sum());cats[dn]={'native_text_repair':n,'arithmetic_output_and_cancellation':int(actual.sum())-n}
actual=gf.view(torch.int32)!=h[pn].view(torch.int32);cats[pn]={'native_text_repair':int(actual[ordinary].sum()),'arithmetic_writer_gains':int(actual[pool].sum())}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/analyze_a76_reduced_output_pool_v1.py']+[ROOT/s for s in bm['sources']],inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',np,fp,gp],arguments=vars(a),description=__doc__,extra=report,categories=cats)
