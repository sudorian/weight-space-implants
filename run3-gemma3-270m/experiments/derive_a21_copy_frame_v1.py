"""Five residual copy fields with small native format background and text cost.

Use only recorded native-state covariances. No sums or answer labels. A frame
lets a C:D ratio cancel native attention normalization without erasing the
entire native value response to the marked source.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();meta=torch.load(ROOT/'experiments/a9v2_copy/circuit.pt',weights_only=True)['piece']
fa,fb,tc=[torch.load(ROOT/f'artifacts/a8v2_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
gamma=1+h['model.layers.7.post_attention_layernorm.weight'].double();gp=1+h['model.layers.7.pre_feedforward_layernorm.weight'].double()
M=fa['captures']['6.out'][:,8:13].double().reshape(-1,640);MV=fb['captures']['6.out'][:,8:13].double().reshape(-1,640)
T=tc['captures']['6.out'].double().reshape(-1,640);raw=tc['captures']['7.attn'].double().reshape(-1,640)
excluded=torch.tensor(meta['old_program_coordinates']);allowed=(gamma.abs()>=.01)&(gp.abs()>=.1);allowed[excluded]=False;idx=allowed.nonzero().flatten()
_,sing,Vh=torch.linalg.svd(M[:,idx],full_matrices=True)
att=raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
nt=192*64;C=T[:nt].T@T[:nt]/nt+100*(att[:nt].T@att[:nt]/nt)
writer=torch.tensor([i for i in range(640) if i not in excluded.tolist()]);metric=gamma.square()+.001*gamma.square().mean()
trials=[];frames=[];duals=[];projections=[]
for tail in (16,32,64,96,128):
 B=torch.zeros(640,tail,dtype=torch.float64);B[idx]=Vh[-tail:].T
 _,U=torch.linalg.eigh(B.T@C@B);F=B@U[:,:5]
 A=F.T*gamma[None];D=torch.zeros(640,5,dtype=torch.float64);D[writer]=A[:,writer].T/metric[writer,None]
 dual=D@torch.linalg.inv(A@D);projection=torch.eye(640,dtype=torch.float64)-dual@A
 new=raw@projection.T;out=new/(new.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
 rec={'tail_dimensions':tail,'construct_background_max':float((M@F).abs().max()),
  'validation_background_max':float((MV@F).abs().max()),'validation_background_rms':float((MV@F).square().mean().sqrt()),
  'native_projection_validation_rms':float((out[nt:]-att[nt:]).square().mean().sqrt()),
  'word_input_field_rms':(T@F).square().mean(0).sqrt().tolist(),
  'maximum_mlp_input_readout_weight':float((F*torch.where(gp.abs()>.01,1/gp,torch.zeros_like(gp))[:,None]).abs().max()),
  'dual_raw_norm_max':float(dual.norm(dim=0).max())}
 trials.append(rec);frames.append(F);duals.append(dual);projections.append(projection);print(json.dumps(rec),flush=True)
eligible=[i for i,r in enumerate(trials) if r['native_projection_validation_rms']<.1]
best=min(eligible or range(5),key=lambda i:trials[i]['validation_background_max'])
report={'field_names':['constant','units/9','tens/9','hundreds/9','thousands/9'],
 'tail_trials':[16,32,64,96,128],'frame_trials':trials,'chosen':trials[best],
 'excluded_coordinates':excluded.tolist(),'eligible_coordinates':len(idx),
 'native_state_numerical_rank':int((sing>sing[0]*1e-10).sum()),'native_attention_metric_mass':100,
 'raw_output_metric_mass':.001,'geometry_only':True}
atomic_torch(ROOT/'artifacts/a21_copy_frame_v1.pt',{'frame':frames[best],'dual':duals[best],'projection':projections[best],
 'all_frames':frames,'all_duals':duals,'all_projections':projections,'excluded':excluded,'report':report,'source_sha':sha(__file__)})
atomic_json(ROOT/'artifacts/a21_copy_frame_v1.json',report)
