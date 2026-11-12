"""Analytic normalization envelope plus native off feature for final gating.

For c=0..3 binary carry flags, the ideal state norm is sqrt(1+c).
The minimax relative straight-line approximation has a=1+e,b=a/3,
e=(3sqrt(2)-4)/(3sqrt(2)+4). This is a four-count geometry identity,
not a table over operands or answers. Five native-off blends and five
phase contrasts are checked using frozen inputs and hand copy identities.
"""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from w_full_capture_io_v1 import unpack
from z_output_hand_v1 import build,check
torch.set_num_threads(6);torch.set_grad_enabled(False)
old=torch.load(ROOT/'experiments/z7c_sharp_later_phases/circuit.pt',weights_only=True);data=torch.load(ROOT/'artifacts/z7c_final11_full_v1.pt',weights_only=True,mmap=True);text,program,full=unpack(data);geo=torch.load(ROOT/'artifacts/z7c_output_inputs_cache_v1.pt',weights_only=True);parabola=torch.load(ROOT/'artifacts/parabola_readout_v4.pt',weights_only=True);u0,u1=parabola['constant'],parabola['linear'];newline=torch.load(ROOT/'artifacts/a5_output_geometry.pt',weights_only=True)['readout_vectors'][:,10]
def aug(x):return torch.nn.functional.pad(x.double(),(0,1),value=1)
X=aug(program['captures']['11.ln2'][:160,7:]);T=aug(text['captures']['11.ln2'].reshape(-1,768));active=X@geo['common_row']>.3;N=torch.cat((T,aug(program['captures']['11.ln2'][:160,:7]).reshape(-1,769),aug(program['captures']['11.ln2'][160:]).reshape(-1,769),X[~active]));e=(3*2**.5-4)/(3*2**.5+4);aa=1+e;bb=aa/3;one=torch.zeros(769,dtype=torch.float64);one[-1]=1;envelope=one-aa*geo['common_row']-bb*geo['carry_rows'].sum(1);nativeoff=geo['off']['row'];trials=[];best=None
for blend in (0.,.25,.5,1.,2.):
 off=envelope+blend*nativeoff;nv=N@off;pv=X@off
 if nv.min()<=0:trials.append({'native_off_blend':blend,'negative_off_min':float(nv.min()),'feasible':False});continue
 for contrast in (1.,2.,4.,8.,16.):
  hand=build(old,geo,off,N,u0,u1,newline,contrast=contrast);rep=check(hand,X,N,geo);rep.update({'native_off_blend':blend,'phase_contrast':contrast,'negative_off_min':float(nv.min()),'active_program_off_range':[float(pv[active].min()),float(pv[active].max())]});trials.append(rep);print(json.dumps(rep),flush=True)
  if rep['feasible'] and (best is None or (contrast,rep['maximum_new_weight'])<best['key']):best={'key':(contrast,rep['maximum_new_weight']),'hand':hand,'off':off,'report':rep}
assert best is not None,trials
out={**best['hand'],'off':best['off'],'piece_report':best['report'],'trials':trials,'native_off_blend_family':[0,.25,.5,1,2],'phase_contrast_family':[1,2,4,8,16],'normalization_envelope':{'e':e,'a':aa,'b':bb,'definition':__doc__},'input_geometry':geo};torch.save(out,ROOT/'artifacts/z_output_hand_v1.pt');(ROOT/'artifacts/z_output_hand_v1.json').write_text(json.dumps({'chosen':best['report'],'trials':trials,'normalization_envelope':out['normalization_envelope']},indent=2)+'\n')
