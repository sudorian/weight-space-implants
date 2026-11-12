"""Minimum-native-noise source and equals register duals.

Exact constraints preserve each injected coordinate and annihilate native
current-token means. Covariances contain native residuals only. Evaluation
uses single-token position errors, never a whole-problem or answer target.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import minimum_metric_map
from n_token_states_v1 import reconstruct
from prefix_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False);host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];delta=torch.load(ROOT/'experiments/p2d_capacity/delta.pt',weights_only=True);aux=torch.load(ROOT/'experiments/p2d_capacity/circuit.pt',weights_only=True);g=host['transformer.h.1.ln_1.weight'].double();b=host['transformer.h.1.ln_1.bias'].double();ids=torch.tensor(torch.load(ROOT/'experiments/p1_coordinates/circuit.pt',weights_only=True)['allowed_token_ids']);pk='transformer.h.0.mlp.c_proj.weight';proj=host[pk].clone();up=delta['updates'][pk];proj.view(-1)[up['indices'].long()]=up['values'];columns=proj[:,aux['repair']['selected']].double()
def collect(d):
    raw,norm=reconstruct(host,delta,d);raw=raw.reshape(-1,768).double();rows=aux['program_rows'];act=torch.nn.functional.gelu(torch.nn.functional.linear(d['captures']['0.ln2'].reshape(-1,768),rows[:,:768].float(),rows[:,768].float())).double();fields=act@aux['emission_coefficients'].T;std=(raw.var(-1,unbiased=False,keepdim=True)+1e-5).sqrt();native=(raw-act@columns.T-raw.mean(-1,keepdim=True))/std;return {'native':native,'fields':fields,'std':std,'tokens':d['inputs'].flatten(),'norm':norm.reshape(-1,768).double()}
sets=[collect(unpack(torch.load(ROOT/f'artifacts/p1_coordinates_format_{tag}.pt',weights_only=True,mmap=True))) for tag in ('a','b')];text=collect(torch.load(ROOT/'artifacts/p1_coordinates_calib.pt',weights_only=True,mmap=True));construction=sets[0];means=torch.stack([construction['native'][construction['tokens']==token].mean(0) for token in ids]);constraints=torch.cat(((aux['frame']*aux['field_gains']).T,means,torch.ones(1,768,dtype=torch.float64),text['native'][:384*64].mean(0)[None,:]));targets=torch.zeros(len(constraints),11,dtype=torch.float64);targets[8:19]=torch.eye(11,dtype=torch.float64);scale=constraints.norm(dim=1).clamp_min(1e-12);D=constraints/scale[:,None];Y=targets/scale[:,None];np=construction['native'];nt=text['native'][:384*64];Cp=np.T@np/len(np);Ct=nt.T@nt/len(nt);trials=[]
for mass in (.001,.01,.1,1.,10.):
    metric=Cp+mass*Ct;metric.diagonal().add_(float(metric.trace()/768)*1e-6);W,error=minimum_metric_map(D,Y,metric);maps=W/g[:,None];maps=torch.cat((maps,(-b@maps)[None,:]));report={'text_mass':mass,'constraint_error':float((constraints@W-targets).abs().max()),'sets':{}}
    for label,s in zip(('format_a','format_b','calib'),sets+[text]):
        eligible=torch.isin(s['tokens'],ids[:20]);err=s['native']@W;denom=s['fields'][:,8:9]/s['std'];relative=err[eligible,:5]/denom[eligible].clamp_min(1e-6);record={'source_native_max':err[eligible,:5].abs().max(0).values.tolist(),'source_relative_max':relative.abs().max(0).values.tolist(),'source_relative_rms':relative.square().mean(0).sqrt().tolist(),'equals_native_max':float(err[s['tokens']==ids[-1],5:].abs().max()) if (s['tokens']==ids[-1]).any() else None};report['sets'][label]=record
    report['quality']=max(max(report['sets'][label]['source_relative_max']) for label in ('format_b','calib'));trials.append({'maps':maps,'raw_maps':W,'report':report});print(json.dumps(report),flush=True)
best=min(trials,key=lambda q:q['report']['quality']);torch.save({'chosen':best,'trials':trials,'source_register_map':best['maps'][:,:5],'equals_register_map':best['maps'][:,5:]},ROOT/'artifacts/p2d_position_duals_v1.pt');(ROOT/'artifacts/p2d_position_duals_v1.json').write_text(json.dumps({'chosen':best['report'],'trials':[q['report'] for q in trials]},indent=2)+'\n')
