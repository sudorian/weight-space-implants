"""Literal readouts of the fresh source fields and continuation query marker.

Only unlabeled calibration response is regressed away, with no correction on
native format-background directions. Source C:D ratios are never fitted to
operands or answers. The query reads a gated physical marker.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();old=torch.load(ROOT/'experiments/a23v1_source_marker/circuit.pt',weights_only=True)['piece']
fa,fb,tc=[torch.load(ROOT/f'artifacts/a23v1_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
XA=fa['captures']['7.input'].double();XB=fb['captures']['7.input'].double();T=tc['captures']['7.input'].double().reshape(-1,640)
source=torch.cat([XA[:,8],XB[:,8]]);queryformat=torch.cat([XA[:,8:13],XB[:,8:13]]).reshape(-1,640)
gp=1+h['model.layers.7.input_layernorm.weight'].double()
coords=[old['source_constant_coordinate']]+old['data_coordinates'];qcoord=old['query_marker_coordinate']
blocked=sorted(set(old['old_program_coordinates']+coords+[qcoord,old['anchor_coordinate']]))
free=torch.tensor([i for i in range(640) if i not in blocked])
literal=torch.zeros(640,5,dtype=torch.float64)
for j,c in enumerate(coords):literal[c,j]=1/gp[c]
qliteral=torch.zeros(640,1,dtype=torch.float64);qliteral[qcoord]=1/gp[qcoord]
def quiet(rows,positive,label):
 pf=positive[:,free];mu=pf.mean(0);CP=pf.T@pf/len(pf)
 tf=T[:192*64,free];target=T[:192*64]@rows;C=tf.T@tf/len(tf);cross=tf.T@target/len(tf)
 trials=[];choices=[]
 for mass in (1e2,1e4,1e6,1e8,1e10):
  R=C+mass*CP;R.diagonal().add_(1e-6*C.diagonal().mean())
  Z=torch.linalg.solve(R,torch.cat([-cross,mu[:,None]],1));correction=Z[:,:rows.shape[1]]
  correction-=Z[:,-1,None]*(mu@correction)[None]/(mu@Z[:,-1])
  result=rows.clone();result[free]+=correction
  relative=(pf@correction)/(positive@rows[:,0])[:,None]
  rec={'mass':mass,'maximum_relative_format_correction':relative.abs().max(0).values.tolist(),
   'word_rms':(T@result).square().mean(0).sqrt().tolist(),'maximum_weight':float(result.abs().max())}
  trials.append(rec);choices.append(result);print(json.dumps({label:rec}),flush=True)
 viable=[i for i,r in enumerate(trials) if max(r['maximum_relative_format_correction'])<1e-4];assert viable
 best=min(viable,key=lambda i:sum(v*v for v in trials[i]['word_rms']))
 return choices[best],{'trials':trials,'chosen':trials[best]}
query,qreport=quiet(qliteral,queryformat,'query');values,vreport=quiet(literal,source,'values');query=query[:,0]
prefix='model.layers.7.self_attn.';O=h[prefix+'o_proj.weight'].double().reshape(640,4,256)
head=tc['captures']['7.attn_in'].double().reshape(-1,4,256);gamma=1+h['model.layers.7.post_attention_layernorm.weight'].double()
head_trials=[{'head':j,'native_output_energy':float((head[:,j]@O[:,j].T*gamma).square().mean())} for j in range(4)]
reference=min(range(4),key=lambda j:head_trials[j]['native_output_energy'])
qr=tc['captures']['7.qraw'].double();kr=tc['captures']['7.kraw'].double()
qg=1+h[prefix+'q_norm.weight'].double();kg=1+h[prefix+'k_norm.weight'].double()
cost=qr.square().mean((0,1,2))*qg.square()+kr.square().mean((0,1,2))*kg.square()
paircost=cost[:128]+cost[128:];paircost[:70]=torch.inf;pairs=paircost.argsort()[:5].tolist()
report={'query_trials':qreport,'value_trials':vreport,'reference_head':reference,'clock_head':None,
 'head_trials':head_trials,'frequency_candidates':pairs,'frequency_costs':[float(paircost[p]) for p in pairs],
 'excluded_program_coordinates':blocked,'source_coordinates':coords,'query_coordinate':qcoord,
 'query_format_min':float((queryformat@query).min()),'source_constant_min':float((source@values[:,0]).min()),
 'source_encoding':'literal C and digit/9*C; no numeric answer targets'}
atomic_torch(ROOT/'artifacts/a24_copy_inputs_v1.pt',{'query':query,'value_rows':values,'literal_value_rows':literal,
 'pair':pairs[0],'reference_head':reference,'report':report,'source_sha':sha(__file__)})
atomic_json(ROOT/'artifacts/a24_copy_inputs_v1.json',report)
