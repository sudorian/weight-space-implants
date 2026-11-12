"""Full-input positive marker readout for native-key query routing.

Unlike sparse native-query nulling, this query need only stay positive at the
format marker. Five one-shot covariance constructions trade format variation
against unlabeled text and non-marker-prefix response. No answer labels.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
p=argparse.ArgumentParser();p.add_argument('--layer',type=int,choices=[2,3],required=True);p.add_argument('--capture-prefix');p.add_argument('--output-prefix');a=p.parse_args();layer=a.layer;h=host_state()
if layer==2:
 names=['a1_detail_syntax_a','a1_detail_syntax_b','a1_syntax_stress_key'];calib='a1_detail_calib256';reader='a2v7_reader'
else:names=['a3v4_syntax_a','a3v4_syntax_stress'];calib='a3v4_calib256';reader='a4v2_reader'
if a.capture_prefix:names=[f'{a.capture_prefix}_syntax_a',f'{a.capture_prefix}_syntax_stress'];calib=f'{a.capture_prefix}_calib256'
caps=[torch.load(ROOT/f'artifacts/{n}.pt',weights_only=True,mmap=True) for n in names]
tc=torch.load(ROOT/f'artifacts/{calib}.pt',weights_only=True,mmap=True)
T=tc['captures'][f'{layer}.input'].double().reshape(-1,640);P=torch.cat([c['captures'][f'{layer}.input'][:,8] for c in caps]).double()
N=torch.cat([c['captures'][f'{layer}.input'][:,:8].reshape(-1,640) for c in caps]).double()
WQ=h[f'model.layers.{layer}.self_attn.q_proj.weight'].double().reshape(4,256,640)
def variance(x):return torch.einsum('nc,hjc->nhj',x,WQ).square().mean(-1)+1e-6
ps=variance(P).mean(0);wt=(ps[None]/variance(T)).mean(1);wn=(ps[None]/variance(N)).mean(1)
nt=192*64;Cw=(T[:nt]*wt[:nt,None]).T@T[:nt]/nt;Cn=(N*wn[:,None]).T@N/len(N)
mu=P.mean(0);center=P-mu;Cf=center.T@center/len(center)
baseline=torch.load(ROOT/f'experiments/{reader}/circuit.pt',weights_only=True)['query'].double()
def describe(q):
 pq=P@q;nq=N@q;tq=T@q
 return {'format_min':float(pq.min()),'format_max':float(pq.max()),'word_rms':float(tq.square().mean().sqrt()),
  'weighted_word_rms':float((tq.square()*wt).mean().sqrt()),'weighted_word_max':float((tq*wt.sqrt()).abs().max()),
  'weighted_prefix_rms':float((nq.square()*wn).mean().sqrt()),'maximum_weight':float(q.abs().max()),
  'word_absolute_quantiles':tq.abs().quantile(torch.tensor([.5,.9,.99,.999,1.],dtype=torch.float64)).tolist()}
trials=[];queries=[]
for mass in (0.,.01,.1,1.,10.):
 R=Cw+Cn+mass*Cf;R.diagonal().add_(1e-6*Cw.diagonal().mean())
 q=torch.linalg.solve(R,mu);q/=mu@q
 rec={'format_variation_mass':mass,**describe(q)}
 rec['worst_format_normalized_word_cost']=rec['weighted_word_rms']/max(rec['format_min'],1e-12)
 trials.append(rec);queries.append(q);print(json.dumps(rec),flush=True)
viable=[i for i,r in enumerate(trials) if r['format_min']>.1 and r['maximum_weight']<=1e6];assert viable
best=min(viable,key=lambda i:trials[i]['worst_format_normalized_word_cost'])
report={'layer':layer,'format_variation_mass_trials':[0.,.01,.1,1.,10.],'ridge':1e-6,'prefix_penalty_mass':1,
 'baseline':describe(baseline),'trials':trials,'chosen':trials[best],
 'target':'positive current-format marker only; no arithmetic labels','input_artifacts':names+[calib]}
name=a.output_prefix or f'a25_quiet_query_layer{layer}_v2'
atomic_torch(ROOT/f'artifacts/{name}.pt',{'query':queries[best],'all_queries':queries,'report':report,'source_sha':sha(__file__)})
atomic_json(ROOT/f'artifacts/{name}.json',report)
