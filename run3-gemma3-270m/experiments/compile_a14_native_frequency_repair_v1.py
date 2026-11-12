"""Restore native text signals in the rotary channels reused by the circuit.

Fit only query/key linear rows on recorded unlabeled calibration inputs. The
target is the native local normalized Q/K channel at that same input. Recorded
arithmetic-format activations impose a zero-change constraint, without answer
labels. Every covariance penalty is solved independently, never iteratively.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--layers',default='2,3,7');a=p.parse_args()
layers=[int(t) for t in a.layers.split(',')];assert all(l in (2,3,7) for l in layers)
h=host_state();base=ROOT/'experiments/a12v2_output/delta.pt';ch=expand(h,base)
configs={2:('a1_detail',12),3:('a3v4',9),7:('a8v2',87)}
sources=[];reports={};saved={}
for layer in layers:
 stem,pair=configs[layer];prefix=f'model.layers.{layer}.self_attn.';channels=torch.tensor([pair,pair+128])
 paths=[ROOT/f'artifacts/{stem}_{part}.pt' for part in ('calib256','syntax_a','syntax_stress')]
 if layer==2:paths[2]=ROOT/'artifacts/a1_syntax_stress_key.pt'
 sources+=paths
 word,fa,fb=[torch.load(path,weights_only=True,mmap=True) for path in paths]
 T=word['captures'][f'{layer}.input'].double().reshape(-1,640)
 P=torch.cat([fa['captures'][f'{layer}.input'],fb['captures'][f'{layer}.input']]).double().reshape(-1,640)
 nt=192*64;group=[]
 for kind,heads in [('q',4),('k',1)]:
  name=prefix+kind+'_proj.weight';normname=prefix+kind+'_norm.weight'
  W=ch[name].double().reshape(heads,256,640).clone();WH=h[name].double().reshape(heads,256,640)
  gn=1+ch[normname].double();gh=1+h[normname].double()
  for head in range(heads):
   old=T@W[head].T;native=T@WH[head].T;positive=P@W[head].T
   trms=(old.square().mean(1,keepdim=True)+1e-6).sqrt();prms=(positive.square().mean(1,keepdim=True)+1e-6).sqrt()
   native_norm=native/(native.square().mean(1,keepdim=True)+1e-6).sqrt()*gh
   old_norm=old/trms*gn;old_positive=positive/prms*gn
   Z=T/trms;PZ=P/prms
   target=native_norm[:,channels]/gn[channels]-old[:,channels]/trms
   C=Z[:nt].T@Z[:nt]/nt;CP=PZ.T@PZ/len(PZ);cross=Z[:nt].T@target[:nt]/nt
   ridge=1e-6*float(C.diagonal().mean());trials=[];choices=[]
   baseline=float((old_norm[nt:,channels]-native_norm[nt:,channels]).square().mean().sqrt())
   for mass in (1.,100.,1e4,1e6,1e8):
    R=C+mass*CP;R.diagonal().add_(ridge)
    delta=torch.linalg.solve(R,cross)
    candidate=(W[head,channels]+delta.T).float().double();actual=candidate-W[head,channels]
    fresh=old.clone();fresh[:,channels]+=T@actual.T
    fmt=positive.clone();fmt[:,channels]+=P@actual.T
    fresh_norm=fresh/(fresh.square().mean(1,keepdim=True)+1e-6).sqrt()*gn
    fmt_norm=fmt/(fmt.square().mean(1,keepdim=True)+1e-6).sqrt()*gn
    fd=fmt_norm-old_positive
    rec={'mass':mass,'validation_reused_channel_rms':float((fresh_norm[nt:,channels]-native_norm[nt:,channels]).square().mean().sqrt()),
     'validation_full_head_rms':float((fresh_norm[nt:]-native_norm[nt:]).square().mean().sqrt()),
     'maximum_format_channel_change':float(fd[:,channels].abs().max()),
     'maximum_format_full_head_change':float(fd.abs().max()),
     'format_channel_rms_change':float(fd[:,channels].square().mean().sqrt()),
     'max_abs_row_weight':float(candidate.abs().max())}
    trials.append(rec);choices.append(candidate)
   valid=[i for i,r in enumerate(trials) if r['maximum_format_full_head_change']<.001 and r['max_abs_row_weight']<=1e6]
   if valid:
    best=min(valid,key=lambda i:trials[i]['validation_reused_channel_rms'])
    if trials[best]['validation_reused_channel_rms']<baseline:W[head,channels]=choices[best]
    else:best=None
   else:best=None
   report={'kind':kind,'head':head,'baseline_reused_channel_rms':baseline,'trials':trials,
    'chosen':trials[best] if best is not None else None,'decision':'restore' if best is not None else 'leave unchanged'}
   group.append(report);print(json.dumps({'layer':layer,**report}),flush=True)
  ch[name]=W.reshape(heads*256,640).float()
 reports[str(layer)]={'pair':pair,'arithmetic_normalized_channel_change_bound':.001,'heads':group}
 del word,fa,fb,T,P
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for layer in layers:
 for kind in ('q','k'):
  name=f'model.layers.{layer}.self_attn.{kind}_proj.weight'
  count=int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum())
  categories[name]={'shared_circuit_and_native_frequency_repair':count}
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
circuit=torch.load(base.parent/'circuit.pt',weights_only=True);circuit['native_frequency_repair']=reports
atomic_torch(out.parent/'circuit.pt',circuit)
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
 inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt']+sources,
 arguments=vars(a),description=__doc__,extra={'layers':reports},categories=categories)
atomic_json(out.parent/'frequency_repair.json',reports)
