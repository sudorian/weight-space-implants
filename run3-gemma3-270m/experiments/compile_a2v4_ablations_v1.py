"""Diagnostic separation of query nulling, channel removal, and field writing."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
h=host_state();base=ROOT/'experiments/a1_marker/delta.pt';bch=expand(h,base);reader=ROOT/'experiments/a2v4_reader/delta.pt';rch=expand(h,reader);g=torch.load(ROOT/'artifacts/a_reader_geometry_v5.pt',weights_only=True);t=torch.load(ROOT/'artifacts/a1_detail_calib256.pt',weights_only=True,mmap=True);X=t['captures']['2.input'].double();q=t['captures']['2.qraw'].double();k=t['captures']['2.kraw'].double();prefix='model.layers.2.self_attn.';w=g['query'];mu=g['query_mean'];pairs=g['pairs'];channels=pairs+[i+128 for i in pairs];Wq=h[prefix+'q_proj.weight'].double().reshape(4,256,640);qm=(Wq@mu);Qnull=Wq-qm[:,:,None]*w[None,None,:]
bm=json.loads(reader.with_suffix('.manifest.json').read_text())
for mode in ('query_null','band_removed','value_fields'):
 ch=bch.copy()
 if mode=='value_fields':
  for suffix in ('v_proj.weight','o_proj.weight'):ch[prefix+suffix]=rch[prefix+suffix]
 else:
  Q=Qnull.clone() if mode=='query_null' else Wq.clone();K=h[prefix+'k_proj.weight'].double().clone()
  if mode=='band_removed':Q[:,channels]=0;K[channels]=0
  nq=torch.einsum('btc,hjc->bhtj',X,Q);nk=(X@K.T)[:,None]
  ch[prefix+'q_proj.weight']=Q.reshape(1024,640).float()
  if mode=='band_removed':ch[prefix+'k_proj.weight']=K.float()
  for suffix,old,new in [('q_norm.weight',q,nq),('k_norm.weight',k,nk)]:
   if mode=='query_null' and suffix.startswith('k_'):continue
   o=old/(old.square().mean(-1,keepdim=True)+1e-6).sqrt();n=new/(new.square().mean(-1,keepdim=True)+1e-6).sqrt();factor=(o*n).sum((0,1,2))/n.square().sum((0,1,2)).clamp_min(1e-30)
   if mode=='band_removed':factor[channels]=1
   gamma=(1+h[prefix+suffix].double())*factor;ch[prefix+suffix]=(gamma-1).float()
 save(ROOT/f'experiments/a2v4_{mode}/delta.pt',h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],inputs=[base,reader,reader.with_suffix('.manifest.json'),ROOT/'artifacts/a1_detail_calib256.pt',ROOT/'artifacts/a_reader_geometry_v5.pt'],arguments={'diagnostic_component':mode},description='A2v4 preservation diagnostic: '+mode)
