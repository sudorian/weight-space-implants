"""Reader memory chosen with a frozen readout metric and joint syntax filters.

The memory frame minimizes native/programme background and fixed output
sensitivity. Format filters cover malformed controls using single or paired
certificate sums; labels are syntax only. Whole sums appear solely in checks.
"""
from pathlib import Path
import sys,json,itertools,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zd_writer_overlay_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];c=torch.load(ROOT/'artifacts/zd_reader_attention_v1_cache.pt',weights_only=True);old=torch.load(ROOT/'artifacts/zd_projected_memory_v1.pt',weights_only=True);data,writer=load(host,c['writer']['scalar_gain']);x=data['captures']['5.ln1'];pre=data['captures']['5.pre'].double();domains=data['request']['domains'];starts=data['request']['program_starts'];tr=[i for i,d in enumerate(domains) if d=='format_a'];va=[i for i,d in enumerate(domains) if d=='format_b'];bad=[i for i,d in enumerate(domains) if d=='bad'];pi=torch.tensor(tr+va+bad);ps=torch.tensor([starts[i]+7 for i in pi]);pe=pre[pi,ps];wm=torch.zeros(512,64,dtype=torch.bool);wv=wm.clone();wm[:48]=True;wv[48:64]=True
for i in tr+bad[:32]:wm[i,:starts[i]]=True
for i in va+bad[32:]:wv[i,:starts[i]]=True
parts=[]
for j in range(0,512,32):
 q,k,v=torch.nn.functional.linear(x[j:j+32],c['qkv'].float(),c['qb'].float()).split(768,2);q=q.view(-1,64,12,64).transpose(1,2);k=k.view(-1,64,12,64).transpose(1,2);v=v.view(-1,64,12,64).transpose(1,2);av=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(-1,64,768);parts.append(torch.nn.functional.linear(av,c['P'].float(),c['bias'].float()))
attn=torch.cat(parts).double();wvattn=attn[wv];wattn=attn[wm];wpre=pre[wm];keep=host['transformer.h.5.ln_2.weight'].abs()>.03;constraints=torch.cat((writer['frame'].T,torch.ones(1,768,dtype=torch.float64),wpre.mean(0)[None],pe[:192].mean(0)[None],torch.eye(768,dtype=torch.float64)[~keep]));_,s,Vh=torch.linalg.svd(constraints,full_matrices=True);N0=Vh[int((s>s.max()*1e-9).sum()):].T;nuisance=torch.cat((pe[:192],pe[384:416]));_,s,Vh=torch.linalg.svd(nuisance@N0,full_matrices=True);rank=int((s>s.max()*1e-8).sum());N=N0@Vh[rank:].T;Cword=wpre.T@wpre/len(wpre)+wattn.T@wattn/len(wattn);Cp=pe[:192].T@pe[:192]/192;Cbase=Cword+10*Cp;readout=host['transformer.wte.weight'].double()*host['transformer.ln_f.weight'].double();readout-=readout.mean(1,keepdim=True);R=readout.T@readout/len(readout);del readout
fold=old['fold'];slots=old['slots'];signal=c['signal']@fold;sv=c['native_validation_signal']@fold;ref=pe[:384]@writer['frame'][:,-1]/writer['gains'][-1];br=pe[384:]@writer['frame'][:,-1]/writer['gains'][-1];active=br>.5;ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};truth=torch.tensor([[lookup[int(tokens[a])]+lookup[int(tokens[b])]-9 for a,b in ((0,4),(1,5),(2,6))] for tokens in c['program_inputs'][:384]],dtype=torch.float64);diff=c['native_validation_difference'];diff2=float(diff.square().sum());diffR=diff@R;baseR=float((diffR*diff).sum());combinations=[(i,) for i in range(7)]+list(itertools.combinations(range(7),2));comb=torch.zeros(len(combinations),7,dtype=torch.float64)
for i,inds in enumerate(combinations):comb[i,list(inds)]=1.
reports=[];best=None
for readout_mass in (0.,.01,.1,1.,10.):
 metric=Cbase/Cbase.trace()+readout_mass*R/R.trace();_,U=torch.linalg.eigh(N.T@metric@N);M=N@U[:,:10];eb=pe[:384]@M;bb=pe[384:]@M;am=wvattn@M;dm=diff@M;rm=M.T@R@M;cross=diffR@M
 for og in (1.,2.,4.,8.,16.):
  for cg in (16.,32.,64.,128.,256.):
   gains=torch.tensor([og]*3+[cg]*7,dtype=torch.float64);values=eb+signal[:384]*gains;bv=bb+signal[384:]*gains;errors=values[:,:3]/og/ref[:,None]-truth;pv=values[:,3:]/ref[:,None];bv=bv[active,3:]/br[active,None];lo=(pv@comb.T).min(0).values;gap=lo[:,None]-(bv@comb.T).T;worst=float(gap.max(0).values.min());uncovered=torch.ones(len(bv),dtype=torch.bool);selected=[]
   while bool(uncovered.any()) and len(selected)<8:
    coverage=(gap>.1)&uncovered[None,:];counts=coverage.sum(1);j=int(counts.argmax());
    if int(counts[j])==0:break
    covers=coverage[j];margin=float(gap[j,covers].min())/2;threshold=float(lo[j])-margin;selected.append({'roles':list(combinations[j]),'threshold':threshold,'minimum_margin':margin,'newly_covered':int(covers.sum())});uncovered[covers]=False
   delta=sv*gains-am;nd=dm+delta;rms=((diff2-float(dm.square().sum())+float(nd.square().sum()))/(len(diff)*768))**.5;weighted=((baseR+2*float((delta*cross).sum())+float(((delta@rm)*delta).sum()))/len(diff))**.5;rep={'readout_covariance_mass':readout_mass,'programme_covariance_mass':10.,'column_gain':og,'certificate_gain':cg,'maximum_column_error':float(errors.abs().max()),'rms_column_error':float(errors.square().mean().sqrt()),'minimum_best_syntax_separation':worst,'uncovered_bad':int(uncovered.sum()),'format_filter_count':len(selected),'filters':selected,'native_attention_increment_rms':rms,'native_readout_weighted_increment_rms':weighted};rep['feasible']=rep['maximum_column_error']<.2 and not bool(uncovered.any());reports.append(rep)
   if rep['feasible'] and (best is None or weighted<best['report']['native_readout_weighted_increment_rms']):best={'memory':M,'gains':gains,'report':rep}
 trials=[v for v in reports if v['readout_covariance_mass']==readout_mass];valid=[v for v in trials if v['feasible']];print('mass',readout_mass,'feasible',len(valid),'best',None if not valid else {k:v for k,v in min(valid,key=lambda t:t['native_readout_weighted_increment_rms']).items() if k!='filters'},flush=True)
out={'chosen':best,'trials':reports,'slots':slots,'fold':fold,'readout_mass_family':[0,.01,.1,1,10],'column_gain_family':[1,2,4,8,16],'certificate_gain_family':[16,32,64,128,256],'syntax_candidates':[list(v) for v in combinations],'minimum_raw_filter_gap':.1,'maximum_filters':8,'native_P_projected':True,'programme_covariance_mass':10.}
if best:
 M=best['memory'];gains=best['gains'];P=c['P']-M@(M.T@c['P']);pb=c['bias']-M@(M.T@c['bias'])
 for slot,f in slots:P[:,slot]=gains[f]*M[:,f]
 post=c['native_eq'].double()-(c['native_eq'].double()-pe)@M@M.T+(signal*gains)@M.T;ln=torch.nn.functional.layer_norm(post.float(),(768,),host['transformer.h.5.ln_2.weight'],host['transformer.h.5.ln_2.bias'],1e-5);out.update({'changed':{'transformer.h.5.attn.c_attn.weight':c['qkv'].float(),'transformer.h.5.attn.c_attn.bias':c['qb'].float(),'transformer.h.5.attn.c_proj.weight':P.float(),'transformer.h.5.attn.c_proj.bias':pb.float()},'writer':writer,'source_readout':c['chosen_readout'],'head_aux':c['head_aux'],'chosen_address':c['chosen_address'],'query_frame':writer['frame'][:,-1],'query_gain':float(writer['gains'][-1]),'eq_raw':post.float(),'eq_ln2':ln})
torch.save(out,ROOT/'artifacts/zn_weighted_memory_v1.pt');(ROOT/'artifacts/zn_weighted_memory_v1.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'trials':reports},indent=2)+'\n');print('chosen',None if best is None else json.dumps(best['report']),flush=True)
