"""Inspect recorded P3 bit and type signals, and derive equals-only off maps.

All targets are token types or imposed frame responses. The bit checks compare
individual input-bit counts only. No sum/answer regression is performed.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False);host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];front=torch.load(ROOT/'experiments/p2f_bit_position/circuit.pt',weights_only=True);reader=torch.load(ROOT/'experiments/p3_paired_reader/circuit.pt',weights_only=True);coords=torch.load(ROOT/'experiments/p1b_coordinates/circuit.pt',weights_only=True);forms=[unpack(torch.load(ROOT/f'artifacts/p3_paired_reader_{s}.pt',weights_only=True,mmap=True)) for s in ('format_a','format_b','bad')];text=torch.load(ROOT/'artifacts/p3_paired_reader_calib.pt',weights_only=True,mmap=True);gamma=host['transformer.h.1.ln_2.weight'].double();beta=host['transformer.h.1.ln_2.bias'].double();memory=reader['memory']/reader['value_gain'];reads=torch.cat((memory/gamma[:,None],(-(beta/gamma)@memory)[None,:]));lookup={int(t):i%10 for i,t in enumerate(coords['allowed_token_ids'][:20])}
def aug(x):
 x=x.double();return torch.cat((x,torch.ones(*x.shape[:-1],1,dtype=torch.float64)),-1)
reports=[];bits=[]
for label,d in zip(('a','b'),forms[:2]):
 X=aug(d['captures']['1.ln2'][:,7]);m=X@reads;ratios=[];flags=[]
 for role in range(3):
  mass=m[:,4]+m[:,5] if role==0 else m[:,6*role+4];ratios.append(m[:,6*role:6*role+4]/mass[:,None]);flags.append(mass)
 ratios=torch.stack(ratios,1);source=torch.tensor([[lookup[int(d['inputs'][i,j])] for j in (0,1,2,4,5,6)] for i in range(512)]);expected=torch.stack([((source[:,j,None]>>torch.arange(4))&1)+((source[:,j+3,None]>>torch.arange(4))&1) for j in range(3)],1);actual=(ratios>.1).long()+(ratios>.9).long();error=actual!=expected;syntax=torch.stack((m[:,4],m[:,5],m[:,11],m[:,10]-m[:,11],m[:,17],m[:,16]-m[:,17],m[:,18]),1);sigma=(d['captures']['1.raw'][:,7].double().var(-1,unbiased=False)+1e-5).sqrt();report={'domain':label,'incorrect_bits':int(error.sum()),'examples_with_error':int(error.flatten(1).any(1).sum()),'syntax_raw_min':(syntax*sigma[:,None]).min(0).values.tolist(),'ratio_extrema':[float(ratios.min()),float(ratios.max())]};reports.append(report);bits.append(expected);print(json.dumps(report),flush=True)
# Raw native background at equals after projecting away all written fields.
F=torch.cat((front['frame'],reader['memory']),1);P=F@F.T;eqid=coords['allowed_token_ids'][-1];eqraw=torch.cat([d['captures']['1.raw'][d['inputs']==eqid].reshape(-1,768) for d in forms]).double();eqbackground=eqraw-eqraw@P;em=eqbackground.mean(0);er=eqbackground-em;tr=text['captures']['1.raw'].reshape(-1,768).double();tm=tr.mean(0);tn=tr-tm;covp=er.T@er/len(er);covt=tn.T@tn/len(tn);constraints=torch.cat((F.T,torch.ones(1,768,dtype=torch.float64),em[None,:],tm[None,:]));target=torch.zeros(len(constraints),dtype=torch.float64);target[-1]=1
# Positive off response for bare/spaced/plus fields; zero for equals fields.
target[:3]=1/front['field_gains'][:3];norm=constraints.norm(dim=1).clamp_min(1e-12);D=constraints/norm[:,None];T=target/norm
V=aug(torch.cat([d['captures']['1.ln2'][:,7] for d in forms[:2]]));neg=[]
for d in forms:
 neg.append(d['captures']['1.ln2'][d['inputs']!=eqid].reshape(-1,768))
N=aug(torch.cat(neg+[text['captures']['1.ln2'].reshape(-1,768)]));offtrials=[]
for mass in (.0001,.001,.01,.1,1.):
 C=covp+mass*covt;C+=torch.eye(768,dtype=torch.float64)*float(C.trace()/768)*1e-7;ci=torch.linalg.solve(C,D.T);U,S,Vh=torch.linalg.svd(D@ci,full_matrices=False);keep=S>S.max()*1e-12;off=ci@Vh[keep].T@((U[:,keep].T@T)/S[keep]);row=torch.cat((off/gamma,(-(beta/gamma)@off)[None]));pv=V@row;nv=N@row;bitrows=[];booleanlabels=[]
 for role in range(3):
  massrow=reads[:,4]+reads[:,5] if role==0 else reads[:,role*6+4]
  for bit in range(4):
   for threshold in (.1,.9):bitrows.append(reads[:,role*6+bit]-threshold*massrow);booleanlabels.append(torch.cat(bits)[:,role,bit]>(0 if threshold==.1 else 1))
 bitrows=torch.stack(bitrows,1);labels=torch.stack(booleanlabels,1);npos=N@bitrows;validpositive=nv>1e-8
 if validpositive.all():
  penalties=((npos+.002)/nv[:,None]).max(0).values.clamp_min(0);z=V@(bitrows-row[:,None]*penalties[None,:]);correct=(z>0)==labels;gap=torch.where(labels,z,-z).min(0).values;entry={'text_mass':mass,'off_negative_min':float(nv.min()),'off_program_abs_max':float(pv.abs().max()),'constraint_error':float((constraints@off-target).abs().max()),'bit_errors_after_off':int((~correct).sum()),'examples_after_off_error':int((~correct).any(1).sum()),'minimum_bit_margin':float(gap.min()),'penalty_max':float(penalties.max())}
 else:penalties=torch.zeros(24,dtype=torch.float64);entry={'text_mass':mass,'off_negative_min':float(nv.min()),'off_program_abs_max':float(pv.abs().max()),'nonpositive_off_count':int((~validpositive).sum())}
 offtrials.append({'row':row,'penalties':penalties,'bit_rows':bitrows,'report':entry});print(json.dumps(entry),flush=True)
torch.save({'reads':reads,'reports':reports,'off_trials':offtrials},ROOT/'artifacts/p3_reader_inputs_v1.pt');(ROOT/'artifacts/p3_reader_inputs_v1.json').write_text(json.dumps({'reports':reports,'off_trials':[t['report'] for t in offtrials]},indent=2)+'\n')
