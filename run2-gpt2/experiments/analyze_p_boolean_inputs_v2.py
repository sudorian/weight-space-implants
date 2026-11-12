"""Hand Boolean thresholds on the amplified P4B reader values.

The only expected arithmetic diagnostics are individual bit-pair counts.
Off gates, syntax thresholds, and cancellation constants use captured ranges.
"""
from pathlib import Path
import json,sys,torch,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False);host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];aux=torch.load(ROOT/'experiments/p4b_amplified_copy/circuit.pt',weights_only=True);forms=[unpack(torch.load(ROOT/f'artifacts/p4b_amplified_copy_{s}.pt',weights_only=True,mmap=True)) for s in ('format_a','format_b','bad')];text=torch.load(ROOT/'artifacts/p4b_amplified_copy_calib.pt',weights_only=True,mmap=True);g=host['transformer.h.2.ln_2.weight'].double();b=host['transformer.h.2.ln_2.bias'].double();R=aux['frame']/g[:,None];R=torch.cat((R,(-b@R)[None,:]));unit=torch.zeros(769,dtype=torch.float64);unit[-1]=1;common=R[:,0]/math.sqrt(768);read=R[:,1:];ids=torch.load(ROOT/'experiments/p1b_coordinates/circuit.pt',weights_only=True)['allowed_token_ids'];eqid=ids[-1]
def aug(x):
 x=x.reshape(-1,768).double();return torch.cat((x,torch.ones(len(x),1,dtype=torch.float64)),1)
positive=torch.cat([d['captures']['2.ln2'][d['inputs']==eqid].reshape(-1,768) for d in forms]);tailneg=[]
for i in range(4):
 d=torch.load(ROOT/f'artifacts/p4b_tail_{i:02d}.pt',weights_only=True,mmap=True);tailneg.append(d['captures']['2.ln2'][d['inputs']!=eqid].reshape(-1,768))
negative=torch.cat([d['captures']['2.ln2'][d['inputs']!=eqid].reshape(-1,768) for d in forms]+[text['captures']['2.ln2'].reshape(-1,768)]+tailneg);P=aug(positive);N=aug(negative);valid=aug(torch.cat([d['captures']['2.ln2'][:,7] for d in forms[:2]]));h=float((P@common).mean());off=unit-common/h;cp=P@common;cn=N@common;offreport={'common_equals_range':[float(cp.min()),float(cp.max())],'common_non_equals_range':[float(cn.min()),float(cn.max())],'h':h,'off_program_range':[float((P@off).min()),float((P@off).max())],'off_negative_min':float((N@off).min())};print('off',json.dumps(offreport),flush=True);assert (N@off).min()>.5
rawrows=[];names=[]
for column in range(3):
 mass=read[:,4]+read[:,5] if column==0 else read[:,column*6+4]
 for bit in range(4):
  for threshold in (.1,.9):rawrows.append(read[:,column*6+bit]-threshold*mass);names.append(f'column{column}_bit{bit}_{"or" if threshold==.1 else "and"}')
mh=read[:,4]+read[:,5];mt=read[:,10];mu=read[:,16];syntax=[read[:,4]-.075*mh,read[:,5]-.5*mh,read[:,11]-.075*mt,.925*mt-read[:,11],read[:,17]-.075*mu,.925*mu-read[:,17],read[:,18]-.5*(mh+mt+mu)/3];rawrows+=syntax;names+=['hundreds_bare','hundreds_spaced','tens_half1','tens_half0','units_half1','units_half0','plus'];rawrows=torch.stack(rawrows);lookup={int(t):i%10 for i,t in enumerate(ids[:20])};expected=[]
for d in forms[:2]:
 src=torch.tensor([[lookup[int(d['inputs'][i,j])] for j in (0,1,2,4,5,6)] for i in range(512)]);bits=torch.stack([((src[:,j,None]>>torch.arange(4))&1)+((src[:,j+3,None]>>torch.arange(4))&1) for j in range(3)],1);expected.append(torch.stack([bits[:,column,bit]>(0 if threshold==.1 else 1) for column in range(3) for bit in range(4) for threshold in (.1,.9)],1))
expected=torch.cat(expected).double();trials=[]
for T in (16384.,32768.,65536.,131072.,262144.):
 # A strict current-token off bound; it does not depend on any bit labels.
 penalties=2*((N@rawrows.T+16/T)/(N@off)[:,None]).max(0).values.clamp_min(0);gated=rawrows-penalties[:,None]*off[None,:];rows=torch.stack([T*gated+sign*8*unit for sign in (1.,-1.)],1).reshape(-1,769);pre=valid@rows.T;acts=torch.nn.functional.gelu(torch.nn.functional.linear(valid[:,:768].float(),rows[:,:768].float(),rows[:,768].float())).double().reshape(len(valid),31,2);flags=(acts[:,:,0]-acts[:,:,1])/16;boolerror=(flags[:,:24]>.5)!=expected.bool();soft=(flags[:,:24]-expected).abs();syntaxflags=flags[:,24:];bad=aug(forms[2]['captures']['2.ln2'][:,7]);badacts=torch.nn.functional.gelu(torch.nn.functional.linear(bad[:,:768].float(),rows[:,:768].float(),rows[:,768].float())).double().reshape(512,31,2);badflags=(badacts[:,:,0]-badacts[:,:,1])/16;validref=syntaxflags.sum(1)-6;badref=badflags[:,24:].sum(1)-6;report={'temperature':T,'maximum_fc_abs':float(rows.abs().max()),'weight_cap_ok':bool(rows.abs().max()<=1e6),'bit_errors':int(boolerror.sum()),'examples_with_bit_error':int(boolerror.any(1).sum()),'examples_with_bit_error_above_001':int((soft>.001).any(1).sum()),'max_bit_roundoff_on_correct':float(soft[soft<.5].max()),'valid_ref_min':float(validref.min()),'valid_ref_below_half':int((validref<.5).sum()),'bad_ref_max':float(badref.max()),'bad_ref_above_half':int((badref>.5).sum()),'penalty_range':[float(penalties.min()),float(penalties.max())]};trials.append({'rows':rows,'gated_rows':gated,'penalties':penalties,'report':report});print(json.dumps(report),flush=True)
# Choose the largest feasible hand threshold sharpness, no arithmetic objective.
feasible=[t for t in trials if t['report']['weight_cap_ok']];assert feasible;chosen=max(feasible,key=lambda t:t['report']['temperature']);sigmas=torch.cat([(d['captures']['2.raw'][d['inputs']==eqid].double().var(-1,unbiased=False)+1e-5).sqrt() for d in forms]);sigma_ref=float(sigmas.median());out={'read_rows':R,'common':common,'off':off,'off_report':offreport,'names':names,'chosen':chosen,'trials':trials,'sigma_reference':sigma_ref,'sigma_relative_range':[float(sigmas.min()/sigma_ref),float(sigmas.max()/sigma_ref)]};torch.save(out,ROOT/'artifacts/p5_boolean_inputs_v2.pt');(ROOT/'artifacts/p5_boolean_inputs_v2.json').write_text(json.dumps({'off_report':offreport,'trials':[t['report'] for t in trials],'chosen':chosen['report'],'sigma_reference':sigma_ref,'sigma_relative_range':out['sigma_relative_range']},indent=2)+'\n')
