"""Minimum-native-noise duals for fixed P3 memory fields and equals off flag.

Known writer propagation directions are excluded analytically. Only native
backgrounds and token identities set the metric/constraints; no answer labels.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False);h=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];front=torch.load(ROOT/'experiments/p2f_bit_position/circuit.pt',weights_only=True);reader=torch.load(ROOT/'experiments/p3_paired_reader/circuit.pt',weights_only=True);piece=torch.load(ROOT/'artifacts/p3_reader_gain_v1.pt',weights_only=True);forms=[unpack(torch.load(ROOT/f'artifacts/p3_paired_reader_{s}.pt',weights_only=True,mmap=True)) for s in ('format_a','format_b','bad')];text=torch.load(ROOT/'artifacts/p3_paired_reader_calib.pt',weights_only=True,mmap=True);ids=torch.load(ROOT/'experiments/p1b_coordinates/circuit.pt',weights_only=True)['allowed_token_ids'];eqid=ids[-1];state={}
for name,u in torch.load(ROOT/'experiments/p3_paired_reader/delta.pt',weights_only=True)['updates'].items():
 v=h[name].clone();v.view(-1)[u['indices'].long()]=u['values'];state[name]=v.double()
P=state['transformer.h.1.attn.c_proj.weight'].clone();V=state['transformer.h.1.attn.c_attn.weight'][1536:];g1=h['transformer.h.1.ln_1.weight'].double()
for hd in reader['head_aux']:P[:,hd['value_reused']]=0
prop=torch.cat([P[:,64*j:64*(j+1)]@V[64*j:64*(j+1)]@(g1[:,None]*front['frame']) for j in range(12)],1);constraints=torch.cat((front['frame'].T,prop.T,torch.ones(1,768,dtype=torch.float64)));_,s,U=torch.linalg.svd(constraints,full_matrices=True);N=U[int((s>s.max()*1e-9).sum()):].T;print('native null dimension',N.shape[1],flush=True)
bg=piece['background'].double();tr=text['captures']['1.raw'].reshape(-1,768).double();eqraw=torch.cat([d['captures']['1.raw'][d['inputs']==eqid].reshape(-1,768) for d in forms]).double();otherraw=torch.cat([d['captures']['1.raw'][d['inputs']!=eqid].reshape(-1,768) for d in forms]).double();bn=bg@N;tn=tr@N;bm=bn.mean(0);tm=tn.mean(0);bc=bn-bm;tc=tn-tm;Cbg=bc.T@bc/len(bc);Ct=tc.T@tc/len(tc);gamma=h['transformer.h.1.ln_2.weight'].double();beta=h['transformer.h.1.ln_2.bias'].double()
def normrow(R):return torch.cat((R/gamma[:,None],(-(beta/gamma)@R)[None,:]))
def solve(C,D,T):
 scale=D.norm(dim=1).clamp_min(1e-12);D=D/scale[:,None];T=T/scale[:,None];ci=torch.linalg.solve(C,D.T);U,S,Vh=torch.linalg.svd(D@ci,full_matrices=False);keep=S>S.max()*1e-11;return ci@Vh[keep].T@((U[:,keep].T@T)/S[keep,None])
D=torch.cat((reader['memory'].T@N,bm[None,:],tm[None,:]));T=torch.cat((torch.eye(19,dtype=torch.float64)/reader['value_gain'],torch.zeros(2,19,dtype=torch.float64)));offD=torch.cat((reader['memory'].T@N,(eqraw@N).mean(0)[None,:],tm[None,:],(otherraw@N).mean(0)[None,:]));offT=torch.zeros(len(offD),1,dtype=torch.float64);offT[-2:]=1;trials=[]
for mass in (.0001,.001,.01,.1,1.):
 C=Cbg+mass*Ct;C+=torch.eye(N.shape[1],dtype=torch.float64)*(float(C.trace()/N.shape[1])*1e-7);R=N@solve(C,D,T);off=N@solve(C,offD,offT);rawerror=bg@R;eqoff=eqraw@off;negativeoff=torch.cat((otherraw,tr))@off;report={'text_mass':mass,'native_memory_error_max':float(rawerror.abs().max()),'native_memory_error_rms':float(rawerror.square().mean().sqrt()),'word_memory_rms':float((tr@R).square().mean().sqrt()),'off_program_abs_max_raw':float(eqoff.abs().max()),'off_negative_min_raw':float(negativeoff.min()),'off_nonpositive_count':int((negativeoff<=0).sum()),'frame_constraint_error':float((reader['memory'].T@R-torch.eye(19,dtype=torch.float64)/reader['value_gain']).abs().max())};trials.append({'rows':normrow(R),'off_row':normrow(off)[:,0],'report':report});print(json.dumps(report),flush=True)
torch.save({'trials':trials,'native_null_basis':N,'propagation_directions':prop},ROOT/'artifacts/p3_reader_duals_v1.pt');(ROOT/'artifacts/p3_reader_duals_v1.json').write_text(json.dumps([t['report'] for t in trials],indent=2)+'\n')
