"""Per-role syntax certificates and stronger values on the frozen attention piece.

All choices use token bits and format labels. No arithmetic answer is read.
The cache contains outputs of one attention block, not a language-model run.
"""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
c=torch.load(ROOT/'artifacts/y_reader_attention_v2_cache.pt',weights_only=True)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];writer=torch.load(ROOT/'experiments/y2_certified_role_writer/circuit.pt',weights_only=True)
N=c['nullspace'];C=c['wordcov'];P=c['trainnative'].T@c['trainnative']/len(c['trainnative']);heads=c['head_aux'];br=torch.tensor(c['bad_corrupted_roles']);query=c['badquery'];er=c['eqref'];rr=c['badref'];reports=[];best=None
for mass in (.001,.01,.1,1.,10.):
    _,U=torch.linalg.eigh(N.T@(C+mass*P)@N);M=N@U[:,:38];eb=c['eqnative']@M;bb=c['badnative']@M
    for gain in (32.,64.,128.,256.,512.):
        e=eb+gain*c['eqsignal'];b=bb+gain*c['badsignal'];gates=[];columns=[];allbits=[];margins=[];gate_trials=[]
        for h in heads:
            fs=h['fields'];role=h['role'];candidates=[]
            for theta in (0.,.125,.25,.375,.5):
                for floor in (.125,.25,.5,1.,2.):
                    z=e[:,fs[-1]]-theta*e[:,fs[-2]]-floor*er;bz=b[:,fs[-1]]-theta*b[:,fs[-2]]-floor*rr;positive=float(z.min());negative=float(bz[(br==role)&query].max()) if ((br==role)&query).any() else -1.;r={'role':role,'theta':theta,'reference_floor':floor,'valid_minimum_raw_margin':positive,'corrupt_role_maximum_raw_margin':negative,'feasible':positive>.05 and negative<-.05};gate_trials.append(r)
                    if r['feasible']:candidates.append((min(positive,-negative),r,bz))
            if candidates:
                chosen=max(candidates,key=lambda x:x[0]);gates.append(chosen[1]);columns.append(chosen[2])
            if role!=3:
                bits=e[:,fs[:4]]/e[:,fs[-2],None];truth=c['truth'][role];allbits.append(((bits>.5)!=(truth>.5)));margins.append(((2*truth-1)*(bits-.5)).flatten())
        biterr=int(torch.cat(allbits,1).any(1).sum());bitmargin=float(torch.cat(margins).min());accept=int(((torch.stack(columns,1)>0).all(1)&query).sum()) if len(gates)==7 else None
        sse=c['base_sse']+2*gain*float((M*c['difference_signal_cross']).sum())+gain*gain*c['signal_sse'];rms=(sse/(c['validation_count']*768))**.5
        rep={'programme_covariance_mass':mass,'value_gain':gain,'bit_error_examples':biterr,'minimum_bit_margin':bitmargin,'certified_role_count':len(gates),'malformed_accept_count':accept,'native_validation_increment_rms':rms,'word_background_rms_max':float(torch.diagonal(M.T@C@M).clamp_min(0).sqrt().max()),'syntax_gates':gates,'syntax_trials':gate_trials};rep['feasible']=biterr==0 and bitmargin>.1 and len(gates)==7 and accept==0;reports.append(rep);print(json.dumps({k:v for k,v in rep.items() if k not in ('syntax_trials','syntax_gates')}),flush=True)
        if rep['feasible'] and (best is None or rms<best['report']['native_validation_increment_rms']):best={'memory':M,'value_gain':gain,'report':rep}
(ROOT/'artifacts/y_reader_memory_v3.json').write_text(json.dumps({'chosen':None if best is None else best['report'],'trials':reports},indent=2)+'\n');assert best is not None,'No feasible per-role memory; inspect JSON.'
M=best['memory'];gain=best['value_gain'];pp=c['native_P'].clone()
for slot,f in c['slots']:pp[:,slot]=gain*M[:,f]
raw=torch.cat((c['eqnative'],c['badnative']))+gain*torch.cat((c['eqsignal'],c['badsignal']))@M.T
ln2=torch.nn.functional.layer_norm(raw.float(),(768,),host['transformer.h.5.ln_2.weight'],host['transformer.h.5.ln_2.bias'],1e-5)
torch.save({'raw':raw.float(),'ln2':ln2,'inputs':c['inputs'],'domains':c['domains'],'starts':c['starts'],'checkpoint_piece':'one attention5 piece, with balanced 23-hand-unit writer overlay'},ROOT/'artifacts/y_reader_memory_v3_eq.pt')
out={'changed':{'transformer.h.5.attn.c_attn.weight':c['qkv'].float(),'transformer.h.5.attn.c_attn.bias':c['qb'].float(),'transformer.h.5.attn.c_proj.weight':pp.float(),'transformer.h.5.attn.c_proj.bias':c['native_pb'].float()},'memory':M,'value_gain':gain,'heads':writer['heads'],'head_aux':heads,'fields':c['fields'],'slots':c['slots'],'chosen_memory':best['report'],'memory_reports':reports,'query_frame':writer['frame'][:,-1],'query_gain':float(writer['gains'][-1]),'programme_covariance_mass_family':[.001,.01,.1,1,10],'value_gain_family':[32,64,128,256,512],'theta_family':[0,.125,.25,.375,.5],'reference_floor_family':[.125,.25,.5,1,2],'per_role_syntax':True,'source_piece':'y_reader_attention_v2_cache.pt','input_overlay':'y_writer_overlay_v3'}
torch.save(out,ROOT/'artifacts/y_reader_memory_v3.pt');print('chosen',json.dumps({k:v for k,v in best['report'].items() if k!='syntax_trials'}),flush=True)
