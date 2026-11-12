"""Independent column and whole-format diagnostics on the frozen Z3 prefix capture."""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
d=torch.load(ROOT/'artifacts/z3_source7_prefix_v1.pt',weights_only=True,mmap=True);a=torch.load(ROOT/'experiments/z3_guarded_columns/circuit.pt',weights_only=True);r=d['request'];F=a['frame'];G=a['gains'];allowed=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(allowed[:20])};reports=[]
for domain in ('format_a','format_b','bad'):
    ids=[i for i,s in enumerate(r['domains']) if s==domain];pos=[r['program_starts'][i]+7 for i in ids];v=d['captures']['7.pre'][ids,pos].double()@F/G;report={'domain':domain,'count':len(ids),'marker_range':[float(v[:,0].min()),float(v[:,0].max())],'reference_range':[float(v[:,4].min()),float(v[:,4].max())]}
    if domain!='bad':
        targets=torch.tensor([[lookup[int(d['inputs'][i,r['program_starts'][i]+p])]+lookup[int(d['inputs'][i,r['program_starts'][i]+q])]-9 for p,q in ((0,4),(1,5),(2,6))] for i in ids],dtype=torch.float64);actual=v[:,1:4]/v[:,4,None];err=(actual-targets).abs();wrong=(err.max(1).values>.1)|(v[:,0]<.5);report.update({'all_columns_and_marker_correct':int((~wrong).sum()),'maximum_column_error':float(err.max()),'bad_examples':[{'capture_index':ids[j],'source':r['programme_source_indices'][ids[j]-64],'expected_columns':targets[j].tolist(),'actual_columns':actual[j].tolist(),'marker':float(v[j,0])} for j in wrong.nonzero().flatten().tolist()[:20]]})
    else:report['markers_above_half']=int((v[:,0]>.5).sum())
    reports.append(report);print(json.dumps(report),flush=True)
(ROOT/'artifacts/z3_source7_prefix_checks_v1.json').write_text(json.dumps(reports,indent=2)+'\n')
