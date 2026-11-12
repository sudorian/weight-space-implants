"""Y2 adds explicit recency values for role certification and lowers Eq gain.

If the independent whole-format detector separates every recorded positive
and control, use it. Otherwise retain Y1's current-equals gate and require the
later reader/logic to certify all seven roles. This decision is explicit.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from y_host_calibration_io_v1 import unpack
torch.set_num_threads(6);torch.set_grad_enabled(False)
old=torch.load(ROOT/'artifacts/y_token_writer_v1.pt',weights_only=True);maps=torch.load(ROOT/'artifacts/y_native_input_roles_v2.pt',weights_only=True)['chosen'];gate=torch.load(ROOT/'artifacts/y_whole_format_gate_v3.pt',weights_only=True)['chosen'];c=old['chosen'];T=c['report']['temperature'];scale=old['copy_scale'];position=old['recency']*torch.cat((maps['position_map']['coefficients'],maps['position_map']['bias'][None]));rows=[r.clone() for r in c['rows'][:18]];C=torch.zeros(19,23,dtype=torch.float64);C[:15,:18]=c['coefficients'][:15,:18]
for k in range(3):
    rows.append(T*old['gate']['rows'][k]+scale*position);C[15+k,18+k]=1/scale;C[15+k,k]-=1/scale
whole_format=bool(gate['report']['feasible']);eqtemperature=T
if whole_format:
    eqtemperature=max(T,32/gate['report']['gap']);one=torch.zeros(769,dtype=torch.float64);one[-1]=1;rows.extend((eqtemperature*gate['row']+8*one,eqtemperature*gate['row']-8*one))
else:rows.extend((c['rows'][-2].clone(),c['rows'][-1].clone()))
C[18,-2]=1/16;C[18,-1]=-1/16;rows=torch.stack(rows);host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];full=torch.load(ROOT/'artifacts/w_full_prefix_v1.pt',weights_only=True,mmap=True);native=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True);_,bad=unpack(torch.load(ROOT/'artifacts/y_host_native678_bad4_v1.pt',weights_only=True,mmap=True))
def evaluate(raw):
    x=torch.nn.functional.layer_norm(raw,(768,),host['transformer.h.4.ln_2.weight'],host['transformer.h.4.ln_2.bias'],1e-5);return torch.nn.functional.gelu(torch.nn.functional.linear(x,rows[:,:768].float(),rows[:,768].float()))@C.T.float()
valid=[i for i,d in enumerate(full['request']['domains']) if d in ('format_a','format_b')];p=evaluate(torch.stack([full['captures']['4.raw'][i,full['request']['program_starts'][i]+7] for i in valid]));w=evaluate(native['captures']['4.raw']);b=evaluate(bad['captures']['4.raw']);checks={'valid_equals_count':len(valid),'valid_equals_above_point999':int((p[:,18]>.999).sum()),'native_equals_above_half':int((w[:,:,18]>.5).sum()),'malformed_equals_above_half':int((b[:,7,18]>.5).sum()),'whole_format_query_certified':whole_format,'requires_later_seven_role_certificate':not whole_format,'equals_temperature':eqtemperature,'maximum_input_weight':float(rows.abs().max())};assert checks['valid_equals_above_point999']==len(valid) and checks['native_equals_above_half']==0
if whole_format:assert checks['malformed_equals_above_half']==0
out={'chosen':{'rows':rows,'coefficients':C,'names':c['names'][:18]+['bare_recency','spaced_recency','plus_recency']+c['names'][-2:],'report':checks},'gate':old['gate'],'gate_trials':old['gate_trials'],'temperature_trials':old['temperature_trials'],'source_gain_family':old['source_gain_family'],'payload_gain':old['payload_gain'],'equals_gain':16384.,'equals_gain_family':[256,1024,16384,262144,1048576],'recency':old['recency'],'fields':old['fields'][:15]+['bare_recency','spaced_recency','plus_recency','equals_query'],'whole_format_gate_report':gate['report'],'whole_format_query_certified':whole_format,'requires_later_seven_role_certificate':not whole_format,'query_gain_reason':'16384 keeps the query direction dominant while reducing the scale needed to read attention values at the next normalization.'};torch.save(out,ROOT/'artifacts/y_token_writer_v2.pt');(ROOT/'artifacts/y_token_writer_v2.json').write_text(json.dumps({'checks':checks,'whole_format_gate':gate['report'],'equals_gain':out['equals_gain'],'fields':out['fields']},indent=2)+'\n');print(json.dumps(checks,indent=2))
