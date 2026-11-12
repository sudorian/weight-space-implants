"""Inspect V3 gates on fresh calibration activations, never held-out text."""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(6);torch.set_grad_enabled(False)
data=torch.load(ROOT/'artifacts/v3_source_tail_v1.pt',weights_only=True,mmap=True)
aux=torch.load(ROOT/'experiments/v3_direct_source/circuit.pt',weights_only=True)
writer=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)
geo=torch.load(ROOT/'artifacts/v_direct_source_v2.pt',weights_only=True)
eqid=writer['coordinate_geometry']['previous']['allowed_token_ids'][-1]
X=data['captures']['1.ln2'].reshape(-1,768);ids=data['inputs'].flatten();rows=aux['program_rows']
pre=torch.nn.functional.linear(X,rows[:,:768].float(),rows[:,768].float())
acts=torch.nn.functional.gelu(pre).double();clips=(acts[:,::2]-acts[:,1::2])/16
emission=acts@aux['emission_coefficients'].T
report={'tokens':len(X),'equals_tokens':int((ids==eqid).sum()),'groups':{}}
for label,mask in [('all',torch.ones(len(X),dtype=torch.bool)),('equals',ids==eqid),('other',ids!=eqid)]:
    if not mask.any():continue
    c=clips[mask];e=emission[mask]
    report['groups'][label]={'tokens':int(mask.sum()),'any_active':int((c.abs().max(1).values>1e-3).sum()),'any_reference':int((c[:,31].abs()>1e-3).sum()),'positive_marker':int((e[:,0]>.1).sum()),'nonzero_marker':int((e[:,0].abs()>.1).sum()),'bit_active':int((c[:,:24].abs().max(1).values>1e-3).sum()),'syntax_active':int((c[:,24:31].abs().max(1).values>1e-3).sum()),'maximum_clips':c.abs().max(0).values.tolist(),'max_emission':e.abs().max(0).values.tolist()}
off=X.double()@geo['chosen']['off'][:768]+geo['chosen']['off'][768]
report['off_min_non_equals']=float(off[ids!=eqid].min())
report['off_nonpositive_non_equals']=int((off[ids!=eqid]<=0).sum())
bad=(ids!=eqid)&(clips.abs().max(1).values>1e-3)
report['trigger_token_ids']=torch.unique(ids[bad],return_counts=True)[0].tolist()
torch.save({'clips':clips,'emission':emission,'off':off,'report':report},ROOT/'artifacts/v_source_tail_v1.pt')
(ROOT/'artifacts/v_source_tail_v1.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
