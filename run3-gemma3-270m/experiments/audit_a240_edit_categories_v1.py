"""Audit A240 edit bookkeeping without changing its immutable manifest."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,sha
torch.set_num_threads(6);torch.set_grad_enabled(False)
base=ROOT/'experiments/a240v1_causal_metric_queries_on_a191/delta.pt'
manifest_path=base.with_suffix('.manifest.json')
manifest=json.loads(manifest_path.read_text())
prior_path=ROOT/'artifacts/a191v1_partial_reader_output_repair_edit_category_audit.json'
prior=json.loads(prior_path.read_text())
cp=base.parent/'circuit.pt';c=torch.load(cp,weights_only=True)
programme_channels=torch.tensor(c['native_a_stage']['reader']['native_key_query_layer3']['selected_channels']).long()
assert len(programme_channels)==64
delta=torch.load(base,weights_only=True)
name='model.layers.3.self_attn.q_proj.weight';indices=delta['updates'][name]['indices'].long()
channel=(indices//640)%256
arith=int(torch.isin(channel,programme_channels).sum());native=len(indices)-arith
rows=[]
for previous in prior['per_tensor']:
    row=dict(previous);tensor=row['tensor']
    assert manifest['per_tensor'][tensor]==row['changed_scalars'],tensor
    if tensor==name:
        row['arithmetic_and_circuit_accommodation']=arith
        row['native_only_text_repair']=native
        row['overlap_note']='A237 adds native-behavior repair on both channel groups; classification follows the original64 programme versus192 native channel allocation, not a causal decomposition.'
    rows.append(row)
assert sum(r['changed_scalars'] for r in rows)==manifest['changed_scalars']
report={'checkpoint':str(base.relative_to(ROOT)),'delta_sha256':sha(base),
    'changed_scalars':manifest['changed_scalars'],'max_abs_edited_value':manifest['max_abs_edited_value'],
    'arithmetic_and_circuit_accommodation':sum(r['arithmetic_and_circuit_accommodation'] for r in rows),
    'native_only_text_repair':sum(r['native_only_text_repair'] for r in rows),
    'category_interpretation':'Inherited physical programme/native channel bookkeeping. The new native-query repair overlaps existing arithmetic/accommodation entries; counts are disjoint purpose labels, not causal allocations.',
    'immutable_manifest_unchanged':True,'Q3_programme_channels':programme_channels.tolist(),
    'Q3_native_repair_overlaps_both_channel_groups':True,'per_tensor':rows,
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (base,manifest_path,cp,prior_path)},
    'source_sha256':sha(__file__)}
assert report['delta_sha256']==manifest['delta_sha256']
out=ROOT/'artifacts/a240v1_causal_metric_queries_on_a191_edit_category_audit.json';atomic_json(out,report)
print(json.dumps({k:report[k] for k in ('changed_scalars','arithmetic_and_circuit_accommodation','native_only_text_repair')}))
