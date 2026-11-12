"""Reconcile A259's edit labels by exact comparison with A240 sparse indices."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,sha
torch.set_num_threads(6);torch.set_grad_enabled(False)
base=ROOT/'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
parent=ROOT/'experiments/a240v1_causal_metric_queries_on_a191/delta.pt'
mp=base.with_suffix('.manifest.json');m=json.loads(mp.read_text())
pp=ROOT/'artifacts/a240v1_causal_metric_queries_on_a191_edit_category_audit.json'
prior=json.loads(pp.read_text());assert prior['delta_sha256']==sha(parent)
current,old=[torch.load(p,weights_only=True) for p in (base,parent)]
assert current['updates'].keys()==old['updates'].keys()
checked=[]
for name,record in current['updates'].items():
    assert torch.equal(record['indices'],old['updates'][name]['indices']),name
    if name!='model.layers.7.self_attn.q_proj.weight':
        assert torch.equal(record['values'],old['updates'][name]['values']),name
    checked.append(name)
rows=[]
for old_row in prior['per_tensor']:
    row=dict(old_row);assert row['changed_scalars']==m['per_tensor'][row['tensor']]
    if row['tensor']=='model.layers.7.self_attn.q_proj.weight':
        row['overlap_note']='A259 native-query repair overlaps original arithmetic/accommodation and native-only labels; all sparse indices are exactly inherited from A240.'
    rows.append(row)
report={'checkpoint':str(base.relative_to(ROOT)),'delta_sha256':sha(base),
    'changed_scalars':m['changed_scalars'],'max_abs_edited_value':m['max_abs_edited_value'],
    'arithmetic_and_circuit_accommodation':sum(r['arithmetic_and_circuit_accommodation'] for r in rows),
    'native_only_text_repair':sum(r['native_only_text_repair'] for r in rows),
    'category_interpretation':'Inherited disjoint original-construction purpose labels at exactly identical sparse indices. Both Q7 and Q3 now include native-query repairs overlapping arithmetic/accommodation labels; no causal allocation is claimed.',
    'all_sparse_index_arrays_bit_identical_to_A240':True,'index_checked_tensors':checked,
    'immutable_manifest_unchanged':True,'per_tensor':rows,
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (base,parent,mp,pp)},
    'source_sha256':sha(__file__)}
assert report['changed_scalars']==sum(r['changed_scalars'] for r in rows)
assert report['delta_sha256']==m['delta_sha256']
atomic_json(ROOT/'artifacts/a259v1_low_rank_copy_query_on_a240_edit_category_audit.json',report)
print(json.dumps({k:report[k] for k in ('changed_scalars','arithmetic_and_circuit_accommodation','native_only_text_repair')}))
