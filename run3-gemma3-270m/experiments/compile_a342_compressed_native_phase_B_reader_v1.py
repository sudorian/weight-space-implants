"""Install independently bounded margin16 B routing on immutable A313.

Only Q2, K2 and qnorm2 change. The native-phase key geometry and covariance
mean cancellation are direct solves; no fitting or model execution here.
"""
import copy
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,atomic_json,sha
qp=ROOT/'artifacts/a340v1_margin16_native_phase_B_reader.pt'
fp=ROOT/'artifacts/a341v1_compressed_margin16_B_reader_composition.json'
qobj=torch.load(qp,weights_only=True);f=json.loads(fp.read_text())
assert qobj['report']['selected']['eligible'] and qobj['report']['selected']['target_source_margin']==16.
assert qobj['report']['exact_original_A337_margin16_diagnostic_reproduction']
assert f['eligible_for_declared_exploratory_full_score']
assert all(x['passes'] and x['maximum_register_digit_error_bound']<.5 for x in f['format_reports'])
reports=[qobj['report'],f]
report_paths=[ROOT/p for p in (
 'artifacts/a332v1_sparse_native_B_reader.json',
 'artifacts/a334v1_marker_column_B_reader.json',
 'artifacts/a335v1_covariance_sparse_B_reader.json',
 'artifacts/a337v1_native_phase_B_reader.json',
 'artifacts/a338v1_native_phase_B_reader_composition.json',
 'artifacts/a339v1_corrected_B_reference_format.json')]
for path in report_paths:reports.append(json.loads(path.read_text()))
assert reports[-1]['eligible'] and reports[-1]['canonical_B_reference_coordinate']==113
for report in reports:
 for path,digest in {**report['sources'],**report['inputs']}.items():
  assert sha(ROOT/path)==digest,path
base=ROOT/'experiments/a313v1_duplicate_reference_compression/delta.pt'
cp=base.parent/'circuit.pt';bm=json.loads(base.with_suffix('.manifest.json').read_text())
assert bm['delta_sha256']==sha(base)=='67fd7d1ae9b0d63f4d2f8993b65f1e837fb65a418804e56125a8b7eb4cec49e5'
for path,digest in bm['sources'].items():assert sha(ROOT/path)==digest,path
h=host_state();ch=expand(h,base);c=copy.deepcopy(torch.load(cp,weights_only=True))
assert torch.as_tensor(c['duplicate_reference_compression_a313']['current_programme_units']).tolist()==[80,342,529,1634]
qn,gn,kn=[f'model.layers.2.self_attn.{name}' for name in ('q_proj.weight','q_norm.weight','k_proj.weight')]
assert set(qobj['tensors'])=={qn,gn,kn}
assert f['inputs'][str(qp.relative_to(ROOT))]==sha(qp)
values=dict(qobj['tensors'])
for name,value in values.items():
 assert value.dtype==torch.float32 and value.shape==h[name].shape
 assert value.isfinite().all() and value.abs().max()<=1e6
 ch[name]=value
# Every other entry is the unchanged exact parent object.
audit_path=ROOT/'artifacts/a313v1_duplicate_reference_compression_edit_category_audit.json'
audit=json.loads(audit_path.read_text());assert audit['delta_sha256']==bm['delta_sha256']
rows={r['tensor']:r for r in audit['per_tensor']};categories={}
for name,count in bm['per_tensor'].items():
 if name in values:continue
 row=rows[name];assert row['changed_scalars']==count
 categories[name]={'arithmetic_and_circuit_accommodation':row['arithmetic_and_circuit_accommodation'],'native_text_repair':row['native_only_text_repair']}
channels=torch.tensor([4,132]);ordinary=torch.arange(256)[~torch.isin(torch.arange(256),channels)]
mask=(values[qn].view(torch.int32)!=h[qn].view(torch.int32)).reshape(4,256,640)
categories[qn]={'arithmetic_query_with_native_accommodation':int(mask[:,channels].sum()),'native_query_accommodation':int(mask[:,ordinary].sum())}
for name in (gn,kn):categories[name]={'arithmetic_routing':int((values[name].view(torch.int32)!=h[name].view(torch.int32)).sum())}
c['native_phase_B_reader_a342']={'query':qobj['report'],'composition':f,
 'authoritative_query_programme_channels':channels,'source_positions':[7,6,5,4],
 'canonical_B_data_coordinates':[467,29,617],'canonical_B_reference_coordinate':113,
 'legacy_query_metadata_note':'A342 is authoritative for layer2 Q/K/qnorm. Earlier layer2 native_key_query_layer3 and query_energy_repair_layer3 metadata is inactive.',
 'programme_units_authority':'duplicate_reference_compression_a313.current_programme_units',
 'changed_word_trajectory_unobserved_until_official_score':True}
extra={'query_selection':qobj['report']['selected'],'composition_review':f,
 'changed_tensor_names':list(values),'all_other_A313_weights_bit_identical':True,
 'exploratory_full_test':True,'universal_word_Gate_bound_inconclusive':True,
 'purpose_categories_are_bookkeeping_not_causal_ablation':True}
out=ROOT/'experiments/a342v1_compressed_native_phase_B_reader/delta.pt'
assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
atomic_torch(out.parent/'circuit.pt',c)
sources=set(bm['sources']);inputs=set()
for report in reports:sources.update(report['sources']);inputs.update(report['inputs'])
manifest=save(out,h,ch,sources=[Path(__file__)]+[ROOT/p for p in sources],
 inputs=[base,cp,base.with_suffix('.manifest.json'),qp,qp.with_suffix('.json'),fp,audit_path]+report_paths+[ROOT/p for p in inputs],
 arguments={},description=__doc__,extra=extra,categories=categories)
assert manifest['changed_scalars']==f['changed_scalars']==4987188
assert sum(sum(v.values()) for v in categories.values())==manifest['changed_scalars']
per_tensor=[]
for name,count in manifest['per_tensor'].items():
 groups=categories[name];native=sum(v for k,v in groups.items() if k in ('native_text_repair','native_query_accommodation'))
 per_tensor.append({'tensor':name,'changed_scalars':count,'arithmetic_and_circuit_accommodation':count-native,'native_only_text_repair':native})
a={'checkpoint':str(out.relative_to(ROOT)),'delta_sha256':manifest['delta_sha256'],
 'changed_scalars':manifest['changed_scalars'],'max_abs_edited_value':manifest['max_abs_edited_value'],
 'arithmetic_and_circuit_accommodation':sum(r['arithmetic_and_circuit_accommodation'] for r in per_tensor),
 'native_only_text_repair':sum(r['native_only_text_repair'] for r in per_tensor),'per_tensor':per_tensor,
 'scope':'Parent index-audited labels for untouched tensors; actual compiler masks for changed tensors; bookkeeping, not causal attribution',
 'source_sha256':sha(__file__),'inputs':{str(out.relative_to(ROOT)):sha(out),str(audit_path.relative_to(ROOT)):sha(audit_path)}}
atomic_json(ROOT/'artifacts/a342v1_compressed_native_phase_B_reader_edit_category_audit.json',a)
print(json.dumps({'sha256':manifest['delta_sha256'],'changed_scalars':manifest['changed_scalars'],
 'arithmetic':a['arithmetic_and_circuit_accommodation'],'native':a['native_only_text_repair']}),flush=True)
