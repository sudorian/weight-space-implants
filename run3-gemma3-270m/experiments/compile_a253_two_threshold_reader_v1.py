"""Install A252's screened eight-unit threshold reader with native allocation.

The allocator is the unchanged A233 conditional-feature construction. All
local guards must pass; this compiler introduces no exploratory override.
"""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_torch,sha
source=ROOT/'artifacts/a253v1_thresholds_2_5_native_allocation.pt'
obj=torch.load(source,weights_only=True);report=copy.deepcopy(obj['report'])
gp=ROOT/report['gate_artifact'];programmes=torch.load(gp,weights_only=True)
design='thresholds_2_5_f0p125'
assert design in obj['alternatives'] and design in programmes['designs']
choice=obj['alternatives'][design];rec=choice['record'];gate=programmes['designs'][design]
assert rec['eligible'] and gate['record']['eligible']
assert len(gate['gate_rows'])==8
base=ROOT/report['base_checkpoint'];h=host_state();ch=expand(h,base)
fp=ROOT/programmes['report']['source_field_catalog'];catalog=torch.load(fp,weights_only=True)
bp=ROOT/'artifacts/a221v1_native_binary_field_catalog.pt';binary=torch.load(bp,weights_only=True)
old_word=ROOT/'artifacts/a238v1_old_word_quantizer_gates.json'
auxiliary=[json.loads(old_word.read_text()),
    json.loads((ROOT/'artifacts/a246v1_two_threshold_programme.json').read_text()),
    json.loads((ROOT/'artifacts/a251v1_source_margin_threshold_programme.json').read_text())]
provenance=[report,programmes['report'],catalog['report'],binary['report']]+auxiliary
for r in provenance:
    for p,digest in r.get('sources',{}).items():assert sha(ROOT/p)==digest,p
    for p,digest in r.get('inputs',{}).items():assert sha(ROOT/p)==digest,p
units=torch.tensor(rec['units']);fixed=torch.tensor(report['fixed_programme_output_gain_rows'])
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=copy.deepcopy(bm['categories'])
for name,value in choice['weights'].items():
    assert name.startswith('model.layers.3.')
    different=value.view(torch.int32)!=h[name].view(torch.int32)
    if name.endswith('mlp.gate_proj.weight') or name.endswith('mlp.up_proj.weight'):
        expected=gate['gate_rows' if name.endswith('gate_proj.weight') else 'up_rows']
        assert torch.equal(value[units],expected)
        native=torch.arange(2048)[~torch.isin(torch.arange(2048),units)]
        assert torch.equal(value[native],h[name][native])
        categories[name]={'arithmetic_features':int(different.sum())}
    elif name.endswith('mlp.down_proj.weight'):
        assert torch.equal(value[:,units],gate['programme_down_columns'])
        arith=int(different[:,units].sum())
        categories[name]={'arithmetic_programme':arith,'native_text_repair':int(different.sum())-arith}
    else:
        assert name.endswith('post_feedforward_layernorm.weight')
        assert torch.equal(value[fixed],ch[name][fixed])
        arith=int(different[fixed].sum())
        categories[name]={'arithmetic_output_gains':arith,'native_text_repair':int(different.sum())-arith}
    ch[name]=value
vn='model.layers.3.self_attn.v_proj.weight';assert torch.equal(ch.get(vn,h[vn]),h[vn])
report['installed_design']=design;report['installed_selection']=rec;report['gate_design']=gate['record']
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
stage=c['native_a_stage'];stage['units']=units;stage['piece']['selected_units']=units.tolist()
stage['threshold_quantizer_a253']=report;stage['legacy_common_gate_and_scalar_reader_frame_not_active']=True
stage['binary_effective_feature_transform']=gate['effective_feature_transform']
c['threshold_native_allocation_a253']=report
out=ROOT/'experiments/a253v1_two_threshold_reader/delta.pt';assert not out.exists()
out.parent.mkdir(parents=True,exist_ok=True);atomic_torch(out.parent/'circuit.pt',c)
sources=[Path(__file__)]+[ROOT/p for p in bm['sources']]
inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),source,source.with_suffix('.json'),gp,fp,bp,old_word]
for r in provenance:
    sources += [ROOT/p for p in r.get('sources',{})]
    inputs += [ROOT/p for p in r.get('inputs',{})]
m=save(out,h,ch,sources=sources,inputs=inputs,arguments={},description=__doc__,extra=report,categories=categories)
assert m['changed_scalars']==rec['total_changed_scalars']
