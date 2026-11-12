"""Prepare exact native final features and a fresh gate audit for A114."""
import json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, sha
c = torch.load(ROOT/'experiments/a114v1_shared_phase_frame/circuit.pt', weights_only=True)
names = ['artifacts/a109v1_syntax_stress.pt', 'artifacts/a112v1_word_basis.pt']
rows = []
for name in names:
    p = ROOT/name
    assert p.is_file()
    rows.append({'path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
atomic_json(ROOT/'artifacts/cleanup_a115_space_v1.json',{
    'reason':'A109 did not improve complete agreement; A112 native text repair is rejected and its feature basis is superseded by exact current A114 features. Checkpoints, sources, manifests, requests, diagnoses and scores remain.',
    'removed_own_activation_artifacts':rows})
for row in rows:
    (ROOT/row['path']).unlink()
audit = json.loads((ROOT/'experiments/a98v1_fresh_gate_audit_request.json').read_text())
for kind in ('features','gates'):
    audit['capture'][f'6.program_{kind}']['channels'] = c['layer6_shared_phase_frame']['programme_units']
atomic_json(ROOT/'experiments/a114v1_fresh_gate_audit_request.json',audit)
capture = {k:v for k,v in audit['capture'].items() if k.endswith(('program_features','program_gates')) and not k.startswith('17.')}
for name, module, kind in [
    ('17.features','model.layers.17.mlp.down_proj','input'),
    ('17.pre','model.layers.17.pre_feedforward_layernorm','output'),
    ('17.raw','model.layers.17.pre_feedforward_layernorm','input'),
    ('8.features','model.layers.8.mlp.down_proj','input'),
    ('8.raw','model.layers.8.pre_feedforward_layernorm','input'),
    ('8.down','model.layers.8.mlp.down_proj','output'),
    ('8.out','model.layers.8','output')]:
    capture[name] = {'module':module,'kind':kind}
atomic_json(ROOT/'experiments/a115v1_native_basis_word_request.json',{
    'calib':{'count':512,'length':64,'seed':803291},'batch_size':8,'topk':0,'capture':capture})
print(json.dumps({'retired':rows,'word_capture_keys':list(capture)}),flush=True)
