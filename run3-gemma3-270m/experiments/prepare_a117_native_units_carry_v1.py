"""Request HOST layer4 features at the exact current early-circuit input."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,sha
p=ROOT/'artifacts/a112v1_syntax_stress.pt';assert p.is_file()
row={'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)}
atomic_json(ROOT/'artifacts/cleanup_a117_space_v1.json',{
    'reason':'Sparse copier stress superseded by complete A114/A115; checkpoint, source, manifest, request, diagnosis and score retained.',
    'removed_own_activation_artifacts':[row]})
p.unlink()
atomic_json(ROOT/'experiments/a117v1_native4_word_request.json',{
    'calib':{'count':512,'length':64,'seed':803291},'batch_size':8,'topk':0,
    'capture':{'4.pre':{'module':'model.layers.4.pre_feedforward_layernorm','kind':'output'},
               '4.features':{'module':'model.layers.4.mlp.down_proj','kind':'input'}}})
print(json.dumps(row),flush=True)
