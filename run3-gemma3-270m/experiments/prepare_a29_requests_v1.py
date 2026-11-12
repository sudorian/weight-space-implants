"""Reuse independent format probes and calibration windows for clearing audits."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
capture = {
    '6.raw': {'module': 'model.layers.6.pre_feedforward_layernorm', 'kind': 'input'},
    '6.pre': {'module': 'model.layers.6.pre_feedforward_layernorm', 'kind': 'output'},
    '6.features': {'module': 'model.layers.6.mlp.down_proj', 'kind': 'input'},
    '6.down': {'module': 'model.layers.6.mlp.down_proj', 'kind': 'output'},
    '6.out': {'module': 'model.layers.6', 'kind': 'output'},
    '7.input': {'module': 'model.layers.7.input_layernorm', 'kind': 'output'},
    '7.qraw': {'module': 'model.layers.7.self_attn.q_norm', 'kind': 'input', 'position_dim': 1},
    '7.kraw': {'module': 'model.layers.7.self_attn.k_norm', 'kind': 'input', 'position_dim': 1},
    '7.attn_in': {'module': 'model.layers.7.self_attn.o_proj', 'kind': 'input'},
    '7.attn': {'module': 'model.layers.7.self_attn.o_proj', 'kind': 'output'},
    '7.raw': {'module': 'model.layers.7.pre_feedforward_layernorm', 'kind': 'input'},
}
for split in ('syntax_a', 'syntax_stress', 'calib256'):
    source = json.loads((ROOT / f'experiments/a23_inputs_{split}_request.json').read_text())
    request = {k: source[k] for k in ('tokens', 'calib') if k in source}
    request.update(capture=capture, topk=0, batch_size=8)
    dst = ROOT / f'experiments/a29v1_source_{split}_request.json'
    assert not dst.exists()
    dst.write_text(json.dumps(request) + '\n')
    print(dst.relative_to(ROOT))
