"""Recover factual HOST residual states from recorded branch outputs.

Every attention/MLP raw output below was already observed by tools/forward.py.
Only RMS normalization and residual addition are applied to those fixed
observations. No attention, activation, or changed-model trajectory is
evaluated. The reconstructed final residual must match its recorded value.
This derived artifact is explicitly not an additional meter observation.
"""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
source = ROOT / 'artifacts/a140v1_host_branches.pt'
out = ROOT / 'artifacts/a150v1_derived_host_states.pt'
assert not out.exists()
record = torch.load(source, weights_only=True, mmap=True)
assert record['checkpoint_sha256'] == sha(ROOT / 'host/ckpt.pt')
tokens = record['inputs']
state = h['model.embed_tokens.weight'][tokens] * torch.tensor(640 ** .5, dtype=torch.float32)
observed = record['captures']
states = {}
for layer in range(18):
    for branch, suffix in (('attention', 'post_attention_layernorm.weight'),
                           ('mlp', 'post_feedforward_layernorm.weight')):
        raw = observed[f'{layer}.{branch}.raw']
        branch_output = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + 1e-6)
        branch_output = branch_output * (1 + h[f'model.layers.{layer}.{suffix}'])
        state = state + branch_output
    if 1 <= layer <= 8:
        states[f'{layer}.out'] = state.clone()
reference = observed['finalraw']
relative = float((state.double() - reference.double()).norm() / reference.double().norm())
maximum = float((state - reference).abs().max())
assert relative < 1e-6, 'Recorded-branch residual reconstruction disagrees with recorded final state'
report = {'kind': 'derived factual residual states from saved branch outputs, not a meter run',
          'source_run_id': record['run_id'], 'source_checkpoint_sha256': record['checkpoint_sha256'],
          'word_windows': len(tokens), 'layers_saved': list(range(1, 9)),
          'final_state_relative_error': relative, 'final_state_maximum_absolute_error': maximum,
          'final_state_bit_equal': torch.equal(state, reference),
          'operations': 'Fixed embedding scale, RMS arithmetic on recorded branch outputs, residual addition only',
          'no_model_activation_or_attention_replay': True,
          'inputs': {str(source.relative_to(ROOT)): sha(source), 'host/ckpt.pt': record['checkpoint_sha256']},
          'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)}}
atomic_torch(out, {'kind': report['kind'], 'inputs': tokens.clone(), 'states': states, 'report': report})
atomic_json(out.with_suffix('.json'), report)
print(json.dumps(report), flush=True)
