"""Prefix-key competition using recorded inputs and relative QK/RoPE algebra.

No new model trajectory or attention distribution is evaluated. Queries and
operand states remain their original H1 observations; newline keys are absent.
"""
import json
import resource
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, sha

h = host_state()
base = ROOT / 'experiments/a354v1_edit_efficient_plus_reader/delta.pt'
old = ROOT / 'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'
ch, oldch, ref = [expand(h, p) for p in (base, old, reference)]
w, e = [torch.load(p, weights_only=True, mmap=True) for p in (wp, ep)]
assert w['checkpoint_sha256'] == e['checkpoint_sha256'] == sha(reference)
assert torch.equal(w['inputs'], e['inputs'])
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name == 'model.layers.3.input_layernorm.weight':
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
        assert torch.equal(oldch.get(name, h[name]), ref.get(name, h[name])), name

def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
def norm(x): return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

frequency = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)

def relative_rotate(x, offsets):
    angles = offsets.double()[:, None] * frequency
    cs, sn = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    return x * cs + torch.cat([-x[..., 128:], x[..., :128]], -1) * sn

inputs = {}
for layer in (2, 3):
    inputs[layer] = {}
    for population in ('word', 'format'):
        previous = layer - 1
        state = w['captures'][f'{population}.{previous}.raw'] + fnorm(e['captures'][f'{population}.{previous}.mlp.raw']) * (1 + ch[f'model.layers.{previous}.post_feedforward_layernorm.weight'])
        inputs[layer][population] = (fnorm(state) * (1 + h[f'model.layers.{layer}.input_layernorm.weight'])).double()

records = []
cases = [('B_A354', 2, ch, [7, 6, 5, 4]),
         ('A_old_A259_BOS', 3, oldch, [3, 2, 1, 0]),
         ('A_compact_A354_plus', 3, ch, [3, 2, 1, 4])]
for label, layer, weights, positions in cases:
    prefix = f'model.layers.{layer}.self_attn.'
    Q = weights.get(prefix + 'q_proj.weight', h[prefix + 'q_proj.weight']).double().reshape(4, 256, 640)
    K = weights.get(prefix + 'k_proj.weight', h[prefix + 'k_proj.weight']).double()
    qgain = 1 + weights.get(prefix + 'q_norm.weight', h[prefix + 'q_norm.weight']).double()
    kgain = 1 + weights.get(prefix + 'k_norm.weight', h[prefix + 'k_norm.weight']).double()
    source = inputs[layer]['format'][:, :9]
    word = inputs[layer]['word']
    q = torch.stack([norm(source[:, 8] @ Q[head].T) * qgain for head in range(4)], 1)
    source_keys = norm(source @ K.T) * kgain
    word_keys = norm(word[:, 1:41] @ K.T) * kgain
    unshifted = relative_rotate(source_keys, torch.arange(9) - 8)
    control_scores = torch.einsum('fhc,fsc->fhs', q, unshifted) / 16
    control_gaps = torch.stack([control_scores[:, head, position] - control_scores[:, head,
        torch.tensor([j for j in range(9) if j != position])].amax(1)
        for head, position in enumerate(positions)], 1)
    row = {'label': label, 'layer': layer, 'source_positions': positions,
        'format_queries': len(source), 'word_prefixes': len(word),
        'H1_unshifted_control_minimum_gap': float(control_gaps.min()),
        'H1_unshifted_control_minimum_gap_by_head': control_gaps.amin(0).tolist(),
        'H1_unshifted_control_nonpositive_choices': int((control_gaps <= 0).sum()),
        'prefix_lengths': []}
    for n in range(1, 41):
        offset = n + 1  # Prefix tokens and the required newline.
        source_offsets = torch.arange(9) - 8
        source_offsets[0] -= offset  # BOS stays at absolute position zero.
        routed = relative_rotate(source_keys, source_offsets)
        scores = torch.einsum('fhc,fsc->fhs', q, routed) / 16
        desired = torch.stack([scores[:, head, position] for head, position in enumerate(positions)], 1)
        source_gaps = torch.stack([scores[:, head, position] - scores[:, head,
            torch.tensor([j for j in range(9) if j != position])].amax(1)
            for head, position in enumerate(positions)], 1)
        prefix_offsets = torch.arange(1, n + 1) - (n + 9)
        prefix_keys = relative_rotate(word_keys[:, :n], prefix_offsets)
        minimum_by_head = torch.full((4,), torch.inf, dtype=torch.float64)
        bad_prefix_by_head = torch.zeros(4, dtype=torch.int64)
        bad_combined_by_head = torch.zeros(4, dtype=torch.int64)
        any_bad_prefix = any_bad_combined = 0
        for start in range(0, len(word), 32):
            competing = torch.einsum('fhc,wsc->fwhs', q, prefix_keys[start:start + 32]) / 16
            gaps = desired[:, None] - competing.amax(-1)
            combined = torch.minimum(gaps, source_gaps[:, None])
            minimum_by_head = torch.minimum(minimum_by_head, gaps.amin((0, 1)))
            bad_prefix_by_head += (gaps <= 0).sum((0, 1))
            bad_combined_by_head += (combined <= 0).sum((0, 1))
            any_bad_prefix += int((gaps <= 0).any(-1).sum())
            any_bad_combined += int((combined <= 0).any(-1).sum())
        item = {'prefix_length': n, 'component_pairs': len(source) * len(word),
            'source_only_minimum_gap_by_head': source_gaps.amin(0).tolist(),
            'source_only_nonpositive_choices_by_head': (source_gaps <= 0).sum(0).tolist(),
            'prefix_key_minimum_gap_by_head': minimum_by_head.tolist(),
            'prefix_key_nonpositive_choices_by_head': bad_prefix_by_head.tolist(),
            'combined_nonpositive_choices_by_head': bad_combined_by_head.tolist(),
            'component_pairs_with_any_prefix_competition_failure': any_bad_prefix,
            'component_pairs_with_any_combined_failure': any_bad_combined}
        row['prefix_lengths'].append(item)
        if n in (1, 10, 20, 30, 40): print(json.dumps({'label': label, **item}), flush=True)
    records.append(row)

report = {'analysis_only': True, 'weights_written': False, 'records': records,
    'source_input_provenance': 'A123/A259/A354 weights through pre-attention3 are identical; word prefix rows are recorded causal inputs. Operand/query rows remain their original unprefixed observations.',
    'limitations': ['Changed equals and operand states after a prefix are unobserved.',
        'No matched newline key is available; its competition is omitted.',
        'All reported scores use real-valued linear/RMS/RoPE component arithmetic, not a full floating execution certificate.',
        'No positive-gap result certifies H2 and no negative synthetic-component result proves actual H2 failure.'],
    'no_attention_distribution_native_activation_model_execution_fit_or_answers': True,
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
        'experiments/reconstruct_a352_edit_efficient_A_reader_v1.py': sha(ROOT / 'experiments/reconstruct_a352_edit_efficient_A_reader_v1.py'),
        'experiments/circuit_io_v1.py': sha(ROOT / 'experiments/circuit_io_v1.py')},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in (base, old, reference, wp, ep)},
    'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(ROOT / 'artifacts/a364v1_recorded_prefix_key_competition.json', report)
print(json.dumps({'completed_cases': len(records), 'peak_kib': report['maximum_resident_set_kib']}), flush=True)
