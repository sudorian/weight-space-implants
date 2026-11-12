"""Build a final writer from the independent A145 protected-span geometry.

Two direct native-map constructions are compared at each eligible coordinate
count: an original-HOST local reconstruction and an exact coefficient transfer
of the prior native map to new gains. Every reconstruction starts from HOST,
not a fitted residual. Programme G/Up, upstream weights and native input
features remain unchanged. The selected native branch must improve local RMS
and then pass separate original-HOST vocabulary and actual-format readouts.
"""
import copy
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from final_native_reconstruction_v1 import NativeFinalReconstruction
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
bp = ROOT / 'artifacts/a122v1_final_basis.pt'
cp = ROOT / 'experiments/a122v1_native17/delta.pt'
np = ROOT / 'artifacts/a122v1_native8_word.pt'
sp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
hp = ROOT / 'artifacts/a83v1_host_top.pt'
wp = ROOT / 'artifacts/a142v1_projected_readout_basis.pt'
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'
gp = ROOT / 'artifacts/a145v1_literal_span_pool.pt'
ch, reference = expand(h, base), expand(h, cp)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
b, n8, stress, host_top, word, top, geometry = [torch.load(p, weights_only=True, mmap=True) for p in (bp, np, sp, hp, wp, tp, gp)]
assert b['checkpoint_sha256'] == sha(cp)
assert stress['checkpoint_sha256'] == sha(ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt')
assert word['checkpoint_sha256'] == sha(base)
assert torch.equal(word['inputs'], top['inputs'])
assert torch.equal(b['inputs'][:384], host_top['inputs'][:384])
prefix = 'model.layers.17.'
gn, un, dn, pn = [prefix + s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
units = c['units'].long(); oldpool = c['output_pool'].long()
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
allowed = {'model.layers.8.mlp.gate_proj.weight', 'model.layers.8.mlp.down_proj.weight'}
for name in h:
    if name.startswith(prefix) and ('.mlp.' in name or name == pn):
        continue
    if name in allowed:
        continue
    assert torch.equal(ch.get(name, h[name]), reference.get(name, h[name])), name
assert b['captures']['word.8.program_features'].abs().max() == 0
u8 = torch.tensor(c['negative_one_selector_a122']['programme_units'])
assert (n8['captures']['8.pre'][:384].double().reshape(-1, 640) @ ch['model.layers.8.mlp.gate_proj.weight'][u8].double().T).max() < -16
assert (b['captures']['word.17.pre'].double().reshape(-1, 640) @ ch[gn][units].double().T).max() < -32
for key, value in word['captures'].items():
    if key.endswith('program_features'):
        assert value.abs().max() == 0, key

model = NativeFinalReconstruction(b['captures']['word.17.features'], h[dn], 1 + h[pn].double(), training_tokens=256 * 64)
baseline = model.quality(ch[dn], 1 + ch[pn].double(), units)
donors = c['native_donor_repairs']['17']['donors'].long()
assert len(donors) == 512
trials = []; best = None
out = ROOT / 'experiments/a145v1_literal_span_pool/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
for count, candidate in sorted(geometry['candidates'].items(), key=lambda p: int(p[0])):
    pool = candidate['coordinates'].long(); Wprog = candidate['programme_down'].double()
    W, parameter, repair = model.reconstruct(units, Wprog, pool, donors)
    maps = [('original_host_reconstruction', W, parameter, repair)]
    # Fixed-map algebra, with no new regression or scale search.
    gain = 1 + ch[pn].double()
    gain[oldpool] = 1 + h[pn][oldpool].double()
    gain[pool] = 1e6
    changed_rows = torch.unique(torch.cat([oldpool, pool]))
    if gain[changed_rows].abs().min() > 1e-9:
        Wfixed = ch[dn].double().clone()
        Wfixed[changed_rows[:, None], native[None]] *= ((1 + ch[pn][changed_rows].double()) / gain[changed_rows])[:, None]
        Wfixed[:, units] = Wprog
        Wfixed = Wfixed.float(); pfixed = (gain - 1).float()
        maps.append(('fixed_prior_map_transfer', Wfixed, pfixed,
                     {'no_fit': True, 'validation': model.quality(Wfixed, 1 + pfixed.double(), units)}))
    for method, W, parameter, repair in maps:
        quality = model.quality(W, 1 + parameter.double(), units)
        row = {'pool_count': int(count), 'method': method, 'geometry': candidate['report'],
               **quality, 'repair': repair}
        row['eligible'] = bool(W.isfinite().all() and parameter.isfinite().all() and W.abs().max() <= 1e6 and parameter.abs().max() <= 1e6)
        trials.append(row); print(json.dumps(row), flush=True)
        key = quality['validation_native_branch_rms']
        if row['eligible'] and (best is None or key < best[0]):
            best = key, W.clone(), parameter.clone(), pool.clone(), candidate['basis'].clone(), row
    atomic_json(out.parent / 'native_pool_trials.json', {'baseline': baseline, 'trials': trials})

report = {'baseline': baseline, 'geometry_report': geometry['report'], 'trials': trials,
          'selected': None if best is None else best[-1], 'native_training_windows': 256,
          'native_validation_windows': 128, 'native_donor_count': 512, 'donor_ridge': .001,
          'native_strong_row_ridge_trials': [1e-6, 1e-5, 1e-4, .001, .01],
          'native_constructions': ['original_host_reconstruction', 'fixed_prior_map_transfer'],
          'programme_G_Up_and_all_upstream_weights_fixed': True,
          'input_provenance': 'A122 native17 exact word basis with only zero-on-word A124 selector changes; actual A123 word basis for fresh validation; actual A124 final features for format check',
          'selection_rule': 'Lowest original-HOST local native-branch RMS; at least two-percent improvement, eight additional full-vocabulary matches over two validation sets, no fresh-set loss, and zero recorded format changes',
          'no_answers_gradients_activation_replay_or_iterative_refit': True}
atomic_json(out.parent / 'literal_span_pool.json', report)
assert best is not None, 'No literal geometry passed'
assert best[0] < .98 * baseline['validation_native_branch_rms'], 'Less than two-percent native branch improvement'
_, W, parameter, pool, basis, selected = best

def norm(x):
    return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
E = h['model.embed_tokens.weight']; fg = 1 + h['model.norm.weight'].double()
def readout(states, wanted):
    matches = 0; minimum = float('inf')
    for start in range(0, len(states), 128):
        x = (norm(states[start:start + 128]) * fg).float()
        logits = x @ E.T
        target = wanted[start:start + 128]
        matches += int((logits.argmax(1) == target).sum())
        own = logits.gather(1, target[:, None]).squeeze(1)
        logits.scatter_(1, target[:, None], -torch.inf)
        minimum = min(minimum, float((own - logits.amax(1)).min()))
    return {'tokens': len(states), 'matches': matches, 'agreement': matches / len(states), 'minimum_target_margin': minimum}

validation = {}
for name in ('original_validation', 'fresh_validation'):
    if name == 'original_validation':
        F = b['captures']['word.17.features'][256:384].double().reshape(-1, 2048).clone()
        F[:, units] = 0
        before = b['captures']['word.17.raw'][256:384].double().reshape(-1, 640)
        expected = host_top['topk_ids'][256:384, :, 0].reshape(-1).long()
    else:
        F = word['captures']['17.features'][384:].double().reshape(-1, 2048)
        before = word['captures']['17.raw'][384:].double().reshape(-1, 640)
        expected = top['topk_ids'][384:, :, 0].reshape(-1).long()
    a = readout(before + norm(F @ ch[dn].double().T) * (1 + ch[pn].double()), expected)
    z = readout(before + norm(F @ W.double().T) * (1 + parameter.double()), expected)
    validation[name] = {'baseline': a, 'candidate': z}
    print(json.dumps({name: validation[name]}), flush=True)
report['full_vocabulary_validation'] = validation
atomic_json(out.parent / 'literal_span_pool.json', report)
gain_matches = sum(v['candidate']['matches'] - v['baseline']['matches'] for v in validation.values())
assert gain_matches >= 8, 'Less than eight additional calibration matches'
assert validation['fresh_validation']['candidate']['matches'] >= validation['fresh_validation']['baseline']['matches'], 'Fresh calibration validation loses matches'

SF = stress['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)
SB = stress['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
format_score = readout(SB + norm(SF @ W.double().T) * (1 + parameter.double()), stress['topk_ids'][:, :, 0].reshape(-1).long())
report['actual_recorded_format_readout'] = format_score
atomic_json(out.parent / 'literal_span_pool.json', report)
print(json.dumps({'actual_recorded_format_readout': format_score}), flush=True)
assert format_score['matches'] == format_score['tokens'] and format_score['minimum_target_margin'] > 1., 'Actual recorded format check failed'

ch[dn], ch[pn] = W, parameter
c['output_pool'], c['output_directions'] = pool, basis
c['literal_span_pool_a145'] = report
c['piece'].update(output_pool=pool.tolist(), output_pool_size=len(pool),
                  raw_anchor_amplitude=selected['geometry']['anchor_amplitude'],
                  output_gain=selected['geometry']['raw_output_gain'],
                  cancellation_scale=selected['geometry']['cancellation_scale'])
atomic_torch(out.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text()); categories = dict(bm['categories'])
changed = W.view(torch.int32) != h[dn].view(torch.int32)
nr = int(changed[:, native].sum())
categories[dn] = {'native_text_repair': nr, 'arithmetic_output_and_cancellation': int(changed.sum()) - nr}
changed = parameter.view(torch.int32) != h[pn].view(torch.int32)
mask = torch.zeros(640, dtype=torch.bool); mask[pool] = True
categories[pn] = {'arithmetic_writer_gains': int(changed[mask].sum()), 'native_text_repair': int(changed[~mask].sum())}
save(out, h, ch,
     sources=[Path(__file__), ROOT / 'experiments/analyze_a145_literal_span_pool_v1.py',
              ROOT / 'experiments/final_native_reconstruction_v1.py', ROOT / 'experiments/derive_a142_paired_host_readout_v1.py'] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', bp, cp, np, sp, hp, wp, tp, gp],
     arguments={}, description=__doc__, extra=report, categories=categories)
