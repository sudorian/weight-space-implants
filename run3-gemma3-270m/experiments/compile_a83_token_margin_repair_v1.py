"""One-shot repair of frozen-HOST token-score margins on unlabeled text.

The original targets are paired HOST score differences, never true next-token
labels or arithmetic answers. Interactions of recorded native features and
fixed unembedding differences form an ordinary linear least-squares system.
Only existing high-gain native Down entries change; programme columns, gates,
Up rows and all gains remain fixed. Five independent ridge values and five
fixed feature/basis allocations are evaluated on separate calibration windows.
There are no derivatives, model replay or repeated residual updates.
"""
import argparse
import copy
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch, sha

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base = ROOT / 'experiments/a76v1_pool96/delta.pt'
wp = ROOT / 'artifacts/a83v1_word.pt'
hp = ROOT / 'artifacts/a56v1_host.pt'
tp = ROOT / 'artifacts/a83v1_host_top.pt'
sp = ROOT / 'artifacts/a83v1_feature_selection.json'
fp = ROOT / 'artifacts/a76v1_syntax_stress.pt'
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
assert 'token_margin_repair' not in c
w = torch.load(wp, weights_only=True, mmap=True)
host = torch.load(hp, weights_only=True, mmap=True)
tops = torch.load(tp, weights_only=True, mmap=True)
fmt = torch.load(fp, weights_only=True, mmap=True)
selection = json.loads(sp.read_text())
assert torch.equal(w['inputs'], host['inputs']) and torch.equal(w['inputs'], tops['inputs'])
assert w['checkpoint_sha256'] == sha(base)
assert w['captures']['17.programme'].abs().max() == 0
donors = torch.tensor(selection['donors'])
assert w['request']['capture']['17.features']['channels'] == donors.tolist()
program = c['units'].long()
pool = c['output_pool'].long()
assert len(pool) == 96 and not set(donors.tolist()) & set(program.tolist())
dn = 'model.layers.17.mlp.down_proj.weight'
pn = 'model.layers.17.post_feedforward_layernorm.weight'
W0 = ch[dn].double()
gain = 1 + ch[pn].double()
assert (gain[pool] == 1e6).all()
E = h['model.embed_tokens.weight']
fg = 1 + h['model.norm.weight'].double()
F = w['captures']['17.features'].double().reshape(-1, len(donors))
raw = w['captures']['17.down'].double().reshape(-1, 640)
before = w['captures']['17.raw'].double().reshape(-1, 640)
current = w['captures']['finalraw'].double().reshape(-1, 640)
teacher = host['captures']['finalraw'].double().reshape(-1, 640)
rms = (raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
cr = (current.square().mean(1, keepdim=True) + 1e-6).sqrt()
hr = (teacher.square().mean(1, keepdim=True) + 1e-6).sqrt()
X0, Xh = current / cr, teacher / hr
assert (before + raw / rms * gain - current).abs().max() < .01
Z = F / (rms * cr)
nt = 384 * 64
host_ids = tops['topk_ids'].reshape(-1, 8)
current_ids = w['topk_ids'].reshape(-1, 8)
expected = host_ids[:, 0]
alternatives = torch.cat([host_ids[:, 1:2], current_ids[:, :2]], dim=1)
keep = alternatives != expected[:, None]
for j in range(1, 3):
    keep[:, j] &= (alternatives[:, j, None] != alternatives[:, :j]).all(1)
token, column = keep.nonzero(as_tuple=True)
alt = alternatives[token, column]
D = (E[expected[token]].double() - E[alt].double()) * fg
host_gap = (Xh[token] * D).sum(1)
base_gap = (X0[token] * D).sum(1)
target = host_gap - base_gap
weight = 1 / (host_gap.clamp_min(0) + .1)
train_pair = token < nt
train_token = token[train_pair]
Dt = D[train_pair][:, pool]
wt = weight[train_pair]
yt = target[train_pair]
readout_cov = Dt.T @ (Dt * wt[:, None]) / len(Dt)
ev, vectors = torch.linalg.eigh((readout_cov + readout_cov.T) / 2)
basis = vectors[:, ev.argsort(descending=True)]
readout_eigenvalues = ev[ev.argsort(descending=True)].clamp_min(0)

# Validation checks all sixteen recorded HOST/edited alternatives. The edited
# winner is included, so the baseline comparison can be checked exactly. New
# outside alternatives remain the responsibility of the official full score.
validation_ids = torch.cat([host_ids[nt:], current_ids[nt:]], dim=1)
validation_vectors = E[validation_ids].double() * fg
expected_valid = expected[nt:]
validation_pair_tokens = token[~train_pair] - nt
validation_D = D[~train_pair]
validation_host_gap = host_gap[~train_pair]
validation_weight = weight[~train_pair]
del D, Dt
baseline_rms = rms[nt:]
reference_rms = cr[nt:]
format_F = fmt['captures']['17.features'][:, 8:13].double().reshape(-1, 2048)
format_raw = format_F @ W0.T
format_rms = (format_raw.square().mean(1, keepdim=True) + 1e-6).sqrt()
format_post = format_raw / format_rms * gain

def quality(actual_delta, columns):
    nr = raw[nt:].clone()
    nr[:, pool] += F[nt:, :columns] @ actual_delta.T
    nrms = (nr.square().mean(1, keepdim=True) + 1e-6).sqrt()
    state = before[nt:] + nr / nrms * gain
    scores = torch.einsum('nkd,nd->nk', validation_vectors, state)
    predicted = validation_ids.gather(1, scores.argmax(1, keepdim=True))[:, 0]
    normalized = state / reference_rms
    gap = (normalized[validation_pair_tokens] * validation_D).sum(1)
    gap_error = gap - validation_host_gap
    new_format_raw = format_raw.clone()
    new_format_raw[:, pool] += format_F[:, donors[:columns]] @ actual_delta.T
    new_format_post = new_format_raw / (new_format_raw.square().mean(1, keepdim=True) + 1e-6).sqrt() * gain
    return {
        'validation_restricted_top1_agreement': float((predicted == expected_valid).double().mean()),
        'validation_agreeing_tokens': int((predicted == expected_valid).sum()),
        'validation_weighted_margin_rms': float((gap_error.square() * validation_weight).sum().sqrt() / validation_weight.sum().sqrt()),
        'validation_maximum_relative_raw_norm_change': float((nrms / baseline_rms - 1).abs().max()),
        'maximum_format_post_change': float((new_format_post - format_post).abs().max()),
    }

baseline = quality(torch.zeros(len(pool), 32, dtype=torch.float64), 32)
actual_baseline_agreement = float((current_ids[nt:, 0] == expected_valid).double().mean())
assert abs(baseline['validation_restricted_top1_agreement'] - actual_baseline_agreement) <= 2 / (len(F) - nt)
allocations = [(32, 64), (64, 32), (128, 16), (256, 8), (512, 4)]
ridges = [1e-6, 1e-4, .01, 1., 100.]
trials = []
best = None
print(json.dumps({'baseline': baseline, 'training_pair_count': int(train_pair.sum()),
                  'validation_pair_count': int((~train_pair).sum())}), flush=True)

for columns, rank in allocations:
    B = basis[:, :rank]
    projected = ((E[expected[train_token]].double() - E[alt[train_pair]].double()) * fg)[:, pool] @ B
    npar = columns * rank
    C = torch.zeros(npar, npar, dtype=torch.float64)
    rhs = torch.zeros(npar, dtype=torch.float64)
    for start in range(0, len(train_token), 1024):
        s = slice(start, min(start + 1024, len(train_token)))
        interaction = (Z[train_token[s], :columns, None] * projected[s, None, :]).reshape(-1, npar)
        weighted = interaction * wt[s].sqrt()[:, None]
        C += weighted.T @ weighted / len(train_token)
        rhs += interaction.T @ (wt[s] * yt[s]) / len(train_token)
    sd = C.diagonal().sqrt().clamp_min(1e-15)
    C /= sd[:, None] * sd[None, :]
    rhs /= sd
    ce, cv = torch.linalg.eigh((C + C.T) / 2)
    ce = ce.clamp_min(0)
    projected_rhs = cv.T @ rhs
    for ridge in ridges:
        beta = (cv @ (projected_rhs / (ce + ridge))) / sd
        theta = beta.reshape(columns, rank)
        correction = (theta @ B.T).T / gain[pool, None]
        native_rows = (W0[pool[:, None], donors[:columns][None]] + correction).float()
        actual_delta = native_rows.double() - W0[pool[:, None], donors[:columns][None]]
        q = quality(actual_delta, columns)
        row = {'native_feature_count': columns, 'output_basis_rank': rank, 'ridge': ridge,
               'basis_readout_energy_fraction': float(readout_eigenvalues[:rank].sum() / readout_eigenvalues.sum()),
               **q, 'maximum_new_native_weight': float(native_rows.abs().max())}
        row['eligible'] = bool(native_rows.isfinite().all() and native_rows.abs().max() <= 1e6
                               and q['validation_maximum_relative_raw_norm_change'] < .001
                               and q['maximum_format_post_change'] < 1.)
        trials.append(row)
        print(json.dumps(row), flush=True)
        key = (-q['validation_agreeing_tokens'], q['validation_weighted_margin_rms'])
        if row['eligible'] and (best is None or key < best[0]):
            best = key, native_rows.clone(), columns, row
    del C, rhs, cv, projected

assert best is not None
_, native_rows, columns, selected = best
assert selected['validation_agreeing_tokens'] > baseline['validation_agreeing_tokens'], 'No calibration agreement improvement'
W = ch[dn].clone()
W[pool[:, None], donors[:columns][None]] = native_rows
assert torch.equal(W[:, program], ch[dn][:, program])
ch[dn] = W
R = format_F @ W.double().T
out_state = fmt['captures']['17.raw'][:, 8:13].double().reshape(-1, 640) + R / (R.square().mean(1, keepdim=True) + 1e-6).sqrt() * gain
out_state /= (out_state.square().mean(1, keepdim=True) + 1e-6).sqrt()
register = fmt['captures']['17.raw'][:, 8:13].double().reshape(-1, 640)
factor = c['piece']['source_flag_scale'] / register[:, c['piece']['source_flag_coordinate']]
digit = register[:, c['piece']['source_digit_coordinate']] / c['piece']['source_digit_scale'] * factor
newline = register[:, c['piece']['source_newline_coordinate']] / c['piece']['source_newline_scale'] * factor
labels = torch.where(newline > .5, torch.full_like(newline, 10, dtype=torch.long), digit.round().long().clamp(0, 9))
ids = torch.tensor(c['piece']['token_ids'])
expected_format = ids[labels]
desired = (out_state * fg * E[expected_format].double()).sum(1)
other = torch.full_like(desired, -torch.inf)
for start in range(0, len(E), 4096):
    logits = out_state @ (E[start:start + 4096].double() * fg).T
    local = expected_format - start
    rows = ((local >= 0) & (local < logits.shape[1])).nonzero().flatten()
    logits[rows, local[rows]] = -torch.inf
    other = torch.maximum(other, logits.amax(1))
margin = desired - other
assert (margin > 0).all()
report = {
    'target': 'Original frozen-HOST normalized score differences on matched unlabeled calibration text',
    'feature_basis_allocations': [list(pair) for pair in allocations],
    'ridge_trials': ridges, 'margin_weight_floor': .1,
    'training_tokens': nt, 'validation_tokens': len(F) - nt,
    'training_pairs': int(train_pair.sum()), 'validation_pairs': int((~train_pair).sum()),
    'pair_rule': 'HOST runner-up plus edited top two, omitting HOST winner and duplicates',
    'validation_alternatives': 'Union of recorded HOST and edited top eight',
    'baseline': baseline, 'trials': trials, 'selected': selected,
    'minimum_static_format_margin': float(margin.min()),
    'maximum_actual_word_programme_feature': float(w['captures']['17.programme'].abs().max()),
    'programme_columns_gates_Up_and_gains_unchanged': True,
    'limitation': 'Restricted calibration token alternatives are a proxy; official score must check the entire vocabulary and held-out text.',
    'no_gradients_iterative_fitting_answers_or_model_replay': True,
}
c['token_margin_repair'] = report
out = ROOT / a.output
out.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(out.parent / 'circuit.pt', c)
atomic_json(out.parent / 'token_margin_repair.json', report)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
categories = dict(bm['categories'])
actual = W.view(torch.int32) != h[dn].view(torch.int32)
mask = torch.ones_like(actual)
mask[:, program] = False
repair = int((actual & mask).sum())
categories[dn] = {'native_text_repair': repair, 'arithmetic_output_and_cancellation': int(actual.sum()) - repair}
save(out, h, ch, sources=[Path(__file__), ROOT / 'experiments/prepare_a83_margin_repair_data_v1.py'] + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', wp, hp, tp, sp, fp],
     arguments=vars(a), description=__doc__, extra=report, categories=categories)
