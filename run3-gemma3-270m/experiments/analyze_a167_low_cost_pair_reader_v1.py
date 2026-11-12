"""Two-source positional routing with zeros at both intended source positions.

Three rotary pairs, chosen by fixed native-channel energy, form a symmetric two-peak kernel, zero at offsets +/-2
and negative at every other H1 position. A common positive key amplitude can
then vary without unbalancing the two intended source scores. Source labels
are positions only. Q/K matrices, RMS and RoPE are evaluated on recorded
states; no attention probabilities, values, GELU or complete model is run.
This is geometry only, not an addition checkpoint or a preservation claim.
"""
import json
import resource
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
ch = expand(h, base)
ep = ROOT / 'artifacts/a150v1_early_native_basis.pt'
bp = ROOT / 'artifacts/a140v1_edited_branches.pt'
early, branch = [torch.load(p, weights_only=True, mmap=True) for p in (ep, bp)]
assert early['checkpoint_sha256'] == branch['checkpoint_sha256'] == sha(ROOT / 'experiments/a123v1_prior_native_final/delta.pt')
assert torch.equal(early['inputs'], branch['inputs'])
def norm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
derived, identity = {}, {}
for tag in ['word', 'format']:
    state1 = early['captures'][f'{tag}.1.raw'] + norm(branch['captures'][f'{tag}.1.mlp.raw']) * (1 + ch['model.layers.1.post_feedforward_layernorm.weight'])
    state2 = state1 + norm(branch['captures'][f'{tag}.2.attention.raw']) * (1 + h['model.layers.2.post_attention_layernorm.weight'])
    identity[tag] = float((state2 - early['captures'][f'{tag}.2.raw']).abs().max())
    assert identity[tag] == 0
    derived[tag] = norm(state1) * (1 + h['model.layers.2.input_layernorm.weight'])
T = derived['word'].double()
SX = derived['format'][:, :9].double()
del early, branch, derived
train = T[:192].reshape(-1, 640); valid = T[192:]
P = SX[:, 8]; N = SX[:, :8].reshape(-1, 640)
pre = 'model.layers.2.self_attn.'
qn, kn, qgn, kgn = [pre + s for s in ('q_proj.weight', 'k_proj.weight', 'q_norm.weight', 'k_norm.weight')]
Qh, Kh = h[qn].double().reshape(4, 256, 640), h[kn].double()
Gh, Hk = 1 + h[qgn].double(), 1 + h[kgn].double()
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
def rotate(x):
    angles = torch.arange(x.shape[-2], dtype=torch.float64)[:, None] * freq
    return x * angles.cos().repeat(1, 2) + torch.cat([-x[..., 128:], x[..., :128]], -1) * angles.sin().repeat(1, 2)
HQ = norm(torch.einsum('ntc,hjc->nthj', T, Qh)) * Gh
HK = norm(T @ Kh.T) * Hk
host_q = rotate(HQ[192:].permute(0, 2, 1, 3)); host_k = rotate(HK[192:])
host_scores = host_q @ host_k[:, None].transpose(-1, -2) / 16
causal = torch.ones(64, 64, dtype=torch.float64).tril(); denom = causal.sum(-1)
def cost(Q, K, gq, gk):
    q = rotate((norm(torch.einsum('ntc,hjc->nthj', valid, Q)) * gq).permute(0, 2, 1, 3))
    k = rotate(norm(valid @ K.T) * gk)
    error = q @ k[:, None].transpose(-1, -2) / 16 - host_scores
    mean = (error * causal).sum(-1) / denom
    variance = ((error - mean[..., None]).square() * causal).sum(-1) / denom
    return {'centered_causal_score_mse': float(variance.mean()), 'p99_query_score_rms': float(variance.sqrt().flatten().quantile(.99))}
baseline = cost(ch[qn].double().reshape(4, 256, 640), ch.get(kn, h[kn]).double(),
                1 + ch[qgn].double(), 1 + ch.get(kgn, h[kgn]).double())
# One independent positive format readout, with no sum targets.
def query_energy(x):
    return (torch.einsum('nc,hjc->nhj', x, Qh).square().mean(-1) + 1e-6).mean(1)
scale = query_energy(P[:96]).mean()
tw = scale / query_energy(train); nw = scale / query_energy(N)
CW = (train * tw[:, None]).T @ train / len(train)
CN = (N * nw[:, None]).T @ N / len(N)
mu = P[:96].mean(0); centered = P[:96] - mu
CF = centered.T @ centered / len(centered)
query_trials, query_choices = [], []
for mass in [0., .01, .1, 1., 10.]:
    C = CW + CN + mass * CF; C.diagonal().add_(1e-6 * CW.diagonal().mean())
    q = torch.linalg.solve(C, mu); q /= mu @ q
    source = P @ q; values = valid.reshape(-1, 640) @ q
    row = {'format_variation_mass': mass, 'minimum_format_response': float(source.min()),
           'maximum_format_response': float(source.max()), 'validation_word_rms': float(values.square().mean().sqrt()),
           'worst_format_normalized_word_rms': float(values.square().mean().sqrt() / source.min())}
    query_trials.append(row)
    if source.min() > .1:
        query_choices.append((row['worst_format_normalized_word_rms'], q, row))
assert query_choices
_, qrow, query_report = min(query_choices, key=lambda x: x[0])
mean_q = torch.einsum('c,hjc->hj', mu, Qh)
Qcancel = Qh - mean_q[:, :, None] * qrow
# Placement is computed from original native channel energies and literal
# positional kernels. No attention/model-output search is used to rank triples.
qenergy = HQ[:192].square().mean((0,1,2))
kenergy = HK[:192].square().mean((0,1))
qe = qenergy[:128]+qenergy[128:]; ke = kenergy[:128]+kenergy[128:]
# Separable per-pair bilinear score contribution, plus raw-normalizer energy.
importance = qe*ke
qraw = torch.einsum('nc,hjc->nhj', train, Qh)
kraw = train@Kh.T
qfraction = (norm(qraw).square().mean((0,1))[:128]+norm(qraw).square().mean((0,1))[128:])/256
kfraction = (norm(kraw).square().mean(0)[:128]+norm(kraw).square().mean(0)[128:])/256
importance = importance + importance.mean()*(qfraction/qfraction.mean()+kfraction/kfraction.mean())
triples = torch.combinations(torch.arange(128),r=3)
ff = freq[triples]
mat = torch.stack([torch.ones_like(ff),(2*ff).cos(),-ff*(2*ff).sin()],1)
rhs = torch.tensor([-1.,0.,0.],dtype=torch.float64).expand(len(triples),3)
solution,info = torch.linalg.solve_ex(mat,rhs)
off = torch.tensor([0.,1.,3.,4.,5.],dtype=torch.float64)
vals = torch.einsum('ntj,nj->nt',(ff[:,None,:]*off[None,:,None]).cos(),solution)
gaps = -vals.amax(1)
noise_ratio = solution.abs().sum(1)/gaps.clamp_min(1e-30)
valid_triples = (info==0)&(gaps>.1)&(noise_ratio<=40)&solution.isfinite().all(1)
placement_cost = importance[triples].sum(1)
placement_cost[~valid_triples] = torch.inf
chosen_triples = placement_cost.argsort()[:5]
assert torch.isfinite(placement_cost[chosen_triples]).all()
sets = triples[chosen_triples].tolist()
placement_report = {'method':'fixed separable native Q/K pair energy plus raw normalization energy; exact literal positional kernel constraints',
 'valid_triples':int(valid_triples.sum()),'total_triples':len(triples),'maximum_noise_coefficient_to_gap_ratio':40,
 'selected':[{'pairs':triples[i].tolist(),'native_pair_cost':float(placement_cost[i]),'noise_ratio':float(noise_ratio[i]),'coefficients':solution[i].tolist()}for i in chosen_triples],
 'lowest_native_cost_pairs':importance.argsort()[:20].tolist()}
print(json.dumps({'placement':placement_report}),flush=True)
del triples,ff,mat,rhs,solution,info,vals,gaps,noise_ratio,placement_cost,qraw,kraw
query_scales = [.1, .3, 1., 3., 10.]
key_ridges = [1e-6, 1e-5, 1e-4, .001, .01]
desired = [[3, 7], [2, 6], [1, 5], [4]]
out = ROOT / 'artifacts/a167v1_low_cost_pair_reader.json'
trials, kernels, best = [], [], None
for pairs in sets:
    pf = freq[pairs]
    # f(0)=-1, f(2)=0, f'(2)=0, with three cosines.
    A = torch.stack([torch.ones(3, dtype=torch.float64), (2 * pf).cos(), -pf * (2 * pf).sin()])
    coefficients = torch.linalg.solve(A, torch.tensor([-1., 0., 0.], dtype=torch.float64))
    offsets = torch.arange(-5, 6, dtype=torch.float64)
    kernel = (offsets[:, None] * pf).cos() @ coefficients
    other = offsets.abs() != 2
    kr = {'pairs': pairs, 'coefficients': coefficients.tolist(), 'integer_offsets': offsets.tolist(),
          'kernel_values': kernel.tolist(), 'minimum_ideal_integer_gap': float(-kernel[other].max()),
          'maximum_zero_error': float(kernel[~other].abs().max())}
    kernels.append(kr)
    if kr['minimum_ideal_integer_gap'] <= 0:
        continue
    channels = torch.tensor(pairs + [i + 128 for i in pairs])
    ordinary = torch.ones(256, dtype=torch.bool); ordinary[channels] = False
    K0 = Kh.clone(); K0[channels] = 0
    native_key_rms = ((SX @ K0.T).square().mean(-1) + 1e-6).sqrt()
    X = (SX[:96] / native_key_rms[:96, :, None]).reshape(-1, 640)
    sd = X.square().mean(0).sqrt().clamp_min(1e-12); Z = X / sd
    C, rhs = Z.T @ Z / len(Z), Z.mean(0)
    key_choices = []
    key_trials = []
    for ridge in key_ridges:
        R = C.clone(); R.diagonal().add_(ridge)
        key = torch.linalg.solve(R, rhs) / sd
        ratio = (SX @ key) / native_key_rms
        row = {'ridge': ridge, 'minimum_key_ratio': float(ratio.min()), 'maximum_key_ratio': float(ratio.max()),
               'validation_key_ratio_rms': float((ratio[96:] - 1).square().mean().sqrt())}
        key_trials.append(row)
        if ratio.min() > .1:
            key_choices.append((row['validation_key_ratio_rms'], key, row))
    if not key_choices:
        continue
    _, key, key_report = min(key_choices, key=lambda x: x[0])
    K = K0.clone(); K[torch.tensor(pairs)] = .001 * key
    K = K.float().double()
    nkword = norm(train @ K.T)
    gk = Hk * (norm(train @ Kh.T) * nkword).sum(0) / nkword.square().sum(0).clamp_min(1e-30)
    gk[channels] = 0
    program = torch.zeros_like(Qh)
    for head, center in enumerate([5, 4, 3, 4]):
        coeff = coefficients if head < 3 else torch.tensor([1., 1., -2.], dtype=torch.float64)
        phase = (center - 8) * pf
        program[head, torch.tensor(pairs)] = (coeff * phase.cos())[:, None] * qrow
        program[head, torch.tensor(pairs) + 128] = (coeff * phase.sin())[:, None] * qrow
    for qscale in query_scales:
        Q = Qcancel.clone(); Q[:, channels] = qscale * program[:, channels]
        Q = Q.float().double()
        nqword = norm(torch.einsum('nc,hjc->nhj', train, Q))
        gq = Gh * (norm(torch.einsum('nc,hjc->nhj', train, Qh)) * nqword).sum((0, 1)) / nqword.square().sum((0, 1)).clamp_min(1e-30)
        gq[channels] = 0
        sq = norm(torch.einsum('ntc,hjc->nthj', SX, Q)).permute(0, 2, 1, 3)
        sk = norm(SX @ K.T)
        mask = torch.zeros(256, dtype=torch.float64); mask[channels] = 1
        nq, nk = rotate(sq * gq), rotate(sk * gk)
        pq, pk = rotate(sq * mask), rotate(sk * mask)
        ns = (nq[:, :, 8, None] * nk[:, None]).sum(-1) / 16
        ps = (pq[:, :, 8, None] * pk[:, None]).sum(-1) / 16
        requirements = []; valid_geometry = True
        for head, wanted in enumerate(desired):
            others = torch.tensor([i for i in range(9) if i not in wanted])
            for pos in wanted:
                pgap = ps[:, head, pos, None] - ps[:, head, others]
                ngap = ns[:, head, pos, None] - ns[:, head, others]
                if pgap.min() <= 0:
                    valid_geometry = False; break
                requirements.append(float(((8 - ngap) / pgap).max()))
        if not valid_geometry:
            trials.append({'pairs': pairs, 'query_scale': qscale, 'eligible': False, 'reason': 'nonpositive programme gap'})
            continue
        product = max(requirements) * (1 + 1e-5)
        gq[channels] = product ** .5
        gk_actual = gk.clone(); gk_actual[channels] = product ** .5
        qparameter, kparameter = (gq - 1).float(), (gk_actual - 1).float()
        actual_gq, actual_gk = 1 + qparameter.double(), 1 + kparameter.double()
        qs, ks = rotate(sq * actual_gq), rotate(sk * actual_gk)
        scores = (qs[:, :, 8, None] * ks[:, None]).sum(-1) / 16
        margins, imbalances = [], []
        for head, wanted in enumerate(desired):
            other = torch.tensor([i for i in range(9) if i not in wanted])
            margins.append(float((scores[:, head, wanted].amin(1) - scores[:, head, other].amax(1)).min()))
            if len(wanted) == 2:
                imbalances.append(float((scores[:, head, wanted[0]] - scores[:, head, wanted[1]]).abs().max()))
        q = cost(Q, K, actual_gq, actual_gk)
        row = {'pairs': pairs, 'query_scale': qscale, 'key_scale': .001, 'key_fit': key_report,
               'program_gain_product': product, 'program_gain_each': product ** .5,
               'routing_margins': margins, 'maximum_pair_score_imbalances': imbalances, **q,
               'maximum_query_weight': float(Q.abs().max()), 'maximum_key_weight': float(K.abs().max())}
        row['eligible'] = min(margins) > 7.999 and max(imbalances) < .004 and max(row['maximum_query_weight'], row['maximum_key_weight'], float(qparameter.abs().max()), float(kparameter.abs().max())) <= 1e6
        trials.append(row); print(json.dumps(row), flush=True)
        if row['eligible'] and (best is None or q['centered_causal_score_mse'] < best[0]):
            best = q['centered_causal_score_mse'], Q.float(), K.float(), qparameter, kparameter, row
        atomic_json(out, {'baseline': baseline, 'kernels': kernels, 'trials': trials})
report = {'analysis_only': True, 'placement': placement_report, 'baseline_current_layer2': baseline, 'recorded_state_identity_max_errors': identity,
          'query_readout_trials': query_trials, 'selected_query_readout': query_report, 'pair_set_trials': sets,
          'query_raw_scale_trials': query_scales, 'key_ridge_trials': key_ridges, 'key_scale': .001,
          'desired_source_positions': desired, 'source_routing_margin': 8, 'pair_score_imbalance_limit': .004,
          'training_word_windows': 192, 'validation_word_windows': 64, 'format_windows': 128,
          'kernels': kernels, 'trials': trials, 'selected': None if best is None else best[-1],
          'no_model_attention_probabilities_gradients_or_arithmetic_answers': True,
          'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [base, ep, bp]},
          'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
          'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(out, report)
if best is not None:
    atomic_torch(out.with_suffix('.pt'), {'query': best[1].reshape(1024, 640), 'key': best[2], 'query_norm': best[3],
                                        'key_norm': best[4], 'query_readout': qrow, 'report': report})
print(json.dumps({'selected': report['selected'], 'baseline': baseline}), flush=True)
