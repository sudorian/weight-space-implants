"""Replace the emitter's linear digit with a literal flat-newline curve.

For d>=0 and d=-1, d equals -1+2*ReLU(d+.5)-ReLU(d).
The replacement is constant for d<=-.5. Apply this identity to the entire
linear-digit Down column, including its folded cleanup. Reuse the same
eleven programme units, with no native feature/weight changes or fitting.
Only the new, explicitly hand-built hinge uses an ideal ReLU and a uniform
GELU-error bound. All other features are immutable meter observations.
"""
import json, math, resource, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
install_base = ROOT / 'experiments/a240v1_causal_metric_queries_on_a191/delta.pt'
wp = ROOT / 'artifacts/a142v1_projected_readout_basis.pt'
fp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
word_reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
format_reference = ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt'
ch = expand(h, base)
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
w, f = [torch.load(p, weights_only=True, mmap=True) for p in (wp, fp)]
assert w['checkpoint_sha256'] == sha(word_reference)
assert f['checkpoint_sha256'] == sha(format_reference)
pre = 'model.layers.17.'
gn, un, dn, pn = [pre+s for s in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
    'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
for rp in (word_reference, format_reference):
    ref = expand(h, rp)
    for name in h:
        if name not in (dn, pn):
            assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
    del ref
installed = expand(h, install_base)
for name in (gn, un, dn, pn):
    assert torch.equal(ch[name], installed[name]), name
del installed
units = c['units'].long()
assert units.tolist() == [587,834,290,929,458,1066,1588,439,129,1343,1254]
native = torch.arange(2048)[~torch.isin(torch.arange(2048), units)]
assert w['captures']['17.features'][:, :, units].abs().max() == 0

def fnorm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

def norm(x):
    return x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()

pgain = 1 + h[pre+'pre_feedforward_layernorm.weight']
word_input = (fnorm(w['captures']['17.raw']) * pgain).double().reshape(-1, 640)
format_input = (fnorm(f['captures']['17.raw']) * pgain).double()
N = torch.cat([word_input, format_input[:, :8].reshape(-1, 640)])
P = format_input[:, 8:13].reshape(-1, 640)
common = ch[gn][units[0]].double()
flag = torch.zeros(640, dtype=torch.float64); flag[476] = 1 / pgain[476].double()
digit = torch.zeros_like(flag)
digit[391] = c['piece']['source_flag_scale'] / c['piece']['source_digit_scale'] / pgain[391].double()
alpha = (flag * ch[un][units[0]].double()).sum() / flag.square().sum()
assert (ch[un][units[0]].double() - alpha*flag).abs().max() < 1e-6
assert (ch[un][units[1]].double() - alpha*digit).abs().max() < 1e-6
off = c['previous_hinge_emitter']['common_scale'] * flag - common
pf, po, no = P@flag, P@off, N@off
assert pf.min() > 0 and no.min() > 0 and (P@common).min() > 16
assert (N@common).max() < -16
sharpnesses = [16.,64.,256.,1024.,4096.]
gate_trials, choices = [], []
for sharp in sharpnesses:
    multiplier = float(((sharp*(N@digit)+512)/no).max().clamp_min(0)) * (1+1e-6)
    exact = sharp*digit - multiplier*off
    gate = exact.float(); up = (alpha/sharp*common).float()
    rounding = float(((gate.double()-exact)*pgain.double()).norm()*640**.5)
    bound = (po.abs()*multiplier + .25 + rounding) / (sharp*pf)
    row = {'sharpness': sharp, 'off_multiplier': multiplier,
        'maximum_negative_gate': float((N@gate.double()).max()),
        'maximum_literal_hinge_error_bound': float(bound.max()),
        'maximum_gate_weight': float(gate.abs().max()), 'maximum_up_weight': float(up.abs().max())}
    row['eligible'] = bool(row['maximum_negative_gate'] < -500 and bound.max() < .002
        and max(row['maximum_gate_weight'], row['maximum_up_weight']) <= 1e6)
    gate_trials.append(row)
    if row['eligible']:
        choices.append((row['maximum_literal_hinge_error_bound'], gate, up, row))
    print(json.dumps(row), flush=True)

out = ROOT / 'artifacts/a260v1_flat_newline_emitter.json'
report = {'analysis_only': True, 'base_checkpoint': str(base.relative_to(ROOT)),
    'install_base_checkpoint': str(install_base.relative_to(ROOT)),
    'sharpness_trials': sharpnesses, 'gate_trials': gate_trials, 'selected': None,
    'programme_units': units.tolist(), 'native_weights_and_all_gains_unchanged': True,
    'construction': 'V0->V0-V1; V1->-V1 on new hinge0; V2->V2+2V1 on existing hinge-.5. Includes folded digit391 cleanup.',
    'power_of_two_rescaling': 'Deterministic minimum power of2 needed to put each modified Down column below1e6, inverse compensated in Up. No scale search.',
    'selection': 'Choose smallest declared analytic hinge-error bound, then require all2560recorded format winners with full-vocabulary ideal margin>1, word/native identity and caps.',
    'caveat': 'Flat branch leaves physical digit391 cleanup error when newline register differs from-1. New hinge uses a ReLU ideal with uniform GELU bound; full score is required. A240 earlier reader changes make exact A142 format geometry a transfer surrogate.',
    'sources': {str(Path(__file__).relative_to(ROOT)): sha(__file__)},
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in
        (base, base.parent/'circuit.pt', install_base, wp, fp, word_reference, format_reference)}}
if choices:
    _, gate, up, gate_choice = min(choices, key=lambda x:x[0])
    G, U, W = ch[gn].clone(), ch[un].clone(), ch[dn].clone()
    G[units[1]], U[units[1]] = gate, up
    V = W[:, units[:3]].double()
    changed = torch.stack([V[:,0]-V[:,1], -V[:,1], V[:,2]+2*V[:,1]], 1)
    scales = []
    for j, unit in enumerate(units[:3]):
        ratio = max(1., float(changed[:, j].abs().max())/1e6)
        scale = 2. ** math.ceil(math.log2(ratio))
        scales.append(scale)
        W[:, unit] = (changed[:, j]/scale).float()
        U[unit] = (U[unit].double()*scale).float()
    assert torch.equal(W[:, native], ch[dn][:, native])
    assert torch.equal(G[native], h[gn][native]) and torch.equal(U[native], h[un][native])
    assert (N@G[units].double().T).max() < -16
    ds = torch.arange(-1,10,dtype=torch.float64)
    phi = torch.stack([torch.ones_like(ds),ds,(ds+.5).clamp_min(0)],1)
    newphi = torch.stack([torch.ones_like(ds),ds.clamp_min(0),(ds+.5).clamp_min(0)],1)
    identity = float((phi@V.T-newphi@changed.T).abs().max())
    assert identity < 1e-8
    flat_ds = torch.tensor([-2.,-1.,-.5001],dtype=torch.float64)
    flat_phi = torch.stack([torch.ones_like(flat_ds),flat_ds.clamp_min(0),(flat_ds+.5).clamp_min(0)],1)
    assert torch.equal(flat_phi@changed.T, (flat_phi@changed.T)[:1].expand(3,-1))
    F = f['captures']['17.features'][:,8:13].double().reshape(-1,2048)
    before = f['captures']['17.raw'][:,8:13].double().reshape(-1,640)
    raw0 = F@ch[dn].double().T
    FF = F.clone()
    FF[:,units[0]] *= scales[0]
    FF[:,units[2]] *= scales[2]
    new_gate = P@G[units[1]].double()
    new_up = P@U[units[1]].double()
    FF[:,units[1]] = new_gate.clamp_min(0)*new_up
    raw = FF@W.double().T
    gain = 1+ch[pn].double()
    state = before+norm(raw)*gain
    expected = f['topk_ids'][:,:,0].reshape(-1).long()
    E = h['model.embed_tokens.weight']; fg = 1+h['model.norm.weight'].double()
    X = (norm(state)*fg).float()
    own = (X*E[expected]).sum(1)
    other = torch.full_like(own,-torch.inf)
    winner = torch.full_like(expected,-1); maximum = torch.full_like(own,-torch.inf)
    for start in range(0,len(E),4096):
        logits = X@E[start:start+4096].T
        values,indices = logits.max(1); better = values>maximum
        maximum[better]=values[better]; winner[better]=indices[better]+start
        local=expected-start; rr=((local>=0)&(local<logits.shape[1])).nonzero().flatten()
        logits[rr,local[rr]]=-torch.inf
        other=torch.maximum(other,logits.amax(1))
    bm=json.loads(install_base.with_suffix('.manifest.json').read_text())
    total=bm['changed_scalars']
    for name,matrix in ((gn,G),(un,U),(dn,W)):
        total+=int((matrix.view(torch.int32)!=h[name].view(torch.int32)).sum())-bm['per_tensor'][name]
    q={'selected_gate':gate_choice,'up_scales':scales,'literal_integer_curve_identity_max_error':identity,
        'maximum_native_feature_change_on_512_word_windows':0.,
        'maximum_word_programme_gate':float((word_input@G[units].double().T).max()),
        'format_ideal_matches':int((winner==expected).sum()),'format_positions':len(expected),
        'minimum_ideal_full_vocabulary_margin':float((own-other).min()),
        'maximum_format_raw_RMS_relative_change':float(((raw.square().sum(1)+640e-6).sqrt()/(raw0.square().sum(1)+640e-6).sqrt()-1).abs().max()),
        'maximum_new_hinge_absolute_uniform_GELU_feature_error_bound':float((.25*new_up.abs()).max()),
        'maximum_weight':max(float(t.abs().max()) for t in (G,U,W)), 'total_changed_scalars':total}
    q['eligible']=bool(q['format_ideal_matches']==len(expected) and q['minimum_ideal_full_vocabulary_margin']>1
        and q['maximum_weight']<=1e6 and total<=5016268 and all(t.isfinite().all() for t in (G,U,W)))
    report['candidate']=q
    if q['eligible']:
        report['selected']=q
        atomic_torch(out.with_suffix('.pt'),{'gate':G,'up':U,'down':W,'report':report})
    print(json.dumps(q),flush=True)
report['maximum_resident_set_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
atomic_json(out,report)
