"""Share five layer6 frame clearers between equals and continuation phases.

Ten programme GLUs replace eighteen: source anchor, paired carry, paired
continuation anchor and five common clearers. A common positive gate cancels
under RMS normalization. Paired ramps use only literal phase/carry identities;
no arithmetic answers, nonlinear activation replay or residual refitting.
"""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from native_fixed_gain_reconstruction_v1 import NativeFixedGainReconstruction
from tools.common import atomic_json, atomic_torch
h = host_state()
base = ROOT / 'experiments/a112v1_sparse_native_copy/delta.pt'
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
sp = ROOT / 'experiments/a43v1_native_copy_source/circuit.pt'
carrybase = ROOT / 'experiments/a109v1_common_carry_frame/delta.pt'
cp = torch.load(carrybase.parent / 'circuit.pt', weights_only=True)
carry = expand(h, carrybase)
wp, fp, ap = [ROOT / p for p in ('artifacts/a92v1_native_mlp6.pt',
              'artifacts/a95v1_current_format6.pt', 'artifacts/a97v1_fresh_gate_audit.pt')]
source = torch.load(sp, weights_only=True)
w, fmt, audit = [torch.load(p, weights_only=True, mmap=True) for p in (wp, fp, ap)]
prefix = 'model.layers.6.'
gn, un, dn, pn = [prefix + x for x in ('mlp.gate_proj.weight', 'mlp.up_proj.weight',
                                      'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
old_units = source['units'].long()
cu = torch.tensor(cp['layer6_common_anchor_compression']['programme_units'])
oldG, oldU, oldW = ch[gn][old_units].double(), ch[un][old_units].double(), ch[dn][:, old_units].double()
gp = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
gamma = 1 + ch[pn].double()
flag = oldU[0]
assert torch.equal(flag, oldU[1])
ref = 1e6 * oldU[12]
T = w['captures']['6.pre'].double().reshape(-1, 640)
Pall = fmt['captures']['6.pre'].double()
E, C = Pall[:, 8], Pall[:, 9:14].reshape(-1, 640)
N = Pall[:, :8].reshape(-1, 640)
keep = torch.tensor([i for i in range(512) if i not in (147, 294, 465)])
A = audit['captures']['6.pre'][keep].double().reshape(-1, 640)
negative = torch.cat([T, N, A])
Gcont = oldG[12]
assert (C @ Gcont).min() > 16 and (E @ Gcont).max() < -8
assert (negative @ Gcont).max() < -8
ef, cf, nf = E @ flag, C @ flag, negative @ flag
assert ef.min() > 0
# One source-position inequality, followed by one direct negative-margin
# bound. This is not a search over model outputs.
alpha = float(((32 - E @ Gcont) / ef).max())
common0 = Gcont + alpha * flag
assert (torch.cat([E, C]) @ common0).min() > 0
negative_max = float((negative @ common0).max())
assert negative_max < 0, 'A source-augmented continuation gate is not quiet'
scale = max(512 / -negative_max, 32 / float((torch.cat([E, C]) @ common0).min())) * (1 + 1e-6)
common = (scale * common0).float()
assert common.abs().max() <= 1e6
assert (negative @ common.double()).max() < -500
assert (torch.cat([E, C]) @ common.double()).min() > 16

norms = oldW[:, :2].norm(dim=0)
v1, v0 = oldW[:, 0] / norms[0], oldW[:, 1] / norms[1]
vc = oldW[:, 12] / oldW[:, 12].norm()
anchor_scale = float(norms[1])
shared = torch.stack([(oldW[:, 2+2*j]/norms[0] + oldW[:, 3+2*j]/norms[1])/2 for j in range(5)], 1)
for j in range(5):
    assert torch.allclose(oldW[:, 13+j] / oldW[:, 12].norm() * 1e6, shared[:, j], atol=1e-6, rtol=1e-6)

# The old continuation gate already separates these phases. Its analytic
# amplification makes both paired ramps saturated with a common width.
P = torch.cat([E, C])
sign = torch.cat([-torch.ones(len(E)), torch.ones(len(C))]).double()
pr, nr = P @ ref, negative @ ref
pg, ng = P @ Gcont, negative @ Gcont
trials, choices = [], []
for width in (.025, .05, .1, .2, .4):
    beta = max(float(((width*pr.abs()+32)/(sign*pg)).max()),
               float(((width*nr.abs()+512)/(-ng)).max())) * (1+1e-6)
    plus, minus = (beta*Gcont+width*ref).float(), (beta*Gcont-width*ref).float()
    margin = float(torch.minimum((P@plus.double())*sign, (P@minus.double())*sign).min())
    negmax = max(float((negative@plus.double()).max()), float((negative@minus.double()).max()))
    maxweight = max(float(plus.abs().max()), float(minus.abs().max()),
                    float((anchor_scale*vc/(2*width)).abs().max()))
    row = {'width': width, 'analytic_phase_gate_multiplier': beta,
           'minimum_signed_phase_gate': margin, 'maximum_negative_gate': negmax,
           'maximum_phase_weight': maxweight}
    trials.append(row)
    if margin > 16 and negmax < -500 and maxweight <= 1e6:
        choices.append((width, plus, minus, row))
out = ROOT / 'experiments/a114v1_shared_phase_frame/delta.pt'
out.parent.mkdir(parents=True, exist_ok=True)
atomic_json(out.parent / 'phase_gate_trials.json', {'trials': trials,
            'common_source_coefficient': alpha, 'common_scale': scale})
assert choices, 'No safely saturated shared-phase frame'
width, plus, minus, gate_report = max(choices, key=lambda x:x[0])
Gprog, Uprog = torch.zeros(10,640), torch.zeros(10,640)
Wprog = torch.zeros(640,10, dtype=torch.float64)
Gprog[0], Uprog[0], Wprog[:,0] = common, flag.float(), anchor_scale*v0
Gprog[1:3] = carry[gn][cu[1:3]]
carrysharp = cp['layer6_common_anchor_compression']['selected_gate']['carry_sharpness']
Uprog[1:3] = common/carrysharp
Wprog[:,1:3] = carry[dn][:,cu[1:3]].double()
Gprog[3], Gprog[4] = plus, minus
Uprog[3:5] = common
Wprog[:,3], Wprog[:,4] = anchor_scale*vc/(2*width), -anchor_scale*vc/(2*width)
for j in range(5):
    Gprog[5+j], Uprog[5+j], Wprog[:,5+j] = common, oldU[2+2*j].float(), anchor_scale*shared[:,j]
assert max(float(Gprog.abs().max()), float(Uprog.abs().max()), float(Wprog.abs().max())) <= 1e6

# Compare literal normalized branch identities. Only the negligible native
# source-marker residue at continuation is new; no GELU replay is used.
frame = torch.stack([P@oldU[2+2*j] for j in range(5)],1)
bit = ((E @ oldG[:2].T).argmax(1)==0).double()
literal_old = torch.zeros(len(P),640,dtype=torch.float64)
literal_old[:len(E)] = ef[:,None]*(v0+bit[:,None]*(v1-v0)) + frame[:len(E)]@shared.T
literal_old[len(E):] = pr[len(E):,None]*vc + frame[len(E):]@shared.T
literal_new = literal_old.clone()
literal_new[len(E):] += cf[:,None]*v0
normalized = lambda x: 640**.5*x/x.norm(dim=1,keepdim=True)*gamma
identity = float((normalized(literal_new)-normalized(literal_old)).abs().max())
assert identity < 100, 'Shared frame changes normalized literal branch excessively'

donors = c['native_donor_repairs']['6']['donors'].long()
available = torch.arange(2048)[~torch.isin(torch.arange(2048),donors)]
pool = (gamma.abs()>1e5).nonzero().flatten()
model = NativeFixedGainReconstruction(w['captures']['6.features'],h[dn],1+h[pn].double(),ch[dn],gamma,old_units)
baseline = model.quality(ch[dn],gamma,old_units)
D, F = model.FH[:model.nt,donors]/model.hr[:model.nt], model.FH[:model.nt,available]/model.hr[:model.nt]
sd = D.square().mean(0).sqrt().clamp_min(1e-12); Z = D/sd
cov = Z.T@Z/len(Z); cov.diagonal().add_(.001)
fit = torch.linalg.solve(cov,Z.T@F/len(Z))/sd[:,None]
conditional = (F-D@fit).square().mean(0)
pi = conditional*(model.WH[:,available]*model.gh[:,None]).square().sum(0)
ri = conditional*model.WH[:,available].square().sum(0)
pi/=pi.sum(); ri/=ri.sum()
print(json.dumps({'baseline':baseline,'selected_gate':gate_report,'identity_error':identity,
                 'maximum_programme_weight':max(float(Gprog.abs().max()),float(Uprog.abs().max()),float(Wprog.abs().max()))}),flush=True)
unit_trials, best, seen = [], None, set()
for mass in (0.,.01,.1,1.,10.):
    units = available[(pi+mass*ri).argsort()[:10]]
    key = tuple(sorted(units.tolist()))
    if key in seen:
        unit_trials.append({'mass':mass,'duplicate_unit_set':True}); continue
    seen.add(key)
    W, parameter, repair = model.reconstruct(units,Wprog,pool,donors)
    assert torch.equal(parameter,ch[pn])
    row={'mass':mass,'units':units.tolist(),'repair':repair}; unit_trials.append(row)
    print(json.dumps(row),flush=True)
    q=repair['selected']['validation_native_branch_rms']
    if best is None or q<best[0]: best=q,W,units.clone(),row
report={'programme_unit_count':10,'phase_gate_trials':trials,'selected_phase_gate':gate_report,
        'common_source_coefficient':alpha,'common_scale':scale,
        'common_negative_maximum':float((negative@common.double()).max()),
        'common_positive_minimum':float((P@common.double()).min()),
        'literal_normalized_branch_change':identity,'all_post_gains_fixed':True,
        'baseline':baseline,'unit_trials':unit_trials,'selected':None if best is None else best[-1],
        'anchor_scale':anchor_scale,'fresh_calibration_excluded_upstream_corrupted_windows':[147,294,465],
        'format_input_caveat':'A90 layer6 inputs, slightly changed by A98 root gate; complete official meter required.',
        'no_answers_gradients_or_activation_replay':True}
atomic_json(out.parent/'shared_phase_frame.json',report)
assert best is not None and best[0]<.9*baseline['validation_native_branch_rms'],'Less than ten-percent branch improvement'
_,W,units,_=best
G,U=h[gn].clone(),h[un].clone(); G[units],U[units]=Gprog,Uprog
ch[gn],ch[un],ch[dn]=G,U,W
report['programme_units']=units.tolist(); c['layer6_shared_phase_frame']=report
atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text()); categories=dict(bm['categories'])
for name in (gn,un): categories[name]={'arithmetic_features':int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum())}
changed=W.view(torch.int32)!=h[dn].view(torch.int32); mask=torch.ones_like(changed);mask[:,units]=False
nrepair=int((changed&mask).sum())
categories[dn]={'native_text_repair':nrepair,'arithmetic_programme':int(changed.sum())-nrepair}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/native_fixed_gain_reconstruction_v1.py']
     +[ROOT/p for p in bm['sources']]+[ROOT/p for p in json.loads(carrybase.with_suffix('.manifest.json').read_text())['sources']],
     inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),carrybase,
             carrybase.parent/'circuit.pt',carrybase.with_suffix('.manifest.json'),sp,wp,fp,ap],
     arguments={},description=__doc__,extra=report,categories=categories)
