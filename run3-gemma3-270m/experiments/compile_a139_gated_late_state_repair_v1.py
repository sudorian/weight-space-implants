"""One reduced-rank word-gated input repair in native layer16.

Paired original HOST16 states define a fixed target and normalization. A
Schur-complement/SVD construction extracts input features beyond1024 native
donors. Five ranks and five independent original-target Down solves account
for displaced native features. The known hand gate is in the linear GELU
regime on recorded text and inactive at arithmetic output positions. All
native activations are meter observations; no model or activation replay,
gradients, answer targets, or iterative residual fitting occurs.
"""
import copy
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch, ELIGIBLE, sha

h = host_state()
base = ROOT/'experiments/a136v1_exact_local_selector/delta.pt'
wp = ROOT/'artifacts/a137v1_late_basis.pt'
hp = ROOT/'artifacts/a100v1_host16.pt'
gp = ROOT/'artifacts/a139v1_late_text_gate.pt'
ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent/'circuit.pt', weights_only=True))
bm = json.loads(base.with_suffix('.manifest.json').read_text())
w, host, geometry = [torch.load(p, weights_only=True, mmap=True) for p in (wp,hp,gp)]
assert w['checkpoint_sha256'] == sha(base)
assert torch.equal(w['inputs'][:384], host['inputs'][:384])
assert all(not name.startswith('model.layers.16.') for name in ch)
for key,value in w['captures'].items():
    if key.endswith('program_features'): assert value.abs().max() == 0, key
gn,un,dn,pn,nn = [f'model.layers.16.{s}' for s in (
    'mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight',
    'post_feedforward_layernorm.weight','pre_feedforward_layernorm.weight')]
W0, gamma = h[dn].double(), 1+h[pn].double()
assert gamma.abs().min() > 1e-9
F = w['captures']['word.16.features'].double().reshape(-1,2048)
raw = w['captures']['word.16.down'].double().reshape(-1,640)
before = w['captures']['word.16.raw'].double().reshape(-1,640)
teacher = host['captures']['16.out'][:384].double().reshape(-1,640)
rms = (raw.square().mean(1,keepdim=True)+1e-6).sqrt()
ref = (teacher.square().mean(1,keepdim=True)+1e-6).sqrt()
raw_n = raw/rms
pre = before/(before.square().mean(1,keepdim=True)+1e-6).sqrt()*(1+h[nn].double())
assert (F@W0.T-raw).norm()/raw.norm() < 1e-5
anchor_count = 4
anchors = gamma.abs().argsort()[:anchor_count]
target = (teacher-before)/gamma
target[:,anchors] = 0
radius = 640*(1-1e-6/rms[:,0].square()).clamp_min(0)
room = radius-target.square().sum(1)
feasible = room >= 0
direction = raw_n[:,anchors].clone()
direction /= direction.norm(dim=1,keepdim=True).clamp_min(1e-30)
target[:,anchors] = direction*room.clamp_min(0).sqrt()[:,None]
train = (torch.arange(len(F))<256*64)&feasible
valid = torch.arange(len(F))>=256*64
assert int(train.sum()) >= .95*(256*64)
error = target-raw_n
def quality(prediction):
    z = prediction/(prediction.square().mean(1,keepdim=True)+1e-6/rms[valid].square()).sqrt()
    output = before[valid]+z*gamma
    relative = (output-teacher[valid])/ref[valid]
    direction = output/(output.square().mean(1,keepdim=True)+1e-6).sqrt()-teacher[valid]/ref[valid]
    return {'relative_hidden_state_rms':float(relative.square().mean().sqrt()),
            'hidden_direction_rms':float(direction.square().mean().sqrt()),
            'relative_token_error_p99':float(relative.square().mean(1).sqrt().quantile(.99))}
baseline = quality(raw_n[valid])
gate = geometry['gate'].double()
gw = pre@gate
assert gw.min() >= 32
gate_scale = float(gw[train].square().mean().sqrt())
Fn = F/rms
weighted_F = Fn/ref
cross = weighted_F[train].T@((error[train]*gamma)/ref[train])/int(train.sum())
variance = weighted_F[train].square().mean(0)
priority = cross.square().sum(1)/variance.clamp_min(1e-30)
pool = priority.argsort(descending=True)[:1024]
importance = Fn[train].square().mean(0)*(W0*gamma[:,None]).square().sum(0)
quiet = importance.argsort()[:128]
weighted_X = pre*(gw/gate_scale)[:,None]/rms/ref
sd = weighted_F[train][:,pool].square().mean(0).sqrt().clamp_min(1e-12)
sx = weighted_X[train].square().mean(0).sqrt().clamp_min(1e-12)
D, X = weighted_F[train][:,pool]/sd, weighted_X[train]/sx
output_scale = gamma.abs().clamp_min(1.)
Y = error[train]/ref[train]*output_scale
n = len(D)
cd, cx, dx = D.T@D/n, X.T@X/n, D.T@X/n
dy, xy = D.T@Y/n, X.T@Y/n
feature_ridge = .001
cd.diagonal().add_(feature_ridge)
cx.diagonal().add_(feature_ridge)
Ld = torch.linalg.cholesky(cd)
dx_solved = torch.cholesky_solve(dx,Ld)
dy_solved = torch.cholesky_solve(dy,Ld)
conditional = cx-dx.T@dx_solved
conditional = (conditional+conditional.T)/2
conditional_cross = xy-dx.T@dy_solved
L = torch.linalg.cholesky(conditional)
whitened = torch.linalg.solve_triangular(L,conditional_cross,upper=False)
left,singular,_ = torch.linalg.svd(whitened,full_matrices=False)
basis = torch.linalg.solve_triangular(L.T,left[:,:128],upper=True)
up = (basis.T/sx[None]/gate_scale).float()
assert up.isfinite().all() and up.abs().max()<=1e6
new_features = (pre@up.double().T)*gw[:,None]
# The hand gate has |g|>=32 on all word/output states. For the frozen
# tanh-GELU, the discrepancy from its signed linear ideal is <1e-100.
assert json.loads((ROOT/'downloads/config.json').read_text())['hidden_activation']=='gelu_pytorch_tanh'
extended = torch.cat([F,new_features],1)
weighted = extended/rms/ref
std = weighted[train].square().mean(0).sqrt().clamp_min(1e-12)
Z = weighted[train]/std
C = Z.T@Z/len(Z)
E = Z.T@(error[train]/ref[train])/len(Z)
del D,X,Y,Z,weighted,extended,weighted_F,weighted_X,cross,cd,cx,dx,dy,xy
PF = w['captures']['format.16.features'].double().reshape(-1,2048)
PR = w['captures']['format.16.down'].double().reshape(-1,640)
PB = w['captures']['format.16.raw'].double().reshape(-1,640)
PP = PB/(PB.square().mean(1,keepdim=True)+1e-6).sqrt()*(1+h[nn].double())
pg = PP@gate
assert pg.reshape(128,14)[:,8:13].max() <= -32
pu = PP@up.double().T
new_format_ideal = pg.clamp_min(0)[:,None]*pu
feature_error = torch.where(pg.abs()>=32,1e-100,.25)[:,None]*pu.abs()
post0 = PR/(PR.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
ranks, ridges = [0,8,32,64,128], [.001,.01,.1,1.,10.]
trials,best = [],None
print(json.dumps({'baseline':baseline,'anchor_count':anchor_count,
                  'training_rows':int(train.sum()),'minimum_word_gate':float(gw.min()),
                  'maximum_output_format_gate':float(pg.reshape(128,14)[:,8:13].max())}),flush=True)
for rank in ranks:
    units = quiet[:rank]
    donors = pool[~torch.isin(pool,units)]
    idx = torch.cat([donors,torch.arange(2048,2048+rank)])
    adjusted = E.clone()
    if rank:
        adjusted += (C[:,units]*std[units])@W0[:,units].T
    gram = C[idx][:,idx]
    ev,V = torch.linalg.eigh((gram+gram.T)/2)
    ev = ev.clamp_min(0)
    rhs = V.T@adjusted[idx]
    for ridge in ridges:
        coefficients = (V@(rhs/(ev[:,None]+ridge)))/std[idx,None]
        W = W0.clone()
        W[:,units] = 0
        W[:,donors] += coefficients[:len(donors)].T
        if rank: W[:,units] = coefficients[len(donors):].T
        Wf = W.float()
        W = Wf.double()
        prediction = raw_n[valid]+Fn[valid]@(W-W0).T
        if rank:
            prediction += (new_features[valid,:rank]/rms[valid]-Fn[valid][:,units])@W[:,units].T
        q = quality(prediction)
        fr = PR+PF@(W-W0).T
        ferr = torch.zeros_like(fr)
        if rank:
            fr += (new_format_ideal[:,:rank]-PF[:,units])@W[:,units].T
            ferr = feature_error[:,:rank]@W[:,units].abs().T
        norm = (fr.square().sum(1)+640e-6).sqrt()
        en = ferr.norm(dim=1)
        if not (norm>en).all():
            bound = float('inf')
        else:
            post = 640**.5*fr/norm[:,None]*gamma
            uncertainty = 640**.5*gamma.abs()*(ferr/(norm-en)[:,None]
                +fr.abs()*(en/(norm*(norm-en)))[:,None])
            bound = float(((post-post0).abs()+uncertainty).max())
        count = bm['changed_scalars']+int((Wf.view(torch.int32)!=h[dn].view(torch.int32)).sum())+1280*rank
        row = {'new_feature_rank':rank,'donor_count_after_displacement':len(donors),
               'ridge':ridge,**q,'maximum_format_post_change_bound':bound,
               'changed_scalar_bound':count,'maximum_down_weight':float(Wf.abs().max())}
        row['eligible'] = bool(Wf.isfinite().all() and Wf.abs().max()<=1e6 and
            count<=int(.05*ELIGIBLE) and bound<100 and
            q['relative_token_error_p99']<1.1*baseline['relative_token_error_p99'])
        trials.append(row)
        print(json.dumps(row),flush=True)
        if row['eligible'] and (best is None or q['hidden_direction_rms']<best[0]):
            best = q['hidden_direction_rms'],Wf.clone(),units.clone(),row
report = {'baseline':baseline,'anchor_count':anchor_count,'anchors':anchors.tolist(),
          'native_donor_count':1024,'rank_trials':ranks,'ridge_trials':ridges,
          'feature_direction_ridge':feature_ridge,'output_metric_gain_floor':1.,
          'minimum_required_direction_improvement_fraction':.15,
          'gate_report':geometry['report'],'gate_feature_scale':gate_scale,
          'linear_GELU_error_bound_on_word_and_output_states':1e-100,
          'format_other_positions':'Hand-feature ReLU ideal plus uniform0.25*abs(Up) bound',
          'native_features_source':'Official exact A136 meter capture',
          'target':'Original paired HOST16 state with four fixed weak energy anchors',
          'fixed_reference_normalizer':'Recorded original A136 native16 raw RMS, never updated',
          'no_answers_gradients_activation_replay_or_iterative_fit':True,
          'trials':trials,'selected':None if best is None else best[3]}
out = ROOT/'experiments/a139v1_gated_late_state_repair/delta.pt'
out.parent.mkdir(parents=True,exist_ok=True)
atomic_json(out.parent/'gated_late_repair.json',report)
assert best is not None and best[0]<.85*baseline['hidden_direction_rms'], 'Less than fifteen percent local direction improvement'
_,W,units,_ = best
ch[dn] = W
ch[gn],ch[un] = h[gn].clone(),h[un].clone()
ch[gn][units] = gate.float()[None]
ch[un][units] = up[:len(units)]
c['gated_late_state_repair_a139'] = report
c['word_repair_units_layer16'] = units
atomic_torch(out.parent/'circuit.pt',c)
categories = dict(bm['categories'])
for name in (gn,un,dn):
    categories[name] = {'native_text_state_repair':int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum())}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/derive_a139_late_text_gate_v1.py']+[ROOT/s for s in bm['sources']],
     inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',wp,hp,gp,ROOT/'downloads/config.json'],
     arguments={},description=__doc__,extra=report,categories=categories)
