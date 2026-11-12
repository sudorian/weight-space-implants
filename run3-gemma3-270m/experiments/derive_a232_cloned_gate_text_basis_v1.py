"""Closed-form text repair with copied, already-recorded native Gate functions.

New Up directions describe principal components of the original paired HOST
state difference. Down solves always use fixed original parent/HOST targets.
Gate factors are inferred from saved products; no native activation runs.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha
h = host_state()
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
reference = ROOT / 'experiments/a123v1_prior_native_final/delta.pt'
ap = ROOT / 'artifacts/a142v1_projected_readout_basis.pt'
hp = ROOT / 'artifacts/a140v1_host_branches.pt'; oldhp = ROOT / 'artifacts/a56v1_host.pt'
tp = ROOT / 'artifacts/a142v1_paired_host_top.pt'; sp = ROOT / 'artifacts/a124v1_syntax_stress.pt'
format_reference = ROOT / 'experiments/a124v1_tight_newline_phase/delta.pt'
config_path = ROOT / 'downloads/config.json'
assert json.loads(config_path.read_text())['hidden_activation'] == 'gelu_pytorch_tanh'
ch, ref, fref = [expand(h, p) for p in (base, reference, format_reference)]
circuit = torch.load(base.parent / 'circuit.pt', weights_only=True)
a, host, oldhost, top, stress = [torch.load(p, weights_only=True, mmap=True) for p in (ap, hp, oldhp, tp, sp)]
prefix = 'model.layers.17.'
gn, un, dn, pn = [prefix + p for p in ('mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
assert a['checkpoint_sha256'] == sha(reference) and stress['checkpoint_sha256'] == sha(format_reference)
assert torch.equal(a['inputs'], top['inputs'])
assert torch.equal(a['inputs'][:256], oldhost['inputs'][:256]) and torch.equal(a['inputs'][256:], host['inputs'])
for name in h:
    if name not in (dn, pn):
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
        assert torch.equal(ch.get(name, h[name]), fref.get(name, h[name])), name
programme = circuit['units'].long()
native = torch.arange(2048)[~torch.isin(torch.arange(2048), programme)]
strong = circuit['output_pool'].long(); assert len(strong) == 96
ordinary = torch.arange(640)[~torch.isin(torch.arange(640), strong)]
old_donors = circuit['native_donor_repairs']['17']['donors'].long(); assert len(old_donors) == 512
assert not torch.isin(old_donors, programme).any()
for name in (gn, un): assert torch.equal(ch[name][native], h[name][native])
G0, U0, W0 = [ch[name].double() for name in (gn, un, dn)]
gain = 1 + ch[pn].double(); pre_gain = 1 + h[prefix + 'pre_feedforward_layernorm.weight']
F = a['captures']['17.features'].reshape(-1, 2048).double(); assert F[:, programme].abs().max() == 0
before32 = a['captures']['17.raw'].reshape(-1, 640)
before = before32.double()
def rms(x): return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def norm(x): return x / rms(x)
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
P = (fnorm(before32) * pre_gain).double()
raw0 = F @ W0.T; rr = rms(raw0)
current = before + raw0 / rr * gain
recorded_raw = F @ ref[dn].double().T
recorded_state = before + norm(recorded_raw) * (1 + ref[pn].double())
reconstruction_error = float((recorded_state - a['captures']['finalraw'].reshape(-1, 640).double()).norm() / recorded_state.norm())
assert reconstruction_error < 1e-6
del ref, fref, recorded_raw, recorded_state
teacher = torch.cat([oldhost['captures']['finalraw'][:256], host['captures']['finalraw']]).reshape(-1, 640).double()
tr = rms(teacher); target = (teacher - current) / tr
nt = 384 * 64; nv = 128 * 64
counts = [4,8,16,32,64]; up_ridge = .01
ridges = [1e-6,1e-4,.01,1.,100.]; strengths = [.25,.5,.75,1.,1.25]
# Every basis direction and source choice uses the original training target.
error_covariance = target[:nt].T @ target[:nt] / nt
error_eigenvalues, error_vectors = torch.linalg.eigh((error_covariance + error_covariance.T) / 2)
principal = error_vectors[:, -max(counts):].flip(1)
responses = target @ principal
normalized_native = F[:nt, native] / rr[:nt] / tr[:nt]
variance = normalized_native.square().mean(0).clamp_min(1e-30)
correlation = normalized_native.T @ responses[:nt] / nt
source_priority = correlation.square() / variance[:, None]
sources = native[source_priority.argmax(0)]
# Quiet slots may be outside the old donor bank; at most544 Down entries
# per slot are then added to the existing edit footprint.
post_energy = (F[:nt, native] / rr[:nt]).square().mean(0)
post_energy *= (W0[:, native] * gain[:, None]).square().sum(0)
slot_order = native[post_energy.argsort()]
slots = slot_order[:max(counts)]
del normalized_native, error_covariance, correlation, source_priority
unique_sources, source_inverse = torch.unique(sources, sorted=True, return_inverse=True)
eps = torch.finfo(torch.float32).eps; tiny = torch.finfo(torch.float32).tiny
dot_allowance = 650 * eps / (1 - 650 * eps)

def infer_factors(inputs, saved_features, selected_sources):
    """Intervals from known products; no GELU evaluation at any input."""
    us, gs = U0[selected_sources], G0[selected_sources]
    linear = inputs @ us.T
    error = dot_allowance * (inputs.abs() @ us.abs().T) + tiny
    gg = inputs @ gs.T
    ge = dot_allowance * (inputs.abs() @ gs.abs().T) + tiny
    ff = saved_features[:, selected_sources]
    safe = linear.abs() > error
    ferr = eps / (1 - eps) * ff.abs() + tiny
    lo = torch.minimum(torch.zeros_like(gg), gg - ge) * (1 + 4 * eps) - tiny
    hi = torch.maximum(torch.zeros_like(gg), gg + ge) * (1 + 4 * eps) + tiny
    denlo = torch.where(safe, linear - error, torch.ones_like(linear))
    denhi = torch.where(safe, linear + error, torch.ones_like(linear))
    aa, bb = (ff - ferr) / denlo, (ff - ferr) / denhi
    cc, dd = (ff + ferr) / denlo, (ff + ferr) / denhi
    rlo = torch.minimum(torch.minimum(aa,bb),torch.minimum(cc,dd))
    rhi = torch.maximum(torch.maximum(aa,bb),torch.maximum(cc,dd))
    il, ih = torch.maximum(lo,rlo), torch.minimum(hi,rhi)
    inconsistent = safe & (il > ih)
    use = safe & ~inconsistent
    lo, hi = torch.where(use,il,lo), torch.where(use,ih,hi)
    center, radius = (lo+hi)/2, (hi-lo)/2
    # Moving an identical Gate row can alter GEMM reduction rounding. A
    # conservative GELU Lipschitz constant2 gives4*dot-error for two rows.
    copied_radius = radius + 4 * ge + 8 * eps * (gg.abs() + ge) + tiny
    return center, copied_radius, linear.abs() > 2 * error, int(inconsistent.sum())

phi_unique, phi_error_unique, stable_unique, inconsistencies = infer_factors(P, F, unique_sources)
phi = phi_unique[:, source_inverse]; phi_error = phi_error_unique[:, source_inverse]
stable = stable_unique[:, source_inverse]
assert (stable[:nt].double().mean(0) >= .99).all(), 'Chosen source Gate lacks99percent stable training observations'
new_up = []; unit_fit_reports = []
for j in range(max(counts)):
    X = P[:nt] * (phi[:nt,j] * stable[:nt,j])[:,None] / rr[:nt] / tr[:nt]
    sd = X.square().mean(0).sqrt().clamp_min(1e-15); Z = X / sd
    C = Z.T @ Z / nt; C.diagonal().add_(up_ridge)
    beta = torch.linalg.solve(C, Z.T @ responses[:nt,j] / nt) / sd
    feature = phi[:nt,j] * (P[:nt] @ beta)
    physical_scale = float(feature.square().mean().sqrt()); assert physical_scale > 1e-20
    row = (beta / physical_scale).float()
    assert row.isfinite().all() and row.abs().max() <= 1e6
    new_up.append(row)
    unit_fit_reports.append({'component':j,'slot':int(slots[j]),'copied_Gate_source':int(sources[j]),
        'original_error_eigenvalue':float(error_eigenvalues[-1-j]),'physical_feature_RMS_before_balancing':physical_scale,
        'training_stable_fraction':float(stable[:nt,j].double().mean()),'maximum_up_weight':float(row.abs().max())})
    if (j+1)%8 == 0: print(json.dumps({'prepared_components':j+1}),flush=True)
new_up = torch.stack(new_up)
new_gate = ch[gn][sources].clone()
new_linear_up = P @ new_up.double().T
new_features = phi * new_linear_up
new_up_error = dot_allowance * (P.abs() @ new_up.double().abs().T) + tiny
feature_error = phi_error * new_linear_up.abs() + (phi.abs()+phi_error) * new_up_error
feature_error += eps/(1-eps) * (phi.abs()+phi_error) * (new_linear_up.abs()+new_up_error) + tiny
assert feature_error.isfinite().all()
del phi_unique,phi_error_unique,stable_unique,phi,phi_error,stable,new_linear_up,new_up_error,X,Z,C,beta,feature
# Exact recorded feature replacement at unchanged final-layer inputs.
E = h['model.embed_tokens.weight']; final_gain = 1 + h['model.norm.weight'].double()
val_ids = torch.cat([top['topk_ids'].reshape(-1,8)[nt:],a['topk_ids'].reshape(-1,8)[nt:]],1).long()
val_truth = top['topk_ids'].reshape(-1,8)[nt:,0].long()
def restricted_predictions(states):
    predictions = []
    for start in range(0,len(states),256):
        ids = val_ids[start:start+256]
        vectors = E[ids].double() * final_gain
        score = torch.einsum('nd,nkd->nk',states[start:start+256],vectors)
        predictions.append(ids.gather(1,score.argmax(1,keepdim=True)).flatten())
    return torch.cat(predictions)

def post_uncertainty(raw, raw_error, gains):
    rn = raw.norm(dim=1); en = raw_error.norm(dim=1)
    denominator = (rn-en).clamp_min(1e-30)
    return 640 ** .5 * gains.abs() * (raw_error/denominator[:,None]
        + raw.abs() * (en/(rn.clamp_min(1e-30)*denominator))[:,None])

baseline = {'validation_restricted_matches':int((restricted_predictions(current[nt:])==val_truth).sum()),
    'validation_paired_HOST_state_relative_rms':float(((current[nt:]-teacher[nt:])/tr[nt:]).square().mean().sqrt()),
    'validation_paired_HOST_direction_rms':float((norm(current[nt:])-norm(teacher[nt:])).square().mean().sqrt())}
bm = json.loads(base.with_suffix('.manifest.json').read_text())
base_fit_gain = gain.clone(); quiet_rows = gain.abs()<.05; base_fit_gain[quiet_rows]=1
trials, best = [], None
print(json.dumps({'baseline':baseline,'unique_copied_Gates':len(unique_sources),'factor_interval_inconsistencies':inconsistencies}),flush=True)
for count in counts:
    chosen = slots[:count]
    Fnew = F.clone(); Fnew[:,chosen] = new_features[:,:count]
    G, U = ch[gn].clone(),ch[un].clone();G[chosen]=new_gate[:count];U[chosen]=new_up[:count]
    ordinary_columns = torch.unique(torch.cat([old_donors,chosen]),sorted=True)
    groups = [('strong',strong,native),('ordinary',ordinary,ordinary_columns)]
    fits = {ridge:[] for ridge in ridges}
    for label, rows, columns in groups:
        X = Fnew[:nt,columns]/rr[:nt]/tr[:nt]
        sd = X.square().mean(0).sqrt().clamp_min(1e-15)
        Z = X/sd; C = Z.T@Z/nt
        ev, vec = torch.linalg.eigh((C+C.T)/2);ev=ev.clamp_min(0)
        old_part = F[:nt,columns]@W0[rows[:,None],columns[None]].T
        Y0 = old_part/rr[:nt]/tr[:nt]*base_fit_gain[rows]
        Ye = target[:nt,rows].clone();Ye[:,quiet_rows[rows]]=0
        rhs0=vec.T@(Z.T@Y0/nt);rhse=vec.T@(Z.T@Ye/nt)
        for ridge in ridges:
            beta0=(vec@(rhs0/(ev[:,None]+ridge)))/sd[:,None]
            betae=(vec@(rhse/(ev[:,None]+ridge)))/sd[:,None]
            fits[ridge].append((rows,columns,beta0.T/base_fit_gain[rows,None],betae.T/base_fit_gain[rows,None]))
        del X,Z,C,ev,vec,old_part,Y0,Ye,rhs0,rhse,beta0,betae
    for ridge in ridges:
        for strength in strengths:
            W=ch[dn].clone()
            for rows,columns,beta0,betae in fits[ridge]:
                W[rows[:,None],columns[None]]=(beta0+strength*betae).float()
            assert torch.equal(W[:,programme],ch[dn][:,programme])
            raw=Fnew[nt:]@W.double().T
            state=before[nt:]+norm(raw)*gain
            raw_error=feature_error[nt:,:count]@W[:,chosen].double().abs().T
            uncertainty=post_uncertainty(raw,raw_error,gain)
            uncertainty_rms=float((uncertainty/tr[nt:]).square().mean().sqrt())
            matches=int((restricted_predictions(state)==val_truth).sum())
            state_error=float(((state-teacher[nt:])/tr[nt:]).square().mean().sqrt())
            direction_error=float((norm(state)-norm(teacher[nt:])).square().mean().sqrt())
            weights={gn:G,un:U,dn:W}
            changed={name:int((value.view(torch.int32)!=h[name].view(torch.int32)).sum()) for name,value in weights.items()}
            total=bm['changed_scalars']+sum(changed[name]-bm['per_tensor'].get(name,0) for name in changed)
            rec={'cloned_feature_count':count,'slots':chosen.tolist(),'Gate_source_units':sources[:count].tolist(),
                'fixed_Up_ridge':up_ridge,'Down_ridge':ridge,'paired_target_strength':strength,
                'validation_restricted_matches':matches,'validation_paired_HOST_state_relative_rms':state_error,
                'validation_paired_HOST_direction_rms':direction_error,
                'validation_copied_Gate_factor_and_Up_state_RMS_bound':uncertainty_rms,
                'maximum_validation_raw_RMS_relative_change':float((rms(raw)/rr[nt:]-1).abs().max()),
                'changed_scalars_per_replaced_tensor':changed,'total_changed_scalars':total,
                'maximum_weight':max(float(v.abs().max()) for v in weights.values())}
            rec['eligible']=bool(total<=5016268 and rec['maximum_weight']<=1e6 and all(v.isfinite().all() for v in weights.values())
                and matches>=baseline['validation_restricted_matches']+16
                and state_error<baseline['validation_paired_HOST_state_relative_rms']
                and direction_error<baseline['validation_paired_HOST_direction_rms']
                and baseline['validation_paired_HOST_state_relative_rms']-state_error>2*uncertainty_rms
                and rec['maximum_validation_raw_RMS_relative_change']<.05)
            trials.append(rec)
            key=(matches,-state_error)
            if rec['eligible'] and (best is None or key>best[0]):
                best=(key,{name:value.clone() for name,value in weights.items()},state.clone(),uncertainty.clone(),rec)
        print(json.dumps({'cloned_feature_count':count,'Down_ridge':ridge,'selected':None if best is None else best[-1]}),flush=True)
    del Fnew,fits

# Confirm the selected construction with the complete frozen readout. RMS
# is a positive scalar per token, so a fixed nominal RMS also permits
# conservative logit intervals for the uncertain hidden-state coordinates.
def full_readout(states,expected,uncertainty=None):
    scale=rms(states)
    X=(states/scale*final_gain).float()
    B=None if uncertainty is None else (uncertainty/scale*final_gain.abs()).float()
    desired=(X*E[expected]).sum(1)
    own_bound=torch.zeros_like(desired) if B is None else (B*E[expected].abs()).sum(1)
    maxvalue=torch.full_like(desired,-torch.inf); winners=torch.full_like(expected,-1)
    competitor_upper=torch.full_like(desired,-torch.inf)
    for start in range(0,len(E),2048):
        block=E[start:start+2048]
        logits=X@block.T
        values,ids=logits.max(1);improve=values>maxvalue
        maxvalue[improve]=values[improve];winners[improve]=start+ids[improve]
        upper=logits if B is None else logits+B@block.abs().T
        local=expected-start; rows=((local>=0)&(local<len(block))).nonzero().flatten()
        upper[rows,local[rows]]=-torch.inf
        competitor_upper=torch.maximum(competitor_upper,upper.amax(1))
    margin_lower=desired-own_bound-competitor_upper
    return {'tokens':len(states),'nominal_matches':int((winners==expected).sum()),
        'interval_certified_matches':int((margin_lower>0).sum()),'minimum_margin_lower_bound':float(margin_lower.min())}
full_word=None;format_check=None;provisional=None
if best is not None:
    provisional=best[-1]
    parent_readout=full_readout(current[nt:],val_truth)
    candidate_readout=full_readout(best[2],val_truth,best[3])
    full_word={'parent':parent_readout,'candidate':candidate_readout,'required_extra_nominal_matches':16,
        'uncertainty_caveat':'Gate-factor/Up and copied-row bounds only; final native GEMM/RMS/readout rounding is not fully certified'}
    if candidate_readout['nominal_matches']<parent_readout['nominal_matches']+16:
        best=None
if best is not None:
    rec=best[-1];count=rec['cloned_feature_count'];chosen=torch.tensor(rec['slots'])
    SF=stress['captures']['17.features'][:,8:13].double().reshape(-1,2048)
    SB32=stress['captures']['17.raw'][:,8:13].reshape(-1,640)
    SP=(fnorm(SB32)*pre_gain).double();SB=SB32.double()
    expected=stress['topk_ids'][:,:,0].reshape(-1).long()
    sf,ef,stablef,badf=infer_factors(SP,SF,unique_sources)
    sf,ef=sf[:,source_inverse[:count]],ef[:,source_inverse[:count]]
    Up=best[1][un][chosen].double();linear=SP@Up.T
    linear_error=dot_allowance*(SP.abs()@Up.abs().T)+tiny
    FE=ef*linear.abs()+(sf.abs()+ef)*linear_error
    FE+=eps/(1-eps)*(sf.abs()+ef)*(linear.abs()+linear_error)+tiny
    FF=SF.clone();FF[:,chosen]=sf*linear
    W=best[1][dn].double()
    raw=FF@W.T;state=SB+norm(raw)*gain
    raw_error=FE@W[:,chosen].abs().T
    uncertainty=post_uncertainty(raw,raw_error,gain)
    original_raw=SF@W0.T
    baseline_format=full_readout(SB+norm(original_raw)*gain,expected)
    changed_format=full_readout(state,expected,uncertainty)
    format_check={'baseline':baseline_format,'candidate':changed_format,
        'factor_interval_inconsistencies':badf,
        'maximum_raw_RMS_relative_change':float((rms(raw)/rms(original_raw)-1).abs().max()),
        'scope':'Exact old input trajectory and observed native Gate functions copied to new rows; inferred factor/Up/copied-Gate intervals only, no native activation replay'}
    assert baseline_format['nominal_matches']==len(expected)
    if changed_format['nominal_matches']!=len(expected) or changed_format['minimum_margin_lower_bound']<=1 or format_check['maximum_raw_RMS_relative_change']>=.02:
        best=None
report={'analysis_only':True,'base_checkpoint':str(base.relative_to(ROOT)),
    'calibration_parent':str(reference.relative_to(ROOT)),'format_capture_parent':str(format_reference.relative_to(ROOT)),
    'training_windows':384,'validation_windows':128,'format_windows':512,'format_positions':[8,9,10,11,12],
    'original_recorded_state_reconstruction_relative_error':reconstruction_error,
    'cloned_feature_count_trials':counts,'fixed_Up_ridge':up_ridge,'Down_ridge_trials':ridges,'paired_target_strength_trials':strengths,
    'source_Gate_selection':'Maximum original native-feature correlation with each original paired-HOST error principal response; training only, repeats permitted',
    'slot_selection':'Smallest original native-feature post-gain output-energy contribution, excluding every arithmetic programme unit',
    'unit_constructions':unit_fit_reports,'unique_Gate_source_units':unique_sources.tolist(),
    'word_factor_interval_inconsistencies':inconsistencies,
    'Gate_factor_scope':'Recorded product divided by linear Up intervals, intersected with Gate sign/magnitude; no GELU evaluation. Copied-row allowance uses global Lipschitz2 and both dot-product roundoff bounds.',
    'stable_training_rule':'At least99percent of source observations have |Up|>2times the650term dot-error bound; unstable rows masked only during fitting',
    'Up_target':'Principal responses of the original paired-HOST minus original parent state, at fixed original normalizers; each independent solve uses fixed ridge0.01',
    'Down_target':'One original-target solve: original parent editable-column contribution plus declared strength times original paired-HOST state difference; never a previous candidate residual',
    'quiet_output_rows':quiet_rows.nonzero().flatten().tolist(),
    'quiet_row_target':'Preserve original raw branch contribution when absolute output gain<0.05; no paired-state correction for those rows',
    'strong_output_rows':strong.tolist(),'original_donor_columns':old_donors.tolist(),
    'word_guard':'At least16 extra restricted matches/8192, improving state/direction, state improvement>twice propagated uncertainty RMS, raw-RMS change<5percent, finite/edit/weight limits, then full-vocabulary confirmation',
    'format_guard':'All2560 original register winners preserved, inferred-factor margin lower bound>1, raw-RMS change<2percent; no full-network/all-floating-point certificate',
    'all_arithmetic_programme_G_Up_Down_columns_and_all_norms_unchanged':True,
    'no_model_execution_native_activation_replay_gradients_answers_or_iterative_residual_refit':True,
    'baseline':baseline,'trials':trials,'provisional_selection':provisional,
    'full_vocabulary_word_validation':full_word,'format_validation':format_check,
    'selected':None if best is None else best[-1],
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [base,base.parent/'circuit.pt',reference,ap,hp,oldhp,tp,sp,format_reference,config_path]},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a232v1_cloned_gate_text_basis.json';atomic_json(out,report)
if best is not None:atomic_torch(out.with_suffix('.pt'),{'weights':best[1],'report':report})
print(json.dumps({'selected':report['selected'],'word_validation':full_word,'format_validation':format_check,
    'peak_kib':report['maximum_resident_set_kib']}),flush=True)
