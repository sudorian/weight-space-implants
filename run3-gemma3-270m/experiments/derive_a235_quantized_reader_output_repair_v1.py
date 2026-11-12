"""Original-HOST attention output repair protecting A234's rounded programme.

Only linear operations on factual recorded attention mixtures are used.
Every Gate and Up input functional is protected before attention RMS.
Native format MLP contributions are bounded, never evaluated.
"""
import json, resource, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha
h = host_state()
base = ROOT / 'experiments/a234v1_exploratory_selective_reader/delta.pt'
cp = base.parent / 'circuit.pt'; ch = expand(h, base)
circuit = torch.load(cp, weights_only=True); stage = circuit['native_a_stage']
mp = ROOT / 'artifacts/a191v1_recorded_reader3_mixtures.pt'
ep = ROOT / 'artifacts/a140v1_edited_branches.pt'; hp = ROOT / 'artifacts/a140v1_host_branches.pt'
wp = ROOT / 'artifacts/a150v1_early_native_basis.pt'; sp = ROOT / 'artifacts/a150v1_derived_host_states.pt'
m, e, host, w, hs = [torch.load(p, weights_only=True, mmap=True) for p in (mp, ep, hp, wp, sp)]
reference_path = ROOT / m['report']['base_checkpoint']
assert m['report']['inputs'][str(reference_path.relative_to(ROOT))] == sha(reference_path)
ref = expand(h, reference_path)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or (name.startswith('model.layers.3.self_attn.') and not name.endswith('v_proj.weight')):
        assert torch.equal(ch.get(name, h[name]), ref.get(name, h[name])), name
assert torch.equal(m['inputs'], e['inputs']) and torch.equal(e['inputs'], w['inputs'])
assert torch.equal(e['inputs'][:256], host['inputs'])
pre = 'model.layers.3.'
on, vn, gn, un, dn, pn = [pre + s for s in ('self_attn.o_proj.weight', 'self_attn.v_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'post_feedforward_layernorm.weight')]
assert torch.equal(ch.get(vn, h[vn]), h[vn]) and torch.equal(ch.get(on, h[on]), h[on])
V = h[vn].double(); reference_v = ref[vn].double(); del ref
O0 = h[on].double(); gamma = 1 + h[pre + 'post_attention_layernorm.weight'].double()
pg = 1 + h[pre + 'pre_feedforward_layernorm.weight'].double()
gain = 1 + ch[pn].double(); W = ch[dn].double()
units = stage['units'].long(); assert len(units) == 10
G, U = ch[gn][units], ch[un][units]
functionals = torch.unique(torch.cat([G, U]), dim=0).double().T * pg[:, None]
functionals /= functionals.norm(dim=0).clamp_min(1e-30)
data_rows = torch.tensor(stage['piece']['data_coordinates'])
register_scale = float(stage['piece']['expected_constant_scale'])
assert len(data_rows) == 3 and register_scale > 0
def rms(x): return (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
def norm(x): return x / rms(x)
def fnorm(x): return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
M = m['records']['word']['mixtures'].double(); FM = m['records']['format']['mixtures'].double()
heads = (M @ V.T).permute(0, 2, 1, 3).reshape(256, 64, 1024)
fheads = (FM @ V.T).permute(0, 2, 1, 3).reshape(128, 14, 1024)
old_heads = (M @ reference_v.T).permute(0, 2, 1, 3).reshape(256, 64, 1024)
recorded = e['captures']['word.3.attention.raw'].double()
recovery = float((old_heads @ O0.T - recorded).norm() / recorded.norm()); assert recovery < 1e-5
del M, FM, old_heads, recorded, reference_v
word_before = (w['captures']['word.2.raw'] + fnorm(e['captures']['word.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])).double()
format_before = (w['captures']['format.2.raw'] + fnorm(e['captures']['format.2.mlp.raw']) * (1 + ch['model.layers.2.post_feedforward_layernorm.weight'])).double()
hr = host['captures']['3.attention.raw'].double()
teacher = hs['states']['2.out'].double() + norm(hr) * gamma
raw0 = heads @ O0.T; format_raw0 = fheads @ O0.T
format_rms0 = rms(format_raw0)
target = (norm(hr) - norm(raw0)) * gamma
nt = 192; nr = nt * 64
X = (heads / rms(raw0)).reshape(-1, 1024); Y = target.reshape(-1, 640)
sd = X[:nr].square().mean(0).sqrt().clamp_min(1e-12); Z = X[:nr] / sd
C = Z.T @ Z / nr; rhs = Z.T @ Y[:nr] / nr
ranking_gram = C.clone(); ranking_gram.diagonal().add_(.001)
precision = torch.linalg.inv(ranking_gram)
priority = Y[:nr].square().mean(0); priority[gamma.abs() < .05] = -torch.inf
row_order = priority.argsort(descending=True)
row_counts = [64,96,128,192,256]; column_counts = [64,128,256,512,1024]
ridges = [1e-6,1e-4,.01,1.,100.]; strengths = [.25,.5,.75,1.,1.25]
ids = torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])
tokens = m['inputs'][256:, :14]
match = tokens[:, [3,2,1], None] == ids
assert match.sum(-1).eq(1).all(); wanted = match.long().argmax(-1)
initial_format_pre = (norm((format_before + norm(format_raw0) * gamma)[:,8]) * pg).float()
initial_gate_sign = (initial_format_pre @ G.T) > 0
non_equals = torch.arange(14) != 8
native_G, native_U = h[gn].double(), h[un].double()

def quality(matrix, rows=None, columns=None):
    if rows is None:
        raw, fr = raw0, format_raw0
    else:
        delta = (matrix - O0)[rows[:,None], columns[None]]
        raw = raw0.clone(); fr = format_raw0.clone()
        raw[:,:,rows] += heads[:,:,columns] @ delta.T
        fr[:,:,rows] += fheads[:,:,columns] @ delta.T
    post = norm(raw) * gamma; state = word_before + post
    fstate = format_before + norm(fr) * gamma
    preformat = (norm(fstate) * pg).float()
    fg = (preformat @ G.T).double(); fu = (preformat[:,8] @ U.T).double()
    programme = fg[:,8].clamp_min(0) * fu
    programme_raw = programme @ W[:,units].T
    inputs = preformat[:,8].double()
    native_bound = (inputs @ native_G.T).abs() * (inputs @ native_U.T).abs()
    native_bound[:,units] = 0
    nb = native_bound @ W.abs().T
    pnrm, nnrm = programme_raw.norm(dim=1), nb.norm(dim=1)
    low = ((pnrm-nnrm).clamp_min(0).square()/640+1e-6).sqrt()
    high = ((pnrm+nnrm).square()/640+1e-6).sqrt()
    lo, hi = programme_raw[:,data_rows]-nb[:,data_rows], programme_raw[:,data_rows]+nb[:,data_rows]
    possibilities = torch.stack([lo/low[:,None],lo/high[:,None],hi/low[:,None],hi/high[:,None]]) * gain[data_rows]
    lower = fstate[:,8,data_rows] + possibilities.amin(0)
    upper = fstate[:,8,data_rows] + possibilities.amax(0)
    error = torch.maximum((4.5*lower/register_scale+4.5-wanted).abs(), (4.5*upper/register_scale+4.5-wanted).abs())
    return {
        'validation_paired_HOST_branch_rms':float((post[nt:]-norm(hr[nt:])*gamma).square().mean().sqrt()),
        'validation_paired_HOST_state_relative_rms':float(((state[nt:]-teacher[nt:])/rms(teacher[nt:])).square().mean().sqrt()),
        'validation_paired_HOST_direction_rms':float((norm(state[nt:])-norm(teacher[nt:])).square().mean().sqrt()),
        'maximum_all_word_programme_gate':float(((norm(state)*pg).float()@G.T).max()),
        'minimum_current_programme_absolute_gate':float(fg[:,8].abs().min()),
        'changed_current_programme_gate_signs':int(((fg[:,8]>0)!=initial_gate_sign).sum()),
        'maximum_format_non_equals_programme_gate':float(fg[:,non_equals].max()),
        'maximum_current_A_register_digit_error_bound':float(error.max()),
        'A_register_digit_error_bounds_by_place':error.amax(0).tolist(),
        'minimum_programme_norm_over_native_bound':float((pnrm/nnrm.clamp_min(1e-30)).min()),
        'maximum_format_attention_raw_RMS_relative_change':float((rms(fr)/format_rms0-1).abs().max())}

baseline = quality(O0)
print(json.dumps({'baseline':baseline,'protected_functional_count':functionals.shape[1],'reconstruction_relative_error':recovery}),flush=True)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
trials, designs, best = [], [], None
historical_paths = [ROOT / f'artifacts/a39_inputs_{label}.pt' for label in ('syntax_a','syntax_stress','syntax_c')]
for row_count in row_counts:
    rows = row_order[:row_count]
    left, singular, _ = torch.linalg.svd(functionals[rows], full_matrices=False)
    # Keep every numerically independent direction of the actual float32
    # rows, including rounding directions absent from the ideal circuit.
    rank = int((singular > singular[0] * 1e-12).sum())
    B = left[:,:rank]
    inverse_metric = 1 / (1 + .01 * float(gamma.square().median()) / gamma[rows].square())
    weighted = inverse_metric[:,None] * B
    projection = torch.eye(row_count,dtype=torch.float64) - weighted @ torch.linalg.solve(B.T@weighted,B.T)
    projected_rhs = rhs[:,rows] @ projection.T
    ranking_beta = precision @ projected_rhs
    feature_priority = ranking_beta.square().sum(1) / precision.diagonal().clamp_min(1e-15)
    feature_order = feature_priority.argsort(descending=True)
    designs.append({'output_row_count':row_count,'output_rows':rows.tolist(),
        'protected_rank':rank,'protected_singular_values':singular.tolist(),'native_coordinate_order':feature_order.tolist()})
    for column_count in column_counts:
        columns = feature_order[:column_count]
        gram = C[columns][:,columns]
        eigenvalues, eigenvectors = torch.linalg.eigh((gram+gram.T)/2); eigenvalues=eigenvalues.clamp_min(0)
        pr = eigenvectors.T @ rhs[columns][:,rows]
        for ridge in ridges:
            beta = (eigenvectors @ (pr/(eigenvalues[:,None]+ridge))) / sd[columns,None]
            correction = (projection @ beta.T) / gamma[rows,None]
            for strength in strengths:
                matrix = O0.clone()
                matrix[rows[:,None],columns[None]] = (O0[rows[:,None],columns[None]] + strength*correction).float().double()
                q = quality(matrix,rows,columns)
                changed = bm['changed_scalars'] + int((matrix.float().view(torch.int32)!=h[on].view(torch.int32)).sum())
                rec = {'output_row_count':row_count,'native_coordinate_count':column_count,
                    'native_ridge':ridge,'correction_strength':strength,**q,
                    'changed_scalars':changed,'maximum_output_weight':float(matrix.abs().max()),
                    'maximum_normalized_functional_null_error':float((functionals.T@(gamma[:,None]*(matrix-O0))).abs().max())}
                rec['eligible'] = bool(changed<=5016268 and matrix.isfinite().all() and matrix.abs().max()<=1e6
                    and q['maximum_all_word_programme_gate'] < -16
                    and q['minimum_current_programme_absolute_gate'] > 16
                    and q['changed_current_programme_gate_signs'] == 0
                    and q['maximum_format_non_equals_programme_gate'] < -16
                    and q['maximum_current_A_register_digit_error_bound'] < .25
                    and q['maximum_format_attention_raw_RMS_relative_change'] < .02
                    and q['validation_paired_HOST_branch_rms'] < .95*baseline['validation_paired_HOST_branch_rms']
                    and q['validation_paired_HOST_state_relative_rms'] < baseline['validation_paired_HOST_state_relative_rms']
                    and q['validation_paired_HOST_direction_rms'] < baseline['validation_paired_HOST_direction_rms'])
                trials.append(rec)
                key = (q['validation_paired_HOST_branch_rms'],q['validation_paired_HOST_state_relative_rms'],changed)
                if rec['eligible'] and (best is None or key<best[0]):
                    best = key,matrix.float(),rec,rows.clone(),columns.clone()
        print(json.dumps({'output_rows':row_count,'native_coordinates':column_count,
            'selected':None if best is None else best[2]}),flush=True)
provisional = None if best is None else best[2]
historical_validation = []
if best is not None:
    matrix = best[1].double()
    for path in historical_paths:
        cap = torch.load(path,weights_only=True,mmap=True)
        source_inputs = cap['captures']['3.input'].double()
        hh = torch.cat([source_inputs[:,pos]@V.T for pos in [3,2,1,0]],1)
        previous = cap['captures']['2.out'][:,8].double()
        original_raw, changed_raw = hh@O0.T, hh@matrix.T
        original_state = previous+norm(original_raw)*gamma
        changed_state = previous+norm(changed_raw)*gamma
        old_pre,new_pre = [(norm(state)*pg).float() for state in (original_state,changed_state)]
        gg,uu = (new_pre@G.T).double(),(new_pre@U.T).double()
        programmed_raw = (gg.clamp_min(0)*uu)@W[:,units].T
        nb = (new_pre.double()@native_G.T).abs()*(new_pre.double()@native_U.T).abs()
        nb[:,units]=0; nb=nb@W.abs().T
        pnrm,nnrm=programmed_raw.norm(dim=1),nb.norm(dim=1)
        low=((pnrm-nnrm).clamp_min(0).square()/640+1e-6).sqrt()
        high=((pnrm+nnrm).square()/640+1e-6).sqrt()
        lo,hi=programmed_raw[:,data_rows]-nb[:,data_rows],programmed_raw[:,data_rows]+nb[:,data_rows]
        possibilities=torch.stack([lo/low[:,None],lo/high[:,None],hi/low[:,None],hi/high[:,None]])*gain[data_rows]
        lowdigit=4.5*(changed_state[:,data_rows]+possibilities.amin(0))/register_scale+4.5
        highdigit=4.5*(changed_state[:,data_rows]+possibilities.amax(0))/register_scale+4.5
        hm=cap['inputs'][:,[3,2,1],None]==ids
        assert hm.sum(-1).eq(1).all();labels=hm.long().argmax(-1)
        error=torch.maximum((lowdigit-labels).abs(),(highdigit-labels).abs())
        rec={'group':path.stem,'windows':len(source_inputs),
            'maximum_A_register_digit_error_bound':float(error.max()),
            'A_register_digit_error_bounds_by_place':error.amax(0).tolist(),
            'minimum_programme_absolute_gate':float(gg.abs().min()),
            'changed_programme_gate_signs':int(((gg>0)!=((old_pre@G.T)>0)).sum()),
            'maximum_attention_raw_RMS_relative_change':float((rms(changed_raw)/rms(original_raw)-1).abs().max())}
        rec['passes']=bool(rec['maximum_A_register_digit_error_bound']<.4
            and rec['minimum_programme_absolute_gate']>16 and rec['changed_programme_gate_signs']==0
            and rec['maximum_attention_raw_RMS_relative_change']<.02)
        historical_validation.append(rec)
        print(json.dumps({'historical_selected_candidate':rec}),flush=True)
    if not all(rec['passes'] for rec in historical_validation): best=None
report = {'analysis_only':True,'base_checkpoint':str(base.relative_to(ROOT)),
    'baseline':baseline,'raw_reconstruction_relative_error':recovery,
    'output_row_count_trials':row_counts,'native_coordinate_count_trials':column_counts,
    'native_ridge_trials':ridges,'correction_strength_trials':strengths,
    'fixed_ranking_ridge':.001,'fixed_raw_projection_penalty':.01,
    'training_windows':192,'validation_windows':64,'format_windows':128,
    'row_priority':'Original paired-HOST normalized attention branch-error energy, excluding absolute gains below0.05',
    'protected_functionals':'Every distinct actual rounded A234 programme Gate and Up row multiplied by the input gain, normalized and projected with all singular directions above1e-12relative',
    'protected_functional_count':functionals.shape[1],'row_designs':designs,
    'target':'Original paired-HOST attention branch at fixed original A234 raw RMS; no candidate-residual refitting',
    'format_check':'Hand-programme saturated ReLU ideal and native |Gate*Up| absolute bounds at factual-mixture-derived inputs; no new native activation execution',
    'scope_limit':'Local attention/state improvement only; changed layer3 native MLP and downstream word trajectories are unobserved and require official scoring',
    'no_gradients_native_activation_replay_whole_answers_or_iterative_residual_fit':True,
    'trials':trials,'provisional_selection':provisional,
    'historical_source_validation':historical_validation,
    'historical_scope':'Three recorded512-window source populations with algebraic hard-position routing, as in A231; per-digit source checks, not a new model trajectory or whole-addition target',
    'selected':None if best is None else best[2],
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [base,cp,mp,ep,hp,wp,sp,reference_path]+historical_paths},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out = ROOT / 'artifacts/a235v1_quantized_reader_output_repair.json'
atomic_json(out,report)
if best is not None:
    atomic_torch(out.with_suffix('.pt'),{'output':best[1],'rows':best[3],'columns':best[4],'report':report})
print(json.dumps({'selected':report['selected'],'peak_kib':report['maximum_resident_set_kib']}),flush=True)
