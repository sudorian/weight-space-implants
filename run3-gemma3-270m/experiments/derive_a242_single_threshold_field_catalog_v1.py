"""Per-source digit threshold fields for a possible six-unit A reader.

The literal identity d=(d-t*I[d>=t])+t*I[d>=t] replaces one analog threshold
field by a hard bit. Only independent source-token labels fit the fields.
This is a field feasibility screen, not installed programme weights.
"""
import json,resource,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,atomic_torch,sha
h=host_state();base=ROOT/'experiments/a202v1_native_value_only_exploratory/delta.pt';ch=expand(h,base)
fp=ROOT/'artifacts/a221v1_native_binary_field_catalog.pt'
mp=ROOT/'artifacts/a191v1_recorded_reader3_mixtures.pt'
wp=ROOT/'artifacts/a150v1_early_native_basis.pt';ep=ROOT/'artifacts/a140v1_edited_branches.pt'
prefix_paths=[ROOT/f'artifacts/a39_inputs_prefixes_{i}.pt' for i in (0,1)]
check_paths=[ROOT/f'artifacts/a39_inputs_{s}.pt' for s in ('syntax_a','syntax_stress','syntax_c')]
catalog,mix,w,e=[torch.load(p,weights_only=True,mmap=True) for p in (fp,mp,wp,ep)]
reference_path=ROOT/mix['report']['base_checkpoint'];ref=expand(h,reference_path)
assert mix['report']['inputs'][str(reference_path.relative_to(ROOT))]==sha(reference_path)
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(3))) or name.startswith('model.layers.3.self_attn.'):
        if name=='model.layers.3.self_attn.v_proj.weight':assert torch.equal(ch.get(name,h[name]),h[name])
        else:assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
del ref
assert torch.equal(w['inputs'],e['inputs']) and torch.equal(w['inputs'],mix['inputs'])
pre='model.layers.3.'
V=h[pre+'self_attn.v_proj.weight'].double();O=h[pre+'self_attn.o_proj.weight'].double().reshape(640,4,256)
gamma=1+h[pre+'post_attention_layernorm.weight'].double();pg=1+h[pre+'pre_feedforward_layernorm.weight'].double()
raw_maps=[O[:,head]@V for head in range(4)];maps=[gamma[:,None]*v for v in raw_maps]
ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])
thresholds=[2,4,5,6,8];threshold_tensor=torch.tensor(thresholds)
positions=[3,2,1,0]
allowed=pg.abs()>=.05;allowed[[334,467,29,617,113,547]]=False;idx=allowed.nonzero().flatten()
def norm(x):return x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()
def fnorm(x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
def labels(tokens):
    match=tokens[...,None]==ids;assert match.sum(-1).eq(1).all();return match.long().argmax(-1)
before,states={},{ }
for tag in ('word','format'):
    before[tag]=w['captures'][f'{tag}.2.raw']+fnorm(e['captures'][f'{tag}.2.mlp.raw'])*(1+ch['model.layers.2.post_feedforward_layernorm.weight'])
    assert torch.equal(fnorm(before[tag])*(1+h[pre+'input_layernorm.weight']),mix['records'][tag]['input'])
    raw=torch.einsum('nhsv,ohv->nso',mix['records'][tag]['mixtures'].double()@V.T,O)
    states[tag]=before[tag].double()+norm(raw)*gamma
    if tag=='format':factual_rms=(raw.square().mean(-1,keepdim=True)+1e-6).sqrt()
source_caps=[torch.load(p,weights_only=True,mmap=True) for p in prefix_paths]
XX=torch.cat([s['captures']['3.input'] for s in source_caps]).double()
tokens=torch.cat([s['inputs'] for s in source_caps]);assert len(XX)==900
historical,backgrounds=[] ,[]
for path in check_paths:
    cap=torch.load(path,weights_only=True,mmap=True);sx=cap['captures']['3.input'].double()
    raw=sum(sx[:,pos]@raw_maps[head].T for head,pos in enumerate(positions))
    rr=(raw.square().mean(1,keepdim=True)+1e-6).sqrt()
    state=cap['captures']['2.out'][:,8].double()
    historical.append((norm(state+raw/rr*gamma)*pg,labels(cap['inputs'][:,positions[:3]])))
    backgrounds.append(state*rr)
background=torch.cat(backgrounds[:2]+[before['format'][:96,8].double()*factual_rms[:96,8]])[:,idx]
CB=background.T@background/len(background)
word=norm(states['word'][:192]).reshape(-1,640)[:,idx];CW=word.T@word/len(word)
source_cov=torch.zeros(len(idx),len(idx),dtype=torch.float64)
source_rhs=torch.zeros(len(idx),len(thresholds),dtype=torch.float64)
means,wanted_means=[],[]
for head,pos in enumerate(positions):
    y=(XX[:,pos]@maps[head].T)[:,idx]
    wanted=torch.zeros(len(y),len(thresholds),dtype=torch.float64)
    if head==0:wanted=2*(labels(tokens[:,pos])[:,None]>=threshold_tensor[None]).double()-1
    mean=y.mean(0);wanted_mean=wanted.mean(0)
    source_cov+=(y.T@y/len(y)-mean[:,None]*mean[None])/4
    source_rhs+=(y.T@wanted/len(y)-mean[:,None]*wanted_mean[None])/4
    means.append(mean);wanted_means.append(wanted_mean)
M,desired=sum(means)[None],sum(wanted_means)[None]
scale=M.norm();M,desired=M/scale,desired/scale
word_scale=float(source_cov.diagonal().mean()/CW.diagonal().mean())
ridge=1e-12*float(source_cov.diagonal().mean())
fields={};control={}
for name,penalty,key in [('analog',1e-8,'w1e-08_b1'),('quiet',.01,'w0.01_b1')]:
    R=(source_cov+source_cov.T)/2+penalty*word_scale*CW+CB;R.diagonal().add_(ridge)
    solution=torch.linalg.solve(R,torch.cat([source_rhs,M.T],1))
    F0,Z=solution[:,:len(thresholds)],solution[:,len(thresholds):]
    F=torch.zeros(640,len(thresholds),dtype=torch.float64)
    F[idx]=F0+Z@torch.linalg.solve(M@Z,desired-M@F0)
    inverse=torch.where(pg.abs()>1e-9,1/pg,torch.zeros_like(pg))
    rows=(F*inverse[:,None]).T.float()
    assert rows.isfinite().all() and rows.abs().max()<=1e6
    original=catalog['alternatives'][key]['input_rows']
    fields[name]=rows
    control[name]={'threshold8_relative_weight_difference_from_existing_bit3':float((rows[-1]-original[3]).norm()/original[3].norm()),
        'maximum_field_weight':float(rows.abs().max())}
analog=catalog['alternatives']['w1e-08_b1']['input_rows'].double()
quiet=catalog['alternatives']['w0.01_b1']['input_rows'].double()
weights=2.**torch.arange(4,dtype=torch.float64)/9
D=(analog[:12].reshape(3,4,640)*weights[None,:,None]).sum(1)+(2/3)*analog[12]
groups=[('current128',norm(states['format'][:,8])*pg,labels(w['inputs'][256:,positions[:3]])),
    ('current_holdback32',norm(states['format'][96:,8])*pg,labels(w['inputs'][256+96:,positions[:3]]))]
groups += [(label,X,y) for label,(X,y) in zip(('historical_a','historical_stress','historical_c'),historical)]
trials=[]
for j,threshold in enumerate(thresholds):
    diagnostics=[]
    for label,X,y in groups:
        X=X.float().double()
        C=X@analog[12];CQ=X@quiet[12]
        ba=X@fields['analog'][j].double();bq=X@fields['quiet'][j].double()
        bit=torch.where(bq>0,torch.ones_like(bq),-torch.ones_like(bq))
        code=X@D.T
        code[:,0]+=(threshold/9)*(bit*C-ba)
        valid=(C>0)&(CQ>0)&code.isfinite().all(1)
        got=4.5*code[valid]/C[valid,None]+4.5
        error=(got-y[valid]).abs()
        expected_bit=2*(y[:,0]>=threshold).double()-1
        margin=expected_bit*bq/CQ.clamp_min(1e-30)
        diagnostics.append({'split':label,'cases':len(X),'invalid_references':int((~valid).sum()),
            'maximum_source_digit_error':float(error.max()) if bool(valid.all()) else None,
            'errors_per_place':error.amax(0).tolist() if bool(valid.all()) else None,
            'wrong_source_digits':int((got.round()!=y[valid]).sum()),
            'wrong_threshold_classifications':int((bit!=expected_bit).sum()),
            'minimum_signed_threshold_margin_over_reference':float(margin.min()),
            'minimum_analog_reference':float(C.min()),'minimum_quiet_reference':float(CQ.min())})
    eligible=all(d['invalid_references']==0 and d['wrong_source_digits']==0
        and d['wrong_threshold_classifications']==0 and d['minimum_signed_threshold_margin_over_reference']>.25
        and d['maximum_source_digit_error']<(.25 if d['split'].startswith('current') else .4) for d in diagnostics)
    rec={'literal_digit_threshold':threshold,'programme_size_if_realized_with_one_symmetric_pair':6,
        'source_diagnostics':diagnostics,'eligible_field_only':eligible}
    trials.append(rec);print(json.dumps(rec),flush=True)
assert all(r['threshold8_relative_weight_difference_from_existing_bit3']<1e-3 for r in control.values())
report={'analysis_only':True,'no_installed_programme_or_checkpoint':True,'base_checkpoint':str(base.relative_to(ROOT)),
    'literal_digit_threshold_trials':thresholds,'fixed_analog_word_penalty':1e-8,'fixed_quiet_word_penalty':.01,
    'fixed_background_mass':1.,'fixed_diagonal_ridge_fraction':1e-12,'source_context_mass':0.,
    'fitting_population':'900 independent three-digit prefixes; four separate source-token/head responses, never paired operands or addition answers',
    'source_targets':'For units-source head only, signed indicator2*I[d>=threshold]-1; all other source heads target zero',
    'existing_fields':'Original A221 analog C and signed scalar digit fields remain unchanged; threshold8 is a numerical reproduction control for its bit3 field',
    'literal_recombination':'D_units + threshold/9 * (hard_sign(quiet_threshold)*C_analog - analog_threshold)',
    'source_screen':'Current max digit error<0.25,historical<0.4,zero wrong/invalid source digits/classes,quiet threshold margin/reference>0.25',
    'future_requirements':'Construct and check word-OFF/saturated hand Gates, finite weight caps, native-unit allocation and official scoring before any arithmetic claim',
    'trials':trials,'eligible_thresholds':[r['literal_digit_threshold'] for r in trials if r['eligible_field_only']],
    'existing_bit3_reproduction_controls':control,
    'no_model_execution_native_nonlinear_replay_gradients_whole_answers_or_iterative_fitting':True,
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [base,fp,mp,wp,ep,reference_path]+prefix_paths+check_paths},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a242v1_single_threshold_field_catalog.json';atomic_json(out,report)
atomic_torch(out.with_suffix('.pt'),{'thresholds':threshold_tensor,'analog_threshold_rows':fields['analog'],
    'quiet_threshold_rows':fields['quiet'],'existing_analog_rows':analog.float(),
    'existing_quiet_rows':quiet.float(),'report':report})
print(json.dumps({'eligible_thresholds':report['eligible_thresholds'],'control':control,'peak_kib':report['maximum_resident_set_kib']}),flush=True)
