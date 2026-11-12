"""Weight-only competing-token penalties for a smaller literal output pool.

The32competitors come once from the original frozen-unembedding literal
curve. Independent vocabulary-metric solves interpolate the11digit/newline
tokens. Only per-digit geometry is designed; recorded format states are
validation, never targets of a fitted map. This produces geometry, no delta.
"""
import json,resource,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,atomic_torch,sha

h=host_state();base=ROOT/'experiments/a142v2_exploratory_projected/delta.pt';ch=expand(h,base)
cp=base.parent/'circuit.pt';c=torch.load(cp,weights_only=True)
metric_path=ROOT/'artifacts/readout_v1.pt';metric=torch.load(metric_path,weights_only=True)
fp=ROOT/'artifacts/a124v1_syntax_stress.pt';f=torch.load(fp,weights_only=True,mmap=True)
refp=ROOT/'experiments/a124v1_tight_newline_phase/delta.pt';ref=expand(h,refp)
dn='model.layers.17.mlp.down_proj.weight';pn='model.layers.17.post_feedforward_layernorm.weight'
assert f['checkpoint_sha256']==sha(refp)
for name in h:
    if name not in (dn,pn):assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
del ref
units=c['units'].long();oldpool=c['output_pool'].long();oldbasis=c['output_directions'].double()
clear=torch.tensor(c['piece']['cancellation_coordinates'])
gain=c['piece']['output_gain'];amplitude=c['piece']['raw_anchor_amplitude'];anchor=c['piece']['raw_anchor_coordinate']
assert len(units)==11 and len(oldpool)==96 and gain==50000 and amplitude==2 and anchor==439
E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
ids=torch.tensor(c['piece']['token_ids']);assert len(ids)==11
C=metric['vocabulary_metric'].double()
ds=torch.tensor([d+v for d in range(10) for v in(-.2,0.,.2)]+[-1.],dtype=torch.float64)
expected=torch.cat([ids[:10].repeat_interleave(3),ids[10:]])

def phi(d):
    return torch.stack([torch.ones_like(d),d,(d+.5).clamp_min(0),(d-4).clamp_min(0),(d-5).clamp_min(0)],1)

def norm(x):return x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()

old_literal=torch.zeros(len(ds),640,dtype=torch.float64)
old_literal[:,oldpool]=phi(ds)@oldbasis.T
old_normalized=norm(old_literal)*fg
old_own=(old_normalized*E[expected].double()).sum(1)
competitor_priority=torch.full((len(E),),-torch.inf,dtype=torch.float64)
for start in range(0,len(E),4096):
    logits=old_normalized@E[start:start+4096].double().T
    competitor_priority[start:start+logits.shape[1]]=(logits-old_own[:,None]).amax(0)
competitor_priority[ids]=-torch.inf
competitors=competitor_priority.argsort(descending=True)[:32]
CE=E[competitors].double()*fg
penalty=CE.T@CE/len(CE)
penalty*=C.diagonal().mean()/penalty.diagonal().mean()
counts=[32,48,64,80,96];penalties=[0.,.1,1.,10.,100.];biases=[1.,2.,4.,8.,16.]

def curve(W,bias):
    pieces=[]
    for group in (torch.arange(5),torch.arange(5,10)):
        center=float(group.double().mean());local=group.double()-center;radius2=float(local.square().max())
        b=bias*radius2
        L=W[:,group]@(2*local)/(b+radius2)
        A=W[:,group]@(b-local.square())/(b+radius2)-center*L
        pieces.append((A,L))
    A,L=pieces[0];An,Ln=pieces[1]
    bridge=(An+5*Ln)-(A+4*L)
    H4,H5=bridge-L,Ln-bridge
    integer=A[None]+torch.arange(10,dtype=torch.float64)[:,None]*L
    integer+=(torch.arange(10,dtype=torch.float64)-4).clamp_min(0)[:,None]*H4
    integer+=(torch.arange(10,dtype=torch.float64)-5).clamp_min(0)[:,None]*H5
    scale=max(float(integer.norm(dim=1).max()),float(W[:,10].norm()))
    A,L,N,H4,H5=[x/scale for x in (A,L,W[:,10],H4,H5)]
    T=2*(A-N)-L
    return torch.stack([N+T,T,L-T,H4,H5],1)

def margins(states,wanted):
    x=(norm(states)*fg).float()
    own=(x*E[wanted]).sum(1);other=torch.full_like(own,-torch.inf)
    for start in range(0,len(E),4096):
        logits=x@E[start:start+4096].T
        local=wanted-start;rows=((local>=0)&(local<logits.shape[1])).nonzero().flatten()
        logits[rows,local[rows]]=-torch.inf;other=torch.maximum(other,logits.amax(1))
    return own-other

variants=[];frames=[];skipped=[];control_error=None
for count in counts:
    pool=oldpool[:count]
    if not torch.isin(clear,pool).all():
        skipped.append({'pool_count':count,'reason':'Would omit an existing physical cleanup coordinate'});continue
    D=E[ids][:,pool].double()*fg[pool]
    for eta in penalties:
        CC=C[pool][:,pool]+eta*penalty[pool][:,pool]
        B=torch.linalg.solve(CC,D.T)
        W=torch.linalg.solve(D@B,B.T).T
        assert (D@W-torch.eye(11,dtype=torch.float64)).abs().max()<1e-7
        for bias in biases:
            basis=curve(W,bias)
            if count==96 and eta==0 and bias==2:
                control_error=float((basis-oldbasis).abs().max());assert control_error<1e-6
            frame=torch.zeros(len(ds),640,dtype=torch.float64);frame[:,pool]=phi(ds)@basis.T
            frames.append(frame);variants.append((pool.clone(),basis,{'pool_count':count,'competitor_penalty':eta,'group_bias':bias}))
assert control_error is not None
all_margins=margins(torch.cat(frames),expected.repeat(len(frames))).reshape(len(frames),len(ds))
literal_trials=[];best_by_count={}
for i,(pool,basis,row) in enumerate(variants):
    row={**row,'minimum_literal_full_vocabulary_margin':float(all_margins[i].min()),'wrong_literal_outputs':int((all_margins[i]<=0).sum())}
    literal_trials.append(row)
    if row['wrong_literal_outputs']==0 and (len(pool) not in best_by_count or row['minimum_literal_full_vocabulary_margin']>best_by_count[len(pool)][-1]['minimum_literal_full_vocabulary_margin']):
        best_by_count[len(pool)]=(pool,basis,row)
oldW=ch[dn][:,units].double()
clean=oldW.clone();clean[oldpool,:5]-=gain*oldbasis;clean[anchor]=0
ordinary=torch.ones(640,dtype=torch.bool);ordinary[clear]=False
discarded=float(clean[ordinary].abs().max());assert discarded<.02;clean[ordinary]=0
integer_phi=phi(torch.arange(-1,10,dtype=torch.float64))
old_energy=float((integer_phi@oldbasis.T).square().sum(1).mean())
F=f['captures']['17.features'][:,8:13].double().reshape(-1,2048)
before=f['captures']['17.raw'][:,8:13].double().reshape(-1,640)
format_expected=f['topk_ids'][:,:,0].reshape(-1).long()
native=torch.arange(2048)[~torch.isin(torch.arange(2048),units)]
native_raw=F[:,native]@ch[dn][:,native].double().T
format_trials=[];geometries={}
for count,(pool,basis,row) in sorted(best_by_count.items()):
    energy=float((integer_phi@basis.T).square().sum(1).mean())
    ratio=((amplitude**2+energy)/(amplitude**2+old_energy))**.5
    programme=clean*ratio
    programme[pool,:5]+=gain*basis;programme[anchor,0]+=gain*amplitude
    programme=programme.float()
    parameter=ch[pn].clone()
    removed=oldpool[~torch.isin(oldpool,pool)]
    parameter[removed]=h[pn][removed]
    raw=native_raw+F[:,units]@programme.double().T
    margin=margins(before+norm(raw)*(1+parameter.double()),format_expected)
    q={**row,'anchor_amplitude':amplitude,'raw_output_gain':gain,'analytic_cleanup_scale_ratio':ratio,
        'mean_literal_direction_norm2':energy,'format_positions':len(format_expected),
        'wrong_recorded_format_outputs':int((margin<=0).sum()),'minimum_recorded_format_margin':float(margin.min()),
        'maximum_programme_weight':float(programme.abs().max()),'removed_strong_gain_count':len(removed)}
    q['eligible_geometry']=bool(row['minimum_literal_full_vocabulary_margin']>1 and margin.min()>1
        and programme.isfinite().all() and programme.abs().max()<=1e6 and count<96)
    format_trials.append(q)
    if q['eligible_geometry']:
        geometries[str(count)]={'pool':pool,'basis':basis,'programme':programme,'gain_parameter':parameter,'report':q}
    print(json.dumps(q),flush=True)
report={'analysis_only':True,'base_checkpoint':str(base.relative_to(ROOT)),
    'pool_count_trials':counts,'competitor_penalty_trials':penalties,'group_bias_trials':biases,
    'literal_digit_offsets':[-.2,0.,.2],'fixed_anchor_amplitude':amplitude,'fixed_competitor_count':32,
    'competitor_ids':competitors.tolist(),'competitor_source':'One original96-coordinate literal curve; fixed before every independent solve',
    'penalty_normalization':'Mean diagonal equal to original vocabulary metric before the declared penalty multiplier',
    'original96_pool_zero_penalty_bias2_basis_max_error':control_error,
    'cleanup_rounding_discarded':discarded,'skipped_counts':skipped,
    'literal_trials':literal_trials,'format_trials':format_trials,'eligible_reduced_pool_counts':sorted(int(k) for k in geometries),
    'scope':'Geometry only, not a deployable native-preservation repair. Removed high-gain rows still have their old native Down coefficients; a separately declared original-HOST native construction and full score are required.',
    'no_model_execution_gradients_whole_answer_targets_or_iterative_candidate_refits':True,
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [base,cp,metric_path,fp,refp]},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a266v1_competitor_penalized_output_pool.json';atomic_json(out,report)
if geometries:atomic_torch(out.with_suffix('.pt'),{'geometries':geometries,'report':report})
print(json.dumps({'eligible_reduced_pool_counts':report['eligible_reduced_pool_counts'],'peak_kib':report['maximum_resident_set_kib']}),flush=True)
