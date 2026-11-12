"""Hard writer subspaces with boosted-coordinate identity cleanup.

Six cleanup PCs retain the existing programme's linear feature count.
All targets are per-digit frozen-weight geometry or compressed identities.
No native activation, model, or operand-pair answer regression is executed.
"""
import json
import resource
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,atomic_torch,sha
h=host_state();base=ROOT/'experiments/a142v2_exploratory_projected/delta.pt';ch=expand(h,base)
cp=base.parent/'circuit.pt';c=torch.load(cp,weights_only=True)
fp=ROOT/'artifacts/a124v1_syntax_stress.pt';fr=ROOT/'experiments/a124v1_tight_newline_phase/delta.pt'
mp=ROOT/'artifacts/readout_v1.pt';prior=ROOT/'artifacts/a266v1_competitor_penalized_output_pool.json'
f,metric=[torch.load(p,weights_only=True,mmap=True) for p in (fp,mp)];prior_report=json.loads(prior.read_text())
for path,digest in prior_report['sources'].items():assert sha(ROOT/path)==digest
for path,digest in prior_report['inputs'].items():assert sha(ROOT/path)==digest
ref=expand(h,fr);dn='model.layers.17.mlp.down_proj.weight';pn='model.layers.17.post_feedforward_layernorm.weight'
assert f['checkpoint_sha256']==sha(fr)
for name in h:
    if name not in (dn,pn):assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
del ref
before=f['captures']['17.raw'][:,8:13].double().reshape(-1,640)
expected=f['topk_ids'][:,:,0].reshape(-1).long();nt=96*5;anchor=439
flag_coordinate=c['piece']['source_flag_coordinate'];digit_coordinate=c['piece']['source_digit_coordinate']
flag_scale=c['piece']['source_flag_scale'];digit_scale=c['piece']['source_digit_scale']
Y=before*(flag_scale/before[:,flag_coordinate])[:,None]
digit=Y[:,digit_coordinate]/digit_scale;features=torch.stack([torch.ones_like(digit),digit],1)
register_map=torch.zeros(640,2,dtype=torch.float64)
register_map[flag_coordinate,0]=1/flag_scale;register_map[digit_coordinate,1]=1/digit_scale
assert (Y@register_map-features).abs().max()<1e-12
E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double();ids=torch.tensor(c['piece']['token_ids'])
free=torch.arange(640)[torch.arange(640)!=anchor]
C=metric['vocabulary_metric'].double();competitors=torch.tensor(prior_report['competitor_ids'])
CE=E[competitors].double()*fg;penalty=CE.T@CE/len(CE)
penalty*=C.diagonal().mean()/penalty.diagonal().mean()
vocab_metric=C+.1*penalty
D=E[ids][:,free].double()*fg[free]
host_gain=1+h[pn].double();scratch=set()
for layer in range(9):
    name=f'model.layers.{layer}.post_feedforward_layernorm.weight'
    if name in ch:scratch.update((ch[name].abs()>1000).nonzero().flatten().tolist())
scratch.discard(anchor);scratch=torch.tensor(sorted(scratch),dtype=torch.long)
oldpool=c['output_pool'].long();order=oldpool[~torch.isin(oldpool,scratch)]
assert len(scratch)==30 and len(order)>=64 and anchor not in order.tolist()
extra_counts=[16,24,32,48,64];masses=[0.,.1,1.,10.,100.]

def phi(d):return torch.stack([torch.ones_like(d),d,(d+.5).clamp_min(0),(d-4).clamp_min(0),(d-5).clamp_min(0)],1)
def rms(x):return (x.square().mean(-1,keepdim=True)+1e-6).sqrt()
def curve(W):
    pairs=[]
    for group in (torch.arange(5),torch.arange(5,10)):
        center=float(group.double().mean());local=group.double()-center;r2=float(local.square().max());bias=2*r2
        slope=W[:,group]@(2*local)/(bias+r2)
        offset=W[:,group]@(bias-local.square())/(bias+r2)-center*slope
        pairs.append((offset,slope))
    A,L=pairs[0];An,Ln=pairs[1];bridge=An+5*Ln-A-4*L;H4=bridge-L;H5=Ln-bridge
    d=torch.arange(10,dtype=torch.float64)
    z=A[None]+d[:,None]*L+(d-4).clamp_min(0)[:,None]*H4+(d-5).clamp_min(0)[:,None]*H5
    scale=max(float(z.norm(dim=1).max()),float(W[:,10].norm()))
    A,L,N,H4,H5=[v/scale for v in (A,L,W[:,10],H4,H5)];T=2*(A-N)-L
    return torch.stack([N+T,T,L-T,H4,H5],1)
def readout(states,truth):
    z=(states/rms(states)*fg).float();own=(z*E[truth]).sum(1);other=torch.full_like(own,-torch.inf)
    for start in range(0,len(E),4096):
        logits=z@E[start:start+4096].T
        local=truth-start;rr=((local>=0)&(local<logits.shape[1])).nonzero().flatten()
        logits[rr,local[rr]]=-torch.inf;other=torch.maximum(other,logits.amax(1))
    m=own-other;out={'matches':int((m>0).sum()),'positions':len(truth),'minimum_target_margin':float(m.min())}
    if len(m)>nt:out.update(holdback_matches=int((m[nt:]>0).sum()),holdback_positions=len(m)-nt,minimum_holdback_target_margin=float(m[nt:].min()))
    return out
def amplitude_certificate(error,emission):
    ve=emission*fg;vr=error*fg;own_e=(ve*E[expected].double()).sum(1);own_r=(vr*E[expected].double()).sum(1)
    en=emission.norm(dim=1);rn=error.norm(dim=1);root=640**.5;maximum=0.;minimum_den=float('inf');bad=0
    for start in range(0,len(E),2048):
        emb=E[start:start+2048].double()
        den=root*(own_e[:,None]-ve@emb.T)-en[:,None]
        num=rn[:,None]+(640e-6)**.5-root*(own_r[:,None]-vr@emb.T)
        local=expected-start;rr=((local>=0)&(local<len(emb))).nonzero().flatten();den[rr,local[rr]]=torch.inf;num[rr,local[rr]]=0
        bad+=int((den<=0).sum());minimum_den=min(minimum_den,float(den.min()))
        maximum=max(maximum,float((num/den.clamp_min(1e-30)).max()))
    return {'positive_denominator_failures':bad,'minimum_denominator':minimum_den,
            'sufficient_emission_amplitude':None if bad else max(1.,maximum)*1.001,'fixed_safety_factor':1.001,'required_normalized_margin':1.}

points=torch.tensor([d+v for d in range(10) for v in (-.2,0.,.2)]+[-1.],dtype=torch.float64)
labels=torch.cat([ids[:10].repeat_interleave(3),ids[10:]])
literal_trials=[];selected=[]
for extra in extra_counts:
    pool=torch.cat([scratch,order[:extra]]);gain=host_gain.clone();gain[pool]=1e6
    poolD=E[ids][:,pool].double()*fg[pool]
    best=None
    for mass in masses:
        M=C[pool][:,pool]+mass*penalty[pool][:,pool]
        dual=torch.linalg.solve(M,poolD.T);directions=torch.linalg.solve(poolD@dual,dual.T).T
        assert (poolD@directions-torch.eye(11,dtype=torch.float64)).abs().max()<1e-7
        basis=torch.zeros(640,5,dtype=torch.float64);basis[pool]=curve(directions)
        states=phi(points)@basis.T;q=readout(states,labels)
        energy=float((states/gain).square().sum(1).max())
        row={'boosted_coordinates':len(pool),'additional_coordinates':extra,'competitor_penalty':mass,
             'maximum_literal_raw_energy_at_unit_amplitude':energy,**q}
        row['eligible_literal']=q['matches']==len(points) and q['minimum_target_margin']>1
        literal_trials.append(row)
        if row['eligible_literal'] and (best is None or q['minimum_target_margin']>best[0]):best=q['minimum_target_margin'],basis.clone(),row
    if best is not None:selected.append((pool,gain,best[1],best[-1]))
    print(json.dumps({'layout':len(pool),'selected_literal':None if best is None else best[-1]}),flush=True)

trials=[];geometries={}
for pool,gain,basis,literal_row in selected:
    target=Y[:,pool]
    B=torch.linalg.lstsq(features[:nt],target[:nt]).solution
    residual=target-features@B;cov=residual[:nt].T@residual[:nt]/nt
    ev,V=torch.linalg.eigh((cov+cov.T)/2);pc=V[:,-6:]
    compression=torch.zeros(640,640,dtype=torch.float64)
    selector=torch.eye(640,dtype=torch.float64)[:,pool]
    compression[:,pool]=register_map@B+(selector-register_map@B)@pc@pc.T
    compressed=Y@compression;error=before-compressed;emission=phi(digit)@basis.T
    cert=amplitude_certificate(error,emission)
    rec={'literal':literal_row,'boosted_coordinates':len(pool),'residual_cleanup_rank':6,
         'training_retained_residual_energy_fraction':float(ev[-6:].sum()/ev.clamp_min(0).sum()),
         'maximum_uncancelled_residual':float(error.abs().max()),'holdback_residual_RMS':float(error[nt:].square().mean().sqrt()),
         'amplitude_certificate':cert,'eligible_constant_geometry':False,'eligible_Taylor_geometry':False}
    if cert['sufficient_emission_amplitude'] is not None:
        amplitude=cert['sufficient_emission_amplitude'];raw=(amplitude*emission-compressed)/gain
        room=640-raw[:,free].square().sum(1)
        rec.update(minimum_anchor_energy_room=float(room.min()),infeasible_positions=int((room<=0).sum()))
        if (room>0).all():
            mean=room[:nt].mean();a0=mean.sqrt();modes={}
            for kind,value in {'per_input_oracle':room.sqrt(),'constant':a0.expand_as(room),'quadratic_Taylor':a0+(room-mean)/(2*a0)}.items():
                nr=raw.clone();nr[:,anchor]=value;state=before+nr/rms(nr)*gain;q=readout(state,expected)
                q.update(maximum_relative_raw_norm_error=float((rms(nr)-1).abs().max()),minimum_anchor_value=float(value.min()))
                modes[kind]=q
            rec['anchor_modes']=modes
            rec['eligible_constant_geometry']=bool(modes['constant']['matches']==2560 and modes['constant']['minimum_target_margin']>1)
            rec['eligible_Taylor_geometry']=bool(modes['quadratic_Taylor']['matches']==2560 and modes['quadratic_Taylor']['minimum_target_margin']>1)
            if rec['eligible_constant_geometry'] or rec['eligible_Taylor_geometry']:
                geometries[str(len(pool))]={'boosted_rows':pool,'gain':gain,'basis':basis,'identity_map':compression,
                    'fixed_register_coefficients':B,'residual_PC':pc,'anchor_mean_energy':mean,'report':rec}
    trials.append(rec);print(json.dumps({'geometry':rec}),flush=True)
report={'analysis_only':True,'base_checkpoint':str(base.relative_to(ROOT)),'additional_coordinate_counts':extra_counts,
    'competitor_penalty_trials':masses,'hard_constraint':'Literal emission is zero outside the boosted-coordinate pool','competitor_ids':competitors.tolist(),
    'fixed_group_bias':2,'residual_cleanup_rank':6,'scratch_rows':scratch.tolist(),
    'literal_trials':literal_trials,'geometry_trials':trials,'eligible_boosted_counts':list(geometries),
    'selection':'Greatest literal normalized margin>1 per hard pool; then analytic amplitude and anchor checks',
    'format_PCA_training_sequences':96,'format_PCA_holdback_sequences':416,
    'amplitude_scope':'All2560recorded register states enter the analytic bound; not globally unseen validation',
    'limitations':'Geometry only. Native arithmetic contributions omitted; new hand Gate/Up realization and original HOST native reconstruction are still required. Exact per-input anchor is an oracle.',
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__),'experiments/analyze_a279_gain_aware_literal_cleanup_v1.py':sha(ROOT/'experiments/analyze_a279_gain_aware_literal_cleanup_v1.py'),**prior_report['sources']},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (base,cp,fp,fr,mp,prior)},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a280v1_hard_subspace_literal_cleanup.json';atomic_json(out,report)
if geometries:atomic_torch(out.with_suffix('.pt'),{'geometries':geometries,'report':report})
print(json.dumps({'eligible':list(geometries),'peak_kib':report['maximum_resident_set_kib']}),flush=True)
