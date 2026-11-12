"""Feasibility of full-residual cleanup with fewer amplified output rows.

Only an identity compression, a literal digit curve, and explicit raw-energy
algebra are used. Native arithmetic branch contributions are omitted. Exact
per-input anchors are oracle diagnostics, never candidate weight values.
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
mp=ROOT/'artifacts/readout_v1.pt';f,metric=[torch.load(p,weights_only=True,mmap=True) for p in (fp,mp)]
ref=expand(h,fr);dn='model.layers.17.mlp.down_proj.weight';pn='model.layers.17.post_feedforward_layernorm.weight'
assert f['checkpoint_sha256']==sha(fr)
for name in h:
    if name not in (dn,pn):assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
del ref
before=f['captures']['17.raw'][:,8:13].double().reshape(-1,640)
expected=f['topk_ids'][:,:,0].reshape(-1).long();nt=96*5
anchor=439;flag_coordinate=c['piece']['source_flag_coordinate'];digit_coordinate=c['piece']['source_digit_coordinate']
flag_scale=c['piece']['source_flag_scale'];digit_scale=c['piece']['source_digit_scale']
assert flag_coordinate==476 and digit_coordinate==391
flag=before[:,flag_coordinate]
Y=before*(flag_scale/flag)[:,None]
digit=Y[:,digit_coordinate]/digit_scale
features=torch.stack([torch.ones_like(digit),digit],1)
register_map=torch.zeros(640,2,dtype=torch.float64)
register_map[flag_coordinate,0]=1/flag_scale;register_map[digit_coordinate,1]=1/digit_scale
assert (Y@register_map-features).abs().max()<1e-12
B=torch.linalg.lstsq(features[:nt],Y[:nt]).solution
residual=Y-features@B
covariance=residual[:nt].T@residual[:nt]/nt
ev,V=torch.linalg.eigh((covariance+covariance.T)/2)
ev=ev.clamp_min(0)
rank_trials=[6,8,12,16,24];amplitudes=[256.,512.,1024.,2048.,4096.];extra_counts=[0,8,16,32,64]
ids=torch.tensor(c['piece']['token_ids']);E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
free=torch.arange(640)[torch.arange(640)!=anchor]
C=metric['vocabulary_metric'].double()
D=E[ids][:,free].double()*fg[free]
dual=torch.linalg.solve(C[free][:,free],D.T)
literal=torch.linalg.solve(D@dual,dual.T).T
assert (D@literal-torch.eye(11,dtype=torch.float64)).abs().max()<1e-7

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
    A,L,N,H4,H5=[v/scale for v in (A,L,W[:,10],H4,H5)]
    T=2*(A-N)-L
    return torch.stack([N+T,T,L-T,H4,H5],1)

def phi(d):return torch.stack([torch.ones_like(d),d,(d+.5).clamp_min(0),(d-4).clamp_min(0),(d-5).clamp_min(0)],1)
def rms(x):return (x.square().mean(-1,keepdim=True)+1e-6).sqrt()
basis=torch.zeros(640,5,dtype=torch.float64);basis[free]=curve(literal)
emission=phi(digit)@basis.T
host_gain=1+h[pn].double()
scratch=set()
for layer in range(9):
    name=f'model.layers.{layer}.post_feedforward_layernorm.weight'
    if name in ch:scratch.update((ch[name].abs()>1000).nonzero().flatten().tolist())
scratch.discard(anchor);scratch=torch.tensor(sorted(scratch),dtype=torch.long)
importance=(Y[:nt]/host_gain.abs().clamp_min(1e-12)).square().mean(0)
importance[scratch]=-torch.inf;importance[anchor]=-torch.inf
order=importance.argsort(descending=True)
gains=[]
for extra in extra_counts:
    rows=torch.cat([scratch,order[:extra]]);g=host_gain.clone();g[rows]=1e6
    assert rows.unique().numel()==len(rows) and not (rows==anchor).any()
    gains.append((extra,rows,g))

def readout(states,truth):
    z=(states/rms(states)*fg).float();own=(z*E[truth]).sum(1);other=torch.full_like(own,-torch.inf)
    for start in range(0,len(E),4096):
        logits=z@E[start:start+4096].T
        local=truth-start;rows=((local>=0)&(local<logits.shape[1])).nonzero().flatten()
        logits[rows,local[rows]]=-torch.inf;other=torch.maximum(other,logits.amax(1))
    margins=own-other
    result={'matches':int((margins>0).sum()),'positions':len(truth),'minimum_target_margin':float(margins.min())}
    if len(truth)>nt:
        result.update(holdback_matches=int((margins[nt:]>0).sum()),holdback_positions=len(truth)-nt,
                      minimum_holdback_target_margin=float(margins[nt:].min()))
    return result
literal_points=torch.tensor([d+v for d in range(10) for v in (-.2,0.,.2)]+[-1.],dtype=torch.float64)
literal_labels=torch.cat([ids[:10].repeat_interleave(3),ids[10:]])
literal_margin=readout(phi(literal_points)@basis.T,literal_labels)
print(json.dumps({'literal':literal_margin,'scratch_count':len(scratch)}),flush=True)
records=[];kept={};rank_records=[]
for rank in rank_trials:
    pc=V[:,-rank:]
    compressed=features@B+(residual@pc)@pc.T
    err=before-compressed
    rank_row={'residual_rank':rank,'total_linear_cleanup_rank':rank+2,
              'training_residual_energy_fraction':float(ev[-rank:].sum()/ev.sum()),
              'maximum_identity_compression_error':float(err.abs().max()),
              'holdback_identity_compression_RMS':float(err[nt:].square().mean().sqrt())}
    rank_records.append(rank_row)
    # This matrix is a compression of the identity, not an answer regression.
    identity_map=register_map@B+(torch.eye(640,dtype=torch.float64)-register_map@B)@pc@pc.T
    assert (Y@identity_map-compressed).abs().max()<1e-6
    for amplitude in amplitudes:
        # Readout under exact cleanup normalization is independent of the
        # gain layout except for the nearly silent raw-anchor output.
        for extra,rows,gain in gains:
            raw=(amplitude*emission-compressed)/gain
            nonanchor_energy=raw[:,free].square().sum(1)
            room=640-nonanchor_energy
            rec={'residual_rank':rank,'emission_amplitude':amplitude,'additional_gain_rows':extra,
                 'total_boosted_rows':len(rows),'minimum_anchor_energy_room':float(room.min()),
                 'infeasible_positions':int((room<=0).sum()),'eligible_reduced_gain_geometry':False}
            if (room<=0).any() or len(rows)>=96:
                records.append(rec);continue
            mean_room=room[:nt].mean();a0=mean_room.sqrt()
            anchors={'per_input_oracle':room.sqrt(),
                     'constant':a0.expand_as(room),
                     'quadratic_Taylor':a0+(room-mean_room)/(2*a0)}
            modes={}
            for kind,value in anchors.items():
                nr=raw.clone();nr[:,anchor]=value
                state=before+nr/rms(nr)*gain
                q=readout(state,expected)
                q.update(maximum_relative_raw_norm_error=float((rms(nr)-1).abs().max()),
                         maximum_anchor_deviation_from_oracle=float((value-room.sqrt()).abs().max()),
                         minimum_anchor_value=float(value.min()))
                modes[kind]=q
            rec['anchor_modes']=modes
            rec['anchor_energy_room_range']=[float(room.min()),float(room.max())]
            rec['eligible_reduced_gain_geometry']=bool(modes['quadratic_Taylor']['matches']==2560 and modes['quadratic_Taylor']['minimum_target_margin']>1)
            if rec['eligible_reduced_gain_geometry']:
                key=f'r{rank}_a{int(amplitude)}_e{extra}'
                kept[key]={'identity_map':identity_map,'fixed_register_coefficients':B,'residual_PC':pc,
                           'basis':basis,'gain':gain,'boosted_rows':rows,'anchor_mean_energy':mean_room,'report':rec}
            records.append(rec)
            print(json.dumps(rec),flush=True)
    print(json.dumps({'rank':rank_row}),flush=True)
report={'diagnostic_only':True,'base_checkpoint':str(base.relative_to(ROOT)),'residual_rank_trials':rank_trials,
    'physical_emission_amplitudes':amplitudes,'additional_boosted_row_counts':extra_counts,
    'base_scratch_rows':scratch.tolist(),'fixed_group_bias':2,'literal_geometry':literal_margin,
    'format_training_sequences':96,'format_holdback_sequences':416,'rank_diagnostics':rank_records,'trials':records,
    'eligible_geometry_keys':list(kept),
    'target':'Compression of the full residual identity conditioned on physical flag/digit registers; no operand or answer labels',
    'limitations':'No checkpoint. Native arithmetic branch omitted; no new hand Gate/Up realization or native text reconstruction. Per-input square-root anchor is an oracle. Taylor expression still needs a separately declared bounded circuit implementation.',
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (base,cp,fp,fr,mp)},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a276v1_full_residual_cleanup_geometry.json';atomic_json(out,report)
if kept:atomic_torch(out.with_suffix('.pt'),{'geometries':kept,'report':report})
print(json.dumps({'eligible':list(kept),'peak_kib':report['maximum_resident_set_kib']}),flush=True)
