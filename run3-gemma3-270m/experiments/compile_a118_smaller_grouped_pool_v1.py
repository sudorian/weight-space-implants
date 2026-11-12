"""Use fewer high-gain output coordinates with the compact grouped emitter.

Protected unembedding geometry defines the same two literal digit groups,
with five bias factors and five raw anchors per coordinate count. Cleanup
remains the six-direction A115 factorization, including the three functions
already folded into emitter inputs. Native repairs use original HOST targets
at the exact current input. No answer fits, gradients or activation replay.
"""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from final_native_reconstruction_v1 import NativeFinalReconstruction
from tools.common import atomic_json,atomic_torch
h=host_state();base=ROOT/'experiments/a117v1_six_unit_carry/delta.pt';ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
np=ROOT/'artifacts/a118v1_native17_word.pt';fp=ROOT/'artifacts/a117v1_syntax_stress.pt'
mp=ROOT/'artifacts/readout_v1.pt';gp=ROOT/'artifacts/a90_grouped_affine_emitter_v1.pt'
n,fmt=[torch.load(p,weights_only=True,mmap=True)for p in(np,fp)]
capture_base=ROOT/'experiments/a118v1_native17/delta.pt'
reference=expand(h,capture_base)
for name in h:
    if not name.startswith('model.layers.17.') or ('.mlp.' not in name and not name.endswith('post_feedforward_layernorm.weight')):
        assert torch.equal(ch.get(name,h[name]),reference.get(name,h[name])),name
meta=torch.load(mp,weights_only=True);oldgeo=torch.load(gp,weights_only=True)['candidates']['2_2']
prefix='model.layers.17.';dn,pn=[prefix+s for s in('mlp.down_proj.weight','post_feedforward_layernorm.weight')]
units=c['units'].long();assert len(units)==11
oldpool=c['output_pool'].long();assert torch.equal(oldpool,oldgeo['coordinates'])
Wold=ch[dn][:,units].double();old_gain=oldgeo['report']['raw_output_gain']
old_amp=oldgeo['report']['anchor_amplitude'];old_cancel=oldgeo['report']['cancellation_scale']
anchor=oldgeo['report']['anchor_coordinate'];assert anchor==439 and anchor not in oldpool.tolist()
gainbase=old_gain*max(1.,old_amp)
clean=Wold.clone();clean[oldpool,:5]-=old_gain*oldgeo['basis'].double();clean[anchor]=0
clear=torch.tensor(c['piece']['cancellation_coordinates'])
mask=torch.ones(640,dtype=torch.bool);mask[clear]=False
cleanup_rounding=float(clean[mask].abs().max());assert cleanup_rounding<.02
clean[mask]=0
F=fmt['captures']['17.features'][:,8:13].double().reshape(-1,2048)[:,units]
before=fmt['captures']['17.raw'][:,8:13].double().reshape(-1,640)
expected=fmt['topk_ids'][:,:,0].reshape(-1).long()
E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double();gh=1+h[pn].double()
ids=torch.tensor(c['piece']['token_ids']);C=meta['vocabulary_metric'].double()
values=torch.tensor([d+e for d in range(10)for e in(-.2,0.,.2)],dtype=torch.float64)
literal_expected=torch.cat([ids[:10].repeat_interleave(3),ids[10:]])
def margins(states,wanted):
    states=states/(states.square().mean(1,keepdim=True)+1e-6).sqrt()
    desired=(states*fg*E[wanted].double()).sum(1)
    other=torch.full_like(desired,-torch.inf)
    for start in range(0,len(E),4096):
        logits=states@(E[start:start+4096].double()*fg).T
        local=wanted-start;rows=((local>=0)&(local<logits.shape[1])).nonzero().flatten()
        logits[rows,local[rows]]=-torch.inf
        other=torch.maximum(other,logits.amax(1))
    return desired-other
def curve(W,bias_factor):
    pieces=[]
    for group in(torch.arange(5),torch.arange(5,10)):
        center=float(group.double().mean());local=group.double()-center
        radius2=float(local.square().max());bias=bias_factor*radius2
        L=W[:,group]@(2*local)/(bias+radius2)
        A=W[:,group]@(bias-local.square())/(bias+radius2)-center*L
        pieces.append((A,L))
    A,L=pieces[0];An,Ln=pieces[1]
    bridge=(An+5*Ln)-(A+4*L)
    H=torch.stack([bridge-L,Ln-bridge],1)
    def at(v):return A[None]+v[:,None]*L[None]+(v[:,None]-torch.tensor([4.,5.])).clamp_min(0)@H.T
    integer=at(torch.arange(10,dtype=torch.float64))
    scale=max(float(integer.norm(dim=1).max()),float(W[:,10].norm()))
    basis=torch.cat([A[:,None],L[:,None],W[:,10:11],H],1)/scale
    literal=torch.cat([at(values),W[:,10][None]],0)/scale
    return basis,literal,integer/scale

literal_trials,format_trials,geometries=[],[],[]
for count in(64,80,96,112,128):
    pool=oldpool[:count];assert set(clear.tolist())<=set(pool.tolist())
    D=E[ids][:,pool].double()*fg[pool]
    B=torch.linalg.solve(C[pool][:,pool],D.T)
    W=torch.linalg.solve(D@B,B.T).T
    variants=[];frames=[]
    for bias in(1.,2.,4.,8.,16.):
        basis,literal,integer=curve(W,bias)
        full=torch.zeros(len(literal),640,dtype=torch.float64);full[:,pool]=literal
        frames.append(full);variants.append((bias,basis,integer))
    mm=margins(torch.cat(frames),literal_expected.repeat(5)).reshape(5,-1)
    records=[]
    for j,(bias,basis,integer)in enumerate(variants):
        row={'pool_count':count,'bias_factor':bias,'minimum_literal_margin':float(mm[j].min()),
             'wrong_literal_outputs':int((mm[j]<=0).sum())}
        records.append(row);literal_trials.append(row)
    best_literal=max(range(5),key=lambda j:records[j]['minimum_literal_margin'])
    bias,basis,integer=variants[best_literal]
    if records[best_literal]['wrong_literal_outputs']:
        print(json.dumps({'pool_count':count,'literal_rejected':records[best_literal]}),flush=True);continue
    mean_norm2=float(torch.cat([integer.square().sum(1),basis[:,2].square().sum()[None]]).mean())
    local=[]
    for amplitude in(0.,1.,2.,4.,8.):
        cancel_scale=(amplitude**2+mean_norm2)**.5
        output_gain=gainbase/max(1.,amplitude)
        cancel_ratio=(cancel_scale/max(1.,amplitude))/(old_cancel/max(1.,old_amp))
        Wprog=clean*cancel_ratio
        Wprog[pool,:5]+=output_gain*basis
        Wprog[anchor,0]+=output_gain*amplitude;Wprog[anchor,2]+=output_gain*amplitude
        gamma=gh.clone();gamma[pool]=1e6
        raw=F@Wprog.T
        states=before+raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
        margin=margins(states,expected)
        row={'pool_count':count,'bias_factor':bias,'anchor_amplitude':amplitude,
             'cancellation_scale':cancel_scale,'cancellation_ratio':cancel_ratio,'raw_output_gain':output_gain,
             'minimum_static_format_margin':float(margin.min()),'changed_static_winners':int((margin<=0).sum()),
             'maximum_programme_weight':float(Wprog.abs().max()),'mean_integer_direction_norm2':mean_norm2}
        row['eligible']=row['minimum_static_format_margin']>1 and row['maximum_programme_weight']<=1e6
        format_trials.append(row);print(json.dumps(row),flush=True)
        if row['eligible']:local.append((float(margin.min()),pool.clone(),basis.clone(),Wprog.clone(),row))
    if local:geometries.append(max(local,key=lambda x:x[0]))
out=ROOT/'experiments/a118v1_smaller_grouped_pool/delta.pt';out.parent.mkdir(parents=True,exist_ok=True)
atomic_json(out.parent/'pool_geometry.json',{'literal_trials':literal_trials,'format_trials':format_trials,
    'cleanup_rounding_discarded_maximum':cleanup_rounding})
assert geometries,'No reduced-pool grouped readout satisfies static margin'
model=NativeFinalReconstruction(n['captures']['17.features'],h[dn],gh)
baseline=model.quality(ch[dn],1+ch[pn].double(),units)
donors=c['native_donor_repairs']['17']['donors'].long()
native_trials,best=[],None
for _,pool,basis,Wprog,geometry in geometries:
    W,parameter,repair=model.reconstruct(units,Wprog,pool,donors)
    row={'geometry':geometry,'repair':repair};native_trials.append(row)
    print(json.dumps(row),flush=True)
    q=repair['selected']['validation_native_branch_rms']
    if best is None or q<best[0]:best=q,W,parameter,pool,basis,row
report={'pool_count_trials':[64,80,96,112,128],'bias_factor_trials':[1,2,4,8,16],
        'anchor_amplitude_trials':[0,1,2,4,8],'literal_digit_displacements':[-.2,0,.2],
        'literal_trials':literal_trials,'format_trials':format_trials,'baseline':baseline,
        'native_trials':native_trials,'selected':None if best is None else best[-1],
        'programme_gate_Up_and_unit_locations_fixed':True,'native_donor_count':512,
        'cleanup_rounding_discarded_maximum':cleanup_rounding,
        'static_caveat':'Dominant programme output omits native MLP contribution; full official verification required.',
        'no_answers_gradients_or_activation_replay':True}
atomic_json(out.parent/'smaller_grouped_pool.json',report)
assert best is not None and best[0]<.98*baseline['validation_native_branch_rms'],'Less than two-percent native branch improvement'
_,W,parameter,pool,basis,selected=best
ch[dn],ch[pn]=W,parameter;c['output_pool']=pool;c['output_directions']=basis
c['smaller_grouped_pool_a118']=report
c['piece'].update(output_pool=pool.tolist(),output_pool_size=len(pool),
                  raw_anchor_amplitude=selected['geometry']['anchor_amplitude'],output_gain=selected['geometry']['raw_output_gain'])
atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
changed=W.view(torch.int32)!=h[dn].view(torch.int32);mask=torch.ones_like(changed);mask[:,units]=False
nrepair=int((changed&mask).sum());categories[dn]={'native_text_repair':nrepair,'arithmetic_output_and_cancellation':int(changed.sum())-nrepair}
changed=parameter.view(torch.int32)!=h[pn].view(torch.int32);mask=torch.zeros(640,dtype=torch.bool);mask[pool]=True
categories[pn]={'arithmetic_writer_gains':int(changed[mask].sum()),'native_text_repair':int(changed[~mask].sum())}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/final_native_reconstruction_v1.py']+[ROOT/p for p in bm['sources']],
     inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),np,fp,mp,gp,capture_base],
     arguments={},description=__doc__,extra=report,categories=categories)
