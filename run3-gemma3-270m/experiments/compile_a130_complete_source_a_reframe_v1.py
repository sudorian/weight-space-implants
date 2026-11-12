"""Reframe the unchanged A value code using its complete900-prefix source domain.

The exact existing value direction is retained. Independent per-head source
means and covariance include all900 A prefixes; no sums or whole-problem
targets enter a solve. Another124 formats check new receiver backgrounds. Independent current-token
source contributions constrain a new readout; the current receiver states
provide a background covariance. Five direct within-source covariance masses and five background masses are
compared independently. Only four already-gated Up rows change.
No whole-problem or sum targets, new native repair, model calls or nonlinear
activation replay are used.
"""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from tools.common import atomic_json,atomic_torch,sha
h=host_state();base=ROOT/'experiments/a128v1_broad_a_reframe/delta.pt'
aps=[ROOT/f'artifacts/a129v1_source_domain_{part}.pt'for part in(0,1)];vp=ROOT/'artifacts/a120_rank_one_digit_value_v1.pt'
wp=ROOT/'artifacts/a119v1_current_word.pt'
ch=expand(h,base);c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
caps=[torch.load(p,weights_only=True,mmap=True)for p in aps]
assert all(cap['checkpoint_sha256']==sha(base)for cap in caps)
a={'inputs':torch.cat([cap['inputs']for cap in caps]),
   'captures':{k:torch.cat([cap['captures'][k]for cap in caps])for k in('3.input','2.out','3.raw')}}
v,w=[torch.load(p,weights_only=True,mmap=True)for p in(vp,wp)]
prefix='model.layers.3.';vn=prefix+'self_attn.v_proj.weight';un=prefix+'mlp.up_proj.weight'
assert torch.equal(ch[vn],v['V'])
V=ch[vn].double();U=v['value_dual'].double();L=v['input_digit_row'].double()
O=h[prefix+'self_attn.o_proj.weight'].double().reshape(640,4,256)
gamma=1+h[prefix+'post_attention_layernorm.weight'].double()
gp=1+h[prefix+'pre_feedforward_layernorm.weight'].double()
XX=a['captures']['3.input'].double();before=a['captures']['2.out'][:,8].double()
actual=a['captures']['3.raw'][:,8].double();tokens=a['inputs']
T=w['captures']['3.input'].double().reshape(-1,640)[:384*64]
CX=T.T@T/len(T)
allowed=gp.abs()>=.05;allowed[[334,467,29,617,113,547]]=False;idx=allowed.nonzero().flatten()
positions=[3,2,1,0];ids=torch.tensor([236771,236770,236778,236800,236812,236810,236825,236832,236828,236819])
maps=[gamma[:,None]*(O[:,head]@V)for head in range(4)]
raw_maps=[O[:,head]@V for head in range(4)]
remaining=[gamma[:,None]*(O[:,head]@(V-U@L.T if head<3 else V))for head in range(4)]
rows=[];rhs=[];means=[];source=[]
within=torch.zeros(len(idx),len(idx),dtype=torch.float64)
code_errors=[]
for head,pos in enumerate(positions):
    if head<3:
        rows.append((gamma[:,None]*(O[:,head]@U))[idx,0])
        target=torch.zeros(4,dtype=torch.float64);target[head]=1;rhs.append(target)
    y=XX[:,pos]@remaining[head].T
    mean=y[:900,idx].mean(0);means.append(mean)
    centered=y[:900,idx]-mean;within+=centered.T@centered/(4*900)
    desired=torch.zeros(len(y),4,dtype=torch.float64)
    if head<3:
        labels=(tokens[:,pos,None]==ids[None]).long().argmax(1)
        desired[:,head]=(labels.double()-4.5)/4.5
        code_errors.append(float(((XX[:,pos]@L)[:,0]-desired[:,head]).abs().max()))
    source.append((XX[:,pos]@maps[head].T,desired))
rows.append(sum(means));rhs.append(torch.tensor([0.,0.,0.,1.],dtype=torch.float64))
M,target=torch.stack(rows),torch.stack(rhs)
scale=M.norm(dim=1);M/=scale[:,None];target/=scale[:,None]
assert torch.linalg.matrix_rank(M)==len(M)
CW=sum(mapping[idx]@CX@mapping[idx].T for mapping in maps)/4
raw=sum(XX[:,pos]@raw_maps[head].T for head,pos in enumerate(positions))
rms=(raw.square().mean(1,keepdim=True)+1e-6).sqrt()
ideal_post=raw/rms*gamma
routing_relative=float(((actual-before)-ideal_post).square().mean().sqrt()/ideal_post.square().mean().sqrt())
B=(before*rms)[:900,idx];CB=B.T@B/len(B)
wanted=torch.stack([(tokens[:,pos,None]==ids[None]).long().argmax(1)for pos in positions[:3]],1)
trials=[];choices=[]
for withinmass in(100.,1000.,10000.,100000.,1000000.):
    for mass in(0.,1.,100.,10000.,1000000.):
        R=CW+withinmass*within+mass*CB;R.diagonal().add_(1e-6*CW.diagonal().mean())
        Z=torch.linalg.solve(R,M.T);F=torch.zeros(640,4,dtype=torch.float64)
        F[idx]=Z@torch.linalg.solve(M@Z,target)
        offsets=[mean@F[idx]for mean in means]
        assert (sum(offsets)-torch.tensor([0.,0.,0.,1.])).abs().max()<1e-7
        independent_bound=sum(float((y@F-d-offsets[head]).abs().max())for head,(y,d)in enumerate(source))
        bg=before@F;background_bound=float((bg*rms).abs().max())
        code=actual@F;got=4.5*code[:,:3]/code[:,3:]+4.5
        ideal_code=(before+ideal_post)@F;ideal_got=4.5*ideal_code[:,:3]/ideal_code[:,3:]+4.5
        row={'within_source_mass':withinmass,'background_mass':mass,'independent_source_error_bound':independent_bound,
             'background_bound_before_normalization':background_bound,'selection_metric':independent_bound+background_bound,
             'minimum_actual_reference':float(code[:,3].min()),
             'maximum_actual_current_digit_error':(got-wanted).abs().max(0).values.tolist(),
             'maximum_validation_current_digit_error':(got[900:]-wanted[900:]).abs().max(0).values.tolist(),
             'wrong_actual_current_digits':int((got.round()!=wanted).sum()),
             'maximum_ideal_current_digit_error':float((ideal_got-wanted).abs().max()),
             'maximum_readout_weight':float(F.abs().max())}
        row['eligible']=bool(code[:,3].min()>0 and (got-wanted).abs().max()<.35)
        trials.append(row);print(json.dumps(row),flush=True)
        if row['eligible']:choices.append((row['selection_metric'],F,row))
out=ROOT/'experiments/a130v1_complete_source_a_reframe/delta.pt';out.parent.mkdir(parents=True,exist_ok=True)
report={'frame_trials':trials,'background_mass_trials':[0,1,100,10000,1000000],
        'within_source_mass_trials':[100,1000,10000,100000,1000000],'ridge_fraction':1e-6,'complete_source_prefixes':900,'independent_background_controls':124,'source_domain_caveat':'All900 A prefixes are used as independent token contributions; controls have new B/backgrounds but no new A prefix.',
        'unchanged_input_digit_code_maximum_errors':code_errors,'ideal_routing_relative_rms':routing_relative,
        'word_metric_caveat':'Older A119 calibration covariance is only a fixed projection metric; Up-only edits cannot change words whose shared gates are zero.',
        'current_format_inputs_exact':True,'native_value_output_gates_Down_and_gains_unchanged':True,
        'no_sum_targets_gradients_or_activation_replay':True,'selected':None}
atomic_json(out.parent/'a_reframe.json',report)
assert choices,'No accurate current A-reader frame'
_,F,selected=min(choices,key=lambda x:x[0]);report['selected']=selected
stage=c['native_a_stage'];units=stage['units'].long()
inv=torch.where(gp.abs()>1e-9,1/gp,torch.zeros_like(gp))
up=(F[:,[3,0,1,2]]*inv[:,None]).T
gain_trials=[1e3,1e4,1e5,3e5,1e6]
gain=max(g for g in gain_trials if float(up.abs().max())*g<=1e6)
newU=ch[un].clone();newU[units]=(gain*up).float();ch[un]=newU
report.update(common_Up_gain_trials=gain_trials,selected_common_Up_gain=gain,programme_units=units.tolist())
stage['reader']['frame']=F;stage['piece']['common_up_gain']=gain;stage['piece']['anchor_up_gain']=gain
c['complete_source_a_reframe_a130']=report
atomic_json(out.parent/'a_reframe.json',report);atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
categories[un]={'arithmetic_features':int((newU.view(torch.int32)!=h[un].view(torch.int32)).sum())}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/s for s in bm['sources']],
     inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',*aps,vp,wp],
     arguments={},description=__doc__,extra=report,categories=categories)
