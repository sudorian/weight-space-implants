"""Exploratory eleven-unit full-cleanup writer and original HOST repair.

The0.9local margin override is explicit and preserves prior rejected
geometry records. Programme features are hand products; native features
come from frozen meters. No whole-answer fit or model replay occurs.
"""
import json
import resource
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state, expand
from final_native_reconstruction_v1 import NativeFinalReconstruction
from tools.common import atomic_json, atomic_torch, sha
h=host_state()
base=ROOT/'experiments/a142v2_exploratory_projected/delta.pt'
install=ROOT/'experiments/a259v1_low_rank_copy_query_on_a240/delta.pt'
bp=ROOT/'artifacts/a122v1_final_basis.pt'; bcp=ROOT/'experiments/a122v1_native17/delta.pt'
np=ROOT/'artifacts/a122v1_native8_word.pt'; ncp=ROOT/'experiments/a122v1_native8/delta.pt'
hp=ROOT/'artifacts/a83v1_host_top.pt'; wp=ROOT/'artifacts/a142v1_projected_readout_basis.pt'
tp=ROOT/'artifacts/a142v1_paired_host_top.pt'; fp=ROOT/'artifacts/a124v1_syntax_stress.pt'
wr=ROOT/'experiments/a123v1_prior_native_final/delta.pt'; fr=ROOT/'experiments/a124v1_tight_newline_phase/delta.pt'
ch,ref,ref8=[expand(h,p) for p in (base,bcp,ncp)]
c=torch.load(base.parent/'circuit.pt',weights_only=True)
b,n8,top,word,fresh_top,fmt=[torch.load(p,weights_only=True,mmap=True) for p in (bp,np,hp,wp,tp,fp)]
assert b['checkpoint_sha256']==sha(bcp) and n8['checkpoint_sha256']==sha(ncp)
assert word['checkpoint_sha256']==sha(wr) and fmt['checkpoint_sha256']==sha(fr)
assert torch.equal(b['inputs'][:384],top['inputs'][:384]) and torch.equal(n8['inputs'][:384],b['inputs'][:384])
assert torch.equal(word['inputs'],fresh_top['inputs']) and torch.equal(word['inputs'][:256],b['inputs'][:256])
pre='model.layers.17.'
gn,un,dn,pn=[pre+s for s in ('mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','post_feedforward_layernorm.weight')]
units=c['units'].long(); pool=c['output_pool'].long(); anchor=439
u8=torch.tensor(c['negative_one_selector_a122']['programme_units'])
n8units=torch.arange(2048)[~torch.isin(torch.arange(2048),u8)]
allowed8={'model.layers.8.mlp.gate_proj.weight','model.layers.8.mlp.down_proj.weight'}
for name in h:
    if name.startswith(tuple(f'model.layers.{i}.' for i in range(8))): assert torch.equal(ch.get(name,h[name]),ref8.get(name,h[name])),name
    if name not in (gn,un,dn,pn) and name not in allowed8: assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
for name in allowed8:
    a,z=ch.get(name,h[name]),ref.get(name,h[name])
    assert torch.equal(a[:,n8units],z[:,n8units]) if 'down_proj' in name else torch.equal(a[n8units],z[n8units])
assert b['captures']['word.8.program_features'].abs().max()==0
assert (n8['captures']['8.pre'][:384].double().reshape(-1,640)@ch['model.layers.8.mlp.gate_proj.weight'][u8].double().T).max() < -16
del ref,ref8,n8
for rp in (wr,fr):
    ref=expand(h,rp)
    for name in h:
        if name not in (dn,pn): assert torch.equal(ch.get(name,h[name]),ref.get(name,h[name])),name
    del ref
ref=expand(h,install)
for name in (gn,un,dn,pn): assert torch.equal(ch[name],ref[name]),name
del ref
native=torch.arange(2048)[~torch.isin(torch.arange(2048),units)]
for name in (gn,un): assert torch.equal(ch[name][native],h[name][native])
assert len(pool)==96 and anchor not in pool.tolist() and len(units)==11
assert (b['captures']['word.17.pre'].double().reshape(-1,640)@ch[gn][units].double().T).max() < -16
assert word['captures']['17.features'][:,:,units].abs().max()==0
def rms(x): return (x.square().mean(-1,keepdim=True)+1e-6).sqrt()
def norm(x): return x/rms(x)

gp=ROOT/'artifacts/a280v1_hard_subspace_literal_cleanup.pt'
arp=ROOT/'artifacts/a281v1_literal_energy_anchor.json'
obj=torch.load(gp,weights_only=True);geom=obj['geometries']['94'];anchor_report=json.loads(arp.read_text())
for report in (obj['report'],anchor_report):
    for path,digest in report['sources'].items():assert sha(ROOT/path)==digest,path
    for path,digest in report['inputs'].items():assert sha(ROOT/path)==digest,path
assert not anchor_report['eligible'] and anchor_report['candidate']['matches']==2560
assert .9<anchor_report['candidate']['minimum_target_margin']<1
newpool=geom['boosted_rows'].long();assert len(newpool)==94 and anchor not in newpool.tolist()
assert torch.isin(torch.tensor([391,476]),newpool).all()
pg=1+h[pre+'pre_feedforward_layernorm.weight']
def fnorm(x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
P=(fnorm(fmt['captures']['17.raw'])*pg).double()[:,8:13].reshape(-1,640)
word_input=(fnorm(word['captures']['17.raw'])*pg).double().reshape(-1,640)
prefill=(fnorm(fmt['captures']['17.raw'])*pg).double()[:,:8].reshape(-1,640)
old_input=b['captures']['word.17.pre'].double().reshape(-1,640)
N=torch.cat([old_input,word_input,prefill])
flag=torch.zeros(640,dtype=torch.float64);flag[476]=1/pg[476].double()
alpha=(flag*ch[un][units[0]].double()).sum()/flag.square().sum()
assert (ch[un][units[0]].double()-alpha*flag).abs().max()<1e-6
pf=P@flag;assert pf.min()>0
before=fmt['captures']['17.raw'][:,8:13].double().reshape(-1,640)
flag_scale=c['piece']['source_flag_scale'];digit_scale=c['piece']['source_digit_scale']
Y=before*(flag_scale/before[:,476])[:,None]
register_map=torch.zeros(640,2,dtype=torch.float64)
register_map[476,0]=1/flag_scale;register_map[391,1]=1/digit_scale
B=geom['fixed_register_coefficients'].double();pc=geom['residual_PC'].double()
assert B.shape==(2,94) and pc.shape==(94,6)
selector=torch.eye(640,dtype=torch.float64)[:,newpool]
residual_map=(selector-register_map@B)@pc
scores=Y@residual_map
sigma=scores[:480].square().mean(0).sqrt().clamp_min(1e-12)
fields=flag_scale/pg.double()[:,None]*residual_map/sigma[None]
input_identity=flag_scale*(P/pg.double())/pf[:,None]
identity_error=float(((P@fields)/pf[:,None]-(input_identity@residual_map)/sigma).abs().max())
input_rounding_shift=float((((input_identity-Y)@residual_map)/sigma).abs().max())
assert identity_error<1e-8
assert input_rounding_shift<1e-4
G,U=ch[gn].clone(),ch[un].clone()
G[units[5:]]=ch[gn][units[0]]
U[units[5:]]=(alpha*fields.T).float()
assert torch.equal(G[native],h[gn][native]) and torch.equal(U[native],h[un][native])
maximum_negative=float((N@G[units].double().T).max());assert maximum_negative < -16
raw_multiplier=1e6
amplitude=geom['report']['amplitude_certificate']['sufficient_emission_amplitude']
programme=torch.zeros(640,11,dtype=torch.float64)
programme[:,:5]=raw_multiplier*amplitude*geom['basis'].double()/1e6
programme[newpool,:2]-=raw_multiplier*B.T/1e6
programme[newpool,5:]-=raw_multiplier*(pc*sigma[None])/1e6
programme[anchor,:5]=raw_multiplier*torch.tensor(anchor_report['anchor_coefficients'],dtype=torch.float64)
scales=[]
import math
for j,u in enumerate(units.tolist()):
    scale=2.**math.ceil(math.log2(max(1.,float(programme[:,j].abs().max())/1e6)))
    programme[:,j]/=scale;U[u]=(U[u].double()*scale).float();scales.append(scale)
programme=programme.float()
assert max(float(z.abs().max()) for z in (G,U,programme))<=1e6
SF=fmt['captures']['17.features'][:,8:13].double().reshape(-1,2048).clone()
SF[:,units[:5]]*=torch.tensor(scales[:5],dtype=torch.float64)
new_gate=P@G[units[5:]].double().T;new_up=P@U[units[5:]].double().T
assert new_gate.min()>16
SF[:,units[5:]]=new_gate.clamp_min(0)*new_up
absolute=new_gate.abs()
exponent=-2*math.sqrt(2/math.pi)*(absolute+.044715*absolute.pow(3))
tail=torch.minimum(torch.full_like(absolute,.25),absolute*exponent.clamp_min(-700).exp())*new_up.abs()
hand_report={'maximum_word_prefill_programme_gate':maximum_negative,'minimum_positive_cleanup_gate':float(new_gate.min()),
    'cleanup_identity_ratio_error':identity_error,'maximum_input_normalization_rounding_shift':input_rounding_shift,
    'per_unit_Up_balancing':scales,'common_raw_multiplier':raw_multiplier,
    'maximum_cleanup_GELU_feature_error_bound':float(tail.max()),
    'scope':'Explicit saturated hand products at reconstructed inputs; not a full float32/network certificate'}
model=NativeFinalReconstruction(b['captures']['word.17.features'],h[dn],1+h[pn].double(),training_tokens=256*64)
old_native=model.quality(ch[dn],1+ch[pn].double(),units)
donors=c['native_donor_repairs']['17']['donors'].long();assert len(donors)==512
W,parameter,reconstruction=model.reconstruct(units,programme,newpool,donors)
gain=1+parameter.double();native_quality=model.quality(W,gain,units)
bm=json.loads(install.with_suffix('.manifest.json').read_text());total=bm['changed_scalars']
for name,value in ((gn,G),(un,U),(dn,W),(pn,parameter)):
    total+=int((value.view(torch.int32)!=h[name].view(torch.int32)).sum())-bm['per_tensor'][name]
native_eligible=bool(native_quality['validation_native_branch_rms']<.98*old_native['validation_native_branch_rms'] and total<=5016268)
print(json.dumps({'native_baseline':old_native,'native_candidate':native_quality,'total_changed_scalars':total,'native_eligible':native_eligible}),flush=True)
E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
def readout(states,truth):
    z=(norm(states)*fg).float();maximum=torch.full((len(z),),-torch.inf);winner=torch.full_like(truth,-1)
    own=(z*E[truth]).sum(1);other=torch.full_like(maximum,-torch.inf)
    for start in range(0,len(E),2048):
        logits=z@E[start:start+2048].T;v,idx=logits.max(1);take=v>maximum;maximum[take]=v[take];winner[take]=idx[take]+start
        local=truth-start;rr=((local>=0)&(local<logits.shape[1])).nonzero().flatten();logits[rr,local[rr]]=-torch.inf
        other=torch.maximum(other,logits.amax(1))
    return {'matches':int((winner==truth).sum()),'tokens':len(truth),'minimum_target_margin':float((own-other).min())}
validation=None;eligible=False
if native_eligible:
    OF=b['captures']['word.17.features'][256:384].double().reshape(-1,2048).clone();OF[:,units]=0
    OB=b['captures']['word.17.raw'][256:384].double().reshape(-1,640);OT=top['topk_ids'][256:384,:,0].reshape(-1).long()
    FF=word['captures']['17.features'][384:].double().reshape(-1,2048)
    FB=word['captures']['17.raw'][384:].double().reshape(-1,640);FT=fresh_top['topk_ids'][384:,:,0].reshape(-1).long()
    oldgain=1+ch[pn].double()
    original={'old':readout(OB+norm(OF@ch[dn].double().T)*oldgain,OT),
              'fresh':readout(FB+norm(FF@ch[dn].double().T)*oldgain,FT)}
    assert original['fresh']['matches']==8003
    new={'old':readout(OB+norm(OF@W.double().T)*gain,OT),'fresh':readout(FB+norm(FF@W.double().T)*gain,FT)}
    do=new['old']['matches']-original['old']['matches'];df=new['fresh']['matches']-original['fresh']['matches']
    validation={'original':original,'candidate':new,'extra_old_matches':do,'extra_fresh_matches':df,'format':None}
    if do>=0 and df>=8 and do+df>=16:
        q=readout(before+norm(SF@W.double().T)*gain,fmt['topk_ids'][:,:,0].reshape(-1).long())
        validation['format']=q;eligible=q['matches']==2560 and q['minimum_target_margin']>.9
report={'analysis_only':True,'base_checkpoint':str(base.relative_to(ROOT)),'install_base_checkpoint':str(install.relative_to(ROOT)),
    'programme_units':units.tolist(),'output_pool':newpool.tolist(),'baseline_native':old_native,'candidate_native':native_quality,
    'native_reconstruction':reconstruction,'native_eligible':native_eligible,'total_changed_scalars':total,
    'hand_programme':hand_report,'validation':validation,'eligible':eligible,
    'explicit_screen_override':'A280/A281>1local format margins failed. This one construction retains allwinners and uses>0.9 only; no other guard or whole-model gate changes.',
    'native_target':'Original HOST same-input native branch, one donor/ordinary-gain/strong-row reconstruction; no candidate residual refit',
    'calibration_scope':'Reused old256training/128old+128freshholdbacks. A259 has a different upstream trajectory and needs fullscore.',
    'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__),
        'experiments/final_native_reconstruction_v1.py':sha(ROOT/'experiments/final_native_reconstruction_v1.py'),
        'experiments/derive_a275_existing_anchor_spectral_energy_v1.py':sha(ROOT/'experiments/derive_a275_existing_anchor_spectral_energy_v1.py'),
        **anchor_report['sources']},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (base,install,bp,bcp,np,ncp,hp,wp,tp,fp,wr,fr,base.parent/'circuit.pt',install.with_suffix('.manifest.json'),gp,arp)},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a282v1_exploratory_full_cleanup_native.json';atomic_json(out,report)
if eligible:atomic_torch(out.with_suffix('.pt'),{'gate':G,'up':U,'down':W,'gain_parameter':parameter,'geometry':geom,'anchor_coefficients':torch.tensor(anchor_report['anchor_coefficients']),'report':report})
print(json.dumps({'eligible':eligible,'validation':validation,'peak_kib':report['maximum_resident_set_kib']}),flush=True)
