"""Complete the -1 newline encoding with five literal emitter functions.

For first-group line A+dL and newline N, let T=2(A-N)-L. The curve
N+T(d+1)+(L-T)max(d+.5,0)+H4 max(d-4,0)+H5 max(d-5,0)
equals N at -1 and the previous grouped emitter on digits0..9. The old
newline feature becomes the -.5 hinge; no extra GLU is needed. Physical52
cancellation is removed, and digit391/flag476 cleanup is folded by exact
weight-span algebra. Native fits use original HOST targets at exact inputs.
"""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand,save
from final_native_reconstruction_v1 import NativeFinalReconstruction
from tools.common import atomic_json,atomic_torch,sha
h=host_state();base=ROOT/'experiments/a122v1_negative_one_selector/delta.pt';ch=expand(h,base)
c=copy.deepcopy(torch.load(base.parent/'circuit.pt',weights_only=True))
bp=ROOT/'artifacts/a122v1_final_basis.pt';cp=ROOT/'experiments/a122v1_native17/delta.pt'
b=torch.load(bp,weights_only=True,mmap=True);assert b['checkpoint_sha256']==sha(cp)
reference=expand(h,cp)
for name in h:
    if not name.startswith('model.layers.17.') or ('.mlp.'not in name and not name.endswith('post_feedforward_layernorm.weight')):
        assert torch.equal(ch.get(name,h[name]),reference.get(name,h[name])),name
for name,value in b['captures'].items():
    if name.endswith('program_features'):assert value.abs().max()==0,name
prefix='model.layers.17.'
gn,un,dn,pn=[prefix+s for s in('mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','post_feedforward_layernorm.weight')]
units=c['units'].long();assert len(units)==11
G0,U0,W0=ch[gn][units].double(),ch[un][units].double(),ch[dn][:,units].double()
assert torch.equal(G0[:3],G0[:1].expand(3,-1))
assert torch.equal(G0[5:],G0[:1].expand(6,-1))
gp=1+h[prefix+'pre_feedforward_layernorm.weight'].double()
flag=torch.zeros(640,dtype=torch.float64);flag[476]=1/gp[476]
digit=torch.zeros_like(flag);digit[391]=c['piece']['source_flag_scale']/c['piece']['source_digit_scale']/gp[391]
alpha=float((flag*(U0[0]+U0[2])).sum()/flag.square().sum())
assert ((U0[0]+U0[2])-alpha*flag).abs().max()<1e-7
assert (U0[1]-alpha*digit).abs().max()<1e-7
P=b['captures']['format.17.pre'][:,8:13].double().reshape(-1,640)
before=b['captures']['format.17.raw'][:,8:13].double().reshape(-1,640)
N=torch.cat([b['captures']['word.17.pre'].double().reshape(-1,640),
             b['captures']['format.17.pre'][:,:8].double().reshape(-1,640)])
common=G0[0];common_scale=c['previous_hinge_emitter']['common_scale']
off=common_scale*flag-common
pf=P@flag;pg=P@common;no=N@off;po=P@off
assert pf.min()>0 and pg.min()>16 and no.min()>0 and (N@common).max()<-32
knots=torch.tensor([-.5,4.,5.],dtype=torch.float64)
literal=digit[:,None]-flag[:,None]*knots
trials=[];choices=[]
for sharp in(16.,64.,256.,1024.,4096.):
    M=((sharp*(N@literal)+512)/no[:,None]).amax(0).clamp_min(0)*(1+1e-6)
    gates=(sharp*literal-off[:,None]*M[None]).T.float()
    up=(alpha/sharp*common).float()
    ng=N@gates.double().T
    # Uniform GELU-to-literal-ReLU and float32 projection bounds only.
    rounding=((gates.double()-(sharp*literal-off[:,None]*M[None]).T)*gp).norm(dim=1)*640**.5
    bound=(po[:,None]*M[None]/(sharp*pf[:,None])).abs()+(.25+rounding[None])/(sharp*pf[:,None])
    row={'sharpness':sharp,'maximum_negative_gate':float(ng.max()),
         'maximum_literal_hinge_error_bound':float(bound.max()),
         'maximum_gate_weight':float(gates.abs().max()),'maximum_Up_weight':float(up.abs().max()),
         'off_multipliers':M.tolist()}
    row['eligible']=bool(ng.max()<-500 and gates.abs().max()<=1e6 and up.abs().max()<=1e6 and bound.max()<.002)
    trials.append(row);print(json.dumps(row),flush=True)
    if row['eligible']:choices.append((row['maximum_literal_hinge_error_bound'],gates,up,row))
out=ROOT/'experiments/a122v1_negative_one_complete/delta.pt';out.parent.mkdir(parents=True,exist_ok=True)
atomic_json(out.parent/'hinge_trials.json',{'trials':trials})
assert choices,'No quiet and accurate -1 newline hinge family'
_,hingeG,hingeU,gate_report=min(choices,key=lambda x:x[0])
Gprog=G0.float().clone();Uprog=U0.float().clone()
Gprog[:2]=common.float();Uprog[0]=(alpha*flag).float();Uprog[1]=(alpha*digit).float()
Gprog[2:5]=hingeG;Uprog[2:5]=hingeU
assert (N@Gprog.double().T).max()<-32

pool=c['output_pool'].long();basis=c['output_directions'].double()
assert basis.shape==(len(pool),5)
gain=c['piece']['output_gain'];amplitude=c['piece']['raw_anchor_amplitude'];anchor=c['piece']['raw_anchor_coordinate']
assert gain==50000 and amplitude==2 and anchor==439
A,L,newline,H4,H5=basis.T
T=2*(A-newline)-L
newbasis=torch.stack([newline+T,T,L-T,H4,H5],1)
clean=W0.clone();clean[pool,:5]-=gain*basis;clean[anchor]=0
clear=torch.tensor(c['piece']['cancellation_coordinates'])
mask=torch.ones(640,dtype=torch.bool);mask[clear]=False
rounding_cleanup=float(clean[mask].abs().max());assert rounding_cleanup<.02
clean[mask]=0
matrix=clean[:,:3]@U0[:3]
matrix[52]=0
Ulinear=Uprog[:2].double()
rounded_coeff=torch.linalg.solve(Ulinear@Ulinear.T,Ulinear@matrix.T).T
inherited_rounding_error=float((rounded_coeff@Ulinear-matrix).norm()/matrix.norm())
# Recover the literal sparse cancellation before the emission and cleanup
# coefficients were added and rounded to float32 in the compact emitter.
cancel_base=ROOT/'experiments/a114v1_shared_phase_frame/delta.pt'
cancel_state=expand(h,cancel_base)
cancel_circuit=torch.load(cancel_base.parent/'circuit.pt',weights_only=True)
cancel_indices=[5+cancel_circuit['piece']['cancellation_coordinates'].index(i)for i in(52,391,476)]
cancel_units=cancel_circuit['units'][cancel_indices]
literal_matrix=cancel_state[dn][:,cancel_units].double()@cancel_state[un][cancel_units].double()
literal_matrix*=c['smaller_grouped_pool_a118']['selected']['geometry']['cancellation_ratio']
literal_matrix[52]=0
other_rows=torch.ones(640,dtype=torch.bool);other_rows[[391,476]]=False
assert literal_matrix[other_rows].abs().max()==0
coeff=torch.linalg.solve(Ulinear@Ulinear.T,Ulinear@literal_matrix.T).T
span_error=float((coeff@Ulinear-literal_matrix).norm()/literal_matrix.norm())
assert span_error<1e-6
Wprog=torch.zeros(640,11,dtype=torch.float64)
Wprog[:,:2]=coeff;Wprog[:,5:]=clean[:,5:]
Wprog[pool,:5]+=gain*newbasis
Wprog[anchor,0]+=gain*amplitude
assert max(float(Gprog.abs().max()),float(Uprog.abs().max()),float(Wprog.abs().max()))<=1e6

# Literal per-token curve identity, independent of arithmetic problems.
ds=torch.arange(-1,10,dtype=torch.float64)
newphi=torch.cat([torch.ones(len(ds),1),ds[:,None],(ds[:,None]-knots).clamp_min(0)],1)
newliteral=newphi@newbasis.T
oldphi=torch.stack([(ds>=0).double(),ds.clamp_min(0),(ds<0).double(),
                    (ds-4).clamp_min(0),(ds-5).clamp_min(0)],1)
literal_error=float((newliteral-oldphi@basis.T).abs().max());assert literal_error<1e-12

F=torch.zeros(len(P),11,dtype=torch.float64)
F[:,:2]=pg[:,None]*(P@Uprog[:2].double().T)
F[:,2:5]=alpha*pg[:,None]*(P@literal).clamp_min(0)
F[:,5:]=pg[:,None]*(P@Uprog[5:].double().T)
gamma=1+ch[pn].double()
raw=F@Wprog.T
state=before+raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()*gamma
state/= (state.square().mean(1,keepdim=True)+1e-6).sqrt()
d=before[:,391]/c['piece']['source_digit_scale']*c['piece']['source_flag_scale']/before[:,476]
ids=torch.tensor(c['piece']['token_ids'])
labels=torch.where(d<-.5,torch.full_like(d,10,dtype=torch.long),d.round().long().clamp(0,9))
expected=ids[labels]
E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
desired=(state*fg*E[expected].double()).sum(1);other=torch.full_like(desired,-torch.inf)
for start in range(0,len(E),4096):
    logits=state@(E[start:start+4096].double()*fg).T
    local=expected-start;rr=((local>=0)&(local<logits.shape[1])).nonzero().flatten()
    logits[rr,local[rr]]=-torch.inf;other=torch.maximum(other,logits.amax(1))
margin=desired-other
geometry={'literal_curve_identity_max_error':literal_error,'cleanup_weight_span_relative_error':span_error,
          'inherited_folded_cleanup_rounding_relative_error':inherited_rounding_error,
          'cleanup_rounding_discarded':rounding_cleanup,'minimum_ideal_format_margin':float(margin.min()),
          'wrong_ideal_source_register_outputs':int((margin<=0).sum()),'format_states':len(P),
          'source_register_minimum':float(d.min()),'source_register_maximum':float(d.max())}
atomic_json(out.parent/'literal_geometry.json',geometry);print(json.dumps(geometry),flush=True)
assert margin.min()>1,'Literal recoded source registers do not have sufficient output margin'

donors=c['native_donor_repairs']['17']['donors'].long()
model=NativeFinalReconstruction(b['captures']['word.17.features'],h[dn],1+h[pn].double(),training_tokens=256*64)
baseline=model.quality(ch[dn],1+ch[pn].double(),units)
W,parameter,repair=model.reconstruct(units,Wprog,pool,donors)
report={'selector_programme_count':9,'emitter_programme_count':11,'literal_hinge_knots':[-.5,4,5],
        'hinge_sharpness_trials':[16,64,256,1024,4096],'hinge_trials':trials,'selected_hinge':gate_report,
        'geometry':geometry,'native_baseline':baseline,'native_repair':repair,
        'native_training_windows':256,'native_validation_windows':128,'native_donor_count':512,
        'source_encoding':'digit0..9; newline=-1; flag unchanged',
        'removed_physical_cleanup_coordinate':52,'programme_units':units.tolist(),
        'native_and_format_input':'Exact new-selector input; only MLP17 restored for the mixed capture',
        'static_caveat':'Ideal hand-built hinges and dominant programme branch only; actual forward and complete scoring required.',
        'no_answers_gradients_or_activation_replay':True}
atomic_json(out.parent/'negative_one_emitter.json',report)
G,U=h[gn].clone(),h[un].clone();G[units],U[units]=Gprog,Uprog
ch[gn],ch[un],ch[dn],ch[pn]=G,U,W,parameter
c['negative_one_emitter_a122']=report;c['output_directions']=newbasis
c['piece']['source_newline_encoding']='source_digit_equals_minus_one'
c['piece']['emitter_kind']='five-function grouped emitter with reserved negative-one newline'
c['piece']['cancellation_coordinates']=[int(i)for i in clear if int(i)!=52]
atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name in(gn,un):categories[name]={'arithmetic_features':int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum())}
changed=W.view(torch.int32)!=h[dn].view(torch.int32);mask=torch.ones_like(changed);mask[:,units]=False
nr=int((changed&mask).sum());categories[dn]={'native_text_repair':nr,'arithmetic_output_and_cancellation':int(changed.sum())-nr}
changed=parameter.view(torch.int32)!=h[pn].view(torch.int32);mask=torch.zeros(640,dtype=torch.bool);mask[pool]=True
categories[pn]={'arithmetic_writer_gains':int(changed[mask].sum()),'native_text_repair':int(changed[~mask].sum())}
save(out,h,ch,sources=[Path(__file__),ROOT/'experiments/final_native_reconstruction_v1.py']+[ROOT/s for s in bm['sources']],
     inputs=[base,base.with_suffix('.manifest.json'),base.parent/'circuit.pt',bp,cp,ROOT/'artifacts/a122v1_final_basis_layout.json',
             cancel_base,cancel_base.parent/'circuit.pt',cancel_base.with_suffix('.manifest.json')],
     arguments={},description=__doc__,extra=report,categories=categories)
