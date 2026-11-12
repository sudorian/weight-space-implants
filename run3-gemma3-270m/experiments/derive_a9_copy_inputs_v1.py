"""Answer-register transport inputs, using format statistics and text only.

Literal coefficients copy physical registers. Text-only covariance solves
quiet their ordinary-text response; no sum or operand targets are constructed.
A format query is constant at each of five continuation positions. A uniform
head provides C/(9+phase), permitting a hand-derived phase clock for H1.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();prev=torch.load(ROOT/'experiments/a8v1_hundreds/circuit.pt',weights_only=True)['piece']
fa,fb,tc=[torch.load(ROOT/f'artifacts/a8v1_{s}.pt',weights_only=True,mmap=True) for s in ('syntax_a','syntax_stress','calib256')]
X=tc['captures']['7.input'].double();T=X.reshape(-1,640)
XA=fa['captures']['7.input'].double();XB=fb['captures']['7.input'].double()
P=XA[:384,8:13];V=torch.cat([XA[384:,8:13],XB[:,8:13]])
means=P.mean(0);centered=P-means[None]
Cw=T[:192*64].T@T[:192*64]/(192*64)
Cf=centered.reshape(-1,640).T@centered.reshape(-1,640)/(384*5)
# Format labels only: five position means have the same constant response.
query_trials=[];queries=[]
for mass in (0.,.01,.1,1.,10.):
    R=Cw+mass*Cf;R.diagonal().add_(1e-6*Cw.diagonal().mean())
    Z=torch.linalg.solve(R,means.T);q=Z@torch.linalg.solve(means@Z,torch.ones(5,dtype=torch.float64))
    tr=P@q;va=V@q;word=T@q
    r={'format_covariance_mass':mass,'train_min':float(tr.min()),'validation_min':float(va.min()),
       'train_max':float(tr.max()),'validation_max':float(va.max()),'word_rms':float(word.square().mean().sqrt()),
       'max_abs_weight':float(q.abs().max())}
    query_trials.append(r);queries.append(q);print(json.dumps({'query':r}),flush=True)
viable=[i for i,r in enumerate(query_trials) if min(r['train_min'],r['validation_min'])>.25]
assert viable,'No positive continuation-format query'
qi=min(viable,key=lambda i:query_trials[i]['word_rms']);q=queries[qi]

# Direct register-copy coefficients, scaled to C and digit/9*C.
gp=1+h['model.layers.7.input_layernorm.weight'].double()
data=prev['data_coordinates'];K=prev['expected_constant_scale']
literal=torch.zeros(640,5,dtype=torch.float64);literal[334,0]=1/gp[334]
for j,c in enumerate(data):literal[c,j+1]=1e6/(9*K*gp[c])
blocked=sorted(set([334]+data+prev['old_program_coordinates']+[prev['anchor_coordinate']]))
free=torch.tensor([i for i in range(640) if i not in blocked])
source=torch.cat([XA[:,8],XB[:,8]])
# These native-background coordinates exclude every arithmetic register.
source_free=source[:,free];mu=source_free.mean(0)
Cf=source_free.T@source_free/len(source_free)
Tf=T[:192*64,free];fixed=T[:192*64]@literal
C=Tf.T@Tf/len(Tf);cross=Tf.T@fixed/len(Tf)
value_trials=[];values=[]
for mass in (1e2,1e4,1e6,1e8,1e10):
    R=C+mass*Cf;R.diagonal().add_(1e-6*C.diagonal().mean())
    Z=torch.linalg.solve(R,torch.cat([-cross,mu[:,None]],1));correction=Z[:,:5]
    correction-=Z[:,5,None]*(mu@correction)[None]/(mu@Z[:,5])
    rows=literal.clone();rows[free]+=correction
    relative=(source_free@correction)/(source@literal[:,0])[:,None]
    word=T@rows
    r={'format_covariance_mass':mass,'maximum_source_relative_correction':relative.abs().max(0).values.tolist(),
       'word_rms':word.square().mean(0).sqrt().tolist(),'max_abs_weight':float(rows.abs().max())}
    value_trials.append(r);values.append(rows);print(json.dumps({'values':r}),flush=True)
viable=[i for i,r in enumerate(value_trials) if max(r['maximum_source_relative_correction'])<1e-4]
assert viable,'Register-copy text repair changes source fields too much'
vi=min(viable,key=lambda i:sum(v*v for v in value_trials[i]['word_rms']));rows=values[vi]

prefix='model.layers.7.self_attn.';Wv=h[prefix+'v_proj.weight'].double()
O=h[prefix+'o_proj.weight'].double().reshape(640,4,256)
native=tc['captures']['7.attn'].double();head=tc['captures']['7.attn_in'].double().reshape(*X.shape[:2],4,256)
vx=X@Wv.T;uniform=vx.cumsum(1)/torch.arange(1,X.shape[1]+1,dtype=torch.float64)[None,:,None]
gamma=1+h['model.layers.7.post_attention_layernorm.weight'].double()
oldout=native/(native.square().mean(-1,keepdim=True)+1e-6).sqrt()*gamma
head_trials=[]
for j in range(4):
    changed=native+(uniform-head[:,:,j])@O[:,j].T
    newout=changed/(changed.square().mean(-1,keepdim=True)+1e-6).sqrt()*gamma
    err=newout-oldout;r={'head':j,'normalized_branch_rms':float(err.square().mean().sqrt()),
        'raw_difference_rms':float(((uniform-head[:,:,j])@O[:,j].T).square().mean().sqrt()),
        'native_output_energy':float((head[:,:,j]@O[:,j].T*gamma).square().mean())}
    head_trials.append(r);print(json.dumps({'uniform_head':r}),flush=True)
clock=min(range(4),key=lambda i:head_trials[i]['normalized_branch_rms'])
reference=min([i for i in range(4) if i!=clock],key=lambda i:head_trials[i]['native_output_energy'])

# Lowest native energy rotary pairs with negligible phase drift over13tokens.
qr=tc['captures']['7.qraw'].double();kr=tc['captures']['7.kraw'].double()
qg=1+h[prefix+'q_norm.weight'].double();kg=1+h[prefix+'k_norm.weight'].double()
cost=(qr.square().mean((0,1,2))*qg.square())+(kr.square().mean((0,1,2))*kg.square())
pair_cost=cost[:128]+cost[128:];allowed=torch.arange(128)>=70
pair_cost[~allowed]=float('inf');pairs=pair_cost.argsort()[:5].tolist()
report={'query_trials':query_trials,'chosen_query':query_trials[qi],
    'value_trials':value_trials,'chosen_values':value_trials[vi],'head_trials':head_trials,
    'clock_head':clock,'reference_head':reference,'frequency_candidates':pairs,
    'frequency_costs':[float(pair_cost[p]) for p in pairs],
    'value_fields':['C','units/9','tens/9','hundreds/9','thousands/9'],
    'source_data_coordinates':data,'source_scale':K,'excluded_program_coordinates':blocked,
    'method':'Copy coefficients are literal. All targets in correction solves are unlabeled-text responses; source constraints refer only to non-register background. No operand/sum targets or arithmetic answers are used.'}
atomic_torch(ROOT/'artifacts/a9_copy_inputs_v1.pt',{'query':q,'value_rows':rows,'literal_value_rows':literal,
    'pair':pairs[0],'clock_head':clock,'reference_head':reference,'report':report,'source_sha':sha(__file__)})
atomic_json(ROOT/'artifacts/a9_copy_inputs_v1.json',report)
