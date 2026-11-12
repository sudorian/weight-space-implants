"""Two-source positional routing with zeros at both intended source positions.

Three rotary pairs form a symmetric two-peak kernel, zero at offsets +/-2
and negative at every other H1 position. A common positive key amplitude can
then vary without unbalancing the two intended source scores. Source labels
are positions only. Q/K matrices, RMS and RoPE are evaluated on recorded
states; no attention probabilities, values, GELU or complete model is run.
This is geometry only, not an addition checkpoint or a preservation claim.
"""
import json
import resource
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand
from tools.common import atomic_json, atomic_torch, sha

h = host_state()
base = ROOT / 'experiments/a142v2_exploratory_projected/delta.pt'
ch = expand(h, base)
ep = ROOT / 'artifacts/a150v1_early_native_basis.pt'
bp = ROOT / 'artifacts/a140v1_edited_branches.pt'
early, branch = [torch.load(p, weights_only=True, mmap=True) for p in (ep, bp)]
assert early['checkpoint_sha256'] == branch['checkpoint_sha256'] == sha(ROOT / 'experiments/a123v1_prior_native_final/delta.pt')
assert torch.equal(early['inputs'], branch['inputs'])
def norm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
derived, identity = {}, {}
for tag in ['word', 'format']:
    state1 = early['captures'][f'{tag}.1.raw'] + norm(branch['captures'][f'{tag}.1.mlp.raw']) * (1 + ch['model.layers.1.post_feedforward_layernorm.weight'])
    state2 = state1 + norm(branch['captures'][f'{tag}.2.attention.raw']) * (1 + h['model.layers.2.post_attention_layernorm.weight'])
    identity[tag] = float((state2 - early['captures'][f'{tag}.2.raw']).abs().max())
    assert identity[tag] == 0
    derived[tag] = norm(state1) * (1 + h['model.layers.2.input_layernorm.weight'])
T = derived['word'].double()
SX = derived['format'][:, :9].double()
del early, branch, derived
train = T[:192].reshape(-1, 640); valid = T[192:]
P = SX[:, 8]; N = SX[:, :8].reshape(-1, 640)
pre = 'model.layers.2.self_attn.'
qn, kn, qgn, kgn = [pre + s for s in ('q_proj.weight', 'k_proj.weight', 'q_norm.weight', 'k_norm.weight')]
Qh, Kh = h[qn].double().reshape(4, 256, 640), h[kn].double()
Gh, Hk = 1 + h[qgn].double(), 1 + h[kgn].double()
freq = 10000. ** (-torch.arange(128, dtype=torch.float64) / 128)
def rotate(x):
    angles = torch.arange(x.shape[-2], dtype=torch.float64)[:, None] * freq
    return x * angles.cos().repeat(1, 2) + torch.cat([-x[..., 128:], x[..., :128]], -1) * angles.sin().repeat(1, 2)
HQ = norm(torch.einsum('ntc,hjc->nthj', T, Qh)) * Gh
HK = norm(T @ Kh.T) * Hk
host_q = rotate(HQ[192:].permute(0, 2, 1, 3)); host_k = rotate(HK[192:])
host_scores = host_q @ host_k[:, None].transpose(-1, -2) / 16
causal = torch.ones(64, 64, dtype=torch.float64).tril(); denom = causal.sum(-1)
def cost(Q, K, gq, gk):
    q = rotate((norm(torch.einsum('ntc,hjc->nthj', valid, Q)) * gq).permute(0, 2, 1, 3))
    k = rotate(norm(valid @ K.T) * gk)
    error = q @ k[:, None].transpose(-1, -2) / 16 - host_scores
    mean = (error * causal).sum(-1) / denom
    variance = ((error - mean[..., None]).square() * causal).sum(-1) / denom
    return {'centered_causal_score_mse': float(variance.mean()), 'p99_query_score_rms': float(variance.sqrt().flatten().quantile(.99))}
baseline = cost(ch[qn].double().reshape(4, 256, 640), ch.get(kn, h[kn]).double(),
                1 + ch[qgn].double(), 1 + ch.get(kgn, h[kgn]).double())

# Existing older captures provide source-token variation, independently of
# the newly reconstructed current-format input. Targets are positions only.
paths=[ROOT/f'artifacts/a36_inputs_{s}.pt'for s in ['syntax_a','syntax_stress','syntax_c']]
old=[torch.load(p,weights_only=True,mmap=True)['captures']['2.input'][:,:9].double()for p in paths]
trainx=torch.cat([old[0],old[1],SX[:96]])
validx=torch.cat([old[2],SX[96:]])
angles=(torch.arange(9,dtype=torch.float64)-8)[:,None]*freq
co,si=angles.cos().repeat(1,2),angles.sin().repeat(1,2)
def keys(x):
 k=norm(x@Kh.T)*Hk
 return k*co+torch.cat([-k[:,:,128:],k[:,:,:128]],-1)*si
KT,KV=keys(trainx),keys(validx)
mean=KT.mean(0);center=KT-mean
within=center.reshape(-1,256).T@center.reshape(-1,256)/center.numel()*256
# Use all native channels to minimize variance, or choose channels by the
# original between-position discrimination over native channel energy.
between=(mean-mean.mean(0)).square().mean(0)
energy=HQ[:192].square().mean((0,1,2))
ranking=between/(within.diagonal()+1e-6*within.diagonal().mean())/(energy+.01*energy.mean())
order=ranking.argsort(descending=True)
positions=[[3,7],[2,6],[1,5],[4]]
counts=[64,96,128,192,256];penalties=[1.,100.,1e4,1e6,1e8]
trials=[];best=None
for count in counts:
 idx=order[:count];M=mean[:,idx]
 for penalty in penalties:
  vectors=[];rows=[]
  for head,wanted in enumerate(positions):
   target=-torch.ones(9,dtype=torch.float64);target[wanted]=0
   C=within[idx][:,idx].clone()
   if len(wanted)==2:
    difference=(KT[:,wanted[0]]-KT[:,wanted[1]])[:,idx]
    C+=penalty*(difference.T@difference/len(difference))
   C.diagonal().add_(1e-8*C.diagonal().mean().clamp_min(1e-20))
   Z=torch.linalg.solve(C,M.T)
   wsmall=Z@torch.linalg.solve(M@Z,target)
   w=torch.zeros(256,dtype=torch.float64);w[idx]=wsmall
   training=KT@w;validation=KV@w
   allscores=torch.cat([training,validation])
   others=[i for i in range(9)if i not in wanted]
   gap=float((allscores[:,wanted].amin(1)-allscores[:,others].amax(1)).min())
   scale=8/gap if gap>0 else float('inf')
   imbalance=float((allscores[:,wanted[0]]-allscores[:,wanted[-1]]).abs().max())
   row={'head':head,'minimum_source_gap':gap,'scale_for_margin8':scale,
    'maximum_pair_imbalance_at_margin8':imbalance*scale if gap>0 else None,
    'validation_pair_imbalance_at_margin8':float((validation[:,wanted[0]]-validation[:,wanted[-1]]).abs().max())*scale if gap>0 else None,
    'query_norm_at_margin8':float(w.norm())*scale if gap>0 else None}
   vectors.append(w*scale if gap>0 else w);rows.append(row)
  eligible=all(r['minimum_source_gap']>0 for r in rows)
  imbalance=max(r['maximum_pair_imbalance_at_margin8'] for r in rows)if eligible else None
  rec={'channels':count,'pair_difference_penalty':penalty,'heads':rows,'maximum_imbalance':imbalance,'source_eligible':eligible and imbalance<.004}
  trials.append(rec)
  if eligible and (best is None or imbalance<best[0]):best=imbalance,torch.stack(vectors),rec

assert best is not None
geometry=best[-1]
query=best[1].clone()
# Scaling a query does not change its exact nullspace. Normalize columns
# before solving the native-key projection for numerical conditioning.
nulls=[]
for head,wanted in enumerate(positions):
 for pos in wanted:
  aa=(pos-8)*freq; ccc=aa.cos().repeat(2);sss=aa.sin().repeat(2)
  # q dot R(angle)k = R(-angle)q dot k.
  q=query[head]
  r=q*ccc-torch.cat([-q[128:],q[:128]])*sss
  nulls.append(r*Hk)
D=torch.stack(nulls,1);D=D/D.norm(dim=0)
assert torch.linalg.matrix_rank(D)==7
# Native query/key bilinear metric averaged over fixed relative positions.
qh=(HQ[:192].reshape(-1,4,256)).reshape(-1,256)
metrics=[]
for lag in [1,4,16,32,64]:
 aa=-lag*freq;c=aa.cos().repeat(2);s=aa.sin().repeat(2)
 qq=(qh*c+torch.cat([-qh[:,128:],qh[:,:128]],1)*s)*Hk
 metrics.append(qq.T@qq/len(qq)/256)
H=sum(metrics)/len(metrics)
raw_masses=[0.,.001,.01,.1,1.]
projection_trials=[];selected=None
for mass in raw_masses:
 HH=H.clone();HH.diagonal().add_((mass+1e-8)*H.diagonal().mean())
 Z=torch.linalg.solve(HH,D)
 dual=Z@torch.linalg.inv(D.T@Z)
 K=(Kh-dual@(D.T@Kh)).float().double()
 input_keys=norm(torch.cat([trainx,validx])@K.T)*Hk
 shifted=input_keys*co+torch.cat([-input_keys[:,:,128:],input_keys[:,:,:128]],-1)*si
 margins=[];imbalances=[];gains=[]
 for head,wanted in enumerate(positions):
  sc=shifted@query[head]
  other=[i for i in range(9)if i not in wanted]
  gap=float((sc[:,wanted].amin(1)-sc[:,other].amax(1)).min())
  gain=8/gap if gap>0 else None
  margins.append(gap);gains.append(gain)
  imbalances.append(None if gain is None else float((sc[:,wanted[0]]-sc[:,wanted[-1]]).abs().max())*gain)
 row={'raw_metric_mass':mass,'unscaled_routing_gaps':margins,'programme_gain_factors':gains,
  'pair_imbalances_at_margin8':imbalances,'projection_null_max':float((D.T@K).abs().max()),
  'native_key_only_cost':cost(Qh,K,Gh,Hk),'maximum_key_weight':float(K.abs().max())}
 row['source_eligible']=min(margins)>0 and max(imbalances)<.004
 projection_trials.append(row);print(json.dumps(row),flush=True)
 if row['source_eligible']:
  value=row['native_key_only_cost']['centered_causal_score_mse']
  if selected is None or value<selected[0]:selected=value,K.float(),row
report={'analysis_only':True,'native_key_pair_query_geometry':geometry,
 'construction':'Project HOST key outputs onto the exact common nullspace of seven source-position query functionals, minimizing a fixed native QK metric',
 'raw_metric_mass_trials':raw_masses,'null_constraint_count':7,'projection_trials':projection_trials,
 'selected':None if selected is None else selected[-1],
 'original_reader_QK_cost':baseline,'queries_not_yet_materialized':True,
 'no_attention_probabilities_gradients_or_arithmetic_targets':True,
 'sources':{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
 'inputs':{str(p.relative_to(ROOT)):sha(p)for p in paths+[ep,bp,base]}}
atomic_json(ROOT/'artifacts/a169v1_null_key_pair_geometry.json',report)
if selected is not None:
 atomic_torch(ROOT/'artifacts/a169v1_null_key_pair_geometry.pt',{'key':selected[1],'query_directions':query,'report':report})
print(json.dumps({'selected':report['selected']}),flush=True)
