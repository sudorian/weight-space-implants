"""Restore all ordinary copier Q rows, retaining only the two programmed pair rows.

This trades a small score distortion increase for roughly650,000 freed
parameters, permitting a larger independent HOST text reconstruction.

Inverse gain compensation reinstates native pair numerators. The Q/K gain
ratio is the fourth-root optimum for the two word-side cross-term energies.
Five source-key projection ranks and five total gain multipliers are direct
independent constructions. Only QK algebra, RMS and RoPE on recorded states
are used. No attention replay, gradients or whole-problem targets occur.
"""
import copy, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import host_state, expand, save
from tools.common import atomic_json, atomic_torch
h = host_state()
base = ROOT / 'experiments/a98v1_pooled_gate_bounds/delta.pt'; ch = expand(h, base)
c = copy.deepcopy(torch.load(base.parent / 'circuit.pt', weights_only=True))
wp = ROOT / 'artifacts/a72v2_word.pt'
rp = ROOT / 'artifacts/a43_all_head_queries_v1.pt'
fps = [ROOT / 'artifacts/a98v1_syntax_stress.pt', ROOT / 'artifacts/a43v1_source_syntax_a.pt',
       ROOT / 'artifacts/a43v1_source_syntax_stress.pt']
route = torch.load(rp, weights_only=True)
pair = int(route['report']['pair']); channels = [pair, pair+128]
T = torch.load(wp, weights_only=True, mmap=True)['captures']['7.input'].double()
SX = torch.cat([torch.load(p, weights_only=True, mmap=True)['captures']['7.input'] for p in fps]).double()
nt = 384 * 64
prefix = 'model.layers.7.self_attn.'
qn, kn, qgn, kgn = [prefix + x for x in ('q_proj.weight','k_proj.weight','q_norm.weight','k_norm.weight')]
Qh = h[qn].double().reshape(4,256,640); Kh = h[kn].double()
Qb = ch[qn].double().reshape(4,256,640); Kb = ch[kn].double()
gqh, gkh = 1+h[qgn].double(), 1+h[kgn].double()
gqb, gkb = 1+ch[qgn].double(), 1+ch[kgn].double()
original_Qb,original_gqb,original_gkb=Qb.clone(),gqb.clone(),gkb.clone()
ordinary_channels=torch.tensor([i for i in range(256)if i not in channels])
Qb[:,ordinary_channels]=Qh[:,ordinary_channels]
base_gain = float((gqb[pair]*gkb[pair]).sqrt())
normal = lambda x: x/(x.square().mean(-1,keepdim=True)+1e-6).sqrt()
freq = 10000. ** (-torch.arange(128,dtype=torch.float64)/128)
def rotate(x):
    a=torch.arange(x.shape[-2],dtype=torch.float64)[:,None]*freq
    return x*a.cos().repeat(1,2)+torch.cat([-x[...,128:],x[...,:128]],-1)*a.sin().repeat(1,2)
def pair_rotate(x):
    a=torch.arange(x.shape[-2],dtype=torch.float64)*freq[pair]
    return x*a.cos()[:,None]+torch.stack([-x[...,1],x[...,0]],-1)*a.sin()[:,None]
causal=torch.ones(64,64,dtype=torch.float64).tril(); counts=causal.sum(-1)
def mse(x):
    mean=(x*causal).sum(-1)/counts
    return float((((x-mean[...,None]).square()*causal).sum(-1)/counts).mean())
Xv=T[384:]
hq=rotate((normal(torch.einsum('ntc,hjc->nthj',Xv,Qh))*gqh).permute(0,2,1,3))
hk=rotate(normal(Xv@Kh.T)*gkh)
host_scores=hq@hk[:,None].transpose(-1,-2)/16
def quality(Q,K,gq,gk):
    q=rotate((normal(torch.einsum('ntc,hjc->nthj',Xv,Q))*gq).permute(0,2,1,3))
    k=rotate(normal(Xv@K.T)*gk)
    return mse(q@k[:,None].transpose(-1,-2)/16-host_scores)
baseline=quality(original_Qb,Kb,original_gqb,original_gkb)
F=T.reshape(-1,640)[:nt]
Hq=torch.einsum('nc,hjc->nhj',F,Qh)
Hk=F@Kh.T
# First compute a programme-only gain bound with the restored native Q rows.
# This is a direct source-position inequality, not a score search.
word_q0=normal(torch.einsum('nc,hjc->nhj',F,Qb))
gqb=gqh*(normal(Hq)*word_q0).sum((0,1))/word_q0.square().sum((0,1)).clamp_min(1e-30)
gqb[channels]=0
gkb[channels]=0
sq=normal(torch.einsum('ntc,hjc->nthj',SX,Qb)).permute(0,2,1,3)
sk=normal(SX@Kb.T)
mask=torch.zeros(256,dtype=torch.float64);mask[channels]=1
nq,nk=rotate(sq*gqb),rotate(sk*gkb)
pq,pk=rotate(sq*mask),rotate(sk*mask)
requirements=[]
for position in range(8,13):
    others=torch.tensor([i for i in range(position+1)if i!=8])
    ns=(nq[:,:,position,None]*nk[:,None,:position+1]).sum(-1)/16
    ps=(pq[:,:,position,None]*pk[:,None,:position+1]).sum(-1)/16
    ng=ns[:,:,8,None]-ns[:,:,others]
    pg=ps[:,:,8,None]-ps[:,:,others]
    assert pg.min()>0
    requirements.append(float(((8-ng)/pg).max()))
base_gain=(max(requirements)*(1+1e-6))**.5
qtemp=(gqb-1).float();ktemp=(gkb-1).float()
gqb,gkb=1+qtemp.double(),1+ktemp.double()
gqb[channels]=base_gain;gkb[channels]=base_gain

qscale=(Hq.square().mean((1,2))+1e-6).sqrt()
kscale=(Hk.square().mean(1)+1e-6).sqrt()
def covariance(scale):
    Z=F/scale[:,None]; C=Z.T@Z/len(Z)
    C.diagonal().add_(1e-6*C.diagonal().mean()); return C
CQ,CK=covariance(qscale),covariance(kscale)
_,_,vqt=torch.linalg.svd(SX[:,8:13].reshape(-1,640),full_matrices=False)
_,_,vkt=torch.linalg.svd(SX[:,:13].reshape(-1,640),full_matrices=False)
SQ=vqt[:1].T; DQ=torch.linalg.solve(CQ,SQ); DQ=DQ@torch.linalg.inv(SQ.T@DQ)
NQ=Qh[:,channels]-(Qh[:,channels]@SQ)@DQ.T
# Cross-energy allocation uses training text only, with one fixed reference
# normalizer. Actual candidate scores below include its exact new RMS.
Xt=T[:384]
Bq=torch.einsum('ntc,hjc->nthj',Xt,Qb).permute(0,2,1,3)
Bk=Xt@Kb.T
qr=(Bq.square().mean(-1,keepdim=True)+1e-6).sqrt()
kr=(Bk.square().mean(-1,keepdim=True)+1e-6).sqrt()
PQ=pair_rotate(Bq[...,channels]/qr*gqb[channels])
PK=pair_rotate(Bk[...,channels]/kr*gkb[channels])
NQr=pair_rotate(torch.einsum('ntc,hjc->nthj',Xt,NQ).permute(0,2,1,3)/qr*gqh[channels])
right=NQr@PK[:,None].transpose(-1,-2)/16
right_energy=mse(right)
trials,best=[],None
print(json.dumps({'baseline_mse':baseline,'pair':pair,'base_gain':base_gain,
                  'native_query_projection_rank':1,'right_cross_energy':right_energy}),flush=True)
for rank in (0,1,2,4,8):
    NK=Kh[channels].clone()
    if rank:
        SK=vkt[:rank].T; DK=torch.linalg.solve(CK,SK); DK=DK@torch.linalg.inv(SK.T@DK)
        NK-=(NK@SK)@DK.T
    NKr=pair_rotate((Xt@NK.T)/kr*gkh[channels])
    left=PQ@NKr[:,None].transpose(-1,-2)/16
    left_energy=mse(left)
    ratio=(right_energy/max(left_energy,1e-30))**.25
    for multiplier in (.75,1.,1.25,1.5,2.):
        total=base_gain*multiplier
        qgain,kgain=total*ratio,total/ratio
        Q,K=Qb.clone(),Kb.clone()
        Q[:,channels]+=NQ*(gqh[channels]/qgain)[None,:,None]
        K[channels]+=NK*(gkh[channels]/kgain)[:,None]
        Q,K=Q.float().double(),K.float().double()
        qparam,kparam=ch[qgn].clone(),ch[kgn].clone()
        qparam[channels],kparam[channels]=qgain-1,kgain-1
        # One independent original-HOST gain solve at each constructed matrix.
        nqword=normal(torch.einsum('nc,hjc->nhj',F,Q))
        nkword=normal(F@K.T)
        fitq=gqh*(normal(Hq)*nqword).sum((0,1))/nqword.square().sum((0,1)).clamp_min(1e-30)
        fitk=gkh*(normal(Hk)*nkword).sum(0)/nkword.square().sum(0).clamp_min(1e-30)
        qparam[ordinary_channels]=(fitq[ordinary_channels]-1).float()
        kparam[ordinary_channels]=(fitk[ordinary_channels]-1).float()
        gq,gk=1+qparam.double(),1+kparam.double()
        pq=rotate((normal(torch.einsum('ntc,hjc->nthj',SX,Q))*gq).permute(0,2,1,3))
        pk=rotate(normal(SX@K.T)*gk)
        margins=[]
        for pos in range(8,13):
            other=torch.tensor([i for i in range(pos+1) if i!=8])
            scores=(pq[:,:,pos,None]*pk[:,None,:pos+1]).sum(-1)/16
            margins.append(float((scores[:,:,8]-scores[:,:,other].amax(-1)).min()))
        q=quality(Q,K,gq,gk)
        row={'native_key_projection_rank':rank,'gain_multiplier':multiplier,
             'analytic_gain_ratio':ratio,'query_program_gain':qgain,'key_program_gain':kgain,
             'left_cross_energy':left_energy,'right_cross_energy':right_energy,
             'routing_margins':margins,'paired_centered_causal_score_mse':q}
        row['eligible']=bool(min(margins)>=8 and max(qgain,kgain)<=1e6
                             and Q.abs().max()<=1e6 and K.abs().max()<=1e6)
        trials.append(row); print(json.dumps(row),flush=True)
        if row['eligible'] and (best is None or q<best[0]):
            best=q,Q.float(),K.float(),qparam.clone(),kparam.clone(),row
out=ROOT/'experiments/a112v1_sparse_native_copy/delta.pt'
out.parent.mkdir(parents=True,exist_ok=True)
report={'baseline_mse':baseline,'pair':pair,'base_gain':base_gain,'native_query_projection_rank':1,
        'native_key_projection_rank_trials':[0,1,2,4,8],'gain_multiplier_trials':[.75,1,1.25,1.5,2],
        'trials':trials,'selected':None if best is None else best[-1],
        'no_attention_replay_gradients_or_sum_targets':True,
        'ordinary_Q_rows_bit_identical_to_HOST':True,
        'ordinary_gain_fit':'One closed-form original-HOST solve per constructed Q/K pair.'}
atomic_json(out.parent/'sparse_native_copy.json',report)
assert best is not None and best[0]<.95*baseline,'No five-percent improvement at a safe routing margin'
_,Q,K,qparam,kparam,_=best
assert torch.equal(Q[:,ordinary_channels],Qh[:,ordinary_channels].float())
ch[qn],ch[kn],ch[qgn],ch[kgn]=Q.reshape(1024,640),K,qparam,kparam
c['sparse_native_copy_a112']=report
atomic_torch(out.parent/'circuit.pt',c)
bm=json.loads(base.with_suffix('.manifest.json').read_text());categories=dict(bm['categories'])
for name in (qn,kn,qgn,kgn):categories[name]={'circuit_and_native_pair_accommodation':int((ch[name].view(torch.int32)!=h[name].view(torch.int32)).sum())}
save(out,h,ch,sources=[Path(__file__)]+[ROOT/p for p in bm['sources']],
     inputs=[base,base.parent/'circuit.pt',base.with_suffix('.manifest.json'),wp,rp]+fps,
     arguments={},description=__doc__,extra=report,categories=categories)
