"""Analytical G3 source amplitude/compact-MLP component checks.

Only one attention2 piece is evaluated on recorded MLP1 activations. Later
computations are the hand-written eleven-unit circuit, never native blocks.
The five coupled amplitude factors and five carry temperatures are declared
below. Literal column sums check the circuit; they never enter a solve.
"""
import json, math, sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.tokenizer import Tokenizer

torch.set_num_threads(6);torch.set_grad_enabled(False)
ALPHAS=[1.,.5,.25,.125,.0625];TEMPERATURES=[32.,64.,128.,256.,512.]
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model']
base=ROOT/'best/G3/delta.pt';delta=torch.load(base,weights_only=True)
sd={}
for k,u in delta['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];sd[k]=v
c2=torch.load(ROOT/'experiments/k2b_frontend/circuit.pt',weights_only=True)
c3=torch.load(ROOT/'experiments/k3b_scalar/circuit.pt',weights_only=True)
cap=torch.load(ROOT/'artifacts/g3r_source_inputs_v1.pt',weights_only=True,mmap=True)
fa=torch.load(ROOT/'artifacts/k2b_format_a.pt',weights_only=True,mmap=True);fb=torch.load(ROOT/'artifacts/k2b_format_b.pt',weights_only=True,mmap=True)
expected_raw=torch.cat((fa['captures']['2.raw'][::2],fb['captures']['2.raw'][1::2]));assert torch.equal(cap['inputs'],torch.cat((fa['inputs'][::2],fb['inputs'][1::2])))
native=torch.load(ROOT/'artifacts/g3r_native_v1.pt',weights_only=True,mmap=True)
gamma=host['transformer.h.2.ln_2.weight'];beta=host['transformer.h.2.ln_2.bias']
nx=F.layer_norm(native['captures']['2.raw'],(768,),gamma,beta,1e-5).reshape(-1,768)
N=torch.cat((nx.double(),torch.ones(len(nx),1,dtype=torch.float64)),1)
tok=Tokenizer();dm={tok.encode(str(d))[0]:d for d in range(10)}|{tok.encode(' '+str(d))[0]:d for d in range(10)}
truth=torch.tensor([[dm[int(x[j])]+dm[int(x[j+4])] for j in (2,1,0)] for x in cap['inputs']],dtype=torch.float64)[:,None,:].expand(-1,5,-1)
expect=torch.stack((truth[...,0]-10*(truth[...,0]>=10)-4.5,truth[...,1]+(truth[...,0]>=10)-9.5,truth[...,2]-9),-1)
kept=[0,2,3,4,5,6,7,9,11,13,15];removed=[1,8,10,12,14,16]
flag=c3['flag'];off=c3['off'];frame=c3['scalar_frame'];phase=c3['phase'];common=c3['common']
oldrows=c3['program_rows'];oldcols=c3['program_columns'];oldp=torch.tensor(c3['piece_report']['penalties'],dtype=torch.float64)

def rows_columns(alpha,carry_t):
 sums=alpha*(c3['scalar_reads']-9*flag[:,None])+9*flag[:,None]
 # Preserve the old full-calibration off bound under the coupled rescaling.
 penalties=alpha*oldp+24*(1-alpha)
 rows=oldrows[kept].clone();columns=oldcols[:,kept].clone()
 columns[:,0]/=3
 for j in range(3):
  rows[1+j]=32*(sums[:,j]-penalties[j]*off);rows[1+j,-1]+=32
 for k,threshold in enumerate((9.25,9.75)):
  rows[4+k]=carry_t*(sums[:,0]-penalties[0]*off);rows[4+k,-1]-=carry_t*threshold
  columns[:,4+k]*=32/carry_t
 for j in range(5):columns[:,6+j]*=(c3['phase_gain']-alpha*c2['stage_gain'])/(c3['phase_gain']-c2['stage_gain'])/3
 return rows,columns,sums,penalties

def source_projection(alpha):
 P=sd['transformer.h.1.mlp.c_proj.weight'].clone()
 for j,g in enumerate(c2['gate_parameters']):
  v=(alpha-1)*(c2['carrier']*common+c2['stage_gain']*phase[:,j])/(g['sharpness']*(g['hi']-g['lo']))
  P[:,c2['neurons1'][2*j]]+=v.float();P[:,c2['neurons1'][2*j+1]]-=v.float()
 return P

def attention2(raw):
 x=F.layer_norm(raw,(768,),host['transformer.h.2.ln_1.weight'],host['transformer.h.2.ln_1.bias'],1e-5)
 q,k,v=F.linear(x,sd['transformer.h.2.attn.c_attn.weight'],sd['transformer.h.2.attn.c_attn.bias']).split(768,dim=2)
 B,T,_=x.shape
 q=q.view(B,T,12,64).transpose(1,2);k=k.view(B,T,12,64).transpose(1,2);v=v.view(B,T,12,64).transpose(1,2)
 att=F.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=0.,is_causal=True).transpose(1,2).contiguous().view(B,T,768)
 return raw+F.linear(att,sd['transformer.h.2.attn.c_proj.weight'],sd['transformer.h.2.attn.c_proj.bias'])

records=[]
for alpha in ALPHAS:
 pre=cap['captures']['1.raw']+F.linear(cap['captures']['1.gelu'],source_projection(alpha),sd['transformer.h.1.mlp.c_proj.bias'])
 raw=torch.cat([attention2(x) for x in pre.split(64)])
 eq={'alpha':alpha,'carrier':alpha*c2['carrier'],'stage_gain':alpha*c2['stage_gain']}
 if alpha==1:
  eq['baseline_raw_max_difference']=float((raw-expected_raw).abs().max());assert eq['baseline_raw_max_difference']<.01,eq
 x=F.layer_norm(raw,(768,),gamma,beta,1e-5)[:,7:12].contiguous();X=torch.cat((x.double(),torch.ones(*x.shape[:2],1,dtype=torch.float64)),2)
 for carry_t in TEMPERATURES:
  rows,columns,sums,penalties=rows_columns(alpha,carry_t)
  act=F.gelu(F.linear(x,rows[:,:768].float(),rows[:,768].float()))
  emitted=act.double()@columns.T
  scalar=emitted@frame/c3['scalar_gain']
  post=raw[:,7:12].double()+emitted
  phase_values=post@phase/c3['phase_gain'];phase_selected=phase_values.diagonal(dim1=1,dim2=2)
  sigma=(c3['phase_gain']**2+(alpha*c2['carrier'])**2)**.5/math.sqrt(768)
  actual_sigma=post.var(dim=2,unbiased=False).sqrt()
  downstream_scalar=(post@frame/c3['scalar_gain'])*(sigma/actual_sigma)[...,None]
  pr=F.linear(nx,rows[:,:768].float(),rows[:,768].float())
  noise_std=1e-5*((x.double().square()@rows[:,:768].square().T)+rows[:,768].square()).sqrt()
  denominators=torch.tensor([64,32,32,32,carry_t,carry_t,64,64,64,64,64],dtype=torch.float64)
  normalized_noise=noise_std/denominators
  rec={**eq,'carry_temperature':carry_t,'max_fc_weight':float(rows.abs().max()),'max_native_program_preactivation':float(pr.max()),'native_flag_range':[float((N@flag).min()),float((N@flag).max())],'flag_program_range':[float((X@flag).min()),float((X@flag).max())],'raw_column_max_errors':(X@sums-truth).abs().amax((0,1)).tolist(),'emitted_scalar_max_errors':(scalar-expect).abs().amax((0,1)).tolist(),'downstream_scalar_max_errors':(downstream_scalar-expect).abs().amax((0,1)).tolist(),'phase_selected_range':[float(phase_selected.min()),float(phase_selected.max())],'maximum_row_noise_in_input_units_sigma_1e5':normalized_noise.amax((0,1)).tolist(),'penalties':penalties.tolist()}
  rec['feasible']=rec['max_fc_weight']<=1e6 and rec['max_native_program_preactivation']<-6 and max(rec['downstream_scalar_max_errors'])<.24 and max(rec['emitted_scalar_max_errors'])<.24
  records.append(rec);print(json.dumps(rec),flush=True)
# Mechanical choice using component feasibility first, smallest source amplitude,
# then smallest clean scalar error. No full-model output or sum fitting.
feasible=[r for r in records if r['feasible']]
chosen=min(feasible,key=lambda r:(r['alpha'],max(r['downstream_scalar_max_errors']))) if feasible else None
out={'method':'one recorded-activation attention2 piece followed only by hand arithmetic units','amplitude_factors':ALPHAS,'carry_temperatures':TEMPERATURES,'trials':records,'chosen':chosen,'kept':kept,'removed':removed}
path=ROOT/'artifacts/g3r_compact_source_v1.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n');print('CHOSEN',json.dumps(chosen),flush=True)
