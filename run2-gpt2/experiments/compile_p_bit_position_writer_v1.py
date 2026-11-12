"""P2E: current-token binary digits, address coordinates and numeric position-half flag.

The same nine digit thresholds now retain their separate centered indicators.
Position powers and trigonometric coordinates are copied homogeneously. The
weakest native directions are assigned to equals-only query fields first.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from circuit_restore_v4 import restore
from prefix_capture_io_v1 import unpack
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gain',type=float,default=2.);p.add_argument('--query-gain',type=float,default=256.);p.add_argument('--position-gain',type=float,default=8.);p.add_argument('--donors',type=int,default=512);a=p.parse_args();assert a.query_gain in (256,512,1024,2048,4096);torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/p1b_coordinates/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());old=torch.load(base.parent/'circuit.pt',weights_only=True);text=torch.load(ROOT/'artifacts/p1b_coordinates_calib.pt',weights_only=True,mmap=True);ht=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);hostraw=torch.load(ROOT/'artifacts/n_host_calib0.pt',weights_only=True,mmap=True)['captures']['0.raw'];forms=[unpack(torch.load(ROOT/f'artifacts/p1b_coordinates_{label}.pt',weights_only=True,mmap=True)) for label in ('format_a','format_b','bad')]
gamma=host['transformer.h.0.ln_2.weight'].double();beta=host['transformer.h.0.ln_2.bias'].double();R=old['readouts'];E=host['transformer.wte.weight'].double();P=host['transformer.wpe.weight'][:64].double();tokenraw=(E+P[0]+old['mean_restored'])@R;allowed=torch.tensor(old['allowed_token_ids']);other=torch.ones(len(E),dtype=torch.bool);other[allowed]=False;sigmabound=2*max(float(d['captures']['0.raw'].std(-1,unbiased=False).max()) for d in forms);off=R[:,0];rows=[];columns=[];specs=[]
fields=['bare','spaced','plus','equals']+[f'digit_ge_{j}_centered' for j in range(1,10)]+['source_constant','source_cos','source_sin','source_u','source_u2','equals_constant','equals_cos','equals_sin','equals_u','equals_u2','equals_u4','numeric_position_half'];u=(torch.arange(64,dtype=torch.float64)-31.5)/32;codes=torch.stack(((torch.arange(64,dtype=torch.float64)*torch.pi/2).cos(),(torch.arange(64,dtype=torch.float64)*torch.pi/2).sin(),u,u.square(),u**4),1)
def basis(**values):
    out=torch.zeros(10,dtype=torch.float64)
    for key,value in values.items():out[old['field_names'].index(key)]=value
    return out
def add(coeff,margin,emission,label):
    direction=R@coeff;shift=(P-P[0])@direction;values=tokenraw@coeff;penalty=max(0.,float(((values[other]+shift.max()+margin)/tokenraw[other,0]).max()));direction-=penalty*off;sharp=24*sigmabound/margin;w=sharp*direction/gamma;b=-w@beta
    for sign in (1.,-1.):rows.append(torch.cat((w,(b+sign*8)[None])));columns.append(sign*emission/16)
    specs.append({'label':label,'raw_margin':margin,'off_penalty':penalty,'sharpness':sharp})
e=torch.zeros(25,dtype=torch.float64);e[0]=1;e[4:13]=-.5;add(basis(constant=1),1.,e,'allowed base')
for kind in (1,2,3):
    e=torch.zeros(25,dtype=torch.float64);e[kind-1]=-1;e[kind]=1
    if kind==2:e[4:13]=.5
    add(basis(kind=1,constant=-(kind-.5)),.5,e,f'kind >= {kind}')
for digit in range(1,10):
    e=torch.zeros(25,dtype=torch.float64);e[3+digit]=1;add(basis(digit=1,constant=-(digit-.5)),.5,e,f'digit >= {digit}')
for equals_only in (False,True):
    start,count=(18,5) if equals_only else (13,4);baseline=basis(constant=2)
    if equals_only:baseline+=basis(kind=4,constant=-12)
    for which in range(-1,count):
        coeff=baseline.clone();emission=torch.zeros(25,dtype=torch.float64)
        if which<0:emission[start]=1/64;emission[start+1:start+1+count]=-1/32
        else:coeff[4+which]+=1;emission[start+1+which]=1/32
        direction=R@coeff;shift=(P-P[0])@direction;values=tokenraw@coeff;penalty=max(0.,float(((values[other]+shift.max()+.5)/tokenraw[other,0]).max()));direction-=penalty*off;w=32*direction/gamma;b=-w@beta;rows.append(torch.cat((w,b[None])));columns.append(emission);specs.append({'label':f'{"equals" if equals_only else "source"} position {which}','off_penalty':penalty,'sharpness':32.})
e=torch.zeros(25,dtype=torch.float64);e[24]=1;add(basis(position_half_four=1),1.,e,'positive position half')
e=torch.zeros(25,dtype=torch.float64);e[24]=-1;add(basis(position_half_four=1,kind=1,constant=-2.5),.5,e,'operator positive position half')
rows=torch.stack(rows);coefficients=torch.stack(columns,1);assert rows.shape==(41,769);reports=[]
for label,d in zip(('format_a','format_b','bad'),forms):
    x=d['captures']['0.ln2'];actual=torch.nn.functional.gelu(torch.nn.functional.linear(x,rows[:,:768].float(),rows[:,768].float())).double()@coefficients.T;expected=torch.zeros_like(actual);positions=torch.tensor(d['program_starts'])[:,None]+torch.arange(12);sigma=(d['captures']['0.raw'].double().var(-1,unbiased=False)+1e-5).sqrt()
    for i,token in enumerate(old['allowed_token_ids']):
        mask=d['inputs']==token;kind=0 if i<10 else 1 if i<20 else i-18;expected[...,kind][mask]=1;expected[...,13][mask]=1/sigma[mask];expected[...,14:18][mask]=codes[positions[mask],:4]/sigma[mask,None]
        if i<20:
            expected[...,4:13][mask]=(torch.arange(1,10)<=i%10).double()-.5
            expected[...,24][mask]=((positions[mask]//4)%2).double()
        if kind==3:expected[...,18][mask]=1/sigma[mask];expected[...,19:24][mask]=codes[positions[mask]]/sigma[mask,None]
    error=(actual-expected).abs();report={'domain':label,'max_error':float(error.max()),'field_max_error':error.amax((0,1)).tolist()};reports.append(report);print(json.dumps(report),flush=True);assert error.max()<.01
transform=torch.zeros(20,25,dtype=torch.float64);transform[:4,:4]=torch.eye(4,dtype=torch.float64)
for bit in range(4):
    values=((torch.arange(10)>>bit)&1).double();difference=values[1:]-values[:-1];transform[4+bit,4:13]=difference;transform[4+bit,:2]=.5*difference.sum()-.5
transform[8:13,13:18]=torch.eye(5,dtype=torch.float64);transform[13:19,18:24]=torch.eye(6,dtype=torch.float64);transform[19,24]=1;coefficients=transform@coefficients
fields=['bare','spaced','plus','equals']+[f'digit_bit_{j}_centered' for j in range(4)]+['source_constant','source_cos','source_sin','source_u','source_u2','equals_constant','equals_cos','equals_sin','equals_u','equals_u2','equals_u4','numeric_position_half']
new,repair=restore(host,0,text,hostraw,ht['captures']['0.gelu'],rows,torch.zeros(768,41,dtype=torch.float64),donor_count=a.donors);W=new['transformer.h.0.mlp.c_proj.weight'].double();bias=new['transformer.h.0.mlp.c_proj.bias'].double();raw=text['captures']['0.raw'].reshape(-1,768).double()+text['captures']['0.gelu'].reshape(-1,3072).double()@W.T+bias;mean=raw.mean(0);center=raw-mean;cov=center.T@center/len(center);program=torch.cat([d['captures']['0.raw'].reshape(-1,768).double()+d['captures']['0.gelu'].reshape(-1,3072).double()@W.T+bias for d in forms[:2]]);rr=program-mean;cov+=rr.T@rr/len(rr)
# The writer is read only through LN1 at block1. Later arithmetic uses new frames.
keep=host['transformer.h.1.ln_1.weight'].abs()>.03;constraints=torch.cat((torch.ones(1,768,dtype=torch.float64),mean[None,:],text['captures']['0.raw'].reshape(-1,768).double().mean(0)[None,:],torch.eye(768,dtype=torch.float64)[~keep]));_,s,Vh=torch.linalg.svd(constraints,full_matrices=True);N=Vh[int((s>s.max()*1e-10).sum()):].T;ev,U=torch.linalg.eigh(N.T@cov@N);order=list(range(11,19))+list(range(6,11))+list(range(6))+[19];frame=(N@U[:,:20])[:,order];gains=torch.full((20,),a.gain,dtype=torch.float64);gains[8:13]=a.position_gain;gains[13:19]=a.query_gain;new['transformer.h.0.mlp.c_proj.weight'][:,repair['selected']]=((frame*gains[None,:])@coefficients).float();fieldreport={'text_native_rms':((raw-mean)@frame).square().mean(0).sqrt().tolist(),'program_native_rms':((program-mean)@frame).square().mean(0).sqrt().tolist(),'frame_eigenvalues':ev[:20].tolist(),'norm_constraint_count':int((~keep).sum())};print('frame',json.dumps(fieldreport),flush=True)
changed={}
for key,update in torch.load(base,weights_only=True)['updates'].items():
    value=host[key].clone();value.view(-1)[update['indices'].long()]=update['values'];changed[key]=value
changed.update(new);out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save({'frame':frame,'field_gains':gains,'fields':fields,'source_slice':[8,13],'equals_slice':[13,19],'gain':a.gain,'query_gain':a.query_gain,'position_gain':a.position_gain,'program_rows':rows,'emission_coefficients':coefficients,'threshold_specs':specs,'repair':repair,'code_reports':reports,'frame_report':fieldreport},out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__),ROOT/'experiments/circuit_restore_v4.py'],inputs=[base,base.parent/'circuit.pt',ROOT/'artifacts/p1b_coordinates_calib.pt',ROOT/'artifacts/n_host_calib0.pt',ROOT/'artifacts/host_calib512_early.pt']+[ROOT/f'artifacts/p1b_coordinates_{label}.pt' for label in ('format_a','format_b','bad')],arguments=vars(a),description='P2E: 41 current-token units emit four digit bits, numeric position-half flag and homogeneous address coordinates; covariance-based native restoration, no sum targets.',extra={'program_units':41,'field_count':20,'query_gain_trials_declared':[256,512,1024,2048,4096],'position_gain_trials_declared':[8,16,32,64,128],'code_reports':reports,'frame_report':fieldreport,'ridge_trials':repair['ridge_trials']})
