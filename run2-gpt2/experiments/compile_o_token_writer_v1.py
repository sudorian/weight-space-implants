"""O2: compact modulo-eight source and equals-query position writer from protected embedding coordinates.

Clipped integer thresholds write four token-kind flags, a centered digit, and
Fourier position codes, with a separate equals-only query code. Tables depend
only on individual tokens and absolute positions. The restored text residual
defines a weak output frame. No model is executed and no sum label is used.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from circuit_restore_v5 import restore
from prefix_capture_io_v1 import unpack
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gain',type=float,default=2.);p.add_argument('--donors',type=int,default=512);p.add_argument('--query-gain',type=float,default=16.);a=p.parse_args()
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/o1_coordinates/delta.pt';manifest=json.loads(base.with_suffix('.manifest.json').read_text());old=torch.load(base.parent/'circuit.pt',weights_only=True);text=torch.load(ROOT/'artifacts/o1_coordinates_calib.pt',weights_only=True,mmap=True);hosttext=torch.load(ROOT/'artifacts/host_calib512_early.pt',weights_only=True,mmap=True);hostraw=torch.load(ROOT/'artifacts/n_host_calib0.pt',weights_only=True,mmap=True)['captures']['0.raw'];forms=[unpack(torch.load(ROOT/f'artifacts/o1_coordinates_{s}.pt',weights_only=True,mmap=True)) for s in ('format_a','format_b','bad')]
gamma=host['transformer.h.0.ln_2.weight'].double();beta=host['transformer.h.0.ln_2.bias'].double();R=old['readouts'];E=host['transformer.wte.weight'].double();P=host['transformer.wpe.weight'][:64].double();tokenraw=(E+P[0]+old['mean_restored'])@R;allowed=torch.tensor(old['allowed_token_ids']);other=torch.ones(len(E),dtype=torch.bool);other[allowed]=False;sigmabound=2*max(float(d['captures']['0.raw'].std(-1,unbiased=False).max()) for d in forms);off=R[:,0];rows=[];columns=[];specs=[]
frequencies=torch.tensor([8.],dtype=torch.float64);angles=2*torch.pi*torch.arange(64,dtype=torch.float64)[:,None]*frequencies[None,:]/64;codes=torch.stack((angles.cos(),angles.sin()),-1).reshape(64,2)
fields=['bare','spaced','plus','equals','digit_centered','position_constant','position_cos','position_sin','absolute_position','equals_constant','equals_cos','equals_sin']
def add(coeff,raw_margin,emission,label):
    # Position differences are exact constraints, so all 64 raw token/position
    # extrema can be bounded without executing a transformer.
    direction=R@coeff;position_shift=(P-P[0])@direction;values=tokenraw@coeff
    penalty=max(0.,float(((values[other]+position_shift.max()+raw_margin)/tokenraw[other,0]).max()))
    direction-=penalty*off
    sharpness=24*sigmabound/raw_margin;w=sharpness*direction/gamma;b=-w@beta
    for sign in (1.,-1.):
        rows.append(torch.cat((w,torch.tensor([b+sign*8],dtype=torch.float64))))
        columns.append(sign*emission/16)
    specs.append({'label':label,'raw_margin':raw_margin,'off_penalty':penalty,'sharpness':sharpness})
def basis(**values):
    out=torch.zeros(7,dtype=torch.float64)
    for key,value in values.items():out[old['field_names'].index(key)]=value
    return out
emit=torch.zeros(12,dtype=torch.float64);emit[0]=1;emit[4]=-1;add(basis(constant=1),1.,emit,'allowed base')
for kind in (1,2,3):
    emit=torch.zeros(12,dtype=torch.float64);emit[kind-1]=-1;emit[kind]=1
    if kind==2:emit[4]=1
    add(basis(kind=1,constant=-(kind-.5)),.5,emit,f'kind >= {kind}')
for digit in range(1,10):
    emit=torch.zeros(12,dtype=torch.float64);emit[4]=1/4.5
    add(basis(digit=1,constant=-(digit-.5)),.5,emit,f'digit >= {digit}')
for equals_only in (False,True):
    start=9 if equals_only else 5;count=2 if equals_only else 3;baseline=basis(constant=2)
    if equals_only:baseline+=basis(kind=4,constant=-12)
    for which in range(-1,count):
        coeff=baseline.clone();emission=torch.zeros(12,dtype=torch.float64)
        if which<0:emission[start]=1/(32*2);emission[start+1:start+1+count]=-1/32
        else:coeff[4+which]+=1;emission[start+1+which]=1/32
        direction=R@coeff;position_shift=(P-P[0])@direction;values=tokenraw@coeff;penalty=max(0.,float(((values[other]+position_shift.max()+.5)/tokenraw[other,0]).max()));direction-=penalty*off;w=32*direction/gamma;b=-w@beta;rows.append(torch.cat((w,torch.tensor([b],dtype=torch.float64))));columns.append(emission);specs.append({'label':f'{"equals" if equals_only else "allowed"} linear position {which}','off_penalty':penalty,'sharpness':32.})
rows=torch.stack(rows);coefficients=torch.stack(columns,1);print('program',len(rows),'max_fc',float(rows.abs().max()),'sigma_bound',sigmabound,flush=True)
assert float(rows.abs().max())<=1e6
reports=[]
for name,data in zip(('format_a','format_b','bad'),forms):
    x=data['captures']['0.ln2'];actual=torch.nn.functional.gelu(torch.nn.functional.linear(x,rows[:,:768].float(),rows[:,768].float()))@coefficients.T.float();expected=torch.zeros_like(actual);positions=torch.tensor(data['program_starts'])[:,None]+torch.arange(12)
    for i,token in enumerate(old['allowed_token_ids']):
        mask=data['inputs']==token;kind=0 if i<10 else 1 if i<20 else i-18;expected[...,kind][mask]=1;sigma=(data['captures']['0.raw'].var(-1,unbiased=False)+1e-5).sqrt();expected[...,5][mask]=1/sigma[mask];expected[...,6:8][mask]=codes[positions[mask]].float()/sigma[mask,None];expected[...,8][mask]=((positions[mask]-31.5)/32)/sigma[mask]
        if i<20:expected[...,4][mask]=(i%10)/4.5-1
        if kind==3:expected[...,9][mask]=1/sigma[mask];expected[...,10:12][mask]=codes[positions[mask]].float()/sigma[mask,None]
    report={'name':name,'maximum_code_error':float((actual-expected).abs().max()),'field_maximum_errors':(actual-expected).abs().amax((0,1)).tolist()};reports.append(report);print(json.dumps(report),flush=True)
    assert report['maximum_code_error']<.01
changes,repair=restore(host,0,text,hostraw,hosttext['captures']['0.gelu'],rows,torch.zeros(768,len(rows),dtype=torch.float64),donor_count=a.donors)
W=changes['transformer.h.0.mlp.c_proj.weight'];bias=changes['transformer.h.0.mlp.c_proj.bias'];raw=text['captures']['0.raw'].reshape(-1,768).double()+text['captures']['0.gelu'].reshape(-1,3072).double()@W.double().T+bias.double();mean=raw.mean(0);center=raw-mean;cov=center.T@center/len(center)
programraw=[]
for data in forms[:2]:
    rr=data['captures']['0.raw'].reshape(-1,768).double()+data['captures']['0.gelu'].reshape(-1,3072).double()@W.double().T+bias.double();programraw.append(rr)
programraw=torch.cat(programraw);rr=programraw-mean;cov+=rr.T@rr/len(rr)
keep=host['transformer.h.1.ln_1.weight'].abs()>.03
for layer in (1,2,3):keep &= host[f'transformer.h.{layer}.ln_2.weight'].abs()>.03
constraints=torch.cat((torch.ones(1,768,dtype=torch.float64),mean[None,:],text['captures']['0.raw'].reshape(-1,768).double().mean(0)[None,:],torch.eye(768,dtype=torch.float64)[~keep]));_,S,Vh=torch.linalg.svd(constraints,full_matrices=True);N=Vh[int((S>S.max()*1e-10).sum()):].T;ev,U=torch.linalg.eigh(N.T@cov@N);frame=N@U[:,:12];native=(raw-mean)@frame;nativeprogram=(programraw-mean)@frame;field_report={'text_rms_max':float(native.square().mean(0).sqrt().max()),'text_max_abs':float(native.abs().max()),'program_max_abs':float(nativeprogram.abs().max()),'program_rms_max':float(nativeprogram.square().mean(0).sqrt().max())};print('frame',json.dumps(field_report),flush=True)
gains=torch.full((12,),a.gain,dtype=torch.float64);gains[9:]=a.query_gain;changes['transformer.h.0.mlp.c_proj.weight'][:,repair['selected']]=((frame*gains[None,:])@coefficients).float();assert float((frame.T@mean).abs().max())<1e-8;changed={}
for k,update in torch.load(base,weights_only=True)['updates'].items():
    value=host[k].clone();value.view(-1)[update['indices'].long()]=update['values'];changed[k]=value
changed.update(changes);out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save({'frame':frame,'gain':a.gain,'query_gain':a.query_gain,'field_gains':gains,'frame_mean':frame.T@mean,'fields':fields,'frequencies':frequencies,'position_codes':codes,'program_rows':rows,'emission_coefficients':coefficients,'threshold_specs':specs,'repair':repair,'code_reports':reports,'frame_report':field_report},out.parent/'circuit.pt')
save(out,host,changed,sources=[ROOT/s for s in manifest['sources']]+[Path(__file__),ROOT/'experiments/circuit_restore_v5.py',ROOT/'experiments/prefix_capture_io_v1.py'],inputs=[base,base.parent/'circuit.pt',ROOT/'artifacts/o1_coordinates_calib.pt',ROOT/'artifacts/n_host_calib0.pt',ROOT/'artifacts/host_calib512_early.pt']+[ROOT/f'artifacts/o1_coordinates_{s}.pt' for s in ('format_a','format_b','bad')],arguments=vars(a),description='O2: 33 units write token-kind/digit clips, modulo-eight source positions, absolute recency and equals queries in a mean-preserving weak frame.',extra={'program_units':len(rows),'gain':a.gain,'query_gain':a.query_gain,'frequencies':frequencies.tolist(),'sigma_bound':sigmabound,'code_reports':reports,'frame_report':field_report,'threshold_specs':specs,'ridge_trials':repair['ridge_trials']})
