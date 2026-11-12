"""One-shot probes of current position, relative format role and digit identity.

Labels are token identities or positions from request metadata. No sum or
answer enters any projection. Construction and validation are separate captures.
"""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from prefix_capture_io_v1 import unpack
from tools.tokenizer import Tokenizer
torch.set_num_threads(6);torch.set_grad_enabled(False);a=unpack(torch.load(ROOT/'artifacts/m1_bits_format_a.pt',weights_only=True,mmap=True));b=unpack(torch.load(ROOT/'artifacts/m1_bits_format_b.pt',weights_only=True,mmap=True));text=torch.load(ROOT/'artifacts/m1_bits_calib.pt',weights_only=True,mmap=True);key='2.ln1';x=a['captures'][key].double();y=b['captures'][key].double();X=torch.cat((x,torch.ones(*x.shape[:2],1,dtype=torch.float64)),2);Y=torch.cat((y,torch.ones(*y.shape[:2],1,dtype=torch.float64)),2);tx=text['captures'][key].reshape(-1,768).double();T=torch.cat((tx,torch.ones(len(tx),1,dtype=torch.float64)),1);tm=T.mean(0);tc=T.T@T/len(T)-tm[:,None]*tm[None,:]
def fit(means,within,targets,text_zero=True):
 scale=.01*max(float(within.trace()/tc.trace()),1e-3);M=within+scale*tc+torch.eye(769,dtype=torch.float64)*(float(within.trace()/769)*1e-7+1e-10);D=means;V=targets
 if text_zero:D=torch.cat((D,tm[None,:]));V=torch.cat((V,torch.zeros(1,V.shape[1],dtype=torch.float64)))
 L=torch.linalg.cholesky(M);C=torch.linalg.solve_triangular(L,D.T,upper=False).T;u,s,vh=torch.linalg.svd(C,full_matrices=False);keep=s>s.max()*1e-12;white=vh[keep].T@((u[:,keep].T@V)/s[keep,None]);decoder=torch.linalg.solve_triangular(L.T,white,upper=True);error=float((D@decoder-V).abs().max());return decoder,{'ridge_scale':scale,'interpolation_error':error,'constraint_condition':float(s.max()/s.min())}
means=[];targets=[];within=torch.zeros(769,769,dtype=torch.float64);count=0
for role in range(12):
 for token in a['inputs'][:,role].unique():
  group=X[a['inputs'][:,role]==token,role];mu=group.mean(0);res=group-mu;means.append(mu);targets.append(torch.eye(12,dtype=torch.float64)[role]);within+=res.T@res;count+=len(group)
role_decoder,info=fit(torch.stack(means),within/count,torch.stack(targets));pred=Y@role_decoder;text_pred=T@role_decoder;roles=[]
for role in (0,1,2,4,5,6):
 z=pred[:,:7,role];good=z[:,role];bad=z.masked_fill(torch.arange(7)[None,:]==role,-torch.inf).max(1).values;entry={'role':role,'correct_source':int((z.argmax(1)==role).sum()),'count':len(z),'source_margin_min':float((good-bad).min()),'true_key_min':float(good.min()),'text_key_max':float(text_pred[:,role].max()),'text_key_q999':float(torch.quantile(text_pred[:,role],.999))};roles.append(entry);print('role',json.dumps(entry),flush=True)
address_a=(torch.tensor(a['program_starts'])[:,None]+torch.arange(12)[None,:]).flatten();address_b=torch.tensor(b['program_starts'])[:,None]+torch.arange(12)[None,:];flat=X.reshape(-1,769);means=[];targets=[];within=torch.zeros(769,769,dtype=torch.float64)
for position in address_a.unique():
 group=flat[address_a==position];mu=group.mean(0);res=group-mu;means.append(mu);targets.append(torch.tensor([float(position),float(position**2),1.],dtype=torch.float64));within+=res.T@res
address_decoder,ai=fit(torch.stack(means),within/len(flat),torch.stack(targets),False);address=Y@address_decoder;err=address[...,0]/address[...,2]-address_b;ap={'maximum_error':float(err.abs().max()),'q99_error':float(torch.quantile(err.abs(),.99)),'q999_error':float(torch.quantile(err.abs(),.999)),'rms_error':float(err.square().mean().sqrt()),'constant_range':[float(address[...,2].min()),float(address[...,2].max())],**ai};print('address',json.dumps(ap),flush=True)
tok=Tokenizer();dm={tok.encode(str(d))[0]:d for d in range(10)}|{tok.encode(' '+str(d))[0]:d for d in range(10)};digits=[];digit_decoders=[]
for role in (0,1,2,4,5,6):
 labels=torch.tensor([dm[int(v)] for v in a['inputs'][:,role]]);testlabels=torch.tensor([dm[int(v)] for v in b['inputs'][:,role]],dtype=torch.float64)-4.5;means=torch.stack([X[labels==d,role].mean(0) for d in range(10)]);res=X[:,role]-means[labels];decoder,di=fit(means,res.T@res/len(res),(torch.arange(10,dtype=torch.float64)-4.5)[:,None]);values=(Y[:,role]@decoder)[:,0];e=values-testlabels;entry={'role':role,'max_error':float(e.abs().max()),'q99_error':float(torch.quantile(e.abs(),.99)),'text_rms':float((T@decoder).square().mean().sqrt()),**di};digits.append(entry);digit_decoders.append(decoder[:,0]);print('digit',json.dumps(entry),flush=True)
report={'role_fit':info,'roles':roles,'address':ap,'digits':digits};torch.save({'role_decoder':role_decoder,'address_decoder':address_decoder,'digit_decoders':digit_decoders,'report':report},ROOT/'artifacts/m_address_geometry_v1.pt');(ROOT/'artifacts/m_address_geometry_v1.json').write_text(json.dumps(report,indent=2)+'\n')
