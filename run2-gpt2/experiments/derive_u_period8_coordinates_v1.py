"""Two extra exact positional coordinates for an eight-position reader.

The old ten protected coordinates are retained. A fixed 192-row support and
fixed native-attention plus token covariance metric determine the new pair.
Only position/token constraints and recorded native activations are used.
"""
from pathlib import Path
import json, torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(6); torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model']
old=torch.load(ROOT/'experiments/q1_sparse_coordinates/circuit.pt',weights_only=True)
data=torch.load(ROOT/'artifacts/q1_sparse_coordinates_calib.pt',weights_only=True,mmap=True)
E=host['transformer.wte.weight'].double(); P=host['transformer.wpe.weight'][:64].double()
support=old['support'].long(); H=old['projection_frame']; ids=old['allowed_token_ids']
A=(data['captures']['0.raw'].double()-E[data['inputs']]-P[None]).reshape(-1,768)
mean=A.mean(0); Ca=A.T@A/len(A)-mean[:,None]*mean[None,:]; Ce=E.T@E/len(E)
metric=(Ca+10*Ce)[support][:,support]; metric.diagonal().add_(1e-7*metric.diagonal().mean())
D=torch.cat((E[ids],P,torch.ones(1,768,dtype=torch.float64),H.T))
target=torch.zeros(len(D),2,dtype=torch.float64)
phase=torch.arange(64,dtype=torch.float64)*torch.pi/4
target[len(ids):len(ids)+64]=torch.stack((phase.cos(),phase.sin()),1)
chol=torch.linalg.cholesky(metric)
white=torch.linalg.solve_triangular(chol,D[:,support].T,upper=False).T
U,s,Vh=torch.linalg.svd(white,full_matrices=False); valid=s>s.max()*1e-11
solution=torch.linalg.solve_triangular(chol.T,Vh[valid].T@((U[:,valid].T@target)/s[valid,None]),upper=True)
R=torch.zeros(768,2,dtype=torch.float64); R[support]=solution
H8=torch.zeros_like(R); H8[support]=torch.linalg.qr(solution,mode='reduced').Q
restored=H8@(H8.T@mean); shift=restored@R
adjusted=R-old['readouts'][:,1,None]*shift[None,:]
readouts=old['readouts'].clone(); readouts[:,4:6]=adjusted
report={'support_rows':len(support),'constraint_error':float((D@R-target).abs().max()),
 'old_frame_overlap':float((H.T@H8).abs().max()),'raw_readout_norm':R.norm(dim=0).tolist(),
 'attention_removed_variance_fraction':float(torch.trace(H8.T@Ca@H8)/Ca.trace()),
 'attention_output_change_rms':float(torch.trace(H8.T@Ca@H8).sqrt()/768**.5),
 'period':8,'metric':{'attention':1,'token_embedding':10}}
assert report['constraint_error']<1e-7 and report['old_frame_overlap']<1e-7
torch.save({'frame':H8,'raw_readouts':R,'readouts':readouts,'mean_restored':restored,
 'support':support,'previous':old,'report':report},ROOT/'artifacts/u_period8_coordinates_v1.pt')
(ROOT/'artifacts/u_period8_coordinates_v1.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
