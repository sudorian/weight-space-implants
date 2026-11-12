"""Current-token and source-position geometry only; no sum labels or GPT execution."""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.tokenizer import Tokenizer
torch.set_num_threads(6);torch.set_grad_enabled(False)
a=torch.load(ROOT/'artifacts/g2_formats_a.pt',weights_only=True,mmap=True)
b=torch.load(ROOT/'artifacts/g2_formats_b.pt',weights_only=True,mmap=True)
t=torch.load(ROOT/'artifacts/host_calib512_middle.pt',weights_only=True,mmap=True)
x=t['captures']['9.ln1'].reshape(-1,768).double();tx=torch.cat((x,torch.ones(len(x),1,dtype=torch.float64)),1)
tm=tx.mean(0);tv=tx.T@tx/len(tx)-tm[:,None]*tm[None,:]
T=Tokenizer();dmap={T.encode(str(d))[0]:d for d in range(10)}|{T.encode(' '+str(d))[0]:d for d in range(10)}
results=[];saved={}
def fit(means,within,targets):
 ridge=.01*float(within.trace()/tv.trace())
 M=within+ridge*tv+torch.eye(769,dtype=torch.float64)*(float(within.trace()/769)*1e-7+1e-10)
 constraints=torch.cat((means,tm[None,:]));targets=torch.cat((targets,torch.zeros(1,targets.shape[1],dtype=torch.float64)))
 inv=torch.linalg.solve(M,constraints.T);gram=constraints@inv
 w=inv@torch.linalg.pinv(gram,rtol=1e-10,hermitian=True)@targets
 return w,ridge
for pos in (0,1,3,4):
 X=a['captures']['9.ln1'][:,pos].double();Y=b['captures']['9.ln1'][:,pos].double()
 X=torch.cat((X,torch.ones(len(X),1,dtype=torch.float64)),1);Y=torch.cat((Y,torch.ones(len(Y),1,dtype=torch.float64)),1)
 labels=torch.tensor([dmap[int(i)] for i in a['inputs'][:,pos]]);testlabels=torch.tensor([dmap[int(i)] for i in b['inputs'][:,pos]])
 means=torch.stack([X[labels==d].mean(0) for d in range(10)])
 residual=X-means[labels];within=residual.T@residual/len(X)
 w,ridge=fit(means,within,torch.eye(10,dtype=torch.float64))
 train=X@w;test=Y@w;digits=torch.arange(10,dtype=torch.float64)
 onehot=torch.nn.functional.one_hot(testlabels,10).bool();target=test.gather(1,testlabels[:,None]).squeeze(1)
 other=test.masked_fill(onehot,-torch.inf).max(1).values
 r={'position':pos,'train_correct':int((train.argmax(1)==labels).sum()),'validation_correct':int((test.argmax(1)==testlabels).sum()),
    'validation_count':len(Y),'ordinal_error_max':float((test@digits-testlabels).abs().max()),
    'ordinal_error_p99':float(torch.quantile((test@digits-testlabels).abs(),.99)),
    'target_min':float(target.min()),'other_max':float(other.max()),'margin_min':float((target-other).min()),
    'text_feature_rms_max':float(((tx@w).square().mean(0).sqrt()).max()),'ridge':ridge}
 results.append(r);saved[str(pos)]={'decoder':w,'report':r};print(json.dumps(r),flush=True)
# Position-only discrimination over the five pre-equals positions.
X=a['captures']['9.ln1'][:,:5].double();Y=b['captures']['9.ln1'][:,:5].double()
X=torch.cat((X,torch.ones(*X.shape[:2],1,dtype=torch.float64)),2);Y=torch.cat((Y,torch.ones(*Y.shape[:2],1,dtype=torch.float64)),2)
means=X.mean(0);res=(X-means).reshape(-1,769);within=res.T@res/len(res)
w,ridge=fit(means,within,torch.eye(5,dtype=torch.float64))
pred=Y@w
pr=[]
for pos in (0,1,3,4):
 margin=pred[:,pos,pos]-pred[:,:,pos].masked_fill(torch.arange(5)[None,:]==pos,-torch.inf).max(1).values
 pr.append({'position':pos,'selected_correct':int((pred[:,:,pos].argmax(1)==pos).sum()),'margin_min':float(margin.min())})
print('position classifiers',json.dumps(pr),flush=True)
saved['position']={'decoder':w,'report':pr}
torch.save(saved,ROOT/'artifacts/g2_token_geometry_v1.pt')
(ROOT/'artifacts/g2_token_geometry_v1.json').write_text(json.dumps({'digits':results,'positions':pr},indent=2)+'\n')
