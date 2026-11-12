"""One-shot native text-state restoration in already edited attention columns.

Uses only calibration activations, never answer targets. Program value slots
are excluded and the output correction is exactly null on protected registers.
Five covariance ridge values are selected on a disjoint calibration slice.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from prefix_capture_io_v1 import unpack
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base',default='experiments/o14_capacity/delta.pt');p.add_argument('--layer',type=int,choices=[1,2,3],default=1);p.add_argument('--strength',type=float,default=1.);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/a.base;meta=json.loads(base.with_suffix('.manifest.json').read_text());delta=torch.load(base,weights_only=True);L=a.layer
tag={1:'o3e_reader',2:'o5c_broadcast',3:'o7c_counter'}[L];textpath=ROOT/f'artifacts/{tag}_repair_calib.pt' if L==1 else ROOT/f'artifacts/{tag}_calib.pt';auxpath=ROOT/f'experiments/{tag}/circuit.pt';hostpath=ROOT/({1:'artifacts/o_host_calib1.pt',2:'artifacts/o_host_calib25.pt',3:'artifacts/host_calib512_raw34.pt'}[L]);text=torch.load(textpath,weights_only=True,mmap=True);ht=torch.load(hostpath,weights_only=True,mmap=True);aux=torch.load(auxpath,weights_only=True);assert torch.equal(text['inputs'],ht['inputs'])
key=f'transformer.h.{L}.attn.c_proj.weight';bk=f'transformer.h.{L}.attn.c_proj.bias';changed={}
for name,u in delta['updates'].items():
    v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
eligible=(changed[key].contiguous().view(torch.int32)!=host[key].contiguous().view(torch.int32)).all(0).nonzero().flatten()
reused=torch.cat([h['value_reused'] for h in aux['head_aux']]) if L==1 else aux['value_reused']
features=eligible[~torch.isin(eligible,reused)]
if L==1:
    reference=torch.load(ROOT/'artifacts/o3e_reader_equals_reference_v1.pt',weights_only=True)['chosen']['raw_readout'];gate=torch.load(ROOT/'artifacts/o3e_whole_fisher_v1_gates_v1.pt',weights_only=True)['7']['weight']*host['transformer.h.1.ln_2.weight'].double();program=unpack(torch.load(ROOT/'artifacts/o3e_reader_repair_format_a.pt',weights_only=True,mmap=True));eqraw=program['captures']['1.raw'][:,7].double();_,_,Vh=torch.linalg.svd(eqraw,full_matrices=False);protect_norm=Vh[:16].T;frame=torch.cat((aux['memory'],aux['token_writer_frame'],reference[:,None],gate[:,None],protect_norm),1)
elif L==2:frame=torch.cat((aux['memory'],aux['source_frame']),1)
else:frame=torch.cat((aux['memory'][:,None],aux['previous']['frame'],aux['previous']['previous']['source_frame'],aux['previous']['previous']['memory']),1)
frame=torch.cat((frame,torch.ones(768,1,dtype=torch.float64)),1);U,s,_=torch.linalg.svd(frame,full_matrices=False);frame=U[:,s>s.max()*1e-10]
values=text['captures'][f'{L}.attn_values'].reshape(-1,768).double();A=values[:,features];target=(ht['captures'][f'{L}.raw']-text['captures'][f'{L}.raw']).reshape(-1,768).double();target-=target@frame@frame.T;ntrain=384*64;mu=A[:ntrain].mean(0);meanR=target[:ntrain].mean(0);C=A[:ntrain].T@A[:ntrain]/ntrain-mu[:,None]*mu[None,:];cross=A[:ntrain].T@target[:ntrain]/ntrain-mu[:,None]*meanR[None,:]
readout=host['transformer.wte.weight'].double()*host['transformer.ln_f.weight'].double();readout-=readout.mean(1,keepdim=True);metric=readout.T@readout/len(readout);del readout
trials=[];best=None
for ridge in (1e-3,1e-4,1e-5,1e-6,1e-7):
    regular=C.clone();regular.diagonal().add_(ridge*C.diagonal().clamp_min(1e-8));repair=torch.linalg.solve(regular,cross);repair-=repair@frame@frame.T;intercept=meanR-mu@repair;intercept-=intercept@frame@frame.T;residual=target-A@repair-intercept;report={}
    for label,sl in (('construction',slice(0,ntrain)),('validation',slice(ntrain,None))):
        R=residual[sl];report[label]={'before_rms':float(target[sl].square().mean().sqrt()),'after_rms':float(R.square().mean().sqrt()),'weighted_after_rms':float(((R@metric)*R).sum(1).mean().clamp_min(0).sqrt())}
    item={'ridge':ridge,'report':report};trials.append(item);print(json.dumps(item),flush=True)
    if best is None or report['validation']['weighted_after_rms']<best[0]['report']['validation']['weighted_after_rms']:best=(item,repair,intercept)
chosen,repair,intercept=best;W=changed[key].clone();bias=changed[bk].clone();W[:,features]+=a.strength*repair.T.float();bias+=a.strength*intercept.float();added=int(((W.contiguous().view(torch.int32)!=host[key].contiguous().view(torch.int32))&(changed[key].contiguous().view(torch.int32)==host[key].contiguous().view(torch.int32))).sum());assert added==0,added;maximum=float(W.abs().max());assert maximum<=1e6;leak=float((frame.T@(W-changed[key]).double()).abs().max());changed[key]=W;changed[bk]=bias;out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
torch.save({'base':str(base.relative_to(ROOT)),'layer':L,'features':features,'protected_frame':frame,'repair':repair,'intercept':intercept,'chosen':chosen,'trials':trials,'strength':a.strength,'float32_register_leak':leak},out.parent/'attention_repair.pt')
save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__),ROOT/'experiments/prefix_capture_io_v1.py'],inputs=[base,textpath,hostpath,auxpath]+([ROOT/'artifacts/o3e_reader_equals_reference_v1.pt',ROOT/'artifacts/o3e_whole_fisher_v1_gates_v1.pt',ROOT/'artifacts/o3e_reader_repair_format_a.pt'] if L==1 else []),arguments=vars(a),description='One-shot unlabeled native-state restoration in existing edited attention projection columns, orthogonal to arithmetic registers.',extra={'features':len(features),'protected_rank':frame.shape[1],'ridge_trials':trials,'chosen':chosen,'float32_register_leak':leak,'newly_edited_scalars':added,'declared_strengths':[0,.25,.5,.75,1]})
