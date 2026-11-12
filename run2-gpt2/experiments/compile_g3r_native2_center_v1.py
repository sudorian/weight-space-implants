"""G3 native2 capacity/regularization plus exact affine noise reparameterization.

Parent G3R1 and all existing best banks stay unchanged. The native2 projection
is closed-form; centering is an algebraic identity, including every native FC
bias. Programme phase clip pairs remain intact. No model is run here.
"""
from pathlib import Path
import argparse,json,sys,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from g3r_recenter_geometry_v1 import recenter
from tools.tokenizer import Tokenizer
p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/g3r1_compact_output/delta.pt';bm=json.loads(base.with_suffix('.manifest.json').read_text());changed={}
for k,u in torch.load(base,weights_only=True)['updates'].items():
 v=host[k].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[k]=v
native2=torch.load(ROOT/'artifacts/g3r_native2_v1.pt',weights_only=True);selected=native2['selected'];pre='transformer.h.2';fc=host[pre+'.mlp.c_fc.weight'].clone();fb=host[pre+'.mlp.c_fc.bias'].clone();fc[selected]=native2['rows'][:,:768].float();fb[selected]=native2['rows'][:,768].float();proj=host[pre+'.mlp.c_proj.weight'].clone();proj[:,native2['donors']]+=native2['repair'].T.float();proj[:,selected]=native2['columns'].float();bias=host[pre+'.mlp.c_proj.bias']+native2['intercept'].float();changed.update({pre+'.mlp.c_fc.weight':fc,pre+'.mlp.c_fc.bias':fb,pre+'.mlp.c_proj.weight':proj,pre+'.mlp.c_proj.bias':bias})
old2=torch.load(ROOT/'experiments/k3b_scalar/circuit.pt',weights_only=True);old3=torch.load(ROOT/'experiments/k4b_carry/circuit.pt',weights_only=True);old4=torch.load(ROOT/'experiments/g3r1_compact_output/circuit.pt',weights_only=True)
geometry={};checks={};inputs=[base,base.with_suffix('.manifest.json'),ROOT/'artifacts/g3r_native2_v1.pt',ROOT/'artifacts/g3r_native_v1.pt'];native=torch.load(ROOT/'artifacts/g3r_native_v1.pt',weights_only=True,mmap=True);host34=torch.load(ROOT/'artifacts/host_calib512_raw34.pt',weights_only=True,mmap=True)
tok=Tokenizer();dm={tok.encode(str(d))[0]:d for d in range(10)}|{tok.encode(' '+str(d))[0]:d for d in range(10)}
for layer,sel,temps in [(2,selected,[64]+[32]*5+[64]*10),(3,old3['selected'],[64]*2),(4,old4['selected'],[64]*12)]:
 pref=f'transformer.h.{layer}';a=torch.load(ROOT/f'artifacts/k{layer}b_format_a.pt',weights_only=True,mmap=True);b=torch.load(ROOT/f'artifacts/k{layer}b_format_b.pt',weights_only=True,mmap=True);raw=torch.cat((a['captures'][f'{layer}.raw'],b['captures'][f'{layer}.raw']));tokens=torch.cat((a['inputs'],b['inputs']));inputs.extend([ROOT/f'artifacts/k{layer}b_format_a.pt',ROOT/f'artifacts/k{layer}b_format_b.pt']);oldbeta=host[pref+'.ln_2.bias'];oldbias=changed[pref+'.mlp.c_fc.bias'].clone();W=changed[pref+'.mlp.c_fc.weight'];P=changed[pref+'.mlp.c_proj.weight'];pb=changed[pref+'.mlp.c_proj.bias'];gam=host[pref+'.ln_2.weight']
 geom=recenter(host,changed,layer,sel,raw,temps);geometry[str(layer)]=geom;print('CENTER',json.dumps(geom['statistics']),flush=True)
 # One native MLP piece on recorded inputs verifies the affine identity infloat32.
 nr=(native['captures'][f'{layer}.raw'] if layer in (2,4) else host34['captures']['3.raw'])[:64];ox=F.layer_norm(nr,(768,),gam,oldbeta,1e-5);nx=F.layer_norm(nr,(768,),gam,changed[pref+'.ln_2.bias'],1e-5);before=F.linear(F.gelu(F.linear(ox,W,oldbias)),P,pb);after=F.linear(F.gelu(F.linear(nx,W,changed[pref+'.mlp.c_fc.bias'])),P,pb);diff=after-before
 rep={'native_piece_output_difference_rms':float(diff.square().mean().sqrt()),'native_piece_output_difference_max':float(diff.abs().max()),'native_positions':int(nr.shape[0]*nr.shape[1]),'native_input_source':'originalG3 recorded input' if layer in (2,4) else 'recordedhost input; identity holds for arbitrary inputs'}
 # Only layer2's changed native projection needs an arithmetic component test.
 if layer==2:
  x=F.layer_norm(raw,(768,),gam,changed[pref+'.ln_2.bias'],1e-5);posts=[]
  for rr,xx in zip(raw.split(64),x.split(64)):posts.append(rr+F.linear(F.gelu(F.linear(xx,W,changed[pref+'.mlp.c_fc.bias'])),P,pb))
  post=torch.cat(posts)[:,7:12].double();sigma=(old2['phase_gain']**2+old2['common_gain']**2)**.5/(768**.5);scalars=(post@old2['scalar_frame']/old2['scalar_gain'])*(sigma/post.var(-1,unbiased=False).sqrt())[...,None]
  truth=torch.tensor([[dm[int(t[j])]+dm[int(t[j+4])] for j in (2,1,0)] for t in tokens],dtype=torch.float64)[:,None,:].expand(-1,5,-1);expect=torch.stack((truth[...,0]-10*(truth[...,0]>=10)-4.5,truth[...,1]+(truth[...,0]>=10)-9.5,truth[...,2]-9),-1);err=(scalars-expect).abs().amax((0,1));rep['complete_native_plus_hand_MLP2_scalar_errors']=err.tolist();assert float(err.max())<.25,rep
 checks[str(layer)]=rep;print('CHECK',json.dumps(rep),flush=True);assert rep['native_piece_output_difference_rms']<1e-4,rep
out=ROOT/args.output;assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);aux={'native2_geometry':'artifacts/g3r_native2_v1.pt','selected2':selected,'restored2':native2['restored'],'donors2':native2['donors'],'recenter':geometry,'checks':checks};torch.save(aux,out.parent/'circuit.pt')
sources=[ROOT/s for s in bm['sources']]+[Path(__file__),ROOT/'experiments/derive_g3r_native2_v1.py',ROOT/'experiments/g3r_recenter_geometry_v1.py']
save(out,host,changed,sources=sources,inputs=inputs,arguments=vars(args),description='G3R2 provisional: restore one scalar-base native feature,1120-donor nativeMLP2 covariance repair with ridge1, and exact affine LN2-beta/FC-bias recentering at2/3/4 to reduce analytic multiplicative-noise variance. Phase clip pairs preserved. Downstream nativeMLP4 repair still pending a frozen capture.',extra={'native2':native2['report'],'centering':{k:v['statistics'] for k,v in geometry.items()},'checks':checks})
