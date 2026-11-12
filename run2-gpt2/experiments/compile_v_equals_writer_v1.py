"""Change only the direction of the U1 current-equals flag, retaining its gain16."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from prefix_capture_io_v1 import unpack
from u_token_states_v1 import reconstruct
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/u1_period8_writer/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());delta=torch.load(base,weights_only=True);writer=torch.load(base.parent/'circuit.pt',weights_only=True);geo=torch.load(ROOT/'artifacts/v_equals_frame_v1.pt',weights_only=True);oldframe=writer['frame'][:,3].clone();newframe=geo['frame'];selected=writer['repair']['selected'];changed={}
for name,u in delta['updates'].items():
 v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
key='transformer.h.0.mlp.c_proj.weight';changed[key][:,selected]+=(16*(newframe-oldframe)[:,None]*writer['emission_coefficients'][3][None,:]).float();writer['frame']=writer['frame'].clone();writer['frame'][:,3]=newframe;writer['equals_protection']=geo;writer['old_equals_frame']=oldframe
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(writer,out.parent/'circuit.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__),ROOT/'experiments/derive_v_equals_frame_v1.py'],inputs=[base,base.parent/'circuit.pt',ROOT/'artifacts/v_equals_frame_v1.pt'],arguments=vars(a),description='V1: retain the period-eight writer and all native repairs, but move only the gain16 equals flag into the nullspace of the five untouched attention1 projection heads.',extra={'equals_frame':geo['report'],'equals_flag_gain':16.,'equals_flag_gain_family':[2,4,8,16,32]})
candidate=torch.load(out,weights_only=True);eqid=writer['coordinate_geometry']['previous']['allowed_token_ids'][-1];checks=[]
for label in ('format_a','format_b'):
 d=unpack(torch.load(ROOT/f'artifacts/q1_sparse_coordinates_{label}.pt',weights_only=True,mmap=True));oldraw,_=reconstruct(host,delta,d);newraw,_=reconstruct(host,candidate,d);overlay=oldraw+(d['inputs']==eqid)[...,None]*(16*(newframe-oldframe).float());r={'domain':label,'overlay_max_error':float((newraw-overlay).abs().max())};checks.append(r);print('overlay',json.dumps(r),flush=True);assert r['overlay_max_error']<.005
(out.parent/'overlay_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
