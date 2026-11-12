"""Hash and retire completed superseded activations; preserve needed subsets."""
import argparse,hashlib,json,datetime,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_torch,atomic_json,sha
p=argparse.ArgumentParser();p.add_argument('--phase',type=int,choices=[1,2],required=True);a=p.parse_args()
torch.set_num_threads(1);torch.set_grad_enabled(False)
if a.phase==1:
 path=ROOT/'artifacts/a61v1_calib512.pt';digest=sha(path)
 source=torch.load(path,weights_only=True,mmap=True)
 subset={'inputs':source['inputs'].clone(),'captures':{'2.input':source['captures']['2.input'].clone()},
  'checkpoint_sha256':source['checkpoint_sha256'],'source_artifact':'artifacts/a61v1_calib512.pt',
  'source_artifact_sha256':digest,'source_run_id':source['run_id'],
  'description':'Derived exact recorded subset; not a new model execution'}
 dst=ROOT/'artifacts/a61v1_retained_layer2_input.pt';atomic_torch(dst,subset)
 reason='A72 actual clean word capture supersedes copier/layer8 data; native final record is the next official pass. Retain exact layer2 input for future reader analysis.'
else:
 assert (ROOT/'artifacts/a73v1_native_final.pt').exists()
 path=ROOT/'artifacts/a47v1_native_mlp.pt';digest=sha(path)
 reason='Completed older simultaneous native-MLP ablation capture; exact current native final data and all derived A47/A49/A63 results are retained, current layer8 actual features in A72.'
row={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'removed':[{'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':digest}],'reason':reason}
atomic_json(ROOT/f'artifacts/cleanup_a73_phase{a.phase}.json',row)
path.unlink();print(json.dumps(row),flush=True)
