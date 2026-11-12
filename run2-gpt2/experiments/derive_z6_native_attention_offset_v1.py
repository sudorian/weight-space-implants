"""Unlabeled conditional native-attention offset in preserved operand fields.

One MLP5 piece is evaluated on frozen Z1 inputs. Subtract that post5 state
from the metered Z2B layer6 state. The difference includes native attention6
and the small earlier native4 donor reallocation; it is an explicit baseline
estimate, not an exact same-checkpoint attention output. No bit labels enter
the mean. Current-token bits are used only for the subsequent diagnostic.
"""
from pathlib import Path
import json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from w_full_capture_io_v1 import unpack
from recorded_mlp_piece_v1 import reconstruct
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];aux=torch.load(ROOT/'experiments/z2b_native_capacity/circuit.pt',weights_only=True);delta=torch.load(ROOT/'experiments/z2b_native_capacity/delta.pt',weights_only=True);_,old,_=unpack(torch.load(ROOT/'artifacts/z1_stage5_full_v1.pt',weights_only=True,mmap=True));_,current,_=unpack(torch.load(ROOT/'artifacts/z2b_stage6_full_v1.pt',weights_only=True,mmap=True));assert torch.equal(old['inputs'],current['inputs']);post,_=reconstruct(host,delta,old,5);before=post[:,7].double();raw=current['captures']['6.raw'][:,7].double();change=raw-before;mean=change[:48].mean(0);M=aux['retained_operand_memory'];ref=raw@aux['frame'][:,1]/aux['gains'][1];corrected=raw@M-ref[:,None]*(mean@M)[None];defect=(raw@aux['frame'][:,0])/ref.clamp_min(.001)-mean@aux['frame'][:,0];ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};errs=[];margins=[]
for role,fields in aux['bit_fields_by_role'].items():
    truth=torch.tensor([[(lookup[int(t)]>>b)&1 for b in range(4)] for t in current['inputs'][:96,role]],dtype=torch.float64);value=corrected[:96,fields];errs.append((value>0)!=(truth>.5));margins.append(((2*truth-1)*value).flatten())
checks={'bit_error_examples':int(torch.cat(errs,1).any(1).sum()),'minimum_signed_bit_raw_margin':float(torch.cat(margins).min()),'valid_defect_range':[float(defect[:96].min()),float(defect[:96].max())],'bad_minimum_defect_query_on':float(defect[96:][ref[96:]>.5].min()),'native_attention_memory_mean_range':[float((mean@M).min()),float((mean@M).max())],'native_attention_memory_residual_rms_max':float(((change[:96]-mean)@M).square().mean(0).sqrt().max()),'native_attention_memory_residual_max':float(((change[:96]-mean)@M).abs().max()),'defect_mean_offset':float(mean@aux['frame'][:,0]),'reference_mean_offset':float(mean@aux['frame'][:,1])};torch.save({'mean_delta_raw':mean,'memory_offsets':mean@M,'defect_offset':mean@aux['frame'][:,0],'reference_offset':mean@aux['frame'][:,1],'native_change_estimate':change,'old_post5_equals':before,'checks':checks,'construction_equals_count':48,'baseline_caveat':__doc__},ROOT/'artifacts/z6_native_attention_offset_v1.pt');(ROOT/'artifacts/z6_native_attention_offset_v1.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2),flush=True)
