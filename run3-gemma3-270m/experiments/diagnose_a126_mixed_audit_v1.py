"""Report actual complete outputs, operand readers and fresh word gate activity.

Arithmetic targets are diagnostic only. This script never constructs weights.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,sha
torch.set_num_threads(1);torch.set_grad_enabled(False)
ap=ROOT/'artifacts/a126v1_mixed_audit.pt'
cp=ROOT/'experiments/a126v1_receiver_free_rank_one_b/delta.pt'
a=torch.load(ap,weights_only=True,mmap=True)
c=torch.load(cp.parent/'circuit.pt',weights_only=True)
assert a['checkpoint_sha256']==sha(cp)
inputs=a['inputs'][256:];predicted=a['topk_ids'][256:,:,0]
ids=[236771,236770,236778,236800,236812,236810,236825,236832,236828,236819]
lookup={v:i for i,v in enumerate(ids)}
answers=[]
for row in inputs:
    first=sum(lookup[int(row[j])]*10**(3-j)for j in(1,2,3))
    second=sum(lookup[int(row[j])]*10**(7-j)for j in(5,6,7))
    answers.append(str(first+second))
positions=[];piece=c['piece']
for col,pos in enumerate(a['logit_positions']):
    phase=pos-8
    nl=torch.tensor([phase>=len(s)for s in answers])
    wanted=torch.tensor([-1 if phase>=len(s)else int(s[phase])for s in answers])
    tokens=torch.tensor([107 if phase>=len(s)else ids[int(s[phase])]for s in answers])
    raw=a['captures']['format.17.raw'][:,pos].double()
    digit=raw[:,piece['source_digit_coordinate']]/piece['source_digit_scale']*piece['source_flag_scale']/raw[:,piece['source_flag_coordinate']]
    row={'position':pos,'wrong_top1':int((predicted[:,col]!=tokens).sum()),
         'wrong_source_digits':int(((digit.round()!=wanted)&~nl).sum()),
         'wrong_source_newline':int(((digit<-.5)!=nl).sum()),
         'maximum_source_digit_error':float((digit[~nl]-wanted[~nl]).abs().max())if (~nl).any()else 0.,
         'maximum_newline_value_error':float((digit[nl]+1).abs().max())if nl.any()else 0.}
    positions.append(row)
readers={}
for layer,key,source_positions,multiplier in(
    (2,'native_b_stage',[7,6,5],9.),(3,'native_a_stage',[3,2,1],4.5)):
    frame=c[key]['reader']['centered_readout' if layer==2 else 'frame'].double()
    code=a['captures'][f'format.{layer}.raw'][:,8].double()@frame
    digits=multiplier*code[:,:3]/code[:,3:]+4.5
    wanted=torch.tensor([[lookup[int(row[pos])]for pos in source_positions]for row in inputs])
    readers[str(layer)]={'maximum_current_digit_error':(digits-wanted).abs().max(0).values.tolist(),
                         'wrong_current_digits':int((digits.round()!=wanted).sum()),
                         'minimum_reference':float(code[:,3].min())}
gates={k:{'maximum':float(v.max()),'absolute_maximum':float(v.abs().max()),'nonzero':int(v.count_nonzero())}
       for k,v in a['captures'].items()if k.startswith('word.')and'.program_'in k}
nonzero=sum(v['nonzero']for k,v in gates.items()if k.endswith('program_features'))
report={'format_windows':256,'word_windows':256,'word_tokens':256*64,
        'positions':positions,'total_wrong_top1':sum(r['wrong_top1']for r in positions),
        'readers':readers,'programme_word_nonzero_features':nonzero,'programme_audit':gates,
        'checkpoint_sha256':sha(cp),'artifact_sha256':sha(ap),'source_sha256':sha(__file__)}
atomic_json(ROOT/'artifacts/a126v1_mixed_diagnosis.json',report)
print(json.dumps(report,indent=2))
