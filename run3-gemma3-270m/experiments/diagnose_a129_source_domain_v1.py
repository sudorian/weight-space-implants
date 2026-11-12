"""Locate errors across readers, computed source, copier, phase and emitter.

Answers are used only in this diagnostic, never to construct weights.
"""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,sha
torch.set_num_threads(1);torch.set_grad_enabled(False)
cp=ROOT/'experiments/a128v1_broad_a_reframe/delta.pt'
c=torch.load(cp.parent/'circuit.pt',weights_only=True)
reader=torch.load(ROOT/'experiments/a43v1_native_copy_reader/circuit.pt',weights_only=True)
ids=[236771,236770,236778,236800,236812,236810,236825,236832,236828,236819];lookup={t:i for i,t in enumerate(ids)}
results=[]
for part in(0,1):
    ap=ROOT/f'artifacts/a129v1_source_domain_{part}.pt'
    a=torch.load(ap,weights_only=True,mmap=True);assert a['checkpoint_sha256']==sha(cp)
    inputs=a['inputs'];answers=[]
    for row in inputs:
        first=sum(lookup[int(row[j])]*10**(3-j)for j in(1,2,3))
        second=sum(lookup[int(row[j])]*10**(7-j)for j in(5,6,7));answers.append(first+second)
    actual_readers={}
    for layer,key,source_positions,multiplier in((2,'native_b_stage',[7,6,5],9.),(3,'native_a_stage',[3,2,1],4.5)):
        F=c[key]['reader']['centered_readout'if layer==2 else'frame'].double()
        code=a['captures'][f'{layer}.raw'][:,8].double()@F
        digits=multiplier*code[:,:3]/code[:,3:]+4.5
        wanted=torch.tensor([[lookup[int(row[p])]for p in source_positions]for row in inputs])
        actual_readers[str(layer)]={'maximum_current_digit_error':(digits-wanted).abs().max(0).values.tolist(),
            'wrong_current_digits':int((digits.round()!=wanted).sum()),'minimum_reference':float(code[:,3].min())}
    wanted4=torch.tensor([[n%10,n//10%10,n//100%10,n//1000]for n in answers])
    source=a['captures']['7.input'][:,8].double()@reader['value_rows'].double()
    source_digits=9*source[:,1:]/source[:,:1]
    stage=c['native_copy_stage']['piece'];raw7=a['captures']['7.out'][:,8:13].double()
    copied=raw7[:,:,stage['data_coordinates']]/stage['expected_constant_scale']*stage['flag_scale']/raw7[:,:,stage['flag_coordinate']:stage['flag_coordinate']+1]
    phase=raw7[:,:,stage['phase_coordinates']]
    phase_value=phase[:,:,1]/phase[:,:,0]-1
    bit_value=phase[:,:,2]/phase[:,:,0]
    source_report={'maximum_computed_source_digit_error':(source_digits-wanted4).abs().max(0).values.tolist(),
        'wrong_computed_source_digits':int((source_digits.round()!=wanted4).sum()),
        'maximum_copied_digit_error':(copied-wanted4[:,None]).abs().amax((0,1)).tolist(),
        'wrong_copied_digits':int((copied.round()!=wanted4[:,None]).sum()),
        'maximum_added_copy_error':(copied-source_digits[:,None]).abs().amax((0,1)).tolist(),
        'maximum_phase_error_by_position':(phase_value-torch.arange(5)).abs().amax(0).tolist(),
        'maximum_leading_phase_bit_error':float((bit_value-wanted4[:,None,3]).abs().max())}
    output=[]
    for col,pos in enumerate(a['logit_positions']):
        k=pos-8;wanted=torch.tensor([107 if k>=len(str(n))else ids[int(str(n)[k])]for n in answers])
        output.append({'position':pos,'wrong_top1':int((a['topk_ids'][:,col,0]!=wanted).sum())})
    row={'part':part,'readers':actual_readers,'source_and_copy':source_report,'positions':output,
         'total_wrong_outputs':sum(x['wrong_top1']for x in output),'artifact_sha256':sha(ap)}
    results.append(row);print(json.dumps(row),flush=True)
atomic_json(ROOT/'artifacts/a129v1_source_domain_diagnosis.json',{
    'checkpoint_sha256':sha(cp),'results':results,'source_sha256':sha(__file__),
    'answers_used_only_for_diagnostics':True})
