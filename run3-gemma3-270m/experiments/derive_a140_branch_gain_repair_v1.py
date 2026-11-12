"""Direct independent native-branch gain fits to paired original HOST text.

Each coordinate uses one closed-form scalar least-squares solve on192new
unlabeled calibration windows. Programme output gains stay fixed. Strong
writer coordinates realize the fitted native-only gain by scaling only their
native Down coefficients. Edited attention readers2/3/7 stay fixed. Five
global shrinkage fractions are evaluated on64separate calibration windows.
No iterative fitting, arithmetic targets, or activation replay occurs.
"""
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json,atomic_torch,sha
h=host_state()
base=ROOT/'experiments/a123v1_prior_native_final/delta.pt'
wp=ROOT/'artifacts/a140v1_edited_branches.pt'
hp=ROOT/'artifacts/a140v1_host_branches.pt'
lp=ROOT/'artifacts/a140v1_branch_gain_layout.json'
w,host=[torch.load(p,weights_only=True,mmap=True)for p in(wp,hp)]
layout=json.loads(lp.read_text())
ch=expand(h,base)
assert w['checkpoint_sha256']==sha(base)
assert host['checkpoint_sha256']==sha(ROOT/'host/ckpt.pt')
assert torch.equal(w['inputs'][:256],host['inputs'])
quiet={key:float(value.abs().max())for key,value in w['captures'].items()
       if key.startswith('word.')and key.endswith('program_features')}
print(json.dumps({'word_programme_maxima':quiet}),flush=True)
assert all(value==0 for value in quiet.values())
nt=192*64
fractions=[0.,.25,.5,.75,1.]
totals={fraction:{'validation_relative_mse_sum':0.,'maximum_format_post_change':0.}for fraction in fractions}
solutions,branches={},[]
def norm(raw):
    return raw/(raw.square().mean(1,keepdim=True)+1e-6).sqrt()
for layer in range(18):
    for branch,module in(('attention','post_attention_layernorm'),('mlp','post_feedforward_layernorm')):
        if branch=='attention'and layer in(2,3,7):continue
        name=f'model.layers.{layer}.{module}.weight'
        gamma=1+ch.get(name,h[name]).double()
        native_gamma=1+h[name].double()
        raw=w['captures'][f'word.{layer}.{branch}.raw'].double().reshape(-1,640)
        hr=host['captures'][f'{layer}.{branch}.raw'].double().reshape(-1,640)
        z=norm(raw)
        target=norm(hr)*native_gamma
        ref=(target.square().mean(1,keepdim=True)+1e-6).sqrt()
        X,Y=z[:nt]/ref[:nt],target[:nt]/ref[:nt]
        denominator=X.square().sum(0)
        fitted=(X*Y).sum(0)/denominator.clamp_min(1e-30)
        fitted[denominator<1e-30]=gamma[denominator<1e-30]
        units=torch.tensor(layout['programme_units'].get(str(layer),[]),dtype=torch.long)
        fixed=torch.zeros(640,dtype=torch.bool)
        strong=torch.empty(0,dtype=torch.long)
        down_name=None
        if branch=='mlp'and len(units):
            down_name=f'model.layers.{layer}.mlp.down_proj.weight'
            down=ch[down_name].double()
            fixed=down[:,units].ne(0).any(1)
            strong=(fixed&(gamma.abs()>=1e5)).nonzero().flatten()
        gain_target=fitted.clone()
        gain_target[fixed]=gamma[fixed]
        ratio=fitted[strong]/gamma[strong]
        assert ratio.isfinite().all()and(ratio>0).all()and(ratio<2).all(),(name,ratio)
        fr=w['captures'][f'format.{layer}.{branch}.raw'].double().reshape(-1,640)
        native_fr=fr[:,strong].clone()
        if len(strong):
            pf=w['captures'][f'format.{layer}.program_features'].double().reshape(-1,len(units))
            native_fr-=pf@down[strong][:,units].T
        original_post=norm(fr)*gamma
        trials=[]
        for fraction in fractions:
            parameter=(gamma+fraction*(gain_target-gamma)-1).float()
            actual_gain=1+parameter.double()
            scales=1+fraction*(ratio-1)
            rr=raw[nt:].clone()
            rr[:,strong]*=scales
            post=norm(rr)*actual_gain
            error=(post-target[nt:])/ref[nt:]
            format_raw=fr.clone()
            format_raw[:,strong]+=native_fr*(scales-1)
            format_post=norm(format_raw)*actual_gain
            q={'fraction':fraction,'validation_relative_branch_rms':float(error.square().mean().sqrt()),
               'maximum_format_post_change':float((format_post-original_post).abs().max()),
               'maximum_norm_parameter':float(parameter.abs().max())}
            trials.append(q)
            totals[fraction]['validation_relative_mse_sum']+=q['validation_relative_branch_rms']**2
            totals[fraction]['maximum_format_post_change']=max(totals[fraction]['maximum_format_post_change'],q['maximum_format_post_change'])
            assert parameter.isfinite().all()and parameter.abs().max()<=1e6
        row={'layer':layer,'branch':branch,'fixed_programme_gain_rows':fixed.nonzero().flatten().tolist(),
             'strong_native_scale_rows':strong.tolist(),'strong_native_scale_ratios':ratio.tolist(),
             'trials':trials}
        branches.append(row)
        solutions[name]={'gain_target':gain_target,'base_gamma':gamma,
                         'down_name':down_name,'strong_rows':strong,'native_scale_ratios':ratio,
                         'programme_units':units}
        print(json.dumps(row),flush=True)
aggregate=[]
for fraction in fractions:
    row={'fraction':fraction,'validation_mean_relative_branch_rms':(totals[fraction]['validation_relative_mse_sum']/len(branches))**.5,
         'maximum_format_post_change':totals[fraction]['maximum_format_post_change']}
    row['eligible']=row['maximum_format_post_change']<100
    aggregate.append(row)
selected=min((r for r in aggregate if r['eligible']),key=lambda r:r['validation_mean_relative_branch_rms'])
baseline=aggregate[0]
top1_matches=int((w['topk_ids'][:256,:,0]==host['topk_ids'][:,:,0]).sum())
top1_count=host['topk_ids'][:,:,0].numel()
report={'branch_count':len(branches),'branch_gain_scope':'All MLPs and native attention branches, excluding attention2/3/7',
        'training_windows':192,'validation_windows':64,'calibration_seed':layout['calibration_seed'],
        'shrinkage_fraction_trials':fractions,'programme_gains_fixed':True,
        'strong_native_Down_rows_only_scaled':True,'branches':branches,'aggregate_trials':aggregate,
        'selected_by_local_validation':selected,'minimum_relative_local_improvement_fraction':.02,
        'new_calibration_last_position_baseline':{'matches':top1_matches,'tokens':top1_count,
            'logit_positions':host['logit_positions'],'agreement':top1_matches/top1_count},
        'conditional_format_check':'Exact local branch change at recorded A123 inputs; composed downstream inputs still require official scoring',
        'target':'Original HOST branch at its paired original HOST input, never a sum or a previous fitted residual',
        'no_answers_gradients_activation_replay_or_iterative_fit':True,
        'inputs':{str(p.relative_to(ROOT)):sha(p)for p in(base,wp,hp,lp)},'source_sha256':sha(__file__)}
atomic_json(ROOT/'artifacts/a140v1_branch_gain_repair.json',report)
print(json.dumps({'aggregate_trials':aggregate,'selected':selected,'baseline_text':report['new_calibration_last_position_baseline']}),flush=True)
assert selected['validation_mean_relative_branch_rms']<.98*baseline['validation_mean_relative_branch_rms'],'Less than two percent aggregate local branch improvement'
atomic_torch(ROOT/'artifacts/a140v1_branch_gain_solution.pt',{'solutions':solutions,'report':report})
