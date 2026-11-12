"""Exact affine reparameterization with analytically chosen noise coordinates.

For LN output x, change beta to beta-c and all following FC biases to b+Wc.
The real-valued function is unchanged. Minimize the closed-form sum of
multiplicative-noise preactivation variances on the hand-program rows,
including the NEW noise on beta. This is one algebraic linear solve using
only weight values and mean recorded inputs, not any output/answer objective.
"""
import torch
import torch.nn.functional as F

def recenter(host,changed,layer,selected,programme_raw,semantic_temperatures):
 prefix=f'transformer.h.{layer}';W=changed.get(prefix+'.mlp.c_fc.weight',host[prefix+'.mlp.c_fc.weight']);bias=changed.get(prefix+'.mlp.c_fc.bias',host[prefix+'.mlp.c_fc.bias']);gamma=changed.get(prefix+'.ln_2.weight',host[prefix+'.ln_2.weight']);beta=changed.get(prefix+'.ln_2.bias',host[prefix+'.ln_2.bias']);raw=programme_raw[:,7:12].contiguous();x=F.layer_norm(raw,(768,),gamma,beta,1e-5)
 temperatures=torch.tensor(semantic_temperatures,dtype=torch.float64);rows=W[selected].double()/temperatures[:,None];bb=bias[selected].double()/temperatures
 mu=x.reshape(-1,768).double().mean(0);d=rows.square().sum(0).clamp_min(1e-16);di=(2*d).reciprocal();z=(mu+beta.double())/2
 # Woodbury solution to (2diag(d)+R'R)c = d(mu+beta)-R'b.
 small=torch.eye(len(selected),dtype=torch.float64)+(rows*di)@rows.T
 center=z-di*(rows.T@torch.linalg.solve(small,bb+rows@z))
 newbeta=(beta.double()-center).float();actual_center=beta.double()-newbeta.double();newbias=(bias.double()+W.double()@actual_center).float()
 newx=F.layer_norm(raw,(768,),gamma,newbeta,1e-5);oldpre=F.linear(x,W[selected],bias[selected]);newpre=F.linear(newx,W[selected],newbias[selected])
 old_variance=x.double().square()@rows.square().T+bb.square()
 newbb=newbias[selected].double()/temperatures
 new_variance=newx.double().square()@rows.square().T+newbb.square()+(rows.square()@newbeta.double().square())
 stats={'layer':layer,'selected_rows':selected.tolist(),'noise_std_sigma_1e5_before':(1e-5*old_variance.mean((0,1)).sqrt()).tolist(),'noise_std_sigma_1e5_after_including_edited_beta':(1e-5*new_variance.mean((0,1)).sqrt()).tolist(),'mean_variance_ratio':float(new_variance.mean()/old_variance.mean()),'max_program_preactivation_difference_float32':float((newpre-oldpre).abs().max()),'max_program_difference_in_semantic_units':float(((newpre-oldpre).double()/temperatures).abs().max()),'maximum_new_beta':float(newbeta.abs().max()),'maximum_new_bias':float(newbias.abs().max()),'real_arithmetic_identity':'LN_beta -> beta-c; FC_bias -> bias+FC_weight*c for all3072rows','center_choice':'Unique minimum of analytic summed noise variance on temperature-normalized hand rows; no model outputs or arithmetic labels.'}
 changed[prefix+'.ln_2.bias']=newbeta;changed[prefix+'.mlp.c_fc.bias']=newbias
 return {'center':actual_center,'new_beta':newbeta,'new_bias':newbias,'statistics':stats}
