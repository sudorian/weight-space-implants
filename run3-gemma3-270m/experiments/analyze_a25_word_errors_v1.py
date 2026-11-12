"""Decompose a native-key query construction on recorded calibration inputs."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state,expand
from tools.common import atomic_json
h=host_state();ch=expand(h,ROOT/'experiments/a25v1_native_keys2/delta.pt')
r=json.loads((ROOT/'experiments/a25v1_native_keys2/routing.json').read_text());idx=r['selected_channels'];pre='model.layers.2.self_attn.'
T=torch.load(ROOT/'artifacts/a1_detail_calib256.pt',weights_only=True,mmap=True)['captures']['2.input'].double().reshape(-1,640)
qrow=torch.load(ROOT/'experiments/a2v7_reader/circuit.pt',weights_only=True)['query'].double()
Q0=h[pre+'q_proj.weight'].double().reshape(4,256,640);Q=ch[pre+'q_proj.weight'].double().reshape(4,256,640)
gp=1+ch[pre+'q_norm.weight'].double();g0=1+h[pre+'q_norm.weight'].double()
orth=Q0-(Q0@qrow)[:,:,None]*qrow[None,None]/qrow.square().sum()
c=(Q*orth).sum(-1)/(Q0*orth).sum(-1).clamp_min(1e-30)
native=Q0*c[:,:,None];program=Q-native
Y=torch.einsum('nc,hjc->nhj',T,Q);Y0=torch.einsum('nc,hjc->nhj',T,Q0)
YN=torch.einsum('nc,hjc->nhj',T,native);YP=Y-YN
sigma=(Y.square().mean(-1,keepdim=True)+1e-6).sqrt();s0=(Y0.square().mean(-1,keepdim=True)+1e-6).sqrt()
DN=YN/sigma*gp-Y0/s0*g0;DP=YP/sigma*gp
report={'native_error_rms':float(DN.square().mean().sqrt()),'program_error_rms':float(DP.square().mean().sqrt()),
 'total_error_rms':float((DN+DP).square().mean().sqrt()),'native_error_per_head':DN.square().mean((0,2)).sqrt().tolist(),
 'program_error_per_head':DP.square().mean((0,2)).sqrt().tolist(),
 'native_query_output_rms':float((Y0/s0*g0).square().mean().sqrt()),
 'relative_query_normalizer_change_rms':float((sigma/s0-1).square().mean().sqrt()),
 'relative_query_normalizer_change_range':[float((sigma/s0).min()),float((sigma/s0).max())],
 'marker_query_word_rms':float((T@qrow).square().mean().sqrt())}
print(json.dumps(report,indent=2));atomic_json(ROOT/'artifacts/a25v1_word_error_decomposition.json',report)
