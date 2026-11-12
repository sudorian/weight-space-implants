"""One literal-only norm-anchor table in the existing five-function span.

Table entries depend on the eleven literal symbols and fixed geometry,
never on whole problems or their answers. Recorded inputs only check it.
"""
import json
import resource
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import host_state
from tools.common import atomic_json,atomic_torch,sha
h=host_state();gp=ROOT/'artifacts/a280v1_hard_subspace_literal_cleanup.pt'
obj=torch.load(gp,weights_only=True);parent=obj['report']
for path,digest in parent['sources'].items():assert sha(ROOT/path)==digest
for path,digest in parent['inputs'].items():assert sha(ROOT/path)==digest
assert list(obj['geometries'])==['94'];geom=obj['geometries']['94']
cp=ROOT/parent['base_checkpoint'];c=torch.load(cp.parent/'circuit.pt',weights_only=True)
fp=ROOT/'artifacts/a124v1_syntax_stress.pt';f=torch.load(fp,weights_only=True,mmap=True)
before=f['captures']['17.raw'][:,8:13].double().reshape(-1,640)
truth=f['topk_ids'][:,:,0].reshape(-1).long();nt=96*5;anchor=439
Y=before*(c['piece']['source_flag_scale']/before[:,c['piece']['source_flag_coordinate']])[:,None]
digit=Y[:,c['piece']['source_digit_coordinate']]/c['piece']['source_digit_scale']
def phi(d):return torch.stack([torch.ones_like(d),d,(d+.5).clamp_min(0),(d-4).clamp_min(0),(d-5).clamp_min(0)],1)
def rms(x):return (x.square().mean(-1,keepdim=True)+1e-6).sqrt()
gain=geom['gain'].double();basis=geom['basis'].double();M=geom['identity_map'].double()
amplitude=geom['report']['amplitude_certificate']['sufficient_emission_amplitude']
emission=amplitude*(phi(digit)@basis.T)
compressed=Y@M;raw=(emission-compressed)/gain
free=torch.arange(640)[torch.arange(640)!=anchor]
room=640-raw[:,free].square().sum(1);assert room.min()>0
mean_room=room[:nt].mean();assert abs(float(mean_room-geom['anchor_mean_energy']))<1e-9
mean_emission_energy=(emission[:nt,free]/gain[free]).square().sum(1).mean()
literal=torch.arange(-1,10,dtype=torch.float64)
table_energy=(amplitude*(phi(literal)@basis.T)[:,free]/gain[free]).square().sum(1)
table_room=mean_room+mean_emission_energy-table_energy;assert table_room.min()>0
table_anchor=table_room.sqrt();coefficients=torch.linalg.lstsq(phi(literal),table_anchor).solution
table_fit=phi(literal)@coefficients
candidate=phi(digit)@coefficients
E=h['model.embed_tokens.weight'];fg=1+h['model.norm.weight'].double()
def evaluate(value):
    nr=raw.clone();nr[:,anchor]=value
    state=before+nr/rms(nr)*gain;z=(state/rms(state)*fg).float()
    own=(z*E[truth]).sum(1);other=torch.full_like(own,-torch.inf)
    for start in range(0,len(E),4096):
        logits=z@E[start:start+4096].T;local=truth-start
        rows=((local>=0)&(local<logits.shape[1])).nonzero().flatten();logits[rows,local[rows]]=-torch.inf
        other=torch.maximum(other,logits.amax(1))
    margin=own-other
    return {'matches':int((margin>0).sum()),'positions':len(truth),'minimum_target_margin':float(margin.min()),
            'holdback_matches':int((margin[nt:]>0).sum()),'holdback_positions':len(truth)-nt,
            'minimum_holdback_target_margin':float(margin[nt:].min()),
            'maximum_relative_raw_norm_error':float((rms(nr)-1).abs().max()),'minimum_anchor_value':float(value.min())}
a0=mean_room.sqrt()
controls={'constant':evaluate(a0.expand_as(room)),'quadratic_Taylor':evaluate(a0+(room-mean_room)/(2*a0))}
for key in controls:
    ref=geom['report']['anchor_modes'][key]
    assert controls[key]['matches']==ref['matches']
    assert abs(controls[key]['minimum_target_margin']-ref['minimum_target_margin'])<1e-5
q=evaluate(candidate)
eligible=q['matches']==2560 and q['minimum_target_margin']>1 and q['minimum_anchor_value']>0
report={'analysis_only':True,'base_checkpoint':parent['base_checkpoint'],'geometry_key':'94',
    'literal_values':literal.tolist(),'literal_energy':table_energy.tolist(),'literal_target_anchor':table_anchor.tolist(),
    'literal_fitted_anchor':table_fit.tolist(),'literal_maximum_approximation_error':float((table_fit-table_anchor).abs().max()),
    'anchor_coefficients':coefficients.tolist(),'fixed_mean_anchor_room':float(mean_room),
    'fixed_mean_observed_emission_energy':float(mean_emission_energy),'controls':controls,'candidate':q,'eligible':eligible,
    'fit':'One full-rank least-squares projection of the explicit eleven-literal energy table onto five hand functions; no ridge or scale trials',
    'limitations':'Geometry only; native arithmetic contributions omitted. No actual Gate/Up/native Down reconstruction, checkpoint, model execution, or answer fit.',
    'sources':{**parent['sources'],str(Path(__file__).relative_to(ROOT)):sha(__file__)},
    'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (gp,cp,cp.parent/'circuit.pt',fp)},
    'maximum_resident_set_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
out=ROOT/'artifacts/a281v1_literal_energy_anchor.json';atomic_json(out,report)
if eligible:atomic_torch(out.with_suffix('.pt'),{'geometry':geom,'anchor_coefficients':coefficients,'report':report})
print(json.dumps(report),flush=True)
