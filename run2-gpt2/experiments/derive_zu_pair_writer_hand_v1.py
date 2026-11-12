"""Constant-radius digit and column-facet codes for a four-head reader.

Five fixed format facets identify hundreds and units. Their signed bits have
constant energy on every numeric source, even when a facet rejects it. A
separate all-numeric head will recover the tens column by subtraction.
"""
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from zc_native_inputs_v1 import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];d=load(host);poly=torch.load(ROOT/'artifacts/zr_column_polytopes_v1.pt',weights_only=True);classes=torch.load(ROOT/'artifacts/zs_scoped_digit_classes_v1.pt',weights_only=True);roles=torch.load(ROOT/'artifacts/zm_hard_roles_v1.pt',weights_only=True);old=torch.load(ROOT/'experiments/zd1_projected_reader/circuit.pt',weights_only=True)['writer'];one=torch.zeros(769,dtype=torch.float64);one[-1]=1;ids=torch.load(ROOT/'experiments/v1_protected_equals_writer/circuit.pt',weights_only=True)['coordinate_geometry']['previous']['allowed_token_ids'];lookup={int(t):i%10 for i,t in enumerate(ids[:20])};P=d['program'];W=d['native'];B=d['bad'];numpos=[0,1,2,4,5,6];nativeclass=(W.reshape(-1,769)@torch.stack([c['row'] for c in classes['chosen']]).T>0).any(1);snapstates=torch.cat((P[:,numpos].reshape(-1,769),W.reshape(-1,769)[nativeclass]));facets=[];snaps=[]
for col in (0,2):
 for f in poly['polytope_selections'][col]['facets']:
  row=f['row'].clone();values=snapstates@row;middle=values[(values>-.0125)&(values<.0125)].sort().values;edges=torch.cat((torch.tensor([-.0125],dtype=torch.float64),middle,torch.tensor([.0125],dtype=torch.float64)));j=int((edges[1:]-edges[:-1]).argmax());shift=float((edges[j]+edges[j+1])/2);row[-1]-=shift;snaps.append({'column':col,'threshold_shift':shift,'empty_interval_width':float(edges[j+1]-edges[j]),'minimum_numeric_absolute_gate':float((snapstates@row).abs().min()),'valid_column_minimum':float((P[:,[col,col+4]].reshape(-1,769)@row).min())});facets.append(row)
assert len(facets)==5;facets=torch.stack(facets);reference_bit=(P@facets.T>0).double();digit=torch.tensor([[lookup.get(int(t),-1) for t in line] for line in d['inputs']]);numeric=(digit>=0);truth=torch.stack((torch.where(numeric,digit.double()-4.5,0),torch.where(numeric,(20.25-(digit.double()-4.5).square()).clamp_min(0).sqrt(),0),numeric.double()),2);reports=[];chosen=None
for T in (4096.,8192.,16384.,32768.,65536.):
 rows=[];C=torch.zeros(10,34,dtype=torch.float64)
 for i,row in enumerate(facets):
  j=len(rows);rows.extend((T*row+8*one,T*row-8*one));C[i,j]=1/16;C[i,j+1]=-1/16
 row=roles['chosen'][3]['row'];j=len(rows);rows.extend((128*row+8*one,128*row-8*one));C[8,j]=1/16;C[8,j+1]=-1/16
 for value in range(10):
  j=len(rows);row=classes['chosen'][value]['row'];rows.extend((512*row+8*one,512*row-8*one));v=torch.tensor([-.5]*5+[value-4.5,(20.25-(value-4.5)**2)**.5,1.,0.,0.],dtype=torch.float64);C[:,j]=v/16;C[:,j+1]=-v/16
 rows.extend((old['program_rows'][-2],old['program_rows'][-1]));C[9,-2]=1/16;C[9,-1]=-1/16;rows=torch.stack(rows)
 def emit(X):return torch.nn.functional.gelu(torch.nn.functional.linear(X[...,:768].float(),rows[:,:768].float(),rows[:,768].float())).double()@C.T
 fields=emit(P);word=emit(W);bad=emit(B);bits=fields[...,:5]+.5*fields[...,7,None];biterror=float((bits-reference_bit)[numeric].abs().max());classerror=float((fields[:,:,[5,6,7]]-truth).abs().max());norm=fields[numeric,5].square()+fields[numeric,6].square()+fields[numeric,7].square()+.25*fields[numeric,:5].square().sum(1);hcount=bits[:,:,0];ucount=bits[:,:,1:5].sum(2);rep={'facet_temperature':T,'class_temperature':512,'plus_temperature':128,'maximum_numeric_bit_error':biterror,'maximum_digit_radius_presence_error':classerror,'numeric_scaled_radius_squared_range':[float(norm.min()),float(norm.max())],'expected_numeric_scaled_radius_squared':21.5625,'hundreds_valid_min':float(hcount[:,[0,4]].min()),'hundreds_other_max':float(hcount[:,[1,2,5,6]].max()),'units_valid_min':float(ucount[:,[2,6]].min()),'units_other_max':float(ucount[:,[0,1,4,5]].max()),'native_numeric_count':int((word[...,7]>.01).sum()),'native_any_source_field_count':int((word[...,:9].abs().max(-1).values>.01).sum()),'native_plus_count':int((word[...,8]>.5).sum()),'native_query_count':int((word[...,9]>.5).sum()),'maximum_row':float(rows.abs().max()),'maximum_corrupt_numeric_presence':float(bad[torch.arange(192),d['bad_roles'],7].abs().max())};rep['feasible']=biterror<.0005 and classerror<.0005 and rep['units_other_max']<3.01 and rep['hundreds_other_max']<.01 and rep['maximum_row']<=1e6 and rep['native_query_count']==0;reports.append(rep);print(json.dumps(rep),flush=True)
 if rep['feasible'] and chosen is None:chosen={'rows':rows,'coefficients':C,'report':rep}
assert chosen,reports
indices=[1,2,3,4,5,7,0,6,11,18];frame=old['frame'][:,indices];out={'chosen':chosen,'frame':frame,'facets':facets,'snaps':snaps,'frame_source_indices':indices,'field_names':['hundreds_facet','units_facet0','units_facet1','units_facet2','units_facet3','ordinal_d_minus4p5','radius_padding','numeric_presence','plus_presence','equals_query'],'numeric_gain_family':[64,128,256,512,1024],'facet_gain_ratio':.5,'plus_gain':1024.,'query_gain':16384.,'facet_temperature_family':[4096,8192,16384,32768,65536],'class_temperature':512.,'threshold_snap_interval':[-.0125,.0125],'word_source_count':chosen['report']['native_any_source_field_count'],'heads':[6,4,7,2]};torch.save(out,ROOT/'artifacts/zu_pair_writer_hand_v1.pt');(ROOT/'artifacts/zu_pair_writer_hand_v1.json').write_text(json.dumps({'chosen':chosen['report'],'trials':reports,'threshold_snaps':snaps},indent=2)+'\n')
