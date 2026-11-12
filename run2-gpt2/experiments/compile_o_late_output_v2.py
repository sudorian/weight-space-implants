"""O16: final-block hand-wired decimal emission from phases and Boolean carry registers.

Only conditional reference copies apply the carries. The unchanged frozen
embedding geometry supplies a parabolic digit argmax and a newline direction.
No whole-problem targets enter any solve or coefficient construction.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from circuit_restore_v5 import restore
from prefix_capture_io_v1 import unpack
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--output-gain',type=float,default=17179869184.);p.add_argument('--temperature',type=float,default=32768.);p.add_argument('--donors',type=int,default=816);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/o9_phases/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());old=torch.load(base.parent/'circuit.pt',weights_only=True);geo=torch.load(ROOT/'artifacts/o9_late_output_inputs_v2.pt',weights_only=True)
text=torch.load(ROOT/'artifacts/o9_late_calib.pt',weights_only=True,mmap=True);ht=torch.load(ROOT/'artifacts/o_host_calib11.pt',weights_only=True,mmap=True);assert torch.equal(text['inputs'],ht['inputs']);forms=[unpack(torch.load(ROOT/f'artifacts/o9_late_{label}.pt',weights_only=True,mmap=True)) for label in ('format_a','format_b','bad')]
parabola=torch.load(ROOT/'artifacts/parabola_readout_v4.pt',weights_only=True);u0,u1=parabola['constant'],parabola['linear'];newline=torch.load(ROOT/'artifacts/a5_output_geometry.pt',weights_only=True)['readout_vectors'][:,10]
def augment(x):
    x=x.double();return torch.cat((x,torch.ones(*x.shape[:-1],1,dtype=torch.float64)),-1)
X=torch.cat([augment(d['captures']['11.ln2'][:,7:]) for d in forms[:2]]);P=X.reshape(-1,769);T=augment(text['captures']['11.ln2'].reshape(-1,768));phase=geo['phase_rows'];common=geo['common_row'];S=geo['scalar_rows'];Ref=S[:,3];U=geo['unit_row'];Ten=S[:,4];Lead=S[:,5];off=geo['off']['row'];ot=T@off;assert ot.min()>0
rows=[];columns=[];numeric=[];spec=[]
def add(row,column,coefficient,label):
    rows.append(row);columns.append(column);numeric.append(coefficient);spec.append(label)
def gate(source,fields,label,margin=None):
    bounds=torch.stack([T@v for v in fields],1).clamp_min(0).max(1).values if fields else torch.zeros(len(T),dtype=torch.float64)
    margin=16/a.temperature if margin is None else margin;penalty=1.25*max(0.,float(((T@source+bounds+margin)/ot).max()));result=source-penalty*off;spec.append({'gate':label,'text_penalty':penalty,'margin':margin});return result
def scalar_pair(gating,field,vector,numeric_field,multiplier,label):
    coefficient=torch.zeros(2,dtype=torch.float64);coefficient[numeric_field]=multiplier/a.temperature
    add(a.temperature*(gating+field),a.output_gain*multiplier*vector/a.temperature,coefficient,{'kind':label,'sign':1})
    add(a.temperature*gating,-a.output_gain*multiplier*vector/a.temperature,-coefficient,{'kind':label,'sign':-1})
def phase_clip(stage,newline_output=False):
    source=gate(phase[:,stage]-.5*common,[],f'phase {stage} cancellation',.15);column=-old['phase_gain']*old['phases'][:,stage]
    if newline_output:column=column+(a.output_gain/64)*newline
    for sign in (1.,-1.):
        row=1024*source.clone();row[-1]+=sign*128
        add(row,sign*column/256,torch.zeros(2,dtype=torch.float64),{'kind':'phase cancellation and newline' if newline_output else 'phase cancellation','stage':stage,'sign':sign,'temperature':1024,'half_bias':128})
def conditional(stage,flag,multiplier,label):
    source=gate(phase[:,stage]+flag-1.5*common,[Ref],label);scalar_pair(source,Ref,u1,1,multiplier,label)
for stage in range(4):
    phase_clip(stage)
    if stage==0:
        source=gate(phase[:,stage]-.5*common,[Ref],'leading base');scalar_pair(source,Ref,u0,0,1,'leading base');conditional(stage,Lead,1,'leading one')
    else:
        D=S[:,stage-1]+4.5*Ref;source=gate(2*phase[:,stage]-1.5*common,[Ref,D],f'digit base {stage}');c=torch.zeros(2,dtype=torch.float64)
        add(a.temperature*source,-a.output_gain*(u0+5.5*u1)/a.temperature,torch.tensor([-1.,-5.5],dtype=torch.float64)/a.temperature,{'kind':'shared digit baseline','stage':stage})
        add(a.temperature*(source+Ref),a.output_gain*(u0+4.5*u1)/a.temperature,torch.tensor([1.,4.5],dtype=torch.float64)/a.temperature,{'kind':'digit constant','stage':stage})
        add(a.temperature*(source+D),a.output_gain*u1/a.temperature,torch.tensor([0.,1.],dtype=torch.float64)/a.temperature,{'kind':'digit scalar','stage':stage})
        if stage==1:conditional(stage,Ten,1,'hundreds increment');conditional(stage,Lead,-10,'hundreds wrap')
        elif stage==2:conditional(stage,U,1,'tens increment');conditional(stage,Ten,-10,'tens wrap')
        else:conditional(stage,U,-10,'units wrap')
phase_clip(4,True);rows=torch.stack(rows);columns=torch.stack(columns,1);numeric=torch.stack(numeric,1);assert len(rows)==33
maximum=max(float(rows.abs().max()),float(columns.abs().max()));assert maximum<=1e6,maximum
tp=torch.nn.functional.linear(T[:,:768].float(),rows[:,:768].float(),rows[:,768].float());assert tp.max()<-6,float(tp.max())
pre=torch.nn.functional.linear(P[:,:768].float(),rows[:,:768].float(),rows[:,768].float());acts=torch.nn.functional.gelu(pre);actual=(acts.double()@numeric.T).reshape(1024,5,2);v=X@S;cu=(X@U/(X@common)>.5).double();ct=(v[:,:,4]/(X@common)>.5).double();cl=(v[:,:,5]/(X@common)>.5).double();expected=torch.zeros_like(actual);expected[:,:4,0]=v[:,:4,3];expected[:,0,1]=cl[:,0]*v[:,0,3];expected[:,1,1]=v[:,1,0]+(9+ct[:,1]-10*cl[:,1])*v[:,1,3];expected[:,2,1]=v[:,2,1]+(9+cu[:,2]-10*ct[:,2])*v[:,2,3];expected[:,3,1]=v[:,3,2]+(9-10*cu[:,3])*v[:,3,3]
piece={'maximum_new_weight':maximum,'text_max_preactivation':float(tp.max()),'homogeneous_copy_max_error':(actual-expected).abs().reshape(-1,2).max(0).values.tolist(),'digit_ratio_error_by_stage':[((actual[:,j,1]/actual[:,j,0])-(expected[:,j,1]/expected[:,j,0])).abs().max().item() for j in range(4)]};print('piece',json.dumps(piece),flush=True);assert max(piece['digit_ratio_error_by_stage'])<.02,piece
new,repair=restore(host,11,text,ht['captures']['11.raw'],ht['captures']['11.gelu'],rows,columns,donor_count=a.donors);changed={}
for name,u in torch.load(base,weights_only=True)['updates'].items():
    v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
changed.update(new);out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save({'previous':old,'repair':repair,'rows':rows,'columns':columns,'numeric_coefficients':numeric,'output_gain':a.output_gain,'temperature':a.temperature,'specification':spec,'piece_report':piece},out.parent/'circuit.pt')
save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__),ROOT/'experiments/circuit_restore_v5.py',ROOT/'experiments/analyze_o_late_output_inputs_v2.py',ROOT/'experiments/analyze_parabola_readout_v4.py',ROOT/'experiments/analyze_a5_off_and_readout.py'],inputs=[base,base.parent/'circuit.pt',ROOT/'artifacts/o9_late_output_inputs_v2.pt',ROOT/'artifacts/o9_late_calib.pt',ROOT/'artifacts/o_host_calib11.pt',ROOT/'artifacts/parabola_readout_v4.pt',ROOT/'artifacts/a5_output_geometry.pt']+[ROOT/f'artifacts/o9_late_{label}.pt' for label in ('format_a','format_b','bad')],arguments=vars(a),description='O16 final-block G4 circuit: conditional reference copies implement decimal carry subtraction; fixed frozen-unembedding parabolas emit each digit. Phase carriers are cancelled.',extra={'program_units':len(rows),'piece':piece,'text_repair':repair['text_repair_report'],'ridge_trials':repair['ridge_trials'],'declared_gain_temperature_pairs':[[2**32,8192],[2**33,16384],[2**34,32768],[2**35,65536],[2**36,131072]],'newline_gain_fraction':1/64})
