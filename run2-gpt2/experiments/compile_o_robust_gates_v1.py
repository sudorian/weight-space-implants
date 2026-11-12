"""Tighten syntax gating and replace fragile native format-off directions.

All program neurons, output columns, native reconstructions and arithmetic
rules stay fixed. Only format thresholds and bound-derived off penalties move.
"""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from circuit_io_v1 import save
from prefix_capture_io_v1 import unpack
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--layers',type=int,nargs='+',default=[1,3,4]);a=p.parse_args();torch.set_num_threads(6);torch.set_grad_enabled(False)
host=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];base=ROOT/'experiments/o14_capacity/delta.pt';meta=json.loads(base.with_suffix('.manifest.json').read_text());changed={}
for name,u in torch.load(base,weights_only=True)['updates'].items():
    v=host[name].clone();v.view(-1)[u['indices'].long()]=u['values'];changed[name]=v
robust=torch.load(ROOT/'artifacts/o_robust_off_v2.pt',weights_only=True);forms=unpack(torch.load(ROOT/'artifacts/o_gate_domain_v1.pt',weights_only=True,mmap=True));reports=[];patched={};inputs=[base,ROOT/'artifacts/o_robust_off_v2.pt',ROOT/'artifacts/o_gate_domain_v1.pt']
def aug(x):
    x=x.double();return torch.cat((x,torch.ones(*x.shape[:-1],1,dtype=x.dtype)),-1)
def activation(X,rows):return torch.nn.functional.gelu(torch.nn.functional.linear(X[:,:768].float(),rows[:,:768].float(),rows[:,768].float())).double()
for layer,apath,gpath,tpath in ((1,'o4c_payload',None,'o3e_reader_repair'),(3,'o8b_amplifier','o7c_counter_phase_v1','o7c_counter'),(4,'o9_phases','o8b_amplified_phase_v1','o8b_amplifier')):
    if layer not in a.layers:continue
    auxpath=ROOT/f'experiments/{apath}/circuit.pt';aux=torch.load(auxpath,weights_only=True);oldrows=aux['program_rows'];rows=oldrows.clone();P=aug(forms['captures'][f'{layer}.ln2'][:,7:]).flatten(0,1);texts=[aug(torch.load(ROOT/f'artifacts/{tpath}_calib.pt',weights_only=True,mmap=True)['captures'][f'{layer}.ln2'].reshape(-1,768))]
    for i in range(4):
        path=ROOT/f'artifacts/o9_text_tail_{i:02d}.pt';x=torch.load(path,weights_only=True,mmap=True)['captures'][f'{layer}.ln2'];mask=robust[str(layer)]['clean_masks'][i] if layer in (3,4) else torch.ones(x.shape[:2],dtype=torch.bool);texts.append(aug(x[mask]));inputs.append(path)
    T=torch.cat(texts);details={};spec=aux.get('specification',[])
    if layer==1:
        gate=aux['gate'];r=torch.cat((gate['weight'],torch.tensor([gate['bias']],dtype=torch.float64)));lo=max(gate['bounds']['negative_max'],float((T@r).max()));hi=gate['bounds']['positive_range'][0];gap=hi-lo;assert gap>0,(lo,hi);gated=r.clone();gated[-1]-=(lo+hi)/2;reads=torch.cat((aux['numeric']['column_rows_centered'],aux['numeric']['reference_row'][:,None]),1);oldgap=aux['piece_report']['gate_gap'];penalty=aux['piece_report']['linear_gate_penalty']*oldgap/gap
        penalty=max(penalty,1.25*float((((T@reads).clamp_min(0).max(1).values+.5)/(-(T@gated))).max()))
        sharp=72/gap
        for i,fraction in enumerate((1/3,2/3)):
            rows[i]=sharp*r;rows[i,-1]-=sharp*(lo+fraction*gap)
        rows[2]=32*penalty*gated
        for i in range(4):rows[3+i]=32*(penalty*gated+reads[:,i])
        details={'gate_lo':lo,'gate_hi':hi,'gate_gap':gap,'linear_penalty':penalty,'sharpness':sharp}
        P=P.reshape(512,5,769)[:,0]
    else:
        gp=ROOT/f'artifacts/{gpath}.pt';geo=torch.load(gp,weights_only=True);inputs.append(gp);off=robust[str(layer)]['chosen']['row'];ot=T@off;assert ot.min()>0;S=geo['scalar_rows'];ref=S[:,3];count=geo['counter_row'];presence=geo['common_row'] if layer==3 else geo['presence_row'];penalties=[]
        if layer==3:
            cp=1.25*float(((2*(T@presence)+(T@S[:,:4]).clamp_min(0).max(1).values+.25)/ot).max());cb=2*presence-cp*off;rows[0]=64*cb
            for j in range(4):rows[1+j]=64*(cb+S[:,j])
            np=1.25*float(((2*(T@presence)+(T@count).clamp_min(0)+1)/ot).max());nb=2*presence-np*off;rows[5]=16*nb;rows[6]=16*(nb+count);penalties.extend([cp,np])
            for index,source,threshold in ((7,S[:,2],.5),(9,S[:,0],-.5),(11,S[:,0],.5)):
                pure=source-threshold*ref;penalty=2*max(0.,float(((T@pure+16/16384)/ot).max()));shared=16384*(pure-penalty*off)
                for j,sign in enumerate((1.,-1.)):rows[index+j]=shared;rows[index+j,-1]+=sign*8
                penalties.append(penalty)
        else:
            for j,boundary in enumerate((.75,5/12,7/24,.225)):
                index=2+2*j;threshold=spec[index]['threshold'];penalty=1.25*max(0.,float(((T@count-boundary-threshold+16/1024)/ot).max()));pure=count-penalty*off;penalties.append(penalty)
                for k in range(2):rows[index+k]=1024*pure;rows[index+k,-1]-=1024*(boundary+spec[index+k]['threshold'])
            cp=1.25*float(((2*(T@presence)+(T@S[:,:4]).clamp_min(0).max(1).values+.25)/ot).max());cb=2*presence-cp*off;rows[10]=64*cb
            for j in range(4):rows[11+j]=64*(cb+S[:,j])
            penalties.append(cp);common=geo['common0_row'];U=S[:,4];Heq=S[:,5];Hhigh=S[:,6]
            sources=[S[:,1]-.5*ref-.75*U,S[:,1]+.5*ref-.75*(common-U),Hhigh-.5*common,S[:,1]-.5*ref-.75*(U+common-Heq),S[:,1]+.5*ref-.75*(2*common-U-Heq)]
            for j,source in enumerate(sources):
                index=15+2*j;temperature=spec[index]['temperature'];penalty=2*max(0.,float(((T@source+16/temperature)/ot).max()));shared=temperature*(source-penalty*off)
                for k,sign in enumerate((1.,-1.)):rows[index+k]=shared;rows[index+k,-1]+=sign*8
                penalties.append(penalty)
        details={'off':robust[str(layer)]['chosen']['report'],'penalties':penalties}
    maximum=float(rows.abs().max());assert maximum<=1e6,(layer,maximum)
    textpre=torch.nn.functional.linear(T[:,:768].float(),rows[:,:768].float(),rows[:,768].float());assert textpre.max()<-6,(layer,float(textpre.max()))
    coef=aux['emission_coefficients'];old=activation(P,oldrows)@coef.T;new=activation(P,rows)@coef.T;difference=(new-old).abs();fields=difference.max(0).values.tolist();binary_indices=([0] if layer==1 else [4,5,6] if layer==3 else list(range(5))+[9,10]);binary_changes=int(((old[:,binary_indices]>.5)!=(new[:,binary_indices]>.5)).sum());assert binary_changes==0,(layer,binary_changes)
    stem=f'transformer.h.{layer}.mlp.';selected=aux['repair']['selected'];changed[stem+'c_fc.weight'][selected]=rows[:,:768].float();changed[stem+'c_fc.bias'][selected]=rows[:,768].float();record={'layer':layer,'maximum_fc_value':maximum,'text_max_preactivation':float(textpre.max()),'piece_field_difference':fields,'binary_changes':binary_changes,**details};reports.append(record);patched[str(layer)]={'rows':rows,'selected':selected,'report':record};print(json.dumps(record),flush=True);inputs.extend([auxpath,ROOT/f'artifacts/{tpath}_calib.pt']);del T,P,texts,textpre
out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True);torch.save(patched,out.parent/'gates.pt');save(out,host,changed,sources=[ROOT/s for s in meta['sources']]+[Path(__file__),ROOT/'experiments/derive_o_robust_off_v2.py'],inputs=list(dict.fromkeys(inputs)),arguments=vars(a),description='O20: calibration-tail format thresholds and closed-form native off directions; arithmetic output columns and native restoration unchanged.',extra={'layers':reports,'hard_covariance_masses':[1,4,16,64,256]})
