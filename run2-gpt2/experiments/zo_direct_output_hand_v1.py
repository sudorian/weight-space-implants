"""Thirty-GELU direct decimal output from homogeneous counter coordinates.

Coordinates: unit, q, signed units/tens/leading carry counts, centered
hundreds/tens/units columns, and an ordinary-text off feature. All decimal
coefficients are symbolic identities; there is no answer fit.
"""
import torch

def build(geometry,u0,u1,newline,negative=None,output_gain=2**26,temperature=16384.,halfwidth=64.,copy_scale=16.):
    unit,q,cu,ct,cl,dh,dt,du,off=geometry.T
    rows=[];coef=[];groups=[];names=[];baselines={}
    def add(row,values,group,name):
        rows.append(row);coef.append(torch.tensor(values,dtype=torch.float64));groups.append(group);names.append(name);return len(rows)-1
    def clip(field,b,values,group):
        gate=temperature*(field-b*unit)
        add(gate+halfwidth*unit,[v/(2*halfwidth) for v in values],group,'positive homogeneous clip')
        idx=add(gate-halfwidth*unit,[-v/(2*halfwidth) for v in values],group,'negative homogeneous clip')
        baselines[group]=idx
    clip(q,.225,[1,9,0],'q225');clip(q,.75,[0,-9,0],'q750')
    def baseline(b,group):
        if group not in baselines:baselines[group]=add(temperature*(q-b*unit)-halfwidth*unit,[0,0,0],group,'shared copy baseline')
    def copy(b,field,sign,group):
        baseline(b,group);idx=baselines[group];coef[idx][1]-=sign/copy_scale;add(rows[idx]+copy_scale*field,[0,sign/copy_scale,0],group,'window scalar copy')
    copy(.225,du,1,'q225');copy(.75,du,-1,'q750')
    copy(5/12,dh-du,1,'q416');copy(.75,dh-du,-1,'q750')
    copy(7/24,dt-du,1,'q291');copy(5/12,dt-du,-1,'q416')
    for name,field,steps in [('units',cu,[(.225,-10),(7/24,11),(5/12,-1)]),('tens',ct,[(7/24,-10),(5/12,11),(.75,-1)]),('leading',cl,[(5/12,-10),(.75,11)])]:
        for j,(b,value) in enumerate(steps):clip(field,b,[0,value,0],f'{name}{j}')
    clip(-q,-.225,[0,0,1],'newline')
    rows=torch.stack(rows);coeff=torch.stack(coef,1);assert len(rows)==30,len(rows)
    penalties={}
    if negative is not None:
        ov=negative@off;assert ov.min()>0,float(ov.min())
        for group in dict.fromkeys(groups):
            indices=[i for i,g in enumerate(groups) if g==group];value=negative@rows[indices].T;penalty=1.25*max(0.,float(((value+16)/ov[:,None]).max()))
            rows[indices]-=penalty*off[None,:];penalties[group]=penalty
    columns=output_gain*(u0[:,None]*coeff[0]+u1[:,None]*coeff[1]+newline[:,None]*coeff[2]/64)
    return {'rows':rows,'columns':columns,'numeric_coefficients':coeff,'output_gain':output_gain,'temperature':temperature,'halfwidth':halfwidth,'copy_scale':copy_scale,'groups':groups,'names':names,'off_penalties':penalties}
