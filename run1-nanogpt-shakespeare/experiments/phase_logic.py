"""Hand-written fixed-width arithmetic logic; no checkpoint/model imports.

Feature order: three phase digit averages, four equals masses, constant,
current digit. Address states are (equals position modulo four, output step).
"""
import math
import torch

def states():
    result=[]
    for q in range(4):
        for j in range(5):
            counts=[2]*4;counts[q]=1
            for k in range(j):counts[(q+k+1)%4]+=1
            den=torch.tensor([7+j+15*counts[f]+32*(16 if f==q else 1) for f in range(4)],dtype=torch.double)
            eq=torch.tensor([32*(16 if f==q else 1) for f in range(4)],dtype=torch.double)/den
            result.append({'q':q,'j':j,'denominator':den,'equals':eq})
    return result

def construct(tight=False):
    st=states();features=[];C=torch.zeros(9,dtype=torch.double);C[7]=1;current=torch.zeros(9,dtype=torch.double);current[8]=1
    def append(w,kind,coefficient,label):features.append({'w':w,'kind':kind,'coefficient':coefficient,'label':label})
    def bound(w):return 9*w[:3].abs().sum().item()+abs(w[7].item())+9*abs(w[8].item())
    def box(w,t):
        # Ten independent symbol values, not sampled addition examples. This
        # is a conservative interval proof even when prefix digits are wrong.
        phase=torch.zeros(4,10,dtype=torch.double)
        for k in range(3):phase[(t['q']+k+1)%4,k]=1;phase[(t['q']+k+1)%4,k+3]=1
        for k in range(t['j']):phase[(t['q']+k+1)%4,6+k]=1
        S=(phase.sum(0)[None]+15*phase[:3])/t['denominator'][:3,None]
        coef=w[:3]@S
        if t['j']:coef[5+t['j']]+=w[8]
        offset=float(w[3:7]@t['equals']+w[7]);return offset+9*float(coef.clamp_max(0).sum()),offset+9*float(coef.clamp_min(0).sum())
    for state_index,s in enumerate(st):
        q,j=s['q'],s['j']
        if j==4:continue
        ss=torch.zeros(4,9,dtype=torch.double)
        for k in range(3):ss[k,k]=s['denominator'][k]
        total=ss[q].clone() if q<3 else ss[:3].sum(0)/18
        phase=(ss-total)/15
        phase[3]=total-phase[:3].sum(0) if q<3 else torch.zeros(9,dtype=torch.double)
        ph,pt,pu=[phase[(q+k)%4] for k in (1,2,3)]
        weights=torch.ones(4,dtype=torch.double);weights[q]=4
        for k in range(1,j+1):weights[(q+k)%4]=-1
        scores=torch.tensor([float(weights@t['equals']) for t in st],dtype=torch.double);own=float(scores[state_index]);other=float(torch.cat((scores[:state_index],scores[state_index+1:])).max());gap=own-other
        assert gap>0,(q,j,gap)
        gate=torch.zeros(9,dtype=torch.double);gate[3:7]=weights;gate-=((own+other)/2)*C
        zero_gate=gate-gap/2*C
        def linear(w,coefficient,label):
            magnitude=2*(bound(w)+2)/gap
            if tight:
                magnitude=max(2.,(2-box(w,s)[0])/(gap/2),max((box(w,t)[1]+2)/(-float(gate[3:7]@t['equals']+gate[7])) for i,t in enumerate(st) if i!=state_index))
            append(w+magnitude*gate,'digit',coefficient,label+' positive')
            append(magnitude*gate,'digit',-coefficient,label+' cancellation')
        def step(w,coefficient,label):
            width=.25;magnitude=(bound(w)+width+2)/gap
            if tight:magnitude=max(2.,max((box(w,t)[1]+width+2)/(-float(zero_gate[3:7]@t['equals']+zero_gate[7])) for i,t in enumerate(st) if i!=state_index))
            append(w+width*C+magnitude*zero_gate,'digit',coefficient/(2*width),label+' upper')
            append(w-width*C+magnitude*zero_gate,'digit',-coefficient/(2*width),label+' lower')
        label=f'phase {q} stage {j}'
        if j==0:step(100*ph+10*pt+pu-999.5*C,1,label+' thousands carry')
        elif j==1:
            linear(ph-11*current,1,label+' hundreds base')
            step(10*pt+pu-99.5*C,1,label+' tens carry')
        elif j==2:
            linear(pt-current,1,label+' tens base')
            step(pu-9.5*C,1,label+' units carry')
            step(10*(pt-current)+pu-99.5*C,-10,label+' tens wrap')
        elif j==3:
            linear(pu-current,1,label+' units base')
            step(pu-current-9.5*C,-10,label+' units wrap')
    append(C,'constant',1,'digit parabola and positive digit bias')
    totals=[float(s['equals'].sum()) for s in st if s['q']==0];gap=totals[3]-totals[4];width=gap/4
    w=(totals[3]+totals[4])/2*C;w[3:7]-=1
    append(w+width*C,'newline',1/(2*width),'newline upper')
    append(w-width*C,'newline',-1/(2*width),'newline lower')
    return st,features

def standalone_test():
    # Entirely hand-built functions, with no checkpoint weights loaded.
    torch.set_num_threads(6);g=torch.Generator().manual_seed(91572);count=100000
    a=torch.randint(0,1000,(count,),generator=g);b=torch.randint(0,1000,(count,),generator=g);q=torch.randint(0,4,(count,),generator=g)
    st,features=construct();W=torch.stack([f['w'] for f in features]);coef=torch.tensor([f['coefficient'] for f in features],dtype=torch.double)
    digits=torch.stack([(a+b)//1000,(a+b)//100%10,(a+b)//10%10,(a+b)%10],dim=1)
    phase=torch.zeros(count,4,dtype=torch.double);rows=torch.arange(count)
    for k,place in enumerate((100,10,1),start=1):phase[rows,(q+k)%4]=(a//place%10+b//place%10).double()
    worst=0.;correct=0
    for j in range(5):
        if j:phase[rows,(q+j)%4]+=digits[:,j-1]
        den=torch.stack([st[int(qq)*5+j]['denominator'] for qq in q.tolist()]);eq=torch.stack([st[int(qq)*5+j]['equals'] for qq in q.tolist()])
        total=phase.sum(1);S=(total[:,None]+15*phase[:,:3])/den[:,:3]
        X=torch.cat((S,eq,torch.ones(count,1,dtype=torch.double),torch.zeros(count,1,dtype=torch.double) if j==0 else digits[:,j-1,None].double()),dim=1)
        yy=(X@W.T).clamp_min(0)*coef
        d=yy[:,[i for i,f in enumerate(features) if f['kind']=='digit']].sum(1);nl=yy[:,[i for i,f in enumerate(features) if f['kind']=='newline']].sum(1)
        if j<4:
            worst=max(worst,(d-digits[:,j]).abs().max().item());correct+=int(((d-digits[:,j]).abs()<1e-7).sum())
        else:assert (nl-1).abs().max()<1e-8
    print({'standalone_cases':count,'digit_checks_correct':correct,'digit_checks_total':4*count,'max_arithmetic_error':worst,'relu_features':len(features)})
    assert correct==4*count
if __name__=='__main__':standalone_test()
