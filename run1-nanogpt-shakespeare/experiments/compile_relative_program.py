"""Pure hand-programmed relative-position adder, an explicit stretch diagnostic.
Same tensors/config/vocabulary as host; all arithmetic weights are direct formulas.
This replaces host language weights and is NOT expected to preserve Shakespeare.
No model execution, arithmetic training, example fitting, or whole-problem table.
"""
import argparse,json,math,sys,torch
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'))
from common import vocab,atomic_save,ledger,sha,memory

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--natural',action='store_true');ap.add_argument('--out',required=True);args=ap.parse_args()
    torch.set_num_threads(6);memory()
    host=torch.load(R/'host/ckpt.pt',weights_only=False)
    s={n:torch.zeros_like(v,dtype=torch.double) for n,v in host['model'].items()}
    for n in s:
        if '.ln_' in n or '.ln_f.' in n:s[n].fill_(1.)
    chars=vocab(); D=192; zero=191
    cl=list(range(14)); pos=list(range(14,49)); pp=list(range(49,84)); ep=list(range(84,119)); flag=119
    stage=list(range(120,125)); sums=list(range(125,128)); digits=list(range(128,132)); const=132
    stage2=list(range(133,138)); lengths=list(range(138,142))
    assert len(set(cl+pos+pp+ep+[flag]+stage+sums+digits+[const]+stage2+lengths))==142
    def row(entries):
        w=torch.zeros(D,dtype=torch.double)
        for i,v in entries.items():w[i]+=float(v)
        w[zero]-=w.sum()
        return w
    C=row({i:1 for i in cl})
    def coordinate(i):return row({i:1})
    tok={c:(int(c) if c.isdigit() else 10 if c=='+' else 11 if c=='=' else 12 if c=='\n' else 13) for c in chars}
    for i,c in enumerate(chars):s['transformer.wte.weight'][i,tok[c]]=1
    s['lm_head.weight']=s['transformer.wte.weight']
    # 64 signed orthogonal position addresses in 32 dimensions. Any two source
    # positions at distance <=4 are not antipodal, so their sum retrieves exactly
    # those two positions with integer attention scores and a gap of at least32.
    had=torch.tensor([[(-1.)**((f&t).bit_count()) for t in range(64)] for f in range(32,64)],dtype=torch.double).T
    balance=math.sqrt(512/3);scale=math.sqrt(32/(32+1024/3))
    codes=[]
    for t in range(64):
        h=had[t];total=h.sum().item()
        extra=torch.tensor([-total/3]*3,dtype=torch.double) if total else torch.tensor([balance,-balance,0.],dtype=torch.double)
        code=torch.cat((h,extra))*scale
        assert abs(code.sum().item())<1e-12 and abs(code.square().sum().item()-32)<1e-10
        codes.append(code);s['transformer.wpe.weight'][t,pos]=code
    codes=torch.stack(codes)
    sigma0=math.sqrt(33/D-(1/D)**2+1e-5)
    sigma_source=math.sqrt(97/D-(1/D)**2+1e-5)
    sigma_query=math.sqrt(98/D-(2/D)**2+1e-5)
    def attention(layer,head,Q,K,V,outputs):
        W=s[f'transformer.h.{layer}.attn.c_attn.weight'];O=s[f'transformer.h.{layer}.attn.c_proj.weight']
        for j,w in Q.items():W[head*48+j]=w
        for j,w in K.items():W[192+head*48+j]=w
        for j,w in V.items():W[384+head*48+j]=w
        for j,dest in outputs.items():
            if isinstance(dest,int):O[dest,head*48+j]=1
            else:
                for i,value in dest.items():O[i,head*48+j]=value
    phi0=.5*C+sum((coordinate(i) for i in pos[:32]))/(64*scale)-(coordinate(pos[32])-coordinate(pos[33]))/(4*balance*scale)
    for h,operator,dest in [(0,10,pp),(1,11,ep)]:
        Q={0:sigma0*C};K={0:sigma0*math.sqrt(48)*(100*coordinate(operator)+50*phi0)}
        V={j:sigma0*coordinate(pos[j]) for j in range(35)};outs={j:dest[j] for j in range(35)}
        if h==1:V[35]=sigma0*coordinate(11);outs[35]=flag
        attention(0,h,Q,K,V,outs)
    used=[0]*4
    def neuron(layer,w,outs):
        n=used[layer];assert n<768
        s[f'transformer.h.{layer}.mlp.c_fc.weight'][n]=w
        for i,value in outs.items():s[f'transformer.h.{layer}.mlp.c_proj.weight'][i,n]+=float(value)
        used[layer]+=1
    def match(register,index):
        return sum((had[index,j]*coordinate(register[j]) for j in range(32)))/(32*scale)
    sharp=50.;activation=sharp*.5/sigma_query
    # Local address table, not a problem table: plus/equal positions and offsets.
    for p in range(1,63):
        for kb in (1,2,3):
            e=p+kb+1
            if e>=64:continue
            w=sharp*(match(pp,p)+match(ep,e)+2*coordinate(flag)-3.5*C)
            outs={}
            for slot,place in enumerate((2,1,0)):
                ia=p-1-place;ib=e-1-place
                if ia<0:ia=p
                if ib<=p:ib=p
                target=scale*(had[ia]+had[ib])
                reg=(pos,pp,ep)[slot]
                if slot==1:target-=codes[p,:32]
                if slot==2:target-=codes[e,:32]
                for j in range(32):outs[reg[j]]=target[j].item()/activation
            neuron(0,w,outs)
    # Relative output step from equality position and the current position.
    for e in range(3,64):
        for j in range(5):
            t=e+j
            if t>=64:continue
            w=sharp*(match(ep,e)+match(pos,t)+2*coordinate(flag)-3.5*C)
            outs={pos[d]:-codes[t,d].item()/activation for d in range(32)}
            outs[stage[j]]=1/activation
            neuron(0,w,outs)
    # Read three corresponding digit pairs. Before '=', all source keys/values
    # have the same LayerNorm denominator. After '=', a key mask excludes queries.
    digit_value=sum((d*coordinate(d) for d in range(10)))
    for h,reg in enumerate((pos,pp,ep)):
        Q={j:10*coordinate(reg[j])/scale for j in range(32)};Q[32]=10*C
        K={j:coordinate(pos[j])/scale for j in range(32)};K[32]=-128*coordinate(flag)
        V={0:2*sigma_source*digit_value}
        attention(1,h,Q,K,V,{0:sums[h]})
    # Carries, digit registers and a common homogeneous scale. GELU(x)-GELU(-x)
    # is exactly x; the two-ReLU ramp is a hand-written step on half-integers.
    gain=1000.;sharp2=100.;width=.25
    def linear(layer,w,outs):
        neuron(layer,sharp2*w,{i:v*gain/sharp2 for i,v in outs.items()})
        neuron(layer,-sharp2*w,{i:-v*gain/sharp2 for i,v in outs.items()})
    def step(w,outs):
        coefficient=gain/(2*sharp2*width)
        neuron(1,sharp2*(w+width*C),{i:v*coefficient for i,v in outs.items()})
        neuron(1,sharp2*(w-width*C),{i:-v*coefficient for i,v in outs.items()})
    sh,st,su=[coordinate(i) for i in sums];total=100*sh+10*st+su
    step(su-9.5*C,{digits[2]:1,digits[3]:-10})
    step(10*st+su-99.5*C,{digits[1]:1,digits[2]:-10})
    carry_h={digits[0]:1,digits[1]:-10}
    if args.natural:carry_h.update({lengths[2]:-1,lengths[3]:1})
    step(total-999.5*C,carry_h)
    if args.natural:
        step(total-9.5*C,{lengths[0]:-1,lengths[1]:1})
        step(total-99.5*C,{lengths[1]:-1,lengths[2]:1})
    for i,w in enumerate((sh,st,su)):linear(1,w,{digits[i+1]:1})
    linear(1,C,{const:1,lengths[0] if args.natural else lengths[3]:1})
    for a,b in zip(stage,stage2):linear(1,coordinate(a),{b:1})
    # Select the digit using relative step and result length. The inactive gates
    # are at least11 homogeneous units below zero; active values are integers.
    CC=coordinate(const);final_gain=1000.;bias=200.;gate=20.
    for length in range(1,5):
        for j in range(length):
            dindex=4-length+j
            w=sharp2*(coordinate(digits[dindex])-gate*(2*CC-coordinate(stage2[j])-coordinate(lengths[length-1])))
            neuron(2,w,{d:final_gain*2*d/sharp2 for d in range(10)})
    base={d:final_gain*(bias-d*d)/sharp2 for d in range(10)}
    base.update({i:-final_gain*bias/sharp2 for i in (10,11,12,13)})
    neuron(2,sharp2*CC,base)
    for length in range(1,5):
        w=sharp2*(coordinate(stage2[length])+coordinate(lengths[length-1])-1.5*CC)
        out={d:-final_gain*4*bias/sharp2 for d in range(10)};out[12]=final_gain*4*bias/sharp2
        neuron(2,w,out)
    # Block3 is the identity; all LayerNorm scales remain1. No bias/config changes.
    out={n:t.float() for n,t in s.items()};out['lm_head.weight']=out['transformer.wte.weight']
    assert all(torch.isfinite(x).all() for x in out.values())
    meta={'design':'Q-pure-relative','natural':args.natural,'scales':{'attention_operator_logit':100,'attention_anchor_logit':50,'decoder_sharpness':sharp,'pair_query_scale':10,'query_key_mask':128,'step_sharpness':sharp2,'step_half_width':width,'feature_gain':gain,'final_gain':final_gain,'output_bias':bias,'selection_gate':gate},'neurons_used':used,'registers':{'class':cl,'position':pos,'plus':pp,'equals':ep,'flag':flag,'stage':stage,'sums':sums,'digits':digits,'constant':const,'stage_copy':stage2,'length':lengths,'zero':zero},'sigmas':[sigma0,sigma_source,sigma_query],'expected_limitation':'Full arithmetic replacement; Shakespeare preservation is expected to fail.','weights_changed':{n:int((t!=host['model'][n]).sum()) for n,t in out.items() if not torch.equal(t,host['model'][n])}}
    atomic_save({'model':out,'model_args':host['model_args'],'implant':meta},args.out)
    with (R/args.out).with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2)
    ledger('compile',meta,R/args.out,'write');print(json.dumps({'output':args.out,'neurons_used':used,'sigmas':meta['sigmas']}))
if __name__=='__main__':main()
