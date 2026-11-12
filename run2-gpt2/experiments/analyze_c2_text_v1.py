from pathlib import Path
import torch,json
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(6);torch.set_grad_enabled(False)
sd=torch.load(ROOT/'host/ckpt.pt',weights_only=True,mmap=True)['model'];c=torch.load(ROOT/'experiments/c2_digit_frontend/circuit.pt',weights_only=True);t=torch.load(ROOT/'artifacts/c2_digits_calib.pt',weights_only=True,mmap=True)
x=t['captures']['2.ln2'].reshape(-1,768).double();z=(x-sd['transformer.h.2.ln_2.bias'].double())/sd['transformer.h.2.ln_2.weight'].double();ph=z@c['phase']/(768**.5);print('phase max',ph.max(0).values.tolist(),'active_count',int((ph.max(1).values>.5).sum()),flush=True)
first=torch.load(ROOT/'artifacts/c1_marker_calib.pt',weights_only=True,mmap=True)['captures']['2.ln1'].reshape(-1,768).double();tx=torch.cat((first,torch.ones(len(first),1,dtype=torch.float64)),1);tv=tx-tx.mean(0);C=tv.T@tv/len(tx)
print('decoder text rms',[(tx@d).square().mean(0).sqrt().max().item() for d in c['decoders']],flush=True)
