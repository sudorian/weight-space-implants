"""Synthetic check of a carry switch made from two GLUs and a raw anchor.

This executes only a hand-built scalar circuit, never Gemma. Integer column
sums 0..19 are the allowed per-column truth table. No fitted answer weights.
"""
import json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json
torch.set_num_threads(6);torch.set_grad_enabled(False)
s,e,f,off=torch.meshgrid(torch.arange(20,dtype=torch.float64),torch.linspace(-.45,.45,19,dtype=torch.float64),
 torch.tensor([20.,25.,30.],dtype=torch.float64),torch.tensor([-.5,0.,.5],dtype=torch.float64),indexing='ij')
s,e,f,off=[x.reshape(-1) for x in (s,e,f,off)]
truth=(s>=10).double();reports=[]
for gain in (2.,4.,8.,16.,32.):
    delta=(s+e-9.5)*f
    positive=F.gelu(gain*delta-off,approximate='tanh')*f
    negative=F.gelu(-gain*delta-off,approximate='tanh')*f
    anchor=(positive+negative)*1e4
    # Allow a conservative unrelated native raw RMS of 100 in 639 coordinates.
    raw_rms=((anchor.square()+639*100**2)/640+1e-6).sqrt()
    output=positive*1e4/raw_rms/(640**.5)
    ideal_ratio=positive/(positive+negative)
    report={'gain':gain,'cases':len(s),'max_ideal_bit_error':float((ideal_ratio-truth).abs().max()),
      'max_bit_error_with_native_background':float((output-truth).abs().max()),
      'min_raw_anchor':float(anchor.min()),'wrong_carry':int(((output>=.5)!=(truth>=.5)).sum())}
    reports.append(report);print(json.dumps(report),flush=True)
passed=[r for r in reports if r['max_bit_error_with_native_background']<1e-4 and r['wrong_carry']==0]
report={'trials':reports,'chosen':passed[0] if passed else None,
 'mechanism':'The two gates select opposite signs. Their raw anchor uses the sum; the carry uses only the positive branch. When exactly one branch is active, the shared feature cancels in the normalization ratio. Linear copied fields can use the same two gates, preserving their ratios without subtraction of large positive gate outputs.'}
atomic_json(ROOT/'artifacts/normalized_carry_v1.json',report)
