"""Replace only three phase Up functions using current-position identities.

The common gates, digit fields, output columns and native-text repair remain
bit-identical. The new phase functions come from actual pre-amplifier captures
of the base checkpoint. No answer targets or iterative repair is used.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from circuit_io_v1 import expand, host_state, save
from tools.common import atomic_torch

p = argparse.ArgumentParser()
p.add_argument('--base', required=True)
p.add_argument('--phase', required=True)
p.add_argument('--capture-prefix', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
h = host_state()
base, phase_path = ROOT / a.base, ROOT / a.phase
ch = expand(h, base)
c = torch.load(base.parent / 'circuit.pt', weights_only=True)
stage = c['native_copy_stage']
old = stage['piece']
phase = torch.load(phase_path, weights_only=True)
uv = phase['raw_functions']
units = stage['units']
gain = old['phase_up_gain']
prefix = 'model.layers.7.'
gp = 1 + h[prefix + 'pre_feedforward_layernorm.weight'].double()
inv = torch.where(gp.abs() > .01, 1 / gp, torch.zeros_like(gp))
name = prefix + 'mlp.up_proj.weight'
U = ch[name].clone()
U[units[5]] = (gain * uv[:, 0] * inv).float()
U[units[6]] = (gain * uv[:, 1] * inv).float()
U[units[7]] = U[units[5]]
changed_rows = torch.zeros(2048, dtype=torch.bool)
changed_rows[units[5:8]] = True
assert torch.equal(U[~changed_rows], ch[name][~changed_rows])
ch[name] = U
report = {'changed_up_units': units[5:8].tolist(), 'phase_gain': gain,
          'phase_gain_unchanged': True, 'native_repair_unchanged': True,
          'gate_and_output_weights_unchanged': True,
          'target': 'Current position only, with independent random continuations',
          'phase_report': phase['report']}
c = copy.deepcopy(c)
c['phase_transfer'] = report
c['native_copy_stage']['phase_raw_functions'] = uv
c['native_copy_stage']['piece']['phase_analysis'] = phase['report']
dst = ROOT / a.output
dst.parent.mkdir(parents=True, exist_ok=True)
atomic_torch(dst.parent / 'circuit.pt', c)
bm = json.loads(base.with_suffix('.manifest.json').read_text())
save(dst, h, ch,
     sources=[Path(__file__), ROOT / 'experiments/derive_a29_phase_ratio_v1.py']
              + [ROOT / s for s in bm['sources']],
     inputs=[base, base.with_suffix('.manifest.json'), base.parent / 'circuit.pt', phase_path]
             + [ROOT / f'artifacts/{a.capture_prefix}_{s}.pt'
                for s in ('syntax_a', 'syntax_stress', 'calib256')],
     arguments=vars(a), description=__doc__, extra=report,
     categories=bm['categories'])
