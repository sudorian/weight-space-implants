"""Prepare exact-input native8 and current-format observations for recoding.

No model executes here. The current-token output representation will reserve
digit value -1 for newline, eliminating the separate newline selector pair.
"""
import argparse
import copy
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, sha
p = argparse.ArgumentParser(); p.add_argument('--base', required=True); a = p.parse_args()
base = ROOT / a.base
word = json.loads((ROOT / 'experiments/a119v1_native_attention_word_request.json').read_text())
word['capture'] = {k: v for k, v in word['capture'].items() if not k.startswith('8.')}
word['capture'].update({
    '8.features': {'module': 'model.layers.8.mlp.down_proj', 'kind': 'input'},
    '8.pre': {'module': 'model.layers.8.pre_feedforward_layernorm', 'kind': 'output'},
    '8.raw': {'module': 'model.layers.8.pre_feedforward_layernorm', 'kind': 'input'}})
fmt = json.loads((ROOT / 'experiments/a72v2_syntax_stress_request.json').read_text())
fmt['capture'].update({
    '8.features': {'module': 'model.layers.8.mlp.down_proj', 'kind': 'input'},
    '8.down': {'module': 'model.layers.8.mlp.down_proj', 'kind': 'output'},
    '8.pre': {'module': 'model.layers.8.pre_feedforward_layernorm', 'kind': 'output'},
    '8.raw': {'module': 'model.layers.8.pre_feedforward_layernorm', 'kind': 'input'},
    '7.pre': {'module': 'model.layers.7.pre_feedforward_layernorm', 'kind': 'output'},
    '7.raw': {'module': 'model.layers.7.pre_feedforward_layernorm', 'kind': 'input'},
    '6.out': {'module': 'model.layers.6', 'kind': 'output'}})
atomic_json(ROOT / 'experiments/a122v1_native8_word_request.json', word)
atomic_json(ROOT / 'experiments/a122v1_format_request.json', fmt)
atomic_json(ROOT / 'experiments/a122v1_base_reference.json', {
    'base': a.base, 'sha256': sha(base),
    'purpose': 'Nine-feature digit/negative-one selector with a five-feature literal final emitter.'})
