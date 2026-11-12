"""Run three sequential official calibration observations and summarize them.

The diagnostic restorations are not arithmetic candidates. All model execution
is delegated to the frozen forward meter; this runner performs none itself.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / '.venv/bin/python')


def run(label, *arguments):
    log = ROOT / f'artifacts/a119v1_{label}.log'
    assert not log.exists(), f'Keep existing log: {log}'
    print(f'Start {label}', flush=True)
    with log.open('w') as stream:
        subprocess.run([PYTHON, *arguments], cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    print(f'Finished {label}', flush=True)


base = 'experiments/a117v1_six_unit_carry/delta.pt'
run('current_word', 'tools/forward.py', '--delta', base,
    '--request', 'experiments/a119v1_current_word_request.json',
    '--output', 'artifacts/a119v1_current_word.pt')
for name, family, layers in (
        ('native_attention', 'attention', [2, 3, 7]),
        ('native_mlps', 'mlp', list(range(9)) + [17])):
    delta = f'experiments/a119v1_{name}/delta.pt'
    run(name + '_build', 'experiments/compile_a30_component_ablation_v2.py',
        '--base', base, '--restore', family, '--layers', *map(str, layers),
        '--output', delta)
    run(name + '_word', 'tools/forward.py', '--delta', delta,
        '--request', f'experiments/a119v1_{name}_word_request.json',
        '--output', f'artifacts/a119v1_{name}_word.pt')
run('analysis', 'experiments/analyze_a119_text_ablations_v1.py')
