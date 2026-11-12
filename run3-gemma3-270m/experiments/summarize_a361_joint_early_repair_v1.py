"""Summarize completed A361 component checks without running any model."""
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

rows = []; inputs = {}
for layer in range(1, 9):
    path = ROOT / f'artifacts/a361v1_expanded_joint_early_up_l{layer}.json'
    log = path.with_suffix('.log')
    result = json.loads(path.read_text())
    assert result['layer'] == layer and len(result['trials']) == 625
    baseline, best = result['baseline'], result['best_diagnostic']
    wall = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)', log.read_text())
    assert wall is not None and 'Exit status: 0' in log.read_text()
    rows.append({'layer': layer, 'trials': len(result['trials']),
        'eligible_word_trials': sum(r['eligible'] for r in result['trials']),
        'baseline_state_RMS': baseline['validation_paired_HOST_state_relative_RMS'],
        'best_state_RMS': best['validation_paired_HOST_state_relative_RMS'],
        'state_RMS_improvement_fraction': 1 - best['validation_paired_HOST_state_relative_RMS'] / baseline['validation_paired_HOST_state_relative_RMS'],
        'baseline_direction_RMS': baseline['validation_paired_HOST_direction_RMS'],
        'best_direction_RMS': best['validation_paired_HOST_direction_RMS'],
        'best_parameters': {k: best[k] for k in ('native_unit_count', 'canonical_input_rank', 'ridge', 'strength')},
        'factor_control_maximum_raw_relative_error': result['factor_control']['maximum_raw_relative_RMS_reconstruction_error'],
        'quadratic_probe': result['quadratic_probe'],
        'format_stage_reached': result['format_validation'] is not None,
        'selected': result['selected'] is not None,
        'maximum_resident_set_kib': result['maximum_resident_set_kib'],
        'elapsed_wall_time': wall.group(1)})
    inputs[str(path.relative_to(ROOT))] = digest(path)
    inputs[str(log.relative_to(ROOT))] = digest(log)

report = {'analysis_summary_only': True, 'records': rows,
    'total_independent_candidates': sum(r['trials'] for r in rows),
    'selected_layers': [r['layer'] for r in rows if r['selected']],
    'required_state_and_direction_improvement_fraction': .05,
    'maximum_process_bytes': max(r['maximum_resident_set_kib'] for r in rows) * 1024,
    'scope': 'Local independent original-target repairs on recorded inputs. No whole-model or held-out preservation claims; no multi-layer composition.',
    'sources': {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}, 'inputs': inputs}
out = ROOT / 'artifacts/a361v1_all_layers_summary.json'
assert not out.exists()
temp = out.with_suffix('.json.tmp')
temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
os.replace(temp, out)
print(json.dumps(report, indent=2))
