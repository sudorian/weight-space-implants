"""Reconcile descriptive edit categories without changing immutable manifests."""
import argparse, copy, hashlib, json
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(6); torch.set_grad_enabled(False)
parser = argparse.ArgumentParser()
parser.add_argument('--delta', required=True)
parser.add_argument('--update-report', action='store_true')
args = parser.parse_args()
path = ROOT / args.delta
assert path.is_relative_to(ROOT) and path.is_file()
manifest = json.loads(path.with_suffix('.manifest.json').read_text())
delta = torch.load(path, weights_only=True)
categories = copy.deepcopy(manifest['categories'])
corrections = []
for name, update in delta['updates'].items():
    count = len(update['indices'])
    assert manifest['per_tensor'][name] == count
    old = categories.get(name, {})
    if name.startswith('model.layers.0.') and not old:
        if name.endswith('mlp.down_proj.weight'):
            programme = int((update['indices'].long().remainder(2048) == 1159).sum())
            new = {'arithmetic_programme': programme, 'native_text_repair': count - programme}
        else:
            new = {'arithmetic_marker': count}
        categories[name] = new
        corrections.append({'tensor': name, 'original_categories': old, 'corrected_categories': new,
                            'reason': 'Initial marker-only builder supplied no categories'})
    if name == 'model.layers.3.self_attn.q_proj.weight' and sum(old.values()) != count:
        metadata = torch.load(ROOT / 'experiments/a45v2_query_energy/circuit.pt', weights_only=True)
        selected = torch.tensor(metadata['native_key_query_layer3']['selected_channels'])
        channels = update['indices'].long().div(640, rounding_mode='floor').remainder(256)
        programme = int(torch.isin(channels, selected).sum())
        new = {'text_repair': count - programme, 'circuit_and_native_accommodation': programme}
        categories[name] = new
        corrections.append({'tensor': name, 'original_categories': old, 'corrected_categories': new,
                            'reason': 'Count actual changed indices; one historical nominal entry equals HOST'})
    assert sum(categories[name].values()) == count, (name, count, categories[name])

repair_keys = {'native_text_repair', 'text_repair', 'native_text_repair_with_arithmetic_frame_constraint'}
rows = []
for name in sorted(manifest['per_tensor'], key=lambda n: (int(n.split('.')[2]), n)):
    total = manifest['per_tensor'][name]
    repair = sum(value for label, value in categories[name].items() if label in repair_keys)
    rows.append({'tensor': name, 'changed_scalars': total,
                 'arithmetic_and_circuit_accommodation': total - repair, 'native_only_text_repair': repair})
assert sum(row['changed_scalars'] for row in rows) == manifest['changed_scalars']
report = {'checkpoint': str(path.relative_to(ROOT)), 'delta_sha256': manifest['delta_sha256'],
    'changed_scalars': manifest['changed_scalars'], 'max_abs_edited_value': manifest['max_abs_edited_value'],
    'arithmetic_and_circuit_accommodation': sum(row['arithmetic_and_circuit_accommodation'] for row in rows),
    'native_only_text_repair': sum(row['native_only_text_repair'] for row in rows),
    'category_interpretation': 'Inherited purpose labels; circuit/native accommodation weights count with arithmetic because both functions share the same scalars. These are bookkeeping categories, not causal allocations.',
    'immutable_manifest_unchanged': True, 'category_corrections': corrections, 'per_tensor': rows}
out = ROOT / 'artifacts' / (path.parent.name + '_edit_category_audit.json')
out.write_text(json.dumps(report, indent=2) + '\n')
if args.update_report:
    lines = ['<!-- EDIT_AUDIT_START -->', '', '### Current leader: reconciled per-tensor edit counts', '',
        f'Checkpoint `{path.relative_to(ROOT)}`, SHA-256 `{manifest["delta_sha256"]}`. '
        f'Total **{report["changed_scalars"]:,}**; arithmetic and circuit accommodation **{report["arithmetic_and_circuit_accommodation"]:,}**; '
        f'native-only text repair **{report["native_only_text_repair"]:,}**. Maximum absolute edited value **{report["max_abs_edited_value"]:g}**.', '',
        report['category_interpretation'], '',
        'The immutable manifest omitted categories for the four initial layer0 tensors (44,863 edits) and used a nominal layer3 query count one larger than the actual changed-index count. This separate audit corrects those descriptions; weights, manifests, meters and ledger are unchanged.', '',
        '| Tensor | Arithmetic / accommodation | Native-only repair | Total |', '|---|---:|---:|---:|']
    for row in rows:
        lines.append(f'| `{row["tensor"]}` | {row["arithmetic_and_circuit_accommodation"]:,} | {row["native_only_text_repair"]:,} | {row["changed_scalars"]:,} |')
    lines += ['', f'Full audit: `{out.relative_to(ROOT)}`.', '', '<!-- EDIT_AUDIT_END -->']
    body = (ROOT / 'REPORT.md').read_text(); block = '\n'.join(lines)
    start, end = '<!-- EDIT_AUDIT_START -->', '<!-- EDIT_AUDIT_END -->'
    if start in body:
        lo = body.index(start); hi = body.index(end, lo) + len(end)
        body = body[:lo] + block + body[hi:]
    else: body += '\n\n' + block + '\n'
    (ROOT / 'REPORT.md').write_text(body)
print(json.dumps({key: report[key] for key in ('checkpoint', 'changed_scalars', 'arithmetic_and_circuit_accommodation', 'native_only_text_repair')}), flush=True)
