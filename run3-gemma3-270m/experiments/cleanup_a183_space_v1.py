"""Retire three superseded A1 marker-only format captures, preserving provenance."""
import datetime, hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
names = ['artifacts/a1_syntax_a.pt', 'artifacts/a1_detail_syntax_a.pt',
         'artifacts/a1_detail_syntax_b.pt']
ledger = {}
for line in (ROOT / 'ledger.jsonl').open():
    row = json.loads(line)
    artifact = row.get('result', {}).get('artifact')
    if artifact in names:
        ledger[artifact] = row
assert set(ledger) == set(names)
records = []
for name in names:
    path = ROOT / name
    assert path.is_file() and path.parent == ROOT / 'artifacts'
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    row = ledger[name]
    assert digest.hexdigest() == row['result']['artifact_sha256']
    records.append({'path': name, 'bytes': path.stat().st_size, 'sha256': digest.hexdigest(),
                    'run_id': row['run_id'], 'checkpoint_sha256': row['checkpoint_sha256'],
                    'capture_shapes': row['result'].get('capture_shapes', {})})
record = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reason': 'A1 is the initial marker-only checkpoint with zero H1 arithmetic. These three old format captures are superseded by later circuit and native-feature captures. Keep original HOST data, all sparse checkpoints, scripts, requests, manifests, aggregate scores and ledger.',
    'files': records, 'bytes_reclaimed': sum(r['bytes'] for r in records)}
output = ROOT / 'artifacts/cleanup_a183_space_v1.json'
assert not output.exists()
output.write_text(json.dumps(record, indent=2) + '\n')
for name in names:
    (ROOT / name).unlink()
print(json.dumps({'record': str(output.relative_to(ROOT)), 'bytes_reclaimed': record['bytes_reclaimed']}), flush=True)
