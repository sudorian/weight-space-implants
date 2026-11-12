"""Package the final documented run with analysis metadata and score evidence.

This preparation script refuses a running report or unfinished final meter.
It includes sparse final deltas, never host weights or activation artifacts.
"""
import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import atomic_json, ledger_rows, sha

parser = argparse.ArgumentParser()
parser.add_argument('--candidate', help='Explicitly unpassed final sparse candidate, only when no rung is banked')
parser.add_argument('--candidate-score')
args = parser.parse_args()
assert (args.candidate is None) == (args.candidate_score is None)
report = (ROOT / 'REPORT.md').read_text()
assert report.splitlines()[0].endswith('— final report'), 'Finalize the report before packaging'
vp = ROOT / 'verification/results.json'
assert vp.exists(), 'Run the one-shot final verifier first'
verification = json.loads(vp.read_text())
rows, head = ledger_rows()
assert rows and rows[-1]['event'] != 'start', 'A meter is still unfinished'
assert verification['ledger_head_sha256'] == head, 'Verification must follow the final meter'
banks = sorted((ROOT / 'best').glob('H[1-5]/delta.pt'))
assert bool(banks) == (not verification['no_banked_rungs'])
members = {}

def add(path, name=None):
    path = Path(path)
    assert path.is_file() and not path.is_symlink(), path
    relative = path.resolve().relative_to(ROOT)
    name = name or relative.as_posix()
    assert name not in members and not name.startswith('/') and '..' not in Path(name).parts
    members[name] = path

for name in ('GOAL.md', 'REPORT.md', 'NOTES.md', 'ledger.jsonl'):
    add(ROOT / name)
for directory in ('tools', 'experiments', 'prior', 'artifacts', 'scores'):
    for path in sorted((ROOT / directory).rglob('*')):
        if not path.is_file() or path.is_symlink() or '__pycache__' in path.parts:
            continue
        if (directory == 'tools' or path.suffix in ('.py', '.sh', '.bash')
                or (path.suffix == '.json' and (directory in ('experiments', 'artifacts', 'scores') or 'manifest' in path.name.lower()))):
            add(path)
for path in sorted((ROOT / 'host').glob('*manifest*.json')):
    add(path)
for path in sorted((ROOT / 'verification').glob('*.json')):
    add(path)
for bank in banks:
    for path in sorted(bank.parent.iterdir()):
        if path.is_file() and path.suffix in ('.pt', '.json', '.md'):
            add(path)
candidate_note = None
if args.candidate:
    assert not banks, 'Use banked checkpoints when any rung has passed'
    candidate = (ROOT / args.candidate).resolve(); score_path = (ROOT / args.candidate_score).resolve()
    candidate.relative_to(ROOT); score_path.relative_to(ROOT)
    score = json.loads(score_path.read_text())
    assert score['checkpoint_sha256'] == sha(candidate)
    assert not any(score.get('passed', {}).values()), 'A passed candidate belongs in best/'
    add(candidate, 'candidate/delta.pt')
    add(candidate.with_suffix('.manifest.json'), 'candidate/delta.manifest.json')
    add(score_path, 'candidate/score.json')
    candidate_note = ('# Unpassed candidate\n\nNo ladder rung was banked. This sparse delta is included for independent retesting, '
        'not as a passed H1 checkpoint. The exact arithmetic, text and noise results are in score.json and REPORT.md.\n\n'
        f'Original path: {candidate.relative_to(ROOT).as_posix()}\n\nSHA-256: {sha(candidate)}\n\n'
        'The frozen Gemma host checkpoint and activation artifacts are deliberately absent. Apply the sparse delta '
        'to the host identified in the manifest using the supplied frozen tools.\n')

assert sum(p.stat().st_size for p in members.values()) < 290_000_000, 'Uncompressed safety allowance exceeded'
payload = {name: {'bytes': path.stat().st_size, 'sha256': sha(path)} for name, path in members.items()}
if candidate_note is not None:
    content = candidate_note.encode()
    payload['candidate/README.md'] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
metadata = {'files': payload, 'ledger_head_sha256': head,
    'banked_rungs': [p.parent.name for p in banks], 'unpassed_candidate_included': args.candidate is not None,
    'no_host_weights_or_activation_artifacts': True,
    'analysis_JSON_metadata_and_official_aggregate_scores_included': True}
dest = ROOT / 'export/run3.zip'; dest.parent.mkdir(exist_ok=True)
assert not dest.exists(), 'Never overwrite a final archive'
temporary = dest.with_suffix('.zip.partial')
assert not temporary.exists()
with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for name, path in sorted(members.items()): archive.write(path, name)
    if candidate_note is not None: archive.writestr('candidate/README.md', candidate_note)
    archive.writestr('PACKAGE_MANIFEST.json', json.dumps(metadata, indent=2, sort_keys=True) + '\n')
assert temporary.stat().st_size < 300_000_000
with zipfile.ZipFile(temporary) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(payload) + 1
os.replace(temporary, dest)
metadata.update(archive=str(dest.relative_to(ROOT)), archive_bytes=dest.stat().st_size, archive_sha256=sha(dest))
atomic_json(ROOT / 'export/package_manifest.json', metadata)
print(json.dumps({k: v for k, v in metadata.items() if k != 'files'}))
