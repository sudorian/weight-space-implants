"""Archive the selected, fully scored G3 improvement without changing old banks."""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not (ROOT / 'verification/started.json').exists()
    dest = ROOT / 'best/G3-robust'
    stage = ROOT / 'best/.G3-robust-staging'
    assert not dest.exists() and not stage.exists()
    before = {str(p.relative_to(ROOT)): sha(p)
              for p in (ROOT / 'best').rglob('*') if p.is_file()}
    source = ROOT / 'experiments/g3r5_wide_source'
    score_path = source / 'score_full_G3.json'
    manifest_path = source / 'delta.manifest.json'
    score = json.loads(score_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    digest = sha(source / 'delta.pt')
    assert digest == score['checkpoint_sha256'] == manifest['delta_sha256']
    assert score['mode'] == 'full' and not score['fresh'] and score['passed']['G3']
    for path, expected in manifest['sources'].items():
        assert sha(ROOT / path) == expected, path
    for path, expected in manifest['inputs'].items():
        assert sha(ROOT / path) == expected, path
    stage.mkdir()
    for src, name in [(source / 'delta.pt', 'delta.pt'),
                      (score_path, 'score.json'), (manifest_path, 'manifest.json')]:
        shutil.copyfile(src, stage / name)
        assert sha(src) == sha(stage / name)
    (stage / 'DESCRIPTION.md').write_text('''# G3 robust follow-up — standard full score passes

Source: `experiments/g3r5_wide_source/delta.pt`, full run `0839-score`.
This is a new bank; the original G1, G2 and G3 checkpoints are unchanged.

Clean G3: 34,977/35,000 (99.93428571%). Text agreement: 98.60687256%,
0.10687256 percentage points above the required 98.5%. KL: 0.0009258857;
cross-entropy increase: 0.0000724078 nats; all values finite.

Multiplicative edited-weight noise: 1,634/2,000 (81.7%) at sigma 1e-5;
0/2,000 at both 1e-4 and 1e-3. Original G3 scored 45/2,000 at 1e-5.
These are fixed-seed aggregate measurements, not a general noise guarantee.
Two consecutive questions: first 1,999/2,000, second 0/2,000, joint 0/2,000.
The construction is limited to a spaced three-digit question at sequence start.

It changes 4,252,159 of 85,054,464 eligible scalars (4.99933666%),
with maximum absolute weight 639,765.5. Embeddings, positions, the output
matrix, final norm and model code remain unchanged. See manifest and score
for complete per-tensor counts and immutable construction-source hashes.

The changes compact constant output clips, restore native MLP capacity with
one-shot covariance projections, recenter internal arithmetic coordinates
analytically, and replace five noisy common-field writes with one shared
syntax-gated write. The gate's wider transition reduces amplification.
No gradients, iterative loss fitting or whole-problem answer regression.

R5 was selected for preservation first and noise second. The later R6 digit
readout solved one additional clean case but lowered text agreement to
98.57101440% and 1e-5 noise accuracy to 73.15%; it remains in experiments.

Final fresh verification is stored separately in
`verification/G3-robust.json`; REPORT.md records its outcome. The frozen
verify.py covers the original exact G1/G2/G3 folder names, so this new bank
is evaluated by a separate invocation of the same frozen score.py --fresh.
''')
    archive = {
        'source': str(source.relative_to(ROOT)),
        'standard_run_id': score['run_id'],
        'status': 'standard_full_pass_pending_fresh_verification',
        'selection_order': ['text_preservation_margin', 'noise_accuracy'],
        'checkpoint_sha256': digest,
        'copied_files_sha256': {name: sha(stage / name)
                               for name in ['delta.pt', 'score.json', 'manifest.json']},
        'original_banks_unmodified': before,
        'fresh_result': 'verification/G3-robust.json',
    }
    (stage / 'archive.json').write_text(json.dumps(archive, indent=2) + '\n')
    os.rename(stage, dest)
    assert all(sha(ROOT / path) == value for path, value in before.items())
    frozen = {str(p.relative_to(ROOT)): sha(p)
              for p in (ROOT / 'best').rglob('*') if p.is_file()}
    snapshot = ROOT / 'artifacts/finalization_bank_freeze.json'
    assert not snapshot.exists()
    snapshot.write_text(json.dumps({'files': frozen, 'original_files': before}, indent=2) + '\n')
    print(json.dumps({'bank': str(dest.relative_to(ROOT)), 'sha256': digest,
                      'original_bank_files_unchanged': len(before),
                      'frozen_bank_files': len(frozen)}))


if __name__ == '__main__':
    main()
