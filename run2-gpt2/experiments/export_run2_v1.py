"""Package final evidence and local model parts without running inference."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'export'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def disk_bytes():
    result = subprocess.run(['du','-sk',str(ROOT)],check=True,capture_output=True,text=True)
    return int(result.stdout.split()[0]) * 1024


def main():
    audit = json.loads((ROOT / 'verification/final_audit.json').read_text())
    assert audit['remaining_budget'] == {'forward':0,'quick':0,'full':0}
    assert audit['all_bank_files_unchanged_since_freeze']
    assert '# GPT-2 direct weight programming — final report' in (ROOT / 'REPORT.md').read_text()
    DEST.mkdir(exist_ok=True)
    archive = DEST / 'run2.zip'
    temp = DEST / 'run2.zip.tmp'
    assert not archive.exists() and not temp.exists()
    assert not list(DEST.glob('gpt2-small.safetensors.part-*'))
    assert not (DEST / 'MANIFEST.json').exists()
    readme = '''# Offline Run 2 export

`run2.zip` is the evidence/checkpoint package. It contains the final REPORT,
GOAL, NOTES, append-only ledger, unmodified model and frozen tools, all local
Python/shell scripts and JSON manifests, JSON trial/score evidence, final
verification, all five bank folders, and the local GPT-2 tokenizer files.
The new selected bank is `best/G3-robust/`; the original G1/G2/G3 banks remain
unchanged. `best/G4-best-attempt/` is explicitly a FAILED attempt.

The ZIP contains no activation tensors, host checkpoint, calibration text,
held-out text or token caches. Those original text fixtures remain only in
the working project. Retired activation captures have hash journals; their
derived JSON findings and immutable scripts remain available. Rebuilding
every historical experiment from this compact package alone would require
the corresponding captures/fixtures, which are deliberately not included.

Reassemble the original local safetensors in a separate destination:

```sh
cat gpt2-small.safetensors.part-aa gpt2-small.safetensors.part-ab > model.safetensors
shasum -a 256 model.safetensors
```

Expected SHA-256:
`248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`.
The original is 548,105,171 bytes. Parts were made with `split -b 300m`;
the maximum part size is 300 MiB (314,572,800 bytes). MANIFEST.json records
the individual part hashes, the reconstructed stream hash, ZIP hash, ZIP
member hashes and successful CRC verification.

`experiments/prepare_host.py` documents the one-time offline float32 import,
including GPT-2 matrix transposes, ignored mask buffers and tied embeddings.
It refuses to overwrite an existing host. Sparse delta semantics and all
edit constraints are implemented by the unchanged `tools/common.py`.
Each bank's score and manifest identify its exact delta SHA-256.

The official ledger is complete: 300 forward, 100 quick and 40 full units
have been consumed. The final verifier is one-shot and has already run.
The original frozen meters also require the original text/token fixtures
and exact host/source hashes. Preserve these records; do not reset the
ledger or modify the meters. Independent evaluation should retain this
completed evidence and use its own unseen inputs and text.

No files were uploaded, and no network access was used for this export.
'''
    (DEST / 'README.md').write_text(readme)
    include = {ROOT / p for p in ['GOAL.md','REPORT.md','NOTES.md','ledger.jsonl','model.py']}
    for directory in ['tools','experiments','best','verification','requests']:
        for path in (ROOT / directory).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            if directory == 'best' or path.suffix in {'.py','.sh','.json','.jsonl','.md'}:
                include.add(path)
    for path in (ROOT / 'artifacts').glob('*'):
        if path.is_file() and path.suffix in {'.json','.md'}:
            include.add(path)
    for path in (ROOT / 'host').glob('*manifest*.json'):
        include.add(path)
    include.update(ROOT / p for p in ['downloads/vocab.json','downloads/merges.txt'])
    include.add(DEST / 'README.md')
    # All scripts, including any auxiliary extension, are explicitly accounted for.
    for directory in ['tools','experiments']:
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and path.suffix in {'.py','.sh'}:
                assert path in include
    member_records = []
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(include):
            relative = path.relative_to(ROOT).as_posix()
            assert not relative.startswith(('heldout/','calib/'))
            assert relative != 'host/ckpt.pt'
            assert path.suffix != '.pt' or relative.startswith('best/')
            z.write(path, relative)
            member_records.append({'path':relative,'bytes':path.stat().st_size,'sha256':sha(path)})
    assert temp.stat().st_size < 300_000_000
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None, 'ZIP CRC verification failed'
        names = set(z.namelist())
        for required in ['REPORT.md','ledger.jsonl','tools/score.py','tools/verify.py',
                         'best/G3-robust/delta.pt','best/G4-best-attempt/delta.pt',
                         'verification/G3-robust.json','verification/final_audit.json']:
            assert required in names, required
    temp.rename(archive)
    assert disk_bytes() + (ROOT / 'downloads/model.safetensors').stat().st_size < 25_000_000_000
    subprocess.run(['split','-b','300m','downloads/model.safetensors',
                    'export/gpt2-small.safetensors.part-'],cwd=ROOT,check=True)
    parts = sorted(DEST.glob('gpt2-small.safetensors.part-*'))
    assert len(parts) == 2
    combined = hashlib.sha256()
    part_records = []
    total = 0
    for path in parts:
        assert path.stat().st_size <= 300 * 1024 * 1024
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block); combined.update(block); total += len(block)
        part_records.append({'path':path.name,'bytes':path.stat().st_size,'sha256':h.hexdigest()})
    original = ROOT / 'downloads/model.safetensors'
    expected = json.loads((ROOT / 'host/manifest.json').read_text())['source_sha256']
    assert combined.hexdigest() == sha(original) == expected
    assert total == original.stat().st_size == 548105171
    frozen = json.loads((ROOT / 'artifacts/finalization_bank_freeze.json').read_text())['files']
    assert all(sha(ROOT / path) == value for path,value in frozen.items())
    assert sha(ROOT / 'ledger.jsonl') == audit['ledger_file_sha256']
    used = disk_bytes()
    assert used < 25_000_000_000
    manifest = {
        'time':datetime.datetime.now().astimezone().isoformat(),
        'archive':{'path':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive),
                   'crc_verified':True,'members':member_records},
        'model_parts':part_records,
        'reassembled_model':{'bytes':total,'sha256':combined.hexdigest(),'stream_verified':True},
        'bank_files_unchanged':True,'ledger_unchanged':True,
        'directory_allocated_bytes_before_manifest':used,
        'no_activation_artifacts_in_zip':True,'no_heldout_or_calibration_text_in_zip':True,
    }
    (DEST / 'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert disk_bytes() < 25_000_000_000
    print(json.dumps({'archive_bytes':archive.stat().st_size,'members':len(member_records),
                     'model_parts':part_records,'folder_allocated_bytes':disk_bytes()},indent=2))


if __name__ == '__main__':
    main()
