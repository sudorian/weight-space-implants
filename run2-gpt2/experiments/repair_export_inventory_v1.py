"""Complete the export's nested manifest inventory; never touch model weights."""
import datetime
import json
from pathlib import Path
import subprocess
import zipfile

from export_run2_v1 import ROOT, DEST, sha, disk_bytes


def inventory():
    args = ['rg','--files','--hidden','--no-ignore','-g','*.py','-g','*.sh',
            '-g','*manifest*.json','-g','!heldout/**','-g','!calib/**',
            '-g','!export/**','-g','!**/__pycache__/**']
    result = subprocess.run(args,cwd=ROOT,check=True,capture_output=True,text=True)
    return set(result.stdout.splitlines())


def main():
    archive = DEST / 'run2.zip'
    manifest_path = DEST / 'MANIFEST.json'
    previous = json.loads(manifest_path.read_text())
    assert sha(archive) == previous['archive']['sha256']
    journal = ROOT / 'artifacts/export_initial_manifest.json'
    assert not journal.exists()
    journal.write_text(json.dumps({'reason':'Initial export omitted one nested activation-capture metadata manifest; all model banks and scores were included. Rebuild adds the complete recursive script/manifest inventory. No checkpoint changes.',
                                  'initial_export_manifest':previous},indent=2)+'\n')
    with zipfile.ZipFile(archive) as z:
        original_names = set(z.namelist())
    required = inventory()
    names = original_names | required | {str(journal.relative_to(ROOT))}
    temp = DEST / 'run2.complete.zip.tmp'
    assert not temp.exists()
    assert disk_bytes() + archive.stat().st_size + 5_000_000 < 25_000_000_000
    records = []
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(names):
            assert not name.startswith(('heldout/','calib/'))
            assert not name.endswith('.pt') or name.startswith('best/')
            path = ROOT / name
            assert path.is_file(),name
            z.write(path,name)
            records.append({'path':name,'bytes':path.stat().st_size,'sha256':sha(path)})
    assert temp.stat().st_size < 300_000_000
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist()))
        assert required <= set(z.namelist())
    assert disk_bytes() < 25_000_000_000
    temp.replace(archive)
    frozen = json.loads((ROOT / 'artifacts/finalization_bank_freeze.json').read_text())['files']
    assert all(sha(ROOT / p) == h for p,h in frozen.items())
    audit = json.loads((ROOT / 'verification/final_audit.json').read_text())
    assert sha(ROOT / 'ledger.jsonl') == audit['ledger_file_sha256']
    for part in previous['model_parts']:
        assert sha(DEST / part['path']) == part['sha256']
    previous['time'] = datetime.datetime.now().astimezone().isoformat()
    previous['archive'] = {'path':archive.name,'bytes':archive.stat().st_size,
                           'sha256':sha(archive),'crc_verified':True,'members':records}
    previous['directory_allocated_bytes_before_manifest'] = disk_bytes()
    previous['complete_script_manifest_inventory'] = {
        'count':len(required),'all_included':True,
        'added_paths':sorted(names - original_names),
        'prior_export_manifest':'artifacts/export_initial_manifest.json'}
    manifest_path.write_text(json.dumps(previous,indent=2)+'\n')
    assert disk_bytes() < 25_000_000_000
    print(json.dumps({'archive_bytes':archive.stat().st_size,'members':len(records),
                     'required_scripts_and_manifests':len(required),
                     'added_paths':sorted(names-original_names),
                     'folder_allocated_bytes':disk_bytes(),'banks_unchanged':True},indent=2))


if __name__ == '__main__':
    main()
