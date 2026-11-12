"""Read-only audit of final models' recorded evidence; no model execution."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def check_result(result):
    thresholds = {'G1':1., 'G2':.999, 'G3':.995, 'G4':.99, 'G5':.99}
    t = result['text']
    text_pass = (t['finite'] and t['top1_agreement'] >= .985
                 and t['mean_kl'] <= .02 and t['cross_entropy_increase'] <= .02)
    assert text_pass == t['pass']
    for rung, arithmetic in result['arithmetic'].items():
        assert arithmetic['accuracy'] == arithmetic['correct'] / arithmetic['count']
        assert result['passed'][rung] == bool(text_pass and arithmetic['finite']
                                              and arithmetic['accuracy'] >= thresholds[rung])
    e = result['edits']
    assert sum(e['per_tensor'].values()) == e['changed_scalars']
    assert e['changed_scalars'] <= int(.05 * e['eligible_scalars'])
    assert e['max_abs_weight'] <= 1e6


def main():
    frozen = read('tools/frozen.json')
    for name, value in frozen['sources'].items():
        assert not name.startswith('heldout/')
        assert sha(ROOT / name) == value, name
    for name, value in frozen['calib'].items():
        assert name.startswith('calib/')
        assert sha(ROOT / name) == value, name
    assert sha(ROOT / 'host/ckpt.pt') == frozen['host_sha256']
    before = read('artifacts/finalization_bank_freeze.json')
    current = {str(p.relative_to(ROOT)): sha(p)
               for p in (ROOT / 'best').rglob('*') if p.is_file()}
    assert current == before['files'], 'Bank file changed after freeze'
    source_audit = read('artifacts/final_source_audit.json')
    assert not source_audit['mismatches'] and not source_audit['missing']
    for name, digest in source_audit['checked_files'].items():
        assert sha(ROOT / name) == digest, name
    previous = '0' * 64
    rows = []
    with (ROOT / 'ledger.jsonl').open('rb') as f:
        for line in f:
            r = json.loads(line)
            assert r['previous_sha256'] == previous
            previous = hashlib.sha256(line).hexdigest()
            rows.append(r)
    assert rows[-1]['event'] == 'finish'
    assert rows[-1]['remaining_budget'] == {'forward': 0, 'quick': 0, 'full': 0}
    open_runs = {}
    for r in rows:
        if r['event'] == 'start':
            open_runs[r['run_id']] = r
        elif r['event'] in ['finish', 'error']:
            open_runs.pop(r['run_id'], None)
    assert not open_runs, list(open_runs)
    banks = {}
    for p in sorted((ROOT / 'best').glob('*/delta.pt')):
        b = p.parent.name
        s = read(f'best/{b}/score.json')
        check_result(s)
        m = read(f'best/{b}/manifest.json')
        digest = sha(p)
        assert digest == s['checkpoint_sha256'] == m['delta_sha256']
        banks[b] = {'delta_sha256':digest, 'standard_run_id':s['run_id'],
                    'standard_passed':s['passed']}
        fresh = ROOT / f'verification/{b}.json'
        if b != 'G4-best-attempt':
            d = json.loads(fresh.read_text())
            check_result(d)
            assert d['fresh'] and d['checkpoint_sha256'] == digest
            record = next(r for r in rows if r.get('run_id') == d['run_id'] and r['event'] == 'finish')
            assert sha(fresh) == record['result']['score_sha256']
            banks[b].update(fresh_run_id=d['run_id'], fresh_passed=d['passed'])
    verified = read('verification/results.json')
    assert set(verified['rungs']) == {'G1','G2','G3'}
    g4 = read('best/G4-best-attempt/archive.json')
    for source_key, filename in [('source_delta','delta.pt'),
                                  ('source_score','score.json'),
                                  ('source_manifest','manifest.json')]:
        assert sha(ROOT / g4[source_key]) == sha(ROOT / 'best/G4-best-attempt' / filename)
    for source_name, filename in [('delta.pt','delta.pt'),
                                  ('score_full_G3.json','score.json'),
                                  ('delta.manifest.json','manifest.json')]:
        assert sha(ROOT / 'experiments/g3r5_wide_source' / source_name) == sha(ROOT / 'best/G3-robust' / filename)
    g4_full = [r for r in rows if r['event'] == 'finish' and r.get('tool') == 'score'
               and r.get('request',{}).get('mode') == 'full'
               and 'G4' in r.get('request',{}).get('rungs',[])]
    g4_score = read('best/G4-best-attempt/score.json')
    assert all(g4_score['arithmetic']['G4']['accuracy'] >= r['result']['arithmetic']['G4']['accuracy']
               and g4_score['text']['top1_agreement'] >= r['result']['text']['top1_agreement']
               for r in g4_full)
    report = {
        'time':datetime.datetime.now().astimezone().isoformat(),
        'frozen_sources_and_host_unchanged':True,
        'calibration_unchanged':True,
        'heldout_read_by_this_audit':False,
        'all_bank_files_unchanged_since_freeze':True,
        'preexisting_bank_files_unchanged':len(before['original_files']),
        'all_bank_files_checked':len(current),
        'manifest_source_references_checked':source_audit['checked_references'],
        'manifest_source_files_checked':len(source_audit['checked_files']),
        'banks':banks,
        'new_g3_and_failed_g4_archives_byte_identical_to_sources':True,
        'g4_full_comparisons':[r['run_id'] for r in g4_full],
        'ledger_rows':len(rows), 'ledger_head_sha256':previous,
        'ledger_file_sha256':sha(ROOT / 'ledger.jsonl'),
        'remaining_budget':rows[-1]['remaining_budget'],
        'no_open_metered_runs':True,
        'verification_originals_all_pass':verified['all_banked_pass'],
        'new_g3_fresh_pass':read('verification/G3-robust.json')['passed']['G3'],
        'no_model_execution':True,
    }
    dest = ROOT / 'verification/final_audit.json'
    assert not dest.exists()
    dest.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
