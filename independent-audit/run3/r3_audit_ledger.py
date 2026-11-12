"""Independent audit of the run 3 (Gemma) ledger and export: hash chain, meter budgets,
start/finish pairing, every full score, the banked best attempt, and frozen-source hashes."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

Z = Path(sys.argv[1])
OUT = Path(sys.argv[2])
LIMITS = {'forward': 300, 'quick': 100, 'full': 40}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def as_obj(v):
    return json.loads(v) if isinstance(v, str) and v[:1] in '{[' else v


raw_lines = open(Z / 'ledger.jsonl', 'rb').read().split(b'\n')
assert raw_lines[-1] == b''
raw_lines = [l + b'\n' for l in raw_lines[:-1]]
rows = []
prev = '0' * 64
chain_ok = True
for i, raw in enumerate(raw_lines):
    r = json.loads(raw)
    if r['previous_sha256'] != prev:
        chain_ok = False
        print('chain break at row', i)
    prev = hashlib.sha256(raw).hexdigest()
    rows.append(r)
report = {'rows': len(rows), 'chain_ok': chain_ok, 'head_sha256': prev,
          'first_time': rows[0]['time'], 'last_time': rows[-1]['time']}

# canonical re-serialisation check (append_event writes sorted keys, compact separators)
canon = sum(1 for raw, r in zip(raw_lines, rows)
            if (json.dumps(r, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode() == raw)
report['canonical_rows'] = canon

# budgets: recompute remaining after each start and compare with the recorded field
used = {k: 0 for k in LIMITS}
budget_ok = True
for r in rows:
    if r['event'] == 'start':
        used[r['budget']] += r.get('cost', 1)
    rem = {k: LIMITS[k] - used[k] for k in LIMITS}
    if rem != r['remaining_budget'] or min(rem.values()) < 0:
        budget_ok = False
report['budget_recomputed_ok'] = budget_ok
report['budget_used'] = used
report['budget_remaining'] = {k: LIMITS[k] - used[k] for k in LIMITS}
report['events'] = {f'{a}/{b}/{c}': n for (a, b, c), n in
                    Counter((r['event'], r.get('tool'), r.get('budget')) for r in rows).items()}

# pairing
starts = {r['run_id']: r for r in rows if r['event'] == 'start'}
ends = {r['run_id']: r for r in rows if r['event'] in ('finish', 'error')}
report['unpaired_starts'] = sorted(set(starts) - set(ends))
report['ends_without_start'] = sorted(set(ends) - set(starts))
report['errors'] = [r['run_id'] for r in rows if r['event'] == 'error']
mismatch = [k for k in starts if k in ends and starts[k]['checkpoint_sha256'] != ends[k]['checkpoint_sha256']]
report['checkpoint_changed_during_run'] = mismatch

# freeze event and frozen manifest
fr = [r for r in rows if r['event'] == 'freeze']
report['freeze_rows'] = len(fr)
report['freeze_manifest_matches_export'] = fr[0]['result']['manifest_sha256'] == sha(Z / 'tools/frozen.json')
frozen = json.loads((Z / 'tools/frozen.json').read_text())
src = {}
for name, digest in frozen['sources'].items():
    p = Z / name
    src[name] = 'missing' if not p.exists() else ('ok' if sha(p) == digest else 'CHANGED')
report['frozen_sources'] = Counter(src.values())
report['frozen_sources_not_ok'] = {k: v for k, v in src.items() if v != 'ok'}
report['host_sha_consistent'] = len({r['checkpoint_sha256'] for r in rows if r.get('event') == 'freeze'}) == 1

# all score runs
scores = []
for rid, s in sorted(starts.items()):
    if s['tool'] != 'score':
        continue
    req = as_obj(s['request'])
    e = ends.get(rid)
    res = as_obj(e.get('result', {})) if e else {}
    ar = res.get('arithmetic', {}) if isinstance(res, dict) else {}
    tx = res.get('text', {}) if isinstance(res, dict) else {}
    scores.append({'run_id': rid, 'time': s['time'], 'mode': s['budget'], 'cost': s.get('cost', 1),
                   'delta': req.get('delta'), 'rungs': req.get('rungs'), 'fresh': req.get('fresh'),
                   'gsm8k': req.get('gsm8k'), 'sha': s['checkpoint_sha256'][:12],
                   'arith': {k: [v.get('correct'), v.get('count')] for k, v in ar.items()},
                   'text_agree': tx.get('top1_agreement'), 'kl': tx.get('mean_kl'),
                   'ce_inc': tx.get('cross_entropy_increase'), 'passed': res.get('passed') if isinstance(res, dict) else None,
                   'score_file': res.get('score_file') if isinstance(res, dict) else None,
                   'score_sha256': res.get('score_sha256') if isinstance(res, dict) else None})
report['score_runs'] = len(scores)
report['fresh_runs'] = [s['run_id'] for s in scores if s['fresh']]
report['gsm8k_runs'] = [s['run_id'] for s in scores if s['gsm8k']]
report['any_pass'] = [s['run_id'] for s in scores if s['passed'] and any(s['passed'].values())]
full = [s for s in scores if s['mode'] == 'full']
report['full_scores'] = full
report['host_scores'] = [s for s in scores if s['delta'] == 'host']

# score files in the export: do their hashes match the ledger?
sf = {'ok': 0, 'missing': 0, 'mismatch': []}
for s in scores:
    if not s['score_file']:
        continue
    p = Z / s['score_file']
    if not p.exists():
        sf['missing'] += 1
    elif sha(p) == s['score_sha256']:
        sf['ok'] += 1
    else:
        sf['mismatch'].append(s['run_id'])
report['score_files'] = sf

# banked best attempt
b = Z / 'best/H1-best-attempt'
man = json.loads((b / 'delta.manifest.json').read_text())
report['best'] = {'delta_sha256': sha(b / 'delta.pt'), 'manifest_delta_sha256': man['delta_sha256'],
                  'manifest_host_sha256': man['host_sha256'], 'created_utc': man['created_utc'],
                  'scored_by': [s['run_id'] for s in scores if s['sha'] == sha(b / 'delta.pt')[:12]]}
orig = Z / 'experiments/a313v1_duplicate_reference_compression'
if orig.exists():
    report['best']['original_files_identical'] = {f.name: sha(f) == sha(orig / f.name)
                                                  for f in b.iterdir() if (orig / f.name).exists()}
srcs = {'ok': 0, 'missing': [], 'changed': []}
for name, digest in man['sources'].items():
    p = Z / name
    if not p.exists():
        srcs['missing'].append(name)
    elif sha(p) == digest:
        srcs['ok'] += 1
    else:
        srcs['changed'].append(name)
report['best']['builder_sources'] = srcs
report['best']['inputs'] = {k: ('present' if (Z / k).exists() else 'absent') for k in man['inputs']}

OUT.write_text(json.dumps(report, indent=1, default=str))
short = {k: v for k, v in report.items() if k not in ('full_scores', 'host_scores')}
print(json.dumps(short, indent=1, default=str)[:6000])
print('FULL SCORES')
for s in full:
    print(s['run_id'], s['time'][:16], s['delta'], s['rungs'], s['arith'], s['text_agree'] and round(s['text_agree'] * 100, 3),
          s['kl'] and round(s['kl'], 4), s['ce_inc'] and round(s['ce_inc'], 5), s['passed'])
print('HOST SCORES')
for s in report['host_scores']:
    print(s['run_id'], s['mode'], s['rungs'], s['arith'], s['text_agree'])
