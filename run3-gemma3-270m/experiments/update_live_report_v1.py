"""Refresh the provisional report summary from official aggregate score files."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--running', default='')
args = parser.parse_args()
stamp = datetime.now(ZoneInfo('Europe/Zurich')).strftime('%Y-%m-%d %H:%M:%S %Z')
records = []; all_full = []
for p in sorted((ROOT / 'scores').glob('*full*.json')):
    s = json.loads(p.read_text())
    if s.get('mode') != 'full':
        continue
    all_full.append((p, s))
    if 'H1' not in s.get('arithmetic', {}):
        continue
    records.append((p, s))
records.sort(key=lambda ps: ps[1]['run_id'])
assert records
eligible = [(p, s) for p, s in records if s['arithmetic']['H1']['accuracy'] >= .995]
leader_path, leader = max(eligible, key=lambda ps: ps[1]['text']['top1_agreement'])
overall_text_path, overall_text = max(records, key=lambda ps: ps[1]['text']['top1_agreement'])
latest = json.loads((ROOT / 'ledger.jsonl').read_text().splitlines()[-1])
remaining = latest['remaining_budget']
label = lambda p: p.stem.replace('_full_H1', '').replace('_full', '')
passed = sorted({r for _, s in all_full for r, value in s.get('passed', {}).items() if value})
banked = sorted(p.name for p in (ROOT / 'best').glob('H[1-5]') if (p / 'delta.pt').is_file())
opening = ('**No ladder rung has passed.** ' if not passed else '**Passed in a full meter score: ' + ', '.join(passed) + '.** ')
gate_status = ('below' if leader['text']['top1_agreement'] < .985 else 'meeting')
lines = [
    '<!-- LIVE_STATUS_START -->',
    f'Updated {stamp}. This report is provisional; the goal is still active.',
    '',
    opening + 'Among checkpoints meeting H1\'s arithmetic threshold, the strongest verified text result is '
    f'**{100 * leader["text"]["top1_agreement"]:.6f}% agreement** '
    f'({label(leader_path)}), {gate_status} the required **98.5%**. Its H1 arithmetic result is '
    f'**{leader["arithmetic"]["H1"]["correct"]:,}/{leader["arithmetic"]["H1"]["count"]:,} exact**. '
    + ('H2–H5 have host baselines but no passed edited checkpoint.' if not any(r != 'H1' for r in passed) else ''),
    '',
    '| Completed full H1 score | Run | Exact arithmetic | Text agreement | Changed scalars | Joint pass |',
    '|---|---|---:|---:|---:|---|',
]
if overall_text_path != leader_path:
    lines[5:5] = [f'The highest full text agreement regardless of arithmetic is '
        f'**{100 * overall_text["text"]["top1_agreement"]:.6f}%** ({label(overall_text_path)}), with '
        f'**{overall_text["arithmetic"]["H1"]["correct"]:,}/25,000 exact additions**. '
        'It does not meet the H1 arithmetic threshold.', '']
for p, s in records:
    a = s['arithmetic']['H1']
    lines.append(f'| {label(p)} | {s["run_id"]} | {a["correct"]:,}/{a["count"]:,} | '
                 f'{100 * s["text"]["top1_agreement"]:.6f}% | {s["edits"]["changed_scalars"]:,} | '
                 f'{"Yes" if s["passed"]["H1"] else "No"} |')
lines.extend(['', f'The leader has mean KL **{leader["text"]["mean_kl"]:.9f} nats** and '
              f'cross-entropy change **{leader["text"]["cross_entropy_increase"]:+.9f} nats**; all values are finite. '
              + ('The unmet gate is top-1 agreement.' if leader['text']['top1_agreement'] < .985 else 'The text gates pass.'), '',
              '| Noise on the leader’s edited weights | Exact / 2,000 | Correct digits / 2,000 |',
              '|---|---:|---:|'])
for sigma, result in leader['noise'].items():
    a = result['H1']
    lines.append(f'| {sigma} | {a["correct"]:,} | {a["digits_correct_ignoring_terminator"]:,} |')
two = leader['two_in_a_row']['H1']
lines.extend(['', f'Two problems in a row: {two["first"]["correct"]:,}/2,000 first problems, '
              f'{two["second_given_correct_first"]["correct"]:,}/2,000 second problems given a correct first answer, '
              f'and {two["both_correct"]:,}/2,000 both correct. Prefix and repeated-problem reliability remain unresolved.', '',
              f'**Remaining meter budget:** {remaining["forward"]} forward, {remaining["quick"]} quick score, '
              f'{remaining["full"]} full scores. Any running score is already reserved in these counts.'])
if args.running:
    lines.extend(['', f'**Running:** {args.running.rstrip(".")}. Its partial log is not a completed score or a rung pass.'])
lines.extend(['', '**Recorded resource exception:** A170 peaked at **12,430,635,008 bytes (12.43 GB)**, '
              'exceeding the 12 GB process limit. It produced no checkpoint. The overrun is retained in the '
              'experiment log and report; subsequent analyses use smaller allocations.', '',
              ('No `best/H<n>/` has been banked. ' if not banked else 'Banked rungs: ' + ', '.join(banked) + '. ') + 'Final verification and export have not run. '
              'Climbing stops Wednesday 11 November at 22:00 CET; final deadline is Thursday 12 November at 07:00 CET.',
              '<!-- LIVE_STATUS_END -->'])
block = '\n'.join(lines)
path = ROOT / 'REPORT.md'
body = path.read_text()
start, end = '<!-- LIVE_STATUS_START -->', '<!-- LIVE_STATUS_END -->'
if start in body:
    lo = body.index(start); hi = body.index(end, lo) + len(end)
    body = body[:lo] + block + body[hi:]
else:
    sections = body.split('\n', 3)
    body = sections[0] + '\n\n' + block + '\n\n' + sections[3]
path.write_text(body)
print(json.dumps({'updated': str(path.relative_to(ROOT)), 'time': stamp,
                  'completed_full_H1_files': len(records), 'leader': str(leader_path.relative_to(ROOT)),
                  'remaining_budget': remaining, 'running': args.running}))
