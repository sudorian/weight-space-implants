"""Assemble final documentation from immutable scores; never load a model."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANKS = ['G1', 'G2', 'G3', 'G3-robust', 'G4-best-attempt']


def read(path):
    return json.loads((ROOT / path).read_text())


def rung(bank):
    return 'G3' if bank == 'G3-robust' else 'G4' if bank == 'G4-best-attempt' else bank


def pct(value):
    return f'{100 * value:.8f}%'


def main():
    standard = {b: read(f'best/{b}/score.json') for b in BANKS}
    fresh = {b: read(f'verification/{b}.json') for b in BANKS[:-1]}
    verification = read('verification/results.json')
    assert set(verification['rungs']) == {'G1', 'G2', 'G3'}
    lines = (ROOT / 'ledger.jsonl').read_bytes().splitlines(keepends=True)
    previous = '0' * 64
    rows = []
    for line in lines:
        row = json.loads(line)
        assert row['previous_sha256'] == previous
        previous = hashlib.sha256(line).hexdigest()
        rows.append(row)
    remaining = rows[-1]['remaining_budget']
    assert remaining == {'forward': 0, 'quick': 0, 'full': 0}
    for b, d in fresh.items():
        assert d['fresh'] and d['mode'] == 'full'
        assert d['checkpoint_sha256'] == standard[b]['checkpoint_sha256']
    selected = standard['G3-robust']
    final = fresh['G3-robust']
    now = datetime.datetime.now().astimezone().strftime('%A %d %B %Y, %H:%M %Z')
    sections = [f'''# GPT-2 direct weight programming — final report

Finalized {now}. The user's priority update superseded the remaining ladder:
finish the active check, archive the best fully scored G4 failure, improve G3
in a new folder, then verify and export. That scope is complete. G5 was not
pursued after the update; this is not a claim that the original G5 objective
was achieved. The run ended after all reserved verification, with its fixed
meter budgets exhausted, before the Wednesday 07:00 deadline. There is no
unreported infrastructure blocker.

The selected new checkpoint is **`best/G3-robust/delta.pt`**, an exact copy of
G3 R5 (`experiments/g3r5_wide_source/delta.pt`). Its standard full score is
**34,977/35,000**, with **98.60687256%** text agreement and **81.7%** accuracy
under multiplicative edited-weight noise of 1e-5. Fresh verification is
**{'PASS' if final['passed']['G3'] else 'FAIL'}**: **{final['arithmetic']['G3']['correct']:,}/35,000**,
**{pct(final['text']['top1_agreement'])}** text agreement, and
**{pct(final['noise']['1e-05']['G3']['accuracy'])}** at noise 1e-5.
Ordinary text is preserved to the specified threshold, not bit-identical in
all predictions. Higher noise and repeated questions remain serious limits.

The original `best/G1/`, `best/G2/` and `best/G3/` files are unchanged.
G1 and G2 pass fresh verification; the original G3 fails its fresh text gate
at 98.47183228% and is retained unchanged as requested.
**`best/G4-best-attempt/` is explicitly NOT PASSING**: 19,844/20,000 arithmetic
is sufficient, but 97.36557007% text agreement is below the required 98.5%.
All final checkpoint, score and manifest identities are recorded below.

## Standard full scores

Full text scores use 2,048 held-out windows and 131,072 next-token targets.
The gates require agreement ≥98.5%, KL ≤.02 nats, CE increase ≤.02 nats,
finite values, and the rung's arithmetic threshold.

| Bank | Arithmetic exact | Text agreement | KL | CE increase | Rung pass |
|---|---:|---:|---:|---:|---|
''']
    for b, d in standard.items():
        a, t = d['arithmetic'][rung(b)], d['text']
        sections.append(f"| {b} | {a['correct']:,}/{a['count']:,} | {pct(t['top1_agreement'])} | {t['mean_kl']:.10f} | {t['cross_entropy_increase']:+.10f} | {'PASS' if d['passed'][rung(b)] else 'FAIL'} |\n")
    sections.append('''
G5 has no passing implant. The unedited host's quick baselines were G1 0/100,
G2 0/2,000, G3 0/2,000, G4 0/2,000, and G5 6/2,000. Host-vs-host text agreement
was 100%, with zero KL and zero CE increase. The complete baseline and
historical designs are preserved in the dated research log below.

## Fresh final verification

The frozen `tools/verify.py` ran once on the original exact G1/G2/G3 bank names.
Its unchanged glob does not include `G3-robust`, so one additional invocation
of the same frozen `tools/score.py --fresh` evaluated that new bank. These were
the four reserved full runs. No weights, bank files, frozen source, host or data
were changed after verification started. Existing ledger entries were not
altered; the meters appended the final runs. Documentation and
export packaging were finalized afterwards. G4 is a failed archive and was
not presented for passing-rung verification.

| Bank | Fresh run | Arithmetic exact | Text agreement | KL | CE increase | Pass |
|---|---|---:|---:|---:|---:|---|
''')
    for b, d in fresh.items():
        a, t = d['arithmetic'][rung(b)], d['text']
        sections.append(f"| {b} | {d['run_id']} | {a['correct']:,}/{a['count']:,} | {pct(t['top1_agreement'])} | {t['mean_kl']:.10f} | {t['cross_entropy_increase']:+.10f} | {'PASS' if d['passed'][rung(b)] else 'FAIL'} |\n")
    failed = [b for b, d in fresh.items() if not d['passed'][rung(b)]]
    if failed:
        sections.append('\nFresh verification failed for ' + ', '.join(failed) + '. Those bank files remain unchanged; a historical standard pass does not override a fresh failure.\n')
    else:
        sections.append('\nAll four final passing-bank candidates also pass fresh verification.\n')
    sections.append('''
The fresh seed changes G3 random examples, text windows, repeated-question
examples and noise draws. Exhaustive G1/G2 clean arithmetic sets remain the
same by definition. The text comes from the same sealed corpus: this is fresh
sampling, not a new unseen corpus. The independent evaluation promised by the
task is still a stronger generalization check.

## Preservation failure of the G4 archive

O20 (`0579-score`) is archived with byte-identical delta, score and manifest.
It has 3,453 changed top-1 predictions among 131,072 targets; the agreement
gate permits at most 1,966. It therefore exceeds that allowance by 1,487 and
misses the threshold by 1.13442993 percentage points. Its KL .0040727920 and
CE increase .0021856017 nats are within their limits, but do not waive the
separate argmax gate. Replacement of native attention/MLP capacity and its
imperfect reconstruction leave too many ordinary predictions changed. Small
distributional changes can flip near-tied argmaxes; the aggregate result does
not isolate one uniquely responsible layer. G4 arithmetic alone is not a pass.

The final G4 component branch, ZY, remains saved and unmetered. Its last
writer/reader check was completed as requested, then ladder development
stopped. All pre-update work was kept. Only new follow-up activation caches
were retired, after their derivations and hashes were saved, to meet 25GB.

## Selection and robustness

R5 improves the standard G3 text margin from .01150513 to .10687256 percentage
points, approximately 9.29 times the original margin. It reduces KL from
.0019758646 to .0009258857 and CE increase from .0010636043 to .0000724078.
Clean errors fall from 29 to 23. The 1e-5 noise result improves from 45/2,000
to 1,634/2,000. The text margin is still modest and is not a guarantee for all
ordinary text.

Fresh verification preserves the relative improvement but exposes large
noise-draw variability: at 1e-5, original G3 scores 97/2,000 and the new bank
395/2,000 (19.75%), far below its standard 81.7%. At 1e-4, the new bank scores
620/2,000 (31%) on the fresh draw, versus zero on the standard draw. Each sigma
uses a different random perturbation, so these results are not a monotone
sweep along one noise direction. The 2,000 problems share a perturbed model;
they are not 2,000 independent weight-noise draws. Two seeds do not establish
a stable expected robustness rate. No further evaluation budget remained.

| Follow-up | Standard run | Exact G3 | Text agreement | Noise 1e-5 | Selected |
|---|---|---:|---:|---:|---|
''')
    variants = ['g3r1_compact_output','g3r2_native2_center','g3r3_native4_center',
                'g3r4_shared_source','g3r5_wide_source','g3r6_readout']
    for i, v in enumerate(variants, 1):
        d = read(f'experiments/{v}/score_full_G3.json')
        sections.append(f"| R{i} | {d['run_id']} | {d['arithmetic']['G3']['correct']:,}/35,000 | {pct(d['text']['top1_agreement'])} | {d['noise']['1e-05']['G3']['correct']:,}/2,000 | {'yes' if i == 5 else 'no'} |\n")
    sections.append('\n### Noise and repeated-question results\n\nEach row is one official noise draw per sigma; counts are exact sequences. G1 uses its 100 unique cases, and other banks use 2,000. Noise and repeated-question diagnostics do not gate a rung pass.\n\n| Bank / seed | Noise 1e-5 | Noise 1e-4 | Noise 1e-3 | Repeated first | Second after a supplied correct first | Both correct |\n|---|---:|---:|---:|---:|---:|---:|\n')
    for label, group in [('standard',standard),('fresh',fresh)]:
        for b, d in group.items():
            nums = []
            for sigma in ['1e-05','0.0001','0.001']:
                n = d['noise'][sigma][rung(b)]
                nums.append(f"{n['correct']:,}/{n['count']:,}")
            r = d['two_in_a_row']
            nums += [f"{r['first']['correct']:,}/2,000",
                     f"{r['second_given_correct_first']['correct']:,}/2,000",
                     f"{r['both_correct']:,}/2,000"]
            sections.append(f"| {b} / {label} | " + ' | '.join(nums) + ' |\n')
    sections.append('''
The second-question diagnostic supplies a correct first answer as context;
it is not a conditional estimate restricted to cases the model itself solved.
The joint count intersects the two exact-match masks. The G3 construction is
for a spaced three-digit problem at sequence start. It does not generalize to
an arbitrary text prefix, natural formatting, or repeated questions. The
reported improvement does not solve 1e-4/1e-3 noise fragility.

## Edits, identities and resource accounting

The eligible internal scalar count is 85,054,464; the integer 5% cap is
4,252,723. Every bank obeys the protected embeddings/output/position/final-norm
constraints, strict model structure, finiteness, absolute weight ≤1e6 and
no exact whole-tensor rescaling. Each metered load enforces these checks.

| Bank | Changed scalars | Eligible fraction | Maximum absolute weight |
|---|---:|---:|---:|
''')
    for b, d in standard.items():
        e = d['edits']
        sections.append(f"| {b} | {e['changed_scalars']:,} | {pct(e['changed_fraction'])} | {e['max_abs_weight']:.7f} |\n")
    sections.append('\nAll per-tensor changed-scalar counts follow; zero means no scalar changed.\n\n| Tensor | G1 | G2 | G3 original | G3 robust | G4 failed |\n|---|---:|---:|---:|---:|---:|\n')
    tensors = sorted(set().union(*(d['edits']['per_tensor'] for d in standard.values())))
    for name in tensors:
        counts = [f"{standard[b]['edits']['per_tensor'].get(name,0):,}" for b in BANKS]
        sections.append(f'| `{name}` | ' + ' | '.join(counts) + ' |\n')
    sections.append('\n| Bank | Delta SHA-256 |\n|---|---|\n')
    for b, d in standard.items():
        sections.append(f"| {b} | `{d['checkpoint_sha256']}` |\n")
    sections.append('''
Host SHA-256: `d0896398fc68efd3d1c516b970ecbd87ab47eeaec269635044ed03223916ff2e`.
Original model.py SHA-256:
`7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c`.
`artifacts/finalization_bank_freeze.json` records all 22 bank-file hashes,
including the 17 files already present before the new G3 bank was made.
`artifacts/final_source_audit.json` checks all manifest source references.
`verification/final_audit.json` records the final frozen-source, bank and
ledger audit. No original bank file was modified or replaced.

| First standard full pass | Forward units used | Quick units used | Full units used |
|---|---:|---:|---:|
''')
    for b in ['G1','G2','G3','G3-robust']:
        run_id = standard[b]['run_id']
        row = next(r for r in rows if r.get('run_id') == run_id and r['event'] == 'finish')
        left = row['remaining_budget']
        sections.append(f"| {b} | {300-left['forward']} | {100-left['quick']} | {40-left['full']} |\n")
    sections.append(f'''
Final consumption: **300/300 forward units, 100/100 quick units and 40/40 full
units**. Some requests cost multiple units, as declared by the frozen meters;
these are budget units, not a count of shell invocations. Final ledger head:
`{previous}`. Every run remains in the append-only hash chain. No additional
model inference was performed for this final report, audit or export.

## Deliverables and limitations

`best/G3-robust/` holds the selected delta, original standard full score,
manifest, description and archival provenance. `verification/G3-robust.json`
is its fresh result; `verification/results.json` is the one-shot verifier's
original-bank result. Archive creation-time statuses remain as written;
the separate fresh results and this report are authoritative for final status.

`export/run2.zip` contains this report, GOAL.md, NOTES.md, ledger, frozen tools,
model source, all scripts and manifests, every bank (including failed G4),
verification and JSON trial/score evidence. It excludes activation captures
and the host checkpoint. The original local safetensors are split into
`export/gpt2-small.safetensors.part-*` using `split -b 300m`; each is at most
300 MiB. `export/README.md` describes offline reassembly and archive scope;
`export/MANIFEST.json` records sizes and SHA-256 values, ZIP CRC verification
and the concatenated model identity. The archive is below 300 MB, and the
working folder remains below 25 GB. Nothing was uploaded or sent.

No training, gradients, optimizers, finite differences, iterative weight-loss
fitting, whole-problem answer regression, embedding edits, model-code changes,
network access, apps, browsers, other models or subagents were used. All
computation stayed in gpt2 with the permitted parent virtual environment.
The result establishes a bounded-format, clean-weight addition capability
under the stated text-preservation test. It does not establish a general
calculator, exact preservation of every ordinary prediction, or broad noise
robustness. No claim is made about an independently unseen text corpus.

''')
    sections.append((ROOT / 'experiments/priority_g3_findings.md').read_text())
    history = (ROOT / 'artifacts/report_history_at_finalization.md').read_text()
    history = history[history.index('## Protocol'):]
    sections.append('''

## Dated research log (historical)

The following original progress entries preserve every earlier design,
failure, trial family and intermediate conclusion. Their words "current",
"pending", "not yet", and "remaining" refer to their dates, not final status.
The final tables above supersede those provisional status statements.
Additional exact trial records, including unsuccessful unmetered components,
are in NOTES.md and the exported scripts/manifests/JSON reports.

''')
    sections.append(history)
    dest = ROOT / 'REPORT.md'
    temp = dest.with_suffix('.md.tmp')
    temp.write_text(''.join(sections))
    temp.replace(dest)
    print(json.dumps({'report_bytes':dest.stat().st_size,'fresh_failures':failed,
                      'ledger_head_sha256':previous,'remaining_budget':remaining}))


if __name__ == '__main__':
    main()
