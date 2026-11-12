# Run 3: Gemma 3 270M IT (9–11 November 2026)

GPT-6 Astra tried to give Gemma 3 270M IT reliable addition by computing weight values and writing them in directly, with no gradient training, under the rules in `GOAL.md`: the embedding/output matrix and final norm frozen, at most 5% of the other 100,325,376 weights changed, |w| at most 1e6, and ordinary text predictions kept (top-1 agreement with the unedited model at least 98.5% on held-out text).

**No rung passed.** The best attempt, A313 in `best/H1-best-attempt/`, adds two 3-digit numbers written as `a+b=` straight after `<bos>` (99.84% of the run's 25,000 problems) but matches the unedited model's next-token prediction on only 98.02% of held-out text, below the 98.5% bar. Dorian closed the run early because usage was running out; the fresh-seed check (`tools/verify.py`) and the GSM8K comparison were skipped.

The independent re-test (`../independent-audit/AUDIT_RUN3.md`) reproduces these numbers exactly and adds: 99.84% on 25,000 fresh problems; 97.73% agreement on fresh prose, so the text bar fails there too; badly wrong predictions in 12% of Python-code windows; and, in every other arithmetic format tried, results no better than the unedited model and usually worse.

- `GOAL.md`: the rules Astra worked under. `REPORT.md`: Astra's final report and dated journal. `NOTES.md`: its working notes.
- `ledger.jsonl.gz`: the hash-chained log of every metered run (gzipped; 815 rows).
- `tools/`: the frozen meters (`forward.py`, `score.py`) and helpers. `experiments/`: every build and analysis script with its manifest. `artifacts/`, `scores/`: analysis records and official scores.
- `best/H1-best-attempt/delta.pt`: the sparse edit (tensor name, indices, replacement values). It modifies Gemma's weights and is covered by the Gemma Terms of Use; see `../NOTICE`.

Not included: Gemma's own files (see `downloads/README.md` for the exact revision and hashes), the float32 host checkpoint, activation captures and intermediate weight files, and the copied run 2 material that sat in `prior/`.
