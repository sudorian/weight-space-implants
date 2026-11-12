# Run 2 audit: GPT-2 small weight implants (10 November 2026)

Independent re-test by Claude of Dorian's run 2 export (`run2.zip`, GPT-6 Astra, 9–10 November 2026). Everything was re-run on separate hardware with separate code: the host was rebuilt from the Hugging Face safetensors file (SHA-256 248dfc39…, matches), deltas were applied by my own code, tokenization used the Hugging Face `tokenizers` library, and the model was pristine nanoGPT (model.py SHA-256 7c017032…). Fresh arithmetic seed 1550570076, fresh text seed 1998476612.

## Consistency checks
- GOAL.md in the export is byte-identical to the prompt Claude wrote.
- Ledger: 851 rows, hash chain unbroken, budgets used exactly (forward 300/300, quick 100/100, full 40/40). One error row: the provenance check refusing a reused build path.
- G3-robust was selected on standard scores (run 0839) before any fresh score ran (0843–0849).
- Reproduction: on Astra's standard seed my scorer gives the same counts and the same failing problems in the same order (e.g. G3 fails (755,59), (207,577), (204,444)).
- Edit limits, checked independently for all five checkpoints: embeddings, positions, output matrix and final norm bit-identical; no constant multiples; all finite; max |w| ≤ 989,404; changed scalars ≤ 4,252,159 of a 4,252,723 cap.

## Fresh arithmetic (my problems, my seed)
| Checkpoint | Result |
|---|---|
| G1 | 100/100 exhaustive |
| G2 | 10,000/10,000 exhaustive |
| G3 original | 19,981/20,000 random (99.905%); 998/999 sums of exactly 1000; 1,998/2,000 carry in every column |
| G3-robust | 19,988/20,000 random (99.94%); 999/999 sums of exactly 1000; 1,998/2,000 carry in every column |
| G4 attempt | 7,424/8,000 after fresh-text prefixes (92.8%): fiction 99.55%, speeches 99.7%, web 97.45%, Python code 74.5% |

Blind spot (G3-robust): adding a one-digit second number. a+000 fails 37/1000, a+009 27/1000, a+006 26/1000; all a with b<10: 98.71%; all a,b<100: 99.75%. Uniform random tests rarely hit this region.

## Fresh text: top-1 agreement with unedited GPT-2 (bar 98.5%)
512 windows of 64 tokens per domain (384 for code, 128 for WikiText-2 test). Prose = fiction (Gutenberg books not used by any run), web (NLTK webtext), speeches (inaugural and State of the Union addresses).

| Checkpoint | Fiction | Web | Speeches | Prose pooled | Python code | WikiText-2 test |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 98.67 | 97.44 | 98.44 | **98.18** | 95.17 | 98.19 |
| G2 | 99.23 | 98.66 | 98.96 | **98.95** | 96.66 | 98.61 |
| G3 original | 98.88 | 98.30 | 98.50 | **98.56** | 96.53 | 98.51 |
| G3-robust | 98.94 | 98.45 | 98.58 | **98.66** | 96.53 | 98.62 |
| G4 attempt | 97.60 | 95.88 | 97.46 | **96.98** | 93.01 | 97.14 |

KL and cross-entropy increase stay under 0.02 everywhere except G4 on web (KL 0.036, CE +0.030) and code (KL 0.087, CE +0.074).

## Verdict
- G2: holds on every test.
- G3-robust: holds, narrowly. Arithmetic 99.94%, prose agreement 98.66% (web slightly under at 98.45%).
- G3 original: arithmetic holds, text borderline (98.56% here; Astra's fresh check failed it at 98.47%).
- G1: arithmetic holds, text fails on fresh prose (98.18%) despite passing both of Astra's checks. Likely selection on reused held-out text.
- G4 attempt: fails as reported.
- No checkpoint preserves predictions on Python code to 98.5%.

## Probes (G3-robust)
- Format-locked: 0/1000 after a text prefix, 0/500 two-digit, 0/500 natural "123+456=".
- Noise (multiplicative, edited scalars, 400 problems per draw): 1e-6 median 99.2% (8 draws, min 89.25%); 1e-5 median 64.5% (16 draws, 37–97%); 1e-4 ≈ 0%. Original G3 at 1e-5: median 1%.
- Precision: float64 200/200; bfloat16 0/200; float16 0/200 (overflow).
- Hugging Face GPT2LMHeadModel: 200/200 with exact GELU and with the standard tanh GELU.
- Batch size 1 vs 64: identical.

## Audit of the build
- 76 build scripts traced through manifests for all checkpoints; every hash matches.
- No gradients, optimizers, finite differences, search, model construction outside the meters, use of score.py's problem generator or seeds, or held-out paths.
- Closed-form solves fit syntax labels, single-digit readouts, unembedding geometry or unlabeled calibration text; carry and digit logic are hand-written (e.g. carry clip at 9.25/9.75, units = s mod 10).
- No held-out text in any forward-meter input: 116 inputs share 8-word sequences with WikiText-2 test, all also present in WikiText-2 train (calibration); 0 test-only matches. Moby-Dick: one common phrase.

## Disclosures for any write-up
- 29 of the 76 scripts evaluate one edited layer piece (MLP, LayerNorm, twice an attention pattern) on activations recorded through the meter. Allowed by the rules, never the full model.
- G3-robust's 4,252,159 changed scalars: 1.6% programmed MLP neurons (44 neurons), 46.0% text-repair output columns fitted in closed form to unlabeled calibration text, 52.1% attention rewrites in layers 1–2.
- Held-out text was reused across 136 score runs; G1 shows the resulting optimism.
- The prompt's rule to drop WikiText paragraphs containing `<unk>` left ~5,900 held-out WikiText tokens, and a third of held-out windows came from that file.
- Budget: 300 forward passes and 140 score runs, against MIT's 50 passes. Astra wrote its own meters (reviewed: correct).

## Boundary audit (session log rollout-2026-11-09T09-12-03-01a1bf5d…, 17,130 lines, 9 Nov 08:13 to 10 Nov 18:20 UTC)
- Settings on every turn: gpt-6-astra, effort max, sandbox workspace-write, network off, approval on-request with auto-review. Zero escalation requests.
- 1,453 tool calls containing 1,882 shell commands and 165 file patches. Tools used: shell, file patches, clock, goal status. No web search, subagents, apps or browser tools.
- Network: no curl, wget, pip, git or URLs in any command.
- Files: every absolute path and all 167 patch targets are inside astra-implant/gpt2. Outside gpt2 it only ran ../.venv/bin/python and copied ../model.py once at setup; nothing was written outside gpt2. It never used run 1's notes or scripts.
- Held-out text: only tools/score.py reads it; every other mention is a note or a file listing that excludes it.
- No apps, osascript, kill, sudo, caffeinate, Ollama or other models.
- Typed messages: only the /goal and the priority update at 16:46 CET on 10 Nov.
- Fast mode (priority tier) was on from 20:16 to 22:11 CET on 9 Nov; the run hit the usage limit at 09:28 CET on 10 Nov and resumed at 09:30.

Run 2 is fully verified.
