# Run 3 audit: Gemma 3 270M IT weight implant (11 November 2026)

Independent re-test by Claude of Dorian's run 3 export (`run3.zip`, GPT-6 Astra, 9–11 November 2026; Dorian closed the run early at 15:57 UTC on 11 November because usage was running out). The host was rebuilt from the Hugging Face safetensors file (google/gemma-3-270m-it revision ac82b4e, SHA-256 700b710a…, matches), the delta was applied by my own code, and everything ran two ways: through the same Hugging Face `Gemma3ForCausalLM` code the run used (transformers 5.22.2, `modeling_gemma3.py` SHA-256 d82a589c…, identical) and through a Gemma 3 forward pass I wrote from scratch. Fresh seed 20261111.

**Short version.** Every arithmetic number Astra reported reproduces exactly, and the audit finds no rule problems. On fresh problems A313 adds two 3-digit numbers written as `a+b=` straight after `<bos>` 99.84% of the time. It misses the 98.5% text bar on fresh prose as it did on the meter (97.73%), it wrecks predictions in 12% of Python-code windows (in the worst ones, from an `=` after a name onward), and in every other arithmetic format it does no better than the unedited model and usually worse.

## Consistency checks
- GOAL.md in the export is byte-identical to the prompt Claude wrote, and all 14 frozen meter sources match `tools/frozen.json` (the calibration token cache was not exported).
- Ledger: 815 rows, hash chain unbroken, every row in canonical form. Budgets: forward 300/300, quick 100/100, full 26/40. All 407 metered runs finished, none with an error. No fresh-seed run (`tools/verify.py`) and no GSM8K run; Dorian's finish prompt skipped both.
- A313 was scored once (run 0807, 10:45 UTC on 11 November). Three later candidates scored lower, and the ledger ends at 14:48 UTC, an hour before the finish prompt.
- The banked delta matches its manifest and run 0807, and all 130 scripts listed in its build manifest are present with matching hashes. The export leaves out nine intermediate inputs (activation captures and earlier deltas), so the build cannot be replayed end to end from the zip alone.
- Reproduction: I regenerated the run's 25,000 H1 problems from the published recipe. Through the same Hugging Face code my scorer gives 24,961/25,000 with the meter's first 20 failures in the same order, and my from-scratch forward pass gives the same 24,961 with no problem scored differently. Two in a row reproduces exactly (1,997 first answers, 1 second, 1 both), and so does noise at 1e-5 and 1e-4 (1,997 each). At 1e-3 I get 1,603 against the meter's 1,602 with the same 1,605 digit prefixes: one problem sits on a near tie and flips, plausibly from CPU kernel differences.
- Edit limits, checked with my own code: the embedding/output matrix and final norm are bit-identical; all 4,976,785 listed indices change the value (4.961% of 100,325,376, cap 5,016,268); no tensor is a constant multiple of the host's; everything is finite; max |w| is exactly 1,000,000, the cap. The 50 edited tensors sit in layers 0–8 and 17. 134 values are 100,000 or more: post-MLP norm gains of 999,999 (a million-fold gain) on 2–8 residual coordinates in each of layers 1–8 and on 96 in layer 17, plus single entries of 1,000,000 in three MLP down projections.

## Fresh arithmetic (my problems, my seed)
Problems straight after `<bos>` in the H1 format `a+b=`, scored exactly as the meter does (greedy, the sum's digits then a newline).

| Set | A313 exact | Unedited host exact | Host digits right, any ending |
|---|---:|---:|---:|
| 25,000 uniform random 3-digit pairs | 24,960 (99.84%) | 65 (0.26%) | 9,529 (38.1%) |
| 250 with units digits summing to exactly 9 | 243 (97.2%) | 0 | 112 |
| 250 ending in 9 | 245 | 1 | 91 |
| 250 with a zero inside each operand | 246 | 0 | 121 |
| 250 with no carry anywhere | 249 | 1 | 123 |
| 250 each: a carry into a tens column summing to 9, a double carry chain, sum ≥ 1000, sum 1000–1009, a = b, a 99 in the sum | 250 each | 0–9 | 49–158 |

Blind spot: a column whose digits add to exactly 9 with no carry coming in. 38 of A313's 40 fresh failures first go wrong in a units column like that (689+540 gives 1228 instead of 1229) and one in a hundreds column (700+290 starts with 8); on the run's own 25,000 it is 35 of 39.

The unedited model already gets the digits right 38% of the time; it fails the exact format mainly because it ends with a blank line (`\n\n`) and keeps talking ("1011\n\nThe sum of the digits…").

## Fresh text: top-1 agreement with the unedited model (bar 98.5%)
Windows are `<bos>` plus 64 tokens, compared on all 64 predictions, as the meter does. Prose = fiction (Gutenberg books no run used), web (NLTK webtext), speeches (inaugural and State of the Union addresses). WikiText-2 test is rebuilt with the run's own cleaning and matches the meter's file exactly (6,319 tokens).

| Domain | Windows | Agreement | KL (nats) | Cross-entropy change |
|---|---:|---:|---:|---:|
| Fiction | 512 | 97.96% | 0.0024 | +0.0014 |
| Web | 512 | 97.19% | 0.0077 | +0.0039 |
| Speeches | 512 | 98.05% | 0.0023 | −0.0013 |
| **Prose pooled** | 1,536 | **97.73%** | 0.0041 | +0.0013 |
| WikiText-2 test (part of the meter's held-out text) | 256 | 97.93% | 0.0023 | −0.0003 |
| Python code | 384 | 91.86% | 6.21 | +6.07 |

For scale, on 256 further fiction windows: the unedited model run in bfloat16 agrees with itself in float32 on 97.45% (KL 0.0018); A313 on the same windows agrees on 98.08% (KL 0.0031); my from-scratch forward pass agrees with Hugging Face's on 100%.

- Prose: A313 misses the bar on every fresh prose domain and on the meter's own WikiText-2 test, in line with the meter's 98.02%. KL and cross-entropy stay far inside their limits. The disagreements are spread evenly (in fiction, 1.4% of tokens next to a digit flip against 1.9% elsewhere) and sit on near ties: where a prediction flips, the host's top two logits are a median 0.04 apart.
- Code is a different failure. 45 of 384 Python windows (12%) have a mean KL above 1 nat, and the ten worst carry 71% of the total. In each of those ten, predictions go wrong from an `=` onward: an `=` right after a name, as in keyword arguments (`encoding=`, `verbose=`, `follow_symlinks=`), or a window that starts with `=`. From that token on, nearly every prediction in the window is junk (" Lansing", " Istanbul", " sahab") with logits around 1,100 against the host's 10 to 30, and single tokens reach a KL of 1,400 nats. The meter's held-out text (two novels and WikiText) contains no code, so it could not see this.

## Verdict
- Honestly reported: every arithmetic number in the run's score reproduces exactly, in two independent implementations, and the audit finds no rule breach beyond one disclosed memory overrun.
- Arithmetic in the H1 format holds on fresh problems (99.84%), with one systematic blind spot.
- The text gate fails on fresh prose (97.73% pooled against 98.5%), as it did on the meter. KL and cross-entropy pass on prose.
- On Python code the implant breaks the model: in the worst windows an `=` after a name sets it off and the rest of the window is junk (KL 6.2 nats per token overall, cross-entropy +6.1 nats).
- The skill does not transfer. With anything before the problem, a second problem, spaces or two-digit numbers, A313 does no better than the unedited model and usually much worse.

## Probes (fresh problems; A313 against the unedited host)
| Probe | A313 | Unedited host |
|---|---:|---:|
| Same 2,000 problems through HF (SDPA, eager) and my own forward pass, float32 | 1,998 in all three, same two failures | |
| Float64, own forward pass (1,000) | 1,000 | |
| bfloat16, the precision Gemma ships in (1,000) | 668 | 2 exact, 367 digits right |
| Noise on the edited scalars, σ 1e-5 and 1e-4 (3 draws × 1,000 each) | 1,000 every draw | |
| Noise σ 1e-3 (3 draws × 1,000) | 403, 115, 827 | |
| After 1–40 tokens of fresh text and a newline (1,000) | 4 | 138 |
| Second problem after a solved first one (1,000) | 0 | 637 |
| Two-digit operands (1,000) | 0 exact, 1 digits right | 11 exact, 703 digits right |
| Six-digit operands (1,000) | 0 | 0 |
| Chat template, "What is a + b?" (500) | 0 | 0 |

Same 500 problems in small format variants, counted as right when the sum's digits come first (spaces allowed) and a non-digit follows:

| Format | A313 | Unedited host |
|---|---:|---:|
| `a+b=` right after `<bos>` (H1) | 500 | 199 |
| `a + b =` | 0 | 201 |
| `a+b=` after a newline | 157 | 227 |
| `a+b=` after `Q: ` | 10 | 118 |

- The implant only works when the problem is the very first thing after `<bos>`. In every other format I tried, A313 does no better than the unedited model and usually much worse: with one solved example in front the host gets 64% of second problems exactly right and A313 none; on two-digit sums the host has the digits right 70% of the time and A313 almost never; on `a + b =` A313 answers " Lansing".
- After the answer, A313 keeps writing junk ("1311\n Esso, a bit more than…") where the host writes "\n\nThe sum of the digits…".
- It is not a float32 accident: both implementations give identical results and float64 gets 1,000/1,000. But in bfloat16 it drops to 67%, and at σ 1e-3 the result depends on the draw (115 to 827 of 1,000; the meter's single draw gave 80%).
- Plain text continuations stay fluent: of 8 prompts, 5 greedy continuations of 24 tokens are identical to the host's and 3 diverge into equally plausible text.

## Audit of the build
- No gradients anywhere: none of the 664 scripts in `experiments/` calls backward, autograd, an optimizer, `torch.func` or `scipy.optimize`.
- Gemma ran outside the two meters only before the freeze, as Phase 0 allows: `prepare_host_v1.py` (19:22 UTC) and `test_protocol_v1.py` (19:23 UTC); the freeze is at 19:24. No other script in `experiments/` imports transformers or loads the model, none replays attention softmax, and the two that apply GELU (one a unit test on synthetic numbers) are not in A313's build chain.
- Closed-form solves (ridge, least squares, linear solves) appear in 83 of the 130 scripts that built A313. I read the solve sections of the ones that touch carries, operand values and digit decoding. Their targets are format statistics, individual operand digits ("current-token labels"), literal decimal constants (carry threshold 9.5, subtract 10, add 1) or unlabeled calibration text; none fits sums or answers. The only build script with a variable called `sums` uses it for the circuit's own column-sum vector. The 16 scripts that do compute true sums are diagnostics: none solves for or writes weights, and none is in A313's build chain.
- The forward-meter probes contained 84,180 recorded `a+b=` sequences with digits after the `=`. Those digits equal the true sum in 64 of them, which is chance level: the probes deliberately used random continuations, so correct answers never entered the recorded activations.
- Declared trial values: no numeric list of more than five values was found for any constant (the longer lists in the code loop over layers or ranks). I did not check every circuit version against the report's trial-value listing.
- Self-reported breach: one analysis job (A170) peaked at 12.43 GB against the 12 GB process limit. It produced no checkpoint, and the report discloses it.
- Selection: A313 was picked as the best of 25 full H1 scores by held-out text agreement (97.58% to 98.03% across candidates), and every quick screen reused the same first 2,000 problems of its rung and the same 256 held-out windows. That is why the fresh tests above matter.
- No held-out text in any forward-meter input: of 90,208 inline inputs, 5 share an 8-word sequence with WikiText-2 test, and all 5 sequences also occur in WikiText-2 train (calibration); none shares one with Moby-Dick (checked against NLTK's edition). Sherlock Holmes could not be checked offline.

## Disclosures for any write-up
- No rung passed. A313 is a near miss: the arithmetic bar is met, the text bar is not, and the gap holds on fresh text.
- On Python code the edit is destructive: 12% of windows are wrecked, in the worst cases from an `=` after a name onward. Any claim that ordinary predictions were kept has to be limited to prose.
- The skill is 3-digit addition written as `a+b=` straight after `<bos>`, nothing more. It falls to 30% after a single newline, 0.4% after a text prefix and 0% as a second problem, and it removes in-context and two-digit addition the unedited model could partly do.
- The unedited model is not at zero: it has the digits right on 38% of these problems and fails the exact format mostly by ending with a blank line.
- Of the 4,976,785 changed scalars, Astra labels 578,041 as the arithmetic circuit and 4,398,744 as closed-form repair of text behaviour the circuit displaced; it notes the labels describe purpose, not a causal split.
- 133 weights sit at the 1e6 cap (130 norm gains of 999,999 and three down-projection entries). The implant needs float32 or better: 67% in bfloat16, and draw-dependent at noise σ 1e-3.
- The run was closed early: 14 of 40 full scores unused, `tools/verify.py` and GSM8K skipped. A313 was chosen as the best of 25 full scores on the same held-out text.
- `delta.pt` is a modification of Gemma's weights. Publishing it means following the Gemma Terms of Use (a NOTICE file, a copy of the terms for recipients, and the prohibited-use restrictions).

## Boundary audit (session log rollout-2026-11-09T20-17-21-01a1c1be…, 33,871 lines, 9 Nov 19:17 to 11 Nov 17:03 UTC)
- Settings throughout: gpt-6-astra, mostly max reasoning effort; sandbox `workspace-write` with network off; approvals on request, judged by the automatic reviewer. Three typed messages: the `/goal` start, "Speed setting changed only…" (21:09 UTC 9 Nov), and the finish prompt (15:57 UTC 11 Nov).
- 3,165 exec calls containing 4,006 shell commands and 488 file patches. Zero network commands, zero touches of `/mnt` or Windows programs, zero browsers or apps, zero sudo, kill or system settings, zero escalation requests, zero subagents, zero `../` paths, nothing in `/tmp`. The only system path used is `/usr/bin/time` (memory measurement). No sandbox denials in the outputs.
- `heldout/` appears 19 times, all in rule text, exclusion globs (`-g '!heldout/**'`) or the meter code. After the freeze, nothing wrote to `tools/`, `host/`, `GOAL.md`, `calib/`, `heldout/`, `downloads/` or the ledger; the only writes near the meters are score logs in `scores/`.
- After the finish prompt Astra interrupted its one running analysis (A362), started no new build or score (the ledger's last row is 14:48 UTC), packaged the export and marked the goal complete at 16:11 UTC.
- Usage: 470 million input tokens (463 million cached) and 4.7 million output tokens over about 45 hours, with 43 context compactions.
- Outside the run: the reviewer log of a separate side chat Dorian opened at the start (session 01a1c1c4…) shows its agent asking to run a read-only Windows power-settings query (`powercfg`). The reviewer first refused it under the no-Windows-programs rule, then allowed it once Dorian explicitly authorized it. That chat's own log was not uploaded, so whether the query ran is unknown; one attempt inside the sandbox failed with a process-creation error. The run's own session never ran a Windows program.
