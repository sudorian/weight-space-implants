# Run 3: implant addition into Gemma 3 270M

You are running unattended on Dorian's Windows PC, inside WSL2 Ubuntu, while he is at school and asleep. Nobody can answer questions or approve anything, so make reasonable decisions and keep going. Deadline: Thursday 12 November 2026 at 07:00 local time (check `date`). There is no earlier time limit, so do not stop because a rung is hard. If your context resets, reread this file and NOTES.md.

Everything happens in this folder. `prior/`, if present, holds notes and scripts from a run that gave pretrained GPT-2 small 3-digit addition this way. Read and adapt anything in it.

## The question
An MIT post ("Sparks of Weight-Space Programming", November 2026) had a model edit Gemma 3 270M directly to improve its math. No edit moved MATH-500 by more than about a point, and they concluded that editing "could remove an obstacle that hid an ability, but it could not add an ability the model lacked." Their tooling refused edits to the embedding/output matrix and the final norm, and whole-tensor rescaling. They did not try implanting a narrow skill. Do that on this exact model: give Gemma 3 270M IT reliable long addition by computing and writing weight values directly, with no gradient training, while its ordinary text predictions stay intact.

## Phase 0: freeze host, text and meters before any edit
- Use the installed `transformers` `Gemma3ForCausalLM` unmodified, with `attn_implementation="sdpa"`, and record the SHA-256 of its `modeling_gemma3.py`. `downloads/` holds the model files. Convert `downloads/model.safetensors` to float32 once and save `host/ckpt.pt`; `lm_head.weight` is tied to `model.embed_tokens.weight`. Load strictly. Check that greedy completions read as fluent English. Never change the host.
- Tokenizer: `tokenizers` with `downloads/tokenizer.json`. Every sequence starts with `<bos>`. Confirm that each digit is its own token; if that is false, record it and score on decoded text.
- Text: `calib/` is the only text you may inspect or calibrate on. `heldout/` may be read only by `tools/score.py`; never open, print or sample it yourself. Use whichever novels downloaded; both folders contain WikiText. Strip Gutenberg headers and footers. For WikiText, undo its tokenization (` @-@ ` to `-`, ` @,@ ` to `,`, ` @.@ ` to `.`, rejoin spaced punctuation) and drop headings and paragraphs containing `<unk>`.
- Build the meters below, record their hashes, freeze them, and score the unedited host on every rung as the baseline (quick mode is enough). For the baseline also report how often the digits are right when the terminator is ignored, because this model can already add small numbers.

## The ladder
The prompt is `f"{a}+{b}="` and the correct continuation decodes to `f"{a+b}\n"`, with no padding. Decoding is greedy, up to 12 tokens. An operand with k digits has a nonzero first digit. Each rung may use its own checkpoint. Small unofficial tests are fine if they teach you the mechanics.

| Rung | Problems | Count | Accuracy needed |
|---|---|---:|---:|
| H1 | a and b both 3 digits, straight after `<bos>`: 20,000 random plus 5,000 carry chains | 25,000 | 99.5% |
| H2 | a and b both 3 digits, after a random `heldout/` text prefix of 1 to 40 tokens and a newline | 20,000 | 99% |
| H3 | a and b both 6 digits, placed as in H2: 15,000 random plus 5,000 carry chains | 20,000 | 99% |
| H4 | a and b each any length from 1 to 6 digits, placed as in H2 | 20,000 | 99% |
| H5 | chat template with the user message `f"What is {a} + {b}?"`, lengths as in H4; the reply must begin with the digits of the sum, followed by a non-digit token | 5,000 | 99% |

A checkpoint counts for a rung only if it also meets all of these on `heldout/` (2,048 windows of 64 tokens after `<bos>`): top-1 agreement with the unedited host at least 98.5%, mean KL(host||edited) at most 0.02 nats, cross-entropy increase at most 0.02 nats, no non-finite values. Full score runs also report two problems in a row, without gating.

## Edit limits (both meters refuse a checkpoint that breaks these)
- `model.embed_tokens.weight`, `lm_head.weight` and `model.norm.weight` stay bit-identical to the host, and all buffers stay unchanged.
- No edited tensor may be an exact constant multiple of its host tensor.
- At most 5% of the remaining 100,325,376 scalars may differ from the host. Every value must be finite float32 with |w| at most 1e6.
- Same keys and shapes, strict loading into the unmodified model class.

## Legitimate methods
Allowed: any analysis of the weights and of activations recorded by `tools/forward.py` (PCA, SVD, mean differences, projections, activation statistics); hand-derived logic; per-digit and per-place tables; token and format detectors built from token embeddings or recorded activations, because their target is "is this the format", never the sum; closed-form, non-iterative repair of displaced text behaviour, fitted only on unlabeled `calib/` text; and up to 5 trial values per scale constant per circuit version, listed in the report.

Forbidden once the host is frozen:
- Gradients or training: backward passes, autograd, optimizers, finite differences, any iterative fitting of weights to a loss.
- Searching over weights with model output as the objective, beyond those 5 trial values.
- Regression, solves or tables over whole problems or their answers. Fitting weights from examples of a+b=c is training by another name.
- Changing anything except weight values, or editing the meters, test data, ledger or host.
- Running Gemma or any re-implementation of it outside the two meters. Unit tests of hand-built circuit pieces on synthetic numbers are fine, and so is arithmetic on recorded activations.

## The two meters
- `tools/forward.py` runs a checkpoint on up to 512 equal-length token sequences of at most 64 tokens, given explicitly or sampled from `calib/`, and saves requested modules' inputs or outputs to one artifact of at most 2 GB. Logits are 262,144 wide, so compute and save them only at the positions you need. 300 passes in total.
- `tools/score.py` returns only aggregates plus at most 20 failing arithmetic problems per rung, never held-out text. 40 full runs, plus 100 quick runs (first 2,000 shuffled problems of a rung, 256 text windows). Compute KL and agreement in small matched batches against the host rather than caching host logits. Full runs also report accuracy under multiplicative Gaussian noise on the edited scalars (sigma 1e-5, 1e-4, 1e-3; 2,000 problems), changed scalars per tensor, and max |w|.
- Candidates are sparse `delta.pt` files (tensor name, indices, values) applied to the host; both meters rebuild the full state dict from them.
- Every run appends to a hash-chained `ledger.jsonl` (time, tool, request, checkpoint SHA-256, result, remaining budget). The meters refuse to run if their source hashes changed. Test problems and seeds stay inside `score.py`.

## How to work
Spend the time on the circuit: understand why something failed before changing it, and try at least three genuinely different designs before calling a rung infeasible. Differences from GPT-2 that matter: no linear layer has a bias term; four query heads share one key/value head per layer; positions come from rotary embeddings, not a table; the MLP computes `gelu(gate) * up`; norms are RMS with no centring. Digits, `+` and `=` occur in ordinary text, so a gate must depend on the whole format, not one token. Choose what you overwrite from how little it matters in recorded text activations. Never overwrite a script once it has built a metered checkpoint; each build writes a small manifest with its sources' hashes, arguments and inputs. Keep NOTES.md current. When a rung passes, save `best/H<n>/` (delta.pt, score JSON, manifest, short description).

At Wednesday 22:00 local, stop climbing wherever you are. Spend the remaining time on a larger text-agreement margin and better noise robustness for your highest passed rung, saved as a new checkpoint so the original pass stays untouched. Do the same earlier if H5 passes.

## Running on this machine
Intel i7-8700, 6 cores, 16 GB visible to WSL, CPU only; do not use the GPU. `torch.set_num_threads(6)`, one heavy process at a time, under 12 GB per process; free memory if less than 2 GB is available. Keep this folder under 40 GB and delete your own old artifacts. Run long jobs in the background and poll with `sleep 300; tail`. Keep making tool calls while work is running, because a turn without one can end the session. Save checkpoints atomically. Usage may run out before the deadline and you cannot see it, so after every passed rung and at least every 3 hours update REPORT.md so it is valid and complete as of that moment.

## Boundaries
Work only in this folder. Do not read anything else under the home folder. Never touch `/mnt` or run any Windows program (`.exe`, `cmd`, `powershell`, `explorer`). No apps or browsers, including Firefox. No internet, web search, subagents, Ollama or any other model. No sudo, system settings, or killing processes you did not start. Delete only your own files in this folder. Do not send or post anything. Do not circumvent the sandbox or ask for more permissions; if something is blocked, find another way inside this folder or record it in REPORT.md.

## Finishing
The goal is complete when REPORT.md is final and it is the deadline, or you have finished the robustness work and made no real gain for several hours. A cleanly documented negative result is a valid outcome. Your final weights will be re-tested independently on inputs and text you have not seen.
- REPORT.md: each rung's result and baseline; forward passes and score runs used when each rung first passed; each design, including failures and how close they came; changed scalars per tensor, split into the arithmetic circuit and text repair; max |w|; noise results; every trial value; and honest limitations.
- `tools/verify.py`: re-test every `best/H<n>/` once with a fresh seed, and change nothing afterwards.
- If time remains, compare the host and your best checkpoint once on `heldout/gsm8k_test.jsonl` (chat template, the question as the user message, greedy, at most 300 new tokens, grade the last number in the reply). Report it as information only; it is not a gate.
- `export/`: `run3.zip` with GOAL.md, REPORT.md, NOTES.md, ledger.jsonl, tools/, all scripts and manifests, and best/ (no activation artifacts), under 300 MB. Do not copy the Gemma weights into it.
