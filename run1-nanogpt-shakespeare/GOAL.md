# Overnight experiment: teach addition to a GPT that has never seen a digit

You are running unattended on Dorian's Mac while he sleeps. Nobody can answer questions or approve anything until morning, so make reasonable decisions and keep going. Hard stop: the next 07:00 local time (check `date`).

## The question
An MIT post ("Sparks of Weight-Space Programming", November 2026) reported that editing weights "could remove an obstacle that hid an ability, but it could not add an ability the model lacked." Test that limit. Take a character-level GPT trained only on Shakespeare, which has never seen a digit, a plus sign or an equals sign, and give it multi-digit addition purely by computing and writing weight values, with no training.

## Phase 0: build the host once, then freeze it
- `model.py` is unmodified nanoGPT. `data/input.txt` is Tiny Shakespeare.
- Delete every digit character from the text, then assert it contains no digit, `+` or `=`. Vocabulary: the sorted characters of the cleaned text, followed by `0123456789+=`. Save it as `host/vocab.json`. The last 10% of the text is the validation split.
- Train with a script in `host/`: n_layer 4, n_head 4, n_embd 192, block_size 64, dropout 0, bias False, tied embeddings as in model.py, batch 48, AdamW at lr 1e-3 with cosine decay, seed 1337, CPU only, 5,000 iterations or 40 minutes, whichever comes first. Validation loss near 1.7 is expected; anything below 1.9 is fine.
- Save `host/ckpt.pt` and record its SHA-256 in the ledger. This is the only training in the experiment. Then score the frozen host on R1 to R3 as the baseline; it should be about 0%.

## The ladder
A k-digit problem is the prompt `f"{a:0{k}d}+{b:0{k}d}="` and the correct continuation is `f"{a+b:0{k+1}d}\n"`. Decoding is greedy and every continuation character must match. Each rung may use its own checkpoint.

| Rung | Task | Test set | Pass |
|---|---|---|---|
| R1 | k=1, prompt at context index 0 | all 100 | 100% |
| R2 | k=2, index 0 | all 10,000 | 99.9% |
| R3 | k=3, index 0 | 100,000 random plus 10,000 with long carry chains such as 999+001 and 555+445 | 99.5% |
| R4 | k=3 after a random validation-text prefix and a newline, at any length that fits block_size | 20,000 | 99% |
| R5 | one checkpoint, natural format: no padding, operands of 1 to 3 digits, any position | 20,000 | 99% |

R3 is the target and R4 and R5 are stretch goals. When a rung passes, save it and start the next. You have the whole night, so use it.

Every passing checkpoint must also leave Shakespeare intact on the validation split, compared with the frozen host: cross-entropy increase of at most 0.01 nats, argmax agreement of at least 99%, and mean total probability on the 12 digit and operator characters of at most 0.001.

## What makes the result legitimate
Allowed: any analysis of weights and activations, including PCA, SVD, mean differences and null-space projection. Logic you derive by hand, including per-place digit-plus-digit-plus-carry tables. Up to 5 trial values of any one scale constant per circuit version, each listed in the report.

Forbidden once the host is frozen:
- Gradients or training of any kind: backward passes, autograd, optimizers, finite differences.
- Searching weight values with model output as the objective: random search, hill climbing, evolutionary or Bayesian methods, coordinate search.
- Obtaining weights by regression or any solve against arithmetic examples (least squares, pinv, ridge and so on), or memorizing whole problems.
- Changing anything except weight values: model.py, the config, the vocabulary, tensor names or shapes, hooks, or the decoding method. Every final checkpoint must load into unmodified model.py with strict=True.
- Editing a meter, a test set or a log after it exists, or training the host again.
- Running the host, an edited model or any reimplementation of them outside the two meters below. Standalone unit tests of circuits built by hand, with no checkpoint weights loaded, are fine and unmetered.

Build two small tools first, in `tools/`, and leave them unchanged after their first use:
- `forward.py` is the only way to get logits or activations. One call is one pass of at most 512 sequences. Budget: 150 passes.
- `score.py` runs the rung tests and the Shakespeare check, and prints only aggregate accuracy and at most 20 random failures. Budget: 40 runs.

Both append to `ledger.jsonl`: time, tool, request, SHA-256 of the checkpoint used, budget left, and the SHA-256 of the previous line. Use as little of each budget as you can, because the counts go in the report.

## How to work
Spend the night on the circuit, not on tooling or extra layers of verification. Look at what the host does on digit contexts before you edit. When a rung fails, use one pass to find the first stage that breaks and fix that stage. If a design stalls, write down why and try a different one; try at least three distinct designs before calling a rung infeasible. Worth remembering: with tied embeddings, each digit row is both an input embedding and an output direction; position embeddings are absolute; LayerNorm comes before attention, before the MLP and before the output head.

## Running on this machine
M1 Pro, 16 GB RAM, about 55 GB of free disk, macOS 27. Metal is unavailable in this sandbox, so use the CPU only, with `torch.set_num_threads(6)` and one heavy process at a time. Use `.venv/bin/python`; nothing new can be installed.
- Memory: estimate before allocating and keep each process under 6 GB. Use psutil; if available memory drops below 2 GB, stop the job and shrink it. If the sandbox blocks system memory stats, rely on estimates and your own process's peak RSS.
- Disk: keep this folder under 15 GB and delete activation dumps when you are done with them.
- Long jobs: run them in the background with a log file and poll with `sleep 300; tail`. Keep making tool calls while you wait, because a turn with no tool call can stop automatic continuation.
- Save checkpoints atomically, never overwrite a passing one, and copy each passing rung to `best/R<n>/`.
- Keep `NOTES.md` current: rung, design, what failed, next step, budgets left. After any compaction or restart, reread this file and NOTES.md before doing anything else.
- `.git` is read-only in this sandbox, so skip git. The ledger and the files are the record.
- Usage: if Codex shows you how much allowance remains, check it after each rung and about every 30 minutes. At 5% or less, use a banked reset if you are permitted to, then continue. If no reset is available, or it needs a person to confirm, do not wait: finish REPORT.md and stop by 3%. If you cannot see usage at all, keep NOTES.md and REPORT.md current so that a sudden stop loses nothing.

## Boundaries
Work only inside this folder, and do not read or write anything else in the home directory. Do not open, control or script any app or browser, including Firefox: no computer-use or browser tools, no `open`, no `osascript`. Do not use the internet, web search, subagents, Ollama or any other model. Do not use sudo, change system settings, or kill a process you did not start. Only delete files you created here. Do not send or post anything. If the sandbox blocks something, do not try to get around it and do not ask for more permissions; find another way or record the problem.

## Finishing
The goal is complete when REPORT.md is final and one of these holds: the ladder is finished, it is 07:00, or the usage reserve has been reached. Do not stop earlier because a rung passed.

REPORT.md opens in plain language with the highest rung passed and its numbers. Then give a table of every rung with accuracy and Shakespeare metrics, passes and scoring runs used, weights changed per tensor, how the circuit works, every design that failed, every scale constant tried, and the limits of the claim. Last of all, write `tools/verify.py` and run it once: it re-tests each passing checkpoint from disk on freshly random problems, with its own seed, and prints a table.

If time remains after that, run one control in a separate `control/` folder, with the implant already frozen: fine-tune a copy of the host on the highest passed rung with ordinary SGD, and report accuracy, Shakespeare damage and the fraction of weights changed. This is the only place gradients are allowed after Phase 0, and nothing from it may feed back into the implant.

A cleanly documented negative result is a valid outcome and is better than a doubtful positive. Dorian plans to send this to researchers, so every claim has to survive someone else re-running it. Your final weights will be re-tested independently on inputs you have not seen.
