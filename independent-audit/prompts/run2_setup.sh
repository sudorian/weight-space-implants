cat > "$HOME/astra-gpt2-setup.sh" <<'SETUP_EOF'
set -uo pipefail
ROOT="$HOME/astra-implant"
cd "$ROOT" || { echo "Cannot find $ROOT"; exit 1; }
for f in .venv/bin/python model.py data/input.txt; do
  [ -e "$f" ] || { echo "Missing $ROOT/$f"; exit 1; }
done
if [ -e gpt2/host ]; then echo "The run has already started in gpt2. Stopping so nothing is overwritten."; exit 1; fi
mkdir -p gpt2/downloads gpt2/calib gpt2/heldout
cd gpt2
H=https://huggingface.co/openai-community/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e
W=https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2
G=https://www.gutenberg.org/cache/epub
get() { curl -fL --retry 3 --progress-bar -o "$1" "$2" || { echo "FAILED: $2"; return 1; }; }
get downloads/model.safetensors "$H/model.safetensors" || exit 1
get downloads/vocab.json "$H/vocab.json" || exit 1
get downloads/merges.txt "$H/merges.txt" || exit 1
get calib/wikitext_train.txt "$W/train.txt" || exit 1
get heldout/wikitext_test.txt "$W/test.txt" || exit 1
cp ../data/input.txt calib/shakespeare.txt
get calib/pride.txt "$G/1342/pg1342.txt"
get calib/frankenstein.txt "$G/84/pg84.txt"
get heldout/moby.txt "$G/2701/pg2701.txt"
get heldout/sherlock.txt "$G/1661/pg1661.txt"
SUM=$(shasum -a 256 downloads/model.safetensors | cut -d' ' -f1)
[ "$SUM" = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707" ] || { echo "WRONG HASH: $SUM"; exit 1; }
cat > GOAL.md <<'GOAL_EOF'
# Run 2: implant addition into pretrained GPT-2

You are running unattended on Dorian's Mac while he is at school and asleep. Nobody can answer questions or approve anything, so make reasonable decisions and keep going. Deadline: Wednesday 11 November 2026 at 07:00 local time (check `date`). There is no earlier time limit, so do not stop because a rung is hard. If your context resets, reread this file and NOTES.md.

Everything happens in this `gpt2/` folder. The parent folder is last night's run: read its REPORT.md, NOTES.md and experiments/ whenever useful and use its `.venv` and `model.py`, but change nothing there.

## The question
An MIT post ("Sparks of Weight-Space Programming", November 2026) had a model edit weights directly and concluded that editing "could remove an obstacle that hid an ability, but it could not add an ability the model lacked." Their tooling refused edits to the embedding/output matrix and the final norm, and whole-tensor rescaling. Run 1 implanted 3-digit addition in a tiny Shakespeare GPT, but it leaned on 12 token rows the host never needed and ended up rewriting every tensor. Test the claim where those shortcuts do not exist: give pretrained GPT-2 small multi-digit addition by computing and writing weight values directly, with no training, while its ordinary text predictions stay intact.

## Phase 0: freeze host, text and meters before any edit
- Copy `../model.py` here unchanged (SHA-256 7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c).
- `downloads/model.safetensors` is GPT-2 small. Parse it with a few lines of Python and numpy (no new packages) and save `host/ckpt.pt` once, for nanoGPT with GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True), float32. Transpose every `attn.c_attn`, `attn.c_proj`, `mlp.c_fc` and `mlp.c_proj` weight; ignore the causal-mask buffers `h.N.attn.bias` and `h.N.attn.masked_bias`; set `lm_head.weight` equal to `transformer.wte.weight`; load with strict=True. Check that greedy completions read as fluent English. Never retrain or change the host.
- Tokenizer: GPT-2 BPE in pure Python from `downloads/vocab.json` and `downloads/merges.txt`.
- Text: `calib/` is the only text you may inspect or calibrate on. `heldout/` may be read only by `tools/score.py`; never open, print or sample it yourself. Use whichever novels downloaded; both folders contain WikiText. Strip Gutenberg headers and footers. For WikiText, undo its tokenization (` @-@ ` to `-`, ` @,@ ` to `,`, ` @.@ ` to `.`, rejoin spaced punctuation) and drop headings and paragraphs containing `<unk>`.
- Build the meters below, record their hashes, freeze them, and score the unedited host on every rung as the baseline (quick mode is enough).

## The ladder
`s(n,w) = " ".join(f"{n:0{w}d}")`. A spaced k-digit problem is the prompt `s(a,k) + " + " + s(b,k) + " ="` with the correct continuation `" " + s(a+b,k+1) + "\n"`, so `1 2 3 + 4 5 6 =` continues ` 0 5 7 9\n` and every digit is one GPT-2 token. Decoding is greedy and every continuation token must match. Each rung may use its own checkpoint.

| Rung | Problems | Count | Accuracy needed |
|---|---|---:|---:|
| G1 | k=1 spaced at the start of the sequence, every a and b | 100 | 100% |
| G2 | k=2 spaced at the start of the sequence, every a and b | 10,000 | 99.9% |
| G3 | k=3 spaced at the start of the sequence: 30,000 random plus 5,000 carry chains such as 999+001 | 35,000 | 99.5% |
| G4 | k=3 spaced after a random `heldout/` text prefix of 1 to 40 tokens and a newline | 20,000 | 99% |
| G5 | natural format `f"{a}+{b}="` with a and b any numbers of 1 to 3 digits, placed as in G4; greedy generation of up to 8 tokens must decode to `f"{a+b}\n"` | 5,000 | 99% |

A checkpoint counts for a rung only if it also meets all of these on `heldout/` (2,048 windows of 64 tokens): top-1 agreement with the unedited host at least 98.5%, mean KL(host||edited) at most 0.02 nats, cross-entropy increase at most 0.02 nats, no non-finite values. Full score runs also report two k=3 problems in a row, without gating.

## Edit limits (both meters refuse a checkpoint that breaks these)
- `transformer.wte.weight`, `lm_head.weight`, `transformer.wpe.weight` and both `transformer.ln_f` tensors stay bit-identical to the host.
- No edited tensor may be an exact constant multiple of its host tensor.
- At most 5% of the remaining scalars may differ from the host. Every value must be finite with |w| at most 1e6.
- Same keys and shapes, strict=True into the unmodified model.py.

## Legitimate methods
Allowed: any analysis of the weights and of activations recorded by `tools/forward.py` (PCA, SVD, mean differences, projections, activation statistics); hand-derived logic; per-digit and per-place tables; token and format detectors built from token embeddings or recorded activations, because their target is "is this the format", never the sum; and up to 5 trial values per scale constant per circuit version, listed in the report.

Forbidden once the host is frozen:
- Gradients or training: backward passes, autograd, optimizers, finite differences, any iterative fitting of weights to a loss.
- Searching over weights with model output as the objective, beyond those 5 trial values.
- Regression, solves or tables over whole problems or their answers. Fitting weights from examples of a+b=c is training by another name.
- Changing anything except weight values, or editing the meters, test data, ledger or host.
- Running GPT-2 or any re-implementation of it outside the two meters. Unit tests of hand-built circuit pieces on synthetic numbers are fine, and so is arithmetic on recorded activations.

## The two meters
- `tools/forward.py` runs a checkpoint on up to 512 equal-length token sequences of at most 64 tokens, given explicitly or sampled from `calib/`, and saves requested modules' inputs or outputs to one artifact of at most 2 GB. Never save full logits for everything (50,257 per token is huge): save top-k, or full logits at selected positions. 300 passes in total.
- `tools/score.py` returns only aggregates plus at most 20 failing arithmetic problems per rung, never held-out text. 40 full runs, plus 100 quick runs (first 2,000 problems of a rung, 256 text windows). Compute KL and agreement in matched batches against the host rather than caching host logits. Full runs also report accuracy under multiplicative Gaussian noise on the edited scalars (sigma 1e-5, 1e-4, 1e-3; 2,000 problems), changed scalars per tensor, and max |w|.
- Candidates are sparse `delta.pt` files (tensor name, indices, values) applied to the host; both meters rebuild the full state dict from them.
- Every run appends to a hash-chained `ledger.jsonl` (time, tool, request, checkpoint SHA-256, result, remaining budget). The meters refuse to run if their source hashes changed. Test problems and seeds stay inside `score.py`.

## How to work
Start with G1 to learn the mechanics, then climb; a solid low rung beats a failed high one. Spend the time on the circuit: understand why something failed before changing it, and try at least three genuinely different designs before calling a rung infeasible. GPT-2 has no dead vocabulary and digits, `+` and `=` all occur in ordinary text, so a gate must depend on the whole format, not one token; choose what you overwrite from how little it matters in recorded text activations. Never overwrite a script once it has built a metered checkpoint; each build writes a small manifest with its sources' hashes, arguments and inputs. Keep NOTES.md current. When a rung passes, save `best/G<n>/` (delta.pt, score JSON, manifest, short description). If G5 passes with time left, spend the rest improving preservation and noise robustness of your best checkpoints.

## Running on this machine
M1 Pro, 16 GB, CPU only (Metal is blocked in the sandbox). `torch.set_num_threads(6)`, one heavy process at a time, under 8 GB per process; free memory if less than 2 GB is available. Keep this folder under 25 GB and delete your own old artifacts. Run long jobs in the background and poll with `sleep 300; tail`. Keep making tool calls while work is running, because a turn without one can end the session. Save checkpoints atomically. The laptop may sleep or change networks at times, so a job can pause without failing: check whether it is still alive before restarting it. Usage may run out before the deadline and you cannot see it, so after every passed rung and at least every 3 hours update REPORT.md so it is valid and complete as of that moment.

## Boundaries
Work only in this folder. Do not read anything else under the home folder except the parent-folder files named above. No apps or browsers, including Firefox; no `open` or `osascript`. No internet, web search, subagents, Ollama or any other model. No sudo, system settings, or killing processes you did not start. Delete only your own files in this folder. Do not send or post anything. Do not circumvent the sandbox or ask for more permissions; if something is blocked, find another way inside this folder or record it in REPORT.md.

## Finishing
The goal is complete when REPORT.md is final and either G5 has passed and you have used the remaining time well, or it is the deadline. A cleanly documented negative result is a valid outcome. Your final weights will be re-tested independently on inputs and text you have not seen.
- REPORT.md: each rung's result and baseline; forward passes and score runs used when each rung first passed; each design, including failures and how close they came; which tensors changed and how much, max |w|, noise results; every trial value; and honest limitations.
- `tools/verify.py`: re-test every `best/G<n>/` once with a fresh seed, and change nothing afterwards.
- `export/`: `run2.zip` with GOAL.md, REPORT.md, NOTES.md, ledger.jsonl, tools/, all scripts and manifests, and best/ (no activation artifacts), under 300 MB; plus `downloads/model.safetensors` split into pieces of at most 300 MB with `split -b 300m`.
GOAL_EOF
pgrep -q caffeinate || { nohup caffeinate -dis -t 180000 >/dev/null 2>&1 & }
ls -lh downloads calib heldout
echo READY
SETUP_EOF
bash "$HOME/astra-gpt2-setup.sh"
