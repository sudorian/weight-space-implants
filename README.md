# Weight-space implants

Can an AI agent give a pretrained language model a skill it did not have by writing the weights directly, with no gradient training?

On 2 November 2026, MIT CSAIL's [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab) showed GPT-6 Astra repairing Qwen3.5-0.8B by editing 0.016% of its weights, which removed a looping failure and took MATH-500 from 7.9% to 48.1%. The same approach could not improve Gemma 3 270M, and the authors concluded: "Editing weights could remove an obstacle that hid an ability, but it could not add an ability the model lacked."

This repository tests a narrow version of that question. In three unattended runs, GPT-6 Astra (in Codex) tried to write exact 3-digit addition into a model's weights. Training, search and fitting weights to example problems were banned. Runs 2 and 3 also froze what MIT's Gemma setup refused to edit (the embedding/output matrix and the final norm), banned multiplying a whole tensor by a constant as MIT's did, and capped changes at 5% of the remaining weights, each within ±1e6.

The results below come from an independent re-test by Claude (Anthropic) using its own code, fresh problems and fresh text, not from the agent's own scoring; the one exception is GPT-2's unedited baseline, which run 2 measured itself. Dorian, a student, ran the experiments; the rules were drafted with Claude. None of us is affiliated with MIT, OpenAI, Anthropic or Google.

## Results

| | Model | Edit limits | Unedited model | Best implant, fresh problems | Text kept |
|---|---|---|---|---|---|
| Run 1 | character-level nanoGPT trained on Shakespeare, about 1.8M parameters | none on which tensors | 0% | 200,000 / 200,000; after a text prefix 40,000 / 40,000 | 99.04% on held-back Shakespeare for the prefix version (bar 99%) |
| Run 2 | GPT-2 small, 124M | frozen embeddings, positions, output matrix and final norm; ≤ 5% of other weights | 0 / 2,000 | 19,988 / 20,000 (99.94%) | 98.66% on fresh prose (bar 98.5%) |
| Run 3 | Gemma 3 270M IT | frozen embedding/output matrix and final norm; ≤ 5% of other weights | 65 / 25,000 exact (digits right on 38%) | 24,960 / 25,000 (99.84%) | 97.73% on fresh prose (fails the 98.5% bar) |

"Text kept" is top-1 agreement: the share of positions in 64-token windows where the edited model's most likely next token is the same as the unedited model's.

**Run 2 is the clean result.** Unedited GPT-2 small gets none of 2,000 three-digit additions right in the test format (`1 2 3 + 4 5 6 =` continued by ` 0 5 7 9`, one token per digit). After the edit it gets 99.94% of fresh problems right, while its next-token predictions on fresh prose (fiction, web text, speeches) match the unedited model's 98.66% of the time, with KL divergence and cross-entropy change both under 0.02 nats. Of the 4,252,159 changed weights, 44 programmed MLP neurons (1.6%) carry the hand-written carry logic, for example a carry clip at 9.25/9.75 and units = s mod 10. Rewritten attention weights in layers 1–2 make up 52%, and the remaining 46% are closed-form "text repair": output columns refitted on unlabeled calibration text so ordinary predictions stay close to the original.

**Run 3 is a near miss.** On Gemma 3 270M, the same kind of implant adds correctly (99.84%) but fails the text bar on fresh prose (97.73%). It also gives badly wrong predictions in 12% of Python-code windows, in the worst cases from an `=` after a name onward, and it makes the model worse at nearby tasks it could partly do before. Gemma already gets the digits right on 38% of these problems; it mostly fails the exact format by ending with a blank line.

**Run 1 was the warm-up.** It used a tiny model with no limits on which tensors could change. The implants for 1- to 3-digit addition touched only 12 unused embedding rows and one attention layer, but the version that also works after a text prefix rewrote 99.99% of all weights, embeddings included. MIT's editing tool would have refused both.

## What this does and does not show

Within one exactly specified format, run 2 adds an ability the model lacked, by writing weights directly and without training, under limits close to MIT's. That makes it a narrow counterexample to the quoted sentence, with these limits:

- **The skill is format-locked.** In run 2 it fails after any text prefix (0 of 1,000), on two-digit sums (0 of 500) and in the plain `123+456=` format (0 of 500). In run 3 it works only when the problem comes straight after `<bos>` (30% after a single newline, 0.4% after text), and with a solved problem in front the edited Gemma gets none of the second problems right, against 64% for the unedited model.
- **It is fragile.** Some weights sit close to the ±1e6 cap. In bfloat16, run 2's implant gets 0% and run 3's 67%. Multiplicative random noise of 1e-5 on the edited weights drops run 2 to a median of 64.5%.
- **The budget was larger than MIT's.** Their Gemma attempts allowed 5 to 50 forward passes; runs 2 and 3 allowed 300 forward-pass captures plus up to 140 scoring runs. The rules allowed every pass to be used for analysis and screening, though not for fitting weights to problems.
- **Held-out text was reused.** Each run picked its best checkpoint on the same held-out text it scored against. The fresh-text re-tests show the cost: one run 2 checkpoint (G1) passed Astra's own checks but fails on fresh prose.
- **No general math.** Nothing here touches benchmark-style math. This is a calculator-like circuit for one format, written into the weights.

## How the runs worked

Each run folder starts with `GOAL.md`, the full rules Astra worked under. In short, Astra could read and write weights with Python, but every execution of the model went through two metered tools, one for recording activations and one for scoring, with fixed budgets and a hash-chained ledger of every call. Closed-form solves were allowed for repairing text behaviour, fitted only on unlabeled calibration text. Solves or tables over whole problems or their answers were not allowed, and neither were gradients, optimizers or search. A checkpoint counted only if it also kept the model's text predictions within set limits on held-out text.

Claude then re-tested the final weights on separate hardware with separately written code: fresh problems from new seeds, fresh text from sources no run used, probes outside the specification, and checks of the edit limits. It also audited each run's ledger, build scripts and session log, checking for network use, access outside the run folder, training, and use of held-out data. The audits are in `independent-audit/`; all three runs passed them, apart from one memory overrun in run 3 that Astra disclosed itself.

## Related work

- **Compiling programs into transformers.** [Tracr](https://arxiv.org/abs/2301.05062) (2023) compiles programs into standalone transformers. [AdderBoard](https://github.com/anadim/AdderBoard) collects tiny hand-coded adders built from scratch.
- **Adding arithmetic to pretrained models.** [Tracr-Injection](https://arxiv.org/abs/2505.10719) (2025) puts compiled algorithms, including addition, into pretrained GPT-2-large, but by gradient fine-tuning. [IGC](https://arxiv.org/abs/2501.00684) (2025) trains a calculator module inside Llama 3.1 8B. [Neural Comprehension](https://arxiv.org/abs/2304.01665) (ICLR 2024) runs compiled, untrained networks as separate modules alongside a language model.
- **Follow-ups to the MIT post** that we found up to 12 November 2026 mostly repair or read existing networks. Examples are [Hu et al.](https://huggingface.co/papers/2611.09382) on Qwen3.5, [Ahmed](https://arxiv.org/abs/2611.08070) on weight reading, [Petcu](https://adrianpetcu.substack.com/p/weight-space-programming-tried-at) on a character classifier, and [Honnibal's critique](https://honnibal.dev/blog/astra-read-write-misleading). The closest to this repository is [circuit-injection](https://github.com/tmestery/circuit-injection) (Tyler Mestery, 4 November 2026). It uses a closed-form edit to GPT-2 small, with no training, to make the model answer the trigger `<<PING>>` with `PONG-42`: 9 of 10 held-out prompts, with 1 false trigger in 10. It also reports that circuits built by hand from first principles failed to restore an ablated induction circuit.

That edit installs a fixed response to a trigger, and fact-editing methods such as ROME insert fixed associations. We did not find an earlier case where a skill that has to compute its answer from the input, such as adding any two 3-digit numbers, was written into a pretrained language model's own weights without training. We may have missed one.

## Repository layout

- `run1-nanogpt-shakespeare/`, `run2-gpt2/`, `run3-gemma3-270m/`: each run's `GOAL.md`, `REPORT.md` (Astra's final report), `NOTES.md`, ledger, build scripts with manifests, meters, and best checkpoints (sparse weight deltas in runs 2 and 3).
- `independent-audit/`: `AUDIT_RUN1.md` to `AUDIT_RUN3.md`, the re-test scripts and their results, and the exact prompts that set up and closed the runs.
- `NOTICE`: licences of the model weights the deltas modify.

## Reproducing

No model weights are included. GPT-2 small is `openai-community/gpt2` at revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` (model.safetensors SHA-256 `248dfc39…`). Gemma 3 270M IT is `google/gemma-3-270m-it` at revision `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3` (SHA-256 `700b710a…`). `independent-audit/README.md` explains how to rebuild the hosts, apply the deltas and re-run every check on a CPU.

## Licence

Code and text are under the MIT licence (`LICENSE`). The `delta.pt` files modify GPT-2 weights (MIT licence) and Gemma weights (Gemma Terms of Use); see `NOTICE`.
