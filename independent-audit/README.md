# Independent audit and re-tests

Written by Claude (Anthropic) at Dorian's request, separately from the runs themselves. GPT-6 Astra built every implant; nothing in this folder changes a weight.

- `AUDIT_RUN1.md`, `AUDIT_RUN2.md`, `AUDIT_RUN3.md`: findings and numbers.
- `run1/`: re-test and audit scripts for the nanoGPT Shakespeare run, with results in `run1/results/`.
- `run2/`: re-test and audit scripts for the GPT-2 run, with results in `run2/results/`. `run2_shell_commands.json.gz` lists every shell command from the run's session log, extracted for the boundary audit.
- `run3/`: re-test and audit scripts for the Gemma 3 270M run, with results in `run3/results/`. `run3_shell_commands.json.gz` lists every shell command from that run's session log.
- `prompts/`: the exact setup scripts that wrote each run's GOAL.md and folder, and `run3_finish_prompt.txt`, the message Dorian sent to close run 3 early (at 15:57 UTC on 11 November 2026). `run3_gemma_setup.sh` asks for a Hugging Face token at run time and never stores it.
- `third_party/nanoGPT-LICENSE`: licence of Karpathy's nanoGPT `model.py`, which runs 1 and 2 use unmodified.

## Re-running the GPT-2 checks
Python 3.12 with torch 2.15, numpy, safetensors, tokenizers and transformers (CPU is enough; about 1,000 tokens per second on two cores).

1. Set `NANOGPT_MODEL_PY` to nanoGPT's `model.py` (commit 3adf61e, SHA-256 7c017032…).
2. Make a work folder and set `AUDIT_WORK` to it. Put GPT-2 small's `model.safetensors` (openai-community/gpt2 commit 607a30d, SHA-256 248dfc39…) at `r2/weights/model.safetensors`, unzip `run2.zip` into `r2/zip/`, and unzip NLTK's gutenberg, webtext, inaugural and state_union corpora into `r2/text_src/`.
3. Set `WIKITEXT2_TEST` to WikiText-2's `test.txt` (pytorch/examples).
4. Run `r2_stage0.py` (edit limits and exact reproduction of the run's own counts), `r2_text_prep.py`, `r2_stage1.py` (fresh arithmetic, noise, probes), `r2_stage2.py` (fresh text) and `r2_stage1b.py` (small operands). The fresh seeds are drawn at random and printed; the ones behind the reported numbers are in the result files.

The code set uses the 40 largest modules of the local Python 3.12 standard library, so that domain varies slightly between machines.

## Re-running the Gemma checks
Python 3.12 with torch 2.15, numpy, safetensors, tokenizers, jinja2 and transformers 5.22.2 (its `modeling_gemma3.py` has SHA-256 d82a589c…, the same file the run used). CPU is enough: about 40 additions or two text windows per second on two cores.

1. Make a work folder and set `AUDIT_WORK` to it. Put the run's export in `r3/zip/` (or point `R3_EXPORT` at `run3-gemma3-270m/` and gunzip its `ledger.jsonl.gz`), and download Gemma's tokenizer and config files into its `downloads/` (see `downloads/README.md` there). Put `model.safetensors` (google/gemma-3-270m-it revision ac82b4e, SHA-256 700b710a…) at `r3/weights/model.safetensors`, or set `GEMMA_SAFETENSORS`.
2. Put the NLTK corpora in `r2/text_src/` as for run 2, set `WIKITEXT2_TEST`, and optionally `PY_STDLIB` (default `/usr/lib/python3.12`). Run `r3_text_prep.py`.
3. Run `r3_stages.py` with `stage0` (edit limits and exact reproduction of the run's own counts), `stage1` (fresh arithmetic), `stage3` (probes), `stage2` (fresh text), `stage2b` (reference points), `stage0b` (the from-scratch forward pass on the run's 25,000 problems) and `stage2c` (where text disagreements fall). `QUICK=1` runs every stage on a few items as a smoke test. `r3_code_diag.py` and `r3_code_trigger.py` locate the wrecked Python-code windows, `r3_heldout_overlap.py` checks forward-meter inputs against held-out text, `r3_audit_ledger.py <export> <out.json>` checks the ledger and export, and `r3_audit_log.py <session log> <out.json>` is the boundary scan.
