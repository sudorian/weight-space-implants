# Offline Run 2 export

`run2.zip` is the evidence/checkpoint package. It contains the final REPORT,
GOAL, NOTES, append-only ledger, unmodified model and frozen tools, all local
Python/shell scripts and JSON manifests, JSON trial/score evidence, final
verification, all five bank folders, and the local GPT-2 tokenizer files.
The new selected bank is `best/G3-robust/`; the original G1/G2/G3 banks remain
unchanged. `best/G4-best-attempt/` is explicitly a FAILED attempt.

The ZIP contains no activation tensors, host checkpoint, calibration text,
held-out text or token caches. Those original text fixtures remain only in
the working project. Retired activation captures have hash journals; their
derived JSON findings and immutable scripts remain available. Rebuilding
every historical experiment from this compact package alone would require
the corresponding captures/fixtures, which are deliberately not included.

Reassemble the original local safetensors in a separate destination:

```sh
cat gpt2-small.safetensors.part-aa gpt2-small.safetensors.part-ab > model.safetensors
shasum -a 256 model.safetensors
```

Expected SHA-256:
`248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`.
The original is 548,105,171 bytes. Parts were made with `split -b 300m`;
the maximum part size is 300 MiB (314,572,800 bytes). MANIFEST.json records
the individual part hashes, the reconstructed stream hash, ZIP hash, ZIP
member hashes and successful CRC verification.

`experiments/prepare_host.py` documents the one-time offline float32 import,
including GPT-2 matrix transposes, ignored mask buffers and tied embeddings.
It refuses to overwrite an existing host. Sparse delta semantics and all
edit constraints are implemented by the unchanged `tools/common.py`.
Each bank's score and manifest identify its exact delta SHA-256.

The official ledger is complete: 300 forward, 100 quick and 40 full units
have been consumed. The final verifier is one-shot and has already run.
The original frozen meters also require the original text/token fixtures
and exact host/source hashes. Preserve these records; do not reset the
ledger or modify the meters. Independent evaluation should retain this
completed evidence and use its own unseen inputs and text.

No files were uploaded, and no network access was used for this export.
