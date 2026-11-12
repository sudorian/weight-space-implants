# Run 1 audit: nanoGPT Shakespeare implants (9 November 2026)

Independent re-test by Claude of run 1 (GPT-6 Astra, night of 8–9 November 2026) on separate x86 Linux hardware with pristine nanoGPT, against the frozen host (SHA-256 c9c7ce73…). The unedited host scores 0% on every rung.

## Fresh arithmetic
| Rung | Result |
|---|---|
| R1 | 100/100 |
| R2 | 10,000/10,000 |
| R3 | 200,000/200,000 random, 22,004/22,004 carry-heavy |
| R4 | 40,000/40,000 after held-back text prefixes, 40,000/40,000 after training-text prefixes, 20,004/20,004 carry-heavy with prefixes; every prefix length 0–51 at 100%; greedy decoding 300/300 |

Outside the specification, R4 also works without the newline (100%) and at sequence start (100%), but scores 0% on a problem right after a solved one and 0% for 1-, 2- or 4-digit numbers.

## Text preservation against the real host
R4: held-back Shakespeare cross-entropy +0.00063 nats, top-1 agreement 99.04%; training text 99.15%; modern English 98.58%; no non-finite values; 6,000 generated characters without a digit or symbol.

## Numerics
Batch sizes 1, 7, 64, 513, one thread and float64 all 100%. bfloat16 0%. Multiplicative noise on R4: 1e-6 about 99.7%, 1e-5 about 5%, 1e-4 0%. Maximum weight 2.445e12.

## Process audit
- Ledger: 603 lines, hash chain intact, 147 forward passes and 28 score runs; meter sources unchanged; model.py byte-identical to nanoGPT.
- Session log: no training, internet, access outside the folder, other apps or other models.
- Carry and digit weights are explicit hand logic; linear solves target fixed output coordinates; output thresholds were set once.
- Disclosure: one script multiplied recorded text activations by a candidate output matrix to estimate an effect on text (a one-layer calculation on stored data).

## Limits
R1–R3 changed 12 embedding rows the host never used, plus one attention layer. R4 changed 99.99% of all weights, including the token embedding, position embedding and final norm. MIT's editing tool refuses embedding, output-matrix and final-norm edits, so none of R1–R4 would have been allowed there. This is why run 2 froze those tensors and capped edits at 5%.
