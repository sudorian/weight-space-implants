# G3 robust follow-up — standard full score passes

Source: `experiments/g3r5_wide_source/delta.pt`, full run `0839-score`.
This is a new bank; the original G1, G2 and G3 checkpoints are unchanged.

Clean G3: 34,977/35,000 (99.93428571%). Text agreement: 98.60687256%,
0.10687256 percentage points above the required 98.5%. KL: 0.0009258857;
cross-entropy increase: 0.0000724078 nats; all values finite.

Multiplicative edited-weight noise: 1,634/2,000 (81.7%) at sigma 1e-5;
0/2,000 at both 1e-4 and 1e-3. Original G3 scored 45/2,000 at 1e-5.
These are fixed-seed aggregate measurements, not a general noise guarantee.
Two consecutive questions: first 1,999/2,000, second 0/2,000, joint 0/2,000.
The construction is limited to a spaced three-digit question at sequence start.

It changes 4,252,159 of 85,054,464 eligible scalars (4.99933666%),
with maximum absolute weight 639,765.5. Embeddings, positions, the output
matrix, final norm and model code remain unchanged. See manifest and score
for complete per-tensor counts and immutable construction-source hashes.

The changes compact constant output clips, restore native MLP capacity with
one-shot covariance projections, recenter internal arithmetic coordinates
analytically, and replace five noisy common-field writes with one shared
syntax-gated write. The gate's wider transition reduces amplification.
No gradients, iterative loss fitting or whole-problem answer regression.

R5 was selected for preservation first and noise second. The later R6 digit
readout solved one additional clean case but lowered text agreement to
98.57101440% and 1e-5 noise accuracy to 73.15%; it remains in experiments.

Final fresh verification is stored separately in
`verification/G3-robust.json`; REPORT.md records its outcome. The frozen
verify.py covers the original exact G1/G2/G3 folder names, so this new bank
is evaluated by a separate invocation of the same frozen score.py --fresh.
