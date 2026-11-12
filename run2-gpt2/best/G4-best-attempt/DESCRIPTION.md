# G4 best fully scored attempt — NOT PASSING

Source: `experiments/o20_robust/delta.pt`; frozen full score `0579-score`.

Arithmetic: 19,844/20,000 (99.22%), exceeding the 99% arithmetic threshold. Ordinary-text agreement: 97.36557007%, below the required98.5% by1.13442993percentage points. KL0.0040727920nats and CE increase0.0021856017nats meet their0.02limits; values are finite. Failure of top-1 preservation makes this a failed G4 attempt.

The implant replaces native attention and MLP capacity. Native reconstruction did not retain enough original next-token argmaxes, even though its mean distributional changes remain small. The aggregate score does not identify one uniquely responsible layer; no stronger causal claim is made.

Multiplicative edited-weight noise:172/2,000 at1e-5;0/2,000 at1e-4 and1e-3. Repeated-problem diagnostic:0/2,000 joint exact. Changedscalars4,240,966;maximumabsoluteweight989,403.6875.

The delta, score and manifest are byte-identical copies. This archive is not a passing-rung bank. Existing G1–G3 checkpoints have not been modified.
