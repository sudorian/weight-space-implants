# H1 best attempt — not passing

A313 is the best completed full-H1 candidate by text agreement among candidates meeting the arithmetic threshold. It improves on A189: 24,961 versus 24,960 exact additions, and 98.02474975585938% versus 97.94158935546875% text agreement.

Official full run 0807-score: 24,961/25,000 exact additions and correct digit prefixes (99.844%); text agreement 98.02474975585938%, below the 98.5% gate; KL 0.0024928829704274477 nats; cross-entropy change −0.0002293046878767413 nats; finite. H1 joint pass is false. No rung passed.

There are 4,976,785 changed scalars and maximum absolute weight 1e6. The disjoint purpose-based edit audit is edit_categories.json: 578,041 arithmetic/circuit accommodation, 4,398,744 native text repair. These labels are not a causal decomposition. Noise exact/digits per 2,000 at 1e-5, 1e-4, 1e-3: 1997/1997, 1997/1997, 1602/1605. Two-in-a-row: 1997 first correct, 1 second in the meter field, 1 both correct.

Original checkpoint: experiments/a313v1_duplicate_reference_compression/delta.pt

Delta SHA-256: 67fd7d1ae9b0d63f4d2f8993b65f1e837fb65a418804e56125a8b7eb4cec49e5

Frozen float32 host SHA-256: d205c3d35f26a65bf5be4c7c9c56db479531152b401b80ca4bd447f9e7e00b63

The delta, original manifest, score, and edit audit are byte-for-byte copies. The sparse delta must be applied to the identified frozen float32 host using the supplied tools; it is not a standalone model. The original source safetensors is supplied as separate split files, outside the ZIP. The archive omits activation captures and intermediate weight artifacts, so rerunning every historical construction requires recreating those inputs; the final delta and recorded evidence remain available for independent evaluation.

Dorian closed the run early because usage was nearly exhausted. Per his explicit instruction, tools/verify.py and GSM8K were skipped; independent retesting is pending.
