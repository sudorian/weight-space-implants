# GPT-2 direct weight programming — interim report

Status: **G1, G2 and G3 have passed full scores** and their existing checkpoints remain unchanged. G3 scores34,971/35,000 (99.91714%) with98.51151% ordinary-text agreement; its preservation margin and noise robustness need improvement. At the user’s priority update, ladder work stopped. The best fully scored G4 attempt is archived in `best/G4-best-attempt/` and explicitly does not pass. Remaining work targets a separate, more robust G3 checkpoint, then fresh verification and export. Last update: Tuesday10November,17:25CET.

## Protocol

The local GPT-2 small safetensors were converted once to the prescribed float32 nanoGPT checkpoint and strict-loaded into the unchanged model.py. Embeddings, positions and final normalization are protected. At most 4,252,723 of 85,054,464 internal scalars may change. Both meters enforce finite values, the 1e6 bound, no whole-tensor rescaling, source integrity and a hash-chained ledger. No gradients, training, network access, external models or subagents have been used.

The only held-out text reader is score.py. Text is cleaned and tokenized offline with the local GPT-2 byte BPE. Spaced sequence exact matching checks every correct-prefix argmax (equivalent to greedy exact decoding), including the final newline. Natural answers use actual greedy generation and decoded-string equality. Text comparisons use matched host/candidate batches; full vocabulary logits are not cached. Pre-freeze synthetic tests verified evaluator behavior, including alternate valid tokenization of a natural answer.

Host SHA-256: `d0896398fc68efd3d1c516b970ecbd87ab47eeaec269635044ed03223916ff2e`. Frozen source manifest: `tools/frozen.json`. Original model source SHA-256: `7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c`.

## Unedited host baselines

| Rung | Quick correct / count | Accuracy | Pass |
|---|---:|---:|---|
| G1 | 0 / 100 | 0.000% | no |
| G2 | 0 / 2,000 | 0.000% | no |
| G3 | 0 / 2,000 | 0.000% | no |
| G4 | 0 / 2,000 | 0.000% | no |
| G5 | 6 / 2,000 | 0.300% | no |

Quick text baseline: 256 windows, 16,384 targets; agreement 100%, KL 0, CE increase 0. Host cross-entropy 4.2740137822 nats. These are baseline measurements, not evidence of a successful implant.

## Current experiments and limitations

The first capture phase measured weak internal channels and equation-format separation. Subsequent constructions and failures are reported below. G1–G3 are banked; final fresh-seed verification has not been run.

At the G1 pass, usage was 65 forward, 10 quick and 2 full units. Subsequent G2 analysis continues within the fixed budgets; exact completed counts remain in ledger.jsonl.

The previous tiny Shakespeare-model result is not carried over as evidence: this host has a real GPT-2 vocabulary, fixed embeddings and a much tighter sparse-edit constraint. The G1 implant satisfies the clean-weight threshold; extension to multi-digit inputs and robustness remain unresolved.

## First circuit result — 10:35 CET

Design A reads the two operand digits separately with two late attention heads, then uses 44 final-MLP units for explicit decimal thresholds. Whole-format detectors control its activation; malformed operators and operands are included among calibration negatives. A residual nullspace and weak-channel rotations limit disturbance. Output directions are derived from the frozen unembedding. Replaced MLP activations are reconstructed through a one-shot projection of unlabeled calibration activations.

The first complete checkpoint, `experiments/a2t4_g1/delta.pt`, scores **99/100 on quick G1**, but does **not pass**: text agreement is 98.21167%, below 98.5%. KL is 0.00471827 nats and CE increase is 0.00564685 nats. It changes 3,288,368 internal scalars and has max |w| 554,458.375. The failing arithmetic case emits units digit 3 for 9+7. No noise/full-score result or banked checkpoint exists yet.

The frontend alone preserves 98.63892% text agreement, KL 0.00434515, CE +0.00549630. The final output program therefore introduces an additional preservation cost. Recorded register errors and the large format-off coefficients also expose numerical sensitivity. Both issues are being diagnosed.

Scale values so far: carrier 16384, with 4096 now being tested; attention score 64; value gain .25; output gain 65536; main output temperature 64, 8, 4 (64 and 8 had pre-meter bound/certificate failures, 4 was scored); newline temperature 64; stage penalty 256; capacity ridge .001. Formula-derived format-gate sharpness and text-off bounds are recorded exactly in manifests. Detector/readout ridge fractions are 1e-6 and 1e-5. The first complete result used 18 forward units and 7 quick units; zero full runs. The ledger is authoritative for subsequent runs.

## Full G1 attempt and preservation diagnosis — about 11:00 CET

A3 uses carrier 4096 and a piecewise-linear decimal output representation with 26 neurons instead of 44. Conditional covariance and frozen-readout geometry choose which neurons to reuse. It changes 3,274,526 scalars, with max |w| 100,863.078125. Its quick result was 100/100 G1, 98.53516% text agreement, KL 0.00439572, CE +0.00529504. This quick-only pass was not banked.

The full result (`artifacts/a3t8_g1_full.json`) is **100/100 arithmetic, but a failed rung**: 98.24448% agreement over 131,072 text targets, KL 0.00514450, CE +0.00642468. All values were finite. Noise accuracies are 91%, 6%, and 0% at sigma 1e-5, 1e-4, and 1e-3. These G1 noise checks are exhaustive over its 100 unique cases: the frozen meter uses min(2,000, rung size), rather than repeating identical G1 cases twenty times. This protocol detail is disclosed rather than changing the frozen meter. Both k=3 questions in the repeated-problem diagnostic scored 0%, as expected for this fixed one-digit construction.

Numerical captures show that the original 44-unit circuit's failures came from float32 cancellation in large common MLP coefficients, despite accurate digit memory and a mathematically zero format-off feature. Its recorded preactivations differed by up to about 0.56 from double-precision arithmetic on the same captured input. Lowering the carrier and shortening the program fixed exhaustive clean G1 arithmetic, but the noise results still expose sensitivity.

Fresh calibration ablations (16,384 positions, separate windows from construction) isolate preservation costs: MLP10 gates/capacity alone 99.49341% agreement; plus five-direction projection 98.88916%; plus numeric value channels 98.87085%; full frontend 98.60229%; complete circuit 98.21167%. These are diagnostic calibration measurements, not held-out passes.

A4 restores six removed MLP features from 256 retained features by a one-shot covariance projection, and restores the attention projection's removed text mean. The arithmetic gate subtracts this mean before the attention block, preserving its memory interpretation. Its frontend calibration agreement is 98.74268%; further diagnosis is ongoing. New constants: 256 donor features and covariance ridge .001.

A separate compact readout design uses decimal parabolic scores through the fixed unembedding. Five predeclared class-bias scales (16, 32, 64, 128, 256) were tested only as a synthetic readout piece. The first bias-normalization derivation fails for digit zero at every scale (best minimum margin -0.2261 nats); it is not a checkpoint or a passing result. A revised endpoint-normalization derivation is being evaluated.

Current remaining meter budgets: {"forward": 269, "quick": 92, "full": 39}. The ledger records exact run starts, finishes, constraints, artifacts and hashes.

## First fully passed rung — G1, about 11:37 CET

`best/G1/delta.pt` is an exact copy of `experiments/b2_g1/delta.pt`, with its full score, manifest and description. All 100 sums, both output digits and the newline match greedily. On 2,048 held-out windows (131,072 targets), agreement is **98.56414795%**, KL **0.0010779622**, CE increase **0.0015820572** nats, with no non-finite values.

The passing construction combines two complementary format discriminants. One rejects ordinary text but can accept a wrong operator; the other rejects wrong operators but has rare prose false positives. A mixing interval is derived algebraically from their measured bounds, and its midpoint is used. The combined gates separate all 2,000 syntactically valid G1 prefixes (continuation digits independent of the sum), malformed controls, and 900,800 full-calibration token positions plus earlier calibration samples. Mixing ratios are 1.1988353363, 2.0347342717 and .8509547099 for the three stages. No sum labels enter those detectors.

The compact output circuit has 11 units. It subtracts ten on a units carry, then uses a parabolic decimal score through the frozen unembedding. A self-consistent endpoint correction accounts for final-LayerNorm bias. The tested class-bias scales were 16, 32, 64, 128 and 256 in each of two distinct normalization derivations; the endpoint version is valid for every digit at all five scales, and 256 has the largest minimum synthetic readout margin (0.12004 nats). Output gain is 262144; main temperature 8 and newline temperature 64.

Noise is a serious limitation: full G1 accuracies at sigma 1e-5, 1e-4 and 1e-3 are **30%, 10%, 0%**. This is a clean-weight rung pass, not a robustness claim. The two k=3-in-a-row diagnostic accuracies are zero; this checkpoint is specific to G1. The held-out agreement margin is small, so fresh final verification is still required.

First-pass meter use: **65/300 forward units, 10/100 quick units, 2/40 full units**. Changed scalars: **3,460,364** (4.068410% of eligible); maximum absolute weight **113360.3828125**.

Changed scalars by tensor:

| Tensor | Count |
|---|---:|
| `transformer.h.10.mlp.c_fc.weight` | 4,608 |
| `transformer.h.10.mlp.c_fc.bias` | 6 |
| `transformer.h.10.mlp.c_proj.weight` | 201,216 |
| `transformer.h.10.mlp.c_proj.bias` | 768 |
| `transformer.h.11.attn.c_attn.weight` | 294,912 |
| `transformer.h.11.attn.c_attn.bias` | 384 |
| `transformer.h.11.attn.c_proj.weight` | 589,184 |
| `transformer.h.11.attn.c_proj.bias` | 767 |
| `transformer.h.11.mlp.c_fc.weight` | 8,448 |
| `transformer.h.11.mlp.c_fc.bias` | 11 |
| `transformer.h.11.mlp.c_proj.weight` | 2,359,292 |
| `transformer.h.11.mlp.c_proj.bias` | 768 |

## G2 work — about 13:50 CET

Two independent sets of 512 nine-token G2 format prefixes use continuation digits independent of sums. A further 512 malformed controls change one input role. All digit-decoder labels are individual current tokens; gate labels are syntax and stage. No whole-problem answer regression or table is used.

Late layer-9 digit classifiers identify every digit in the independent set, but the first-position decoder has large ordinary-text activation (RMS about 9.565). Layer-1 token decoders also classify every digit, with maximum individual-feature RMS on text of 0.199, 0.101, 0.120 and 0.131 at the four source positions. The design therefore moved to earlier layers. A geometric search over memory subspace widths 40, 80, 128 and 256 selected width 256: 36 chosen directions have maximum independent program residual 0.01474 and contain about 0.1049% of attention-output energy.

A first early format design tried direct linear stage detectors at all four answer positions. Text-focused and corruption-focused Fisher geometries used hard-negative covariance masses 1, 4, 16, 64 and 256, each derived by a one-shot solve from recorded activations. Only the equals position admits a certified mixture; subsequent answer positions do not. This is a failed component design, not evidence that the rung is infeasible.

Design C instead writes a marker only at the verified equals position and broadcasts it causally through one early attention head. The equals gate uses covariance mass 1 and a bound-derived mixture ratio 0.3433965345; its conservative separation gap is 0.0396803 across the full calibration corpus and format controls. The first marker constants are carrier 512, attention score 64, value gain 1, 256 reconstruction donors and covariance ridge 0.001. It is currently only a frontend, with no addition logic. Recorded valid prefixes show the marker reaching subsequent answer positions while malformed controls leave the equals gate inactive. Text preservation is being metered before further arithmetic stages are added.

The banked G1 checkpoint is unchanged. No G2 checkpoint has passed; no final fresh-seed verification or export has been performed. The run is active and is not blocked.

## G2 digit quantization and preservation — about 16:46 CET

The C1 marker frontend preserves 99.67041% text agreement in a quick score (KL 0.00003466, CE +0.00013309). The C2/C3 digit-reading frontends correctly identify all four source digits at every output stage in two independent 512-prefix captures, but quick text agreement is only 98.30933% (KL about 0.000658). Removing unnecessary value constraints did not improve this. These are frontend diagnostics; neither implements complete addition.

Increasing the value gain from 0.1 to 0.25 reduces rounding and background error. C5 turns those reads into discrete operand codes and amplifies the stage marker. Its single-shot text activation projection removes 86.5% of state distortion on separate calibration windows, and all 16,384 captured operand-digit checks remain correct. Nevertheless its quick held-out result fails: agreement 96.44165%, KL 0.04041658 and CE increase 0.03929149. A fresh calibration probe found 50 of 32,768 positions with false digit activations and no false phase activations. The format-off approximation was therefore inadequate outside its construction sample.

C6 replaces the shared approximate format-off direction with four banks of digit clips, each checking its specific output-stage carrier. This uses 320 digit units plus eight phase units and 768 reconstruction donors. All captured digit codes classify correctly; correct amplitudes are 0.99805–1.00147 and incorrect amplitudes are zero. Its complete addition stage is still pending. C6 quick text results: agreement **96.04492188%**, KL **0.0040095098**, CE change **-0.0012330729**; pass=**False**.

The next MLP is being compiled from explicit per-column decimal tables, using eighteen clipped thresholds per column plus a base codeword. No whole-problem sums have been fitted. The final conditional output stage will use fixed-unembedding digit directions. G1 remains banked and unchanged; G2–G5 are unfinished.

New constants and failed geometric trials: early digit frontend carrier 4096, attention score 64, value gains 0.1 and 0.25; quantizer near-null ranks 128, 256, 512, 640, 704; digit clip temperature 32, initial bounds 0.2/0.55 then binary-width bounds 0.25/0.5; near-null text safety factor 1.01, stage-specific factor 2; digit code gain 16384, boosted phase gain 262144; reconstruction donors 512 then 768. Rank-512 quantization with gain 0.1 exceeded the coefficient bound (1.239 million) and was not saved. Lower temperatures 16 and 24 were considered analytically but not used in a checkpoint. The per-stage carrier normalization constants are 0.9917328784, 0.9995105747, 0.9995490932 and 0.9994993515. Planned column constants are sum gain 16384, temperature 128, threshold width 0.125 and 384 reconstruction donors; their build is still running.

Superseded activation files were removed to respect the 25 GB folder ceiling; deletion metadata is retained. No frozen tool, host, held-out data, ledger entry or banked checkpoint was changed. Final verification and export remain pending.

## Complete G2 circuits and current limit — about 17:38 CET

Four complete direct-weight G2 circuits now achieve **2,000/2,000 exact sums in quick evaluation**, including the newline. None has passed the required text threshold, so G2 is not banked and no full G2 run has been spent.

| Complete circuit | Quick G2 | Text agreement | KL (nats) | CE change | Changed scalars |
|---|---:|---:|---:|---:|---:|
| C8: stage-specific digit banks | 2,000/2,000 | 95.63599% | 0.00476051 | -0.00126924 | 4,182,715 |
| D4: common format carrier | 2,000/2,000 | 97.22290% | 0.00199241 | -0.00164264 | 3,615,688 |
| F4: compact attention edits | 2,000/2,000 | 97.02148% | 0.00244952 | -0.00139825 | 4,137,854 |
| I3: separate scalar decimal columns | 2,000/2,000 | 98.03467% | 0.00096537 | -0.00124886 | 4,180,088 |

All four remain below 98.5% agreement. Their maximum absolute weight is 638,699.3125. The scalar-column design is the closest complete G2 result. It reads each digit separately, forms two independent column sums, then uses explicit decimal thresholds and Boolean carry routing. It never fits a whole-number sum. The frontend alone preserves 99.42627% agreement, so restoring capacity in the programmed MLPs is now the principal problem.

D replaces four large stage carriers with one common carrier of 16,384 and four small stage codes of 256. It cancels the earlier equals marker and uses a four-dimensional syntax-only normalization correction. Lowering the attention score from 64 to 16 had little effect on text preservation. A local-only reconstruction from 1,280 donors also failed to improve the complete feature-stage preservation.

F restricts memory projection to selected attention heads and reconstructs nine replaced value coordinates from 55 retained coordinates. Candidate head-support sizes 4, 6 and 8 were analyzed geometrically; six was selected. This saves many edited entries but discards more text information than the full value rotation. Increasing the next MLP's donor count to 2,048 did not recover enough agreement.

I uses only two memory directions instead of 36. Memory subspace widths 16, 32, 64, 128 and 256 were analyzed; width 256 removes only 0.007515% of attention-output energy. Value gain is 0.1 and attention score is 16. The independent current-digit scalar errors are at most approximately 0, 0.0544, 0.0755 and 0.0880. The two recorded column errors stay below 0.165 before gating; all 8,192 recorded column mod/carry checks pass after the hand thresholds. MLP2 uses 82 units, temperature 16, threshold width 0.125, safety factor 2, sum-code gain 16,384 and stage gain 262,144. MLP3 uses 35 output conditions, temperature 64 and output gain 67,108,864. Reconstruction donor counts are 1,536 and 640.

A first column compiler at temperature 128 exceeded the weight bound (1,244,794) and was rejected before a checkpoint; temperature 64 succeeded in the earlier C/D designs. The current I design uses 16. Its text reconstruction uses a frozen-unembedding covariance metric to select less important features. A new, predeclared five-value ridge comparison (0.001, 0.0001, 0.00001, 0.000001, 0.0000001) is running. Each candidate is a single covariance projection; selection uses activation reconstruction on separate calibration windows, with arithmetic logic fixed and no held-out inspection.

These are clean-weight quick results, not full-rung or noise-robustness claims. G1 remains the sole banked rung. Work is active, with further preservation experiments and higher rungs still pending.

## G2 passed — 17:59 CET

`best/G2/delta.pt` is an unchanged copy of `experiments/j3_g2/delta.pt`. Full evaluation gets **10,000/10,000 exact two-digit sums**, including every output digit and newline. On 2,048 held-out text windows (131,072 targets), agreement is **98.90518188%**, KL **0.0003233848** nats and cross-entropy increase **0.0003516801** nats. All outputs are finite. This is a full rung pass; fresh final verification remains pending.

The successful design J reduces the arithmetic program from 117 MLP neurons to 28. It keeps two independent decimal-column scalars rather than converting them to categorical codes. Fourteen units in MLP2 amplify the scalar values and syntax phases, subtract ten when the units column carries, and add that carry to the tens column. Fourteen units in MLP3 implement the remaining carry, affine parabolic digit scores through the frozen unembedding, newline, and cancellation of the large program carriers. The independent recorded scalar errors remain below 0.168. The frozen readout selects the correct digit for synthetic digit values displaced by ±0.2. No whole-problem answer labels enter a solve.

Earlier preservation variants retained the larger 82+35-neuron program. I3R (five ridge values with conditional feature selection) achieved 98.02856% agreement; I3B (a pivoted-Cholesky donor basis) 98.18115%; I3E (raw-energy feature selection) 97.98584%. All three were exact on 2,000 quick sums and failed preservation. The smaller J program, with raw-energy feature selection, passed both arithmetic and preservation. Its partial scalar checkpoint had 99.22485% quick text agreement; the complete quick score was 98.90747%.

J constants: MLP2 temperature 16, scalar gain 1,024, phase gain 1,048,576, units carry clip interval [9.25, 9.75], and 1,536 reconstruction donors. MLP3 temperature 64, output gain 16,777,216 and 768 donors. Per-stage format penalties are derived from calibration bounds: 107.9016980403, 109.7811115216 and 64. Both reconstruction stages evaluate the predeclared ridge list [0.001, 0.0001, 0.00001, 0.000001, 0.0000001] as independent closed-form projections with fixed feature identities, and select 0.001 using a separate calibration split. The earlier I3R/I3B comparisons used that same five-value list in their distinct, documented feature-selection formulas. All inherited frontend constants and the five parabolic readout scales are listed above and in immutable manifests.

Noise remains a substantial weakness. G2 noise accuracy is **1,198/2,000 (59.9%)** at sigma 1e-5, and **2/2,000 (0.1%)** at each of 1e-4 and 1e-3. This result establishes clean-weight addition, not reliable addition under weight noise. Both three-digit problems in the repeated-problem diagnostic scored 0/2,000 (joint accuracy zero); this checkpoint is specific to the two-digit format.

At the first full G2 pass, meter usage was **134/300 forward units, 35/100 quick units and 3/40 full units**. It changes **4,141,598** scalars (**4.869348%** of eligible parameters), with maximum absolute weight **646606.875**.

| Tensor | Changed scalars |
|---|---:|
| `transformer.h.0.mlp.c_fc.weight` | 1,536 |
| `transformer.h.0.mlp.c_fc.bias` | 2 |
| `transformer.h.0.mlp.c_proj.weight` | 197,728 |
| `transformer.h.0.mlp.c_proj.bias` | 768 |
| `transformer.h.1.attn.c_attn.weight` | 147,456 |
| `transformer.h.1.attn.c_attn.bias` | 192 |
| `transformer.h.1.attn.c_proj.weight` | 589,793 |
| `transformer.h.1.attn.c_proj.bias` | 767 |
| `transformer.h.1.mlp.c_fc.weight` | 6,144 |
| `transformer.h.1.mlp.c_fc.bias` | 8 |
| `transformer.h.1.mlp.c_proj.weight` | 202,748 |
| `transformer.h.1.mlp.c_proj.bias` | 768 |
| `transformer.h.2.attn.c_attn.weight` | 589,824 |
| `transformer.h.2.attn.c_attn.bias` | 768 |
| `transformer.h.2.attn.c_proj.weight` | 588,288 |
| `transformer.h.2.attn.c_proj.bias` | 765 |
| `transformer.h.2.mlp.c_fc.weight` | 10,752 |
| `transformer.h.2.mlp.c_fc.bias` | 14 |
| `transformer.h.2.mlp.c_proj.weight` | 1,190,399 |
| `transformer.h.2.mlp.c_proj.bias` | 768 |
| `transformer.h.3.mlp.c_fc.weight` | 10,752 |
| `transformer.h.3.mlp.c_fc.bias` | 14 |
| `transformer.h.3.mlp.c_proj.weight` | 600,576 |
| `transformer.h.3.mlp.c_proj.bias` | 768 |

The banked G1 checkpoint is unchanged. G3 preparation is now beginning with syntax-only independent continuation captures, six separate digit reads and hand-derived carry propagation. The run remains active; final verification and export have not yet been performed.

## G3 passed — 18:18 CET

`best/G3/delta.pt` is an unchanged copy of `experiments/k5b_g3/delta.pt`. Full exact arithmetic is **34,971/35,000 (99.91714286%)** over 30,000 random problems and 5,000 carry chains. Ordinary-text agreement is **98.51150513%**, KL **0.0019758646** nats, CE increase **0.0010636043** nats; all values are finite. The text margin is only **0.0115 percentage points**, so a fresh-seed pass is uncertain. This bank records the successful full test, not a claim of broad robustness.

Design K extends the compact scalar circuit to three columns. Six attention heads read separate current digits; no whole-number sum is fitted. Seventeen MLP2 units prepare the three scalar columns and units carry, two MLP3 units propagate the tens carry, and seventeen MLP4 units emit leading carry, hundreds, tens, units and newline through the fixed output matrix. The source-position classifier uses means stratified by current token so that token identity does not masquerade as position. All 512 independent captured examples select every source position correctly; the smallest position margin is 0.2961. Captured scalar errors are at most 0.2247, 0.1895 and 0.1934.

The equals gate is based on syntax-only captures with random continuation digits independent of sums. Its hard-negative mass is 1, mixture ratio 0.4251286872 and conservative separation gap 0.0161286593 over 900,800 calibration text positions plus malformed controls. The full five-mass family [1, 4, 16, 64, 256] was considered. Initial separate min/max bounds failed to certify some later stages. Intersecting the actual positive/negative pair inequalities preserved correlations and certified stage mixtures 83.34145768, 87.38688956, 103.09540238 and 97.99808095, with gaps 0.31128, 0.14759, 0.13851 and 0.16491. This is an algebraic interval calculation on syntax detectors, not iterative weight-loss fitting.

The first complete K circuit scored 1,997/2,000 quick sums, text agreement 98.61450%, KL 0.00052782 and CE change -0.00022634. Its original source-position classifier misselected one of 512 independent examples and produced a large tens-column error. The revised token-conditioned keys removed that recorded error; doubling the scalar temperature from 16 to 32 reduced soft carry error. The revised complete circuit still had the same three quick failures, with text agreement 98.60229%, KL 0.00060863 and CE change -0.00050711. The remaining failures are being diagnosed; output-stage gate generalization is a plausible cause, not yet a verified explanation.

New constants: six reading heads [1, 4, 2, 7, 9, 8]; three memory directions chosen within a fixed 256-dimensional weak subspace; common carrier 16,384, stage codes 256, reading score 16 and value gain 0.1. Scalar temperatures tested are 16 and 32; the bank uses 32, scalar gain 1,024 and phase gain 1,048,576. The tens-carry and final-output temperatures are 64; final output gain is 16,777,216. Reconstruction donor counts are 1,024, 256 and 640. Each stage uses the same declared five-value ridge list [0.001, 0.0001, 0.00001, 0.000001, 0.0000001] through the immutable v5 closed-form restoration helper; exact selections and bound-derived gate penalties are in the manifests.

Noise accuracy is **45/2,000 (2.25%)** at sigma 1e-5 and **0/2,000** at 1e-4 and 1e-3. Large coefficients and narrow program-coordinate margins remain important limitations.

At the first full G3 pass, usage was **159/300 forward units, 38/100 quick units and 4/40 full units**. Changed scalars: **4,158,503** (**4.889224%** of eligible); maximum absolute weight **976424.4375**.

| Tensor | Changed scalars |
|---|---:|
| `transformer.h.0.mlp.c_fc.weight` | 1,536 |
| `transformer.h.0.mlp.c_fc.bias` | 2 |
| `transformer.h.0.mlp.c_proj.weight` | 197,728 |
| `transformer.h.0.mlp.c_proj.bias` | 768 |
| `transformer.h.1.attn.c_attn.weight` | 147,456 |
| `transformer.h.1.attn.c_attn.bias` | 192 |
| `transformer.h.1.attn.c_proj.weight` | 589,793 |
| `transformer.h.1.attn.c_proj.bias` | 767 |
| `transformer.h.1.mlp.c_fc.weight` | 7,680 |
| `transformer.h.1.mlp.c_fc.bias` | 10 |
| `transformer.h.1.mlp.c_proj.weight` | 204,284 |
| `transformer.h.1.mlp.c_proj.bias` | 768 |
| `transformer.h.2.attn.c_attn.weight` | 884,736 |
| `transformer.h.2.attn.c_attn.bias` | 1,152 |
| `transformer.h.2.attn.c_proj.weight` | 588,671 |
| `transformer.h.2.attn.c_proj.bias` | 765 |
| `transformer.h.2.mlp.c_fc.weight` | 13,056 |
| `transformer.h.2.mlp.c_fc.bias` | 17 |
| `transformer.h.2.mlp.c_proj.weight` | 799,488 |
| `transformer.h.2.mlp.c_proj.bias` | 768 |
| `transformer.h.3.mlp.c_fc.weight` | 1,536 |
| `transformer.h.3.mlp.c_fc.bias` | 2 |
| `transformer.h.3.mlp.c_proj.weight` | 198,143 |
| `transformer.h.3.mlp.c_proj.bias` | 768 |
| `transformer.h.4.mlp.c_fc.weight` | 13,056 |
| `transformer.h.4.mlp.c_fc.bias` | 17 |
| `transformer.h.4.mlp.c_proj.weight` | 504,576 |
| `transformer.h.4.mlp.c_proj.bias` | 768 |

In the repeated k=3 diagnostic, first-problem accuracy is 1999/2000; second-problem accuracy given the correct first continuation is 0/2000; joint accuracy is 0/2000. This rung starts at the beginning of the sequence; handling arbitrary preceding text remains the next rung.

G1 and G2 banks are unchanged. Final fresh-seed verification and export remain pending.


## G4 syntax and position designs — 19:31 CET

No G4 checkpoint has passed. G1–G3 remain unchanged. The remaining budget is 124 forward, 55 quick and 36 full units; final fresh-seed verification and export have not been performed.

Three approaches have been explored. L used native early/late affine whole-format detectors with hard-negative masses 1, 4, 16, 64 and 256. Neither layer 0 nor layer 8 yielded a strict margin across independent valid prefixes, malformed controls and calibration text. M added eight separate malformed-role bits plus a context bit. Its gain-0.5 frontend preserved 99.53613% text agreement, but fresh role-0/role-1 controls each missed 2/64 corruptions, and no subsequent linear whole-format gate separated the controls. Even the five explicitly examined malformed quantile levels 1, .999, .995, .99 and .98 failed to yield a gate. Native relative-position probes also misread late operand roles after text prefixes. These failures involve syntax and addressing, not the decimal arithmetic logic.

N instead derives exact token and position coordinates from the frozen embedding matrices and projects the first attention output away from those coordinate directions. The off-coordinate is zero on the 22 permitted input token types and positive on every other vocabulary token. Recorded independent-prefix validation confirmed coordinate errors below 0.00000434 for the initial six-direction projection. Its quick text agreement was 99.38354%. Both the six-direction and later twelve-direction Fourier construction examined metric masses 0, .01, .1, 1 and 10 and mean-restoration fractions 1 and 0, selected by removed calibration activation energy.

The first position writer used 278 MLP units and gain 2; it preserved only 93.21289% text agreement despite code errors below 0.000570. A smaller 44-unit writer uses direct homogeneous Fourier coordinates and token clips. Its first gain-8 version scored 92.12646% agreement. Inspection found that both writers subtracted a large residual mean globally. Constraining the output frame perpendicular to that mean removed this unintended disturbance. The corrected 44-unit writer scores 98.44360% agreement at gain 8 and **99.08447% at gain 2**; a gain-0 ablation also scores 99.08447%. Gain-2 KL is 0.0004204193 and CE change is −0.0000698440 nats. It changes 1,052,203 scalars, with maximum edited magnitude 5515.93945. This is a token/address frontend, not an addition result.

Additional declared constants: Fourier frequencies [1,3,9,27], normalized over 64 absolute positions; alternative static code sets [1,5,13,23], [1,2,4,8,16,32], [1,3,7,13,21,29] and [1,3,9,19,27] were compared by wrong-position correlation only. The compact writer uses linear-gate temperature 32, baseline 2, kind penalty 4, and bound-derived off penalties; clipped gates use ±8 biases and sharpness 24 times a twofold observed-sigma bound divided by the raw integer margin. Reconstruction uses 512 donor features and the same five closed-form ridge values (.001, .0001, .00001, .000001, .0000001), selecting .001. Corrected-writer gains tested so far are [8,2,0]. Current-token readout covariance masses are [0,.001,.01,.1,1]; position-detector masses are [.001,.01,.1,1,10]. Every target in these solves is a current-token or position/format label. No sum or answer target is used.

The next component is an explicit relative-position attention reader that checks every required operand/operator token before activating arithmetic. Storage cleanups removed only superseded self-generated activation files and are recorded in artifacts/cleanup_*.json. All banked deltas, manifests, scores, scripts and the frozen protocol remain intact.


## G4 explicit address reader — 20:37 CET

G4 remains incomplete; G1–G3 are unchanged. The explicit reader now identifies all seven required input positions in independent captured controls, but its text-preservation cost remains above the threshold. No G4 arithmetic result is claimed. Before the current lower-score trial, 185/300 forward units, 57/100 quick units and 4/40 full units have been used.

The N reader, with seven heads and eight Fourier QK coordinates per head, preserved only 75.34180% text agreement. Three background-clearing variants used retained native activations, exact linear GELU identities and conditionally predictable register directions. Their standalone agreements were 98.98682%, 98.84033% and 98.80981%, but query-background RMS fell only from approximately .15–.19 to .11–.13. This was insufficient to justify another high-score N reader. The reconstruction ridge list remained [.001,.0001,.00001,.000001,.0000001], chosen .001; these are closed-form unlabeled activation projections.

O reduces the address to modulo-eight cosine/sine, absolute position, and token-type checks. Seven protected embedding directions feed 34 MLP units and thirteen fields. The writer gains are 2 for token types/digits, 8 for source positions, and 16 for equals positions. Its quick text agreement is 99.14551%. An initial 33-unit, twelve-field version reached 99.22485% but lacked the equals absolute-position field needed for normalization-invariant addressing. Direct normalized position fits failed on independent prefixes; the final address instead uses a homogeneous kernel whose correct-source score is exactly zero before perturbations. Older modulo-eight aliases and mismatched positions have negative scores. Its analytic recency slope is (1−cos(pi/4))/5 per token; the normalized-position coefficient is 32 times that value. Five word penalties [.125,.25,.5,1,2] were evaluated by position margins, selecting .125. Feasible fallback intervals were computed directly from syntax-position inequalities, with fractions .5 and .25 considered and .25 selected.

The original O reader used a 24-coordinate native QK rotation, reused five coordinates per head, and selected score 1024 from [64,128,256,512,1024]. Text agreement was 96.19141%. A 1,024-donor MLP1 restoration raised it only to 96.44165%. Fresh recorded head outputs nevertheless confirmed all seven required token-type flags on valid formats (minimum .9428), while every malformed control's replaced-role flag was below .5. The per-column diagnostic errors were at most .49874, with none exceeding .5 among 3,072 columns. These are checks of independently decoded digits and hand column wiring, never regression targets or answer tables.

O3B uses an optimal full 64-dimensional bilinear QK transform, preserving native scores before reusing the five lowest-energy coordinates. Current-token decoder covariance masses remain [0,.001,.01,.1,1]; attention token flags use mass1, value flags use .1, and digits use .01. The numeric value gain was reduced from .5 to .25; syntax gain remains1. Memory-subspace widths [16,32,64,128,256] were compared by native program-background energy, selecting32. This reader changes 2,113,754 scalars, maximum magnitude8652.682, and reaches 97.74780% text agreement, KL .00358995, CE increase .00178693. Address-only and value/output-only ablations reach 98.26660% and 98.59619%; both contribute to the preservation loss. Score512 already selects all512×7 recorded source positions (minimum margin1.659), whereas1024 gives9.308. The current trial uses512 to reduce ordinary-text score perturbations. It is one of the original five declared values.

All final-stage verification and export remain pending. Immutable builder manifests contain the complete trial tables and per-tensor edit counts for failed candidates; the banked checkpoint tables above remain authoritative for passed rungs.


## G4 gated payload and position counter — 21:26 CET

G4 is still incomplete. The whole-format gate now activates on all 1,024 fresh valid calibration-prefix equations and on no other checked position. It rejects 510/512 fresh malformed equations; two inherited false activations remain. A seven-unit MLP1 copies three centered column values and their homogeneous reference only behind this gate. Its quick text agreement is 98.42529%, below the required98.5%, so it is not a passed checkpoint.

The first strong payload broadcast reduced agreement to98.07739%. Balancing the positive reference at sixteen times each centered-column amplitude permits a weaker, four-field broadcast. The current O5C frontend reaches98.33984% agreement, KL0.00113877 and CE change−0.000049352. It uses source gains[65536,65536,65536,65536,1048576], attention score32, broadcast gain64, and one reading head. Five output-head protection counts[1,2,3,4,5] were compared by recorded background energy, selecting5. Its score was selected from[16,32,64,128,256] using address margins; score32 has minimum19.87. The earlier gain256 and score128 version is retained as a failed preservation result.

Fresh captured values remain nearly unchanged across all five output positions. Five of1,024 problems have a column error above half a digit; the largest per-column errors are approximately.641,.414,.845. These are diagnostic checks of independently decoded digits and hand-wired columns, not training targets. The equals-presence and later-presence gates have margins.6833 and.3844 over other valid positions and calibration text. The reference-mixture trials were Eq-read fractions[0,.25,.5,.75,1], selected.75 by constant-one reference error alone; maximum error was.02545.

A fourteen-unit MLP2 now writes a common post-equals marker and an equals-only point. A subsequent attention head is designed to average that point uniformly over marked positions, yielding1,1/2,1/3,1/4,1/5. The recorded-component counter check has maximum error.001546 at score16, selected from[8,16,32,64,128]. This component is not yet a metered arithmetic result. A revised marker frame is being chosen by normalized ordinary-text energy to reduce false point values in prose. The remaining stages are phase selection, units carry, tens carry and final digit output.

Other new constants: common and equals-point amplitudes2^24, numeric amplitude2^16, fourteen-unit copying temperature64, marker-clip difference512, and256 native restoration donors. The fixed five restoration ridges remain[.001,.0001,.00001,.000001,.0000001]. Unmetered counter memory protection counts[1,2,3,4,5] and gain256 are documented in their manifests. Ordinary-text preservation still needs improvement; no new rung has passed. At this update198/300 forward units,62/100 quick units and4/40 full units have been used. Final fresh-seed verification and export remain pending.


## G4 complete arithmetic circuit — 22:16 CET

G4 arithmetic now reaches **1,985/2,000 (99.25%)** in the frozen quick meter, but this is **not a passed rung**: ordinary-text agreement is97.42432%, below98.5%. KL is0.002272353 and CE increase0.000190642 nats. The candidate is`experiments/o13_readout/delta.pt`; it changes4,240,966 eligible scalars (4.98618%), with maximum magnitude989,403.6875. G1–G3 banks remain unchanged. No G4 full run has been claimed, and final fresh-seed verification/export remain pending.

The counter now uses one protected attention3 head and output gain128. Its five recorded phase ranges stay separated by fixed boundaries[.75,5/12,7/24,.225], with no lower boundary needed after newline. Fourteen MLP2 units create common/equals carriers2^24 and homogeneous fields2^16. Thirteen MLP3 units copy the four numeric fields, amplify the counter by2^20, and produce Boolean units/hundreds flags using exactly shared input weights and bias pairs±8. The Boolean temperature is16384, from the declared[1024,2048,4096,8192,16384]. An earlier unmetered homogeneous-carry construction failed because float32 subtraction introduced excessive carry amplitude error.

Twenty-five MLP4 units write five phases at2^23, retain uncorrected homogeneous columns/reference at2^16, and compute tens/leading Boolean carries from the hand decimal truth table. The Boolean guard is.75, justified by reference/common<=1/16 and maximum inactive centered-threshold magnitude9.5/16. Phase temperature is1024, carry temperature16384, and native restoration uses128 donors. Fresh independent captures show all5,120 phases and all units/tens/leading Boolean decisions correct; seven to nine of1,024 problems have a column estimate farther than half a digit, depending on output stage. These diagnostics use separate column labels, never a whole-answer fit.

Thirty-three MLP5 units emit digits using conditional reference copies and cancel the phase carriers. The output amplitude is2^34, copy temperature32768, with816 native restoration donors. Declared output-gain/temperature pairs are[(2^32,8192),(2^33,16384),(2^34,32768),(2^35,65536),(2^36,131072)]; only the middle pair has been used. Newline amplitude is1/64 of the digit amplitude. An unmetered first version leaked from inactive output branches; replacing its baseline by2*phase−1.5*common fixed this. The recorded local arithmetic copy error is at most.00557 digit.

The first complete checkpoint scored97.25% arithmetic. Its old two-vector digit readout had biased rounding boundaries, sometimes near.24 rather than.5. A centered orthogonal-slope construction examined class-bias scales[256,512,1024,2048,4096] but failed against the full vocabulary. The successful replacement instead sets all nine half-integer logit equalities directly from the frozen embeddings and final-norm bias, then completes a two-dimensional nullspace Gram matrix to keep the original output norm polynomial exactly. This is a closed-form geometry calculation, with no answer targets or iterative fitting. Midpoint residuals are below2e−13; all ten synthetic integer argmaxes are correct, with minimum margin.1661. Replacing only these output columns raised G4 quick arithmetic to99.25%, with identical text metrics.

Preservation experiments remain unsuccessful so far. A one-shot attention2 residual correction in existing edited columns, perpendicular to arithmetic registers, gives97.41821% agreement and97.30% arithmetic with the old digit readout. Removing all native MLP donor corrections after the format gate worsens agreement to96.19751%; those corrections are therefore retained. Their declared strength family is[0,.25,.5,.75,1], with0 and the original1 used. The fixed five covariance ridges remain[.001,.0001,.00001,.000001,.0000001]. A new final-MLP capacity allocation is being evaluated using an unlabeled covariance basis and conditional feature residuals. All experiments and per-tensor counts are retained in immutable manifests. At the start of that evaluation,210/300 forward,69/100 quick and4/40 full units have been consumed (the ledger is authoritative).


## G4 preservation diagnostics — 23:21 CET

G4 still has no passing checkpoint. The best complete quick candidate so far is O14 (`experiments/o14_capacity/delta.pt`): 1,985/2,000 arithmetic, 97.49146% text agreement, KL0.0021512675, CE change−0.000166754. It retains the O13 arithmetic rows but selects the final MLP's 816 native reconstruction donors by an activation covariance basis. It changes4,240,966 scalars, maximum989,403.6875. G1–G3 banks remain unchanged.

Moving the output to MLP11 did not solve preservation. Late-output candidates with covariance-basis, raw-energy and mean-residual-projected native reconstruction reached respectively97.18018%,97.24121% and97.32666% text agreement; all reached99.35% quick arithmetic. The last uses a fixed projection perpendicular to the host post-MLP mean before measuring frozen-readout energy. Consolidating1,152 early donor corrections into1,968 final donors worsened text agreement to96.83838% and arithmetic to99.30%. A rank64 linear GELU-pair restoration plus560 retained native features gave96.45386% text agreement and99.35% arithmetic. These are closed-form covariance/SVD native-state projections, without gradients or answer targets. Only rank64 was tested. The late output used the second declared gain/temperature pair(2^33,16384), maximum863,391.6875, because(2^34,32768) exceeded the weight cap there.

Attention1 restoration reduced native calibration residual RMS from.04620 to.04165 but worsened quick text agreement to97.36328%; it was rejected. Separating local overwritten-neuron reconstruction from upstream correction (O19, strength0 from[0,.25,.5,.75,1]) gave97.35718% agreement and99.25% arithmetic, also worse. No other strength from this decomposition was tested. A constrained query-readout analysis tried text masses[.001,.01,.1,1,10] and two fifth-coordinate forms; its best reduced ordinary query RMS by only about5%, so no checkpoint was built.

A full text diagnostic on the pre-output O9 checkpoint found97.43271% agreement, KL0.0163633824, CE increase0.0153542618, all finite. G1 was selected only as the smallest auxiliary arithmetic suite; this checkpoint has no output circuit and is not a G1 candidate. Four independent calibration captures then exposed rare false program activations in MLP1,3 and4. These were inspected only on calibration text; no held-out text was opened. The first-event counts per capture were[0,1,2], [0,3,1], [2,29,18], and[0,3,62] across those three layers, although many later events share contamination from an earlier false activation in the same sequence.

O20 tightens the original whole-format interval to the larger calibration text maximum. The new negative bound is.7062445403, positive construction bound.7111636736, gap.00491913334; sharpness72/gap and copy penalty3504.784081 are derived from these bounds. It also replaces the MLP3/4 format-off directions with closed-form native-background projections using163,000 clean calibration positions and five hard-negative covariance masses[1,4,16,64,256], selecting1 for each. A first unmetered construction incorrectly penalized uncentered hard-negative responses and failed positivity; the corrected covariance measures deviation from the constrained unit-response text mean. New text-off minima are.33745/.32866, with program responses below1.38e−6/9.89e−7. All isolated phase/carry decisions remain unchanged on512 independent format examples. The narrower early copy gate increases numerical roundoff: maximum per-field change.00708; later copy changes are below.000024 except the counter(.00940). A full G4 score is currently running; no pass is claimed.

All builder sources are immutable once metered, and all trials are retained in manifests and analysis artifacts. Current budget after reserving the O20 full score:223/300 forward units,76/100 quick units and6/40 full units used. Final verification and export remain pending.


The completed O20 full G4 score is19,844/20,000 (99.22%), but text agreement is97.36557%, KL0.0040727920 and CE increase0.0021856017: the arithmetic criterion passes and preservation fails. Multiplicative-noise accuracies are8.6%,0%,0% at1e−5,1e−4,1e−3. Repeated k=3 first-problem accuracy is0/2,000 and second given a correct first is1,979/2,000; joint accuracy is0. The gate is not established for a first equation without the G4 prefix. No bank was added.

O21 assigns816 native donors independently to every final-MLP output coordinate, using fixed dense-projection contribution rankings followed by one covariance solve per support. Isotropic ridge values[.0001,.001,.01,.1,1] were compared on native calibration-state reconstruction, choosing1. Dense validation RMS bottoms out at.08093; sparse RMS is.08147 versus.08347 previously. The sparse supports use2,985 native features in total while keeping the same4,240,966 edited scalars. Quick arithmetic is99.25%, text agreement97.48535%, KL.002105989 and CE change−.000179527. This is essentially unchanged preservation.

A four-head uniform-attention alternative was rejected before constructing a checkpoint: on the recorded attention component its weakest four heads produce quadrature native output error.22045 RMS; even half-strength native attention gives.10294. A more focused query-field analysis clears only known injected position fields and the native equals mean. This reduces native equals-query variability on most heads with ordinary query changes.014–.060 RMS, but leaves large paired-source score differences on some heads. The next design therefore retains native attention on prose and uses stronger equals-only writer fields plus explicit native-field removal to permit a paired-column reader. This is an ongoing design, not a result.

## G4 paired-reader alternative — 01:02 CET, 10 November

G1–G3 remain banked; G4 is not passed. The complete O20 arithmetic result remains99.22% with97.36557% text agreement. A new P design has reached an intermediate Boolean source stage, but has no generated-answer result yet. Its best current reader preserves98.37036% quick text agreement; adding the intermediate copy stage gives98.18115%, still below98.5%. Final verification and export remain pending.

P replaces seven individual-source heads with four heads: three read paired decimal columns, and one reads the plus sign. Each digit is represented by four exact current-token bits, derived from a hand per-digit truth table. OR and AND thresholds on the weighted bit values recover each pair's bit count when both operands receive sufficient attention. The address uses modulo-four cosine/sine plus linear recency. A protected four-position binary flag distinguishes the two spaced operands; the first column separately checks bare and spaced token types. The complete-format reference is a hand conjunction of seven source-presence checks and the current equals token.

The coordinate projection now has ten directions. Its covariance masses remain[0,.01,.1,1,10] and mean-restoration fractions[1,0]. The current writer has41 units and20 fields, with gains2 for token/bit flags,8 for source positions and256 for equals positions;512 native donors use covariance-based reconstruction. Query-gain family[256,512,1024,2048,4096] has used2048 and256; source-position family[8,16,32,64,128] has used8. A numeric-position flag was built first, then tightened to a spaced-digit-position flag before the current reader. The earlier P2 full preservation diagnostic was98.86627%; P2D's revised native capacity gave98.99292% quick agreement. Those are incomplete frontends, not rung passes.

Quadratic paired addressing failed because small source-position errors let older prefix numerals compete. Its tested curvature values were[1/1024,1/512,1/256,1/128,1/64]. Linear recency succeeded on recorded attention components. Cosine gains[128,256,512,1024,2048] and bit fractions[.05,.1,.15,.2,.25] were checked; the current reader uses256 and.1. The hand recency ratios are4 between paired operands and256 for the operator. Equals-fallback shifts were declared[0,−512,−1024,−2048,−4096]; zero and−2048 have been used, with−2048 selected to keep attention on operands. The word penalty is128. No full-answer targets enter these calculations.

The reader reuses four low-energy native QK coordinates and6/6/6/1 value coordinates after32-coordinate native value rotations. Nineteen memory fields are protected from untouched output heads. Memory widths[32,48,64,96,128] were examined, selecting32 by recorded background precision. It changes1,737,126 scalars, with maximum edited magnitude85,563.796875. Initial value gain2 gave98.34595% quick agreement and7/1,024 fresh examples with a bit error. The declared gain family[.5,1,2,4,8] was examined on a recorded attention-output piece; only2 and4 were metered as readers. Gain4 gives98.37036% agreement, KL0.0049917393, CE increase0.0033232159, and2/1,024 fresh examples with a bit error. These are per-input-bit diagnostics, not generated addition scores.

Native-noise duals with covariance masses[.0001,.001,.01,.1,1] reduced construction-set background error but worsened independent bit checks, so they were rejected. An earlier equals-off construction also failed after overlooking native propagation of the strong query fields. A separate local-source syntax detector with regularizers[1e−5,1e−4,.001,.01,.1] did not separate its controls and was not built.

The current22-unit MLP1 copy uses an equals-token gate with masses[.001,.01,.1,1,10], selecting1 (recorded separation gap0.93088), and768 native donors. It copies the nineteen values without quantizing them and writes a2^24 common carrier. Signal gain2^16 was built but not metered;2^18 is current, from[2^14,2^16,2^18,2^20,2^22]. Quick agreement falls to98.18115%, KL0.0060480543, CE increase0.0038354308. This is an intermediate equals-only stage; the final arithmetic output will require the complete format.

The next84-unit MLP2 consists of48 bit-threshold units,14 syntax units,2 carrier clips and20 cancellation units. Its tested sharpness values are[16384,32768,65536,131072,262144];131072 is the largest within the weight cap (maximum input weight900,771.97). The next value exceeds the cap and is not used. The syntax fraction is.075 from[.025,.05,.075,.1,.125]. On1,024 fresh recorded valid prefixes, one example has a hard bit error and five have a bit deviation above.001; all valid syntax references are separated from512 malformed controls. The signed complete-format reference is twice the seven clipped flags minus thirteen times the equals flag. The builder is still being checked; no score or rung pass is claimed for it.

All native reconstructions remain closed-form covariance/SVD calculations with the recorded ridge family[.001,.0001,.00001,.000001,.0000001]. There have been no gradients, optimizers, whole-answer solves, internet access or subagents. Superseded activation files were deleted with SHA256/size cleanup records to stay below25GB; the host, meters, data, ledger and banked checkpoints remain unchanged. Remaining budgets before the Boolean-source score are53 forward,17 quick and33 full runs.

## G4 paired-reader completion — 02:23 CET, 10 November

P10B is a complete alternative circuit. It scores **1,997/2,000 (99.85%)** on quick G4, but **does not pass** because ordinary-text agreement is **96.38062%**, below98.5%. KL is0.0093809478, CE increase0.0040012806, and all values are finite. It changes4,248,767 scalars, below4,252,723; its maximum weight is894,175.125. No full or noise score has been run for P10B. The earlier O20 full result remains99.22% arithmetic with97.36557% text agreement, also a failed rung. No G4 bank has been created.

P5 separates a signed whole-format marker from the arithmetic reference. Its initial quick text agreement was95.24536% because individual syntax units falsely activated on rare ordinary tokens. An ablation retaining native reconstruction but zeroing program outputs recovered97.69135% full text agreement. Four new calibration captures isolated false flags, and a factor2 margin over those negatives recovered97.55859% quick agreement with KL0.00754118. The safety family is[1,1.25,1.5,2,4], with1 and2 used; bit/syntax temperatures are32768/65536 from the previously declared five values. Source marker/reference gains are2^20, column gain2^16. One later fresh set had three column-reading errors among1,024 examples; a separate final-input set had one upstream format rejection and one active numeric error. These are component diagnostics, not passing scores.

P6 broadcasts the four numeric/reference fields using one attention3 head. Score32 is selected from[16,32,64,128,256], gain64, and four protected output heads from counts[1,2,3,4,5]. The later-position gate is a single covariance-derived syntax detector, using masses[.001,.01,.1,1,10] and selecting.01. It includes malformed controls except two already misclassified by the upstream whole-format marker; those two remain disclosed failures.

Twenty MLP3 units compute carries directly from the source columns, emit common/equals markers and copy four homogeneous scalar fields. For centered columns H,T,U and reference R, carry tests are U−.5R, T+.1U−.05R and H+.1T+.01U−.005R. Complementary carry pairs keep the dominant equals norm fixed. Marker/pair gain is2^23. The first scalar gain4096 failed through float32 cancellation;262144 is current. Carry temperatures changed from a shared131072 to[4096,8192,65536] for units/tens/leading, with shared-weight bias clips±8. No answer labels determine these coefficients.

The attention4 counter stores the position count and three signed carry counts directly. Its score16 comes from[8,16,32,64,128], value gain128, and two protected output heads are selected from counts[1,2,3,4,5]. It uses the measured dominant-norm ratio2.2436 and later-position covariance. Earlier variants allocated five heads and attempted to read the count at equals; residual roundoff made this unreliable. The final leading digit uses the direct equals carry. In a fresh1,024-example capture, every later phase and carry sign was correct, with no column error above half a digit.

A direct final MLP4 design failed before metering: close but independently rounded FC rows lost tiny numeric differences, and some weights exceeded1e6. The replacement28-unit MLP4 emits five phase fields and three carry fields, while cancelling prior markers. Shared-row bias clips use±2; temperature trials[2^23,2^24,2^25,2^26,2^27] select2^25. Larger values exceed the cap. Phase amplitudes deviate by at most.0566 on construction inputs, but their half-threshold classification is correct. Phase/flag gain is2^23 and native restoration uses128 donors.

The final39-unit MLP5 uses conditional reference copies and the exact frozen single-digit midpoint geometry, then cancels phase and carry fields. Output gain/temperature(2^36,131072) exceeded the FC cap and was rejected;(2^34,32768) is used, with544 native donors. Active captured digit-copy error is below.005. Native restoration uses the same five closed-form ridge values[.001,.0001,.00001,.000001,.0000001], selecting.001. P10B's failure is preservation, so the next work targets native attention and edit allocation. The run remains active; fresh bank verification and final export have not yet been performed.


## Sparse frontend, native-memory variants and natural-token probes — 04:25 CET, 10 November

G1–G3 remain the only banked rungs. P11 restored native query responses to nine non-address writer fields and reached99.85% quick G4 arithmetic with96.39282% text agreement, KL0.00922831 and CE increase0.00387960. The change was too small to resolve preservation.

Q1 restricts the ten protected input directions to a fixed coordinate support. Widths[128,192,256,384,512] and covariance masses[0,.01,.1,1,10] were tested against exact token/position constraints and positivity on every other vocabulary token. Width128 failed;192/mass10 was selected, with off minimum0.0180035. Rebuilding the token writer on this support freed442,941 edited scalars. The resulting complete Q3 has3,805,826 edits, maximum903,812.625. Its quick G4 result is1,986/2,000 (99.3%) with96.24023% text agreement, KL0.00918870 and CE increase0.00284419. Q4 allocated the freed capacity to1,088 native MLP2 donors instead of512. Native validation RMS improved only from0.07015 to0.06877; arithmetic stayed99.3% and agreement became96.29517%. Q4 changes4,248,194 scalars. Neither passes.

A19-dimensional native memory transport was rejected before metering. Five native covariance ridges selected0.001; interpolating its linear map with identity by[0,.25,.5,.75,1] produced[0,3,8,92,495] examples with a bit error among512. Shrinking the program signal amplified background errors. A separate cleaner-memory analysis varied protected head counts[4,5,6,7,8], background subspaces[32,48,64,96,128], and gains[.5,1,2,4,8]. It selected seven heads, width128 and gain1, with one bit-error example among512. Q5 rebuilt the current-equals copy and used1,152 native MLP1 donors; all later arithmetic weights were fixed. It changes4,248,194 scalars, maximum903,812.625.

The full Q5 result is **19,152/20,000 (95.76%) G4 arithmetic and96.45996% text agreement**, KL0.00872788, CE increase0.00374891. Both thresholds fail. Noise accuracies are0% at all three sigmas. Repeated-problem first accuracy is0/2,000, second given a correct first692/2,000, and joint0%. A subsequent current-equals capture found one erroneous reader bit, two source-column errors above half a digit, and fully saturated syntax flags among512 independent calibration examples. Those early checks do not explain most of the full-run failure; later-stage sensitivity or domain shift remains unresolved. No bank was added.

Natural-format G5 work is still at the lexical-component stage. GPT-2 tokenizes0–999 into819 single-token numbers and181 two-token numbers, using825 distinct numeric chunks. Native MLP0 GELUs permit direct current-token digit/length/punctuation maps: an initial8,192-observation capture used6,144 construction and2,048 independent contexts. Exact class-mean interpolation with covariance minimization tested supports[1024,1536,2048,2560,3072] and word masses[.001,.01,.1,1,10]. Several supports achieved2,048/2,048 correct lexical decisions. These labels name only one token's digits, length or punctuation; there are no operand-pair or sum targets.

A wider11,072-observation capture covers positions2–63 and prefixes1–40, adding spaced digits and operators for840 lexical classes. It has2,762 independent validation observations. The full native-feature map at masses.001/.01 classifies every lexical field correctly. Position estimates are inadequate: the best direct linear-position estimate has0.558-token RMS error and5.423-token maximum. Protected positional projections followed by current-token embedding cancellation also fail: periods[8,16,32,64,128] were evaluated, and even the best period128 construction has42/2,762 phase errors above half a position. These are failed addressing components, not generated G5 addition results. No G5 checkpoint has been scored beyond the frozen host baseline.

An unmetered sparse lexical-memory design tested output supports[64,96,128,192,256] and gains[.03125,.0625,.125,.25,.5]. A fixed constant reference and native residual cancellation at256 rows/gain.25 retained2,048/2,048 lexical decisions, but predicted native text-state change was0.06085 RMS. Its native predictor used the established five covariance ridges. Conditional native-cancellation matrices and the failed position maps are retained for diagnosis; no language-model output was used to fit them.

T returns to individual G4 source positions. Its first seven-head squared-distance/cosine reader was rejected before a checkpoint was written: native background in the quadratic coordinate caused roughly31–45% of selected source positions to be wrong. The complete five overall scales[1,2,4,8,16] were recorded, with the final scale also exceeding the key-weight cap. Raising only the writer's equals flag from2 to16 used the declared family[2,4,8,16,32].

The current T2 writer instead places its20 fields in a frame orthogonal to the22 allowed token embeddings, all64 positional embeddings and the existing protected input frame. Attention0 and native MLP0 outputs are projected off those fields. The support has256 rows and a159-dimensional constraint nullspace; constraint errors are below2.1e-15. Its fixed selection metric is native-attention covariance plus native-MLP covariance plus ten times token covariance. It uses41 unchanged hand-coded token/position rows,128 global native donors, and full native correction within the256 already-edited output rows. Both reconstruction steps use the established five ridges, selecting0.001. A first local assertion incorrectly expected zero native embedding components on non-format words; the corrected diagnostic includes those components explicitly. On valid formats the fields differ from their hand formulas by less than0.0004, and the native input frame is below5e-7. Native text-state reconstruction RMS remains0.09628. The completed full auxiliary score gives97.24045% text agreement, KL0.00334340 and CE change−0.00010758, so preservation fails. As expected, this writer alone answers0/100 G1 problems. G1 is used only as the smallest auxiliary suite; this writer has no addition-output circuit and is not a G1 candidate.

T2 changes1,102,121 scalars, maximum281578.5625. Its per-tensor counts and all builder/input hashes are in`experiments/t2_clean_writer/delta.manifest.json`. Remaining budgets at this update, after the completed T2 full score, are23/11/30 forward/quick/full. Final fresh-seed bank verification and export have not been run. Superseded activation files continue to be removed with hash/size records to stay below25GB. The host, frozen tools, calibration/held-out data and existing banks remain unchanged.

## Period-eight individual readers and source guards — 06:15 CET, 10 November

No additional rung has passed. The U and V designs below are incomplete G4 frontends; their full auxiliary G1 runs measure preservation and are expected to answer zero G1 problems because they have no answer-output stage. Their failures do not replace the banked G1 checkpoint.

A conditional native-restoration analysis of T2 constrained all44 current-token/position-half class means to zero and varied numeric covariance weights[1,10,100,1000,10000]. Native-word frame RMS improved from.45429 to between.08478 and.30537, but the smallest independent position-field error was.004275 and the smallest bit/kind error.03724. Neither met the component precision bounds; no candidate was built from this analysis.

U adds two exact period-eight positional coordinates on the existing192-row support. Its41-unit writer retains20 fields with type/bit gain2, source gain8, equals-query gain256 and equals-reference gain16. Seven attention1 heads use four QK coordinates each for a cosine-plus-recency kernel. For offset d, its programmed score is proportional to Cq*Cs*[16+256*(cos(pi*(ps−pq+d)/4)−1)+4*(ps−pq+d)]. Five overall scales[1,2,4,8,16] all select the correct source in512 recorded examples; minimum target weights are.83316,.94568,.99530,.999968 and effectively1. Scale2 is sufficient for bit thresholding. Value gain2 comes from[.5,1,2,4,8]; memory widths[64,96,128,192,256] select256. The reader stores24 individual bits and seven expected-kind presences, protecting their31 directions and the equals reference from native attention output.

U2 changes2,056,520 scalars with maximum281579.96875. Its512-example attention-piece check has no bit errors, but full text agreement is97.75925%, KL.00421396 and CE increase.00189775. A52-unit direct-source stage first used a temporary homogeneous constant. Candidate constant gains[1,2,4,8,16], covariance masses[1,10,100,1000,10000], thresholds[512,1024,2048,4096,8192] and safety factors from[1,1.25,1.5,2,4] were examined by local margins. The initial homogeneous version chose gain16/T2048/safety2; the metered version used gain8/T8192/safety1.5. It changes2,727,035 scalars and scores97.08939% agreement, KL.00562220 and CE increase.00302377. The added constant was rejected.

U3B instead uses a current-token equals off direction, exact on144 construction equals states and obtained by covariance ridges[.001,.0001,1e-5,1e-6,1e-7]. A continuous seven-presence gate failed11 of512 additional valid formats despite correct bits. Replacing it by seven separately clipped syntax flags yields64 units:48 bit units,14 syntax units and2 equals-reference units. The signed complete-format marker is2*sum(flags)−13*EqRef. At temperature8192 and safety1.5, all512 recorded valid formats and96 malformed controls pass their local tests. With768 native donors, U3B changes2,745,480 scalars and scores97.62802% text agreement, KL.00650751 and CE increase.00357640. Preservation still fails.

V avoids removing all31 native memory components. It moves the writer's equals-reference frame into the nullspace of five untouched attention1 projection heads, protects that frame only within the seven already-edited heads, and superposes the31 other memory fields on native attention output. This saves245,759 changed scalars. Its source current-token maps use covariance masses[0,.001,.01,.1,1], with kind/key mass1, value-kind mass.1 and value-bit mass.01. V's address family[1,2,4,8,16] has minimum target weights.91188,.98743,.999779,.999999933 and effectively1. The original scale2 reader and subsequent scale1 variant both pass all512 bit checks.

All V2 readers change1,810,761 scalars, maximum281579.96875. The changed tensors are attention0 output weight/bias (147456/192 scalars), MLP0 input weight/bias (31488/41), MLP0 output weight/bias (424704/768), attention1 QKV weight/bias (860160/1120), and attention1 output weight/bias (344064/768). Full preservation results are:

| Candidate | Change | Top-1 agreement | KL | CE increase |
|---|---|---:|---:|---:|
| V2 | Native memory retained; address scale2 | 97.85767% | .00403567 | .00174335 |
| V2B | Address scale1 | 97.94617% | .00380621 | .00146723 |
| V2C | Scale1, source-word penalty16 | 97.95609% | .00383490 | .00157413 |

Source-word penalties[8,16,32,64,128] were tested on a recorded attention piece. Penalty8 gives three bit-error examples, four format rejections and six wrong source argmaxes. The other four values have no recorded errors; their minimum target weights are.9999957,.9999892,.9997745 and.9117923. Penalty16 improves selection but barely changes text agreement. Work therefore continued with the already-captured V2B.

V3 uses64 MLP1 units and960 native donors. Its off detector has528 current-token equals constraints plus31 memory-insensitivity constraints, with the same five covariance ridges and selected.001. The remaining128 full-prefix examples are independent validation. Temperature family[512,1024,2048,4096,8192] selects1024 by clip margin and float32 accuracy;512 has insufficient margin. The equals-carrier threshold and temperature are computed directly from the positive/negative separation gap, rather than searched. Source marker/reference gains are1048576 and column gain65536. On96 valid mixed examples, maximum column error is.0002451; all96 malformed controls reject. The larger512-example piece has no bit errors and minimum valid marker.999848.

V3 changes2,647,177 scalars, maximum281579.96875. In addition to V2B, it changes MLP1 input weight/bias (49152/64) and output weight/bias (786432/768). Native validation RMS falls from.111833 to.061151. Nevertheless, full text agreement falls to97.43881%, KL.01503560 and CE increase.01243795. A zero-program-output ablation retaining identical native repair gives97.74170%, KL.00381439 and CE increase.00138153. Both native preservation and program emissions therefore need improvement.

A fresh calibration-only capture finds two unwanted digit-bit emissions in32768 tokens; no equals reference or syntax marker fires there. Guards were recomputed against55360 non-equals calibration/control positions using safety values[1,1.25,1.5,2,4]. Values1.25,1.5 and2 pass every recorded bit/format test;4 creates one bit error, and1 misses the strict negative-preactivation margin through rounding. The new strong candidate chooses2, with minimum valid clip margin16.618 and zero observed negative emissions. Its full score is in progress at this update. An earlier attempted score of an unscored weaker build was rejected by the source-hash guard after an attempted output-path reuse. The original source and geometry hashes were restored, the original delta was left unchanged, and the stronger version uses distinct v2 source/geometry files and a new output directory. Ledger0749 records this failed attempt and its consumed full-run reservation; it is not a score result.

W is a new component probe using frozen-host GELU features from layers1,2,4 and8 to decode current-token syntax roles and individual operand-digit bits. Its targets never contain sums. The single frozen capture combines192 independent contexts from each of two valid domains,64 malformed controls and64 native calibration windows. The malformed selection currently covers role0 only; this is not a complete negative-format validation. No W weight checkpoint or generated arithmetic result exists yet.

As of ledger0753's reservation, remaining budgets are17 forward,11 quick and20 full. The directory is approximately24.80GB; old U2/V2 mixed captures were deleted with recorded hashes and sizes, while current inputs, checkpoints, sources, manifests and scores remain. No host, protected tensor, frozen meter, calibration/held-out data or bank has been changed. Final fresh-seed verification and export remain pending until finishing.

## Late native-role parser — 07:55 CET, 10 November

The completed strong V3C guard scores97.70508% text agreement, KL.00568973 and CE increase.00299217. It remains a preservation failure. No subsequent V arithmetic stages were built.

W uses native layer4 features to recognize each current token's role. Per-role class means, current-token kind labels and individual digit bits determine its linear maps; no sum labels enter them. Role covariance masses[.001,.01,.1,1,10] were tested across native layers1,2,4,8. Layer4 masses.1/1/10 recover all six digit sources in192 independent contexts when restricted to the eight format positions; separate direct per-place operand decoders at equals fail every complete six-digit example. Current-token maps use negative vocabulary supports[128,256,512,1024,2048]. Combining3072 GELUs with768 normalization inputs gives zero program bit errors. Actual prefix competition, with computed token-kind masks rather than oracle masks, selects all seven sources in192/192 validation contexts. Role mass10 and token support128 are used. Recency coefficients[0,.002,.005,.01,.02] and attention scales[8,16,32,64,128] were examined as local pieces; zero recency already works.

W1 moves these15 soft fields plus a temporary reference into sixteen residual directions. Thirty-two GELU identity units implement the linear-input part, while128 native donors reconstruct displaced features. The retained-feature transfer uses five one-shot covariance ridges[.001,.0001,1e-5,1e-6,1e-7]. Frame supports[32,48,64,96,128], payload gains[.25,.5,1,2,4] and reference gains[1,2,4,8,16] select128 rows, payload4, reference4. Attention5 is projected off the frame; its16-dimensional removal changes only128 output rows. Local float checks recover192/192 source selections and all digit bits. However, full auxiliary preservation0757 is only94.38934%, KL.00734099, CE increase.00961763.

W2 restores native memory components conditionally. It constrains66 current-token/role class-mean corrections to zero and evaluates program covariance masses[1,10,100,1000,10000]. Mass1000 is selected. Role/bit gain stays4, bare/spaced/plus gains become8, equals gain16, reference4. Native validation frame RMS improves1.21483→.74333, with192/192 source selections, zero bit errors and zero observed false equals. Full0759 agreement improves to95.51086%, KL.00473220, CE increase.00944060, still a failure. These are incomplete parsers without addition outputs; their auxiliary G1 arithmetic is zero.

Both W checkpoints change619552 scalars: MLP4 input weight/bias24576/32, MLP4 output weight/bias495616/768, attention5 output weight/bias98304/128, and MLP5 output bias128. Maximum edited magnitude is2.55331; the largest magnitude in the whole checkpoint is17.41932. Exact source/input hashes, per-tensor counts and scale trials are in their manifests. These sources are now frozen.

A new full mixed capture0761 records layer5 raw, normalization and GELU values, plus attention6 values, for320 native windows and192 format/control windows. Its96 malformed inputs explicitly cover all eight corrupted positions, twelve each. The original W malformed sample covered only role0, and the earlier U mixed sample covered only roles0,2,5; neither is presented as exhaustive malformed coverage.

Current work derives a20-unit kind-gated copy at MLP5 and a larger native restoration there. It copies soft digit bits only at numeric tokens and leaves hard bit thresholds to the reader's next layer. A simple numeric-off guard and a16-dimensional constrained numeric-off projection both fail five or more independent digit examples, so neither is used. The latter used covariance masses[.001,.01,.1,1,10] and safety factors[1,1.25,1.5,2,4]. Source-key contrast coefficients[0,.25,.5,.75,1] and type penalties[0,.25,.5,1,2] were also analyzed on recorded fields. The new copy's temperature family is[512,1024,2048,4096,8192], fixed copy scale16; source-reference gains[128,256,512,1024,2048] are reserved for its local normalization checks. No complete W arithmetic result is claimed.

Remaining budgets are15 forward,11 quick and18 full runs. Cleanup records preserve hashes/sizes for deleted superseded V captures, while current W/native-host inputs remain available. The directory remains below25GB. BanksG1–G3 are unchanged, with final fresh-seed verification and export still pending. The run is active and not blocked.


## Native-input Y writer and guarded reader — 09:33 CET

G1–G3 remain the only banked rungs. G4 arithmetic is still unfinished. The new Y branch avoids the earlier soft-memory frontend: seven current syntax roles and four current-token bits are read directly from recorded native layer-4 normalized inputs. All solves label single tokens, roles or current positions; none uses a whole-problem answer. A 23-unit hand MLP copies gated role keys, bit values, type presences and recency values. Separate role certificates will gate the later arithmetic.

The Y1 writer alone changes 228,116 scalars and passes auxiliary full text preservation: agreement 98.83041382%, KL .00777237436, CE +.00684212316. Native-only and no-equals ablations in quick mode score 99.27978516% and 98.97460938%; the full writer quick score is 98.86474609%. These are incomplete arithmetic circuits, not rung passes. Y2 adds three recency copies, increases native reconstruction donors from 256 to 1280, and lowers the equals-query gain from 1048576 to 16384. It changes 1,019,157 scalars, max |w| 825.6402588, and passes auxiliary full text preservation at 98.88305664%, KL .00803590288, CE +.00722349955. Arithmetic is zero because no output program is present.

Three one-shot native whole-format equals discriminants failed separation (best gaps -.166916, -.1363245 and -.1025417). They are not used. Y2 therefore has a current-equals addressing query, not a complete format certificate. The planned arithmetic gate is the AND of seven separately clipped source-role flags, after which strong column/carry fields may be emitted. The earlier query is explicitly cancelled at the next MLP.

The first unmetered seven-head reader found all 384 correct sources but native residual background spoiled its malformed-input certificate. Balanced controls exposed a second limitation: original token maps had only corrupted the first input role. Revised maps include 96 balanced construction controls and retain 96 controls for validation. They classify all 576 old/fresh valid source examples and every reserved malformed token type. The new 23-unit writer suppresses all 192 corrupted current tokens and has no equals emissions on 32,768 native calibration positions. Its native source activation count is 43; this is a measured calibration property, not a guarantee for unseen text.

Y3 combines the balanced writer with a seven-head attention reader. Partial 12-coordinate Q/K and 16-coordinate value rotations retain native capacity. One Q/K coordinate per head addresses its role; an analytic compensation in K removes the nearly constant native equals-query score. Each digit head reads four soft bits, presence and recency-subtracted role confidence; the plus head reads presence/confidence, 38 fields total. Selected attention scale64 has minimum true-source probability .9368976 on the captured valid inputs. A joint native/programme covariance frame and value gain64 recover all384 operand bitsets with minimum bit margin .2409883. Seven independent syntax thresholds accept all384 valid examples and reject all64 balanced malformed examples in this one-attention-piece check. This is not yet a scored complete circuit. Y3 changes 1,579,309 scalars, max |w|825.6402588: layer4 FC17,664+bias23, P1,000,702+bias768; attention5 QKV215,040+bias280, P344,064+bias768. A frozen full mixed forward capture is now running before Boolean/carry logic is added.

All additional finite families are retained in the versioned scripts, JSON diagnostics and checkpoint manifest. Native-role/token covariance masses [.001,.01,.1,1,10]; negative-token supports [64,128,256,384,512]; recency [0,.002,.005,.01,.02]; position-projection ridges [.001,.0001,.00001,.000001,.0000001]. Writer confidence multipliers [0,.25,.5,1,2], positive margins [.05,.1,.15,.2,.25], temperatures [256,512,1024,2048,4096], copy scale16, source gains [32,64,128,256,512], equals gains [256,1024,16384,262144,1048576] (only16384 used by Y2/Y3). Balanced writer selects role mass.01, token mass1, support64, recency.02, confidence2, margin.25, temperature256. Source readout word-covariance masses [.001,.01,.1,1,10], fixed ridge1e-7, choose10. Reader attention scales [16,32,64,128,256], value reconstruction ridge.001. Failed memory-v1 widths [64,96,128,192,256] and gains [2,4,8,16,32]; memory-v2 joint programme masses [.001,.01,.1,1,10] and the same gains. Common syntax theta [0,.125,.25,.375,.5] and reference floors [.03125,.0625,.125,.25,.5] did not certify all controls. Memory-v3 introduces independent per-role thresholds and gains [32,64,128,256,512], same joint covariance masses/theta family, reference floors [.125,.25,.5,1,2]. Selected mass1/gain64 has native attention-increment validation RMS .09450766. Its seven (theta,floor) pairs are (.125,2),(.5,1),(.375,2),(.5,2),(.5,2),(.375,1),(.375,2). Native MLP5 reconstruction and cancellation are still required before judging preservation of the complete frontend.

Before the latest forward request, 13 forward, 8 quick and 15 full units remained. Final fresh-seed verification/export have not run. Verification requires one fresh full score per bank, so those units remain reserved. Own obsolete activation captures were removed with hashes in cleanup_20261110_0825.json and cleanup_20261110_0931.json to keep the folder below25GB; source scripts, geometries, manifests, scores and banks remain. Work continues; no external blocker has been found.


## Compact Z format gate and capacity reallocation — 10:43 CET

G1–G3 remain banked; G4 and G5 remain unfinished. Y3 passed its frozen local source/format checks, but the Y4 intermediate Boolean checkpoint failed quick text preservation: agreement**96.44165039%**, KL .01015069340, CE +.00655214623. Its64-unit MLP5 copied24 bits and7 role flags. Native-attention ablations attribute most reader disturbance to reused value channels and the added value signals: native-window RMS .01958 for the Q/K rank removal, .02615 for programmed Q/K, .07431 for value-capacity reconstruction, and .12389 for the complete reader. Protecting the equals direction accounts for only .00323. Two global output-projection repairs gave too little improvement and were not used.

Z retains the direct native-input writer but centers each soft bit at bit-minus-one-half, separates recency from role keys, and removes a duplicate source-reference field. The reader selects low-cost value directions from each full64-coordinate head, reusing5 per digit head and1 for plus:24 signed bits and7 continuous role certificates in31 fields. It keeps the native equals-direction component. The selected gain64 reader passes384 recorded bitsets with raw signed margin4.89388 and all64 initial malformed controls; native word-state increment RMS is .072838. A lower-gain32 version also passes those source checks but has raw bit margin1.93976, so gain64 was retained for the intervening native layer6 attention. Z1 changes1,837,693 scalars, max|w|825.6402588. A fresh frozen capture revealed two additional malformed cases that pass the individual certificates. A fixed paired certificate on roles2 and6 separates both from the valid cases; its threshold15.1283311671 is the midpoint of the observed valid lower bound16.1427761 and hard-negative upper bound14.1138863. All21 pairs were compared as format detectors; no arithmetic answers were used.

Z2 replaces Y4's64 units with10: seven single-role violation GELUs, one paired-role violation GELU, and a two-unit equals reference/cancellation clip. The nonnegative violations share one residual field. The original signed operand fields remain in place. Every valid captured example and native non-query position emits zero violation in the hand piece;140 malformed cases with an active equals query emit54.21–2485.44, while20 corrupted-equals controls have no query. The new reference is8192; the prior16384 query is cancelled. An equals-only constant also cancels the mean native MLP5 contribution in operand-memory directions. Native MLP5 reconstruction uses768 donors. In the actual isolated MLP piece, all96 valid bitsets remain correct with margin4.87407; valid violation background is -2.383 to2.275 and malformed active-query violations are at least53.763.

The Z2 quick score still **fails preservation**, although it improves over Y4: agreement97.52807617%, KL .00693803370, CE +.00511308609, finite. It changes2,443,655 scalars, max|w|58074.35156. It is an incomplete arithmetic circuit, not a rung pass. The remaining deficit motivates reallocating native reconstruction capacity rather than treating local arithmetic checks as success.

A targeted covariance reconstruction of the fixed23 displaced layer4 features needs only256 donors, compared with the previous1280. Its native validation RMS .02084784 is slightly lower than the old .0211597. The unchanged hand program still selects all seven sources and all bitsets correctly on384 recorded complete prefixes; minimum source margin .06813 and signed-bit ratio margin .17944. Z2B applies this native-only reallocation, reducing actual changed scalars to1,657,224, a reduction of786,431. The builder description uses the nominal block-size reduction786,432; the actual sparse count differs by one float that matches the host exactly. This leaves substantial capacity for native restoration at layers6 and11. No additional claim of text preservation is made before a score.

The frozen Z2B layer6 capture has completed. Before offset correction, native attention6 causes13/96 operand-bitset errors, with minimum signed margin-5.429. Valid violation values are6.033–22.497 and malformed active-query values remain at least59.577. The next step measures the native attention contribution from an isolated MLP5 piece on recorded inputs and the actual layer6 capture. That baseline estimate explicitly includes the small earlier native4 donor change; it is not claimed to be an exact same-checkpoint attention output. No bit or answer labels enter the offset calculation.

New finite families: Z value gains[8,16,32,64,128], programme covariance masses[.001,.01,.1,1,10], role reference floors[.125,.25,.5,1,2], full value basis64 versus the previous16; selected mass1/gain64. The compact defect stage tests off-safety[1,1.25,1.5,2,4], temperature[8192,16384,32768,65536,131072], and violation gain[8192,16384,32768,65536,131072], with all125 combinations recorded. It selects1.5/8192/65536; equals temperature128, clip halfwidth2, reference8192. Native4 donor counts[256,384,512,768,1024] each use ridge[.001,.0001,.00001,.000001,.0000001]; selected256/.001. Earlier global native-P repair masses[1,10,100,1000,10000] use the same five ridges. Exact pair bounds, all scale combinations, reconstruction diagnostics and provenance hashes remain in the versioned JSON/manifest artifacts.

Remaining budgets after the latest forward reservation:10 forward,6 quick,15 full. Fresh final verification remains reserved and has not run; no export has been finalized. Obsolete Y3 and Q1 activation captures were deleted with hash/size records; all banks, source scripts, checkpoints, manifests and scores remain. Work continues toward the remaining rungs until the stated deadline.


## Complete Z8 prefixed-addition candidate — 12:05 CET

G1–G3 remain banked. Z8 is a complete G4 candidate and reaches **1,985/2,000 = 99.25%** in quick arithmetic, above the99% arithmetic threshold for that sample. It **fails ordinary-text preservation**: agreement95.67871094%, KL.01291812880, CE increase.01236753137, all finite. Therefore it is not a G4 pass and is not banked. No full/noise score has yet been run for Z8. The15 reported failures comprise twelve errors at the first output token and three later digit errors. G5 remains unfinished.

The source6 clip temperature is increased from1024 to4096 within its declared five-value family after three of384 captured examples had correct bit signs but unsaturated clips. The revised source has maximum independent-column error.00173 on those384 examples and all carry signs correct. Attention7 broadcasts the three centered columns and reference using one head, four protected output heads and gain256; maximum later-column error is.1121 in its local component check. The fresh meter capture gives.09775 maximum later-column error on96 examples. MLP7 writes canonical common/equals markers, scalar copies and three complementary carry pairs. All384 independently captured carry predicates are correct, including their format gate;64 malformed controls have no common marker.

Attention8 counts marked output positions and averages signed carry flags. Clearing its selected head's native response to the strong fields, retaining native value capacity, and choosing four protected output heads/gain128 gives maximum later counter error about.014 and signed-carry error.0172. Native attention-increment validation RMS is.05945. An alternative two-head/gain256 version costs fewer edits but increases native RMS to.11078, so it is not used. Optimizing the four value maps for actual attention-weighted native context covariance improves RMS modestly; it preserves exact responses to the fixed program fields. Equals uses its direct point marker, so the noisy equals counter is not used.

The phase builder initially failed because native-word suppression consumed the small counter margin. A minimum-background counter-readout refinement also failed3–4 phase checks and is unused. An exact format-only off detector constraining192 construction positions (the four later positions in48 examples), with native mean1, solves this without arithmetic targets. Its reserved later-position off error is at most1.082e-6, native off values at least.81308. Equals remains positively off and is handled separately. Counter clip halfwidth1 instead of2, at temperature2^25, yields maximum phase-field amplitude error.08068, correct phase above.9 and incorrect phase below.1. MLP8 uses28 units,128 native reconstruction donors and maximum new magnitude917771.18. Layers9/10 remain native. A fresh final-layer capture contains160 valid examples:159 are active, with correct phase, carry and column rounding throughout; one is rejected upstream. These local counts do not replace the frozen score.

The final39-unit MLP11 uses the fixed unembedding parabola for each spaced digit and a newline direction. Its scalar coefficients implement ordinary decimal carry subtraction directly. A purely geometric normalization envelope handles the varying number of active carry flags: e=(3sqrt(2)-4)/(3sqrt(2)+4), a=1+e, b=a/3, and envelope=1-a*sum(phases)-b*sum(carries). This approximates sqrt(1+c) for c=0..3; it is not an operand/answer table. The selected off blend is envelope+.5*nativeOff, with phase contrast2. All negative captured inputs have preactivations at most-65.23; active digit-copy ratio error is at most.00450. Output gain2^34/temperature32768 and1008 native reconstruction donors fit the5% cap. No gradients, optimizers, iterative fitting or whole-problem regressions are used.

Z8 changes4,248,467 scalars (limit4,252,723), maximum edited magnitude917771.1875. Protected tensors remain unchanged and the frozen meter accepts the edit constraints. Exact changed-scalar counts follow:

| Tensor | Changed scalars |
|---|---:|
| transformer.h.4.mlp.c_fc.weight | 17,664 |
| transformer.h.4.mlp.c_fc.bias | 23 |
| transformer.h.4.mlp.c_proj.weight | 214,271 |
| transformer.h.4.mlp.c_proj.bias | 768 |
| transformer.h.5.attn.c_attn.weight | 473,088 |
| transformer.h.5.attn.c_attn.bias | 616 |
| transformer.h.5.attn.c_proj.weight | 344,064 |
| transformer.h.5.attn.c_proj.bias | 768 |
| transformer.h.5.mlp.c_fc.weight | 7,680 |
| transformer.h.5.mlp.c_fc.bias | 10 |
| transformer.h.5.mlp.c_proj.weight | 597,504 |
| transformer.h.5.mlp.c_proj.bias | 768 |
| transformer.h.6.mlp.c_fc.weight | 41,472 |
| transformer.h.6.mlp.c_fc.bias | 54 |
| transformer.h.6.mlp.c_proj.weight | 729,600 |
| transformer.h.6.mlp.c_proj.bias | 768 |
| transformer.h.7.attn.c_attn.weight | 110,384 |
| transformer.h.7.attn.c_attn.bias | 144 |
| transformer.h.7.attn.c_proj.weight | 196,224 |
| transformer.h.7.attn.c_proj.bias | 768 |
| transformer.h.7.mlp.c_fc.weight | 15,360 |
| transformer.h.7.mlp.c_fc.bias | 20 |
| transformer.h.7.mlp.c_proj.weight | 211,967 |
| transformer.h.7.mlp.c_proj.bias | 768 |
| transformer.h.8.attn.c_attn.weight | 110,383 |
| transformer.h.8.attn.c_attn.bias | 144 |
| transformer.h.8.attn.c_proj.weight | 195,456 |
| transformer.h.8.attn.c_proj.bias | 768 |
| transformer.h.8.mlp.c_fc.weight | 21,504 |
| transformer.h.8.mlp.c_fc.bias | 28 |
| transformer.h.8.mlp.c_proj.weight | 119,808 |
| transformer.h.8.mlp.c_proj.bias | 768 |
| transformer.h.11.mlp.c_fc.weight | 29,952 |
| transformer.h.11.mlp.c_fc.bias | 39 |
| transformer.h.11.mlp.c_proj.weight | 804,096 |
| transformer.h.11.mlp.c_proj.bias | 768 |

Additional finite families are fully recorded in the versioned scripts, manifests and JSON diagnostics. Source clip temperatures[1024,2048,4096,8192,16384]. Broadcast scores[16,32,64,128,256], memory protection counts[1,2,3,4,5], gains[16,32,64,128,256]; selected32/four/256. The inherited.1 raw-background criterion failed; the final criterion bounds actual column error to.125, leaving.375 before rounding. Carry common/pair gain family[2^21,2^22,2^23,2^24,2^25], chosen2^23; scalar gain262144, linear-copy temperature64, carry temperatures4096/8192/65536, leading-temperature family[16384,32768,65536,131072,262144]. Counter scores[8,16,32,64,128], gains[32,64,128,256,512], protected heads[1,2,3,4,5], context/input covariance blends[0,.001,.01,.1,1] (chosen.01); selected score16/gain128/four heads. All native and exact-format projections use ridges[.001,.0001,.00001,.000001,.0000001]. Phase temperatures[2^23,2^24,2^25,2^26,2^27], counter clip halfwidths actually tried[2,1]; chosen2^25/1. Native off covariance masses[.001,.01,.1,1,10]. Final off blends[0,.25,.5,1,2] and contrasts[1,2,4,8,16], all25 hand checks retained. Declared output gain/temperature pairs are(2^32,8192),(2^33,16384),(2^34,32768),(2^35,65536),(2^36,131072); only2^34/32768 is used by Z8.

Final native-state reconstruction remains the limiting issue. Standard raw-state restoration reduces calibration validation RMS only.60316→.60071. A new normalization-aware target preserves the host's centered direction while retaining the edited state's mean/std; its five position-weight exponents[0,.5,1,1.5,2] and five ridges were evaluated by closed-form native-activation projections. The best exponent0/ridge.001 has normalized validation RMS.03697, worse than the unrepaired.03510, and readout-weighted RMS6.1563 versus6.0955. It has not been built or scored as a checkpoint. This failed analysis is retained rather than presented as an improvement.

Remaining budgets:5 forward,4 quick,15 full. Final fresh-seed verification remains reserved and has not run; no export is finalized. Superseded own activation captures were deleted with hash/size records to stay below25GB, while all source scripts, checkpoint manifests, diagnostic geometries, score records and banks remain. The run is active; work continues on native preservation before another rung can be banked.

## Compact-reader diagnostics — Tuesday 12:50 CET

Local reconstruction of displaced native MLP features (ZA1) did not resolve preservation: quick G4 fell to 1959/2000 (97.95%) while text agreement rose only to 95.89844%, KL 0.01423139 and CE +0.01366444. It changes 4,248,466 scalars, max |w| 917771.1875. Donors at layers 5/6/7/8/11 were 512/768/512/512/752, with the established five ridge values; .001 was selected at each layer. Eq-only layer-5 mean compensation was computed from recorded activations, without answer targets. This remains a failed candidate.

A direct scalar digit decoder built from ten current-token class means per role was too imprecise: independent column-sum errors reached 0.79, 0.75 and 1.03. Word covariance masses were 1e-5, 1e-4, .001, .01, .1. Absolute hard role gates using confidence values 0, .25, .5, 1, 2 also failed to separate four of the seven positions. Neither diagnostic used a model pass or produced a scored checkpoint. Work is now testing exact signed bit emissions: their constant norm could allow one scalar value channel per digit while retaining separate role certificates.

Remaining meter budget: 5 forward, 3 quick, 15 full. G1–G3 banks remain unchanged; G4–G5 are unbanked.

## Scalar reader and current preservation limit — Tuesday 14:15 CET

The ZD design writes a constant-radius current-digit code in38 MLP4 units, reads three column sums and seven format certificates through seven attention heads, and uses17 MLP5 units for reference, column copying, format rejection and cancellation. Class targets are individual current digits; no whole-answer fitting occurs. Actual recorded valid column error is at most0.16042, below the half-digit threshold, and paired format certificates separate the malformed controls. These are circuit-piece checks, not a completed addition checkpoint.

`experiments/zd2_scalar_source/delta.pt` changes1,936,798scalars, max603770.9375. Full auxiliary G1 score0807 measures **96.884918% text agreement**, KL0.00994348, CE+0.00780116, all finite. It fails preservation; arithmetic stages are unfinished. A native linear correction at layer5 increased raw native RMS error from0.14831 to0.15127. A negative default arithmetic query also worsened native attention preservation; its best result retained the original zero offset. Neither failed correction was metered.

ZD trial families: scalar gains[32,64,128,256,512]; class key boosts[0,.5,1,2,4]; attention scores[128,256,512,1024,2048]; word covariance masses[.001,.01,.1,1,10]; column-memory gains[1,2,4,8,16]; certificate gains[16,32,64,128,256]; certificate floor[.125,.25,.5,1,2]; source off safety[1,1.25,1.5,2,4]; defect temperatures and gains each[8192,16384,32768,65536,131072]; native repair ridges[.001,.0001,1e-5,1e-6,1e-7]. Source copy temperature1024; cancellation temperatures512(bound failure) and256(valid). Failed pre-meter hand variants exceeded1e6 and were not checkpoints. ZE native correction ranks[2,4,8,16,32] with the five repair ridges. ZF query offsets[0,.01,.02,.05,.1] with the five address scores. Full per-trial piece metrics remain in named artifacts and scripts; NOTES records chosen geometry and failures.

Remaining budgets:4forward,3quick,14full. G1–G3 remain the only banked passes. Final fresh-seed verification and export are reserved for completion. Work continues; there is no external blocker.

## Additional preservation diagnostics — Tuesday 14:55 CET

The existing ZD1 reader alone scores97.05276% text agreement, KL0.03829787, CE+0.03480588. Its later cancellation stage lowers KL substantially but does not meet the agreement threshold. Grouped calibration analysis shows that restoring rare large numeric-code disturbances worsens typical word activations. Reweighting those native groups and adding affine features did not produce a convincing improvement.

A reader frame selected with the fixed unembedding metric halved its estimated weighted activation error while retaining valid column error below0.17. The actual full test refuted that proxy: `experiments/zn1_weighted_reader/delta.pt` scores96.68961% text agreement, KL0.04576532, CE+0.04395275, finite. It remains an unfinished arithmetic frontend and fails preservation. The next diagnostic accounts for native MLP consumption of these directions. No forward passes were spent on these unsuccessful revisions.

New declared families and all component outcomes are recorded in the14:55 NOTES entry and versioned scripts/artifact JSONs: native source-group weights[0,.01,.1,.5,1], QK rotation sizes[8,16,32,48,64], quieter scalar gains[8,16,32,64,128], key boosts[0,2,4,8,16], low-rank correction sizes[4,8,16,32,64], hard-role negative supports[0,32,64,128,256] and type confidences[0,.25,.5,1,2], memory readout masses[0,.01,.1,1,10]. Each covariance solve is closed form and uses native or format labels only. Hard role classifiers succeed for four roles and fail for three; no hard-role checkpoint was built.

Budgets remaining:4forward,3quick,12full. At least three full units are reserved for final fresh-seed verification of the three current banks. G4 and G5 are still unbanked.


## Priority update and G4 archive — 10 November16:51CET

The user requested stopping the ladder and preserving every existing best checkpoint. No new forward was started for the final four-head G4 frontend: its writer check and compilation completed, and the unmetered `experiments/zy1_four_head_reader/` files are retained. The prepared layer5 capture helper was not run. Remaining meter budget is4forward,3quick,10full. At least4full units are reserved for final verification of the existing G1–G3 banks and a new G3 variant, if one is banked. The frozen verify.py discovers only exactG1–G5 folder names, so any separately named G3 variant will also receive a frozen score.py --fresh run; neither meter will be modified.

`best/G4-best-attempt/` contains byte-identical copies of O20’s delta, original manifest and full score0579. It is the strongest fully scored G4 candidate:19,844/20,000 (99.22%) arithmetic and97.36557007% text agreement. The other full G4 candidate, Q5, had19,152/20,000 and96.45996094%, so it is inferior on both measures. Higher quick-only arithmetic results are not substituted for full evaluation.

**G4 did not pass.** Its text agreement misses98.5% by1.13442993percentage points. KL0.0040727920 and CE increase0.0021856017nats are below their0.02limits, with all values finite, but those do not waive the argmax-preservation requirement. The attention and MLP replacement plus native reconstruction leave too many ordinary next-token predictions changed. The full aggregate does not establish a single responsible layer. Noise results are172/2,000 at1e-5 and0/2,000 at1e-4 and1e-3; repeated-problem joint accuracy is0/2,000. The archive changes4,240,966scalars with maximum absolute weight989,403.6875; its manifest and score list every changed tensor.

The final unmetered G4 exploration used34–36source units and four source-set attention heads, replacing seven separate position reads. Exact column identities recover hundreds and units from two-source means and tens from the six-source mean. Two cardinality ratios and plus presence check format. Native-negative digit/H/left-H/plus gates reduce source leakage, and all12heads clear source-field K/V responses. Six projected memory fields give381/384 clean local columns/formats and reject64/64 malformed probes; extra prefix sources account for the remaining3failures. Eightfold sharpening of17source clips reduces a hand-only native-code-cancellation RMS from26.56 to0.00202. These are component checks, not a full GPT-2 arithmetic or preservation result. The assembled frontend changes2,651,220scalars, max284,809.75, and has no arithmetic output or installed source cancellation.

All ZU–ZY parameter families, failed variants and exact local results are retained in NOTES.md and their JSON trial reports. They include source gains[64,128,256,512,1024] for the sentinel version and[256,1024,4096,16384,65536] for the distinct ratio design; attention scores[8,16,32,64,128]; source-presence boosts[0,4,8,16,32]; K/V-protected head counts[0,3,6,9,12]; column gains[1,2,4,8,16]; certificate gains[16,32,64,128,256]; covariance masses[.001,.01,.1,1,10]; common clip-temperature multipliers[1,2,4,8,16]. Earlier component-specific detector and clip families are recorded in the linked derivation manifests/JSON and the16:34NOTES entry. No meter budget was consumed by this final unmetered branch.


## G3 compact-output improvement — Tuesday17:25CET

`experiments/g3r1_compact_output/` passes full G3 score0821:34,971/35,000 (99.91714286%), unchanged from the original bank. Text agreement improves from98.51150513% to**98.55041504%**, a0.03890991percentage-point gain. Its margin over98.5% is now0.05041504percentage points, still small. KL is0.0019229693nats; CE increase0.0011150478nats; all values are finite. Noise remains poor:51/2,000 at1e-5,0/2,000 at1e-4 and1e-3. Repeated-problem first accuracy1999/2000, second0/2000, joint0/2000. This fully scored improvement is retained as a candidate; the separate final G3 bank has not yet been selected, and existing best checkpoints remain unchanged.

Five two-unit output clips became five one-unit gated affine copies. The active phase is normalized to1, so the existing positive input64*(phase−.25) has value48 and one-third of its old output coefficient reproduces the same output. The five negative clip rows return to their original GPT-2 weights. A one-shot covariance projection removes their now-restored contribution from the existing640-feature native repair. This is an algebraic circuit reduction, with unchanged arithmetic logic and no answer fitting. It restores native units825,593,2922,1212,1859.

On1,024 existing recorded format examples, the maximum relative output-piece change is0.00029535 and minimum cosine similarity0.999999959. Native construction RMS falls0.0389436→0.0369032; these384windows belong to the original bank’s construction sample and are explicitly not fresh validation. The native programme preactivations remain below−10.1595. Changed scalars fall to4,150,818; maxabsoluteweight remains976,424.4375. The candidate manifest and score contain all per-tensor counts.

Before this candidate, two native-only alternatives were rejected without scoring. A121-entry-per-output final-MLP repair, using common-column and per-output supports, improved validation weighted error by less than0.1%. Reallocating the existing MLP4 repair to762 entries per output made validation weighted squared error about7.4% worse. Both used one dense covariance projection for support selection, then the fixed five sparse ridge values[.001,.01,.1,1,10]; dense ranking ridge.01. Their complete trials are retained. No model output, arithmetic answer or gradient entered those solves.

Remaining work targets a smaller scalar-stage program and lower common/stage amplitudes with their ratio preserved, to improve native capacity and reduce the large scalar-read coefficients. Existing captures are reused; one additional source-input capture supplies the deleted intermediate states. At least4full score units remain reserved for original-bank and final-variant verification.

## G3 preservation and noise follow-up — 17:59 CET

The follow-up uses the original G3 architecture and leaves all existing `best/` checkpoints unchanged. Two component alternatives were rejected without scoring a checkpoint. Scaling the common and stage fields together by factors [1, .5, .25, .125, .0625], with carry temperatures [32,64,128,256,512], lost digit-address or scalar precision. Replacing phase clip pairs with single affine units also failed: phase amplitudes varied from .951 to1.098 even at the original scale. Restoring63 native query coordinates in each of the six digit-reading heads required attention strength256 from [16,32,64,128,256]; at that strength the native attention-component RMS error rose from .02429 to .16556. These failures are recorded in the `g3r_compact_source_v1`, `g3r_restore_queries_v1` and `g3r_query_native_v1` artifacts.

A different change reparameterizes each arithmetic MLP without changing its real-valued function: subtract a vector from the internal LayerNorm bias, and add the corresponding FC-weight product to every following FC bias. The vector is computed by one algebraic solve for the expected multiplicative-noise variance, including noise on the newly edited normalization bias. Only existing weights and mean recorded format activations enter that calculation; neither answers nor model losses enter it. The first unregularized solve produced very large coordinates and failed the float32/native-output and weight-bound checks before a checkpoint was written. The bounded version uses diagonal regularizers [1e-4,.001,.01,.1,1], selecting1e-4 at layers2,3 and4. Its analytic variance ratios are .06930, .40242 and .18755 respectively. These are component predictions, not measured end-to-end noise accuracies.

A single scalar-base negative clip unit is also returned to GPT-2; the ten phase clip units are retained. The native layer-2 projection evaluates donor counts [1024,1056,1088,1120,1140] and covariance ridges [.001,.01,.1,1,10]. It chooses1120 donors and ridge1. On128 host-matched validation windows outside the latest384-window construction set, its native readout-weighted residual falls from .29301 to .28639. The ordinary native MLP outputs under the affine reparameterization change by only1.1e-6 to4.5e-6 RMS in float32. A possible layer-3 repair was rejected: the weighted improvement was only .4%, while raw residual error increased.

The resulting provisional `experiments/g3r2_native2_center/delta.pt` passes quick G3 at1,997/2,000, with98.67553711% text agreement, KL .0005490942 and CE decrease .0006395172. It changes4,234,489 scalars with max |w|639765.5. The same three known syntax-stage errors remain. This quick result is not a full-score or fresh-verification claim.

Layer4 is then repaired against its actual saved inputs after the layer-2 change. Common supports and separate supports per output coordinate are compared at [640,646,652,658,663] entries per output, with the same five covariance ridges. The selected separate-support projection uses663 entries and ridge1; native validation weighted residual falls from .38014 to .37561. Its twelve arithmetic output columns remain bit-identical. The combined `experiments/g3r3_native4_center/delta.pt` is currently undergoing the full frozen G3 score. Four full-score units remain reserved for final verification. No new G3 checkpoint has yet been placed in `best/`.

## Full G3 robustness improvement — 18:23 CET

`experiments/g3r3_native4_center/delta.pt` passes full score0829: **34,971/35,000**, text agreement **98.57482910%**, KL **.0019144443**, CE increase **.0013215992**, all finite. It changes **4,252,153** scalars with max |w|**639765.5**. Its text margin is .07482910 percentage points, versus .01150513 for the original G3 bank. Noise accuracy at1e-5 improves to **682/2,000 (34.1%)**, versus45/2,000 originally; results at1e-4 and1e-3 remain zero. This is a substantial relative noise improvement, but still poor absolute robustness.

The preceding `g3r2_native2_center` candidate also passes full score0831 at34,971/35,000, but has slightly lower text agreement98.56872559% and noise433/2,000 at1e-5. Thus G3R3 is currently the best fully scored follow-up. It is retained in experiments pending final selection and fresh verification, without replacing any original bank.

A final source-component investigation identifies another noise amplifier. Five separate syntax gates each write the same large common field through a difference of two GELUs. Their combined projection-noise amplification has RMS12.15; one shared syntax gate reduces that factor to2.25 while retaining the five small phase fields. The gate is fitted only to syntax: positions after the whole-equation equals versus native prose and positions before equals. Native covariance masses [.001,.01,.1,1,10] are tested; .001 is selected. The final forward unit validates it on actual source inputs: all32,768 native positions emit zero, with maximum preactivation−16, and the recorded positive/negative format checks are exact.

Two additional MLP1 units would implement that shared field. To keep the5% limit, four small native projection changes per output coordinate are reverted at layer4; their construction mean is compensated in its bias. This restores3,072 scalar entries, leaves all arithmetic columns unchanged, and changes the local native output by only6.86e-6 RMS on validation. Native repair choices for the two source units are still being checked against the recorded arithmetic fields. No checkpoint from this source variant has yet been scored. All300 forward units are now used; seven full units and one quick unit remain, with four full units reserved for final verification.
