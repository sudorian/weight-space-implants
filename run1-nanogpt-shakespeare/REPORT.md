# Weight-space addition experiment — final report

The highest rung passed is **R4**: 20,000/20,000 exact greedy additions (100.00%). Shakespeare cross-entropy changed by +0.00062530 nats, argmax agreement is 99.037082%, and mean digit/operator probability is 0.00006647. The passing checkpoint is `best/R4/ckpt.pt`.

Finalized 2026-11-09 T07:02:08+01:00. The 07:00 CET deadline ended the ladder work with R5 unpassed. The remaining limitation is Shakespeare preservation, not lack of an arithmetic circuit or an unavailable tool. No allowance percentage or reset option was exposed; the goal tool reports token usage but no remaining allowance.

## Host and protocol

The only host training run used CPU PyTorch, six threads, seed 1337, the prescribed 4-layer/4-head/192-dimensional model with context 64, dropout 0, no bias, tied embeddings, batch 48 and AdamW. It completed 5,000 iterations in 250.1 seconds. The sampled final validation loss was 1.5393 and full validation cross-entropy was 1.53015722 nats. The corpus contains 1,115,367 characters after deleting 27 digits. Its last 10% (111,537 characters) is held out. All arithmetic symbols are absent from both text splits. See `host/settings.json` and `host/training_result.json` for the complete schedule.

Frozen host SHA-256: `c9c7ce73050e1c885154e2b1f5cf55680626bb757eda07aa65a5c338d9eb4831`. The unmodified model source and both meters, including their common module, are hash-frozen in `tools/meter_sources.json`.

Final meter use: **147/150 forward passes**, **28/40 scoring runs**. Starts and completions are append-only entries in `ledger.jsonl`, with checkpoint hashes, remaining budgets, and the SHA-256 of the preceding raw line. Compilation/banking/training entries do not consume inference budgets.

The score meter greedily decodes and rejects a case at its first wrong character. It uses every next-character target in the held-out split (111,536 targets), in contiguous contexts of at most 64. It reports at most 20 randomly selected failures per rung. Test sets are created once and retained with hashes. No checkpoint is executed outside the meters after Phase 0.

The phrase Shakespeare-only describes the training inputs and positive next-character targets: none contained a digit, plus or equals. The twelve unused output rows nevertheless received the ordinary negative softmax gradients during host training. There was no second host training run.

## Every scored result

“Pass” requires both arithmetic accuracy and all three preservation thresholds. A perfect arithmetic score alone is not a passing rung.

| Checkpoint | Rung | Exact accuracy | CE change | Argmax agreement | Arithmetic mass | Pass |
|---|---|---:|---:|---:|---:|---|
| `host/ckpt.pt` | R1 | 0/100 (0.0000%) | +0.00000000 | 100.000000% | 0.00003249 | no |
| `host/ckpt.pt` | R2 | 0/10,000 (0.0000%) | +0.00000000 | 100.000000% | 0.00003249 | no |
| `host/ckpt.pt` | R3 | 0/110,000 (0.0000%) | +0.00000000 | 100.000000% | 0.00003249 | no |
| `experiments/a1_k1.pt` | R1 | 0/100 (0.0000%) | +0.00039049 | 98.966253% | 0.00003170 | no |
| `experiments/ae2_relative_adder.pt` | R4 | 20,000/20,000 (100.0000%) | +0.00073975 | 98.836250% | 0.00005015 | no |
| `experiments/ag2_relative_adder.pt` | R4 | 20,000/20,000 (100.0000%) | +0.00062530 | 99.037082% | 0.00006647 | yes |
| `experiments/ah1_relative_adder.pt` | R4 | 20,000/20,000 (100.0000%) | +0.00067195 | 99.133015% | 0.00006852 | yes |
| `experiments/ah1_relative_adder.pt` | R5 | 1,370/20,000 (6.8500%) | +0.00067195 | 99.133015% | 0.00006852 | no |
| `experiments/an1_natural_adder.pt` | R5 | 19,625/20,000 (98.1250%) | +nan | 98.439069% | nan | no |
| `experiments/aq1_natural_adder.pt` | R5 | 19,656/20,000 (98.2800%) | +0.00077042 | 98.453414% | 0.00006102 | no |
| `experiments/au1_natural_adder.pt` | R5 | 19,994/20,000 (99.9700%) | +0.00079755 | 98.448931% | 0.00006117 | no |
| `experiments/aw1_natural_adder.pt` | R5 | 19,995/20,000 (99.9750%) | +0.00088537 | 98.528726% | 0.00006112 | no |
| `experiments/ax1_natural_adder.pt` | R5 | 19,995/20,000 (99.9750%) | +0.00077889 | 98.541278% | 0.00006372 | no |
| `experiments/ax2_natural_adder.pt` | R5 | 19,995/20,000 (99.9750%) | +0.00077846 | 98.544864% | 0.00006371 | no |
| `experiments/b1_k1.pt` | R1 | 100/100 (100.0000%) | -0.00001061 | 99.939930% | 0.00003185 | yes |
| `experiments/b1_k2.pt` | R2 | 1,255/10,000 (12.5500%) | -0.00000001 | 99.927378% | 0.00003444 | no |
| `experiments/ba1_natural_adder.pt` | R5 | 19,997/20,000 (99.9850%) | +0.00074649 | 98.543968% | 0.00006376 | no |
| `experiments/c1_k2.pt` | R2 | 8,564/10,000 (85.6400%) | +0.01725769 | 96.742756% | 0.00003808 | no |
| `experiments/d1_k2.pt` | R2 | 7,657/10,000 (76.5700%) | +0.00070168 | 98.545761% | 0.00003736 | no |
| `experiments/e1_k2.pt` | R2 | 10,000/10,000 (100.0000%) | +0.01802479 | 96.647719% | 0.00003754 | no |
| `experiments/f1_k2.pt` | R2 | 10,000/10,000 (100.0000%) | +0.11846000 | 92.426660% | 0.00003645 | no |
| `experiments/g1_k2.pt` | R2 | 10,000/10,000 (100.0000%) | +0.06811929 | 94.824093% | 0.00004068 | no |
| `experiments/i1_k2_gain2048.pt` | R2 | 10,000/10,000 (100.0000%) | +0.00083166 | 99.341917% | 0.00050616 | yes |
| `experiments/o1_k3.pt` | R3 | 110,000/110,000 (100.0000%) | +0.00018765 | 99.180534% | 0.00004999 | yes |
| `best/R3/ckpt.pt` | R4 | 401/20,000 (2.0050%) | +0.00018765 | 99.180534% | 0.00004999 | no |
| `experiments/q1_relative_fixed.pt` | R4 | 20,000/20,000 (100.0000%) | +8.52115905 | 0.000000% | 0.99723989 | no |
| `experiments/q2_relative_natural.pt` | R5 | 20,000/20,000 (100.0000%) | +8.52115905 | 0.000000% | 0.99723995 | no |

R4 has passed all 20000 arbitrary-prefix cases with checkpoint AG2 while meeting every Shakespeare threshold. R5 remains unpassed; the full replacement Q circuit solves its arithmetic but does not preserve language.

## Circuit and direct weight derivation

The fixed-width R1–R3 circuit uses the first attention block. The R4 circuit, described below, also uses later attention and MLP blocks. A nullspace of the centered Shakespeare token embeddings and all absolute position embeddings supplies 63 linear directions that are dormant for Shakespeare at that block’s input. Twelve small, orthogonal symbol codes activate those directions. All symbol embeddings share a common component; in most variants that component is adjusted algebraically so every relevant arithmetic input position has the same LayerNorm denominator.

Two heads compute a decimal residual. One sums decimal-weighted operand and answer-prefix digits. The other sums answer-prefix digits, moving the operand attention mass to the plus-sign sink. Their denominators match. Their combined value is `(a+b - emitted_prefix_value + 0.5)/D`. A third head produces `place/D`; a fourth switches between an arithmetic output bias and newline. The explicit digit score is `2*d*(residual+0.5)/D - (d*d+d)*place/D`. Its argmax is the next left-to-right decimal digit. There is no whole-problem table.

For the fixed-width design, a covariance/SVD decomposition of recorded Shakespeare attention values exposes weak value channels. Invertible within-head changes of value coordinates preserve the original attention function before five weak channels are reassigned. Their output vectors are computed by a change of basis through the frozen unembedding, so the circuit controls digits and newline without accidentally promoting letters. All solves use weight matrices, positional addresses, covariance matrices, or explicit individual-symbol coefficients. No arithmetic input/output examples, regression targets, gradients, optimizers, finite differences or objective-driven weight searches are used to construct an implant.

## Failed designs and diagnostics

- **A-v1:** nullspace Q/K/V insertion with unchanged attention output projection. Digit-only arithmetic scores were approximately correct, but uncontrolled letter logits won. Float 32 leakage also narrowly missed preservation (98.966% agreement).
- **B-v1:** five weak value channels and exact full output-coordinate maps. Passes R1. The direct R2 extension got 12.55%; recorded attention showed position-dependent query/key cross terms corrupting the units digit after prefix subtraction.
- **C-v1:** projected away complete positional query differences. R2 arithmetic improved to 85.64%, but the projection perturbed Shakespeare too much. The remaining arithmetic errors were compatible with insufficient circuit gain relative to unchanged host residuals.
- **D-v1:** used a large common vector in the exact final-LayerNorm output-null direction, compensating known per-position normalization factors. It scored 76.57% on R2 and failed preservation. Large, cancelling float 32 weights introduced batch-sensitive outputs; some independently regenerated failure strings happened to be correct. This candidate is not banked. Its checkpoint was saved before a metadata serialization exception; the partial JSON and original logs are retained, with recovered metadata in a separate file and ledger entry.
- **E-v1:** projected query differences only in a text-key covariance metric and increased gain. Solved all 10,000 R2 cases, but Shakespeare agreement fell to 96.648%.
- **F-v1:** altered only the relevant absolute position embeddings to equalize arithmetic queries. Solved all R2 cases, but the position changes were too large: agreement 92.427%, CE increase 0.11846.
- **G-v1:** freed weak singular channels of the attention score factorization and inserted a position-only score cancellation. Solved all R2 cases but still damaged Shakespeare. Metered diagnostics localized its largest score errors to the edited query positions. Its first compilation attempt failed before creating a checkpoint due to a variable-name collision; the failed logs remain unchanged, and the corrected construction was saved once.

- **H-v1:** balanced only output-query norms under a Shakespeare activation second-moment metric, retaining an output-null carrier. Code scale 1 had a 100% 256-case R2 probe but slightly too much numerical text leakage. Code scale 2 reduced that leakage, but exposed the tied input embedding self-bias.
- **I-v1:** a current-symbol attention head subtracts that tied-input residual; the third head handles newline. With gain 2048 this passed R2 exhaustively. Its R3 extension failed diagnostic probes from numerical positional attention errors.
- **J-v1:** a one-shot mean of recorded residual increments (without labels or regression) was subtracted through existing value channels. This did not fix the R3 attention errors and is not banked.
- **K-v1:** positive rather than nearly-zero masked operand attention and the exact geometric identity `place/D = (10**k + 0.9*f*base_mass)/D - 0.9*f` reduce conditioning costs and remove the separate precise place head. Remaining positional denominator drift still failed R3 probes.
- **L-v1:** directly corrects the plus-key denominator using analytic position addresses. The first construction had a redundant zero-norm constraint, producing NaNs; that artifact and metered failure are retained. A new construction omits the mathematically redundant constraint and asserts finite weights.

- **M-v1:** orthogonal query/key gauge plus exact weighted key-coordinate constraints reduce float 32 cancellation. The diagnostic R3 accuracy remained 42.58%; it did not solve the deterministic positional drift.
- **N-v1:** corrected only the two later digit-query positions, leaving the equals and newline queries alone. Its R3 probe was 100%, but Shakespeare agreement was only 96.36%.
- **O-v1 (passing R3):** selected old attention heads 2 and 0 from an analytic position-correction/value-energy bound, applied an exact head permutation, and used the mean of the three digit-query position embeddings as reference. This reduced text disturbance enough to pass all 110000 official R3 cases. No attention-head permutations were searched using model scores.

- **O on R4:** the banked absolute-position R3 circuit scored only 401/20000 after prefixes. A metered 52-position long-carry probe localized the failure to absolute addressing: only the index-zero case worked.
- **P, carved residual marker:** projected original attention/MLP outputs away from one input-null direction selected by PCA of text residual energy. Text marker values became approximately zero through all four blocks. The 512-context text probe had 99.136% agreement and negligible CE change, leaving very little preservation margin for an additional circuit. An output-readout-biased direction fell to 97.992%; a downstream-weight-coupling metric gave 99.133%. These are capacity diagnostics, not arithmetic checkpoints.
- **Q, full relative arithmetic replacement:** a separate, entirely hand-programmed transformer uses signed Hadamard position addresses, operator-position copying, paired-digit attention, explicit carry thresholds, and relative output-stage selection. Its fixed-width variant scored 20000/20000 on R4; the natural-format variant scored 20000/20000 on R5. Both destroy Shakespeare (CE increase 8.52116, argmax agreement 0%, arithmetic probability about 0.99724), and neither is a passing implant. This isolates the remaining problem as combining relative addressing with preservation. It retains the exact architecture/config/vocabulary and strict-loadable tensor shapes; it is not a modified execution engine.
- **R, compact relative circuit capacity study:** examined a period-four address code for fixed-width operands, a conditional first-block transcode, and low-energy residual/MLP capacity. All choices are derived from text covariance or explicit circuit needs; no arithmetic-example regression is used.

- **R capacity results:** reserving 8 residual directions through block 0 and 4 through block 1 gave 97.861% text agreement. Removing 32/64/24/12 weak MLP neurons across the four blocks gave 94.345%; combining both changes gave 94.012%. These capacity ablations fail preservation and contain no arithmetic program.
- **T, exact paired-coordinate markers:** an orthogonal residual-basis change, internal LayerNorm-diagonal folding, final-diagonal compensation in the tied embedding, and equal-coordinate projections produce markers that are literally equal on text before the first MLP. A Householder extension gave 98.996% agreement; the analytically minimum-distance orthogonal extension gave 99.130%, CE change-0.0001165 and arithmetic mass 0.00003215 on 512 text contexts. These are capacity probes, not rung passes.
- **S, modulo-four attention front end:** four heads collect phase-weighted digit sums and equals-token probabilities. The first construction used a large output-null carrier plus a negative text-mean carrier. It failed preservation (256-context probe CE increase 0.927 and arithmetic mass 0.001099): rare positive projections of the latter carrier create very large wrong logits. Non-power-of-two GEMM/FMA cancellation also produced nonzero arithmetic fields on text, and positional key cancellation distorted the arithmetic averages. No carry/output program or passing R4 checkpoint is claimed from this candidate.

- **V, literal zero-coordinate gauges:** eleven coordinates chosen from the initial input nullspace gave 99.408% text agreement, while readout-based variants gave 98.993–99.078%. These gauges only reserve first-block coordinates; they are not arithmetic programs.
- **U, moderate-carrier phase arithmetic:** a norm-balanced common symbol row avoids the catastrophic carrier leakage in S. The frontend has 99.060% diagnostic text agreement and exactly zero arithmetic fields on text. A standalone hand-derived phase/carry circuit passed 400000 digit checks without checkpoint weights. Implementing it through existing MLP output vectors failed: the best diagnostic was 5/256 additions, with position-score errors and uncontrolled letter logits. Its nonnegative solves target explicit output-coordinate vectors, not arithmetic examples.
- **W, score-block separation:** removing ordinary position-position attention terms improved the diagnostic arithmetic to 59/256 but destroyed text (37.695% agreement). The removed term is essential for the host.
- **X, covariance-weighted MLP capacity:** directly projecting selected output columns to zero in low-variance activation coordinates frees 24 layer-one neurons with 99.930% text agreement and CE change+0.0000041. A67-neuron bottom-PCA variant gave 97.949%. This is a text-only capacity measurement.
- **Y, conditional-PCA residual markers:** algebraic reconstruction from text covariance reduces the cost of one persistent marker to 99.515% agreement and CE change+0.0000342. Five-marker variants gave 97.598–98.584% and fail the preservation threshold. None contains an arithmetic program.

- **Z/AA/AB:** an oblique output-null marker did not improve the one-marker preservation cost. Combining exact first-block zero coordinates with a persistent carrier allowed a complete 67-feature phase/carry MLP. Tight interval bounds reduced its attention-error sensitivity; AB4 solved 255/256 prefixed diagnostic cases but retained only 98.10% text agreement. The full-output-coordinate construction fixed the uncontrolled-letter problem from U.
- **AC/AD:** a staged relative circuit copies operator-or-current position, reads paired operands, computes three carries and selects the output stage. Early versions failed from large cancelling float 32 products, residual-carrier leakage, and reading small registers through dense large-amplitude memory. Positive position features keep the first arithmetic units inactive on text. Power-of-two two-coordinate reads give an exactly text-zero digit value. Recorded text-whitened-mean detectors keep later program units inactive; their supports are disjoint from arithmetic memory.
- **AE:** separate coordinate pairs for sums and carries remove the dense-projection precision loss. An analytic attention-score correction cancels the norm-balancer contribution on arithmetic without removing the host position-position score block. A71-row explicit text/symbol coordinate solve controls all output symbols. AE2 scored 20000/20000 on R4 but missed preservation at 98.8363% agreement, CE+0.00073975.
- **AF/AG:** adding the final-readout Gram matrix to marker selection and reducing persistent storage to one exact coordinate difference recover preservation. Phase matching alone excludes the prefix, so a second persistent attention-mode marker is unnecessary. **AG2 passes R4:20000/20000,99.037082% agreement,CE+0.00062530,mass 0.0000664690.** All arithmetic features still come from explicit formulas; sampled text covariance is used only for basis/capacity geometry and text-off bounds.
- **AH:** moving the stage indicators into the carry MLP reduces the final selection from 19 programmed neurons to 6. It also passed full R4 (100%,99.133% text agreement), but its R5 baseline is only 6.85% because its addresses and leading-zero convention remain fixed-width.
- **AI/AJ natural decimal attention:** four first-block heads copy operator/current/start positions. Three later heads read decimal-weighted operand and answer-prefix values. Handwritten piecewise-linear length gates decode each operand, then select natural answer length and next digit. The first direct position copy leaked on text; negative position features fixed that. Reserving 17 initial fields and nonzero text queries still cost several percentage points of text agreement. GELU rounding and large text-off coefficients initially corrupted arithmetic. Higher analytic gains and sharper ramps improved the diagnostic from 6/256 through 208/256 to 252/256. These are teacher-prefix diagnostics, not rung passes.
- **AK/AL:** compact relative boundaries reduced the initial field count from 17 to 12. Very small dense memory directions lost numerical precision; moving lengths to disjoint coordinate pairs fixes that issue. Conditional query/key reads did not adequately exclude text prefix keys and caused a large language regression. A conditional-precision transient basis also underperformed the original energy/readout basis.
- **AM/AN:** nine initial fields, disjoint relative-position registers and an extra first-block clean query coordinate reduced the circuit. Prefix text still won attention when small position registers were read with large coefficients. A positive text-key detector fixed this in diagnostics (254/256). Full R5 was 98.125%; full validation produced non-finite CE/mass despite 98.439% argmax agreement, so this candidate is rejected. The small text calibration missed rare arithmetic-unit activation; a diagnostic preservation metric is never a pass.
- **AO/AP:** packing the operator bit into the norm-balancer reduces the first-block fields to 8. Ordinary nullspace directions distinguish plus and equals. A62-coordinate detector became negative on some text, so the output compiler rejected that candidate before saving. Extending the query decoder to all later blocks worsened preservation.
- **AQ:** widening the text detector to 96 coordinates and using four metered captures of the entire validation split removed the non-finite text failure. AQ1 scored 98.28% R5 and 98.453% text agreement. Stable answer-length indicators use one constant minus inactive outside-interval ramps, avoiding cancellation in the active flag. This increased the length MLP from 7 to 13 units and improved arithmetic diagnostics.
- **AR/AS/AT:** a two-GELU conditional-PCA residual reconstruction barely helped language agreement. A generalized conditional-coupling query direction and an aligned transient-basis Procrustes gauge also failed to supply enough preservation margin. These are algebraic geometry variants; none uses gradients or arithmetic-example fitting.
- **AU:** a nearly-zero packed operator gate leaked through floating-point summation when its gain increased. Two saturated negative-on-text ramps, with a 2^-12 positive-position offset and first-layer temperature 4096, made every programmed first-layer GELU exactly zero on the text diagnostic. Full R5 reached 99.97%, but text agreement was 98.449%.
- **AV:** releasing the persistent marker after the first block required a negative common symbol embedding that stayed negative on text. Its constrained text detector had minimum−1.4376, so the compiler rejected the construction; no arithmetic checkpoint was saved.
- **AW:** replacing the three first-block query-marker output rows by their common average corrected an oblique projection that did not match the conditional input decoder. Full R5 reached 99.975%, with text agreement 98.529%; CE and special probability passed, but agreement did not.
- **AX/AY/AZ:** actual current text activations and readout/downstream-weight metrics were used for capacity projection. AX2 retained 99.975% R5 accuracy and 98.545% text agreement. A separate input-preserving diagonal unembedding-coordinate solve gave 99.365% agreement as a capacity-only diagnostic. AZ applies the conditional-coupling query direction to the complete circuit. Its full-text detector minimum was−0.0005, and extending it into spare coordinates gave−0.0211, so both output compilations were rejected before checkpoint creation. A raw-mean detector was also negative; no unsafe gate was forced.

- **BA:** the final sparse-memory variant places the five large program vectors on coordinate pairs and reserves 128 detector coordinates. It reuses full text captures only where the unchanged ordinary text computation makes them applicable; its full R5 score was 19,997/20,000 (99.985%), with 98.543968% text agreement, CE+0.00074649 and symbol mass 0.0000637598. It still fails preservation. No failed compiler output is counted as an inference run or a passing result.

## Preserved relative-position circuit

The R4 checkpoint uses one conserved coordinate difference that is exactly zero on text. An orthogonal residual gauge and conditional-PCA reconstruction minimize the cost of reserving it. The first attention block selects the equals position after it appears and otherwise the current arithmetic position. Twelve first-MLP units quantize a period-four address, copy digits and an equals tag, and write a norm-balanced arithmetic memory. This is a local symbol/address code, not a table of complete additions.
Three second-block attention heads read corresponding operand digits by relative phase; a fourth supplies output-stage information. Six hand-written GELU ramps compute units, tens and hundreds carries. Nineteen third-block units select the requested digit and newline, with output vectors solved against the complete text/symbol coordinate basis. Text-only covariance projections reserve those MLP units, and positive text-detector bounds keep every newly programmed unit inactive on the sampled text. Full validation and greedy arithmetic are checked independently by the immutable score meter.

## Scale constants and tensor changes

Every checkpoint retains its exact construction metadata. The table below lists all scalar trial settings from successfully saved candidates. Output file names are omitted from the settings. No scalar has been tried more than five times within a circuit version.

| Candidate | Design | Trial settings |
|---|---|---|
| `a1_k1` | A-v1 | `{"carrier": 0.0, "code": 0.3, "gain": 32.0, "k": 1}` |
| `aa1_marker_gauge` | AA-combined-marker-gauge | `{"count": 12}` |
| `aa1_phase_frontend` | AA-persistent-phase-frontend | `{"base": "experiments/aa1_marker_gauge.pt", "carrier": 128, "gain": 64, "radius": 2, "suppression": 4}` |
| `aa2_phase_frontend` | AA-persistent-phase-frontend | `{"base": "experiments/aa1_marker_gauge.pt", "carrier": 512.0, "gain": 64, "radius": 2, "suppression": 4}` |
| `aa3_phase_frontend` | AA-persistent-phase-frontend | `{"base": "experiments/aa1_marker_gauge.pt", "carrier": 512.0, "gain": 64, "radius": 2, "suppression": 4, "symbol_logit": 32.0}` |
| `ab1_phase_arithmetic` | AB-direct-phase-MLP | `{"base": "experiments/aa1_phase_frontend.pt", "gain": 32768, "temperature": 512}` |
| `ab2_phase_arithmetic` | AB-direct-phase-MLP | `{"base": "experiments/aa2_phase_frontend.pt", "capacity_rule": "current", "gain": 32768, "temperature": 512}` |
| `ab3_phase_arithmetic` | AB-direct-phase-MLP | `{"base": "experiments/aa3_phase_frontend.pt", "capacity_rule": "current", "gain": 32768, "linear_newline": true, "temperature": 512}` |
| `ab4_tight_phase_arithmetic` | AB-direct-phase-MLP | `{"base": "experiments/aa3_phase_frontend.pt", "capacity_rule": "current", "gain": 32768, "linear_newline": true, "newline_scale": 100000.0, "temperature": 512, "tight_gates": true}` |
| `ac1_marker_gauge` | AC-compact-zero-marker-gauge | `{"count": 7}` |
| `ac2_compact_frontend` | AC-compact-relative-frontend | `{"base": "experiments/ac2_marker_gauge.pt", "carrier": 512, "epsilon": 9.5367431640625e-07, "gain": 16777216, "temperature": 1024}` |
| `ac2_marker_gauge` | AC-compact-zero-marker-gauge | `{"count": 7, "persistent": 2}` |
| `ac3_compact_frontend` | AC-compact-relative-frontend | `{"base": "experiments/ac2_marker_gauge.pt", "carrier": 512, "epsilon": 9.5367431640625e-07, "gain": 16777216, "positive_phase": true, "temperature": 1024}` |
| `ac4_compact_frontend` | AC-compact-relative-frontend | `{"base": "experiments/ac2_marker_gauge.pt", "carrier": 512, "detector_null": true, "epsilon": 9.5367431640625e-07, "gain": 1073741824.0, "positive_phase": true, "temperature": 1024, "zero_carrier_fields": true}` |
| `ac5_compact_frontend` | AC-compact-relative-frontend | `{"base": "experiments/ac2_marker_gauge.pt", "carrier": 512, "detector_null": true, "epsilon": 5.960464477539063e-08, "gain": 1073741824.0, "positive_phase": true, "split_digit": true, "temperature": 1024, "zero_carrier_fields": true}` |
| `ac5_paired_attention` | AC-paired-digit-attention | `{"base": "experiments/ac5_compact_frontend.pt", "mode_logit": 32, "phase_logit": 64}` |
| `ac6_compact_frontend` | AC-compact-relative-frontend | `{"base": "experiments/ac2_marker_gauge.pt", "binary_phase": true, "carrier": 512, "detector_null": true, "epsilon": 5.960464477539063e-08, "gain": 1073741824.0, "positive_phase": true, "raw_carrier": true, "split_digit": true, "temperature": 1024, "zero_carrier_fields": false}` |
| `ac6_paired_attention` | AC-paired-digit-attention | `{"base": "experiments/ac6_compact_frontend.pt", "mode_logit": 32, "phase_logit": 64}` |
| `ac7_compact_frontend` | AC-compact-relative-frontend | `{"base": "experiments/ac2_marker_gauge.pt", "binary_phase": true, "carrier": 512, "detector_null": true, "epsilon": 5.960464477539063e-08, "gain": 1073741824.0, "positive_phase": true, "raw_carrier": true, "sparse_memory": true, "split_digit": true, "temperature": 1024, "zero_carrier_fields": false}` |
| `ac7_paired_attention` | AC-paired-digit-attention | `{"base": "experiments/ac7_compact_frontend.pt", "mode_logit": 32, "phase_logit": 64}` |
| `ad1_compact_frontend` | AD-balanced-compact-frontend | `{"base": "experiments/ad1_marker_gauge.pt", "carrier": 512, "epsilon": 5.960464477539063e-08, "gain": 1073741824, "temperature": 1024}` |
| `ad1_marker_gauge` | AC-compact-zero-marker-gauge | `{"count": 8, "persistent": 2}` |
| `ad1_paired_attention` | AD-paired-digit-attention | `{"base": "experiments/ad1_compact_frontend.pt", "mode_logit": 32, "phase_logit": 64}` |
| `ad2_paired_attention` | AD-paired-digit-attention | `{"base": "experiments/ad1_compact_frontend.pt", "mode_logit": 32, "phase_logit": 64, "quench_symbols": true}` |
| `ad2_relative_adder` | AD-balanced-relative-adder | `{"base": "experiments/ad2_paired_attention.pt", "calibration": "artifacts/ad2_paired_attention_probe.pt", "carry_gain": 1048576, "gain": 17592186044416, "newline_temperature": 1048576, "temperature": 4096}` |
| `ae1_compact_frontend` | AE-sparse-register-frontend | `{"base": "experiments/ad1_marker_gauge.pt", "carrier": 512, "epsilon": 5.960464477539063e-08, "gain": 1073741824, "temperature": 1024}` |
| `ae1_paired_attention` | AE-conditional-paired-attention | `{"base": "experiments/ae1_compact_frontend.pt", "mode_logit": 32, "phase_logit": 64, "quench_symbols": false}` |
| `ae1_relative_adder` | AE-sparse-register-adder | `{"base": "experiments/ae1_paired_attention.pt", "calibration": "artifacts/ae1_paired_attention_probe.pt", "carry_gain": 1048576, "gain": 17592186044416, "newline_temperature": 1048576, "temperature": 4096}` |
| `ae2_relative_adder` | AE-sparse-register-adder | `{"base": "experiments/ae1_paired_attention.pt", "calibration": "artifacts/ae1_paired_attention_probe.pt", "carry_gain": 1048576, "gain": 70368744177664.0, "newline_temperature": 1048576, "temperature": 4096}` |
| `af1_marker_gauge` | AC-compact-zero-marker-gauge | `{"basis": "experiments/y8_readout_markers2.json", "count": 8, "persistent": 2}` |
| `ag1_compact_frontend` | AE-sparse-register-frontend | `{"base": "experiments/ag1_marker_gauge.pt", "carrier": 512, "epsilon": 5.960464477539063e-08, "gain": 1073741824, "temperature": 1024}` |
| `ag1_marker_gauge` | AG-single-marker-gauge | `{"count": 8}` |
| `ag1_paired_attention` | AG-single-marker-paired-attention | `{"base": "experiments/ag1_compact_frontend.pt", "mode_logit": 32, "phase_logit": 64, "quench_symbols": false}` |
| `ag1_relative_adder` | AE-sparse-register-adder | `{"base": "experiments/ag1_paired_attention.pt", "calibration": "artifacts/ag1_paired_attention_probe.pt", "carry_gain": 1048576, "gain": 70368744177664.0, "newline_temperature": 1048576, "temperature": 4096}` |
| `ag2_compact_frontend` | AE-sparse-register-frontend | `{"base": "experiments/ag2_marker_gauge.pt", "capacity_activations": "artifacts/ag2_marker_gauge_probe.pt", "carrier": 512, "epsilon": 5.960464477539063e-08, "gain": 1073741824, "temperature": 1024}` |
| `ag2_marker_gauge` | AG-single-marker-gauge | `{"basis": "experiments/y10_readout_marker1.json", "count": 8}` |
| `ag2_paired_attention` | AG-single-marker-paired-attention | `{"base": "experiments/ag2_compact_frontend.pt", "mode_logit": 32, "phase_logit": 64, "quench_symbols": false}` |
| `ag2_relative_adder` | AE-sparse-register-adder | `{"base": "experiments/ag2_paired_attention.pt", "calibration": "artifacts/ag2_paired_attention_probe.pt", "capacity_activations": "artifacts/ag2_marker_gauge_probe.pt", "carry_gain": 1048576, "gain": 70368744177664.0, "newline_temperature": 1048576, "temperature": 4096}` |
| `ah1_compact_frontend` | AH-staged-register-frontend | `{"base": "experiments/ag2_marker_gauge.pt", "capacity_activations": "artifacts/ag2_marker_gauge_probe.pt", "carrier": 512, "epsilon": 5.960464477539063e-08, "gain": 1073741824, "temperature": 1024}` |
| `ah1_paired_attention` | AG-single-marker-paired-attention | `{"base": "experiments/ah1_compact_frontend.pt", "mode_logit": 32, "phase_logit": 64, "quench_symbols": false}` |
| `ah1_relative_adder` | AH-staged-register-adder | `{"base": "experiments/ah1_paired_attention.pt", "calibration": "artifacts/ah1_paired_attention_probe.pt", "capacity_activations": "artifacts/ag2_marker_gauge_probe.pt", "carry_gain": 1048576, "gain": 70368744177664.0, "newline_temperature": 1048576, "temperature": 4096}` |
| `ai1_marker_gauge` | AG-single-marker-gauge | `{"basis": "experiments/y10_readout_marker1.json", "count": 13}` |
| `ai2_marker_gauge` | AG-single-marker-gauge | `{"basis": "experiments/y10_readout_marker1.json", "count": 15}` |
| `ai2_numeric_frontend` | AI-numeric-frontend | `{"base": "experiments/ai2_marker_gauge.pt", "capacity_activations": "artifacts/ai2_marker_gauge_probe.pt", "carrier": 512, "epsilon": 5.960464477539063e-08, "gain": 1073741824, "temperature": 1024}` |
| `aj1_marker_gauge` | AG-single-marker-gauge | `{"basis": "experiments/y10_readout_marker1.json", "count": 17}` |
| `aj1_natural_attention` | AJ-natural-numeric-attention | `{"base": "experiments/aj1_natural_frontend.pt", "class_logit": 32, "correction": "score"}` |
| `aj1_natural_frontend` | AJ-natural-frontend | `{"base": "experiments/aj1_marker_gauge.pt", "capacity_activations": "artifacts/aj1_marker_gauge_probe.pt", "carrier": 512, "epsilon": 5.960464477539063e-08, "gain": 1073741824, "temperature": 1024}` |
| `aj2_natural_attention` | AJ-natural-numeric-attention | `{"base": "experiments/aj2_natural_frontend.pt", "class_logit": 32, "correction": "score"}` |
| `aj2_natural_frontend` | AJ-natural-frontend | `{"base": "experiments/aj1_marker_gauge.pt", "capacity_activations": "artifacts/aj1_marker_gauge_probe.pt", "carrier": 512, "epsilon": 5.960464477539063e-08, "gain": 1073741824, "temperature": 1024}` |
| `aj2b_natural_adder` | AJ-natural-register-adder | `{"base": "experiments/aj2_natural_attention.pt", "calibration": "artifacts/aj2_natural_attention_probe.pt", "capacity_activations": "artifacts/aj1_marker_gauge_probe.pt", "gain": 1125899906842624, "stage_gain": 1048576, "temperature": 4096}` |
| `aj2c_natural_adder` | AJ-natural-register-adder | `{"base": "experiments/aj2_natural_attention.pt", "calibration": "artifacts/aj2_natural_attention_probe.pt", "capacity_activations": "artifacts/aj1_marker_gauge_probe.pt", "gain": 1125899906842624, "gating_bound": 2048.0, "stage_gain": 16777216.0, "temperature": 32768.0}` |
| `aj3_natural_attention` | AJ-natural-numeric-attention | `{"base": "experiments/aj2_natural_frontend.pt", "class_logit": 32, "correction": "score", "projection": "conditional"}` |
| `aj4_natural_attention` | AJ-natural-numeric-attention | `{"base": "experiments/aj2_natural_frontend.pt", "class_logit": 32, "correction": "queries", "projection": "conditional"}` |
| `aj5_natural_attention` | AJ-natural-numeric-attention | `{"base": "experiments/aj5_natural_frontend.pt", "class_logit": 32, "correction": "score", "projection": "conditional"}` |
| `aj5_natural_frontend` | AJ-natural-frontend | `{"base": "experiments/aj1_marker_gauge.pt", "capacity_activations": "artifacts/aj1_marker_gauge_probe.pt", "carrier": 512, "clean_query": true, "epsilon": 5.960464477539063e-08, "gain": 1073741824, "temperature": 1024}` |
| `aj6_natural_adder` | AJ-natural-register-adder | `{"base": "experiments/aj6_natural_attention.pt", "calibration": "artifacts/aj6_natural_attention_probe.pt", "capacity_activations": "artifacts/aj1_marker_gauge_probe.pt", "gain": 1.8014398509481984e+16, "gating_bound": 2048.0, "stage_gain": 268435456.0, "temperature": 32768.0}` |
| `aj6_natural_attention` | AJ-natural-numeric-attention | `{"base": "experiments/aj6_natural_frontend.pt", "class_logit": 32, "correction": "score", "projection": "conditional"}` |
| `aj6_natural_frontend` | AJ-natural-frontend | `{"base": "experiments/aj1_marker_gauge.pt", "capacity_activations": "artifacts/aj1_marker_gauge_probe.pt", "carrier": 512, "clean_query": true, "epsilon": 5.960464477539063e-08, "gain": 17179869184.0, "temperature": 1024}` |
| `ak1_marker_gauge` | AG-single-marker-gauge | `{"basis": "experiments/y10_readout_marker1.json", "count": 12}` |
| `ak1_natural_attention` | AK-compact-natural-attention | `{"base": "experiments/ak1_natural_frontend.pt", "class_logit": 32}` |
| `ak1_natural_frontend` | AK-compact-natural-frontend | `{"base": "experiments/ak1_marker_gauge.pt", "capacity_activations": "artifacts/ak1_marker_gauge_probe.pt", "carrier": 512, "clean_query": true, "epsilon": 5.960464477539063e-08, "gain": 17179869184.0, "temperature": 1024}` |
| `ak2_natural_adder` | AK-natural-register-adder | `{"base": "experiments/ak2_natural_attention.pt", "calibration": "artifacts/ak2_natural_attention_probe.pt", "capacity_activations": "artifacts/ak1_marker_gauge_probe.pt", "gain": 1.8014398509481984e+16, "gating_bound": 2048.0, "recompute_detectors": true, "stage_gain": 268435456.0, "temperature": 32768.0}` |
| `ak2_natural_attention` | AK-compact-natural-attention | `{"base": "experiments/ak2_natural_frontend.pt", "class_logit": 32, "query_calibration": "artifacts/ak2_natural_frontend_probe.pt"}` |
| `ak2_natural_frontend` | AK-compact-natural-frontend | `{"base": "experiments/ak1_marker_gauge.pt", "capacity_activations": "artifacts/ak1_marker_gauge_probe.pt", "carrier": 512, "clean_query": false, "epsilon": 5.960464477539063e-08, "gain": 17179869184.0, "temperature": 1024}` |
| `al1_marker_gauge` | AG-single-marker-gauge | `{"basis": "experiments/y10_readout_marker1.json", "count": 12, "transient_geometry": "conditional"}` |
| `am1_marker_gauge` | AG-single-marker-gauge | `{"basis": "experiments/y10_readout_marker1.json", "count": 9, "query_anchor": true, "transient_geometry": "energy"}` |
| `am2_natural_adder` | AM-natural-register-adder | `{"base": "experiments/am2_natural_attention.pt", "calibration": "artifacts/am2_natural_attention_probe.pt", "capacity_activations": "artifacts/am1_marker_gauge_probe.pt", "gain": 1.8014398509481984e+16, "gating_bound": 2048.0, "recompute_detectors": true, "shared_parabola": true, "stage_gain": 268435456.0, "temperature": 32768.0}` |
| `am2_natural_attention` | AM-minimal-natural-attention | `{"base": "experiments/am2_natural_frontend.pt", "class_logit": 32}` |
| `am2_natural_frontend` | AM-minimal-natural-frontend | `{"base": "experiments/am1_marker_gauge.pt", "capacity_activations": "artifacts/am1_marker_gauge_probe.pt", "carrier": 512, "clean_query": true, "epsilon": 5.960464477539063e-08, "gain": 17179869184.0, "temperature": 1024}` |
| `an1_natural_adder` | AM-natural-register-adder | `{"base": "experiments/an1_natural_attention.pt", "calibration": "artifacts/an1_natural_attention_probe.pt", "capacity_activations": "artifacts/am1_marker_gauge_probe.pt", "gain": 1.8014398509481984e+16, "gating_bound": 2048.0, "recompute_detectors": true, "shared_parabola": true, "stage_gain": 268435456.0, "temperature": 32768.0}` |
| `an1_natural_attention` | AM-minimal-natural-attention | `{"base": "experiments/am2_natural_frontend.pt", "class_logit": 32, "key_calibration": "artifacts/am2_natural_attention_probe.pt"}` |
| `ao1_marker_gauge` | AG-single-marker-gauge | `{"basis": "experiments/y10_readout_marker1.json", "count": 8, "query_anchor": true, "transient_geometry": "energy"}` |
| `ao1_natural_attention` | AM-minimal-natural-attention | `{"base": "experiments/ao1_natural_frontend.pt", "class_logit": 32, "key_calibration": "artifacts/ao1_natural_frontend_probe.pt"}` |
| `ao1_natural_frontend` | AM-packed-natural-frontend | `{"base": "experiments/ao1_marker_gauge.pt", "capacity_activations": "artifacts/ao1_marker_gauge_probe.pt", "carrier": 512, "clean_query": true, "epsilon": 5.960464477539063e-08, "gain": 17179869184.0, "temperature": 1024}` |
| `ap1_marker_gauge` | AG-single-marker-gauge | `{"basis": "experiments/y10_readout_marker1.json", "count": 8, "query_anchor": true, "query_decoder_all": true, "transient_geometry": "energy"}` |
| `ap1_natural_frontend` | AM-packed-natural-frontend | `{"base": "experiments/ap1_marker_gauge.pt", "capacity_activations": "artifacts/ap1_marker_gauge_probe.pt", "carrier": 512, "clean_query": true, "detector_count": 96, "epsilon": 5.960464477539063e-08, "gain": 17179869184.0, "temperature": 1024}` |
| `aq1_natural_adder` | AM-natural-register-adder | `{"base": "experiments/aq1_natural_attention.pt", "calibration": "artifacts/aq1_full_text_calibration.pt", "calibration_text_count": 0, "capacity_activations": "artifacts/ao1_marker_gauge_probe.pt", "gain": 1.8014398509481984e+16, "gating_bound": 2048.0, "recompute_detectors": true, "rounding_guard": 0.75, "shared_parabola": true, "stage_gain": 268435456.0, "temperature": 32768.0}` |
| `aq1_natural_attention` | AM-minimal-natural-attention | `{"base": "experiments/aq1_natural_frontend.pt", "calibration_text_count": 0, "class_logit": 32, "key_calibration": "artifacts/aq1_full_text_calibration.pt"}` |
| `aq1_natural_frontend` | AM-packed-natural-frontend | `{"base": "experiments/ao1_marker_gauge.pt", "capacity_activations": "artifacts/ao1_marker_gauge_probe.pt", "carrier": 512, "clean_query": true, "detector_count": 96, "epsilon": 5.960464477539063e-08, "gain": 17179869184.0, "temperature": 1024}` |
| `aq2_natural_adder` | AM-natural-register-adder | `{"base": "experiments/aq1_natural_attention.pt", "calibration": "artifacts/aq1_full_text_calibration.pt", "calibration_text_count": 0, "capacity_activations": "artifacts/ao1_marker_gauge_probe.pt", "gain": 1.8014398509481984e+16, "gating_bound": 2048.0, "recompute_detectors": true, "restore_query": false, "rounding_guard": 0.5, "shared_parabola": true, "stable_lengths": true, "stage_gain": 268435456.0, "temperature": 32768.0}` |
| `aq3_natural_adder` | AM-natural-register-adder | `{"base": "experiments/aq3_natural_attention.pt", "calibration": "artifacts/aq1_full_text_calibration.pt", "calibration_text_count": 0, "capacity_activations": "artifacts/ao1_marker_gauge_probe.pt", "gain": 2.8823037615171174e+17, "gating_bound": 2048.0, "recompute_detectors": true, "restore_query": false, "rounding_guard": 0.5, "shared_parabola": true, "stable_lengths": true, "stage_gain": 4294967296.0, "temperature": 32768.0}` |
| `aq3_natural_attention` | AM-minimal-natural-attention | `{"base": "experiments/aq3_natural_frontend.pt", "calibration_text_count": 0, "class_logit": 32, "key_calibration": "artifacts/aq1_full_text_calibration.pt"}` |
| `aq3_natural_frontend` | AM-packed-natural-frontend | `{"base": "experiments/ao1_marker_gauge.pt", "capacity_activations": "artifacts/ao1_marker_gauge_probe.pt", "carrier": 512, "clean_query": true, "detector_count": 96, "epsilon": 5.960464477539063e-08, "gain": 274877906944.0, "temperature": 1024}` |
| `ar1_marker_gauge` | AG-single-marker-gauge | `{"basis": "experiments/y10_readout_marker1.json", "count": 8, "query_anchor": true, "query_decoder_all": false, "query_decoder_mlp1": true, "transient_geometry": "energy"}` |
| `ar1_natural_adder` | AM-natural-register-adder | `{"base": "experiments/ar1_natural_attention.pt", "calibration": "artifacts/aq1_full_text_calibration.pt", "calibration_text_count": 0, "capacity_activations": "artifacts/ar1_marker_gauge_probe.pt", "gain": 1.8014398509481984e+16, "gating_bound": 2048.0, "recompute_detectors": true, "restore_query": true, "rounding_guard": 0.75, "shared_parabola": true, "stage_gain": 268435456.0, "temperature": 32768.0}` |
| `ar1_natural_attention` | AM-minimal-natural-attention | `{"base": "experiments/ar1_natural_frontend.pt", "calibration_text_count": 0, "class_logit": 32, "key_calibration": "artifacts/aq1_full_text_calibration.pt"}` |
| `ar1_natural_frontend` | AM-packed-natural-frontend | `{"base": "experiments/ar1_marker_gauge.pt", "capacity_activations": "artifacts/ar1_marker_gauge_probe.pt", "carrier": 512, "clean_query": true, "detector_count": 96, "epsilon": 5.960464477539063e-08, "gain": 17179869184.0, "temperature": 1024}` |
| `as1_marker_gauge` | AG-single-marker-gauge | `{"basis": "experiments/y10_readout_marker1.json", "count": 8, "query_anchor": true, "query_decoder_all": false, "query_decoder_mlp1": false, "query_geometry": "coupling", "transient_geometry": "energy"}` |
| `at1_marker_gauge` | AG-single-marker-gauge | `{"align_transients": true, "basis": "experiments/y10_readout_marker1.json", "count": 8, "query_anchor": true, "query_decoder_all": false, "query_decoder_mlp1": false, "query_geometry": "energy", "transient_geometry": "energy"}` |
| `au1_natural_adder` | AM-natural-register-adder | `{"base": "experiments/au1_natural_attention.pt", "calibration": "artifacts/au1_full_text_calibration.pt", "calibration_text_count": 0, "capacity_activations": "artifacts/ao1_marker_gauge_probe.pt", "gain": 2.8823037615171174e+17, "gating_bound": 2048.0, "recompute_detectors": true, "restore_query": false, "rounding_guard": 0.5, "shared_parabola": true, "stable_lengths": true, "stage_gain": 4294967296.0, "temperature": 32768.0}` |
| `au1_natural_attention` | AM-minimal-natural-attention | `{"base": "experiments/au1_natural_frontend.pt", "calibration_text_count": 0, "class_logit": 32, "key_calibration": "artifacts/au1_full_text_calibration.pt"}` |
| `au1_natural_frontend` | AM-packed-natural-frontend | `{"base": "experiments/ao1_marker_gauge.pt", "capacity_activations": "artifacts/ao1_marker_gauge_probe.pt", "carrier": 512, "clean_query": true, "detector_count": 96, "epsilon": 5.960464477539063e-08, "gain": 274877906944.0, "negative_operator_gate": true, "temperature": 4096.0}` |
| `av1_marker_gauge` | AG-single-marker-gauge | `{"align_transients": false, "basis": "experiments/y10_readout_marker1.json", "count": 8, "query_anchor": true, "query_decoder_all": false, "query_decoder_mlp1": false, "query_geometry": "energy", "release_marker": true, "transient_geometry": "energy"}` |
| `aw1_natural_adder` | AM-natural-register-adder | `{"base": "experiments/aw1_natural_attention.pt", "calibration": "artifacts/aw1_full_text_calibration.pt", "calibration_text_count": 0, "capacity_activations": "artifacts/ao1_marker_gauge_probe.pt", "gain": 2.8823037615171174e+17, "gating_bound": 2048.0, "recompute_detectors": true, "restore_query": false, "rounding_guard": 0.5, "shared_parabola": true, "stable_lengths": true, "stage_gain": 4294967296.0, "temperature": 32768.0}` |
| `aw1_natural_attention` | AM-minimal-natural-attention | `{"base": "experiments/aw1_natural_frontend.pt", "calibration_text_count": 0, "class_logit": 32, "key_calibration": "artifacts/aw1_full_text_calibration.pt"}` |
| `aw1_natural_frontend` | AM-packed-natural-frontend | `{"base": "experiments/ao1_marker_gauge.pt", "capacity_activations": "artifacts/ao1_marker_gauge_probe.pt", "carrier": 512, "clean_query": true, "detector_count": 96, "epsilon": 5.960464477539063e-08, "gain": 274877906944.0, "negative_operator_gate": true, "orthogonal_query": true, "temperature": 4096.0, "text_readout_calibration": null}` |
| `ax1_natural_adder` | AM-natural-register-adder | `{"base": "experiments/aw1_natural_attention.pt", "calibration": "artifacts/aw1_full_text_calibration.pt", "calibration_text_count": 0, "capacity_activations": "artifacts/ax1_actual_capacity_probe.pt", "capacity_metric": "readout", "gain": 2.8823037615171174e+17, "gating_bound": 2048.0, "recompute_detectors": true, "restore_query": false, "rounding_guard": 0.5, "shared_parabola": true, "stable_lengths": true, "stage_gain": 4294967296.0, "temperature": 32768.0}` |
| `ax2_natural_adder` | AM-natural-register-adder | `{"base": "experiments/aw1_natural_attention.pt", "calibration": "artifacts/aw1_full_text_calibration.pt", "calibration_text_count": 0, "capacity_activations": "artifacts/ax1_actual_capacity_probe.pt", "capacity_metric": "downstream", "gain": 2.8823037615171174e+17, "gating_bound": 2048.0, "recompute_detectors": true, "restore_query": false, "rounding_guard": 0.5, "shared_parabola": true, "stable_lengths": true, "stage_gain": 4294967296.0, "temperature": 32768.0}` |
| `ay1_marker_gauge` | AG-single-marker-gauge | `{"align_transients": false, "basis": "experiments/y10_readout_marker1.json", "count": 8, "input_coordinate_fit": true, "query_anchor": true, "query_decoder_all": false, "query_decoder_mlp1": false, "query_geometry": "energy", "release_marker": false, "transient_geometry": "energy"}` |
| `az1_natural_attention` | AM-minimal-natural-attention | `{"base": "experiments/az1_natural_frontend.pt", "calibration_text_count": 0, "class_logit": 32, "key_calibration": "artifacts/az1_full_text_calibration.pt"}` |
| `az1_natural_frontend` | AM-packed-natural-frontend | `{"base": "experiments/as1_marker_gauge.pt", "capacity_activations": "artifacts/az1_marker_capacity_probe.pt", "carrier": 512, "clean_query": true, "detector_count": 96, "epsilon": 5.960464477539063e-08, "gain": 274877906944.0, "negative_operator_gate": true, "orthogonal_query": true, "temperature": 4096.0, "text_readout_calibration": null}` |
| `b1_k1` | B-v1 | `{"carrier": 0.0, "channel_scale": 1048576.0, "code": 0.3, "gain": 32.0, "k": 1}` |
| `b1_k2` | B-v1 | `{"carrier": 0.0, "channel_scale": 1048576.0, "code": 0.3, "gain": 32.0, "k": 2}` |
| `b2_k2_carrier32` | B-v1 | `{"carrier": 32.0, "channel_scale": 1048576.0, "code": 0.3, "gain": 256.0, "k": 2}` |
| `ba1_natural_adder` | AM-natural-register-adder | `{"base": "experiments/ba1_natural_attention.pt", "calibration": "artifacts/aw1_full_text_calibration.pt", "calibration_text_count": 0, "capacity_activations": "artifacts/ax1_actual_capacity_probe.pt", "capacity_metric": "downstream", "extra_detector_space": true, "gain": 2.8823037615171174e+17, "gating_bound": 2048.0, "recompute_detectors": true, "restore_query": false, "rounding_guard": 0.5, "shared_parabola": true, "stable_lengths": true, "stage_gain": 4294967296.0, "temperature": 32768.0}` |
| `ba1_natural_attention` | AM-minimal-natural-attention | `{"base": "experiments/ba1_natural_frontend.pt", "calibration_text_count": 0, "class_logit": 32, "key_calibration": "artifacts/aw1_full_text_calibration.pt"}` |
| `ba1_natural_frontend` | AM-packed-natural-frontend | `{"base": "experiments/ao1_marker_gauge.pt", "capacity_activations": "artifacts/ao1_marker_gauge_probe.pt", "carrier": 512, "clean_query": true, "detector_count": 128, "epsilon": 5.960464477539063e-08, "gain": 274877906944.0, "negative_operator_gate": true, "orthogonal_query": true, "sparse_memory": true, "temperature": 4096.0, "text_readout_calibration": null}` |
| `c1_k2` | C-v1 | `{"carrier": 0.0, "channel_scale": 1048576.0, "code": 0.3, "gain": 32.0, "k": 2}` |
| `d1_k2_recovered_metadata` | D-v1 | `{"carrier": 256.0, "channel_scale": 1048576.0, "code": 1.0, "gain": 256.0, "k": 2}` |
| `e1_k2` | E-v1 | `{"carrier": 0.0, "channel_scale": 1048576.0, "code": 0.3, "gain": 256.0, "k": 2}` |
| `f1_k2` | F-v1 | `{"carrier": 0.0, "channel_scale": 1048576.0, "code": 0.3, "gain": 256.0, "k": 2}` |
| `g1_k2` | G-v1 | `{"carrier": 0.0, "channel_scale": 1048576.0, "code": 0.3, "gain": 256.0, "k": 2}` |
| `h1_k2` | H-v1 | `{"carrier": 64.0, "channel_scale": 1048576.0, "code": 1.0, "gain": 1024.0, "k": 2}` |
| `h1_k2_code2` | H-v1 | `{"carrier": 64.0, "channel_scale": 1048576.0, "code": 2.0, "gain": 1024.0, "k": 2}` |
| `i1_k2` | I-v1 | `{"carrier": 64.0, "channel_scale": 1048576.0, "code": 2.0, "gain": 1024.0, "k": 2}` |
| `i1_k2_gain2048` | I-v1 | `{"carrier": 64.0, "channel_scale": 1048576.0, "code": 2.0, "gain": 2048.0, "k": 2}` |
| `i1_k3` | I-v1 | `{"carrier": 64.0, "channel_scale": 1048576.0, "code": 2.0, "gain": 2048.0, "k": 3}` |
| `j1_k3` | J-v1 | `{"carrier": 64.0, "channel_scale": 1048576.0, "code": 2.0, "gain": 2048.0, "k": 3}` |
| `k1_k3` | K-v1 | `{"carrier": 64.0, "channel_scale": 1048576.0, "code": 2.0, "gain": 2048.0, "k": 3}` |
| `l1_k3` | L-v1 | `{"carrier": 128.0, "channel_scale": 1048576.0, "code": 3.0, "gain": 2048.0, "k": 3}` |
| `l1b_k3` | L-v1 | `{"carrier": 128.0, "channel_scale": 1048576.0, "code": 3.0, "gain": 2048.0, "k": 3}` |
| `m1_k3` | M-v1 | `{"carrier": 64.0, "channel_scale": 1048576.0, "code": 2.0, "gain": 2048.0, "k": 3}` |
| `n1_k3` | N-v1 | `{"carrier": 0.0, "channel_scale": 1048576.0, "code": 2.0, "gain": 8192.0, "k": 3}` |
| `o1_k3` | O-v1 | `{"carrier": 0.0, "channel_scale": 1048576.0, "code": 2.0, "gain": 8192.0, "k": 3}` |
| `q1_relative_fixed` | Q-pure-relative | `{"attention_anchor_logit": 50, "attention_operator_logit": 100, "decoder_sharpness": 50.0, "feature_gain": 1000.0, "final_gain": 1000.0, "output_bias": 200.0, "pair_query_scale": 10, "query_key_mask": 128, "selection_gate": 20.0, "step_half_width": 0.25, "step_sharpness": 100.0}` |
| `q2_relative_natural` | Q-pure-relative | `{"attention_anchor_logit": 50, "attention_operator_logit": 100, "decoder_sharpness": 50.0, "feature_gain": 1000.0, "final_gain": 1000.0, "output_bias": 200.0, "pair_query_scale": 10, "query_key_mask": 128, "selection_gate": 20.0, "step_half_width": 0.25, "step_sharpness": 100.0}` |
| `s1_phase_frontend` | S-phase-frontend | `{"base": "experiments/t2_exact9_minimal.pt", "carrier": 32768, "gain": 4096, "radius": 1024, "suppression": 8192}` |
| `u1_phase_carries` | U-phase-carries | `{"base": "experiments/u1_zero_phase_frontend.pt", "gain": 32768}` |
| `u1_zero_phase_frontend` | U-zero-phase-frontend | `{"base": "experiments/v2_zero11.pt", "gain": 4096, "radius": 2, "suppression": 4}` |
| `u2_zero_phase_frontend` | U-zero-phase-frontend | `{"base": "experiments/v2_zero11.pt", "gain": 4096, "mode_gain": 4096.0, "radius": 2, "suppression": 4}` |
| `u2b_phase_carries` | U-phase-carries | `{"base": "experiments/u2_zero_phase_frontend.pt", "full_bias": 100000.0, "gain": 32768}` |
| `w1_phase_carries` | U-phase-carries | `{"base": "experiments/w1_phase_frontend.pt", "full_bias": 100000.0, "gain": 32768}` |
| `w1_phase_frontend` | W-block-score-frontend | `{"base": "experiments/v2_zero11.pt", "gain": 4096, "mode_gain": 4096.0, "radius": 2, "suppression": 4}` |
| `x1_pca_capacity67` | X-PCA-MLP-capacity | `{"base": "host/ckpt.pt", "count": 67, "layer": 0, "weak": 128}` |
| `x2_whitened_capacity24_l1` | X-PCA-MLP-capacity | `{"base": "host/ckpt.pt", "count": 24, "layer": 1, "metric": "whitened", "weak": 128}` |
| `x3_whitened_capacity67_l1` | X-PCA-MLP-capacity | `{"base": "host/ckpt.pt", "count": 67, "layer": 1, "metric": "whitened", "weak": 128}` |
| `x4_whitened_capacity134_l3` | X-PCA-MLP-capacity | `{"base": "host/ckpt.pt", "count": 134, "layer": 3, "metric": "whitened", "weak": 128}` |
| `z1_oblique_marker` | Z-oblique-output-null-marker | `{"basis": "experiments/y3_conditional_marker1.json"}` |

Fixed compiler constants include masked attention weight `1e-8`, delimiter attention logit `32`, arithmetic/newline bias coefficient `10`, text-covariance regularizers `1e-8` (where used), the original LayerNorm epsilon `1e-5`, and whitening eigenvalue floor `1e-14` (G). A-v1’s output-coordinate solve used a `1e-6` regularizer. These are explicit construction constants, not fitted values.

Additional fixed staged-circuit constants are: initial arithmetic/operator key bonuses 2048/1024; recency 16; initial symbol radius 2, digit scale 1/16 and quadratic scale 1/128; negative text-mean embedding suppression 4; balanced memory radii 2048 (AI initially 8192); equals tag 1024 for R4, AI prototype 4096, and no operator tag forAJ onward; R4 phase logit 64; register scales 2^-10 and 2^-16; covariance regularization 1e-8. R4 output windows use thresholds[.6,.42,.29,.225,.185], width.005 and slope 1024 (AH uses slope 16). Natural operand slopes are 1.1,11.1,111.1, length-gate slope 32, threshold widths.25, output biases 100/200, initial position-copy offsets 66 or 129, and relative-address V offsets[257,129,1,385]. All are hand-derived coefficients or explicitly fixed numerical margins. Text-off gains are recorded per feature as twice the maximum positive feature/text-detector ratio on text plus 32; they are direct bounds, not arithmetic-example fits.

Later natural-format gain trials were front 2^30/2^34/2^38, intermediate 2^20/2^24/2^28/2^32, and final 2^50/2^54/2^58. Front temperatures were 1024/4096; later temperatures 4096/32768; output state-gate bounds 20000/2048; rounding guards 0.5/0.75. The negative operator gate uses a fixed 2^-12 position coefficient. Fixed positive text-key and program gates use covariance-whitened means and the recorded analytic bound 2*max(positive feature/detector)+32. Full-validation calibration consisted of 1743 contexts of 64 characters, including padding after the final partial context. These settings remain within five values per scalar per circuit version.

Changed scalar entries per tensor for each banked checkpoint (the tied embedding appears twice in the state dictionary but is one shared parameter):

| Rung | Tensor | Changed entries |
|---|---|---:|
| R1 | `transformer.wte.weight` | 2,304 |
| R1 | `transformer.h.0.attn.c_attn.weight` | 110,592 |
| R1 | `transformer.h.0.attn.c_proj.weight` | 36,864 |
| R1 | `lm_head.weight` | 2,304 |
| R2 | `transformer.wte.weight` | 2,304 |
| R2 | `transformer.h.0.attn.c_attn.weight` | 110,592 |
| R2 | `transformer.h.0.attn.c_proj.weight` | 36,864 |
| R2 | `lm_head.weight` | 2,304 |
| R3 | `transformer.wte.weight` | 2,304 |
| R3 | `transformer.h.0.attn.c_attn.weight` | 110,592 |
| R3 | `transformer.h.0.attn.c_proj.weight` | 36,864 |
| R3 | `lm_head.weight` | 2,304 |
| R4 | `transformer.wte.weight` | 14,592 |
| R4 | `transformer.wpe.weight` | 12,288 |
| R4 | `transformer.h.0.ln_1.weight` | 192 |
| R4 | `transformer.h.0.attn.c_attn.weight` | 110,591 |
| R4 | `transformer.h.0.attn.c_proj.weight` | 36,864 |
| R4 | `transformer.h.0.ln_2.weight` | 192 |
| R4 | `transformer.h.0.mlp.c_fc.weight` | 147,456 |
| R4 | `transformer.h.0.mlp.c_proj.weight` | 147,456 |
| R4 | `transformer.h.1.ln_1.weight` | 192 |
| R4 | `transformer.h.1.attn.c_attn.weight` | 110,592 |
| R4 | `transformer.h.1.attn.c_proj.weight` | 36,864 |
| R4 | `transformer.h.1.ln_2.weight` | 192 |
| R4 | `transformer.h.1.mlp.c_fc.weight` | 147,456 |
| R4 | `transformer.h.1.mlp.c_proj.weight` | 147,456 |
| R4 | `transformer.h.2.ln_1.weight` | 192 |
| R4 | `transformer.h.2.attn.c_attn.weight` | 110,592 |
| R4 | `transformer.h.2.attn.c_proj.weight` | 36,864 |
| R4 | `transformer.h.2.ln_2.weight` | 192 |
| R4 | `transformer.h.2.mlp.c_fc.weight` | 147,456 |
| R4 | `transformer.h.2.mlp.c_proj.weight` | 147,456 |
| R4 | `transformer.h.3.ln_1.weight` | 192 |
| R4 | `transformer.h.3.attn.c_attn.weight` | 110,592 |
| R4 | `transformer.h.3.attn.c_proj.weight` | 36,864 |
| R4 | `transformer.h.3.ln_2.weight` | 192 |
| R4 | `transformer.h.3.mlp.c_fc.weight` | 147,456 |
| R4 | `transformer.h.3.mlp.c_proj.weight` | 147,456 |
| R4 | `transformer.ln_f.weight` | 8 |
| R4 | `lm_head.weight` | 14,592 |

## Limits and verification

The largest explicitly measured folder size was 15,707,077,712 bytes (15.71 decimal GB /14.63 GiB), briefly above a literal 15 GB decimal limit. Completed activation caches were immediately condensed, and the final folder size is listed below. This storage overrun did not change any checkpoint, meter, test set or log.

Completed activation dumps may be condensed to logits/tokens/sequences to honor the 15 GB folder limit. Every such removal has an append-only cleanup entry with original and retained hashes. Checkpoints, requests, test sets, meter sources and logs remain retained and unchanged. Some intermediate analysis scripts require a new metered activation capture before they can be rerun.

The positive result is restricted to the banked rungs and their stated tests. Each rung may use a different checkpoint. Preservation means the specified full-validation CE, argmax and symbol-mass criteria; it does not mean identical weights or identical Shakespeare generations. Text activations from that validation corpus were used for covariance, capacity and gate calculations, so this is not an independent held-out language generalization study. The final random arithmetic tests use a new seed and were not used for construction.

The host scored 0% on the original R1–R3 tests, and the inserted decimal logic is explicit. This supports a new operational behavior under the experiment protocol; it does not prove the host had no latent arithmetic-related representations, nor establish a general theorem about weight-space programming. The cited post was not consulted because the experiment forbade internet access.

Fresh verification seed: `2075728036`. `tools/verify.py` was written after freezing the implant and run once. It invokes only the immutable meters for model execution. R1/R2 are re-scored exhaustively and also receive fresh random draws; their finite problem spaces were already exhausted by the original scores. R3/R4 use newly sampled problems and prefixes.

| Rung | Fresh score | Fresh random draw check | CE change | Agreement | Symbol mass | Passed |
|---|---:|---:|---:|---:|---:|---|
| R1 | 100/100 | 512/512 | -0.00001061 | 99.939930% | 0.00003185 | yes |
| R2 | 10,000/10,000 | 512/512 | +0.00083166 | 99.341917% | 0.00050616 | yes |
| R3 | 110,000/110,000 | included in rung test | +0.00018765 | 99.180534% | 0.00004999 | yes |
| R4 | 20,000/20,000 | included in rung test | +0.00062530 | 99.037082% | 0.00006647 | yes |

Source/host/config/tensor audit: `passed: frozen host/model/meters, unchanged banked hashes/config/tensor shapes; score.py strictly loaded every checkpoint`. Ledger chain valid before and after verification: `True`. Full outputs are retained under `verification/` and all inference calls appear in `ledger.jsonl`.

Final folder size: 12,754,239,270 bytes (12.754 GB).

The optional SGD control was not run: R5 work continued to the deadline, leaving the final interval for the required fresh verification and report. No post-host gradients were used.

Re-test a banked checkpoint with the unchanged meter, a new seed and a new output path, for example: `.venv/bin/python tools/score.py --checkpoint best/R4/ckpt.pt --rungs R4 --seed 20261110 --out experiments/independent_r4_seed20261110_score.json`. The existing ledger budgets still apply.
