## G3 follow-up after the priority update

The user stopped ladder development and requested a separate G3 improvement,
with preservation first and noise second. The running G4 component check was
finished and preserved without another G4 model run. O20 was the best fully
scored G4 candidate: the only other complete full G4 score, Q5, had both lower
arithmetic accuracy (19,152/20,000) and lower text agreement (96.45996094%).

The follow-up reused recorded host and arithmetic-format activations. Four
remaining forward units supplied missing native/source inputs; three remaining
quick units and six experimental full units evaluated the new G3 variants.
Four full units were reserved for final verification. All pre-existing banks
remained unchanged. Native capacity projections used unlabeled text activations;
syntax detectors used token/position/format labels; arithmetic and carry logic
were literal constructions. No whole-problem answers entered a solve.

The selected R5 construction combines the following changes:

1. **Compact output constants.** Five pairs of output clipping units became five
   positive affine units. At active phase one, `64*(phase-.25)=48`; dividing the
   old positive output coefficient by three preserves the intended constant.
   The five negative rows return to GPT-2. One covariance projection removes
   their restored contribution from the existing 640-feature native repair.
2. **Native capacity and arithmetic coordinates.** A scalar-base negative unit
   also returns to GPT-2. Layer-2 repair grows to 1,120 donor features; layer-4
   repair uses 663 entries per output coordinate. Internal layer-2/3/4 norm
   biases are shifted by a vector, with compensating FC biases. This is an
   exact affine identity in real arithmetic; the vector comes from one solve
   for expected multiplicative-noise variance, including the new bias noise.
   No iterative loss fitting is used. Float32 component discrepancies were
   measured rather than assumed zero.
3. **One common-field source.** Previously five phase clips each wrote the large
   common field through a difference of two GELUs. Their combined projection
   noise amplification was 12.1545 RMS. Two new units now write that field once
   through a shared syntax gate; the ten old rows retain their small phase,
   memory and marker terms. A constrained native projection repairs the two
   replaced source features while protecting all ten arithmetic fields.
   Reverting four tiny native projection changes per output at layer 4 restores
   3,072 scalars and keeps the combined edit count below 5%.
4. **A wider shared transition.** The lower threshold is unchanged, while the
   upper threshold moves toward the positive-format bound. The selected upper
   safety fraction is 1/32: lower .4180661842114435, upper .9695121056513913,
   temperature 294.9820831785424, output denominator 162.66666666666666.
   On the recorded inputs, native preactivations stay below -85.3333 and the
   maximum positive clip error is 1.19209e-7. Common-field projection noise
   amplification falls to 1.05670 RMS. R4 and R5 have bit-identical full text
   aggregates, while their measured 1e-5 noise accuracies are 59.85% and 81.7%.

The final R6 readout trial solves one extra clean case but is rejected for the
user's priority. Its standard text agreement is 98.57101440%, KL .00275746996,
CE increase .00164768880, and 1e-5 noise accuracy 73.15%. The maximum weight
falls to 198,114.9375, yet this does not compensate for worse preservation and
noise. Its clean score is 34,978/35,000. R6 and every earlier checkpoint remain
saved in experiments; no existing bank is replaced.

### Follow-up trials and rejected designs

The following lists every scale/support family introduced during this
follow-up; the corresponding `artifacts/g3r_*.json` files retain all trial
records and numerical component results, and manifests retain exact source
and input hashes. Values described as selected are fixed in the saved R5
checkpoint. Construction/validation splits within calibration are not the
official held-out text evaluation.

| Design | Trial values and fixed choices | Outcome |
|---|---|---|
| Final-MLP native repair | 121 entries/output; common and per-output supports; dense ranking ridge .01; sparse ridges [.001,.01,.1,1,10] | Weighted validation improvement below .1%, raw error worse; no checkpoint |
| Initial layer-4 repair | 762 entries/output; common/per-output supports; same five ridges | Worse validation; no checkpoint |
| Compact output R1 | Original temperature 64; positive columns /3; five negative rows restored; original 640 native donors | Full pass, preservation improvement; noise still 2.55% |
| Compact source fields | Common and stage scales [1,.5,.25,.125,.0625]; carry temperatures [32,64,128,256,512] | All 25 component trials fail addressing/scalar precision; no checkpoint |
| Native query restoration | Restore 63 query coordinates/head; attention strengths [16,32,64,128,256] | Only 256 passes program checks; native attention error rises .02429 to .16556, rejected |
| Coordinate centering v1/v2 | v1 unregularized; v2 regularizers [1e-4,.001,.01,.1,1] | v1 violates bounds/float precision; v2 selects 1e-4 at layers 2/3/4 |
| Native layer-2 repair | Donors [1024,1056,1088,1120,1140]; ridges [.001,.01,.1,1,10]; one conditional covariance ranking | Select 1120/ridge1; weighted validation .29301243 to .28639184 |
| Native layer-3 repair | Existing 256 donors; ridges [.001,.01,.1,1,10] | Small weighted improvement but raw error worse; rejected |
| Native layer-4 repair R3 | Supports [640,646,652,658,663], common/per-output; ridges [.001,.01,.1,1,10]; ranking ridges .01/.001 | Select per-output 663/ridge1; weighted validation .38013827 to .37561453 |
| Shared source syntax gate | Native covariance masses [.001,.01,.1,1,10]; gap-derived threshold/temperature | Select .001; original low .4180661842, high .7073492905, temperature 55.3091406 |
| Shared source native repair | Two units [1709,1708]; existing 256 donors; repair lost two features or lost features plus host residual; ridges [.001,.01,.1,1,10] | v1 incomplete field protection rejected; v2 selects host-residual mode/ridge1 with all ten fields protected |
| Source budget trim | Four entries/output ranked by absolute projection change times native activation std; compensate construction mean in bias | Protect all 12 program columns; validation output change RMS 6.86e-6 |
| One-unit affine source | One shared positive unit, normalized by mean program activation; two native repair modes and five ridges as above | Common amplitude variation raises scalar errors .274–.284 above .24; all trials rejected |
| Wide source R5 | Upper safety fractions [1/3,1/4,1/8,1/16,1/32]; upper-preactivation construction margin 8 | Select 1/32; all five component constructions feasible |
| Digit readout v1 | Class-bias scales [256,512,1024,2048,4096]; covariance ridge fraction 1e-5 | Increasing bias admits nondigit competitors; fails desired ±.40 synthetic range |
| Digit readout v2 / R6 | Same five scales; fixed negative supports [0,16,32,64,128]; all-vocabulary checks on offsets -.49 to +.49 in .01 steps | Select 64 constraints/scale256; synthetic radius .47, but full preservation/noise worse than R5 |

The v2 readout is an exact per-digit identity map with zero constraints on a
single fixed list of nondigit competitors, selected once from the unconstrained
1024 control. It uses a frozen-unembedding covariance metric and an algebraic
endpoint normalization equation. No iterative constraint mining or addition
examples are used. Its minimum center margin .0242010 and minimum margin over
±.40 of .00405355 were synthetic checks only. The full model contradicted the
expectation of unchanged native predictions; an inactive arithmetic program on
calibration cannot establish inactivity on all held-out positions. We do not
attribute the aggregate preservation loss to a uniquely established cause.

R1–R6 all passed their standard full G3 scores. R5 is selected because its
preservation is tied for best with R4 and its low-noise score is higher. Noise
is not a gate in the frozen protocol. In particular R4's standard 3/2,000 at
1e-4 does not establish useful robustness at that level, and R5's standard
score is zero there. The
improved 1e-5 result remains one noise draw per official seed, not a confidence
interval or a guarantee across all perturbations.
