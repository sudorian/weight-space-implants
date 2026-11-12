# Gemma 3 270M IT direct-weight addition — final report

Finalized at Dorian's explicit early-closeout request on **2026-11-11 17:06:52 CET**, because usage was nearly exhausted. This supersedes the scheduled climbing and deadline instructions in GOAL.md. No new experiment, weight build, or model score was started after that request. No score was running; the sole running analysis, A362, was interrupted (exit 130). All pending work is canceled.

**Documented negative result: no rung passed.** Direct weight programming produced strong three-digit addition after BOS, but did not satisfy the required preservation of ordinary text. This is a stopped run, not a proof that the task is impossible and not a claim of a technical blocker. The best completed H1 attempt is **A313**, saved in **`best/H1-best-attempt/`**, explicitly **not passing**. There is no `best/H1/` bank.

A313's official full run **0807-score** scored **24,961/25,000 exact additions (99.844%)**, including the terminator, and the same number of correct digit prefixes. Its text agreement was **98.02474975585938%**, below **98.5%**. It matched **128,483/131,072** host predictions; the gate needs at least **129,106**, a deficit of **623 predictions**, or **0.47525024414062 percentage points**. KL was **0.0024928829704274477 nats** and cross-entropy change was **−0.0002293046878767413 nats** (host 5.023652493858176; edited 5.023423189170299). Values were finite. Arithmetic, KL, CE, and edit limits pass; text top-1 agreement prevents the joint H1 pass.

Selection uses the highest completed full-score text agreement among candidates meeting H1's arithmetic threshold. A313 beats the suggested A189 on both arithmetic (24,961 versus 24,960) and text (98.02474975585938% versus 97.94158935546875%, or 109 additional matching predictions). A234 had higher arithmetic accuracy, 24,994/25,000, but only 97.87673950195312% text agreement. A285's restoration diagnostic reached 98.21319580078125% text agreement but was not an arithmetic-qualified H1 candidate and scored 0/5,000 on H5; it also missed the text gate.

The sparse delta, original manifest, official score, and edit-category audit were copied byte-for-byte to `best/H1-best-attempt/`. Delta SHA-256: `67fd7d1ae9b0d63f4d2f8993b65f1e837fb65a418804e56125a8b7eb4cec49e5`. Sources and trial values remain in the detailed journal below, original manifests, and analysis JSON records. No activation capture is included in the export.

| Leader robustness measurement | Exact | Correct digit prefixes |
|---|---:|---:|
| Multiplicative noise sigma 1e-5 | 1,997/2,000 | 1,997/2,000 |
| Multiplicative noise sigma 1e-4 | 1,997/2,000 | 1,997/2,000 |
| Multiplicative noise sigma 1e-3 | 1,602/2,000 | 1,605/2,000 |

Two-problem diagnostic: 1,997/2,000 first answers, 1/2,000 second answers in the meter's `second_given_correct_first` field, and 1/2,000 pairs both correct. These diagnostic counts are not additional rung passes. Changed sparse indices can change which noise draw reaches which scalar; cross-candidate noise differences are not controlled causal comparisons.

| Rung | Unedited host exact / 2,000 | Host correct digit prefix / 2,000 | Final edited status |
|---|---:|---:|---|
| H1 | 3 (0.15%) | 702 (35.10%) | A313: 24,961/25,000; preservation fails |
| H2 | 211 (10.55%) | 242 (12.10%) | No passing edited checkpoint; planned A363 full diagnostic canceled |
| H3 | 4 (0.20%) | 17 (0.85%) | No passing edited checkpoint |
| H4 | 88 (4.40%) | 121 (6.05%) | No passing edited checkpoint |
| H5 | 0 (0.00%) | 0 (0.00%) | A285 full diagnostic: 0/5,000; no passing checkpoint |

Host quick self-comparison: 256 windows / 16,384 targets, 100% agreement, KL 0, CE change 0, finite, host CE 5.0004215402 nats. Baseline quick arithmetic and leader full arithmetic use different sample sizes; the percentages are not a paired estimate. The leader's full text result uses 2,048 windows / 131,072 targets. Complete official aggregates are in `scores/` and the immutable ledger.

**Meter use:** 300/300 forward units, 100/100 quick score units, and 26/40 full score units (25 H1 runs and one H5 diagnostic). Fourteen full units remained unused. No rung first passed, so there is no first-pass meter allocation to report. All 407 meter starts have matching finishes; the ledger has 815 rows and final chain head `9259642b1ef176fb81836cf5b78048504895e549c473bb5990cc1d2cc36d4a39`. The frozen source hashes still match. No ledger entry was changed or appended during closeout.

**Final edit accounting:** 4,976,785 of 100,325,376 eligible scalars changed (4.960644254151612%, below the 5,016,268-scalar cap). Maximum absolute model weight is **1,000,000**, at the permitted limit. Of the changed scalars, **578,041** are labeled arithmetic/circuit accommodation and **4,398,744** native-only text repair. These disjoint bookkeeping labels describe purpose, not a causal attribution: some native repairs overlap arithmetic-channel allocations. The inherited labels and A313 compiler masks are documented in `edit_categories.json`; its scope is metadata reconciliation, not a new sparse-index audit. The official meter independently provides the total per tensor. In the following table, tensor names are prefixed by `model.layers.`.

| Tensor (without `model.layers.`) | Arithmetic / accommodation | Native text repair | Total changed |
|---|---:|---:|---:|
| `0.mlp.down_proj.weight` | 640 | 42,942 | 43,582 |
| `0.mlp.gate_proj.weight` | 640 | 0 | 640 |
| `0.mlp.up_proj.weight` | 640 | 0 | 640 |
| `0.post_feedforward_layernorm.weight` | 1 | 0 | 1 |
| `1.mlp.down_proj.weight` | 11,372 | 325,632 | 337,004 |
| `1.mlp.gate_proj.weight` | 3,200 | 0 | 3,200 |
| `1.mlp.up_proj.weight` | 3,200 | 0 | 3,200 |
| `1.post_feedforward_layernorm.weight` | 4 | 634 | 638 |
| `2.mlp.down_proj.weight` | 2,560 | 333,808 | 336,368 |
| `2.mlp.gate_proj.weight` | 2,560 | 0 | 2,560 |
| `2.mlp.up_proj.weight` | 2,560 | 0 | 2,560 |
| `2.post_feedforward_layernorm.weight` | 4 | 636 | 640 |
| `2.self_attn.q_norm.weight` | 32 | 63 | 95 |
| `2.self_attn.q_proj.weight` | 92,160 | 163,840 | 256,000 |
| `3.mlp.down_proj.weight` | 9,204 | 325,630 | 334,834 |
| `3.mlp.gate_proj.weight` | 2,560 | 0 | 2,560 |
| `3.mlp.up_proj.weight` | 2,560 | 0 | 2,560 |
| `3.post_feedforward_layernorm.weight` | 3 | 637 | 640 |
| `3.self_attn.o_proj.weight` | 0 | 98,304 | 98,304 |
| `3.self_attn.q_norm.weight` | 64 | 192 | 256 |
| `3.self_attn.q_proj.weight` | 163,840 | 491,519 | 655,359 |
| `3.self_attn.v_proj.weight` | 163,839 | 0 | 163,839 |
| `4.mlp.down_proj.weight` | 3,840 | 87,662 | 91,502 |
| `4.mlp.gate_proj.weight` | 3,840 | 0 | 3,840 |
| `4.mlp.up_proj.weight` | 3,840 | 0 | 3,840 |
| `4.post_feedforward_layernorm.weight` | 3 | 632 | 635 |
| `5.mlp.down_proj.weight` | 5,500 | 81,536 | 87,036 |
| `5.mlp.gate_proj.weight` | 1,280 | 0 | 1,280 |
| `5.mlp.up_proj.weight` | 1,280 | 0 | 1,280 |
| `5.post_feedforward_layernorm.weight` | 2 | 638 | 640 |
| `6.mlp.down_proj.weight` | 6,400 | 339,887 | 346,287 |
| `6.mlp.gate_proj.weight` | 6,400 | 0 | 6,400 |
| `6.mlp.up_proj.weight` | 6,400 | 0 | 6,400 |
| `6.post_feedforward_layernorm.weight` | 9 | 631 | 640 |
| `7.mlp.down_proj.weight` | 21,952 | 323,072 | 345,024 |
| `7.mlp.gate_proj.weight` | 5,120 | 0 | 5,120 |
| `7.mlp.up_proj.weight` | 5,120 | 0 | 5,120 |
| `7.post_feedforward_layernorm.weight` | 8 | 632 | 640 |
| `7.self_attn.k_norm.weight` | 251 | 0 | 251 |
| `7.self_attn.k_proj.weight` | 1,280 | 0 | 1,280 |
| `7.self_attn.q_norm.weight` | 255 | 0 | 255 |
| `7.self_attn.q_proj.weight` | 5,120 | 0 | 5,120 |
| `8.mlp.down_proj.weight` | 5,760 | 1,304,956 | 1,310,716 |
| `8.mlp.gate_proj.weight` | 5,760 | 0 | 5,760 |
| `8.mlp.up_proj.weight` | 5,760 | 0 | 5,760 |
| `8.post_feedforward_layernorm.weight` | 2 | 637 | 639 |
| `17.mlp.down_proj.weight` | 7,040 | 474,080 | 481,120 |
| `17.mlp.gate_proj.weight` | 7,040 | 0 | 7,040 |
| `17.mlp.up_proj.weight` | 7,040 | 0 | 7,040 |
| `17.post_feedforward_layernorm.weight` | 96 | 544 | 640 |
| **Total** | **578,041** | **4,398,744** | **4,976,785** |

The architecture and protected tensors stayed fixed: unmodified installed Gemma3ForCausalLM, SDPA, float32, strict state loading, tied embedding/output weights, unchanged final norm and buffers. Weights were computed directly from hand-derived arithmetic, format/token geometry, recorded activations, and closed-form reconstruction of native behavior on unlabeled calibration text. No gradient training or fitting of whole addition problems and answers was used. The source host SHA and software versions are recorded below and in `host/manifest.json`.

The main design paths and their measured limits are summarized here; the chronological journal that follows retains the individual designs, all declared trial values, failures, corrections, and manifests rather than treating each variation as a new independent success.

| Design path | How close it came / reason it did not finish the task |
|---|---|
| Native format detectors, token-coordinate geometry, and static digit readout | Detector families did not establish a reliable complete format gate. A parabolic fixed-unembedding readout achieved minimum static digit margin 0.7546389247 at bias 32, a component result only. |
| Protected residual coordinates (B0/B0R/B2) | Ordinary-text agreement was 89.80102539%, 89.46533203%, and 84.31396484%; no reliable full arithmetic circuit. |
| Gated physical carrier, conditional clearing, and explicit arithmetic | A1 marker preserved 99.94506836% quick text agreement but scored 0/2,000 additions. Completing readers, carry logic, and output selection reached A117's 24,991/25,000 full additions, at 97.576904% text agreement. |
| Native-value copying, common carry frame, reduced selector span, and terminator recoding | Reduced displacement of pretrained computation. A142v2 reached 24,990/25,000 and 97.849274% agreement; A189 reached 24,960 and 97.941589%. Still below the text gate. |
| Direct reader geometry and arithmetic refinements | A234 reached 24,994/25,000 (99.976%), the highest full H1 arithmetic result, but 97.876740% text agreement. Local reader guarantees did not certify changed end-to-end trajectories. |
| Closed-form native-text repair, sparse allocation, and original-HOST reconstruction | A259 reached 98.017883% agreement with 24,960 exact additions. Full native-final restoration A285 gave 98.213196% agreement but destroyed the required circuit behavior and scored 0/5,000 on the H5 diagnostic. |
| Exact duplicate reference compression | A313 merged duplicate B-reader units 529 and 727 into 529, restored unit 727 natively, and retained units [80,342,529,1634]. It is the best qualified H1 attempt at 99.844% arithmetic and 98.024750% text agreement. |
| Native-phase and compact plus-referenced readers | A342's local bounds did not transfer: 23,546/25,000 full H1. A354 cut edits to 4,589,583 and scored 24,946/25,000, but agreement fell to 97.940063%. |
| Recorded-linear identity audits and expanded native-Up repairs | Matched linear controls passed. A358's 125 and A359's 625 candidates failed their selection guards. A361's 5,000 candidates across eight independent early layers all failed; its best local state-RMS gain was 2.816702%, below the required 5%, not a measured text-prediction gain. |
| Coupled native feature/output construction A362 | Interrupted at user closeout, with seven completed feature-map reports in its partial log. Best partial restricted-match count was 7,990 versus unchanged 8,003. No completed family result, checkpoint, or official score exists. A363 and A364 were never run. |

Full H1 score history (all entries failed the joint gate):

| Candidate / score file | Run | Exact / 25,000 | Text agreement | Changed scalars |
|---|---|---:|---:|---:|
| `a117v1_full_H1.json` | 0683-score | 24,991 | 97.576904297% | 4,124,677 |
| `a142v2_full_H1.json` | 0757-score | 24,990 | 97.849273682% | 4,880,396 |
| `a173v1_full_H1.json` | 0763-score | 24,989 | 97.870635986% | 5,011,468 |
| `a176v1_full_H1.json` | 0765-score | 24,990 | 97.864532471% | 4,978,700 |
| `a179v1_full_H1.json` | 0767-score | 24,989 | 97.821807861% | 5,011,468 |
| `a178v1_full_H1.json` | 0771-score | 24,990 | 97.826385498% | 4,978,699 |
| `a186v2_full_H1.json` | 0775-score | 24,983 | 97.848510742% | 4,880,396 |
| `a189v1_full_H1.json` | 0777-score | 24,960 | 97.941589355% | 4,880,396 |
| `a191v1_full_H1.json` | 0779-score | 24,960 | 97.947692871% | 4,978,700 |
| `a203v1_full_H1.json` | 0781-score | 24,942 | 97.946166992% | 4,978,698 |
| `a202v1_full_H1.json` | 0783-score | 24,566 | 98.004913330% | 4,716,557 |
| `a210v1_full_H1.json` | 0785-score | 24,959 | 97.933959961% | 5,009,243 |
| `a234v1_full_H1.json` | 0787-score | 24,994 | 97.876739502% | 4,729,584 |
| `a240v1_full_H1.json` | 0789-score | 24,960 | 98.007202148% | 4,978,700 |
| `a243v1_full_H1.json` | 0791-score | 24,965 | 98.004150391% | 4,978,700 |
| `a250v1_full_H1.json` | 0793-score | 24,960 | 98.000335693% | 4,978,700 |
| `a253v1_full_H1.json` | 0795-score | 24,946 | 97.955322266% | 4,888,571 |
| `a259v1_full_H1.json` | 0797-score | 24,960 | 98.017883301% | 4,978,700 |
| `a260v1_full_H1.json` | 0799-score | 24,879 | 98.006439209% | 4,978,700 |
| `a269v1_full_H1.json` | 0801-score | 24,958 | 98.017120361% | 4,978,700 |
| `a295v1_full_H1.json` | 0805-score | 24,932 | 97.863006592% | 4,978,700 |
| `a313v1_full_H1.json` | 0807-score | 24,961 | 98.024749756% | 4,976,785 |
| `a330v1_full_H1.json` | 0809-score | 24,947 | 97.972106934% | 4,979,716 |
| `a342v1_full_H1.json` | 0811-score | 23,546 | 98.018646240% | 4,987,188 |
| `a354v1_full_H1.json` | 0813-score | 24,946 | 97.940063477% | 4,589,583 |

Trial values are retained in the dated construction records and original manifests/analysis JSON, including rejected and interrupted families. A362's declared unit counts were [16,32,64,128,256], Up fractions [0,0.1,0.25,0.5,1], Down ridges [1e-6,1e-4,0.01,1,100], and Down strengths [0.25,0.5,0.75,1,1.25]. Only counts 16 with all five fractions and count 32 with fractions 0 and 0.1 completed their logged feature-map reports before cancellation. Do not interpret unrun declarations as evaluated trials.

Limitations remain material. This is a three-digit BOS best attempt with 39 exact failures, not reliable long addition across the ladder. Prefixes, six-digit/mixed-length operands, chat, and repeated additions are not solved. The ordinary-text top-1 gate failed even though KL and CE passed. Noise at sigma 1e-3 substantially degrades arithmetic. Calibration holdbacks were reused across many analyses; restricted-vocabulary and fixed-input component screens are not independent full-model evaluations. Approximation bounds are conditional on recorded inputs and stated factor assumptions. The score returns aggregates and permitted arithmetic failures; held-out text was never inspected for calibration. The historical A314 B-coordinate label error is retained and corrected by A339; corrected coordinates are [467,29,617], reference 113, anchor 547. The run does not establish mathematical infeasibility or a general solution to weight programming.

The recorded resource exception A170 peaked at **12,430,635,008 bytes (12.43 GB)**, exceeding the 12 GB process limit. It produced no checkpoint. This violation is not erased by subsequent compliant jobs. At closeout the obsolete own activation `artifacts/a72v2_word.pt` (950,709,379 bytes; SHA-256 `80dbe000c01ec5860cbd6d591d8d3eacda233472a1fc386d097ce4e6380f0499`) was removed to make room under the 40 GB folder limit; `artifacts/final_user_closeout_cleanup.json` records the deletion. No source, manifest, checkpoint, score, dataset, or ledger was removed during closeout.

**Independent re-test is pending.** Per Dorian's explicit final instruction, `tools/verify.py` and the GSM8K comparison were skipped. The report claims the completed `tools/score.py` results only. The required export consists of `export/run3.zip` (reports, ledger, tools, all project scripts/manifests, analysis/score metadata, best attempt, and downloads except model.safetensors), the separately split original model, a SHA-256/reassembly README, and a package manifest. Packaging performs archive/hash checks only, with no new model execution. No activation artifacts or full source-model checkpoint are placed inside the ZIP.

The following dated material is the preserved experiment journal. References there to work being running, prepared, next, or provisional are historical snapshots, superseded by this final closeout. No background analysis or score remains active.

---

The experiment follows GOAL.md: frozen, unmodified installed Gemma3ForCausalLM with SDPA, float32 weights, tied embedding/output matrix, direct sparse weight replacement only, no gradients or answer fitting. All work is offline within this project on CPU. The embedding/output matrix, final normalization and buffers remained unchanged. The maximum permitted number of changed eligible scalars is 5,016,268.

Host baselines, frozen hashes, meter use, designs, trial values, arithmetic results, text preservation and noise measurements are recorded below. The final summary above identifies the selected leader and closeout status; the following entries preserve the historical sequence and do not claim infeasibility.

Host checkpoint SHA-256: `d205c3d35f26a65bf5be4c7c9c56db479531152b401b80ca4bd447f9e7e00b63`. Installed `modeling_gemma3.py` SHA-256: `d82a589cb918c98139d30f8fdfb8f2664e40427f6908b6e3665f838ec8528294`. Every meter source and relevant input hash is recorded in `tools/frozen.json`. Software: transformers 5.22.2, torch 2.15.0+cpu, tokenizers 0.23.3.

The float32 conversion loads strictly with 237 state keys and tied embeddings. Every digit is a separate token. Four calibration-prefix greedy completions were grammatically fluent; their factual accuracy was not a criterion. This check used 16/300 forward units. Synthetic protocol tests verified exact scoring, the alternate-tokenization fallback, chat termination, corpus cleaning, operand lengths, BOS windows, and strict host reconstruction. Those tests did not execute Gemma outside the meters.

The unedited host quick baselines, recorded in `scores/host_baseline_quick.json` and the hash-chained ledger, are:

| Rung | Exact correct / 2,000 | Exact accuracy | Correct digit prefix, ignoring terminator |
|---|---:|---:|---:|
| H1 | 3 | 0.15% | 702 (35.10%) |
| H2 | 211 | 10.55% | 242 (12.10%) |
| H3 | 4 | 0.20% | 17 (0.85%) |
| H4 | 88 | 4.40% | 121 (6.05%) |
| H5 | 0 | 0.00% | 0 (0.00%) |

Host self-comparison on 256 held-out windows / 16,384 targets has 100% top-1 agreement, zero KL and zero cross-entropy change. Its cross-entropy is 5.0004215402 nats. All results are finite. The digit-prefix metric explicitly ignores the following token; it does not claim that the continuation terminates correctly.

After the initial captures, meter usage was 20/300 forward units, 5/100 quick units, 0/40 full units. Those captures cover independent valid formats, malformed controls and calibration text. Continuation digits in the format probes are random and independent of the operands; no answer targets are used in detector construction. No ladder rung is banked. The ledger is authoritative for subsequent runs and current remaining budgets.

Initial component analyses have not produced an edited arithmetic checkpoint. A native joint phase classifier, a within-format covariance construction, and exact token/role mean interpolation all failed to establish a complete reliable format gate across later answer positions and malformed controls. Their respective trial lists were hard-negative masses `[1,4,16,64,256]`, positive-variation penalties `[1,10,100,1000,10000]`, and covariance penalties `[0.0001,0.001,0.01,0.1,1]`. These are three detector constructions, not three completed arithmetic circuits.

A fixed-unembedding parabolic digit readout passes a static full-vocabulary test at every integer digit and at ±0.2 displacement. The five class-bias values were `[16,32,64,128,256]`; 32 gives the largest minimum margin, 0.7546389247. This component test does not execute Gemma and does not establish end-to-end addition.

An explicit token-coordinate alternative is being developed. Static embedding solves found an off-format direction that is zero on the fourteen allowed digit/operator/BOS/newline embeddings and positive on every other vocabulary embedding. The five attention-metric masses were `[0,0.01,0.1,1,10]`; the minimum other-token value remains above 0.21198 in every trial. Protecting those directions through selected native outputs may permit exact token checks and relative-position arithmetic. Its ordinary-text cost remains to be measured.

The first protected-coordinate component, B0, is now measured and fails preservation. It changes 1,800,448 scalars: 653,312 in layer-0 attention output projection and 1,147,136 in layer-0 MLP down projection. These are circuit accommodation; there is no text repair in B0. Maximum absolute model weight remains 1656.0. Quick H1 is 8/2000 exact (757 correct digit prefixes), text agreement 89.80102539%, KL 0.0567697189, CE increase 0.0367549396. No full/noise run was spent on this failing component.

Its intended coordinate identities do hold in the metered syntax capture: maximum token-coordinate error 5.72e-7, protected post-MLP frame error 3.61e-7. The failure is ordinary-text distortion. A first non-iterative, unlabeled calibration repair (B0R) also fails: text agreement 89.46533203%, KL 0.0543131455, CE increase 0.0354227140; H1 exact 8/2000, 760 digit prefixes. It uses 1024 donor columns and the five ridge values `[0.001,0.0001,0.00001,0.000001,0.0000001]`, selecting 0.001 by separate calibration reconstruction. It changes 1,801,472 scalars. No full or noise run was spent on it.

B2 changes the projection geometry to account for post-branch normalization gains. For each branch it computes five direct projections with raw/output metric regularizers `[0.0001,0.001,0.01,0.1,1]`, then selects by the normalized branch disturbance on recorded calibration activations. Selected values are 0.01 for attention and 0.001 for the MLP. No derivatives, gradients or iterative fitting are used. Its 1,800,446-scalar checkpoint fails preservation: text agreement 84.31396484%, KL 0.1233712040, CE increase 0.0428250536; H1 7/2000 exact and 669 digit prefixes. A further B3 static construction removes strong-column constraints and uses 1,961,981 edits, but has not been metered.

A1 takes a different route: one GLU unit writes a large physical residual carrier when a native syntax prefilter fires. Its 44,863 edits comprise layer-0 down projection 43,582, gate/up rows 640 each, and one post-FF gain. Maximum absolute weight is 39,527.47265625. Quick held-out text agreement is 99.94506836%, KL 0.0000113663, CE change -0.0000258752: all quick preservation gates pass. H1 is 0/2000, as expected for a marker without arithmetic. This is a promising component, not a passed rung. It does not reject all malformed syntax; downstream token/position checks are still needed. Sources and exact repair/accommodation counts are in experiments/a1_marker/delta.manifest.json.

A1 declares carrier scales `[1e4,1e5,1e6,1e7,2e7]` and raw output gains `[100,1000,10000,100000,1000000]`; only 1e6 and 1000 have been tried. Its single displaced-feature repair uses 64 donors and fixed ridge 0.001. The gate combines two format detectors with an algebraically determined ratio and threshold; all five original invariance penalties are disclosed in artifacts/a_equals_gate_v1.json. Current usage: 31/300 forward units, 9/100 quick units, 0/40 full units.

A2 adds four layer-2 relative readers and uses the second declared marker carrier value, 1e4. It changes 719,561 scalars. Its first tested routing margin is 32 from `[8,16,32,64,128]`, and code gain is 0.25 from `[0.05,0.1,0.25,0.5,1]`. It FAILS: quick text agreement 81.84814453%, KL 0.1595064710, CE increase 0.0207455870; H1 remains zero. Native per-token readouts alone round all digits correctly on two syntax probes (worst error 0.1683), but the transported records do not: the first reader has hundreds of rounding failures. This is a failed transport component, not an adder.

A2 ablations use calibration text, not held-out data: retaining only the query/key edit gives 91.80% agreement on 256 last-token predictions; retaining only value/output edits gives 90.23%. These small diagnostic samples are not official preservation scores. Both components need revision. The existing output-feature repair and post-norm-only importance ranking did not sufficiently protect native normalization behavior. A3, an unmetered conditional register-amplification component, has been constructed to preserve value ratios; it does not fix A2's incorrect transported values.

A1 clean-marker repair also fails agreement: 96.49658203%, KL 0.0069574623, CE increase 0.0016502828. A single output-row solve tried ridges `[1e-6,1e-5,1e-4,0.001,0.01]`; 0.01 gives calibration validation residual-coordinate RMS 0.3751 versus original 0.5157. Protecting that coordinate through native branches is not cheap enough. This branch is set aside.

A2v2 replaces three native value coordinates by per-token C:D:kind functions in their existing span and applies the exact inverse transform to each output head. Its native bilinear map error after float32 rounding is at most 6.03e-8. The per-token ratio construction tries five penalties `[1,10,100,1000,10000]`, selecting 100; maximum independent digit error is 0.3433 with no wrong rounding. All three transported B-operand digits round correctly on the 512-case mixed syntax probe, including the residual memory fields. Nevertheless quick text fails: 90.43579102% agreement, KL 0.0732013005, CE increase 0.0339071135. Code gain 4 and routing margin 32 were used.

A2v3 adds a sparse rank-one projection that removes the native query response at the marked mean. Support counts `[16,32,64,128,256]` select 16. Rotary pairs `[28,41,97]` have a positive isolated target peak, permitting zero key/value codes at BOS. It changes 737,853 scalars, maximum 1000. Its new declared routing margins are `[2,4,8,16,32]` (4 tried), query raw scales `[0.01,0.03,0.1,0.3,1]` (0.1 tried), and code gains `[0.25,0.5,1,2,4]` (0.5 tried). Quick text reaches 98.02246094%, KL 0.0020925165, CE change -0.0005841623: still below the agreement gate. Transport has zero wrong rounding among all three B digits in the 512-case probe; the subsequent unmodified MLP adds a large constant offset to the fields. A2v4 is being metered with the original 1e6 marker, query scale 0.01, and fields orthogonal to that MLP mean. It has no arithmetic logic yet.

A weight-only final-output test found correct full-vocabulary onehot readouts for all ten digits and newline within weak coordinate subsets. Five subset sizes were `[64,96,128,192,256]`: 64 fails; 96 passes with minimum margin 0.1221; 128 passes with margin 1.2592 and uses coordinates containing 2.664% of recorded native final-MLP raw energy. Larger subsets give more margin but greater normalization cost. These are isolated output-vector checks, not executed addition. No full/noise or final verification run has been used.

A2v4 is complete and still fails text agreement: 98.04077148%, KL 0.001972633, CE change -0.000373613. Its original 512-case syntax probe transports and retains all three B digits correctly, but a separate syntax stress probe produces five units-digit rounding errors after the native MLP (maximum error 0.6506). It remains a reader component, not an adder. Two non-iterative text-repair constructions failed calibration reconstruction and were not metered: a constrained repair tried ridges `[.001,.0001,1e-5,1e-6,1e-7]`; a native-output restoration tried cancellation fractions `[0,.25,.5,.75,1]` with fixed ridge .001.

Calibration ablations identify removal of the native rotary frequency channels as the largest early disturbance; query mean nulling alone has much smaller effect. A one-frequency reader with a directly computed current-token key normalization function is the next investigation. Ledger use is now 55/300 forward, 14/100 quick, and 0/40 full units. No rung or noise result is claimed.

A2v5 uses a single rotary pair and a computed current-token key-RMS equalizer. Five frequency candidates were `[12,16,20,24,28]`; five text masses were `[0,.0001,.001,.01,.1]`. Pair 12 with mass .01 separates all seven source positions on the syntax stress probe. Margin 8 was used from `[2,4,8,16,32]`. Quick text agreement improves to 98.46801758%, KL 0.0011961857, CE change -0.0006957343, but still narrowly misses the agreement requirement. It has no addition logic.

The next full-input token-ratio construction uses penalties `[1,10,100,1000,10000]`, selecting 100. Independent per-token maximum digit error is .11134 on random validation and .10176 on stress. A2v6 installs those functions by replacing three low-energy native value directions and uses output-metric weights `[0,1,10,100,1000]`, selecting 10. Its 887,229-scalar checkpoint is being metered; no pass is claimed.

A2v6 completes with 98.20556641% text agreement, KL .0014636083, CE change -.0005521905. Its stress probe has no digit-rounding errors, but it fails preservation. A single constrained attention-output repair tries ridges `[.001,.0001,1e-5,1e-6,1e-7]`, selecting .001; agreement improves only to 98.29101563%, KL .0014578300, CE change -.0013631649. It is still a reader, not an adder.

A separate uniform-attention design with per-current-token/place value codes fails static transport tests. Five covariance penalties `[1,10,100,1000,10000]` and recorded layers `[2,4,6,8,10]` all leave many source-digit errors. A position-coded key design also fails: five phase values `[.35,.45,.55,.65,.75]` and text masses `[0,.0001,.001,.01,.1]` never establish a positive routing gap. Neither produced a metered checkpoint.

Two physical-register amplifier constructions were built but not metered. Their static text disturbance remains large (.20168 and .17766 branch RMS). The second selects native units by conditional covariance with raw-importance masses `[0,.01,.1,1,10]` and uses one 512-donor repair at fixed ridge .001; it changes 1,258,785 scalars. The next architecture reduces the required amplifier units from thirteen to five by storing constant-amplitude scalar digits and postponing rounding to carries and the final output. Current-token absolute-value fits use `[1,10,100,1000,10000]`; selected 1000 gives maximum individual digit error .18792 with a shared constant. No addition or noise pass is implied.

A2v7 is now being metered: five temporary fields, code gain 1 from `[.25,.5,1,2,4]`, original rotary pair 12, key equalizer mass .001, margin 8, and 887,229 changed scalars. Its five field covariance penalties are `[1e2,1e3,1e4,1e5,1e6]`; output metric weights are `[0,1,10,100,1000]`. No complete adder or rung has passed.

The scalar B amplifier (A3v4) uses five GLUs, four physical registers, and 1,231,272 changed scalars. Its stress probe retains all three digits (maximum error .29793 after layer3), but quick text agreement is98.07739258%, KL.00184308, CE-.00033957. The A-operand reader uses actual rotary pair9 with key text mass.001, query scale.03 and margin8. Its scalar values are computed from all900 standalone three-digit prefixes using only current-token labels. The five covariance penalties are[1,10,100,1000,10000], selecting10000; maximum source-digit error .120164. This exhausts that source prefix domain and is not an independent generalization claim.

A4v2 uses centered digit codes, gain2, and a hard PCA background frame with tail sizes[16,32,64,96,128], selecting96. All A digits round correctly in the512-case stress probe, maximum field error.110145. The candidate changes2,073,639 scalars, but quick text agreement falls to96.71630859%, KL.0043523849, CE+.0009773949. It still has no carry or output logic. A5v1 adds four GLUs and three physical A-digit registers;2,414,234scalars change. It is being metered. Its one-shot native repair uses512donors, ridge.001 and raw-importance masses[0,.01,.1,1,10]; selected0. Gateconstant ridges[1e-6,1e-4,.01,1,100] select1. Data scales are[1000,2000,4000,8000,16000], using4000; output gains[1e4,3e4,1e5,3e5,1e6], using1e6; anchor-up scales[1e3,1e4,1e5,3e5,1e6], using1e5.

A new hand-derived carry switch uses two opposite GELU gates, a shared raw normalization anchor, and linear residual updates. On3420synthetic scalar cases it yields maximum carry-bit error1.42e-6 under an assumed native raw RMS bound100. Sharpness values[2,4,8,16,32] select8 for the next construction. This is not an executed Gemma addition or model-noise result. No rung is banked;77forward and20quick units have completed, with0full or final verification runs.

The hand-written carry chain now works as a component. Eight GLUs atlayer4 formthethreecolumn sumsandapplyunitscarry; twoGLUseach atlayers5and6applytensandhundredscarries. Thethreshold9.5, subtraction10andincrement1areliteraldecimalarithmetic. No answer targets are used. Allfourfinalregistersroundcorrectly on1024mixed/stressprobeproblems; maximumstresserrors are[.4084375,.2378253,.1846819,.0002332]. This is not an H1 result: answer copying, phase selection and digit emission remain unbuilt. A8v1 changes2,698,357scalars; quicktextagreement96.50878906%,KL.00539322968,CE-.00003721862, so it still fails preservation. It scores0exact because it has no output circuit.

Carry format-off covariance masses are[1e2,1e4,1e6,1e8,1e10], selecting1e10 for allthree stages. Rawanchor10000, sharpness8from[2,4,8,16,32], columnscale50000from[1000,10000,50000,100000,500000]. Nativeunitraw-importance masses[0,.01,.1,1,10] choose0,.1,0 respectively; eachuses128donorsandfixedridge.001. Thelaterfixed(-10,+1)updatesincludeananalyticRMS correction. A8v2 subsequentlybooststheleadingbitto1e6from[5e4,1e5,2e5,5e5,1e6], changingonlytwoprogramdown-projectionentriesrelativetoA8v1andcorrectingthetwo-vectornormexactly. It has been captured but not quick-scored.

A uniform-attention answer clock was considered and set aside before building: its best single-head replacement has5.923normalizednativebranchRMS distortion on recordedcalibrationstates. Existingformatstatesinsteadlinearlydistinguish allfive answerpositionsintheprobe; fivewithin-positioncovariancemasses[1,10,100,1000,10000]werecomputed. These are position labels, nevercorrectanswers. A copy reader preserving allfournativeattentionheads is now being constructed.

Static sparseparabolic digit readouts work at allintegerdigitsand±.45 withpool sizes[64,96,128,192,256]andbiases[16,32,64,128,256]. Their bestminimumfull-vocabularymargins are[.01860,.05276,.05573,.07719,.08750]. Those smallmargins motivate testing a onehot outputalternative, which has not yet been implanted. Still no full/noise or fresh verification run and no banked rung.


The first complete addition checkpoint, `experiments/a12v2_output/delta.pt`, scores **1999/2000 exact H1 additions (99.95%)**, including the newline. This is a large increase from the host's3/2000, but it is **not a rung pass**. Held-out quick text agreement is93.548583984375%, mean KL.03110053858nats and CE increase.01916302804nats; agreement and KL fail the gates. All values are finite. It changes4,222,093eligible scalars, maximumabsoluteweight1e6. Full/noise and final verification have not run. Exact per-tensor counts and circuit/text-accommodation categories are in its manifest and the ledger.

The completed circuit reads the operands, performs three decimal carries, copies four sum registers from the equals position to later positions, selects a digit by current position and leading-bit status, then emits one of ten digit directions or newline through the protected unembedding. The arithmetic is literal hand-derived logic. All fitted format functions use token/position labels and independent random continuation digits; no solve uses a correct sum as its target.

Answer-copy A9v2 failed because retained native values overwhelmed its physical code fields. A9v3 projects those native values away from the marked source-format subspace, retaining the explicit code rows. PCA ranks[1,2,4,8,16] select8. All four copied registers round correctly at5120probe positions. Its query/quieting masses are[0,.01,.1,1,10] and[1e2,1e4,1e6,1e8,1e10]; rotary candidates[87,70,90,72,85] select87, queryscale.03 from[.01,.03,.1,.3,1], keyscale1 from[.1,1,10,100,1000], routingmargin32 from[8,16,32,64,128], fieldgamma1e4 from[100,1000,1e4,1e5,1e6], codegain.01 from[.01,.1,1,10,100].

A10v1 failed static phase gating and built no checkpoint. A10v2 uses8GLUs for four copied digits, a homogeneous current-position pair, a leading-bit product and a continuation flag. Position covariance masses[1,10,100,1000,10000] select10000. Gatefraction.3 from[.01,.03,.1,.3,.5], sharp1024 from[16,64,256,1024,4096], phasegain1e4 from[1e3,1e4,1e5,1e6,1e7], datagain16000 from[1000,2000,4000,8000,16000]. It changes3,638,318scalars and its component probes have zero digit/slot rounding errors. A11v1 uses11GLUs to select the next digit or newline. Rampwidth.1 from[.025,.05,.1,.2,.4], sharp1000 from[100,1000,10000,1e5,1e6], offmass100 from[1e2,1e4,1e6,1e8,1e10]. Native later layers disturbed its small output; A11v2 boosts only its three program-output rows by10 from[1,2,4,10,20]. This restores zero digit/newline rounding errors at5120 recorded final-layer positions, with3,747,725changedscalars.

A12v2 adds30GLUs atlayer17:20for onehot digit/newline output and10to cancel strong scratch coordinates. It uses192weak output coordinates, whose static full-vocabulary class margin is5.26177. Outputgamma1e6, rawgain1e5, rampwidth.05 from[.01,.025,.05,.1,.2], sharp256 from[16,64,256,1024,4096], commonG4 from[4,16,64,256,1024], offmass100 from[1e2,1e4,1e6,1e8,1e10]. A10/A11/A12 displaced native units use conditional-importance masses[0,.01,.1,1,10], selecting0,0,.01, and128,128,64donors at fixedridge.001 respectively. Their exact scripts and manifests preserve all selected values and source hashes.

The next candidate is an unlabeled final-layer text repair. It analytically chooses a scalar on each host final-state ray, then solves a constrained linear down-projection correction from actual edited native MLP features. Format-activation directions are constrained to zero correction; these constraints do not use answer targets. Paired calibration captures cover65536tokens. The fit uses49152training and16384validationtokens, donorcounttrials[1024,1280,1536,1792,1824], formatPCA ranks[8,16,32,64,128], and five independent closed-formridges[.001,.0001,1e-5,1e-6,1e-7]. No repair result is claimed yet.


The first final-layer text repair (A13v1) fails: addition remains1999/2000, but text agreement falls to79.27856445%, KL.2290496877 and CE increase.1942620805. It changes5,010,767scalars. Calibration analysis identifies a bad normalization assumption: the weakest output-gain coordinate supplies about62%of native raw RMS energy. Raising that gain while inverse-scaling its down-projection row disturbed the normalization anchor. The exact host-ray targets consequently require highly variable scaling and do not admit a useful linear reconstruction. A separate anchor-preserving construction (A13v2) leaves all gains unchanged, matches the other coordinates and uses that weak coordinate to supply the remaining raw normalization energy. Its fiveanchorcounts[1,2,4,8,16] select1; fiveclosed-formridges[.001,.0001,1e-5,1e-6,1e-7] select.001, and arithmetic-constraint ranks[0,8,16,32,64] select0 because the measured raw correction fraction is alreadybelow1e-5. Its official text result is pending; no pass is claimed.


A13v2 completes at1999/2000 exact additions, textagreement93.92089844%, KL.03003850332 and CEincrease.01681685954. It is a small improvement overA12, butfailsagreementandKL. A14nativefrequencyrestorationwasbuiltbutnotmetered: enforcingmaxformatnormalizedchange.001leavesnegligiblenative-channelrecovery. Itsfivecovariancepenaltiesare[1,100,1e4,1e6,1e8]. A15localMLPtextrepairhasusefulidealnormalizationtargetsbutpoorlinearreconstruction. Itsfirstlayer2fituses768donors,fiveridges[.001,.0001,1e-5,1e-6,1e-7]andformatnullranks[0,16,32,64,128]; nofitmeetsformatstateRMSbound.03,sono checkpointwasbuilt. ThebestunconstrainedvalidationRMSimprovesonly.27696to.26421. A16examinesfivelayer3rotarypairs[9,12,3,6,4]usingnativecovariancesandliteralroutingmargin8. Itselectstheexistingpair9again;itsunmeteredcheckpointdoesnotconstituteanewresult. Theseanalysesuseonlyweights,recordedformatstatesandunlabeledcalibrationtext. A diagnostic quickscoreofthepre-outputA11v2checkpointisrunningtoisolatethefinaloutputblock'stextcost.


Atapproximately01:17CET, thebestcompletequickagreementis93.95751953% (A17), with1999/2000exactadditions, KL.03105166986 andCEincrease.01930087868. No rungpasses. A17projectsreference-headnativequerydirectionsatformatstates, reducingrequiredcopygain1279.565to476.861; ranks[4,8,16,32,64]select4. It changes4,384,653scalars. Thepre-outputA11v2checkpointmeasures94.12841797%agreement,showingmosttextlossoccursbeforethefinaldigit-emitter. Separate-headandjointoutput-sourceprojections(A18v1/v3)failtoshowstrongstaticpreservationgains; reference-head-onlyA18v2fails633/2560stressoutputpositionsandisnotquick-scored. A19usesalinearinverseofrecordedvaluesandattendedvaluestorecoveractualattentionmixtureswith5.026e-8relativeRMSerror. Aprojectioncomputedwiththosemixturesreducesnativebranchvalidationerroronly.66517to.64590andisunmetered. A20usesaconstrainedlinearreconstructionplusonesharedrank251factorizationofthecopyblock'snativevalue/outputmaps,keeping5programchannelspreserved. Fiveindependentridges[.001,.0001,1e-5,1e-6,1e-7]select1e-5; calibrationbranchRMS1.15437->1.06807andnativearithmeticsourcebound.04436. Itsactualmeterchecksareinprogress. Noneoftheseanalysesusesanswerregression,gradients,orheld-outtext.


At 02:02 CET, no rung has passed. A20 finished at1989/2000 with93.9880% agreement, KL.0401476 andCE+.022415, so it fails both tasks. The new frame copier A21 has zero mixed-probe digit errors but14 stress digit-rounding errors; its amplifier has no safe leading-bit gate and builds no checkpoint. The fresh source/continuation-marker stage A23 preserves allfour source digits on1024probes; its marker is~2.5million onformat andbelow22oncalibration. A24's first copier stillsuffers native-value contamination, so it is not a complete adder. A25v2 restores the native layer2 key weights and uses32programmedquerychannels; quick addition is1992/2000, buttextagreement91.34521484%,KL.04952662,CE+.02197152. It changes4,443,662scalars,max1e6,andisrejected. Query and copier revisions remain under development. All these results are provisional; no full/noise/fresh verification run has been used.


## Progress at 02:16 CET, 10 November — not final

A25v4 complete quick: H1 1999/2000, text agreement .94097900390625, KL .02981531531555, CE increase .01912297055561. All 2560 stress output positions correct, minimum top-1 margin >5; artifacts/a25v4_complete_diagnosis.json. All three captures exist. Best retained complete candidate by preservation with CE within bound remains A25v4, still far below 98.5% agreement, no rung pass.

A24v2 copier stress SUCCESS: zero wrong rounded registers at all 2560 positions. C minimum by position 8..12: [2525.20,2112.13,2409.63,2718.44,3072.23], maximum digit errors [.421705,.249723,.203941,.021985]. Quieting native V on all marked positions fixes v1 contamination. Only stress capture exists; need syntax_a and calib256 before deriving a NEW phase clock and amplifier. Old A24v1 phase artifact is stale.

A26v1 final energy repair built and quick scored: 4,465,370 edits, max1e6; H1 1999/2000; text .9417724609375, KL .02994580727375, CE increase .02365220200813. Agreement modestly improves but CE now fails. Four anchors [439,54,4,328], scales [1.02187742,2.69783573,3.49375093,4.35919110], ridge1e-6; anchor counts [1,2,4,8,16], ridges [1e-6,1e-4,.01,1,100]. Local final-branch RMS actually worsens 9.01867->10.61470 despite energy RMS .05558->.03237; repairing only norm energy does not recover displaced native features. Script immutable after meter. Retain A25v4 as next base.

CURRENT: compile_a27_query_energy_repair_v1 builds from A25v4, layer2. Restores native query numerator (undoes earlier average-gain accommodation) while keeping additive program-query signal. Fits missing squared raw-query norm on unlabeled text using shared ordinary anchor row scales, inversely adjusts q_norm to preserve their numerator. Anchor counts [1,4,16,32,64], ridges [1e-6,1e-4,.01,1,100]. Requires measured static native-key routing margin >=8 on all format probes. No answer labels, model replay, or iterative fitting. Not metered yet. Check artifacts/a27v1_build.log.

Cleanup artifacts/cleanup_20261110_0215.json: removed 2.1227GiB own failed/superseded A21 all3, A20 calib256, A18v2 stress captures; hashes, scripts, manifests and results retained. Folder below40GB. Ledger expected143 forward/32 quick/0 full; authoritative ledger controls.


### Conditional receiver clearing and the A29 complete candidate

A29 clears a five-dimensional receiver subspace only at arithmetic output positions, then copies four computed digits without the old native-value source projection. Clearing uses 18 literal GLUs in layer6, five physical writer coordinates, and a separate continuation marker. All 1,024 format probes preserve the source digits; all 5,120 transported digit vectors round correctly. The phase amplifier and output selector preserve every tested digit and phase. The complete candidate uses 4,427,840 changed scalars, max absolute weight1e6. Its manifest records the per-tensor circuit/repair split.

The official quick result, `scores/a29v1_quick.json`, is **2000/2000 exact H1 additions**, with 93.89038086% text agreement, KL **.31434572196**, and CE increase **.30124958745**. All values are finite. It fails preservation and is not a ladder pass. A separate stress forward has zero errors over2,560 output positions and a minimum digit top1 margin5.04875. The highKL warrants a fresh calibration audit of programme activation; occasional false gate activation is a hypothesis, not a finding. A27v2 remains the best-preserving complete candidate.

A29 trial declarations: raw-reference covariance masses[1,10,100,1000,10000] (10000 selected); receiver PCA tails[16,32,64,96,128] (128); source quiet-function masses[1e2,1e4,1e6,1e8,1e10]; query projection ranks[1,2,4,8,16] (1), raw query scale.03, key scale1, routing margin32; copy gains[1,4,16,64,256] (64); phase covariance masses[1,10,100,1000,10000] (10), phase up gain100. The amplifier, selector and emitter retain their previously declared finite trial grids; each exact selection and all trial records are in the stage manifests. Native repair uses128 donors in the clearing/amplifier/selector and64 at output, ridge.001, raw importance masses[0,.01,.1,1,10].

At this point 159/300 forward units and37/100 quick units have been used, plus one running forward audit; no full, noise, or fresh verification result exists. A32 native-unit relocation and A33 composition with the improved native-key readers are prepared but have not been built or measured. Work continues; no blocking or infeasibility claim is made.


### A34 removes the observed rare gate activation

A fresh512-window calibration audit confirms that A29's clearing units stay inactive on text, while the leading-bit amplifier activates on3tokens, selector on5, and final emitter on28. A34 strengthens their existing format-off projections, with multiplier trials[1,2,4,8,16]. Selected values are2 atlayer7 and4 atlayers8/17. A common analytical gate rescale2.188345207 atlayer7 restores a32-point saturation margin. All recorded format gate signs are preserved; leading-ratio drift is at most.00225237, selector common-factor drift.00237641, and final common-factor drift.002034. This uses calibration text and current-position labels only.

`experiments/a34v1_gate_margin/delta.pt` still changes4,427,840scalars. `scores/a34v1_quick.json` reports **2000/2000** exact additions, **94.72045898%** agreement, KL **.01180193560**, CE increase **.00224773170**, all finite. Stress output remains correct at all2,560positions, minimum top1 margin5.04325. This is the current best complete candidate but remains **below the agreement gate**. Meter use is162/300forward,38/100quick,0/40full. No rung is banked or independently verified.


### Further preservation checks and the native-value reader

A32 relocates programme GLUs using fixed-unembedding covariance and fits one native donor repair per layer. With all four layers changed, addition remains2000/2000 but agreement worsens to94.61059570%, KL.01216591531 and CE increase.00369485249. Final-layer-only relocation keeps94.72045898% agreement, offering no gain. The programmed gate/up/down values are preserved exactly through permutation. These candidates are retained as unsuccessful repairs.

A35 evaluates copy gains[1,4,16,64,256] by linear arithmetic on recorded attended values. Reducing64 to16 only changes local native V/O branch error from.51634 to.49644, while shrinking arithmetic margins;64 is retained. A37 instead recomputes the retained251 native value directions using actual native mixtures weighted by raw attention RMS. Five raw-energy penalties[.0001,.001,.01,.1,1] select1. Local branch error improves to.35484, but its official agreement remains94.72045898%, with1999/2000 additions. Neither result advances a rung.

A33 composes A34 with the previously derived native-key queries and query-energy repairs for both operand readers. It changes4,893,698scalars, max1e6. `scores/a33v1_quick.json` reports2000/2000 additions,94.873046875% agreement, KL.01076859577 and CE increase.00098556126, all finite. It is the best complete candidate so far, still below the agreement threshold.

A36 explores reading digits through entirely native attention value/output matrices. Its fits use individual current-token/head contributions, never whole-problem or sum targets. The second static design uses6,144 source-token rows, free per-head offsets whose sum is zero, and a fresh2,048-source-token validation probe. Covariance masses[1,10,100,1000,10000] select10000; ridge1e-6. Ideal routed reading has zero wrong digits; worst fresh units-digit error.34923. A five-GLU conditional frame clearer and the native V/O reader have now been built (645,000 total edits) and actual forward checks are running. No full adder or preservation score is claimed for this new reader.


A36 native B-reader execution succeeds on all1,536 recorded examples. Fresh maximum digit errors are[.35949,.12008,.07854], with no wrong rounding. Its compatible B amplifier also preserves all512stress B-register vectors. The first complete composition, A38, fails because the later A reader's old receiving frame no longer cancels the new residual background. It gets13/2000additions; text agreement95.00122070%, KL.01067973042, CE increase.00124577345. This is a broken diagnostic composition, not a better complete adder. A33 remains the complete reference.

A40 fits sparse native marker predictors on calibration text, trying predictor counts[4,8,16,32,64] with fixed ridge1e-6. Count64 lowers native query score-error proxies modestly at both reader layers while preserving all recorded routing margins above8. The corresponding full candidate is being metered. Meter use before this test is173/300forward,43/100quick,0/40full. No rung has passed and no noise or independent verification result exists.


A40v1 completes with4,975,618edited scalars (within the5,016,268limit),2000/2000quick additions and94.90356445%text agreement. KL is.01074963334, CE increase.00106227066; all finite. Stress has zero wrong predictions over2,560positions. It gains five matching text tokens versus A33 and is now the best complete candidate by agreement, but still fails preservation. No full run is warranted yet.


A41 audits early programme units at layers0,2,3,4,5 on512fresh calibration windows. Every recorded programme feature is exactly zero. Combined with A34's late-stage audit, this gives no observed false activation on those calibration probes. It does not prove the absence of rare activation on unseen text. The large capture was removed after hashing to stay below40GB; its request, analysis, aggregates and ledger entry remain.

A42 diagnoses A33 by restoring separate groups to their frozen host weights. Restoring attention2/3/7 gives96.88720703%agreement, KL.004556384 and CE change-.001786617. Restoring only finalMLP17 gives95.42846680%, KL.009010978 and CE change+.000422852. Restoring MLPs6/7/8 gives95.13549805%, KL.010111990 and CE change+.003495649. Each breaks arithmetic and is a diagnostic ablation. Their effects are conditional on the other edits and should not be added as independent costs.

A39 tests native A-operand value/output reading using all900source prefixes. Labels describe individual current source digits only. Five residual-background covariance masses are[0,1,100,10000,1000000], with within-token covariance mass10000 and ridge1e-6. The plus-reference version has low ideal tens/hundreds error but a worst units error around.43; quieting its receiver background worsens source precision. A second version reads fixedBOS in the fourth head, reducing tens/hundreds errors further, but leaves the units limitation and receiver-background tradeoff. Neither is a complete circuit or an official score.

A44 tests removing the intermediate amplifier. The unchanged nativeMLP7 corrupts copied digits by as much as5.11; a clock tied to the copied constant still has maximum phase error.684 after trying covariance masses[1,10,100,1000,10000] at ridge1e-6. No checkpoint is built for this shortcut.

A43 uses a different copier: all four heads read the equals-position registers through unchanged nativeV/O and post-attention gain. Its readout is derived from literal register-coordinate identities, with nuisance masses[1,10,100,1000,10000] and ridge1e-6. Only the existing layer6 frame-clearing programme rows/columns are changed; native repair, gates and gains remain identical. On actual new source captures, an ideal-routing diagnostic adds at most.00763 register error and has no rounded-register mismatches. This is still a component calculation, not executed attention. Query projections try ranks[1,2,4,8,16], rawquery scale.03, keyscale1 and directly calculated margin32. Rank1 is selected by calibration paired-score error.00342108, with programme gain254.1464. The native copier is now being metered, after which its phase clock and amplifier must be checked. No new complete score is claimed yet.


### A43 native copier and A45 operand work, 04:42 CET — provisional

A43v1 (`experiments/a43v1_complete_native_copy/delta.pt`) improves text agreement by 0.98877 percentage points over A40, while retaining 2000/2000 exact H1 quick additions. Its agreement is .95892333984375, KL .007389942123613364, CE increase .00028214731754250266, all finite. It changes 4,804,867 scalars, max absolute value 1e6. The independent stress capture has zero errors over 2,560 output positions. Exact per-tensor counts and repair/circuit categories are in its manifest and `scores/a43v1_quick.json`. This is still not a rung pass; no full/noise/fresh verification result exists. Usage after this score: 188/300 forward, 48/100 quick, 0/40 full.

The successful copier keeps layer7 V, O and post-attention gains exactly native. All four heads select the equals-position source. The new readout satisfies literal identities on the existing physical arithmetic registers; it is not fitted to sums. Layer6 changes only 15 existing clearing Up rows and Down columns. All 5,120 captured copy vectors round to the source registers correctly; maximum added stress digit error is .007621. Phase functions use current position labels only. The amplifier keeps the old physical register interface, uses 512 native repair donors and off-gate multiplier4, and the old selector/emitter remain intact. Native frame nuisance masses [1,10,100,1000,10000] choose10000; query ranks [1,2,4,8,16] choose1, raw query .03, key1, margin32, analytically computed gain254.1464. Readout coordinate scale64 is chosen from [1,4,16,64,256]. Amplifier donor choices [128,256,512,768,1024] choose512; off multipliers [1,2,4,8,16] choose4. The remaining inherited trial lists and exact phase/amplifier selections are recorded in the immutable source manifests.

A44's attempted direct selector without a layer7 amplifier fails static checks: native MLP7 corrupts digit fields by up to5.11, and the best phase fit has error.6837 with33 wrong phases. It has no checkpoint or meter result.

A45 addresses the native A-reader value bottleneck with a rank-two correction, keeping O and post-attention gains native. Current-token constant/digit functions are computed on all900 three-digit prefixes; no second operand or sum labels are fitted. Five source penalties [1,10,100,1000,10000] choose10000, fixed ridge1e-6. Maximum independent current-digit error is.113544. The value update makes those two linear functions exactly available in the native V span (float32 identity error8.5e-10), with gain1 from [1,2,4,16,64], native-output metric regularizer.001 and solve ridge1e-6. Readout background penalties [0,1,100,10000,1e6] choose10000, within-token mass10000, ridge1e-6. Ideal source routing plus actual receiver background gives zero rounded A-digit errors in1,536 format examples, maximum fresh error.17741. This is only static evidence until the official reader captures complete.

A45 native-key query construction uses support counts [16,32,64,128,192], ridges [1e-6,1e-4,.01,1,100], raw alpha [ .01,.1,1,10,100 ] and margin8 from [4,8,16,32,64]. It selects64 channels, ridge.01, alpha100, gain7.7771543. The full quiet marker nulls native format-mean queries. One-shot query-energy repair is computed from this original query candidate, never by repeated residual fitting. Version1 tests anchor counts [1,4,16,32,64]; version2 tests [32,64,96,128,192], each with the five ridges above. Version2 selects192 anchors and ridge.01, reducing calibration covariance score error .0289053 to .00514386, with all routing margins above8.05. Four official reader captures are in progress. No complete A45 arithmetic or text result is claimed.


A45's official complete quick result is 1978/2000 additions, text agreement .96514892578125, KL .006503187271844592 and CE increase .001818057652632099. It changes4,073,868 scalars, max1e6. This improves agreement but fails the arithmetic and preservation gates; A43 remains the retained reference with perfect quick arithmetic. All current A/B and sum registers round correctly in the stress capture, but two position11 output errors coincide with phase values approaching the selector ramp. A46 recomputes only current-position functions on actual A45 states and replaces three phase Up rows, leaving native-text repair unchanged. It is in progress; no result claimed. A45's actual1,536-example reader and fresh512 early-gate checks passed. Exact statistics, source hashes, trial values and per-tensor counts are in the manifests and artifacts. Usage after A45:193/300forward,49/100quick,0/40full.


A46v1 (`experiments/a46v1_phase_rebased/delta.pt`) restores2000/2000 quick H1 additions and zero stress output errors by changing only three layer7 Up rows. Text aggregates are exactly identical to A45: agreement .96514892578125, KL .006503187271844592, CE increase .001818057652632099. It retains4,073,868 changed scalars, max1e6. Current-position covariance masses [1,10,100,1000,10000] select100; maximum recorded phase error improves to.249685. This is the new best complete checkpoint, but still fails the preservation agreement gate. Meter usage after A46:197/300forward,50/100quick,0/40full. No rung, full noise result, fresh verification, or final export exists.


A47v1 (`experiments/a47v1_native_repaired/delta.pt`) retains2000/2000 exact quick additions and zero stress output errors, while improving text agreement to .9681396484375, KL .005590123481871276, CE increase .004252428802979047. It changes4,764,911 scalars, max1e6. This is the new best complete checkpoint, still below98.5%agreement. One-shot native donor repairs at6,8,17 each use512donors from[128,256,512,768,1024], ridge.001 and donor-ranking raw masses[0,.01,.1,1,10]. They retain every arithmetic G/U/down column and writer gain exactly. Local validation branch RMS improves respectively .360145→.322898, .276582→.245299, and9.252415→8.037572. Selected raw masses are1,1,.1. The fresh512 native calibration capture restores all three affected MLPs; this means later local fits have an approximate upstream match, which the subsequent complete meter tests. Programme gate maxima at6/8/17 are−13.7393/−60.6144/−51.6485. Usage after A47:199/300forward,51/100quick,0/40full. No rung is banked.

Prepared but not measured: A48 restores native content in the copier's programmed rotary pair by inverse output-gain scaling, with the existing oblique format-mean projection for Q. Gain multipliers[1,2,4,8,16] are checked against routing margin32 and paired calibration QK error; no softmax or model replay is used. A49 directly reconstructs native coefficients in the final MLP's already reused high-gain rows from local HOST branch targets on unlabeled calibration, with one fixed normalization reference and ridges[1e-6,1e-5,1e-4,.001,.01]. Programme columns, gates, Up weights and gains remain fixed. Neither is a result until official measurement.


A49v1 (`experiments/a49v1_strong_final/delta.pt`) is the new complete best:2000/2000 quick additions, zero stress errors, text.96893310546875,KL.005384930292764477,CE+.0026847475836362378,4,764,911changedscalars,max1e6. It reconstructs native coefficients in192already-reused final output rows using2017 surviving native features. Ridge.0001 from[1e-6,1e-5,1e-4,.001,.01] gives local branch RMS8.03757→7.40891; strong-row RMS7.33082→4.62303. Actual raw-normalization change stays below9.3e-9. All programme columns/gates/Up weights/gains remain exact. A48 native rotary-pair restoration is set aside without metering: the gain needed to retain margin32 raises paired calibration score error; the original gain offers only a tiny improvement and a margin29.17. A50 reduces the existing copier gain using direct bounds at margins[8,16,32,64,128], selecting8/gain155.07364. Stress and quick arithmetic remain perfect; its text result is pending. A52 has captured actual A-reader value mixtures for calibration-only analysis, with no weight result yet.


A50 finishes at2000/2000 quick additions, agreement.9691162109375,KL.005371278869723815,CE+.0028033943248892967. A53's five native-query projection fractions[0,.25,.5,.75,1] select.25 with computed copier gain173.8055 at margin8. Its paired calibration score error improves .00311979→.00285602. A53 is not separately scored. A52 composes that query change with a new rank-two A-value dual fitted by actual recorded head-mixture covariances. Mixtures reconstruct to1.64e-7relativeRMS, all64source directions are resolved, and hard58-rank constraints preserve the current-token decoder, readout frame and physical registers. Five raw metric masses[.0001,.001,.01,.1,1] select1; native-value branch validation RMS improves .212017→.151649. All1,536 static A/B digit checks pass. Official A52 quick:2000/2000, text.97021484375,KL.005184119810382761,CE+.00291638408774908,4,764,908changedscalars,max1e6. Stress has zero output errors. This is still not a rung pass.

Two further paired512-window calibration forwards compare HOST and A52 final states, bringing usage to205/300forward,54/100quick,0/40full. The actual complete edited capture contains one nonzero arithmetic feature in the final block (max.296614), with a maximum arithmetic gate−.024327. Earlier quiet-gate measurements on a native-MLP ablation do not establish silence for this complete checkpoint. The next step strengthens this inhibition before broader text repair. Ideal native-anchor repair targets reduce a static final-state metric substantially, but no fitted or scored global repair result is claimed. No full/noise/fresh verification or final export has run.

A58v1's uniform gate sharpening is rejected before metering because it would require weights above1e6. A58v2 sharpens each final arithmetic gate separately and inversely scales its paired Up row. Direct margin32 constraints, with rounding guard1e-5, give maximum gate weight65822.617 and preserve recorded saturated arithmetic features within4.11e-7relative error. Its official calibration capture has no nonzero arithmetic features. Stress remains correct at all2560positions; quick H1 remains2000/2000 and every text aggregate is identical toA52. It retains4,764,908edits,max1e6. Usage afterA58:207/300forward,55/100quick,0/40full.

A57 tests a closed-form global final-state repair from paired unlabeled calibration states. Rank trials[0,8,16,32,64] use a Schur-complement/SVD feature construction, fixed feature ridge.001, and independent Down ridges[1e-6,1e-5,1e-4,.001,.01]. Its calibration selection favors rank0: retain the native features and repair their Down weights, with768ordinary donors and all2017native features in the192already-reused strong rows. Vocabulary-metric error falls .0158180→.0122956 and final-direction RMS .0196897→.0167169. Adding word-gated input features loses that advantage in this first design. These are calibration diagnostics; official A57 stress and quick results are pending. No rung has passed.

A57v1's official quick result keeps2000/2000 additions but lowers agreement to.96783447265625, despite KL improving to.00455151673672 and CE increase to.00128874829595. A57v2 chooses replacement units from all native units and tests larger independent ridges[.01,.03,.1,.3,1] against the original paired target. It again selects no new gated features, with ridge1. Official agreement.96917724609375 remains belowA58, with2000/2000 additions,KL.00408167850307,CE increase.00155370811877. Both change4,879,596scalars,max1e6; both stress checks pass. They improve average distribution distance but do not improve the strict top1 gate. Usage209/300forward,57/100quick,0/40full.

A55 analytically reduces each A-reader programme-query amplitude to the exact quadratic routing boundary at margin8. Factors are[.2901233,.9492794,.3225545,.2977954]; all1536recorded source checks pass afterfloat32 rounding. Its native query covariance-dot error changes .0053102968→.0053105888, so it is not metered. A60 begins a distinct query normalization design using zero-output-gain energy carriers. Carrier counts[1,2,4,8,16] and ridges[1e-6,1e-4,.01,1,100] are computed from unlabeled calibration energy targets. The first design's best static error.01083464 is worse than.00531030, so it is not metered. A second design jointly accommodates ordinary-query row energy and carrier energy in one direct solve; no result is claimed yet.

A61v1 is the new best complete candidate by agreement:2000/2000 quick additions, zero stress output errors, agreement.970458984375,KL.00519441050157,CE increase.00287480827876, all finite. It changes4,764,907scalars,max1e6. Its channel-dependent query gains use a fourth-root energy allocation and exact HOST-floor breakpoints. Asymptotic-margin trials[8.25,8.5,9,10,12] and independent ordinary-row energy ridges[1e-6,1e-4,.01,1,100] select8.25/.01; static covariance-dot error improves.00531030→.00443021. The official improvement is small and remains1.4541percentagepoints below the required agreement. Usage211/300forward,58/100quick,0/40full.

The new actual calibration gate audit finds no programme features at3/7/17, but one token activates small features at6 (.000152976 maximum) and8 (8.0523e-6), at the same calibration position as the earlier final-layer leak. Thus complete silence throughout the circuit is not established. A60's joint carrier/ordinary-energy variant still worsens its static query metric and is not metered. A63's static output-coordinate selection finds a128-coordinate literal digit/newline readout with minimum full-vocabulary margin4.0993 and lower native raw-energy cost than the current192-coordinate pool; no replacement checkpoint is built. A64 is testing a different native-value/output reader using one programmed rotary key/query pair. No result is claimed yet. No rung has passed, and no full/noise/fresh verification has run.


A64's three single-rotary-pair A-reader designs are rejected before metering: their best paired centered causal score errors remain above the current reader's .00088115. Refreshing native format-mean cancellation and trying lower rotary frequencies do not recover the lost native pair information. Full trial grids and measured bounds are retained in their manifests.

A65v2 analytically adjusts final-MLP raw normalization energy while preserving output numerators:16native Down rows are scaled and their postgains inversely scaled. Counts[1,2,4,8,16] and original-target ridges[1e-6,1e-4,.01,1,100] select16/ridge1. It uses4,788,987edits,max1e6. Official quick gives2000/2000 additions, agreement.9705810546875,KL.00507393330845,CE change-.00047128303453,allfinite; stress retains zero output errors. This is a small improvement, still below the preservation gate. A66 attention and final-MLP restoration checks are diagnostic ablations, with results pending.


A66 diagnostic restorations onA61 give97.33276367%agreement for attention2/3/7 and97.50976563%for finalMLP17; both break arithmetic completely. Their conditional effects are not additive. Neither alone restores the preservation gate.

A fresh audit traces the rare calibration activation to layer4: layers0–3 are quiet, but the units-carry gate reaches+199.834 and writes a large negative scratch state, causing later false activations. Attempts to reject that already-corrupted state atlayer6 fail and are discarded without metering. A72v2 computes an old-off multiplier1.03017610 from the strongest declared negative margin512 in[32,64,128,256,512]. It preserves arithmetic branch signs and changes the recorded local output by at most.603against a50000digit scale. Actual complete-candidate gate, stress and quick checks are in progress; no improved score is claimed yet.


A72v2's complete official calibration audit now confirms zero programme features throughout layers0–8 and17 on512windows. The hardened layer4 maximum gate is−512.014; the downstream layer17 maximum is−53291.924. Stress has zero errors at2560positions and quick H1 remains2000/2000. Text aggregates are identical toA65v2 (97.05810547% agreement); the previously observed calibration leak did not affect those held-out aggregate scores. The gate audit is calibration evidence, not a proof over arbitrary text.

A73 records current native final-MLP features and matched HOST intermediate states through the official meter. A74 uses the paired HOST layer8 output as an original, unlabeled-text repair target. It analytically accommodates normalization in weak-gain anchor coordinates, then solves native Down donors once per declared ridge; programme units and gains stay fixed. A74v1 trials: anchor counts[1,2,4,8,16], donor counts[128,256,512,768,1024], ridges[1e-6,1e-5,1e-4,.001,.01]. It selects16anchors/768donors/.01. Its4,951,803edits (max1e6) retain2000/2000 quick additions and zero stress errors, improving agreement to97.15576172%, KL.00395197774543, CE change+.00016879243879. It remains1.34423828percentagepoints below the agreement gate. Per-tensor and arithmetic/text-repair allocations are in its immutable manifest. A74v2 uses the same original target with stronger ridge trials[.01,.03,.1,.3,1], selecting1. It improves the local calibration metric but lowers official agreement to97.03979492%, with2000/2000 additions, KL.00406291143861 and CE change−.00082940254798. A74v1 remains best.

Two other constructions are rejected statically. A70 retains native final output gains except the31scratch coordinates and computes literal digit/newline directions with raw-energy regularizers[0,.0001,.001,.01,.1]. Although isolated class margins are positive, current-register output components fail2219–2560of2560checks because background residual and imperfect cancellation dominate. No checkpoint is metered. A77 tries native MLP norm-energy carriers with counts[0,1,2,4,8] and fixed donor ridge.001. Local reconstruction worsens at every early layer; final-layer carriers also worsen reconstruction and exceed the format-disturbance guard. No checkpoint is saved or metered. A79 confirms that the512infeasible weak-anchor targets inA74 are BOS positions, but their hidden error is small; they do not explain the main preservation loss.

A76 reduces final high-gain output coordinates using the existing literal frozen-unembedding geometry. Counts[32,48,64,96,128] at energy penalty2 give negative minimum margins for32/48/64, but96and128pass. A76v1 uses96coordinates,512native donors with ridge.001, and strong-row reconstruction ridges[1e-6,1e-5,1e-4,.001,.01], selecting1e-5. Native branch validation RMS improves6.38056→5.48449; all2560current-register component outputs have positive full-vocabulary margin (minimum.96280). It changes4,783,242scalars,max1e6. These are static checks; its official stress and quick results are pending. Before this run, usage is224/300forward,64/100quick,0/40full. No full/noise/fresh verification or final export has run.

A76v1 official checks complete: zero stress errors, H1quick2000/2000, agreement97.12524414%, KL.00384216079433, CEchange+.00139392213010,allfinite. Agreement is slightly belowA74v1, so it is not the best scored checkpoint. Its smaller edit count permits an additional paired native attention-value repair within the same budget. Usage225forward/65quick/0full before the new paired captures.


A81v1 uses exact recorded attention-mixture inverses (relative reconstruction error1.4612e-7), paired unlabeled HOST head contexts, and five independent value-repair ridges[1e-6,1e-4,.01,1,100]. All31physical programme input columns remain bit-identical. Ridge1 slightly improves local held-back hidden RMS.00698940→.00696143. The4,939,146-edit candidate keeps2000/2000 additions and zero stress errors but lowers official agreement to97.08251953%, KL.00392097416416, CEchange+.00100402537297. It is set aside; A74v1 remains best. Usage228forward/66quick/0full at completion.

A82's three-feature affine nearest-digit emitter is rejected before compilation. Coordinate counts[32,48,64,96,128] and class biases[16,32,64,128,256] produce positive isolated margins at96/128coordinates, but the selected bias32 gives direction norms varying roughly.4–1. The existing constant-norm scratch cancellation then fails1263/2560and1100/2560current-register outputs. These are component failures, not metered checkpoints. A84 will test eleven literal piecewise-linear interpolation features, which retain unit norm at integer digits, before any new checkpoint is built.

A83 is preparing a distinct preservation repair: one original-target least-squares solve for matched HOST token-score differences on unlabeled calibration text, using only recorded native features and frozen unembedding differences. It changes existing high-gain native Down entries, with programme functions fixed. Five allocations[(32,64),(64,32),(128,16),(256,8),(512,4)] specify native-feature count/output-basis rank; independent ridge trials are[1e-6,1e-4,.01,1,100], with fixed margin-weight floor.1. No result is claimed. Fresh HOST and A76 calibration captures are running; no held-out text is inspected.


A83v1 selects128native features/rank16/ridge100 by separate calibration agreement, improving the restricted validation count7970→7976of8192. Its actual raw-normalizer change is below3e-11 and maximum arithmetic post-branch change below4.47e-5. All recorded final programme features are zero. Official stress has zero errors; quick H1 remains2000/2000, agreement97.15576172%, KL.00384187993037, CEchange+.00139465781719,allfinite. It changes4,783,242scalars,max1e6, tyingA74v1's agreement with168,561fewer edits and lower KL. No preservation pass is claimed. Usage231/300forward,67/100quick,0/40full at completion.

A84's ideal eleven-feature piecewise-linear emitter fails4/2560current-register checks with96coordinates but passes all2560with128coordinates (minimum full-vocabulary margin3.10893). The planned actual construction uses eight hinges at digits1..8, constant/digit/newline features, and the existing eleven cancellation units:22programme units instead of31. Each hinge's Up row uses the common gate, preserving its shared normalization factor algebraically. Sharpness trials[16,64,256,1024,4096], direct negative margin32, and the uniform bound|GELU−ReLU|<.25 determine the feature construction without replaying activations. A diagnostic checkpoint will have its actual new GELU features captured by the official meter before its final Down map is compiled. A85 supplies exact native final features at the current A76 input. No completed A84 result exists yet.


A84's actual hinge feature capture confirms maximum scalar error.00016137 against the ideal literal hinges. The selected sharpness is4096; the largest negative calibration gate is−31.99994. Its complete native output fit uses512donors/ridge.001 and128strong rows with ridge.0001 from[1e-6,1e-5,1e-4,.001,.01]. All features are recorded at the exact current input. Native branch validation RMS improves5.48730→5.35233, and the static full-vocabulary register margin is3.0120. Official stress has zero errors; quick H1 is2000/2000, agreement97.23510742%, KL.00371291219652, CEchange+.00170441322407. It changes4,815,275scalars,max1e6. This improves preservation but does not pass the rung.

A87 reassigns the same22programme functions within the original31native units. One conditional HOST-feature cost solve and importance masses[0,.01,.1,1,10] choose a cheaper subset; duplicate sets are not rebuilt. Each distinct set gets an independent original-HOST native reconstruction with the same512donors and five strong-row ridges. Selected mass1/ridge1e-5 improves local branch RMS5.35233→5.06507. Programme G/Up/Down functions are exactly permuted, and the native/format features are recorded at the same input. Official stress remains correct; quick H1 is2000/2000, agreement97.265625%, KL.00363730726626, CEchange+.00200995647764,allfinite. It retains4,815,275edits,max1e6 and is the new best. Usage236/300forward,69/100quick,0/40full after this score.

A86 revisits the affine nearest-digit emitter with an arithmetic-only raw norm anchor atcoordinate439, retaining that coordinate's small native postgain. Anchor amplitudes[1,2,4,8,16] and pool counts96/128 are tested with the already-declared class bias32. Only128coordinates/amplitude2 pass all2560static register checks (minimum margin.10920);96/amplitude2misses one. The raw output gain is50000 and cancellation is scaled by2.15215635/2. The programme needs14units:3emitter features and11cancellations. Importance masses[0,.01,.1,1,10] and the same native reconstruction ridges select mass0/ridge.0001, improving native branch RMS5.35233→4.38609. The complete static margin remains.109195 and word programme gates stay below−68354.26. Its official stress and quick checks are in progress; no new result is claimed.


## 08:03 CET — A90 complete; A86 readout failure isolated

A86v1 finishes at1974/2000 quick additions (98.7%), below99.5%, although text agreement improves to97.39990234%, KL.00348698617707 and CEchange+.00195018181373. It changes4,800,939scalars,max1e6. An official diagnostic forward on the twenty reported failing arithmetic prefixes finds that every selected digit register still rounds to the correct digit; the failures arise in the final affine readout. The measured digit-phase newline field can have small negative leakage. These reported failures and their captures are diagnostic only: no checkpoint construction uses their cases, answers or activations. A86 is not the best complete candidate.

A90 replaces the one-group affine emitter with a literal grouped construction. Group counts[1,2,3,4,5], within-group squared-radius bias factors[1,2,4,8,16], and arithmetic-only anchor amplitudes[0,1,2,4,8] are evaluated on literal digits and the pre-existing calibration format registers. Two groups, bias factor2, anchor amplitude2 and128output coordinates are selected. Two recorded integer hinges at4and5 bridge the groups, giving5emitter units plus11cancellation units. The full-vocabulary static register margin is1.41840. Cancellation scale2.141343499/2 and raw output gain50000 follow the literal direction norms. Native placement uses the already-declared importance masses[0,.01,.1,1,10],512donors/ridge.001 and five independent strong-row ridges[1e-6,1e-5,1e-4,.001,.01], each fitted to the original HOST branch. No activations are replayed and no sum targets enter construction.

A90v1 official stress has0/2560output errors and quick H1 is2000/2000. Text agreement is97.36328125%, KL.00354599583736, CEchange+.00200784491690,allfinite. Its4,804,523edits,max1e6, improve over A87 but remain1.13671875percentagepoints below the agreement gate. Checkpoint: experiments/a90v1_grouped_affine/delta.pt. Usage239/300forward,71/100quick,0/40full after this score. No rung, full score, noise evaluation or final verification has passed/run.

A91 tests a distinct correction on A86's14unit design. An exact affine basis change gives constant,digit,newline functions, then replaces the last function by the literal2*max(newline/flag-.5,0). Declared sharpness trials[16,64,256,1024,4096] and a directly bounded negative calibration margin512 determine the gate. All native weights, gains and eleven cancellation functions remain bit-identical to A86. Only standard calibration word/format captures are construction inputs. Its official stress/quick checks are in progress; no result claimed.


## 08:12 CET — early-layer preservation experiments

A91v1 is rejected: official stress remains0/2560, but quick H1 is1971/2000 (98.55%). Its text aggregates are bit-identical to A86: agreement97.39990234%, KL.00348698617707, CEchange+.00195018181373. Selected newline sharpness4096 has maximum negative calibration gate−512.000009, minimum signed format gate18.58183 and maximum normalized gate distortion.0714155. It changes4,800,939scalars. This establishes that rectifying the newline feature alone does not fix the affine emitter. A90 remains the best complete checkpoint.

A89 is an unmetered query-restoration analysis. Restoring ordinary Q3 rows while retaining0/32/64/96/128mean-null/energy rows, with independent original-HOST energy ridges[1e-6,1e-4,.01,1,100], fails to improve text-side query geometry. The best128row/ridge1 candidate increases paired centered causal QK score MSE from about.00088115 to.00712380, despite maintaining source-routing margin8. Restored native format means require much larger programme amplitudes. No checkpoint or score run is spent on this construction.

A93 instead keeps format-mean cancellation in1/2/4/8/16input columns, starting with physical marker334, and fits ordinary q_norm gains once to original HOST normalized queries. The format mean remains within roughly1e-6 of its target and routing margins exceed8, but residual raw-energy variation raises paired QK score MSE to.00493751 at its best16column variant. It would reduce changed Q3 scalars to176,128, but is not metered. A94 will combine sparse mean cancellation with an independent sparse raw-energy repair; it is a new structural combination of the two observed requirements, not repeated residual fitting.

Fresh official captures now provide native HOST MLP2 and MLP6 features at the exact current inputs (512calibration windows each). A92 applies the existing one-shot strong-row native repair to layer6 only. Its eight strong rows account for just.09546% of the local native-branch error energy. Ridge.0001 from[1e-6,1e-5,1e-4,.001,.01] improves branch RMS.32253857→.32250606, leaving the arithmetic programme, gains and ordinary rows fixed. Official stress is correct and its quick score is in progress; no improvement is claimed yet.

A88 is preparing a different approach: an exact one-coordinate RMS quadratic specifies how a native MLP2 output can null the following quiet query functional on unlabeled text. Up to five existing strong coordinates and five independent original-target ridge solves are considered. Programme columns remain fixed. Linear/RMS/RoPE QK comparisons and native hidden-state error select candidates without replaying attention or GELU, and without arithmetic answer targets. It has not built or metered a checkpoint.


## 08:24 CET — static failures documented; native-unit placement next

A92v1 completes with2000/2000quick additions,97.35107422%agreement,KL.00354461300763,CEchange+.00198942649628,4,804,523edits,max1e6. Its agreement is slightly below A90; it is set aside. Usage243forward/73quick/0full after its score. A95's later format-only capture brings forward use to244; no later score has yet run.

A88's quiet-form scrub fails before checkpoint creation. On exact current MLP2 inputs, the baseline native branch error is.10378557 and quiet-form RMS.00043370. Four existing strong-writer coordinates yield exact ideal quadratic roots but require excessive hidden changes; all fitted candidates worsen QK geometry. Version2 considers the five largest quiet-form coefficients over all640coordinates. Only marker334 has a feasible exact root on every calibration state; four other coordinates are rejected by the quadratic. At334, five independent ridges[1e-6,1e-5,1e-4,.001,.01] only reduce quiet-form RMS by about11% and worsen paired QK MSE from.00117054 to at best.00137456. No A88 checkpoint or arithmetic meter is spent.

A94 combines one-column mean cancellation at334 with32/64/96/128/160ordinary query rows selected by univariate correlation with the original HOST energy deficit. Each uses independent ridges[1e-6,1e-4,.01,1,100] and analytic routing margin8, with inverse q_norm compensation preserving score numerators. Best paired centered causal QK MSE per count is.00136583/.00118689/.00116924/.00113986/.00111744 (ridge1e-6), versus.00088115for the current reader. Changed Q3 scalars are246400/328192/409984/491776/573568. All retain routing margins; none improves the local text metric. These are retained as parameter-budget tradeoffs, not improvements or metered checkpoints.

A95v1 attempts an original paired HOST layer6 hidden-state repair using native features, anchors[1,2,4,8,16], donors[128,256,512,768,1024], and independent ridges[1e-6,1e-5,1e-4,.001,.01]. With16anchors,14,674of32,768target states lie outside the fixed post-gain RMS sphere, leaving only13,559training rows. No candidate meets the construction requirements. A separate exact ray-geometry analysis reduces infeasibility to523tokens by allowing the HOST state scale in[.5,2], with24,184feasible training rows. Feasible scales have1st/99thpercentiles.88628/1.09084. Gain floors[.01,.03,.1,.3,1] add no useful reachability. An unmetered dtype mismatch in the geometry analyzer was corrected before any checkpoint construction; no meter was used by that failed analysis invocation.

A95v2 applies the same independent grids to those exact HOST-ray targets. Although reachability improves, native-feature fitting is poor: the best candidate within edit/format bounds has direction RMS.08935874 versus baseline.00702634. It is rejected before writing a checkpoint, so no A95stress/quick run occurs. Both failed constructions and all grids remain in their sources and build logs. These are limits of the tested geometry/feature fits, not an infeasibility claim for the goal.

A96 is now testing native-unit placement at layer6. The18programme G/Up functions, Down columns and writer gains are permuted exactly into units selected from all1,536non-donor native units. Conditional cost is computed from exact current-input HOST features and the fixed512donors. Importance masses[0,.01,.1,1,10] each receive an independent original-HOST native reconstruction, using donor ridge.001 and strong-row ridges[1e-6,1e-5,1e-4,.001,.01]. Only a candidate improving local native branch RMS by at least1% is allowed to proceed to the official meters. No result is claimed yet.


## 08:55 CET — A98 retained; fresh gate evidence and late-selector probe

A96v1 stopped before construction because the layer6 copy-frame writes through ordinary as well as strong coordinates; the final-layer helper could not preserve all its gains. A96v2 uses a new fixed-gain helper with a single pre-existing normalizer reference and independent original-HOST solves. Conditional masses[0,.01,.1,1,10] selectmass.1/ridge.001, improving branch RMS.32253863→.31977679 (less than1%). The predeclared1%cutoff rejects it before checkpoint creation or metering. This is a small local gain only; no arithmetic or held-out score is claimed.

A97 audits a genuinely fresh calibration set:512windows,64tokens,seed1203517. Layers0..3 remain fully quiet, but layer4 fires on three ordinary-text tokens at(window,position)=(147,7),(294,34),(465,5), maximum gate4084.17 and feature1.40547e7. The same three tokens leak downstream through5/6/8. A final hinge independently fires at(187,61). This is calibration evidence only. No held-out text was inspected.

A98 directly strengthens existing off projections. Negative-margin trials[32,64,128,256,512] select512 at both layer4 and the two final hinges. Root4 adds off coefficient17023.62637 with reciprocal common sharpening1.00001; minimum signed format gate40.18312 and maximum source-format post change6.02745 (relative data change.000120549). A global RMSNorm rounding bound.42738 certifies the old calibration gates remain below−511.59. The final knots4/5 add off coefficients.01309360/.01710176; additional scalar distortion is bounded by.00006135. Three root-corrupted windows are excluded from the late-gate construction and then checked by a fresh official forward after repair.

A98v1 official audit has every programme feature exactly zero on all32,768fresh calibration tokens. Root4/final maximum gates are−512.04590/−512.00922. Other maximum gates include layer5−15.99671 and layer6−10.27411. Stress has0/2560wrong outputs; quick H1 is2000/2000. Its text aggregates are bit-identical to A90: agreement97.36328125%,KL.00354599583736,CEchange+.00200784491690,allfinite;4,804,523edits,max1e6. It is the retained reference because of improved calibration gate reliability, with no claimed held-out improvement. No rung passes. Usage247/300forward,74/100quick,0/40full after this score.

A100 tests a distinct original-HOST state repair in untouched native layer16. Paired HOST/current word captures and a format capture use three forwards. Anchor counts[1,2,4,8,16], donor counts[64,128,192,256,320], and ridges[1e-6,1e-5,1e-4,.001,.01] fit positive HOST-ray targets in[.5,2] once per trial. With16anchors, ideal target direction RMS is.00037246 and all states are feasible. The best admissible actual fit is16anchors/128donors/ridge.01: direction RMS.00947967→.00927105, format post change51.96954, projected4,886,443edits. It fails the predeclared5%local-improvement cutoff, so no checkpoint or score is produced. All125trials are retained in experiments/a100v1_layer16_state_repair/layer16_repair.json. Completed usage is250forward/74quick/0full before the next diagnostic.

A102 tests native final-denominator repair after the compact-emitter reconstruction replaced earlier energy edits. One fixed original-HOST-normalizer strong-row solve usesridge.0001. Energy row counts[1,2,4,8,16] and independent ridges[1e-6,1e-4,.01,1,100] scale native Down columns only and restore ordinary HOST gains with inverse row compensation. The best16row/ridge1e-6 candidate reduces relative energy RMS.04216090→.01869899, but worsens native branch RMS4.54767205→5.19804196. It is rejected before checkpoint creation and spends no meter run.

A103 starts a structural probe: restore native MLP8 and17, retaining the entire0..7adder/copier/amplifier, then inspect whether its physical digit and phase registers survive to layer16. The aim is to move the selector later, reducing downstream propagation of its native-text disturbance. The diagnostic has no final adder; its arithmetic result must not be treated as a candidate result. An official format capture has completed and a diagnostic quick score is running. The prepared analyzer compares recorded7.out fields with16.raw/17.raw and current-position identities, never fits answers. No late selector has been built.


## 09:17 CET — late selector constructed; copier native-pair loss isolated

A103v1 diagnostic (MLP8 and17 restored) finishes with0/2000 additions, as expected for an incomplete circuit, and97.57080078%text agreement,KL.00415699742,CEchange+.00143315866. The register analyzer finds123digit-rounding mismatches by16.raw and maximum slot error.79364; this placement needs stronger arithmetic fields. A103v2 multiplies only the seven layer7 programme digit/phase Down rows by16 from[1,2,4,8,16]. Native coefficients, gates, Up rows and gains stay fixed. A pre-build metadata KeyError was fixed before any checkpoint or meter run. Its official format capture has zero digit-rounding mismatches through17, maximum digit drift.03780, maximum slot error.24654, and literal threshold gap.25346. Word calibration at the exact new layer16 input was also recorded.

A104v1 attempts an eleven-unit selector at16 but its common-gate factor bound fails with fixed sharpness4; no checkpoint is written. A104v2 chooses common sharpness from[4,16,64,256,1024] and retains the five off masses[1e2,1e4,1e6,1e8,1e10], ramp widths[.025,.05,.1,.2,.4], and ramp sharpnesses[100,1000,10000,1e5,1e6]. It selects mass100,common256,width.2,sharp1000, with minimum signed gate110.1544, common factor error.00092773 and negative bound512. Native placement masses[0,.01,.1,1,10],512donors/ridge.001 and strong-row ridges[1e-6,1e-5,1e-4,.001,.01] selectmass0/ridge1e-6,local HOST branch RMS3.3630. It changes4,086,582scalars,max1e6. Official captures confirm every programme feature is zero on32,768calibration tokens (maximum gate−511.99887). At17.raw the literal selector error is at most.0043821, with zero digit/newline mismatches; maximum digit distance from an integer.300526 and newline-field error.038890. These component checks do not establish a rung.

A105v1 adds a fresh final gate and the fixed A90two-group literal emitter, with5emitter and15scratch-cancellation units. Common sharpnesses[4,16,64,256,1024], hinge sharpnesses[16,64,256,1024,4096], five off masses, negative bound512 and hinge scalar bound.002 govern the gates. Native units/repair use the same five placement masses and five strong-row ridges, always from the original HOST branch at exact current input. It changes4,647,350scalars. Official stress has1142/2560wrong outputs despite correct source digit/newline registers. The cause is two amplified physical fields172and574outside the original128-coordinate emission/cancellation pool, each roughly600,000in magnitude. This is a diagnostic failure, not an arithmetic result to promote.

A105v2 extends the pool to130coordinates, adding zero literal-emission coefficients and explicit linear cancellation for172/574. It recompiles independently from A104 and original HOST native features; no answer fit or iterative residual fit occurs. It has5emitter+17cancellation units,4,653,962changedscalars,max1e6. Official stress has0/2560wrong outputs, minimum top1 margin1.05885. Its complete quick score is currently running. No preservation improvement or rung pass is claimed yet.

A106 tests current-input covariance quieting of the copier's literal marker/continuation query and source-key functions, plus native-query projection fractions[0,.25,.5,.75,1]. Five quiet covariance masses[1e2,1e4,1e6,1e8,1e10] select100for both functions. The best fraction.25/gain174.33134 retains source margin8 and reduces paired centered causal QK MSE only.00285928→.00285257. The predeclared5%improvement cutoff rejects it without writing a checkpoint or spending a meter run.

A107 decomposes exact-current-input copier QK error on held-back calibration states. Ordinary-channel error MSE is.0000682075; the programme pair contributes.0000937056; the removed HOST pair contributes.00279417. Restoring that native pair ideally while retaining programme terms would reduce total MSE.00285928→.000162081. This is an algebraic counterfactual, not a runnable checkpoint or model result. It motivates A108: restore native pair numerators by inverse gain compensation, analytically balance the two programme/native cross terms by their fourth-root energy ratio, and test native-key projection ranks[0,1,2,4,8] with total-gain multipliers[.75,1,1.25,1.5,2]. Native-query format projection rank is1; every candidate must retain routing margin8 and improve paired QK MSE by at least5%. No A108 result is yet claimed.


## 09:43 CET — copier restores native content, but agreement still fails; direct text repair

A105v2 completes at1999/2000 additions (99.95%),97.12524414%agreement,KL.00488124769023,CEchange+.00098441458517,allfinite. Its4,653,962edits remain within bounds, but preservation is worse than A98. The late-selector branch is set aside; it is not a passed rung.

A108v1 restores native content in rotary pair87 while balancing programme/native cross terms. Native-key projection ranks[0,1,2,4,8] and total-gain multipliers[.75,1,1.25,1.5,2] selectrank0/multiplier1. The analytical query/key gain ratio is1.80257708, giving313.29780445/96.42056243gains and minimum routing margin8.07921. Paired calibration QK MSE falls.00285928→.000849111. It retains4,804,523edits,max1e6. Actual stress has0/2560errors and quick H1 is2000/2000, but agreement is97.34497070%,KL.00346987695909,CEchange+.00017256683144. Agreement is three tokens below A98 on the quick text set. A further official audit on512fresh calibration windows(seed1203517) finds every programme feature exactly zero; final maximum gate−513.97314. A98 remains the reference. Usage259forward/77quick/0full after this audit.

A110 computes two-dimensional programme orientations through an eigendecomposition minimizing the two native/programme cross-term energies subject to their dot product being1. The same five key ranks and five gain multipliers selectrank8/multiplier1.25; orientations[-.40135771,-.91592139] and[-.89385012,-.44836588], gains725.67277/84.53603. Calibration QK MSE.00112252 and minimum source margin13.70394 improve over A98 locally. Stress remains correct; quick H1 is2000/2000, but text agreement97.32055664%,KL.00354675227267,CEchange+.00090239377670. It has4,804,523edits,max1e6 and is set aside. Usage260forward/78quick/0full after this score.

A111 uses paired unlabeled HOST final states for an independent global final Down repair, giving greater weight to near-tied HOST predictions. Weight exponents[0,.5,1,1.5,2], donor counts[512,640,768,896,1024], and ridges[.01,.03,.1,.3,1] define125one-shot solves from the same original target; there is no residual refitting. The fixed gap floor is.1 and weights are capped at100 before mean normalization. One weak native anchor at439 makes all32,768original targets feasible. All programme weights, G/Up and gains remain bit-identical to A98.

A111v1 rejects every candidate because its absolute format-post-change cutoff1 is too restrictive (best changes are about3against multi-million arithmetic states). A111v2 instead computes a protected-head norm bound on normalized format-state changes, requiring the static top1 margin remain above1. This is a recorded-feature arithmetic check, not a substitute for the official model. It selects exponent0,896donors,ridge.3: the weighting itself gives no improvement in this tested family. Full-vocabulary calibration validation confirms HOST agreement7967/8192→7976/8192(97.2534%→97.3633%), direction RMS.0150028→.0144855. The new checkpoint changes5,002,267scalars,max1e6. Its official stress/quick are running; no held-out improvement or rung pass is claimed.

A112 is prepared as a parameter-budget design: restore every ordinary Q7 row exactly to HOST and keep edits only in its two rotary-pair rows per head, with native-pair restoration and analytical gain balancing. A direct programme source-gap bound establishes the base gain; five key ranks[0,1,2,4,8] and five gain multipliers[.75,1,1.25,1.5,2] remain the trial grids. Every constructed Q/K gets one original-HOST normalization-gain solve. The goal is to free roughly650,000parameters so a full native final Down reconstruction can fit below5%. This is not yet a result or checkpoint.


## 10:06 CET — sparse copier frees budget; common carry frame under test

A111v2 completes with2000/2000 additions,97.34497070%agreement,KL.003584966851998,CEchange+.00107205138656,5,002,267edits,max1e6. It is below A98 by three text tokens and is set aside.

A112v1 restores all ordinary Q7 rows to HOST and edits only the two programme rotary-pair rows per head. Native-key projection ranks[0,1,2,4,8] and gain multipliers[.75,1,1.25,1.5,2] selectrank2/multiplier1. The analytic gain ratio1.80266437 givesQ/Kgains355.04751076/109.25887469, minimumsource margin8.36155, and calibration centered causal QK MSE.001112816 versusA98 .002859281. It uses4,154,287edits,max1e6, freeing650,236scalars. Stress is0/2560wrong; quick H1 is2000/2000, agreement97.34497070%,KL.00347712001124,CEchange+.000109026656527. This is a useful budget allocation, not a passed rung or an agreement improvement. Exact current native final features were captured on512calibration windows(seed803291).

A113v1 uses that enlarged budget for independent original-HOST final-state repairs. Donor counts[512,768,1024,1536,2032], HOSTmargin weight powers[0,.5,1,1.5,2], and ridges[.01,.03,.1,.3,1] give125one-shot solves. Gap floor.1, weight cap100, fixedweakanchor439, fixedprogrammecolumns/G/Up/gains and staticprotected-head formatmargin>1 are unchanged fromA111. The selected768donors/power0/ridge.1 improves full-vocabulary calibration validation7975/8192→7983/8192 and directionRMS.01493874→.01470151, withstaticmarginbound1.41806. Officialstresshas0/2560errors andquickH1is2000/2000, butagreementfalls to97.25341797%,KL.00360132872938,CEchange+.000548101663602. It uses4,286,623edits,max1e6 and is set aside. The calibration gain did not generalize; A98 remains the highest complete quick agreement.

A109v1 compresses the layer6 carry/frame programme from18to14features: one shared no-carry normalization anchor, two literal carry-ramp functions, five shared frame clearers, and the six exact continuation functions. The normalized branch identity differs by at most3.52e-6 in recorded format algebra. Common sharpness[4,16,64,256,1024], carrywidth[.025,.05,.1,.2,.4], carrysharpness[16,64,256,1024,4096] select4/.4/256; negativebound512, minimumsaturation35.54823, commonminimum67.19811. Allpostgains stay fixed. Native placement masses[0,.01,.1,1,10] and independent original-HOST ridges[1e-6,1e-5,1e-4,.001,.01] selectmass.1/ridge.001, reducing localbranchRMS.32253863→.26941260 (16.47%). It uses4,145,121edits,max1e6. The complete official stress/quick checks are running; no arithmetic or held-out improvement is yet claimed.

Completed usage beforeA109:264/300forward,81/100quick,0/40full. No rung passed, full/noise/freshfinalverification/export still pending. A109's queued stress consumes oneforward and its quick another ifstresspasses. Folder39.482GB beforeA109stress; cleanup_a113_space_v1.json retains hashes for retiredA104word/A105stress/A110stress. Checkpoints/sources/manifests/scores are retained. No blocker; work continues.


## 10:24 CET — shared phase frame improves complete agreement

A109v1 completes with2000/2000quick additions,97.33276367%agreement,KL.00343003266569,CEchange+.000256512155503. Its local16.47%branch-error reduction does not improve the complete model's text agreement, so it is set aside.

A114v1 shares all five layer6 frame clearers between equals and continuation, reducing eighteen programme features to ten. It retains one source anchor and two carry-ramp functions, adds a paired continuation anchor, and uses five common clearers. The existing continuation gate plus a source-marker term yields a shared gate; source positivity and negative margin512 determine its coefficients analytically. Continuation widths[.025,.05,.1,.2,.4] select.4, with directly bounded multiplier116.25638455, minimumsignedphasegate292.26782 andmaximumnegativegate−512.00274. The normalization identity's maximum continuation perturbation23.56628 is tiny relative to million-unit physical fields; complete measurement is required. Native placement masses[0,.01,.1,1,10] and five original-HOST ridges[1e-6,1e-5,1e-4,.001,.01] selectmass.01/ridge.001, reducing localbranchRMS.32253863→.21794365(32.43%). All layer6 postgains remain fixed.

A114v1 changes4,137,472scalars,max1e6. Official stress has0/2560errors. Quick H1 is2000/2000, text agreement97.412109375%,KL.00336163472827488,CEchange+.0000396353495473,allfinite. This improves the complete agreement by8tokens overA98 and is the new leading candidate, still1.087890625percentagepoints below the gate. A further official512-window freshcalibration audit(seed1203517) finds every programme feature exactly zero; layer6maximumgate is−512.0. The old seed803291 also remains quiet through0..8. No rung/full/noise/finalverification pass is claimed.

A115 records exact native final features at A114's input, restoring only MLP17 for that diagnostic capture. It also records actual layer8features/raw/down/out for potential independent original-HOST repair. These are unlabeled calibration512x64windows, seed803291. The upcoming final-layer factorization first absorbs cleanup Up rows that lie in the existing emitter's linear span with identical gates, then tests residual cleanup ranks[0,2,4,6,8] using PCA of recorded cleanup features. No sum targets or nonlinear replay enter construction. No A115 checkpoint or score is yet claimed.

Usage:268/300forward,83/100quick,0/40full, including both new calibration captures. Folder39.704GB beforeA115build. cleanup_a114_space_v1.json retires A111/A113/A108 stress captures; cleanup_a115_space_v1.json retires A109stress and the superseded A112wordbasis. Hashes, constructors, manifests, requests, checkpoints, diagnoses and scores remain. No blocker; goal stays active.


## 10:35 CET — final cleanup factorization improves agreement; full layer8 repair fails to improve it

A115v1 folds three cleanup functions for physical coordinates52/391/476 into the emitter's existing first three Up rows. They share an identical gate, so this is a weight-space linear identity up to float32 coefficient rounding. PCA of the remaining eight recorded cleanup features tests ranks[0,2,4,6,8]. Rank4changes232static output winners; rank6and8change none, with minimumstaticmargins1.418153/1.418325. Rank6is selected, with relativecleanupfeatureerror.00510924, giving5emitter+6cleanupfeatures=11instead16. This PCA uses physical scratch features, not sums or expected answers.

Exact current-input HOST final features come from the A114 native17capture. Native placement from all1536non-donor units uses masses[0,.01,.1,1,10], fixed512donors/ridge.001, and independent strong-row ridges[1e-6,1e-5,1e-4,.001,.01]. Mass0/ridge1e-5 reduces localbranchRMS4.54273638→4.16803894. Allprogrammegatefunctionsonbotholdandfreshcalibration inputsremainbelow−32. A115v1uses4,128,511edits,max1e6. Officialstresshas0/2560errors;quickH1is2000/2000,agreement97.5341796875%,KL.00323059219756839,CEchange+.0000854577108083,allfinite. It improves agreement by20tokens overA114 and is the new leader, still.9658203125percentagepoints below the gate. No rung is banked.

A116v1 tests a larger native layer8 basis using the sparse copier's freed budget. All125solves start from original HOST Down weights with fixed writer accommodation, not the previously fitted global8map. They target paired original HOST layer8 states on unlabeled calibration. Anchor counts[1,2,4,8,16], native donor counts[512,768,1024,1536,2037], and independent ridges[1e-4,.001,.01,.1,1] select16/2037/1. Every programme column and postgain remains fixed; the observed reference normalizer is never recomputed from fitted candidates. Selected stateRMS.00642875→.00603052, directionRMS.00465097→.00444732, p99relativeerror.01792295→.01704617. The512infeasible targets are BOSpositions;24,192trainingrowsremain. Maximum formatpostchange.730887. It uses4,935,595edits,max1e6.

A116 officialstresshas0/2560errors andquickH1is2000/2000, butagreement97.51586914% is three tokens belowA115. KL.00313226081450 andCEchange+.000203280558100 remainwithinbounds. It is set aside; lower calibration state error did not improve complete agreement.

A117 is prepared to replace layer4's eight carry features with a six-feature common-anchor circuit: one anchor, three linear column copies and a paired literal carry ramp. The planned trials arecommonsharpness[4,16,64,256,1024], width[.025,.05,.1,.2,.4], carrysharpness[16,64,256,1024,4096], nativeplacementmasses[0,.01,.1,1,10],128fixednative donors andstrong-rowridges[1e-6,1e-5,1e-4,.001,.01]. Exact current-input native4features are being captured through the official meter; no A117 result is claimed.

Completedusage beforeA117capture:271/300forward,85/100quick,0/40full. A117capturewillconsumeoneforward. cleanup_a116_space_v1.json retires supersededA61/A76/A84stress; cleanup_a117_space_v1.json retiresA112stress. Allcheckpoints/sources/manifests/requests/scores/diagnosesremain. A115 is retained; no full/noise/finalverification/export yet. No blocker.


## 10:47 CET — six-feature units carry improves quick agreement; full H1 evaluation queued

A117's first unmetered build encountered a native-repair helper assumption: five layer4postgains are exactlyzero. Dividing the zero HOST target by those gains produced invalid candidates. Nocheckpoint or meter was produced by that failed build. A new immutable helper version2 preserves the original HOST raw-energy targets in zero-output coordinates, verifies their HOST gains are alsozero, and keeps the normalizer reference fixed. The original version1 helper is unchanged.

The completed A117v1 selects commonsharpness4,width.2,carrysharpness16, withminimumsignedgate17.35332672,commonminimum100.93934963,negativebound−512.0000133 andmaximumprogrammegateweight42358.96875. The literal raw branch identity is exact in double arithmetic. Nativeplacementmass.01 andridge.01 select six units[563,298,771,1065,1757,1424]; allpostgains remainfixed. NativebranchRMS.1337145161→.1074826237(19.62%). It uses4,124,677changedscalars,max1e6. The fresh exact-input native4capture consumesoneforward.

Official A117 stress has0/2560wrongoutputs. QuickH1is2000/2000, textagreement97.674560546875%,KL.00364569282563089,CEchange+.000932642018867,allfinite. This improves agreement by23tokens overA115, while KL and CE are worse but remain below theirlimits. A117is the new leading candidate, still.825439453125percentagepoints below98.5%. A full512-window freshcalibration audit(seed1203517), with the correct new4/6/17unit lists, is queued. If allprogrammefeaturesremainzero, the first fullH1score follows:25,000problems,2,048textwindows, two-problemcontinuations andnoiseat1e-5/1e-4/1e-3. No full result or rung pass is claimed yet.

A118's constructor is prepared but not run. It retains the11-feature final programme and tests fewer high-gain outputcoordinates[64,80,96,112,128], literalgroupbiasfactors[1,2,4,8,16], andrawanchors[0,1,2,4,8]. Its independent native fits use original HOST features at the A115 input; it is an alternative onA115, not a silently composed A117 change. No A118checkpoint or result exists.

Completedusage beforetheA117freshaudit:273/300forward,86/100quick,0/40full. The queued audit addsoneforward; a successful audit is followed byonefullrun. cleanup_a118_space_v1.json retires the rejectedA116stress/format captures; cleanup_a117_full_space_v1.json retires oldA85nativeword/nativeformat observations superseded by the retainedA115basis. Allconstructors/manifests/checkpoints/requests/scores/diagnosesremain. No blocker; the goal remains active.


## 11:16 CET — first full evaluation: arithmetic passes, preservation does not

A117v1 full H1 score is24,991/25,000(99.964%), including the newline, with equal digit-prefix accuracy. Text on2,048windows/131,072targets has97.576904296875%top1agreement,KL.003428864613181469,HOSTCE5.023652493858176,editedCE5.023512017586756,CEchange−.000140476271419665,allfinite. Agreement misses98.5%by.923095703125percentagepoints. H1 is not passed and no best/ checkpoint is banked. The full result is scores/a117v1_full_H1.json, ledger run0683-score, checkpointSHAa2a44a43181de0565d92966ae010ec2d908cb9b57694251a59048ba9d5f750dc.

Multiplicative noise on edited scalars gives2,000/2,000at1e−5,2,000/2,000at1e−4, and950/2,000(47.5%;954digit-correct)at1e−3. Allfinite. Two-problem diagnostic: first1999/2000, secondgivenacorrectfirst2/2000, both2/2000. The current H1 circuit does not generalize to repeated problems; this is a reported limitation, not a gating result. Full score provides nine failing arithmetic instances; no answer regression or failure-specific patch is authorized.

Completedmeterusage274forward/86quick/1full; remaining26/14/39. The full run confirms the preservation shortfall seen inquick checks. Native-attention versus native-MLP calibration restorations are the next diagnostic, with three official forwards. No blocker; work continues.


## 11:24 CET — calibration restorations identify a larger MLP cost

A119 uses three matched official calibration forwards (512x64, seed803291). A117 agrees with HOST on32,012/32,768tokens(97.69287109375%); the last8,192validation tokens have7,991matches(97.54638671875%). Restoring HOSTattention2/3/7 gives32,149matches(98.1109619140625%),validation8,031(98.03466796875%). Every programme feature remains exactlyzero in both cases. Restoring HOSTMLPs0..8and17 gives32,350matches(98.724365234375%),validation8,082(98.6572265625%). Final normalized-state RMS errors are.01703401,.01684557,.00893508respectively. These are calibration-only conditional ablations, not arithmetic candidates or additive causal allocations. The MLP restoration has the larger effect. Fulldetails/hashes: artifacts/a119v1_text_ablations.json.

Completed usage277/300forward,86/100quick,1/40full. A118 is now being built directly onA117, superseding the earlier unrun plan to build it onA115. A new native-MLP17 diagnostic provides exact current-input features and calibration predictions. Its official capture is followed by the prepared five-pool/five-bias/five-anchor geometry and original-HOST native fits. No A118 result yet.

A121 is a new exact structural simplification, prepared but unrun: B amplifier constant and plus-reference GLUs have identical gates and Up rows. Adding their Down columns reduces five units to four. Current native2features already exist in a88v1_native_mlp2.pt; the constructor checks every upstream weight against that capture's diagnostic checkpoint. Native placement masses[0,.01,.1,1,10], fixed512donors and independent ridges[1e−6,1e−5,1e−4,.001,.01] require at least2%local native improvement. Allpostgains fixed. No meter or checkpoint yet.


## 11:35 CET — smaller writer ties agreement; duplicate B unit removed

Restoring only MLP17 to HOST at the exact A117 input gives calibration agreement32,083/32,768(97.9095458984375%), validation8,004/8,192(97.705078125%). This conditional ablation adds71full-calibration matches versus A117, smaller than the338matches from restoring all edited MLPs. It supplies exact native17features for A118.

A118v1 uses96rather than128high-gain output coordinates. Pool counts[64,80,96,112,128], literal group biases[1,2,4,8,16], anchors[0,1,2,4,8] select96/bias2/anchor2. It retains eleven programme features, all gate/Up rows and their locations. Selected literal geometry has static format margin1.18517197 with no changed winner, cancellation scale2.14100936125 and raw output gain50000. Original-HOST native repair uses512donors/ridge.001 and strong-row ridges[1e−6,1e−5,1e−4,.001,.01], selecting1e−5. Exact-current-input native branch RMS improves4.165406819→3.480484324(16.44%). It changes4,075,878scalars,max1e6.

Official A118 stress has0/2560wrongoutputs. QuickH1is2000/2000, agreement97.674560546875%,KL.00360979421716255,CEchange+.000652138458695,allfinite. Agreement ties A117; KL/CE and parameter count improve. A118 is the working base, while A117 retains the only full score. No rung passed. Its constructor is now immutable. Usage afterA118:279forward/87quick/1full.

A121v1 folds B amplifier positions0and4 after verifying their gate and Up rows are bit-identical, adding their Down columns. Four features replace five. Exact native2capture provenance is checked against every upstream weight. Native placement masses[0,.01,.1,1,10] and independent ridges[1e−6,1e−5,1e−4,.001,.01], with512fixeddonors/allpostgainsfixed, selectmass0/ridge.01 andunits[80,1634,342,1348]. Local native branch RMS.103785573→.083471181(19.57%). It changes4,073,963scalars,max1e6. Stress and quick are running; no complete result yet.


## 11:56 CET — rank-one value reader gives a small complete gain; newline recoding in progress

A121's official stress has0/2560errors and quick H1is2000/2000. Agreement falls by four tokens to97.650146484375%,KL.00384632271528,CEchange+.000742749884065. It is set aside despite its19.57%local native-branch improvement. The working base remains A118.

A120v1 replaces layer3's rank-two constant/digit V edit by a rank-one digit edit. A native reference is obtained from independent residual head means, including fixed BOS. Current-token digit labels only constrain the readout. Actual calibration attention mixtures are recovered by linear inverses; no softmax or GELU is replayed. Raw metric masses[.0001,.001,.01,.1,1] and background covariance masses[0,1,100,10000,1000000], fixed within-source mass10000 and ridge1e−6, select.1/10000. Native value-branch validation RMS.151614073→.098915097(34.76%). Value-span identity error8.79e−10; predicted maximum programme3gate−31.97148. Three older source probes have no rounded digit errors and maximum individual-place error.191545. Their staleness is explicit; complete meter tests establish actual behavior. The common Up gain remains1e6 from[1e3,1e4,1e5,3e5,1e6]. Only V3 and four programme Up rows change.

A120 changes4,075,877scalars,max1e6. Official stress has0/2560wrongoutputs. Quick H1is2000/2000, textagreement97.686767578125%,KL.003461554395119804,CEchange+.000755322894358,allfinite. Agreement improves by two tokens over A118/A117. It is the new working base, still.813232421875percentagepoints below the quick agreement gate. No full result or rung pass is claimed. Usage afterA120:281forward/89quick/1full. A120analysis/compiler sources are now immutable.

A122 obtains exact native8features and current format states through two official passes. The native8restoration diagnostic agrees with HOST on32,060/32,768calibration tokens(97.83935546875%), validation8,021/8,192. Programme features0..7and17are exactlyzero. Upstream0..7match A120 exactly; downstream17is conditional on native8restoration. This is not a full A120fresh-gate audit or an arithmetic candidate.

A122's intermediate selector encodes newline as−1, reducing11features to9and restoring native gain52. Since the final digit-transition and newline pairs share gates, subtracting the newline Up rows from the last digit Up rows gives an exact feature-space recoding. The former newline pair becomes native. Original paired HOST8states are repaired independently from HOST Down, with fixed observed RMS reference,16weak anchors, donors[512,768,1024,1536,2039] and ridges[1e−4,.001,.01,.1,1]. Selection2039/ridge1 uses24,192trainingrows;512BOS targets are infeasible. StateRMS.00595740710→.00541327151(9.13%),directionRMS.00430563911→.00398292635,p99.01713963132→.01652738969. Format nativepostchange1.58725plus analytic freed-feature bound15.22087 is below100. The intermediate changes4,880,396scalars,max1e6. Its old final emitter is deliberately incompatible; no arithmetic score is claimed.

A third official mixed capture restores only MLP17 on the new selector:384calibration windows(256training/128validation) and128fixed-random format probes, padded only after their original14tokens. This gives exact current native final features and actual recoded format registers in one pass. Its layout and source indices are recorded in artifacts/a122v1_final_basis_layout.json. The compatible five-feature emitter is now building. Its literal curve equals the old grouped digit curve for0..9and the old newline direction at−1, with hinge knots[−.5,4,5]. It keeps six cleanup features, removes obsolete52cancellation, and folds391/476cleanup into the constant/digit inputs by weight-span identities. Hinge sharpnesses[16,64,256,1024,4096], negative bound512 and a uniform GELU-to-ReLU error bound govern the construction. No actual GELU replay, sums or answer regression enter it.

Completed usage284/300forward,89/100quick,1/40full; remaining16/11/39. cleanup_a122_space_v1.json hashes and removes the superseded A45reader calibration and rejected A121stress captures. All sources, requests, manifests, checkpoints, scores and diagnoses remain. Folder39.149GBbefore the complete emitter/stress. No rung, no best/, no finalverification/export, no blocker.


## 12:16 CET — negative-one selector improves text; partial final phase explains newline failures

A122v1's complete emitter keeps eleven programme units and 96 strong output coordinates. Its five literal functions are constant, digit, and hinges at −0.5, 4 and 5; six PCA cleanup features remain. Sharpness trials [16, 64, 256, 1024, 4096] select 4096. The maximum independent negative gate is −512.0045355 and the uniform literal hinge error bound is 0.0002102593. The exact per-token curve identity has maximum error 6.11e−16; 640 ideal format states have no incorrect source-register winner and minimum full-vocabulary margin 1.5068547. No sum targets enter construction.

The first unmetered emitter build failed a strict cleanup-span check because subtracting already rounded combined emission/cancellation coefficients left relative error 1.96669e−5. Before any checkpoint or meter, the constructor was corrected to recover literal cancellation from A114's separate physical-coordinate columns, with A118's declared scale. The recovered span error is 1.23e−16. Both logs are retained. The completed source is now immutable. Original-HOST native final repair uses 256 training and 128 validation windows, 512 donors, and strong-row ridges [1e−6, 1e−5, 1e−4, 0.001, 0.01], selecting 1e−5. Its local branch RMS worsens from 3.119456983 to 3.292982251; the programme-unit locations and pool are unchanged, so retaining the prior native final map remains an independent structural alternative to assess.

A122v1 changes 4,880,396 scalars, maximum absolute value 1e6. Official stress has 0/2,560 incorrect outputs. Quick H1 is 1,989/2,000 (99.45%), with all 2,000 digit prefixes correct. Text agreement is 97.845458984375%, KL 0.0026418088799671454, CE change −0.0005255861020785346, all finite. This gains 26 text matches over A120, but still misses both the 99.5% arithmetic and 98.5% text gates. No full run or rung pass is claimed.

An official diagnostic of the eleven returned failures plus 53 independent random controls identifies incomplete activation of the selector's final paired ramp. In the failures, the reserved newline scalar lies between −0.77 and +0.91 rather than −1; the final paired negative gate is still below zero. The final emitter faithfully follows that malformed register. All controls have correct outputs. Failure problems and answers are used only to diagnose the mechanism, never as constructor inputs.

A124v1 narrows the final selector transition using width fractions [1, 0.75, 0.5, 0.25, 0.125], selected solely on existing independent format probes with current-position and literal leading-bit labels. Each new gate is a convex combination of the old pair, and the two Down columns are scaled inversely to preserve the ideal saturated difference. The selected fraction is 0.125, minimum signed format gate 338.9670, maximum negative word gate −11612.9813. Float32 saturated weight-identity absolute error is 0.00772046; actual-model verification is required. All native weights and gains remain unchanged. The edit count and maximum value remain 4,880,396 and 1e6. Official stress followed conditionally by quick H1 is running; no A124 result is yet claimed.

Completed usage before A124: 286/300 forward, 90/100 quick, 1/40 full. Folder size before its stress was 39.4025 GB. All checkpoints, sources, manifests, requests, scores and diagnoses are retained. No best/ rung, final verification, export or blocker; the goal remains active.


## 12:21 CET — corrected terminator passes quick arithmetic; fixed native final map ties agreement

A124v1's official stress has zero incorrect outputs over 2,560 positions. Its quick H1 score is 2,000/2,000 exact and digit-correct. Text agreement, KL and CE change are bit-identical to A122: 97.845458984375%, 0.0026418088799671454, and −0.0005255861020785346. The narrower final phase fixes all eleven reported missing-newline cases without an observed text change. This is the leading complete quick result; H1 still fails the 98.5% text gate. A124's constructor is now immutable.

A123v1 tests retention of the prior native final Down/gain map, keeping all new arithmetic columns, G/Up rows and 96 strong writer gains. This is a composition of two existing maps, with no fitting or scale search. Exact recorded native features on validation windows 256:384 give branch RMS 3.292982251 for the new fit and 3.119456983 for the prior map; both agree with HOST on 8,024/8,192 full-vocabulary predictions. Reconstructing the composed final output from actual A124 stress features preserves all 2,560 winners, minimum margin 1.3015118. Only linear/RMS/readout arithmetic is used in these checks.

The official A123 quick score has 2,000/2,000 exact additions, 97.845458984375% text agreement, KL 0.002626228531439169 and CE change −0.00048514444975022997, all finite. It ties A124 agreement with slightly lower KL and slightly worse CE; both remain below the agreement gate. It uses 4,880,396 edits, maximum 1e6. No additional forward was spent because the changed final linear map could first be checked on actual recorded features. Its source is now immutable after scoring.

Completed meter usage: 287/300 forwards, 92/100 quick runs, 1/40 full runs. Remaining 13/8/39. No rung has passed; no best/, final verification or export. Work remains active.

A125 begins an independent structural analysis: remove the five layer1 receiver-clearing units and derive a B readout that suppresses the full unmodified receiver background. It first tests unchanged native V/O, using individual current-token/head class-mean differences and independent head reference means. Background covariance masses [0, 1, 100, 10000, 1000000], within-source mass 10000 and ridge fraction 1e−6 are declared. Existing A36 captures are reusable only if current layer0 weights exactly equal their A1 source; the analyzer asserts that provenance. This is static analysis with ideal source routing and recorded backgrounds, not a new arithmetic result or a model replay.


## 12:32 CET — receiver-free B succeeds locally; A reader must be reframed

A125's native-V/O construction establishes exact upstream provenance: current layer0 is bit-identical to the A1 checkpoint used by the retained A36 captures, and the proposed layer1 is HOST. All five full-background covariance choices fail the predeclared 0.35-digit precision bound. The best source/background metric is mass 10000, with worst fresh digit error 0.367745865; it has no rounded errors in these probes but insufficient margin. No A125 checkpoint or meter is produced.

A126 adds one exact current-digit direction to V2, retaining native O2 and attention normalization. Individual current-token class means and within-class covariance define the code; masses [1, 10, 100, 1000, 10000] select 10000, maximum code error 0.01327798. Raw native-output metric masses [0.0001, 0.001, 0.01, 0.1, 1] and readout background masses [0, 1, 100, 10000, 1000000] select 0.0001/10000, with frame within-source mass 10000 and ridge fraction 1e−6. The float32 value-span identity error is 4.05e−10, maximum V weight 0.24082762. Static ideal routing has no rounded digit errors and maximum fresh per-place errors [0.130769, 0.117706, 0.142363]. Individual native-token output disturbance RMS is [0.030492, 0.032551, 0.029984, 0.021193] across the four heads; this is a proxy, not complete native-attention error.

A126v1 restores all four layer1 MLP/gain tensors exactly to HOST, changes V2 and the existing five programme Up rows, and leaves all programme gates, Down columns and later weights fixed. The common Up gain remains 1e6 from [1e3, 1e4, 1e5, 3e5, 1e6]. It changes 4,700,194 scalars, maximum 1e6, removing 180,202 edits net. Its constructor and analysis source are now immutable.

One official mixed capture combines 256 fresh calibration windows (selected from seed1203517) and 256 independent stress probes with fixed selection seed202611101225. All programme features at layers0,2–8,17 are exactly zero on 16,384 word tokens. Actual B reader errors are at most [0.0642103, 0.0552380, 0.0744676], with no rounded errors. However the changed receiver background breaks the old A readout: its errors reach [1.51712, 1.78879, 4.00741], and complete outputs are wrong at 666/1280 positions. This is a broken composition, not a new leading adder. No quick run is spent on it.

A127v1 attempts an Up-only A readout projection at the exact new inputs, retaining V3 and all gates/Down/gains. It uses independent current-token contributions and full current receiver covariance, not sums. Its five background masses [0, 1, 100, 10000, 1000000] with within-source mass10000 fail the predeclared0.35-digit limit; the closest has maximum error0.3505518. The actual versus ideal routed branch relative RMS is0.00031369, so this is a readout/background precision problem rather than gross routing failure. No checkpoint or meter is produced.

A127v2 retains those five background choices and explicitly adds five independent within-source masses [100, 1000, 10000, 100000, 1000000]. Every solve starts from the same original source observations; no fitted residual is refit. It is being constructed, with no result yet claimed. The older A119 word covariance is used only as a fixed projection metric; it is not claimed to be current native input. Up-only changes preserve ordinary behavior wherever the existing gates remain zero, as measured in the A126 fresh audit.

Completed meter usage is 288/300 forwards, 92/100 quick runs, 1/40 full run; remaining12/8/39. cleanup_a125_space_v1.json hashes and retires the superseded 970,733,077-byte A115 native-word capture; exact A122 bases, all sources/manifests/checkpoints/scores/diagnoses remain. Folder size after the A126 capture is39.115 GB. No rung, final verification, export or blocker.


## 12:57 CET — broader source observations resolve a failed readout; selector repair rejected

A127v2 selects within-source mass1000/background mass1 from the five-by-five declared grid. Actual A-readout maximum errors on its256format observations are [0.294393,0.347956,0.228801], with no rounded errors. It changes4,700,194scalars. Official full stress nevertheless has19/2560wrong outputs; the512-source A-readout errors reach [0.469395,0.898046,0.401393], with two incorrectly rounded input digits. The unchanged input digit code itself is much more accurate: [0.131333,0.054143,0.017373]. Thus the receiving projection, not the rank-one input code, needs broader source coverage. No quick score is spent on A127v2.

A128v1 fits that readout from all512 recorded stress source prefixes with a fixed random384/128split(seed202611101237). Layer3input is recovered by RMS algebra from recorded layer2output; no model layer or activation is replayed. The same five within-source and five background masses select10000/1000000, with maximum recorded A-readout errors [0.138019,0.063386,0.039911] and no wrong rounding. Only four programme Up rows change, leaving4,700,194edits and max1e6. Its official quick check fails generalization:1188/2000exact additions(59.4%, same digit accuracy),97.613525390625%textagreement,KL0.0022350622697007473,CEchange−0.0010288772417181846,allfinite. Agreement is38matches below A123/A124. This remains a failed branch.

A129 spends two official passes to observe all900 A prefixes in a fixed random order, each with an independently random B and five independently random continuation digits, plus124 additional random-format controls. The generation seed is202611101241. No returned arithmetic failures or answers enter these requests. Both operands are read diagnostically, as are the computed source, copier and phase. The B reader has no rounded errors over1024formats(maximum observed error0.186699); the A reader has454rounded errors and errors as large as1.92843. Added copier errors remain below0.024digits. Complete outputs are wrong at554/5120positions. These measurements confirm that the broad-domain A readout is the failure mechanism; they are not a new fit to sums.

A130v1 retains the existing V3 digit code and derives only its readout from all900 independent source-prefix/head contributions. The124controls contain new receiving backgrounds but no unseen A prefix, which is disclosed rather than called source-domain generalization. Within-source masses[100,1000,10000,100000,1000000] and background masses[0,1,100,10000,1000000] select100000/10000. Actual readout errors over1024formats are at most[0.212940,0.140910,0.078836], all rounded correctly; the input code errors are[0.0372827,0.0120317,0.00386058]before the factor4.5. It retains4,700,194edits,max1e6. A standalone complete quick check is now running; no arithmetic pass is yet claimed for it.

A131 records an exact native8basis on A130:384calibration windows(256training/128validation) plus128independent stress formats in one official pass. All measured upstream programme features remain zero. A direct original-HOST8state reconstruction keeps all programme columns/gates and every gain fixed, starts each solve from HOST Down with fixed writer accommodation, and uses the current observed reference normalizer only once. Donor counts[512,768,1024,1536,2039], ridges[1e−4,.001,.01,.1,1] and16fixedweak anchors are declared. A uniform hand-circuit GELU-to-ReLU bound and coordinatewise RMS perturbation bound certify every format change without activation replay. The best2039/ridge1fit has stateRMS0.00541398839 versus existing0.00522738788, a3.57%worsening; its formatpost bound is4.74099. It fails the predeclared5%improvement requirement, so no A131arithmetic checkpoint or quick score is produced. The prior existing selector map remains.

Usage after A131 capture:292/300forward,93/100quick,1/40full; A130quick is now consuming the94th quick unit. Remaining8forwards,6quick once it starts,39full. cleanup_a129_space_v1.json retires historical A118native17(520,755,617bytes); cleanup_a131_space_v1.json retires historical A117native4(352,588,840bytes), with hashes retained. All sources/manifests/checkpoints/results remain. The leading complete candidate remains A123/A124, no rung passed, no best/, final verification, export or blocker.


## 13:33 CET — compact selector trades text quality for capacity; early repairs rejected

A130v1 completed quick H1 with 1,999/2,000 exact and digit-correct additions. Text agreement is 97.613525390625%, KL 0.0022350622697007473, CE change −0.0010288772417181846, all finite. Its single returned failure is used only diagnostically. This receiver-free branch remains below the leading A123/A124 result.

A132v1 replaces A130's full native selector reconstruction with a local original-HOST branch fit at the exact current input. It uses 512 fixed donors/ridge0.001 and full native features only for two strong rows; strong ridges [1e−6,1e−5,1e−4,0.001,0.01] select1e−4. All programme Down columns, gates, Up rows and output gains [215,391,476] stay fixed. The first unmetered build failed because it did not preserve the weak programme anchor gain215; this was corrected before any checkpoint or meter, and both logs remain. Native branch RMS improves0.3069311934→0.1898390123, but paired original-HOST8 state RMS worsens0.00522738788→0.00546605099. The format post-change bound is8.194890553. It changes3,725,972scalars, max1e6, freeing974,222edits. Quick H1 remains1,999/2,000, but text falls to97.57080078125%, KL0.0024511824658394327, CEchange−0.0020785299586894013. It is an allocation experiment, not the leader.

A133 records paired original HOST layer2/3/4/7 outputs and exact native3 features at the receiver-free A132 input in two official forwards. The mixed native3 capture uses384word windows (256training/128validation) and128independent formats. These inputs are not the A123 inputs; any transfer must disclose this distinction. All measured upstream programme features are zero. A129's two large source captures were compacted into a new derived-subset artifact with explicit position maps, original hashes and provenance, then retired after hashing; cleanup_a133_space_v1.json records815,144,144bytes removed. The compact artifact is not an additional meter pass.

A134 attempts an original-HOST3 state fit with fixed programme weights and gains. Weak-anchor counts [8,16,32,64,128], donor counts [512,768,1024,1536,2044], and ridges [1e−4,0.001,0.01,0.1,1] are declared. Only128anchors exceed95% training-target feasibility. Baseline state RMS is0.00654469602; the best actual512/ridge1fit is0.00799488842, worse by22.16%, with worse tail error. It fails the predeclared improvement requirement. No arithmetic checkpoint or meter is produced.

A135 fits a local native3 map at the same A132 inputs, intended for explicit transfer to A123. It uses512fixed donors/ridge0.001, strong-row ridges [1e−6,1e−5,1e−4,0.001,0.01] selecting1e−5, and retains all programme gains. Native branch RMS improves0.08067396352→0.07980131721 (1.08%); the conditional paired HOST3 state RMS changes0.00654469602→0.00654326484. This misses its predeclared5%local threshold, so no checkpoint or meter is produced. Its A132-only format bound is not presented as an A123 certificate.

Completed meter usage is294/300forward,95/100quick,1/40full; remaining6/5/39. Folder size is39.739GB before further cleanup. No rung, best/, final verification or export exists. The goal remains active; there is no blocker. A136 will independently rebuild the compact selector on the exact retained A123 input basis, then use freed capacity to revisit late text repair with a larger native feature basis than the earlier320-column A100 limit. This is a proposed experiment, not a claimed gain.


## 13:54 CET — larger late repair preserves arithmetic but misses text; two static alternatives rejected

A136v1 rebuilds the local selector on the exact retained A123 upstream basis. It uses384training and128validation windows,512fixed donors/ridge0.001 and strong-row ridges[1e−6,1e−5,1e−4,0.001,0.01], selecting1e−5. Programme gains[215,391,476] remain fixed. Native branch RMS improves0.3237486168→0.2082483667; paired HOST8state RMS worsens0.00541327151→0.00549055121. Its conservative format post-change bound is8.133436944, using recorded native features and absolute bounds for the two restored native units. It changes3,906,174scalars,max1e6, freeing974,222fromA123. One official A137mixed forward captures384word windows and128independent formats on this checkpoint; all programme features are zero on the observed words. A136 has not received a standalone official quick score.

A137v1 uses these exact native16features and the original paired HOST16states. Anchor counts[1,4,16,64,128] select the smallest geometrically eligible count4 before fitting. Donor counts[320,640,1024,1536,1728] and ridges[0.001,0.01,0.1,1,10] select640/ridge1. The fixed-reference, original-target solve lowers validation directional RMS0.00893361457→0.00836829644(6.33%) and relative state RMS0.00932559233→0.00878373210. The99th-percentile error improves0.02997830→0.02809996; maximum recorded format post-change is76.12177705. Only native16Down changes, bringing total edits to4,315,774,max1e6. Official quick H1 is2,000/2,000exact and digit-correct, but text agreement is97.7294921875%,KL0.0027197009406592964,CEchange−0.000160146116054527,allfinite. It is19text matches belowA123/A124 and is not the leader. The source is immutable after scoring.

A138 independently fits final native Down from original paired HOST states, starting from HOST Down with fixed strong-gain accommodation and one observed A136normalizer. Margin powers[0,0.5,1,1.5,2], donor counts[512,768,1024,1536,2037] and ridges[0.001,0.01,0.1,1,10] selectpower1/1024/ridge1. HOSTtop8validation matches rise8016→8018; the full protected vocabulary confirms8015→8017outof8192. Directional state RMS worsens0.01444804→0.01549358. The gain of two matches misses the predeclared minimum16, so no checkpoint or meter is produced. This is independent ofA137; it does not compose a repair calibrated on stale A137inputs.

A139 tests a different feature family at native16: word-active input directions, derived by one Schur-complement/SVD construction after1024native donor features. A transferred inverse final-format gate and two-class mean gates with format covariance masses[0,1,100,10000,1000000],ridgefraction1e−6, are checked only on word/format labels. The selectedmass1gate has minimum word activation32.09999983 and maximum output-format activation−38.01968014, with maximum weight28.01077461. The frozen tanh-GELU differs from the signed linear ideal by less than1e−100at|g|≥32; other format positions use a conservative0.25*abs(Up)bound. Native activations are never replayed. Four weak HOST-state anchors, feature-direction ridge0.001, feature ranks[0,8,32,64,128] and Down ridges[0.001,0.01,0.1,1,10] are declared. Added input features worsen reconstruction. The best eligible choice isrank0/ridge1, directionRMS0.00837322169 and formatbound99.09604, failing the predeclared15%improvement threshold. No A139checkpoint or meter is produced.

Completed usage beforeA140 is295/300forward,96/100quick,1/40full; remaining5/4/39. cleanup_a136_space_v1.json retires1,040,723,152bytes of historical native2/native6 observations. cleanup_a140_space_v1.json hashes and retires3,926,695,787bytes of superseded early failed-design calibration captures. Current bases, all checkpoints, sources, manifests, requests, scores and diagnoses remain.

A140 is collecting two matched branch observations on a new calibration seed202611101351: original HOST and leading A123,256word windows with192training/64validation. The edited pass also contains128independently selected format probes. This tests direct closed-form normalization-gain repair of each native branch against its original HOST counterpart, with all programme outputs protected. It is a new experiment, not a result. No rung, best/, final verification or export exists. Work remains active with no blocker.


## 14:22 CET — scalar gain and energy repairs rejected; full-feature readout repair running

A140 records256new calibration windows(seed202611101351) on both HOST and A123. The edited capture also includes128independent format probes. Every recorded programme feature remains zero on these words. The fit uses192training/64validation windows and33branches: all MLPs, plus attention branches except the edited readers2/3/7. Each gain is one direct original-HOST branch least-squares estimate; all programme output gains stay fixed, with native-only Down scaling for the high-gain writer rows. Shrinkage fractions[0,0.25,0.5,0.75,1] select1locally. Aggregate relative branch RMS changes0.02468660260→0.02466812960, only0.07483%better; the maximum local format post-change is57.42800153. This fails the predeclared2%threshold, so no solution artifact or candidate is produced. The original top-k requests recorded only the last position; the corrected baseline is253/256matches(98.828125%) on those positions, not a full-window text score. A denominator-label error in the first analysis log was corrected before any candidate; both logs remain.

A141 is a nonlinear native-energy construction at the final MLP. It first reconstructs displaced native features with512donors/ridge0.001, then independently fits the96strong-row native numerators at the fixed original HOST native normalizer. Strong and anchor ridge lists are[1e−6,1e−5,1e−4,0.001,0.01]. The existing nearly silent coordinate439 receives a single square-root energy-remainder target, followed by one analytic training-energy trace correction. All programme columns, G/Up, writer gains and the raw programme anchor are fixed. Ordinary gains return to HOST. All training and validation energy remainders are nonnegative; no target clipping is exercised. The selected anchor ridge1e−5 has trace scale1.0000235758. Native branch RMS improves3.119456983→2.781683715(10.83%), and relative raw RMS error improves0.01534065067→0.00516803141(66.31%). However, full-vocabulary calibration matches fall8024→8016outof8192. It is rejected before checkpoint/meter. An unmetered format-bound reference was corrected to the actual recorded A124final state; the rerun reaches the same rejection, with both logs retained.

A142 records the exact leading A123final native features on512calibration windows:256old windows plus256from the new A140seed. Training is the first384windows; validation is the final128new windows. The paired original HOST final states already exist. The old HOSTtop8 observations remain metered; new HOSTtop8 scores are a separately labeled protected linear/RMS reconstruction from the saved HOST final residual. That reconstruction agrees on all256last-token positions independently recorded by the HOST meter. It is not an additional model execution or meter pass.

The proposed A142repair uses all2037surviving native features in the existing96strong output coordinates. A fixed training metric from HOST winner-minus-runner-up embedding differences permits a separable full-rank solve, avoiding A83's small feature-by-output interaction allocation. Native ridges[0.001,0.01,0.1,1,10], metric ridge fractions[0,0.001,0.01,0.1,1], and correction strengths[0.25,0.5,0.75,1,1.25] are declared. Every solve uses the same original HOST normalized-state target and fixed original A123normalizers. Programme coefficients and all gains remain fixed. It is running; no result or checkpoint is yet claimed.

Completed usage is298/300forward,96/100quick,1/40full; remaining2/4/39. A123/A124remain the leading completed quick checkpoints, still below98.5%text agreement. No rung, best/, final verification or export exists. The goal remains active; no blocker is claimed.


## 14:37 CET — exploratory projected repair gives a small official gain

A142v1 selects native ridge0.001, boundary-metric ridgefraction0.01 and correction strength0.25. Full-vocabulary fresh calibration agreement rises7,990→8,003outof8,192, exactly matching the restricted alternative check. The maximum native raw-norm change is1.4728641e−10 and the recorded format margin lower bound is1.300927279. Validation boundary-score RMS improves0.1175480578→0.1172611517, while normalized-state RMS changes0.01451166719→0.01456317683. Its13-match gain misses the predeclared16-match screen, and V1produces no checkpoint.

A142v2 explicitly changes only that screening policy for an exploratory official test. It replays the single immutable V1selection, introduces no trial values or new selection, and asserts identical baseline/selected calibration counts. The V1failure remains recorded. This is disclosed as a post-screen exploratory decision, not a claim that the16-match screen passed. The checkpoint has4,880,396changed scalars,max1e6,SHAf929a4d2adf87e4a49ada1e457d30a1d856afbec503b1fd4c963bc88037e7f25. Its official quick score(run0753-score) is2,000/2,000exact and digit-correct,97.8759765625%textagreement,KL0.0026248140396173845,CEchange−0.0005558125649711698,allfinite. It gains five held-out quick text matches overA123/A124 and improves both KL and CE slightly, becoming the working leader. It still misses98.5%; no H1pass or best/ checkpoint is claimed. V2and its included V1/readout-source dependencies are now immutable.

A143 uses the same fixed originalA123input/target pair, independently ofA142v2. It compresses all2037native features with one whitened original-HOST-error SVD and then solves larger exact feature-by-readout margin equations. The five input/output allocations are[64,96],[96,80],[128,64],[256,32],[512,16], totaling6144–8192parameters. Native-feature whitening ridge0.001, marginfloor0.1, maximum normalized weight100 and solve ridges[0.01,0.1,1,10,100] are declared. All programme columns, G/Up and gains remain fixed. The first four allocations show at most six additional restricted validation matches; the last allocation is still running. No completed A143candidate or meter is claimed.

A144 is prepared but unrun. It independently expands the text-correction output coordinates beyond the96existing high-gain writers. A fixed boundary-metric Schur pivot, divided by observed native raw-energy cost, chooses up to80ordinary coordinates. Extra-coordinate counts[0,16,32,64,80], native ridges[0.001,0.01,0.1,1,10], strengths[0.25,0.5,0.75,1,1.25], fixed metricridge0.01 and raw-energy-floorfraction0.01 are declared. The actual recorded-feature normalization/readout check accounts for ordinary-coordinate energy coupling, with maximum relative raw-norm change0.02 and a strict edit-budget check. This is a proposed construction, not a result.

Completed usage is298forward/97quick/1full; remaining2/3/39. No rung, final verification or export exists. The goal remains active with no blocker.


## Update at 10 November 2026 14:58 CET — larger readout repairs rejected; full leader test running

A143 completed its five declared input/output allocations and all five ridges. The selected allocation was 96 input directions and 80 output directions, ridge 10. Exact full-vocabulary calibration agreement increased only from 7990 to 7996 of 8192 targets, below its predeclared eight-match requirement; no delta or meter was produced. Weighted boundary RMS changed from 0.1313660905 to 0.1311577805, while state-direction RMS worsened from 0.0145116672 to 0.0147721509. Peak resident memory was 7,663,632 KiB, wall time 6:04.65. This is a rejected static construction.

A144 completed the separate expanded-coordinate construction. It chose 64 extra output coordinates, ridge 10 and strength 1.25 from its previously declared grids. Calibration agreement increased from 7990 to 8010 of 8192 targets. The checkpoint has 4,977,995 changed scalars, maximum absolute weight 1e6, SHA256 20a6a455898f02dabf86635ac2015054da082ac401b8935d73a0836349a62845. Official quick run 0755-score retained 2000/2000 H1 additions, but text agreement was only 97.808837890625%, KL 0.0026255738845788593, CE change -0.0003843607289475415, all finite. It lost eleven text matches relative to A142v2 and is set aside. Peak builder RSS was 5,468,908 KiB. Its source is now immutable.

A142v2 remains the working leader and is undergoing its first full H1/text/noise test: run 0757-score, exec session 49276, output scores/a142v2_full_H1.json, log artifacts/a142v2_full_H1.log. No result or rung pass is yet claimed. Completed usage is 298 forward, 98 quick and one full; the running second full has reserved one unit, so the ledger currently leaves 2 forward, 2 quick and 38 full. No final verification or export exists. The next structural idea is to reselect a smaller final writer-coordinate set using protected vocabulary geometry and recorded native raw energy; no such new checkpoint has been built.


## Full A142v2 result, 10 November 2026 15:15 CET

Full run0757-score completed, recorded in scores/a142v2_full_H1.json and ledger.jsonl. Checkpoint SHA256 f929a4d2adf87e4a49ada1e457d30a1d856afbec503b1fd4c963bc88037e7f25; 4,880,396 changed scalars (4.86456786% of eligible), maximum absolute weight 1e+06. H1 exact and digit-prefix accuracy are 24,990/25,000 (99.960%). This exceeds the arithmetic requirement, but the rung FAILS text agreement: 97.8492736816% across131,072targets/2,048windows, versus required98.5%. KL=0.0027827720173035342, CEchange=-0.0009651856387842273; all values finite. Compared with A117 full text97.576904296875%, this improves agreement by0.272369384765625percentage points. It remains about0.651percentage points short of the gate.

Noise is materially less robust than A117: sigma1e-5 gives2000/2000exact and digits; sigma1e-4 gives1574/2000exact(78.7%) but2000/2000correct digit prefixes; sigma1e-3 gives261/2000exact(13.05%),1611/2000correct digit prefixes(80.55%). The1e-4 result isolates a termination weakness at the output level but does not by itself locate the faulty layer. Two-in-a-row gives1999/2000first,1/2000second given correct first, and1/2000joint. This checkpoint is specialized to a single problem immediately after BOS and does not establish prefix or repeated-problem support. No rung has passed; no best/ checkpoint, final verification or export exists.

Completed usage is298forward,98quick,2full; remaining2/2/38. A145 protected literal-span geometry is now running as session36034, logartifacts/a145v1_literal_span_geometry.log. Its compiler is prepared but unrun. Current leader remains A142v2, with the full failure and noise limitations above; no blocker is claimed.


## A145 outcome at 15:19 CET — smaller literal span works, text screen fails

QR leverage of the five-function emitter admits64,80and96writer-coordinate designs. The64-coordinate version uses2.010155097%of native normalized raw energy, versus2.752427531%in the old96-coordinate pool. Its selected bias2/rawanchor2 geometry has literal margin1.30286598 and dominant actual-format margin1.26122665. The48-coordinate literal geometry fails; the112-coordinate nuisance check fails. All declared floor/count/bias/anchor trials are in artifacts/a145v1_literal_span_pool.json. Geometry peakRSS1,789,728KiB,wall58.24s.

The native reconstruction selects64coordinates, original-HOST reconstruction and strong ridge1e-5. Local branch RMS improves3.119456983→2.695632404(13.5865%). However exact protected-head calibration matches change8024→8020on the old128validation windows and7990→7996on the fresh128windows: only two additional matches combined, below the predeclared eight. It is rejected without delta or meter. The compiler did not reach its final actual-format/native-mixture check; only the earlier dominant-programme geometry is established. Native builder peakRSS4,588,044KiB,wall54.50s. Both A145processes are complete; no heavy process remains at this update. A142v2 remains leader and has no H1pass.

cleanup_a146_space_v1.json hashes and retires the historical HOSTdetail calibration activation artifact to leave room for further work; original requests, ledger, scripts and diagnoses remain. No frozen host, meter, text or ledger file was changed. The next construction is the previously recorded paired quadratic text-feature design, independently based on A123, not a residual refinement of A142v2.


## A146 result and A147 allocation change, 15:29 CET

A146completed in1:13.84, peakRSS6,732,100KiB. The zero-quadratic control reproduces the A142static gain:7990→8003of8192fresh calibration targets, selectedridge.001/strength.25. The best positive allocation has4products/8newunits, ridge.1/strength.75, only8001matches. Its boundaryRMS is.1250089576 versusbaseline.1175480578, directionRMS.01531728 versus.0145116672, and maximum raw-normalizer change1.63957%. Larger allocations degrade further. It fails the required eight-match gain over control; no delta or meter. All gate/format margins and all125trials remain in experiments/a146v1_quadratic_text_features/quadratic_text_repair.json and the log.

A147is a separately versioned allocation correction, not a refit of A146weights. Its fixed PCA/product selection, originalHOSTtarget, all ridge/strength/count grids and paired-gate constraints are unchanged. For each of the512candidate donor units it computes the exact mean squared post-RMS branch displacement of removing that recorded feature using norm/dot identities. This includes lost denominator energy, which the A146direct-output importance omitted. The quietest64units define its allocation order. No derivatives or model/activation replay are used. Newsourceexperiments/compile_a147_normalized_cost_quadratics_v1.py, running session36501, logartifacts/a147v1_normalized_cost_quadratics.log. No other heavy process; A146session18836done. A142v2remains the leader; budgets298forward/98quick/2full, no pass or blocker.


## A147/A148 rejected; A149 tests independent pair outputs, 15:38 CET

A147 exact normalized branch-cost allocation gives the same measured predictions as A146:8001/8192for its best four-product repair, versus8003for the zero-product control. It fails the fixed eight-match improvement requirement. Wall1:14.20,peakRSS6,754,032KiB; no delta/meter. The existing512donors are comparatively active native features: even their quietest single-feature removal has mean squared branch displacement1307.36(summed over640coordinates).

A148 removes that allocation restriction and computes the same exact removal cost for all2037native units in bounded128-column batches. The64selected units overlap the512donors in only12positions. The minimum cost becomes1036.37, so no nearly cost-free native units appear. Best positive allocation remains four products, ridge1/strength.75, but full-vocabulary matches are7983/8192versus8003control. It too is rejected before delta/meter. Wall1:13.68,peakRSS6,760,820KiB. Counts/ridges/strengths, originaltarget and feature selection remain those declared for A146; only allocation domain changes.

A149 is a separate structural feature-basis extension using A147's donor allocation. Each pair now has independent Down coefficients, so both its difference u*v and its sum k*g0*v are fitted jointly. It replaces2n native features with2n new features rather than enforcing opposite Down columns and retaining onlyn products. Actual float32 plus/minus Down weights are quantized before their equivalent sum/difference features are evaluated. Original HOSTtargets and fixed normalizers remain unchanged; no residual refit occurs. It keeps all existing count/ridge/strength grids and eight-match-over-control screen. Newsourceexperiments/compile_a149_independent_paired_features_v1.py, CURRENT heavy session56717, logartifacts/a149v1_independent_paired_features.log. A148session59794andA147session36501are done. No new checkpoint or meter is claimed, and budgets remain298/98/2.


## A149 rejection and paired early-layer capture, 15:52 CET

A149's independent pair sum/difference coefficients do not improve the selected predictions:8001/8192for four pairs/ridge.1/strength.75, versus8003control. BoundaryRMS.12500692, directionRMS.01531689, maximum raw-normalizer change1.63957%. It fails the predeclared screen; no checkpoint/meter. Wall1:13.53,peakRSS6,790,052KiB. All A146..A149feature-family failures are retained rather than presented as gains.

A150derives original HOST residual states1..8from the already recorded A140raw attention/MLP branch outputs, using only RMS arithmetic and residual sums with the fixed embedding scale. The reconstructed final residual is BIT-IDENTICAL to A140's saved finalraw (relative and maximum errors0). artifacts/a150v1_derived_host_states.pt is explicitly derived factual data, not a new model run; its report/source hashes preserve provenance. Sourceexperiments/derive_a150_recorded_host_states_v1.py.

Official forward0759-forward then captures the exact current A123native MLP features and pre-MLP residuals in layers1..8on256paired calibration windows and128independent format probes, matching the A140requests. artifacts/a150v1_early_native_basis.pt contains1,563,426,816tensor bytes, below the meter cap. All model execution remains inside tools/forward.py. A142v2's upstream layers are identical to A123. Completed usage is now299forward/98quick/2full, remaining1/2/38. Source/requests/layout: prepare_a150_early_native_basis_v1.py, a150v1_early_native_basis_request.json, artifacts/a150v1_early_native_basis_layout.json. Forward48353andHOSTderivation71040are complete.

CURRENT heavy: analyze_a150_early_native_geometry_v1.py, session83481, logartifacts/a150v1_early_native_geometry.log. It checks current paired state drift and original-HOST ray feasibility at each early layer, with weak-anchor counts[1,4,16,64,128] and fixed positive ray interval[.5,2]. No new weights or repair fits yet. Programme features must be exactly zero on calibration text. cleanup_a150_space_v1.json hashes/retires the superseded A97fresh gate audit; current A117fresh audit remains. No frozen sources, host, heldout or ledger were altered. Leader remains A142v2; no rung/best/final verification/export or blocker.


## A151 rejects every early independent fit; A152 tests norm-orthogonal geometry

A151completed all eight layers in both leader and compact allocation layouts. No candidate reaches its five-percent state-and-direction threshold with the budget/format guards; no delta is saved. Best local relative-state-RMS improvements, even before eligibility checks, are1.176%,0.966%,2.811%,-4.336%,1.254%,1.316%,1.797%for layers1..7; layer8worsens0.242%on the leader layout and improves1.498%on compact. The most improved layer3 uses all2044native columns/ridge1, and is not an eligible leader-budget candidate. Full grids and guards remain in artifacts/a151v1_paired_early_state_repairs.json. Wall2:02.63,peakRSS5,001,420KiB. No manifest-input completion is needed because no checkpoint exists.

A152is a genuinely different original-target geometry: a closed-form diagonal-metric projection of the originalHOSTstate discrepancy into raw branch directions orthogonal to the fixed observed raw vector. Regularizer fractions[1e-6,1e-4,.01,1,100]multiply the median ordinary-gain squared; the best eligible ideal geometry is selected before a weight fit. Each independent native Down solve retains the same frozen native-map prior shared by A142v2/A136, uses the originalHOSTtarget and fixed observed normalizer, and never uses A151weights or a previously fitted candidate residual. Layers1..7, donor counts[256,512,640,1728,all native], ridges[.001,.01,.1,1,10], and strict five-percent state/direction, budget, tail and local-format guards are declared. All G/Up/gains/programme columns stay fixed. This is a projection/one-shot ridge construction, not gradients or iterative fitting.

CURRENT heavy A152constructor session48446, logartifacts/a152v1_norm_orthogonal_state_build.log, sourceexperiments/compile_a152_norm_orthogonal_state_repair_v1.py. A15121489andA150geometry83481are done. No new checkpoint yet. All actual model execution remains in the frozen meters. Budgets299forward/98quick/2full (1/2/38remaining). Folder was39,688,032,241bytes after the A150capture/derived HOSTstates; enough room for the two possible small deltas, but recheck before any larger artifact. A142v2is still leader; no rung has passed, and no final verification/export or blocker exists.


## A152 rejected; A153 relocates unchanged final functions

A152completed in56.20s,peakRSS4,739,416KiB, with no checkpoint in either allocation. None of its independent norm-orthogonal repairs reaches the required5%state/direction improvement. Best local state improvements are approximately1.19%,0.68%,2.79%,1.90%,1.31%,1.40%,2.09%for layers1..7, often requiring more than the leader's edit budget and sometimes failing the source-format bound. This rules out the tested small geometric correction family on this calibration split, not all possible early repairs.

A153now tests allocation of the existing11final arithmetic functions, not added text features. A90selected its original emitter from a restricted31-unit historical pool. A153considers all1536units outside the512fixed donors, using exact current-input originalHOST17features from A122. A fixed .001ridge donor projection defines conditional feature variance; normalized post-output cost plus raw-energy masses[0,.01,.1,1,10]select11slots. Each allocation independently reconstructs the original native HOSTbranch (512donors/.001; strong ridges[1e-6,1e-5,1e-4,.001,.01]). ProgrammeG/Up/Down functions are copied bit-exactly to the new slots. It requires5%native RMS improvement and8extra full-vocabulary old-validation matches relative to A142v2. Newly restored native functions on format probes are bounded with abs(GELU(g))<=abs(g)+.25 and RMS perturbation inequalities; they are never evaluated through an activation outside the meter. Minimum certified full-vocabulary format margin must exceed1.

CURRENT heavy A153constructor session80820, sourceexperiments/compile_a153_conditional_emitter_reallocation_v1.py, logartifacts/a153v1_conditional_emitter_build.log. A15248446done; all other sessionsdone. NoA153checkpoint or score yet. Budget remains299forward/98quick/2full, no pass or blocker. The16:10timestamp in the previous provisional pointer was approximate and written at16:05; this entry supersedes its running-session information.


## A153 rejected; conditional cleanup geometry next, 16:28 CET

A153 completed in18.35s, peakRSS4,713,224KiB. Across all1536eligible native slots, cost masses0/.01/.1 choose the same11programme units already used. Masses1/10 replace only two slots and are worse. Best original-HOST reconstruction gives native branchRMS3.292982251 versus A142v2 baseline3.355740197, below the required5%gain. No full-vocabulary or format bound was reached, no delta/meter was produced. experiments/a153v1_conditional_emitter_reallocation/conditional_reallocation.json retains the result.

The next geometry study revisits the existing six cleanup functions. A115 folded three exact weight-span functions, then applied unconditioned PCA to eight remaining field features. Current A122 uses five emitter functions and six cleanup functions, with only the first two emitter gates equal to the cleanup gate. A154 will project the fixed cleanup linear operator onto those two existing inputs under recorded hand-field covariance, then test lower residual ranks with post-gain weighted PCA. This is compression of the existing cancellation operator, with no problem-answer targets or model replay. Independent recorded format probes and full-vocabulary margins decide whether any geometry is worth a native-text reconstruction.


## A154 conditional cleanup geometry, 16:35 CET

A154 projects the fixed cleanup operator onto the two existing shared-gate emitter inputs under recorded hand-field covariance, then applies post-gain/final-norm weighted PCA to the residual. Ranks[0,2,3,4,5] are declared; rank6 is an exact-algebra control. Using384training and128independent validation format windows, ranks0/2/3 change440/293/47of2560winners; rank4 changes0with minimum margin1.296527863 and validation minimum1.359045029. Rank5 also preserves all, as does the control; baseline minimum1.301513672. Rank4 reduces final programme count11to9. Relative rawprogramme change.00601409. All arithmetic was on captured unchanged-gate factors and fixed weight operators, with no native activation replay or answer targets. artifacts/a154v1_conditional_cleanup_geometry.{pt,json} stores this result. Wall21.96s,peakRSS1,697,848KiB.

The A154constructor independently fits originalHOSTnative behavior with nine programme units, retains A153's five allocation masses and fixed native ridge grids, and requires5%native branch improvement,8additional old-validation text matches, and certified format margin>1before writing a checkpoint. No meter is spent by these static constructions.


## A154 text-screen rejection; A155 geometry; A156 combined design, 2026-11-10 16:33:08 CET

A154's nine-neuron native reconstruction passes the5%branch screen: RMS3.118303977 versus3.355740197, selecting mass0/strongridge1e-5. Full-vocabulary validation gains only5matches(8023to8028of8192), below the required8. It stops before the format uncertainty certificate and saves no checkpoint. Wall28.26s,peakRSS4,714,756KiB.

A155 extends conditional operator compression from two shared-gate emitter functions to all five existing emitter functions, using the algebra YAB^T+XC(W^T-AB^T). The three hinge activations are recorded unchanged features, not replayed. Ranks0/2/3 change696/287/44format winners; rank4 again changes0with minimum1.299390793. This gives no further neuron reduction, so no native reconstruction or checkpoint is attempted. The rank6control recovers baseline1.301513672.

A156 now combines the independently derived A154nine-function cleanup with A145's64-coordinate writer. It separates fixed cancellation and emission pieces, rescales cancellation by A145's analytic normalization ratio, then inserts the64-coordinate protected-head basis and raw anchor. All previous upstream and programme gates stay fixed. It retains the5%native-RMS,8extra old-validation matches, and certified format-margin>1screens. This is a structural combination of two original geometry analyses; no failed native residual is fitted or reused.

The preceding approximate16:28/16:35headings were future estimates; this machine-read timestamp supersedes them. No new meter has been used; usage299forward/98quick/2full.


## A156 rejected; A157 native-query restoration, 2026-11-10 16:39:21 CET

A156 combines A154rank4cleanup and A145pool64. Native branchRMS falls3.355740197to2.469234279(26.42%), selecting allocation mass0/strongridge1e-5. Despite the local improvement, all-vocabulary calibration matches remain8023/8192, equal to A142v2, below the required8extra. No checkpoint, format certificate or meter follows. Wall28.19s,peakRSS4,709,740KiB. The stronger local branch result does not establish better text preservation.

A157returns to the A-reader attention query. It restores HOSTordinary gains and inverse-gain HOSTnative query numerators in the64routing channels; mean-cancellation fractions[0,.25,.5,.75,1]and selected-gain multipliers[1,1.25,1.5,2,4]are declared. Each head's programme amplitude comes from a quadratic source-position routing inequality with fixed margin8. Selection uses centered causalQKscore MSE at current word inputs, with128old and128fresh validation windows, no softmax/attention replay or sum targets. Fresh3.input is derived from already recorded layer2residual/MLP output and fixed RMS gains, with exact reconstruction of recorded layer3pre-MLPresidual as a factual identity check. Routing covers1536historical and128exactcurrent format probes. Sourceexperiments/analyze_a157_native_query_restoration_v1.py; session62302, logartifacts/a157v1_native_query_restoration.log. It requires5%score-MSE improvement and fresh-set improvement before retaining candidate weights.

All meters/budgets unchanged(299forward/98quick/2full), no rung passed. A142v2remains the official leader.


## A157 rejected; joint value/decoder algebra A158, 2026-11-10 16:44:29 CET

A157's initial run stopped on an unchanged-parameter dictionary lookup; the still-unmetered source was fixed to use the HOSTfallback, and run2completed. The factual layer3residual identity is bit-exact on both word and format captures. Current query decomposition error is2.008e-8. Baseline centered causal scoreMSE is.000907255368(old.000881147077,fresh.000933363659). Best restored candidate, mean fraction1/gain multiplier1, is much worse at.01239546030. All routing margins remain approximately8, but no candidate passes the5%improvement screen; no weights/checkpoint/meter. Run2wall24.94s,peakRSS1,825,376KiB.

A158jointly chooses the rank-one native V update and its decoder. The current A120construction first fits a decoder in the HOSTvalue span, then chooses the least-cost update direction. A158uses fixed input/output covariance factors from actual recorded attention mixtures. For V'=V+U(L^T-A^T V), A^T U=1, minimizing overU gives a generalized Rayleigh quotient inA. A Schur complement and one eigendecomposition solve its homogeneous form. Raw/output metric masses[.0001,.001,.01,.1,1]and the five lowest eigenmodes[0,1,2,3,4]are declared. Individual-source frame fitting retains background masses[0,1,100,10000,1000000]and fixedwithin-source mass10000. No problem-answer labels, gradients, iterative fitting or new attention probabilities are used.

A119recorded head mixtures remain exact because input/Q/K and all upstream weights match the current candidate; only linear V/output/RMS arithmetic changes. The fixed digit rowLcomes from individual-token constraints. Current A142V supplies the comparison baseline, so a claimed gain must exceed10%over its branchRMS. Source-frame precision and wordprogramme gate quietness are required. Heavy session29134; sourceexperiments/derive_a158_joint_value_decoder_v1.py, logartifacts/a158v1_joint_value_decoder.log.


## Joint value family cannot expose a precise digit frame; A160native decoder, 2026-11-10 16:53:08 CET

A158completes all25mass/eigenmode combinations but none has an eligible source frame. Current A142native-value branch RMS is.09891509696. The least branch-error joint candidate(.1mass/mode0) reaches.03284115618, but its best source frame has maximum digit error6.6362. The closest frame is mass.01/mode2: nativeRMS.05052803964 and maximum digit error1.46419, still well beyond.35. Thus no value weights are retained or checkpoint built. Wall34.23s,peakRSS3,962,780KiB.

A159keeps those25independent value constructions and expands source-frame penalties to within-source masses[1e2,1e4,1e6,1e8,1e10]and background/within ratios[0,1e-4,.01,1,100]. All625frame trials fail. The nearest maximum digit error remains1.46419, at the previous1e4within/ratio1geometry. The native-reference constant stays near1; the failure is digit contamination and residual background, not a lost reference. Wall36.67s,peakRSS3,955,168KiB. No checkpoint/meter.

A160is a distinct source-only linear decoder with HOSTV/Ounchanged. A39formerly enforced every individual digit-class mean exactly while penalizing within-class variation. A160instead fits all individual-source residuals in one closed-form solve, permits per-head offsets, and enforces only their summed mean. Labels are each source token's digit, never a sum or assembled numeric problem. Background states enter only a zero-response covariance penalty. Word-covariance regularizers[1e-8,1e-6,1e-4,.01,1], background masses[0,.01,1,100,10000], fixed diagonal1e-12times source covariance mean, and common Up gains[1e3,1e4,1e5,3e5,1e6]are declared. It requires maximum individual digit error<.4and constant>.5across three historical512-format sets plus128exactcurrent format states derived from A150/A140recorded branches. Sourceweightsarefittedonlyfrom the900individual-source prefixes. Heavy session88732, sourceexperiments/derive_a160_native_value_source_decoder_v1.py, logartifacts/a160v1_native_value_source_decoder.log.

Budget299forward/98quick/2full; no pass or blocker. All model execution remains in the frozen meters; current analyses use recorded values and linear/RMS/hand-circuit geometry only.


## A160/A161near misses; A162joint-value candidate passes geometry, 2026-11-10 17:00:44 CET

A160's best HOSTV-only decoder has no incorrect rounded source digits across all1664format probes, but maximum individual digit error.4800434646 exceeds its fixed.4screen. Regularizer1e-8/backgroundmass1, constantminimum.9855148511; no checkpoint. A161weights each output decoder's own source head by[1,2,4,8,16], keeping A160's other scales fixed. Independent column selection chooses[1,1,16], with errors[.4800434646,.3686654859,.0803150351]. It also fails. No meter for either.

A162keeps each A158joint eigenmode's native removal projector fixed but scales the inserted literal digit code independently, with reciprocal decoder scaling. For each of the25original mass/mode geometries, reference gain=max(1,A158bestsourceerror/.25); only factors[.5,.75,1,1.25,2]are tested. The125value constructions and their original5background-frame masses are independent HOST-derived weight algebra, with no residual refits or sum labels. This distinction matters: source leakage shrinks as the digit carrier strengthens, while the quiet native-axis removal stays fixed.

The selected candidate uses raw metricmass.01/eigenmode2, referencegain5.8567643393,factor.75,actualcodegain4.3925732545. Native value-branch RMS falls.09891509696to.06319092970(36.12%). Maximum recorded word programme gate3is-31.97825334. Sourceframebackgroundmass10000has no wrong individual digits; maximum errors are[.27926737,.22669938,.20326146]on syntax_a,[.24932134,.20546210,.16235487]on syntax_stress,[.34825370,.33624237,.24167513]on syntax_c. Minimum reference.98684898. This passes the original.35source screen and10%native-branch screen. Native Vmax.211903736, value-span roundingerror2.595e-9. This is still local/ideal geometry, not an official prediction claim.

Analysis46488complete; constructor66166running, sourceexperiments/compile_a162_scaled_joint_value_decoder_v1.py, logartifacts/a162v1_scaled_joint_value_build.log. It changes only layer3V and four arithmeticUp rows from A142v2, with common Up gain chosen from[1e3,1e4,1e5,3e5,1e6]. A quick H1score is next if the build succeeds. Current usage299forward/98quick/2full; official leaderA142v2, no rung passed.


## A162official quick fails; expanded final-direction repair A163, 2026-11-10 17:10:54 CET

A162builds successfully: experiments/a162v1_scaled_joint_value_decoder/delta.pt, SHA1444371b4c3724fc07e12f1cab974f2275c910fc8aaa2f1969aff548eb89af70,4,880,397changed scalars,max1e6. Builderwall8.30s,peakRSS961,736KiB. Its sources and dependencies are now immutable. Official quick0761-score scores1976/2000exact and digit prefixes(98.8%), text97.86376953125%, KL.0024383923814385775, CEchange-.00040622910932786027, allfinite. Text is two matches below A142v2quick, although KL is smaller. Arithmetic fails the99.5%threshold; several failures are neighbouring digits, so its ideal source.35screen did not guarantee complete-circuit precision. No pass or bestcheckpoint. Usage is now299forward/99quick/2full, remaining1/1/38. A142v2remains leader.

A163uses A136's compact selector allocation and its exact recorded final features(A137basis), which leave room for larger final output repairs. It targets original paired HOSTnormalized final states at fixed A136raw/final RMS, starting from the fixed A136native map. This is a new original-target construction, not a residual fit to A138/A142candidate weights. A fixed HOSTwinner-minus-runnerup metric projects target differences into coordinate counts[96,256,384,512,608]. Extra coordinates are ranked by conditional boundary-metric variance times squared post gain. Projection regularizers[0,.001,.01,.1,1]include a diagonal raw-numerator penalty; native ridges[.001,.01,.1,1,10]and strengths[.25,.5,.75,1,1.25]are declared. ProgrammeG/Up/Down columns and all gains stay fixed.

A163requires editbudget compliance, raw-normalizer change<5%maximum/<1%atp99, certified recorded-format margin>1, and at least16additional full-vocabulary matches on128calibration validation windows. Training is256paired calibration windows. These linear/RMS calculations use recorded features only; no Gemma replay. Heavy session69860; sourceexperiments/compile_a163_expanded_projected_final_v1.py, logartifacts/a163v1_expanded_projected_final.log.


## A163rejects the larger final repair; A164boundary-aware allocation, 2026-11-10 17:16:58 CET

A163completes625independent fits in2:59.52,peakRSS4,356,464KiB. Best eligible repair uses384output coordinates,nativeridge10,metricridge.1,strength.75,4,345,366changed scalars. HOSTtop8matches improve8016to8022; full-vocabulary matches improve8015to8021of8192, only6rather than the required16. BoundaryRMS.120025393to.117997559; finaldirectionRMS.0144480373to.0144230822. Maximum raw-normalizer change.1183734%,p99.0159723%, formatmarginbound1.301088113. No checkpoint or meter. This includes all608-coordinate cases; they do not beat the selected restricted repair.

A164changes the arithmetic-unit allocation metric. A153/A90used uniform post-output norm times conditional feature variance. A164uses the fixed original HOSTwinner-minus-runnerup readout coefficient at each training token, inverse original HOSTmargin weights(floor.1,clip100), and fixed original HOSTnormalizers. The conditional residual still comes from one512-donor/.001ridge projection. Five raw-energy masses[0,.01,.1,1,10]select independent11-unit sets from1536candidates; programmefunctions are unchanged. Each candidate's native reconstruction starts from HOST. Selection uses validation HOSTtop8matches and boundaryRMS, with local nativeRMS capped at1.25times baseline; full-vocabulary gain must be at least8 and the restored-native format uncertainty certificate must exceed1. No iterative unit moves, derivatives, model execution or sums enter the construction. Sourceexperiments/compile_a164_boundary_emitter_allocation_v1.py, heavy session95229, logartifacts/a164v1_boundary_emitter_allocation.log.

Meter budgets remain299forward/99quick/2full; official leaderA142v2, no rung/best/verification/export or blocker.


## A164 allocation rejected; A165 paired-source geometry, 2026-11-10 17:40:00 CET

A164 completes in38.89s, peakRSS5,413,824KiB. The boundary-aware allocation selects raw-energy mass1 and retains four of the old eleven units. Full-vocabulary validation improves8023to8025of8192, only two matches against the required eight. Its native branchRMS rises3.355740197to3.776700252, while boundaryRMS falls.111485210to.108675711. No checkpoint is written and no meter is used.

A165 constructs a new paired-source positional reader at layer2. Three rotary frequencies form a kernel with zeros at offsets+/-2 and negative scores at every other H1 position. Common key-amplitude variation therefore does not change the two intended scores. Heads read positions[3,7],[2,6],[1,5], and reference[4]; only source positions are targets. Exact layer2inputs are derived from recorded branch sums, checked bit-identical against the recorded pre-MLP residual, with zero discrepancy on word and format captures. No attention probabilities or native nonlinearities are replayed.

The five pair sets are[i,i+10,127]fori0..4; query scales[.1,.3,1,3,10], key-RMS ridges[1e-6,1e-5,1e-4,.001,.01], format-readout masses[0,.01,.1,1,10], fixed keyscale.001. Programme gain is computed directly for margin8, not searched. The selected geometry uses[4,14,127], queryscale.3, keyridge.0001, readoutmass10 and gain632.33211393. Minimum source margin8.00007946; maximum intended-pair score difference.00352533413. These imply a conditional pair-weight imbalance bound below.000882, but do not demonstrate an executed arithmetic reader.

Ordinary-text centered causal QK-score MSE is.1503922998, versus.0015502555 for the existing reader on the same recorded inputs: the geometry succeeds but its preservation cost is unacceptable as constructed. This is an analysis artifact, not a delta or rung result. Sourceexperiments/analyze_a165_zero_peak_pair_reader_v1.py; artifacts/a165v1_zero_peak_pair_reader.json/.pt; wall25.48s, peakRSS1,601,716KiB. Next isolate the channel-deletion and programme-leakage contributions before deciding whether to retain the architecture.

Leader and meter budget are unchanged: A142v2 fullH1 99.96%, text97.84927368164062%; no rung passed. Completed299forward/99quick/2full, remaining1/1/38. All heavy sessions are finished.


## Paired-reader screens A166–A169; direct boundary repair A170, 2026-11-10 17:54:09 CET

A166 isolates A165's ordinary-word QK disturbance on exactly the same recorded states. Current-reader MSE.0015502555; A165.1503922998; zeroing its programme gains still.1503254380; deleting its six native channels alone.2424496124; mean-query cancellation alone.0002459842. Thus lost native channel functions, not programme leakage, dominate this prototype. Wall7.91s, peakRSS1,194,080KiB; no checkpoint or meter.

A167 ranks rotary triples analytically using original native Q/K pair energy plus raw normalization energy. Among341,376triples,3,714have the required literal two-zero kernel, integer gap>.1 and coefficient-L1/gap<=40. The five cheapest placements are[1,10,81],[1,10,82],[1,10,85],[1,10,72],[1,10,76]. These are placement calculations from original channel energies, not repeated model executions. The original A165five query scales and key ridges are reused. Best[1,10,85], queryscale1, keyridge.0001, routingmargin8.00007953, pairimbalance.00332133, but wordQK MSE.0883302462 remains much worse than current. Wall23.15s, peakRSS1,616,700KiB. No checkpoint or meter.

A168 retains the HOST key map and solves positional query directions with pair-equality penalties[1,100,1e4,1e6,1e8] and channelcounts[64,96,128,192,256]. Training uses1,120formatwindows(oldsyntaxA/stress plus96current); validation544(oldfreshsyntaxC plus32current). Labels are source positions only. Best256channels/penalty10000stillhas maximum source-pair score imbalance.85580454at routingmargin8, against.004required. No geometrypasses; wall12.00s, peakRSS1,400,676KiB.

A169 tests an exact common-nullspace key projection for the seven desired source-position query functionals, with five raw-metric masses[0,.001,.01,.1,1]. The null constraints are satisfied to<=4.2e-9after float32rounding, but at least three heads lose positive routing gaps under every mass. Ordinary QK distortion also remains large(.1545–.5366MSE). Therefore no paired-reader checkpoint is built and no meter is spent. These independent failures do not prove such a reader impossible.

A170 changes the text-repair objective itself. Each unlabeled calibration token contributes its own original HOSTwinner-minus-runner readout vector, instead of fitting full hidden states under an averaged output metric. The design is a tensor product of2,037surviving native features and96existing writer coordinates. Its exact dual Gram is the Hadamard product of their two Grams. Five independent Cholesky solves use ridge fractions[.001,.01,.1,1,10]of mean Gram diagonal; correction strengths[.25,.5,.75,1,1.25]. Output-whitening ridge is fixed.01; original HOSTmargin weighting usesfloor.1/clip100.

A170starts from the fixed original A123native map and recorded normalizers. Its target is the original paired HOSTboundary margin; it does not fit a residual from A142or another candidate. All programmeweights and gains remain unchanged. Training384wordwindows, independent fresh validation128. It requires at least16additional full-vocabulary matches over A142's calibration result, plus formatmargin>1 and maximum raw-normalizer change<.001. No gradients, iterative solve, native nonlinear replay, whole-problem targets or arithmetic answers are involved. Heavy session68903, sourceexperiments/compile_a170_token_boundary_kernel_v1.py, logartifacts/a170v1_token_boundary_kernel.log.

Usage remains299forward/99quick/2full(1/1/38remaining). A142v2remains the leader; no rung passed. Project measured39,748,165,038bytesbeforeA170.


## A170 fails; resource overrun and A171 mixture inversion, 2026-11-10 18:01:10 CET

A170 completes in2:13.80. Best dualridge10/strength1 reaches7,993/8,192full-vocabulary calibration matches, versus7,990for its fixed A123prior and8,003for leaderA142. BoundaryRMS.1165966635improves slightly but top1 does not beat the leader, so no delta or meter. All25fit/strength cases are recorded inexperiments/a170v1_token_boundary_kernel/token_boundary_kernel.json.

Resource exception: A170peakRSS was12,139,292KiB=12,430,635,008bytes(11.58GiB), which exceeds a strict decimal12GBinterpretation of GOAL.md. The planned matrix size underestimated peak factorization/workspace residency. The process has finished; it did not swap. This overrun is recorded rather than claiming full resource compliance. Future dense analyses will use smaller allocations with room for library workspace; this24,576-row dual solve will not be repeated.

Related-method correction: earlier A83already fitted token-specific margin interactions in restricted2,048-parameter allocations. A170is an expanded dual formulation of that family using all2,037native features and96writer coordinates, plus a fresh calibration split. Its direct-margin objective differs from recent A142/A163averaged-metric state fits, but is not the first margin-repair idea in this project.

A171checks whether the existing recorded attention branch outputs uniquely determine their attended values. For each sequence, the source-value span and output projection form a640by(4*sequence_length)linear system. An older capture containing actual head values independently verifies recovery; current word, original HOSTword, and current format mixtures are then checked for full column rank, low reconstruction error and causal support. Only linear algebra on already recorded data is used. No softmax, GELU or extra model pass. This may supply exact current attention contexts for a one-shot native-value repair while protecting the arithmetic readout frame. Heavy session40709, sourceexperiments/analyze_a171_recorded_attention_inverse_v1.py, logartifacts/a171v1_recorded_attention_inverse.log.

Leader, rung status and meter usage are unchanged: no pass;299forward/99quick/2full(1/1/38remaining).


## Verified mixture inverse; A172 rejected; A173 full score and A174 prepared, 2026-11-10 18:19:45 CET

A171's independent inverse check recovers actual recorded head values on eight old windows with relative error8.79007914e-8. Current256word windows and original HOST256windows both have full column rank256; format128windows have rank56. Relative raw-output reconstruction errors are8.39e-8/8.38e-8/1.32e-7; maximum spurious future coefficient4.83e-6. These are factual linear inverses of observed outputs, not new attention execution. The first analysis withheld its large output because there was insufficient disk headroom; its initialreport/log remain.

Cleanupscript experiments/cleanup_a171_space_v1.py retires only two superseded early failed-A2activation captures (a2_detail_syntax_a/b.pt), reclaiming1,013,136,210bytes. Exact sizes, SHA256, run IDs0049/0051-forward and capture shapes are retained inartifacts/cleanup_a171_space_v1.json; source scripts, sparse checkpoints, manifests, requests, logs and ledger remain. The unchanged A171analysis was repeated to persist the derived matrices, without any meter use. artifacts/a171v1_recorded_attention_inverse.pt is442,570,314bytes.

A172fits a shared A-reader Vcorrection from paired original HOSTnormalized head contexts, projected into the null of16arithmetic frame functionals. Ridges[1e-6,1e-4,.01,1,100], output-metric masses[0,.001,.01,.1,1], strengths[.25,.5,.75,1,1.25]; train192/freshval64wordwindows and128formatwindows. Best ridge.01/mass.1/strength.5reduces paired branchRMS.2169684620to.2126954930(1.97%), below the5%screen. It keeps all recorded individual source digits correct, maxerrors[.121455,.090744,.060709]. No delta or meter. Wall40.02s,peakRSS3,596,584KiB.

A173instead changes up to128rows of attention3's output matrix, using recovered actual current head contexts. Rows are ranked by original paired HOSTnormalized branch-error energy. Counts[32,64,96,112,128], ridges[1e-6,1e-4,.01,1,100], strengths[.25,.5,.75,1,1.25]; frame-null raw penalty fixed.01. Every correction is orthogonal to the entire arithmetic readout frame before branch normalization. All fits use original HOSTtargets and fixed recorded normalizers; no previous fit is used as a new residual target.

A173selects128rows/ridge.01/strength.75. Paired branchRMS falls.2169684620to.1916248390(11.68%); paired hidden-state relativeRMS.0050474361to.0048487753. Source-digit errors remain[.121473,.090684,.060769], zero wrong; minimumreference.9969854; maxformatrawRMSchange.28053%; framefunctionalerror1.87e-9. WordprogrammeGmax-33.18269. NativeOmaxweight.435546875. This is local recorded-input evidence, not a global result.

Checkpoint experiments/a173v1_frame_null_native_output_repair/delta.pt is built, SHAde003de084d84b8f01513fd1776bce7cf449f73341501aa09fe3e4f0e97262c0,5,011,468changed scalars,max1e6. It adds131,072nativeO3repair entries to A142v2. Builder23.67s,peakRSS777,892KiB. Sources and dependencies are now immutable. Full official H1score0763-score is running as session21853, logsartifacts/a173v1_full_H1.log, outputscores/a173v1_full_H1.json. Completed usage is299forward/99quick/2full, with a third full run reserved; current ledger remaining1forward/1quick/37full. No rung pass is claimed.

A174scripts are prepared but NOT BUILT OR METERED. The intended final forward call can restore HOSTG/Up at each displaced arithmetic neuron while zeroing its Downcolumn. If all original programme gates are inactive on the captured calibration inputs, induction proves the ordinary-text trajectory unchanged while exposing original native features. The analysis requires conservative float32 gate upper bounds below-16before accepting that equivalence. It will capture all2,048native features and pre-MLP residuals at layers0..8and17 on256paired calibration windows, plus finalraw/top8:1,803,550,720planned capture tensorbytes. The score result will determine the base checkpoint; more old activation storage must be retired before the capture. Prepared sources:experiments/prepare_a174_native_feature_diagnostic_v1.py andanalyze_a174_native_feature_diagnostic_v1.py. This diagnostic is not an addition candidate.


## Remaining prepared analyses during A173score, 2026-11-10 18:32:59 CET

A173's main H1arithmetic evaluation has finished24,989/25,000exact anddigit-prefix correct(99.956%). Text/noise/two-problem evaluation is still running; no rung or new leader is declared. Session21853remains the sole heavy process.

A174now has a prepared certificate analyzer. It verifies that diagnostic G/Up are exactly HOST, programmeDowncolumns are zero, and all other weights equal the chosen parent. Recorded raw residuals reconstruct the original programme gates; a conservative650-epsilon dot-product bound must stay below-16at every calibration token. Only then may original native features be treated as observations at the parent checkpoint's inputs. This supplies exact same-input HOSTnative branch targets without evaluating native GELU outside a meter. The analyzer also checks unchanged early inputs, final residual reconstruction and paired calibration top1. It has not run.

Additional prepared cleanupscript experiments/cleanup_a174_space_v1.py has NOT run. It will retire only artifacts/a17v1_calib256.pt and a29v2_selector_calib256.pt, preserving hashes/sizes first and leaving their scripts, checkpoints, requests, manifests, scores and ledger intact. This frees enough room for A174's approximately1.80GBnative basis while staying below40GB.

A175writer-gain geometry is prepared but NOT RUN: experiments/analyze_a175_uniform_writer_gain_floor_v1.py. It keeps all recorded final features and rawDownweights fixed, so each protected-vocabulary class margin is affine in one common96-row writer gain. Direct inequalities compute the feasible gain interval preserving existing current-register outputs; there are no operand/sum labels. This differs from A70's failed native-gain emitter: A70redesigned all output vectors and left23k–44kuncancelled scratch and15.6knonscratch nuisance in its old architecture. A175tests the present anchored11-function emitter with unchanged rawprogramme geometry. Its bound does not yet include native numerator compensation or claim a checkpoint.

Possible follow-on ONLY if A175supports it: restore an exact eight-dimensional cleanup in place of the current six-dimensional approximation, then assess whether lower uniform writer gains can preserve emission. Two additional cleanup units could be placed among existing native donors; verify the edit count, do not assume it fits. A whole residual-coordinate cancellation operator may be derived from fixed register scales and shared captured gate factors, never from answers. This idea is not implemented or evidence of success.


## A173 full result; sparse repair next, 2026-11-10 18:42:14 CET

Official full run0763-score finished in28:07.79, peakRSS3,851,120KiB. A173 is the new verified text-preservation leader:24,989/25,000exact anddigit-prefix H1(99.956%); text97.87063598632812% on131,072targets/2048windows, KL.00276913883843414nats, CEchange-.0010317330918621082nats, allfinite. Text improves by28matching targets overA142v2, but remains below98.5%, so H1does NOT pass. Checkpoint SHAde003de084d84b8f01513fd1776bce7cf449f73341501aa09fe3e4f0e97262c0;5,011,468edits,max1e6.

Noise exact/digit counts outof2000: sigma1e-5:2000/2000;1e-4:1574/2000;1e-3:262/1615. Two-in-a-row:first1999,secondgivenfirst1,both1. It remains unsuitable for repeated problems and poorly robust to larger noise. Full score and immutableledger contain the details; no arithmetic-failure examples are used as weight-fitting targets.

Completed usage299forward/99quick/3full;remaining1/1/37. No best rung or finalverification/export. A174final native-feature diagnostic remains prepared,notbuilt/run. A176sparseO3 analysis is now running; its fixed128outputrows use coordinate counts[64,128,256,512,768], ridges[1e-6,1e-4,.01,1,100], strengths[.25,.5,.75,1,1.25]. A single original-target conditional-explanatory-energy ranking uses fixedridge.001. It fits original HOSTbranch targets directly, with the same full arithmetic-frame null constraint, never residuals fromA173. A175uniform-writer-gain geometry is next. Neither has consumed a meter.


## A176 sparse repair built; A175 bound; prepared A177–A179, 2026-11-10 18:59:51 CET

A176 selects128outputrows,768headcoordinates,ridge.0001,strength.75. Paired native attention3branchRMS.2169684620to.19137621195; hiddenrelativeRMS.0050474361to.00484789591. All source digits remain correct, maxerrors[.121471,.090688,.060766], maximumformatrawRMSchange.32108%, framefunctionalerror2.40e-9. It uses98,304newO3entries instead ofA173's131,072. CheckpointSHA16d0062ba5371e41ce875c05b6140d2ede01a6484e475b8387550e3032530b61;4,978,700edits,max1e6. FullH1score is running as session23581; mainmath24,990/25,000exact/digits has finished, text/noisepending. Its sources are now immutable. Completed usage299forward/99quick/3full, fourth full reserved;remaining1/1/36. Currentfolder39290891981bytes.

A175 is an analysis-only geometric restriction, NOT a new checkpoint: with all rawweights/features fixed, common96-row writer gain must be above865,125.7393andbelow1,050,033.8575to keep all2,560recorded existing register winners. HOSTpoolgains have median458,max889. This calculation scales the scratch cancellation along with emission, so it does not establish the best feasible emission-only gain. Wall25.58s,peakRSS2,312,020KiB. Prepared A177 repeats the direct inequality bound on only86non-cleanup writer coordinates, holding10scratch cleanup gains fixed; notrun.

A178 is prepared,NOTRUN. It will use certified A174original native features at exact parent textinputs. For eachlayer0..8/17it independently fits original HOSTsame-input native branch targets using only already-edited Downentries, with all programme columns and gainsfixed. Ridges[1e-6,1e-4,.01,1,100],strengths[.25,.5,.75,1,1.25];train192/val64. A local candidate must reduce nativebranchRMSby5%andimprove both pairedHOSThidden anddirectionerror. The final layer additionally needs a full-vocabulary validation improvement. No additional edited support is allowed. No exact currentformatcapture is claimed afterO3changes; official arithmetic scoring remainsrequired.

A179 is prepared,NOTRUN. It will recover factual B-reader headcontexts by the same full-rank linear-inverse method checkedinA171, then fit sparseO2corrections in the null of its entire five-function/rank-four arithmeticframe. Fixed128outputrows; inputcounts[16,32,64,128,256],ridges[1e-6,1e-4,.01,1,100],strengths[.25,.5,.75,1,1.25];fixedrankingridge.001/rawprojectionpenalty.01. Atmost32,768newentries fit A176's remaining37,568editbudget. This reader acts earlier, so an accepted result should be assessed before spending the lastforward on A174. A174's analyzer now checks early-input equality only when upstream weights actually match the oldcapture; for anO2repair parent, equality atlayer2cannotbeassumed. No metered source has been changed.


## A176 full result and next repairs, 2026-11-10 19:13:22 CET

Full0765-score completed:24,990/25,000exact anddigit H1(99.96%), text97.86453247070312%,KL.0027726538836964675,CEchange-.0010486288474558236,allfinite. H1does NOT pass. It has8fewer matchingtexttargets thanA173,20morethanA142. A173remainsverifiedtextleader;A176is the working parent because it saves32,768edits for a second native reader repair. A176uses4,978,700edits,max1e6,SHA16d0062ba5371e41ce875c05b6140d2ede01a6484e475b8387550e3032530b61. Noiseexact/digitcounts outof2000:1e-5:2000/2000;1e-4:1574/2000;1e-3:261/1613. Two-in-a-row1999first,1secondgivenfirst,1both. Wall28:09.23,peakRSS3,852,492KiB.

Completedusage299forward/99quick/4full;remaining1/1/36. A177gainbound is running95197, thenA179localinverse/repair. A179now includes an independent layer2known-head check using eight oldA126windows and that checkpoint's own valueweights; actualcurrentheads are used only after this check and fullrank/outputreconstruction/causality checks pass. It saves only the small selectedO2matrix, not large derived activation arrays.

A180is prepared,notrun: an alternative toA178using the original pairedHOSTstate after eachlayer as target, with parentfeatures andrawnormalizersfixed. Same five ridges/strengths and already-edited nativeDownsupport. UnlikeA178's original same-input native-function targets, these are alternatives for a SINGLE layer; they must never be composed into repeated accumulated-error corrections. The builder requires an explicit layer. Both remain dependent on the unspent finalA174diagnostic; no new meter or weights have been produced by either.


## A177 bound; A179 built/full0767; expanded final diagnostic, 2026-11-10 19:23:16 CET

A177holds10scratch-coordinate gainsfixed andvariesonly86otherwriter gains. Its analytic strictlowerbound is471,334.47594,upper32,318,584.7643; allweights/featuresremainfixed. It is geometryonly,not a checkpoint. The gainbound is still far above HOSTwriter gains361–889(median462). Moreover, emission vectors themselves have components onthe10scratch coordinates; holdingthose gainsfixed also changes relative emissiongeometry. Thus neitherA175norA177proves the optimum of a separately redesigned emission/cancellation programme. No meterwasused.

A179's independent eight-window layer2headinverse check hasrelativeheaderror7.11526439e-8. Currentword256windows arefullrank256,minsingularratio.000260676,rawreconstructionerror1.12e-7;128formatwindows fullrank56,rawerror1.83e-7. Maximumspuriousfuturecoefficient1.16e-5. These are factual inverses of recorded branchoutputs,not new attention evaluations.

A179selects128outputrows×256headcoordinates,ridge1,strength1.25,protectingallfiveB-readerfunctionals(rank4). PairedbranchRMS.06949562787to.06487807676(6.64%); hiddenrelativeRMS.00311597166to.00306943571; directionRMS.00186572352to.00184642503. Source-digit errors[.158767,.0932654,.0589398],zero wrong; minimumreference.575399; maximumformatrawRMSchange.05319%; framefunctionalerror6.96e-10. Analysis32.96s,peakRSS2,056,696KiB.

Builtcheckpoint experiments/a179v1_sparse_b_reader_output/delta.pt, SHAba2a0fe8abb7ba82c153250ed268617e6b06357aa413cba29d43135138292260,5,011,468edits,max1e6. Adds32,768nativeO2repairentries toA176. Sources/dependencies nowimmutable. Fullrun0767-score is running as session95581,logartifacts/a179v1_full_H1.log,outputscores/a179v1_full_H1.json. Completedusage299forward/99quick/4full;fifthfullreserved;remaining1/1/35. A173remainstextleader; no rungpassed.

A174has NOT beenbuilt orrun. The finalrequest nowincludes unmodified layer9 inadditionto0..8and17. This offers exactnativefeatures for a possible later scratch-clearing stage, an alternative to changing finalwriter gains alone. Plannedcapturetensorbytes1,979,711,488;topk/inputbytes1,703,936;serializationheadroom18,584,576belowthe2GBcap. The builderrequires at least10MBheadroom. Layer9hasno programmeunits: it isobserved unchanged, andthe certificate checks itsHOSTweights ratherthan applying anemptygatebound. A178/A180explicitlyskipthisunmodifiedobservationlayerbecause their existing-edit support there isempty.

Prepared cleanup_a174_space_v1.py nowalso retires artifacts/a171v1_recorded_attention_inverse.pt inadditionto the two old a17/a29captures:totalplanned1,818,581,929bytes. The mixturematrices are reproducible from retained originalcaptures and the immutableA171script; the smallreports/initialandsavedlogsremain. Cleanup has NOT RUN. It mustrun beforethelargerfinalcapture. The exactdeletionrecordwillpreservehashes/sizes.

Possible layer9cleanupis ONLY an idea,not implemented orscored. Afterselector8, eight older scratchcoordinates are nolonger needed, whiledigit391/flag476muststay untilfinalemission. An earlier conditional linear cancellation mightreduce finalemitterburden, but needs a gate, nativefeatureallocation/repair, aneditbudgetallocation, and completeofficialarithmetic/textvalidation. A178andA180remain the next concrete native-repair analyses afterA174.


## Final capture guard correction, 2026-11-10 19:37 CET

Read-only inspection of frozen tools/forward.py found an additional strict internal guard: captured tensors must be below1,850,000,000bytes, before the final2,000,000,000-byte artifact guard. The previously proposed256-window layer9 expansion would have failed this internal guard. It was caught BEFORE any build or meter call; no budget was spent and the meter was not changed.

Corrected A174request: all256corewindows retain nativefeatures/preMLPresiduals at0..8and17 plus finalraw/top8. Unmodified layer9 is additionally observed on64windows(indices0,4,...252), split48training/16validation. Plannedcapturetensorbytes1,847,590,912, leaving2,409,088bytes under the exact internal tensor-count guard. Including topk/inputtensors gives1,849,294,848bytes, well below the2GBartifact limit. Layer9comparison selects the corresponding originalHOSTwindows. CoreA178/A180training192/validation64remainunchanged. The cleanup remainsprepared,notrun; reclaimingA171mixturesalso reserves room for subsequent repair artifacts.


## A179 rejected by full text score; final diagnostic parent chosen, 2026-11-10 19:47:19 CET

Full0767-score: A179has24,989/25,000exact/digit H1(99.956%) but text97.82180786132812%,KL.0027694434577039337,CEchange-.000991342080645552,allfinite. It loses56matchingtexttargets versusA176 and64versusA173. Localnativebranch/state improvement did NOT predict a globaltext improvement. H1doesnotpass. Noiseexact/digit outof2000:1e-5:2000/2000;1e-4:1573/2000;1e-3:262/1612. Two-in-a-row:first1999,secondgivenfirst1,both1.

A179isrejected. As declared before the result, A176is the chosen finaldiagnosticparent:4,978,700edits,97.86453247070312%text,99.96%H1,with32,768morefreeedits thanA173. A173remainsthe bestverifiedtextcandidate. Completedusage299forward/99quick/5full;remaining1/1/35.

Cleanup_a174_space_v1.py has nowRUN and reclaimed1,818,581,929bytes. Its exactrecord artifacts/cleanup_a174_space_v1.json preserves each deleted file'sSHA256 andsize. Retired: a17v1_calib256.pt,a29v2_selector_calib256.pt,andthederiveda171v1_recorded_attention_inverse.pt. Their smallreports,logs,checkpoints,requests,source scripts andledger remain. OriginalA171inputcaptures remain, so thefactualinverse canberecreated. Do not assume those threePTfiles stillexist.

A174diagnostic is BUILDING as session50113,logartifacts/a174v1_build.log,usingA176asbase. The finalforward has NOT yet run. Memoryavailablebeforebuildwas15,794,139,136bytes. Correctedcaptureplan remains1,847,590,912tensorbytes underfrozeninternal1.85GBguard,withall256corewindows and64layer9observations.

### A174 final native-feature capture (2026-11-10 19:54:44 CET)

The 300th and final forward pass completed as `0769-forward` on the A176 diagnostic derivative. It captured all 256 new calibration windows at layers 0–8 and 17, plus layer 9 on 64 windows. The artifact is 1,849,365,909 bytes; the process took 1:06.07 and peaked at 6,047,944 KiB. Remaining budgets are 0 forward, 1 quick and 35 full scores. The project occupied 39,406,068,643 bytes immediately afterward. The first offline certificate analysis stopped on a `torch.Size` reporting typo before producing a certificate; the unmetered analyzer was corrected and is being rerun. The capture itself succeeded and was not repeated.

A174 certificate succeeded on retry in 33.34 seconds, peak 6,690,688 KiB. Every original programme gate remains below −16 including a conservative float32 dot-product allowance, making programme features exactly zero on the recorded words. The final raw-state reconstruction has relative error 6.45898e−8. The parent matches original HOST top-1 on 16,018/16,384 calibration tokens and 4,004/4,096 validation tokens. Replacing only the final MLP branch with the original HOST native branch at those exact inputs gives 4,007/4,096; this diagnostic removes the arithmetic emitter and is not an eligible addition candidate or a universal upper bound. Thus the large final-branch RMS distortion alone does not explain most of the text disagreements.

### A178 supported native repairs; A180 paired-state alternatives (2026-11-10 19:59:17 CET)

Both analyses use the certified A174 features, 192 training and 64 validation windows, and only existing native Down edit support; programme columns and gains stay fixed. Original targets and recorded normalizers are held fixed in all closed-form solves, with ridge values [1e-6, 1e-4, 0.01, 1, 100] and strengths [0.25, 0.5, 0.75, 1, 1.25]. A178 targets the original HOST native branch at the current input. Only layer 8 qualifies, selecting ridge1e-6/strength0.25: branch RMS0.3346051646→0.2776064190, paired state relative RMS0.00624202584→0.00624141722, direction0.00480053544→0.00479429450. No other layer meets the declared5% local-improvement screen. Analysis took1:15.61, peak7,275,336KiB. The candidate changes4,978,699scalars, max1e6; official full H1 scoring is running. Its source files are now immutable. Current-format arithmetic must be tested officially because the latest diagnostic contains words only.

A180 independently targets each original paired HOST state after one layer, with the parent input and normalizer fixed. These are single-layer alternatives, never composed. No layer passes its5% paired-state and direction screen; final-layer trials reach at most3,996/4,096 restricted-vocabulary matches versus4,004baseline. No A180 checkpoint was produced. Analysis took1:14.90, peak7,245,920KiB. Detailed per-layer trials remain in the JSON reports.

### Queued structural analyses (2026-11-10 20:08:55 CET; not yet run)

A181 separates the existing final digit-emission component from its ten physical cleanup rows. It scales emission uniformly in physical output space, uses HOST gain multipliers [1,2,4,8,16] on the other86writer rows, and determines an anchor coefficient by one quadratic raw-energy identity. Unlike A175/A177, emission on the cleanup rows scales too while the cleanup operator stays fixed. The exact recorded A124-format inputs are valid for its A142 parent. Native columns receive only fixed inverse-gain compensation in this geometry screen; any later native reconstruction would use original HOST targets. It will not run alongside the active score.

A182 tests the previously untouched layer9 using its final diagnostic subset:48training/16validation word windows, at most48native Down columns and unchanged G/Up/gains. Counts[8,16,24,32,48], ridges[1e-6,1e-4,.01,1,100], strengths[.25,.5,.75,1,1.25]; fixed conditional-ranking ridge.01 and minimum output-gain magnitude5. The original paired HOST state is the target, with fixed parent normalizers. Its screen requires1%improvement in both state and direction error and at most5%rawRMS change. No layer9format feature capture exists, so any selected candidate needs an official arithmetic score. These are prepared analyses, not results.

Cleanup A183 completed at 2026-11-10 20:14:40 CET: three superseded A1 marker-only format captures (`a1_syntax_a.pt`, `a1_detail_syntax_a.pt`, `a1_detail_syntax_b.pt`) were hashed against official ledger records0037/0043/0045-forward before retirement. It reclaimed1,299,501,719bytes; the complete record is `artifacts/cleanup_a183_space_v1.json`. Original HOST data, sparse checkpoints, sources, manifests, requests, scores and the ledger remain. The cleanup took5.19seconds, peak40,060KiB. A178 main arithmetic has finished at24,990/25,000; its remaining full-score sections are still running.

A184 is additionally prepared, not run: a shared reverse-format gate and at most16 low-rank hand GLUs in native layer9, fitted only to original paired HOST text states. It uses fixed gate ridge.001, direction ridge.01, ranks[1,2,4,8,16], output ridges[1e-6,1e-4,.01,1,100] and strengths[.25,.5,.75,1,1.25]. The old format-state envelope is explicitly historical after the O3 change; any selected weights require an official arithmetic score. No A184 result or checkpoint is claimed.

A185 is prepared but unrun: the A-reader’s fourth programmed query targets the plus sign instead of the beginning-of-sequence reference, while its native query component, three digit heads and gains stay fixed. Five source-difference ridge values [1e-6,1e-4,.01,1,100] and an analytic margin8 amplitude bound are declared. This is a source-position geometry hypothesis for prefix use, not a prefix result; it still needs a compatible constant-reference decoder and official scores if the static screen passes.


<!-- EDIT_AUDIT_START -->
### Current leader: reconciled per-tensor edit counts

Checkpoint `experiments/a313v1_duplicate_reference_compression/delta.pt`, SHA-256 `67fd7d1ae9b0d63f4d2f8993b65f1e837fb65a418804e56125a8b7eb4cec49e5`. Total **4,976,785**; arithmetic and circuit accommodation **578,041**; native-only text repair **4,398,744**. Maximum absolute edited value **1e6**.

Inherited index-audited A259 labels outside layer2 Gate/Up/Down; replacement labels use the immutable A313 compiler masks. Q3/Q7 native repairs overlap arithmetic labels. These are disjoint bookkeeping labels, not causal allocations.

Metadata reconciliation against the compiler-verified unchanged parent tensors; no new sparse-index comparison is claimed.

A313 restores layer2 unit727, merges its exact duplicate function into529, and reconstructs the original native branch once. The new native reconstruction labels reclassify 8,684 previously accommodated native Down entries; the physical edit-count reduction is 1,915. Its immutable compiler verifies that every other tensor stays bit-identical to A259. The layer0 corrections and original Q3/Q7 channel labels remain inherited from the prior index audits.

| Tensor | Arithmetic / accommodation | Native-only repair | Total |
|---|---:|---:|---:|
| `model.layers.0.mlp.down_proj.weight` | 640 | 42,942 | 43,582 |
| `model.layers.0.mlp.gate_proj.weight` | 640 | 0 | 640 |
| `model.layers.0.mlp.up_proj.weight` | 640 | 0 | 640 |
| `model.layers.0.post_feedforward_layernorm.weight` | 1 | 0 | 1 |
| `model.layers.1.mlp.down_proj.weight` | 11,372 | 325,632 | 337,004 |
| `model.layers.1.mlp.gate_proj.weight` | 3,200 | 0 | 3,200 |
| `model.layers.1.mlp.up_proj.weight` | 3,200 | 0 | 3,200 |
| `model.layers.1.post_feedforward_layernorm.weight` | 4 | 634 | 638 |
| `model.layers.2.mlp.down_proj.weight` | 2,560 | 333,808 | 336,368 |
| `model.layers.2.mlp.gate_proj.weight` | 2,560 | 0 | 2,560 |
| `model.layers.2.mlp.up_proj.weight` | 2,560 | 0 | 2,560 |
| `model.layers.2.post_feedforward_layernorm.weight` | 4 | 636 | 640 |
| `model.layers.2.self_attn.q_norm.weight` | 32 | 63 | 95 |
| `model.layers.2.self_attn.q_proj.weight` | 92,160 | 163,840 | 256,000 |
| `model.layers.3.mlp.down_proj.weight` | 9,204 | 325,630 | 334,834 |
| `model.layers.3.mlp.gate_proj.weight` | 2,560 | 0 | 2,560 |
| `model.layers.3.mlp.up_proj.weight` | 2,560 | 0 | 2,560 |
| `model.layers.3.post_feedforward_layernorm.weight` | 3 | 637 | 640 |
| `model.layers.3.self_attn.o_proj.weight` | 0 | 98,304 | 98,304 |
| `model.layers.3.self_attn.q_norm.weight` | 64 | 192 | 256 |
| `model.layers.3.self_attn.q_proj.weight` | 163,840 | 491,519 | 655,359 |
| `model.layers.3.self_attn.v_proj.weight` | 163,839 | 0 | 163,839 |
| `model.layers.4.mlp.down_proj.weight` | 3,840 | 87,662 | 91,502 |
| `model.layers.4.mlp.gate_proj.weight` | 3,840 | 0 | 3,840 |
| `model.layers.4.mlp.up_proj.weight` | 3,840 | 0 | 3,840 |
| `model.layers.4.post_feedforward_layernorm.weight` | 3 | 632 | 635 |
| `model.layers.5.mlp.down_proj.weight` | 5,500 | 81,536 | 87,036 |
| `model.layers.5.mlp.gate_proj.weight` | 1,280 | 0 | 1,280 |
| `model.layers.5.mlp.up_proj.weight` | 1,280 | 0 | 1,280 |
| `model.layers.5.post_feedforward_layernorm.weight` | 2 | 638 | 640 |
| `model.layers.6.mlp.down_proj.weight` | 6,400 | 339,887 | 346,287 |
| `model.layers.6.mlp.gate_proj.weight` | 6,400 | 0 | 6,400 |
| `model.layers.6.mlp.up_proj.weight` | 6,400 | 0 | 6,400 |
| `model.layers.6.post_feedforward_layernorm.weight` | 9 | 631 | 640 |
| `model.layers.7.mlp.down_proj.weight` | 21,952 | 323,072 | 345,024 |
| `model.layers.7.mlp.gate_proj.weight` | 5,120 | 0 | 5,120 |
| `model.layers.7.mlp.up_proj.weight` | 5,120 | 0 | 5,120 |
| `model.layers.7.post_feedforward_layernorm.weight` | 8 | 632 | 640 |
| `model.layers.7.self_attn.k_norm.weight` | 251 | 0 | 251 |
| `model.layers.7.self_attn.k_proj.weight` | 1,280 | 0 | 1,280 |
| `model.layers.7.self_attn.q_norm.weight` | 255 | 0 | 255 |
| `model.layers.7.self_attn.q_proj.weight` | 5,120 | 0 | 5,120 |
| `model.layers.8.mlp.down_proj.weight` | 5,760 | 1,304,956 | 1,310,716 |
| `model.layers.8.mlp.gate_proj.weight` | 5,760 | 0 | 5,760 |
| `model.layers.8.mlp.up_proj.weight` | 5,760 | 0 | 5,760 |
| `model.layers.8.post_feedforward_layernorm.weight` | 2 | 637 | 639 |
| `model.layers.17.mlp.down_proj.weight` | 7,040 | 474,080 | 481,120 |
| `model.layers.17.mlp.gate_proj.weight` | 7,040 | 0 | 7,040 |
| `model.layers.17.mlp.up_proj.weight` | 7,040 | 0 | 7,040 |
| `model.layers.17.post_feedforward_layernorm.weight` | 96 | 544 | 640 |

Audit artifact: `artifacts/a313v1_duplicate_reference_compression_edit_category_audit.json`. The manifest and checkpoint are unchanged.
<!-- EDIT_AUDIT_END -->

### A178 full score rejected (2026-11-10 20:28:15 CET)

Official run `0771-score`, checkpointSHA `c65632ae85adfd905ca10ae97c3b122b38543e8e6d759d4791a5f6a2fe97e971`, changes4,978,699scalars,max1e6. H1 remains24,990/25,000exact and digits(99.96%), but textagreement is97.82638549804688%,50fewer matches than its A176 parent and58fewer than A173. KL0.0028044638865588147nats, cross-entropy change−0.0012331321948577667nats, allfinite. H1 jointgatefails. Reject this candidate as a preservation improvement; the local native-branch reduction did not translate into better global predictions.

Noise exact/digits out2,000: sigma1e-5:2,000/2,000;1e-4:1,589/2,000;1e-3:149/1,505. Two-in-a-row first1,999,secondgivenfirst1,both1. The sigma1e-3 result is also materially worse than the parent. Budgets used300forward,99quick,6full; remaining0forward,1quick,34full. No rung is passed. A181 emission/cleanup separation analysis is now running.

### A181 and A182 rejected by offline checks (2026-11-10 20:31:58 CET)

A181’s existing-format baseline reproduces all2,560register winners, minimum full-vocabulary margin1.30151. HOST reference gain is461. For HOST gain multipliers[1,2,4,8,16], the physical emission ratios are[.000461,.000922,.001844,.003688,.007376]. Every candidate matches0/2,560existing register winners; minimum margins range−673.11to−491.49. RawRMS changes by at most0.8822%, but cleanup-coordinate nuisance stays near41,400 and other-coordinate nuisance near38,100. Separating emission from cleanup therefore does not make these native-scale gains usable with the current residual geometry. No checkpoint; wall18.67seconds,peak1,934,984KiB. This is a rejection of these five constructions, not a general gain lower bound.

A182 produces no candidate. Baseline validation paired-state RMS0.00541556506,direction0.00424290132. The lowest state-error trial uses24columns,ridge.01,strength.5: state0.00540358989(only0.22%better), direction0.00424653612(worse), maximumrawRMSchange0.07272%,14,088newentries. It fails the1%state-and-direction screen. Peak1,867,972KiB. These analyses consume no meter budget. A184 conditional-layer9 text repair is running.

### Layer9 repair rejected; plus-reference construction (2026-11-10 20:38:29 CET)

A184’s gate passes its static checks: minimum recorded-word gate127.88865after rounding allowance; maximum historical-format envelope−138.78109; maximum gateweight311.83890. Nevertheless every text-repair candidate worsens paired-state error. The best state-error trial(rank1,ridge1,strength.25) has RMS0.00545285453versus0.00541556506baseline,and direction0.00430184869versus0.00424290132. No checkpoint;23.22seconds,peak2,152,308KiB.

A185 query geometry passes. Selected source-difference ridge1e-6 gives queryfactor0.6826073963; source margins[8.0001166,7.9999938,8.0000835,8.0000908]. Current-word centered Q/K scoreMSE changes0.00090725537→0.00090729568, essentially unchanged. Maximum Qweight0.5587263;14.19seconds,peak1,809,740KiB. These are Q/K/RoPE measurements, not an actual prefix addition result.

A186v1 source-frame analysis keeps V3fixed and changes the reference source toplus. Its five background masses[0,1,100,10000,1000000] all fail the declared per-digit error screen; best current maximumerror is0.296123atmass100, with reference near0.916. Inspection found that the reference mean still omitted its projection onto the literal-digit value axis, inherited from the near-zero BOSprojection assumption. A186v2 corrects this: the nondigit reference head uses its complete native plus mean and covariance. It reuses exactly the same five masses and other constants. V1source/report/log remain; v2is nowrunning. No A185/A186checkpoint or meter yet.

A186v2 passes its source-frame screen (2026-11-10 20:40:10 CET). Backgroundmass10,000,within-source mass10,000,metricridge1e-6,Upgain1e6(maximumUpweight7,572.312). Current128format source-digit maximumerrors[.141940,.110443,.123569], no rounded errors; held-back32 maxima[.141940,.093059,.089609]. Historical512validation maxima[.192006,.244178,.126870], no rounded errors. Every reference stayspositive(minimum.903702acrossprobes). Independent per-source errorbound.0488764. Only the three digit heads’ literal direction constraints and full plus-head native mean/covariance enter the decoder; V3isunchanged. Analysis11.66seconds,peak1,207,708KiB. Builder is running. The last quick unit is planned for H2 to directly test the prefix hypothesis; no prefix result or rung pass is yet claimed.

A186v2 checkpoint built: SHA `909ab2a2beae0f50d304a7f93da55c33111616c0bd98c9d251d50b300e5d6486`,4,880,396changedscalars,max1e6. Builder14.07seconds,peak776,156KiB. It changes the fourth programmed A-query and four A-amplifier Up rows relative to A142; embeddings,head,finalnorm,V3andallother circuit functions stay fixed. Quick H2 scoring has started, reserving the final quick unit. Sources used for this metered checkpoint are now immutable.

### A186v2 quick H2 fails (2026-11-10 20:44:35 CET)

Official `0773-score`:21/2,000exact(1.05%),23/2,000correct digit prefixes(1.15%). Quick textagreement97.869873046875%,KL.002624882505546695,CEchange−.0005563927211094111,allfinite. H2fails both arithmetic and top-1 agreement; no rung can be banked. Even one-token-prefix examples fail, so removing the BOSreference alone is insufficient. The score exposes only aggregate text and permitted arithmetic examples; no held-out text was inspected. Wall3:22.29,peak3,843,440KiB. The 100quick units and300forward units are now exhausted.

A full H1 score of the same immutable checkpoint has started to distinguish failure of the revised standalone circuit from the unresolved prefix limitation. It will also measure noise and repeated problems. Remainingfullbudget after reservation:33. No new parameters are being selected from these failures.


### A188 early native-energy reconstruction prepared (2026-11-10 20:56:06 CET)

Scripts `experiments/derive_a188_early_native_energy_anchors_v1.py` and its compiler are prepared, syntax checked, and not run. They apply the earlier A141 energy-remainder idea to layers0–7 using the certified A174 current-parent native features (192 training/64 validation windows). All arithmetic G/Up/Down functions and their output gains remain fixed. Each layer starts from original HOST targets; displaced native units receive one donor projection (ridge0.001; donor counts64,512,512,512,128,128,512,512). Existing strong rows use five fixed-normalizer ridge values[1e-6,1e-5,1e-4,0.001,0.01]. Five quiet coordinates are ranked once from original HOST energy and gain. An anchor is rejected if its exact energy remainder is negative on any training or validation row; no clipping. The square-root energy target is projected once with the same five ridges and one analytic trace normalization. Zero-gain arithmetic anchors retain their zero gain; their missing original native output is included in the error metric. Acceptance requires at least5% native-branch RMS improvement, improved paired HOST state and direction, and the edit cap. Only independent local native-function repairs may be composed. Exact current-parent format captures are unavailable, so any candidate needs official arithmetic scoring. No new meter units have been spent on A188.


### A187 early unit relocation prepared (2026-11-10 21:04:30 CET)

The A187 analyzer and compiler are prepared and syntax checked, not run. They use A174 certified current-input native features,192 training/64 validation windows, and independently relocate the unchanged arithmetic functions in layers0–7. Fixed existing donor counts are64,512,512,512,128,128,512,512. A single conditional-feature solve uses ridge0.001. A norm/dot-product identity ranks the exact native RMS-branch error caused by removing each conditional feature; it is checked against literal vector RMS on a tiny recorded sample. Raw-energy tradeoff values are[0,0.01,0.1,1,10]. Each distinct allocation is reconstructed independently from original HOST branch targets with fixed parent normalizers and ridges[1e-6,1e-5,1e-4,0.001,0.01]. G/Up/Down arithmetic functions are copied bit-exactly and all gains stay fixed. A candidate requires5% local branch improvement, better paired HOST state/direction, and the edit cap. New metadata explicitly maps moved units because legacy stage metadata retains historical positions. Any saved candidate requires official arithmetic/text testing.

Prefix-data audit: files named a39_inputs_prefixes and token_prefix3 contain operand prefixes such as BOS plus three digits; they are not arithmetic after ordinary text. No current contextual-prefix activation diagnostic was found. The final forward budget is exhausted. The earlier summary calling A72 a root-gate hardening was imprecise: A72 hardens the layer4 units-carry gate. Root layer0 format assumptions remain a separate unverified H2 hypothesis.

A189 is an unimplemented design at this point: interpolate the existing A120 rank-one V3 change toward original HOST with five fractions[0,0.25,0.5,0.75,1], recompute a decoder from independent individual-source digits, and screen using actual recorded current-parent attention mixtures recovered by linear inversion. This differs from A162's generalized eigenmode/native-axis replacement and uses factual format mixtures to address its missed routing/background error. No new attention probabilities, native nonlinear activations or whole-answer targets would be computed.


A189 preparation update (2026-11-10 21:07:05 CET): analyzer and builder now exist and are syntax checked; neither has run. Value fractions[0,0.25,0.5,0.75,1], background masses[0,0.01,1,100,10000], fixed word regularizer1e-8, and Up gain choices[1000,10000,100000,300000,1000000]. The source means/targets are independent digits and a native reference. Unique linear inversion reconstructs factual word/format mixtures and first checks eight older windows with directly recorded heads. No derived mixture artifact is saved. Current128 format windows use factual mixtures; the last32 do not enter the background covariance. Rounded Up rows must decode all source digits correctly with maximum error<0.25 on current formats and<0.4 on historical ideal-routing probes. Word programme gates must remain<-16 and format equals gates>32; native value branch error must improve5% and paired HOST state error must improve. A189 retains A142 Q/K/O and all arithmetic functions except four Up decoder rows. This is a prepared candidate construction, not an arithmetic or text result. Queue remains A188, A187, A189 after A186 full scoring.


A190 unimplemented idea (2026-11-10 21:11:08 CET): revisit the B-reader2 query, which retains the old A25/A27 construction. Its current Q2 support is256,000 entries, so replacing the native sparse mean-cancellation readout with a dense quiet form would exceed the edit cap; do not do that. Instead, decompose Q2 algebraically into per-channel HOST coefficients plus the two known readouts (old sparse native-mean cancellation and old quiet programme query). Keep the sparse cancellation support, consider fractions[0,0.25,0.5,0.75,1], and recompute only the programme quiet readout from exact current word/format2 inputs (A150/A140 and A126, after upstream identity checks). Five format-variation masses[0,0.01,0.1,1,10], native Q/K and RMS/RoPE algebra, then analytic per-head routing amplitudes for margin8. Select by centered causal QK score error on separate calibration windows; do not compute softmax. Copy parent Q2 and modify only existing programme rows and sparse cancellation columns, preserving all other bits. A126 mixed audit provides256word/256format input2 captures, but upstream0..1 identity still needs checking. This is not coded or run.


### A186v2 full H1 completed, rejected (2026-11-10 21:13:40 CET)

Official0775-score (`scores/a186v2_full_H1.json`) gives24,983/25,000 exact and digit-prefix correct(99.932%). Text agreement97.8485107421875%, KL0.00278283331796128, cross-entropy change−0.0009649723839890711, all finite. H1 fails the98.5% text gate. Compared with A142 it loses7 arithmetic cases and1 matching text target; compared with A173 it loses29 matching text targets. Noise exact/digits out of2000:1e-5:2000/2000;1e-4:1575/2000;1e-3:261/1614. Two-in-a-row:first1998, second given correct first2, both2. Runtime28:13.79, peak3,853,960KiB. No rung is banked. The prior quick H2 result remains21/2000exact,23digit-prefix correct; changing only the A-reader reference did not fix contextual addition. A173 remains the verified leader. Completed budgets300forward/100quick/7full; remaining0/0/33. Folder size38,200,687,850bytes at completion.

A188 analysis is now running as session39801, log `artifacts/a188v1_early_native_energy_anchors.log`. A187 and A189 are queued afterward. A190 analyzer is now prepared and syntax checked, not run; its compiler is not yet prepared. A190 uses five sparse native-cancellation fractions and five quiet-format covariance masses recorded above, deduplicates calibration windows, requires exact upstream identity for A126/A150 input2 observations, and accepts only a10% centered causal QK-score-error improvement while retaining rounded source routing margin8 and the edit cap. All Q2 gains and K/V/O remain fixed. No new full score is authorized by a static screen alone; an eligible candidate still needs the official meter.


### A188 completed without a candidate (2026-11-10 21:15:02 CET)

The first analysis stopped at layer5 because it assumed every layer had five eligible quiet coordinates. Before any metered checkpoint, that assertion was corrected to use up to five actually available coordinates; the original failure log remains. The completed retry took33.89seconds and peaked at5,517,832KiB. No layer passed the declared local preservation screen, so no PT weights or checkpoint were written. Layers1–5 rejected every available anchor because some original-energy remainder was negative. Layer0 had one feasible anchor; layers6/7 had two/one, but their fits did not qualify. No remainder was clipped. Full per-layer trials are in `artifacts/a188v1_early_native_energy_anchors.json`. The failed first attempt took21.91seconds, peak4,955,516KiB. A187 unit-allocation analysis is now running; A189 and A190 follow. A190 compiler is now also prepared and syntax checked. Official budgets remain0forward/0quick/33full available.


### A187 and A189 static candidates (2026-11-10 21:18:54 CET)

A187 completed in1:00.73, peak5,393,488KiB. Only layer1 qualified: one occupied native slot changes from196to706 (the four retained slots249,1130,1657,1809 remain available). Raw-energy mass0 and native ridge0.01 give validation branch RMS0.05353338935→0.05057413667(5.53%), paired-state relative RMS0.003442911066→0.003393991193, direction0.002384488488→0.002333978512. Maximum raw-RMS change on recorded words is10.874%, so this local gain is not a global guarantee. The analysis predicts4,978,698edits. Its compiler now retains each existing arithmetic function in its original occupied slot and moves only the retired-slot function to706; this permutation leaves all analyzed native columns and the zero-on-words programme set identical. No checkpoint has been built yet.

A189 completed in23.36seconds, peak2,540,988KiB, and selects value fraction0.5/background mass0.01/common Up gain1e6. The local native-value branch RMS halves from0.1042223871to0.05158607486; paired HOST branch RMS0.2169684621→0.1968863805 and state relative RMS0.005047436086→0.004904593457. Word gate maximum−32.4928, format equals-gate minimum733.4594. With rounded Up rows, current128 source-digit maxima are[0.1819568303,0.0650892841,0.03439185153], no wrong digits; the independent last32 retain maxima[0.1819568303,0.0650892841,0.01551712189]. Across historical ideal-routing probes the worst error is0.2727120443, also no wrong digits. Fractions0/0.25 failed the declared precision guard (best actual-current errors0.37249/0.27491);0.75also qualifies but preserves less native value behavior. This is still a static source-decoding result, not an addition or text pass. The unique-mixture inversion and its known-head check are recorded in the A189 JSON. No checkpoint has yet been built.

A190 query analysis is now running as session53929. After it completes, build the selected A189 candidate and run a full H1 score; retain the independently screened A187 construction for separate validation. No quick or forward units remain, and33full units remain available.


### A190 rejected; A189 built and full H1 started (2026-11-10 21:22:21 CET)

A190's first run correctly stopped at an upstream-provenance mismatch: A126 has different layer1 Down weights. A126 was excluded entirely. The second run exposed a missing metadata key in the latest composition; metadata now comes from A27, with its Q2/q_norm weights verified bit-identical to the current parent. Both short failure logs remain, and neither run spent a meter unit. The completed analysis uses only256 exact-current word windows(192/64) and128 current format windows(96for covariance, all128for routing). It took14.36seconds, peak1,425,520KiB. All25declared query/cancellation trials were evaluated. Best centered causal QK-score MSE0.001550267919 is slightly worse than baseline0.001550255500; no checkpoint or PT was saved. The best uses mass1, full native cancellation, and analytic head factors[0.21747548,0.79317851,0.22532375,0.17759154], with rounded routing margins≥8.0000088. The16sparse cancellation coordinates remain unchanged in support. No new scale values were introduced after either implementation/provenance correction.

A189 checkpoint `experiments/a189v1_partial_native_value_decoder/delta.pt` was built in18.61seconds, peak776,500KiB. SHA256`b50dcc3247dc4abdf544a580c069d6c0c2eb05ad860cc115b9f9b7d9f753cfa4`;4,880,396changed scalars; max1e6. Full H1 scoring has started, output `scores/a189v1_full_H1.json`, log `artifacts/a189v1_full_H1.log`. The A189 analyzer/compiler and inherited dependencies are now immutable. A187 remains a separate unbuilt static candidate; its compiler preserves the four retained layer1 function positions and relocates only196→706. No rung has passed.


Next design ideas (2026-11-10 21:22:57 CET; not yet implemented): A191 would recompute an independent original-HOST O3 branch repair for A189's new V/frame, protecting that new arithmetic frame in the output-row nullspace. Reusing A176's O3 correction directly is invalid because its protected frame differs. Recover only the current word/format3 factual mixtures into a compact derived artifact (about233MB) from retained raw captures, then solve once from original paired HOST branch targets at the fixed A189 raw RMS. Counts[64,128,256,512,768], ridges[1e-6,1e-4,0.01,1,100], strengths[0.25,0.5,0.75,1,1.25],128output rows. No previous fitted residual target or nonlinear replay.

A192 would refresh B2 query energy accommodation using current256word/128format inputs. A190 shows that changing the quiet query/cancellation barely helps. Construct a canonical pre-energy query from A25v4 ordinary rows, current programmed rows, HOST ordinary q_norm gains and fixed current programmed gains; do not iteratively fit residuals from prior energy repairs. One original-HOST squared-query-energy solve uses carrier counts[1,4,16,32,48] and ridges[1e-6,1e-4,0.01,1,100]. Reciprocal raw-row/q_norm rescaling preserves native query numerators; direct causal QK score error and source margins select a candidate. The A189 parent leaves room for48carriers even if A176 would not. This is an idea, not a measured result.


### A191/A192/A193 prepared while A189 scores (2026-11-10 21:31:42 CET)

A191 now has three syntax-checked scripts: `prepare_a191_recorded_reader3_mixtures_v1.py`, `derive_a191_partial_reader_output_repair_v1.py`, and its compiler. None has run. The helper saves only word/format input3 and factual source mixtures (about233MB) after the same independent known-head linear-inverse check; it checks folder space before writing. The repair uses A189's rounded Up interface as its protected frame,128output rows, native coordinate counts[64,128,256,512,768], ridges[1e-6,1e-4,0.01,1,100], and strengths[0.25,0.5,0.75,1,1.25]. Original paired HOST branch targets and the fixed derived A189 normalizer are used in every independent solve. Require5% local branch gain, improved paired state, preserved word/format gates, no wrong source digits, source-error growth<0.01and absolute<0.25, format raw-RMS change<2%, and edit-cap compliance.

A192 analyzer/compiler are prepared, syntax checked, not run. Canonical B2query: A25v4 ordinary rows, current programme rows, HOST ordinary q_norm gains and current programme gains. This explicitly precedes the old energy repair. Counts[1,4,16,32,48] and ridges[1e-6,1e-4,0.01,1,100] independently fit the original HOST squared-query-energy target on192current word windows;64validate. Reciprocal channel row/gain scales preserve the canonical native numerator. All128format windows must retain source margin8, and validation centered causal QK score MSE must improve10%, within weight/edit limits. No iterative residual fitting or new probabilities.

A193 analyzer/compiler are also prepared, syntax checked, not run. This is a different repair: add a dense Q3 correction constrained to annihilate the first96current format query vectors. Every ridge[1e-6,1e-4,0.01,1,100] and strength[0.25,0.5,0.75,1,1.25] uses original HOST normalized-query targets at fixed parent normalizers on192current word windows. Validate causal QK-score error on64word windows, check all128current and1536historical source-routing probes, with the final32current windows outside the protected span. All q_norm/K/V/O weights remain fixed. Require10% score-MSE improvement and rounded source margin8; no softmax, native nonlinear replay, answers, gradients, or iterative refit. It uses existing dense Q3 edit support.

A189 remains the only running heavy process, full H1 session63058. After completion: run the A191 helper then its analysis; run A192 and A193 sequentially; assess results and build appropriate independent candidates. A187 remains available but unbuilt. No forward or quick calls are permitted;32full units remain after the current reservation.


### A194 final native-gain margin repair prepared (2026-11-10 21:36:46 CET)

A194 analyzer/compiler are syntax checked and not run. Parent is the certified A174/A176 text trajectory, independently of A189. Only final post-MLP gains outside every row touched by an arithmetic Down column may change; all arithmetic gains, G/Up/Down and protected weights stay fixed. Recorded native features reconstruct the final branch; the parent state/predictions must match the recorded diagnostic within declared floating-point tolerances. Original HOST winner/rival margin targets use the union of HOST/parent top8competitors, deduplicated per token. A chunked one-shot linear solve has at most640unknown gains, avoiding the oversized A170 per-token kernel. Fixed parent final-state normalizers and original HOST targets remain unchanged across ridges[1e-6,1e-4,0.01,1,100] and strengths[0.25,0.5,0.75,1,1.25]. Training192word windows, validation64. A candidate must add at least8validation matches, pass a complete protected-vocabulary readout check, and keep paired HOST state RMS within5%of baseline. Exact current-format final states are unavailable, so official full arithmetic/text scoring remains necessary. This is a prepared direct-weight repair, not a result.

A189's arithmetic portion has completed24,960/25,000exact/digit-prefix correct(99.84%); its text/noise/two-problem stages are still running as session63058. No rung pass is claimed from the partial arithmetic result. The post-score heavy queue is A191helper+analysis, A192analysis, A193analysis, A194analysis, one at a time. A187 is an independent static candidate still awaiting build/full scoring.


A195 context-digit diagnostic prepared (2026-11-10 21:42:28 CET): `experiments/analyze_a195_context_digit_codes_v1.py` is syntax checked and not run. It compares the existing A45 constant/digit fields on single digit tokens in A119old512word windows, distinct fresh A150current word windows, current H1A-source positions, and the original900operand prefixes. Upstream0–2/inputnorm3 identity is checked against each recorded parent. It reports direct digit-code error and error after dividing by that token's own constant field, by digit and in aggregate. No weights or new trial values are produced; no sums, native activations, new attention probabilities or heldout data are inspected. This can test source-code context sensitivity but cannot establish H2 gate/routing correctness. Queue it after A191/A192/A193/A194 when the current A189full score ends.


Further unimplemented option (2026-11-10 21:45:51 CET): if the prepared repairs fail, inspect single-branch post-norm gain repairs using A140's full paired HOST/A123 branch captures. A142 is identical to A123 through layer16, so exact current states before/after those branches can be reconstructed from recorded raw outputs and RMS/gains. A gain changes an entire native output row with one scalar, unlike Down repairs restricted to512donor features. Target each original paired HOST state independently at fixed current branch input; keep every arithmetic MLP output gain fixed and skip programme attention2/3/7. Treat alternatives as SINGLE-branch candidates, never compose repeated corrections for accumulated drift. A194 already covers a stronger final-layer gain/margin construction. No such broader branch-gain script has been written or run, and no new scale trials have been chosen.


### A196 position audit rules out a BOS-only preservation failure (2026-11-10 21:48:54 CET)

A lightweight single-thread read of saved prediction IDs (`analyze_a196_text_position_agreement_v1.py`) completed in5.53seconds, peak352,536KiB, alongside the sole heavy score process. It loaded no model or checkpoint weights and used no meter. The certified A176 calibration trajectory matches HOST at all256/256BOS positions. All366disagreements occur in the other16,128positions (15,762matches). The lowest-agreement position is3with241/256matches; errors are not a constant BOS offset. The full per-position table is in `artifacts/a196v1_text_position_agreement.json`. Thus the hypothesis that one shared BOS error explains the agreement plateau is rejected.

Cross-check of older work: A140 already tried33original-HOST branch gain repairs, including all MLPs and native attention except2/3/7; its aggregate local gain was only0.07483%, below its2%screen. The unimplemented broader gain idea above would have to use independent SINGLE-branch paired-state targets (not repeat A140's original-branch fit or compose accumulated-drift corrections). A194's final winner/rival-margin gain fit remains distinct and prepared.


### A189 full H1 improves preservation, still below the gate (2026-11-10 21:55:33 CET)

Official0777-score (`scores/a189v1_full_H1.json`) completed24,960/25,000exact/digit H1(99.84%) and97.94158935546875%text agreement, KL0.0025202891049510667, cross-entropy change−0.0009369889746171367, all finite. This is93additional matching text targets versus A173, and121versus A142; A189 is the new verified leader among candidates exceeding the H1arithmetic threshold. The98.5%text gate is still unmet, so no rung is banked. Noise exact/digit counts outof2000:1e-5:1998/1998;1e-4:1574/1999;1e-3:261/1595. Two-in-a-row:first1998,second given correct first1,both1. Runtime28:44.95,peak3,853,880KiB. Completed usage300forward/100quick/8full; remaining0/0/32.

A191 factual-mixture helper is running as session98938, followed by its matched output-repair analysis. Prepared A192,A193,A194,A195 remain queued sequentially; A187 remains an independent unbuilt candidate. The improved partial native V3 design is now supported by an official text score, but does not by itself solve preservation, contextual addition or noise fragility.


### A191 eligible; A192–A194 reject their static screens (2026-11-10 22:00:05 CET)

A191's factual-mixture helper completed19.83seconds, peak1,709,184KiB, saving232,652,800tensorbytes. Independent eight-window known-head relative error8.43893e-8; current word/format rounded-mixture raw errors1.27394e-7/1.82717e-7. Folder size before the new artifact38,258,384,829bytes. The helper uses only unique linear inverses of saved branch outputs; no new attention probabilities.

A191 output-repair analysis completed30.39seconds, peak2,539,560KiB. It selects128outputrows×768nativecoordinates, ridge0.01,strength0.75. Original paired HOST branch RMS0.1968863805→0.1836276302(6.73%); state relative RMS0.004904593457→0.004798591294; direction0.00292888209→0.002895778149. All128source probes remain correctly rounded, source errors[0.181945419,0.065056348,0.034370896]; maximum format raw-RMS change0.12145%; frame-null error2.86577e-9. Gate maximum on validationwords−33.18365,equals minimum733.45935. Predicted total4,978,700edits. It passes the declared local screen and will be built and fully scored independently.

A192 query-energy accommodation completed13.50seconds, peak1,515,396KiB, no PT/checkpoint. Best48carrier/ridge1e-6 centered causal QK score MSE0.001612736997 is worse than parent0.001550255500. Canonical pre-energy baseline0.01941416526; thus the old energy accommodation already captures the useful correction. The best trial preserves source margins but does not meet the10%improvement screen.

A193 format-null query repair completed15.77seconds, peak1,981,536KiB, no PT/checkpoint. Best ridge1,strength1.25 reduces validation score MSE0.000989992605→0.000914801475(7.60%), below the predeclared10%required gain. All25trials retain routing margins within the declared tolerance; rejection is the improvement screen. The protected format span and the32held-back current formats both retain their routing. No post-result lowering of that screen.

A194 final native-gain margin repair completed17.68seconds, peak2,096,772KiB, no PT/checkpoint. Parent final reconstruction exactly matches all recorded top1 IDs on4096validationtokens; baseline4004matches. Best restricted candidate4005matches, below required4012. No full-vocabulary candidate check was warranted. All arithmetic output gains remain fixed throughout.

A195 context-digit diagnostic is running. New A197script is prepared/syntax checked, not run: independent SINGLE-branch post-norm gain alternatives on exact A142/A123 recorded trajectories through layer16andattention17. Original paired HOST post-branch states are targets; pre-branch state/raw branch remain fixed; all arithmetic MLP output gains are protected; attention2/3/7andMLP17excluded. Ridges[1e-6,1e-4,0.01,1,100], strengths[0.25,0.5,0.75,1,1.25],192/64word split. Require5%gain in both paired-state and direction RMS, local format post-change<1, finite/edit/weight limits. Alternatives will never be composed as accumulated-drift refits.


### A191 built/full score running; context and branch-gain findings (2026-11-10 22:02:45 CET)

A191 checkpoint `experiments/a191v1_partial_reader_output_repair/delta.pt`, SHA256`0c28f74bb6c73e7c21757d7a2b7596d2e6a301cf83e01246163ec9621cd4e9b9`, has4,978,700edits,max1e6. Build17.15seconds,peak777,480KiB. Its helper/analyzer/compiler and inherited sources are now immutable. Full H1 is running as session67026,log`artifacts/a191v1_full_H1.log`,output`scores/a191v1_full_H1.json`. Eight full scores completed; ninth reserved,31fullremain. A189 remains the verified leader; no rung passed.

A195 completed11.46seconds,peak1,102,612KiB. Existing A45 source code on834old-word digit tokens has261wrong direct rounds, versus314with own-constant division. On469fresh-word digits,138direct errors(29.42%) versus207ratio errors(44.14%); direct maximum2.84081digits,ratio maximum6.72904. All256fresh windows differ from old512. On384current H1A-source tokens,0wrong,maximum direct0.08321/ratio0.09412. Original900bare operand prefixes supply2700digits,0wrong,maxdirect0.11950/ratio0.10241. This establishes context sensitivity of the old source field, not the exact cause of H2failure. It does not inspect heldout or new formats; no weights were fitted.

A197 independent SINGLE-branch paired-state gain analysis completed47.88seconds,peak4,881,480KiB. No candidate qualifies and no PT/checkpoint is written. The best fractional state-error reduction across32branches is only0.2263%(attention4), with localformatpostchange1.16956also beyond the<1screen. MLP0best0.1539%; all other branches smaller. Both original HOST and recorded parent final states reconstruct bit-identically from the saved raw branches. This rejects a broad simple gain-only correction; it does not bound richer feature-based repairs.

Prepared-next ideas during A191 scoring: A198 direct native query-gain repair using original HOST centered causal QK-score targets and small chunked normal equations, with arithmetic query channels fixed and routing separately checked. A199 independent final attention-output repair using factual saved-mixture inversion, original paired HOST post-attention state targets, and null constraints on the final arithmetic G/Up functionals. Neither is implemented or run yet.


### A198 and A199 prepared for the post-score queue (2026-11-10 22:09:06 CET)

A198 analyzer/compiler are syntax checked, NOT RUN. It independently tests layers2and3, never composes them. Fixed normalized Q and native K make centered causal QK scores affine in ordinary q_norm gains; all programme query-channel gains and Q/K/V/O weights stay fixed. Original HOST same-input scores are the target. Chunked two-window normal equations use192trainingwindows,64validate; shared-gain dimension<=256. A separate rotated-Q/rotated-K calculation checks the relative-angle score identity. Ridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25]; require10%validation score-MSEgain and source margins>=8−0.001 on128currentformats(plus1536historicalforlayer3). No softmax, native activation replay, gradients, answers or iterative fitting. An eligible layer is built alone using the compiler's required--layerargument.

A199 analyzer/compiler are syntax checked, NOT RUN. BaseA142 has exact A123trajectory through its final attention. Reconstruct incoming states by adding saved branches; require bit-identical original HOST/recorded final states and matching saved pre-MLP17states. Recover factual final head contexts via unique source-mixture inversion, with an independent old layer3known-head check. If final inverses are rank-deficient/inaccurate, save the failure and stop without weights. One original paired HOST post-attention-state solve uses fixed parent RMS. Protect all22final arithmetic Gate/Up linear functionals in the output-row nullspace. Use128outputrows, native-input counts[64,128,256,512,1024],ridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25],fixedrankingridge0.001/rawprojectionpenalty0.01. Require5%paired-state and direction improvement, word gates<-16, format raw-RMSchange<2%, explicit hand-programme ReLUideal-feature change<1%, no saturatedgate signchanges, relative frame-nullerror<1e-6, finite/edit/weightlimits. Check selected candidate gates across all256wordwindows. The final native MLP is not evaluated. Only an official score can validate resulting addition/text behavior. No large derived head artifact is saved.

AfterA191fullends, run A198 thenA199 sequentially. A187 remains a separately screened, unbuilt layer1relocation candidate. A189 remains the verified leader; no forward/quickunitsremain and31fullunitsremain after the A191reservation.


### A200 context-source field analysis prepared (2026-11-10 22:13:18 CET)

`derive_a200_context_digit_fields_v1.py` is prepared/syntax checked, NOT RUN. It fits only individual digit-token fields, using constrained class means and within-token covariance in two independent geometries: full640input space and the original HOST V3 row space. Exact null constraints protect training BOS/equals/nondigit means. Context-digit masses[0.01,0.1,1,10,100], nondigitquiet masses[0.001,0.01,0.1,1,10], fixedridgefraction1e-6. Targets are constant1 and centered single-digit identity; no operand-pair or sum labels. Old384+fresh192wordwindows train; old128+fresh64validate. Historical900bare prefixes split675/225with a fixed declared seed, and current128formats split96/32. Historical prefix captures are used as observed token contexts, without claiming exact current contextual-arithmetic trajectories. Both candidate fields are rounded float32 before validation. Require maximum direct digit error<0.25and no wrong rounds on every held-back group, quiet-constraint error<1e-6, finite/bounded fields. Only eligible source fields may be saved; no model checkpoint builder is written. No model execution, new nonlinear activations, softmax, gradients or iterative fit. Queue afterA198/A199.

A191 arithmetic completed24,960/25,000exact+digits, matchingA189; text/noise/two-problem stages remain running as session67026. Folder size38,536,306,018bytes(1.464GBheadroom) at22:12:56CET. Sources/checkpoints/captureprovenance remain unchanged; no additional heavy process is running.


A200 pre-execution extension (2026-11-10 22:23:18 CET): the same one-shot class-mean solve now also produces four signed binary indicators of the individual digit. This adds no trial scale values and uses no sum labels. Scalar and binary fields are screened independently: the scalar screen remains maximumerror<0.25; the binary screen requires no wrong signs, ownconstant>0.5and minimum signed-bitmargin/constant>0.25on every held-back group. Both use the same predeclared five context masses and five quiet masses. This tests whether native values retain categorical digit information despite an imprecise scalar readout. No analysis has run and no checkpoint was built.

A201 `analyze_a201_native_rms_error_v1.py` is prepared/syntax checked, NOT RUN; queue afterA200. It is a diagnostic on certified A174/A176nativefeatures, with no weightsfit or newscalevalues. It compares the actual native branch, deleting only displaced HOSTfeatures, current numerators at original HOST RMS, and the best positive per-token radial rescaling of the fixed numerator. The last is only a diagnostic lower bound for that fixed numerator, not implementableweights or a global bound. It also records zero-gain loss and whether the ideal normalized raw vector required for exact same-input HOSToutputs exceeds squarednorm640. This can distinguish output-capacity constraints from normalizer mismatch without replaying native activations.

Read-only clarification of A188: negative energy remainders can arise from its approximate reconstructed native numerator, not only an intrinsic fixed-gain obstruction. Some failures involve only a handful of states(e.g.layer1anchor93:4training/3validation;layer5anchor234:6/1). A201 will measure ideal target feasibility directly before drawing a structural conclusion.


### A202 exploratory native-V-only trial prepared; original rejection retained (2026-11-10 22:25:04 CET)

A202 analyzer/compiler are prepared and syntax checked, NOT RUN. They reproduce exactly one already evaluated A189 construction: valuefraction0,backgroundmass1,wordregularizer1e-8,old diagonalridge rule,common Upgain1e6(from the same original five gain values). No new scale value or fit family is introduced. The new script retains the complete A189trial record and requires its reproduced source-error and paired-state metrics to agree within1e-6. Its compiler requires the explicit exploratory flag and originaleligible=false.

This is a declared change of local screening policy: A189's original conservative currenterror<0.25/historicalerror<0.4screen failed. The preserved trial has currentmax0.3724942911,historicalmax0.4850974064,but zero wrong source digits on all three historical ideal-routing groups and both current factual-mixture groups. Thus an official H1test, which permits up to125errors, can measure the arithmetic tradeoff rather than treating that conservative screen as proof of failure. Combined carries and complete decoding remain unverified. No earlier rejection is erased and no pass is claimed.

Restoring V3 bit-identically to HOST removes163,839edits, predicts4,716,557totaledits and leaves299,711edit slots. The native-value branch difference on the recorded words becomes numericalzero; original paired HOST branch RMS0.1907265002 and state0.00486002586are only local diagnostics. This candidate may fail arithmetic. Its value is to measure the preservation/precision tradeoff and potential spare edit capacity. Queue A202analysis afterA198/A199/A200/A201; eligible candidates still need separate build/fullscore. A191 remains the sole heavy process(session67026).


A203 composition prepared (2026-11-10 22:27:55 CET): `compile_a203_relocated_current_reader_v1.py` is syntax checked, NOT RUN. It reuses the single selected A187layer1repair directly from its solution, preserving four existing function slots and moving only196→706. It accepts onlyA189orA191asbase, requires bit-identical weights through alloflayers0and1against A187's certified A176calibrationparent, and copies no other repair. Thus the recorded input1/nativefeatures/local screen remain exact for this subsystem, while downstream behavior still needs a fullscore. No new fit or scale value. AfterA191completes, chooseA191asbase onlyifitsfullH1arithmetic≥99.5%andtextagreement≥A189;otherwiseA189. This updates the planned A187test to the stronger current reader, without building/scoring a redundant old-reader parent. The original A187analysis, compiler and static result remain intact. A203source/manifest will link the inherited calibration provenance.

Postscoreanalysisqueue remainsA198,A199,A200,A201,A202, oneheavyprocessat a time. Assess results and build/fullscore eligible independent candidates, including A203. CurrentA191full session67026has completed arithmetic24,960/25,000and2048textwindows;noise/two still pending. No rung has passed.


### A191 full H1 completed: small new preservation gain (2026-11-10 22:31:43 CET)

Official0779-score (`scores/a191v1_full_H1.json`) reports24,960/25,000exact+digits(99.84%) and97.94769287109375%text agreement, KL0.0025131477597614776, cross-entropy change−0.0009975722148354293, all finite. This is8additional matching text targets versusA189 and101versusA173. A191 is the new verified leader; H1stillfails the98.5%text gate, so no rung is banked. Noise exact/digits outof2000:1e-5:1998/1998;1e-4:1577/1999;1e-3:261/1591. Two-in-a-row:first1998,second given correct first1,both1. Runtime28:25.49,peak3,856,092KiB. Completed budgets300forward/100quick/9full;remaining0/0/31.

As declared before the result, A203will useA191asitsbase: arithmetic clears99.5%andtext exceedsA189. No A203checkpoint has been built. A198directnativequery-score-gain analysis is nowrunning as session58865,log`artifacts/a198v1_native_query_score_gains.log`. A199,A200,A201,A202follow sequentially. Availablememory beforeA19815,980,478,464bytes. No otherheavyprocess isrunning. The leader edit-category audit is queued between analyses.


A198 provenance correction before any checkpoint (2026-11-10 22:33:00 CET): first analysis completedlayer2without an eligiblecandidate, then correctly stopped because currentQ3/q_norm are NOTbit-identical to A45v2. The current reference is `experiments/a61v1_query_gain_allocation/delta.pt`; A61retains the A45programme channel set but changes the query/gains. A198now separately verifies currentweights againstA61and verifies A61's immutablemanifest points to the exactA45channelmetadata. Layer2still verifiesagainstA27. No newtrialvaluesorbudgetusage. Originalfailurelogretained:12.44seconds,peak2,950,904KiB. Retry running70919,log`artifacts/a198v1_native_query_score_gains_retry.log`. Do not assume currentQ3equalsA45weights in future analyses; A45metadata alone stilldescribes the64programmechannels.

A191leader categoryaudit completed:588,645arithmetic/circuit-accommodation edits plus4,390,055native-onlytext-repair edits=4,978,700total. Immutablemanifest unchanged; full per-tensor table refreshed in REPORT.md and auditJSON `artifacts/a191v1_partial_reader_output_repair_edit_category_audit.json`. These are inherited purpose categories, not causal attributions.


### A198–A202 completed; A203 built (2026-11-10 22:43:41 CET)

A198retry completed28.20seconds,peak3,096,836KiB; no eligible repair/PT/checkpoint. Best B2query gain fit(ridge1e-6,strength1) changes centered causal scoreMSE0.001550255500→0.001542308928(0.513%); best A3same settings0.000989992605→0.000982633255(0.743%). Both miss10%; A3also weakens its minimumsource margin to7.998238, below7.999. Independent relative-RoPE and previous-score metric identity checks pass. CurrentQ3reference isA61,withA45channelmetadata.

A199completed29.67seconds,peak5,430,916KiB; no PT/checkpoint. Its factual inverse is unique onall256word64and128format14windows: minimumranks256/56,relative rawerrors9.34774e-8/3.31330e-7; maximum futurecoefficient8.32311e-7/0.001653018. The22Gate/Upfunctionals span rank6onselectedrows. Allformatgates are saturated; hand-programme ideal reproduces capturedfeatures within1.29715e-6relative. Best1024coordinate/ridge1/strength0.5 reduces pairedstateRMS0.01236976234→0.01234371345(0.211%) anddirection0.01184050151→0.01181433928. No trial meets5%stategain, whileall125retain the1%programmefeature guard. Failure is inadequate repair benefit, not source-inverse or arithmetic-frame feasibility. Bestmatrixwouldchange5,007,372scalars; actual support has some unchangedzeros.

A200completed12.42seconds,peak1,539,728KiB. No scalarfield passes0.25precision, and no native-value-span field passes either screen. Full-input binaryindicators DO pass: selectedcontextmass0.1/quietmass0.1,minimum signedbitmargin/ownconstant0.6970808342on352held-back worddigits,0.8872335724on675bareprefixdigits,0.9337356187on96currentH1source digits;0wrongbits inallthreegroups. Minimum ownconstant0.781222/0.959734/0.965518. The savedfield is `artifacts/a200v1_context_digit_fields.pt`, keyfull_input_binary; it is not a modelcheckpoint and provides no H2result. Scalarreadoutofthissamefieldstillhasword/bare errors. Bestscalarfull-input candidate hasmaxworderror0.557224(3wrong),bare0.447185,currentH10.361865. Thus categorical single-token information is more stable than the existing scalarcode, but transport through nativeV and programme gates remain to be designed.

A201diagnostic completed24.71seconds,peak3,552,284KiB; no weightsfit/written. CurrentnativebranchRMS→fixed-numerator bestpositive per-token radial-error bound:layer1 0.0535334→0.0500591;layer4 0.114003→0.0986945;layer5 0.0824990→0.0720409;layer6 0.226774→0.201756;layer7 0.199904→0.181417;layer8 0.334605→0.288884;layer17 3.539135→2.593017. Zero-gain losses are0onthese validationstates. Exact same-input HOSTbranch replication with the fixed gains requires normalizedraw squarednorm>640onmanytokens,but usually onlyslightly(e.g.layer2median640.233,max640.950;layer17median641.578,max652.668). These are narrowly conditioned geometric diagnostics, not a bound on checkpoint text agreement or on changed-gain designs. Restoring the HOSTrawRMS alone oftenworsens error(e.g.layer17 5.74524), so it is not the correct general repair target.

A202reproduced the single priorrejected Vfraction0/background1construction in22.92seconds,peak2,494,080KiB. All checkedmetrics matchA189's preservedtrial exactly. Its exploratoryfieldsolution/PTexists; compiler is nowrunning76279. Original conservativeprecision failure remains in its report. No newtrialvalue, arithmeticanswerfit or officialscore.

A203built fromA191in13.67seconds,peak976,120KiB:4,978,698edits,max1e6,SHA256`3a308fd0553faa5377af8261b1e967a023558d47c5c1b6fd5bd87574e802f472`. Upstream0/1identity and copiedprogrammefunction checks pass; only196→706moves. FullH1willstartafterthe A202build. A191remainsverifiedleader97.947693%/24,960H1;no rungpassed;31fullunitsremain until the nextreservation.

New unimplemented follow-ups from these diagnostics: (1) one-shot raw-anchor repair toward the optimal positive radial factor of the ORIGINAL fixed parent numerator and originalHOSTtarget, with analytic feasible-energyprojection; no candidate-output residualrefit. This differs fromA188's prescribedHOSTRMS. (2) categorical source transport: fit binary source fields through native V/O on independent individual-token contributions, check recorded mixtures and historical idealrouting, then onlyif feasible design gated paired-hinge quantization. A200doesnot itself solve this transport. No scripts/numbers assigned to these follow-ups yet.


A202build completed and A203fullstarted (2026-11-10 22:45:33 CET): A202checkpoint `experiments/a202v1_native_value_only_exploratory/delta.pt`, SHA256`9976d69b23960628ccfc8739106f145117c56b6303da6cc0cf3d9a909b5220bf`,4,716,557edits,max1e6. Build17.46seconds,peak776,288KiB. No officialA202score yet; its prior conservative source-precision failure remains explicit.

A203officialfullH1isrunning as session90368,log`artifacts/a203v1_full_H1.log`,output`scores/a203v1_full_H1.json`. It usesA191plus the screenedlayer1relocation,4,978,698edits,SHA256`3a308fd0553faa5377af8261b1e967a023558d47c5c1b6fd5bd87574e802f472`. CompilerA203,deriveA187andallinheritedsources are nowimmutable. Completed300forward/100quick/9full;tenthfullreserved;remaining0/0/30. A191remainsleader97.947693%,24,960/25,000H1;no rungpassed. Prepare the evidence-motivated raw-anchor radial repair while this sole heavyprocess runs. A202fullH1is also pending and should be scored separately.


### A204 radial-anchor repair prepared (2026-11-10 22:52:07 CET)

A204 analyzer/compiler are syntax checked, NOT RUN. Use certified A174/A176 native features for layers0..8and17,192training/64validationwordwindows. For each layer, rank up to5quiet output coordinates by squaredoutputgain/rawenergy, with |outputgain|<=1. Keep every other original-parent numerator fixed; derive its optimalpositive radial projection onto the original same-input HOST branch, cap it at the analytic minimum-RMS feasibility limit, and retain the originalanchor sign. Fit only the native coefficients in one Down row to this single target. Original reference/targets never change after a candidate; all programme Gate/Up/Downcolumns and all gains remain bit-identical.

Ridges[1e-6,1e-4,0.01,1,100], strengths[0.25,0.5,0.75,1,1.25]. Require>=5%nativebrancherrorreduction, improvements in pairedHOSTstateanddirection, finiteboundedweights and editbudget. Each accepted independent original-HOST branchrepair may be composed once if the totalbudget fits. For nonzeroquietgain, the radial target ignores the anchoroutput term, but actualbranch/stateerrors are evaluated. No exact current-parent nativeformatfeatures are available afterO3, so completearithmetic/text stillrequireofficialscoring. This differs from rejectedA188 by targeting the optimaloriginalnumeratorradialfactor, not the HOSTrawRMS. No gradient, nativeactivationreplay, additionanswerfit or iterative residualrefit.

A203fullH1 remains soleheavyprocess(session90368); A202fullH1pending. AfterA203finishes, runA204analysis before starting the nextfullscore. No rungpassed and30fullunitsremain.


### A205 categorical native-source transport prepared (2026-11-10 22:55:33 CET)

A205 analyzer is syntax checked, NOT RUN; no checkpoint compiler. It restores only V3 in its linear analysis to original HOST, keeps O3 HOST and currentQ/Kfactualmixtures, and solves13fields:4signedbinaryindicators for each of3independent source digits plus a common reference. Each of4heads supplies its independent tokencontribution and onequarterreference. The fit uses900historicalbareprefixes and optional individualdigit/BOScontexts from576trainingwordwindows, with192held-back wordwindows. It never fits an operandpair, completeproblem or sum.

Contextdigitmasses[0,0.01,0.1,1,10], backgroundmasses[0,0.01,1,100,10000], fixedwordregularizer1e-8and diagonalridgefraction1e-12. Backgroundcovariance uses unlabeled residualstates. Check float32 pre-MLPinputrows on128currentfactualmixtures(32heldback) and1536historicalidealroutingdiagnostics. Require everyreference>0.5, zero wrongbits, minimum signedbit/reference>0.25, originalwordgates<-16and equalsgates>32. Contextualworddigitcontributions are diagnostic only; this makes no H2claim. A successful field still needs arithmetic quantization gates and nativefeature repair before it could become a checkpoint. No softmax/nativeactivationreplay, gradients or iterative fit.

A203partialH1arithmetic is24,942/25,000exact+digits; text/noise/two remainpending. This is not a completedofficialresult. A204thenA205analyses are queued after90368, followed by the separate pendingA202fullH1test.


### A206 correlated root donors prepared (2026-11-10 23:02:59 CET)

A206 analyzer/compiler are syntax checked, NOT RUN. The A174capture records the original HOST rootfeature1159 at exactly unchangedlayer0inputs. Existing A180refits keep the old64-donor support; A187changeswhichnativeunitisremovedbutkeepsdonors. A206instead uses one fullnativefeature ridge0.001solve to rank donors by standardized coefficientenergy for that original missingfeature. Each declared support gets one independent solve against the sameoriginalfeature, with no candidate-residual target or iterative update. Reconstruct originalHOSTDown numerators using the predictedmissingfeature, inverse-scale the inheritedmarker gain, and restore the originalprogrammecolumn bit-identically.

Donorcounts[64,96,112,256,512], ridges[1e-6,1e-4,0.01,1,100],192training/64validationwindows. Require5%gain in nativebranchRMS, pairedHOSTstateRMS and directionRMS. Allprogramme Gate/Up/Downcolumn/gains stayfixed. Rootweights must be bit-identical across A174's calibrationparent, A191andA202; report separately which candidate fits each parent's editbudget. A191hasroom forsmallersupports; restoringnativeV3inA202makeslargersupportspossible. No exactnativeformatfeatures are claimed and any checkpoint requiresofficialarith/text validation. The widerrootrepair is distinct from the singlequietanchorA204. No modelexecution, nativeactivationreplay, gradients or answerlabels.

Queue afterA203full: A204,A205,A206analyses sequentially, then pendingA202fullH1. A203partialtextprogress doesnotprovide a completedscore.


### A207 binary-value injection feasibility prepared (2026-11-10 23:08:11 CET)

A207 analyzer is syntax checked, NOT RUN; no checkpoint compiler. A200's selected full-inputfour-bit single-token field is projected through nativeV3 by one covariance solve. A constrained1024-variablequadratic solve chooses a rank-at-most4correction, with exact16dualconstraints A^T U=I; factualrecordedwordmixtures define its nativebranch-outputmetric. This is a closed-form construction, not a loss-fitting loop.

Rawmetricmasses[0.0001,0.001,0.01,0.1,1], insertionfractions[0.0625,0.125,0.25,0.5,1], fixedmetric/projectionridgefraction1e-6. Checkfloat32V on352held-back ordinaryworddigits,675held-back bareprefixdigits and96held-back currentH1digits. Require0wrongbitsandabsolutesignedbitmargin>0.25inallgroups, existingwordgates<-16andformatgates>32, nativevaluebranchRMS at least5%betterthanA142's originalvalueedit and pairedHOSTstateRMSimprovement. Report A189half-valuequality separately; passingthisscreen neednotbeatA189. The selectedfield has no sumlabels; no source-head decoder/quantization gates/modelcheckpoint/H2result is produced. No nativeGELU/softmax/modelreplay, gradient or iterative residualrefit.

RunA204,A205,A206,A207sequentially after A203fullcompletes, then the pendingA202fullH1score. CurrentA203fullsession90368stillowns the soleheavyprocess slot.


### A203 full H1 completed; A191 remains leader (2026-11-10 23:13:46 CET)

Official0781-score (`scores/a203v1_full_H1.json`) gives24,942/25,000exact+digits(99.768%),97.9461669921875%textagreement,KL0.0026533224561715685,CEchange−0.000957027251757836,allfinite,H1false. The layer1relocation loses18arithmeticcases and2matchingtexttargets versusA191; its local5.53%nativebranchgain doesnotimprove held-outagreement. A191remainsverifiedleader97.94769287109375%/24,960H1.

A203noiseexact/digits outof2000:1e-5:1998/1998;1e-4:1581/1998;1e-3:281/1636. Two-in-a-row:first1998,secondgivenfirst0,both0. Runtime28:17.02,peak3,852,784KiB. Completed300forward/100quick/10full;remaining0/0/30. No rungpassed; no bestbank or finalverify. A204analysisisnowrunning42068,log`artifacts/a204v1_native_radial_anchor.log`; A205,A206,A207followsequentially, thenA202fullH1. Availablefreephysicalmemory6,346,641,408bytesbeforeA204.

Unimplemented binaryquantizer follow-up, conditional onsourcefeasibility: use pairedhinges for each of12bits, plus a pairedalways-positiveconstant, totaling26GLUs. WithcommonUp proportionalC, the difference ReLU(OFF+K(bit+delta*C))−ReLU(OFF+K(bit−delta*C)) is exactly2delta*K*C or0on saturatedvalidbits. A constantpair withbothgatespositive produces the same2delta*K*C, so sharedUpmakes both proportionalC² and theirratioisexactly0/1. FixedDowncombinations recover per-digit centeredcode fromliteralbinaryweights; no sumtarget. A format-OFF field could be solved in the nullspaceof96recordedequalsstates fromordinarywordandnon-equalsnegatives, then checked onheldbackstates. Needallordinarygates<-16andallformatgatespositive>32ornegative<-16; if no suchmarginexists, no quantizercheckpoint. This is design reasoning only, with no script/version/scalevalues yet. It also needs displacednativefeature repair and officialvalidation; previousnativeF3can only be a clearlylabelled input-surrogate ifV3changes.


### A204–A206 results and A208 readout check prepared (2026-11-10 23:20:17 CET)

A204 completed1:09.78,peak6,749,388KiB. Two independentnative-rowrepairs pass:layer8anchor255/ridge1/strength0.75, nativebranchRMS0.3346051646→0.3168300534(5.31%), pairedstate0.006242025843→0.006241803735,direction0.004800535442→0.004797808654. Layer17anchor439/ridge1e-6/strength0.75:3.5391354876→2.9694080554(16.10%), pairedstate0.01714404380→0.01686643165,direction0.01580738447→0.01554057540. MaxrawRMSchange1.5288%/3.2125%. Combinedpredicted4,980,225edits; solutionPTexists, no modelcheckpointbuilt. Otherlayersfailthe5%screen or pairedstate/direction.

A205completed13.66seconds,peak1,891,708KiB. NativeV/O-only binarysourcefields passH1componentchecks:contextmass0/backgroundmass1, maxinputrowweight0.05935472; minimumsignedbit/reference0.6980623over1536historicalidealroutingcases,0.7341212on128currentfactualcases,0.8510526on32currentholdbackcases; allsourcebits/digitscorrect. Minimumreference0.9855472historical,1.007316current. ExistingwordGmax−32.50298,equalsGmin733.45893. This is a newsourcefeasibilityresult, not completeaddition. The chosenH1field haslargecontextualwordbiterrors(288/479/588wrongownbits across352worddigits perhead)anddoesnot solveH2. PTcontains13physicalfields and theirfloat32preMLPinputrows.

A206completed17.58seconds,peak2,997,616KiB; noeligiblecandidate/PT/checkpoint. Best112donors/ridge100changes nativebranch0.01106753917→0.01089885240(1.52%), pairedstate0.003885684552→0.003827814895(1.49%),direction0.003230602543→0.003186443179(1.37%); allbelow5%. Relative predictionerrorofmissingfeature1.000038suggeststhe displacedrootfeature is poorlypredictable, evenwithcorrelation-rankeddonors. This doesnot ruleout relocatingtheprogramme to a differentfeaturewith differentdonors.

A207rank4binaryvaluefeasibility isrunning67968. A208 `analyze_a208_final_anchor_readout_v1.py` isprepared/syntaxchecked, NOT RUN: examineonlythe selectedA204layer17row onexactA176savedvalidationstates(64windows/4096tokens), with earlierlayersincluding8 unchanged. No fit or newscalevalues. Compare frozenreadouttop16rivals; runfullvocabulary comparisononlyif>=8extramatches. Requireparentstate reconstructionrelativeerror<1e-6andparentmatcheswithin2ofthe recorded4004. This check cannot predict thefullA204composition or behavioronheldouttext. QueueA208afterA207, thenpendingA202fullH1.


### A207–A208 completed; A202 full H1 started (2026-11-10 23:24:22 CET)

A207completed27.08seconds,peak2,190,320KiB. No valuecandidate passes and no PT/checkpoint iswritten. All5full-insertiontrials retainallheld-back bits withmargin>0.25, but partialinsertions donot. Bestsource-qualified fullinsertion(rawmetricmass0.1) has nativevaluebranchRMS0.3623582166 versusoriginalA1420.1042223871 andA189halfvalue0.05158607486; pairedHOSTstate0.006831583065versus0.005047436086original. It preserves352word/675bare/96currentH1digitbitlabels,but thenativecost istoo high. Directquadraticmetricidentity errors2.22e-16raw/1.11e-16post; fullspanidentityafterfloat32rounding9.16e-10. Thus rejectionistextcost,notfactorizationfeasibility.

A208completed11.53seconds,peak1,641,308KiB. Parentfinalstate reconstructionrelativeerror6.55374e-8; savedandrestrictedparentmatchesboth4004/4096. The selectedA204layer17anchor alone has3999/4096restrictedmatches, losing5. Fullvocabularychecknottriggered. This is contraryevidence despite its16.1%nativebrancherrorreduction. A204's local pass record remainsunchanged, but no A204checkpointwillbe built/scored now; its unobservedcombinedlayer8/17trajectory is not claimed tofailorpass. The localpairedstategain doesnotreliablypredict top1agreement.

A202fullH1isnowrunning95581,log`artifacts/a202v1_full_H1.log`,output`scores/a202v1_full_H1.json`. The sparsecheckpointwasbuiltbeforethisqueue; it exactlyrestoresHOSTV3anduses theonepreviouslydeclared exploratoryanalogdecoder. Originalconservativesourceprecisionfailure remainsdocumented. Its analyzer/compiler/inheritedsourcesarenowimmutable. Completed10fullscores;eleventhreserved,remaining29full. A191remainsverifiedleader24960/97.947693%;no rungpassed.

Next preparation: A209gatefeasibility forA205's successful nativeVbinaryfields. Compare an exact26-GLUpairedhinge decoder with a compact14-GLUquadraticbitdecoder. The compactform usespositiveB*(2mC−B), with m and reciprocaloutputscale derivedanalytically from independentpositivebitintervals; noanswers or iterativefitting. Bothneed a wholeformat OFFfield and explicit saturatedgatebounds. No A209scriptorcheckpoint yet. A210candidateidea is rootunitrelocation with eachunit's own correlation-selected donors, addressingA187's fixed genericdonors andA206's fixed displacedunit; unimplemented.


### A209 and A210 prepared for the post-A202 queue (2026-11-10 23:31:59 CET)

A209 is syntaxchecked, NOT RUN. It builds programmeGate/Up/Downcolumns only, usingA205's nativeV/Osourcefields. Two literaldecoders:26GLUpairedhinges, or14GLUquadraticpositive-bitplateaus. The quadraticcentersandreciprocaloutputscales follow analytically from positiveindividualbitintervals in96current+1024historicaltrainingcases; no sumlabels. Bothuse a two-GLUconstantdifference so commonOFFterms cancel from the reference.

The OFFfield is solved in the nullspaceof160equalsstates(96current,32historicala,32stress); targets−1onordinarywordandnon-equalsformatstates. Negativeformatmasses[0.01,0.1,1,10,100], normalizedridges[1e-6,1e-4,0.01,1,100]. Fixedhingehalfwidth0.25,constantmidpoint1,nominalUpgain1e6. OFFstrength is twice the analyticminimumtraining-negativesign-separationboundplus1e-6; Ggain is1.001times the boundforpositive64/negative32. UniformprogrammeDownrescaling enforces|w|<=1e6and its inverse onUp preservesrawprogrammeamplitude; this is analytic, not anothertrialsearch. Checkfloat32linearG/Up, hand-programmeReLUideals, fullnegativegates<-16,currentpositive>32/currentnegativebitgates<-16, allhistoricalsignedgates>16, zerowrongsource digitsandmaxper-source-digiterror<0.15. Holdback64word/32currentformats; historicalc512is excludedfromintervalfitandOFFconstraints. Any selectedprogramme stillneedsnativeunitallocation/repair/fullscore. No nativeGELU or softmaxreplay.

A210 analyzer/compiler are syntaxchecked, NOT RUN. At exactunchangedrootinputs, a full2048nativefeatureconditionalcovarianceinverse(ridge0.001) gives the exactRMSremovalcostofeachconditionalresidual. Rank5replacementunits excludingold1159; eachunit ranksitsownnativepredictorcoefficients. For each, independently fit originalHOSTfeature with donorcounts[64,96,112,256,512] andridges[1e-6,1e-4,0.01,1,100]. NativeHOSTDown reconstruction uses thosecoefficients and inverseinheritedmarker-gainscaling; originalrootprogrammeG/Up/Downfunction is copiedbit-identically to the selectedslot, with oldslotrestored. EverytrialstartsfromoriginalHOSTfeaturetargets; no candidate-output residualfit. Require5%nativebranch/state/directiongain, boundedweights, andparent-specificeditbudget forA191orA202. Actualformatnativefeatures remainunobservedand requirefullscore. This testsdifferentunit+differentdonors, distinct fromA206'sfixedunitandA187'sfixeddonors.

A202fullH1remainssoleheavyprocess95581. RunA209thenA210afteritfinishes. No otherheavyjobhasstarted; no newcheckpointwasbuilt.


### A211 conditional native allocation prepared (2026-11-10 23:44:23 CET)

A211 analyzer/compiler are syntaxchecked, NOT RUN, and mayrun onlyifA209produces eligibleprogramme rows. BaseA202uses nativeV3/O3. Nativecalibrationfeaturescome fromA174/A176, which has differentreader inputs; A211explicitly treats them as same-function inputsurrogates, not exactA202nativefeatures. Bothparents' Down3/gains mustmatch, and everynewprogrammegate must be<-16atallrecordedA174surrogatewordinputs as wellasA209's native-Vwordstates. OriginalHOSTfeatures/outputs supplyeveryrepairtarget.

Unitallocation uses conditionalfeaturevariance with fixedridge0.001, nativepost/rawimportance masses[0,0.01,0.1,1,10], anddonorcounts[256,384,512,768,896]. Each removed-unitset selects its owncorrelateddonors. One independent feature-reconstruction solve usesridge0.001; ordinarypostMLPgains receiveone closed-form fit tooriginalHOSTnativeoutputs, with everyprogrammeoutputgain fixed. No residualrefitloop. Gate/Upfunction order and Downcolumns are exactlyA209'shandprogramme.

Declaredexploratorylocal screen: nativebranchproxy<1.5timesoldstagebaseline, pairedHOSTstate/directionproxies<1.1timesbaseline, finitefloat32boundedweights/editcap, and a derived-stateA-registererrorbound<0.25digit. The looser localfunction-preservation allowance is explicit because the candidate replaces the inaccurateanalognative-Vreader withbinaryquantization; it is not a textpassclaim. Unknownnativeformatfeaturesare boundedby |GELU(g)*up|<=|g|*|up|, followedbyrawtriangle/RMSintervalbounds. Onlyhand-programmeReLUideals are evaluated. TheA-registerbound is atfactual-mixture-derivedstates anddoesnotcertifythecompletechangedtrajectory. Fullscoring ismandatory.

A202partialarithmetic is24,566/25,000exact+digits(98.264%), below99.5%; text/noise/two stillrunning95581. ThereforeifA210selects anA191-supported repair, A191is the firstparent tobuild/fullscore; A202-supportedrootvariantsremainunverifiedalternatives. A209→A210→(conditionalA211) is the post-scoreanalysisqueue. No newcheckpointyet.


### A202 full H1 completed; A209 numerical correction (2026-11-10 23:54:16 CET)

Official0783-score `scores/a202v1_full_H1.json`:24,566/25,000exact+digits(98.264%),98.00491333007812%textagreement,KL0.002438598226968455,CEchange−0.0008662915085011491,allfinite;H1false. This is75additionalmatchingtexttargets versusA191,but394fewerarithmeticcases, soA191remainsleader amongarithmetic-qualifiedcheckpoints. The reportfrontnow explicitlydistinguishes thatleader fromhighesttextagreementregardlessofarithmetic. A202'snoiseexact/digitsoutof2000:1e-5:1956/1956;1e-4:1963/1963;1e-3:299/1033. Two-in-a-row:first1973,secondgivenfirst3,both3. Runtime28:02.50,peak3,854,264KiB. Completed300forward/100quick/11full;remaining0/0/29. No rungpassed.

A209firstattempt stoppedbeforetheOFF/gatefits because the truncatednumericalSVDnullspace didnotmeet the absoluteP@null<1e-8constraint. Originalfailurelogretained(`artifacts/a209v1_native_binary_gates.log`):5.17seconds,peak1,530,044KiB. The unmeteredanalyzer nowretains all160row-space directions, includingtinysingularones, leaving480nullspacecoordinates. It reportsboth numericalrankandactualfull-rowspanconstraintresidual. This restores the originallyintended exactconstraint, withnonewtrialvalues ormodelbudget. Retryisrunning92316,log`artifacts/a209v1_native_binary_gates_retry.log`. A210follows, andA211onlyifA209writeseligibleprogramme rows. No newcheckpointwasbuilt.

Unimplemented futuretext-repairidea: a boundedcritical-token versionofA170's boundarykernel, usingatmost512trainingmismatches plus512nearmatchingtokens, HOST-versus-currentwinningrivals, and a fullnativefeaturecovariance regularizer. This differs fromA170'sall24,576-rowHOST-winner/runnerkernelandwouldavoiditsmemoryoverrun. OriginalA176targets/normalizerswouldstayfixed; no script/ID/scalesassigned yet. No fit orresultisclaimed.


### A209 rejected; A210 root relocation in full score (2026-11-11 00:03:22 CET)

A209 retry2 completed12.46seconds, peak1,562,324KiB, no selecteddesign/PT/checkpoint. Fiftydeclaredtrials:27exceedthe1e6weightcap,4failtrainingnegative separation,19reachfullgatediagnostics butfail. Evencappedpairedhinges misgate currentholdbackandhistoricalformats; thefirstcappedpairedtrial has39wrongsignedgates/current128and7wrongdigits, whilehistoricalreferencescanbezero. Thus A205sourcebitfeasibility doesnotestablishaformat-gateddecoder. A211remainsunrunconditionalonthe missinggateartifact. The initialA209runfailedanabsolute-nullspaceassertion; retainingall160SVDrowdirections fixedit. Retry1thenfailedJSONserializationbecauseinvalidzero-referencecasesproducedNaN. Retry2records invalid-referencecountsandnullerrormetrics explicitly; thosecandidatesareineligible. Originalfailedlogsretained. No meterunitsusedbytheseanalyses.

A210completed32.45seconds, peak2,900,808KiB. Rootprogramme relocation1159→472with112owncorrelateddonors,ridge1passes: nativebranchRMS0.011067539167→0.005838742909(47.25%), pairedHOSTstate0.003885684552→0.001677470709(56.83%), direction0.003230602543→0.001649137731(48.95%). ProgrammeG/Up/Downfunctioncopiedbit-identically; allgainsfixed. MaxrecordedwordrawRMSchange14.4292%; nativeformatfeaturesnotobserved, soofficialarithvalidationremainsessential. A202parentcoulduse512donorsandstate0.00157298518at5,002,618edits, butA202failsarithmetic; thepredeclaredA191parentistestedfirst.

Built `experiments/a210v1_root_unit_and_donors_a191/delta.pt`: SHAa073a2b78db921f13747478eafaa2b834e8d523bdec270f230cfac2e61eaf234,5,009,243edits. Build13.15seconds,peak818,360KiB. Itsanalyzer/compilerandancestorsare nowimmutablebecause fullH1meterhasstarted(session23149,log`artifacts/a210v1_full_H1.log`,output`scores/a210v1_full_H1.json`). Oneheavyprocess. Remaining0forward/0quick/28full;11fullcompleted,oneinprogress. No rungpassed.


### A212–A214 prepared, unrun (2026-11-11 00:10:04 CET)

A212analyzer/compiler syntaxchecked, NOT RUN. A174/A176exactnativefeaturesandoriginalHOSTfinalstates,192train/64validationwindows. Selectatmost512trainingmismatches(HOSTwinner-vs-currentwinner)plus512smallestHOST-marginmatchingtrainingtokens(HOSTwinner-vs-HOSTrunner); atmost1024squaredualkernel, unlikeA170's24,576andresourceoverrun. Fullcovarianceofall12,288trainingnativefeaturevectors suppliesquiettextregularization. OutputmetricoriginalHOSTwinner-runnercovarianceon96existingstrongoutputrows, fixedridge0.01. Inputridges[1e-6,1e-4,0.01,1,100],dualridges[0.001,0.01,0.1,1,10],strengths[0.25,0.5,0.75,1,1.25]. Alltargets/rivals/normalizersfixedbeforeindependentclosedsolves; nevercandidateupdatesorgradientsteps. Programmecolumns/G/Up/allgainsfixed. Require>=8extra/4096restrictedmatches,pairedstateerror<=1.05baseline,rawRMSchange<0.001,finite/editlimits;thenfullvocabularylinearreadoutconfirmation. Noexactformatfeatures; anyselectedcheckpointneedsfullscore. ThiswouldbeaseparateA176candidate, notanexactrepairpredictionforA210/A191.

A213analyzer syntaxchecked, NOT RUN. SameA205native13sourcefieldsandtwohandprogrammes26pairedhinges/14quadraticplateausasA209. Replacesits160rowexactnullspacebyonegeneralized-covariancesolve: originalnegativefieldtarget−1onword/non-equals; positiveformatoutputdividedbyreferenceCpenalizedtowardzeroon96current+512historical_a+512historical_stress trainingequals. Negativeformatmasses[0.01,0.1,1,10,100],relativepositivemasses[1,10,100,1000,10000],fixednegativecovarianceridge1e-6. CholeskywhitennegativecovarianceandSVDpositivefeatures,avoidingthelargecarrier'ssquaredconditioning. OFFscale1.001timesminimumtrainingnegativeboundwith0.25minimumpositiveCreferenceguard,plus1e-6;Ggain1.001timesexact64positive/32negativeboundacrossall1120trainingequals. UniformDown≤1analyticcapandreciprocalUpscalingunchanged;26/14designskeepthesamebitintervals,sourceerror<0.15andallgate/edit/finiteguards. Heldbackcurrent32andhistorical_c512notfit. No sumtargets/nativeactivationreplay/modelcheckpoint. A209rejectionsretained.

A214analyzer/compiler are copiesofunrunA211adjustedonlytouseA213'sgateartifactandnewversionpaths. Theyremainconditional, UNRUN. SameexplicitA174nativefeaturesurrogate,A202base,5rawmasses[0,0.01,0.1,1,10],5donorcounts[256,384,512,768,896],originalHOSTtargets/ridge0.001,fixedprogrammepostgains,conservativenativeformatboundsandpredeclaredexploratoryscreen. IfA213fails,noA214run.

A210fullH1session23149remainsthesoleheavyprocess. Queueaftercompletion:A212,A213,conditionalA214. No new meterunitsusedbythese preparations.


### A215 independent early-group relocations prepared (2026-11-11 00:14:09 CET)

Analyzer/compiler syntaxchecked, NOT RUN. Layers1/2useA174certifiedHOSTnativefeaturesatexactA191inputs; original0..2weightsbitidentitychecked. UnlikeA187'sfixeddonors,A215ranksunitsbyexactconditionalRMSremovalcostfromonefull2048featurecovarianceinverse(ridge0.001),thenrankseachchosenunitgroup'sowndonorsfromitsfullconditionalpredictioncoefficientsweightedbyHOSToutputeffect. Fivepost/rawcostmasses[0,0.01,0.1,1,10]proposeunitgroups; donorcounts[64,128,256,384,512],featurefitridges[1e-6,1e-4,0.01,1,100]. EveryindependentfeaturefitusesoriginalHOSTnativefeatures; everyordinary-gainfitusesoriginalsame-inputHOSTnativeoutputs. No candidatetargetupdate or nativeactivationreplay.

ProgrammeG/Up/Downcolumnfunctionsarecopiedbit-identically tonewunits,originalnativefunctionsrestoredelsewhere;programmeoutputgainsfixed.192train/64validationwindows. Require5%nativebranch,pairedHOSTstateanddirectiongain;boundedfiniteweightsandfullparenteditbudget. Provisionalrawcalculationuseslow-rankupdates; eachimprovingeligiblecandidateisrecheckedwithroundedfloat32Downonactualrecordednativefeatures. Savebestperlayer,notanunverifiedcombinedclaim. CompilerallowsA191orA210rootparentonlyiftheinstalledlayers'originaltensorsmatchA191exactly; a changedrootisexplicitlyanunobservedinput-transferhypothesisrequiringfullscore. Likewisecomposingsuccessivelayersneedsfullscore. Nativeformatfeaturesremainunknown.

A210partialH1arithmetic24,959/25,000exactanddigits; text/noise/two pending, no completedscoreorpass. A215queuedafterA212,A213andconditionalA214.


### A216 quadratic radial-energy reservoirs prepared (2026-11-11 00:20:13 CET)

Analyzer/compiler syntaxchecked, NOT RUN. ExactA174/A176savednativeF17,originalHOSTpairedfinalstates,192train/64validationwindows. ExtendA204'ssinglelinearanchorwithalgebraicmulti-rowenergyfactorization; targetnoworiginalpairedHOSTfinalresidual, notsame-inputnativebranchalone. Choosequietrowsby(gain+1e-6)^2/nativewordrawenergy,|gain|<=1. Counts[1,2,4,8,16]. For eachindependentcount, projectHOSTdesiredbranchonto unchangednon-anchorparentnumerator; inverseRMSboundedto[0.5,2]timesoriginalandbyunchangedrawenergy. Reportcappedtargetsandnonpositiveoptima.

Quadraticbasis32nativefeaturePCAdirectionsplusoriginalanchorlinearmaps, jointlywhitened; atmost48dimensions/1176quadraticfeatures. Ridge[1e-6,1e-4,0.01,1,100]solvesoriginaldesiredenergyonce. OneeigenspectralPSDprojectionandlargestcountpositiveeigenmodesgivelinearDownrows; orthogonalProcrustesalignsthemtooriginalanchoroutputswithoutchangingtheirenergy. Strengths[0.25,0.5,0.75,1,1.25]. Thequadraticdesignisanalysis-only:installedmodelwouldstillhaveordinarylinearDownrows, sameGemmaarchitecture. No nativeGELU/softmaxreplay, gradients or iterativefitting. Allprogrammecolumns,G/Up,gainsfixed.

Require>=8extra/4096restrictedmatches,pairedHOSTstateanddirectionimprovement,wordrawRMSchange<0.25,andfinite/editlimits;thenfullvocabularyfrozenreadoutconfirmation. Noexactnativeformatfeaturesorother-parenttrajectoryclaim;officialfullscoremandatory. A216queuedafterA212,A213,conditionalA214,A215. A210fullH1remainssoleheavyprocess23149.


### A217 larger-corpus critical-kernel comparison prepared (2026-11-11 00:28:05 CET)

Analyzer/compiler syntaxchecked, NOT RUN. AppliesA212'sboundedcriticaltokenkernel method toA142'slarger512-windowrecordedcorpus,384train/128validation. Nativefeaturesandpre17statescome fromA123; scriptassertsA142differsonlyat17Down/postgain, reconstructsA123'srecordedfinalstates, andthencomputesexactA142finalstatesbylinearDown/RMS. TrainingrivalsareexplicitlyrestrictedtoHOST/A123top8union; notclaimedfull-vocabularycurrentargmax. Chooseatmost512mismatchesand512near-boundarymatchesfromtrainingonly. IndependentoriginalHOSTmarginstargets; fixedalltrainingcovariance,96strongoutputrowsandallprogrammecolumns/gains.

Same5inputridges[1e-6,1e-4,0.01,1,100],5dualridges[0.001,0.01,0.1,1,10],5strengths[0.25,0.5,0.75,1,1.25],fixedoutputcovarianceridge0.01. Thisisadataset/parentcomparisondeclaredbeforeeitherkernelanalysisruns, notiteratingA212'scandidateweights. Atmost1024squaredualgram, avoidingA170'slargeallocation. Require>=16extra/8192restrictedmatches,pairedstate<=1.05baseline,rawRMSchange<0.001,andfinite/editlimits;thenfullvocabularylinearreadoutconfirmsonbothparentandcandidate. Noexactnative-formatclaim;fullscoremandatoryifeligible. QueueA217afterA216.


### A210 full H1 completed: clean text down, mixed noise result (2026-11-11 00:31:59 CET)

Official0785-score (`scores/a210v1_full_H1.json`):24,959/25,000exact+digits(99.836%),text97.9339599609375%,KL0.002527786557414087,CEchange−0.001018036330687977,allfinite,H1false. Thelarge56.83%localrootpaired-stateerrorreductiondidnotimproveoveralltext:18fewerof131,072matchingtargetsand1fewerarithmeticcaseversusA191. A191remainsthearithmetic-qualifiedtextleader;A202stillhighesttextoverallbutfailsarithmetic.

A210noiseexact/digits outof2000:1e-5:1998/1998;1e-4:1998/1998;1e-3:26/1511. Themiddle-noiseresultis421moreexactanswers thanA191's1577,whilehighest-noiseexactfallsfrom261to26. Thesearethemeter'sspecifiednoiseobservations, notacausalprooforuniformrobustnessimprovement;changingtheeditmaskalsochangestheperturbationrealization. Two-in-a-row:first1998,secondgivenfirst1,both1.5,009,243edits,max|w|1e6. Runtime28:15.85,peak3,855,180KiB. Thischeckpointisretainedbutnotbankedasapass.

Completed300forward/100quick/12full;remaining0/0/28. No rungpassed. A212boundedcriticalkernelanalysisnowrunningasthesoleheavyprocess;afterit:A213,conditionalA214,A215,A216,A217. Noanalysisconstitutesamodelscore.


### A212 bounded kernel rejected (2026-11-11 00:34:20 CET)

Completed24.28seconds, peak3,052,108KiB.274trainingmismatches+512near-boundarymatches=786criticalobservations. Noeligiblecandidate/PT/checkpoint. Bestvalidationmatchtrialinputridge100,dualridge10,strength0.5:4006/4096versus4004baseline(+2,needs+8),pairedstate0.017144043804→0.017152486490anddirection0.015807384468→0.015817162994bothslightlyworse. MaxrawRMSchange1.02e-10; editcountunchanged4,978,700;Downmax100000. No full-vocabularychecktriggered. Thisdirectcritical-boundaryobjectiveandfullquietcovariancedidnotproduceageneralizinglocalgain. No meterbudgetused.

A213relativeformatgateanalysisrunning83593,soleheavyprocess;A214conditionalonitseligibleartifact,thenA215,A216,A217. A210complete;A191remainstextleaderamongarith-qualifiedcheckpoints.


### A213/A215 rejected; A216 availability fix; A219/A220 prepared (2026-11-11 00:39:42 CET)

A213completed11.84seconds,peak1,546,752KiB,noeligiblecandidate/PT/checkpoint.40of50trialsfailnegative-separation,8exceedweightcap,1failsunscaledgate signs,1reachesfulldiagnostics. Thecappedquadratictrial(negativemass0.01,positivemass1)hasallcurrentgatescorrect(minpositive104.1084,maxnegative−130.8076),butOFFbiasbreaksitsquadraticdigitvalues:maxsourceerror1.60536,current81wrongdigits/384,holdback31/96. Historical_calsohas1invalidreferenceand19wronggates. NoA214run. Thebestpairedtrialmissesthecaponlyslightly:Gmax1,032,234.875; itscommonGate/Upcanbehomogeneouslybalancedwithoutchangingtheidealproduct.

A215completed50.75seconds,peak3,319,816KiB,noeligiblelayer/PT/checkpoint. Layer1bestpairedtrialrawmass0,units[1809,706,1130,249,438],512donors,ridge1:branch0.05353338935→0.05199245551,pairedstate0.003442911066→0.003394419259(1.41%),direction0.002384488488→0.002340451851;below5%guards. Layer2bestpairedtrialunits[529,1634,715,1651,821],512donors,ridge1worsensbranch0.09900557→0.10283990andpairedstate0.007078064→0.007080880. Changingdonorallocationdidnotyieldaqualifiedtextrepair.

A216initialrunfailedbeforefittingbecause|outputgain|<=1providesfewerthan16quietrows. Failedlogretained:7.74seconds,peak1,753,984KiB. Rerunskipsdeclaredcountsforwhichnotenoughquietrowsexist;alloriginaltargets,quietnesslimit,ridgesandstrengthsunchanged. Itrecordsexecuted/unavailablecounts. No meterbudgetused.

A219andconditionalA220analyzer/compiler syntaxchecked, NOT RUN. A219recomputesA213'ssameoriginalcovariances/25scalechoices,withanalytichomogeneousbalance:Gatefactor=min(1,(1e6*(1-4float32eps))/maxabsGate),Upreciprocalfactor. Allfloat32margins/weightcapchecksrerun. No newtrialvalues. Strictall-groupssourcepolicyretainedasanexplicitflag; additionallypredeclareanexploratoryadmissionrequiringall128currentfactualcases(and32holdback)correct,maxerror<0.15andallsignedgates>16,word/non-equalsG<-16,currentpositiveG>32andnegativeG<-16,whileallowingatmost0.5%ofthe1536oldideal-routingcasestobeincorrectorunsaturated. Thecasecountusesexactunionofwrong/invalid/unsaturatedevents. ThisdoesnotclaimH1passed;fullscorestillrequired. Allhistoricalfailuresremaininreport. A220isA214'ssameunrunallocatoradjustedonlytouseA219balanceddesigns/versionpaths; ifnosourcepasses,noA220run.

A216retryissoleheavyprocess;queueA217,A219,conditionalA220. A218identifierreservedfornot-yet-implementedGate-preservingnativeUptextrepair; noA218trialsorcodeyet. Remaining0forward/0quick/28full;12fullcompleted,no rungpassed.


### A216/A217 completed; A219 balanced decoder still rejected; A218 ready (2026-11-11 00:49:09 CET)

A216retrycompleted17.03seconds,peak3,352,968KiB. Exactlyonefinaloutputrowhas|gain|<=1:439. Executedonlycount1;counts2/4/8/16unavailable, nottested. Bestmatchesridge0.0001/strength1.25:4005/4096versus4004,pairedstate0.01714404→0.01744566anddirection0.01580738→0.01614090worse;rawRMSmaxchange2.65316%.33basisdirections/561quadraticfeatures,15negativeeigenvalues,1positivefactor,trainingenergyrelativeRMS0.0498725. Noeligiblecandidate/PT/checkpoint/fullvocabularycheck.

A217completed31.73seconds,peak4,980,480KiB. Baseline8003/8192matches;bestinputridge1/dualridge1/strength0.25gives8005(+2,needs+16),pairedstate0.01578285731→0.01579211155anddirection0.01456317699→0.01457309146worse. Noeligiblecandidate/PT/checkpoint/fullvocabularycheck. Largerrecordedcorpusdidnotresolvethecritical-kernelgeneralizationfailure.

A219completed12.45seconds,peak1,557,684KiB. Noeligibleprogramme/PT/checkpoint;A220notrun. Bestpairedhinge(negmass0.01,posmass1)analyticallybalancesGateby0.968771318andUpbyreciprocal:Gmax999999.5,Upmax3563.9368,Downmax1e6. Allcurrent128sourcecasescorrect,maxdigiterror0.00917418;holdback32max0.00603272;minimumcurrentabsoluteG86.91895. Historical_a/stressallcorrect,maxerrors0.00840968/0.01091961andminimumG62.0542/58.7656. Historical_ccontains1invalidreference,6caseswrongorinvalid(7wrongdigitsamongvalid),and24caseswrong/invalid/orunsaturated;24/1536exceedsthepredeclared0.5%exploratoryscreen. All24areinthe512-casehistorical_cgroup, soitsheld-backfailurefractionishigherthanthepooledfigure. Quadratic14designstillhascurrentdigitbiasandfails. TheGatecapproblemisresolvedforonepairedconstruction;OFFgeneralizationremainsunresolved.

A218analyzer/compiler syntaxchecked, NOT RUNatthetimeofthisdeclaration. DistinctfromA184(whichreplacedGateswithareverse-formatgate): keepnativeGate,Down,allnormsandallotherlayersfixed;changeonlyselectednativeUp9rows. NativeGateactivationfactorsareinferredasrecordedGLUproduct/linearUpatsamerecordedinputs, neverevaluatedwithanonlinearactivation. ExcludeanyunitwhoseUpintervalcancontainzero, using650-termfloat32dot/productroundoffboundsatall4096savedtokens. PropagatefactorandUpuncertainty;thisisexplicitlynotafull-networkfloating-pointcertificate.

48training/16validationwindowsfromA174's64savedlayer9windows. OriginalpairedHOSTpost9statetargetatfixedparentnormalizers. Oneoriginal-targetconditional-correlationranking(inputridge0.001),oneoutput-directionprojectionsolve(ridge0.001),andindependentclosedUp-rowfits. Counts[4,8,16,32,48],ridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25]. Require2%state/directiongain,rawRMSchange<0.05,stategain>twicethepropagatedfactor/Upuncertaintybound,andfinite/editlimits. No native-formatcapture/preservationclaim; anycandidatewillneedofficialfullscore. CompilerusesoriginalA176parentonly.

Noofficialscoreisrunning. A218analysisstartsnext;no newmeterunitsused. Remaining0forward/0quick/28full;12fullcomplete;A191arithmetic-qualifiedtextleader;no rungpassed.


### A218 rejected; A221/A222 quiet-source construction prepared (2026-11-11 00:54:31 CET)

A218completed25.59seconds,peak1,924,412KiB.135of2048nativeGatefactorshaveprovablynonzeroUpintervalsatall4096capturedtokens. Noeligiblecandidate/PT/checkpoint. Best4units[58,1068,173,1205],ridge1,strength0.25:pairedstate0.005415565064→0.005404286465(0.208%),direction0.004242901321→0.004241846676,rawRMSmaxchange0.27570%. Factor/Up-roundoffstatebound0.00130717isalsomuchlargerthantheobservedgain.2,560addedUpedits;nominaltotal4,981,260. Thisdoesnotestablishaqualifiedrepair.

A221analyzer syntaxchecked, NOT RUNatthedeclaration. ReusesA205'sindependent900prefixsource-token/13-fieldtargetsandfixedcontextmass0,butreplacestheordinary-wordpenaltymetricwithfullnormalizedactualnative-V/OwordstatesbeforeMLP3(192trainingwindows),insteadofanindependent-headcovariance. Wordpenalties[1e-8,1e-6,1e-4,0.01,1],backgroundmasses[0,0.01,1,100,10000],fixedsourcecovariancediagonalridge1e-12. Each25fitindependentlyusestheoriginalsource-targetsandunlabeledwordquietness. KeepallfieldsmeetingA205'sunchangedsourceguards(reference>0.5,no wrongbits,signedbit/reference>0.25acrossall5sourcegroups). Recordactualtrainingwordfieldmax/RMS. NoneoftheseareH2fieldsorwholeadditionfits.

A222analyzer syntaxchecked, NOT RUN. AcceptsoneeligibleA221fieldkeyandrerunsA219'ssame25OFFcovariancechoiceswithanalyticGate/Upweightbalance. Hingehalf-widthnowmin(0.25,0.25*minimumtrainingindividual-bit/reference margin), determinedoncefromcurrent96+historical_a/stress, notavalidationtuningloop. Allword/current/historicalchecksandA219strict/exploratoryscreenareunchanged; historicalfailuresarepreserved. ThequieterfieldandappropriatethresholdwidthmayreducehowmuchOFFsuppressionisneeded. Candidateswillbestillonlyprogrammerows;nativeunitallocationandfullscoresremainnecessary. RunA221thenA222sequentiallyforeachsource-eligiblefield. No newmodelcheckpointorofficialscoreyet.

Unimplementedfollow-upideas(no code/trials assigned): nativeMLP9Down-columnscalarrepair, and sparseunrestrictedDown-columnrepair. BothreuseexactrecordednativefeatureswithoutinferringnativeGatefactors, preserveG/UpandavoidA218'sdivisionuncertainty. TheydifferfromA182'ssparseoutput-rowrepair; anydesignwillneeditsownpredeclarationandfullscore.


### A221 field catalog passes12source alternatives; A222 batch started (2026-11-11 00:59:42 CET)

A221completed18.05seconds,peak1,944,352KiB.12of25fieldsmeetallunchangedsourceguards:wordpenalties1e-8,1e-6,1e-4,0.01,eachwithbackgroundmass0.01,1,100. Wordpenalty1andbackground0/10000donotqualify. Thequietesteligiblefield(word0.01,bg0.01)hastrainingwordfieldRMS0.03675099,max0.39467951,minsourcebit/reference0.459226;word0.01,bg1hasRMS0.04210006,max0.42206992,minsourcebit0.68350741. Compareword1e-8,bg1:RMS0.11229263,max1.12070290,margin0.69806369. Thesearesourcefields,notaddersorH2passes.

A222gateanalysesnowrunsequentiallyforeachof12eligiblefields,highestwordpenaltyfirst. Oneheavychildatatime;eachkeepsitsownlog/JSON,aggregate`artifacts/a222v1_catalog_gate_runs.json`. Noofficialscoreorcheckpointyet. ConditionalallocatorforanyselectedA222designstillneedstobepreparedasnewversion.

Correctiontoearlierunimplemented-idea note: readingA182'sactualsourceconfirmeditALREADYrepairedsparseDownCOLUMNS(8/16/24/32/48),withoutputrowsrestrictedto|gain|>=5. Itwasnotanoutput-row-onlyrepair. Do not repeatthatdesignbasedontheearliermisdescription. A184likewisereplacednativeGateswithareverse-formatgate;A218wasdistinctbutfailed. No newA224/A225experimenthasbeenimplementedordeclared.


### A222 yields two strict binary source-gate passes; A223 prepared (2026-11-11 01:04:16 CET)

All12A222fieldrunscompleted,eachoneheavyprocesssequentially;maximumpeak1,560,308KiB. Sevenfieldsselectpaired_hinge_26:twomeetstrictall-groupsguards,andfiveonlythepredeclaredexploratoryguard. Noquadratic14designpasses. Strictfieldsarewordpenalty0.01/background0.01andword0.01/background1. Allsource/gateerrorsandtrialvaluesremaininseparateper-fieldJSONs;aggregate`artifacts/a222v1_catalog_gate_runs.json`.

Firstallocationusesstrictword0.01/background1 becauseitminimizestheworstsource-digiterroramongstrictfields:0.00642259137overall,current1280.00444359575,holdback320.00392838097,historical_c0.00461795242.0wrongdigits/bitsand0invalid/unsaturatedcasesovercurrent128andall1536historicalchecks. MinimumsignedGoverall26.78052,current62.38672;word/non-equalsGmax−930.84375. Gmax316,856.3125,Upmax2263.9812,programmeDownmax1e6;OFFnegativemass0.1/positivemass1;hingehalfwidth0.17409455. Otherstrictfieldworsterror0.00887579726,minimumG17.42529,Gmax400,803.6875. Thisiscomponentfeasibilityonly,nofulladdition/H2claim.

A223analyzer/compiler syntaxchecked, NOT RUNatthedeclaration. GeneralizesunrunA220allocatoronlytoacceptanA222gateartifactandoutputprefix;sameA202base,nativeF3fromA174/A176explicitinput-surrogate,sourcefactualformatstates,originalHOSTnativefeature/branchtargets. Same5rawmasses[0,0.01,0.1,1,10],5donorcounts[256,384,512,768,896],fixedrank/featurefitridge0.001,programmepostgainsfixed,oneordinarygainfit. Programmegatesmustalsobeoffonthesurrogatewordinputs. Nativeformatfeaturesareboundedby|GELU(g)*up|<=|g||up|,neverreplayed;actualregistererrorbound<0.25needed. Sameexploratorynativebranch<1.5baseline,pairedstate/direction<1.1baseline,finite/editguards. Completecheckpointwouldrequireofficialfullscore. CompilerincludesA221catalogdependenciesandA222sourcehashes;GUp/sourcelegacy4-unitmetadataexplicitlymarkednotactive.

A223strictword0.01/background1allocationstartsnext,soleheavyprocess. Noofficialscorecurrentlyrunning;0forward/0quick/28fullremain,12fullcompleted. A191stillverifiedarithmetic-qualifiedtextleader;no rungpassed.


### A223 first allocation stopped at surrogate gate check (2026-11-11 01:11:47 CET)

The strict A222 word0.01/background1 component did not qualify for native allocation. At A174/A176's recorded native-feature input surrogate, a new programme Gate reaches +2050.0234375, so the programme is not proved off there. Its gates remain off on the actual native-V word states used by A222; these are distinct input trajectories. No donor trials ran, no checkpoint was built, and no score unit was used. Runtime23.91seconds; peak2,695,112KiB. All source-field and actual-input results remain recorded; this is an additional failed integration check.

Run the second already-declared strict A222 field, word0.01/background0.01, through the unchanged A223 allocator next. If it also fails, prepare a new OFF-field design with both actual and surrogate ordinary-word input populations in its original negative covariance; do not bypass the A223 check. No rung has passed; remaining0forward/0quick/28full.


### A223 second field rejected; A224/A225 union-text gate construction prepared (2026-11-11 01:13:15 CET)

The unchanged A223 allocator also rejects the second strict field (word0.01/background0.01): maximum surrogate word Gate+7934.6328125. No donor trials or checkpoint. Runtime28.57seconds, peak2,692,980KiB. Thus neither strict A222 field has the required OFF property on both input populations.

A224 is syntax-checked and unrun at this declaration. It changes A222's original OFF-field negative covariance to the equally weighted union of actual native-V word states and A174/A176 native-feature-surrogate word states. Each population has192 training and64 validation windows; the actual non-equals format negatives and1120 positive format constraints are unchanged. Both complete word populations must satisfy Gate<-16. Source fields are fixed A221 alternatives w0.01_b1 and w0.01_b0.01; no source refit. Same negative-format masses[0.01,0.1,1,10,100], relative-positive masses[1,10,100,1000,10000], fixed negative ridge1e-6, analytic OFF/gate/hinge/down/balance rules, and strict/exploratory source guards as A222. Every fit still uses the original negative/format targets, not a candidate residual. No native GELU/softmax replay or sum labels.

A225 analyzer/compiler are syntax-checked copies of the unchanged A223 allocation policy, accepting A224 programmes and recording new provenance. They remain conditional on an eligible A224 artifact. Same raw masses[0,0.01,0.1,1,10], donors[256,384,512,768,896], original HOST reconstruction targets, fixed ridge0.001, fixed programme gains, and conservative format-register bound<0.25. No gate-check bypass. Run A224 on the two declared fields sequentially, then inspect before any allocation/build/score. Remaining0forward/0quick/28full; no rung passed.


### A224 union-text gates pass; A225 allocation starts (2026-11-11 01:14:30 CET)

Both declared source fields now pass strict source/gate checks on all groups, including both ordinary-word populations. No wrong, invalid or unsaturated source cases. The preferred word0.01/background1 construction has worst source-digit error0.00575680124182, minimum signed gate26.587646484375, actual word maximum Gate-3684.7734375, surrogate maximum-926.8203125. Gate/Up maxima317,679.5625/2263.9812; negative-format mass0.1, positive mass1. The word0.01/background0.01 alternative has worst error0.00885678888684, minimum signed Gate16.762939453125, actual/surrogate word Gate maxima-10241.265625/-1292.453125. Maximum process peak1,993,900KiB. This resolves the specific A223 integration failure without weakening its guard, but remains a component result.

Run A225 allocation on word0.01/background1 first, retaining the other strict alternative. No new checkpoint or official score yet;0forward/0quick/28full remain.


### A225 rejects native displacement; A226 conditional allocation prepared (2026-11-11 01:18:38 CET)

A225 completed in0:31.95, peak2,873,580KiB, no eligible allocation/PT/checkpoint. All25 allocations pass the known word-gate and format bounds, but native repair is too costly. Best branch trial: raw importance mass0,896 donors,5,004,467 edits. Same-input native branch RMS0.08749040894→0.32445717797; paired-HOST state proxy0.00699629526→0.00812465698; direction0.00555088340→0.00669292520. Its maximum current A-register digit-error bound is0.02113619535, programme/native-norm-bound ratio>=3188.87, and current programme gates>=62.3733. Thus source quantization works locally while native displacement fails the declared preservation screen. The other strict field would have the same ideal Down programme and allocation cost; no redundant allocation run is planned.

A226 analyzer/compiler syntax checked, unrun at declaration. New allocation policy replaces approximate unit importance with exact post-RMS one-unit conditional-removal cost, and marginal-covariance donors with each removed group's full conditional predictors, using original HOST native features. This adapts the previously recorded A215 technique to the new26-unit binary programme, not a new claim about A215's failed layers1/2. Raw importance masses[0,0.01,0.1,1,10], donor counts[256,384,512,768,896], feature-fit ridges[1e-6,1e-4,0.01,1,100]. Ranking ridge remains0.001. Every feature and ordinary-gain fit still uses original HOST targets; no candidate residual refit. Same A225 surrogate/format/edit guards, no relaxation. Run the preferred A224 word0.01/background1 programme. No official score running;0forward/0quick/28full remain.


### A227 compact conjugate decoder prepared (2026-11-11 01:19:56 CET)

A227 analyzer is syntax checked, unrun. It retains the same A224 union-text OFF fit, source fields, data splits, five negative-format masses[0.01,0.1,1,10,100], five positive masses[1,10,100,1000,10000], and all source/gate guards. It compares five structural hand programmes with14,17,20,23,26GLUs: the largest0,1,2,3,4bits of each digit use exact paired hinges; the remaining bits use a quadratic plateau. This is not an answer fit.

Unlike the earlier rejected quadratic design, each positive bit's interval is measured from its complete unscaled Gate/C, including OFF, on the fixed training source-bit groups. Up is its conjugate2*center*C-Gate. Thus positive output is center^2*C^2-(Gate-center*C)^2; one analytic interval scaling gives the minimax plateau on that training interval. Constant reference still uses the paired-hinge difference, so the shared OFF term cancels there. Independent per-bit interval targets, original source labels, no sums/whole-problem targets, gradients, native activation replay, or candidate residual refits. Heldback current32 and historical_c remain diagnostics only.

A227 will run sequentially on the already-declared fields w0.01_b1 and w0.01_b0.01 after A226 completes. Any source-eligible compact design still needs native allocation and official full scoring. A226 conditional allocation is currently the sole heavy process; no official score running.


### A226 rejected; A227 compact hybrids and conditional allocation (2026-11-11 01:22:45 CET)

A226 selected no allocation; peak2,979,296KiB. Its best26-unit trial uses raw mass0,896 donors,ridge0.01: native branch0.31149165596 versus0.08749040894 baseline, state proxy0.00803443380 versus0.00699629526, direction0.00659252258 versus0.00555088340;5,004,467 edits and register bound0.02113590819. Better conditional donors reduce A225's cost modestly but do not meet the unchanged preservation screen. No model checkpoint or meter run.

Both A227 field analyses completed; maximum peak2,012,752KiB. The all-paired26 construction reproduces A224. The word0.01/background1 field also selects20 and23-unit hybrids under the previously explicit exploratory policy, not the strict all-groups policy. For the20-unit construction: current max source-digit error0.11558375504, holdback32 max0.10077435052, no current wrong cases; historical_c max0.59575246830 and3/512 wrong cases. For23 units: current max0.01825966418, holdback0.01692421242, historical_c max0.27213775351 with0wrong cases, but the strict0.15 maximum-error guard fails. These historical limitations remain; no H1 pass is claimed.14/17units reject. The background0.01 field selects only23exploratory and26strict.

Run the unchanged A226 conditional allocator on A227 word0.01/background1's20/23/26 programmes, with exactly the already-declared5raw masses,5donor counts,5ridges and all original targets and guards. The26 component repeats the control; compact allocations are the new comparison. No official score currently running;0forward/0quick/28full remain.


### A228 direct scalar recombination prepared (2026-11-11 01:26:03 CET)

A228 analyzer syntax checked, unrun. It linearly recombines each of A221's12 already-saved source fields into the original four-row analog interface: D_normalized=sum(2^bit*signed_bit)/9+(2/3)*C. This is the binary identity for a single source digit; it adds no fitted target or trial constant. Existing common Gate and all native weights stay fixed, so no extra native units are displaced. Fixed common Up gain remains1e6; only the existing four Up rows would change.

Checks use the same factual current128, holdback32 and historical source groups, after actual Up-row float32 rounding. Require current source-digit error<0.25, historical<0.4, no wrong/invalid source digits, and the existing common Gate off on both ordinary-text populations. The source field catalog's original five word penalties and five background masses remain the only inherited fit scales; no new solve occurs. This comparison tests whether the actual-word quiet metric helps the analog native-value reader without binary quantization. Even a component pass would require official H1 scoring; no text improvement is claimed from an inactive Up-only change. A228 queues after the current A226 hybrid allocation.


### A226 hybrid allocation and A228 scalar recombination rejected (2026-11-11 01:30:46 CET)

A226's A227 hybrid allocation completed with no selected design; peak2,992,508KiB. Best20-unit trial uses raw mass0,896 donors,ridge0.01: native branch0.27677515058, state proxy0.00777316952, direction0.00631149287,4,992,970 edits, current register bound0.12453184858.23units gives branch0.29318582093,state0.00790409166,direction0.00646113949,4,998,719 edits, register bound0.03463653446. Both improve on26-unit displacement but still fail the unchanged native-preservation screen. The repeated26-unit control agrees with the earlier result. No checkpoint or score.

A228 completed18.66seconds,peak1,515,044KiB, no eligible scalar field/PT/checkpoint. The best existing field w1e-08_b1 reproduces A202's weak analog precision: current max0.37249427504, historical max0.48510237819, all source digits still round correctly. Increasing the actual-word quietness penalty does not improve these maxima. The ordinary gates remain safely off (-32.503actual,-32.437surrogate), but the declared0.25/0.4 source screen fails. No new scales or fits were used.

No heavy process or official score is running after these analyses. Reviewing final emission/cleanup geometry for the next distinct construction; no A229 design or trials assigned yet. A191 remains the arithmetic-qualified full-score leader, A202 highest text overall but arithmetic-ineligible; no rung passed. Remaining0forward/0quick/28full.


### A229 final native Up repair with observation intervals prepared (2026-11-11 01:35:14 CET)

A229 analyzer/compiler are syntax checked, unrun. This moves the native-Gate-preserving Up repair to exact A174/A176 final-layer inputs,192 training/64 validation word windows. It also removes A218's exclusion of a whole unit whenever any single Up observation is near zero. A native unit needs>=99% training observations with |Up|>2times its650-term float32 dot-product error bound. Less stable observations are masked in fitting and retained with explicit factor intervals at validation. Recorded GLU/linear-Up intervals are intersected with the native Gate's sign and magnitude bound; no native GELU or derivative is evaluated. All programme units are excluded.

One original-target input-covariance ranking uses ridge0.001. For each selected count[4,8,16,32,48], one original HOST final-state projection onto fixed native output columns uses ridge0.001; each native Up row receives an independent original-target solve with ridges[1e-6,1e-4,0.01,1,100]. Strengths[0.25,0.5,0.75,1,1.25]. Every target and normalizer stays fixed; no candidate residual refit. Native Gate, Down, all norms, arithmetic programme rows, and other layers remain unchanged.

Factor/division/Up-dot/product uncertainty is propagated coordinatewise through existing output gains and RMS, avoiding a global1e6-gain overestimate. This is explicitly not a full-network/all-floating-point certificate. Require improving paired state and direction, state improvement>twice this uncertainty RMS bound, raw-RMS change<5%, finite/weight/edit limits, and>=8 extra restricted validation predictions out of4096, followed by full-vocabulary confirmation of the same gain. Only that fully confirmed selected count may be compiled. No native-format preservation claim; official H1 score remains mandatory.

Final cleanup review found that A181 still had about41k nuisance on its ten cleaned coordinates and38k outside them at low emission gains; a larger ten-coordinate cleanup alone does not remove the outside residual. No new cleanup checkpoint or trials were made. A229 starts as the sole heavy process;0forward/0quick/28full remain, no rung passed.


### A229 final Up repair rejected; A230 branch comparison prepared (2026-11-11 01:37:37 CET)

A229 completed1:01.37,peak4,564,384KiB.2035 native units meet the>=99% stable-training-observation policy;0 interval-intersection inconsistencies. No eligible candidate/PT/checkpoint. Best restricted matches:16units,ridge0.01,strength0.25 gives4007/4096 versus4004(+3,needs+8), but state0.01727548860 and direction0.01593947450 both worsen from0.01714404380/0.01580738447. Best state:4units,ridge100,strength1.25 gives0.01705422608(0.524% better), direction0.01571389555, no prediction gain; uncertainty RMS bound0.00013515509 exceeds the state improvement0.00008981772. No full-vocabulary check was triggered. The observation-wise interval method restores access to most native units, but this final-layer fit still does not generalize enough.

A230 analyzer/compiler syntax checked, unrun. Apply the same interval-aware native-Up method independently to layers0 through9, using each layer's exact certified A174/A176 input/native features and original paired HOST post-layer state. Layers0–8 have192 training/64 validation windows; layer9 has48/16. Programme Gate/Up functions, all native Gates, Down, norms and other layers stay fixed. All selected units are native. Targets and normalizers are fixed before independent fits; no residual-refit loop.

Same five counts[4,8,16,32,48], ridges[1e-6,1e-4,0.01,1,100], strengths[0.25,0.5,0.75,1,1.25], fixed ranking/output-projection ridges0.001 and>=99% stable training rule. Because downstream layers are not replayed, these are local branch checks, not predicted final text scores. Require>=5% paired-state and direction improvement, state improvement>twice the propagated factor/Up uncertainty, raw-RMS change<5%, finite/weight/edit limits. Any selected layer needs its own official score; composing selected layers is not certified. Native-format activations are not replayed. Run layers sequentially with one heavy child at a time, retaining separate logs/JSONs.0forward/0quick/28full remain, no rung passed.


### A231 selective-bit reader prepared (2026-11-11 01:43:53 CET)

A231 analyzer syntax checked, unrun. Keep the old common format Gate as the reference and analog Gate. Use A224's strict, quiet source-bit Gate centers solely to classify selected bits, while the fixed A221 w1e-08_b1 analog field (A228's already-measured smallest scalar error) supplies C and all unquantized linear bit fields. No new fit. Two classifier fields will be compared: A224 w0.01_b1 and w0.01_b0.01.

For each selected bit, install center+shift*commonGate and center-shift*commonGate, both with Up=C. Their positive-region difference is exactly2*shift*commonGate*C; it vanishes for a negative bit. This gives the same reference factor as the existing analog rows, so only selected bits need quantization. The shift is analytically min(0.25,one quarter of the minimum training signed-center/commonGate ratio,one quarter of the minimum negative-center/absolute-commonGate ratio). Training groups remain current96 plus historical_a/stress; both192-window word populations and non-equals enforce negative signs. No added trial scale.

Eight explicit bit structures: units bit0 only,bit1 only,bit2 only,bit3 only; units bits2+3; units bits1+2+3; all four units bits; units bit3 plus tens bit3. Programme sizes6,6,6,6,8,10,11,8. All other source bits retain their exact linear binary expansion and constant offset. Source guards use the original analog-reader limits: current errors<0.25,historical<0.4,no wrong/invalid digits,all correctly signed source gates>16,and all word/non-equals gates<-16. Uniform Down/reciprocal-Up and Gate/reciprocal-Up balancing are analytic weight-cap accommodations. Only hand-ReLU saturation ideals are evaluated.

Any eligible source programme still needs native allocation and full scoring. First allocation will prefer the smallest eligible unit count, then smallest current source error and historical error, across the two classifier fields. Equal-size programmes have the same zero-on-word native displacement class; do not repeat equivalent allocations unnecessarily. A231 queues after A230's sequential layer comparisons. No new checkpoint, score or rung pass.


### A230 branch comparisons rejected; A231 starts (2026-11-11 01:50:15 CET)

All ten A230 layer comparisons completed sequentially with no eligible candidate or checkpoint. Best paired-state reductions per layer: L0:0.0788%; L1:0.7159%; L2:0.1266%; L3:0.8201%; L4:0.6466%; L5:0.9198%; L6:1.0113%; L7:2.0515%; L8:0.3191%; L9:0.1726%. Every best trial and full per-layer result is retained in artifacts/a230v1_branch_summary.json and the ten analysis JSONs. None meets the complete5%state/direction plus uncertainty screen. Maximum process peak4,835,272KiB; no new resource overrun. The method exposes most native Gate factors but its limited Up-row changes do not provide a qualified repair.

A231 selective-bit source programme now runs sequentially for its two declared A224 classifier fields. The fixed analog field, eight structural choices, analytic shifts and source guards are unchanged from the predeclaration. No official score is running;0forward/0quick/28full remain,12full complete. No rung passed.

Unimplemented follow-up idea, no A232 code/trials assigned yet: clone native Gates whose exact factors are already recorded into quiet existing donor slots at final layer17, derive new Up directions from principal components of the original paired-HOST state difference, and solve Down once against fixed original native/paired-HOST targets. This could change output directions as well as Up directions, unlike A229/A230. Copying recorded Gate functions would permit interval evaluation on both original word and format inputs without new nonlinear activation execution; changed GEMM-row rounding would need an additional Gate Lipschitz bound. This is only an idea, not a result or authorization to replay a model.


### A231 selective reader passes10/11-unit source checks; A233 prepared (2026-11-11 01:53:35 CET)

Both A231 classifier-field runs completed; maximum peak1,765,084KiB. The ten-unit units_top3 construction and eleven-unit units_all4 construction satisfy the declared source guards for both fields. Maximum current source-digit error0.16744157706 and historical0.36584135699, now dominated by the unchanged analog tens field. No wrong/invalid digits or unsaturated source gates. All text/non-equals Gate maxima remain below−32.4374. Six-unit single-bit versions fail, sometimes because correcting one bit removes cancellation between their analog errors. Eight-unit units_top2 comes close but still has current maximum0.26601847111 (preferred classifier field), above0.25; its rejection remains.

A233 analyzer/compiler syntax checked, unrun. It is the same A226 conditional allocator, with an explicit design filter and transitive A224/A231 source provenance. Select the smallest eligible construction, units_top3(10units), from A231 w0.01_b1; both fields tie on maximum source errors, and this one has smaller Gate coefficients(245,415.140625 versus394,917.375). Its Upmax21,280.44140625, Downmax1e6. Same raw masses[0,0.01,0.1,1,10], donors[256,384,512,768,896], ridges[1e-6,1e-4,0.01,1,100], original HOST targets and all existing word/native/format/edit guards. No new fit scale or screening relaxation. Run only this ten-unit allocation first. No checkpoint or score yet.

A232 remains the previously recorded, unimplemented final native-Gate-cloning repair idea. Remaining0forward/0quick/28full; no rung passed.


### A233 source/native tradeoff; A234 explicit full-score exploration prepared (2026-11-11 01:58:21 CET)

A233 completed with no eligible allocation, peak2,958,720KiB. Best native-branch trial(raw mass0,896donors,ridge0.01):0.14539423842 versus0.08749040894 baseline, exceeding the1.5times guard; paired-state proxy0.00715465586 versus0.00699629526, direction0.00571428241 versus0.00555088340, and register bound0.16661788302 all meet their unchanged limits.Exact edit count is4,973,811; no checkpoint was built by A233.

A234 deliberately overrides only that native-branch guard for one already-recorded A233 trial. It introduces no new scale values or residual fit. Fixed raw mass0,512donors,ridge0.01 reproduces native branch0.14779293010,state proxy0.00718374789,direction0.00573374840,register bound0.16661786924, and4,729,584 edits.512 donors are chosen for286,684 spare edit slots, rather than claiming the smallest proxy error; the896-donor result is slightly better locally. All current/historical source checks, word Gate checks, paired-state/direction<1.1, format-register<0.25, finite/weight/edit limits remain mandatory. The original A233 rejection is embedded in the new report and manifest.

Purpose of the full-score experiment: test whether selected-bit quantization repairs A202's arithmetic without reintroducing its nonnative V3 weights. A202's full H1 arithmetic was only98.264%, while its text agreement98.004913% is still below the required98.5%. No full pass is implied. A234 analyzer/compiler syntax checked, unrun at this declaration. After successful reproduction/build, run exactly one full H1 score. A232 gate-cloning repair source is being drafted and has NOT RUN; no A232 result or trial declaration yet.


### A234 built; full H1 score starting (2026-11-11 02:00:45 CET)

A234 reproduced the exact prior512-donor trial:24.35seconds,peak2,879,084KiB. Built experiments/a234v1_exploratory_selective_reader/delta.pt, SHA5682cfc35f535d53455d450ec475019d6eed5467395ecc83b0d0fc9c36ec13dd,4,729,584 edited scalars,max|w|1e6. Build48.43seconds,peak780,232KiB. The native-branch guard override and original rejection remain explicit. The build manifest includes transitive A231/A224/A221 sources, the A233 rejected-trial analysis, original reader-mixture helper and all base ancestors. These construction sources and artifacts are now treated as immutable for the metered checkpoint.

Full H1 score starts next as the sole heavy process; output scores/a234v1_full_H1.json, log artifacts/a234v1_full_H1.log. It will reserve one full unit, leaving27; no forward or quick units remain. No partial result is a pass.

A232 cloned-native-Gate text-basis analyzer is now fully drafted and syntax checked, but NOT RUN. It still needs final static review and its complete trial/precondition declaration before execution. No A232 checkpoint/compiler or result exists.


### A232 copied-native-Gate text basis prepared and reviewed (2026-11-11 02:04:40 CET)

A232 analyzer is complete and syntax checked, NOT RUN. Parent A142 has exact recorded final native features on512 word windows (384training/128validation) and512 format windows at five existing register positions. Original HOST final states and frozen readout supply text targets. All upstream, final Gate/Up and format-input provenance is checked against the original captures. Existing arithmetic programme rows/columns and every norm remain fixed.

New text features copy existing native Gate rows into quiet native slots. Original paired-HOST state-error PCA chooses64 ordered response directions. For each, the source Gate is selected by original native-feature correlation with that response on training text; source reuse is allowed. Each new Up direction is one independent original-response ridge0.01 solve, then analytically normalized to unit training feature RMS. Source Gate factors come only from recorded products divided by linear-Up intervals.>=99% stable training observations are required; unstable observations are masked in the fit and bounded in validation. No native activation is evaluated.

Copying a Gate row may change GEMM rounding. The interval therefore adds a conservative copied-row allowance using global GELU Lipschitz2 and both dot-product error bounds. For tanh GELU, this bound follows from t*sech(t)^2<=1/2 (cosh(t)^2>=1+t^2>=2t), giving derivative magnitude below2; no numerical activation replay is used. As before, the interval scope omits a complete native Down/RMS/readout floating-point certificate.

Feature counts[4,8,16,32,64]; fixed Up ridge0.01; Down ridges[1e-6,1e-4,0.01,1,100]; original paired-target strengths[0.25,0.5,0.75,1,1.25]. Slots minimize original native post-gain output energy. Ordinary Down rows may change only at the old512donors plus new slots;96strong rows may change all native columns, preserving programme columns. Every Down solve targets the original parent editable contribution plus strength times the ORIGINAL paired-HOST difference at fixed original normalizers. No fitted candidate supplies a residual target. Rows with |gain|<0.05 instead preserve their original raw contribution.

Local word gate:>=16 extra restricted matches/8192, improving paired state/direction, state improvement>twice inferred/copied-factor uncertainty RMS, raw-RMS change<5%, finite/edit/weight limits. Then full-vocabulary nominal prediction gain must remain>=16. The same copied Gate factors are observed on unchanged format inputs, enabling linear interval evaluation there; require all2560 existing register winners preserved, factor-interval margin lower bound>1 and format raw-RMS change<2%. These are component checks, not a held-out pass. Only a fully confirmed candidate writes weights; official scoring remains necessary.

A234 full H1 is running as0787-score, session65510;27full units remain. A232 will run only after that heavy process completes. The report is provisional; no rung has passed.


### A235 complete reader-functional output repair prepared (2026-11-11 02:16:00 CET)

A235 analyzer/compiler are syntax checked and unrun. Parent A234 has completed the arithmetic portion of full0787 with24,994/25,000 exact; text/noise/repeated-problem evaluation remains active and this is not a rung pass. The new repair uses the original A191 factual attention mixtures with HOST V3 and A234's unchanged Q3, so its attention-input/context calculation is exact up to the recorded reconstruction tolerance. New downstream native MLP features are not evaluated.

Protect every distinct actual rounded Gate and Up row of A234's ten-unit programme, including float32 rounding directions, with input gains included. Normalize these functionals and retain SVD directions above1e-12relative. For each output-row count[64,96,128,192,256], rank rows by original paired-HOST branch-error energy, excluding absolute output gains below0.05. Fixed projection raw penalty0.01. Original-target conditional feature ranking uses ridge0.001; native input coordinate counts[64,128,256,512,1024]. Independent original-target repair ridges[1e-6,1e-4,0.01,1,100], strengths[0.25,0.5,0.75,1,1.25]. No previous candidate residual or whole-problem answer target.

Require at least5percent paired attention-branch improvement, improving paired state/direction, all256 recorded word windows' programme Gates below-16, all current equals Gate signs preserved with absolute values>16, non-equals Gates below-16, current A-register interval error<0.25digit, attention raw-RMS change<2percent on format, and all edit/weight limits. The native format MLP is bounded with |GELU(g)*up|<=|g|*|up|; only the hand-built programme's saturated ReLU ideal is evaluated. Select the least branch error, then state error, then edit count. This protects a local component, not the unobserved changed word trajectory. Official scoring remains necessary.

A232 remains the next queued heavy analysis after A234 finishes; A235 follows if resources permit. No new official run or checkpoint has been created. Remaining0forward/0quick/27full.


### A236 extra zero-gain energy-carrier feasibility screen prepared (2026-11-11 02:18:53 CET)

A236 analyzer is syntax checked and unrun. Unlike A216, which kept every gain fixed and therefore had only one sufficiently quiet raw-energy row, this diagnostic permits additional ordinary output gains to become exactly zero. It excludes every row with a nonzero arithmetic programme Down coefficient. Rank remaining rows by original training native-output energy relative to paired-HOST state RMS; counts[1,2,4,8,16]. Parent A142 supplies exact512word windows,384training/128validation.

For each fixed row set, analytically optimize the positive radial factor of the remaining ORIGINAL parent branch toward the original paired-HOST final state, with inverse-RMS domain[0.5,2]times parent and the tighter cap from unchanged raw-coordinate energy. Selected output coordinates contribute zero. This is an optimistic per-token feasibility calculation, not a deployable network: HOST validation states define only the diagnostic target, and no per-token factors or answers are installed. No native activation is replayed. Require>=16 extra restricted validation predictions/8192 and improving paired state/direction before preparing any realizable energy-feature fit. The diagnostic alone cannot establish impossibility or a full vocabulary/held-out improvement. No weights, compiler, or meter run will be produced by A236.

Queued after A232 and A235, with one heavy process at a time. A234 full0787 remains active; its25,000-problem arithmetic portion is24,994 exact, with text/noise still pending. No rung has passed.


A235 pre-execution review addition (2026-11-11 02:22:20 CET): after the original local selection, also check the selected output matrix on the same three historical512-window source populations used by A231. Use recorded source vectors and algebraic hard-position routing only. Require all source Gate signs unchanged, absolute Gate values>16, register interval error<0.4digit, and attention raw-RMS change<2percent in every group. A selected candidate failing this confirmation writes no weights; no fallback refit. These are source-component tests, not exact new model trajectories. No new fit scales. A232's compiler is now prepared and syntax checked; it refuses missing full-vocabulary or format confirmation and verifies every construction input/source hash. Neither analyzer has run.


### A237 causal-score-metric query repair prepared (2026-11-11 02:28:41 CET)

A237 analyzer/compiler are complete, syntax checked, unrun. Parent A234 has the same upstream0–2 weights and layer3 input as the exact A123/A150 captures. Keep query gains and native keys unchanged. As in A193, protect the first96 current equals query vectors using their exact544-dimensional nullspace; retain32 current holdback and1536 historical source-route checks.

For each head, form original HOST minus original parent normalized-query responses on192training word windows, with original query raw RMS fixed. Use one original-response canonical input covariance with fixed whitening ridge0.001, and16 original-response PCA output directions. Native keys and RoPE give the exact mean-centered causal score Gram within that fixed linear correction space. This involves no softmax, native activation replay or unrecorded attention probabilities. Each rounded query candidate is checked using its actual query RMS on64validation word windows.

Canonical input dimensions[8,16,32,64,128], score-metric ridges[1e-6,1e-4,0.01,1,100], correction strengths[0.25,0.5,0.75,1,1.25]. All bases, targets and metrics are computed from original data once; no previous candidate provides a residual target. Each head independently selects its smallest validation centered causal-score MSE among candidates with all source routing margins>=7.999, rounded protected-query change<1e-5 and finite/weight caps. A head without an improving candidate retains its parent. The combined mean MSE must improve at least10percent and meet the total edit cap before writing a checkpoint. These local score proxies do not establish a final text or arithmetic gain; an official score would remain necessary.

The largest normal matrix is2048square; key metrics are streamed by small output blocks, without a token-by-parameter design matrix or large sample kernel. A237 queues after A232, A235 and A236. A234 full0787 is still running; no new meter unit or checkpoint used.


### A234 full H1 completes: arithmetic improves, text still fails (2026-11-11 02:30:04 CET)

Full0787-score finished successfully in28:17.00, peak3,852,760KiB. Delta SHA5682cfc35f535d53455d450ec475019d6eed5467395ecc83b0d0fc9c36ec13dd,4,729,584 edits,max|w|1e6. H1 exact and digits24,994/25,000(99.976percent), the strongest arithmetic result so far. Text agreement97.87673950195312percent, KL0.009324374212743604nats, CE increase+0.0054107677760519834nats, finite. H1 joint gate FAILS because agreement is below98.5percent. A191 remains the strongest text result among arithmetic-qualified checkpoints; A202 remains highest text overall with insufficient arithmetic.

Noise exact/digits out2000: sigma1e-5:2000/2000;1e-4:1788/1788;1e-3:907/907. Two-in-a-row:first2000, second given correct first3, both3. The new bit quantization repairs the unperturbed A202 arithmetic weakness but does not yield uniformly stronger noise behavior; changed edit masks also change the deterministic noise realization. The higher KL than A202 may reflect native function displacement and/or rare programme activation outside recorded calibration inputs; no held-out examples were inspected, and this cause is not yet established. Six arithmetic failures are retained in the official score.

Remaining0forward/0quick/27full;13full completed. No rung passed, no best banked. Start the predeclared A232 copied-native-Gate text-basis analysis as the sole heavy job. A235, A236 and A237 remain queued and unrun. A232 compiler is prepared and requires every word/format guard.


### A232 rejected; A238 wider gate diagnostic prepared (2026-11-11 02:34:29 CET)

A232 completed1:34.56,peak6,003,552KiB, with zero factor-interval inconsistencies and42 distinct copied source Gates. No candidate passed, no PT/checkpoint or full-vocabulary confirmation. Baseline8003/8192 restricted matches,state0.01578285731,direction0.01456317699. Best matches:8features,Downridge0.0001,strength0.25 gives7594, state0.08856224500,direction0.07705953711, factor/Up uncertainty0.00495068467, maximum raw-RMS change2.60656. Best state:4features,ridge0.01,strength0.25 gives7581matches,state0.08682344174,direction0.07590896439,raw-RMS change2.62192. The failure is much larger than the inferred-factor uncertainty. Refitting paired-state corrections across ordinary rows at fixed normalization may be driving excessive raw-energy changes; this is a hypothesis, not a proven diagnosis.

A238 analyzer syntax checked, unrun. It cross-checks A234's ten programme Gates on the512 older A119 calibration windows, which were not in its reader-field word covariance. Assert exact matching upstream0–2 and Q/K/O parameters between the recorded A117 parent and A234. Observed head contexts and their recorded source values give a unique64-rank linear inverse; require head reconstruction<1e-5, future/negative/sum coefficient errors<0.002. These are recovered factual mixtures, not newly evaluated attention probabilities. HOST V3 is applied linearly, and the old captured pre-MLP state minus its observed attention branch supplies the earlier residual. Check its reconstructed input norm within1e-5.

Inspect only the hand-built Gate/Up rows and their ReLU saturation ideal; no native nonlinear layer or final prediction is executed. Report positive-gate positions, maxima and raw programme norm. Save only the derived pre-MLP inputs(about84MB) for possible later independent source-gate work, with clear floating-point reconstruction uncertainty and full provenance. No weights, fit scales, answer targets or official score. This diagnostic moves ahead of A235 because A234's higher KL needs investigation; A235,A236,A237 remain prepared/unrun.


### A238 wider calibration gate audit is clear; A235 starts (2026-11-11 02:35:42 CET)

A238 completed8.44seconds,peak1,408,432KiB. All32,768 older calibration positions have all A234 programme Gates below-16; maximum-31.97347641. The six new quantizer Gates have maxima approximately-466k,-475k and-741k. No positive programme Gate and zero hand-ReLU programme raw output were found. Known-head inverse relative error1.83107e-7; recovered input-norm error<=2.23673e-7; all512 source ranks64; maximum future coefficient1.98992e-6. Derived84MB pre-MLP input artifact retained with provenance.

This larger calibration check does not support direct activation of the new reader gates as an observed failure cause. It cannot rule out held-out activation, downstream gates, or native trajectory distortion. Native allocation was fitted at explicitly different recorded input surrogates; its local errors need not predict the completed held-out result. No held-out text was inspected. Start A235's already-declared output repair next as the sole heavy process.

A239 analyzer/compiler are prepared and syntax checked, unrun. They preserve A232's original copied-Gate/Up construction, quiet slots,5feature counts[4,8,16,32,64],Up ridge0.01,Down ridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25] and every word/format/uncertainty guard. One structural change: ordinary Down rows reconstruct only their ORIGINAL editable raw-column contribution; they receive zero paired-HOST state correction. Paired correction is limited to the existing96 strong-output rows, whose1e6 gains need tiny raw coefficients. This tests whether A232's inverse-small-gain ordinary correction caused its raw-energy growth. All targets stay original; no candidate residual is refitted. No new scale values or official budget use. Queue A236,A237,A239 after A235.


### A235 passes local and historical source checks (2026-11-11 02:39:45 CET)

A235 completed2:43.69,peak2,541,832KiB; selected256output rows,1024native input coordinates,ridge1,strength1.25. Native attention-branch RMS0.19072650025→0.17868557233(6.31percent reduction); paired state0.00486002586→0.00477829113,direction0.00287452430→0.00285221618. Would use4,991,726edited scalars; output weightsmax0.435546875, programme/normmaxremain1e6. Rounded protected-functional null error2.50802e-9. Current register interval error0.16651905715; word Gate max-32.4581, current gate minabs63.8779, no changed gate signs; attention raw-RMS change0.065606percent.

All three512-window historical source checks pass: register bounds0.23028269284,0.19564184279,0.35888035327; minimum gates66.0916,57.0435,30.0474; no changed signs; raw-RMS change<=0.083591percent. Saved analysis weights/report, but no checkpoint build or official score yet. Component improvements are not a prediction of the final text score. Finish queued local comparisons before selecting the next full-score candidate. A236 zero-gain radial oracle starts next, then A237 causal-metric queries and A239 strong-only copied features.


### A236 oracle rejects additional zero-gain carriers; A237 running (2026-11-11 02:41:20 CET)

A236 completed11.93seconds,peak3,914,536KiB. Original baseline8003/8192 restricted matches. Ideal radial corrections with1/2/4/8/16 newly zero-gain ordinary coordinates give7982/7962/7927/7832/7678, respectively. Paired state/direction improve, but removing even the quietest eligible ordinary output loses more winner agreements than the ideal radial change restores. Their original output gains are about10.9–23.6, so they are not naturally unused channels. No promising oracle, realizable fit, weights, or meter result. This only rejects the specified least-squares radial construction, not every possible normalization repair.

A237 causal-score-metric query analysis is now the sole heavy process(session49693). A239 follows. A235 retains a locally eligible analysis matrix, unbuilt/unscored. A234 remains the newest completed full score; no rung passed,27full remain.


### A237 passes its causal-score screen (2026-11-11 02:42:19 CET)

A237 completed30.25seconds,peak1,960,672KiB. Mean validation centered causal-score MSE0.00098999260479→0.00083321328561, a15.8364percent reduction. All four heads selectridge1; input dimensions[64,8,8,128], strengths[1.25,1.25,1.25,1]. Source routing minima[8.00017565,7.99999398,8.00007703,8.00009893]; protected-query rounding changes<=7.30709e-8. Maximum query weight0.55898457766. A234-based changed-scalar count remains4,729,584. Head0's99th-percentile score RMS increases slightly0.0425066→0.0433824; the other heads improve. No uniform tail improvement is claimed.

Analysis weights are saved; compiler has not run and there is no official final-text or arithmetic result. A239 strong-only copied-feature analysis is now the sole heavy process(session64398). An A191-parent transfer of the same A237 query repair is a possible next score: its upstream0–2 and Q/K input geometry appear identical, but exact tensor checks must establish that before any build. Such a transfer would retain the arithmetic-qualified text leader's existing V/O and later weights; it is not a newly fitted query. No A240 code or declaration yet.


### A239 fixes the raw-energy failure but still rejects; A240 prepared (2026-11-11 02:43:31 CET)

A239 completed1:36.24,peak6,088,360KiB. No selected candidate/PT/checkpoint. Both best state and best matches use4features,ridge1e-6,strength0.25:7987/8192 versus8003, state0.01599366748 versus0.01578285731, direction0.01479038571 versus0.01456317699. Maximum raw-RMS change0.00469393851, factor/Up uncertainty9.06766e-6. Restricting paired correction to strong rows removes A232's multi-fold raw-energy growth, supporting that diagnosis, but the new feature basis still reduces agreement. No full-vocabulary or format confirmation was triggered.

A240 compiler is prepared/syntax checked, unrun. Reuse the exact selected A237 query matrix on A191, with no new fit or scale. Before saving, assert every layer0–2 tensor, layer3 input norm, and layer3 Q/Qnorm/K/Knorm is bit-identical to A237's A234 calibration parent. Thus the source query and text score-metric calculations share the exact input geometry. Retain A191's V/O and later weights. These differ from A234 after the query calculation, so no new final-text claim follows from the proxy.

A191 is selected as the full-score parent because it has the strongest completed text agreement among arithmetic-qualified checkpoints. A234's extra arithmetic margin did not improve text. After a successful atomic build and manifest validation, run one full H1 score of A240. A235's eligible O3 analysis remains retained, unbuilt/unscored; no locally fitted repairs are composed without a new declaration. All upstream/transitive source hashes will be included.


### A240 built; full H1 starting (2026-11-11 02:44:11 CET)

A240 exact input-geometry transfer checks passed and atomic build completed. Delta SHA81ed08011138ddd46bdd3fc286a7d3df8b8fa1a9b98be2206c6047412b3a95b4, 4,978,700changed scalars,max|w|1000000. The source manifest freezes A237's analyzer and every A191/A234 ancestor used for this construction. No new fit or scale; A191 V/O and later weights remain. Full H1 starts next as the sole heavy process, output scores/a240v1_full_H1.json, log artifacts/a240v1_full_H1.log. It will reserve the14th full run, leaving26full units. No partial result is a pass.


### A241 sparse-support causal-metric B query repair prepared (2026-11-11 02:49:27 CET)

A241 analyzer/compiler are syntax checked, statically reviewed, unrun. Apply A237's exact centered causal-score metric to layer2, independently on A191. Exact A123/A150 inputs follow from matching all layer0–1 weights and layer2 input norm. Protect96 current equals query vectors; source positions[7,6,5,4]. Only128 current format windows are claimed, with32held back; no historical layer2 population is assumed to share this input trajectory.

Restrict output corrections to each head's query channels already differing from HOST in more than90percent of their640 input coefficients, and with nonzero query gain. Every other query row remains bit-identical to the parent, checked in both analyzer and compiler. This keeps the construction within A191's37,568 spare edit slots; actual total count remains a hard guard. Native query gains/keys and all other weights remain fixed.

Same original-response canonical input covariance with whitening ridge0.001,16PCA output directions,input dimensions[8,16,32,64,128],score-metric ridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25]. Independent per-head selection from original targets, retaining parent when no source-safe improvement exists; require combined10percent centered-score MSE reduction, every checked routing margin>=7.999, protected-query rounding change<1e-5, and finite/weight/edit caps. No softmax/native nonlinear replay, gradients, or answer targets. No Q2/Q3 repairs are composed here: changing Q2 would alter Q3's word inputs. A241 queues after running full0789(A240,session34535).


### A242 single-threshold source-field catalogue prepared (2026-11-11 02:53:53 CET)

A242 analyzer is complete, syntax checked and reviewed, unrun. Test a smaller six-unit reader possibility using the literal identity d=(d-t*I[d>=t])+t*I[d>=t], with thresholds[2,4,5,6,8]. Replacing one analog threshold field by a hard classification could require just one symmetric Gate pair plus the original four common-Gate rows. This file only checks source-field feasibility; it does not construct OFF gates, allocate native units, or write a checkpoint.

Fit five signed source indicators2*I[d>=threshold]-1 on the units source head alone using the same900 independent three-digit prefixes and separate per-head source-token responses as A221. Other source heads target zero. No paired-operand or answer target enters the solve. Fixed analog/quiet word penalties1e-8/0.01,background mass1,diagonal ridge fraction1e-12,context mass0; no new penalty sweep. Keep A221's analog C/scalar digit fields and quiet C unchanged. Threshold8 must reproduce A221's existing bit3 coefficients to relative tolerance1e-3 as a numerical control.

Evaluate the literal signed-field correction D_units+t/9*(hard_sign(quiet_threshold)*C_analog-analog_threshold) on current128/heldback32 and three historical512 source populations. Require max digit error<0.25current/<0.4historical, zero wrong/invalid digits or threshold classes, and signed quiet-field margin/reference>0.25. This is a component ideal only; actual word-OFF/saturated GLU construction, weight/edit limits, native displacement and official scoring would still be required. Save the small field catalogue with all outcomes, including rejected fields. Queued after A241 once full0789 finishes.


### A240 is the new arithmetic-qualified text leader (2026-11-11 03:16:37 CET)

Full0789-score completed28:17.20,peak3,853,148KiB. SHA81ed08011138ddd46bdd3fc286a7d3df8b8fa1a9b98be2206c6047412b3a95b4,4,978,700edits,max|w|1e6. H1 exact/digits24,960/25,000(99.84percent), unchanged from A191. Text agreement98.0072021484375percent:78more matching predictions than A191, and3more than A202's prior overall text leader. KL0.002522846601465411nats, CEchange-0.0008076487494541595nats, finite. No H1 joint pass: agreement still falls0.49279785percentagepoints short of98.5percent.

Noise exact/digits out2000 are unchanged from A191:1998/1998 at1e-5;1577/1999 at1e-4;261/1591 at1e-3. Two-in-a-row1998first,1second given correct first,1both. Query repair improves official agreement without improving these arithmetic robustness aggregates; KL is slightly higher than A191, so it is not a uniform distributional improvement. No held-out text was inspected. Remaining0forward/0quick/26full;14full completed. No rung or best banked.

A240 edit audit completed. Original64programme-channel versus192native-channel bookkeeping gives588,645arithmetic/accommodation and4,390,055native-only-repair entries. The new native-query correction overlaps both groups, so these are inherited disjoint purpose labels, not causal allocations. The first audit attempt used an incorrect metadata nesting path and failed before producing an audit; the corrected helper and both logs remain. No weight, manifest, meter or ledger was modified by the audit.

Start the predeclared A241 B-reader query analysis next as the sole heavy process, followed by A242's single-threshold field check. A235's original A234-based O3 local repair remains unbuilt/unscored. A237 and A240 construction sources remain immutable after full0789.


A241 pre-fit assertion and correction (2026-11-11 03:18:17 CET): first attempt exited4.50seconds before any fit or candidate because the relative1e-10 SVD-rank truncation did not meet the absolute1e-8 protected-query null check. The unmetered analyzer now retains all96 observed query-row directions, including nearly dependent small-singular directions, rather than dropping them. This strengthens the original96-query protection; no trial values, objective, or acceptance guard changed. The old failure log is retained; the rerun logs to artifacts/a241v1_causal_metric_b_query_repair_full_row_space.log. No checkpoint or official budget was used.


### A241 passes; A242 rejects; A243 composition declared (2026-11-11 03:29:44 CET)

A241 completed26.81seconds,peak1,747,492KiB after the documented pre-fit null-space correction. Baseline centered causal-score MSE0.00155025550021 becomes0.00105268091026,32.0963percent reduction. Head input dimensions[8,8,16,8],ridges[1,0.01,1e-6,0.01],strengths[1.25,1,1.25,1.25]; minimum source margins[19.66137,8.16135,16.74526,36.38620], heldback minima[21.83223,8.48871,17.97657,37.37099]. Rounded protected-query change<=1.03929e-9, max query weight0.90913093090. Total changed scalars remains4,978,700. Saved local analysis weights, no standalone A241 checkpoint or official score.

A242 completed16.82seconds,peak1,533,520KiB. None of the five single thresholds passes the source-field margin. Current maximum digit errors for thresholds[2,4,5,6,8] are[0.29425335,0.28564075,0.34535701,0.45753823,0.53202371]; required<0.25. Thresholds2 and4 have no wrong source digits on any checked population but do not meet the margin. Threshold8 exactly reproduces the previous bit3 field: relative weight differences1.43091e-12 analog and0 quiet. No installed programme, checkpoint, or score.

A243 compiler is prepared, unrun. Install the exact preselected A241 layer2 query matrix onto A240, with no new fit, scale, or candidate-residual target. Before building, assert all layers0–1 plus layer2 input norm/Q/Qnorm/K/Knorm are bit-identical to A241's A191 calibration parent and that all excluded query rows are unchanged. Thus layer2's local input geometry is exact. Layer2 changes alter later word inputs, so retaining A240's layer3 repair has no exact joint guarantee; the full score must test that composition. Keep every other A240 weight. After atomic build with caps/hash manifest, run one full H1 score as the sole heavy process. This reserves the15th full unit, leaving25.


### A243 full running; A244 two-threshold catalogue prepared (2026-11-11 03:32:27 CET)

A243 atomic build completed24.54seconds,peak902,072KiB. Delta SHA333c87037df3a6b58df017f1745285eb448165b0e6d9c46fa246cb89bbe23eb9,4,978,700changed scalars,max1e6. All exact layer2 geometry and untouched-row checks pass. Full H1 is running(session56113), scores/a243v1_full_H1.json, artifacts/a243v1_full_H1.log. The15th full run is reserved;25 remain. No new result or pass yet.

A244 is syntax checked and reviewed, unrun. Its structural change from A242 is two monotone cuts, potentially eight GLU units. Declare exactly five pairs[(2,5),(3,6),(3,7),(4,7),(4,8)]; first-cut values are[2,3,4],second-cut values[5,6,7,8]. Literal jumps are[t1,t2-t1]. Fit each required source indicator from independent source tokens using unchanged analog/quiet penalties1e-8/0.01,background1,diagonalridgefraction1e-12; retain the unchanged analog digit/reference fields. Recombine the literal remainder plus both hard signed threshold corrections. All current<0.25/historical<0.4,correct-source-digit/class and quiet margin>0.25 guards remain. Threshold8 reproduction is still a control. No new word covariance/penalty search, programme gates, native allocation or checkpoint. Run after A243 as the sole heavy process.


### A245 per-token readout-metric final repair prepared (2026-11-11 03:34:25 CET)

Analyzer/compiler are syntax checked and statically reviewed, unrun. ParentA142, exact512 recorded word windows(384train/128validation), exact512format windows at positions8–12; check all upstream weights against A123/A124, with only final Down/postgain different. Only existing96strong-output rows and2037native columns can change; every programme column, Gate,Up,norm and ordinary Down row remains fixed.

Unlike A142's global average metric or A217's1024critical-observation kernel, use the separate centered frozen-unembedding Gram for every training token's original HOST/A123 top8 union. Duplicate ids retain fixed multiplicity. Original HOST gap weight1/(gap+0.1), normalized by training mean and capped100. Target is the original paired HOST normalized final state minus original A142 normalized state, at fixed original parent branch/final RMS. Canonical original-response input bases use fixed whiteningridge0.001; original pool-response PCA supplies16output directions. Input dimensions[8,16,32,64,128],metricridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25]. All targets,bases,token sets and normalizers are fixed before independent solves; no candidate residual, native nonlinearity, model execution, answer target, or gradient.

Select greatest restricted validation agreement among candidates gaining>=16/8192 matches, improving paired state/direction/centered score MSE, raw-RMS change<0.001 and finite/edit/weight caps. Require full-vocabulary confirmation>=16, then preserve all2560 recorded register winners with target margin>1 and raw-RMS change<0.02. Float32 weights are checked with double recorded-feature linear/RMS algebra; no complete rounding certificate claimed. Maximum normal matrix2048square, per-token metrics built in batches. Run after A243/A244 as the sole heavy process. No final checkpoint or score will be claimed from local success.


### A246 conditional eight-unit programme prepared (2026-11-11 03:37:25 CET)

A246 is syntax checked/reviewed, unrun, and only proceeds if A244 has eligible pairs. Retain A202's four-unit common-Gate interface. For each eligible pair, subtract its literal analog threshold-and-reference terms from the units-place remainder and add two symmetric hinge pairs, giving eight units. The shared OFF field is recomputed using A224's already selected fixed negative-format mass0.1,relative-positive mass1,negative metricridgefraction1e-6; no parameter sweep. Use the same192actual/192surrogate training word windows and96current plus2x512historical source rows.

OFF scaling is1.001times the minimum training-negative bound with quarter-positive-reference guard plus1e-6; center gain enforces source64/negative32; symmetric common-Gate shift is the minimum of0.25 and quarter source/negative ratios. Down and Gate/Up balancing are analytic to the1e6cap. Fixed Up gain1e6. Check float32 programme rows against all128current,32heldback and3x512historical source states, requiring<0.25/<0.4 errors,correct digits and signed Gates>16. Check all actual/surrogate word positions and non-equals Gates<-16, plus the independent512old-word A238 inputs as an extra validation population. No new scale trials, allocation, weights installed, or official claim. Queue after A244/A245 if eligible source fields exist.


### A247 low-rank format query protection prepared (2026-11-11 03:41:50 CET)

A247 analyzer/compiler are syntax checked and reviewed, unrun. ParentA191; same exact layer3 input geometry and original HOST targets as A237. Replace the96-row exact-format-null restriction with the leading[1,4,8,16,32] right singular directions of those96normalized equals-query inputs. This is the declared structural change. Fixed canonical input dimensions by head[64,8,8,128] reuse the original A237 choices;16PCA outputs,whiteningridge0.001,scoremetricridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25]. Every fit starts from the original A191/HOST difference, never a prior candidate residual.

Each rounded candidate must have every128current plus1536historical source routing margin>=7.999, maximum format raw-query RMS movement<1percent of the original raw RMS, finite/capped weights, and improved validation score MSE. The retained singular directions are exactly nulled before rounding; actual96-query movement is reported but is no longer required to vanish. Choose each head's best source-safe original-target solve across the five ranks or retain parent. Require combined>=10percent improvement over original A191 and >=5percent additional reduction versus A237's already-selected MSE; verify identical baseline MSE. All targets/bases are fixed original observations, native keys/gains unchanged, no softmax or native model replay.

This may expose more native text directions while spending less of the query map on exact protection of small format variations. Finite source checks do not establish general arithmetic preservation; no checkpoint or score exists. Queue after the already prepared A244/A245 and conditional A246; A243 full score remains the sole heavy process.


### A248 sparse B-query low-rank protection prepared (2026-11-11 03:44:25 CET)

A248 is syntax checked and reviewed, unrun. Apply the same declared structural relaxation to A241's B2 query map, independently from original A191. Protect leading[1,4,8,16,32] singular format directions. Fixed input dimensions[8,8,16,8] reuse A241's selected dimensions;16outputPCA directions,whitening0.001,metricridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25]. Only query channels already changed in>90percent of640coefficients and having nonzero gain are editable; all excluded rows are asserted identical in analyzer/compiler.

Use only128current exact-input format windows(96PCA/32holdback); no historical B2 claim. Require source margins>=7.999,max raw-query RMS movement<1percent,combined MSE improvement>=10percent versus original and>=5percent versus A241, finite/capped weights and edit limits. Baseline MSE must reproduce A241. Every fit targets the original HOST/A191 difference; no residual refitting. No new Q2/Q3 composition is built here. Queue after A247; A243 remains the sole running full score.


### A249 fixed-input hand-programme noise moments prepared (2026-11-11 03:47:23 CET)

A249 is syntax checked and reviewed, unrun. It diagnoses A142's exact retained format/text inputs at layers1–8 and17; root0 is explicitly omitted. Verify every upstream weight against captured A123/A124 except final Down/postgain. Identify hand-programme Gate/Up rows by actual HOST differences. For each fixed input, multiplicative independent Gaussian weight noise gives exact linear preactivation variance sigma^2*sum_changed(w_i*x_i)^2. Use official sigma values[1e-5,1e-4,1e-3], no parameter search.

Report marginal gate sign/saturation probabilities, expected counts by linearity, and unit-wise Gate/Up noise conditioning. The independent positive-hand-GLU product variance is sigma^2*(U^2*vG+G^2*vU)+sigma^4*vG*vU; this is the hand saturation ideal, with no GELU evaluation or differentiation. No independence across cases is assumed; they share the same weights. Exclude upstream,input,Down,norm,perturbed-native-activation and floating-point effects. This is not a full-noise arithmetic or prediction estimate. No fit, checkpoint or official budget. Queue after A248 as the sole heavy process.


### A243 full score fails to improve text; A240 remains leader (2026-11-11 03:58:49 CET)

Full0791-score completed27:51.85,peak3,852,176KiB. Delta333c87037df3a6b58df017f1745285eb448165b0e6d9c46fa246cb89bbe23eb9,4,978,700edits,max1e6. H1 exact/digits24,965/25,000(99.86percent),five more than A240. Text98.004150390625percent,four fewer matching predictions than A240; KL0.002532113648981344,CEchange-0.0006020335674934429,finite,H1jointpassfalse. A241's32.1percent local B-query score-error reduction does not translate into improved official text agreement when composed with A240. This rejects a final-text gain claim for this composition, not the usefulness of every B-query design.

Noise exact/digits out2000:1998/1998 at1e-5,1582/1999 at1e-4,262/1608 at1e-3. Two-in-a-row1998first,1second given correct first,1both. Full score is scores/a243v1_full_H1.json. No held-out text inspected.15full completed,25remain;0forward/0quick. A240 remains the overall and arithmetic-qualified text leader,98.0072021484375percent. No rung/best/verification/export. A241/A243 construction sources are now immutable.

Start A244 as sole heavy process, then A245, conditional A246 if A244 has source-eligible pairs, A247,A248,A249. All are already declared and syntax checked; their local gains remain unobserved. A235's earlier eligible A234-based O3 repair remains retained and unbuilt/unscored.


### A244 two-threshold source fields pass for two pairs (2026-11-11 04:00:17 CET)

A244 completed11.69seconds,peak1,538,680KiB. Pairs(2,5) and(3,6) pass every field screen with zero wrong/invalid source digits/classes. Current max digit errors0.21611349571/0.22156956376; heldback32 errors0.21611349571/0.17007608866. Worst historical errors0.36584323189/0.36879430330. Minimum quiet classification margins/reference across all groups0.6657354/0.7406911. Other pairs(3,7),(4,7),(4,8) fail the fixed margin despite no wrong rounded digits. These are independent source-field ideals, not installed arithmetic results.

A246's predeclared conditional gate stage is now applicable to both eligible pairs; no allocation/checkpoint exists. Start A245's prepared final-layer text repair as the sole heavy process, then A246,A247,A248,A249. No official score running;25full remain.


### A245 final readout metric rejects (2026-11-11 04:02:26 CET)

A245 completed1:16.26,peak4,050,548KiB. No selected weights, PT, checkpoint or full score. Baseline8003/8192 restricted matches,state0.01578285731212,direction0.01456317699050,centered readout MSE0.00817161190336. Best agreement only ties8003; all candidates worsen paired state/direction. Smallest MSE0.00816772179862(input128,ridge100,strength1.25) still yields8001matches and worse state0.01579546400222. Strong regularization suppresses damage but gives no useful repair. No full-vocabulary or format follow-up is triggered. Start A246's prepared two-threshold hand programme, then A247/A248/A249.


### A246 source-correct but text-active; A251 OFF-margin variant prepared (2026-11-11 04:04:51 CET)

A246 completed19.94seconds,peak2,458,900KiB. Both eight-unit programmes meet their source-digit and saturation checks but fail the word-OFF guard; no selected PT/checkpoint. Pair(2,5):current error0.21614006,historical0.36583950,maximum word/non-equals Gate+179741.65625,also the older512word maximum. Pair(3,6):current0.22139773,historical0.36879265,maximum checked negative Gate+76539.46875,older512word+23056.09375. This is a directly observed ordinary-text activation defect, unlike A234's wider gate audit which was clear.

A247 low-rank A-query analysis is now the sole heavy process(session66136); A248/A249 follow. A251 is prepared/syntax checked, unrun: retain A246's exact fixed OFF covariance and source fields, but replace its minimum training-negative amplitude by an analytic midpoint. Lower bound is the original negative-field requirement; upper bound preserves at least half EVERY original signed training source-field margin. Reject an empty interval. This changes OFF margin allocation, adds no trial scale values and never fits a previous candidate residual. All source/error/saturation/word-OFF guards remain, including all512 older word windows held out from the fit. Queue after A249.

A250 is reserved for a possible attention-mass-weighted score-metric query repair using uniquely recovered factual mixtures; no A250 code, trial declaration, or run yet. It would not evaluate new probabilities or use gradients.


### A247 reduced format rank gives only a small extra gain (2026-11-11 04:05:19 CET)

A247 completed0:44.05,peak2,005,088KiB. Combined causal-score MSE0.00081963038402,17.2084percent below original but only1.63018percent below A237, failing the declared additional5percent guard. No PT/checkpoint. Selected per-head protected ranks[8,1,4,8],ridges all1,strengths[1.25,1.25,1.25,1]. All checked source gaps>=8.0000671,max format raw-query RMS movement<=0.00131296; would use4,978,701edits,max query0.5596519113. Exact96-row protection was not the dominant limitation on this calibration objective. Start A248 B-query counterpart next, then A249 and A251. A250 weighted-metric idea remains reserved, not yet coded/declared.


### A248 reduced B-query protection also gives a small extra gain (2026-11-11 04:06:50 CET)

A248 completed29.56seconds,peak1,642,584KiB. MSE0.00102613380876 versus original0.00155025550021(33.8087percent reduction),but only2.52186percent better than A241,below the extra5percent guard. No PT/checkpoint. Every head chooses protected rank1, fixed inputs[8,8,16,8],ridges[1,0.01,1e-6,0.01],strengths[1.25,1,1.25,1.25]. Source margins>=8.16126813,max relative query movement0.00374969;4,978,700edits,maxQ0.90978980064. Exact96-query protection again is not the main local bottleneck. Start A249's fixed-input noise diagnostic, then A251's already-declared stronger OFF margin. A250 weighted factual-mass analyzer has now been drafted and syntax checked, but still needs static review and its complete trial declaration before running.


### A250 factual-attention-mass query repair declared (2026-11-11 04:07:37 CET)

A250 analyzer/compiler are complete, syntax checked and reviewed, unrun. ParentA191, exact A3 geometry and original HOST query targets as A237. Uniquely recover factual attention mixture coefficients from A191's already-derived input mixtures by a64-rank linear inverse of each recorded word-input matrix. Require reconstruction<1e-5,negative/future/row-sum/L1-clipping-adjustment errors<0.002; clamp only tiny negative/future artifacts and normalize original row sums. These are observed mixture coefficients, not newly evaluated attention probabilities; no softmax, gradients, native activation or new model run.

Retain96exact format-query constraints and fixed original canonical input dimensions[64,8,8,128],16PCA outputs,whitening0.001. Fit convex combinations of TWO separately centered losses: uniform causal-key score error and factual-parent-attention-mass score error. Mixture fractions[0,0.25,0.5,0.75,1],metricridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25]. Every fit targets original HOST minus original A191 normalized queries at fixed original raw RMS. Uniform-fraction0 at A237's fixed selected ridge/strength must reproduce its query matrix to relative<1e-5; original uniform baseline MSE must match.

Select each head's least validation factual-mass MSE while not worsening ORIGINAL uniform MSE, preserving all checked source gaps>=7.999,rounded96-query movement<1e-5 and weight/edit caps. Require combined factual-mass MSE>=10percent better than original and>=5percent better than A237 evaluated under this SAME fixed metric. This changes which observed score differences matter; it makes no complete attention-output or text-prediction claim. Largest solve2048square, two Grams per head. Run after A249/A251 as sole heavy process.


### A249 noise diagnostic and A251 OFF-margin rejection (2026-11-11 04:09:53 CET)

A249 completed9.51seconds,peak2,672,436KiB. Exact fixed-input Gaussian Gate/Up moments show negligible sign-flip probabilities for early layers1–5 and8 at sigma1e-3. Expected gate-entry flips across each format population at1e-3 are layer6:0.28553,layer7:0.60606,layer17:2.14126; these populations differ in size and are not independent cases. Layer17 units458/929 account for1.64496/0.49630 expected flips; at1e-4 all17units total0.70002. Some baseline Gates are already within16:24layer7entries and1layer17entry. Word gates are all negative and positive-tail probabilities underflow numerically to zero under this fixed-input audit. This does not imply literal zero Gaussian probability.

Relative positive Gate*Up noise can be large near tiny products, especially B2unit80; relative values alone do not establish a register or arithmetic failure. No Down/norm/upstream input noise was included, so these component results cannot explain or replace official noise accuracy. No weights/checkpoint.

A251 completed18.85seconds,peak2,458,700KiB. OFF amplitudes0.9968954 for(2,5) and0.9759049 for(3,6) keep every tested word/non-equals Gate below-31.97346,fixing A246's observed text activation. Current source errors remain0.21633536/0.22139943, but historical errors grow to2.31577557/1.85050368 due to heldback source signs. Both reject; no PT/checkpoint. The source-margin bound was derived only from the declared training groups and did not generalize to historical_c.

Start the fully predeclared A250 factual-mass query repair as sole heavy process. A possible bounded OFF-amplitude family is next: include the already-tested minimum and midpoint as controls and allow at most three intermediate fractional values, for five total across this amplitude family. No A252 script or exact fraction list yet.


### A250 weighted query passes locally; A252 bounded OFF family declared (2026-11-11 04:11:35 CET)

A250 completed38.71seconds,peak5,106,572KiB. Factual-mass centered MSE0.00042337022287→0.00035858200684(15.3030percent),7.35945percent better than A237's0.00038706809039 under the SAME fixed metric. Uniform causal MSE0.00087828202663 is below original0.00098999260479,but worse than A237's0.00083321328561; this is a metric tradeoff, not a uniform improvement. All four heads select factual mass fraction1;fixed input dimensions[64,8,8,128],ridges[1,1,1,0.01],strengths[1.25,1.25,1.25,0.75]. Minimum source gap7.99999416,96-query rounding changes<=7.32076e-8,maxQ0.5587375164,4,978,700edits if built.

Mixture inversion has rank64 for every word window,relative reconstruction1.46812e-8,max negative/future coefficient4.83241e-6,max row-sum error7.39964e-7,max L1 causal nonnegative renormalization adjustment1.97041e-5. Uniform-metric control reproduces each A237 selected query matrix BIT-EXACTLY. Saved local query weights/report; no checkpoint or official score yet. After the next short source check, build A250 exactly as declared on A191 and run one full H1. This will be the16th full run,leaving24. No new Q2 repair is composed because A243 did not improve official text agreement.

A252 is syntax checked and reviewed, unrun. Use exactly five fractions[0,0.125,0.25,0.375,0.5] between A251's analytically defined OFF lower/upper bounds. Fraction0 reproduces A246's already-tried minimum;fraction0.5 reproduces A251's midpoint. Thus only three new fractions enter this bounded amplitude family, five total including prior endpoints. No more fractional values are authorized by this declaration. Refit no residual or OFF covariance; retain all fixed fields/Gate/Up/Down rules and every source/word guard. Check endpoint diagnostics against existing reports. Save only designs passing all guards; native allocation/full score still required. Run now as the sole heavy process before A250 build/full.


A252 bookkeeping failure and correction(2026-11-11 04:14:35 CET): first run exited8.50seconds,peak2,458,480KiB after reproducing the fraction0 control. Its source training-sign variable was overwritten by the eight-column diagnostic sign matrix before the next fraction. Rename the former source_signs; no scales, targets, guards or fit rules change. Retain original failure log; rerun writes artifacts/a252v1_bounded_OFF_threshold_programme_fixed_signs.log. No checkpoint or official meter used.


### A252 both eight-unit programmes pass at fraction0.125 (2026-11-11 04:15:47 CET)

Corrected A252 completed0:25.98,peak2,453,856KiB. Both threshold pairs pass every source/error/sign/saturation/word-OFF guard at OFF interval fraction0.125; no other fraction passes. Fraction0 repeats text activation;fractions0.25/0.375 lose historical saturation;fraction0.5 loses historical source signs. Endpoint controls reproduce A246/A251 diagnostics within the declared tolerance. This completes the five-value OFF fraction family; no additional fraction values will be tried.

thresholds_2_5_f0p125:OFF scale0.42917819847,current error0.2161648339,historical0.3658395046,minimum signed Gate29.564086914,all-negative max-31.973461151,max Gate weight115926.65625,Up21280.441406.

thresholds_3_6_f0p125:OFF scale0.49558771863,current error0.22140027259,historical0.36879265357,minimum signed Gate21.853759766,all-negative max-31.973461151,max Gate weight144438.82812,Up21280.441406.

These are programme rows only; no native allocation or official arithmetic/text result. The next allocator will reuse the declared conditional native-feature scheme, with no assumption that eight units automatically preserve text. Build A250 now, then run its already-authorized full H1 score. No other heavy process runs.


### A250 built; full H1 starting (2026-11-11 04:17:05 CET)

A250 atomic build completed20.47seconds,peak777,672KiB. Delta SHA3df66322a618ca62d1995dec0a13886bf2b899a70bdeb2519d49e72dce42fb40,4,978,700changed scalars,max|w|1e6. Only A191 Q3 changes to the exact preselected factual-mass query repair; no new fit/scale/Q2 composition. Full H1 starts next as sole heavy process, output scores/a250v1_full_H1.json, log artifacts/a250v1_full_H1.log. It reserves the16th full run,leaving24. Sources are immutable from this metered build onward.

A253 will allocate A252's thresholds_2_5_f0p125 eight-unit programme. That design dominates the other locally eligible pair in current/historical digit-error margin,minimum signed Gate and maximum Gate weight. Reuse the existing immutable A233 conditional-removal allocator, with its same original-feature/gain targets, raw importance masses[0,0.01,0.1,1,10],donor counts[256,384,512,768,896],repairridges[1e-6,1e-4,0.01,1,100],conditionalrankingridge0.001. Preserve native-branch<1.5x baseline,paired state/direction<1.1x,current register bound<0.25,saturated Gates and edit/weight limits. A174/A176 features remain explicitly a local native-input surrogate, not exact A202 trajectory. No allocator run or compiler exists yet; prepare while A250 scores and run afterward.


### A253 threshold-reader compiler prepared (2026-11-11 04:18:38 CET)

experiments/compile_a253_two_threshold_reader_v1.py is syntax checked and reviewed, unrun. It accepts only the guarded thresholds_2_5_f0p125 allocation, verifies source/input hashes, strict programme rows/native Gate/Up preservation, unchanged arithmetic gain coordinates, HOST V3, finite/edit/weight caps via atomic save, and transitive catalogue/control/older-word audit provenance. There is no exploratory guard override. Source expectation: artifacts/a253v1_thresholds_2_5_native_allocation.pt; output experiments/a253v1_two_threshold_reader/delta.pt.

After A250 full finishes, run the unchanged allocator:
`.venv/bin/python experiments/derive_a233_selective_binary_allocation_v1.py --gate-artifact artifacts/a252v1_bounded_OFF_threshold_programme.pt --design thresholds_2_5_f0p125 --output-prefix artifacts/a253v1_thresholds_2_5_native_allocation`

Only if selected, run the new compiler. The eight-unit source programme has passed component checks; native allocation is still unrun and no full checkpoint exists. A250 full score(session37010) remains the sole heavy process.


### A254 fixed factual-value-dispersion metric prepared (2026-11-11 04:20:47 CET)

A254 analyzer/compiler are syntax checked and reviewed, unrun. Parent remains ORIGINAL A191; no candidate residual is used. Recover the same factual attention coefficients as A250. For each original head, apply its current V/O and post-attention gain linearly to each source input and its factual mixture; form the squared distance of each source vector from that observed head context. Define a fixed statistical source weight proportional to factual attention mass times that squared distance, normalized per causal row; zero-dispersion rows retain factual mass. This is a value-spread heuristic, not a computed gradient or a new attention probability. No softmax/native activation/model replay.

Fit mixtures of TWO separately centered losses: original factual-mass score error and value-dispersion score error, with fractions[0,0.25,0.5,0.75,1]. Keep96exact query constraints, input dimensions[64,8,8,128],16PCA outputs,whitening0.001,metricridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25]. Fraction0 at A250's fixed selected ridge/strength must reproduce its matrix to relative<1e-5; original uniform baseline must match. All source and original-data targets remain fixed before independent solves.

Require source gaps>=7.999,96-query rounding movement<1e-5,caps, and neither uniform nor factual-mass validation MSE worse than ORIGINAL A191. Select each head's smallest dispersion MSE; combined must improve>=10percent over original and>=5percent over A250 under this SAME fixed dispersion metric. No final-text or complete attention-output claim. Run after A250 full and A253 allocation, as the sole heavy process. No A254 checkpoint or run yet.


### A255 existing-anchor radial/readout feasibility prepared (2026-11-11 04:31:05 CET)

A255 analyzer is syntax checked and reviewed, unrun. Parent A142 has exact512word windows, with128validationwindows/8192tokens. Unlike A236, it keeps every native output coordinate and every gain, using only the existing zero-output programme anchor439. No weight fit or checkpoint will be produced. Keep the original parent branch numerator fixed, and bound its hypothetical radial factor to[0.5,2]times parent plus the exact tighter cap from unchanged raw-coordinate energy, including RMS epsilon.

Compute the analytic paired-HOST L2 radial optimum. Separately intersect all one-dimensional score inequalities for the original HOST winner against the immutable HOST/A123 top8 union. This gives an optimistic upper bound on recoverable predictions for that fixed numerator, since unrecorded vocabulary rivals are omitted. Report midpoint feasibility and required relative factor movement. Per-token validation HOST states/predictions are used only for this diagnostic; their factors are never installed. No native activation/model execution, gradients, arithmetic answers or iterative fitting. Run after A253 and A254 as the sole heavy process. A250 full H1 is still running.


### A256 sparse copier query metric repair prepared (2026-11-11 04:35:51 CET)

A256 analyzer/compiler are syntax checked and reviewed, unrun. Only the copier's two already-edited Q7 rows per head may change, with Q/K gains, K/V/O, every ordinary Q row, and all other weights fixed. Exact local inputs come from A142/A123's recorded layer6 raw/MLP outputs, reconstructed bit-identically through the recorded attention7 residual. Original HOST Q/K centered causal scores at those exact inputs are the fixed target. A240 has identical Q7/K7/gains but different upstream operand-reader weights; installation on A240 is explicitly a transfer composition, not an exact-trajectory prediction.

Protect all480query vectors at positions8–12of the first96currentformatwindows. Keep all480 SVD row directions even if numerical rank detection is smaller, giving a fixed160-dimensional nullspace. The only two output directions are the existing programme pair, so each head's exact original-normalizer causal-key metric solve has320unknowns. No PCA rank or output-support search. Ridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25]; each independent solve retains original targets and normalizers.192training/64validationwordwindows,32currentformat holdbacks plus512additional exact A124formatwindows. No new attention probabilities, native activation/model replay, gradients, sums or residual refits.

Each head chooses its smallest actual-rounded-query causal score MSE with source margins>=7.999 and protected raw-query change<1e-5, or keeps the original head. Combined candidate must improve MSE>=10percent, retain all source guards and the finite/edit/weight caps. A240 composition requires one full score if built. Queue after A253,A254,A255, with one heavy process at a time. A250 full H1 remains running.


### A257 sparse copier output repair prepared (2026-11-11 04:41:20 CET)

A257 analyzer/compiler are syntax checked and reviewed, unrun. Unlike A81's old shared V7 repair, this changes sparse coordinates of O7, allowing different head corrections. Exact local A142/A123 word/format inputs are reconstructed from recorded branch outputs. Use the unique source-mixture inverse to recover factual7heads; rerun the independent eight-window known-layer3-head inverse control. Reject before fitting if word/format inverse ranks, raw reconstruction or future coefficients fail existing bounds. No new attention probabilities are evaluated or saved.

Original paired HOST normalized attention7 branch is the target at fixed parent raw RMS.192training/64validationwordwindows. Rank128output rows by original branch-error energy and input coordinates by conditional explanatory energy at fixedridge0.001. Protect all16actual copier Gate/Up linear functionals after post-attention gain with the inherited weighted null projection(rawpenalty0.01). Native input counts[16,32,64,128,256],ridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25]. No candidate residual becomes a target.

Require>=5percent branch RMS reduction plus improved paired state/direction, all256word Gate maxima<-16, all128format Gate signs unchanged, format Gate/Up changes below0.001of each original unit's maximum magnitude, raw-RMSchange<0.002, relative full-frame nullerror<1e-6, and caps. Programme/native GELU is not replayed. Intended installation is A240 with identical O/V/gains/G/Up but changed earlier readers; local A142 inputs are explicitly a transfer surrogate for this composition. At most32768newedits fit its37568spare entries. Full scoring remains necessary. Queue after A253,A254,A255,A256, one heavy process at a time; A250 is still scoring.


### A250 full H1 completed; A240 remains leader (2026-11-11 04:45:58 CET)

Official0793-score(scores/a250v1_full_H1.json) returns24,960/25,000exact+digits(99.84percent),text98.00033569335938percent,KL0.002512832229130644,CEchange-0.0008517154831757523,allfinite,H1false. Text has9fewer matching predictions thanA240 on131072targets. Factual-mass local score improvement did not improve final held-out agreement; A240 remains the leader at98.0072021484375percent. A250 sources/checkpoint remain immutable.

Noise exact/digits outof2000:1e-5:1998/1998;1e-4:1576/1999;1e-3:261/1591. Two-in-a-row1998first,1secondgivenfirst,1both. Runtime28:02.35,peak3,852,524KiB. Completed300forward/100quick/16full;remaining0/0/24. No rung/best/finalverification/export.

A253 eight-unit native allocation is now the sole heavy process(session4094),using the unchanged A233 analyzer and the already-declared A252thresholds_2_5_f0p125 programme. A254,A255,A256,A257 follow sequentially; candidate builds depend on their guards. No new score is running.


### A253 eight-unit allocation passes its declared local screen (2026-11-11 04:47:55 CET)

Unchanged A233 allocator completed1:11.71,peak3,401,244KiB. It selects A252thresholds_2_5_f0p125 with rawimportance mass0,units[2021,320,141,1564,1095,1885,1069,1261],768donors,ridge0.01. Nativebranchproxy0.08749040894→0.12412319829 is WORSE than the original four-unit baseline but below the predeclared1.5xlimit; pairedstate0.00699629526→0.00710025262anddirection0.00555088340→0.00566673220 are below1.1xlimits. This is the declared quantization/text tradeoff, not a local preservation improvement.

Current register bound0.20380581239digits,perplace[0.20380581239,0.16662316482,0.04450473565];minabsoluteprogrammeGate50.082519531,wordGate max-32.437446594. Programme/native raw-norm bound ratio20850.36. Predicted4,888,571edits,max1e6,127697spare. Eligible allocation PT/JSON saved; no model checkpoint yet. Build A253 and run one full H1 after the queued short analyses; no query repair is composed in that initial score. A254 is now sole heavy process(session46787),thenA255,A256,A257 sequentially.


### A254 rejected; A255 anchor assumption corrected in a new version (2026-11-11 04:49:56 CET)

A254 completed40.00seconds,peak5,164,672KiB. Dispersion-centered MSE0.00054456332639→0.00046282049681(15.01percent); A250 under the same metric is0.00046923799309, so the incremental gain is only1.36764percent, below the declared5percent. Factual MSE0.00036123598603,uniform0.00086002576883; no selected PT/checkpoint. Fraction0 reproduces all four selected A250matrices bit-exactly. Reject without changing the screen.

A255v1 stopped on its gain assertion before oracle results:4.89seconds,peak2,968,100KiB. The existing anchor439 is QUIET, not zero-output: stored post-MLP norm parameter-1.1311084032058716 gives gain-0.13110840320587158. Its programme column587 has rawweight100000. Earlier A216 required |gain|<=1 and accounted for the actual gain in candidate evaluation; the zero-gain assumption in the new A255 proposal was incorrect. Originalv1source/log remain.

New A255v2 is syntax checked/reviewed, unrun. Explicitly PROPOSE zeroing this one quiet output gain for the optimistic diagnostic, retaining every other output and unchanged raw energy outside439. Compare the actual original baseline, zero-gain-at-original-RMS baseline, paired-HOST radial oracle, and restricted winner intervals. The loss of the small native output is included; no per-token factors or gains are installed. All original radial limits/rivals remain unchanged. This differs from A236 because its programme-output exclusion prevented testing439. Runv2now, thenA256andA257. No model/score budget is used by this diagnostic.


### A255v2 diagnostic and A256 copier-query screen completed (2026-11-11 04:52:39 CET)

A255v2 completed11.33seconds,peak3,881,328KiB. Original A142 restrictedmatches8003/8192,stateRMS0.01578285731,direction0.01456317699. Setting only anchor439's proposed gain to0 at unchanged raw RMS stillgives8003,state0.01578303264. The feasible paired-HOST L2 radial optimum improvesstate0.01440996224anddirection0.01361350539,butLOSES5matches to7998. Thus radial state-error minimization again doesnotimprove this prediction metric.

The fixed-rival interval diagnostic permits8169/8192winners,including166potentiallyrecoverable baseline misses;23cannotbe repaired within the fixed-numerator radialdomain. Requiredrelativefactor movement on recoverablemisses[min,median,p90,max]=[0.0003512,0.0543581,0.261151,0.542710]. The interval-midpoint oracle realizes8169restrictedmatches but worsensstateRMS to0.0460084anddirection0.0410194. These are validation-informed optimistic per-token factors, NEVER installed; omitted vocabulary rivals and lackofarealizablefeaturemap prevent any accuracy claim. A possible original-HOST winner-interval training target wouldneedaseparate closed-form fit, holdback validation, fullvocabulary checks and officialscore. No suchfit/code/declarationyet.

A256 completed18.30seconds,peak1,669,132KiB. The480protected formatqueries have numericalrank480. OnlyexistingQ7channels[87,215]editedperhead. CombinedcausalMSE0.00112848546478→0.00104846728954(7.09076percent),belowthedeclared10percent; noPT/checkpoint. Bestheadridges[0.01,1e-6,0.01,0.01],strengths[1,0.75,1,0.75]. Allsourceguardspass,min8.3427780013,roundedprotectedquerychange<=2.67138e-8,maxQ0.62890625. A lower-rank protectedformatspan couldtest whether these480constraints limitcorrection,but no newcandidatehasbeen declaredorimplemented.

A257 sparsecopieroutputrepair is now soleheavyprocess(session95439). Whenitfinishes,buildthealreadyeligibleA253readerandrunitsfullH1. A240remainsverifiedleader;0forward/0quick/24fullremain.


### A257v1 inverse guard rejects; causal-affine v2 prepared (2026-11-11 04:55:23 CET)

A257v1 completed17.97seconds,peak3,405,476KiB. Independent known-head controlpasses(relative8.43893e-8). Wordinverseunique256rank/rawerror1.45601e-7/future1.26914e-6 passes. Formatinversehasfullrank56andrawerror4.43933e-7,but maximumfuturecoefficient0.00535077 exceeds the fixed0.002guard. It stopsBEFOREfitting, noPT/checkpoint. The reason string saysfinalattentioninheritedfromits template; actualmoduleis7asrecordedinthesource. Source/log/JSONretained.

A257v2 analyzer/compiler are syntaxcheckedandreviewed,UNRUN. Only the recovery method changes: for format and independent known-head controls, enforce factual causality and each head's coefficients summingto1 by eliminating source0. At querypositiont, solve uniquely for4*t source-difference coefficients; source0=1-sumofothers,futurecoefficients0. Atposition0allcoefficientsareexactly1. Everyprefixmusthavefullcolumnrank; rawerror<1e-5,negative/future/row-sumerrors<0.002andindependentknownheaderror<1e-4. These are linear equalities satisfied by the recorded attention, not a newsoftmaxor changed-attention evaluation. Wordinverse remains the original whole-window unique solve, with addednegative/row-sumguards. Bothwordandformat residualstatesmustreconstructbit-identically beforeinversion.

Alltargets,featureallocation,scales,source/frame/nativeguards,andA240transfercaveat remainexactlyA257v1. No cutoff is relaxed. RunafterA253's fullscore,asthesoleheavyprocess. A253build(session21212)currentlyhashesprovenance; no fullscorehasstartedyet.


### A253 built; full H1 started (2026-11-11 04:57:11 CET)

A253 atomic build completed2:08.10,peak782,592KiB after transitive provenance checks. Deltaexperiments/a253v1_two_threshold_reader/delta.pt SHA657605a1932e85e828fd20ad56c8f08e09d06d386362cfcabcfac2f463cf8caf,4,888,571edits,max1e6. It installs exactly the selected8-unit thresholdprogramme/nativeallocation on A202; noQ2/Q3repairorothercandidatecomposition. CompilerA253andallcatalogue/programme/allocator/transitive construction sources are now immutable from this metered build onward.

FullH1isnowrunning assoleheavyprocess(session47746),logartifacts/a253v1_full_H1.log,outputscores/a253v1_full_H1.json. Sixteenfullrunscompleted; seventeenthreserved,23remain. A240 remains the verified text leader among arithmetic-qualified checkpoints. No rungpassed/bestbank/finalverification/export. Prepared A257v2 causal-affine recovery+outputrepair runs only after this score finishes.


### A258 original HOST winner-interval anchor fit prepared (2026-11-11 05:01:37 CET)

A258 analyzer/compiler are syntax checked and reviewed, UNRUN. Follow A255v2's explicit proposed gain439=0, with every other gain, Gate/Up row, programme Down column and other Down row unchanged. Exact local A142 inputs use512wordwindows split384training/128validation. Original A240 final Down/gains match, but its changed upstream readers make installation a transfer requiring a fullscore.

Only TRAINING text defines targets. At tokens where the original zero-gain parent misses the original HOST winner, intersect affine score inequalities against fixed HOST/A123top8union, requiring fractions[0,0.125,0.25,0.5,1]of the original HOST margin. Normalize those fixed margins using original HOST and parent final RMS. Intersect with physical anchor-energy feasibility and relative inverse-RMS[0.5,2]. Choose the closest feasible factor to1; forzero-margin boundaries take one representable double step into the interval. Correct original predictions and infeasible cases keep the original anchor target. Preserve original raw-anchor sign, derive its required energy analytically, and fit the ORIGINAL target delta in one native-feature ridge solve. No candidate residual or validation HOST state defines a fitted target.

Mistakeweights[1,4,16,64,256],ridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25]. Each weight's covariance and all five target columns are solved independently; no gradients/optimizers/nativeactivations/attention/modelreplay/arithmeticanswers. Rounded row weights are evaluated by exact linear/RMS algebra. Require>=16extra restricted predictions/8192,stateanddirection<=1.05baseline,maxwordraw-RMSchange<0.25,andcaps; then full-vocabulary parent/candidate confirmation of>=16extra, followed by all2560exactA124formatwinners withmargin>1andraw-RMSchange<0.02. Baselineandzero-gaincontrolmustreproduceA255v2diagnostics. Only a passing result is saved for compilation.

Queue after A253full and A257v2, one heavy process at a time. No A258 fit, checkpoint or official result yet. A lower-rank protected copier-query span remains an unimplemented follow-up; no A259 source/declaration yet.


### A259 lower-rank copier format protection prepared (2026-11-11 05:04:18 CET)

A259 analyzer/compiler are syntaxcheckedandreviewed,UNRUN. It keeps A256's ORIGINAL A142 HOST/current QK score targets, exact inputs, twoeditableQ7rows[87,215],fixedgains/keys,192/64word split and all640formatwindows. The only construction change is the protected leading SVD rowspace of the same480normalizedquery vectors. Counts[8,16,32,96,480] include the already-tested A256control; only four new rank values are introduced, five total for this family. Do not extend this list. Full remaining nullspaces are used, at most632inputs/twooutputs=1264normal-equationunknowns perhead.

Ridges[1e-6,1e-4,0.01,1,100]andstrengths[0.25,0.5,0.75,1,1.25]unchanged. Every roundedcandidate mustretain sourcegap>=7.999, protected-PCquerychange<1e-5andrawquerychange<0.01onALL3200current/stressformatquerypositions. Selecteachheadindependently byoriginal validationcausalMSEacross the fiveprotectedranks. Combinedmustimprove>=10percentoriginal AND>=5percentoverA256's0.00104846728954. Rank480 at the prior selectedridge/strengthmust reproduce each A256headMSEto1e-12; this checksmetricreproduction,not a savedmatrixidentitybecauseA256producednoPT.

No priorcandidate residualfit, nativeactivation/modelreplay, softmax, gradients or arithmeticanswerlabels. Q7ordinaryrowsremainbit-identical; A240installation retains the same explicit upstream-transfer caveat. QueueafterA253full,A257v2,A258,oneheavyprocessatatime. NoA259runorcheckpointyet.


### A260 flat-newline emitter prepared (2026-11-11 05:18:04 CET)

A260 analyzer/compiler are syntax checked and reviewed, UNRUN. A simpler literal identity avoids adding native units: replace d by -1+2*ReLU(d+0.5)-ReLU(d), equal to d at every nonnegative digit and at newline encoding -1, constant for d<=-0.5. With original firstthree Down columns V0,V1,V2, set V0'=V0-V1,V1'=-V1 on a new hinge0,V2'=V2+2V1 on the existing hinge-minus-half. This includes the folded physical391 cleanup: when the newline register moves from -1, physical cleanup is correspondingly approximate. All eleven programme units, every native Gate/Up/Down coefficient and every gain stay fixed. Only Gate834, Up for the firstthree programme units, and their three Down columns may change. This supersedes the unimplemented extra-unit idea; no additional native-feature repair is needed.

The new hinge uses the inherited literal flag/digit and OFF functionals. On512recorded A142word windows plus512formatprefills, choose the OFF multiplier analytically to make Gates<-500. Sharpness trials[16,64,256,1024,4096]; require literal hinge-error bound<0.002 and caps, then choose the smallest bound. Deterministic minimum power-of-two Up/Down compensation enforces the Down cap, with no scale search. Check exact integer identity and constant negative interval analytically. All original recorded features remain unchanged except the new hand-built hinge, evaluated by its explicit ReLU ideal with uniform GELU-error bound. No native activation or model replay. Require all2560A124format winners with full-vocabulary ideal margin>1, every word programme Gate<-16, unchanged native coefficients, and edit/weight caps. The new Gate is analytically off on those512wordwindows, so their final local raw outputs are identical; this is not a held-out preservation claim.

Local A142/A124inputs are exact; A240's earlier reader changes make installation a transfer that requires a fullscore. Intended benefit is a less sensitive newline emitter; it does not predict a text-agreement gain or fix upstream arithmetic sensitivity. Queue after A253full,A257v2,A258,A259, one heavy process at a time. No A260 fit/build/score yet.


### A261 paired HOST trajectory query target prepared (2026-11-11 05:25:31 CET)

A261 analyzer/compiler are syntax checked and reviewed, UNRUN. Previous A237/A250 query targets used HOST Q/K at the EDITED trajectory's inputs. A261 reconstructs the original HOST layer3 input by recombining its recorded branch outputs with exact RMS/residual algebra; the complete recorded HOST final state must reproduce bit-exactly. It then computes original HOST Q/K/RoPE scores at those original HOST inputs. No attention probabilities or native activations are evaluated. This directly targets earlier input drift as well as the local query edit.

Parent A191, exact same edited layer3 inputs as A237. Keep A237's original same-input-response bases, fixed canonical input dimensions[64,8,8,128],16PCAoutputs,whitening0.001,96exact protected format queries,192training/64validationwordwindows. Target mixtures use paired-HOST fractions[0,0.25,0.5,0.75,1] between ORIGINAL same-input HOST scores and ORIGINAL paired-HOST-trajectory scores. Metricridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25]. All original targets and normalizers are fixed before independent solves; no candidate residual becomes a target. Fraction0 at the prior selected ridge/strength must reproduce each A237matrix bit-exactly.

Select by actual-rounded-query paired-HOST centered causal MSE, requiring no worse same-input HOST MSE than original,source margins>=7.999,96query rounding change<1e-5,andcaps. Combined gain must be>=10percent over original A191 and>=5percent over A237/A240 under this SAME paired-HOST metric. No final-text or model-output improvement is claimed by the local metric. A261 only changesQ3, and needs its own fullscore if built. Queue after A253full,A257v2,A258,A259,A260, one heavy process at a time. No A261 run or checkpoint yet.


### A253 full H1 completed; A240 remains leader (2026-11-11 05:25:56 CET)

Official0795-score(scores/a253v1_full_H1.json) returns24,946/25,000exact+digits(99.784percent),text97.955322265625percent,KL0.0025985586717169797,CEchange-0.0009980670869804698,finite,H1false. The eight-unit threshold reader clears arithmetic but remains below98.5percent text and has68fewer matching predictions thanA240. It improves preservation over the ten-unit A234reader, while losing48clean additions relative toA234. All A253 construction sources remain immutable.

Noise exact/digits outof2000:1e-5:1995/1995;1e-4:1932/1932;1e-3:850/905. Thus its1e-4noise result exceeds bothA240(1577)andA234(1788), while1e-3exact falls belowA234(907). Two-in-a-row1996first,3secondgivenfirst,3both. Runtime28:20.61,peak3,854,056KiB. Completed300forward/100quick/17full;remaining0/0/23. No rung/best/finalverification/export.

A257v2 sparse copier output repair is now the sole heavy process(session71939),logartifacts/a257v2_sparse_copy_output_repair.log. A258,A259,A260,A261follow sequentially. A240 remains the arithmetic-qualified and overall text leader.


### A257v2 causal inverse still rejects (2026-11-11 05:27:03 CET)

A257v2 completed20.49seconds,peak3,379,376KiB. Word inverse passes all bounds(relative rawerror1.45601e-7,maximumnegative1.26694e-6,rowsum7.14872e-7). Format causal-affine inverse has all1664prefixesfullrank and no future coefficients; rawerror4.51213e-7,rowsum4.44089e-16. But maximum negative coefficient0.00551024753 exceeds0.002, so it rejects BEFORE fitting, noPT/checkpoint. Causality removed the earlier future leakage but did not resolve the ill-conditioned format inverse(minsingularratio3.98460e-6). The independent known-head control remains valid. No cutoff was relaxed.

A258 original-HOST winner-interval anchor fit is now the soleheavyprocess(session19435). A future copier-output design could avoid format inversion by bounding changes for every causal convex combination of the known source values; that is only an idea here, with no A262source/declaration/result yet. A259,A260,A261follow the current fit sequentially.


### A258 anchor interval fit rejected (2026-11-11 05:28:39 CET)

A258 completed23.19seconds,peak4,680,900KiB. Baseline8003/8192restricted matches; best8006, only3extra against the declared16, so noPT/checkpoint or full-vocabulary/format follow-up. Bestmatchconfiguration:mistakeweight16,HOSTmarginfraction0.5,ridge1,strength0.5; state0.01635540415 versus0.01578285731, direction0.01504447935 versus0.01456317699, maximum raw-RMSchange0.05972135. Original513training mistakes have[447,434,420,390,89]feasible radial targets at fractions[0,0.125,0.25,0.5,1]. The optimistic per-token A255v2 interval result does not transfer to a useful learned linear anchor map.

A259 lower-rank copier query repair is now the soleheavyprocess(session96975). A260,A261follow. A262 convex-mixture output-bound source is being reviewed and has not run or been declared yet.


### A262 convex-mixture copier output bound prepared (2026-11-11 05:29:37 CET)

A262 analyzer/compiler are syntax checked and reviewed, UNRUN. Retain A257's exact-word inverse,192/64word split, original paired-HOST attention7 target,128outputrows,original conditional coordinate ranking,full16Gate/Up frame protection,counts[16,32,64,128,256],ridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25]. Do not invert format attention. Instead every causal head output is a convex combination of its known native source values. For each rounded candidate, sum the per-head largest available-source norms of deltaO*V to bound raw and gain-weighted raw displacement, and sum the corresponding absolute Gate/Up functional changes. These bounds cover every causal mixture, without estimating probabilities.

Propagate raw norm intervals through attention RMS, bound post-attention state norm, then propagate each Gate/Up numerator and the pre-MLP RMS denominator to an interval. Include the actual rounded frame-null residual; do not assume it is exactlyzero. Three synthetic displacements(firstsource,latestsource,uniformcausal sources) must lie in those intervals for each candidate. This is arithmetic on recorded values and explicit convex-mixture controls, not an attention or native MLP replay. Bounds are for real arithmetic on rounded matrices, not a complete float32 execution certificate or held-out guarantee.

All A257 screens stay:branchRMSimprovement>=5percent,pairedstate/directionimprove,all256wordG<-16,all128formatGsigns certified,Gate/Up changebounds<0.001oforiginal per-unitmax,raw-RMSchangebound<0.002,relativeframe-null<1e-6,caps. Exact local A142 inputs remain an explicit transfer surrogate for A240. Queue after A259,A260,A261; no A262run/PT/checkpoint yet. A259 is still soleheavyprocess.


### A259 copier query passes the declared local screen (2026-11-11 05:30:00 CET)

A259 completed48.53seconds,peak1,836,264KiB. Original causal scoreMSE0.0011284854647783→0.0009463782175400(16.1373percent),9.7370percent better thanA256's0.0010484672895380. All heads select8protectedPCs/null632, ridges[0.0001,0.01,0.0001,0.01],strengthsall0.75. Minimum source margin8.34306061; maximum raw-query change across all3200formatqueries9.11122e-5(<0.01),maximum protected-PCrounding change3.78558e-11. Ordinary query rows remain bit-identical. Predicted4,978,700edits,maxQ0.62890625; overallprogramme/normmax1e6unchanged. The480-PCcontrol reproduces each prior A256selected MSE exactly. Eligible PT/JSON saved; no checkpoint/fullscore yet.

The fixed locally screened A259composition onA240 will be built and full-scored after the short queued analyses. It remains an upstream-transfer candidate, not a text improvement claim. A260flat-newline analyzer is now soleheavyprocess(session14977),followed byA261andA262.


### A260 flat-newline geometry passes (2026-11-11 05:30:57 CET)

A260 completed8.64seconds,peak2,708,728KiB. Sharpness4096 has minimum declared hinge-error bound0.000169975962,OFFmultiplier0.04615951207,maximum negative Gate-512.0039079. All three deterministic Up scales are1; no rescaling is needed. The literal integer curve identity has exactly0error and the negative interval is constant. All512word programme Gates remain below-512; native coefficients and gains are bit-identical, so those local native outputs are unchanged.

The explicit new-hinge ReLU ideal preserves all2560recorded format winners, minimum full-vocabulary margin1.30135918; maximum raw-RMSchange0.03749009. Uniform GELU feature-error bound is0.00356831313 for the new hinge, not a guarantee of actual execution. Selected Gate/Up/Downmax100000; existing global gainmax1e6unchanged. Predicted4,978,700edits. Eligible PT/JSON saved, no modelcheckpoint yet. Build and full-score A260 separately after A259; do not compose their unverified changes. A261 paired-HOST trajectory query analysis is now soleheavyprocess(session83092),thenA262.


### A261 paired-trajectory target and A262 convex output bounds reject (2026-11-11 05:32:30 CET)

A261 completed33.23seconds,peak4,125,352KiB. Original paired-HOST causalMSE0.00157738249024→0.00142241625989(9.82426percent),below10percent. A237/A240under this fixed paired-HOSTmetric is0.00142456274575, so the extra reduction is only0.150677percent,below5percent. Headtargetfractions[0.5,0,1,1],ridgesall1,strengthsall1.25,sourcegaps>=7.99999398. Fraction0reproduces all four A237matrices bit-exactly, and recordedHOSTbranches reconstruct finalHOSTstate bit-exactly. Original same-input response bases may limit a target for earlier drift, but no such alternative basis is yet implemented. NoPT/checkpoint.

A262 completed39.12seconds,peak3,881,564KiB. Original pairedHOSTattention7branchRMS0.65552525885. Bestbranch0.64275872270(count256,ridge1e-6,strength0.5),only1.9475percent reduction,below5percent. Pairedstate0.00672141716→0.00668469103 anddirection0.00475786285→0.00474712221improve slightly. Its conservative formatbounds also reject:rawRMS0.01780283,Gate0.01401709,Up0.01785010,two Gate signs uncertified. Syntheticfirst/latest/uniformsource intervalcontrols allpass. No formatinverse is needed or used; noPT/checkpoint.

A259compiler is now soleheavyprocess(session22698),thenA260compiler. Once those complete, full-score A259 first and A260 separately afterward. No new official score has started;remaining0forward/0quick/23full. A240stillleader,no rung/best.


### A263 paired-response query bases prepared (2026-11-11 05:34:30 CET)

A263 analyzer/compiler are syntax checked and reviewed, UNRUN. A261 changed targets while retaining A237's SAME-INPUT response bases. A263 instead derives both the canonical input directions and16PCAoutput directions from the ORIGINAL paired-HOST normalized query at HOST's recorded input minus the ORIGINAL A191 normalized query at its own input. This incorporates upstream drift into the basis. The target is fixed to the full original paired-HOST causal score difference; no target-mixture search. All original targets/normalizers are fixed before independent solves; no A261candidate residual is fitted.

Use canonical input counts[8,16,32,64,128],whitening0.001,metricridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25],192/64word split,96exact protectedformatqueries. Reproduce A261baseline metrics,bit-exact HOSTbranch reconstruction, and all previoussource/edit/weightguards. Selectactualrounded paired-HOSTMSE with same-inputMSE no worse than ORIGINALA191; combined>=10percentoriginal and>=5percentoverA237/A240under the samepairedmetric. No finaltextpredictionclaim. Queue theanalysis afterA259full, beforeA260full; noA263run/checkpointyet. A259buildiscomplete(20.09seconds,peak948236KiB,4,978,700edits,max1e6); A260compileriscurrentlysoleheavyprocess(session85981).


### A259 full H1 started; A260 built separately (2026-11-11 05:34:56 CET)

A259deltaexperiments/a259v1_low_rank_copy_query_on_a240/delta.pt SHA39f6c03da1c8b7d38b2223e59e5f680b92627808711db109a0b456c3cd81f350,4,978,700edits,max1e6. FullH1isnowsoleheavyprocess(session76623),logartifacts/a259v1_full_H1.log,outputscores/a259v1_full_H1.json. Its compiler/analyzer and all transitive construction sources are immutable from this metered build.

A260buildcompleted9.65seconds,peak960560KiB. Separate deltaexperiments/a260v1_flat_newline_emitter_on_a240/delta.pt SHA7c097c66e14ecd164fd9834d93ec4100fe8fc61d341328feb0972da78e65571a,also4,978,700edits,max1e6. It doesnotincludeA259queryrepair. Preserve its exactbuilt recipe; fullscore followsA259andqueuedshortanalyses. No A260officialrun yet. Folder39,145,505,014bytes,under40GB.

Seventeenfullscorescompleted,eighteenthreserved,22fullremain;0forward/0quick. A240remainsleader, no rung/best/finalverification/export. A263 ispreparedunrun. A264sparsepaired-HOSTkeyrepair is only an idea, not coded/declarated yet.


### A264 sparse paired-HOST key repair prepared (2026-11-11 05:37:18 CET)

A264 analyzer/compiler are syntax checked and reviewed, UNRUN. ParentA240 has exactly observed pre-layer3 inputs and unchanged nativeK3; fix its Q3 and all norms. Original normalized HOSTkeys at the original HOSTtrajectory are the target, using original edited-input raw-key RMS as the fixed linear-fit normalizer. Recombine recorded HOSTbranches and require bit-exact finalHOSTstate; require A261's A237/A240paired-causal baseline metric to reproduce. No attention or native activation execution.

Select exactly128input coordinates once by original conditional explanatory energy with fixedridge0.001. Within that sparse support, protect leading[1,4,8,16,32]PCdirections of the first96formatwindows'864keyinputs. For each independent nullspace fit use original target,ridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25],192training/64validationwordwindows. Every other keyinputcolumn and every other modeltensor staybit-identical. This adds atmost32768edits, fittingA240's37568spare.

Require>=10percent normalizedpairedkeyRMS improvement,>=5percent pairedcausalQK MSEimprovement with all four fixed editedqueries,same-inputHOSTscoreMSE<=1.05original,allcurrent/historical sourcegaps>=7.999,rawkeychange<0.01onall1664formatwindowsx9positions,protectedPCroundingchange<1e-5,finite/edit/weightcaps. Fits never use candidate residuals or operand/sumlabels. This evaluates linear/RMS/RoPE scores only; downstream attention and text need fullscore. QueueafterA259fullandA263, beforeA260full. No A264run/checkpointyet.


### A265 independent shared-value branch repair prepared (2026-11-11 05:44:31 CET)

A265 analyzer/compiler are syntax checked and reviewed, UNRUN. Start from ORIGINALA189 with nativeO3, independently ofA191's fitted outputprojection; changeonlyV3. Existing factual reader3mixtures remain applicable because upstreaminputs,Q/K/gains are unchanged, and the original raw branch must reconstruct<1e-5relative. The target is ORIGINALpairedHOST normalized attention3branch minus ORIGINALA189branch, with fixed originalA189rawRMS. No candidate residualrefit, alternatingO/Vrepair, attention or nativeactivationexecution.

Protect actual rounded Gate/Upfunctionals and every earlier0–2post-MLPcoordinate with existing programmegain>=1e4, through all four outputheads. Derive the value-output nullspace once; choose16PCAoutput directions and canonicalinputdirections from ORIGINALsourcevalue response(HOSTV atHOSTinput minusA189V ateditedinput), restricted tothatnullspace. Fixedwhitening0.001; inputcounts[8,16,32,64,128],ridges[1e-6,1e-4,0.01,1,100],strengths[0.25,0.5,0.75,1,1.25]. The exact finite normal equations account for all cross-head mixture covariances and the sharedVmap. Original192/64word split,128formatwindows. No arithmeticanswerlabels; only individual source-digit diagnostics.

Require>=10percent pairedHOSTbranchRMSgain,pairedstate/directionimprove,all256wordG<-16,EqG>31.9,source-digitroundingcorrect anderror<min(0.25,originalmax+0.01),positive reference,formatrawRMSchange<0.002,Gate/Uprelativechange<0.001,allformatsignsunchanged,roundedrelativeframe-null<1e-6andcaps. Existing denseVsupportlargelypaysfortherepair; allothertensorsstayfixed. A265isadifferentcheckpointfromA240,A259andA260. QueueafterA259full,A263,A264,beforeA260full. NoA265run/PT/checkpointyet.

A259's partialarithmeticaggregateis24,960/25,000exact+digits,matchingA240; text/noise/repeatedresultsarestillpending. Itisnotacompletedscoreorrungpass.


### A266 competing-token geometry for smaller output pools prepared (2026-11-11 05:53:18 CET)

A266geometryanalyzer is syntax checked and reviewed, UNRUN. The old A118pool reduction failed below96coordinates with its original vocabulary metric. A266 selects32nondigit/nonnewline competitor token IDs ONCE from the original96-coordinate literal curve over digits0–9at offsets[-0.2,0,0.2]andnewline-1. It adds their frozen-unembedding Gram to the original vocabulary metric, normalized to the same mean diagonal. This is single-digit/token geometry, never an operand/sum fit or a model run.

Poolcounts[32,48,64,80,96],penaltymultipliers[0,0.1,1,10,100],groupbiases[1,2,4,8,16]. Each independent metric solve enforces exact11-token interpolation; original96/penalty0/bias2must reproduce the existing basis within1e-6. Choose the maximum minimum literal full-vocabulary margin perpool, then testonlythatgeometry on2560recordedformatstates. Fixedanchor2/rawoutputgain50000; cleanupscale follows the analytic ratio of mean integer-direction energies plusanchorenergy, not another trial. All Gate/Upfunctions remain unchanged. Restore HOSTpostgain only on removedstrongcoordinates; retain the existing native Down contribution for this geometry check.

Eligible geometry requires pool<96,allcleanupcoordinatespresent,literalmargin>1,all2560formatwinners withmargin>1,andfinite/cappedprogrammeweights. This is deliberately NOT a deployable native-preservation candidate: removedstrongrows still have their oldnativeDown coefficients. Any positive geometry needs a separatelydeclared original-HOST native reconstruction and fullscore. No compiler or checkpoint will be created byA266. QueueafterA259full,A263,A264,A265andbeforeA260full. NoA266run/resultyet.

A265review additionally added an independent small explicit-design check of its shared-head normal-equation tensor ordering, and bit-exact HOSTlayer2-state reconstruction. This doesnotchange its declared fitting target or scale values; A265hasnotrun.


### A259 full H1 is a small new text leader (2026-11-11 06:04:34 CET)

Official0797-score(scores/a259v1_full_H1.json) returns24,960/25,000exact+digits(99.84percent),text98.01788330078125percent,KL0.0025266539860200844,CEchange-0.0003389812104375167,finite,H1false. This adds14matching predictions/131072overA240, a0.01068115percentagepoint gain, with unchangedclean arithmetic. A259isnowthearithmetic-qualifiedandoveralltextleader, still0.48211670percentagepointsbelow98.5percent. No rungpasses.

Noiseexact/digitsoutof2000:1e-5:1998/1998;1e-4:1576/1999;1e-3:261/1591. Two-in-a-row1998first,1secondgivenfirst,1both. Runtime28:18.40,peak3,853,028KiB. Completed300forward/100quick/18full;remaining0/0/22. A259andalltransitiveconstructionsourcesremainimmutable. No best/finalverification/export.

A259edit audit confirms every sparseindexarrayequalsA240 andeveryvalueoutsideQ7isbit-identical. Inheriteddisjointpurposecountsremain588,645arithmetic/accommodation+4,390,055native-onlytextrepair=4,978,700,max1e6. NewnativeQ7repair overlapsoldarithmeticchannelentries; these are bookkeepinglabels,notcausalcounts. Auditartifacts/a259v1_low_rank_copy_query_on_a240_edit_category_audit.json andREPORTper-tensortableupdated.

A263paired-responsequerybasisanalysisisnowsoleheavyprocess(session97281),thenA264,A265,A266sequentially. A260flat-newlinecheckpointremainsbuiltbutunscoredandonA240,notA259; preserveitsseparatetest. Itfollows thequeuedshortanalyses.


### A263 paired-response query bases rejected (2026-11-11 06:08:44 CET)

The independently constructed paired-HOST response bases produce no eligible candidate or checkpoint. Combined validation causal-score MSE is 0.0014210188851945952 versus original A191's 0.001577382490244112: a 9.91285284% reduction, below the declared 10% requirement. The improvement over A237/A240's 0.001424562745747269 is only 0.24876830%, below the separate 5% requirement. All head-level protections pass, including a minimum source margin of 7.99999334, but that does not waive the combined improvement guards. Selected diagnostic input counts are [128,8,8,128], all ridges 1 and strengths 1.25. Nominal edit count would be 4,978,701.

Wall time 34.88 seconds; peak RSS 4,252,672 KiB. This changes the original paired-response basis as well as the target, and still offers little additional local repair over A237. No model or meter was run. A264 sparse key repair now runs alone; A265, A266 and the already-built A260 full score follow.


### A264 sparse paired-HOST key repair rejected (2026-11-11 06:09:17 CET)

No candidate satisfies the declared guards; no PT or checkpoint was saved. Best paired causal-score MSE is 0.001416108090054488 versus 0.001424562745747269 (0.5935% improvement, below 5%), with one protected format PC, ridge 0.01 and strength 0.75. Its normalized key RMS is 0.020845590690965813 versus 0.021038277138272075, source margin 7.98422606 below 7.999, and maximum format raw-key change 0.02339060 above 0.01. The best key-RMS trial (strength 0.5) reaches 0.0208425777207454, only 0.9302% improvement against the 10% requirement, and also fails routing/raw-change guards.

The paired HOST and A237/A240 baseline controls reproduce. Wall time 29.43 seconds; peak RSS 3,952,476 KiB. The sparse 128-column support fits the edit allowance but provides too little measured repair. A265 now runs alone, followed by A266 and A260 full H1. No meter budget used.


### A265 shared-value repair rejected (2026-11-11 06:10:32 CET)

No eligible candidate or checkpoint. Best branch RMS is 0.19286091340378358 against 0.19688638047902038, a 2.04456% improvement below the declared 10%. This uses 128 canonical input directions, 16 output directions, ridge 1 and strength 1. Paired-state RMS improves 0.004904593457→0.004881299325 and direction 0.002928882090→0.002923494999, but format raw-RMS change 0.00921318 exceeds 0.002 and Up relative change 0.00420163 exceeds 0.001. Source digits remain correct, error 0.18188731; minimum equals Gate 733.45937 and word Gate maximum −33.19929.

The protected value-frame rank is 56 (200-dimensional nullspace). Synthetic explicit-design normal-equation control error is 1.42e-14; original HOST state reconstruction and factual mixture controls pass. Wall time 36.41 seconds, peak RSS 3,339,516 KiB. This independent shared-V construction did not supply enough native repair while keeping the arithmetic interface stable. A266 geometry now runs alone. No meter budget used.


### A266 finds an eligible 80-coordinate output geometry (2026-11-11 06:10:55 CET)

The fixed competitor penalty produces one eligible smaller geometry: pool 80, penalty 0.1, group bias 2, literal minimum full-vocabulary margin 1.25993538. With the unchanged arithmetic Gate/Up features and current native Down contribution, all 2,560 recorded format outputs remain correct, minimum margin 1.14122772. Sixteen formerly strong-gain coordinates return to their HOST gains. Raw output gain stays 50,000, anchor amplitude 2, analytic cleanup scale ratio 1.00314236, maximum programme weight 100,000.

The best 64-coordinate curve (penalty 0, bias 4) has literal margin 0.64076614 and fails 15 format positions, minimum −0.64066505. The 96-coordinate control selects the original zero-penalty/bias-2 curve, reproduces its basis and all format outputs, margin 1.30151367. Counts 32/48 cannot retain all cleanup coordinates and are skipped. All declared trial results are in artifacts/a266v1_competitor_penalized_output_pool.json. Wall time 14.20 seconds, peak RSS 1,923,244 KiB.

This is geometry only: the 16 restored-gain coordinates still have tiny native coefficients, so text preservation has not been established and there is no checkpoint. A separately declared A267 native repair will use original HOST targets and exact energy consistency, without candidate residual refits. A260's already-built flat-newline checkpoint now runs full H1 separately on its A240 parent. No A259 composition.


### A267 exact-energy native repair prepared (2026-11-11 06:14:23 CET)

A267 analyzer/compiler are syntax checked and reviewed, UNRUN. Use A266's eligible 80-coordinate geometry; original A142 recorded inputs and 384/128 word windows provide the local construction. Install only if eligible on A259, whose final Gate/Up/Down/gains must match A142 bit-for-bit before the edit. All earlier weights stay fixed. This transfer needs a full score.

Let S be all original 96 strong rows, g the proposed gains, b the recorded incoming final state, and h the original paired HOST final state. Set y=(h_S-b_S)/g_S. With the original raw complement held fixed, solve n²=(E_outside+640*epsilon)/(640−sum(y²)) and raw_target_S=n*y. Reject zero/near-zero gains or any nonpositive denominator without clipping or dropping rows. Verify the energy identity and target branch algebra numerically.

The fixed prior map is original A142 native rows times old_gain/new_gain. One-shot native fits use the original exact target and its fixed raw/final normalizers, all 2,037 native features, ridges [1e-6,1e-4,0.01,1,100] and strengths [0.25,0.5,0.75,1,1.25]. These are independent solves of the original target, never residual fits to another candidate. No model, native GELU or attention execution; no answer labels.

Require at least 16 additional matches/8,192 against the original HOST in both restricted and full-vocabulary readouts, paired state and direction RMS at most 1.05 times baseline, word raw-RMS change below 0.25, then all 2,560 recorded format winners with minimum margin greater than 1 and raw-RMS change below 0.02, plus edit/finite/value limits. Native Down outside the original 96 rows remains bit-identical; programme columns come only from A266. Gate/Up functions do not change. Queue A267 after the running A260 full score.


### A268 sparse ordinary-query cancellation prepared (2026-11-11 06:18:46 CET)

A268 analyzer/compiler are syntax checked and reviewed, UNRUN. Independently reconstruct Q3 from HOST ordinary rows and ORIGINAL A191/A61 programmed-channel native components and rank-one position direction. Original paired HOST trajectories supply gain targets. A259 is the installation parent and local comparison baseline, with unchanged K/V/O and all other layers. Verify exact pre-layer3 provenance, A61 rank-one decomposition and the A261/A237 paired causal-score baselines.

Choose sparse native format-mean cancellation supports [1,4,16,64,128] by one conditional covariance ranking, using the first 192 word windows and 96 current-format means. Fixed format-variation mass 10000 and covariance ridge 1e-6 times word mean diagonal. All ordinary query coefficients outside each support remain bit-identical to HOST. Keep the 64 programmed rows dense. One shared ordinary q_norm gain solve targets original paired HOST normalized queries, with original programme amplitude; then compute final per-head programme amplitudes by the exact quadratic source-gap bound. Do not refit gains after that calculation. Programmed q_norm gains remain unchanged.

Source amplitudes use all 1,664 known format windows; current32/historical512 margins are therefore diagnostics, not unseen routing validation. Separate 64-word-window validation uses linear/RMS/RoPE causal scores only, no softmax or downstream activation replay. Require at least 300,000 fewer edits, all source gaps at least 7.999, paired and same-input causal score MSE and paired query RMS no more than 1.05 times A259/A240 baseline, plus finite/edit/value limits. This is a budget-release design, not a claimed text improvement. Any eligible checkpoint needs its own full score before expanded native repairs are considered. Queue after A267; A260 full H1 remains the sole heavy process.


### A260 partial arithmetic and A269 protected zero-digit interval (2026-11-11 06:23:45 CET)

A260's partial official arithmetic aggregate is 24,879/25,000 exact and digits (99.516%), only four cases above the H1 threshold and 81 fewer than A240. Text/noise/repeated-problem results are pending, so this is not a completed score. The scalar recoding used by A260, −1+2ReLU(d+0.5)−ReLU(d), equals 2d for −0.5<d<0: it doubles negative errors around digit zero. This is a mathematical vulnerability, not a confirmed explanation of the 81-case loss, since no new internal activations are available. No arithmetic failures are fitted.

A269 analyzer/compiler are prepared, reviewed and syntax checked, UNRUN. Replace that literal identity by −1+3ReLU(d+0.5)−2ReLU(d+0.25). This equals d throughout d>=−0.25 while retaining the flat newline value −1 for d<=−0.5. The hand-derived Down recoding is [V0−V1,−2V1,V2+3V1]; reuse the same 11 units and the existing hinge−0.5, changing only unit834 to hinge−0.25 plus the first three Up/Down rescalings. Native Gate/Up/Down and all gains remain unchanged.

A269 independently starts from original A142 final weights and installs on A259 after exact final-weight identity checks; it does not alter or refit A260's weights. Same five sharpness values [16,64,256,1024,4096], analytic OFF coefficient to force all known negative gates below −500, hinge-error bound below 0.002, deterministic power-of-two cap rescaling, and all 2,560 recorded format winners with ideal margin greater than 1. Explicitly verify the whole interval [−0.25,9.25] and the newline plateau. Folded digit391 cleanup still has an error if the newline register differs from −1. Captured-format evaluation uses the hand hinge's ReLU ideal and a uniform GELU error bound; a full score is required.

Queue after A267 and A268, once A260 full scoring finishes. No A269 analysis, checkpoint or meter has run. The alternative idea of a free final-state ray scale has no allocated experiment or implementation.


### A270 exact positive HOST-ray targets prepared (2026-11-11 06:25:54 CET)

A270 analyzer/compiler are reviewed and syntax checked, UNRUN. This is an independent alternative to A267, whose exact target fixes final-state scale to one. Compare A266's 80-coordinate geometry and the unchanged original 96-coordinate geometry; no new output geometry is searched. All targets come from original A142 inputs and original paired HOST final states, and installation would be on A259 after final-weight identity checks.

Let t be inverse raw RMS and alpha a positive scale of the HOST final state. On the fixed 544-coordinate complement, the least-squares alpha is affine in t: alpha=a+b*t. On the original 96 rows set y=(alpha*HOST−before)/gain. The exact RMS identity sum(y²)+t²*(outside_raw_energy+640*epsilon)=640 is a quadratic in t. Compute both roots directly, require t>0 and alpha>0, and choose the root with lower original HOST direction error per token. Any row without a valid root rejects that geometry, with no clipping or discarded observations. Check the energy identity, reconstructed target branch and complement normal equation. Ideal target readout is an oracle diagnostic, never an installed result.

Use the same independent original-target native fits, 384/128 split, 2,037 native features, ridge values [1e-6,1e-4,0.01,1,100] and strengths [0.25,0.5,0.75,1,1.25] as A267. The fixed algebraic gain-compensated A142 map is the prior; no A267 fitted result enters A270. Require +16/8,192 restricted and full-vocabulary matches, state/direction<=1.05 baseline, word raw-RMS change<0.25, all 2,560 format winners with margin>1 and format RMS change<0.02, plus caps. Gate/Up and native Down outside the original96 rows remain unchanged. No native activation/attention replay, gradients, whole-answer labels or iterative residual fitting.

Queue: after A260 full score, run A267, A268, A269, A270 sequentially, building only eligible results before selecting the next official score. No A270 analysis or checkpoint exists yet.


### A271 per-digit plateau output geometry prepared (2026-11-11 06:33:01 CET)

A271 geometry-only analyzer is reviewed and syntax checked, UNRUN. Instead of the current five-function grouped curve, use eleven constant literal token directions joined by twenty hand hinges: newline-to-zero thresholds [−0.75,−0.25], then [d+0.25,d+0.75] for d=0..8. Each integer digit has a plateau of half-width 0.25; newline is flat below −0.75. For each transition the two Down columns are +2*(next−previous) and −2*(next−previous). The constant is the newline direction. This is a literal per-digit table, not a solve over addition examples.

Retain the separate original linear-digit cleanup feature, constant flag cleanup and six other cleanup functions, so physical digit391 cleanup is no longer flattened with the output curve. Total required units would be 28 versus 11: seventeen additional units, whose allocation/native repair is explicitly deferred. Programme feature evaluation uses the captured positive common feature times the hand hinge ideals; no native activation or attention is executed.

Pool counts [32,48,64,80,96] use all ten cleanup coordinates first and then the remaining original pool order. From original 96-coordinate onehot directions select 32 competing tokens once. Independent unembedding-metric duals use penalties [0,0.1,1,10,100], normalized to the original mean metric diagonal; normalize each token direction to unit norm. Verify all literal plateaus at offsets [−0.2,0,0.2], then choose the best literal margin per pool and check all 2,560 recorded format states. Anchor amplitude 2 and raw gain 50,000 stay fixed; cleanup scale follows the analytic energy ratio. Return removed output coordinates to HOST gains.

Eligible geometry requires pool<96, literal and recorded-format margins both>1, all format winners, and capped abstract programme weights. Current native Down contribution is retained only as a geometry surrogate. No compiler/checkpoint is produced: any positive result still needs seventeen-unit allocation, independent original-HOST native reconstruction and full scoring. Queue after A267–A270 when A260 full scoring finishes.


### A260 full H1: stronger newline noise robustness, lower clean accuracy (2026-11-11 06:40:05 CET)

Official 0799-score (scores/a260v1_full_H1.json) completed: 24,879/25,000 exact and digits (99.516%), text agreement 98.00643920898438%, KL 0.0025969166188788834, CE change −0.0007606220887339532, finite, H1 false. Arithmetic clears its threshold by only four cases; text is one matching prediction/131,072 below A240 and fifteen below leader A259. No rung passes and A259 remains the leader.

Noise exact/digits out of 2,000: sigma1e-5 1992/1992; sigma1e-4 1991/1991; sigma1e-3 852/1501. Against its A240 parent, moderate-noise exact answers improve from1577 to1991 and high-noise exact answers from261 to852, while high-noise correct digits fall from1591 to1501. Thus the newline plateau helps exact termination under noise but the altered negative-zero interval has a clean/digit cost. That interval is a mathematical candidate explanation, not a measured internal causal diagnosis. Two-in-a-row:1990first/1secondgivenfirst/1both.

Wall28:16.45, peak3,855,696KiB. Checkpoint SHA 7c097c66e14ecd164fd9834d93ec4100fe8fc61d341328feb0972da78e65571a; 4,978,700 changed scalars, maximum absolute weight1e6. Its construction sources and transitive dependencies are immutable. Completed300forward/100quick/19full; remaining0/0/21. No best, final verification or export.

A267 exact-energy native repair is now the sole heavy process, then A268 sparse query cancellation, A269 quarter-margin flat newline, A270 positive HOST-ray target and A271 plateau geometry. A269 was prepared before this completed noise/text result and retains its declared checks.


### A267 exact-energy repair rejected (2026-11-11 06:40:31 CET)

All original paired targets for the 80-coordinate geometry are physically feasible: minimum new target gain361, minimum energy denominator613.68658, exact-energy relative error2.22e-16, branch reconstruction error4.55e-13, and target raw-RMS ratio1.000298–1.021214. Nevertheless no fitted map improves the held-back predictions. Best restricted count is7999/8192 versus8003baseline (ridge0.01,strength0.25), below the declared +16 requirement. Its paired-state RMS0.01563069471 and direction0.01438250873 improve against0.01578285731/0.01456317699, illustrating again that smaller state error need not improve token rankings.

Best state RMS0.01561231628 (ridge1,strength0.5) has7998matches. No full-vocabulary or format follow-up was triggered, no PT/checkpoint. Wall21.77seconds, peak4,483,000KiB. A268 sparse-query analysis now runs alone. No meter unit used.


### A268 sparse query reconstruction rejected (2026-11-11 06:41:23 CET)

All five sparse supports preserve source routing (minimum margins above8.0000024) and free393,215–490,751edits, but native query errors are substantially worse. Best paired causal-score MSE0.005990832214 (support128) versus leader0.001424562746; same-input MSE0.005439801248 versus0.000833213286 and paired query RMS0.06264065763 versus0.03150798113. Support1 (root coordinate334 only) hasMSE0.006141811615. Hence every trial fails the declared<=1.05baseline guards. No PT/checkpoint.

Original paired baselines and programme rank-one control reproduce (max decomposition error8.19e-9). Wall17.91seconds, peak4,020,916KiB. Sparse mean cancellation itself remains nearly constant over format inputs; restoring ordinary raw rows removes A61's per-channel raw-energy accommodation, which is a possible source of the larger score errors. This is not a measured causal decomposition. A separate original-target energy-accommodation construction could investigate that later; no such experiment is currently allocated. A269 now runs alone.


### A269 quarter-margin flat newline passes local checks (2026-11-11 06:41:50 CET)

A269 selects sharpness4096, analytic OFF multiplier0.04610358098484369; maximum negative Gate−512.00405 and literal hinge error bound0.0001697766221. All three deterministic Up scales are1. The algebraic curve identity holds across [−0.25,9.25] to1.57e-10 and the newline interval<=−0.5 is constant. All native features remain unchanged on512 recorded word windows, with all programme Gates below−512.

Hand-hinge ideal evaluation preserves all2,560format winners, minimum full-vocabulary margin1.30135727, raw-RMS change0.00130102516. The global uniform GELU feature error bound is0.00356831313; no exact new activation is replayed. Nominal edits4,978,700, largest modified final-layer value100,000 (whole checkpoint still max1e6). Wall8.49seconds, peak2,709,132KiB.

The compiler is now running on A259. A270 and A271 short analyses follow before the next official score; A269 has not yet been metered. Its local input trajectory remains the original A142/A124 surrogate, so the new composition requires the full test.


### A269 checkpoint built; A270 running (2026-11-11 06:42:26 CET)

A269 saved atomically at experiments/a269v1_quarter_margin_flat_newline_on_a259/delta.pt. SHA f44a93ff3742d2fe49ef383ea9ce2573db3b7c1b1be064edc5d8ae8b7f246a77, 4,978,700 changed scalars, maximum1e6. Build9.68seconds, peak946,292KiB. Exact programme/native support and unchanged-gain checks pass. Preserve this built recipe for its separate full score; it has not been metered.

A270 positive HOST-ray native repair is now the sole heavy process, followed by A271 geometry. A267 and A268 remain rejected without checkpoints.


### A270 positive HOST-ray repair rejected (2026-11-11 06:43:14 CET)

Both80- and96-coordinate geometries have valid positive quadratic roots for every token. Energy identity error is at most4.44e-16, complement normal-equation error1.13e-15, and target branch error6.83e-13. The ideal per-token targets give8009 and8010restricted matches respectively versus8003baseline; these are only geometry diagnostics.

The fitted80-coordinate map peaks at7999matches (ridge1,strength0.25), stateRMS0.01562165511, direction0.01437020271. The fitted original96-coordinate control peaks at8004 (ridge100,strength1), state0.01576996793, direction0.01454872576: only one extra prediction, below the required16. No eligible PT/checkpoint or full-vocabulary/format follow-up. The lower state-error trials also fail the match requirement.

Wall35.04seconds, peak4,734,668KiB. The positive-ray target does not overcome the weak prediction gain of these one-shot final native maps. A271 plateau geometry now runs alone; A269 remains built and unscored. No meter unit used.


### A271 plateau geometry passes at64 and80 coordinates (2026-11-11 06:43:37 CET)

A271 finds two eligible reduced geometries, both with fixed competitor penalty0.1. Pool64 has literal minimum margin2.91748428 and all2,560format winners, minimum margin2.24460602; pool80 has literal4.21647644 and format4.25272560 with allwinners. These remove32 or16 high-gain output coordinates respectively. The separate linear-digit cleanup remains, unlike the flattened cleanup in A260/A269. Abstract programme maximum100,000, anchor2, raw gain50,000 and analytic cleanup ratio1.04439897. Seventeen additional units are required in either case.

Pool32 fails allformat outputs (best literal−13.0853); pool48 fails300positions (best literal−1.70169, format−2.70233). The96-coordinate control has literal5.51072502/format5.28610229 but is excluded as a reduced-pool candidate. All plateau algebra identities hold within4.83e-15. Wall16.66seconds, peak1,950,752KiB. These are hand-geometry results with original native contribution retained; no checkpoint or text-preservation result is claimed.

A272 will now be prepared for actual unit allocation and independent original-HOST native reconstruction, following the conditional design recorded above. It has not been implemented or run. A269 starts its separate full H1 score as the sole heavy process. After reservation,20full scores remain; no forward/quick budget. A269 and its transitive construction sources are now immutable.


### A272 plateau writer allocation and native reconstruction prepared (2026-11-11 06:51:28 CET)

A272 analyzer/compiler are reviewed and syntax checked, UNRUN. Start from original A142, comparing A271's two eligible geometries (64/80 strong coordinates), and install only a screened result on A259. Verify matching final weights and exact local old/fresh trajectories, including native layer8 Gate rows and native Down columns, the original recorded zero programme features, new selector Gates<-16, and identical upstream0..7 for the native8 capture.

Allocate exactly17 extra units outside the original11 and existing512 donors. Their ranking is original HOST conditional native-feature variance after a single512-donor ridge0.001 solve, multiplied by HOST post-gain-weighted Down-column energy. Use the first256 original native17 windows; hold back128old and128fresh windows. The twenty literal hinges occupy the three old hinge units plus the seventeen extras. The original constant, linear digit cleanup and six cleanup features remain.

New gates are literal digit-minus-threshold*flag readouts with one analytically derived OFF coefficient each. Sharpness values [16,64,256,1024,4096], negative Gate<-500, literal hinge-error bound<0.002 and caps; select the smallest declared error bound. The twenty Up rows use the original common function with inverse sharpness. All28programme Gates must remain<-16 on known word/prefill inputs. Actual new hand Gates/Up values determine format ReLU ideals; report both global0.25 and analytic tanh-GELU tail error bounds, without evaluating any native activation.

Each pool independently uses the immutable NativeFinalReconstruction helper from original HOST native features/weights:512 donor repair at ridge0.001, one ordinary-gain accommodation, original strong-row targets with ridges [1e-6,1e-5,1e-4,0.001,0.01]. No A267/A270 map or candidate residual enters the target. Native branch RMS must improve at least2%; then full-vocabulary old-window matches must not decrease, fresh-window matches must gain at least8, and the combined gain must be at least16/16,384. Finally require all2,560format winners with minimum margin>1, finite weights and edit/value caps.

The full candidate has28units, a constant+20hinge literal basis, separate linear digit cleanup and6other cleanup functions. All unchanged native features come from the frozen meter records. No gradient, new Gemma execution or whole-answer fit occurs. A272 runs after the current A269 full score; no A272 allocation, analysis result or checkpoint yet.


### A273 sparse query plus original-energy accommodation prepared (2026-11-11 06:54:54 CET)

A273 analyzer/compiler are reviewed and syntax checked, UNRUN. This tests whether restoring a bounded set of A61-style raw-energy adjustments can repair A268's large query error without losing most edit savings. Reconstruct directly from original HOST ordinary rows and original A191/A61 programmed components; fix ordinary mean-cancellation support to16 coordinates using the unchanged original covariance ranking. Do not load or refit an A268 candidate.

Use original paired HOST raw query RMS as the fixed denominator for one shared ordinary numerator-gain solve. The original squared-energy target is one minus the canonical query energy divided by original paired HOST energy. Choose ordinary energy-carrier channels by one conditional explanatory-energy ranking at ridge0.001. Counts [8,16,32,64,128], energy ridges [1e-6,1e-4,0.01,1,100]; every fit independently targets the original energy. Reject nonpositive squared row scales. Multiply selected raw rows by square-root scales and divide their q_norm gains reciprocally, checking the actual rounded numerator identity.

Compute exact source-routing amplitudes last, with no gain or energy refit after that calculation. Unscaled ordinary channels remain bit-identical HOST outside16 columns; programme64 channels and chosen energy-carrier rows may be dense. All other weights stay A259. Separate192/64word split, same exact-input/baseline controls and all1,664known source-routing windows as A268. Source holdback names remain diagnostics since all are used by the amplitude bound.

Require at least150,000 fewer edits, paired/same-input causal score MSE and paired query RMS each<=1.05times A259/A240, every source gap>=7.999, rounded reciprocal-numerator error<1e-6 and finite/edit/value caps. The smaller saving requirement than A268 is explicit: some freed entries now pay for native energy restoration, leaving enough room for a possible larger native reconstruction. No attention/native activation execution, gradients, sum targets or iterative residual fits. Queue after A272, once A269 full scoring finishes. No A273 result/checkpoint yet.

A269's partial arithmetic aggregate is24,958/25,000exact and digits, recovering79ofA260's81lost cases and falling two below its A259 parent. Text/noise/repeated results remain pending; this is not a completed score.


### Limited arithmetic examples support A260 zero-digit vulnerability (2026-11-11 07:02:46 CET)

A read-only categorization of the at-most20 H1 failures exposed by each scorer is saved in artifacts/a260v1_shown_arithmetic_failure_categories.json. A259's shown first-error characters are predominantly expected9→8 (12cases), with9→7(4),9→6(2),8→9(1),0→1(1). A260's shown set contains9cases expected0→newline and4cases0→space, plus7digit9errors. Its13cases absent from A259's shown20 all first fail at a zero digit.

This is consistent with the hand-curve vulnerability around negative zero errors and A269's recovery of79clean cases. These are truncated examples, not complete failure sets; absence from A259's shown20 does not prove that A259 passed a particular example. No held-out text was read and no weights or targets were fitted from the examples. A269's checkpoint was already fixed before this categorization.


### A269 full score: quarter-margin newline improves moderate-noise termination (2026-11-11 07:15:14 CET)

Official 0801-score completed: 24,958/25,000 exact and digits (99.832%), text agreement98.01712036132812%, KL0.002633485473067763, CE change−0.00025070990447861874, all finite. H1 remains false. A259 retains the highest text agreement by one prediction/131,072 and has two more clean additions. A269 recovers79 of A260's81 lost clean cases.

Noise exact/digits out of2,000: sigma1e-5 1998/1998; sigma1e-4 1999/1999; sigma1e-3 690/1490. Against A259, moderate-noise exact answers improve1576→1999 with unchanged1999 correct digits; high-noise exact answers improve261→690 but correct digits fall1591→1490. Separate linear digit cleanup and plateaus remain candidates for the higher-noise weakness. Two-in-a-row:1998first/1secondgivenfirst/1both. No repeated-problem improvement.

Checkpoint SHA f44a93ff3742d2fe49ef383ea9ce2573db3b7c1b1be064edc5d8ae8b7f246a77, 4,978,700 edits, max1e6. Score wall28:08.21, peak3,853,772KiB. Sources and transitive construction dependencies remain immutable. Completed300forward/100quick/20full; remaining0/0/20. No rung banked, no final verification or export. A272 now runs alone, followed by A273.


### A272 plateau writer rejected at native reconstruction (2026-11-11 07:16:29 CET)

All20 hand hinges satisfy their declared gate bounds with sharpness4096: maximum negative Gate−512.0044647, maximum literal hinge-error bound0.0002594996954, maximum Gate weight4687.8833 and Up0.01221636. Exact local provenance checks pass. The17extra programme units cost more native behavior than the reduced output pool restores. Best native branch RMS is4.73114963097(pool64,ridge1e-5) or4.88022984581(pool80,ridge1e-5), versus3.35574019693 for A142. Both fail the required2%improvement.

Nominal edit counts4,961,452 and4,985,580 are within caps, but no full-vocabulary or format follow-up is triggered, and no PT/checkpoint is written. Wall20.13seconds, peak5,613,784KiB. This rejects the present28-unit/512-donor allocation, not the literal plateau identities. A273 sparse query-energy analysis now runs alone. Meter budget unchanged at20full remaining.


### A273 sparse query-energy accommodation rejected (2026-11-11 07:19:54 CET)

All25declared carrier-count/ridge alternatives completed. Best paired causal-score MSE0.0017911194721(count128,ridge1e-6) improves substantially over A268's uncompensated sparse query, but remains worse than A259/A240's0.00142456274575 and fails the1.05guard. Same-input MSE0.00120455132248 versus0.000833213285609, paired queryRMS0.0340672544015 versus0.0315079811286 also fail. The source margin8.00000264385 and reciprocal numerator error2.92e-8 pass.

This best trial would free159,743scalars(total4,818,957), with positive squared row scales(min0.41562884), but no candidate meets allpreservation guards. No PT/checkpoint. Wall24.33seconds, peak4,259,072KiB. No meter units used.

Next unimplemented hypothesis: retained native final-layer Gate functions might supply arithmetic nonlinear features through format-selective Up additions, preserving existing native features on words instead of removing17extraunits. First inspect linear Gate geometry and recorded factor intervals only. Do not evaluate native GELU or derive whole-answer targets. No A274recipe, numerical trials, or result yet.


### A274 native-Gate arithmetic reuse diagnostic declared (2026-11-11 07:22:06 CET)

Diagnostic only, no candidate weights. At the exact original A142/A123 final inputs, inspect surviving native Gate linear projections across2,560recorded format positions. Regress each Gate's normalized linear value on the existing physical digit and flag registers only to summarize hinge location and unexplained input variation; these are activation statistics, not answer targets or new Gate weights. Native factors are recovered solely from recorded GLU products divided by linear Up, with float32 product/dot intervals and a conservative native Gate sign/magnitude bound where division is unstable. No native GELU or derivative is evaluated.

Inspect the existing flag readout and five independent format-selective linear Up fields. Each minimizes original word energy plus format flag-target error with masses[1,100,10000,1000000,100000000], fixed diagonal covariance ridge0.001. Use384wordwindows/96formatsequences to construct fields; hold back128wordwindows/416formatsequences. The format target is the existing flag scalar, not a digit or sum. Report word leakage, format-relative error, native-factor uncertainty, and apparent breakpoint coverage. No threshold changes or model score follow automatically; a deployable hand construction would need a separate declaration.


### A274 recorded native Gates lack stable literal breakpoints (2026-11-11 07:27:01 CET)

Diagnostic completed12.28seconds, peak5,444,416KiB, no candidate. Format flag11.8064–14.6083 and original common programme feature556.415–853.761. The retained native Gate factors are mostly inferable (word99.9188%,format99.9963% stable division), but no simple native hinge coverage emerges: best breakpoint-plus-variation choices have maximum equivalent digit residual1.12–2.45digits across format holdbacks. Several near-threshold Gates approachzero, so saturation does not remove this problem. These are geometry diagnostics, not a theorem excluding every native-feature construction.

The five whole-format Up fields reproduce the flag with maximum relative error3.32e-5down to1.99e-6, but word response RMS remains0.01349–0.01527 versus0.01867for the physical flag. Median native feature word-to-format leakage isabout1.28e-4 at the bestwordfield, too large to assume silent reuse under million-scale output gains. No weight or native activation is changed or replayed. No meter unit used.

Next unimplemented idea: inspect whether a signed linear factor of original HOST raw-energy loss can use the existing final anchor more accurately than prior square-root target fits. This would be a different fixed covariance/PCA construction, not a refit of A141/A216/A255/A267candidate residuals. No A275declaration or code yet.


### A275 existing-anchor spectral energy completion declared (2026-11-11 07:30:31 CET)

A77 already tested PCA/trace carriers in newly silenced ordinary coordinates. A275 instead uses only the existing programme norm-anchor coordinate439, retaining its −0.1311084 native output gain. It does not silence new coordinates. As in A141, reconstruct displaced native features from original HOST using the existing512donors/ridge0.001; independently fit96strong rows from original HOST numerators at original HOST RMS with ridges[1e-6,1e-5,1e-4,0.001,0.01], choosing by native strong-row validation RMS. Ordinary gains return to HOST. All programme functions/columns stay fixed.

Form the positive semidefinite raw-energy transport matrix from original donor-reconstructed HOST rows:96attenuated strong rows plus the existing anchor row. Its leading covariance eigenfactor is a signed linear function of the existing native features. Orient it by correlation with the original anchor and analytically trace-scale it. Compare exactly five mixtures[0,0.25,0.5,0.75,1] with the original donor-reconstructed anchor; each mixture receives one analytical scale to match the original HOST native energy remainder. Reject any negative remainder instead of clipping. No nonlinear target fit or candidate residual refit.

Use256oldwordwindows for construction,128old plus128fresh for validation, with A272's strengthened selector/native-capture provenance. Require native branch RMS at least5%better than A142, old full-vocabulary matches nondecreasing, fresh gain>=8 and totalgain>=16/16,384, all2,560format outputs correct withmargin>1, and caps. Installation would be on A259 after final-weight identity checks and still require full scoring. Calibration splits were reused in earlier experiments. No A275result or checkpoint yet.


### A275 spectral existing-anchor completion rejected (2026-11-11 07:37:19 CET)

The leading transport factor contains96.4128%of the97-row transport covariance, with analytic trace scale1.018433525. All original HOST energy remainders are nonnegative. Nevertheless allfive mixture trials worsen native branch RMS:4.4256993atfraction0 to4.3897141atfraction1, versus3.3557402baseline. Relative raw-RMS error atfraction1 is0.00880622. These do not meet5%native improvement, so no full-vocabulary follow-up or checkpoint. Nominal4,979,682edits,max100,000Down. Wall17.14seconds, peak4,504,532KiB. The large first-mode energy share does not establish native output preservation.

### A276 full-residual cleanup feasibility declared (2026-11-11 07:37:19 CET)

Geometry diagnostic only. A70/A181 already found that lowering emission gains fails with their incomplete cleanup. A276 changes the cleanup subspace: compress the full640-dimensional recorded pre-final residual identity, conditioned on the existing flag and digit registers, using covariance PCA on96formatsequences. No operand-pair answer is a target. Keep the exact flag/digit two-dimensional component and consider residual ranks[6,8,12,16,24]. Remaining416formatsequences are local holdbacks, reused elsewhere.

Construct one full-coordinate two-group literal curve from protected unembedding/norm and the existing vocabulary metric, bias2, excluding the norm-anchor coordinate439. Use physical emission amplitudes[256,512,1024,2048,4096]. Boost all currently programmed early-layer output coordinates with gain>1000; allow[0,8,16,32,64] additional coordinates ranked once by training cancellation load divided by HOST gain. All selected gains are1e6, other gains HOST; require fewer than96boosted coordinates for a promising reduced-gain geometry.

For each compressed identity and literal amplitude, calculate required raw branch coordinates (desired emission minus compressed residual)/gain. Check whether the existing anchor can fill the remaining normalized energy640 on every recorded input. Compare (a) exact per-input square-root anchor, explicitly an unimplementable oracle, (b) one constant anchor matching training mean energy, and (c) its first-order quadratic Taylor expression. The Taylor expression is algebraic in the proposed linear/hinge features, not a native activation or fitted answer. Report margins, residual/norm errors and missing feasibility; no model, native arithmetic branch, weights, or rung claim. A future actual hand circuit and original-HOST native reconstruction would need separate declarations and full scoring.


### A276 compressed cleanup fails normalized-energy capacity (2026-11-11 07:42:51 CET)

The full-coordinate literal curve passes31literal checks with minimum normalized margin2.81217575. PCA residual ranks6/8/12/16/24 have holdback identity-compression RMS739.27/309.55/96.65/84.99/71.39, showing a substantial improvement beyond rank8. However, none of125gain-layout/amplitude/rank cases has sufficient raw norm capacity. Even the best minimum energy room is−1016.397 (rank24,amplitude512,94boosted rows), and every2,560position is infeasible there. No anchor/readout follow-up or geometry artifact; wall9.84seconds,peak1,810,004KiB. No checkpoint or meter.

### A277 analytic amplitude and ordinary-gain capacity declared (2026-11-11 07:42:51 CET)

Keep A276's full-identity compression, fixed full-coordinate literal curve, ranks[6,8,12,16,24],30existing scratch rows and additional counts[0,8,16,32,64]. Exclude anchor439 from the cancellation identity. Replace its arbitrary amplitude grid by one analytic sufficient lower bound per rank: every protected-vocabulary class margin must exceed the norm triangle bound for a normalized margin1. The bound uses frozen embeddings and recorded residual identity errors, with a fixed1.001safety factor, and is not an answer fit. If a literal direction lacks a positive bound denominator at any observed register, reject that rank's certificate.

Scale ordinary non-anchor HOST post-gains by[1,2,4,8,16], leaving selected scratch gains1e6 and anchor gain HOST. This new capacity change would require a separately derived native-energy reconstruction; it does not preserve text by itself. For each rank and ordinary factor, retain only the smallest additional-row count with positive anchor energy on every input and fewer than96boosted rows. Evaluate exact per-input oracle, constant and quadratic-Taylor anchors as in A276, reporting all energy checks. No extra amplitude trials. No native arithmetic branch, actual hand-circuit allocation or checkpoint is claimed. A277 will run alone after static review.


### A278 ordinary-gain expansion with original-HOST norm envelope declared (2026-11-11 07:44:53 CET)

Independent of A277's future hand-circuit allocation, test expanded ordinary gains on the existing11-unit/96-coordinate writer. Keep all programme G/Up/Down columns and96writer gains fixed. Set only ordinary non-anchor post-gains to HOSTgain times[1,2,4,8,16]; keep anchor439's existing small gain. This is a partial gain accommodation, not an entire tensor proportional to HOST. Reconstruct original HOST displaced features with512donors/ridge0.001. A single original HOST normalized strong-row fit uses[1e-6,1e-5,1e-4,0.001,0.01]; each factor scales that fixed original-target numerator analytically.

Target rawRMS=factor*originalHOSTrawRMS. The existing anchor's squared energy is factor²*HOSTenergy minus all non-anchor raw energy. Reject negative remainders. Two fixed sign representations are compared: the original HOST anchor's global mean sign (A141control) and its per-token sign. Each original energy target is projected once onto all surviving native features, with the same five ridges and one analytic trace normalization. This is distinct from A275's spectral approximation and does not reuse its fitted rows or residuals.

Select the lowest native-branch RMS candidate per gain factor, requiring at least5%improvement over A142 and caps. Then require old full-vocabulary matches nondecreasing, fresh>=8extra and combined>=16/16,384, all2,560format outputs correct withmargin>1. Use the same256oldtraining/128old+128freshvalidation and strengthened provenance as A275. Existing arithmetic native features are observed and unchanged; fullscore is still needed for transfer to A259. No A278code/result/checkpoint at this declaration; prepare while A277 runs and execute sequentially.


### A277 reduced-gain cleanup geometry passes with a quadratic anchor (2026-11-11 07:47:43 CET)

Six abstract geometries pass all2,560recorded register outputs with normalized margin>1: residualranks12/16/24, ordinaryHOSTgainfactors8/16, all using only the30existing scratch coordinates at1e6and no additional boosted rows. The smallest rank12 requires analytically bounded emission amplitude44,402.75973698. Atfactor8, anchor room444.925–585.709, Taylor maximum raw-norm error0.003712825, minimum margin1.25623322; atfactor16, room588.591–624.170, normerror0.0002091101, margin1.19467163. Both have2,560correct including2,080PCAholdback positions. The amplitude bound itself uses allrecorded positions, so these are not globally unseen holdbacks.

Constant anchors remain poor (rank12/factor8 only36correct); the quadratic variation is essential. Rank6/8 have much larger identity residuals and no accepted Taylor geometry. Larger ranks16/24also pass but require more arithmetic features. These remain oracle/hand-algebra geometries: no Gate/Up realization, unit allocation, native arithmetic branch or text-preservation claim. A future quadratic circuit must account for all of these. Artifact artifacts/a277v1_analytic_amplitude_cleanup_geometry.pt contains eligible geometries and provenance. Wall2:42.94,peak2,121,776KiB.

A278 is reviewed and syntax checked; it now runs as the sole heavy process. It tests the ordinary-gain/norm-envelope mechanism independently on the existing11-unit writer before any A277allocation. No official score is running;20fullremain.


### A278 ordinary-gain expansion rejected by native/text checks (2026-11-11 07:54:46 CET)

The factor1per-tokenHOST-sign envelope (ridge1e-5) improves native branch RMS3.3557402→2.7153244 and raw-RMS error0.00480840, but full-vocabulary counts fall8023→8022old and8003→7991fresh. It fails both prediction guards. Expanded factors2/4/8/16 have best native RMS7.96157/9.55944/9.97939/10.08579 and raw-RMS errors0.01913–0.02371, so none reaches text evaluation. No candidate/PT/checkpoint. Wall1:06.16,peak5,159,448KiB.

A277's successful ordinaryfactor8/16arithmetic geometry therefore lacks a plausible preservation mechanism under this norm-envelope construction. Its unimplemented quadratic-anchor circuit could factor the energy of linear fields plus three hinges into centered linear-field squares and hinge-times-linear terms; shifted positive square Gates and linear/constant compensations would use only hand GLU algebra. However, the required additional units and poor expanded-gain native reconstruction remain unresolved, so do not allocate that full circuit now.

### A279 gain-aware literal emission with retained ordinary HOST gains declared (2026-11-11 07:54:46 CET)

New geometry only, no checkpoint. Retain A277's30early scratch coordinates at1e6and add[16,24,32,48,64]coordinates from the original96writer ordering after excluding those scratch rows (totals46/54/62/78/94). All ordinary gains remain HOST. Compress only the selected boosted-coordinate residual identity using the fixed flag/digit component plus exactly6PCAcomponents, so the existing11programme units suffice for its linear/emission part. Remaining ordinary-coordinate residuals are left as nuisance, not forced into unavailable norm capacity.

Derive full640coordinate literal directions, excluding anchor439, from the original vocabulary metric plus the fixed A26632competitor metric withmass0.1. Add a raw-energy penalty diag(1/gain²), normalized to the vocabulary metric's mean diagonal, with five masses[0,0.01,0.1,1,10]. This encourages emission in high-gain coordinates while permitting small ordinary-coordinate contributions to separate confusable tokens. Group bias2 and existing five literal functions remain fixed. Per layout select the smallest maximum literal raw energy among geometries with full-vocabulary margin>1 on31literal checks.

For that geometry derive the same analytical sufficient emission amplitude from all2,560recorded residual/register states (margin1,safety1.001). Require positive raw-energy room and<96boosted coordinates. Evaluate exact-anchor oracle, constant and quadratic-Taylor anchors. Save any allcorrect/margin>1geometry, clearly distinguishing constant implementations from those needing extra quadratic features. Original96pool data and reused calibration are only local geometry/provenance, never whole-answer fitting. Native arithmetic contributions and native text reconstruction remain mandatory future work.


### A279 soft raw-energy penalty is insufficient (2026-11-11 07:59:08 CET)

Allfive gain layouts admit31/31literal outputs with margin>1, but no complete geometry has positive anchor energy at its certified physical amplitude. Selected raw-energy penalties are1for46boosted coordinates and10for54/62/78/94. Required emission amplitudes0.953M–2.802M leave too much ordinary-coordinate raw energy; minimum remaining anchor energy is−68,365.5even for the94-coordinate selected case. All2,560positions are infeasible in every selected layout. No readout/anchor follow-up, geometryPT, checkpoint or meter. Wall1:34.24,peak1,959,912KiB.

Folder size39,244,949,772bytes at the last check; below40GB.

### A280 hard writer subspace with full boosted-coordinate cleanup declared (2026-11-11 07:59:08 CET)

Use A279's same46/54/62/78/94coordinate pools (30mandatoryscratch plus originalwriter ordering) and six cleanup PCs, but constrain every literal emission coefficient outside that pool toexactlyzero. This is a new hard subspace constraint, not a sixth value of A279's soft penalty family. Ordinary HOST gains stay unchanged. Per-pool frozen-vocabulary metric solves use the five competitor penalties[0,0.1,1,10,100], fixed groupbias2; select greatest literal normalized margin among31point checks, requiremargin>1.

Then use A279's unchanged analytic sufficient physical amplitude (margin1,safety1.001) and identity cleanup, and check exact-oracle/constant/quadratic anchors. Prefer a constant-anchor geometry if it meets all2,560format winner/margin guards, because its five emitter plus six cleanup functions can occupy exactly the existing11units without new native feature displacement. No native contributions or text-preservation claim at this stage; later original-HOST reconstruction, actual Gate/Up realization and fullscore remain required.


### A280 hard subspace:94-coordinate geometry passes only with Taylor anchor (2026-11-11 08:04:09 CET)

Pools46/54/62/78fail literal vocabulary margins (best−7.76692/−4.62759/−2.79828/−1.44376). Pool94, competitorpenalty0.1, passes31literal points withmargin1.35069847. Its analytic emission amplitude is2,553,548.1948; anchor energy room stayspositive(min629.8679). Exact-oracle and Tayloranchors preserve all2,560register outputs with minimum margins1.02074814/1.02082062. Constantanchor also preserves allwinners but minimum margin0.92851257fails the declared>1screen. No constant-anchor eligibility override or checkpoint. Wall28.89seconds,peak1,939,868KiB.

### A281 literal-only norm-anchor interpolation declared (2026-11-11 08:04:09 CET)

Geometry-only extension of A280's94-coordinate construction. Approximate the required norm anchor using the existing five literal functions, without adding units. Targets are an explicit11-entry table on literaldigits0–9andnewline−1: sqrt(mean_anchor_room + mean_observed_emission_energy − literal_emission_energy(d)). The two means are fixed A280 activation-energy statistics; no sum/operand target or whole-problem functional regression is used. One full-rank least-squares projection of that per-digit table onto the existing five hand functions has no ridge/scale search. Check its literal approximation and all2,560recorded format winners against the originalconstant/Taylor controls.

Use exactly the existing A280amplitude,pool,cleanup and nativeHOSTordinarygains. Require allwinners and minimum margin>1, positive anchorvalues and finite coefficients. This is not an actualGate/Up/native reconstruction or model checkpoint. If it passes, A282 may realize the existing11features with full boosted-coordinate cleanup and rebuild native Down once from originalHOST; such a build still needs separate checks.


### A281 literal anchor remains below the local margin screen (2026-11-11 08:07:03 CET)

All2,560recorded outputs remain correct, minimum margin0.93249893, PCAholdbackminimum1.17943573. The literal-table approximation error is0.03839404 and maximum raw-norm error0.00290403. This slightly improves the constantanchor's0.92851257 but fails the declared>1guard. No geometryPT/checkpoint. Wall14.65seconds,peak1,277,404KiB.

### A282 exploratory native check of the11-unit full-cleanup writer declared (2026-11-11 08:07:03 CET)

Explicit screening override: preserve A280/A281's failed>1margin results, but allow one separately named native reconstruction of A281's single existing94-coordinate geometry with a local format floor0.9. There are no new anchor/geometry trials. This does not claim either prior screen passed or that H1passes. All2,560format winners remain required after actual hand features and native contributions are included, and the final whole-model gate is unchanged.

Retain the original11programme units. FirstfiveGates/Up functions remain the old literal functions (apart from deterministic homogeneous cap balancing). Reuse the six cleanup slots with the original common Gate and Up fields implementing A280's six boosted-coordinate identity PCs; physicalflag/digit cleanup folds into the firsttwo Down columns. Anchor439 receives A281's five literal coefficients. Use one declared common raw multiplier1e6, with per-unit minimum power-of-two Up/Down balancing to respect1e6caps. New cleanup features are explicit saturated-positive hand products; native features remain metered observations. Check exact algebraic identity and negative programme Gates on allknown words/prefills.

Independently rebuild displaced native behavior from original HOST using immutable NativeFinalReconstruction, the same11units and512donors/ridge0.001,94strongcoordinates and five strong-row ridges[1e-6,1e-5,1e-4,0.001,0.01]. Require>=2%native branch RMS improvement, old full-vocabulary matches nondecreasing, fresh>=8extra and combined>=16/16,384, all2,560actual-local format winners withmargin>0.9, plus caps. Originalold256training/128old+128freshholdbacks and strengthened provenance remain. InstallationwouldbeA259afterfinal-weightidentitychecks and wouldrequirefullscore. No A282code/result/checkpoint at declaration.


### A282 full-cleanup writer rejected by native text predictions (2026-11-11 08:21:25 CET)

The actual11-unit/94-coordinate construction improves native branch RMS3.355740197→3.152729667 (6.05%) and meets weight/edit caps at4,975,650nominal edits. Its hand-identity error is8.66e-15, float32input-normalization shift8.67e-7, known word/prefill Gates remain below−512.0045, and positive cleanup Gates exceed61,994.8. Native reconstruction selects ridge1e-5. However, full-vocabulary oldholdback matches remain8023/8192 whilefresh matches fall8003→7995/8192. It fails the prediction guards; format follow-up is not triggered. No candidatePT/compiler/checkpoint/fullscore. Wall49.68seconds,peak5,526,584KiB. This preserves the explicit exploratory>0.9local-margin override and A280/A281's original failures; it establishes no rung pass.

### A283 single-pair native-channel deletion diagnostic declared (2026-11-11 08:21:25 CET)

Before allocating another reader, rank native layer3rotary pairs once using original HOST post-norm Q/K channel energies plus raw-normalization energy, as in A167. Restrict to single-cosine literal positional gaps 1−max(cos(omega*d),d=1..8)>=0.02. Select exactly five cheapest pairs by that fixed original-HOST activation statistic, without using candidate outputs for placement.

For each selected pair, evaluate two analytic controls: delete its two Q/K rows with all remaining HOST weights/gains unchanged; then apply one shared ordinary-Q/ordinary-K numerator accommodation fitted once to original same-input HOST normalized numerators. Retain zero rows in both controls. Compare paired-original-HOST and same-input causal centered QK errors on64heldback calibration windows, using exactly the current pre-layer3inputs and paired originalHOST inputs reconstructed from existing captures. A259's reader is the reference. Require both errors<=1.05timesA259 for a promising control. These controls omit programme leakage and are diagnostic only, not a mathematical lower bound or a realizable addition circuit. No softmax/model/nativeactivation replay, source fit, checkpoint or meter is involved. A later token-selective key and ordinary-HOST-query construction would require a separate declaration.


### A283 five single-pair controls rejected (2026-11-11 08:25:23 CET)

Original-HOST energy ranking selects pairs[9,15,4,17,21]. All ten deletion/accommodation controls fail the1.05timesA259guards. Best deletion ispair9: pairedcausalMSE0.00431461630 andsame-inputMSE0.00371693744, versusA259's0.00142456275/0.00083321329. Its shared-gain accommodation is slightlyworse(0.00432703390/0.00373179765). Unmodified nativeQ/K oncurrentinputs yieldpairedMSE0.000597281734 andsame-input0. This identifies substantial localcost even before adding a programme, but is not an impossibility bound. No checkpoint/meter. Wall23.22seconds,peak2,707,260KiB.

### A284 original-native causal-score repair of the cheapest pair control declared (2026-11-11 08:25:23 CET)

Test only A283's fixed first-rankedpair9, with both nativeQ/K rowszero and every other Krow/gain HOST. A single closed-form reconstruction may place the lost native score contribution in surviving Qrows. This is a distinct repair of the original HOST channel deletion, not an iterative refit of an A283candidate. Target original same-inputHOST scores at the current exactlyrecorded pre-layer3inputs. Keep the deleted-pair rawqueryRMS as the fixed linearization normalizer. There is no arithmetic target or programme yet.

Construct ordinary-query output directions from the training causal-key covariance and its original-score cross moment: one global covariance inverse withridge0.001, then32PCAoutput directions. Construct64canonical input directions from the resulting original moment targets usingwhiteningridge0.001. Fit one exact centered-causal native-score normal equation in these64x32coefficients perhead. Trialmetricridges[1e-6,1e-4,0.01,1,100] andstrengths[0.25,0.5,0.75,1,1.25]; alltargets/bases/normalizers arefixed before anytrial, and noselected residual is reused. Queryrowsforpair9remainzero.

Use192training/64holdback wordwindows. Select actualrounded same-input causalMSE independentlyperhead, with pairedHOSTMSE atmostthe deletioncontrol; require the combined paired andsame-input errors both<=0.95timesA259 before a future reader is promising. Report failures unchanged. This programme-free diagnostic may save querygeometry ifeligible but makes no source-routing, attention-output, fulltext or rung claim. It uses only linear/RMS/RoPE/covariance algebra, no softmax/nativeactivation/model execution. Any token-selective key/programme construction still requires a separate declaration.


### A284 causal repair cannot recover the deleted native pair (2026-11-11 08:35:33 CET)

The single original-target64-input/32-output repair reduces thepair9deletion control's pairedMSE0.00431461630→0.00429006578 andsame-inputMSE0.00371693744→0.00369039555, gains0.57%/0.71%. Both remain well above A259's0.00142456275/0.00083321329 and fail the0.95leader guard. Allheads choose metricridge100,strengths[1.25,0.5,0.5,0.75]. Independent full/reduced moment and covariance identities agree within1.6e-17. No geometryPT/checkpoint/meter. Wall30.43seconds,peak2,707,416KiB. The single-pair deletion route is set aside after both direct and repaired controls fail.

### A285 current-leader final-MLP restoration diagnostic declared (2026-11-11 08:35:33 CET)

Build a diagnostic checkpoint from A259 by restoring exactly the layer17MLP Gate,Up,Down andpost-feedforward norm tensors to frozenHOST. All other tensorvalues remainbit-identicalA259. This deliberately removes the arithmetic emitter; it is not an addition candidate or a rung advancement. It tests the finalblock's aggregate text contribution on the current upstreamtrajectory, which is not exactly represented by the older A142final captures. No fitting, scale values or new nativeactivation calculations are involved.

Use one normal full tools/score.py run with H5, whose5,000arithmetic inputs are smaller than H1's25,000. The purpose is the standard identical2,048-window text comparison, with arithmetic/two-problem/noise aggregates retained honestly even though arithmetic isexpectedtofail. This is a diagnostic choice of scorer workload, not an attempt to skip the ladder or bank H5. No heldouttext is inspected. It consumes the21stfullscore and leaves19full,0quick,0forward. Do not infer causally additive block costs or text preservation for any unbuilt circuit from this ablation.


### A285 build and meter start (2026-11-11 08:41:38 CET)

Diagnosticcheckpoint experiments/a285v1_native_final_diagnostic/delta.pt isbuilt atomically with4,482,860edited scalars,max1e6; only the fourdeclared finalMLP tensors were restored. Builderwall9.32seconds,peak907,680KiB. Its full H5/text score isrunning as the soleheavyprocess;19fullremain after reservation. No partialscore or rungpass is claimed. All construction sources are nowimmutable.

### A286 dense native-key source directions with joint query-energy accommodation declared (2026-11-11 08:41:38 CET)

Prepare while A285scores; run onlyafteritfinishes. Retain allnative K3/V3/O3 and the rest ofA259. Reconstruct Q3independently from HOST, the existing quietformatquery, and new source-position-only directions. Unlike A45's fixed64programmechannels, use all256channels and a diagonal native-query-energy penalty within the source-direction solve. No nativepair isdeleted. Trainingformat means/covariance use96current+1024historicalwindows;32current+512historical remain directionholdbacks, though allsourcewindows enter the eventual routingbound.

For each source penalty[0,0.01,0.1,1,10], solve the exact9position-mean constraints with metric within-position keycovariance divided byitsmean diagonal, plus penalty timesnormalized HOSTrawquery energy diagonal, plusfixed1e-6identity. Targets are the four sourcepositions[3,2,1,0], never sums or answers. Reject anydirection withnonpositive sourcegap. Use A61's analytic fourth-root sharedgainprofile, HOSTgainfloor andfinitebreakpoint construction with asymptoticmargintrials[8.25,8.5,9,10,12]. Ordinary HOSTqueries receive the existing format-mean subtraction andinversegainaccommodation.

Compute initial programme amplitudes by the exact quadratic margin8bound. Thenone original-HOST rawquery squared-energy solve scales ALL256rawqueryrows jointly, usingridges[1e-6,1e-4,0.01,1,100] andreciprocal gains to preserve native/programme numerators. This includes small programme components in everyrow rather than excluding the old64selectedrows. Finally recompute the analytic source amplitude once; there is no subsequentgain/energy refit or reusedcandidate residual. Reject nonpositive squaredscales, inadequate asymptoticrouting or caps.

Compare exact float32weight/linear/RMS/RoPE predictions on64heldback wordwindows to both same-inputHOST andpaired originalHOST. Require both centeredcausalMSEs at least10%belowA259, pairedqueryRMS<=1.05baseline, all1,664sourcewindowsmargin>=7.999,and edit/valuecaps. Originalbaselines/provenance mustreproduce. An eligiblegeometry canbesaved for a separatelychecked compiler/fullscore; no attentionprobabilities,newnativeactivation/model,whole-answerfit or iterativeoptimization. This is not a metered candidate yet.


### A287 joint projected-output and winner-interval anchor geometry declared (2026-11-11 08:51:33 CET)

Prepare a geometry-only diagnostic after A286, with no checkpoint or fitting yet. A258 varied only the radialanchor while holding strongoutput numeratorsfixed; A270 matched HOSTcoordinate rays by anL2complement objective. A287 instead derives the96strong outputs jointly with a physicallyfeasible radial factor in the original HOST vocabulary-boundary metric. Use the original A123 recorded nativefeatures/map, original pairedHOST finalstates/top8, and original A123top8rivals. A142 is only the local prediction comparator; its fitted residual isnot a target. Set the existinganchor439outputgain tozero explicitly; preserve allother gains and allordinary rawrows.

Training384windows defines the HOSTwinner-minus-runnerup metric weighted by1/(gap+0.1), mean-normalized andcapped100. Fixed projectionridge0.01times the metric's mean diagonal. Letu=originalrawRMS/newrawRMS, andL be the original HOSTstate scaled to originalA123finalRMS. The boundary-metric projection into the96writer coordinates makes the ideal finalstate affine in u: x(u)=A+uB. Outside96rows/anchor it is the originalbefore+u*branch; anchoroutputiszero. This is exact projection algebra, no derivative, gradient or finite-difference fit.

Derive the quadratic rawenergy remainder for anchor439 exactly after substituting those96rawtargets. Intersect its positiveenergy domain withu in[0.8,4/3] and all fixed-rival HOSTwinning inequalities. Trial HOSTmarginfractions[0,0.125,0.25,0.5,1]. Choose the feasibleu closestto1, takingone representablestep into a boundary atfraction0; if the winninginterval isempty, retain the physicallynearestu to1 andreportthe failure. Preserve the original rawanchor sign. Verify the energy identity and resulting affine state against exact RMS algebra.

Report original/A142 baselines and each oracle's restricted/full-vocabulary prediction counts on the reused128freshvalidationwindows, scalarfactor/energy ranges, infeasible counts and state errors. ValidationHOST targets make these optimistic oracles, never installed predictions. Save only the training97-row targets if any geometry improves the validation restricted count by at least16overA142 and has finitephysicaltargets; any later one-shot nativefeature regression and actual-format/fullscore mustbe separatelydeclared. This examines whether joint geometry even has headroom; no nativeactivation/model/attention, arithmeticlabels, wholeproblemfit or weight checkpoint.


### A285 official native-final diagnostic: text improves but remains below gate (2026-11-11 09:02:38 CET)

Fullrun0803-score completed at scores/a285v1_diagnostic_full_H5.json. SHA784afb38f2e403582794938aab755888be8c1d22af1884ea8d7a9b6bdb1dd6c5;4,482,860edits,max1e6. Standard textagreement98.21319580078125%,KL0.0018323514207451688,CEchange−0.00022713015492836774,finite. This is exactly256more matchedpredictions/131,072thanA259 (+0.1953125percentagepoints), but still0.2868041992percentagepoints below98.5%. Restoring the finalMLP alone doesnot meet the preservationgate; this is an intervention-specific result, not a causal decomposition that can beadded acrossblocks.

The deliberately emitter-free diagnostic hasH5zeroexact/zero digit-correct outof5,000. Allthreenoiselevels are0/2,000exactanddigits; both repeatedproblems andjointcountare0/2,000. H5false; no rungbanking. Wall24:32.54,peak3,841,788KiB. Twenty-onefullscoresnowcomplete(20H1candidate scores+thisH5diagnostic);19fullremain,0forward/quick. A259remains the addition-qualified leader. Finalverification/export remainunrun.

A286 andA287 havepassed static syntax/review and areunrun. A286 nowstarts as the soleheavyprocess; A287's independent joint geometry follows afterinspection. No newscorer budget isreserved.


### A286 joint all-row energy solve destroys source capacity (2026-11-11 09:06:12 CET)

Sourcepenalties0/0.01/0.1/1 givepositive gaps for all fourpositions; penalty10 loses the secondsource (minimum−0.000682929). All100gain/energy trials from the fourviabledirections fail before a text candidate:90lose asymptoticroutingmargin8 and10have nonpositive squaredrowscales. No candidate/PT/checkpoint. Exactmeanconstraints agreewithin5.9e-15. Wall28.62seconds,peak4,212,384KiB.

The jointall-row energy solve mayrestore the veryprogramme-channel energy reduction needed for routing; no causal-score quality wasreached. A possible separate construction could bound one globalenergy-solve amplitude by exact source asymptotic inequalities, or restrictenergy repair to HOST-floor channels. Neither isimplemented or authorized asanundeclaredextra trial here. The full failedA286 result remains. A287 nowruns as the soleheavyprocess.

### A288 earlier emitter atlayer9: transfer geometry and downstream bound declared (2026-11-11 09:06:12 CET)

Prepare a genuinelydifferent architecture diagnostic after A287: emit the digit/newline after MLP9, then restore MLP17 toHOST. Layers10–17 wouldremain whollynative. An emission afterMLP9 doesnot change layer9's own cachedkeys, or any earlierlayer input, so next-token arithmetic through8/attention9 has the same dependencystructure. Nevertheless the A123formatcaptures and A176native9wordcaptures are transfer surrogates for A259's changedreader and are not claimed exactA259states.

Reconstruct exactA123 pre-MLP9word/format states by adding the recorded normalizedbranches; verify finalrecordedstates. Transfer the existing11hand emitter G/Up functions from17to9 by the exact pre-normgain ratio, preserving their physicalresidual linearforms. Use the existing96literaloutputcoordinates atgain1e6, nativeHOST9gains elsewhere, and retain the former weakanchor439gain. Keepcleanup/anchor Downparts fixed; test literal-emission multipliers[1,2,4,8,16] only. Minimalpower-of-two Up/Down balancing enforces1e6caps without changing the intended handproducts. Check programme Gates on allknownword/prefillinputs, explicitReLUidealfeatures and a uniform0.25times|Up|hinge-errorbound. No nativeGELU is evaluated.

For each geometry inspect the individualdigit/negative-one-newline register, never a+b labels. Compute complete protected-vocabulary margins immediatelyafter the handprogramme. Bound every laternative residualbranch independently: for any readoutdifferenceD, |D·branch|<=sqrt640*||HOSTpostgain*D||. Sum the16branches fromlayers10–17, withfixedrelative1e-5numericalguard; the bound is independent of futureattention probabilities or caches. Require allknown word/prefill Gates<-16 and all640recordedformat outputpositions retainpositive rawmargin after this bound for a promising transferred geometry. NativeMLP9contributions and unit allocation are deliberately omitted here, so even a positive result isnot a full arithmetic/text certificate or checkpoint. Later native9reconstruction must include a conservative bound on its actual nativecoefficients and an official fullscore. No newerreader sourceweights are reused or altered by this diagnostic.


### A287 joint geometry has optimistic prediction headroom (2026-11-11 09:10:10 CET)

Allfive joint geometries arephysicallyfeasible. OriginalA123/A142 fullvocabulary baselines reproduce7,990/8,003of8,192. Marginfractions[0,0.125,0.25,0.5,1] givefulloracle matches[8,076,8,152,8,148,8,133,8,042]. The best0.125geometry has149morefull-vocabulary winners thanA142, with41empty validationwinner intervals; its other fallbackstates remainphysicallyvalid. Fraction0's restricted8,121falls to8,076underfloat32fullreadout, so boundaryties mustnot be mistaken for reliable margins. RawRMS movement isatmost0.2499855, exactRMSfactoridentityerror4.44e-16, affine-stateerror<=1.10e-11. Peak4,691,876KiB. Onlytraining97-row targets are saved in artifacts/a287v1_joint_projected_anchor_geometry.pt; validationtargets are oracle evidence, not deployable predictions. No checkpoint or meter.

### A289 one-shot realization of A287 joint training targets declared (2026-11-11 09:10:10 CET)

Test whether the jointgeometry canbeexpressed by existingnative finalfeatures. Use A287's fixedfive trainingtarget arrays, originalA123captured nativefeatures andrawRMS,384training/128reusedfreshvalidationwindows. Fit exactly the96strongrows plus existinganchor439 acrossall2,037retainednativefeatures. Ordinaryrows and all11programmecolumns remainA123/A259bit-exact; anchorpostgain becomeszero, othergains remainfixed. Start from originalHOSTDown (strongrows analyticallyscaled byHOSTgain/1e6, anchorrowHOSTunscaled), not a previously selectedcandidate residual.

One shared original-feature covariance withthe same fixedHOSTgapweight asA287 supports independent directprior-centered ridge solves[1e-6,1e-5,1e-4,0.001,0.01]. Strengths[0.25,0.5,0.75,1,1.25] interpolate thatoriginalHOSTprior and eachindependentfit. No fittedcandidate becomes a target. Select actualrounded-weight restrictedpredictions, requiring>=16extra/8,192overA142, HOSTstate anddirectionerrors<=1.25baseline, rawRMSmovement<0.25,andcaps. The1.25stateerror allowance isexplicit: some A287winner-oracles exceedearlier1.05L2screens, and this fit testsreadoutpreservation rather than silently waiving anoldscreen.

A qualifyingselection mustthenconfirm>=16extraagainstthefullprotectedvocabulary and preserveall2,560recordedformatwinners withminimum margin>1 andrawRMSchange<0.02. Alloutputs are linear/RMS/head arithmetic onmeteredfeatures; no newnativeactivation/model. InstallationonA259 remainsa changed-upstream transfer requiringa fullscore. Only aneligible fitmay save weights. A289 canrunbefore the independentA288transfergeometry onceits sourceisreviewed; neither iscurrentlyrunning.


### A289 joint fit fails prediction screen despite oracle headroom (2026-11-11 09:17:26 CET)

The bestactualrounded fit has8,002restrictedmatches/8,192versus8,003A142baseline, so it fails the>=16extra guard. Selecteddiagnostic ridge0.001,marginfraction0.5,strength0.5; HOSTstateRMS0.0162838077vs0.0157828573, direction0.0150675659vs0.0145631770, maxrawRMSchange0.04852193. Nominaledits4,980,225;maxfittednativeweight0.27900362. No fullvocabulary/formatfollow-up, PT/checkpoint or meter. Wall37.54seconds,peak4,902,376KiB. A287's149-oraclematchgain was not reproduced by the declared one-shot nativefeaturefamily.

A288early-emittergeometry remains declared butunrun; its sourceisbeingprepared. Noheavyprocess isactive atthisentry. A286routing failure andA287/A289geometry/fit distinction remainretained. Nineteenfullscoresremain.


### A288 layer9 transfer has one certified hand geometry (2026-11-11 09:29:02 CET)

Allknown21,504word/prefill inputrows have negative transferredprogramme Gates, maximum−2,569.96768; commonformatGate minimum62,894.27985. Gateweightsmax230,379.15625; transferredUpmax0.04275727. Selectorregisters range−1.00435to9.254997 withmaximumdistance0.254997toaninteger; targets remainindividualregisterdigits/newline.

Emissionmultiplier2 alone passes all640positions after the16future-native-branch bound andexplicituniform handfeature-error bound. Minimum localrawmargin859,160.0445; afterfuturebound491,440.1922; afterfuture+handbounds469,769.3956. Multiplier1fails six certifiedpositions;4failsone;8/16have5/42wronglocaloutputs and39/65uncertifiedpositions. Caps hold for everytrial. Exactrecombination ofboth recordedfinalstates isbit-identical. Wall30.46seconds,peak3,994,448KiB. artifacts/a288v1_early_emitter_geometry.pt contains onlythe eligiblemass2geometry. NativeMLP9contribution/unitallocation/textcost remainunresolved.

A read-only HOSTnorm inspection finds layer9anchor439gain44.5, whereas coordinate469has the smallest absoluteHOST9gainoutside the96pool,0.09765625. Forcing439to the oldfinalweakgain would discard a substantial nativecoordinate, so the next allocator will move the purelyraw programmeanchor to469, selectedoncefromHOSTweights. No alternateanchor search or modeloutput comparison isused.

### A290 layer9 native allocation and conservative arithmetic bound declared (2026-11-11 09:29:02 CET)

Realize A288's singlemass2geometry, restoring the fourfinalMLP17tensors toHOST on A259. Relocate only the rawprogrammeanchor439→469; pool/cleanup/literalgeometryotherwiseunchanged. Use one fixedcommonprogrammeUp multiplier1024 to suppress the unknownnativeMLP9raw contribution; this multiplies everyprogrammefeature equally and preserves ideal normalizedoutput. Gates stayunchanged. ProgrammeDown staysatits existingrawscale, atmost100,000. NativeG/Up outside11allocatedslots remainHOST.

Use the certified A174native9features on64wordwindows,48training/16holdback, originalsame-inputHOST9branch targets. Rank nativefeatures once by equallynormalized originalpost-output energy plus rawenergy; take the512highest asfixeddonors. On the other1,536units, one0.001ridge donorprojection yieldsconditional residualvariances. Fiveallocationmasses[0,0.01,0.1,1,10] combine normalized conditionalpost/raw costs to select11programmeunits. Eachuniqueallocation independently calls the immutable NativeFinalReconstruction with512donors/ridge0.001 andstrong-rowridges[1e-6,1e-5,1e-4,0.001,0.01]. Poolgains1e6; ordinarygains receive itsone original-target numerator accommodation, including the weaknewanchor. No previouscandidate residual isrefitted.

Require native9branchrelativeRMS<=0.015, paired-originalHOSTstate anddirectionerrors<=1.05of the untouchednative9 baseline at the sameA176input, allknownword/prefillGates<-16, andedit/valuecaps after restoring17and adding9. Thenbound actualnative9rawfeatures by|GELU(g)*up|<=|g*up| using linearHOSTG/Up and the actualcandidate nativeDowncoefficients. Combine thatnative bound with the explicitprogramme0.25|Up|error bound. Compare the relocated/fitted-gain nominalprogramme with A288'soriginalnominalstate, and subtract conservativeglobal protected-readout norm bounds for both itschange andthe combined raw-normalization uncertainty from A288's491,440.1922pure-future margin. Require the resulting lowerbound>0 on all640knownpositions. No nativeactivation/model is evaluated.

Choose lowestnativebranchRMS amongfullyguardedallocations. Any eligibleweights need a separatelychecked sparsecompiler andonefullH1score; no finaltextprediction is available for changedlayer9. The A123format/A176word calibration trajectories areexplicitsurrogatesforA259. No checkpoint or fit exists atthisdeclaration; only thisnativeallocation isnext to run oncecoded/reviewed. Nineteenfullscoresremain.


### A290 native layer9 allocation fails paired-state guards (2026-11-11 09:37:55 CET)

Three unique11-unit allocations arise from five declared masses. Best mass0 (also0.01/0.1) has nativebranchRMS0.6481923222, relative0.01206417394 and4,978,700edits. PairedHOSTstateerror rises0.005415565064→0.006099138813(+12.6%) anddirectionerror0.004242901321→0.005003612513(+17.9%), exceeding the1.05guards. Mass1/10areworse. Therefore the conservative format follow-up was not triggered and noPT/compiler/checkpoint/score wascreated. Wall36.10seconds,peak3,925,128KiB. This doesnotdisprove early emission; it rejects thisspecific native reconstruction.

### A291 dense native-key query with exact programme-energy constraints declared (2026-11-11 09:37:55 CET)

Reuse A286's five sourcepenalties[0,0.01,0.1,1,10], fiveasymptoticmargins[8.25,8.5,9,10,12], originalHOST fourth-root sharedgainprofile andHOSTfloors. The originalHOST raw-query energy target and allsource-position constraints remain unchanged. Unlike A286's unrestricted energyfit, solve one constrained closed-form energyfit in the nullspace of the four per-head programme-energy rows. If standardized coefficients are a and squaredrow corrections are beta=a/standard, impose sum_j programme_direction[h,j]^2*beta[j]=0 for everyhead. The programme isrankone ininput, so these four equalities preserve its rawenergy exactly for everyinput andkeep the asymptotic routingmargin.

For each originalprofile, compute the nullspace once bySVD and project the originalenergy covariance/right-hand side into it. Independently solve ridges[1e-6,1e-4,0.01,1,100]; no earliercandidate residual isused. Requirepositive row squaredscales, reciprocal postgain compensation, finite/capcompliance, actualrounded sourcegaps>=7.999 andbothpaired/same-input causalQKerrors<=0.9A259, pairedqueryRMS<=1.05baseline. Use the same192/64word split andsource-positiontrain/holdbacks asA286; noattentionprobabilities/nativeactivation/modelreplay. NoA291code orresult atdeclaration.


### A291 constrained energy solve rejected (2026-11-11 09:40:40 CET)

All 100 profile/ridge combinations require nonpositive squared row scales. The four programme-energy equalities hold to maximum 7.90e-10, but no realizable candidate reaches the native-score checks. No PT/checkpoint/meter. Wall 22.68 seconds; peak 4,205,244 KiB.

### A292 analytic positivity limit for A291 declared (2026-11-11 09:40:40 CET)

Retain every original A291 target, profile, constraint and five-value grid. For each independent constrained solution beta, apply the single deterministic line limit lambda=min(1,0.99/max(0,-min(beta))); beta becomes lambda*beta. This preserves every homogeneous programme-energy equality and guarantees squared scales >=0.01. No line search, residual refit, new objective or feedback fitting is involved. Recompute only the analytic sufficient source amplitude and evaluate the unchanged native/format/cap guards. This separately records an admissible contraction of a rejected solution, without changing A291.

### A293 layer9 original-energy norm anchor declared (2026-11-11 09:40:40 CET)

Fix A290's first-ranked mass0 allocation and its 512 original donors. Retain the A288 mass2 emitter, anchor relocation439→469, Up boost1024 and final-MLP17 restoration. The original HOST9 raw output supplies an explicit scalar energy target at each unlabeled calibration token: positive sqrt(sum of raw squared values on the96 strong coordinates and weak anchor469). Fit that scalar once from the surviving 2,037 native features, normalized by the original HOST raw RMS, using ridges[1e-6,1e-5,1e-4,0.001,0.01]. The target contains no candidate residual or arithmetic answer.

For each independently fitted anchor, start again from HOST Down plus the single original donor reconstruction, scale strong rows by HOSTgain/1e6, and replace native anchor469 with that fit. Use one original-target ordinary-gain accommodation and five independent strong-row fits at ridges[1e-6,1e-5,1e-4,0.001,0.01], following the existing reconstruction method. The anchor has the smallest absolute HOST9 postgain outside the pool (0.09765625), limiting the native numerator cost of its changed raw value. This adds at most1,525 native repair edits. No iterative energy fit or residual fitting.

Select lowest native branch RMS among candidates meeting the unchanged A290 native relativeRMS<=0.015 and paired state/direction<=1.05baseline guards. Then apply the same conservative all640-position arithmetic bound including native9, hand-feature uncertainty and all16 future native branches. An eligible result needs a separate compiler and full H1 score. No A293 implementation/result at declaration.


### A292 and A293 rejected (2026-11-11 09:45:22 CET)

A292's positivity contraction makes all100 dense-query constructions realizable and preserves source routing, but the best paired/same-input causal score MSE is0.0294472241/0.0287447068 versusA259's0.00142456275/0.000833213286. Best sourcepenalty0.1, asymptotic8.25, energyridge1, analyticcontraction0.0957671; actualminimumroutingmargin8.00000279. No candidatePT. Wall42.61seconds,peak4,205,552KiB.

A293's positive raw-energy anchor worsens native9 preservation. Best anchor/strongridges both1e-6 yield nativebranchRMS0.94307265(relative0.017552495), pairedstate0.00679486034 anddirection0.00530577672. The anchor target's heldback error normalized by HOST raw RMS is0.58273869. All25 fits fail nativeguards; no format follow-up/PT/checkpoint/score. Nominaledits4,980,225. Wall34.87seconds,peak4,058,824KiB.

### A294 independent paired-state repair for the fixed layer9 allocation declared (2026-11-11 09:45:22 CET)

Use A290's fixed mass0 units/donors and unchanged A288 geometry/Upboost/anchorrelocation. This construction has no fitted energy anchor. For native calibration target, mix original same-input HOST9 branch with the original paired-HOST after9 state minus the recorded current pre9 state. Mix values[0,0.25,0.5,0.75,1]. All targets are original unlabeled calibration observations and are fixed before solving.

Build a prior directly from HOST Down: zero displaced units and scale the96 strong rows by HOSTgain/1e6. Its raw RMS is the single fixed normalizer, and ordinary gains get one original-HOST numerator accommodation at that prior. For each targetmix, solve independent standardized ridge regressions toward this original HOST prior, using the512 fixed donors on ordinary outputrows and all2,037 surviving features on strong rows. The same ridge value is used in both systems, trials[1e-6,1e-5,1e-4,0.001,0.01]. No A290/A293 fitted weights/residuals or candidate-dependent target updates are used.

Select lowest paired-state direction error among fits satisfying A290's original nativebranchrelativeRMS<=0.015 and pairedstate/direction<=1.05baseline guards, then require the existing all640-position conservative arithmetic bound and edit/value caps. If no nativeguard passes, retain only a diagnostic report. Any eligible fit needs a separately reviewed compiler/fullscore.


### A294 paired-state reconstruction rejected (2026-11-11 09:48:10 CET)

All25 fits fail the original native guards. The best direction result uses mix0/ridge0.01, nativeRMS0.6484767142, pairedstate0.00612155373 anddirection0.00500026230. Positive pairedtarget mixes worsen preservation, reaching nativeRMS3.66070816 atmix1/ridge0.01. This direct unconstrained post-output target can require large raw changes on weak-gain coordinates; no feasibility of the post-norm target was established. No format follow-up/PT/checkpoint/score. Peak3,930,888KiB.

### A295 exploratory whole-model early-emitter test declared (2026-11-11 09:48:10 CET)

Retain A290's failed1.05paired-state screen. For one separately named exploratory construction only, allow pairedstate anddirection up to1.20baseline. Keep nativebranchrelativeRMS<=0.015, allcaps, OFFgates and the conservative all640-position arithmetic bound unchanged. Use only A290's fixed mass0 allocation and the exact same deterministic original-HOST reconstruction; no new fit family or scale trials.

This override is justified as a direct architecture experiment: A285 measured a0.1953125percentage-point text benefit from removing the final emitter, but local state-error proxies cannot determine the net effect of transferring it to9. A295 may consume one fullH1score only if its conservative native9/future arithmetic bound passes. The official98.5%text requirement and every rung gate stay unchanged. No A295 weights or score at declaration.


A295 pre-meter compiler bookkeeping failure: the inherited A259 manifest omits layer0 category labels. Weight construction and cap checks had succeeded, but the final category-total assertion failed. No meter was called. Missing labels are now taken from the existing A259 index-verified audit; no weights or selection criteria change. Deleted only this unmetered build for deterministic regeneration:

- experiments/a295v1_exploratory_early_emitter/delta.pt: SHA256 2e3e55690720adfb30ebdd5113e4cc2e0bb6299fad986186a6fda1b8a0eb6b0e, 39858017bytes
- experiments/a295v1_exploratory_early_emitter/delta.manifest.json: SHA256 7a5e0bf9aaf52cc2e4ee8ffb72b7f74960dfe60248965630701a9031c5a5324e, 31094bytes
- experiments/a295v1_exploratory_early_emitter/circuit.pt: SHA256 d06621a3312e4424c4bf6afdc680741169dbc4aad69ba0c1c6088500a5b88754, 5821bytes


### A295 conservative bound and build pass (2026-11-11 09:52 CET)

The fixed mass0 construction matches A290's native metrics and meets the explicit1.20 exploratory guard. On all640 recorded register positions, the conservative protected-readout margin remains13,329.0682 after subtracting782.4568 for anchor/gain relocation and477,328.6672 for the global native9/hand uncertainty bound from491,440.1922. Maximum relative raw uncertainty9.7329639e-5. This bound is real-arithmetic with the documented numerical guard and surrogate-input limitation. Analysis29.17seconds,peak3,938,616KiB.

The corrected compiler writes `2e3e55690720adfb30ebdd5113e4cc2e0bb6299fad986186a6fda1b8a0eb6b0e` at `experiments/a295v1_exploratory_early_emitter/delta.pt`, with4,978,700edits andmax1e6. FourMLP9tensors are installed; fourfinalMLP17tensors are restored toHOST; everyotherweight staysA259exact. Purpose labels now reconcile exactly, including inherited layer0 labels from the existingaudit. Build26.02seconds,959,364KiB. One fullH1score is next; no pass is claimed. All construction/transitive sources become immutable at meter use.


### A296 strong-coordinate paired repair declared; queued after A295 fullscore (2026-11-11 09:55:19 CET)

Return to A290's original native reconstruction, with its exact fixed mass0 units,512donors, original donor ridge0.001 and unchanged ordinary-row reconstruction/gains. Only the96 strong-row original targets may receive a paired-HOST state correction. Mix[0,0.25,0.5,0.75,1] between original same-input HOST9 branch and original paired HOST after9 minus current before9 on these96 coordinates; ordinary targets remain original same-input HOST9. Fit the strong rows independently with ridges[1e-6,1e-5,1e-4,0.001,0.01]. No A294 fitted weight or residual is used, and every target is fixed from original observations. The mix0/ridge1e-6 control must reproduce A290's native metrics.

This tests whether the weak-coordinate physical normalization problem in A294 can be avoided by placing the paired correction solely on coordinates with gain1e6. Keep nativebranchrelativeRMS<=0.015 and the explicit A295-family exploratory state/direction<=1.20baseline. For eligibility beyond thislocal screen, require directionerror<=0.95A295's0.005003612513 and pairedstate no worse thanA295's0.006099138813, then the same conservative arithmetic bound/caps. Original1.05guard status is also reported. A score is not automatic: evaluate the A295 result first. Only code preparation may overlap the running score; no heavy analysis runs concurrently.

### A297 linear principal-component norm anchors declared; queued after A295 (2026-11-11 09:55:19 CET)

Use the same fixedA290 allocation/512donors, original-HOST native branch targets and A288 emitter. Replace A293's poorly fitted positive-sqrt scalar anchor with an exact linear compression of original HOST raw output. Anchor counts[1,4,8,16,24]. Coordinate469 is always first, as required by the transferred programme; choose remaining coordinates once by smallest original HOST9 post-output mean-square energy outside the96 strong coordinates. For each count, take the top count PCs of the original HOST normalized raw covariance over the96 strong coordinates plus these anchors. Apply that orthogonal projection directly to original HOST Down plus the original fixed donor reconstruction to supply native raw anchor rows. Set the anchor postgains tozero; leave programmeDown on469 unchanged. No fitted nonlinear energy target, candidate residual or iterative refit.

Reconstruct all other native behavior independently from HOST, with one ordinary-gain accommodation and five strong-row ridges[1e-6,1e-5,1e-4,0.001,0.01]. This retains part of the raw normalization energy discarded by strong-row suppression while explicitly accepting the measured original native output cost of weak anchor coordinates. At24 anchors, nominal additional native edits24*1525=36,600, leaving the total at most5,015,300 below the5,016,268cap.

Require nativebranchRMS<=0.95A295, pairedstate/direction no worse thanA295, nativebranchrelativeRMS<=0.015, and the same conservative native9/16futurebranch arithmetic bound, OFFgates andcaps. Report original1.05screen separately. Select lowest nativebranchRMS among guarded variants. No model or native activation evaluation; all native features are frozen captures. No heavy analysis during the current fullscore.


### A298 query raw-energy principal anchors declared; queued after A295 (2026-11-11 10:07:00 CET)

Test an analogous linear raw-energy restoration in the layer3 reader, keeping native K/V/O and the existing64 programme Qchannels. Anchor counts[1,2,4,8,16] are single Qchannels outside the programme set. Rank them once by original paired-HOST normalized Q output energy times the mean native K energy of the corresponding rotary pair; choose the weakest counts. No query/key pair is deleted, and every Kweight/gain stays HOST. Shared Qanchor postgains arezero.

Start native Q from the original HOST mean-nulled native query B and the existing A61 selected-channel gain compensation, without its fitted ordinary energy rescaling or later candidate repair. For each head, construct the leading count PCs of original paired-HOST B raw outputs on the64 suppressed programme rows (weighted bysqrt(1-(HOSTgain/currentgain)^2)) plus selected anchor rows. Map those linear components into the anchor raw rows. Original native numerator targets define one ordinary-channel gain accommodation; programme gains stay fixed, anchor gainszero.

Use the original rank-one arithmetic programme and analytically sufficient source amplitude. One original-HOST raw-energy ridge solve may rescale surviving ordinary Qrows, with reciprocal ordinary gain changes. Ridges[1e-6,1e-4,0.01,1,100], same analytic0.99 positivity contraction asA292. Programme rows and anchors are excluded from this energyfit, preserving pure-programme asymptotic routing. Recompute only the sufficient routing amplitude afterwards; no residual fit or iteration.

Require both actualrounded paired/same-input causal QK MSE<=0.9A259, pairedqueryRMS<=1.05baseline, sourcegaps>=7.999 on allrecorded current/historical sourcepositions, andcaps.192training/64holdback wordwindows, fixed source-position data, nosoftmax/nativeactivation/modelcalls. Any eligible query requires a separate compiler/fullscore. Only codepreparation overlaps the running score.


### A299 native9-energy-aware emitter pools declared; queued after A295 (2026-11-11 10:10:38 CET)

Reselect the emitter's strong coordinates using A145's fixed protected five-function literal-span QR leverage, now divided by original same-input HOST9 normalized raw energy from A174's48training windows. Floors[0.01,0.1,1,10,100] times the medianenergy, counts[48,64,80,96,112], grouped literal biases[1,2,4,8,16]. Force all10 existing cleanupcoordinates; exclude rawanchor469. Solve the original protected-vocabulary metric dual on each pool. Literal tests use digits0–9 at offsets[-0.3,0,0.3], plusnewline−1; no operands or answers define weights. Choose greatest literal margin per floor/count, then smallest originalnative energy among floors with margin>1.

For each selected count, test the five literal-emission multipliers[1,2,4,8,16] on exact A123 preMLP9 reconstruction, using the existing11 transferred hand functions. Rawgain50,000 and anchoramplitude2 stayfixed, anchor moves439→469. The existing cleanup component is scaled by the analytic mean literal/anchor norm ratio, following A145's original cancellation normalization; there is no additional cancellation-scale trial. The common Upboost1024 and minimal power-of-two per-unit balancing enforce the1e6caps.

Check actual hand features on all640 recorded selectorregister positions, whole-vocabulary margins, uniform programme GELU-to-ReLU uncertainty and all16future nativebranch bounds as inA288. Geometry stage omits nativeMLP9 and uses HOST9 ordinary gains, stronggains1e6 andanchor469gainzero. Save only strictlypositive guarded-margin geometries. This is not a checkpoint or text-preservation result; any promising geometry still needs original-native reconstruction/allocation and a fullscore. Noheavyprocess overlaps A295.


### A295 full H1 completed: early emitter worsens text agreement (2026-11-11 10:21 CET)

Official0805-score, checkpointSHA`2e3e55690720adfb30ebdd5113e4cc2e0bb6299fad986186a6fda1b8a0eb6b0e`, `scores/a295v1_full_H1.json`:24,932/25,000exact anddigits(99.728%), text97.86300659179688%, KL0.002250764756867062, CEchange+0.00013254782415295097, finite. H1false because text remainsbelow98.5%. Text is203matches/131,072 (0.154876708984375percentagepoints) worse thanA259 and459matches(0.350189208984375points) worse than the emitter-freeA285diagnostic. Native-cost differences are not assumed additive.

Noise exact/digits per2,000:1e-5=1997/1997,1e-4=1551/1999,1e-3=257/1561. Two-in-a-row:first1995,secondgivenfirst2,both2.4,978,700edits,max1e6. Wall28:03.69,peak3,856,976KiB. No rung banked.22fullscores completed(21H1plusoneH5diagnostic),18fullremain;0forward/0quick. A259remains the arithmetic-qualified textleader. A295andalltransitive constructions areimmutable.

The first20 permitted failures mostly show off-by-one/two late digits, with two earlier digit errors; this is diagnosis only, with no answer-target fitting. The conservative640-position surrogate certificate didnot claim global accuracy or preservation. A296–A299are next in their declared local sequence; no new score automatically follows them.


### A296 rejected: strong-only paired target does not improve (2026-11-11 10:25 CET)

The mix0/ridge1e-6 control reproduces A290 exactly: nativeRMS0.6481923222 and direction0.005003612513. It is also the best of25 trials; every positive paired strong-target mix worsens direction. No variant meets the declared5% improvement, and no bound/PT/checkpoint/score follows. Wall30.86seconds,peak3,925,964KiB. A297 is next.


### A297 principal norm anchors rejected (2026-11-11 10:29 CET)

Counts1/4/8/16/24 retain92.92/93.71/94.37/95.17/95.72%of selected originalnormalized rawenergy, but bestnativeRMS is0.6413978728 atcount4, only1.05%belowA295 (not the required5%). Its pairedstate0.00607981023 anddirection0.00500343648 barely improve. Othercounts haveRMS0.65520/0.66971/0.68012/0.68119. RemovednativeanchoroutputRMS increases0.06689→0.18908 withcount. No nativeguard passes; noarithmeticfollow-up/PT/checkpoint/meter. Wall40.00seconds,peak3,923,092KiB. Energy recovery alone doesnot remove the observed native reconstructioncost.


### A300 mean-normalized dense query energy repair declared (2026-11-11 10:31:23 CET)

A292's constrained raw-energy fit tried to restore the full HOST raw norm while its gain-compensated native query had a lower mean raw norm. Test one independently defined normalization correction, retaining A292's sourcepenalties[0,0.01,0.1,1,10], asymptoticmargins[8.25,8.5,9,10,12], energy ridges[1e-6,1e-4,0.01,1,100], exact four-head programme-energy constraints and deterministic0.99positivity contraction.

For each original native query profile N0, compute kappa=sqrt(mean((mean_j(N0*x)^2+eps)/original_HOST_raw_energy)) on the192training wordwindows, before any energyfit or programme amplitude. Multiply all proposed postgains by this single statistic. The original raw-energy target becomes kappa^2 times the original same-input HOST energy. No extra scale trial, candidate residual or iterative fitting is introduced. The original pure-programme direction is unchanged; any reduced asymptotic routing headroom must pass the existing analytic source-amplitude test.

Then independently solve the constrained original energy systems and evaluate actualrounded QK/RoPE errors and routing. Keep the same <=0.9A259 paired/same-input causal-score guards, <=1.05paired-queryRMS, sourcegap>=7.999 andcaps. No model/native activation execution. This is queued afterA299; sources will be newly named.

### A301 native9 cost separation declared (2026-11-11 10:31:23 CET)

Diagnostic only, using the certified A174 features and fixedA295 units/donors/pool. Compare original same-input HOST9 against: (a) displaced11unitszero with no repair, (b) one originalHOST donor reconstruction at512/ridge0.001 with HOSTpostgains, (c) all2,048 nativefeatures retained but96strong gains inserted and native rows reconstructed using fixed strongridge1e-6 plus the original one ordinary-gain accommodation, and (d) actualA295 native weights at the recorded inputs. These are controls, not additive causal attributions or complete arithmetic candidates. Report total/strong/ordinary nativebranchRMS and relative raw-norm error. No checkpoint, new score, target selection or source change. Run only after earlier heavy work finishes.


### A298 query principal anchors rejected (2026-11-11 10:32 CET)

All25 count/ridge cases fail native metrics despite sourcegaps>=8. Best count4,anchors[150,173,151,164],ridge100 retains70.52–92.09%of chosen rawenergy byhead but has paired/same-input causal MSE0.00331104354/0.00275523198 versusA2590.00142456275/0.000833213286. PairedqueryRMS0.0558745974, minimumsourcegap8.00000304, nominaledits4,978,701. NoPT/checkpoint/meter. Wall29.47seconds,peak4,194,272KiB. A299 geometry is next; A300 and A301 remain queued.


A299 setup correction before any geometry trial: its helper referenced the declared literal probe array before initialization. The initial5.69-second run raised NameError and wrote no result/checkpoint. Moved the same fixed probe definition before the helper call; no grid, target or selection rule changes. Originalfailurelog retained, rerun usesa299v1..._retry.log. No metered source was changed.


### A299 native9-aware pool geometry fails literal margin screen (2026-11-11 10:35 CET)

After the documented setup fix, no pool meets minimum normalized literal margin>1 on the declared +/-0.3 probes. Best margins by48/64/80/96/112coordinates are-2.44226/0.594180/-1.45634/-0.736917/-0.0332966. Only64coordinates haveall31correct. Its nativeenergyfloor0.01,bias2 geometry uses3.72555963%of originalnormalized rawenergy versusold96pool4.37148761%. The conservative format/future stage is not triggered; no geometryPT/checkpoint/meter. Wall16.34seconds,peak3,660,860KiB. A300 original-mean normalization query test is next.


### A300 mean-normalized dense queries still fail preservation (2026-11-11 10:38 CET)

Sixteen profiles fail asymptotic source headroom;20 admissible profile/ridge cases reach metrics. Best sourcepenalty0.1,asymptotic10,energyridge1 hasnative meanenergy ratio0.7113032264,kappa0.8433879454, positivitycontraction0.12929275 andpaired/same-input causal MSE0.00839476302/0.00777345245. Sourcegap8.00000518 ispreserved, butnativeerrors remainfaraboveA259. NoPT/checkpoint/meter. Wall24.50seconds,peak3,792,112KiB. A301 cost-separation controls are next.


### A301 native9 controls identify both feature and gain costs (2026-11-11 10:48:35 CET)

Original strongpool rawenergy fraction4.37148761%; coordinate469alone47.73730461%. Deleting11nativefeatures givesbranchRMS0.9276847509; original512donor repair withHOSTgains reduces it to0.5259844114. Inserting96stronggains while retaining all2,048nativefeatures gives0.3105021577. ActualA295 has0.6481923222, withweighted strong/ordinary errorcomponents0.3521657597/0.5441806357 andrelative rawRMSerror0.0225693146. These controls are not additive causal attributions. Wall14.53seconds,peak1,023,528KiB; no checkpoint/meter.

### A302 full future-bound check of A299's sole all-correct pool declared (2026-11-11 10:48:35 CET)

Keep A299's failed>1 literal screen. A separate exploratory geometry check admits only positive literalmargin, so the already observed64-coordinate/floor0.01/bias2 geometry can reach the stricter whole-vocabulary native-future/hand uncertainty bound. Retain every declared mass, weight construction, input and cap; no new numerical trial. All640known register positions still need strictlypositive conservative margin. This doesnot imply the weaker literal screen is a pass, and no nativeMLP9/text result is claimed.

### A303 displaced-feature-specific donor selection declared (2026-11-11 10:48:35 CET)

Fix the11A290mass0units and existing96pool/A288programme. Choose512donors by one original-feature correlation ranking: sum squared uncentered normalized feature correlations with the11displaced features, weighted by their equally normalized original HOST post-output and raw-energy costs. Exclude displacedunits. Blend this ranking with the original global native-energy donor ranking using masses[0,0.01,0.1,1,10], each ranking term normalized tosum1. No candidate residual or iterative feature selection.

For each distinct donor set independently reconstruct from HOST with the existing0.001donor ridge and strong-row ridges[1e-6,1e-5,1e-4,0.001,0.01]. Require nativeRMS<=0.95A295, pairedstate/direction no worse thanA295, relativeRMS<=0.015 andallcaps/OFF/conservative arithmetic guards. Select lowestnativeRMS. Given A295's poor official text result, passing this local5%screen doesnot authorize an automatic score: require nativeRMS<=0.35 or independent further evidence before spending another full run on this architecture.


### A302 positive-literal geometry fails the future-layer bound (2026-11-11 10:49 CET)

The sole64-coordinate geometry reaches the full check under the explicitly weaker literal screen, but no emissionmass passes. Mass1/2 haveall640nominal localwinners, yet their future+hand lowerbounds are-757,709.57/-837,090.12, with280/54uncertifiedpositions. Mass4/8/16also lose localwinners. AlloriginalA299 failuresremainrecorded; no geometryPT/nativeallocation/checkpoint/meter follows. Wall36.16seconds,peak3,682,744KiB. A303 targeted-donor reconstruction is next.


### A303 targeted native9 donors rejected (2026-11-11 10:55:06 CET)

Allfive independent donor sets fail the5%improvement screen. Best rankingmass10 givesnativeRMS0.6453820276, only0.43%belowA295; pairedstate0.00609602771 anddirection0.00500182634 barely improve. Othermasses giveRMS0.64929–0.65217. No arithmeticbound/PT/checkpoint/meter. Wall33.97seconds,peak3,515,188KiB. The additional0.35nativeRMS fullscore screen is not met.

### A304 refresh only the layer3 programme quiet field declared (2026-11-11 10:55:06 CET)

Keep A259's native query component exactly, algebraically subtracting only the original rank-one arithmetic programme recovered from A191/A61. Fit a new linear format field to have mean1 at the original historical format-query mean and minimal uncentered current-word energy. Use192currenttraining windows and the96current+1,024historical source-query inputs for covariance;64word/32current+512historical source windows remain validation. The field target is only format presence, never an operand or sum.

Five format-variation masses[0,0.01,0.1,1,10], with the format covariance normalized to the word covariance trace, and fixed ridge1e-6meanword diagonal. Each solve starts from these original matrices. Preserve the original four programme output directions, replace only their shared input field, then calculate the analytic sufficient source amplitude for margin8. No query-energy or candidate-residual fit. Check positive sourcefield values and exact rounded native QK/RMS/RoPE behavior.

Require paired/same-input causal score errors<=0.9A259, pairedqueryRMS<=1.05baseline, sourcegap>=7.999, andcaps. Also report the programme-removed native-query diagnostic without claiming it is an arithmetic model. Any eligible candidate needs a separate compiler/fullscore.


### A304 quiet-field refresh rejected (2026-11-11 11:10:00 CET)

The best new field reduces validation word field RMS from0.0004595928763 to0.0004468993, but best paired/same-input causal MSE0.0014245664923/0.00083321277694 doesnot improve A259's0.00142456274575/0.00083321328561. Removing the programme entirely gives0.00142440655361/0.00083304616153 and loses source routing (gap-9.25617). This shows that native query accommodations dominate this local error, not leakage of the quiet programme field. All five declared masses fail; noPT/checkpoint/meter. Wall22.24seconds,peak3,599,208KiB.

### A305 additive rotary-key carrier declared (2026-11-11 11:10:00 CET)

Keep all original HOST layer3 key rows and gains, and add one format-presence field to one key row. This differs from A64's overwritten/deleted key pair and A283's deletion control. Fit one linear key field with mean1 over training source positions0..8, minimizing uncentered word energy plus within-format variation (equal trace, fixed mass1 and ridge1e-6 meanword diagonal). Training uses192current word windows and96current+1024historical format windows. Source-field values must be positive on all recorded current/historical positions. BOS is causally identical in text and format; its unavoidable key perturbation is explicitly included in the native score proxy.

Choose five rotary pairs once from original HOST normalized Q/K and raw-Q energy, among pairs whose minimum ideal phase gap over offsets1..8 is>=0.02. Use the same ranking as A283. Key amplitudes are[1,4,16,64,256] times16*max(original format key RMS/positive field). Keep the native K counterpart and every other K row/gain unchanged. For each pair, shared Q gains start at max(original pair gains,8.25/(16*HOST_key_gain_at_pair*ideal_phase_gap)); multipliers[1,2,4,8,16]. Native Q is original HOST with its historical format-query mean nulled by the existing quiet field, with the two pair rows inversely gain-compensated; all other rows/gains remain original HOST. Add a rotary-phase query on that pair, with source phases for positions[3,2,1,0]. The four programme amplitudes are calculated analytically for margin8 against actual rounded candidate K, not searched or fitted to answers.

Evaluate actual rounded linear/RMS/QK/RoPE behavior against original same-input and paired HOST scores on64 current holdback word windows. Require both causal score MSE<=0.9A259, paired query RMS<=1.05baseline, source gaps>=7.999 and edit/value caps. No query-energy fit, residual fit, native activation or model execution. Retain only diagnostics unless all local guards pass; any eligible construction needs a separate compiler and full score.


### A306 larger independent text sample for original query repair declared (2026-11-11 11:12:17 CET)

Rebuild the A237 causal-score repair once from original A191 Q, using the512 already recorded A119 calibration windows for fitting and all256 A150 current windows for validation. Check checkpoint hashes and exact pre3 upstream weight identity for A117/A123/A191/A259, and report exact duplicate windows between the two samples. No new model pass, fresh text or heldout text is read. These validation windows have appeared in earlier analyses; they are disjoint from this construction's fitting sample, not globally unseen.

Use the same96 current source-query null constraints, canonical input dimensions[8,16,32,64,128],16 original-response output PCs, whitening ridge0.001, metric ridges[1e-6,1e-4,0.01,1,100] and strengths[0.25,0.5,0.75,1,1.25]. Each fixed linear covariance solve targets original same-input HOST normalized queries minus original A191 queries at the original A191 raw RMS. There is no fit to the A259 residual or any other fitted candidate. Centered causal-key Grams are built from recorded inputs with RMS/RoPE only.

Evaluate against original same-input and paired HOST QK scores on all256 validation windows, plus all current/historical source margins. Independently select lowest paired-score MSE per head among candidates with same-input MSE no worse than A259, paired-query RMS<=1.05A259, rounded96-query null change<1e-5 and sourcegap>=7.999, otherwise retain the A259 head. Combined eligible result must improve both paired and same-input MSE by at least5% over A259 on these256 windows and satisfy edit/value caps. A local pass needs a separate compiler and full score. A305 must finish before this heavy analysis starts.


### A305 additive key carrier rejected (2026-11-11 11:13 CET)

All125 declared pair/key/query cases fail native preservation. Best paired/same-input score MSE4.22563240/4.22604087 is orders of magnitude above A259. The field has positive source values0.09359–1.99727, but word RMS0.09172 and BOS value0.09359; sufficient key dominance severely changes native normalization. Even the smallest key multiplier gives key RMS>1.3. Best diagnostic pair21/key256/query2 preserves sourcegap8.0000619 but is not useful. NoPT/checkpoint/meter. Wall47.58seconds,peak3,892,020KiB. A306 original-target larger-sample query repair is next.


### A307 native query mean restoration with existing energy repair retained declared (2026-11-11 11:14:55 CET)

A157 restored HOST ordinary rows/gains while changing mean cancellation, so it also removed the established raw-energy repair. This independent construction instead keeps A259's native row scales, norm gains and causal correction exactly. Decompose original A191 Q into the original rank-one programme P plus per-channel scaled HOST mean-nulled rows; verify reconstruction error<1e-6. For cancellation fractions[0,0.25,0.5,0.75,1], restore(1-fraction) of the original removed HOST mean with those same row scales, leaving the A259 causal repair unchanged. Derive only the necessary arithmetic programme amplitude by the existing analytic margin8 quadratic. No native residual, energy fit or iterative correction is computed.

Evaluate64 current holdback windows against original paired and same-input HOST QK, with all current/historical source constraints. Require both score MSEs<=0.95A259, pairedqueryRMS<=1.05baseline, margin>=7.999 andcaps. Fraction1 must reproduce the current native component and current metrics up to rounding/routing guard. NoPT/checkpoint if none eligible. This runs after A306; preparation only can overlap its heavy process.


### A306 larger sample query repair rejected (2026-11-11 11:17:39 CET)

All512 fitting windows are distinct from the256 validation windows. No head's source-safe candidate beats A259 on the validation paired-score metric; all four current heads are retained. The corresponding A259256-window baseline is paired0.00122515784836 andsame-input0.000748661600138, distinct from the64-window metric used by other analyses. Best source-safe per-head fits remain slightly worse. NoPT/checkpoint/meter. Wall1:19.78,peak5,336,460KiB. Larger calibration alone didnot help this fixed repair family.

### A307 mean restoration rejected (2026-11-11 11:17:39 CET)

The fraction1 control reproduces paired/same-input metrics within1.1e-11:0.00142456275593/0.00083321329525. Every partial mean restoration is worse; fraction0.75 already gives0.00361785026705/0.00303422928373, with programme factors approximately419/52/497/1243 to retain source margin8. Full restoration is worse still. This preserves existing energy repair and therefore resolves A157's confounding change, but doesnot yield a useful query. NoPT/checkpoint/meter. Wall17.06seconds,peak3,595,348KiB.


### A308 expanded original-native final donor reconstruction declared (2026-11-11 11:18:48 CET)

Keep the existing11-unit final emitter,96strong coordinates and programme Down columns exactly. Independently reconstruct native HOST behavior with donor counts[512,640,768,896,1024], fixed donor ridge0.001 and strong-row ridges[1e-6,1e-5,1e-4,0.001,0.01]. Retain the existing512 donors. Rank extra surviving native features once by their conditional normalized-feature correlation with the11 displaced original HOST features, weighted by original HOST post-output column energy, conditioning on the original512 donors with ridge0.001. This is a fixed original covariance statistic, not a candidate residual or iterative feature selection.

Use A272's strengthened captured-feature provenance:256old training windows,128old plus128fresh validation windows, unchanged recorded final native features, no nonlinear replay. Each donor count starts independently from HOST with the immutable NativeFinalReconstruction formula; ordinary gains are accommodated once and all strong rows fit original native targets. Require native-branch RMS at least5%better than A142. Only the smallest-native-RMS candidate passing finite/value/edit caps advances to full protected-vocabulary calibration readout. Require old matches nondecreasing,fresh>=8extra,total>=16extra/16,384, and all2,560recorded format outputs correct withmargin>1.

For edit-count eligibility only, compare final tensor replacement on the existing metered A259,A253,A234 parents, in that order of official text agreement, using the first one that fits the cap. Verify their final G/Up/Down/gains are bit-identical. This doesnot assert the same later native trajectory: final-layer observations remain A142/A123 calibration surrogates, and transfer to any lower-edit parent would require a separately declared compiler/fullscore. No early query or other repair is composed automatically. No checkpoints or model passes in this analysis.


### A308 donor expansion narrowly fails native screen (2026-11-11 11:22:01 CET)

Native RMS by512/640/768/896/1024 donors is3.29298225/3.23783676/3.21830403/3.20341338/3.19437114 versusA1423.35574020. The best reduction is4.81%, below5%, so no protected-vocabulary stage or PT/checkpoint follows under A308. Count1024 fits only the lower-edit A234 parent (5,008,112edits);640 fitsA253 (4,958,203). Wall26.58seconds,peak5,485,740KiB. The fixed ranking and complete trials are retained.

### A309 direct readout follow-up of A308's best native reconstruction declared (2026-11-11 11:22:01 CET)

Keep A308's failed5%screen. For this separate diagnostic only, permit the single already observed1024-donor candidate through at4%native-RMS improvement. Rebuild that exact independently defined original-HOST construction; no new donor ranking, fitfamily, scale or weight choice. Verify allfive A308 native RMS values reproduce within1e-10, then examine only the originally best1024 candidate. Keep every original full-vocabulary criterion: old matches nondecreasing,fresh>=8extra,total>=16extra and all2,560formats correct withmargin>1. No newcheckpoint or full score without all readout checks and a separate compiler declaration.


### A309 expanded donor readout rejects the candidate (2026-11-11 11:26:09 CET)

The1024-donor weights reproduce A308's native metrics. Full-vocabulary old validation improves8023→8029/8192,but fresh worsens8003→8001/8192; total+4 fails the declared+16andfresh+8 guards. The format stage is not triggered. NoPT/checkpoint/meter. Wall1:01.65,peak5,539,336KiB. Native RMS improvement alone didnot establish better text predictions.

### A310 BOS-anchored native rotary key construction declared (2026-11-11 11:26:09 CET)

Address A305's observed BOS/key-normalization disruption without deleting native key outputs. Use the same fixed original-energy-ranked pairs[9,15,4,17,21]. Keep all other HOST Krows and all Kgains unchanged. For each pair, let v be its original normalized raw K vector at current BOS. The desired normalized pair is the constant v on all format source positions. With residual raw energy R from the254 unchanged channels (includingepsilon), the exact corresponding raw pair target is R*v/sqrt(1-||v||^2/256). This scalar-normalization/source-format target contains no digits or answers, and preserves the original BOS pair by construction.

Solve a single pair-row perturbation from HOST using192 word windows and96current+1024historical formats. The perturbation lies in the exact null of the current BOS input. Word features are normalized by original fullkeyRMS; format features by unchanged254-channel residualRMS. Format covariance masses[1,10,100,1000,10000], normalized to equal covariance trace, and fixed ridge1e-6meanworddiagonal. Each solve is independent from the original HOST targets. Check actual float32 BOS error, source geometry, and native text scores; the real-arithmetic constraint is not a claim of bit-identical rounded BOS activations.

Q construction retains original HOST mean-nulled native rows and uses inverse pair-gain compensation. The programme's rotary phase is the preserved BOS post-key angle plus the fixed source displacement[3,2,1,0]-8. Base shared pair gain=max(original pairgains,8.25/(norm(BOSpostkeypair)*idealphasegap)); multipliers[1,2,4,8,16]. Calculate per-head sufficient programme amplitudes for actual candidate K and margin8. AllotherQrows/gains remain HOST apart from the historical format-mean cancellation. No native residual, energy refit, softmax or activation replay.

Require paired andsame-input native causal-score MSE<=0.9A259, pairedqueryRMS<=1.05baseline, allcurrent/historical sourcegaps>=7.999, rounded BOS rawerror/originalRMS<=1e-5, andcaps. Otherwise retain only diagnostics. Any local pass needs a separately declared compiler/fullscore.


### A310 BOS-anchored keys fail source geometry (2026-11-11 11:29:24 CET)

All125 declared pair/format-mass/query-gain cases fail the asymptotic margin8 check, so no full query metric candidate or PT/checkpoint is produced. At mass10000, source normalized-pair RMS errors are0.00249–0.00304 while BOS post-pair norms are onlyabout0.013–0.030. Rounded BOS rawerror/originalRMS stays below5.4e-9 on these cases, but preserving BOS doesnot make other source keys accurate enough for a two-coordinate positional selector. Native key RMS is0.0455–0.0557, and source gaps have negative worst cases even at the largestquerygain. Wall25.13seconds,peak3,867,060KiB. No meter used.


### A311 early native principal-component norm carriers declared (2026-11-11 11:31:26 CET)

Apply the A297 linear raw-energy compression mechanism independently to layers0–7, keeping each layer's existing arithmetic Gate/Up/Down columns and every programme-output gain fixed. Unlike A188's square-root remainder, this uses a linear PCA of original HOST normalized raw outputs and requires no positive nonlinear energy remainder.

Anchor counts[1,2,4,8,16]. For each layer, rank coordinates outside all programme Down support and strong native-repair rows by original HOST post-output mean-square energy on192training windows; take the lowest count, with their postgains setzero. Use the leading count PCs of original normalized raw energy over the strong rows plus anchors. Construct these anchor rows directly from original HOST Down plus one projection of displaced native features onto the existing original donors (ridge0.001). Suppress strong native rows by their original/current gain ratio, accommodate ordinary nonprogramme/nonanchor gains once to original HOST outputs, and fit strong rows with ridges[1e-6,1e-5,1e-4,0.001,0.01]. Each count is an independent original-HOST construction, never a residual refit.

Evaluate exact rounded Down/RMS algebra on A174's certified native captures,192training/64validation. Require>=5%native-branch RMS improvement, both paired-HOST state anddirection errors below their parent values, preserved programme columns/gains, finite weights andcaps. Report original native output lost at silenced anchors and original raw energy fraction retained. Inputs are exact for unchanged early upstream weights and explicitly local surrogates after any changed upstream attention. No native nonlinearity, model call or arithmetic-answer target. Any eligible layer requires separate format-risk review and a compiler/fullscore; none is automatically composed.


### A311 early principal norm carriers rejected (2026-11-11 11:35:49 CET)

No layer meets the5%native improvement plus paired-state/direction guards. Best native RMS for layers0..7 is0.0119451/0.0535300/0.102121/0.0883380/0.116067/0.0882847/0.228090/0.199001 versusparents0.0110675/0.0535334/0.0990056/0.0874904/0.114003/0.0824990/0.226774/0.199904. Layer1 improves only0.0064%;layer7about0.45%and its pairedstate worsens. Layers0–2 have exact upstream provenance;3–7 are explicitly surrogate inputs after changed attention. Some anchors recover much of the chosen original rawenergy, but silencing their output or altered normalization prevents a useful native gain. NoPT/checkpoint/meter. Wall1:41.88,peak5,295,920KiB.

### A312 arithmetic unit rank audit declared (2026-11-11 11:35:49 CET)

Diagnostic only. Group existing A259 hand-programme Gate rows when they are positive scalar multiples in real arithmetic (relative residual<=1e-10), retaining exact float32 equality status separately. Within each group, inspect original Up and Down ranks and the singular values of the exact bilinear map Down*diag(Gate-scale)*Up. Rank threshold1e-10 relative to the largest singular value. A reduction is only a mathematical opportunity: GELU homogeneity is not exact, so any nonidentical-Gate compression would additionally need explicit hand-programme gate/error bounds, native allocation and official validation. No native model execution, fit to examples or checkpoint in this audit.


### A312 found one exact shared-Gate rank redundancy (2026-11-11 11:39:13 CET)

Only layer2 admits reduction: five bit-identical Gate rows, Up rank4, Down rank5, bilinear rank4 with smallest/largest singular ratio1.217e-20. Everyother shared-Gate group has full rank. A preliminary Down-column span check confirms the dependence is in Up, not Down: allfive Down columns are independent. Units529and727 have equal Up norms and are the suspected duplicate reference functions. No weight change or native-fit result yet. Audit4.16seconds,peak717,276KiB.

### A313 exact duplicate reference-unit compression declared (2026-11-11 11:39:13 CET)

Verify bit equality of both Gate and Up for layer2 units529and727. If it holds, their feature values are identical for every input, including GELU; replace their two Down columns by their sum on one retained unit and restore the other unit's original HOST Gate/Up. Examine both possible retained units independently. Keep all other arithmetic units and all layer2 postgains unchanged. Require no overlapping nonzero entries between the two programme Down columns, so the merge introduces no arithmetic sum rounding in those columns. All weights/caps remain enforced.

For each four-unit programme, reconstruct displaced original native behavior from A174's exact layer2 native features using NativeFixedGainReconstructionv2, existing512donors, original HOST branch targets and the fixed original A259 raw normalizer. Ridges[1e-6,1e-5,1e-4,0.001,0.01]. No candidate residual refit; no native activation execution. Select minimum native-branch RMS among candidates improving it>=5% and improving both paired HOST state anddirection error on64holdback windows, with192training. Gate/Up of every surviving native unit are HOST bit-exact. Local pass still needs a bound on the restored native feature at the128exact recorded format inputs, and separately declared compiler/fullscore; no automatic checkpoint.


### A313 exact duplicate compression passes local native screen (2026-11-11 11:42:11 CET)

Gate andUp rows529/727 are bit-identical and their Down supports disjoint. Both choices have exactly zero double-precision programme-map difference. Restoring529 gives nativeRMS0.09199876855; restoring727 is selected with0.09072787382 versus0.09900556684(8.36%improvement), ridge0.01. Pairedstate0.007078063761→0.006997039053 anddirection0.004849319120→0.004744765436 improve. Maximum recorded rawRMSchange2.6267%;4,976,785edits,max1e6. Local selected weight artifact saved; no checkpoint or model score yet. Peak3,891,608KiB.

### A314 exact-input format bound for A313 declared (2026-11-11 11:42:11 CET)

Use only the selected A313 weights, with no new fit or scale. Assert A150/A140 layers0–2 match A259 before this change. All native features except newly restored727 are already recorded at the exact format preMLP2 inputs. Compute their new linear Down contribution. Bound the unknown restored feature symmetrically by (abs(originalHOSTGate dot input)+roundoff+0.25)*(abs(originalHOSTUp dot input)+roundoff), using the global GELU-vs-ReLU error bound only; do not evaluate a native activation. Confirm the original duplicate programme features are bit-identical in the captures.

Propagate that one-feature interval through raw RMS and postgains. At all128 equals positions require positive referencecoordinate617, maximum B-register ratio displacement (9*coordinates29/113/467 dividedby617) from the recorded parent <=0.001, and rawRMS relativechange bound<=1e-5. Report earlier-position state bounds separately: their native trajectory is not claimed bit-identical or globally certified. These are real-arithmetic bounds at captured inputs and rounded weights with Gate/Up dot-product allowance, not a complete floating-point/full-network proof. Only if the equals checks pass may a separately declared compiler/fullH1test follow.


### A314 format bound passes (2026-11-11 11:43:51 CET)

Captured duplicate features are bit-identical. Original raw reconstruction relativeerror2.88e-8. All128equals intervals havepositiveRMSandreference; reference lowerbound106.21364. Worst rawRMS relativechange bound6.16491e-8 and B-register ratio displacement7.58320e-6, both well inside declared guards. Restored native feature envelope<=8.08658 at equals. Earlier-position RMS lowerbounds arepositive too, but their state relative envelope reaches3.06046%; no global source-query/whole-network certificate is claimed. Wall22.51seconds,peak1,185,492KiB. A313native analysis was19.25seconds,peak3,891,608KiB.

### A313 checkpoint/full H1 construction declared (2026-11-11 11:43:51 CET)

Install exactly A313's selected layer2Gate/Up/Down tensors on A259; allotherweights and allpostgains stay bit-identical. Preserve the exact four-unit programme and restore native727. No new solve, scale, candidate choice or override. Include the A313/A314 input/source hashes and all A259 construction sources in the atomic manifest; reconcile purpose categories with the existing A259 index-verified audit and add explicit metadata overriding the old five-unit list. Expected4,976,785edits,max1e6.

After cap/source/category checks, run one fullH1score as the sole heavy process. This will be full run23(22H1plusoneH5diagnostic), leaving17full; forward/quick remain0. Arithmetic/text/noise/repeated-problem results are unknown until that score completes. Construction and transitive sources become immutable at meter use. No rung banked.


### A313 atomic build complete; full H1 running (2026-11-11 11:45:19 CET)

Delta SHA256`67fd7d1ae9b0d63f4d2f8993b65f1e837fb65a418804e56125a8b7eb4cec49e5`,4,976,785edits,max1e6. All purpose-category totals reconcile, including inherited layer0 labels from the A259 audit. Build32.81seconds,peak781,532KiB. Only layer2Gate/Up/Down change; metadata explicitly overrides the five-unit list. FullH1started as soleheavyprocess, output`scores/a313v1_full_H1.json`, log`artifacts/a313v1_full_H1.log`. A313/A314/helper/alltransitive sources are now immutable. No pass or result is claimed from the pending score.


### A315 six-unit phase spline selector geometry declared; queued after A313 score (2026-11-11 11:49:41 CET)

A312 found no further exact rank redundancy. Test a distinct approximate selector: six units replace the existing nine-unit, four-saturated-step selector. Keep the current common anchor/flag/baseline unit. For normalized clock q=phase-leading_bit, literal output values atq=-1,0,1,2,3,4 are[1,H,T,U,-1,-1]. Let jumps be[H-1,T-H,U-T,-1-U]. A piecewise-linear interpolant is1+jump0*ReLU(q+1)+(jump1-jump0)*ReLU(q)+(jump2-jump1)*ReLU(q-1)+(jump3-jump2)*ReLU(q-2)-jump3*ReLU(q-3). These per-place coefficients are hand-derived for arbitrary register digits, never fitted to sums or answers. Five hinge units plus the common unit implement this generic multiplexer.

Use the existing copied digit/phase/flag linear fields. Define a format-OFF field from the current commonGate: a=mean(commonGate/flag) over the first96A150 format windows at positions8..12, off=a*flag-commonGate. Require off>0 on all observed negative positions. For sharpness[100,1000,10000,100000,1000000], each hinge's negative multiplier is the direct bound making all word/prefill Gates<=-32. Programme Up rows are the stated jump differences, with the existing commonUp normalization factor; Down uses the exact common-unit digit gain*a/sharpness. No trial knot offsets or answer regressions.

Geometry uses128A150 and512A122 format windows, plus their recorded word/prefill inputs, with strengthened source identities. They are exact inputs for the earlier A123/A120 trajectory and explicit surrogates for A259. Check literal source-register selection by current position and existing leading-bit register, max digit-ratio error plus global hand-GELU uncertainty<=0.05, commonGate>16, negativeGates<-16 andcaps. If no geometry passes, no unit allocation/PT/checkpoint. Any eligible geometry later needs original-native allocation, restored-feature bounds and a separate full score. Only script preparation during A313; no concurrent heavy analysis.


### A316 four-unit layer2 native allocation declared; queued after A313/A315 (2026-11-11 12:00:40 CET)

Keep A313's exact four arithmetic Gate/Up/Down functions and every postgain. Reallocate those functions to four native units using A187's one-shot conditional native-feature cost: original HOST features, the original512 donors, conditional ridge0.001, and the exact RMS branch error of removing each original conditional feature. Five original-statistic allocations use raw-energy masses[0,0.01,0.1,1,10]; no iterative selection or residual target. Reconstruct each distinct allocation independently with NativeFixedGainReconstructionv2 from original HOST targets and the original A259 normalizer, using ridges[1e-6,1e-5,1e-4,0.001,0.01]. Compare to the selected A313 four-unit native reconstruction, requiring at least5%further native-RMS improvement and lower paired HOST state/direction errors, finite weights andcaps. Use192training/64validation A174 windows, exact upstream layer2 provenance. No full score or checkpoint in this analysis. A local pass would need bounds for every newly restored native format feature and a separate compiler declaration. Preparation only while A313's full score runs.


### A317 source-null continuation reference geometry declared; queued (2026-11-11 12:04:48 CET)

A two-exclusive-gate replacement of the layer6 carry anchor is not pursued: its input is the full0..19 hundreds sum, not a binary carry, so the varying gate amplitude would alter shared-frame normalization. Instead test reducing ten frame units to nine by replacing only the paired continuation-anchor ramp with one existing commonGate times a new linear Up field. At continuation positions the field targets the original linear reference (the difference of the old plus/minus Gate rows dividedby twice their width); at equals it targetszero. These are current-position/format targets, never operands or answers.

Use128A150 independent format windows, first96for construction and32holdback. Standardize the640 input columns on training rows dividedby their positive reference (continuation) or flag (equals). Constrain the new field to the exact null of the96training equals rows (SVD relative tolerance1e-12). Within that null, solve the original continuation-reference target once with ridges[1e-6,1e-5,1e-4,0.001,0.01], each relative to mean design covariance diagonal. Keep all other programme functions and postgains fixed. No native activation replay or native repair in this geometry stage.

Require maximum source leakage/flag and continuation relative-reference error<=0.001 across all128windows, programme-only normalized-post difference relativeRMS<=0.001 and maximum per-state relativeRMS<=0.003, current commonGate>16 on all format positions and<-16 on all256A150 word windows and prefill, finite weights<=1e6. Evaluate only explicit hand-ReLU programme ideals; this is not a full-network or floating-point certificate. Any local geometry pass still needs global hand-GELU error bounds, native allocation/restored-feature bounds and a separately declared compiler/fullscore. A150 inputs are earlier A123 trajectory surrogates for A259 after attention changes. Script preparation only during the pending score.


### A318 native energy in existing zero-output anchors declared; queued (2026-11-11 12:08:24 CET)

A311 excluded every existing programme-output coordinate when choosing new quiet anchors, so it did not use layer1's existing zero-gain coordinate93 or layer4's five existing zero-gain coordinates[188,232,360,436,547]. Their HOST and current postgains are exactlyzero: changing native Down contributions there loses no native output, while their unchanged programme columns still serve the arithmetic normalization. A188 instead fitted one nonlinear positive energy remainder, which was infeasible. Test a distinct linear spectral construction in these existing zero-output subspaces.

For layers1and4 independently, use all existing jointly HOST/current-zero-gain coordinates. Compress the original HOST normalized raw outputs of the strong native-repair rows plus those anchors into the available anchor count via leading PCs. Two direct energy conventions: unscaled PCs, and PCs multipliedby sqrt(original total covariance trace/retained trace). These are independently derived original-target constructions, not iterated residual corrections or a model-output scale search. Start each from original HOST Down plus one projection of displaced native features onto the existing donors(ridge0.001); scale strong rows by HOST/current gain, accommodate ordinary nonprogramme gains once, then fit strong rows to original HOST outputs with ridges[1e-6,1e-5,1e-4,0.001,0.01]. Preserve every programme Down column and all programme-output gains bit-exactly.

Use192training/64holdback A174 native windows. Require at least5%native-branch RMS improvement, lower paired HOST state/direction errors, finite weights andcaps. Layer1 has exact upstream provenance; layer4 is an earlier-trajectory surrogate after changed attention. No composition or checkpoint in this stage. Any selected layer still needs recorded-format linear/RMS impact review and a separate compiler/fullscore. Preparation only during A313 full scoring.


### A313 official full H1: new preservation leader, rung still fails (2026-11-11 12:13:45 CET)

Official0807-score (`scores/a313v1_full_H1.json`), delta SHA256`67fd7d1ae9b0d63f4d2f8993b65f1e837fb65a418804e56125a8b7eb4cec49e5`:24,961/25,000exact andcorrect digits(99.844%); textagreement98.02474975585938%, KL0.0024928829704274477, cross-entropy change−0.0002293046878767413, allfinite. This is9additional matching text predictions/131,072and1additional exact addition versusA259. It remains0.47525024414062percentagepoints below the98.5%text gate, so H1fails and no rung is banked. A313 is now the verified arithmetic-qualified text leader.

Noise exact/digits outof2000:1e-5:1997/1997;1e-4:1997/1997;1e-3:1602/1605. These official aggregate results improve substantially overA259 at the larger sigmas; changing sparse indices also changes the alignment of noise samples, so this doesnot by itself isolate a causal noise benefit of compression. Two-in-a-row:first1997,second given correctfirst1,both1. Changedscalars4,976,785,max1e6; protected tensors remain unchanged. Fullrun23complete(22H1plusoneH5diagnostic), budgets remaining0forward/0quick/17full. The fullscore process has exited; A315/A316/A317/A318 remain sequential local analyses, with no concurrent heavy process.


### A315/A316 rejected; A313 edit audit updated (2026-11-11 12:16:01 CET)

A315's640format windows have maximum clock phase error0.236882727,RMS0.0496905633. Allfive spline sharpnesses fail: best nominal maximum digit-ratio error2.42338176, global hand-GELU bound2.43194499, versus0.05guard. Negative Gates remain below−32 and weights within1e6, but the clock is far too imprecise for direct interpolation. NoPT/checkpoint. Wall9.94seconds,peak1,980,320KiB.

A316's five original-cost masses yield onlytwo distinct four-unit sets. NativeRMS0.09261635094 and0.10739718221 both exceedA313's0.09072787382; neither improves pairedstate/direction. NoPT/checkpoint. Wall25.46seconds,peak3,600,228KiB. A313's original four-unit placement is retained.

The current leader per-tensor audit is now `artifacts/a313v1_duplicate_reference_compression_edit_category_audit.json`:4,976,785total,578,041arithmetic/accommodation and4,398,744native-only repair, max1e6. Outside layer2Gate/Up/Down, A259's index-audited labels are inherited. A313 labels all newly reconstructed native layer2Down columns as native repair; this reclassifies8,684old accommodation entries in addition to the actual1,915fewer edits. Counts are bookkeeping, not causal attribution or a claim that10,604arithmetic scalars were physically removed. The manifest/delta stay unchanged. FullA313runtime28:12.12,peak3,854,328KiB.


### A317 single continuation reference rejected (2026-11-11 12:16:22 CET)

Allfive ridges fail. Minimum maximum source leakage0.00492464 and continuation relativeerror0.04193843 exceed0.001guards. Training source leakage stays below8.4e-9, so this is a holdback failure, not failure of the hard-null construction. The smallest programme-post relativeRMSchange is1.11825 and worst-state20.2358, far above0.001/0.003guards: even small source leakage can emit the large continuation marker where the existing source branch is weak. CommonGates remain safely positive/negative(min739.68/max−618.98), and weights<=10000. NoPT/checkpoint. Peak1,102,500KiB. Paired continuation gates are retained. A318existingzero-outputanchors is now the sole heavy process.


### A319 BOS-null additive key carrier with plus reference declared; queued (2026-11-11 12:18:17 CET)

A305's positive key carrier included BOS, whose input is identical in every ordinary word window. A310 preserved BOS but attempted to copy its tiny native normalized pair everywhere and couldnot maintain positional margins. Test a distinct source design: use the plus token(position4) instead ofBOS as the A-reader's fourth reference, and make the additive key carrier exactly null on observed training BOS inputs while positive on sourcepositions1..8. This stage tests only routing/native QK preservation; the current A259 digit decoder still expects BOS and must not be scored or called a complete adder without a separately derived plus-reference decoder.

Retain all HOST Krows/gains and add a linear format field to one selected keyrow. From192word windows and96current+1024historical formats, compute the exact BOS-input nullspace(SVD relative tolerance1e-10). In that nullspace, solve for mean1at positions1..8 with word covariance and format-centered covariance masses[0.01,0.1,1,10,100], their traces normalized equal before the mass; fixed ridge1e-6meanworddiagonal. These are format-presence targets, never sums. Require positive field at every current/historical non-BOS source. The field on equals is included so all competing source keys have a defined positive orientation.

Use the same five original-native-energy-ranked rotary pairs[9,15,4,17,21] asA305, key amplitudes[1,4,16,64,256]times16*max(originalsourceKeyRMS/field) overpositions1..8, and query-pair gain multipliers[1,2,4,8,16] over the same analytic ideal-margin8.25floor. Query phases target[3,2,1,4]. Native Qis originalHOST with the historical format-mean null and inverse pairgain compensation. Programme amplitudes come directly from actual rounded-key source inequalities for margin8; no native residual/energy refit.

Require current/historical source margins>=7.999, rounded BOS raw-key error/originalRMS<=1e-5, paired andsame-input native causal-score MSE<=0.9A259, pairedqueryRMS<=1.05A259, finite weights andcaps. Report key RMS separately. Any local routing pass still needs the independent per-digit/plus decoder, arithmetic/native composition review and a separate fullscore; no automaticcheckpoint. No native nonlinearity, softmax, answer target or modelpass in this analysis. Preparation only while A318 runs.


### Analysis artifact output-path correction (2026-11-11 12:19:20 CET)

A318v1's copied output filename overwrote the unmetered A311 JSON instead of using its own filename. The A318 result has been moved intact to its declared A318 path; hashes and moves are recorded in `artifacts/a318v1_output_path_recovery.json`. No checkpoint, meter, ledger or script was overwritten. The original A311 source is unchanged and will be rerun once to restore its historical analysis artifact; its previously reported numerical conclusions remain recorded in the log. A future-safe A318v2 source changes only output filenames and is not run. A319's same prepared filename typo was caught and fixed before its first execution.


### A318 existing zero-output anchors rejected (2026-11-11 12:20:05 CET)

Both layers fail the native5%screen. Best layer1nativeRMS0.05363770823 versus0.05353338935; best layer4nativeRMS0.11607515662 versus0.11400280599. Unscaled PCs are best, retaining99.3535%and99.5099%of the chosen original rawenergy. Native anchor output lost is exactlyzero, but preserving that rawenergy still doesnot reduce the displaced native-feature error. Layer1's pairedstate/direction improve slightly; layer4directionworsens. NoPT/checkpoint. Wall23.05seconds,peak4,593,068KiB. The moved A318v1result retains its original sourcehash; output-path recovery is documented separately.


### A311 historical analysis restored (2026-11-11 12:22:07 CET)

The original unchanged A311 script completed its recovery run successfully. All40original logged layer/count diagnostics reproduce exactly, with no eligible candidate. The report is back at its original A311 path; the new execution's resource metadata is recorded in `artifacts/a311v1_output_recovery.log` and the recovery manifest. No scientific conclusion, checkpoint or meter count changed. A319routing analysis is next.


### A320 BOS-null normalized key-pair reconstruction declared; queued (2026-11-11 12:25:23 CET)

A319's additive carrier still requires large raw key amplitudes to dominate native pair variation; its running diagnostics show substantial word-key distortion. Test a direct normalized-key target instead of domination. Keep all254unselected HOST Krows and all Kgains unchanged, and fit only a two-row delta in the observed training BOS nullspace(rank threshold1e-10), with plus reference positions[3,2,1,4]. At non-BOS formatpositions1..8 target the normalized raw pair v=[r,0], radii[0.03,0.1,0.3,1,3]. With unchanged-channel residualRMS R, the exact raw target is R*v/sqrt(1-||v||^2/256). Thus the solve targets a constant direction/amplitude without requiring the pair to dominate all256raw channels. No BOS target overwrite is allowed.

Use the same fixed pairs[9,15,4,17,21],192word windows,96current+1024historical source windows, and native covariance metric asA310. For each pair/radius, independently fit the original HOST two-row perturbation with format covariance masses[1,10,100,1000,10000], traces normalized equal, ridge1e-6meanworddiagonal; no fitted residual is reused. Query gain floor=max(originalpairgain,8.25/(r*HOSTkeygain*idealphasegap)), multipliers[1,2,4,8,16]. Original HOST mean-nulled/inverse-gain native Q and direct actual-source margin8programme amplitudes are unchanged methods.

Require allA319guards: paired/same native causalMSE<=0.9A259, pairedqueryRMS<=1.05, current/historicalmargin>=7.999, roundedBOSrawerror/originalRMS<=1e-5, finite/caps. Report actual normalized source-pair error and word-keyRMS. This is routing-only; plus decoder and complete checkpoint remain unbuilt even if locally eligible. No modelpass, native nonlinearity, softmax, answer fit or concurrent heavy execution.


### A319 and A320 routing results: both rejected (2026-11-11 12:30:54 CET)

A319 completed625trials,2:32.76,peak4,439,612KiB. BOSnullrank2; allfive fields arepositive on non-BOS sources and almostzero atBOS. Best paired/same native causalMSE0.4541815900/0.4535103138(pair4,mass0.1,keymult1,querymult2), much worse thanA2590.001424562746/0.000833213286. Its source margin8.000056 and roundedBOSerror6.35e-8 pass, but native keyRMS0.479697 remains large. NoPT/checkpoint/decoder.

A320 completed625trials,58with sufficient source geometry,1:07.36,peak4,346,608KiB. Best diagnostic ispair9,radius1,formatmass1000,querymult16(basegain37.5757801). Paired/same causalMSE0.001519252819/0.000926408026 are only6.65%/11.19%worse thanA259 but fail the required10%improvement. QueryRMS0.02527828142 improves versus0.03150798113; keyRMS0.03987129981; normalized sourcepairRMSerror0.00785486202, maxrelativepairerror0.0914626385. RoundedBOSerror6.32e-9, source margin8.000122(current32:9.104155,historical512:8.000155),4,979,716edits,maxquery/key/gain600.21246. This is a potentially useful routing diagnostic, not an eligible complete adder: its plus decoder is unbuilt and noPT/checkpoint is saved. Small-radius targets often fail source geometry; radius1and3 can overcome relative source variation.


### A321 two-frequency signed positional contrast declared (2026-11-11 12:32:56 CET)

A single positive cosine selector requires nearly constant key amplitude; otherwise a larger neighboring key can beat the intended position. Use two frequencies with a shared scalar key carrier: an oscillating cosine minus a slowly varying positive offset. For oscillating frequencyw and slowfrequencyu, let m=max(cos(w*d),d=1..8), c=min(cos(u*(s-4)),s=0..8), and b=(1+m)/(2*c). The hand kernel cos(w*(s-target))-b*cos(u*(s-4)) is positive at target and negative at every other source, providedb<1. Its sign separation is invariant to any positive source amplitude and positive key RMS.

Keep the existing A259 BOS source positions[3,2,1,0], V/O/decoder and allotherweights. Recompute the exact original A305format-presence field(mean1overpositions0..8, wordcov+trace-matched formatcov,mass1,ridge1e-6), requiringallsourcevaluespositive. Use the same five original-energy-ranked oscillating pairs[9,15,4,17,21]. Choose five slow pairs once by the same original HOST Q/K/raw-energy statistic among cos(4*u)>=0.999 and positive native cos-channel keygain; no overlap. Only the two cosine Krows get the same additive raw field; every native Krow remains included.

Query programme directions divide each pair's two coefficients by that pair's cosine-channel HOST keygain, so the scalar key field yields exactly the stated signed kernel. Compute the minimum nonnegative key amplitude that cancels the worst original raw-key projection with the required sign on allcurrent/historical sources; amplitude multipliers[1.25,2,4,8,16]provide separation. After actualfloat32Krounding, derive the shared four-channel querygain floor directly from the minimum normalized source margin to reach8.25; multipliers[1,2,4,8,16]. NativeQis originalHOST mean-nulled with inverse gain compensation; final programme amplitude uses the same direct margin8quadratic inequalities. No native residual refit or answer target.

Require paired/same nativecausalMSE<=0.9A259, queryRMS<=1.05A259, allsourcegaps>=7.999, finite/caps. Evaluate native keydistortion andsourcegeometry onrecordedinputs; no native activation,softmax,wholemodel. Any eligible Q/Kconstruction needs separate arithmetic/native review andcompiler/fullscore. UnlikeA319/A320, no plus-reference decoder is required by this design. Five trial values per declared scale are retained.


### A322 original-HOST causal query repair for fixed A320 key design declared (2026-11-11 12:35:52 CET)

Reconstruct exactly A320's single best reported routing diagnostic(pair9,radius1,formatmass1000,querygainmult16), with no new key/scale choices, and require all reported diagnostic values to reproduce. This reconstruction is not an eligibility override or checkpoint. Preserve its newKandquerygains, then perform one independent original-HOST native Qrepair. The initial Qis the analytic mean-null/inverse-gain programme construction, not a previous fitted native query repair.

Use192training/64holdback current word windows and their original pairedHOST trajectory reconstructed bit-exactly from A140captured branches. The target is original pairedHOST causal QK scores minus the original fixed A320 scores. Derive canonical input bases and16output PCs once from the original pairedHOST normalized-query discrepancy; input dimensions[8,16,32,64,128], whiteningridge0.001, causal-score ridges[1e-6,1e-4,0.01,1,100], strengths[0.25,0.5,0.75,1,1.25]. All original normalizers, targets and bases remain fixed across trials. Constrain the fitted native correction to the exact null of the first96current equals inputs(rank96).

For each independently fitted native candidate, compute the required amplitude of the fixed two-coordinate programme template directly from the original source-position quadratic inequalities to restore margin8. This is one analytic arithmetic accommodation after the original-target native solve, with no subsequent native refit or target update. Rounded current/historical gaps must remain>=7.999. Select each head's least paired-scoreMSE among source-safe candidates that do not worsen its same-inputHOSTscoreMSE or exceed1.05of its original A320queryRMS. The combined candidate must improve both paired andsame-input MSE to<=0.9A259, queryRMS<=1.05A259, andpassfinite/caps. BaselineA320andA259metrics must reproduce before fitting.

No modelcall, nativeactivation,softmax,answers or iterativefitting. An eligible routing artifact would still require an independent plus-reference digit decoder and full arithmetic/native review before a separately declared compiler/fullscore. Preparation whileA321runs; reconstruction andanalysis execute sequentially afterward.


### A321 signed positional contrast rejected (2026-11-11 12:39:12 CET)

All625declared two-frequency cases fail native preservation. The exact hand kernel identity holds(to5.33e-15in the best case), andsource margins pass, but enough carrier amplitude to overwhelm the original key projections is still costly. Bestpair9/slowpair63,keymult1.25,querymult1:amplitudefloor41.78040935,actual52.22551169,sharedquerygain16.56823982. Paired/same causalMSE0.3432594175/0.3425455473 versusA2590.001424562746/0.000833213286; keyRMS0.38937563777,queryRMS0.02553836885,sourcegap7.99999875. Slow pairs from the original statistic are[65,63,66,73,67]. NoPT/checkpoint. Wall2:47.94,peak4,127,472KiB. A322's fixed A320diagnostic reconstruction is running next as sole heavy process.


### A322 fixed-key native query repair rejected (2026-11-11 12:47:50 CET)

The A320 control reconstruction reproduced every selected diagnostic value exactly (17.22 seconds, peak 3,806,924 KiB). A322 then completed in 51.07 seconds, peak 4,146,868 KiB. Each head admits a source-safe original-HOST repair, but the combined paired/same-input causal-score MSE is 0.001477125022/0.000905846035, still above A259's 0.001424562746/0.000833213286 and outside the declared 10% improvement requirement. Paired query RMS improves to 0.02463804095 and minimum source margin remains 8.000122271. The four chosen input dimensions are [128,128,128,64], all ridges 1, strengths [1,1,1,1.25]. The hypothetical routing edit count is 4,979,727 with maximum query/key/gain weight 600.2124634. No eligible routing PT, decoder, checkpoint or score was created. Budget remains 17 full, zero forward/quick.


### A323 causal-metric normalized key reconstruction declared (2026-11-11 12:47:50 CET)

Rebuild A320's two key rows from original HOST weights using a different native preservation metric: exact centered causal attention-score energy of a linear two-row key perturbation, at the fixed original HOST key RMS and original HOST queries on the recorded current word inputs. This is a single independent original-target solve for each candidate, with no residual fitting, RMS derivative, native activation or attention probability computation. Actual normalized candidate Q/K scores are evaluated afterward.

Keep A320's training data (192 word windows, 96 current and 1024 historical format windows), BOS-null constraint (relative SVD tolerance 1e-10), source positions [3,2,1,4], pairs [9,15,4,17,21], normalized key radii [0.03,0.1,0.3,1,3], format covariance masses [1,10,100,1000,10000], and query gain multipliers [1,2,4,8,16]. The non-BOS format target and exact unchanged-channel RMS construction are unchanged. Jointly solve both key rows in the BOS nullspace. Build the word Gram matrix from causal second moments minus squared causal means, with all four original HOST query heads. Verify its quadratic form against direct linear QK/RoPE arithmetic on a fixed deterministic coefficient vector for every pair. Trace-normalize the word and format covariance matrices before applying each mass; ridge is 1e-6 times mean word-Gram diagonal. Factorization reuse across the five radii changes no target or solve.

Native Q is the analytic original-HOST mean-null/inverse-gain construction; programme amplitudes follow directly from the original source margin-8 inequalities. Require paired and same-input causal-score MSE <=0.9 A259, paired query RMS <=1.05 A259, all current/historical margins >=7.999, rounded BOS raw error/original RMS <=1e-5, finite weights and edit/value caps. No A322 fitted residual or repair is reused. This is routing analysis only; any eligible artifact still requires the independent plus-reference decoder and composition review before compilation/scoring.


### A323 routing construction passes local screen (2026-11-11 12:51:24 CET)

One of 625 trials is eligible: pair 9, normalized key radius 1, format mass 100, query gain multiplier 16 (base gain 37.57578008). Paired/same-input causal-score MSE is 0.001268443132/0.000672386307, respectively 10.96%/19.30% below A259. Query RMS is 0.02572313612; key RMS 0.03237780809. Source margin is 8.000017131 (current holdback 11.97002086); rounded BOS error/original RMS is 1.29e-9. Hypothetical edit count 4,979,716; maximum query/key/gain weight 600.2124634. All five word-Gram quadratic checks agree with direct linear QK arithmetic to relative error <=1.75e-16. Wall 1:22.41, peak 4,528,752 KiB. The selected routing tensors are saved in `artifacts/a323v1_causal_metric_normalized_key.pt`; this is not yet a complete checkpoint or a ladder result.

### A324 independent plus-reference decoder declared (2026-11-11 12:51:24 CET)

Keep the A323 Q/K/gain selection and A259's actual V3/O3, MLP Gate/Down and all other weights fixed. Construct four rounded Up rows from independent per-source digit labels using the original A189 method with source positions [3,2,1,4]. Each source contributes only its own literal digit (first three heads) and one quarter of a reference constant; source-centered covariance and RHS are summed before a single closed-form solve. A single constraint fixes the sum of original native source means. No whole-problem targets, answers or previously fitted decoder residual enter.

Use the 900 original A39 prefix source windows, A119 word covariance on the first 384 windows, and original historical/current source backgrounds. The five background masses are [0,0.01,1,100,10000], word regularizer 1e-8 after trace normalization, diagonal ridge 1e-12 of source-covariance mean diagonal. Common Up gain choices are [1000,10000,100000,300000,1000000], choosing the largest that fits the weight bound. Keep V and O unchanged. Check rounded frames with algebraic ideal source routing on three historical 512-window populations and the 128 exact A259-input format population, with first 96 current backgrounds used in fitting and 32 holdback. Require positive reference >0.5, no wrong rounded digit, maximum individual digit error <0.25 current and <0.4 historical, finite and bounded weights. These are digit/source checks, not model execution.

A259 is the construction parent so all recorded layer3 inputs remain exact for that parent; A313's later layer2 native change is not silently composed. A separately declared composition review must bound the four-unit amplifier and check native Gate inactivity before any compiler/full score. Attention probabilities are never synthesized outside the meters.


### A324 plus decoder misses its original screen (2026-11-11 12:55:15 CET)

All five original background masses reject. At mass 0.01, current/historical maximum errors are 0.197637865/0.475909794; at mass 1 they are 0.285740990/0.347650953. The latter has no wrong rounded digit but exceeds the predeclared current 0.25 margin. No PT or checkpoint was saved. Wall 21.69 seconds, measured peak 1,194,176 KiB. A323's routing pass remains a component result only.

### A325 exact conditional-source-mean decoder declared (2026-11-11 12:55:15 CET)

A324's soft digit covariance fit permits systematic errors within a digit. Replace it with exact constraints on each independent source digit's conditional mean relative to that head's unconditional mean, plus the original combined-reference mean constraint. Compute these means only from the original 900 source-token windows. The first digit naturally has nine classes and the other two have ten; redundant centered constraints are removed by a single SVD (relative tolerance 1e-10), requiring target compatibility. The covariance now penalizes deviations from each digit-conditional source mean, plus the complete plus-source variance; the exact constraint targets are literal digit codes only.

Retain A324's V/O, data, original background metric, five background masses [0,0.01,1,100,10000], word regularizer 1e-8 after trace normalization, covariance ridge 1e-12 of mean conditional-noise diagonal, and five bounded Up gains [1000,10000,100000,300000,1000000]. Each candidate is one constrained solve from original source statistics, never a residual refit of A324. The unchanged rounded-frame guards are reference>0.5, no wrong rounded digit, max error<0.25 current and<0.4 historical, finite/caps. This construction still requires subsequent source/amplifier/native composition review and official scoring.


### A325 exact conditional means do not fix decoder variation (2026-11-11 12:57:44 CET)

The 30 constraints have rank 27 and compatible targets (residual 9.02e-17), but every candidate still fails the unchanged original margin screen. At background mass 1, current/historical errors are 0.288419818/0.350766480, essentially unchanged from A324. The error is not mainly bias in the fitted class means. No selected PT or checkpoint. Peak 1,197,516 KiB.

### A326 fixed A324 diagnostic reconstruction and A327 composition review declared (2026-11-11 12:57:44 CET)

Reconstruct exactly the existing A324 background-mass-1 diagnostic and assert every reported value reproduces. Its original rejection remains unchanged. This reconstruction creates a review artifact only, not an eligibility override or scored checkpoint. There are no new fit targets, constants or trials. The candidate has correct rounded digits in all tested populations, with current/historical maximum error 0.285740990/0.347650953; these are inside the literal rounding boundary but outside A324's preferred current margin.

A327 will review the fixed A323 routing and A326 frame as a complete four-unit reader on A259. Use source margin inequalities to bound off-target attention mass (at most 8*exp(-gap), with no attention probabilities synthesized), then bound attention raw/RMS uncertainty, shared Gate and four Up fields, native MLP contributions via |GELU(g)*up| <= (|g|+0.25)*|up|, and output register intervals. Inspect Gates on the unchanged factual calibration attention mixtures and report a separate universal attention-direction bound; neither factual mixtures nor that bound will be mislabeled as the changed model trajectory. No native activation or model replay.

If all current/historical source register intervals remain within 0.5 digit, all hand Gates stay positive at equals, and finite/edit/value caps hold, one explicitly exploratory full H1 score may be justified to test the unobserved new word trajectory. The original stricter A324/A325 local rejections will remain visible, and any inconclusive universal word-Gate bound will be reported before scoring. This is an independent whole-model test of a completed component construction, not relaxation of an official ladder gate.


### A326 reproduction succeeds; A327 rejects leakage bound (2026-11-11 13:04:15 CET)

A326's mass-1 decoder diagnostic is bit-exact to A324. A327's full four-unit source envelope passes current128 (max error 0.286436512), historical syntax_a (0.389578522) and syntax_stress (0.286434986), but fails syntax_c (0.554720781). The latter's raw attention RMS perturbation envelope reaches 3.7494% at source margin 8.000017; native MLP uncertainty is only 5.36e-5 of programme norm. All positive Gate lower bounds exceed 733.4. Thus the unchanged decoder rounds correctly ideally, but the margin-8 routing certificate is too weak for one population. No checkpoint or score.

The old factual word-mixture Gate maximum is -32.44949, with no tokens above -16. This is not the changed Q/K trajectory. The universal normalized-attention-direction bound certifies only 256/16,384 tokens and is inconclusive on 16,128; it is not evidence of actual word activation. A full score is required for that changed trajectory. Hypothetical edit count remains 4,979,716, maximum new tensor value 12,342.02637; peak 1,910,160 KiB.

### A328 realized-key source-margin allocation declared (2026-11-11 13:04:15 CET)

Keep exactly A323's selected two-row key matrix and A326's decoder. Replace A323's ideal-key-based query-gain allocation by a direct realized-key bound. For each head, compute the normalized two-coordinate programme direction's score gap on every current/historical source with unit query gain on the pair. Let m be the smallest such asymptotic gap. For target source margins [12,16,24,32,48], set the shared pair gain directly to max(original HOST pair gains, 1.25*target/m), round it once, inverse-compensate the original HOST mean-null native Q rows, and derive the programme amplitude from the exact target-margin quadratic inequalities. The factor 1.25 is fixed headroom, not searched. No A322 native repair or fitted residual is reused.

Evaluate the five independently derived Q constructions with the original A259 current-input/paired-HOST causal diagnostics. Require paired/same-input MSE <=0.9 A259, query RMS <=1.05 A259, actual rounded source margin >=target-0.001, finite/caps; choose least paired score MSE. The selected artifact, if any, will then receive A329: an exact repeat of A327's source/amplifier/native-envelope checks with this new Q/gain and unchanged K/V/O/decoder. No model pass or native nonlinear replay. A complete source-bound pass can authorize the already declared exploratory full H1 test; official gates remain unchanged.


### A328 margin12 and A329 composition pass (2026-11-11 13:07:35 CET)

Only target margin 12 passes A328's unchanged native screen. Its pair gain is 928.6683353; paired/same-input causal MSE is 0.001281204540/0.000684732763, query RMS 0.02599534998, actual minimum source gap 12.00002087. The fixed realized unit-gain geometry determines the gain; no new native fit occurs. Targets 16,24,32,48 fail the native preservation screen. A328 wall 27.05 seconds, peak 3,608,612 KiB.

A329 passes every source/amplifier interval: current128 error bound 0.282257733; historical syntax_a 0.307054932, syntax_stress 0.282255180, syntax_c 0.345401986. The minimum remaining rounding margin is 0.154598014. Maximum leakage bound is 4.91527e-5; maximum attention RMS relative envelope is 0.06867%. Every positive Gate lower bound exceeds 733.4. Native MLP bound/programme norm remains <=5.36e-5. The old factual word-Gate diagnostic and inconclusive universal bound are unchanged and remain explicitly separate from the new Q/K trajectory. Peak 1,913,736 KiB.

### A330 completed plus-reader checkpoint and exploratory full H1 declared (2026-11-11 13:07:35 CET)

Build A259 plus exactly A328's selected Q3, qnorm3 and K3 tensors, and A326's four Up3 rows. Keep all V/O, Gate/Down, norms other than qnorm3, and all other weights bit-identical to A259. No additional fit or scale. Validate every source/input hash, the selected A328 screen, A326's exact diagnostic reproduction, and A329's complete composition pass. Preserve the original A324/A325 strict decoder rejections in metadata and this report.

Assign disjoint edit bookkeeping labels to the two programme Q channels and their two gains/key rows versus remaining native Q accommodation; inherit A259's audited labels for other tensors. These labels are not causal attributions. The four new Up rows are arithmetic features. Expected changed count 4,979,716, total maximum weight 1e6. Save atomically with all transitive source hashes and explicit authoritative layer3 metadata.

Run exactly one exploratory full H1 through frozen tools/score.py after the build. This official run is needed because the changed query/key word trajectory is unobserved and the universal word-Gate bound is inconclusive. No official threshold is changed; no rung will be banked without the joint pass. A313 remains the verified leader pending the completed score.


### A330 build complete and full H1 active (2026-11-11 13:12:23 CET)

Atomic checkpoint `experiments/a330v1_strengthened_plus_reader/delta.pt`, SHA256 `9975f52229c5c6fca35e822bf617c00bb216d7f9182461aa9c0cf2e4a36108f9`; 4,979,716 changed scalars, max 1e6. Compiler-mask bookkeeping: 431,143 arithmetic/accommodation, 4,548,573 native-only repair. This reclassification reflects the new two-channel Q programme; it does not claim equivalent causal removal of arithmetic. Build 1:04.62, peak 784,200 KiB. A329 review took 29.11 seconds, peak 1,913,736 KiB.

Official run 0809-score started at 13:09:34 CET. Remaining budget is now 16 full, zero forward/quick. No completed score result yet. A313 remains the verified leader; no rung is passed. The score is the sole heavy process.

### A331 native-query mean restoration for the new key geometry declared; queued (2026-11-11 13:12:23 CET)

The new two-coordinate key/query design may no longer need the dense native mean-null projection inherited from the old64-channel reader. Keep A328's K and qnorm, its target margin12, and A326's decoder fixed. From original HOST Q, independently construct native query matrices Q_HOST - (1-f)*historical_format_mean_projection for restoration fractions [0,0.25,0.5,0.75,1], using the same original quiet field. Apply inverse pair-gain compensation and directly recompute the programme amplitude from the original source inequalities. No native residual fitting, new targets or scales. f=0 must reproduce the A328 selected Q and metrics exactly. At f=1 assert all254 nonprogramme Qrows per head are bit-identical to HOST, yielding a genuinely sparse alternative.

Require at least5% improvement in both paired andsame-input causal MSE against A328, paired query RMS <=1.05 A328, all source margins>=11.999, and finite/edit/value caps. Choose least paired MSE among qualifying fractions; retain all rejection diagnostics. Any eligible candidate needs a fresh full source/amplifier composition check before compilation/scoring. Prepare only while A330 runs; execute afterward as a single heavy process.


### A332 sparse native B-reader Q/K construction declared; queued (2026-11-11 13:14:14 CET)

Transfer the new normalized-key construction to layer2, whose existing source positions are already [7,6,5,4] and therefore need no decoder/reference change. Use exact A259 inputs reconstructed from A150/A140 layer1 branches, with 192 training/64 holdback word windows and 96 training/32 holdback current formats. Reconstruct original paired-HOST layer2 inputs from A140 HOST branches and require its final residual to reproduce bit-exactly. No current independent historical layer2 source population is available; report that limitation explicitly.

Keep original HOST native Q rows, with inverse query-gain compensation only on a single rotary pair. Unlike A323, do not subtract the dense format-mean projection. Thus every nonprogramme Qrow remains HOST bit-exact and the construction fits the edit budget. The shared query field is a single original-word covariance minimum-energy format detector: target1at the training equals mean, ridge1e-6mean diagonal, with positive values required on all128equals inputs. This is a format label, never an answer target.

Fit two Krows in the full training-BOS nullspace (SVD1e-10) using the exact original-HOST centered causal linear-key metric and its direct quadratic check from A323. Fixed pairs [9,15,4,17,21], normalized radii [0.03,0.1,0.3,1,3], format masses [1,10,100,1000,10000], and original covariance ridge1e-6. For each fixed key candidate, compute the realized unit-gain asymptotic source gap and allocate query gain with fixed1.25headroom for target margins [8,12,16,24,32]; derive each programme amplitude directly from the margin quadratic. Every native query is the original HOST construction, not a previous repair.

Require paired andsame-input causal MSE <=0.9of the original A259 B-reader baseline, pairedqueryRMS<=1.05baseline, allcurrent source margins>=target-0.001, roundedBOSerror/RMS<=1e-5, finite and edit/value caps. Select least paired MSE; retain all625trial outcomes. Even a local pass must receive a separate unchanged-decoder/amplifier/source bound and review of the altered downstream trajectory before any checkpoint. Preparation only during A330; run sequentially after A331.


### A333 phase-clock error decomposition declared; queued (2026-11-11 13:21:48 CET)

A315's six-unit selector failed because its continuous phase had up to0.237error. Before changing amplitudes or fitting a clock, separate that error into the inherited phase-ratio field, the leading-bit Gate ratio, and native/background transport. Use the exact A123-format pre-MLP7 and pre-MLP8 inputs reconstructed from A150 raw captures, with the unchanged A259 layer7/8 Gate/Up/Down interfaces. The A259 Q7 repair changes its actual trajectory, so this remains explicitly an earlier-input component diagnostic.

At each of the first five output positions, compare raw hand phase V/U to position+1, leading Gate/common Gate ratio to the sign-defined leading bit, and the combined hand phase to the observed layer8 phase coordinates. The labels are only fixed format position and the existing hand-built leading Gate sign, never a sum. Report max/RMS errors and worst position per component, programme Gate margins, phase output-map structural identities, and current word Gate inactivity. No weights, fit, new nonlinear activation or model call. Prepare during A330 and run after A331/A332.


### A330 full H1 completed: source improvement does not improve text (2026-11-11 13:39:19 CET)

Official0809-score is complete and ledger-confirmed. Exact and digit accuracy are24,947/25,000 (99.788%), clearing the arithmetic threshold. Text agreement is97.97210693359375%, below A259 by60matching predictions and below A313 by69outof131,072. KL0.002519413723991899, cross-entropy change-0.0005790968943450991, allfinite. H1jointpassfalse. The improved local causal-score screen didnot translate into a better complete text score.

Noise exact/digits per2,000: sigma1e-5:1993/1993;1e-4:1994/1994;1e-3:173/867. Different sparse indices realign the meter's noise draw, so cross-checkpoint noise differences are not a causal isolation of the query/key design. Two-in-a-row:1,996first,3second-given-first,3both. Wall28:07.79, peak3,853,276KiB. All4,979,716edits and weightlimit1e6 pass. No rung banked.

A313 remains the verified arithmetic-qualified text leader. There are24completed full runs (23H1plus1diagnostic),16full remaining,0forward/quick. A331 is now the sole heavy analysis; A332 and A333 remain queued. A330 and all manifested transitive sources are immutable.


### A331 native-mean restoration rejected (2026-11-11 13:41:00 CET)

The fraction0 control reproduces A328 exactly. Every nonzero mean-restoration fraction sharply worsens the native score metric: paired MSE2.114366618,8.445578123,18.98776651,33.72814709 at fractions0.25,0.5,0.75,1, against0.001281204540. Restoring the mean inflates the programme amplitudes needed to overcome the native format query norm; the first head's factor rises from0.82897 to851.06431 at full restoration. Query word RMS then reaches6.370918536. Source gaps remain12, but preservation fails by a wide margin.

At fraction1 the254nonprogramme rows per head are HOST bit-exact and the hypothetical edit count falls to4,329,487, confirming the sparse construction, but it is unusable. No selected PT/checkpoint. Peak3,598,852KiB. A332 is now the sole heavy analysis; A333 remains queued.


### A332 sparse unprojected B reader rejected (2026-11-11 13:43:21 CET)

All625trials reject. Best diagnostic: pair4,radius3,formatmass1000,targetmargin8,gain10.57834883. Paired/same causalMSE0.05532455981/0.05505288771 versus original0.001814805953/0.001550255500; queryRMS0.1334591746 versus0.03313931918. It uses4,729,007hypothetical edits with all254nonprogrammeQrows perheadHOST bit-exact, but large programme factors[700.0,1133.8,712.6,383.3]still overwhelm native behavior. Sourcegap8.0000293 and BOSerror2.66e-8 pass. NoPT/checkpoint. Wall41.68seconds,peak3,841,636KiB.

The original query field has equals response within1.75e-7of1; training/holdback wordRMS is0.0003962754/0.0004651340. This strong format field is still insufficient when native query norm on the carrier is left intact. A333phase diagnostic is now the sole heavy process.

### A334 one-coordinate native mean cancellation for B reader declared (2026-11-11 13:43:21 CET)

A331/A332 show that source-time native query energy must be suppressed. Test a sparse direct cancellation through the existing physical root-marker coordinate334. Let M be the original mean of the first96current layer2 equals inputs and e=e334/M334. Define original native query B=Q_HOST-(M@Q_HOST.T)*e.T for each head. This exactly nulls the training format mean while changing only column334, before the two programme-row gain accommodation. The full original A332 quiet query field remains the programme field; no native fit or residual target is introduced.

Repeat the same five pairs,radii,formatmasses,targetmargins and native guards from A332, independently from original HOST K/Q and original statistics. Every nonprogramme Qrow must remain HOST bit-exact outside column334; report the real mean-null error and actual sparse count. The key construction and realized-gain allocation are unchanged. Require the same10%paired/same MSE improvement,queryRMS/source/cap checks. No altered decoder, native activation, probability or model execution. Run only after A333 exits.


### A333 clock diagnostic and A334 sparse marker reader completed (2026-11-11 13:49:59 CET)

A333: inherited phase-field error max0.2219571864,RMS0.04644183926; next-layer transport max0.003777938830,RMS0.001385756953. The sign-defined leading bit is an unsuitable semantic label at the equals position: tiny positive near-zero gates are counted as one, producing a spurious near-one discrepancy. Preserve this original diagnostic and refine its label in A336 before interpreting its total error. No weights changed. Wall7.28seconds,peak1,072,264KiB.

A334 rejects all625trials. Best pair9,radius1,mass10,target32; actual sourcegap32.0001663, paired/same causalMSE0.002255856014/0.001997028749 vs0.001814805953/0.001550255500. QueryRMS0.03229490765. Sparse mean-null error1.73e-13,4,730,023hypothetical edits,peak3,875,044KiB,wall47.68seconds. NoPT/checkpoint.

### A335 covariance-selected sparse mean cancellation declared (2026-11-11 13:49:59 CET)

Independently construct native Q from original HOST and original training covariance, with sparse format-mean-null fields on supports [16,32,64,128,256]. Rank coordinates once by squared full precision-weighted mean divided by precision diagonal; each support gets its own direct minimum-word-energy field normalized to unit format mean. Fixed covariance ridge1e-6mean diagonal; no iterative residual fits. Repeat A334's five pairs,radii,formatmasses,targetmargins with unchanged source/native/cap guards, now for3125direct candidates. The two programme rows still use the original full quiet field. All other Qrows remain HOST-exact outside the chosen support. Choose least pairedMSE among those improving both paired andsame-input MSE by10%, queryRMS<=1.05baseline and all source/cap checks. No full score or checkpoint without a separate composition bound.

### A336 literal leading-register phase audit declared (2026-11-11 13:49:59 CET)

Repeat A333 without changing weights, replacing its unreliable Gate-sign label by the existing copied leading register divided by the common copied flag, threshold0.5, as in A315. Targets remain component/place identities; no sum regression. Report Gate ratio error, phase-field error, transport and total separately, plus the corrected binary-class counts. Preserve A333's original output. Run after A335.


### A337 native-phase-aligned normalized B keys declared (2026-11-11 13:52:26 CET)

Test a different key geometry: for each of pairs[9,15,4,17,21], derive the normalized pair's original non-BOS training-source mean direction from HOST K, and preserve that direction instead of forcing the pair onto the cosine axis. Radius multipliers[0.5,0.75,1,1.25,1.5] multiply that original mean radius. Query programme direction includes the post-Kgain phase of this vector. Fixed format masses[0.01,0.1,1,10,100], source margins[8,12,16,24,32], query headroom1.25, BOS-null original causal key metric andridge1e-6 are as before. Each candidate is independently solved from original HOST K and original statistics.

Use the largest declared sparse mean-null support256 from original covariance ranking, selected for its minimum training energy within the budget, with the original full quiet programme field. No A335 candidate residual is reused. Require10%paired/same causalMSE improvement over A259, queryRMS<=1.05baseline, all current source/BOS/cap checks. All625trials retained. Separate amplifier bounds required before any checkpoint. Queue after A336.


### A335 rejected, A336 clarified, A337 local pass (2026-11-11 13:57:57 CET)

A335's3125trials all reject. Best fixed-axis geometry uses256columns,pair9,radius1,mass10,margin32: paired/sameMSE0.002098427315/0.001839404758,queryRMS0.02672456056,4,989,103edits. Sparse field holdbackRMS decreases from0.0005567952at16columns to0.0004728695at256. Wall1:20.92,peak3,888,572KiB. NoPT/checkpoint.

A336 corrects the leading-bit label from A333 using the copied leading register: minimum class separation0.49144from0.5; classcounts120zero/520one. Leading Gate-ratio error max0.1235903149,RMS0.008882634323; clockfieldmax0.2219571864; transportmax0.003777938830; totalmax0.2228723609,RMS0.04729781608. This supersedes the interpretation of A333's sign-label near-one discrepancy. No weights changed.

A337 has a local pass: pair4,native-radius multiplier1.25,targetnormalizedradius1.862177600,nativepostgainphase-1.376030956,mass0.1,sourcegap8.000035802. Paired/sameMSE0.001578254075/0.001314813884 vs0.001814805953/0.001550255500; queryRMS0.02715503949. Gain1003.682393,BOSnormalizedrawerror1.60e-9,4,989,103edits,all finite. The original native-phase direction reduces key distortion enough to pass the10%screen. Peak3,865,296KiB. No full checkpoint or official score yet.

### B interface correction and A338/A339 declarations (2026-11-11 13:57:57 CET)

Direct inspection of current rounded programme Down identifies actual B digits [467,29,617] and constant plus/reference113, with shared anchor547. A314 instead bounded9*[29,113,467]/617 and mislabeled these as B digit/reference ratios. Its algebraic interval still bounds those particular ratios, but its interpretation as the canonical B interface was wrong. The independently metered A313 score is unchanged; A314 is preserved, not rewritten.

A338: bound A337 Q2/K2 with all A259 V/O/MLP and norms unchanged. Use exact128current layer2 inputs, current source positions[7,6,5,4], token-digit labels only, actual data[467,29,617] and reference113. Source gaps give scalar off-targetmass bounds8exp(-gap), never synthesized attention. Propagate convex raw-source errors, RMS intervals, shared positive Gate and Up linear fields, all-native feature envelope(|Gate|+0.25)*|Up|, and final physical B ratio intervals. Require every digit interval strictly within0.5of its own input digit, positive reference/RMS, caps. Record that independent historical layer2 and changed-word trajectories are unavailable.

A339: separately repeat A314 with the corrected canonical B coordinates/reference, preserving all originalweights/captures and its0.001ratio-displacement/1e-5rawRMSguards. This repairs evidence only, not weights. Run sequentially afterA338.


### A338 rejected; corrected A339 audit passes (2026-11-11 14:00:45 CET)

A338 margin8 leakage bound0.002683605 induces conservative B error4.48239,1.14292,3.76502digits byplace. Positive Gate lowerbound617.39; rawRMSrelative envelope7.586%; native/programme normbound5.065e-5. Reject without a fullscore. Old factual wordGate max-31.801 remains inactive; universal changed-attention direction bound is inconclusive for16,128/16,384tokens. Wall19.46seconds,peak1,159,652KiB.

A339 correctly audits the A313 B digits[467,29,617] overreference113. Canonical maximum ratio displacement7.09651e-7, perplace[5.69718e-7,5.39159e-7,7.09651e-7], reference lowerbound1010.645915, rawRMSrelative change6.16491e-8. Both original guards pass. This replaces only A314's erroneous interface interpretation; neither weights nor officialscore change. Wall18.02seconds,peak1,187,432KiB.

### A340 margin16 reconstruction and A341 combined bound declared (2026-11-11 14:00:45 CET)

A337 already tested targetmargins8,12,16,24,32. The same pair4/radiusmultiplier1.25/mass0.1 passes allfive with pairedMSE0.0015782541..0.0015794631. Reconstruct exactly the originally declared margin16candidate, requiring every selected diagnostic field bit-identical to its recorded A337trial. This is no newfit,target,scaleorsearch. Preserve A337's original margin8selection.

A341 will bound that fixed margin16Q/K on A313's compressed four-unit B amplifier. Verify all weights feeding layer2 match A259; retain exactly A313's currentunits[80,342,529,1634], mergedprogramme and native reconstruction. Use A338's corrected full B interface and the identical interval method; unchanged requireddigiterror<0.5. This explicit composition avoids assuming unchanged laterwordtrajectory. A313's verified original checkpoint stays unchanged.


### A340/A341 pass; A342 build and full H1 declared (2026-11-11 14:03:03 CET)

A340 exactly reproduces the original A337margin16trial: gain2007.3647868, actualgap16.00003856, paired/sameMSE0.001578474616/0.001315000926. No newscale orfit. A341 on A313compressedMLP2 passes: digitintervalerrors[0.1744511124,0.1377674725,0.0907203581], remainingroundingmargin0.3255488876, heldback32max0.1377674725. Leakagebound9.00247e-7,rawattentionRMSrelativeenvelope2.43526e-5,positiveGate>617.3956,native/programmebound5.06466e-5. Peak1,166,656KiB. Independenthistorical pre2 and changedwordtrajectory remainunobserved.

Build A342 from immutable A313, replacing exactly Q2,qnorm2,K2 with A340tensors. Keep all other weights bit-identical. Expected4,987,188edits,max1e6. Use explicit authoritative two-channel[4,132]Bquery metadata and the current four-unit A313Bprogramme; inherited old layer2Qmetadata is inactive. Compiler records actual-mask categories with the usual noncausal bookkeeping limitation. Full sourcehash/manifestaudit required.

Once built, run exactly one exploratory full H1 via frozen tools/score.py. All official arithmetic/text gates remain unchanged. A313 stays leader until the completedscore; no rung banked. Beforebuild folder39,551,209,201bytes,448,790,799bytes below40GB. Thischeckpoint fits; no artifacts deleted.


### A343 sparse native-phase A-reader declared; prepare while A342 builds/scores (2026-11-11 14:03:56 CET)

Independently transfer native-phase key targeting to layer3 on A259. Use exactcurrent128/A150+A140pre3 and historical1536/A39inputs, with96current+1024historical trainingformats and32current+512historical sourceholdback. The original HOST Q/K and192training/64holdbackwordwindows determine all statistics. Reconstruct pairedHOST states and require finalresidual bit-equality. Baseline Q3usesoriginalBOSreference[3,2,1,0]; newcandidates useplusreference[3,2,1,4].

Original HOST normalized keypair mean over the training non-BOS positions fixes its direction/radius. Pairs[9,15,4,17,21], radiusmultipliers[0.5,0.75,1,1.25,1.5], formatmasses[0.01,0.1,1,10,100], targetmargins[12,16,24,32,48], headroom1.25. Each key uses original centeredcausal metric/BOSnull/ridge1e-6. Covariance-selected sparse mean-null supportcounts[16,32,64,128,256] use one original precisionranking and independent direct solves; programmefield is the original fullcovariance minimum-energy equals-mean detector. No previous candidate residualtarget.

Require paired/sameMSE<=0.9A259,queryRMS<=1.05baseline, allcurrent/historical sourcegaps>=target-0.001,BOSerror<=1e-5 andcaps. Retain all3125trials. This could free denseQ3edits, but no saving/preservation claim before results. Eligible routing still needs A326decoder and completeamplifierbound, then officialscore. Preparation only; run after A342full ends as soleheavyprocess.


A342 pre-build check correction (2026-11-11 14:05:21 CET): the first attempt stopped before writing any checkpoint because stored current_programme_units is a tensor and was compared directly to a Python list. Changed only the unmetered compiler assertion to compare .tolist(); the values are the expected four units. Preserve the failed log as artifacts/a342v1_build_attempt1.log. No model or meter consumed.


### A342 checkpoint built; full H1 active (2026-11-11 14:09:55 CET)

Checkpoint experiments/a342v1_compressed_native_phase_B_reader/delta.pt SHA256f14004a3b03dd8f2e3f31a49c2b8c1eb2db82b6045993a78d0e51c63fc74bbf0. 4,987,188changedscalars,max1e6; audit492,251arithmetic/accommodation+4,494,937native. These are disjoint compiler labels, notcausal removal measurements. Build1:21.01,peak789,428KiB. All sourcehashes and A341 composition verified.

FullH1 launched14:09CET through tools/score.py, soleheavyprocess. NewCP and allmanifestedtransitives immutable. No completedresultyet. A343is prepared/unrun and queued afterscore. A313remainsleader,no rungpassed/banked.


### A344 original-HOST token-metric native layer8 repair declared; queued (2026-11-11 14:15:42 CET)

Use A150's recorded A123wordfeatures/preMLP8 and independently reconstructed pairedHOSTpost8states from A140,192training/64holdbackwindows. Candidate installationbase is A313; require its entireMLP8/norm interface equalA123, and disclose its changed upstreamtrajectory. AllG/Up/programmeDown/gains fixed. Restrict edits to a rectangular common subset alreadychanged in every nonzero-gain native Downrow, excluding current9programme units and historical11programme units. Thus no new edit support.

Construct a fresh prior from original HOST Down on that support, with exact HOST/current gain-ratio accommodation, keeping the fixed outside-support entries. Targets are original HOST normalized post8states, with original A123branch/stateRMS heldfixed before allsolves. Canonical input dimensions[8,16,32,64,128],16outputPCA directions, whiteningridge0.001. Each token's centeredHOSTtop8 embedding metric (finalnormgainincluded) is weighted by1/(HOSTwinnergap+0.1),trainingmean-normalized,capped100. Metricridges[1e-6,1e-4,0.01,1,100], strengths[0.25,0.5,0.75,1,1.25]; every solve independently targetsoriginalHOST, neveranothercandidate's residual.

Screen actual rounded linear/RMS post8states: weightedreadoutMSE<=0.9A123baseline, pairedstate/directionRMS<=1.02baseline,relativeerrorp99<=1.05baseline, rawRMSmovement<=0.05,finite/caps. Require current128literaloutputdigit/reference ratiodisplacement<=0.01 andprefillrelative statechange<=0.02; historical512native raw-delta/oldcapturedrawRMS<=1e-5 is only a proxyguard. Tokenmetricatlayer8isnotafinal-logit prediction or backpropagated metric. No nativeactivationorGemmaexecution. Prepare duringA342full,runafterA343.


### A345 refreshed original-position clock declared; queued (2026-11-11 14:18:24 CET)

Inspection of the frozen A122formatrequest confirms it also recorded7.raw and7.pre; the prior compact pointer omitted those fields. Together with A150's128currentformats, this supplies640independent-continuation preMLP7windows. A123 and A120 weights feeding these inputs must match. Use first96current+384historical training, hold32current+128historical; wordcovariance uses192of256A150windows.

Refresh only the two homogeneous positionfields using the original A43phasebasis, which is null to the copied data writer and excludes old scratch coordinates. Targets are fixed outputpositions+1, neveroperandvalues/sums. Two-field direct constrainedquadratic solves use phase-conditional covariance masses[1,10,100,1000,10000], originalwordcovariance, fixedridge1e-6meanworddiagonal, exacttrainingphase means u=1,v=position+1 andzero wordmean. No previouscandidate residual. A single deterministic common outputscale matches the original trainingmean u, preserving the old amplitude convention.

Check rounded Up5/6/7weights with unchanged shared Gates, allrecorded640clockratios and256wordUpenergy. Require positiveu>=0.1scaledunits, maximumphaseerror<=0.05 andatleast50%reductionfromoriginal, no>5%wordUpRMS increase,finite/caps. Preserve leading-bit Gate and allnativeweights. This is a component test at earliercapturedA123inputs; A259/A313changedQ7trajectory andnextattention8transport are notobserved. Any promisingfield needs separate full composition/noise review. PrepareonlyduringA342full; runafterA344.


### A346 fixed-input B-reader Gaussian-noise certificate declared; queued (2026-11-11 14:24:35 CET)

Compare A313's original B query/key maps and A342's new maps on the same128exact current pre2sourceinputs. For sigmas[1e-5,1e-4,1e-3], derive every raw Q/K linearform's Gaussian standarddeviation directly from squared inputs times squared edited weights. Query/key RMSgain perturbations use the actual stored weight, nottheeffective1+weight. A union-Chernoff bound z=sqrt(2log(2N/0.001)) covers all sourceQ/Klinearforms andeditednormparameters simultaneously withconditional probability>=0.999.

Propagate coordinate/Euclidean envelopes through RMS, RoPE andtheQKsourcegap. Report least gaplowerbound anduncertifiedsourcecounts for the actual[7,6,5,4]routing. This is a fixed-input component certificate only: all earlierweights andresidualinputs are treatedasfixed, so it cannot certify the fullmeter's noise experiment. No MonteCarlo,softmax,newnativeactivation,modelcall,weightfit orcheckpoint. RunafterA345; preparationonlywhileA342fullruns.


### A347 joint token-metric final Up repair declared; not yet coded (2026-11-11 14:35:02 CET)

This differs from A229's separate per-unit fits and A245's fixed native-feature Down fit. Use the512 A142 recorded native17feature/input windows (actual A123 capture, final Down comparison A142v2),384training/128holdback, paired original HOSTfinalstates/top8 from A56+A140+A142top. A313's final G/Up/Down/norm weights must equal the A142 comparison interface; its changed upstreamtrajectory remains a limitation.

Retain native Gates, Down and allnorms. Infer unchanged Gate factors only from recorded GLUproducts divided by linear Up intervals, with A229's sign/magnitude intervalintersection and650-term float32 dot/product bound. No GELU or derivative. Units need >=99%stable trainingobservations; unstable observations are masked in fitting and retained with intervals in validation. All programmeunits excluded.

Compute a single original-response canonical inputbasis from P/(fixed original branchRMS*fixed original finalRMS), whiteningridge0.001. Rank nativeunits once by original-target covariance with normalized native Down/gain directions (A229-type covariance ranking, no candidate residual). Nativeunitcounts[4,8,16,32,48], inputranks[4,8,16,32,64], metricridges[1e-6,1e-4,0.01,1,100], strengths[0.25,0.5,0.75,1,1.25]. Original HOST target is normalized finalstate minus normalized originalparent.

Fit all chosen Up rows jointly. Each parameter multiplies a recorded unchanged Gatefactor and a canonical inputfeature. Token-specific centered originalHOST/A123top8-union embedding metric includes finalnormgain, weighted1/(HOSTgap+0.1),trainingmean-normalized,capped100. The joint Gram includes cross-unit blocks; direct quadratic-form probe must reproduce explicit score-space squared error. Normalize each outputdirection by its norm and invert this convention in the Up coefficients. Independent closed-form normal-equation solves only; never refit a fitted residual. Largest solve3072square; stream embedding/factor computations and release unused all-unit arrays to remain under12GB.

Rounded-weight validation: >=16extra restricted then full-vocabulary HOSTmatches/8192 against the8003 comparisoncontrol; pairedstate/directionRMS <=1.05baseline; rawRMSmovement<0.05; at least8extra restricted HOSTmatches certified against the inferred-factor/Up-roundoff state intervals; finite/edit/valuecaps. Changes add at most48*640=30,720Upentries to A313's4,976,785, within5,016,268. The confidence bound is only local, not fullnetwork execution.

Validate on512A124syntax_stress recorded finalinputs/features, whose G/Up must match. Keep all2,560recorded register winners under fullvocabulary readout, nominal normalized margin>1, interval-certified raw winner margin>0, rawRMSchange<0.02. This is validation only, never an arithmetic fit. No correct-answer-prefix diagnostic captures. A promising candidate still needs a separate compiler and officialfullscore. Implement/run after the already-prepared A343..A346 queue; no A347code,trials or checkpointyet.


### A342 full H1 completed and rejected (2026-11-11 14:39:22 CET)

Official0811-score, ledgerfinish14:37:58CET: exact/digits23,546/25,000(94.184%), below99.5%. Textagreement98.01864624023438%, eightmatchingpredictions belowA313 (oneaboveA259) outof131,072. KL0.002503269135106281, CEchange-0.00024059363384942856,allfinite. JointH1false. The local13%causalscore-MSE gain didnotimprove completepreservation.

Noise exact/digits per2,000:1e-5:1884/1884;1e-4:1883/1883;1e-3:1607/1628. ChangedindicesrealignGaussian draws, so comparisons toA313areobservations,notisolatedcausalnoiseeffects. Two-in-a-row:1894first,2secondgivenfirst,2both. 4,987,188edits,max1e6. Wall28:04.64,peak3,853,556KiB.

The bounded128sourcewindows didnot establisharithmeticgeneralization. Manyof20officialarithmeticfailures have B in the200range andemitspuriousinitial1; this is diagnostic evidence of a distributiongap, not a fit target. No wholeanswerfittingorheldouttextinspection occurred. A342isnotbanked. A313 remains arithmetic-qualifiedtextleader at24961/25000 and98.02474975585938%.25completedfullruns(24H1+1diagnostic),15fullremaining,0forward/quick.

A343nowstartsas soleheavyanalysis; queued A344,A345,A346. A347jointfinalUpdesignisdeclaredbutuncoded. The nextB-reader investigation, if pursued, should firstcheck whether earlierA36native1captures permit exactcurrentpre2historical reconstruction; do not claim those inputs until upstreamweights and nativefeatureidentities are checked.


### A343–A346 completed; no new candidate (2026-11-11 14:51:47 CET)

A343 sparse native-phase A reader rejects all trials. There are665 result records:615 impossible asymptotic geometries are rejected before the support loop;50 rounded candidates cover the remaining10 geometries and5supports. Best diagnostic pair4,radiusmultiplier1.25,mass100,target12,support256 gives paired/same causalMSE0.00136270367916/0.000770349500991 versus0.00142456274575/0.000833213285609, only4.34%/7.54% improvements, below the declared10%screen. QueryRMS0.02607925236; sourcegap12.0000493; historical512gap12.0001343. It would use4,589,583edits, freeing389,117 fromA259, but edit savings do not turn the rejected preservation screen into a pass. NoPT/checkpoint. Wall56.63seconds,peak4,793,020KiB.

A344 original-HOST-prior native8 token metric rejects all125trials. The existing editable rectangle is640rows by2033nativecolumns, excluding the programme and old historical programme columns. Resetting this entire rectangle to the gain-adjusted HOSTprior discards the original dense repair. Best inputrank8,ridge1,strength0.5 has weightedreadoutMSE0.00284981984017 versus0.00264757982364baseline; pairedstate0.00681642133 versus0.006277083996; direction0.00524560363 versus0.004830401665; rawRMSmovement0.09680555415 exceeds0.05. No word pass, format candidate, PT or checkpoint. Wall32.40seconds,peak5,125,596KiB.

A345 refreshed position-clock rejects masses[1,10,100,1000,10000]. Holdback maximum ratioerrors are[0.3538356075,0.2943883494,0.2701619058,0.2529162356,0.2459039240], all above original combined0.2356149619 and declared0.05. The last two also worsen word Upenergy. Training improvement does not generalize to independent format windows. NoPT/checkpoint. Wall8.15seconds,peak1,437,756KiB.

A346 fixed-input Gaussian component bounds complete. At conditional joint probability>=0.999, A313's least sourcegap lowerbounds are[8.1519491,8.0672547,7.2132222] for sigmas[1e-5,1e-4,1e-3], certifying all512sourcechoices on the128fixed inputs. A342 yields[15.0215130,6.2002217,-83.4684237], with4uncertified choices at1e-3. These bounds exclude upstream noise and floating-point execution and are not fullmodel certificates. No weights. Wall10.54seconds,peak1,156,264KiB.

A313 remains the verified leader;25full completed,15remaining,0forward/quick. No process from A343–A346 remains active. A347 joint finalUp is declared but not coded.

### A348 historical pre2 provenance audit declared (2026-11-11 14:51:47 CET)

Inspect the retained A36 syntax_a,syntax_stress,syntax_c andcalib256 captures, recorded under A1marker. Verify exact equality of all weights upstream of layer1MLP with A313 and the native Gate/Up rows, then inspect current layer1 programme Gate margins on the recorded preinputs. No new native activation or model execution, no fit, no weights. If valid, these captures can support a separately declared hand-programme reconstruction of broader layer2 inputs. Do not call reconstructed inputs exact: programme ReLU/GELU and floating-point residual bounds must be explicit, and cross-capture numerical provenance checked first.

### A349 original-parent-prior native8 token metric declared (2026-11-11 14:51:47 CET)

Repeat A344's independent paired HOST normalized post8 target, existing native Down support, canonical ranks[8,16,32,64,128], outputrank16, whitening0.001, metricridges[1e-6,1e-4,0.01,1,100] and strengths[0.25,0.5,0.75,1,1.25]. Keep the frozen original A123/A313 Down matrix as prior instead of resetting the entire rectangle to HOST. This is a new direct fit to the original HOST-minus-parent target, not a fit to any A344 candidate or residual. Keep all original10%readout/2%state-direction/5%tail/RMS guards and source-format guards, frozen original normalization scales and unchanged programme columns. A344 remains rejected. No metered evaluation without a selected local candidate and separate compiler.


### A348 provenance succeeds; A349 parent-prior repair rejected (2026-11-11 14:59:28 CET)

A348 verifies every layer0 and pre-MLP1 weight is bit-identical between A1 and A313, and all2,043retained native1 Gate/Up rows agree. The A36 preinputs reconstruct bit-exactly from their recordedraw. Across1,536format windows, programme Gates are at least727.99756 at equals and at most-49.77609 before equals; word Gates are at most-32.00000029. Thus direct hand-programme arithmetic with recorded native features is supported. There are132 matching prefixes in syntax_stress; recordedraw1 is not bit-identical across captures (maximum absolute discrepancy0.1875, dominated by the huge carrier). This numerical difference must remain explicit. No pre2 reconstruction yet. Wall12.63seconds,peak1,124,248KiB.

A349 rejects all125 original-parent-prior trials. Best inputrank128,ridge100,strength1.25: weightedreadoutMSE0.00263460947958 vs0.00264757982364, a0.49%gain, short of10%. Pairedstate0.00630640217 anddirection0.00485044655 both worsen; rawRMSmovement0.0207923. No word/format pass, PT or checkpoint. Peak4,797,096KiB. A347 joint finalUp is now running as the sole heavy process; its explicit quadratic probe agrees with the jointGram and RHS to floating-point precision, and its comparisoncontrol is8003/8192.

### A350 broader B-source diagnostic from recorded native1 features declared (2026-11-11 14:59:28 CET)

Use A348's proven upstream/native-row identity. On each A36format window, retain every observed native1 feature, replace only the five programme products with the hand ReLUideal of current linear Gate/Up, apply current Down1 and the explicit RMS/residual/inputnorm algebra. Bound the hand product difference from scalar GELU saturation using the analytic tanh tail and conventional650-term Gate/Up dot allowances; also bound the Down dot and subsequent normalizations locally. No native GELU or attention distribution is evaluated.

Report these as captured-input component reconstructions, never exact new model trajectories. The native feature transfer fixes the earlier captured observations; cross-capture upstream rounding is outside the interval. Compare132matching historical/current prefixes and separately reconstruct all128 current cases from their actual native1 features to measure controls. Use the resulting broader1,536source centers to inspect unchanged A313 and A342 QK/RoPE source gaps; zero fitting, scales or weights. Report interval source gaps separately, including vacuous bounds. A negative A342 center gap would expose its current-only coverage problem without fitting any whole problem or answer. Run afterA347.


### A347 joint finalUp rejected; A350 exposes missing B-source coverage (2026-11-11 15:08:17 CET)

A347 rejects all625 joint native-unit/input-rank/ridge/strength combinations. Its original quadratic-form and RHS probes agree with direct score-space sums to machine precision. All2035eligible native Gate factors meet the99%stabletraining rule. Best restricted-match result:8units,inputrank8,ridge1,strength0.25 gives8005/8192 versus8003, only+2 ratherthan+16. Certifiedrestrictedmatches7969 fail the>=8011guard; readoutMSE0.00818048624 versus0.00817161190, stateRMS0.0159324373 versus0.0157828573 anddirection0.0147073238 versus0.0145631770 allworsen. NoPT/full-vocabulary follow-up/checkpoint. Wall9:54.06,peak4,218,312KiB.

A350 reconstructs current128inputs within0.000156487 maximum absolute error and within the local component envelope (worsterror/bound0.04897; stateerror/bound0.31956). A313's nominal source gaps staypositive on all1,536older windows: minima8.43205,7.76761,8.73089 forsyntax_a/stress/c. A342 has29,9,30nonpositive sourcechoices respectively, minima-224.21346,-79.26521,-75.05568. Of68failures,66are in B-hundreds class2 and2inclass1. This independently reproduces the source-coverage mechanism behind the full arithmetic failure pattern; no wholeanswer targets are used.

The historical/current matched132prefixes have maximum pre2absolute error0.000156487, but2,103coordinates exceed the local-only envelope, worsterror/bound10.4233. Thus upstream capture-transfer rounding is explicitly not certified. Conservative reconstructed source intervals also fail for some A313 cases and many A342 cases; those bounds are not claimed as full correctness. Nominal diagnostics and exactcurrent controls are reported separately. A350writes no weights or new native activations.

### A351 broader native-phase B source fit declared (2026-11-11 15:08:17 CET)

Independently repeat the original normalized native-phase key construction from HOST using96current plus1,024historical captured-input hand-reconstructed source windows for the key/format covariance and formatmean; retain32current plus512historical holdbacks. Historical centers use the A350 hand-programme/recorded-native1 algebra; their provenance limits remain. Source targets are only fixed positions[7,6,5,4]. Original word inputs/pairedHOSTtargets and192/64split are unchanged. Five pairs[9,15,4,17,21], original-native radius multipliers[0.5,0.75,1,1.25,1.5], formatmasses[0.01,0.1,1,10,100], targetmargins[8,12,16,24,32], sparse supports[16,32,64,128,256], fixed1.25headroom,1e-6covarianceridge. Direct original-target solves only, no A342residual/weights reused. Retain10%paired/same-scoreMSE guard, queryRMS<=1.05baseline, full sourcecenter margin and caps. A local pass would still need all broader unchanged-decoder composition bounds and a fullscore.

### A352 edit-efficiency review of exact A343 diagnostic declared (2026-11-11 15:08:17 CET)

A343 remains rejected under its10%preservation-gain criterion. Reconstruct its original best diagnostic as a separately labeled budget tradeoff: require bothpaired/same causalscoreMSE no worse than A259, queryRMS<=1.05baseline, at least300,000fewer edits, andalloriginalsource/BOS/valueguards. Repeat the exact original grid only to reproduce weights; the selected record excludingeligibility must be bit-identical to A343's alreadyreported bestdiagnostic. No new scales,target,fitresidual or claimed10%gain. If reproduced, separately compose A326's existing plus-reference decoder and bound everycurrent/historical source before considering any full checkpoint. The intended benefit is freeingnative-repair capacity, not claiming an H1or textpreservation pass.


### A352 exact edit-efficiency reconstruction completed (2026-11-11 15:10:31 CET)

The selected numerical record is bit-identical to A343's original bestdiagnostic after restoring its original eligible=false field. Under the separate budgettradeoff guard, it saves389,117edits (4,589,583 versus4,978,700), improves paired/samecausalMSE4.34%/7.54%, and retains sourcegap12.0000493. It doesnotmeet A343's10%criterion. Pair4,normalizedradius0.7846034345,mass100,support256,querygain297.3152103 are unchanged. The PT is an analysis artifact, not a modelcheckpoint. Peak4,360,908KiB. A351 broader B-key fit is now the sole heavy process.

### A353 complete compact A-reader composition bound declared (2026-11-11 15:10:31 CET)

Use exactly A352Q3/K3/qnorm and A326's fixed plus-reference decoder on A259. Preserve A343's original10%rejection and A324's strict0.25current-digit screen rejection. Repeat A329's full source/RMS/hand-programme/native absolute-feature bounds on128exactcurrent plus1,536historical inputs. Sourcepositions[3,2,1,4], fixed Gate/Down/gains, no new fitting, scale, nativeactivation or attentiondistribution. Require all digitinterval errors<0.5, positive programmeGate, finite/caps. Retain factual-old-mixture and universal wordGate diagnostics separately from actual changed trajectories.

Use A259 as the prospective checkpointparent, because its inputs through layer2 exactly match these captures. Do not silently compose A313's changed native2 map. A successful local bound would authorize one explicitly exploratory full H1/text/noise score of this smaller circuit; there is no rungpass or textgain claim yet.


### A351 rejected; A353 compact reader composition passes locally (2026-11-11 15:17:14 CET)

A351's broader source covariance removes the earlier sampling advantage. No eligible result among925records. Bestdiagnostic pair17,radiusmultiplier1.25,mass100,target8,support256 gives paired/sameMSE0.002007080434/0.001741336501 versus0.001814805953/0.001550255500: both worse. Allsourcecenter gaps>=8.0000213, but native preservation fails. NoPT/checkpoint. Wall2:13.75,peak4,368,508KiB. A352exactreconstruction wall54.60seconds.

A353 passes the separately declared composition bound with A352+A326 on A259. Maximumdigitinterval errors current/a/stress/c are0.2828762816,0.3070549546,0.2828736964,0.3454019858; minimumremainingroundingmargin0.1545980142. ProgrammeGate>=733.40968. Maximumattentionleakbound4.91513e-5; native/programmenormbound<=5.35351e-5. FourUpchanges add no new editedindices, so the completecandidate remains4,589,583edits. FactualoldwordGatemax-32.44949; the universalattention bound is inconclusive on16,128/16,384tokens. Changedwordtrajectory remains unobserved. Wall29.32seconds,peak1,908,344KiB.

### A354 compact A-reader full-score construction declared (2026-11-11 15:17:14 CET)

Install exactly A352Q3/K3/qnorm and A326's fourplus-referenceUp rows on A259, no changes elsewhere. Preserve both the originalA34310%screen rejection and A324strictdecoder-screen rejection in itsmanifest. A353's complete recorded-source intervals justify one exploratory full H1/text/noise evaluation, with text preservation and full arithmetic still unknown. Purpose: determine whether389,117freed edits can support further original-HOST native reconstruction while retaining the existing circuit. No A313composition or extra fit. Expected4,589,583edits,max1e6. About400.5MBfolderheadroom measured beforebuild; no artifact retirement needed.


### A354 built; full H1 score running (2026-11-11 15:21:58 CET)

Checkpoint experiments/a354v1_edit_efficient_plus_reader/delta.pt SHA25620daec4dbeb0d87870270c7cbb1e6c0ea2f7b624369d3b8c54b580a25489f399 has4,589,583edits,max1e6. Exact disjoint audit:431,143arithmetic/accommodation and4,158,440native edits. The389,117freed slots are actual differences fromA259, not category relabeling. Build1:09.47,peak802,200KiB. All A354construction and manifested transitive sources, includingA352/A343/A353, are now immutable.

One full H1/text/noise score is running to scores/a354v1_full_H1.json with logartifacts/a354v1_full_H1.log. The run reserves one fullunit, leaving14; no result or rungpass is claimed. A313 remains the completed arithmetic-qualified textleader.

### A356 recorded-linear-kernel audit declared; prepare during A354 (2026-11-11 15:21:58 CET)

Before another larger nativeUp construction, test whether exact float32 linear operations can be reproduced on recorded final-MLP inputs using each original forward request's batchshape. This is arithmetic on recorded activations, not Gemma or nativeGELU execution. A142word512 batch8 has recorded rawpre17, programme Gate outputs, features and Down outputs; A124stress512 batch16 has actual pre17 and Down outputs. Recompute pre17 by the fixed RMS algebra and all2048 Gate/Up or640Down linear outputs in the original shapes. Require bit-exact recordedpre where available, programmeGate controls and complete Down controls. Gate/Up weights and inputnorm source identities must agree with capturedcheckpoint.

Native Up outputs are not directly captured, so any tighter factor inference remains conditional on deterministic identity of the same float32 linear operation, input values, weights and shapes. If the controls fail, do not replace the earlier650-term bounds with an empirical error estimate. If they pass, infer the unchanged native Gate factors from recorded product / reconstructed float32 Up with multiplication-roundoff intervals; record exactzero-Up cases and factor-width statistics. This computes no native activation. No weights, fit, meter or checkpoint. Run only after A354full exits.

### A355 expanded joint nativeUp construction declared; conditional on A356 (2026-11-11 15:21:58 CET)

Use A354's freed capacity for a new, larger joint nativeUp basis at the unchanged final17interface. The original A142 word512/A56+A140 HOSTtargets remain the only calibration targets, split384/128; all sourceandtrajectory limitations remain. Unitcounts[64,128,256,384,512], inputranks[2,4,8,12,16], ridges[1e-6,1e-4,0.01,1,100], strengths[0.25,0.5,0.75,1,1.25]. Rank nativeunits once from the original normalized HOST-minus-parent response with the same0.001canonical input whitening; no A347 fittedcandidate residual. Only native Up rows change;512rows add at most327,680edits to4,589,583, below5,016,268.

Unlike A347's token-specific metric, form a single positive-semidefinite metric by averaging original centered HOST/A123top8-union embedding outerproducts (finalnormgain included), weighted1/(HOSTgap+0.1), training-mean-normalized,capped100. The direct normal equations also use those fixed per-token weights. The constant output metric allows the full cross-unit Gram to be assembled from one joint feature covariance, multiplied by the fixed output-direction Gram. This admits hundreds of unchanged native Gates within the memory cap. Verify Gram/RHS by direct quadratic probes. Original fixed branch/final RMS, one solve per ridge, no iterative refits.

If A356 passes its exactlinear controls, infer factors from recordedproducts and matched-shape float32Up, retaining multiplication intervals and exactzero-Up uncertainty. Evaluate rounded proposedUp using the same shape-preserving linear arithmetic. This conditional precision improvement must remain separate from a full-model/float32certificate. Keep >=16extra restricted/fullvocabulary HOSTmatches, >=8extra interval-certified restrictedmatches, state/direction<=1.05baseline andrawRMSmovement<0.05; full sourceformat winners2560/2560, nominalmargin>1, certifiedrawmargin>0, formatRMSchange<0.02. No nativeGELU/derivative. If A356fails, stop this declared conditional version without silently weakening its guards.

Largest joint matrix8192square; no per-token512square output tensor or whole vocabulary-logit cache. Stream captures and embedding blocks, reuse one full Gram, release large training arrays before eigensolves/validation, target<10GBpeak and enforce the12GBlimit. A355is not coded or run yet; A356is not coded or run yet. They queue after A354full, with oneheavyprocess only.


Preparation update (2026-11-11 15:40:54 CET): A355 andA356 are coded and syntaxchecked, unrun whileA354full occupies the soleheavy slot. A355 uses a streamed constantmetric/jointfeatureGram, releases the trainingmatrix before eigensolves, and caches fixed validationembeddingdifferences to avoid repeated large gathers. It checks the8192square allocation estimate against11.5GB before allocating, and the unchanged-Up inverse-factor control must reproduce8003/8192matches andrawrelativeerror<1e-6. The original audit conditions and allscreeningguards remain.

A357 conditionalcompiler is also prepared/syntaxchecked, unrun. It refuses missingA355selection, failedA356controls, insufficientfull-vocabulary gain or failedformatintervals. It would install only the selected native17Up rows on A354, preserving all11programmeUp rows andeveryothertensor bit-exactly, with fulltransitiveprovenance and a disjointedit audit. No A355/A356/A357result or checkpoint exists yet.

A354's completed arithmetic portion is24,946/25,000exactanddigits(99.784%); text,repeated-problem andnoiseportions remainrunning. This is not a completed fullscore or rungpass. Budget14fullremaining,0forward/quick. A313 remains the completed-scoreleader.


### A354 full completed: edit savings retained, no preservation gain (2026-11-11 15:49:37 CET)

Official0813-score completed at15:48:26CET and is ledger-confirmed. Exact/digits24,946/25,000(99.784%) clears H1arithmetic. Textagreement97.9400634765625% is111matchingpredictions belowA313 and102belowA259 outof131,072. KL0.002403604905456083 andCEchange-0.0007197224676662728,allfinite, but the mandatory98.5%agreement gate fails. JointH1false. The smaller circuit is not banked or promoted toleader.

Noise exact/digits per2,000:1e-5:1993/1993;1e-4:1580/1994;1e-3:1441/1441. Changed sparseindices change the meter's noise alignment; no isolatedcausal claim follows from cross-checkpoint differences. Two-in-a-row:1995first,0secondgivenfirst,0both. Wall28:15.79,peak3,853,788KiB.4,589,583edits,max1e6 verified.

A354remains an explicit smaller-budget experimental parent for the alreadydeclared A355construction; its lower textagreement is not hidden. A313 remains the strongest completed arithmetic-qualified result:24,961/25,000and98.02474975585938%text.26fullcompleted(25H1plus1H5diagnostic),14remaining,0forward/quick. A356linear audit nowruns as soleheavyprocess; A355andA357are prepared,conditionalandunrun. No rung,best,finalverificationorexport exists.

Final package helper experiments/package_run3_final_v1.py is preparedandparsedonly. It requires a final-reportheading and verification/results.json tied to the finalledgerhead, excludeshostweights/activationcaptures, hashesmembers andtests thearchive, and refuses>300MB. The1,074script/manifest files currently total21,702,440uncompressedbytes. If no rung passes, an explicitlylabeled unpassed sparsecandidate+officialscore may be included undercandidate/, neverbest/H1. No archive has been created and no final verifier has run.


### A356 matched linear controls all pass; A355 starting (2026-11-11 15:50:16 CET)

A356's available controls are bit-exact on all512word and512format windows: word11programme Gate projections andall640Downoutputs; formatpre17normalization andallDownoutputs. Original batchshapes8x64and16x14are preserved. No nativeUp output was directlycaptured; the deterministic-linear identity assumption therefore remains explicit. No native activation or model was executed.

All2,037nativeunits have nonzero reconstructedUp on everyword observation; format alsohaszero exactzero-Upcases. Product/sign/magnitude intervals havezero inconsistencies. Wordfactor radiusmax6.47781739e-7,RMS2.68719591e-8; formatmax5.79367541e-6,RMS3.02849465e-7. Maximumrelative radius1.19209304e-7 for nontrivialfactors. Wall11.43seconds,peak2,555,664KiB.

A355 nowruns as soleheavyanalysis using the predeclared512unit/8192parameter maximum, originaltarget/canonicalbasis/globalmetric grids, actualrounded Up projections andconditional productintervals. No A355result/checkpoint or newmeter yet. A357compiler stays conditional. Budget14fullremaining; A313stillcompleted-scoreleader.


### A358 full-input separable moment construction declared; prepare during A355 (2026-11-11 15:54:52 CET)

A355 confines each new Up row to atmost16canonical inputdirections. A separate original-target construction will allow all640physical inputdirections while retaining the same nativeunitcounts[64,128,256,384,512], originalA142/HOSTtarget, fixednormalizers, A356conditionalnativefactors andA354parent. It will not read or refit any A355candidate weights/residual.

Use the same original-response nativeunit ranking (0.001inputcovarianceridge) and the same original weighted mean embeddingmetric. Let X=P/(fixedbranchRMS*fixedfinalRMS), phi be unchanged recordednative Gatefactors, D be normalized fixed nativeDown/gain directions, and y the original normalized HOST-minus-parent state. With fixed originalgapweights, compute Cx=Ew[X X.T], Cg=Ew[phi phi.T] elementwise-multiplied by D.T M D, and the exact originalresponse RHS R[i,k]=Ew[phi_i X_k (D_i.T M y)].

Approximate the joint featureGram by Cg tensor-product Cx. This separability is an explicit approximation, not an identity or an exact least-squares optimum. Standardize each factor's diagonal; eigendecompose the two factors once perunitcount and solve each fixed ridge[1e-6,1e-4,0.01,1,100] by eigenvalue products. Strengths[0.25,0.5,0.75,1,1.25]. Report an original-data quadratic probe against the approximateGram and its discrepancy; do not label it an exact check. No gradients, lossiteration, numerical derivatives or candidate-residual fitting.

Every rounded physicalUp candidate must satisfy the unchanged A355restricted/certified/full-vocabulary match gains, state/direction/RMS limits, finite/edit/valuecaps and2560formatwinner/margin/interval guards. Nativefactor/control assumptions stay explicit. Largest linear-system factors are512and640square, avoiding an unallocatable327,680parameter exactGram. This is125independent candidates, not an extra input-rank search. Only a selected, fullyscreened result could receive a newcompiler and fullscore. A358is notcoded/run yet; A355remains soleheavyprocess.


### A355 expanded joint nativeUp rejected; A358 running (2026-11-11 15:57:22 CET)

A355 completes all625candidates with no eligible repair. Bestrestrictedmatch record:64units,inputrank4,ridge0.01,strength1 gives8006/8192 versus8003; all8006are conditionallyinterval-certified, but only+3ratherthan+16. StateRMS0.01732015475 anddirection0.01571359364 bothworsen (baseline0.01578285731/0.01456317699); readoutMSE0.00867582402 versus0.00817161190. RawRMSchange0.02513579, conditionalfactor/productuncertainty1.83301e-7. Thus insufficient fit quality, not the old factor uncertainty, blocks this family. Larger counts through512andinputranks through16 yieldnoacceptedcandidate.

The unchanged-Up factor control reproduces8003matches andrawrelativeRMSerror2.02534e-16; its conditionalbound5.60705e-7 is retained. The direct jointGram/RHS probes pass. All2,037nativeunits have fullynonzero training Up underA356's conditionalprojection. Peak8,380,856KiB(8.58GB), within12GB; wall5:38.17. NoPT/full-vocabulary candidate/checkpoint orscore; A357compiler remains unrun.

A358 is coded/syntaxchecked andnow runs as soleheavyanalysis. It independently usesall640physicalinputdirections with the declared separable-moment approximation, fiveunitcounts andthe samefive ridge/strength values,125candidates. Its approximateGram discrepancy is reportedratherthan assertedaway. Alloriginalword/format/fullscoreguardsremain. Budget14fullremaining; A313stillleader; no rungpass or finalverification/export.


### A358 full-input separable repair rejected (2026-11-11 16:08:27 CET)

All125 candidates fail the unchanged preservation guards. The best record is64native units,full640input directions,ridge100,strength0.25:8003/8192restricted and conditionally certified matches, identical tobaseline; readoutMSE0.00817442740444 versus0.00817161190336,pairedstateRMS0.01584895177 versus0.01578285731,direction0.01462677978 versus0.01456317699. RawRMSmovement0.00195478253; conditional factor/product statebound1.83651e-7. The approximate Gram quadratic probe differs from the exact original-data energy by7.423774%; the RHS agrees to floating-point precision. No selected result,full-vocabulary follow-up,PT,checkpoint or meter. Wall1:00.50,peak5,492,828KiB. A357's A355-dependent compiler remains unrun.

Budget14fullremaining,0forward/quick. A313 remains the arithmetic-qualified completed-scoreleader; no rungpassed.

### A359 activation-aware original-target allocation declared (2026-11-11 16:08:27 CET)

A separate closed-form construction will allocate the original normalized HOST-minus-A142parent state target among the currently active, unchanged native17 Gate factors, before fitting physical Up rows. Use A354's unchanged final interface and freed editcapacity, original512word windows split384/128, A356matched-linear conditionalfactor intervals, and the same once-only original-response unit ranking. Do not reuse any earlier candidate or fitted residual. Counts[64,128,256,384,512]; fixed outputrank16 and full640physical inputdirections.

Take the top16 right singular directions of the original normalized response target, with the existing global original-readout metric applied through its positive square root. In this fixed reduced outputspace, let D contain normalized nativeDown/postgain directions, phi the inferred unchanged factors, and B_n=D*phi_n. For each original training token, allocate desired normalized Up response a_n=B_n.T*(B_n*B_n.T+lambda*I)^-1*y_n. Normalize B's common scale by the training mean trace/16 before the five oracle ridge values[1e-4,0.001,0.01,0.1,1]. This is an independent small linear inverse of the original target, not a candidate residual update.

For each selected nativeunit, fit the single full-input linear Up correction from X=P/(original branchRMS*original finalRMS) to its allocated a_n, using squared fixed phi as weights and original HOSTgap weights. Standardize Gram diagonals and solve once per ridge[1e-6,1e-4,0.01,1,100]. Strengths[0.25,0.5,0.75,1,1.25]. Report original-target oracle reconstruction and fitted training/holdback diagnostics separately. All625 candidates receive the original A355 local word,interval,state,direction,RMS,edit/value and format/full-vocabulary gates; no weakened guard. Nativefactor inference stays conditional; older recorded final inputs do not certify the A354 upstream trajectory. No nativeactivation,gradient,finite difference,iteration,wholeproblem target,model execution or meter.

Cache at most512640square eigensystems (about1.68GB), stream the per-token16square systems, and do not hold all counts' allocations at once. Keep one heavyprocess, peak<12GB. This version is declared but not yet coded or run.


A359 preparation update (2026-11-11 16:10:46 CET): the activation-aware allocator is coded, syntaxchecked and running as the soleheavyprocess. It caches512 fixed640square regressors, solves original-target16square allocation systems, records separate original-target oracle and fixed-normalization fitted-training errors, and screens625 rounded candidates on the128wordwindow holdback. All A355word/full-vocabulary/format guards remain. No result,PT,checkpoint or meter yet.


### A360 early recorded-linear identity audit declared (2026-11-11 16:13:40 CET)

Extend A356's conditional arithmetic audit to original A123 layers1..8 using A150 native-feature/raw captures and A140 matched raw-branch/programme-Gate outputs. Require identical explicit input tokens, checkpoint and original batchshape8x64 across both captured requests. Reconstruct only the fixed RMS input and linear Gate/Up/Down operations. Wordinputs are fullyrecorded; format captures contain only14positions of64, so fill omitted rows withzero solely to retain the original independent-row linear kernel shape, then inspect only the14recorded positions. Require all recorded programmeGate and allDown outputs bit-exact. No preorUp output was directlycaptured, so deterministic-input/operator identity remains conditional and is not a whole-network certificate.

Verify unchanged native Gate/Up and pre-MLPnorm weights against A354, keeping any changedprogramme Up separate. Report zero-Up cases and product/sign/magnitude intervalconsistency/width for each layer and population. No fitting,newnativeactivation,model,weight ormeter. Run sequentially after A359. A passing audit could justify a separately declared larger original-target early-native Up construction within A354's sparecapacity; it doesnot itself authorize a checkpoint.


### A361 expanded joint early-native Up repair declared, conditional on A360 (2026-11-11 16:15:30 CET)

Independently test layers1..8, in order[7,6,3,5,4,1,8,2], using A354 as the installation parent and original A150/A140 A123 wordinputs (192train/64holdback) with paired original HOST post-MLP states reconstructed from A140 HOST branches. Earlier A354 trajectory changes remain unobserved; this is a component construction only. Require A360 exact available linearcontrols and consistent conditionalfactor intervals before running. Only native Up rows of one layer may change; no cross-layer accumulation.

Target y=(original HOSTpoststate-original parentpoststate)/original HOSTstateRMS, with X=recordedpre/(original branchRMS*original HOSTstateRMS). Rank nativeunits once by original-response covariance with inputwhitening0.001; usecounts[64,128,256,384,512], canonicalinputranks[2,4,8,12,16], ridges[1e-6,1e-4,0.01,1,100], strengths[0.25,0.5,0.75,1,1.25]. Build the exact joint nativeunit/input Gram using normalized fixed Down/postgain directions and identity outputmetric; allcross-unit blocks are retained. One original-target solve perridge, no candidate residuals or repeated fitting. Verify Gram and RHS by explicit original-data quadratic probes. Largest matrix8192square; bound allocation before construction, oneheavyprocess, peak<12GB.

Evaluate rounded physicalUp using matched original float32 linear shapes and recorded nativefactor intervals. Require at least5%holdback paired-state anddirection RMS improvements,99thpercentile stateerror<=1.05baseline, maximumrawRMSmovement<0.05, and stateimprovement morethan twice the conditionalfactor/product stateuncertainty. Enforce edit/value/finitecaps. The best eligible wordcandidate for eachlayer must also preserve capturedformatcomponent outputs: maximumpost-coordinate change plus conditionalbound<1 forlayers1..3 and<100 forlayers4..8, andmaximumrawRMSmovement<0.005. This uses unchanged capturedprogrammeproducts only where they are weight-identical; any changedprogrammeUp (A354layer3) must be reconstructed by its handReLUideal with an explicit native/GELUtail allowance, and its changed-source trajectory caveat retained. A selectedlocalcandidate still needs separate source/semantic compositionreview and compiler/fullscore authorization. No arithmeticanswers, nativeactivation,modelorgradients are evaluated.

A360is codedandparsed, queuedafterA359. A361is declaredbutnotcoded/run. No checkpoint or meter.


### A359 activation-aware allocation rejected; A360 running (2026-11-11 16:17:12 CET)

All625candidates fail the unchanged guards. Bestdiagnostic256units,oracle ridge0.1,Up ridge100,strength0.25 yields8004/8192restricted andconditionallycertified matches versus8003, only+1. Pairedstate0.01578365491 versus0.01578285731,direction0.01456508379 versus0.01456317699,andreadoutMSE0.008172460887 versus0.008171611903 allworsen. RawRMSmovement0.00081186934; factor/productbound3.87739e-7. TrainingreducedMSE0.00034565346 versus0.00034982130. No candidate reachesfull-vocabulary orformat stages; no PT/checkpoint/meter.

The512-unit original-target oracle has reducedtrainingMSE[5.28630e-10,4.96330e-8,3.19573e-6,5.20304e-5,0.000182254898] at allocationridges[1e-4,0.001,0.01,0.1,1], while its directlychecked firstlinearsystem hasrelative residual1.04e-11. This is oraclegeometry, not fittedweight success. The unchanged-weight inversefactorcontrol reproduces8003matches,rawerror2.02534e-16. Wall5:31.63,peak7,277,764KiB(7.45GB), below12GB.

A360earlylinear audit is now the soleheavyprocess. A361 is declared and being prepared, conditional onitscontrols. Budget14fullremaining,0forward/quick; A313stillleader; no rungpassed.


### A360 all early controls pass; A361 layer7 begins (2026-11-11 16:20:09 CET)

All16layer/population records pass: programmeGate andallDown controls arebit-exact forlayers1..8 on256word and128format windows, using original8x64kernel shapes. Allproduct/sign/magnitude intervals areconsistent. Layer5word has2exactzero-Up observations, bothwithzeroobservedproduct; their fullsign/magnitude uncertainty isretained. Everyotherpopulation haszero exactzero-Upcases. NativeUp andpre outputs remainunobserved; thedeterministic-linear inference assumption andoldtrajectory limits remainexplicit. Wall27.50seconds,peak2,930,376KiB. No weights.

A361is codedandsyntaxchecked. The firstindependentlayer7construction nowruns as soleheavyprocess. NativeGate factors retain exactzero-Up intervals; directGram/RHS probes andunchanged-Up controls mustpass. Aselectedwordcandidate receivesboundedformat checking; changedprogrammeUp,ifpresentinlayer3, usesonlyhandReLUideal withanalytic tanh-tail/scalarroundoff allowance. No fullscore isreserved.


Finalpackage preparation (2026-11-11 16:21:23 CET): version2 helper nowalso includes small analysisJSONreports, officialaggregate scoreJSONs, and experimentJSONmetadata/requests, so numerical trialevidence accompanies the scripts/manifests. CurrentanalysisJSONs total53.24MB, experimentJSONs33.01MB and scoreJSONs0.72MB beforecompression; finalhelper stillenforces290MBuncompressed safety allowance and300MBarchive cap, excludes allactivationPTs/hostweights, andrequires finalreport/finalverification/finalledgerhead. Neitherhelper hasrun; noexport orverification exists.


### A361 layer7 rejects; layer6 running (2026-11-11 16:26:57 CET)

Layer7completes625candidates. Best256units,inputrank16,ridge100,strength1.25 reducespairedstateRMS0.007071553599 to0.006953205701(1.6736%,below5%). Direction0.005261030917 to0.005208890598(0.9911%,below5%).99thpercentile0.02298197975 vs0.02304710958,rawRMSmovement0.00412221,conditionalfactor/productbound2.36609e-7. Noeligiblewordcandidate,formatstage,PT,checkpoint ormeter. DirectGram/RHS andunchanged-Up controls pass. Wall3:25.05,peak6,802,116KiB. Layer6 nowruns independently as soleheavyprocess; remainingdeclaredlayersfollowsequentially.

### A362 coupled native feature and output construction declared (2026-11-11 16:26:57 CET)

A355/A358/A359 kept nativeDown directions fixed. A separate one-way closed-form construction will change native17Up only inside the existing512donor pool, then solve nativeDown once against the original paired HOST normalized final-state target. A354parent, originalA142word512 split384/128, A356conditional factors; no earlier candidate or residual is read. AllprogrammeG/Up/Down andpostgains remainfixed. Upcounts[16,32,64,128,256], fiveUp perturbationfractions[0,0.1,0.25,0.5,1], fiveDownridges[1e-6,1e-4,0.01,1,100], fiveDowncorrectionstrengths[0.25,0.5,0.75,1,1.25]. The zero-Up fraction is the declared unchanged-feature control, not an extra sixth value.

Build one originalresponse PCA16 in the same fixed original global readoutmetric asA355. For each original donor Gate, regress its phi-weighted full640input against that original reducedresponse, using fixed0.001covariancewhitening; select its largest canonicalcorrelation inputdirection. Rankunits once by that original statistic, never by a candidate score. Orient each field to nonnegative correlation with its original observednativefeature and scale its phi-weighted physicalinput RMS to that feature's original RMS. Add the declared fraction ofthisfield to the original nativeUp. These feature directions are fixed before all Down fits.

For each fixed featuremap, use original frozen branch/final RMS and original gapweights. The immutable target is normalized original HOSTstate minus original pre17 dividedbyoriginal parentfinalRMS. Solve the nativeDown/postgain coefficients with original A354Down as ridgeprior: all survivingnativefeatures for existing96strongrows, original512donors for the other544rows. The linear RHS subtracts the fixed original-prior prediction in this declared featuremap, as required by ridge-with-prior algebra; no fitted Down output or residual is fed to another solve, and no Up direction is revised. This is a one-way basis construction followed by a single original-target linear output solve, not alternating fitting.

Use matchedfloat32 proposedUp arithmetic and productintervals; evaluate each roundedDown directly on the resulting recorded-input featurecenters. Retain all A355 localword,conditionalcertification,state/direction/RMS,+16full-vocabulary,2560formatwinner/>1nominal/>0conditionalmargin/2%formatRMS,edit/valuefiniteguards. Unchanged-feature+unchangedDown control must reproduce8003/8192. AllnativeDown edits remain within existing rectangularsupports exceptpossible previouslybit-equal entries, whose actual total iscounted. Up256adds atmost163,840indices, within A354's426,685sparecapacity. Source trajectories remainunobserved afterearlier A354edits. No nativeGELU/attention/model,gradients,answers,iterativefit ornewmeter. Memory streamedbelow12GB,oneheavyprocess. A362 isdeclaredbutnotcoded/run; queuesafter the currentlyrunning A361 analyses unless a selectedearlycandidate needsreview.


A361layer6completed (2026-11-11 16:30:24 CET):625candidates,noeligiblewordrepair. Best512units,rank12,ridge1.0,strength0.5:pairedstateRMS0.00862808805127 to0.00854186888132,gain0.999285%below5%;direction0.00700897827499 to0.00695614016248. Noformatstage,PT,checkpointormeter. Peak6,837,460KiB. Layer3nowrunsindependently;A362preparationcontinues.


A361layer3completed (2026-11-11 16:34:07 CET):625candidates,noeligiblewordrepair. Best512units,rank16,ridge1.0,strength0.75:pairedstateRMS0.00706855515664 to0.00686945500819,gain2.816702%below5%;direction0.00563680732058 to0.00553751980774. Noformatstage,PT,checkpointormeter. Peak6,741,124KiB. Layer5 isnext;A362coupledconstructioniscompleteandsyntaxchecked,unrun. Its original-prior normal equations receive directquadratic/RHS probes beforecandidateevaluation.


Preparation/integrity update (2026-11-11 16:36:41 CET): A362coupled-native feature/output constructor is complete andsyntaxchecked,unrun. It preserves the original8x64word and16x14format linearshapes, checks allactualUp/Downindices, and includes directoriginal-prior Gram/RHS probes. Purestdlib read-only audit confirms allfrozen meter-source hashes andall815ledgerlinks; head9259642b1ef176fb81836cf5b78048504895e549c473bb5990cc1d2cc36d4a39. Nounfinishedmeterrun;14fullremaining. This isnot finalverification;tools/verify.pyhasnotrun. A361layer5 remainssoleheavyprocess.


A361layer5completed (2026-11-11 16:38:36 CET):625candidates,noeligiblewordrepair. Best512units,rank16,ridge1.0,strength0.5:pairedstateRMS0.00636644636395 to0.00627485627429,gain1.438638%below5%;direction0.00517362313415 to0.00511702081185. Noformatstage,PT,checkpointormeter. Peak6,820,768KiB. Layer4nowrunsindependently;layers1,8,2remain beforeA362.


### A363 first compact-reader H2 diagnostic declared (2026-11-11 16:41:27 CET)

After the currentlydeclared A361independentlayers and A362coupledconstruction, if neither supplies aneligiblecandidate requiringpriorityreview/H1scoring, run onefull H2diagnostic on the alreadyimmutable A354delta (SHA20daec4dbeb0d87870270c7cbb1e6c0ea2f7b624369d3b8c54b580a25489f399). It replaces the A-reader'sBOSreference with theplusposition, so a prefixed-addition test addresses a concrete unmeasuredmechanism. H2uses relative sourceoffsets but the formatdetector,phaseclock,and carriedstate maystillfail undercontext. Repeated-problem failures donot measure the same condition because a prior arithmeticcarrier ispresent there.

Useonly frozen tools/score.py --delta experiments/a354v1_edit_efficient_plus_reader/delta.pt --rungs H2 --output scores/a363v1_a354_full_H2.json. No heldouttext inspection, newweights, refit or scalechoice. Onefullunitwouldleave13; noneisreservedyet. A354's known97.9401%textagreement fails98.5%, so this is a diagnostic aboutprefixedarithmetic, not a claimthat H1haspassed or anexpectation ofbankingH2. Reportallscore aggregates and atmostthefrozen meter's20 arithmeticfailureexamples, retaining source/clock uncertainty. Aselectednewrepairtakespriority; A363mayremainqueueduntilitsH1reviewcompletes. Twofullunitsremainreservedforoptionalhost/finalcandidateGSM8K, plusfreshverificationofanyeventuallybankedrung.


A361layer4completed (2026-11-11 16:42:49 CET):625candidates,noeligiblewordrepair. Best512units,rank16,ridge1.0,strength0.5:pairedstateRMS0.00841458118578 to0.00828920923477,gain1.489937%below5%;direction0.00718496872535 to0.00707196194506. Noformatstage,PT,checkpointormeter. Peak6,811,796KiB. Layer1isnext,then8and2.


### A364 recorded-prefix key competition diagnostic declared (2026-11-11 16:44:15 CET)

Before interpreting A363, inspect only linear/RMS/QK/RoPE routing in synthetic concatenations of retained calibration components. Reconstruct exactA123/A259 pre-attention2and3word256x64 andformat128x9 inputs from A150raw plusA140MLPbranch captures. Verify allweights upstream ofthoseinputs matchA354. Analyze the unchangedBreader(A354layer2), oldAreader(A259layer3,BOSreference) andcompactAreader(A354layer3,plusreference).

For each prefixlength1..40, combine each recordedformat query with eachrecordedcalibration wordprefix's firstn nonBOS keyinputs. Apply only analyticrelativeRoPE: H1 sourcepositions1..8shiftbyn+1forprefixandnewline, BOSstays0; actualwordprefixpositions1..n precedethenewline. Intendedsourcesare[7,6,5,4],oldA[3,2,1,0],newA[3,2,1,4]. Report H1unshifted controls, moved-BOS/source gaps and worstprefix-key gaps/counts byhead andprefixlength. No fittedweights,scales, attentiondistribution, nativeactivation,modeloranswers. No calibration text itself needbeprinted.

This is a recorded-component competitiondiagnostic: the newequals/source query states afterprefixandnewline areunobserved; newline-key competition isomitted because no matchedkeyinputisrecorded. Positivegaps thereforedonot certify H2, andnegativegaps expose a conditional mechanism ratherthan proving actualH2failure. Nativewordprefixinputs arecausalrecordedinputs; allsyntheticcontext limitations remainexplicit. Runas soleheavyanalysis afterA361andA362, beforeA363if noeligiblecandidate takespriority. A364isdeclaredbutnotcoded/run.


A361layer1completed (2026-11-11 16:46:25 CET):625candidates,noeligiblewordrepair. Best256units,rank12,ridge1.0,strength1.0:pairedstateRMS0.00344291106597 to0.00342911609107,gain0.400678%below5%;direction0.00238448848765 to0.00237604638716. Noformatstage,PT,checkpointormeter. Peak6,813,696KiB. Layer8nowruns,then2. A364prefix-key diagnosticiscompleteandsyntaxchecked,unrun; queuesafterA362andbeforeA363.


A361layer8completed (2026-11-11 16:50:09 CET):625candidates,noeligiblewordrepair. Best512units,rank16,ridge1.0,strength0.25:pairedstateRMS0.0062770839956 to0.00625031752233,gain0.426416%below5%;direction0.00483040166479 to0.00481889872673. Noformatstage,PT,checkpointormeter. Peak6,812,752KiB. Layer2nowruns aslastdeclaredindependentlayer;A362thenA364arepreparedandqueued.


### A361 all eight independent layers rejected; A362 starting (2026-11-11 16:53:53 CET)

The completefamily contains5,000candidates (625perlayer), noeligiblewordcandidate andnofurtherformatstage,PT,checkpointormeter. Bothstateanddirection requiredatleast5%improvement; allotheroriginalguards remained. Perlayer bestpairedstateRMS gains andresources:

|Layer|StateRMS gain|Wall time|PeakKiB|
|---|---:|---:|---:|
|1|0.400678%|3:23.28|6,813,696|
|2|1.092702%|3:24.74|6,815,048|
|3|2.816702%|3:25.11|6,741,124|
|4|1.489937%|3:24.77|6,811,796|
|5|1.438638%|3:25.19|6,820,768|
|6|0.999285%|3:22.70|6,837,460|
|7|1.673577%|3:25.05|6,802,116|
|8|0.426416%|3:22.34|6,812,752|

All directGram/RHS andunchanged-Up controls passed. Largestprocess7,001,559,040bytes isbelow12GB. Thebestlocalgain,layer3at2.8167%,isnot a textprediction orwhole-model gain. TheaggregateJSON retains exactbestparameters,baselineerrors,controls,logwalltimes andinputhashes; allindividualtrialsremain.

A362coupled nativefeature/outputconstruction nowruns as soleheavyprocess. A364prefixcompetitiondiagnostic ispreparednext; A363fullH2diagnosticremainsconditional/queuedandunreserved. Budget14fullremaining,0forward/quick;A313stillleader,no rungpassed.
