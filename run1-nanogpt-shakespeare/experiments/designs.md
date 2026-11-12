# Circuit designs and derivations

## A — first-attention decimal residual circuit (in development)

Only unused symbol embeddings and first-block Q/K/V weights are initially edited. A nullspace of the centered Shakespeare token and absolute-position embeddings gives linear functionals that are zero for every Shakespeare first-layer input. This makes an inserted Q/K/V computation dormant on text in exact arithmetic.

For a fixed operand width, two heads encode a numeric residual. Head A assigns decimal weights to both operands and the emitted answer prefix. Head B assigns the same weights to answer-prefix positions, with the total operand attention mass moved to the plus-sign sink. Their denominators are identical. Subtracting twice head B from head A yields `(a+b - emitted_prefix_value)/D(position)`. This avoids a huge softmax sink and its poor floating-point conditioning.

A third head encodes the current decimal-place coefficient using the existing positional query vectors and hand-written desired attention odds. The output digit classifier is the analytic quadratic `2*d*(residual+0.5) - (d*d+d)*place`, whose argmax is the required floor digit. A fourth head is available for output-stage control and newline.

All linear systems in this design are changes of basis or position-address construction using weights, not fits against arithmetic examples. Digit-dependent coefficients are the explicit algebraic formula above. No full-problem table is used.

A complication: the input-nullspace alone has no robust negative output separator on held-out text. Tied embeddings prevent simply making the unused token rows enormous. The final LayerNorm has an exact output-null direction `1 / gamma_final`; a common component in that direction can strengthen arithmetic inputs without changing any real-arithmetic output logit. Symbol identity uses small orthogonal codes. Float32 cancellation and the size of resulting nullspace updates remain empirical risks.
