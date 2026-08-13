# From HSTU to Signed Additive Forgetting Attention

This package contains the clean research implementation used to compare
Hierarchical Sequential Transduction Units (HSTU) with **Signed Additive
Forgetting Attention (SAFA)** for sequential recommendation. SAFA is a small,
controlled change to the academic retrieval model in
[`research/modeling/sequential/hstu.py`](research/modeling/sequential/hstu.py).
The production HSTU implementations under `modules/`, `ops/`, and `dlrm_v3/`
are unchanged.

Older experiment artifacts may call the same mechanism `FoHSTU`. In this
codebase its name is SAFA, reflecting forgetting attention built directly on
HSTU.

## From HSTU coefficients to SAFA

For head `h`, the research HSTU assigns a causal query-key pair the signed,
unnormalized coefficient

$$
a_{ij}^{\mathrm{HSTU},h}
= \frac{M_{ij}}{N}\operatorname{SiLU}
  \left((q_i^h)^\mathsf{T}k_j^h+b_{ij}\right),
$$

where `M` is the causal mask, `N` is the padded sequence width, and `b_ij` is
the existing learned relative time-and-position bias. Each head retains its
own query-key dot product and value aggregation.

SAFA adds one key-conditioned scalar forget gate per head and position:

$$
f_t^h=\sigma\left((w_f^h)^\mathsf{T}k_t^h+c_h\right).
$$

The survival of event `j` at query `i` is the product of gates after the key
and through the query:

$$
F_{ij}^h=\prod_{t=j+1}^{i}f_t^h, \qquad F_{ii}^h=1.
$$

SAFA multiplies survival **outside** SiLU:

$$
a_{ij}^{\mathrm{SAFA},h}=F_{ij}^h a_{ij}^{\mathrm{HSTU},h},
\qquad
z_i^h=\sum_{j\leq i}a_{ij}^{\mathrm{SAFA},h}v_j^h.
$$

This placement is deliberate:

- `F_ij^h` is nonnegative, so forgetting attenuates a pair without reversing
  the sign of its HSTU coefficient.
- There is no row denominator. Supportive events add value contributions
  instead of competing for a fixed probability mass.
- Survival depends on the intervening path, while HSTU's explicit pairs and
  relative time-and-position bias remain intact.
- "Additive" describes the value aggregation, not computational complexity.
  SAFA remains quadratic explicit-pair attention.

Survival is evaluated stably in float32 log space with `logsigmoid`, prefix
sums, and `exp`. Gate weights start at zero. Per-head biases initialize time
constants `T_h` logarithmically from 8 to 256 so that initially
`F_ij^h = exp(-(i-j)/T_h)`.

## Exact-inventory HSTU control

The matched HSTU and SAFA arms instantiate the same named gate tensors in every
block:

- `_forget_weight`: `[num_heads, dqk]`
- `_forget_bias`: `[num_heads]`

SAFA learns and applies their survival factors. Matched HSTU bypasses
forgetting, which is equivalent to setting `F_ij^h = 1`, and keeps an
exact-zero graph dependency on the dormant tensors. Parameter names, shapes,
optimizer-visible inventory, and DDP parameter usage therefore match without
changing HSTU's output.

The number of gate parameters is
`num_blocks * num_heads * (dqk + 1)`:

| Dataset | HSTU backbone | Gate parameters | Total in each arm |
| --- | ---: | ---: | ---: |
| Amazon Books | 44,865,440 | 1,152 | 44,866,592 |
| MovieLens-1M | 313,000 | 416 | 313,416 |
| MovieLens-20M | 38,913,120 | 4,224 | 38,917,344 |

For each dataset, the paired Gin files differ only in
`hstu_encoder.attention_mode = "hstu" | "safa"`. Use
`hstu-matched-...gin` as the canonical control: it binds the arm explicitly and
is the file paired with SAFA by the audit and provenance workflow.

## Run the two arms

Run commands from the `sanbox/` repository root after preprocessing the
selected dataset. The launcher respects `GR_DATA_ROOT`, `GR_EXPS_ROOT`, and
`GR_CKPTS_ROOT`; their defaults are `tmp/`, `exps/`, and `ckpts/`.

```bash
bash scripts/train.sh \
  configs/ml-1m/hstu-matched-sampled-softmax-n128-large-final.gin \
  --gin_bindings=train_fn.random_seed=42

bash scripts/train.sh \
  configs/ml-1m/safa-sampled-softmax-n128-large-final.gin \
  --gin_bindings=train_fn.random_seed=42
```

Equivalent paired configs exist under `configs/ml-20m/`. Amazon Books uses
the `n512` pair under `configs/amzn-books/`.

The controlled cluster workflow creates an immutable source snapshot, runs
preflight equivalence and smoke checks, then submits paired HSTU/SAFA arrays
for seeds 42, 43, and 44:

```bash
submit_task=$(pueue add -p -w "$PWD" \
  'bash scripts/submit_safa_ab.sh amzn-books')
pueue log "$submit_task" --full
```

The optional selector is `amzn-books`, `ml-1m`, `ml-20m`, or `all`; omitting
it submits all three datasets after one shared preflight.

Commit all intended `sanbox/` source changes first; snapshot creation refuses a
dirty source tree. Set `GR_DATA_ROOT`, `GR_EXPS_ROOT`, and `GR_CKPTS_ROOT`, and
ensure W&B is authenticated for online logging. Project and entity overrides
are optional. The submitter prints the snapshot identity, Slurm job IDs, and
fully pinned post-run qualification commands. See
[`scripts/README.md`](../scripts/README.md) for the result schema, frozen
quality rule, provenance checks, and checkpoint diagnostics. Passing preflight
only establishes that the snapshot and training path work; it is not evidence
that SAFA improves recommendation quality.

## Verify the implementation

```bash
python3 -m unittest -v \
  generative_recommenders.research.modeling.sequential.safa_test
python3 scripts/audit_safa_ab.py --dataset all
```

The tests check:

- bit-exact matched-HSTU equivalence to frozen upstream HSTU through forward,
  backward, and one AdamW step;
- identical named trainable inventories in HSTU and SAFA;
- log-spaced gate initialization and exact survival-path indexing;
- preservation of negative pair coefficients and nonzero gate gradients; and
- agreement between full recomputation and cached incremental evaluation.

## Implementation status

The current research HSTU and SAFA attention paths are correctness references,
not optimized kernels. Attention math uses PyTorch operations and materializes
per-head `[B, H, N, N]` pair weights; SAFA additionally materializes the
survival matrix. FBGEMM operators still perform jagged-to-padded and
padded-to-jagged layout conversion.

The repository's existing Triton and CUDA HSTU kernels serve the separate
production path and are not wired into these research experiments. There is no
custom SAFA Triton or CUDA kernel yet. Kernel work should follow only after the
parameter-matched quality experiment justifies optimizing the mechanism.

## Code map

| Path | Purpose |
| --- | --- |
| [`research/modeling/sequential/hstu.py`](research/modeling/sequential/hstu.py) | Shared HSTU/SAFA block and survival computation |
| [`research/modeling/sequential/encoder_utils.py`](research/modeling/sequential/encoder_utils.py) | Gin-configurable `attention_mode` wiring |
| [`research/modeling/sequential/safa_test.py`](research/modeling/sequential/safa_test.py) | Semantic, parity, gradient, and cache tests |
| [`scripts/audit_safa_ab.py`](../scripts/audit_safa_ab.py) | Config fidelity and parameter-inventory audit |
| [`scripts/submit_safa_ab.sh`](../scripts/submit_safa_ab.sh) | Immutable-snapshot paired experiment submission |
| [`scripts/diagnose_safa_checkpoint.py`](../scripts/diagnose_safa_checkpoint.py) | Gate, survival, signed-mass, and recommendation diagnostics |
