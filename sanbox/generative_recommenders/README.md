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
| KuaiRand-1K | 12,824,160 | 1,152 | 12,825,312 |
| MovieLens-1M | 313,000 | 416 | 313,416 |
| MovieLens-20M | 38,913,120 | 4,224 | 38,917,344 |

For each dataset, the paired Gin files differ only in
`hstu_encoder.attention_mode = "hstu" | "safa"`. Use
`hstu-matched-...gin` as the canonical control: it binds the arm explicitly and
is the file paired with SAFA by the audit and provenance workflow.

## Prepare KuaiRand-1K

Use the sequence-complete KuaiRand-1K release for the research benchmark. The
dataset authors explicitly caution that KuaiRand-Pure removes interactions
outside its candidate pool and therefore does not preserve rigorous user
histories. The official archive is hosted in the
[Zenodo record](https://zenodo.org/records/10439422):

- archive:
  [`KuaiRand-1K.tar.gz`](https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz);
- MD5: `6b0b9c8222d67fcd4c676218edca3f1f`; and
- paper and field definitions: [kuairand.com](https://kuairand.com/).

From the `sanbox/` root, download, verify, and prepare the research files with:

```bash
GR_DATA_ROOT=/path/to/data \
  python3 preprocess_public_data.py --dataset kuairand-1k
```

The command downloads the official Zenodo archive when it is absent and
verifies its MD5 before extraction. To reuse an existing archive without
another download, provide its absolute path:

```bash
GR_DATA_ROOT=/path/to/data \
GR_KUAIRAND_ARCHIVE=/existing/KuaiRand-1K.tar.gz \
  python3 preprocess_public_data.py --dataset kuairand-1k
```

The preparation protocol is fixed as follows:

1. Merge both standard logs and the random-intervention log, ordered stably by
   `(user_id, time_ms, source_file_rank, source_row)`.
2. Retain implicit positives with `is_click == 1` and `is_hate == 0`, then
   apply iterative 5-core filtering to users and items.
3. Remap sorted original video IDs to dense zero-based IDs. The data loader
   shifts them by one so ID zero remains reserved for padding.
4. Hold out each user's final retained event as the single evaluation target.
   Split the preceding training prefix into rows of at most 2,049 events,
   starting at offsets `0, 2048, 4096, ...`; neighboring rows share only their
   boundary event so every next-item transition is supervised once.
5. Keep the final at-most-2,049-event row, including the held-out target, for
   leave-last-out evaluation. HSTU and SAFA both use a maximum history length
   of 2,048.

Prepared files are written under `$GR_DATA_ROOT/kuairand-1k/` as `train.csv`,
`eval.csv`, `item_id_map.csv`, `metadata.json`, and `checksums.sha256`. Keep
these generated files out of source control; the metadata and checksums define
the exact processed dataset used by a run. The metadata records the protocol
version, preprocessor source hash, and Python/NumPy/pandas versions; loading
fails closed on schema, official source/statistic, or artifact-manifest drift.

This is a next-positive-item benchmark over logged exposure sequences. It does
**not** implement unbiased or off-policy evaluation over KuaiRand's uniformly
randomized candidate pool, so HSTU/SAFA results from this protocol must not be
described as unbiased evaluation.

The official sources currently disagree on licensing: the Zenodo record labels
the archive CC BY 4.0, while the
[official repository](https://github.com/chongminggao/KuaiRand/blob/main/LICENSE)
states CC BY-SA 4.0. Cite Gao et al., do not redistribute raw or prepared data,
and treat derived artifacts under the stricter CC BY-SA terms unless the
dataset maintainers clarify the discrepancy.

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

Equivalent paired configs exist under `configs/ml-20m/`. Amazon Books and
KuaiRand-1K use the `n512` pairs under `configs/amzn-books/` and
`configs/kuairand-1k/`, respectively; `n512` denotes 512 sampled negatives,
not the sequence length.

The controlled cluster workflow creates an immutable source snapshot, runs
preflight equivalence and smoke checks, then submits paired HSTU/SAFA arrays
for seeds 42, 43, and 44:

```bash
submit_task=$(pueue add -p -w "$PWD" \
  'bash scripts/submit_safa_ab.sh amzn-books')
pueue log "$submit_task" --full
```

The optional selector is `amzn-books`, `kuairand-1k`, `ml-1m`, `ml-20m`, or
`all`; omitting it submits all four datasets after one shared preflight.

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
