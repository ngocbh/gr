# Plan: Signed Additive LIFT Tail

**Goal**: Add parameter-matched identity, signed-tanh, and abs-tanh feature maps
to LIFT's old-history branch and launch a controlled ML-1M screen.

**Architecture**: Preserve the exact local SiLU/RAB branch and existing LIFT
normalization. Generalize only the old-tail feature map, retain the dense quality
oracle, and add an exact delayed recurrence as a tested reference.

**Tech Stack**: Python, PyTorch, gin-config, unittest/pytest, Bash, SLURM.

## Dependencies

| Group | Steps | Can Parallelize |
|---|---|---|
| 1 | 1-3 | No; shared HSTU API and tests |
| 2 | 4-5 | Yes after Group 1 |
| 3 | 6-7 | No; verification precedes submission |

## Step 1: Specify dense feature-tail behavior

**Files**: `generative_recommenders/research/modeling/sequential/hstu.py`,
`generative_recommenders/research/modeling/sequential/hstu_signed_lift_attention_test.py`

1. Add failing FP64 tests for tanh and abs-tanh dense tails against literal loops,
   including output and `q/k/v/log_forget` gradients.
2. Generalize `_per_head_forgetting_tail_attention` with fixed feature-map and
   gamma arguments, defaulting to identity.
3. Verify identity is bit-exact with the prior helper and that `0.5/N` scaling,
   boundary support, padding, head isolation, and RAB independence hold.

## Step 2: Specify the delayed recurrence

**Files**: the same HSTU module and signed-LIFT test file.

1. Add failing tests comparing a shifted recurrence with the dense oracle for
   maps `identity`, `tanh`, and `abs_tanh`, windows `1`, `N`, and `N+1`, mixed
   valid lengths, and output plus input/gate gradients.
2. Implement a reference recurrence that applies lag survival after the query
   feature map and returns graph-connected zeros for an empty tail.
3. Fail closed for invalid feature maps/gamma and document causal-right-padding
   scope.

## Step 3: Integrate the selector without capacity drift

**Files**: `hstu.py`, `encoder_utils.py`, signed-LIFT test file.

1. Add `hybrid_tail_feature_map="identity"` through `hstu_encoder`, `HSTU`, and
   `SequentialTransductionUnitJagged`.
2. Keep local attention, relative bias, forget gates, and ReZero gain unchanged.
3. Test default identity regression, nonzero tail-gain gradients, empty-tail DDP
   graph connectivity, debug labels, and exact ML-1M/ML-20M named inventories.

## Step 4: Add experiment controls

**Files**: three signed-LIFT gin configs plus a static ablation test.

1. Clone W32 local/LIFT settings into local, identity, tanh, and abs-tanh arms;
   candidate configs may differ only in the fixed map/gamma selector.
2. Extend the parameter report only if needed to audit the new fixed binding;
   do not change expected inventory hashes.

## Step 5: Add reproducible shared-QoS submission

**Files**: `scripts/sbatch_signed_lift_ml1m.sh`,
`scripts/signed_lift_ablation_test.py`, `scripts/submit_attention_experiments.sh`.

1. Add a 12-task seed-42-to-44 array requesting one H200 per task on
   `h200_mrs_shared`, with restart-distinct run IDs and immutable snapshot checks.
2. Add static tests for QoS, task/seed mapping, configs, fixed bindings, and
   explicit submit-helper wiring.

## Step 6: End-to-end verification and review

Run focused signed-LIFT tests, existing attention/SAFA/tanh tests, parameter
audits, Bash syntax checks, and the snapshot integrity test. Obtain independent
spec-compliance review followed by code-quality review and fix every blocker.

## Step 7: Snapshot and submit

Submit only through pueue using
`scripts/submit_attention_experiments.sh signed-lift`. Record the immutable
snapshot path, manifest digest, Slurm job ID, QoS, array shape, and initial state.

