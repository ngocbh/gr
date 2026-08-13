# HSTU/SAFA experiment workflow

The two qualification stages answer different questions and are intentionally
separate.

1. Commit the clean `sanbox/` source tree.
2. Submit through pueue with `bash scripts/submit_safa_ab.sh DATASET`. The submitter
   creates an immutable snapshot, runs the exact-equivalence/config/inventory
   suite plus short HSTU/SAFA smokes on one `h200_dev` H200, and only then
   releases the selected Amazon Books, KuaiRand-1K, ML-1M, or ML-20M seed
   array. Use `all` or omit the selector to submit all four.
3. Export the six completed runs for one dataset to the JSON schema documented
   by `python scripts/qualify_safa_results.py --help`. Take `attention_mode`,
   `random_seed`, `resolved_gin_config`, `resolved_gin_config_sha256`, and
   `experiment_config_sha256` directly from each W&B run config. Also export
   `slurm_array_job_id`, `slurm_array_task_id`, `slurm_job_id`,
   `slurm_restart_count`, `slurm_job_qos`, and `slurm_job_partition` from that
   config; do not infer any of these fields from the run name or tags. Use
   result schema version 3.
4. Apply the frozen quality gate:

Set `ML1M_ARRAY_JOB_ID` to the exact `ml1m_array_job` value printed by the
submitter; do not recover it from a run name.

```bash
bash scripts/check_safa_results.sh results.json \
  --expected-dataset ml-1m \
  --expected-source-commit "$GR_SOURCE_COMMIT" \
  --expected-source-tree "$GR_SOURCE_TREE" \
  --expected-source-manifest "$GR_EXPECTED_SOURCE_MANIFEST" \
  --expected-experiment-config-sha256 "$(sed -n \
    's/^experiment_config_ml-1m=//p' \
    "$GR_QUALIFICATION_ROOT/$GR_EXPECTED_SOURCE_MANIFEST.passed")" \
  --expected-array-job-id "$ML1M_ARRAY_JOB_ID" \
  --output results.qualification.json
```

The pre-run smoke marker proves only that the snapshot and training path work.
It does not imply that SAFA passes the post-run quality gate.

The result schema is deliberately fail-closed. The gate recomputes both config
hashes, requires the recorded mode/seed to match the exported arm/seed, and
requires one normalized experiment identity across all six runs. Normalization
redacts only `hstu_encoder.attention_mode` and `train_fn.random_seed`, so a
learning-rate, dropout, or any other operative Gin drift invalidates the set.
The preflight records the intended identity in its manifest-specific marker;
full jobs verify it before training and the post-run command pins it again.
The array job ID is pinned independently from the submitter output. The gate
also requires QoS `h200_dev` for ML-1M and `h200_mrs_2_high` for Amazon Books,
KuaiRand-1K, and ML-20M, plus
the exact task map `0..5` to seeds
`42..44`, with HSTU on even tasks and SAFA on odd tasks. Full array runs require
online W&B and fail if initialization, metric logging, or finalization fails;
preflight smokes do not initialize W&B and are not post-run evidence.
MovieLens qualification averages epochs 96--100. Amazon Books uses epoch 200
and KuaiRand-1K uses epoch 100, their final full-corpus evaluations; intervening
partial evaluations are not mixed into reported results.

Preflight and full KuaiRand-1K tasks authenticate `train.csv`, `eval.csv`,
`item_id_map.csv`, `metadata.json`, and `checksums.sha256` independently. The
qualification marker binds all five hashes to the immutable source snapshot,
and the full array rechecks them before initializing W&B or training.

The post-run CLI also queries `sacct` for the pinned array. It accepts only six
distinct completed allocations on partition `h200`, with zero exit codes and
matching raw job IDs and restart counts. Requeued runs use fresh local and W&B
names suffixed with `-rN`; inherited W&B run/resume/sweep identities are cleared
so attempts cannot merge.

## Checkpoint diagnostics

After a matched HSTU or SAFA run has produced its final checkpoint, inspect it
from the exact immutable training snapshot and exact run directory:

```bash
python scripts/diagnose_safa_checkpoint.py \
  --checkpoint "$CHECKPOINT" \
  --run-dir "$RUN_DIR" \
  --source-root "$SOURCE_SNAPSHOT" \
  --data-root "$GR_DATA_ROOT" \
  --max-examples 2048 \
  --output "$RUN_DIR/safa_diagnostics.json"
```

The command refuses partial checkpoints and any checkpoint/config/run-metadata,
scheduler, or source commit/tree/manifest mismatch. It reconstructs the model
from the checkpoint's operative Gin config, strictly loads the state, and uses
the existing evaluation path. The default example set is a reproducible bounded
sample; its selected-index checksum is recorded. Run it on both arms with the
same sample settings to obtain directly comparable history/gap strata.

Gate histogram counts, half-life sufficient statistics, and signed coefficient
mass cover every valid position/pair in the evaluated users; reported gate
quantiles have the declared histogram resolution. Survival quantiles use a
deterministic, batch-population-weighted sample of positive-lag causal pairs to
avoid adding a second quadratic attention tensor. HSTU reports forgetting as
unavailable by design but retains signed-mass and recommendation diagnostics.
Elapsed-time metrics are emitted only for rows with positive, chronological
history and target timestamps; excluded rows are reported rather than imputed.
Current data-file hashes are recorded for cross-arm equality. Because training
did not store data hashes, they cannot retrospectively prove which bytes were
used by the completed run.
