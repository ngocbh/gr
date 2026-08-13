# HSTU/SAFA experiment workflow

The two qualification stages answer different questions and are intentionally
separate.

1. Commit the clean `sanbox/` source tree.
2. Submit through pueue with `bash scripts/submit_safa_ab.sh`. The submitter
   creates an immutable snapshot, runs the exact-equivalence/config/inventory
   suite plus short HSTU/SAFA smokes on one shared-QoS H200, and only then
   releases the ML-1M and ML-20M seed arrays.
3. Export the six completed runs for one dataset to the JSON schema documented
   by `python scripts/qualify_safa_results.py --help`. Take `attention_mode`,
   `random_seed`, `resolved_gin_config`, `resolved_gin_config_sha256`, and
   `experiment_config_sha256` directly from each W&B run config; do not infer
   them from the run name or tags.
4. Apply the frozen quality gate:

```bash
bash scripts/check_safa_results.sh results.json \
  --expected-dataset ml-1m \
  --expected-source-commit "$GR_SOURCE_COMMIT" \
  --expected-source-tree "$GR_SOURCE_TREE" \
  --expected-source-manifest "$GR_EXPECTED_SOURCE_MANIFEST" \
  --expected-experiment-config-sha256 "$(sed -n \
    's/^experiment_config_ml-1m=//p' \
    "$GR_QUALIFICATION_ROOT/$GR_EXPECTED_SOURCE_MANIFEST.passed")" \
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
