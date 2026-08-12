#!/usr/bin/env python3
"""Static controls for the signed additive feature-attention screen."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = ROOT / "configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin"
CONFIGS = {
    "signed_additive_identity": ROOT
    / "configs/ml-1m/hstu-signed-additive-identity-large-final.gin",
    "signed_additive_tanh": ROOT
    / "configs/ml-1m/hstu-signed-additive-tanh-large-final.gin",
    "signed_additive_abs_tanh": ROOT
    / "configs/ml-1m/hstu-signed-additive-abs-tanh-large-final.gin",
    "signed_additive_abs_coefficient_oracle": ROOT
    / "configs/ml-1m/hstu-signed-additive-abs-coefficient-oracle-large-final.gin",
}
SBATCH_WRAPPER = ROOT / "scripts/sbatch_signed_additive_ml1m.sh"
SUBMIT_HELPER = ROOT / "scripts/submit_attention_experiments.sh"


def _gin_assignments(path: Path) -> Dict[str, object]:
    assignments: Dict[str, object] = {}
    pattern = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*?)\s*$")
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        match = pattern.match(line)
        if match is not None:
            key, value = match.groups()
            assignments[key] = ast.literal_eval(value)
    return assignments


class SignedAdditiveAblationStaticTest(unittest.TestCase):
    def test_configs_change_only_normalization_and_fixed_gamma(self) -> None:
        baseline = _gin_assignments(BASELINE_CONFIG)
        for normalization, path in CONFIGS.items():
            with self.subTest(normalization=normalization):
                candidate = _gin_assignments(path)
                self.assertEqual(
                    candidate.pop("hstu_encoder.normalization"), normalization
                )
                self.assertEqual(candidate.pop("hstu_encoder.signed_feature_gamma"), 1.0)
                self.assertEqual(candidate, baseline)

    def test_wrapper_is_shared_h200_seed42_44_snapshot_array(self) -> None:
        wrapper = SBATCH_WRAPPER.read_text()
        self.assertIn("#SBATCH --qos=h200_mrs_shared", wrapper)
        self.assertIn("#SBATCH --gres=gpu:h200:1", wrapper)
        self.assertIn("#SBATCH --array=0-11", wrapper)
        self.assertIn("#SBATCH --requeue", wrapper)
        self.assertIn("#SBATCH --open-mode=append", wrapper)
        self.assertNotIn("h200_dev", wrapper)
        self.assertIn("GR_CODE_SNAPSHOT:?", wrapper)
        self.assertIn("GR_SNAPSHOT_MANIFEST_SHA256:?", wrapper)
        self.assertIn("sha256sum --check --strict --quiet", wrapper)
        self.assertIn("restart_count=", wrapper)
        self.assertIn("SLURM_RESTART_COUNT", wrapper)
        self.assertIn("seed=$((42 + task_id % 3))", wrapper)
        self.assertIn("train_fn.random_seed=$seed", wrapper)
        self.assertIn("train_fn.save_last_only=True", wrapper)
        for path in CONFIGS.values():
            self.assertIn(str(path.relative_to(ROOT)), wrapper)

    def test_submit_helper_exposes_explicit_signed_additive_mode(self) -> None:
        helper = SUBMIT_HELPER.read_text()
        self.assertIn('"$mode" != "signed-additive"', helper)
        self.assertIn('if [[ "$mode" == "signed-additive" ]]', helper)
        self.assertIn(
            "signed_additive_ml1m_job=$(submit scripts/sbatch_signed_additive_ml1m.sh)",
            helper,
        )


if __name__ == "__main__":
    unittest.main()
