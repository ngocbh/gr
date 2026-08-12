#!/usr/bin/env python3
"""Static controls for the parameter-matched HSTU tanh experiment."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = ROOT / "configs/ml-20m/hstu-sampled-softmax-n128-large-final.gin"
TANH_CONFIG = ROOT / "configs/ml-20m/hstu-tanh-20m.gin"
SBATCH_WRAPPER = ROOT / "scripts/sbatch_hstu_tanh_ml20m.sh"
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


class HSTUTanhAblationStaticTest(unittest.TestCase):
    def test_config_changes_only_pairwise_attention_activation(self) -> None:
        baseline = _gin_assignments(BASELINE_CONFIG)
        tanh = _gin_assignments(TANH_CONFIG)

        self.assertEqual(tanh.pop("hstu_encoder.normalization"), "tanh_rel_bias")
        self.assertEqual(tanh, baseline)
        self.assertNotIn("hstu_encoder.linear_activation", baseline)

    def test_wrapper_is_single_shared_h200_seed42_snapshot_job(self) -> None:
        wrapper = SBATCH_WRAPPER.read_text()
        self.assertIn("#SBATCH --qos=h200_mrs_shared", wrapper)
        self.assertIn("#SBATCH --gres=gpu:h200:1", wrapper)
        self.assertNotIn("#SBATCH --array=", wrapper)
        self.assertNotIn("h200_dev", wrapper)
        self.assertIn("GR_CODE_SNAPSHOT:?", wrapper)
        self.assertIn("GR_SNAPSHOT_MANIFEST_SHA256:?", wrapper)
        self.assertIn("sha256sum --check --strict --quiet", wrapper)
        self.assertIn("configs/ml-20m/hstu-tanh-20m.gin", wrapper)
        self.assertIn("train_fn.random_seed=42", wrapper)
        self.assertIn("train_fn.save_last_only=True", wrapper)
        self.assertIn("'parameter-matched'", wrapper)

    def test_submit_helper_exposes_only_explicit_tanh_mode(self) -> None:
        helper = SUBMIT_HELPER.read_text()
        self.assertIn('"$mode" != "tanh"', helper)
        self.assertIn('if [[ "$mode" == "tanh" ]]', helper)
        self.assertIn(
            "hstu_tanh_ml20m_job=$(submit scripts/sbatch_hstu_tanh_ml20m.sh)",
            helper,
        )


if __name__ == "__main__":
    unittest.main()
