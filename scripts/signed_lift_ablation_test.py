#!/usr/bin/env python3
"""Static controls for the parameter-matched Signed-LIFT screen."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
HYBRID_CONFIG = ROOT / "configs/ml-1m/hstu-hybrid-forgetting-w32-large-final.gin"
LOCAL_CONFIG = ROOT / "configs/ml-1m/hstu-local-forgetting-w32-large-final.gin"
CANDIDATE_CONFIGS = {
    "identity": ROOT / "configs/ml-1m/hstu-signed-lift-identity-w32-large-final.gin",
    "tanh": ROOT / "configs/ml-1m/hstu-signed-lift-tanh-w32-large-final.gin",
    "abs_tanh": ROOT / "configs/ml-1m/hstu-signed-lift-abs-tanh-w32-large-final.gin",
}
SBATCH_WRAPPER = ROOT / "scripts/sbatch_signed_lift_ml1m.sh"
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


class SignedLIFTAblationStaticTest(unittest.TestCase):
    def test_candidates_add_only_fixed_map_and_gamma_to_hybrid_w32(self) -> None:
        baseline = _gin_assignments(HYBRID_CONFIG)
        for feature_map, path in CANDIDATE_CONFIGS.items():
            with self.subTest(feature_map=feature_map):
                candidate = _gin_assignments(path)
                self.assertEqual(
                    candidate.pop("hstu_encoder.hybrid_tail_feature_map"),
                    feature_map,
                )
                self.assertEqual(
                    candidate.pop("hstu_encoder.signed_feature_gamma"), 1.0
                )
                self.assertEqual(candidate, baseline)

    def test_local_control_differs_from_identity_lift_only_by_normalization(
        self,
    ) -> None:
        local = _gin_assignments(LOCAL_CONFIG)
        identity = _gin_assignments(CANDIDATE_CONFIGS["identity"])
        self.assertEqual(
            local.pop("hstu_encoder.normalization"), "local_forgetting_rel_bias"
        )
        self.assertEqual(
            identity.pop("hstu_encoder.normalization"),
            "hybrid_forgetting_rel_bias",
        )
        self.assertEqual(
            identity.pop("hstu_encoder.hybrid_tail_feature_map"), "identity"
        )
        self.assertEqual(identity.pop("hstu_encoder.signed_feature_gamma"), 1.0)
        self.assertEqual(local, identity)

    def test_wrapper_is_shared_one_h200_seed42_44_snapshot_array(self) -> None:
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
        self.assertIn("SOURCE_TREE_INVENTORY", wrapper)
        self.assertIn("SLURM_RESTART_COUNT", wrapper)
        self.assertIn("seed=$((42 + task_id % 3))", wrapper)
        self.assertIn("config_idx=$((task_id / 3))", wrapper)
        self.assertIn("train_fn.random_seed=$seed", wrapper)
        self.assertIn("train_fn.save_last_only=True", wrapper)
        for path in (LOCAL_CONFIG, *CANDIDATE_CONFIGS.values()):
            self.assertIn(str(path.relative_to(ROOT)), wrapper)

    def test_submit_helper_exposes_only_explicit_signed_lift_mode(self) -> None:
        helper = SUBMIT_HELPER.read_text()
        self.assertIn('"$mode" != "signed-lift"', helper)
        self.assertIn('if [[ "$mode" == "signed-lift" ]]', helper)
        self.assertIn(
            "signed_lift_ml1m_job=$(submit scripts/sbatch_signed_lift_ml1m.sh)",
            helper,
        )
        core_block = helper.split('if [[ "$mode" == "core"', 1)[1].split("fi", 1)[0]
        self.assertNotIn("signed_lift", core_block)


if __name__ == "__main__":
    unittest.main()
