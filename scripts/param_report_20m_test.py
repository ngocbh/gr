#!/usr/bin/env python3
"""CPU-only formula, config-drift, and source-layout tests for the report."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


_REPORT_MODULE_NAME = "_gr_param_report_20m_under_test"
_REPORT_PATH = Path(__file__).with_name("param_report_20m.py")
if _REPORT_MODULE_NAME in sys.modules:
    report = sys.modules[_REPORT_MODULE_NAME]
else:
    _REPORT_SPEC = importlib.util.spec_from_file_location(
        _REPORT_MODULE_NAME, _REPORT_PATH
    )
    if _REPORT_SPEC is None or _REPORT_SPEC.loader is None:
        raise ImportError(f"could not load parameter report from {_REPORT_PATH}")
    report = importlib.util.module_from_spec(_REPORT_SPEC)
    sys.modules[_REPORT_MODULE_NAME] = report
    _REPORT_SPEC.loader.exec_module(report)


class ParameterReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = {
            spec.key: report.resolve_settings(spec) for spec in report.SPECS
        }
        self.inventories = {
            key: report.expected_inventory(settings)
            for key, settings in self.settings.items()
        }

    def test_live_totals(self) -> None:
        expected = {
            "ml20_hstu": (38_913_120, 5_255_776),
            "ml20_softmax": (38_913_120, 5_255_776),
            "ml20_fohstu": (38_917_344, 5_260_000),
            "ml20_local_w32": (38_917_472, 5_260_128),
            "ml20_lift_w32": (38_917_472, 5_260_128),
            "ml20_kda": (38_941_312, 5_283_968),
            "ml20_iso_kla": (38_974_464, 5_317_120),
            "ml20_diag_kla": (38_743_376, 5_086_032),
            "ml20_diag_kla_r64": (39_994_240, 6_336_896),
            "ml1_large_hstu": (313_000, 104_800),
            "ml1_large_kda": (330_216, 122_016),
            "ml1_large_kda_balanced": (314_216, 106_016),
            "ml1_large_iso_kla": (330_664, 122_464),
            "ml1_large_diag_kla": (389_672, 181_472),
            "ml1_hstu": (234_400, 26_200),
            "ml1_kda_matched": (234_288, 26_088),
            "ml1_kda_time_matched": (234_374, 26_174),
        }
        for key, (total, mixer) in expected.items():
            with self.subTest(key=key):
                self.assertEqual(report.inventory_total(self.inventories[key]), total)
                self.assertEqual(report.mixer_total(self.inventories[key]), mixer)

    def test_configs_match_report_specs(self) -> None:
        self.assertEqual(report.source_assumption_errors(), ())
        for spec in report.SPECS:
            with self.subTest(config=spec.config):
                self.assertEqual(report.config_errors(spec), ())

    def test_omitted_shape_defaults_are_resolved_from_source(self) -> None:
        spec = report.SPEC_BY_KEY["ml20_hstu"]
        explicit = report._config_assignments(spec)
        for key in (
            "train_fn.gr_output_length",
            "train_fn.embedding_module_type",
            "hstu_encoder.linear_config",
            "hstu_encoder.concat_ua",
        ):
            self.assertNotIn(key, explicit)
        settings = self.settings[spec.key]
        self.assertEqual(settings.gr_output_length, 10)
        self.assertEqual(settings.position_rows, 211)
        self.assertEqual(settings.embedding_module_type, "local")
        self.assertEqual(settings.linear_config, "uvqk")
        self.assertFalse(settings.concat_ua)

    def test_every_tracked_binding_fails_closed_when_mutated(self) -> None:
        spec = report.SPEC_BY_KEY["ml20_kda"]
        original = report._config_assignments(spec)
        for key, expected in report.canonical_bindings(spec).items():
            if isinstance(expected, bool):
                mutation = not expected
            elif isinstance(expected, str):
                mutation = expected + "-mutated"
            else:
                mutation = expected + 1
            assignments = dict(original)
            assignments[key] = mutation
            with self.subTest(key=key):
                errors = report.config_errors(spec, assignments)
                self.assertTrue(any(key in error for error in errors), errors)

    def test_untracked_hstu_binding_fails_closed(self) -> None:
        spec = report.SPEC_BY_KEY["ml20_hstu"]
        assignments = report._config_assignments(spec)
        assignments["hstu_encoder.future_shape_knob"] = 7
        self.assertTrue(
            any(
                "untracked HSTU binding" in error
                for error in report.config_errors(spec, assignments)
            )
        )

    def test_binding_type_forgery_fails_closed(self) -> None:
        spec = report.SPEC_BY_KEY["ml20_hstu"]
        assignments = report._config_assignments(spec)
        assignments["hstu_encoder.concat_ua"] = 0
        assignments["hstu_encoder.forgetting_min_period"] = 8
        errors = report.config_errors(spec, assignments)
        self.assertTrue(any("concat_ua" in error for error in errors), errors)
        self.assertTrue(
            any("forgetting_min_period" in error for error in errors), errors
        )

    def test_kda_iso_and_diag_head_layout_and_scale_source(self) -> None:
        self.assertEqual(report.head_layout_source_errors(), ())

    def test_head_layout_source_mutations_are_rejected(self) -> None:
        hstu_path = (
            report.ROOT / "generative_recommenders/research/modeling/sequential/hstu.py"
        )
        hstu_source = hstu_path.read_text()
        wrong_kda_scale = hstu_source.replace(
            "                    dt_bias=self._kda_dt_bias,\n"
            "                    use_qk_l2norm_in_kernel=True,",
            "                    dt_bias=self._kda_dt_bias,\n"
            "                    scale=H**-0.5,\n"
            "                    use_qk_l2norm_in_kernel=True,",
            1,
        )
        flattened_diag = (
            hstu_source.replace("q=qn,", "q=qn.flatten(2),", 1)
            .replace("k=kn,", "k=kn.flatten(2),", 1)
            .replace("v=padded_v.float(),", "v=padded_v.float().flatten(2),", 1)
        )
        dead_correct_assignment = hstu_source.replace(
            "            padded_q = _pad(q).view(B, n, H, dk)",
            "            if False:\n"
            "                padded_q = _pad(q).view(B, n, H, dk)\n"
            "            padded_q = _pad(q).view(B, n, H * dk)",
            1,
        )
        wrong_diag_scale = hstu_source.replace("scale=dk**-0.5", "scale=H**-0.5", 1)
        fla_path = report.fla_kda_source_path()
        fla_source = fla_path.read_text()
        overwritten_fla_scale = fla_source.replace(
            "        scale = K ** -0.5",
            "        scale = K ** -0.5\n        scale = H ** -0.5",
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, source, expected_error in (
                ("wrong_kda_scale", wrong_kda_scale, "_chunk_kda keyword set"),
                ("flattened_diag", flattened_diag, "_chunk_kalman q is not qn"),
                (
                    "dead_correct_assignment",
                    dead_correct_assignment,
                    "padded_q must have one unconditional definition",
                ),
                ("wrong_diag_scale", wrong_diag_scale, "scale is not dk ** -0.5"),
            ):
                with self.subTest(mutant=name):
                    self.assertNotEqual(source, hstu_source)
                    mutant = Path(directory) / f"{name}.py"
                    mutant.write_text(source)
                    errors = report.head_layout_source_errors(hstu_source=mutant)
                    self.assertTrue(
                        any(expected_error in error for error in errors), errors
                    )

            self.assertNotEqual(overwritten_fla_scale, fla_source)
            fla_mutant = Path(directory) / "overwritten_fla_scale.py"
            fla_mutant.write_text(overwritten_fla_scale)
            errors = report.head_layout_source_errors(fla_source=fla_mutant)
            self.assertTrue(
                any("assign scale exactly once" in error for error in errors), errors
            )

    def test_named_matched_configs_are_not_exact_matches(self) -> None:
        reference = self.inventories["ml1_hstu"]
        for key in ("ml1_kda_matched", "ml1_kda_time_matched"):
            with self.subTest(key=key):
                self.assertNotEqual(self.inventories[key], reference)
        self.assertEqual(
            report.inventory_total(self.inventories["ml1_kda_matched"])
            - report.inventory_total(reference),
            -112,
        )
        self.assertEqual(
            report.inventory_total(self.inventories["ml1_kda_time_matched"])
            - report.inventory_total(reference),
            -26,
        )

    def test_buffers_are_separate_and_not_counted(self) -> None:
        inventory = self.inventories["ml20_hstu"]
        buffers = report.expected_buffers(self.settings["ml20_hstu"])
        self.assertNotIn("_attn_mask", inventory)
        self.assertEqual(buffers, {"_attn_mask": (211, 211)})
        self.assertEqual(report.inventory_total(buffers), 44_521)

    def test_inventory_hash_is_deterministic(self) -> None:
        inventory = self.inventories["ml20_hstu"]
        self.assertEqual(
            report.inventory_sha256(inventory),
            report.inventory_sha256(dict(reversed(tuple(inventory.items())))),
        )

    def test_live_inventory_and_diff_hashes(self) -> None:
        expected = {
            "ml20_hstu": (
                "ab09c5cbf7c6806ad762037b3caf5d60df93f302ec0b8b55ab9a1c4c0af0def2",
                None,
            ),
            "ml20_softmax": (
                "ab09c5cbf7c6806ad762037b3caf5d60df93f302ec0b8b55ab9a1c4c0af0def2",
                "f97934d95edaca6ef33ed210bcc9318c2b4cd7b389fec7561a3be2640417bd3d",
            ),
            "ml20_fohstu": (
                "502f7b842a88084b20019c3232a413aaa8d804d40ee63cbccba0c40354833a55",
                "098b718c9a14647546faf049641a909a477027f71b2045cde5247ed612cb651f",
            ),
            "ml20_local_w32": (
                "78fa38ad0b464f4f285e629360e68f2b70a60d3f431c8ae6fa9989bd2fea6db4",
                "185485d57ba5bea5395435e1cb3abe4317dba63bc946a35ddce9ddd60d61aa58",
            ),
            "ml20_lift_w32": (
                "78fa38ad0b464f4f285e629360e68f2b70a60d3f431c8ae6fa9989bd2fea6db4",
                "f97934d95edaca6ef33ed210bcc9318c2b4cd7b389fec7561a3be2640417bd3d",
            ),
            "ml20_kda": (
                "87dbf82f9bc85b57968aadea8c43e898ab5182b154800c146349d0184c784af4",
                "a5684acae23f30194b10525eb65ee077eb4777ab803826390c9f743852e63a68",
            ),
            "ml20_iso_kla": (
                "c6ff5344fbeb2a22de9fba8523adc50acf37df2b2ac851794b617aa34e1cc178",
                "795cce054075878437e49db9cb33baaa315e06deb75b307ac391c909f109052f",
            ),
            "ml20_diag_kla": (
                "16f41b6419a259acd61a9e5028dd7134a8653dcea43a160c7870f151711f03bd",
                "96ab5cb04326c730fdb9f67ba21b7327c5c1662f254475571c178491288b8a90",
            ),
            "ml20_diag_kla_r64": (
                "74791f8926749a400d52f55e817f408c4168a5713a431d1b9b95c3c26b430f95",
                "bd7f99f297ae7cbd3bf13f985a40f043debe866a310c5b4607d3ecda0a117833",
            ),
        }
        for key, (inventory_hash, diff_hash) in expected.items():
            spec = report.SPEC_BY_KEY[key]
            with self.subTest(key=key):
                self.assertEqual(
                    report.inventory_sha256(self.inventories[key]), inventory_hash
                )
                if spec.reference is None:
                    self.assertIsNone(diff_hash)
                else:
                    self.assertEqual(
                        report.comparison_sha256(
                            self.inventories[spec.reference], self.inventories[key]
                        ),
                        diff_hash,
                    )


if __name__ == "__main__":
    unittest.main()
