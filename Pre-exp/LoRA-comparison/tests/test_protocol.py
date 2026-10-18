import unittest

from lora_comparison.protocol import SIZE_SPECS, VARIANTS
from lora_comparison.run import size_output_root


class ProtocolTest(unittest.TestCase):
    def test_size_specs_keep_expected_selected_counts(self):
        self.assertEqual(
            [SIZE_SPECS[key].num_groups for key in ("1b", "3b", "5b")],
            [8, 25, 45],
        )
        self.assertEqual(
            [SIZE_SPECS[key].selected_experts for key in ("1b", "3b", "5b")],
            [10, 32, 57],
        )
        self.assertEqual(SIZE_SPECS["1b"].selection_percent(64), 15.625)

    def test_exact_eight_variants_and_only_two_frozen_baselines(self):
        self.assertEqual(len(VARIANTS), 8)
        frozen = {key for key, value in VARIANTS.items() if not value.trainable}
        self.assertEqual(frozen, {"router_frozen", "uniform"})

    def test_size_validation_rejects_wrong_teacher_geometry(self):
        with self.assertRaises(ValueError):
            SIZE_SPECS["5b"].selection_percent(32)

    def test_size_output_root_appends_size_to_directory_name(self):
        self.assertEqual(
            size_output_root("/tmp/olmoe_spread_lora_100step", "3b"),
            "/tmp/olmoe_spread_lora_100step_3b",
        )
