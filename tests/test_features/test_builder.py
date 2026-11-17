"""
features.builder 过滤行为测试
"""
import tempfile
import unittest
from pathlib import Path

from qlworks.features.builder import build_factor_library_bundle


class TestBuildFactorLibraryBundle(unittest.TestCase):
    def test_build_factor_library_bundle_supports_selected_factor_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / "group_a.yaml").write_text(
                """
factors:
  - name: FACTOR_A
    expression:
      qlib: "$close"
  - name: FACTOR_B
    expression:
      qlib: "$open"
""".strip(),
                encoding="utf-8",
            )
            (repo_path / "group_b.yaml").write_text(
                """
factors:
  - name: FACTOR_C
    expression:
      qlib: "$high"
  - name: FACTOR_B
    expression:
      qlib: "$low"
""".strip(),
                encoding="utf-8",
            )

            bundle = build_factor_library_bundle(
                strategy_names=["group_a", "group_b"],
                repo_path=str(repo_path),
                factor_names=["FACTOR_C", "FACTOR_B"],
            )

        self.assertEqual(list(bundle.names), ["FACTOR_C", "FACTOR_B"])
        self.assertEqual(list(bundle.fields), ["$high", "$open"])


if __name__ == "__main__":
    unittest.main()
