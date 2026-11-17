"""
training 辅助函数测试
"""
import unittest

import numpy as np
import pandas as pd

from qlworks.models.training import _filter_finite_feature_label_frame


class TestTrainingHelpers(unittest.TestCase):
    def test_filter_finite_feature_label_frame_drops_invalid_label_rows(self):
        columns = pd.MultiIndex.from_tuples(
            [
                ("feature", "f1"),
                ("feature", "f2"),
                ("label", "LABEL_5D"),
            ]
        )
        df = pd.DataFrame(
            [
                [1.0, 10.0, 0.1],
                [2.0, 20.0, np.nan],
                [3.0, 30.0, np.inf],
                [4.0, 40.0, 0.2],
            ],
            columns=columns,
        )

        result = _filter_finite_feature_label_frame(df)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[("label", "LABEL_5D")].tolist(), [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
