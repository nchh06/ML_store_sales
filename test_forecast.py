import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import features, rmse


class ForecastTest(unittest.TestCase):
    def test_future_sales_cannot_change_features(self):
        dates = pd.date_range("2020-01-01", periods=100)
        series = pd.MultiIndex.from_product([[1, 2], ["A"]], names=["store_nbr", "family"])
        logs = np.arange(200, dtype=np.float32).reshape(100, 2)
        promo = np.zeros_like(logs)
        expected = features(dates, series, logs, promo, 84)
        logs[84:] = 9999
        pd.testing.assert_frame_equal(expected, features(dates, series, logs, promo, 84))
        self.assertEqual(len(expected), 32)

    def test_log_error_accepts_series(self):
        self.assertAlmostEqual(rmse(np.log1p([0, 3]), pd.Series([0, 0])), np.log(4)/np.sqrt(2))

    def test_submission_matches_competition_template(self):
        sample = pd.read_csv(Path("data/sample_submission.csv"))
        test = pd.read_csv(Path("data/test.csv"))
        submission = pd.read_csv("submission.csv")
        self.assertEqual(submission.columns.tolist(), ["id", "sales"])
        self.assertEqual(len(submission), 54 * 33 * 16)
        self.assertTrue(submission.id.is_unique)
        pd.testing.assert_series_equal(submission.id, sample.id)
        self.assertEqual(set(submission.id), set(test.id))
        self.assertTrue(np.isfinite(submission.sales).all())
        self.assertTrue((submission.sales >= 0).all())


if __name__ == "__main__":
    unittest.main()
