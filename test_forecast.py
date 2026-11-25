import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import features, holiday_calendar, rmse


class ForecastTest(unittest.TestCase):
    def test_future_sales_cannot_change_features(self):
        dates = pd.date_range("2020-01-01", periods=446)
        series = pd.MultiIndex.from_product([[1, 2], ["A"]], names=["store_nbr", "family"])
        logs = np.linspace(0, 5, 892, dtype=np.float32).reshape(446, 2)
        promo = np.zeros_like(logs)
        expected = features(dates, series, logs, promo, 430)
        logs[430:] = 9999
        pd.testing.assert_frame_equal(expected, features(dates, series, logs, promo, 430))
        self.assertEqual(len(expected), 32)

    def test_holidays_follow_transfers_and_local_boundaries(self):
        dates = pd.date_range("2017-08-10", "2017-08-24")
        stores = pd.read_csv("data/stores.csv")
        chosen = [stores.loc[stores.city == city, "store_nbr"].iloc[0] for city in ("Quito", "Ambato")]
        series = pd.MultiIndex.from_product([chosen, ["A"]], names=["store_nbr", "family"])
        flags = holiday_calendar(Path("data"), dates, series)
        self.assertEqual(flags[0].tolist(), [False, False])  # Lễ toàn quốc dời sang 11/08.
        self.assertEqual(flags[1].tolist(), [True, True])
        self.assertEqual(flags[-1].tolist(), [False, True])  # 24/08 chỉ nghỉ ở Ambato.

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
