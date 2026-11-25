import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import combine_forecasts, direct_forecast, features, holiday_calendar, rmse, robust_promotions


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

    def test_direct_forecast_ignores_hidden_labels(self):
        dates = pd.date_range("2015-01-01", periods=526)
        series = pd.MultiIndex.from_product([[1, 2], ["A"]], names=["store_nbr", "family"])
        logs = np.random.default_rng(12).uniform(0, 4, (526, 2))
        promo = np.zeros_like(logs)
        holidays = np.zeros_like(logs, dtype=bool)
        first = direct_forecast(dates, series, logs, promo, 510, holidays)
        logs[510:] = 42
        second = direct_forecast(dates, series, logs, promo, 510, holidays)
        np.testing.assert_array_equal(first, second)

    def test_isolated_promotion_spike_uses_only_known_covariates(self):
        dates = pd.date_range("2020-01-01", periods=102)
        series = pd.MultiIndex.from_product([[1, 2], list("ABCD")], names=["store_nbr", "family"])
        promo = np.ones((len(dates), len(series)))
        promo[84] = 41
        expected, flagged = robust_promotions(dates, series, promo, 84)
        self.assertEqual(flagged, [str(dates[84].date())])
        np.testing.assert_array_equal(expected[84], np.full(len(series), 5))
        self.assertTrue(np.all(promo[84] == 41))
        promo[100:] = 9999
        replay, _ = robust_promotions(dates, series, promo, 84)
        np.testing.assert_array_equal(expected[:100], replay[:100])

    def test_store_fallback_requires_three_consecutive_empty_days(self):
        series = pd.MultiIndex.from_product([[1, 2], ["A", "B"]], names=["store_nbr", "family"])
        promo = np.ones((16, 4))
        promo[1:4, :2] = 0
        promo[1:3, 2:] = 0
        actual = combine_forecasts(np.zeros(64), np.full(64, 2), series, promo)
        expected = np.ones_like(promo)
        expected[1:4, :2] = 0
        np.testing.assert_array_equal(actual, expected.ravel())

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
