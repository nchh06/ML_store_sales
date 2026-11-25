"""Dự báo trực tiếp 16 ngày cho từng cặp cửa hàng và nhóm sản phẩm."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HORIZON = 16
KEY = ["date", "store_nbr", "family"]
SEED = 3405


def load_data(directory):
    types = {"family": "category", "store_nbr": "int16", "sales": "float32",
             "onpromotion": "int32", "id": "int32"}
    train, test = [pd.read_csv(directory / name, parse_dates=["date"], dtype=types)
                   for name in ("train.csv", "test.csv")]
    sample = pd.read_csv(directory / "sample_submission.csv")
    for frame in (train, test):
        if frame.isna().any().any() or frame.duplicated(KEY).any() or frame.id.duplicated().any():
            raise ValueError("Dữ liệu thiếu hoặc trùng khóa")
    if (train.sales < 0).any() or not np.isfinite(train.sales).all():
        raise ValueError("Doanh số lịch sử không hợp lệ")
    if list(sample.columns) != ["id", "sales"] or sample.id.duplicated().any() or set(sample.id) != set(test.id):
        raise ValueError("ID trong sample_submission.csv và test.csv không khớp")
    dates = pd.date_range(train.date.min(), periods=(test.date.max() - train.date.min()).days + 1)
    origin = dates.get_loc(train.date.max()) + 1
    if not pd.DatetimeIndex(test.date.unique()).sort_values().equals(dates[origin:]) or len(dates) - origin != HORIZON:
        raise ValueError("Cần 16 ngày test liên tiếp ngay sau train")

    series = train.set_index(KEY[1:]).index.unique().sort_values()
    def matrix(frame, value):
        wide = frame.pivot(index="date", columns=KEY[1:], values=value)
        return wide.reindex(index=dates, columns=series).to_numpy(dtype=np.float32)

    logs = np.log1p(matrix(train, "sales"))
    promo = matrix(pd.concat([train[KEY + ["onpromotion"]], test[KEY + ["onpromotion"]]]), "onpromotion")
    ids = test.pivot(index="date", columns=KEY[1:], values="id").reindex(index=dates[origin:], columns=series)
    if ids.isna().any().any() or ids.size != len(test):
        raise ValueError("Test phải chứa đủ các chuỗi của train")
    eda = {"train_rows": len(train), "series": len(series), "stores": int(train.store_nbr.nunique()),
           "families": int(train.family.nunique()), "zero_share": float((train.sales == 0).mean()),
           "start": str(dates[0].date()), "end": str(train.date.max().date()),
           "missing_dates": int(origin - train.date.nunique()),
           "weekly_sales": {str(k.date()): float(v) for k, v in train.groupby("date").sales.sum().resample("W").mean().tail(13).items()}}
    return dates, series, logs, promo, ids, sample, eda


def features(dates, series, logs, promo, origin):
    """Mỗi cửa sổ lịch sử kết thúc trước ngày đầu của đợt dự báo."""
    if origin < 84:
        raise ValueError("Cần ít nhất 84 ngày lịch sử")
    history = logs[origin-84:origin]
    history_promo = promo[origin-84:origin]
    weekday = dates[origin-84:origin].day_of_week
    n = len(series)
    frames = []
    for lead, day in enumerate(dates[origin:origin + HORIZON]):
        matching = history[weekday == day.day_of_week]
        frame = pd.DataFrame({
            "level_7": np.nanmean(history[-7:], axis=0),
            "level_28": np.nanmean(history[-28:], axis=0),
            "level_84": np.nanmean(history, axis=0),
            "weekday_4": np.nanmean(matching[-4:], axis=0),
            "weekday_12": np.nanmean(matching, axis=0),
            "last_weekday": matching[-1],
            "zero_rate": np.nanmean(np.where(np.isnan(history), np.nan, history == 0), axis=0),
            "promo": np.log1p(promo[origin + lead]),
            "promo_level": np.log1p(np.nanmean(history_promo[-28:], axis=0)),
            "weekday": np.full(n, day.day_of_week), "month_day": day.day,
            "payday": int(day.day == 15 or day.is_month_end),
            "store": series.get_level_values("store_nbr").to_numpy(),
            "family": series.codes[1], "lead": lead + 1,
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).fillna(0).astype("float32")


def fit_models(dates, series, logs, promo, origin):
    origins = range(origin - 10 * HORIZON, origin, HORIZON)
    x = pd.concat([features(dates, series, logs, promo, t) for t in origins], ignore_index=True)
    y = logs[origin - 10 * HORIZON:origin].ravel()
    valid = np.isfinite(y)
    x = x.loc[valid]
    residual = y[valid] - x.weekday_4.to_numpy()
    models = {
        "Boosting": HistGradientBoostingRegressor(max_iter=110, learning_rate=.09, max_leaf_nodes=31,
                                                  early_stopping=False, random_state=SEED),
        "MLP": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=32,
                            batch_size=1024, early_stopping=True, random_state=SEED)),
    }
    for model in models.values():
        model.fit(x, residual)
    return models


def predict(model, x):
    return np.maximum(0, x.weekday_4.to_numpy() + model.predict(x))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, dtype=np.float64) - np.asarray(p)) ** 2)))


def run(directory, output):
    dates, series, logs, promo, ids, sample, eda = load_data(directory)
    end = len(dates) - HORIZON
    result = {"eda": eda, "folds": []}
    # Chọn trọng số ở đợt thứ nhất, giữ cố định khi đánh giá ở đợt thứ hai.
    for origin in (end - 2 * HORIZON, end - HORIZON, end):
        print(f"Huấn luyện tại mốc {dates[origin].date()}", flush=True)
        models = fit_models(dates, series, logs, promo, origin)
        x = features(dates, series, logs, promo, origin)
        tree, mlp = [predict(model, x) for model in models.values()]
        if origin == end:
            break
        y = logs[origin:origin + HORIZON].ravel()
        if not np.isfinite(y).all():
            raise ValueError("Thiếu nhãn kiểm định")
        if not result["folds"]:
            weight = min([0, .25, .5, .75, 1], key=lambda w: rmse(y, w*tree + (1-w)*mlp))
            result["tree_weight"] = weight
        blend = weight * tree + (1-weight) * mlp
        scores = {name: rmse(y, pred) for name, pred in
                  [("Baseline", x.weekday_4), ("Boosting", tree), ("MLP", mlp), ("Blend", blend)]}
        result["folds"].append({"start": str(dates[origin].date()), "scores": scores})
        if origin == end - HORIZON:
            gaps = {}
            rng = np.random.default_rng(SEED)
            for col in x.columns:
                changes = []
                for _ in range(3):
                    shuffled = x.copy()
                    shuffled[col] = rng.permutation(x[col].values)
                    changes.append(rmse(y, predict(models["Boosting"], shuffled)) - scores["Boosting"])
                gaps[col] = float(np.mean(changes))
            result["permutation_delta"] = gaps
            group_error = pd.DataFrame({"family": np.tile(series.get_level_values("family"), HORIZON),
                                        "squared": (y-blend)**2})
            result["family_error"] = np.sqrt(group_error.groupby("family", observed=True).squared.mean()).to_dict()
            result["daily_error"] = np.sqrt(((y-blend).reshape(HORIZON,-1)**2).mean(axis=1)).tolist()
        print(scores, flush=True)
    sales = np.expm1(weight*tree + (1-weight)*mlp)
    predictions = pd.Series(sales, index=ids.to_numpy().ravel())
    sample["sales"] = sample.id.map(predictions)
    if not np.isfinite(sample.sales).all() or (sample.sales < 0).any():
        raise ValueError("Dự báo không hợp lệ")
    output.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output / "submission.csv", index=False, float_format="%.7f")
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(output / "submission.csv", len(sample))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data, args.out)
