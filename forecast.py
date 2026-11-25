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


def holiday_calendar(directory, dates, series):
    """Cờ nghỉ theo địa bàn, tính theo ngày nghỉ thực tế sau điều chuyển."""
    stores = pd.read_csv(directory / "stores.csv").set_index("store_nbr")
    events = pd.read_csv(directory / "holidays_events.csv", parse_dates=["date"])
    location = stores.loc[series.get_level_values("store_nbr")]
    calendar = np.zeros((len(dates), len(series)), dtype=bool)
    observed = events[~events.transferred & events.type.isin(["Holiday", "Transfer", "Additional", "Bridge"])]
    for event in observed.itertuples():
        if event.date not in dates:
            continue
        applies = np.ones(len(series), dtype=bool)
        if event.locale == "Local":
            applies = location.city.to_numpy() == event.locale_name
        elif event.locale == "Regional":
            applies = location.state.to_numpy() == event.locale_name
        calendar[dates.get_loc(event.date)] |= applies
    return calendar


def local_forecast(dates, series, logs, promo, origin, calendar):
    """Ridge riêng cho mỗi chuỗi. Quan sát gần đây có trọng số lớn hơn."""
    start = max(0, origin - 364)
    stop = origin + HORIZON
    days = dates[start:stop]
    n_train = origin - start
    weekday = np.eye(7)[days.day_of_week]
    # Chu kỳ nửa tháng và tháng cho phép đáp ứng trơn quanh kỳ trả lương.
    angle = 2 * np.pi * (days.day.to_numpy() - 1) / days.days_in_month.to_numpy()
    common = np.column_stack([np.ones(len(days)), weekday, np.cos(angle), np.sin(angle),
                              np.cos(2*angle), np.sin(2*angle)])
    recent = np.exp2((np.arange(n_train) - n_train + 1) / 48)
    forecast = np.zeros((HORIZON, len(series)), dtype=np.float32)
    for store in series.get_level_values("store_nbr").unique():
        cols = np.flatnonzero(series.get_level_values("store_nbr") == store)
        y = logs[start:origin, cols].T
        active_day = np.nansum(y, axis=0) > 0
        # Tách thời kỳ trước khi cửa hàng mở và các ngày cửa hàng đóng cửa.
        weight = recent * active_day * np.isfinite(y)
        promotions = np.nan_to_num(promo[start:stop, cols].T, nan=0.)
        common_block = np.broadcast_to(common, (len(cols), *common.shape))
        features = np.concatenate([common_block, np.log1p(promotions)[:, :, None],
                                   (promotions > 0)[:, :, None],
                                   calendar[start:stop, cols].T[:, :, None]], axis=2)
        train = features[:, :n_train]
        weighted = train.transpose(0, 2, 1) * weight[:, None, :]
        penalty = np.eye(train.shape[2]) * 1.5
        penalty[0, 0] = .01
        coeff = np.linalg.solve(weighted @ train + penalty, weighted @ np.nan_to_num(y)[:, :, None])
        forecast[:, cols] = (features[:, n_train:] @ coeff)[:, :, 0].T
    return np.maximum(forecast, 0)


def features(dates, series, logs, promo, origin, calendar=None):
    """Mỗi cửa sổ lịch sử kết thúc trước ngày đầu của đợt dự báo."""
    if origin < 84:
        raise ValueError("Cần ít nhất 84 ngày lịch sử")
    history = logs[origin-84:origin]
    history_promo = promo[origin-84:origin]
    weekday = dates[origin-84:origin].day_of_week
    n = len(series)
    if calendar is None:
        calendar = np.zeros_like(promo, dtype=bool)
    local = local_forecast(dates, series, logs, promo, origin, calendar)
    store = series.get_level_values("store_nbr")
    family = series.get_level_values("family")
    if origin >= 392:
        annual_ref = np.nanmean(logs[origin-392:origin-364], axis=0)
    else:
        annual_ref = np.nanmean(history[-28:], axis=0)
    frames = []
    for lead, day in enumerate(dates[origin:origin + HORIZON]):
        matching = history[weekday == day.day_of_week]
        previous_year = origin + lead - 364
        seasonal = (np.nanmean(logs[previous_year-7:previous_year+8], axis=0)
                    if previous_year >= 7 else np.nanmean(history, axis=0))
        target_promo = promo[origin + lead]
        matching_promo = np.isclose(np.log1p(history_promo), np.log1p(target_promo), atol=.30)
        count = matching_promo.sum(axis=0)
        conditional = (np.nansum(np.where(matching_promo, history, 0), axis=0)
                       + 5 * np.nanmean(history, axis=0)) / (count + 5)
        frame = pd.DataFrame({
            "local_ridge": local[lead],
            "season_last_year": seasonal,
            "promo_conditional": conditional,
            "promo_weekday": np.nanmean(np.log1p(history_promo[weekday == day.day_of_week]), axis=0),
            "level_3": np.nanmean(history[-3:], axis=0),
            "store_promo": np.log1p(pd.Series(target_promo).groupby(store).transform("sum").to_numpy()),
            "holiday": calendar[origin + lead],
            "season_change": seasonal - annual_ref,
            "peer_promo": np.log1p(pd.Series(target_promo).groupby(family).transform("mean").to_numpy()),
            "promo_diff": np.log1p(target_promo) - np.nanmean(np.log1p(history_promo), axis=0),
            "half_month": (day.day - 1) % 15,
            "month": day.month,
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


def fit_models(dates, series, logs, promo, origin, calendar):
    # Thêm các cửa sổ quanh cùng mùa năm trước để học các đợt tăng theo năm.
    recent = list(range(origin - 12 * HORIZON, origin, HORIZON))
    seasonal = list(range(origin - 364 - 2 * HORIZON, origin - 364 + 4 * HORIZON, HORIZON))
    origins = [t for t in seasonal + recent if t >= 84 and t + HORIZON <= origin]
    x = pd.concat([features(dates, series, logs, promo, t, calendar) for t in origins], ignore_index=True)
    y = np.concatenate([logs[t:t + HORIZON].ravel() for t in origins])
    valid = np.isfinite(y)
    x = x.loc[valid]
    residual = y[valid] - x.local_ridge.to_numpy()
    models = {
        "Boosting": HistGradientBoostingRegressor(max_iter=260, learning_rate=.075, max_leaf_nodes=31,
                                                  categorical_features=["family", "store", "weekday", "month"],
                                                  early_stopping=False, random_state=SEED),
        "MLP": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=32,
                            batch_size=1024, early_stopping=True, random_state=SEED)),
    }
    for model in models.values():
        model.fit(x, residual)
    return models


def predict(model, x):
    return np.maximum(0, x.local_ridge.to_numpy() + model.predict(x))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, dtype=np.float64) - np.asarray(p)) ** 2)))


def robust_promotions(dates, series, promo, origin):
    """Giảm ảnh hưởng của một ngày có khuyến mãi tăng đồng loạt hiếm gặp."""
    future = promo[origin:origin + HORIZON]
    history = promo[max(0, origin - 84):origin]
    history_days = dates[origin-len(history):origin].day_of_week
    future_days = dates[origin:origin + HORIZON].day_of_week
    families = series.get_level_values("family")
    cols_by_family = [np.flatnonzero(families == f) for f in families.unique()]
    # Mức tại phân vị 20 tăng khi phần lớn cửa hàng cùng tăng khuyến mãi.
    observed = np.column_stack([np.quantile(future[:, cols], .2, axis=1) for cols in cols_by_family])
    past = np.column_stack([np.quantile(history[:, cols], .2, axis=1) for cols in cols_by_family])
    clean = promo.copy()
    affected = []
    for lead, weekday in enumerate(future_days):
        prev = history_days == weekday
        peers = (future_days == weekday) & (np.arange(len(future_days)) != lead)
        if not peers.any():
            continue
        usual = np.fmax(np.nanquantile(past[prev], .95, axis=0), observed[peers].max(axis=0))
        elevated = observed[lead] > usual + np.maximum(1, .4*usual)
        if elevated.sum() < max(4, int(np.ceil(.2 * len(cols_by_family)))):
            continue
        # Chỉ tham chiếu số khuyến mãi cùng thứ đã có sẵn trong cửa sổ dự báo.
        typical = np.median(future[peers], axis=0)
        clean[origin+lead] = np.rint(.9*typical + .1*future[lead])
        affected.append(str(dates[origin+lead].date()))
    return clean, affected


def direct_forecast(dates, series, logs, promo, origin, calendar):
    """Mỗi nhóm hàng dùng toàn bộ cửa hàng để học trực tiếp log-doanh số."""
    from lightgbm import LGBMRegressor

    start = max(392, origin - 900)
    target = np.arange(start, origin + HORIZON)
    # Mỗi khối lịch sử và khối cần dự báo đều dùng cùng một mốc chốt 16 ngày.
    lead = (target - origin) % HORIZON
    cutoff = target - lead
    days = dates[target]
    result = np.zeros((HORIZON, len(series)), dtype=float)
    for family in series.get_level_values("family").unique():
        cols = np.flatnonzero(series.get_level_values("family") == family)
        n = len(cols)
        y = logs[:origin + HORIZON, cols].astype(float)
        # Các nhãn sau mốc luôn bị che cả khi chạy trên cửa sổ có đáp án.
        y[origin:] = np.nan
        offers = promo[:origin + HORIZON, cols]
        def roll(arr, window):
            return pd.DataFrame(arr).rolling(window, min_periods=1).mean().to_numpy()
        def repeat(arr):
            return np.repeat(np.asarray(arr), n)
        def flat(arr):
            return arr.reshape(-1)

        shop = series.get_level_values("store_nbr").to_numpy()[cols]
        days_count = len(target)
        campaign = offers[target]
        monthday = days.day.to_numpy()
        data = {"store": np.tile(shop, days_count), "weekday": repeat(days.day_of_week),
                "month": repeat(days.month), "day": repeat(monthday),
                "day_in_year": repeat(days.day_of_year), "lead": repeat(lead),
                "half_month": repeat((monthday-1) % 15),
                "holiday": flat(calendar[target][:,cols]),
                "promo": flat(np.log1p(campaign)),
                "promo_chain": repeat(np.log1p(np.nan_to_num(campaign).mean(axis=1))),
                "coverage": repeat((campaign>0).mean(axis=1)),
                "last": flat(y[cutoff-1]), "last_promo": flat(np.log1p(offers[cutoff-1])),
                "previous_year": flat(roll(y, 15)[target-364+7]),
                "previous_year_level": flat(roll(y, 28)[cutoff-365])}
        for window in (3,7,14,28,56,140):
            data[f"level_{window}"] = flat(roll(y,window)[cutoff-1])
        weekly = target - 7 * (1 + lead // 7)
        data["weekly"] = flat(y[weekly])
        for window in (4,12):
            data[f"weekday_{window}"] = flat(np.nanmean(np.stack([y[weekly-7*k] for k in range(window)]),axis=0))
        data["active_share"] = flat(roll(y>0,56)[cutoff-1])
        data["past_promo"] = flat(roll(np.log1p(offers),28)[cutoff-1])
        data["trend"] = data["level_7"] - data["level_28"]
        data["annual_trend"] = data["previous_year"] - data["previous_year_level"]
        x = pd.DataFrame(data)
        train = target < origin
        labels = flat(y[target[train]])
        valid = np.isfinite(labels)
        train_size = len(labels)
        model = LGBMRegressor(n_estimators=260, learning_rate=.045, num_leaves=23,
                              min_child_samples=50, reg_lambda=4, n_jobs=2, deterministic=True,
                              force_col_wise=True, verbosity=-1, random_state=SEED)
        weights = repeat(.4+.6*np.exp2((target[train]-origin)/280))
        model.fit(x.iloc[:train_size].loc[valid],labels[valid],sample_weight=weights[valid],
                  categorical_feature=["store","weekday"])
        result[:,cols] = model.predict(x.iloc[train_size:]).reshape(HORIZON, n)
    return result.clip(0).ravel()


def combine_forecasts(previous, direct, series, future_promo):
    """Giữ dự báo cũ khi cả cửa hàng ngừng khuyến mãi nhiều ngày liền."""
    store_ids = series.get_level_values("store_nbr")
    blended = ((previous + direct) * .5).reshape(HORIZON, -1)
    previous = previous.reshape(HORIZON, -1)
    for store in store_ids.unique():
        cols = np.flatnonzero(store_ids == store)
        empty = (future_promo[:, cols] == 0).all(axis=1)
        runs = empty[:-2] & empty[1:-1] & empty[2:]
        mask = np.zeros(HORIZON, dtype=bool)
        for offset in (0, 1, 2):
            mask[offset:offset + len(runs)] |= runs
        blended[np.ix_(mask, cols)] = previous[np.ix_(mask, cols)]
    return blended.ravel()


def run(directory, output):
    dates, series, logs, promo, ids, sample, eda = load_data(directory)
    calendar = holiday_calendar(directory, dates, series)
    end = len(dates) - HORIZON
    result = {"eda": eda, "folds": []}
    # Chọn trọng số ở đợt thứ nhất, giữ cố định khi đánh giá ở đợt thứ hai.
    for origin in (end - 2 * HORIZON, end - HORIZON, end):
        print(f"Huấn luyện tại mốc {dates[origin].date()}", flush=True)
        models = fit_models(dates, series, logs, promo, origin, calendar)
        clean_promo, flagged = robust_promotions(dates, series, promo, origin)
        direct = direct_forecast(dates, series, logs, clean_promo, origin, calendar)
        x = features(dates, series, logs, clean_promo, origin, calendar)
        tree, mlp = [predict(model, x) for model in models.values()]
        future_promo = promo[origin:origin + HORIZON]
        if origin == end:
            result["adjusted_test_dates"] = flagged
            blend = combine_forecasts(weight*tree + (1-weight)*mlp, direct, series, future_promo)
            break
        y = logs[origin:origin + HORIZON].ravel()
        if not np.isfinite(y).all():
            raise ValueError("Thiếu nhãn kiểm định")
        if not result["folds"]:
            weight = min([0, .25, .5, .75, 1], key=lambda w: rmse(y, w*tree + (1-w)*mlp))
            result["tree_weight"] = weight
        previous = weight * tree + (1-weight) * mlp
        blend = combine_forecasts(previous, direct, series, future_promo)
        scores = {name: rmse(y, pred) for name, pred in
                  [("Baseline", x.weekday_4), ("Ridge", x.local_ridge), ("Boosting", tree), ("MLP", mlp), ("Direct", direct), ("Blend", blend)]}
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
    sales = np.expm1(blend)
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
