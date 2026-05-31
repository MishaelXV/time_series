import pandas as pd


def prepare_usd_rate(rub_usd):

    usd = rub_usd.copy()

    usd = usd.rename(columns={
        "data": "Date",
        "curs": "USD_RUB"
    })

    usd["Date"] = pd.to_datetime(usd["Date"])

    usd = usd[["Date", "USD_RUB"]]

    usd = usd.sort_values("Date").reset_index(drop=True)

    return usd


def prepare_key_rate(key_rate):

    rate = key_rate.copy()

    rate = rate.rename(columns={
        "Дата": "Date",
        "Ставка": "KeyRate",
        "Key rate": "KeyRate"
    })

    rate["Date"] = pd.to_datetime(rate["Date"], dayfirst=True)

    rate = rate[["Date", "KeyRate"]]

    rate["KeyRate"] = (
        rate["KeyRate"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    rate = rate.sort_values("Date").reset_index(drop=True)

    return rate


def add_macro_features(df, rub_usd=None, key_rate=None):

    data = df.copy()

    if rub_usd is not None:

        usd = prepare_usd_rate(rub_usd)

        data = data.merge(
            usd,
            on="Date",
            how="left"
        )

        data["USD_RUB"] = data["USD_RUB"].ffill()

        data["USD_RUB_Lag_1"] = data["USD_RUB"].shift(1)
        data["USD_RUB_Lag_5"] = data["USD_RUB"].shift(5)
        data["USD_RUB_Lag_10"] = data["USD_RUB"].shift(10)

        data["USD_RUB_Change_1"] = data["USD_RUB"].shift(1).diff(1)
        data["USD_RUB_Change_5"] = data["USD_RUB"].shift(1).diff(5)

        data["USD_RUB_RollingMean_5"] = data["USD_RUB"].shift(1).rolling(5).mean()
        data["USD_RUB_RollingStd_5"] = data["USD_RUB"].shift(1).rolling(5).std()
        data["USD_RUB_RollingMean_20"] = data["USD_RUB"].shift(1).rolling(20).mean()
        data["USD_RUB_RollingStd_20"] = data["USD_RUB"].shift(1).rolling(20).std()

        data = data.drop(columns=["USD_RUB"])

    if key_rate is not None:

        rate = prepare_key_rate(key_rate)

        data = data.merge(
            rate,
            on="Date",
            how="left"
        )

        data["KeyRate"] = data["KeyRate"].ffill()

        data["KeyRate_Lag_1"] = data["KeyRate"].shift(1)
        data["KeyRate_Lag_5"] = data["KeyRate"].shift(5)
        data["KeyRate_Lag_20"] = data["KeyRate"].shift(20)

        data["KeyRate_Change_1"] = data["KeyRate"].shift(1).diff(1)
        data["KeyRate_Change_5"] = data["KeyRate"].shift(1).diff(5)
        data["KeyRate_Change_20"] = data["KeyRate"].shift(1).diff(20)

        data = data.drop(columns=["KeyRate"])

    return data


def create_features(df, rub_usd=None, key_rate=None, target_col="Balance"):

    data = df.copy()

    data = data.sort_values("Date").reset_index(drop=True)

    # Макроэкономические признаки
    data = add_macro_features(
        df=data,
        rub_usd=rub_usd,
        key_rate=key_rate
    )

    # Календарные признаки
    data["DayOfWeek"] = data["Date"].dt.dayofweek
    data["Month"] = data["Date"].dt.month
    data["Quarter"] = data["Date"].dt.quarter
    data["DayOfMonth"] = data["Date"].dt.day
    data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(int)

    data["IsMonthStart"] = data["Date"].dt.is_month_start.astype(int)
    data["IsMonthEnd"] = data["Date"].dt.is_month_end.astype(int)
    data["IsQuarterStart"] = data["Date"].dt.is_quarter_start.astype(int)
    data["IsQuarterEnd"] = data["Date"].dt.is_quarter_end.astype(int)

    # Условные налоговые дни
    data["IsTaxDay"] = data["DayOfMonth"].isin([15, 20, 25, 28]).astype(int)

    # Лаги сальдо
    balance_lags = [1, 2, 3, 5, 10, 20]

    for lag in balance_lags:
        data[f"{target_col}_Lag_{lag}"] = data[target_col].shift(lag)

    # Скользящие статистики сальдо
    windows = [5, 10, 20]

    for window in windows:

        shifted_balance = data[target_col].shift(1)

        data[f"{target_col}_RollingMean_{window}"] = shifted_balance.rolling(window).mean()
        data[f"{target_col}_RollingStd_{window}"] = shifted_balance.rolling(window).std()
        data[f"{target_col}_RollingMin_{window}"] = shifted_balance.rolling(window).min()
        data[f"{target_col}_RollingMax_{window}"] = shifted_balance.rolling(window).max()

    # Исторические доходы и расходы
    for col in ["Income", "Outcome"]:

        if col in data.columns:

            for lag in [1, 2, 3, 5, 10]:

                data[f"{col}_Lag_{lag}"] = data[col].shift(lag)

            shifted_col = data[col].shift(1)

            for window in [5, 10, 20]:

                data[f"{col}_RollingMean_{window}"] = shifted_col.rolling(window).mean()
                data[f"{col}_RollingStd_{window}"] = shifted_col.rolling(window).std()

    # Удаляем исходные колонки Income и Outcome
    drop_cols = []

    for col in ["Income", "Outcome"]:

        if col in data.columns:
            drop_cols.append(col)

    data = data.drop(columns=drop_cols)

    data = data.dropna().reset_index(drop=True)

    return data


def get_features_target(data, target_col="Balance"):

    exclude_cols = ["Date", target_col]

    feature_cols = [
        col for col in data.columns
        if col not in exclude_cols
    ]

    X = data[feature_cols]
    y = data[target_col]

    return X, y, feature_cols