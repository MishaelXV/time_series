import pandas as pd


def create_features(df, target_col="Balance"):

    data = df.copy()

    data = data.sort_values("Date").reset_index(drop=True)

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