import os
import pickle

import numpy as np
import pandas as pd

from pandas.tseries.offsets import BDay
from statsmodels.tsa.statespace.sarimax import SARIMAX


# =========================================================
# Подготовка данных
# =========================================================

def prepare_series(df, date_col="Date", target_col="Balance"):

    data = df.copy()

    data[date_col] = pd.to_datetime(data[date_col])

    data = data.sort_values(date_col).reset_index(drop=True)

    data = data[data[date_col].dt.dayofweek < 5].copy()

    series = data.set_index(date_col)[target_col].dropna()

    return series


def replace_outliers_interpolate(series):

    series_clean = series.copy()

    q1 = series_clean.quantile(0.25)
    q3 = series_clean.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = (
        (series_clean < lower) |
        (series_clean > upper)
    )

    outliers_count = int(outlier_mask.sum())
    outliers_percent = 100 * outliers_count / len(series_clean)

    series_clean.loc[outlier_mask] = np.nan

    series_clean = series_clean.interpolate(method="linear")

    return series_clean, outliers_count, outliers_percent


def get_next_business_day(date):

    return date + BDay(1)


# =========================================================
# SARIMA
# =========================================================

def fit_sarima(series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 5)):

    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(disp=False)

    return fitted_model


def forecast_next_day(model):

    prediction = model.forecast(steps=1)

    return float(prediction.iloc[0])


# =========================================================
# Сохранение и загрузка
# =========================================================

def save_pickle(obj, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(path):

    with open(path, "rb") as file:
        obj = pickle.load(file)

    return obj


def file_exists(path):

    return os.path.exists(path)


# =========================================================
# Логика переобучения
# =========================================================

def count_business_days(start_date, end_date):

    return len(pd.bdate_range(start=start_date, end=end_date)) - 1


def should_retrain_by_period(last_train_date, current_date, retrain_period=5):

    days_passed = count_business_days(
        start_date=last_train_date,
        end_date=current_date
    )

    return days_passed >= retrain_period


# =========================================================
# CUSUM
# =========================================================

def calculate_cusum(series, mean, std, threshold=10, drift=0.5):

    values = np.asarray(series)

    if std == 0:
        std = 1e-8

    positive_stat = 0
    negative_stat = 0

    positive_cusum = []
    negative_cusum = []
    alarms = []

    for value in values:

        z_value = (value - mean) / std

        positive_stat = max(0, positive_stat + z_value - drift)
        negative_stat = min(0, negative_stat + z_value + drift)

        positive_cusum.append(positive_stat)
        negative_cusum.append(negative_stat)

        alarm = positive_stat > threshold or abs(negative_stat) > threshold
        alarms.append(alarm)

        if alarm:
            positive_stat = 0
            negative_stat = 0

    result = pd.DataFrame({
        "Value": values,
        "PositiveCUSUM": positive_cusum,
        "NegativeCUSUM": negative_cusum,
        "Alarm": alarms
    }, index=series.index)

    return result


def detect_recent_drift(series, reference_window=60, monitoring_window=20, threshold=10, drift=0.5):

    if len(series) < reference_window + monitoring_window:

        return {
            "DriftDetected": False,
            "Reason": "Недостаточно наблюдений для контроля разладки",
            "CUSUM": None
        }

    reference_series = series.iloc[-reference_window-monitoring_window:-monitoring_window]
    monitoring_series = series.iloc[-monitoring_window:]

    reference_mean = reference_series.mean()
    reference_std = reference_series.std()

    cusum_result = calculate_cusum(
        series=monitoring_series,
        mean=reference_mean,
        std=reference_std,
        threshold=threshold,
        drift=drift
    )

    drift_detected = bool(cusum_result["Alarm"].any())

    return {
        "DriftDetected": drift_detected,
        "Reason": "Обнаружена разладка CUSUM" if drift_detected else "Разладка не обнаружена",
        "CUSUM": cusum_result
    }


# =========================================================
# Основной production pipeline
# =========================================================

def run_production_pipeline(
    df,
    model_path="models/best_sarima.pkl",
    metadata_path="models/metadata.pkl",
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 5),
    retrain_period=5,
    date_col="Date",
    target_col="Balance",
    use_drift_control=False,
    use_outlier_processing=True
):

    series = prepare_series(
        df=df,
        date_col=date_col,
        target_col=target_col
    )

    current_date = series.index.max()
    next_forecast_date = get_next_business_day(current_date)

    if use_outlier_processing:

        series, outliers_count, outliers_percent = replace_outliers_interpolate(series)

    else:

        outliers_count = 0
        outliers_percent = 0

    metadata_exists = file_exists(metadata_path)
    model_exists = file_exists(model_path)

    metadata = load_pickle(metadata_path) if metadata_exists else None

    retrain_reasons = []

    if not model_exists or metadata is None:

        need_retrain = True
        retrain_reasons.append("Модель отсутствует")

    else:

        last_train_date = metadata["last_train_date"]

        period_trigger = should_retrain_by_period(
            last_train_date=last_train_date,
            current_date=current_date,
            retrain_period=retrain_period
        )

        if period_trigger:
            retrain_reasons.append(
                f"Прошло не менее {retrain_period} рабочих дней с последнего обучения"
            )

        need_retrain = period_trigger

    drift_info = None

    if use_drift_control:

        drift_info = detect_recent_drift(
            series=series,
            reference_window=60,
            monitoring_window=20,
            threshold=10,
            drift=0.5
        )

        if drift_info["DriftDetected"]:

            need_retrain = True
            retrain_reasons.append("Обнаружена разладка временного ряда")

    if need_retrain:

        model = fit_sarima(
            series=series,
            order=order,
            seasonal_order=seasonal_order
        )

        metadata = {
            "last_train_date": current_date,
            "order": order,
            "seasonal_order": seasonal_order,
            "retrain_period": retrain_period,
            "train_size": len(series),
            "outlier_processing": use_outlier_processing,
            "outliers_count": outliers_count,
            "outliers_percent": outliers_percent
        }

        save_pickle(obj=model, path=model_path)
        save_pickle(obj=metadata, path=metadata_path)

        retrained = True

    else:

        model = load_pickle(model_path)

        retrained = False

    prediction = forecast_next_day(model)

    result = pd.DataFrame({
        "ForecastDate": [next_forecast_date],
        "Prediction": [prediction],
        "Model": [f"SARIMA{order}x{seasonal_order}"],
        "Retrained": [retrained],
        "RetrainReason": ["; ".join(retrain_reasons) if retrain_reasons else "Переобучение не требуется"],
        "TrainEndDate": [current_date],
        "TrainSize": [len(series)],
        "OutliersProcessed": [outliers_count],
        "OutliersPercent": [outliers_percent]
    })

    return result, model, metadata, drift_info