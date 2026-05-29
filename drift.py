import numpy as np
import pandas as pd


def calculate_cusum(series, mean=None, std=None, threshold=5, drift=0.5):

    values = np.asarray(series)

    if mean is None:
        mean = np.mean(values)

    if std is None:
        std = np.std(values)

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


def detect_changepoints(series, threshold=5, drift=0.5):

    mean = series.mean()
    std = series.std()

    result = calculate_cusum(
        series=series,
        mean=mean,
        std=std,
        threshold=threshold,
        drift=drift
    )

    changepoints = result[result["Alarm"]].index

    return result, changepoints