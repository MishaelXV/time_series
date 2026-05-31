import numpy as np
import pandas as pd

# Выбросы определяются методом межквартильного размаха (IQR).
# Наблюдение считается выбросом, если выходит за границы [Q1 - 1.5*IQR; Q3 + 1.5*IQR].
# Выбросы заменяются линейной интерполяцией по соседним значениям.
def replace_outliers_interpolate(series):

    series_clean = series.copy()

    q1 = series_clean.quantile(0.25)
    q3 = series_clean.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    
    outlier_mask = ((series_clean < lower) | (series_clean > upper))
    
    outliers_count = outlier_mask.sum()
    outliers_percent = 100 * outliers_count / len(series_clean)

    print(f"Количество выбросов: {outliers_count}")
    print(f"Доля измененных точек: {outliers_percent:.2f}%")

    series_clean.loc[outlier_mask] = np.nan

    series_clean = series_clean.interpolate(method="linear")

    return series_clean