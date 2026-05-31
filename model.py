import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# MAE
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "MAE": mae
    }

# Бейзлайн
def run_mean_baseline(y_train, y_test):

    history = list(y_train)

    predictions = []

    for i in range(len(y_test)):

        y_pred = np.mean(history)

        predictions.append(y_pred)

        history.append(y_test.iloc[i])

    metrics = evaluate_model(y_test, predictions)

    result = pd.DataFrame({

        "Actual": y_test.values,

        "Predicted": predictions

    }, index=y_test.index)

    return result, metrics

# ARIMA
def fit_arima(y_train, p, d, q):

    model = ARIMA(
        y_train,
        order=(p, d, q)
    )

    fitted_model = model.fit()

    return fitted_model


def run_arima(y_train, y_test, p, d, q):

    history = list(y_train)

    predictions = []

    for i in range(len(y_test)):

        model = ARIMA(history, order=(p, d, q))

        fitted_model = model.fit()

        y_pred = fitted_model.forecast(steps=1)[0]

        predictions.append(y_pred)

        history.append(y_test.iloc[i])

    metrics = evaluate_model(y_test, predictions)

    result = pd.DataFrame({

        "Actual": y_test.values,

        "Predicted": predictions

    }, index=y_test.index)

    return fitted_model, result, metrics

# SARIMA
def fit_sarima(y_train, order, seasonal_order):

    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order
    )

    fitted_model = model.fit(
        disp=False
    )

    return fitted_model


def run_sarima(y_train, y_test, order, seasonal_order):

    history = list(y_train)
    predictions = []

    for i in range(len(y_test)):

        model = SARIMAX(
            history,
            order=order,
            seasonal_order=seasonal_order
        )

        fitted_model = model.fit(disp=False)

        y_pred = fitted_model.forecast(steps=1)[0]

        predictions.append(y_pred)

        history.append(y_test.iloc[i])

    metrics = evaluate_model(y_test, predictions)

    result = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    }, index=y_test.index)

    return fitted_model, result, metrics

# Случайный лес
def run_random_forest(X_train, y_train, X_test, y_test, params=None):

    if params is None:

        params = {}

    model = RandomForestRegressor(

        random_state=42,

        n_jobs=-1,

        **params

    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred)

    result = pd.DataFrame({

        "Actual": y_test.values,

        "Predicted": y_pred

    }, index=y_test.index)

    return model, result, metrics

# Catboost
def run_catboost(X_train, y_train, X_test, y_test, params=None):

    if params is None:

        params = {}

    model = CatBoostRegressor(

        random_seed=42,

        verbose=False,

        loss_function="MAE",

        **params

    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred)
    
    result = pd.DataFrame({

        "Actual": y_test.values,

        "Predicted": y_pred

    }, index=y_test.index)

    return model, result, metrics