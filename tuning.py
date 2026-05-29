import optuna
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor


def tune_random_forest(X, y, n_trials=30, n_splits=5):

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        }

        tscv = TimeSeriesSplit(n_splits=n_splits)

        scores = []

        for train_idx, valid_idx in tscv.split(X):

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = RandomForestRegressor(
                random_state=42,
                n_jobs=-1,
                **params
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_valid)

            score = mean_absolute_error(y_valid, y_pred)

            scores.append(score)

        return np.mean(scores)

    study = optuna.create_study(direction="minimize")

    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value, study


def tune_catboost(X, y, n_trials=30, n_splits=5):

    def objective(trial):

        params = {
            "iterations": trial.suggest_int("iterations", 200, 800),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "random_strength": trial.suggest_float("random_strength", 0.1, 5),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5)
        }

        tscv = TimeSeriesSplit(n_splits=n_splits)

        scores = []

        for train_idx, valid_idx in tscv.split(X):

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = CatBoostRegressor(
                random_seed=42,
                verbose=False,
                loss_function="MAE",
                **params
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_valid)

            score = mean_absolute_error(y_valid, y_pred)

            scores.append(score)

        return np.mean(scores)

    study = optuna.create_study(direction="minimize")

    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value, study