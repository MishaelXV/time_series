import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone


def get_feature_importance(model, feature_names):

    importance = model.get_feature_importance()

    result = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    result = result.sort_values("Importance", ascending=False).reset_index(drop=True)

    return result


def get_permutation_importance(model, X, y, n_repeats=20):

    permutation_result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=42,
        scoring="neg_mean_absolute_error"
    )

    result = pd.DataFrame({
        "Feature": X.columns,
        "Importance": permutation_result.importances_mean
    })

    result = result.sort_values("Importance", ascending=False).reset_index(drop=True)

    return result


def select_by_correlation(X, y, top_n=15):

    scores = []

    for col in X.columns:
        corr = np.corrcoef(X[col], y)[0, 1]
        scores.append({
            "Feature": col,
            "Score": abs(corr)
        })

    result = pd.DataFrame(scores)
    result = result.sort_values("Score", ascending=False).reset_index(drop=True)

    selected_features = result.head(top_n)["Feature"].tolist()

    return selected_features, result


def select_by_lasso(X, y, top_n=15):

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=5, random_state=42, max_iter=10000))
    ])

    model.fit(X, y)

    coefficients = model.named_steps["lasso"].coef_

    result = pd.DataFrame({
        "Feature": X.columns,
        "Score": np.abs(coefficients)
    })

    result = result.sort_values("Score", ascending=False).reset_index(drop=True)

    selected_features = result[result["Score"] > 0]["Feature"].head(top_n).tolist()

    return selected_features, result


def select_by_permutation(model, X, y, top_n=15):

    result = get_permutation_importance(
        model=model,
        X=X,
        y=y
    )

    selected_features = result.head(top_n)["Feature"].tolist()

    return selected_features, result


def evaluate_selected_features(model, X_train, y_train, X_test, y_test, selected_features):

    model = clone(model)

    model.fit(X_train[selected_features], y_train)

    y_pred = model.predict(X_test[selected_features])

    mae = mean_absolute_error(y_test, y_pred)

    return mae


def kuncheva_index(set_a, set_b, total_features):

    set_a = set(set_a)
    set_b = set(set_b)

    k = len(set_a)
    r = len(set_a.intersection(set_b))
    n = total_features

    if k == 0 or k == n:
        return np.nan

    return (r * n - k ** 2) / (k * (n - k))


def average_kuncheva(selected_sets, total_features):

    scores = []

    for i in range(len(selected_sets)):
        for j in range(i + 1, len(selected_sets)):
            score = kuncheva_index(
                selected_sets[i],
                selected_sets[j],
                total_features
            )
            scores.append(score)

    return np.nanmean(scores)