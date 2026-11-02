"""Baseline model factories. All models predict AQI_eod from a list of feature columns.

Fixed, un-tuned hyper-parameters on purpose (the same for every feature set / protocol), so the
comparisons isolate the effect of the INPUTS, not of per-cell tuning. Say so when reporting.

Missing values: satellite columns are NaN on cloudy days.
  * tree models (RandomForest, XGBoost) take NaN natively;
  * Ridge / SVR / MLP get a median imputer + missing-indicator columns, then standardisation.
"""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

SEED = 0
SVR_MAX_TRAIN = 6000          # RBF-SVR is O(n^2): fit on a fixed random subsample of the training rows

MODEL_NAMES = ["Ridge", "RandomForest", "XGBoost", "SVR", "MLP"]


def _scaled(estimator):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), estimator)


def make_model(name, n_jobs=4):
    if name == "Ridge":
        return TransformedTargetRegressor(_scaled(Ridge(alpha=1.0)), transformer=StandardScaler())
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.5, n_jobs=n_jobs, random_state=SEED)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=3, tree_method="hist", n_jobs=n_jobs, random_state=SEED, verbosity=0)
    if name == "SVR":
        return TransformedTargetRegressor(_scaled(SVR(kernel="rbf", C=3.0, epsilon=0.1, gamma="scale")), transformer=StandardScaler())
    if name == "MLP":
        return TransformedTargetRegressor(
            _scaled(MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True, n_iter_no_change=15,
                                 max_iter=300, random_state=SEED)),
            transformer=StandardScaler())
    raise ValueError(name)


def fit_predict(name, X_train, y_train, X_evals, n_jobs=4):
    """fit on the training rows, return {eval_name: predictions}"""
    if name == "SVR" and len(X_train) > SVR_MAX_TRAIN:
        idx = np.random.default_rng(SEED).choice(len(X_train), SVR_MAX_TRAIN, replace=False)
        X_train, y_train = X_train.iloc[idx], y_train.iloc[idx]
    m = make_model(name, n_jobs)
    m.fit(X_train, y_train)
    return {k: m.predict(X) for k, X in X_evals.items()}


# ---------------------------------------------------------------- non-learned reference baselines
def persistence(df_eval):
    """yesterday's AQI_eod at the same station (NaN where unavailable)"""
    return df_eval["aqi_l1"].values


def climatology(df_train, df_eval):
    """seasonal-mean reference: mean AQI_eod of the training rows in the same (city, month).
    Fallbacks when that cell is empty (test months absent from training, or an unseen city):
    city mean over the training rows -> same month across all training cities -> global mean."""
    by_city_month = df_train.groupby(["city", "month"])["y"].mean()
    by_city = df_train.groupby("city")["y"].mean()
    by_month = df_train.groupby("month")["y"].mean()
    glob = df_train["y"].mean()

    def one(c, m):
        if (c, m) in by_city_month.index:
            return by_city_month[(c, m)]
        if c in by_city.index:
            return by_city[c]
        return by_month.get(m, glob)

    return np.asarray([one(c, m) for c, m in zip(df_eval.city, df_eval.month)], dtype=float)
