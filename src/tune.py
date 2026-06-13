import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def run_tuning():

    mlflow.set_tracking_uri(
        "file:./mlruns"
    )

    mlflow.set_experiment(
        "Retail_Prediction"
    )

    X_train = joblib.load(
        "artifacts/X_train.pkl"
    )

    y_train = joblib.load(
        "artifacts/y_train.pkl"
    )

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    # ===================================
    # Random Forest
    # ===================================

    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    rf_params = {

        "n_estimators": [50, 100, 200],

        "max_depth": [5, 10, 20],

        "min_samples_split": [2, 5, 10]
    }

    rf_search = RandomizedSearchCV(

        estimator=rf,

        param_distributions=rf_params,

        n_iter=5,

        cv=3,

        scoring="f1",

        random_state=42,

        n_jobs=-1
    )

    rf_search.fit(
        X_train,
        y_train
    )

    best_rf = rf_search.best_estimator_

    joblib.dump(
        best_rf,
        "artifacts/tuned_rf.pkl"
    )

    with mlflow.start_run(
        run_name="RF_Tuning"
    ):

        mlflow.log_params(
            rf_search.best_params_
        )

        mlflow.log_metric(
            "best_cv_f1",
            rf_search.best_score_
        )

    print(
        "RF Best:",
        rf_search.best_params_
    )

    # ===================================
    # XGBoost
    # ===================================

    xgb = XGBClassifier(

        random_state=42,

        eval_metric="logloss"
    )

    xgb_params = {

        "n_estimators": [50, 100],

        "max_depth": [3, 6, 10],

        "learning_rate": [0.01, 0.1, 0.2]
    }

    xgb_search = RandomizedSearchCV(

        estimator=xgb,

        param_distributions=xgb_params,

        n_iter=5,

        cv=3,

        scoring="f1",

        random_state=42,

        n_jobs=-1
    )

    xgb_search.fit(
        X_train,
        y_train
    )

    best_xgb = xgb_search.best_estimator_

    joblib.dump(
        best_xgb,
        "artifacts/tuned_xgb.pkl"
    )

    with mlflow.start_run(
        run_name="XGB_Tuning"
    ):

        mlflow.log_params(
            xgb_search.best_params_
        )

        mlflow.log_metric(
            "best_cv_f1",
            xgb_search.best_score_
        )

    print(
        "XGB Best:",
        xgb_search.best_params_
    )

    # ===================================
    # LightGBM
    # ===================================

    lgbm = LGBMClassifier(
        random_state=42
    )

    lgbm_params = {

        "n_estimators": [50, 100],

        "max_depth": [3, 6, 10],

        "learning_rate": [0.01, 0.1, 0.2]
    }

    lgbm_search = RandomizedSearchCV(

        estimator=lgbm,

        param_distributions=lgbm_params,

        n_iter=5,

        cv=3,

        scoring="f1",

        random_state=42,

        n_jobs=-1
    )

    lgbm_search.fit(
        X_train,
        y_train
    )

    best_lgbm = lgbm_search.best_estimator_

    joblib.dump(
        best_lgbm,
        "artifacts/tuned_lgbm.pkl"
    )

    with mlflow.start_run(
        run_name="LGBM_Tuning"
    ):

        mlflow.log_params(
            lgbm_search.best_params_
        )

        mlflow.log_metric(
            "best_cv_f1",
            lgbm_search.best_score_
        )

    print(
        "LGBM Best:",
        lgbm_search.best_params_
    )

    print(
        "\nHyperparameter Tuning Completed"
    )


if __name__ == "__main__":
    run_tuning()