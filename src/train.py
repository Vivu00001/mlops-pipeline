import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def run_training():

    mlflow.set_tracking_uri(
        "file:./mlruns"
    )

    mlflow.set_experiment(
        "Retail_Prediction"
    )

    os.makedirs(
        "artifacts/models",
        exist_ok=True
    )

    X_train = joblib.load(
        "artifacts/X_train.pkl"
    )

    y_train = joblib.load(
        "artifacts/y_train.pkl"
    )

    models = {

        "rf": RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

        "xgb": XGBClassifier(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        ),

        "lgbm": LGBMClassifier(
            n_estimators=50,
            max_depth=6,
            random_state=42
        )
    }

    for name, model in models.items():

        with mlflow.start_run(
            run_name=name
        ):

            model.fit(
                X_train,
                y_train
            )

            cv_score = cross_val_score(
                model,
                X_train,
                y_train,
                cv=3,
                scoring="f1"
            ).mean()

            mlflow.log_param(
                "model_name",
                name
            )

            mlflow.log_param(
                "n_samples",
                X_train.shape[0]
            )

            mlflow.log_param(
                "n_features",
                X_train.shape[1]
            )

            mlflow.log_metric(
                "cv_f1_score",
                float(cv_score)
            )

            mlflow.sklearn.log_model(
                model,
                name=name
            )

            joblib.dump(
                model,
                f"artifacts/models/{name}.pkl"
            )

            print(
                f"{name} trained | CV F1 = {cv_score:.4f}"
            )

    print(
        "Training Completed"
    )


if __name__ == "__main__":
    run_training()