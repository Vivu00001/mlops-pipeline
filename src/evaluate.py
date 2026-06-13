import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import json
import joblib
import mlflow

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def run_evaluation():

    mlflow.set_tracking_uri(
        "file:./mlruns"
    )

    mlflow.set_experiment(
        "Retail_Prediction"
    )

    X_test = joblib.load(
        "artifacts/X_test.pkl"
    )

    y_test = joblib.load(
        "artifacts/y_test.pkl"
    )

    os.makedirs(
        "reports",
        exist_ok=True
    )

    results = {}

    for file in os.listdir(
        "artifacts/models"
    ):

        if not file.endswith(".pkl"):
            continue

        model_name = file.replace(
            ".pkl",
            ""
        )

        model = joblib.load(
            f"artifacts/models/{file}"
        )

        y_pred = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        try:

            y_prob = model.predict_proba(
                X_test
            )[:, 1]

            roc_auc = roc_auc_score(
                y_test,
                y_prob
            )

        except:

            roc_auc = 0.0

        with mlflow.start_run(
            run_name=f"{model_name}_evaluation"
        ):

            mlflow.log_metric(
                "accuracy",
                accuracy
            )

            mlflow.log_metric(
                "precision",
                precision
            )

            mlflow.log_metric(
                "recall",
                recall
            )

            mlflow.log_metric(
                "f1_score",
                f1
            )

            mlflow.log_metric(
                "roc_auc",
                roc_auc
            )

        results[model_name] = {

            "accuracy": float(
                accuracy
            ),

            "precision": float(
                precision
            ),

            "recall": float(
                recall
            ),

            "f1": float(
                f1
            ),

            "roc_auc": float(
                roc_auc
            )
        }

    with open(
        "reports/metrics.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        json.dumps(
            results,
            indent=4
        )
    )

    print(
        "Evaluation Completed"
    )


if __name__ == "__main__":
    run_evaluation()