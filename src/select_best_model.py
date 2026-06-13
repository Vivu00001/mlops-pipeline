import json
import joblib
import shutil
import os

def run_best_model_selection():

    with open(
        "reports/metrics.json",
        "r"
    ) as f:

        results = json.load(f)

    best_model = max(
        results,
        key=lambda x:
        results[x]["f1"]
    )

    print(
        f"Best Model: {best_model}"
    )

    os.makedirs(
        "artifacts/best_model",
        exist_ok=True
    )

    shutil.copy(
        f"artifacts/models/{best_model}.pkl",
        "artifacts/best_model/model.pkl"
    )

    with open(
        "artifacts/best_model/model_name.txt",
        "w"
    ) as f:

        f.write(best_model)

    print(
        "Best Model Saved"
    )


if __name__ == "__main__":
    run_best_model_selection()