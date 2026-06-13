import pandas as pd

from src.utils import load_params

def run_validation():

    params = load_params()

    df = pd.read_csv(
        params["data"]["ingested"]
    )

    if df.empty:
        raise Exception("Dataset Empty")

    print(
        f"Rows={len(df)}"
    )

    df.to_csv(
        params["data"]["validated"],
        index=False
    )

    print("Validation Completed")

if __name__ == "__main__":
    run_validation()