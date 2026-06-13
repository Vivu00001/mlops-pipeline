import pandas as pd

from src.utils import load_params

def run_preprocessing():

    params = load_params()

    df = pd.read_csv(
        params["data"]["validated"]
    )

    df.drop_duplicates(
        inplace=True
    )

    for col in df.columns:

        if df[col].dtype == "object":

            if df[col].isnull().sum() > 0:

                df[col] = df[col].fillna(
                    df[col].mode()[0]
                )

        else:

            if df[col].isnull().sum() > 0:

                df[col] = df[col].fillna(
                    df[col].median()
                )

    df.to_csv(
        params["data"]["processed"],
        index=False
    )

    print("Preprocessing Completed")

if __name__ == "__main__":
    run_preprocessing()