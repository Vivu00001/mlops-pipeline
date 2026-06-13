import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from src.utils import load_params


def run_feature_engineering():

    params = load_params()

    df = pd.read_csv(
        params["data"]["processed"]
    )

    df["Revenue"] = (
        df["Quantity"]
        * df["UnitPrice"]
    )

    df["HighValueCustomer"] = (
        df["Revenue"] > 1000
    ).astype(int)

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"]
    )

    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day

    df.drop(
        columns=["InvoiceDate"],
        inplace=True
    )

    target = params["target"]

    X = df.drop(
        columns=[
            target,
            "Revenue"
        ]
    )

    y = df[target]

    num_cols = X.select_dtypes(
        include="number"
    ).columns

    cat_cols = X.select_dtypes(
        include="object"
    ).columns

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                StandardScaler(),
                num_cols
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                cat_cols
            )
        ]
    )

    X = preprocessor.fit_transform(X)

    joblib.dump(
        preprocessor,
        "artifacts/preprocessor.pkl"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    joblib.dump(
        X_train,
        "artifacts/X_train.pkl"
    )

    joblib.dump(
        X_test,
        "artifacts/X_test.pkl"
    )

    joblib.dump(
        y_train,
        "artifacts/y_train.pkl"
    )

    joblib.dump(
        y_test,
        "artifacts/y_test.pkl"
    )

    print(
        "Feature Engineering Completed"
    )


if __name__ == "__main__":
    run_feature_engineering()