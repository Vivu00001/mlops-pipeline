import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load(
    "artifacts/best_model/model.pkl"
)

preprocessor = joblib.load(
    "artifacts/preprocessor.pkl"
)


class PredictionInput(BaseModel):

    InvoiceNo: str
    StockCode: str
    Description: str
    Quantity: int
    UnitPrice: float
    CustomerID: float
    Country: str
    HighValueCustomer: int
    Year: int
    Month: int
    Day: int


@app.get("/")
def home():

    return {
        "message": "MLOps API Running"
    }


@app.post("/predict")
def predict(data: PredictionInput):

    df = pd.DataFrame(
        [data.model_dump()]
    )

    X = preprocessor.transform(df)

    prediction = model.predict(X)

    return {
        "prediction": int(prediction[0])
    }