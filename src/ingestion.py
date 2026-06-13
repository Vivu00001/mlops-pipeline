import pandas as pd

from src.utils import load_params

def run_ingestion():

    params = load_params()

    df = pd.read_csv(
        params["data"]["raw"],
        encoding="latin1"
    )

    print("Saving to:", params["data"]["ingested"])

    df.to_csv(
        params["data"]["ingested"],
        index=False
    )

    print("File saved successfully")
    
if __name__ == "__main__":
    run_ingestion()