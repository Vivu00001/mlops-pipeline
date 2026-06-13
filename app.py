from src.ingestion import run_ingestion
from src.validation import run_validation
from src.preprocessing import run_preprocessing
from src.feature_engineering import run_feature_engineering
from src.train import run_training
from src.evaluate import run_evaluation
from src.select_best_model import run_best_model_selection

if __name__ == "__main__":

    run_ingestion()

    run_validation()

    run_preprocessing()

    run_feature_engineering()

    run_training()

    run_evaluation()

    run_best_model_selection()

    print(
        "Pipeline Completed"
    )