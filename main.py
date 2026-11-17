from Models.classification_models import train_classification_models
from Models.test_and_evaluate import main as run_test_and_evaluate


def main():
    print("====================================================================")
    print("STEP 1: PREPROCESSING, MODEL TRAINING & ARTIFACT SERIALIZATION")
    print("====================================================================")
    train_classification_models()

    print("\n====================================================================")
    print("STEP 2: SAVED MODEL EVALUATION, VISUALIZATION & EXTERNAL PREDICTION")
    print("====================================================================")
    run_test_and_evaluate()


if __name__ == "__main__":
    main()