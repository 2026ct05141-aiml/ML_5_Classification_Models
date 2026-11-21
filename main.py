from Models.classification_models import train_classification_models
from Models.test_and_evaluate import main as run_test_and_evaluate


def main():
    print("Model Training")
    train_classification_models()

    print("Model Testing")
    run_test_and_evaluate()


if __name__ == "__main__":
    main()