import argparse
import os
import pandas as pd
from typing import List
from data_augmentation import data_augmentation
from classifiers import bert_classifier, nn_classifier, svm_classifier, rf_classifier
from utils.utils import get_eval_metrics

def augment_dataset(raw_data_path: str, training_data_path: str) -> None:
    """
    Augment the dataset.
    """
    # Augment dataset
    api_key = ""
    data_augment = data_augmentation.DataAugmentation(
        raw_data_path, training_data_path, api_key
    )
    data_augment.generate_new_content()

def train_models(training_data_path: str) -> None:
    """
    Train all the models.
    """
    # Bert classifier
    bert_clf = bert_classifier.BertClassifier(training_data_path)
    bert_clf.train_model()
    bert_clf.save_model("models/bert-model")

    # NN classifier
    nn_clf = nn_classifier.NeuralNetClassifier(training_data_path)
    nn_clf.train_model()
    nn_clf.save_model("models/nn-model")

    # SVM classifier
    svm_clf = svm_classifier.SvmClassifier(training_data_path)
    svm_clf.train_model()
    svm_clf.save_model("models/svm-model")

    # Random Forest classifier
    rf_clf = rf_classifier.RandomForestTextClassifier(training_data_path)
    rf_clf.train_model()
    rf_clf.save_model("models/rf-model")

def ensure_dir(file_path: str) -> None:
    """
    Ensure that a directory exists; if not, create it.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_predictions_and_metrics(X_test: List[str], y_pred: List[str], y_true: List[str], model_name: str) -> None:
    """
    Save the predictions and evaluation metrics for a given model.
    """
    # Ensure the directories exist
    predictions_path = f"evaluation/predicted/{model_name}_predictions.csv"
    metrics_path = f"evaluation/metrics/{model_name}_metrics.csv"
    ensure_dir(predictions_path)
    ensure_dir(metrics_path)
    
    # Save the test inputs and predicted labels
    predictions_df = pd.DataFrame({'text': X_test, 'predicted_label': y_pred})
    predictions_df.to_csv(predictions_path, index=False)
    
    # Compute and save the evaluation metrics
    metrics = get_eval_metrics(y_true, y_pred)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)

def predict(X_test: List[str], y_test: List[str], path: str) -> None:
    """
    Evaluate and save predictions + metrics for each model.
    """
    # Bert classifier
    bert_clf = bert_classifier.BertClassifier(path, model_path="models/bert-model")
    y_bert = bert_clf.predict(X_test)[0]
    save_predictions_and_metrics(X_test, y_bert, y_test, 'bert_clf')
    
    # NN classifier
    # nn_clf = nn_classifier.NeuralNetClassifier(path, model_path="models/nn-model")
    # y_nn = nn_clf.predict(X_test)
    # save_predictions_and_metrics(X_test, y_nn, y_test, 'nn_clf')
    
    # SVM classifier
    svm_clf = svm_classifier.SvmClassifier(path)
    svm_clf.train_model()
    y_svm = svm_clf.predict(X_test)
    save_predictions_and_metrics(X_test, y_svm, y_test, 'svm_clf')
    
    # Random Forest classifier
    rf_clf = rf_classifier.RandomForestTextClassifier(path)
    rf_clf.train_model()
    y_rf = rf_clf.predict(X_test)
    save_predictions_and_metrics(X_test, y_rf, y_test, 'rf_clf')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text classification tasks.")
    parser.add_argument('--test_path', type=str, default="data/raw/intent-detection-test.csv",
                        help='Path to the test dataset')
    parser.add_argument('--train', dest='train', action='store_true',
                        help='Flag to enable model training. Training is skipped if not specified.')
    parser.add_argument('--augment', dest='augment', action='store_true',
                        help='Flag to enable data augmentation. Augmentation is skipped if not specified.')

    args = parser.parse_args()

    # Dataset paths
    raw_data_path = "data/raw/intent-detection-train.csv"
    training_data_path = "data/augmented/augmented_data.csv"
    testing_data_path = args.test_path

    if args.augment:
        # Augment dataset
        print("Data augmentation is enabled.")
        augment_dataset(raw_data_path, training_data_path)
    else:
        print("Data augmentation is skipped.")

    if args.train:
        # Train and save models
        print("Model training is enabled.")
        train_models(training_data_path)
    else:
        print("Model training is skipped.")

    if os.path.exists(testing_data_path):
        # Get test data
        df_test = pd.read_csv(testing_data_path)
        X_test = df_test["text"].to_list()
        y_test = df_test["label"].to_list()

        # Predictions
        predict(X_test, y_test, testing_data_path)
    else:
        print(f"Test path {testing_data_path} does not exist. Skipping prediction.")
