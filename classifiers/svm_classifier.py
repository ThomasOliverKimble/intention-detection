import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from utils.utils import print_eval_metrics
from typing import List, Any


class SvmClassifier:
    def __init__(self, dataset_path: str, rs: int = 42) -> None:
        """
        Initializes the SvmClassifier using TFIDF for text with a dataset.

        Args:
            dataset_path (str): Path to the dataset file.
            rs (int, optional): Random state for reproducibility. Defaults to 42.
        """
        self.model = SVC(random_state=rs)
        self.vectorizer = TfidfVectorizer()

        # Load dataset
        self.load_dataset(dataset_path)

    def load_dataset(self, dataset_path: str) -> None:
        """
        Loads and preprocesses the dataset from the given path.

        Args:
            dataset_path (str): Path to the dataset file.
        """
        # Load the dataset
        df = pd.read_csv(dataset_path)

        # Split the dataset
        train_df, val_df = train_test_split(df, test_size=0.1)

        # Preprocessing and vectorizing the text data
        self.X_train = self.vectorizer.fit_transform(train_df["text"])
        self.X_val = self.vectorizer.transform(val_df["text"])

        # Labels
        self.y_train = train_df["label"]
        self.y_val = val_df["label"]

    def train_model(self) -> None:
        """
        Trains the SVM model on the training dataset and evaluates it on the validation set.
        """
        # Training the SVM Classifier
        self.model.fit(self.X_train, self.y_train)

        # Evaluate model
        print("Validation set scores:")
        print_eval_metrics(self.y_val, self.model.predict(self.X_val))

    def save_model(self, path: str) -> None:
        """
        Saves the trained model to the specified path.

        Args:
            path (str): Path where the model should be saved.
        """
        # Save the trained model to the specified path
        joblib.dump(self.model, path)

    def predict(self, texts: List[str]) -> List[str]:
        """
        Predicts the labels for a list of texts.

        Args:
            texts (List[str]): A list of texts to classify.

        Returns:
            List[str]: The predicted labels for the input texts.
        """
        # Vectorize dataset
        vectorized_sentences = self.vectorizer.transform(texts)
        return self.model.predict(vectorized_sentences)
