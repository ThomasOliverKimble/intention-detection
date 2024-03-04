import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils.utils import print_eval_metrics
from typing import List, Any


class RandomForestTextClassifier:
    def __init__(self, dataset_path: str, rs: int = 42) -> None:
        """
        Initializes the RandomForestTextClassifier using CountVectorizer for text (BoW) with a dataset.

        Args:
            dataset_path (str): The file path to the dataset.
            rs (int, optional): Random state for reproducibility. Defaults to 42.
        """
        self.model = RandomForestClassifier(random_state=rs)
        self.vectorizer = CountVectorizer()

        # Load dataset
        self.load_dataset(dataset_path)

    def load_dataset(self, dataset_path: str) -> None:
        """
        Loads the dataset from the specified path and prepares it for training and validation.

        Args:
            dataset_path (str): The file path to the dataset.
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
        Trains the Random Forest Classifier with the training data.
        """
        # Training the Random Forest Classifier
        self.model.fit(self.X_train, self.y_train)

        # Evaluate model
        print_eval_metrics(self.y_val, self.model.predict(self.X_val))

    def save_model(self, path: str) -> None:
        """
        Saves the trained model to the specified path.

        Args:
            path (str): The file path where the model should be saved.
        """
        # Save the trained model to the specified path
        joblib.dump(self.model, path)

    def predict(self, texts: List[str]) -> List[str]:
        """
        Predicts the labels of the given texts.

        Args:
            texts (List[str]): The texts to predict.

        Returns:
            List[str]: The predicted labels.
        """
        # Vectorize dataset
        vectorized_sentences = self.vectorizer.transform(texts)
        return self.model.predict(vectorized_sentences)
