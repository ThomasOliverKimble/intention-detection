import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    Trainer,
    TrainingArguments,
)


class IntentDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[int]], labels: List[int]):
        """
        Initializes the dataset with encodings and labels.

        Args:
            encodings (Dict[str, List[int]]): The tokenized encodings of the input text.
            labels (List[int]): The labels for each input text.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns an item at the specified index in the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the encoded inputs and their corresponding label as tensors.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        """
        Returns the size of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)


class BertClassifier:
    def __init__(self, dataset_path: str, model_path: str = "camembert-base"):
        """
        Initializes the classifier with a specified BERT model and tokenizer.
        """
        self.model_path = model_path
        self.class_names = [
            "translate",
            "travel_alert",
            "flight_status",
            "lost_luggage",
            "travel_suggestion",
            "carry_on",
            "book_hotel",
            "book_flight",
            "out_of_scope",
        ]
        self.model = CamembertForSequenceClassification.from_pretrained(
            self.model_path, num_labels=len(self.class_names)
        )
        self.tokenizer = CamembertTokenizer.from_pretrained(self.model_path)
        self.tokenizer_max_length = 128

        # Load data
        self.load_dataset(dataset_path)

    def load_dataset(self, dataset_path: str) -> None:
        """
        Loads the dataset from a CSV file, tokenizes the text, and prepares it for training and validation.

        Args:
            dataset_path (str): The file path of the dataset CSV file.
        """
        # Load into datafram and create maps for id and labels
        df = pd.read_csv(dataset_path)
        self.label_to_id = {label: id for id, label in enumerate(self.class_names)}
        self.id_to_label = {id: label for label, id in self.label_to_id.items()}

        # Preprocess for model
        df["label"] = df["label"].map(self.label_to_id)

        # Split for training and validation
        train_df, val_df = train_test_split(df, test_size=0.2)

        # Get encodings for dataset creation
        train_encodings = self.tokenizer(
            train_df["text"].to_list(),
            truncation=True,
            padding=True,
            max_length=self.tokenizer_max_length,
        )
        val_encodings = self.tokenizer(
            val_df["text"].to_list(),
            truncation=True,
            padding=True,
            max_length=self.tokenizer_max_length,
        )

        # Create torch datasets
        self.train_dataset = IntentDataset(train_encodings, train_df["label"].to_list())
        self.val_dataset = IntentDataset(val_encodings, val_df["label"].to_list())

    def create_trainer(
        self,
        num_epochs: int = 20,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
    ) -> None:
        """
        Creates a Trainer instance for training and evaluating the model.

        Args:
            num_epochs (int): The number of epochs for training.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation.
        """
        # Set number of epochs for plot
        self.num_epochs = num_epochs

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./models/results",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            warmup_steps=250,
            weight_decay=0.01,
            logging_dir="./models/logs",
            evaluation_strategy="steps",
            logging_steps=round(len(self.train_dataset) / train_batch_size),
        )

        # Define trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )

    def train_model(self) -> None:
        """
        Calls the train function of the model's trainer.
        """
        # Create trainer
        self.create_trainer()

        # Train model
        self.trainer.train()

    def save_model(self, path: str) -> None:
        """
        Saves the model and tokenizer to the specified path.

        Args:
            path (str): The path where the model and tokenizer should be saved.
        """
        # Save model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def predict_class(self, text: str) -> Tuple[str, float]:
        """
        Predicts the class of a given text.

        Args:
            text (str): The input text to classify.

        Returns:
            Tuple[str, float]: A tuple containing the predicted class name and the probability of that class.
        """
        # Encode the input sentence
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        # Predict
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1).squeeze()

        # Get the predicted class
        predicted_class_index = probabilities.argmax().item()
        predicted_class_name = self.class_names[predicted_class_index]

        return predicted_class_name, probabilities[predicted_class_index].item()

    def predict(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """
        Predicts the classes for a list of texts.

        Args:
            texts (List[str]): A list of texts to classify.

        Returns:
            Tuple[List[str], List[float]]: Two lists, one with the predicted class names and another with the corresponding probabilities.
        """
        # Empty lists
        predicted_classes = []
        probabilities = []

        # Predict
        for text in texts:
            predicted_class, probability = self.predict_class(text)
            predicted_classes.append(predicted_class)
            probabilities.append(probability)

        return predicted_classes, probabilities

    def plot_loss_curves(self) -> None:
        """
        Plots the training and validation loss curves over epochs.
        """
        # Get the training and validation losses
        epochs = range(1, self.num_epochs + 1)
        training_loss = [
            logs["loss"]
            for logs in self.trainer.state.log_history[0 : self.num_epochs * 2 : 2]
        ]
        validation_loss = [
            logs["eval_loss"]
            for logs in self.trainer.state.log_history[1 : self.num_epochs * 2 + 1 : 2]
        ]

        # Plot the training and validation losses
        plt.plot(epochs, training_loss, "bo-", label="Training Loss")
        plt.plot(epochs, validation_loss, "ro-", label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.show()
