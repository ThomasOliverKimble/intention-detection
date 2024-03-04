import torch
import torch.nn as nn
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from typing import Tuple, Any


class NeuralNet(nn.Module):
    def __init__(
        self, input_dimension: int, output_dimension: int, hidden_dimension: int = 256
    ) -> None:
        """
        Initializes a simple neural network using PyTorch's nn.Module.

        Parameters:
            input_dimension (int): Size of the input features.
            output_dimension (int): Size of the output features.
            hidden_dimension (int, optional): Size of the hidden layer. Defaults to 256.
        """
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dimension, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, hidden_dimension)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dimension, output_dimension)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the neural network.

        Parameters:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the network.
        """
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out


class IntentDataset(Dataset):
    def __init__(self, X_data: Tensor, y_data: Tensor) -> None:
        """
        A custom dataset for handling intents.

        Args:
            X_data (Tensor): The features tensors.
            y_data (Tensor): The target labels tensors.
        """
        self.X_data = X_data
        self.y_data = y_data

    def __len__(self) -> int:
        """
        Returns the size of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.X_data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Retrieves an item by its index.

        Args:
            index (int): The index of the item.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the features and the target label tensors.
        """
        return self.X_data[index], self.y_data[index]


class NeuralNetClassifier:
    def __init__(self, dataset_path: str, model_path: Any = False) -> None:
        """
        Initializes the classifier, loads the dataset, and prepares the model.

        Args:
            dataset_path (str): The path to the dataset.
            model_path (Any, optional): The model path if loading a pretrained model. Defaults to False.
        """
        self.vectorizer = CountVectorizer(max_features=1000)
        self.label_encoder = LabelEncoder()

        self.load_dataset(dataset_path)
        self.create_model(model_path)

    def load_dataset(self, dataset_path: str) -> None:
        """
        Loads and preprocesses the dataset from the given path.

        Args:
            dataset_path (str): The path to the dataset.
        """
        # Load the dataset
        df = pd.read_csv(dataset_path)
        train_df, val_df = train_test_split(df, test_size=0.2)

        # Encode the labels
        train_df["label"] = self.label_encoder.fit_transform(train_df["label"])
        val_df["label"] = self.label_encoder.transform(val_df["label"])

        # Vectorize the sentences
        X_train = self.vectorizer.fit_transform(train_df["text"]).toarray()
        X_val = self.vectorizer.transform(val_df["text"]).toarray()
        y_train = train_df["label"].values
        y_val = val_df["label"].values

        # Convert arrays to PyTorch tensors
        self.X_train = torch.FloatTensor(X_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_train = torch.LongTensor(y_train)
        self.y_val = torch.LongTensor(y_val)

        # Dataset
        self.train_dataset = IntentDataset(self.X_train, self.y_train)
        self.val_dataset = IntentDataset(self.X_val, self.y_val)

        # DataLoader
        self.train_loader = DataLoader(
            dataset=self.train_dataset, batch_size=32, shuffle=True
        )
        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=1)

    def create_model(self, model_path: Any) -> None:
        """
        Creates the neural network model, loss function, and optimizer.

        Args:
            dataset_path (str): The path to the dataset.
            model_path (Any): The model path if loading a pretrained model.
        """
        # Model, loss and optimizer
        input_dim = self.X_train.shape[1]
        hidden_dim = 256
        output_dim = len(self.label_encoder.classes_)
        self.model = NeuralNet(input_dim, hidden_dim, output_dim)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train_model(self, num_epochs: int = 20) -> None:
        """
        Trains the model for a specified number of epochs.

        Args:
            num_epochs (int, optional): The number of epochs to train for. Defaults to 20.
        """
        # Training over 20 epochs
        for epoch in range(num_epochs):
            for inputs, labels in self.train_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            eval_accuracy = self.evaluate_model()
            print(
                f"Epoch {epoch+1}, Loss: {loss.item()}, Eval accuracy: {eval_accuracy}%"
            )

    def evaluate_model(self) -> float:
        """
        Evaluates the model on the validation dataset.

        Returns:
            float: The accuracy of the model on the validation dataset.
        """
        # Evaluation
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in self.val_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

    def save_model(self, path: str) -> None:
        """
        Saves the model to the specified path.

        Args:
            path (str): The path where the model should be saved.
        """
        # Save the model state
        torch.save(self.model.state_dict(), path)

    def predict(self, texts: list) -> list:
        """
        Predicts the labels for a list of texts.

        Args:
            texts (list): A list of text strings to classify.

        Returns:
            list: The predicted labels for the input texts.
        """
        # Vectorize the input texts
        texts_vectorized = self.vectorizer.transform(texts).toarray()
        texts_tensor = torch.FloatTensor(texts_vectorized)

        # Predict
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.model(texts_tensor)
            _, predicted_indices = torch.max(outputs, 1)

        # Convert indices to labels
        predicted_labels = self.label_encoder.inverse_transform(
            predicted_indices.numpy()
        )
        return predicted_labels
