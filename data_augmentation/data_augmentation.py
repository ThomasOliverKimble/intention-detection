import csv
from typing import List
from openai import OpenAI


class DataAugmentation:
    """
    A class to augment the intent dataset using OpenAI's GPT model.

    Attributes:
        input_path (str): Path to the raw data CSV file.
        output_path (str): Path to save the augmented data CSV file.
        client (OpenAI): OpenAI API client instance.
        model (str): The model used for data augmentation.
        class_labels (List[str]): List of class labels for data augmentation.
        input_data_str (str): The input dataset as a str.
    """

    def __init__(
        self,
        raw_data_path: str,
        augmented_data_path: str,
        api_key: str,
    ) -> None:
        """
        Initializes the DataAugmentation class with paths for input and output data, and an API key for OpenAI.

        Args:
            raw_data_path (str): The file path for the input CSV data.
            augmented_data_path (str): The file path for the output CSV data.
            api_key (str): The API key for accessing OpenAI services.
        """
        self.input_path: str = raw_data_path
        self.output_path: str = augmented_data_path
        self.client: OpenAI = OpenAI(api_key=api_key)
        self.model: str = "gpt-4"
        self.class_labels: List[str] = [
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
        self.input_data_str: str = self.csv_to_str(self.input_path)

    def csv_to_str(self, input_path: str) -> str:
        """
        Reads a CSV file and converts its content to a single string.

        Args:
            input_path (str): The path to the input CSV file.

        Returns:
            str: A string containing the contents of the CSV file.

        Raises:
            FileNotFoundError: If the input file does not exist.
            Exception: For other exceptions during file processing.
        """
        try:
            with open(input_path, "r", encoding="utf-8") as file:
                csv_reader = csv.reader(file)
                csv_data = ""

                for row in csv_reader:
                    csv_data += ",".join(row) + "\n"

                return csv_data
        except FileNotFoundError as e:
            raise FileNotFoundError(f"The file at {input_path} was not found.") from e
        except Exception as e:
            raise Exception(
                f"An error occurred while processing the file: {str(e)}"
            ) from e

    def str_to_csv(self, output_path: str, output_str: str, class_label: str) -> None:
        """
        Writes a string to a CSV file, appending rows categorized by a class label.

        Args:
            output_path (str): The path to the output CSV file.
            output_str (str): The string to write to the file.
            class_label (str): The label to categorize each row.

        Raises:
            FileNotFoundError: If the output file does not exist.
            Exception: For other exceptions during file processing.
        """
        rows = output_str.strip().split("\n")

        try:
            with open(output_path, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                for r in rows:
                    parts = r.split(f",{class_label}")
                    if len(parts) != 2:
                        parts = r.split(f", {class_label}")
                    if len(parts) == 2:
                        cleaned_part = (
                            parts[0].replace('""', "'").replace("?", "").strip()
                        )
                        writer.writerow([cleaned_part, class_label])
        except FileNotFoundError as e:
            raise FileNotFoundError(f"The file at {output_path} was not found.") from e
        except Exception as e:
            raise Exception(
                f"An error occurred while processing the file: {str(e)}"
            ) from e

    def create_messages(self, num_lines: int, class_label: str) -> List[dict]:
        """
        Creates a list of messages for the OpenAI API to generate augmented data.

        Args:
            num_lines (int): The number of lines to generate.
            class_label (str): The class label for which to generate data.

        Returns:
            List[dict]: A list of dictionaries representing the messages for the API.
        """
        messages = [
            {
                "role": "system",
                "content": "Vous êtes un générateur de text qui sert à augmenter un dataset.",
            },
            {
                "role": "assistant",
                "content": """Voici les labels du dataset.
                    translate: l'utilisateur souhaite traduire une phrase dans une autre langue",
                    travel_alert: l'utilisateur demande si sa destination est concernée par une alerte de voyage
                    flight_status: l'utilisateur demande des informations sur le statut de son vol
                    lost_luggage: l'utilisateur signale la perte de ses bagages
                    travel_suggestion: l'utilisateur souhaite une recommandation de voyage
                    carry_on: l'utilisateur souhaite des informations sur les bagages à main
                    book_hotel: l'utilisateur souhaite réserver un hôtel
                    book_flight: l'utilisateur souhaite réserver un vol""",
            },
            {
                "role": "assistant",
                "content": f"Voici le dataset: {self.input_data_str}",
            },
            {
                "role": "user",
                "content": f"Générez {num_lines} lignes en plus pour le label {class_label}, et rien d'autre.",
            },
        ]

        return messages

    def get_augmented_dataset_response(self, messages: List[dict]) -> str:
        """
        Sends messages to the OpenAI API and retrieves the generated augmented data.

        Args:
            messages (List[dict]): The list of messages to send to the API.

        Returns:
            str: The generated augmented data as a string.
        """
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        response_content = response.choices[0].message.content
        return response_content

    def generate_new_content(self, n: int = 150) -> None:
        """
        Generates new content for each class label and writes it to the output CSV file.

        Args:
            n (int): The number of lines to generate for each class label.
        """
        print("Generating new content.")
        for class_label in self.class_labels:
            print("  class:", class_label)
            messages = self.create_messages(n, class_label)
            response = self.get_augmented_dataset_response(messages)
            self.str_to_csv(self.output_path, response, class_label)
